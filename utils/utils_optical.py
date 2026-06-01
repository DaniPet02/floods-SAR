import ee
import geemap
import pandas as pd
from pathlib import Path

from utils import utils_setup as us
from utils import utils_SAR as su


# =============================================================================
# Sentinel-2 preprocessing
# =============================================================================

def preprocess_s2_collection(collection):
    """Scale Sentinel-2 SR bands, mask clouds with SCL and add NDWI/MNDWI/NDVI."""
    bands = ["B2", "B3", "B4", "B8", "B11"]

    def preprocess(image):
        image = ee.Image(image)
        scl = image.select("SCL")

        clear = (
            scl.eq(4)       # vegetation
            .Or(scl.eq(5))  # bare soil
            .Or(scl.eq(6))  # water
            .Or(scl.eq(7))  # unclassified / clear enough
        )

        reflectance = image.select(bands).divide(10000).updateMask(clear)

        ndwi = reflectance.normalizedDifference(["B3", "B8"]).rename("NDWI")
        mndwi = reflectance.normalizedDifference(["B3", "B11"]).rename("MNDWI")
        ndvi = reflectance.normalizedDifference(["B8", "B4"]).rename("NDVI")

        return (
            reflectance
            .addBands([ndwi, mndwi, ndvi])
            .copyProperties(image, image.propertyNames())
        )

    return collection.map(preprocess)


def make_s2_composite(collection, method="median"):
    """Build a simple Sentinel-2 composite from a preprocessed collection."""
    method = method.lower()

    if method == "median":
        return collection.median()

    if method == "mosaic":
        return collection.sort("CLOUDY_PIXEL_PERCENTAGE", False).mosaic()

    if method in ["least_cloudy", "first"]:
        return ee.Image(collection.sort("CLOUDY_PIXEL_PERCENTAGE").first())

    raise ValueError("method must be 'median', 'mosaic', or 'least_cloudy'.")

# =============================================================================
# Small summaries
# =============================================================================

def s2_metadata_df(collection, label):
    """Return a compact metadata table for a Sentinel-2 collection."""
    def image_to_feature(image):
        image = ee.Image(image)

        return ee.Feature(None, ee.Dictionary({
            "label": label,
            "date": ee.Date(image.get("system:time_start")).format("YYYY-MM-dd HH:mm"),
            "cloudy_pixel_percentage": image.get("CLOUDY_PIXEL_PERCENTAGE"),
            "mgrs_tile": image.get("MGRS_TILE"),
            "system_index": image.get("system:index"),
        }))

    fc = ee.FeatureCollection(collection.toList(collection.size()).map(image_to_feature))
    features = fc.getInfo().get("features", [])

    if not features:
        return pd.DataFrame()

    df = pd.DataFrame([feature["properties"] for feature in features])

    return df.sort_values(["label", "date"]).reset_index(drop=True)


def valid_pixel_percentage(image, region, band="B3", scale=20):
    """Estimate valid-pixel percentage over an AOI using the mask of one band."""
    valid_area = (
        image.select(band)
        .mask()
        .gt(0)
        .multiply(ee.Image.pixelArea())
        .rename("valid_area")
    )

    total_area = ee.Image.pixelArea().rename("total_area")

    valid = valid_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
        tileScale=4,
    ).get("valid_area").getInfo()

    total = total_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
        tileScale=4,
    ).get("total_area").getInfo()

    if valid is None or total in [None, 0]:
        return 0.0

    return float(valid / total * 100)


def optical_processing_df(event_key, s2, optical, optical_params):
    """Compact vertical summary of the Sentinel-2 optical configuration."""
    aoi = optical["aoi"]

    rows = [
        ("Pre-event S2 images", s2["pre"].size().getInfo()),
        ("Post-event S2 images", s2["post"].size().getInfo()),
        ("Composite method", optical_params.get("composite_method", "median")),
        ("Optical index", optical_params.get("optical_index", "MNDWI")),
        ("Optical threshold", optical_params.get("optical_threshold", 0.0)),
        ("Pre valid pixels", f"{valid_pixel_percentage(optical['pre_comp'], aoi):.1f}%"),
        ("Post valid pixels", f"{valid_pixel_percentage(optical['post_comp'], aoi):.1f}%"),
    ]

    return pd.DataFrame(rows, columns=["parameter", "value"])


# =============================================================================
# Optical products
# =============================================================================

def make_optical_inputs(event_key, s2, aoi, composite_method="median"):
    """Create Sentinel-2 pre/post composites and optical change products."""
    s2_pre = preprocess_s2_collection(s2["pre"])
    s2_post = preprocess_s2_collection(s2["post"])

    pre_comp = make_s2_composite(s2_pre, method=composite_method).clip(aoi)
    post_comp = make_s2_composite(s2_post, method=composite_method).clip(aoi)

    delta_ndwi = post_comp.select("NDWI").subtract(pre_comp.select("NDWI")).rename(
        f"{event_key}_delta_NDWI"
    )

    delta_mndwi = post_comp.select("MNDWI").subtract(pre_comp.select("MNDWI")).rename(
        f"{event_key}_delta_MNDWI"
    )

    delta_ndvi = post_comp.select("NDVI").subtract(pre_comp.select("NDVI")).rename(
        f"{event_key}_delta_NDVI"
    )

    return {
        "aoi": aoi,
        "s2_pre": s2_pre,
        "s2_post": s2_post,
        "pre_comp": pre_comp,
        "post_comp": post_comp,
        "delta_ndwi": delta_ndwi,
        "delta_mndwi": delta_mndwi,
        "delta_ndvi": delta_ndvi,
    }


def make_optical_masks(
    event_key,
    optical,
    permanent_water,
    optical_index="MNDWI",
    optical_threshold=0.0,
):
    """Create valid-pixel, optical-water and new-water masks."""
    optical_index = optical_index.upper()
    post_comp = optical["post_comp"]

    optical_valid = (
        post_comp
        .select("B3")
        .mask()
        .rename(f"{event_key}_optical_valid")
        .toByte()
    )

    optical_water = (
        post_comp
        .select(optical_index)
        .gt(optical_threshold)
        .rename(f"{event_key}_optical_water")
        .toByte()
    )

    optical_new_water = (
        optical_water
        .And(permanent_water.unmask(0).Not())
        .rename(f"{event_key}_optical_new_water")
        .toByte()
    )

    return {
        "optical_valid": optical_valid,
        "optical_water": optical_water,
        "optical_new_water": optical_new_water,
    }


def make_fusion_map(
    event_key,
    sar_confidence,
    optical_masks,
    permanent_water,
    sar_confidence_threshold=3,
):
    """Create a simple categorical SAR-optical comparison map."""
    sar_flood = sar_confidence.gte(sar_confidence_threshold)
    sar_low = sar_confidence.gt(0).And(sar_confidence.lt(sar_confidence_threshold))

    optical_water = optical_masks["optical_water"].unmask(0).gt(0)
    optical_valid = optical_masks["optical_valid"].unmask(0).gt(0)
    permanent_water = permanent_water.unmask(0).gt(0)

    fusion = ee.Image(0).rename(f"{event_key}_sar_optical_fusion")

    fusion = fusion.where(
        optical_valid.Not().And(sar_flood.Not()).And(sar_low.Not()),
        5,
    )

    fusion = fusion.where(sar_low, 6)
    fusion = fusion.where(sar_flood, 1)

    fusion = fusion.where(
        optical_valid.And(optical_water).And(sar_flood.Not()).And(sar_low.Not()),
        2,
    )

    fusion = fusion.where(
        optical_valid.And(optical_water).And(sar_flood),
        3,
    )

    fusion = fusion.where(permanent_water, 4)

    return fusion.toByte()


def make_agreement_table(
    event_key,
    sar_confidence,
    optical_masks,
    aoi,
    optical_index="MNDWI",
    optical_threshold=0.0,
    sar_confidence_threshold=3,
):
    """Compare SAR flood and optical new-water areas over valid optical pixels."""
    optical_valid = optical_masks["optical_valid"].unmask(0).gt(0)

    sar = (
        sar_confidence
        .gte(sar_confidence_threshold)
        .And(optical_valid)
    )

    optical = (
        optical_masks["optical_new_water"]
        .unmask(0)
        .gt(0)
        .And(optical_valid)
    )

    intersection = sar.And(optical)
    union = sar.Or(optical)

    sar_area = us.mask_area_km2(sar, aoi)
    optical_area = us.mask_area_km2(optical, aoi)
    intersection_area = us.mask_area_km2(intersection, aoi)
    union_area = us.mask_area_km2(union, aoi)

    return pd.DataFrame([{
        "event": event_key,
        "optical_index": optical_index,
        "optical_threshold": optical_threshold,
        "sar_confidence_threshold": sar_confidence_threshold,
        "sar_area_km2_valid_optical": sar_area,
        "optical_area_km2": optical_area,
        "intersection_km2": intersection_area,
        "union_km2": union_area,
        "iou": intersection_area / union_area if union_area > 0 else 0.0,
        "dice": (
            2 * intersection_area / (sar_area + optical_area)
            if (sar_area + optical_area) > 0
            else 0.0
        ),
    }])


# =============================================================================
# Map
# =============================================================================

def make_optical_map(
    event,
    places,
    optical,
    optical_masks,
    sar_confidence,
    fusion_map=None,
    sar_confidence_threshold=3,
):
    """Create a simple interactive Sentinel-2 optical map."""
    aoi = event["aoi"]
    vis = us.get_vis_params()

    m = geemap.Map(center=event["center"], zoom=event["zoom"])

    # Base optical layers
    m.addLayer(
        optical["post_comp"],
        vis["true_color"],
        "S2 post-event true color",
        True,
    )

    m.addLayer(
        optical["pre_comp"],
        vis["true_color"],
        "S2 pre-event true color",
        False,
    )

    # Main optical flood-related evidence
    m.addLayer(
        optical_masks["optical_new_water"].selfMask(),
        {
            "min": 0,
            "max": 1,
            "palette": ["0000ff"],
        },
        "Optical new water",
        True,
    )

    # Optional SAR comparison layer, switched off by default
    m.addLayer(
        sar_confidence.gte(sar_confidence_threshold).selfMask(),
        {
            "min": 0,
            "max": 1,
            "palette": ["00ffff"],
        },
        f"SAR flood confidence ≥ {sar_confidence_threshold}",
        False,
    )

    m.addLayer(
        us.styled_aoi(aoi, color=event.get("aoi_color", "red")),
        {},
        event.get("aoi_label", "AOI"),
    )

    if places is not None:
        m.addLayer(
            us.styled_places(places, aoi),
            {},
            "Places",
        )

    m.addLayerControl()
    return m

def make_optical_area_table(event_key, optical_masks, aoi):
    """Create a compact area table for Sentinel-2 optical water masks."""
    rows = [
        ("optical_water", optical_masks["optical_water"]),
        (
            "optical_new_water_without_permanent_water",
            optical_masks["optical_new_water"],
        ),
    ]

    return pd.DataFrame([
        {
            "event": event_key,
            "method": method,
            "area_km2": us.mask_area_km2(mask, aoi),
        }
        for method, mask in rows
    ])

def export_optical_vs_sar_png(optical_results, sar_results, region, output_dir, month, dimensions=900):
    """Export optical new water and final SAR flood mask side by side."""
    vis = us.get_vis_params()
    post_rgb = optical_results["post_comp"].visualize(**vis["true_color"])

    optical_rgb = post_rgb.blend(
        optical_results["optical_new_water"].selfMask().visualize(
            min=0, max=1, palette=["0000ff"], opacity=0.75
        )
    )

    sar_rgb = post_rgb.blend(
        sar_results["sar_flood"].selfMask().visualize(
            min=0, max=1, palette=["00ffff"], opacity=0.75
        )
    )

    return us.export_side_by_side_ee_images(
        optical_rgb,
        sar_rgb,
        region,
        {},
        Path(output_dir) / f"{month}_optical_new_water_vs_sar_flood.png",
        "S2 post-event + optical new water",
        "S2 post-event + final SAR flood mask",
        dimensions,
    )

# =============================================================================
# Public pipeline
# =============================================================================

def run_optical_event(
    event_key,
    event,
    s2,
    sar_confidence,
    permanent_water,
    places=None,
    output_dir=None,
    make_map=True,
    optical_index="MNDWI",
    optical_threshold=0.0,
    composite_method="median",
    sar_confidence_threshold=3,
    max_cloud_pre=None,
    max_cloud_post=None,
    pre_window_override=None,
    post_window_override=None,
    **unused_params,
):
    """
    Run a compact Sentinel-2 optical workflow.

    max_cloud_* and window_override parameters are accepted for compatibility with
    cfg.OPTICAL_PARAMS, but filtering is already done in utils_setup.load_event_datasets.
    """
    aoi = event["aoi"]

    optical = make_optical_inputs(
        event_key=event_key,
        s2=s2,
        aoi=aoi,
        composite_method=composite_method,
    )

    optical_masks = make_optical_masks(
        event_key=event_key,
        optical=optical,
        permanent_water=permanent_water,
        optical_index=optical_index,
        optical_threshold=optical_threshold,
    )

    fusion_map = make_fusion_map(
        event_key=event_key,
        sar_confidence=sar_confidence,
        optical_masks=optical_masks,
        permanent_water=permanent_water,
        sar_confidence_threshold=sar_confidence_threshold,
    )

    optical_params = {
        "optical_index": optical_index,
        "optical_threshold": optical_threshold,
        "composite_method": composite_method,
    }

    df_processing = optical_processing_df(
        event_key=event_key,
        s2=s2,
        optical=optical,
        optical_params=optical_params,
    )
    
    df_area = make_optical_area_table(
        event_key=event_key,
        optical_masks=optical_masks,
        aoi=aoi,
    )

    df_s2_metadata = pd.concat(
        [
            s2_metadata_df(s2["pre"], f"{event_key}_pre"),
            s2_metadata_df(s2["post"], f"{event_key}_post"),
        ],
        ignore_index=True,
    )

    df_agreement = make_agreement_table(
        event_key=event_key,
        sar_confidence=sar_confidence,
        optical_masks=optical_masks,
        aoi=aoi,
        optical_index=optical_index,
        optical_threshold=optical_threshold,
        sar_confidence_threshold=sar_confidence_threshold,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df_processing.to_csv(
            output_dir / f"{event_key}_optical_processing_summary.csv",
            index=False,
        )

        df_area.to_csv(
            output_dir / f"{event_key}_optical_area_summary.csv",
            index=False,
        )

        df_s2_metadata.to_csv(
            output_dir / f"{event_key}_s2_metadata.csv",
            index=False,
        )

        df_agreement.to_csv(
            output_dir / f"{event_key}_sar_optical_agreement.csv",
            index=False,
        )

    optical_map = (
        make_optical_map(
            event=event,
            places=places,
            optical=optical,
            optical_masks=optical_masks,
            sar_confidence=sar_confidence,
            fusion_map=fusion_map,
            sar_confidence_threshold=sar_confidence_threshold,
        )
        if make_map
        else None
    )

    return {
        "optical": optical,
        "masks": optical_masks,
        "df_processing": df_processing,
        "df_s2_metadata": df_s2_metadata,
        "df_agreement": df_agreement,
        "fusion_map": fusion_map,
        "map": optical_map,
        "df_area": df_area,

        **optical,
        **optical_masks,
    }

def export_optical_new_water_png(
    optical_results,
    region,
    output_dir,
    month,
    dimensions=900,
):
    """Export post-event Sentinel-2 true color with optical new-water overlay."""
    vis = us.get_vis_params()

    post_rgb = optical_results["post_comp"].visualize(**vis["true_color"])

    optical_new_water_overlay = (
        optical_results["optical_new_water"]
        .selfMask()
        .visualize(
            min=0,
            max=1,
            palette=["0000ff"],
            opacity=0.65,
        )
    )

    return us.export_single_ee_image_png(
        image=post_rgb.blend(optical_new_water_overlay),
        region=region,
        vis_params={},
        output_path=Path(output_dir) / f"{month}_optical_new_water_overlay.png",
        title="Sentinel-2 post-event true color + optical new water",
        dimensions=dimensions,
    )
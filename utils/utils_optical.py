import ee
import geemap
import pandas as pd
from pathlib import Path

from utils import utils_setup as us


# ============================================================
# OPTICAL VISUALIZATION
# ============================================================

def get_optical_vis_params():
    return {
        "true_color": {
            "bands": ["B4", "B3", "B2"],
            "min": 0.0,
            "max": 0.30,
        },
        "false_color": {
            "bands": ["B8", "B4", "B3"],
            "min": 0.0,
            "max": 0.45,
        },
        "ndwi": {
            "min": -0.5,
            "max": 0.8,
            "palette": ["8c510a", "ffffff", "01665e", "003c8f"],
        },
        "mndwi": {
            "min": -0.5,
            "max": 0.8,
            "palette": ["8c510a", "ffffff", "80cdc1", "003c8f"],
        },
        "ndvi": {
            "min": -0.2,
            "max": 0.8,
            "palette": ["8c510a", "ffffbf", "1a9850"],
        },
        "delta_index": {
            "min": -0.5,
            "max": 0.5,
            "palette": ["a50026", "ffffff", "313695"],
        },
        "optical_water": {
            "min": 0,
            "max": 1,
            "palette": ["ffffff", "0000ff"],
        },
        "optical_valid": {
            "min": 0,
            "max": 1,
            "palette": ["ff0000", "00ff00"],
        },
        "fusion": {
            "min": 0,
            "max": 6,
            "palette": [
                "ffffff",  # 0 no flood
                "1f78b4",  # 1 SAR only
                "33a02c",  # 2 optical only
                "e31a1c",  # 3 SAR + optical
                "08306b",  # 4 permanent water
                "bdbdbd",  # 5 cloud / no optical data
                "ff7f00",  # 6 uncertain low-confidence SAR
            ],
        },
        "sar_confidence": {
            "min": 0,
            "max": 3,
            "palette": ["ffffff", "ffff00", "ff9900", "ff0000"],
        },
    }


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _ee_fc_to_df(fc):
    info = fc.getInfo()
    features = info.get("features", [])

    if len(features) == 0:
        return pd.DataFrame()

    return pd.DataFrame([feature["properties"] for feature in features])


def _s2_metadata_df(collection, label):
    def image_to_feature(img):
        img = ee.Image(img)

        return ee.Feature(None, ee.Dictionary({
            "label": label,
            "date": ee.Date(img.get("system:time_start")).format("YYYY-MM-dd HH:mm"),
            "cloudy_pixel_percentage": img.get("CLOUDY_PIXEL_PERCENTAGE"),
            "mgrs_tile": img.get("MGRS_TILE"),
            "system_index": img.get("system:index"),
        }))

    fc = ee.FeatureCollection(
        collection.toList(collection.size()).map(image_to_feature)
    )

    df = _ee_fc_to_df(fc)

    if not df.empty:
        df = df.sort_values(["label", "date"]).reset_index(drop=True)

    return df


def _preprocess_s2_collection(collection):
    bands = ["B2", "B3", "B4", "B8", "B11", "B12"]

    def preprocess(image):
        scl = image.select("SCL")

        clear = (
            scl.neq(0)
            .And(scl.neq(1))
            .And(scl.neq(3))
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))
        )

        reflectance = (
            image
            .select(bands)
            .divide(10000)
            .updateMask(clear)
        )

        ndwi = reflectance.normalizedDifference(["B3", "B8"]).rename("NDWI")
        mndwi = reflectance.normalizedDifference(["B3", "B11"]).rename("MNDWI")
        ndvi = reflectance.normalizedDifference(["B8", "B4"]).rename("NDVI")

        return (
            reflectance
            .addBands([ndwi, mndwi, ndvi])
            .copyProperties(image, ["system:time_start", "system:index"])
        )

    return collection.map(preprocess)


def _valid_pixel_percentage(valid_mask, region, scale=60):
    area_valid = (
        valid_mask
        .unmask(0)
        .gt(0)
        .multiply(ee.Image.pixelArea())
        .rename("valid_area")
    )

    area_total = (
        ee.Image.pixelArea()
        .rename("total_area")
    )

    valid_value = area_valid.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
        tileScale=4,
    ).get("valid_area").getInfo()

    total_value = area_total.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
        tileScale=4,
    ).get("total_area").getInfo()

    if valid_value is None or total_value in [None, 0]:
        return 0.0

    return float(valid_value / total_value * 100)


def _compute_change(post_comp, pre_comp, index_name, event_key):
    return (
        post_comp
        .select(index_name)
        .subtract(pre_comp.select(index_name))
        .rename(f"{event_key}_delta_{index_name}")
    )


def _optical_water_mask(
    composite,
    index_name,
    threshold,
    permanent_water,
    remove_permanent,
    name,
):
    water = composite.select(index_name).gt(threshold)

    if remove_permanent:
        water = water.And(permanent_water.unmask(0).Not())

    return water.rename(name).toByte()


def _create_fusion_map(
    sar_confidence,
    optical_water,
    optical_valid,
    permanent_water,
    name,
):
    sar_flood = sar_confidence.gte(2)
    sar_low = sar_confidence.eq(1)

    optical_water = optical_water.unmask(0).gt(0)
    optical_valid = optical_valid.unmask(0).gt(0)
    permanent_water = permanent_water.unmask(0).gt(0)

    fusion = ee.Image(0).rename(name)

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


def _area_km2(mask, region, scale=10):
    area_image = (
        mask
        .unmask(0)
        .gt(0)
        .multiply(ee.Image.pixelArea())
        .divide(1e6)
        .rename("area_km2")
    )

    value = area_image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
        tileScale=4,
    ).get("area_km2").getInfo()

    return 0.0 if value is None else float(value)


def _agreement_metrics(
    sar_confidence,
    optical_new_water,
    optical_valid,
    region,
    scale=10,
):
    sar = sar_confidence.gte(2).unmask(0).And(optical_valid.unmask(0))
    optical = optical_new_water.unmask(0).gt(0).And(optical_valid.unmask(0))

    intersection = sar.And(optical)
    union = sar.Or(optical)

    sar_area = _area_km2(sar, region, scale=scale)
    optical_area = _area_km2(optical, region, scale=scale)
    intersection_area = _area_km2(intersection, region, scale=scale)
    union_area = _area_km2(union, region, scale=scale)

    iou = intersection_area / union_area if union_area > 0 else 0.0
    dice = (
        2 * intersection_area / (sar_area + optical_area)
        if (sar_area + optical_area) > 0
        else 0.0
    )

    return {
        "sar_area_km2_valid_optical": sar_area,
        "optical_area_km2": optical_area,
        "intersection_km2": intersection_area,
        "union_km2": union_area,
        "iou": iou,
        "dice": dice,
    }

def _make_optical_map(
    event,
    places,
    post_comp,
    optical_valid,
    optical_water,
    optical_new_water,
    sar_confidence,
    fusion_map,
    delta_ndwi,
    delta_mndwi,
    delta_ndvi,
    vis,
    optical_index,
    optical_threshold,
):
    aoi = event["aoi"]
    m = geemap.Map(center=event["center"], zoom=event["zoom"])

    raster_layers = [
        {"image": post_comp, "vis": vis["true_color"], "name": "S2 post-event true color", "shown": True},
        {"image": post_comp, "vis": vis["false_color"], "name": "S2 post-event false color NIR", "shown": False},
        {"image": post_comp.select("NDWI"), "vis": vis["ndwi"], "name": "S2 post-event NDWI", "shown": False},
        {"image": post_comp.select("MNDWI"), "vis": vis["mndwi"], "name": "S2 post-event MNDWI", "shown": False},
        {"image": post_comp.select("NDVI"), "vis": vis["ndvi"], "name": "S2 post-event NDVI", "shown": False},
        {"image": delta_ndwi, "vis": vis["delta_index"], "name": "ΔNDWI = post - pre", "shown": False},
        {"image": delta_mndwi, "vis": vis["delta_index"], "name": "ΔMNDWI = post - pre", "shown": False},
        {"image": delta_ndvi, "vis": vis["delta_index"], "name": "ΔNDVI = post - pre", "shown": False},
        {"image": optical_valid.selfMask(), "vis": vis["optical_valid"], "name": "Optical valid pixels", "shown": False},
        {"image": optical_water.selfMask(), "vis": vis["optical_water"], "name": f"Optical water: {optical_index} > {optical_threshold}", "shown": False},
        {"image": optical_new_water.selfMask(), "vis": vis["optical_water"], "name": "Optical new water without permanent water", "shown": False},
        {"image": sar_confidence.selfMask(), "vis": vis["sar_confidence"], "name": "SAR confidence map", "shown": False},
        {"image": fusion_map.selfMask(), "vis": vis["fusion"], "name": "SAR-optical fusion", "shown": True},
    ]

    us.add_layers(m, raster_layers)
    m.addLayer(ee.FeatureCollection([ee.Feature(aoi)]).style(color=event.get("aoi_color", "red"),fillColor="00000000",width=2,),{},event.get("aoi_label", "AOI"),)
    m.addLayer(places.filterBounds(aoi).style(color="000000",pointSize=5,pointShape="circle",),{},"Places",)
    m.addLayerControl()

    return m


def mask_s2_clouds_scl(image):
    """
    Mask clouds, shadows, cirrus, snow and invalid pixels using Sentinel-2 SCL.

    Keeps:
    4 = vegetation
    5 = bare soil
    6 = water
    7 = unclassified / clear enough
    """
    image = ee.Image(image)

    scl = image.select("SCL")

    clear = (
        scl.eq(4)       # vegetation
        .Or(scl.eq(5))  # bare soil
        .Or(scl.eq(6))  # water
        .Or(scl.eq(7))  # unclassified
    )

    return image.updateMask(clear)


def add_optical_indices(image):
    """
    Add NDWI, MNDWI and NDVI to a Sentinel-2 image.
    Assumes reflectance bands are already scaled to 0-1.
    """
    image = ee.Image(image)

    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = image.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")

    return image.addBands([ndwi, mndwi, ndvi])


def preprocess_s2_collection(collection, apply_cloud_mask=True):
    """
    Scale Sentinel-2 SR reflectance bands and optionally apply SCL cloud masking.

    Important:
    - Optical reflectance bands are divided by 10000.
    - SCL is kept unscaled because it contains integer classification labels.
    """

    optical_bands = [
        "B1", "B2", "B3", "B4", "B5", "B6",
        "B7", "B8", "B8A", "B9", "B11", "B12"
    ]

    def preprocess(image):
        image = ee.Image(image)

        scaled_optical = image.select(optical_bands).divide(10000)
        scl = image.select("SCL")

        out = (
            scaled_optical
            .addBands(scl)
            .copyProperties(image, image.propertyNames())
        )

        if apply_cloud_mask:
            out = mask_s2_clouds_scl(out)

        out = add_optical_indices(out)

        return out

    return collection.map(preprocess)


def make_best_available_s2_composite(
    collection,
    method="median",
):
    """
    Build a cloud-masked Sentinel-2 composite from the available valid pixels.

    method:
    - "median": robust standard multi-temporal composite
    - "mosaic": uses the collection order after sorting by cloud percentage
    """
    method = method.lower()

    if method == "median":
        return collection.median()

    if method == "mosaic":
        return collection.sort("CLOUDY_PIXEL_PERCENTAGE").mosaic()

    raise ValueError("method must be either 'median' or 'mosaic'.")


def load_s2_best_available_composite(
    aoi,
    dates,
    event,
    window_key,
    max_cloud=100,
    apply_cloud_mask=True,
    composite_method="median",
):
    """
    Load Sentinel-2 SR Harmonized images for a selected date window and create
    a best-available optical composite.
    """
    start, end = dates[window_key]

    collection_raw = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
    )

    collection_proc = preprocess_s2_collection(
        collection_raw,
        apply_cloud_mask=apply_cloud_mask,
    )

    composite = make_best_available_s2_composite(
        collection_proc,
        method=composite_method,
    ).clip(aoi)

    return {
        "collection_raw": collection_raw,
        "collection_proc": collection_proc,
        "composite": composite,
        "start": start,
        "end": end,
        "count": collection_raw.size(),
        "window_key": window_key,
        "max_cloud": max_cloud,
        "apply_cloud_mask": apply_cloud_mask,
        "composite_method": composite_method,
    }


def valid_pixel_percentage(image, region, band="B3", scale=20):
    """
    Estimate percentage of valid pixels in a composite over a region.
    Uses the mask of the selected band.
    """
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
    ).get("valid_area")

    total = total_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
        tileScale=4,
    ).get("total_area")

    return ee.Number(valid).divide(ee.Number(total)).multiply(100)


# ============================================================
# PUBLIC OPTICAL PIPELINE
# ============================================================

def run_optical_event(
    event_key,
    event,
    dates,
    sar_confidence,
    permanent_water,
    places,
    output_dir,
    max_cloud_pre=95,
    max_cloud_post=100,
    optical_index="MNDWI",
    optical_threshold=0.0,
    pre_window_override=None,
    post_window_override=None,
    composite_method="median",
    make_map=True,
    vis=None,
):
    if vis is None:
        vis = get_optical_vis_params()

    aoi = event["aoi"]

    pre_window_key = pre_window_override or event["pre"]
    post_window_key = post_window_override or event["post"]

    pre_dates = dates[pre_window_key]
    post_dates = dates[post_window_key]

    # Best-available cloud-masked Sentinel-2 composites.
    pre_data = load_s2_best_available_composite(
        aoi=aoi,
        dates=dates,
        event=event,
        window_key=pre_window_key,
        max_cloud=max_cloud_pre,
        apply_cloud_mask=True,
        composite_method=composite_method,
    )

    post_data = load_s2_best_available_composite(
        aoi=aoi,
        dates=dates,
        event=event,
        window_key=post_window_key,
        max_cloud=max_cloud_post,
        apply_cloud_mask=True,
        composite_method=composite_method,
    )

    pre_comp = pre_data["composite"]
    post_comp = post_data["composite"]

    # Keep collections for metadata and optional inspection.
    s2_pre_raw = pre_data["collection_raw"]
    s2_post_raw = post_data["collection_raw"]
    s2_pre = pre_data["collection_proc"]
    s2_post = post_data["collection_proc"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_s2_metadata = pd.concat(
        [
            _s2_metadata_df(s2_pre_raw, f"{event_key}_pre"),
            _s2_metadata_df(s2_post_raw, f"{event_key}_post"),
        ],
        ignore_index=True,
    )

    if not df_s2_metadata.empty:
        df_s2_metadata = df_s2_metadata.sort_values(["label", "date"]).reset_index(drop=True)

    pre_valid = pre_comp.select("B3").mask().rename(f"{event_key}_pre_valid").toByte()
    post_valid = post_comp.select("B3").mask().rename(f"{event_key}_post_valid").toByte()

    pre_valid_percentage = _valid_pixel_percentage(pre_valid, aoi, scale=60)
    post_valid_percentage = _valid_pixel_percentage(post_valid, aoi, scale=60)

    df_valid_summary = pd.DataFrame([
        {
            "event": event_key,
            "window": "pre",
            "window_key": pre_window_key,
            "start": pre_dates[0],
            "end": pre_dates[1],
            "valid_pixel_percentage_aoi": pre_valid_percentage,
        },
        {
            "event": event_key,
            "window": "post",
            "window_key": post_window_key,
            "start": post_dates[0],
            "end": post_dates[1],
            "valid_pixel_percentage_aoi": post_valid_percentage,
        },
    ])

    delta_ndwi = _compute_change(post_comp, pre_comp, "NDWI", event_key)
    delta_mndwi = _compute_change(post_comp, pre_comp, "MNDWI", event_key)
    delta_ndvi = _compute_change(post_comp, pre_comp, "NDVI", event_key)

    optical_valid = post_valid

    optical_water = _optical_water_mask(
        post_comp,
        index_name=optical_index,
        threshold=optical_threshold,
        permanent_water=permanent_water,
        remove_permanent=False,
        name=f"{event_key}_optical_water",
    )

    optical_new_water = _optical_water_mask(
        post_comp,
        index_name=optical_index,
        threshold=optical_threshold,
        permanent_water=permanent_water,
        remove_permanent=True,
        name=f"{event_key}_optical_new_water",
    )

    fusion_map = _create_fusion_map(
        sar_confidence=sar_confidence,
        optical_water=optical_water,
        optical_valid=optical_valid,
        permanent_water=permanent_water,
        name=f"{event_key}_sar_optical_fusion",
    )

    metrics = _agreement_metrics(
        sar_confidence=sar_confidence,
        optical_new_water=optical_new_water,
        optical_valid=optical_valid,
        region=aoi,
        scale=10,
    )

    metrics["event"] = event_key
    metrics["optical_index"] = optical_index
    metrics["optical_threshold"] = optical_threshold
    metrics["pre_window_key"] = pre_window_key
    metrics["post_window_key"] = post_window_key
    metrics["composite_method"] = composite_method

    df_agreement = pd.DataFrame([metrics])

    df_s2_metadata.to_csv(output_dir / f"{event_key}_s2_cloud_validity_audit.csv", index=False)
    df_valid_summary.to_csv(output_dir / f"{event_key}_s2_valid_pixel_summary.csv", index=False)
    df_agreement.to_csv(output_dir / f"{event_key}_sar_optical_agreement.csv", index=False)

    optical_map = None

    if make_map:
        optical_map = _make_optical_map(
            event=event,
            places=places,
            post_comp=post_comp,
            optical_valid=optical_valid,
            optical_water=optical_water,
            optical_new_water=optical_new_water,
            sar_confidence=sar_confidence,
            fusion_map=fusion_map,
            delta_ndwi=delta_ndwi,
            delta_mndwi=delta_mndwi,
            delta_ndvi=delta_ndvi,
            vis=vis,
            optical_index=optical_index,
            optical_threshold=optical_threshold,
        )

    return {
        "df_s2_metadata": df_s2_metadata,
        "df_valid_summary": df_valid_summary,
        "df_agreement": df_agreement,
        "s2_pre_raw": s2_pre_raw,
        "s2_post_raw": s2_post_raw,
        "s2_pre": s2_pre,
        "s2_post": s2_post,
        "pre_comp": pre_comp,
        "post_comp": post_comp,
        "optical_valid": optical_valid,
        "optical_water": optical_water,
        "optical_new_water": optical_new_water,
        "delta_ndwi": delta_ndwi,
        "delta_mndwi": delta_mndwi,
        "delta_ndvi": delta_ndvi,
        "fusion_map": fusion_map,
        "metrics": metrics,
        "map": optical_map,
        "pre_window_key": pre_window_key,
        "post_window_key": post_window_key,
        "pre_window": (pre_data["start"], pre_data["end"]),
        "post_window": (post_data["start"], post_data["end"]),
        "pre_image_count": pre_data["count"],
        "post_image_count": post_data["count"],
        "composite_method": composite_method,
    }
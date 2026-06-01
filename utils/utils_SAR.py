import ee
import geemap
import pandas as pd
import numpy as np
from pathlib import Path
from utils import utils_setup as us


# =============================================================================
# Small helpers
# =============================================================================

def otsu_threshold(image, region, band, scale=30, max_buckets=128):
    """Compute an Otsu threshold from the histogram of one image band."""
    hist = image.select(band).reduceRegion(
        reducer=ee.Reducer.histogram(maxBuckets=max_buckets),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
        tileScale=4,
    ).get(band).getInfo()

    if hist is None:
        return None

    counts = np.array(hist["histogram"], dtype=float)
    means = np.array(hist["bucketMeans"], dtype=float)

    if counts.sum() == 0 or len(means) < 2:
        return None

    total = counts.sum()
    total_sum = (counts * means).sum()
    total_mean = total_sum / total

    scores = []

    for i in range(1, len(means)):
        count_0 = counts[:i].sum()
        count_1 = total - count_0

        if count_0 == 0 or count_1 == 0:
            scores.append(-np.inf)
            continue

        sum_0 = (counts[:i] * means[:i]).sum()
        mean_0 = sum_0 / count_0
        mean_1 = (total_sum - sum_0) / count_1

        scores.append(
            count_0 * (mean_0 - total_mean) ** 2
            + count_1 * (mean_1 - total_mean) ** 2
        )

    return float(means[1:][int(np.argmax(scores))])

def clean_mask(mask, permanent_water, slope, slope_threshold=5, min_connected_pixels=10):
    """Remove permanent water, steep terrain and very small isolated patches."""
    clean = (
        mask.unmask(0)
        .gt(0)
        .And(permanent_water.unmask(0).Not())
        .And(slope.lt(slope_threshold))
    )

    if min_connected_pixels and min_connected_pixels > 1:
        connected = (
            clean.selfMask()
            .connectedPixelCount(maxSize=100, eightConnected=True)
            .unmask(0)
        )
        clean = clean.And(connected.gte(min_connected_pixels))

    return clean.toByte()


# =============================================================================
# SAR products
# =============================================================================

def make_sar_inputs(event_key, s1, aoi, speckle_radius=30):
    """Create VV/VH composites and change products."""
    pre_vv = us.median_composite(s1["pre"], band="VV", aoi=aoi)
    post_vv = us.median_composite(s1["post"], band="VV", aoi=aoi)

    pre_vh = us.median_composite(s1["pre"], band="VH", aoi=aoi)
    post_vh = us.median_composite(s1["post"], band="VH", aoi=aoi)

    if speckle_radius is not None and speckle_radius > 0:
        pre_vv = pre_vv.focal_median(speckle_radius, "circle", "meters")
        post_vv = post_vv.focal_median(speckle_radius, "circle", "meters")
        pre_vh = pre_vh.focal_median(speckle_radius, "circle", "meters")
        post_vh = post_vh.focal_median(speckle_radius, "circle", "meters")

    pre_vv = pre_vv.rename(f"{event_key}_pre_VV")
    post_vv = post_vv.rename(f"{event_key}_post_VV")
    pre_vh = pre_vh.rename(f"{event_key}_pre_VH")
    post_vh = post_vh.rename(f"{event_key}_post_VH")

    delta_vv = post_vv.subtract(pre_vv).rename(f"{event_key}_delta_VV")
    delta_vh = post_vh.subtract(pre_vh).rename(f"{event_key}_delta_VH")

    ratio_vv = (
        ee.Image(10).pow(post_vv.divide(10))
        .divide(ee.Image(10).pow(pre_vv.divide(10)))
        .rename(f"{event_key}_ratio_VV")
    )

    ratio_vh = (
        ee.Image(10).pow(post_vh.divide(10))
        .divide(ee.Image(10).pow(pre_vh.divide(10)))
        .rename(f"{event_key}_ratio_VH")
    )

    return {
        "pre_vv": pre_vv,
        "post_vv": post_vv,
        "pre_vh": pre_vh,
        "post_vh": post_vh,
        "delta_vv": delta_vv,
        "delta_vh": delta_vh,
        "ratio_vv": ratio_vv,
        "ratio_vh": ratio_vh,
    }

def make_flood_masks(
    event_key,
    sar,
    permanent_water,
    slope,
    aoi,
    delta_threshold=-2.5,
    ratio_threshold=0.7,
    vv_post_threshold=-17,
    vh_delta_threshold=-1.5,
    vh_ratio_threshold=0.8,
    slope_threshold=5,
    min_connected_pixels=10,
    confidence_threshold=3,
    use_otsu_in_confidence=True,
):
    """
    Create VV, VH and Otsu flood candidates, then combine them into confidence.

    VH is used as a supporting signal only when both VH decrease and VH ratio agree.
    Otsu is computed only over physically plausible negative-change areas.
    """
    flood_delta_vv = clean_mask(
        sar["delta_vv"].lt(delta_threshold),
        permanent_water,
        slope,
        slope_threshold,
        min_connected_pixels,
    ).rename(f"{event_key}_flood_delta_VV")

    flood_ratio_vv = clean_mask(
        sar["ratio_vv"].lt(ratio_threshold),
        permanent_water,
        slope,
        slope_threshold,
        min_connected_pixels,
    ).rename(f"{event_key}_flood_ratio_VV")

    flood_low_vv = clean_mask(
        sar["post_vv"].lt(vv_post_threshold),
        permanent_water,
        slope,
        slope_threshold,
        min_connected_pixels,
    ).rename(f"{event_key}_flood_low_post_VV")

    flood_delta_vh = clean_mask(
        sar["delta_vh"].lt(vh_delta_threshold),
        permanent_water,
        slope,
        slope_threshold,
        min_connected_pixels,
    ).rename(f"{event_key}_flood_delta_VH")

    flood_ratio_vh = clean_mask(
        sar["ratio_vh"].lt(vh_ratio_threshold),
        permanent_water,
        slope,
        slope_threshold,
        min_connected_pixels,
    ).rename(f"{event_key}_flood_ratio_VH")

    vv_evidence = (
        flood_delta_vv
        .Or(flood_ratio_vv)
        .Or(flood_low_vv)
    )

    flood_vh_support = (
        flood_delta_vh
        .And(flood_ratio_vh)
        .And(vv_evidence)
        .rename(f"{event_key}_flood_VH_support")
        .toByte()
    )

    otsu_domain = (
        sar["delta_vv"]
        .updateMask(permanent_water.unmask(0).Not())
        .updateMask(slope.lt(slope_threshold))
        .updateMask(sar["delta_vv"].lt(0))
    )

    otsu_value = otsu_threshold(
        image=otsu_domain,
        region=aoi,
        band=f"{event_key}_delta_VV",
    )

    otsu_valid = bool(otsu_value is not None and otsu_value < 0)

    if otsu_value is None:
        flood_otsu = ee.Image(0).rename(f"{event_key}_flood_otsu_VV").toByte()
    else:
        flood_otsu = clean_mask(
            sar["delta_vv"].lt(otsu_value).And(sar["post_vv"].lt(vv_post_threshold)),
            permanent_water,
            slope,
            slope_threshold,
            min_connected_pixels,
        ).rename(f"{event_key}_flood_otsu_VV")

    sar_confidence = (
        flood_delta_vv.unmask(0)
        .add(flood_ratio_vv.unmask(0))
        .add(flood_low_vv.unmask(0))
        .add(flood_vh_support.unmask(0))
    )

    if use_otsu_in_confidence and otsu_valid:
        sar_confidence = sar_confidence.add(flood_otsu.unmask(0))

    sar_confidence = sar_confidence.rename(f"{event_key}_sar_confidence").toByte()

    sar_flood = (
        sar_confidence
        .gte(confidence_threshold)
        .rename(f"{event_key}_sar_flood")
        .toByte()
    )

    return {
        "flood_delta_vv": flood_delta_vv,
        "flood_ratio_vv": flood_ratio_vv,
        "flood_low_vv": flood_low_vv,
        "flood_delta_vh": flood_delta_vh,
        "flood_ratio_vh": flood_ratio_vh,
        "flood_vh_support": flood_vh_support,
        "flood_otsu": flood_otsu,
        "sar_confidence": sar_confidence,
        "sar_flood": sar_flood,
        "otsu_threshold": otsu_value,
        "otsu_valid": otsu_valid,
    }

def make_area_table(event_key, masks, aoi, confidence_threshold=3):
    """Create a compact area table for SAR flood candidates."""
    rows = [
        ("delta_VV", masks["flood_delta_vv"]),
        ("ratio_VV", masks["flood_ratio_vv"]),
        ("low_post_event_VV", masks["flood_low_vv"]),
        ("delta_VH", masks["flood_delta_vh"]),
        ("ratio_VH", masks["flood_ratio_vh"]),
        ("VH_support_delta_AND_ratio", masks["flood_vh_support"]),
        ("otsu_delta_VV", masks["flood_otsu"]),
        (f"SAR_confidence_gte_{confidence_threshold}", masks["sar_flood"]),
    ]

    return pd.DataFrame([
        {
            "event": event_key,
            "method": method,
            "area_km2": us.mask_area_km2(mask, aoi),
        }
        for method, mask in rows
    ])


# =============================================================================
# Map
# =============================================================================

def make_sar_map(event, places, sar, masks):
    """Create a simple interactive SAR map."""
    aoi = event["aoi"]
    vis = us.get_vis_params()

    vh_vis = {"min": -30, "max": -5}
    ratio_vis = {"min": 0.3, "max": 1.5, "palette": ["blue", "white", "red"]}
    confidence_vis = {
        "min": 1,
        "max": 6,
        "palette": ["yellow", "orange", "red"],
    }

    m = geemap.Map(center=event["center"], zoom=event["zoom"])

    layers = [
        (sar["pre_vv"], vis["s1_vv"], "S1 VV pre-event", False),
        (sar["post_vv"], vis["s1_vv"], "S1 VV post-event", True),
        (sar["pre_vh"], vh_vis, "S1 VH pre-event", False),
        (sar["post_vh"], vh_vis, "S1 VH post-event", False),

        (sar["delta_vv"], vis["s1_change"], "ΔVV = post - pre", False),
        (sar["delta_vh"], vis["s1_change"], "ΔVH = post - pre", False),
        (sar["ratio_vv"], ratio_vis, "VV ratio = post/pre", False),
        (sar["ratio_vh"], ratio_vis, "VH ratio = post/pre", False),

        (masks["flood_delta_vv"].selfMask(), vis["flood"], "Flood candidate: ΔVV", False),
        (masks["flood_ratio_vv"].selfMask(), vis["flood"], "Flood candidate: VV ratio", False),
        (masks["flood_low_vv"].selfMask(), vis["flood"], "Flood candidate: low post VV", False),
        (masks["flood_delta_vh"].selfMask(), vis["flood"], "Flood candidate: ΔVH", False),
        (masks["flood_ratio_vh"].selfMask(), vis["flood"], "Flood candidate: VH ratio", False),
        (masks["flood_otsu"].selfMask(), vis["flood"], "Flood candidate: Otsu ΔVV", False),

        (masks["sar_confidence"].selfMask(), confidence_vis, "SAR confidence", False),
        (masks["sar_flood"].selfMask(), vis["flood"], "Final SAR flood mask", True),
    ]

    for image, layer_vis, name, shown in layers:
        m.addLayer(image, layer_vis, name, shown)

    m.addLayer(
        us.styled_aoi(aoi, color=event.get("aoi_color", "red")),
        {},
        event.get("aoi_label", "AOI"),
    )

    if places is not None:
        m.addLayer(us.styled_places(places, aoi), {}, "Places")

    m.addLayerControl()
    return m


# =============================================================================
# Public pipeline
# =============================================================================

def run_sar_event(
    event_key,
    event,
    s1,
    permanent_water,
    slope,
    places=None,
    output_dir=None,
    make_map=True,
    confidence_threshold=3,
    orbit_pass_override=None,
    relative_orbit_override=None,
    **params,
):
    """
    Run the compact Sentinel-1 SAR flood workflow.

    orbit_pass_override and relative_orbit_override are accepted here only
    because they come from cfg.SAR_PARAMS, but the actual orbit filtering has
    already been applied in the setup section when loading Sentinel-1.
    """
    aoi = event["aoi"]

    speckle_radius = params.pop("speckle_radius", 30)

    sar = make_sar_inputs(
        event_key=event_key,
        s1=s1,
        aoi=aoi,
        speckle_radius=speckle_radius,
    )
    masks = make_flood_masks(
        event_key=event_key,
        sar=sar,
        permanent_water=permanent_water,
        slope=slope,
        aoi=aoi,
        confidence_threshold=confidence_threshold,
        **params,
    )

    df_area = make_area_table(
        event_key=event_key,
        masks=masks,
        aoi=aoi,
        confidence_threshold=confidence_threshold,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df_area.to_csv(
            output_dir / f"{event_key}_sar_area_summary.csv",
            index=False,
        )

    sar_map = make_sar_map(event, places, sar, masks) if make_map else None

    return {
        "sar": sar,
        "masks": masks,
        "df_area": df_area,
        "map": sar_map,

        **sar,
        **masks,
    }


# =============================================================================
# SAR notebook summaries and figures
# =============================================================================

def sar_processing_df(month, s1, sar_params, sar_results):
    """Compact vertical summary of the SAR processing configuration."""
    otsu = sar_results["otsu_threshold"]

    rows = [
        ("Orbit pass", sar_params.get("orbit_pass_override")),
        ("Relative orbit", sar_params.get("relative_orbit_override")),
        ("Pre-event S1 images", s1["pre"].size().getInfo()),
        ("Post-event S1 images", s1["post"].size().getInfo()),
        ("Speckle radius", f"{sar_params.get('speckle_radius', 30)} m"),
        ("Slope threshold", f"{sar_params.get('slope_threshold')}°"),
        ("Minimum connected pixels", sar_params.get("min_connected_pixels")),
        ("Confidence threshold", sar_params.get("confidence_threshold", 3)),
        ("Otsu ΔVV threshold", None if otsu is None else round(otsu, 3)),
        ("Otsu valid", sar_results["otsu_valid"]),
    ]

    return pd.DataFrame(rows, columns=["parameter", "value"])

def vv_score(sar_results):
    """Return the number of VV-based flood indicators active per pixel."""
    return (
        sar_results["flood_delta_vv"].unmask(0)
        .add(sar_results["flood_ratio_vv"].unmask(0))
        .add(sar_results["flood_low_vv"].unmask(0))
        .rename("vv_score")
    )

def sar_consistency_df(sar_results, aoi):
    """Check whether the final SAR flood mask is supported by VV evidence."""
    vv_score_gte_2 = vv_score(sar_results).gte(2)
    final_flood = sar_results["sar_flood"]

    rows = [
        ("VV_score >= 2", vv_score_gte_2),
        ("Final SAR flood", final_flood),
        ("Final flood supported by VV_score >= 2", final_flood.And(vv_score_gte_2)),
        (
            "Final flood NOT supported by VV_score >= 2",
            final_flood.And(vv_score_gte_2.unmask(0).Not()),
        ),
    ]

    df = pd.DataFrame([
        {
            "metric": name,
            "area_km2": round(us.mask_area_km2(mask, aoi), 2),
        }
        for name, mask in rows
    ])

    return df

def export_delta_vv_png(sar_results, region, output_dir, month, dimensions=900):
    """Export standalone ΔVV diagnostic figure."""
    vis = us.get_vis_params()

    return us.export_single_ee_image_png(
        image=sar_results["delta_vv"],
        region=region,
        vis_params=vis["s1_change"],
        output_path=Path(output_dir) / f"{month}_sar_delta_vv.png",
        title="SAR change evidence: ΔVV = VV post-event − VV pre-event",
        dimensions=dimensions,
    )

def export_vv_support_vs_final_png(sar_results, region, output_dir, month, dimensions=900):
    """
    Export a clearer comparison between:
    - VV-supported candidate areas that were discarded by the final SAR mask
    - Final SAR flood mask retained by the workflow
    """
    vis = us.get_vis_params()

    post_vv_rgb = sar_results["post_vv"].visualize(**vis["s1_vv"])

    # Strong VV-based support
    vv_support = vv_score(sar_results).gte(2)

    # Final SAR flood mask
    final_flood = sar_results["sar_flood"]

    # Areas supported by VV but NOT retained in the final result
    vv_only_not_final = vv_support.And(final_flood.Not())

    # Left overlay: discarded VV-supported candidates
    vv_only_overlay = vv_only_not_final.selfMask().visualize(
        min=0,
        max=1,
        palette=["ff9900"],   # orange
        opacity=0.9,
    )

    # Right overlay: final confirmed flood
    final_flood_overlay = final_flood.selfMask().visualize(
        min=0,
        max=1,
        palette=["00ffff"],   # cyan
        opacity=0.9,
    )

    return us.export_side_by_side_ee_images(
        left_image=post_vv_rgb.blend(vv_only_overlay),
        right_image=post_vv_rgb.blend(final_flood_overlay),
        region=region,
        vis_params={},
        output_path=Path(output_dir) / f"{month}_vv_support_vs_final_sar_flood.png",
        left_title="VV-supported candidates discarded by the final SAR mask",
        right_title="Final confirmed SAR flood mask",
        dimensions=dimensions,
    )
import ee
import geemap
import pandas as pd
import numpy as np
from pathlib import Path

from utils import utils_setup as us


# ============================================================
# SAR VISUALIZATION
# ============================================================

def get_sar_vis_params():
    return {
        "s1_vv": {
            "min": -25,
            "max": 0,
        },
        "delta_vv": {
            "min": -5,
            "max": 5,
            "palette": ["0000ff", "ffffff", "ff0000"],
        },
        "ratio": {
            "min": 0.3,
            "max": 1.5,
            "palette": ["0000ff", "ffffff", "ff0000"],
        },
        "flood": {
            "min": 0,
            "max": 1,
            "palette": ["ffffff", "0000ff"],
        },
        "confidence": {
            "min": 0,
            "max": 3,
            "palette": ["ffffff", "ffff00", "ff9900", "ff0000"],
        },
        "permanent_water": {
            "min": 0,
            "max": 1,
            "palette": ["ffffff", "0000ff"],
        },
        "slope": {
            "min": 0,
            "max": 20,
            "palette": ["ffffff", "ffff00", "ff9900", "ff0000"],
        },
        "rivers": {
            "min": 0,
            "max": 1,
            "palette": ["ffffff", "0000ff"],
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


def _s1_metadata_df(collection, aoi, label, scale=1000):
    def image_to_feature(img):
        img = ee.Image(img)

        mean_angle = img.select("angle").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True,
            tileScale=4,
        ).get("angle")

        return ee.Feature(None, ee.Dictionary({
            "label": label,
            "date": ee.Date(img.get("system:time_start")).format("YYYY-MM-dd HH:mm"),
            "orbit_pass": img.get("orbitProperties_pass"),
            "relative_orbit": img.get("relativeOrbitNumber_start"),
            "polarizations": ee.List(img.get("transmitterReceiverPolarisation")).join("/"),
            "mean_incidence_angle": mean_angle,
            "resolution_meters": img.get("resolution_meters"),
            "system_index": img.get("system:index"),
        }))

    fc = ee.FeatureCollection(
        collection.toList(collection.size()).map(image_to_feature)
    )

    df = _ee_fc_to_df(fc)

    if not df.empty:
        df = df.sort_values(["label", "date"]).reset_index(drop=True)

    return df


def _s1_group_candidates(df_pre, df_post):
    if df_pre.empty or df_post.empty:
        return pd.DataFrame()

    pre_counts = (
        df_pre
        .groupby(["orbit_pass", "relative_orbit"])
        .size()
        .reset_index(name="pre_count")
    )

    post_counts = (
        df_post
        .groupby(["orbit_pass", "relative_orbit"])
        .size()
        .reset_index(name="post_count")
    )

    candidates = pd.merge(
        pre_counts,
        post_counts,
        on=["orbit_pass", "relative_orbit"],
        how="inner",
    )

    if candidates.empty:
        return candidates

    candidates["balanced_count"] = candidates[["pre_count", "post_count"]].min(axis=1)
    candidates["total_count"] = candidates["pre_count"] + candidates["post_count"]

    return candidates.sort_values(
        ["balanced_count", "total_count"],
        ascending=False,
    ).reset_index(drop=True)


def _select_s1_geometry(
    aoi,
    pre_dates,
    post_dates,
    df_candidates,
    orbit_pass_override=None,
    relative_orbit_override=None,
):
    if orbit_pass_override is not None and relative_orbit_override is not None:
        selected = {
            "selection_mode": "manual_override",
            "orbit_pass": orbit_pass_override,
            "relative_orbit": int(relative_orbit_override),
        }

        s1_pre = us.load_sentinel1(
            aoi,
            *pre_dates,
            orbit_pass=orbit_pass_override,
            relative_orbit=relative_orbit_override,
        )

        s1_post = us.load_sentinel1(
            aoi,
            *post_dates,
            orbit_pass=orbit_pass_override,
            relative_orbit=relative_orbit_override,
        )

        return s1_pre, s1_post, selected

    if df_candidates.empty:
        selected = {
            "selection_mode": "all_images_fallback",
            "orbit_pass": "ALL",
            "relative_orbit": "ALL",
        }

        return (
            us.load_sentinel1(aoi, *pre_dates),
            us.load_sentinel1(aoi, *post_dates),
            selected,
        )

    best = df_candidates.iloc[0]

    selected = {
        "selection_mode": "automatic_best_group",
        "orbit_pass": best["orbit_pass"],
        "relative_orbit": int(best["relative_orbit"]),
    }

    s1_pre = us.load_sentinel1(
        aoi,
        *pre_dates,
        orbit_pass=selected["orbit_pass"],
        relative_orbit=selected["relative_orbit"],
    )

    s1_post = us.load_sentinel1(
        aoi,
        *post_dates,
        orbit_pass=selected["orbit_pass"],
        relative_orbit=selected["relative_orbit"],
    )

    return s1_pre, s1_post, selected


def _otsu_threshold(image, region, band, scale=30, max_buckets=128):
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

    if counts.sum() == 0:
        return None

    total = counts.sum()
    total_sum = (counts * means).sum()
    total_mean = total_sum / total

    thresholds = means[1:]
    bss_values = []

    for i in range(1, len(means)):
        count_0 = counts[:i].sum()
        count_1 = total - count_0

        if count_0 == 0 or count_1 == 0:
            bss_values.append(-np.inf)
            continue

        sum_0 = (counts[:i] * means[:i]).sum()
        mean_0 = sum_0 / count_0
        mean_1 = (total_sum - sum_0) / count_1

        bss_values.append(
            count_0 * (mean_0 - total_mean) ** 2
            + count_1 * (mean_1 - total_mean) ** 2
        )

    if len(bss_values) == 0:
        return None

    return float(thresholds[int(np.argmax(bss_values))])


def _clean_mask(mask, permanent_water, slope, slope_threshold, min_connected_pixels, name):
    candidate = mask.unmask(0).gt(0)
    not_permanent_water = permanent_water.unmask(0).Not()
    low_slope = slope.lt(slope_threshold).unmask(0)

    flood = candidate.And(not_permanent_water).And(low_slope)

    connected = flood.selfMask().connectedPixelCount(
        maxSize=100,
        eightConnected=True,
    )

    return flood.And(connected.gte(min_connected_pixels)).rename(name).toByte()


def _confidence_from_masks(mask_list, name):
    score = ee.Image(0)

    for mask in mask_list:
        score = score.add(mask.unmask(0).gt(0))

    return (
        ee.Image(0)
        .where(score.eq(1), 1)
        .where(score.eq(2), 2)
        .where(score.gte(3), 3)
        .rename(name)
        .toByte()
    )


def _area_km2(mask, region, scale=10):
    area_image = (
        mask.unmask(0)
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

def _make_sar_map(
    event,
    places,
    rivers_image,
    permanent_water,
    slope,
    vis,
    pre_vv,
    post_vv,
    delta_vv,
    ratio_vv,
    flood_delta,
    flood_ratio,
    flood_vv_post,
    flood_otsu,
    sar_confidence,
    delta_threshold,
    ratio_threshold,
    vv_post_threshold,
):
    aoi = event["aoi"]

    m = geemap.Map(center=event["center"], zoom=event["zoom"])

    raster_layers = [
        {"image": pre_vv, "vis": vis["s1_vv"], "name": "S1 VV pre-event", "shown": False},
        {"image": post_vv, "vis": vis["s1_vv"], "name": "S1 VV post-event", "shown": True},
        {"image": delta_vv, "vis": vis["delta_vv"], "name": "ΔVV = post - pre", "shown": False},
        {"image": ratio_vv, "vis": vis["ratio"], "name": "VV ratio = post/pre", "shown": False},
        {"image": flood_delta.selfMask(), "vis": vis["flood"], "name": f"Flood ΔVV < {delta_threshold} dB", "shown": False},
        {"image": flood_ratio.selfMask(), "vis": vis["flood"], "name": f"Flood ratio < {ratio_threshold}", "shown": False},
        {"image": flood_vv_post.selfMask(), "vis": vis["flood"], "name": f"Flood VV post < {vv_post_threshold} dB", "shown": False},
        {"image": flood_otsu.selfMask(), "vis": vis["flood"], "name": "Flood Otsu diagnostic only", "shown": False},
        {"image": sar_confidence.selfMask(), "vis": vis["confidence"], "name": "SAR confidence map", "shown": True},
        {"image": permanent_water.clip(aoi), "vis": vis["permanent_water"], "name": "Permanent Water > 80%", "shown": False},
        {"image": slope.clip(aoi), "vis": vis["slope"], "name": "Slope", "shown": False},
    ]

    us.add_layers(m, raster_layers)


    m.addLayer(rivers_image.clip(aoi), vis["rivers"], "HydroSHEDS rivers", False)
    m.addLayer(ee.FeatureCollection([ee.Feature(aoi)]).style(color=event["aoi_color"],fillColor="00000000",width=2,),{},event["aoi_label"],)
    m.addLayer(places.filterBounds(aoi).style(color="000000",pointSize=5,pointShape="circle",),{},"Places",)
    m.addLayerControl()

    return m


def make_plain_s1_pre_post(
    aoi,
    dates,
    event,
    load_sentinel1_fn,
    orbit_pass=None,
    relative_orbit=None,
    band="VV",
):
    """
    Load Sentinel-1 pre/post collections and create plain median composites
    for dataset visualization.

    No flood-specific processing is applied here.
    """

    s1_pre = load_sentinel1_fn(
        aoi,
        *dates[event["pre"]],
        orbit_pass=orbit_pass,
        relative_orbit=relative_orbit,
    )

    s1_post = load_sentinel1_fn(
        aoi,
        *dates[event["post"]],
        orbit_pass=orbit_pass,
        relative_orbit=relative_orbit,
    )

    pre_img = (
        s1_pre
        .select(band)
        .median()
        .clip(aoi)
        .rename(f"pre_{band}_plain")
    )

    post_img = (
        s1_post
        .select(band)
        .median()
        .clip(aoi)
        .rename(f"post_{band}_plain")
    )

    return {
        "s1_pre": s1_pre,
        "s1_post": s1_post,
        "pre_img": pre_img,
        "post_img": post_img,
        "pre_count": s1_pre.size(),
        "post_count": s1_post.size(),
        "band": band,
        "orbit_pass": orbit_pass,
        "relative_orbit": relative_orbit,
    }


def mask_coverage_percent(image, region, band="VV", scale=100):
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


def audit_s1_candidate_coverage(
    load_sentinel1_fn,
    event,
    dates,
    candidates_df,
    display_aoi,
    scale=100,
):
    rows = []

    for _, row in candidates_df.iterrows():
        orbit_pass = row["orbit_pass"]
        relative_orbit = int(row["relative_orbit"])

        pre_dates = dates[event["pre"]]
        post_dates = dates[event["post"]]

        s1_pre = load_sentinel1_fn(
            event["aoi"],
            *pre_dates,
            orbit_pass=orbit_pass,
            relative_orbit=relative_orbit,
        )

        s1_post = load_sentinel1_fn(
            event["aoi"],
            *post_dates,
            orbit_pass=orbit_pass,
            relative_orbit=relative_orbit,
        )

        pre_img = s1_pre.select("VV").median()
        post_img = s1_post.select("VV").median()

        rows.append({
            "orbit_pass": orbit_pass,
            "relative_orbit": relative_orbit,
            "pre_count": s1_pre.size().getInfo(),
            "post_count": s1_post.size().getInfo(),
            "pre_coverage_analysis_%": mask_coverage_percent(pre_img, event["aoi"], scale=scale).getInfo(),
            "post_coverage_analysis_%": mask_coverage_percent(post_img, event["aoi"], scale=scale).getInfo(),
            "pre_coverage_display_%": mask_coverage_percent(pre_img, display_aoi, scale=scale).getInfo(),
            "post_coverage_display_%": mask_coverage_percent(post_img, display_aoi, scale=scale).getInfo(),
        })

    return pd.DataFrame(rows)


def build_sar_pre_post_filtered_png_images(
    pre_vv,
    post_vv,
    permanent_water=None,
    slope=None,
    flood_delta=None,
    flood_ratio=None,
    final_flood_mask=None,
    sar_confidence=None,
    slope_threshold=5,
    apply_permanent_water_mask=False,
    apply_slope_mask=False,
    show_flood_delta_overlay=False,
    show_flood_ratio_overlay=False,
    show_final_flood_overlay=True,
    show_confidence_overlay=False,
    overlay_on_post_only=True,
    sar_vis=None,
):
    """
    Build pre/post Sentinel-1 VV RGB images for static PNG export.

    The base images are the plain VV backscatter images.
    Optional masks can hide permanent water or steep areas.
    Optional overlays can show flood candidates or SAR confidence.
    """

    if sar_vis is None:
        sar_vis = {
            "min": -25,
            "max": 0,
        }

    valid_mask = ee.Image(1)

    if apply_permanent_water_mask and permanent_water is not None:
        valid_mask = valid_mask.And(permanent_water.unmask(0).Not())

    if apply_slope_mask and slope is not None:
        valid_mask = valid_mask.And(slope.lt(slope_threshold).unmask(0))

    pre_base = pre_vv.updateMask(valid_mask).visualize(**sar_vis)
    post_base = post_vv.updateMask(valid_mask).visualize(**sar_vis)

    left_img = pre_base
    right_img = post_base

    def add_overlay(base_img, mask, palette, opacity=0.65, min_value=0, max_value=1):
        if mask is None:
            return base_img

        overlay = (
            mask
            .selfMask()
            .visualize(
                min=min_value,
                max=max_value,
                palette=palette,
                opacity=opacity,
            )
        )

        return base_img.blend(overlay)

    # Blue = flood candidates from ΔVV
    if show_flood_delta_overlay and flood_delta is not None:
        if overlay_on_post_only:
            right_img = add_overlay(
                right_img,
                flood_delta,
                palette=["0000ff"],
                opacity=0.65,
            )
        else:
            left_img = add_overlay(left_img, flood_delta, palette=["0000ff"], opacity=0.65)
            right_img = add_overlay(right_img, flood_delta, palette=["0000ff"], opacity=0.65)

    # Cyan = flood candidates from ratio
    if show_flood_ratio_overlay and flood_ratio is not None:
        if overlay_on_post_only:
            right_img = add_overlay(
                right_img,
                flood_ratio,
                palette=["00ffff"],
                opacity=0.55,
            )
        else:
            left_img = add_overlay(left_img, flood_ratio, palette=["00ffff"], opacity=0.55)
            right_img = add_overlay(right_img, flood_ratio, palette=["00ffff"], opacity=0.55)

    # Blue = final selected flood mask, usually SAR confidence >= 2
    if show_final_flood_overlay and final_flood_mask is not None:
        if overlay_on_post_only:
            right_img = add_overlay(
                right_img,
                final_flood_mask,
                palette=["0000ff"],
                opacity=0.70,
            )
        else:
            left_img = add_overlay(left_img, final_flood_mask, palette=["0000ff"], opacity=0.70)
            right_img = add_overlay(right_img, final_flood_mask, palette=["0000ff"], opacity=0.70)

    # Yellow/orange/red = SAR confidence levels
    if show_confidence_overlay and sar_confidence is not None:
        if overlay_on_post_only:
            right_img = add_overlay(
                right_img,
                sar_confidence,
                palette=["ffff00", "ff9900", "ff0000"],
                opacity=0.70,
                min_value=1,
                max_value=3,
            )
        else:
            left_img = add_overlay(
                left_img,
                sar_confidence,
                palette=["ffff00", "ff9900", "ff0000"],
                opacity=0.70,
                min_value=1,
                max_value=3,
            )
            right_img = add_overlay(
                right_img,
                sar_confidence,
                palette=["ffff00", "ff9900", "ff0000"],
                opacity=0.70,
                min_value=1,
                max_value=3,
            )

    return {
        "left_image": left_img,
        "right_image": right_img,
    }


# ============================================================
# PUBLIC SAR PIPELINE
# ============================================================

def run_sar_event(
    event_key,
    event,
    dates,
    permanent_water,
    slope,
    rivers_image,
    places,
    vis,
    output_dir,
    make_map=True,
    delta_threshold=-2.5,
    ratio_threshold=0.7,
    vv_post_threshold=-17,
    slope_threshold=5,
    min_connected_pixels=10,
    orbit_pass_override=None,
    relative_orbit_override=None,
    use_otsu_in_confidence=False,
):
    aoi = event["aoi"]
    pre_dates = dates[event["pre"]]
    post_dates = dates[event["post"]]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    s1_pre_all = us.load_sentinel1(aoi, *pre_dates)
    s1_post_all = us.load_sentinel1(aoi, *post_dates)

    df_pre = _s1_metadata_df(s1_pre_all, aoi, f"{event_key}_pre")
    df_post = _s1_metadata_df(s1_post_all, aoi, f"{event_key}_post")

    df_metadata = pd.concat([df_pre, df_post], ignore_index=True)
    if not df_metadata.empty:
        df_metadata = df_metadata.sort_values(["label", "date"]).reset_index(drop=True)

    df_candidates = _s1_group_candidates(df_pre, df_post)

    s1_pre, s1_post, selected = _select_s1_geometry(
        aoi=aoi,
        pre_dates=pre_dates,
        post_dates=post_dates,
        df_candidates=df_candidates,
        orbit_pass_override=orbit_pass_override,
        relative_orbit_override=relative_orbit_override,
    )

    selected["pre_image_count"] = s1_pre.size().getInfo()
    selected["post_image_count"] = s1_post.size().getInfo()

    
    pre_vv = s1_pre.select("VV").median().clip(aoi).rename(f"{event_key}_pre_VV")
    post_vv = s1_post.select("VV").median().clip(aoi).rename(f"{event_key}_post_VV")
    pre_vh = s1_pre.select("VH").median().clip(aoi).rename(f"{event_key}_pre_VH")
    post_vh = s1_post.select("VH").median().clip(aoi).rename(f"{event_key}_post_VH")

    delta_vv = post_vv.subtract(pre_vv).rename(f"{event_key}_delta_VV")
    ratio_vv = ee.Image(10).pow(post_vv.divide(10)).divide(ee.Image(10).pow(pre_vv.divide(10))).rename(f"{event_key}_ratio_VV")
    raw_delta = delta_vv.lt(delta_threshold)
    raw_ratio = ratio_vv.lt(ratio_threshold)
    raw_vv_post = post_vv.lt(vv_post_threshold)

    otsu_threshold = _otsu_threshold(
        delta_vv,
        region=aoi,
        band=f"{event_key}_delta_VV",
    )

    otsu_valid = bool(otsu_threshold is not None and otsu_threshold < 0)

    if otsu_threshold is None:
        raw_otsu = ee.Image(0).rename(f"{event_key}_flood_otsu_raw").toByte()
    else:
        raw_otsu = delta_vv.lt(otsu_threshold)

    flood_delta = _clean_mask(
        raw_delta,
        permanent_water,
        slope,
        slope_threshold,
        min_connected_pixels,
        f"{event_key}_flood_delta_clean",
    )

    flood_ratio = _clean_mask(
        raw_ratio,
        permanent_water,
        slope,
        slope_threshold,
        min_connected_pixels,
        f"{event_key}_flood_ratio_clean",
    )

    flood_vv_post = _clean_mask(
        raw_vv_post,
        permanent_water,
        slope,
        slope_threshold,
        min_connected_pixels,
        f"{event_key}_flood_vv_post_clean",
    )

    flood_otsu = _clean_mask(
        raw_otsu,
        permanent_water,
        slope,
        slope_threshold,
        min_connected_pixels,
        f"{event_key}_flood_otsu_clean",
    )

    confidence_masks = [flood_delta, flood_ratio, flood_vv_post]

    if use_otsu_in_confidence and otsu_valid:
        confidence_masks.append(flood_otsu)

    sar_confidence = _confidence_from_masks(
        confidence_masks,
        f"{event_key}_sar_confidence",
    )

    df_area = pd.DataFrame([
        {
            "event": event_key,
            "method": f"delta_VV_lt_{delta_threshold}_dB",
            "used_in_confidence": True,
            "area_km2": _area_km2(flood_delta, aoi),
        },
        {
            "event": event_key,
            "method": f"ratio_lt_{ratio_threshold}",
            "used_in_confidence": True,
            "area_km2": _area_km2(flood_ratio, aoi),
        },
        {
            "event": event_key,
            "method": f"VV_post_lt_{vv_post_threshold}_dB",
            "used_in_confidence": True,
            "area_km2": _area_km2(flood_vv_post, aoi),
        },
        {
            "event": event_key,
            "method": "otsu_delta_VV_failed" if otsu_threshold is None else f"otsu_delta_VV_lt_{otsu_threshold:.3f}_dB",
            "used_in_confidence": bool(use_otsu_in_confidence and otsu_valid),
            "area_km2": _area_km2(flood_otsu, aoi),
        },
        {
            "event": event_key,
            "method": "sar_confidence_medium_high",
            "used_in_confidence": True,
            "area_km2": _area_km2(sar_confidence.gte(2), aoi),
        },
        {
            "event": event_key,
            "method": "sar_confidence_high",
            "used_in_confidence": True,
            "area_km2": _area_km2(sar_confidence.eq(3), aoi),
        },
    ])

    df_metadata.to_csv(output_dir / f"{event_key}_s1_metadata_audit.csv", index=False)
    df_candidates.to_csv(output_dir / f"{event_key}_s1_orbit_candidates.csv", index=False)
    df_area.to_csv(output_dir / f"{event_key}_sar_area_methods.csv", index=False)

    sar_map = None

    if make_map:
        sar_map = _make_sar_map(
            event=event,
            places=places,
            rivers_image=rivers_image,
            permanent_water=permanent_water,
            slope=slope,
            vis=vis,
            pre_vv=pre_vv,
            post_vv=post_vv,
            delta_vv=delta_vv,
            ratio_vv=ratio_vv,
            flood_delta=flood_delta,
            flood_ratio=flood_ratio,
            flood_vv_post=flood_vv_post,
            flood_otsu=flood_otsu,
            sar_confidence=sar_confidence,
            delta_threshold=delta_threshold,
            ratio_threshold=ratio_threshold,
            vv_post_threshold=vv_post_threshold,
        )

    return {
        "df_candidates": df_candidates,
        "df_metadata": df_metadata,
        "df_area": df_area,
        "selected": selected,
        "otsu_threshold": otsu_threshold,
        "otsu_valid": otsu_valid,
        "s1_pre": s1_pre,
        "s1_post": s1_post,
        "pre_vv": pre_vv,
        "post_vv": post_vv,
        "pre_vh": pre_vh,
        "post_vh": post_vh,
        "delta_vv": delta_vv,
        "ratio_vv": ratio_vv,
        "flood_delta_clean": flood_delta,
        "flood_ratio_clean": flood_ratio,
        "flood_vv_post_clean": flood_vv_post,
        "flood_otsu_clean": flood_otsu,
        "sar_confidence": sar_confidence,
        "map": sar_map,
    }
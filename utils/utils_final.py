import ee
import geemap
import pandas as pd
from pathlib import Path

from utils import utils_setup as us


WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

VULNERABILITY_CLASSES = {
    0: "No / very low vulnerability",
    1: "Low vulnerability",
    2: "Medium vulnerability",
    3: "High vulnerability",
}

RISK_CLASSES = {
    0: "No / very low risk",
    1: "Low risk",
    2: "Medium risk",
    3: "High risk",
}


def get_risk_vis_params():
    return {
        "hazard": {
            "min": 0,
            "max": 3,
            "palette": ["ffffff", "ffffcc", "fd8d3c", "bd0026"],
        },
        "vulnerability_score": {
            "min": 0,
            "max": 5,
            "palette": ["ffffff", "ffffcc", "fed976", "fd8d3c", "e31a1c", "800026"],
        },
        "vulnerability_class": {
            "min": 0,
            "max": 3,
            "palette": ["ffffff", "ffffcc", "fd8d3c", "bd0026"],
        },
        "risk_score": {
            "min": 0,
            "max": 9,
            "palette": ["ffffff", "ffffcc", "fed976", "fd8d3c", "e31a1c", "800026"],
        },
        "risk_class": {
            "min": 0,
            "max": 3,
            "palette": ["ffffff", "ffffcc", "fd8d3c", "bd0026"],
        },
        "distance_to_river": {
            "min": 0,
            "max": 5000,
            "palette": ["0000ff", "00ffff", "ffffff"],
        },
        "binary": {
            "min": 0,
            "max": 1,
            "palette": ["ffffff", "ff0000"],
        },
        "water_occurrence": {
            "min": 0,
            "max": 100,
            "palette": ["ffffff", "add8e6", "0000ff"],
        },
    }


def _ee_fc_to_df(fc):
    info = fc.getInfo()
    features = info.get("features", [])

    if len(features) == 0:
        return pd.DataFrame()

    return pd.DataFrame([feature["properties"] for feature in features])


def _load_rivers_fc(aoi):
    return (
        ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers")
        .filterBounds(aoi)
    )


def create_hazard_from_sar_confidence(sar_confidence, name="hazard"):
    return (
        sar_confidence
        .unmask(0)
        .rename(name)
        .toByte()
    )


def create_hazard_from_fusion(fusion_map, name="hazard"):
    fusion = fusion_map.unmask(0)

    return (
        ee.Image(0)
        .where(fusion.eq(2), 1)
        .where(fusion.eq(6), 1)
        .where(fusion.eq(1), 2)
        .where(fusion.eq(3), 3)
        .rename(name)
        .toByte()
    )


def create_hazard_layer(hazard_input, hazard_source, name="hazard"):
    source = hazard_source.lower()

    if source == "fusion":
        return create_hazard_from_fusion(hazard_input, name=name)

    if source == "sar":
        return create_hazard_from_sar_confidence(hazard_input, name=name)

    raise ValueError("hazard_source must be either 'fusion' or 'sar'.")


def _create_low_elevation_mask(dem, region, percentile=25, scale=30):
    threshold = dem.reduceRegion(
        reducer=ee.Reducer.percentile([percentile]),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
        tileScale=4,
    ).get("DEM")

    threshold = ee.Number(threshold)

    return {
        "threshold": threshold,
        "mask": dem.lte(threshold).rename("low_elevation"),
    }


def _distance_to_rivers(rivers_fc, max_distance_m=5000, line_width=2):
    river_raster = (
        ee.Image()
        .byte()
        .paint(featureCollection=rivers_fc, color=1, width=line_width)
        .selfMask()
    )

    return river_raster.distance(
        kernel=ee.Kernel.euclidean(max_distance_m, "meters"),
        skipMasked=False,
    ).rename("distance_to_river_m")


def create_vulnerability_layers(
    dem,
    slope,
    worldcover,
    water_occurrence,
    rivers_fc,
    region,
    elevation_percentile=25,
    slope_threshold=2,
    river_distance_threshold=1000,
    historical_water_threshold=10,
):
    low_elev_data = _create_low_elevation_mask(
        dem=dem,
        region=region,
        percentile=elevation_percentile,
        scale=30,
    )

    low_elevation = low_elev_data["mask"]
    low_slope = slope.lt(slope_threshold).rename("low_slope")

    distance_to_river = _distance_to_rivers(
        rivers_fc=rivers_fc,
        max_distance_m=river_distance_threshold,
        line_width=2,
    )

    near_river = distance_to_river.lt(river_distance_threshold).rename("near_river")

    exposed_landcover = (
        worldcover.eq(40)
        .Or(worldcover.eq(50))
        .rename("exposed_landcover")
    )

    historical_water = (
        water_occurrence
        .unmask(0)
        .gt(historical_water_threshold)
        .rename("historical_water")
    )

    vulnerability_score = (
        low_elevation.unmask(0)
        .add(low_slope.unmask(0))
        .add(near_river.unmask(0))
        .add(exposed_landcover.unmask(0))
        .add(historical_water.unmask(0))
        .rename("vulnerability_score")
        .toByte()
    )

    vulnerability_class = (
        ee.Image(0)
        .where(vulnerability_score.eq(1), 1)
        .where(vulnerability_score.gte(2).And(vulnerability_score.lte(3)), 2)
        .where(vulnerability_score.gte(4), 3)
        .rename("vulnerability_class")
        .toByte()
    )

    return {
        "low_elevation_threshold": low_elev_data["threshold"],
        "low_elevation": low_elevation,
        "low_slope": low_slope,
        "distance_to_river": distance_to_river,
        "near_river": near_river,
        "exposed_landcover": exposed_landcover,
        "historical_water": historical_water,
        "vulnerability_score": vulnerability_score,
        "vulnerability_class": vulnerability_class,
    }


def create_risk_map(hazard, vulnerability_class, name="risk"):
    risk_score = (
        hazard.unmask(0)
        .multiply(vulnerability_class.unmask(0))
        .rename(f"{name}_score")
        .toByte()
    )

    risk_class = (
        ee.Image(0)
        .where(risk_score.gte(1).And(risk_score.lte(3)), 1)
        .where(risk_score.gte(4).And(risk_score.lte(6)), 2)
        .where(risk_score.gt(6), 3)
        .rename(f"{name}_class")
        .toByte()
    )

    return {
        "risk_score": risk_score,
        "risk_class": risk_class,
    }


def area_by_worldcover(binary_mask, worldcover, region, event_key, scale=10):
    mask = binary_mask.unmask(0).gt(0)

    area_image = (
        ee.Image.pixelArea()
        .divide(1e6)
        .rename("area")
        .updateMask(mask)
        .addBands(worldcover.rename("class"))
    )

    result = area_image.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="class"),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
        tileScale=4,
    ).get("groups").getInfo()

    rows = []

    if result is not None:
        for item in result:
            class_value = int(item["class"])
            rows.append({
                "event": event_key,
                "worldcover_class": class_value,
                "worldcover_label": WORLDCOVER_CLASSES.get(class_value, "Unknown"),
                "area_km2": float(item["sum"]),
            })

    return pd.DataFrame(rows)


def area_by_class(class_image, class_dict, region, event_key, scale=10):
    area_image = (
        ee.Image.pixelArea()
        .divide(1e6)
        .rename("area")
        .addBands(class_image.unmask(0).rename("class"))
    )

    result = area_image.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="class"),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
        tileScale=4,
    ).get("groups").getInfo()

    rows = []

    if result is not None:
        for item in result:
            class_value = int(item["class"])
            rows.append({
                "event": event_key,
                "class": class_value,
                "label": class_dict.get(class_value, "Unknown"),
                "area_km2": float(item["sum"]),
            })

    return pd.DataFrame(rows)


def roi_statistics(
    hazard,
    risk_class,
    worldcover,
    roi_zones,
    event_key,
    scale=20,
):
    """
    Compute flood/risk/exposure statistics over predefined ROI polygons.

    This is preferable to city-point buffers because flood impacts usually occur
    over floodplain areas, agricultural zones, river corridors, or municipal
    surroundings, not necessarily at the exact city-centre point.
    """

    hazard_mask = hazard.gte(2).unmask(0)
    high_risk_mask = risk_class.eq(3).unmask(0)

    builtup = worldcover.eq(50).unmask(0)
    cropland = worldcover.eq(40).unmask(0)

    pixel_area_km2 = ee.Image.pixelArea().divide(1e6)

    area_image = ee.Image.cat([
        hazard_mask.multiply(pixel_area_km2).rename("flood_area_km2"),
        high_risk_mask.multiply(pixel_area_km2).rename("high_risk_area_km2"),
        hazard_mask.And(builtup).multiply(pixel_area_km2).rename("builtup_flooded_km2"),
        hazard_mask.And(cropland).multiply(pixel_area_km2).rename("cropland_flooded_km2"),
    ])

    def add_event_property(feat):
        return ee.Feature(feat).set("event", event_key)

    roi_zones = roi_zones.map(add_event_property)

    stats_fc = area_image.reduceRegions(
        collection=roi_zones,
        reducer=ee.Reducer.sum(),
        scale=scale,
        tileScale=4,
    )

    df = _ee_fc_to_df(stats_fc)

    if df.empty:
        return df

    keep_cols = [
        "event",
        "name",
        "flood_area_km2",
        "high_risk_area_km2",
        "builtup_flooded_km2",
        "cropland_flooded_km2",
    ]

    df = df[[col for col in keep_cols if col in df.columns]]

    numeric_cols = [
        "flood_area_km2",
        "high_risk_area_km2",
        "builtup_flooded_km2",
        "cropland_flooded_km2",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    return df.sort_values("flood_area_km2", ascending=False).reset_index(drop=True)


def make_final_risk_map(
    event,
    places,
    hazard,
    vulnerability,
    risk_score,
    risk_class,
    water_occurrence,
    vis,
):
    aoi = event["aoi"]

    raster_layers = [
        {"image": hazard.selfMask(), "vis": vis["hazard"], "name": "Flood hazard", "shown": False},
        {"image": vulnerability["vulnerability_score"].clip(aoi), "vis": vis["vulnerability_score"], "name": "Vulnerability score", "shown": False},
        {"image": vulnerability["vulnerability_class"].selfMask().clip(aoi), "vis": vis["vulnerability_class"], "name": "Vulnerability class", "shown": False},
        {"image": risk_score.selfMask().clip(aoi), "vis": vis["risk_score"], "name": "Risk score", "shown": False},
        {"image": risk_class.selfMask().clip(aoi), "vis": vis["risk_class"], "name": "Risk class", "shown": True},
        {"image": vulnerability["distance_to_river"].clip(aoi), "vis": vis["distance_to_river"], "name": "Distance to river", "shown": False},
        {"image": vulnerability["low_elevation"].selfMask().clip(aoi), "vis": vis["binary"], "name": "Low elevation", "shown": False},
        {"image": vulnerability["low_slope"].selfMask().clip(aoi), "vis": vis["binary"], "name": "Low slope", "shown": False},
        {"image": vulnerability["near_river"].selfMask().clip(aoi), "vis": vis["binary"], "name": "Near river", "shown": False},
        {"image": vulnerability["exposed_landcover"].selfMask().clip(aoi), "vis": vis["binary"], "name": "Exposed land cover", "shown": False},
        {"image": water_occurrence.clip(aoi), "vis": vis["water_occurrence"], "name": "JRC water occurrence", "shown": False},
    ]

    m = geemap.Map(center=event["center"], zoom=event["zoom"])
    us.add_layers(m, raster_layers)

    m.addLayer(ee.FeatureCollection([ee.Feature(aoi)]).style(color=event.get("aoi_color", "red"), fillColor="00000000", width=2), {}, event.get("aoi_label", "AOI"))
    m.addLayer(places.filterBounds(aoi).style(color="000000", pointSize=5, pointShape="circle"), {}, "Places")
    m.addLayerControl()

    return m


def run_final_risk_event(
    event_key,
    event,
    hazard_input,
    hazard_source,
    dem,
    slope,
    worldcover,
    water_occurrence,
    places,
    output_dir,
    elevation_percentile=25,
    slope_threshold=2,
    river_distance_threshold=1000,
    historical_water_threshold=10,
    roi_buffer_m=3000,
    make_map=True,
    vis=None,
):
    if vis is None:
        vis = get_risk_vis_params()

    aoi = event["aoi"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rivers_fc = _load_rivers_fc(aoi)

    hazard = create_hazard_layer(
        hazard_input=hazard_input,
        hazard_source=hazard_source,
        name=f"{event_key}_hazard",
    )

    vulnerability = create_vulnerability_layers(
        dem=dem,
        slope=slope,
        worldcover=worldcover,
        water_occurrence=water_occurrence,
        rivers_fc=rivers_fc,
        region=aoi,
        elevation_percentile=elevation_percentile,
        slope_threshold=slope_threshold,
        river_distance_threshold=river_distance_threshold,
        historical_water_threshold=historical_water_threshold,
    )

    risk = create_risk_map(
        hazard=hazard,
        vulnerability_class=vulnerability["vulnerability_class"],
        name=f"{event_key}_risk",
    )

    df_exposure = area_by_worldcover(
        binary_mask=hazard.gte(2),
        worldcover=worldcover,
        region=aoi,
        event_key=event_key,
        scale=10,
    )

    df_vulnerability_area = area_by_class(
        class_image=vulnerability["vulnerability_class"],
        class_dict=VULNERABILITY_CLASSES,
        region=aoi,
        event_key=event_key,
        scale=30,
    )

    df_risk_area = area_by_class(
        class_image=risk["risk_class"],
        class_dict=RISK_CLASSES,
        region=aoi,
        event_key=event_key,
        scale=10,
    )

    roi_zones = event.get("roi_zones", None)

    if roi_zones is not None:
        df_roi = roi_statistics(
            hazard=hazard,
            risk_class=risk["risk_class"],
            worldcover=worldcover,
            roi_zones=roi_zones,
            event_key=event_key,
            scale=20,
        )
    else:
        df_roi = pd.DataFrame()

    df_exposure.to_csv(output_dir / f"{event_key}_final_exposure_by_worldcover.csv", index=False)
    df_vulnerability_area.to_csv(output_dir / f"{event_key}_final_vulnerability_area.csv", index=False)
    df_risk_area.to_csv(output_dir / f"{event_key}_final_risk_area.csv", index=False)
    df_roi.to_csv(output_dir / f"{event_key}_final_roi_statistics.csv", index=False)

    final_map = None

    if make_map:
        final_map = make_final_risk_map(
            event=event,
            places=places,
            hazard=hazard,
            vulnerability=vulnerability,
            risk_score=risk["risk_score"],
            risk_class=risk["risk_class"],
            water_occurrence=water_occurrence,
            vis=vis,
        )

    return {
        "event": event_key,
        "hazard_source": hazard_source,
        "hazard": hazard,
        "vulnerability": vulnerability,
        "vulnerability_score": vulnerability["vulnerability_score"],
        "vulnerability_class": vulnerability["vulnerability_class"],
        "risk_score": risk["risk_score"],
        "risk_class": risk["risk_class"],
        "df_exposure": df_exposure,
        "df_vulnerability_area": df_vulnerability_area,
        "df_risk_area": df_risk_area,
        "df_roi": df_roi,
        "low_elevation_threshold": vulnerability["low_elevation_threshold"],
        "map": final_map,
    }
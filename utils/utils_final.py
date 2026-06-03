import ee
import geemap
import pandas as pd
from pathlib import Path
from utils import config as cfg
from utils import utils_setup as us


# =============================================================================
# Hazard, vulnerability and risk products
# =============================================================================

def make_hazard_layer(sar_confidence, confidence_threshold=3, name="hazard"):
    """
    Convert SAR confidence into a 0-3 flood hazard class.

    Classes:
        0 = no / very low hazard
        1 = low hazard
        2 = medium hazard
        3 = high hazard
    """
    confidence = sar_confidence.unmask(0)

    return (
        ee.Image(0)
        .where(
            confidence.gt(0).And(confidence.lt(confidence_threshold)),
            1,
        )
        .where(
            confidence.gte(confidence_threshold)
            .And(confidence.lt(confidence_threshold + 1)),
            2,
        )
        .where(
            confidence.gte(confidence_threshold + 1),
            3,
        )
        .rename(name)
        .toByte()
    )

def make_vulnerability_layers(
    dem,
    slope,
    worldcover,
    water_occurrence,
    rivers,
    region,
    elevation_percentile=25,
    slope_threshold=2,
    river_distance_threshold=1000,
    historical_water_threshold=10,
):
    """Create simple vulnerability factors and a 0-3 vulnerability class."""

    rivers_raster = (
        ee.Image()
        .byte()
        .paint(featureCollection=rivers, color=1, width=2)
        .selfMask()
    )

    distance_to_river = rivers_raster.distance(
        kernel=ee.Kernel.euclidean(river_distance_threshold, "meters"),
        skipMasked=False,
    ).rename("distance_to_river_m")

    near_river = distance_to_river.lt(river_distance_threshold).rename("near_river")

    elevation_threshold = ee.Number(
        dem.updateMask(near_river).reduceRegion(
            reducer=ee.Reducer.percentile([elevation_percentile]),
            geometry=region,
            scale=30,
            maxPixels=1e13,
            bestEffort=True,
            tileScale=4,
        ).get("DEM")
    )

    low_elevation = (
        dem.lte(elevation_threshold)
        .And(near_river)
        .rename("low_elevation")
    )

    low_slope = slope.lt(slope_threshold).rename("low_slope")

    historical_water = (
        water_occurrence
        .unmask(0)
        .gt(historical_water_threshold)
        .rename("historical_water")
    )

    cropland = worldcover.eq(40)
    builtup = worldcover.eq(50)

    exposed_landcover = (
        cropland.Or(builtup)
        .rename("exposed_landcover")
    )

    landcover_exposure_score = (
        cropland.multiply(1)
        .add(builtup.multiply(2))
    )

    vulnerability_score = (
        low_elevation.unmask(0)
        .add(low_slope.unmask(0))
        .add(near_river.unmask(0))
        .add(landcover_exposure_score.unmask(0))
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
        "elevation_threshold": elevation_threshold,
        "low_elevation": low_elevation,
        "low_slope": low_slope,
        "distance_to_river": distance_to_river,
        "near_river": near_river,
        "exposed_landcover": exposed_landcover,
        "historical_water": historical_water,
        "vulnerability_score": vulnerability_score,
        "vulnerability_class": vulnerability_class,
    }

def make_risk_layers(hazard, vulnerability_class, name="risk"):
    """Combine flood hazard and vulnerability into risk score and risk class."""
    risk_score = (
        hazard.unmask(0)
        .multiply(vulnerability_class.unmask(0))
        .rename(f"{name}_score")
        .toByte()
    )

    risk_class = (
        ee.Image(0)
        .where(risk_score.gte(1).And(risk_score.lte(2)), 1)
        .where(risk_score.gte(3).And(risk_score.lte(5)), 2)
        .where(risk_score.gte(6), 3)
        .rename(f"{name}_class")
        .toByte()
    )

    return {
        "risk_score": risk_score,
        "risk_class": risk_class,
    }


# =============================================================================
# Tables
# =============================================================================

def area_by_class(class_image, class_labels, region, event_key, scale=10):
    """Compute area grouped by a categorical class image."""
    area_image = (
        ee.Image.pixelArea()
        .divide(1e6)
        .rename("area")
        .addBands(class_image.unmask(0).rename("class"))
    )

    groups = area_image.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="class"),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
        tileScale=4,
    ).get("groups").getInfo()

    rows = []

    if groups is not None:
        for item in groups:
            class_value = int(item["class"])
            rows.append({
                "event": event_key,
                "class": class_value,
                "label": class_labels.get(class_value, "Unknown"),
                "area_km2": float(item["sum"]),
            })

    return pd.DataFrame(rows).sort_values("class").reset_index(drop=True)

def exposure_by_worldcover(hazard, worldcover, region, event_key, scale=10):
    """Compute medium/high flood hazard area grouped by WorldCover class."""
    hazard_mask = hazard.gte(2)

    area_image = (
        ee.Image.pixelArea()
        .divide(1e6)
        .rename("area")
        .updateMask(hazard_mask)
        .addBands(worldcover.rename("class"))
    )

    groups = area_image.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="class"),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
        tileScale=4,
    ).get("groups").getInfo()

    rows = []

    if groups is not None:
        for item in groups:
            class_value = int(item["class"])
            rows.append({
                "event": event_key,
                "worldcover_class": class_value,
                "worldcover_label": cfg.WORLDCOVER_CLASSES.get(class_value, "Unknown"),
                "area_km2": float(item["sum"]),
            })

    return (
        pd.DataFrame(rows)
        .sort_values("area_km2", ascending=False)
        .reset_index(drop=True)
    )

def final_processing_df(event_key, risk_params, final_results):
    """Create a compact vertical summary of the final workflow settings."""
    rows = [
        ("Event", event_key),
        ("Hazard source", "Sentinel-1 SAR confidence"),
        ("Elevation percentile", risk_params.get("elevation_percentile")),
        (
            "Low-elevation threshold",
            round(final_results["elevation_threshold"].getInfo(), 2),
        ),
        ("Slope threshold", f"{risk_params.get('slope_threshold')}°"),
        (
            "River distance threshold",
            f"{risk_params.get('river_distance_threshold')} m",
        ),
        (
            "Historical water threshold",
            f"{risk_params.get('historical_water_threshold')}%",
        ),
    ]

    return pd.DataFrame(rows, columns=["parameter", "value"])


# =============================================================================
# Map and figures
# =============================================================================

def make_final_map(event, places, hazard, vulnerability, risk):
    """Create a simple interactive final risk map."""
    aoi = event["aoi"]
    vis = us.get_vis_params()

    layers = [
        (hazard.selfMask().clip(aoi), vis["hazard"], "Flood hazard class", False),
        (
            vulnerability["vulnerability_class"].selfMask().clip(aoi),
            vis["vulnerability_class"],
            "Vulnerability class",
            False,
        ),
        (
            risk["risk_class"].selfMask().clip(aoi),
            vis["risk_class"],
            "Final risk class",
            True,
        ),
        (
            vulnerability["distance_to_river"].clip(aoi),
            vis["distance_to_river"],
            "Distance to river",
            False,
        ),
        (
            vulnerability["low_elevation"].selfMask().clip(aoi),
            vis["binary_red"],
            "Low elevation",
            False,
        ),
        (
            vulnerability["low_slope"].selfMask().clip(aoi),
            vis["binary_red"],
            "Low slope",
            False,
        ),
        (
            vulnerability["near_river"].selfMask().clip(aoi),
            vis["binary_red"],
            "Near river",
            False,
        ),
        (
            vulnerability["exposed_landcover"].selfMask().clip(aoi),
            vis["binary_red"],
            "Exposed land cover",
            False,
        ),
        (
            vulnerability["historical_water"].selfMask().clip(aoi),
            vis["binary_red"],
            "Historical water occurrence",
            False,
        ),
    ]

    m = geemap.Map(center=event["center"], zoom=event["zoom"])

    for image, layer_vis, name, shown in layers:
        m.addLayer(image, layer_vis, name, shown)

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

def export_hazard_vs_risk_png(final_results, region, output_dir, month, dimensions=900):
    """Export hazard and final risk class side by side."""
    vis = us.get_vis_params()

    return us.export_side_by_side_ee_images(
        left_image=final_results["hazard"],
        right_image=final_results["risk_class"],
        region=region,
        vis_params=vis["risk_class"],
        output_path=Path(output_dir) / f"{month}_hazard_vs_final_risk.png",
        left_title="Flood hazard class",
        right_title="Final risk class",
        dimensions=dimensions,
    )


# =============================================================================
# Public pipeline
# =============================================================================

def run_final_event(
    event_key,
    event,
    sar_confidence,
    dem,
    slope,
    worldcover,
    water_occurrence,
    rivers,
    places=None,
    output_dir=None,
    make_map=True,
    confidence_threshold=3,
    elevation_percentile=25,
    slope_threshold=2,
    river_distance_threshold=1000,
    historical_water_threshold=10,
    roi_buffer_m=None,
    **unused_params,
):
    """Run the final SAR-based hazard, vulnerability and risk workflow."""
    aoi = event["aoi"]

    hazard = make_hazard_layer(
        sar_confidence=sar_confidence,
        confidence_threshold=confidence_threshold,
        name=f"{event_key}_hazard",
    )

    vulnerability = make_vulnerability_layers(
        dem=dem,
        slope=slope,
        worldcover=worldcover,
        water_occurrence=water_occurrence,
        rivers=rivers,
        region=aoi,
        elevation_percentile=elevation_percentile,
        slope_threshold=slope_threshold,
        river_distance_threshold=river_distance_threshold,
        historical_water_threshold=historical_water_threshold,
    )

    risk = make_risk_layers(
        hazard=hazard,
        vulnerability_class=vulnerability["vulnerability_class"],
        name=f"{event_key}_risk",
    )

    df_exposure = exposure_by_worldcover(
        hazard=hazard,
        worldcover=worldcover,
        region=aoi,
        event_key=event_key,
        scale=10,
    )

    df_vulnerability_area = area_by_class(
        class_image=vulnerability["vulnerability_class"],
        class_labels=cfg.VULNERABILITY_CLASSES,
        region=aoi,
        event_key=event_key,
        scale=30,
    )

    df_risk_area = area_by_class(
        class_image=risk["risk_class"],
        class_labels=cfg.RISK_CLASSES,
        region=aoi,
        event_key=event_key,
        scale=10,
    )

    result = {
        "hazard": hazard,
        "vulnerability": vulnerability,
        "vulnerability_score": vulnerability["vulnerability_score"],
        "vulnerability_class": vulnerability["vulnerability_class"],
        "risk_score": risk["risk_score"],
        "risk_class": risk["risk_class"],
        "elevation_threshold": vulnerability["elevation_threshold"],
        "df_exposure": df_exposure,
        "df_vulnerability_area": df_vulnerability_area,
        "df_risk_area": df_risk_area,
    }

    risk_params = {
        "elevation_percentile": elevation_percentile,
        "slope_threshold": slope_threshold,
        "river_distance_threshold": river_distance_threshold,
        "historical_water_threshold": historical_water_threshold,
    }

    result["df_processing"] = final_processing_df(
        event_key=event_key,
        risk_params=risk_params,
        final_results=result,
    )

    result["map"] = (
        make_final_map(
            event=event,
            places=places,
            hazard=hazard,
            vulnerability=vulnerability,
            risk=risk,
        )
        if make_map
        else None
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result["df_processing"].to_csv(
            output_dir / f"{event_key}_final_processing_summary.csv",
            index=False,
        )

        df_exposure.to_csv(
            output_dir / f"{event_key}_final_exposure_by_worldcover.csv",
            index=False,
        )

        df_vulnerability_area.to_csv(
            output_dir / f"{event_key}_final_vulnerability_area.csv",
            index=False,
        )

        df_risk_area.to_csv(
            output_dir / f"{event_key}_final_risk_area.csv",
            index=False,
        )

    return result
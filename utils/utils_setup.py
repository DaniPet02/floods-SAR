import ee
import geemap
import pandas as pd
import requests
from io import BytesIO
from pathlib import Path
from IPython.display import Image as IPyImage
from PIL import Image, ImageDraw
import json
import hashlib
_CACHE = {}


def cache_key(stage, month, params=None, version="v1"):
    text = json.dumps(
        {"stage": stage, "month": month, "params": params or {}, "version": version},
        sort_keys=True,
        default=str,
    )
    return f"{stage}_{month}_{hashlib.md5(text.encode()).hexdigest()[:8]}"

def cached(stage, month, params, compute_fn, use_cache=True, version="v1"):
    key = cache_key(stage, month, params, version)

    if use_cache and key in _CACHE:
        print(f"Using cached {stage} for {month}.")
        return _CACHE[key]

    print(f"Computing {stage} for {month}.")
    _CACHE[key] = compute_fn()
    return _CACHE[key]

def clear_cache(stage=None):
    if stage is None:
        _CACHE.clear()
    else:
        for key in list(_CACHE):
            if key.startswith(f"{stage}_"):
                del _CACHE[key]


# =============================================================================
# Dataset loading
# =============================================================================

def load_event_datasets(aoi, dates, event, sar_params=None, optical_params=None):
    """
    Load Sentinel-1 and Sentinel-2 collections for one selected flood event.

    The function loads pre-event, post-event and recovery windows.
    Auxiliary datasets are loaded separately because they are static layers,
    not event-specific satellite acquisitions.
    """
    sar_params = sar_params or {}
    optical_params = optical_params or {}

    orbit_pass = sar_params.get("orbit_pass_override")
    relative_orbit = sar_params.get("relative_orbit_override")

    s1 = {}
    s1_all = {}
    s1_orbits = {}
    s2 = {}

    for window in ["pre", "post", "recovery"]:
        # ---------------------------------------------------------------------
        # Sentinel-1 SAR
        # ---------------------------------------------------------------------
        start_date, end_date = dates[event[window]]

        s1_base = (
            ee.ImageCollection("COPERNICUS/S1_GRD") # Sentinel-1 Ground Range Detected (GRD) product, which is the most commonly used for flood mapping due to its balance of resolution and coverage. 
            # It provides dual-polarization data (VV and VH) at 10m resolution, which are essential for our flood detection algorithm.
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq("instrumentMode", "IW")) # Interferometric Wide swath mode, which is the default mode
            .filter(ee.Filter.eq("resolution_meters", 10))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        )

        s1_all[window] = s1_base

        s1_selected = s1_base

        if orbit_pass is not None:
            s1_selected = s1_selected.filter(
                ee.Filter.eq("orbitProperties_pass", orbit_pass)
            )

        if relative_orbit is not None:
            s1_selected = s1_selected.filter(
                ee.Filter.eq("relativeOrbitNumber_start", int(relative_orbit))
            )

        s1[window] = s1_selected

        # Orbit diagnostics are useful only for pre/post comparison.
        if window in ["pre", "post"]:
            s1_orbits[f"{window}_ASC"] = s1_base.filter(
                ee.Filter.eq("orbitProperties_pass", "ASCENDING")
            )
            s1_orbits[f"{window}_DESC"] = s1_base.filter(
                ee.Filter.eq("orbitProperties_pass", "DESCENDING")
            )

        # ---------------------------------------------------------------------
        # Sentinel-2 optical
        # ---------------------------------------------------------------------
        optical_date_key = optical_params.get(
            f"{window}_window_override",
            event[window],
        )

        optical_start, optical_end = dates[optical_date_key]

        max_cloud = {
            "pre": optical_params.get("max_cloud_pre", 95),
            "post": optical_params.get("max_cloud_post", 100),
            "recovery": optical_params.get("max_cloud_recovery", 95),
        }[window]

        s2[window] = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") # Sentinel-2 Level-2A Surface Reflectance product, 
            # which provides atmospherically corrected reflectance values at 10m resolution, essential for accurate water detection using spectral indices.
            .filterBounds(aoi)
            .filterDate(optical_start, optical_end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        )

    availability = {
        "S1 pre-event ALL": s1_all["pre"],
        "S1 post-event ALL": s1_all["post"],
        "S1 recovery ALL": s1_all["recovery"],
        "S1 pre-event selected orbit": s1["pre"],
        "S1 post-event selected orbit": s1["post"],
        "S1 recovery selected orbit": s1["recovery"],
        "S1 pre-event ASC": s1_orbits["pre_ASC"],
        "S1 post-event ASC": s1_orbits["post_ASC"],
        "S1 pre-event DESC": s1_orbits["pre_DESC"],
        "S1 post-event DESC": s1_orbits["post_DESC"],
        "S2 pre-event raw": s2["pre"],
        "S2 post-event raw": s2["post"],
        "S2 recovery raw": s2["recovery"],
    }

    return {
        "s1": s1,
        "s1_all": s1_all,
        "s1_orbits": s1_orbits,
        "s2": s2,
        "availability": availability,
    }

def load_auxiliary_datasets(aoi):
    """
    Load static auxiliary layers used for flood post-processing and interpretation.

    These layers are not tied to a specific event date:
    - JRC Global Surface Water
    - permanent water mask
    - ESA WorldCover
    - Copernicus DEM
    - slope
    - HydroSHEDS rivers
    """
    jrc_water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater") # JRC Global Surface Water dataset, which provides historical water occurrence and change information at 30m resolution, 
    # crucial for identifying areas that are frequently flooded or have permanent water presence, thus improving the accuracy of our flood risk assessment.
    water_occurrence = jrc_water.select("occurrence").unmask(0)
    permanent_water = water_occurrence.gt(80).rename("permanent_water")

    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map") # ESA WorldCover dataset, which provides global land cover classification at 10m resolution for the year 2020,

    ghsl_population = (
        ee.ImageCollection("JRC/GHSL/P2023A/GHS_POP")
        .sort("system:time_start", False)
        .first()
        .select("population_count")
        .clip(aoi)
        .rename("population_count")
    )

    worldcover_classes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    worldcover_remap = worldcover.remap(
        worldcover_classes,
        list(range(len(worldcover_classes))),
    ).rename("WorldCover_remap")

    dem_collection = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM") # Copernicus DEM GLO-30, which provides global digital elevation data at 30m resolution, 
    # essential for identifying low-lying areas and understanding flood dynamics in our risk assessment.
    dem = dem_collection.mosaic().setDefaultProjection(
        dem_collection.first().projection()
    )

    slope = ee.Terrain.slope(dem).rename("slope")

    rivers = (
        ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers") # HydroSHEDS Free-Flowing Rivers dataset, which provides global river network data at high resolution,
        .filterBounds(aoi)
    )

    rivers_image = ee.Image().byte().paint(
        featureCollection=rivers,
        color=1,
        width=2,
    )

    return {
        "jrc_water": jrc_water,
        "water_occurrence": water_occurrence,
        "permanent_water": permanent_water,
        "worldcover": worldcover,
        "worldcover_remap": worldcover_remap,
        "dem": dem,
        "slope": slope,
        "rivers": rivers,
        "rivers_image": rivers_image,
        "population": ghsl_population,
    }


# =============================================================================
# Tables and quick summaries
# =============================================================================

# dataset availability
def collection_sizes(collections, output_path=None):
    """
    Return a table with the number of images in each Earth Engine collection.

    This is useful at the beginning of the notebook to justify the selected
    pre-event and post-event windows.
    """
    rows = [
        {
            "Dataset window": name,
            "Number of images": collection.size().getInfo(),
        }
        for name, collection in collections.items()
    ]

    df = pd.DataFrame(rows)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


# =============================================================================
# Visualization parameters
# =============================================================================

def get_vis_params():
    """Centralized visualization parameters for maps and PNG exports."""
    return {
        # Sentinel-1 SAR
        "s1_vv": {
            "min": -25, # backscatter
            "max": 0,
        },
        "s1_change": {
            "min": -5,
            "max": 5,
            "palette": ["blue", "white", "red"],
        },

        "s2_true_color": {
            "bands": ["B4", "B3", "B2"], # Red, Green, Blue for natural color visualization
            "min": 0,
            "max": 3000,
        },

        "true_color": {
            "bands": ["B4", "B3", "B2"],
            "min": 0.0,
            "max": 0.30,
        },
        "false_color": {
            "bands": ["B8", "B4", "B3"], # NIR, Red, Green for vegetation and water detection
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
        "sar_confidence": {
            "min": 0,
            "max": 5,
            "palette": ["ffffff", "ffff00", "ff9900", "ff0000", "990000"],
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
                "ff7f00",  # 6 low-confidence SAR
            ],
        },
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
        "binary_red": {
            "min": 0,
            "max": 1,
            "palette": ["ffffff", "ff0000"],
        },
        "population": {
            "min": 0,
            "max": 1000,
            "palette": ["white", "yellow", "orange", "red"],
        },
        "water_occurrence": {
            "min": 0,
            "max": 100,
            "palette": ["white", "lightblue", "blue"],
        },
        "permanent_water": {
            "min": 0,
            "max": 1,
            "palette": ["white", "blue"],
        },
        "dem": {
            "min": 0,
            "max": 500,
            "palette": ["blue", "green", "yellow", "brown", "white"],
        },
        "slope": {
            "min": 0,
            "max": 20,
            "palette": ["white", "yellow", "orange", "red"],
        },
        "worldcover": {
            "min": 0,
            "max": 10,
            "palette": [
                "006400",  # trees
                "ffbb22",  # shrubland
                "ffff4c",  # grassland
                "f096ff",  # cropland
                "fa0000",  # built-up
                "b4b4b4",  # bare/sparse vegetation
                "f0f0f0",  # snow/ice
                "0064c8",  # permanent water
                "0096a0",  # herbaceous wetland
                "00cf75",  # mangroves
                "fae6a0",  # moss/lichen
            ],
        },
        "rivers": {
            "min": 0,
            "max": 1,
            "palette": ["ffffff", "0000ff"],
        },
        "flood": {
            "min": 0,
            "max": 1,
            "palette": ["00ffff"],
        },
    }

def styled_aoi(aoi, color="red", width=2):
    """Return an AOI geometry as a transparent styled FeatureCollection."""
    return ee.FeatureCollection([ee.Feature(aoi)]).style(
        color=color,
        fillColor="00000000",
        width=width,
    )

def styled_places(places, aoi, color="000000", point_size=4, point_shape="circle"):
    """Return place points inside an AOI as a styled FeatureCollection."""
    return places.filterBounds(aoi).style(
        color=color,
        pointSize=point_size,
        pointShape=point_shape,
    )

def make_context_map(event, aoi, places): # simple map showing AOI and places
    """
    Create a simple context map showing only the study area and relevant places.
    This map is only used for orientation at the beginning of the notebook.
    """
    m = geemap.Map(center=event["center"], zoom=event["zoom"])

    m.addLayer(
        styled_aoi(
            aoi,
            color=event.get("aoi_color", "red"),
        ),
        {},
        event.get("aoi_label", "AOI"),
    )

    m.addLayer(
        styled_places(
            places,
            aoi,
        ),
        {},
        "Places",
    )

    m.addLayerControl()
    return m


# =============================================================================
# Composite helpers
# =============================================================================

def median_composite(collection, band=None, aoi=None): # build a median composite from an Earth Engine ImageCollection
    """
    Build a median composite from an Earth Engine ImageCollection.

    Parameters:
        collection: ee.ImageCollection
        band: optional band name or list of bands
        aoi: optional clipping geometry
    """
    if band is not None:
        collection = collection.select(band)

    image = collection.median()

    if aoi is not None:
        image = image.clip(aoi)

    return image

def mask_area_km2(mask, region, scale=10): # compute area in km² for quantifying flood extent
    """Compute the area of a binary mask in km²."""
    area = (
        mask.unmask(0)
        .gt(0)
        .multiply(ee.Image.pixelArea())
        .divide(1e6)
        .rename("area_km2")
    )

    value = area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
        tileScale=4,
    ).get("area_km2").getInfo()

    return 0.0 if value is None else float(value)


# =============================================================================
# PNG export helpers
# =============================================================================

def export_side_by_side_ee_images( # export two Earth Engine image previews as a single side-by-side PNG
    left_image,
    right_image,
    region,
    vis_params,
    output_path,
    left_title="Pre-event",
    right_title="Post-event",
    dimensions=900,
): 
    """Export two Earth Engine image previews as a single side-by-side PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    left = download_ee_thumbnail(
        image=left_image,
        region=region,
        vis_params=vis_params,
        dimensions=dimensions,
    )

    right = download_ee_thumbnail(
        image=right_image,
        region=region,
        vis_params=vis_params,
        dimensions=dimensions,
    )

    left, right = match_heights(left, right)

    gap = 20
    title_h = 48

    canvas = Image.new(
        "RGB",
        (left.width + right.width + gap, left.height + title_h),
        "white",
    )

    canvas.paste(left, (0, title_h))
    canvas.paste(right, (left.width + gap, title_h))

    draw = ImageDraw.Draw(canvas)
    draw.text((12, 14), left_title, fill="black")
    draw.text((left.width + gap + 12, 14), right_title, fill="black")

    canvas.save(output_path)

    return IPyImage(filename=str(output_path))

def download_ee_thumbnail(image, region, vis_params, dimensions=900): # download earth engine image for png export
    """Download an Earth Engine thumbnail and return it as a PIL RGB image."""
    url = image.clip(region).getThumbURL({
        **vis_params,
        "region": region,
        "dimensions": dimensions,
        "format": "png",
    })

    response = requests.get(url)
    response.raise_for_status()

    return Image.open(BytesIO(response.content)).convert("RGB")

def match_heights(left, right): # resize two images to the same height while keeping aspect ratio
    """Resize two PIL images to the same height while keeping aspect ratio."""
    target_h = min(left.height, right.height)

    left = left.resize(
        (int(left.width * target_h / left.height), target_h)
    )

    right = right.resize(
        (int(right.width * target_h / right.height), target_h)
    )

    return left, right

def export_single_ee_image_png( # export one Earth Engine image preview as PNG and display it in the notebook
    image,
    region,
    vis_params,
    output_path,
    title=None,
    dimensions=900,
):
    """Export one Earth Engine image preview as PNG and display it in the notebook."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = download_ee_thumbnail(
        image=image,
        region=region,
        vis_params=vis_params,
        dimensions=dimensions,
    )

    if title is not None:
        canvas = Image.new("RGB", (img.width, img.height + 44), "white")
        canvas.paste(img, (0, 44))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 12), title, fill="black")
        img = canvas

    img.save(output_path)

    return IPyImage(filename=str(output_path))
import ee
import geemap
import pandas as pd


def load_sentinel1(aoi, start_date, end_date, orbit_pass=None, relative_orbit=None):
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("resolution_meters", 10))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )

    if orbit_pass is not None:
        collection = collection.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))

    if relative_orbit is not None:
        collection = collection.filter(ee.Filter.eq("relativeOrbitNumber_start", int(relative_orbit)))

    return collection


def load_sentinel2_raw(aoi, start_date, end_date, max_cloud=95):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
    )


def load_auxiliary_datasets(aoi):
    jrc_water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    water_occurrence = jrc_water.select("occurrence").unmask(0)
    permanent_water = water_occurrence.gt(80).rename("permanent_water")

    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    worldcover_classes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    worldcover_remap = worldcover.remap(
        worldcover_classes,
        list(range(len(worldcover_classes)))
    ).rename("WorldCover_remap")

    dem_collection = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM")
    dem = dem_collection.mosaic().setDefaultProjection(dem_collection.first().projection())
    slope = ee.Terrain.slope(dem).rename("slope")

    rivers = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(aoi)
    rivers_image = ee.Image().byte().paint(
        featureCollection=rivers,
        color=1,
        width=2
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
    }


def collection_sizes(collections, output_path=None):
    rows = [
        {"Dataset window": name, "Number of images": collection.size().getInfo()}
        for name, collection in collections.items()
    ]

    df = pd.DataFrame(rows)

    if output_path is not None:
        df.to_csv(output_path, index=False)

    return df

def get_vis_params():
    return {
        "s1_vv": {
            "min": -25,
            "max": 0
        },
        "s1_difference": {
            "min": -5,
            "max": 5,
            "palette": ["blue", "white", "red"]
        },
        "s2_true_color": {
            "bands": ["B4", "B3", "B2"],
            "min": 0,
            "max": 3000
        },
        "water_occurrence": {
            "min": 0,
            "max": 100,
            "palette": ["white", "lightblue", "blue"]
        },
        "permanent_water": {
            "min": 0,
            "max": 1,
            "palette": ["white", "blue"]
        },
        "dem": {
            "min": 0,
            "max": 500,
            "palette": ["blue", "green", "yellow", "brown", "white"]
        },
        "slope": {
            "min": 0,
            "max": 20,
            "palette": ["white", "yellow", "orange", "red"]
        },
        "worldcover": {
            "min": 0,
            "max": 10,
            "palette": [
                "006400",
                "ffbb22",
                "ffff4c",
                "f096ff",
                "fa0000",
                "b4b4b4",
                "f0f0f0",
                "0064c8",
                "0096a0",
                "00cf75",
                "fae6a0"
            ]
        },
        "rivers": {
            "min": 0,
            "max": 1,
            "palette": ["ffffff", "0000ff"]
        },
    }


def add_layers(map_obj, layer_specs):
    for spec in layer_specs:
        map_obj.addLayer(
            spec["image"],
            spec.get("vis", {}),
            spec["name"],
            spec.get("shown", True)
        )


def build_event_raster_layers(
    event_key,
    event,
    s1_vv,
    s2_rgb,
    water_occurrence,
    permanent_water,
    vis,
    include_seasonal_reference=False,
    show_s2_previews_by_default=False,
    show_water_on_event_map=False,
    seasonal_layers=None,
):
    layers = [
        {
            "image": s1_vv[event_key]["pre"],
            "vis": vis["s1_vv"],
            "name": f"S1 VV {event['name']} pre-event",
            "shown": True,
        },
        {
            "image": s1_vv[event_key]["post"],
            "vis": vis["s1_vv"],
            "name": f"S1 VV {event['name']} post-event",
            "shown": False,
        },
        {
            "image": s2_rgb[event_key]["pre"],
            "vis": vis["s2_true_color"],
            "name": f"S2 RGB {event['name']} pre-event preview",
            "shown": show_s2_previews_by_default,
        },
        {
            "image": s2_rgb[event_key]["post"],
            "vis": vis["s2_true_color"],
            "name": f"S2 RGB {event['name']} post-event preview",
            "shown": False,
        },
        {
            "image": water_occurrence.clip(event["aoi"]),
            "vis": vis["water_occurrence"],
            "name": "JRC Water Occurrence",
            "shown": show_water_on_event_map,
        },
        {
            "image": permanent_water.clip(event["aoi"]),
            "vis": vis["permanent_water"],
            "name": "Permanent Water > 80%",
            "shown": False,
        },
    ]

    if (
        event_key == "sept"
        and include_seasonal_reference
        and seasonal_layers is not None
    ):
        layers.extend([
            {
                "image": seasonal_layers["s1_vv"],
                "vis": vis["s1_vv"],
                "name": "S1 VV seasonal reference 2023",
                "shown": False,
            },
            {
                "image": seasonal_layers["s1_difference"],
                "vis": vis["s1_difference"],
                "name": "S1 VV 2024 pre - 2023 seasonal ref",
                "shown": False,
            },
            {
                "image": seasonal_layers["s2_rgb"],
                "vis": vis["s2_true_color"],
                "name": "S2 RGB seasonal reference 2023",
                "shown": False,
            },
        ])

    return layers


def make_setup_event_map(
    event_key,
    events,
    aoi_style,
    places_style,
    rivers_image,
    s1_vv,
    s2_rgb,
    water_occurrence,
    permanent_water,
    vis,
    include_seasonal_reference=False,
    show_rivers_by_default=False,
    show_s2_previews_by_default=False,
    show_water_on_event_map=False,
    seasonal_layers=None,
):
    event = events[event_key]

    raster_layers = build_event_raster_layers(
        event_key=event_key,
        event=event,
        s1_vv=s1_vv,
        s2_rgb=s2_rgb,
        water_occurrence=water_occurrence,
        permanent_water=permanent_water,
        vis=vis,
        include_seasonal_reference=include_seasonal_reference,
        show_s2_previews_by_default=show_s2_previews_by_default,
        show_water_on_event_map=show_water_on_event_map,
        seasonal_layers=seasonal_layers,
    )

    m = geemap.Map(center=event["center"], zoom=event["zoom"])

    add_layers(m, raster_layers)

    m.addLayer(
        rivers_image.clip(event["aoi"]),
        vis["rivers"],
        "HydroSHEDS rivers",
        show_rivers_by_default
    )

    m.addLayer(
        aoi_style,
        {},
        event.get("aoi_label", "AOI")
    )

    m.addLayer(
        places_style,
        {},
        "Places"
    )

    m.addLayerControl()

    return m

def make_auxiliary_map(
    center,
    zoom,
    aoi,
    aoi_styles,
    places_style,
    rivers_image,
    auxiliary,
    vis,
    default_layer="jrc",
):
    m = geemap.Map(center=center, zoom=zoom)

    layer_specs = [
        {
            "key": "jrc",
            "image": auxiliary["water_occurrence"].clip(aoi),
            "vis": vis["water_occurrence"],
            "name": "JRC Water Occurrence",
        },
        {
            "key": "permanent_water",
            "image": auxiliary["permanent_water"].clip(aoi),
            "vis": vis["permanent_water"],
            "name": "Permanent Water > 80%",
        },
        {
            "key": "worldcover",
            "image": auxiliary["worldcover_remap"].clip(aoi),
            "vis": vis["worldcover"],
            "name": "ESA WorldCover",
        },
        {
            "key": "dem",
            "image": auxiliary["dem"].clip(aoi),
            "vis": vis["dem"],
            "name": "Copernicus DEM",
        },
        {
            "key": "slope",
            "image": auxiliary["slope"].clip(aoi),
            "vis": vis["slope"],
            "name": "Slope",
        },
    ]

    for spec in layer_specs:
        m.addLayer(
            spec["image"],
            spec["vis"],
            spec["name"],
            spec["key"] == default_layer
        )

    m.addLayer(
        rivers_image.clip(aoi),
        vis["rivers"],
        "HydroSHEDS rivers",
        False
    )

    for name, style in aoi_styles.items():
        m.addLayer(style, {}, name)

    m.addLayer(places_style, {}, "Places")
    m.addLayerControl()

    return m
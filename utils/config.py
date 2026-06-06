import ee


def get_project_config():
    aoi_emilia_romagna = ee.Geometry.Rectangle([10.7, 43.7, 12.8, 45.1])
    aoi_romagna = ee.Geometry.Rectangle([11.3, 44.0, 12.5, 44.8])
    aoi_bologna = ee.Geometry.Rectangle([11.35, 44.35, 11.85, 44.85])
    aoi_sept_lugo_bagnacavallo = ee.Geometry.Rectangle([11.72, 44.30, 12.10, 44.68])
    aoi_sept_faenza = ee.Geometry.Rectangle([11.72, 44.20, 11.98, 44.38])
    aoi_sept_forli = ee.Geometry.Rectangle([11.92, 44.12, 12.18, 44.30])

    places = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([11.8833, 44.2856]), {"name": "Faenza"}),
        ee.Feature(ee.Geometry.Point([11.8000, 44.3190]), {"name": "Castel Bolognese"}),
        ee.Feature(ee.Geometry.Point([11.9100, 44.4214]), {"name": "Lugo"}),
        ee.Feature(ee.Geometry.Point([11.9781, 44.4142]), {"name": "Bagnacavallo"}),
        ee.Feature(ee.Geometry.Point([12.2035, 44.4184]), {"name": "Ravenna"}),
        ee.Feature(ee.Geometry.Point([11.3426, 44.4949]), {"name": "Bologna"}),
        ee.Feature(ee.Geometry.Point([11.3436, 44.3896]), {"name": "Pianoro"}),
        ee.Feature(ee.Geometry.Point([11.408, 44.470]), {"name": "San Lazzaro di Savena"}),
        ee.Feature(ee.Geometry.Point([11.534, 44.538]), {"name": "Budrio"}),
        ee.Feature(ee.Geometry.Point([11.361, 44.577]), {"name": "Castel Maggiore"}),
        ee.Feature(ee.Geometry.Point([11.938, 44.385]), {"name": "Cotignola"}),
        ee.Feature(ee.Geometry.Point([12.041, 44.222]), {"name": "Forli"}),
        ee.Feature(ee.Geometry.Point([12.020, 44.455]), {"name": "Traversara"}),
    ])

    dates = {
    "sept_pre": ("2024-08-20", "2024-09-15"),
    "sept_post": ("2024-09-18", "2024-09-27"),
    "sept_recovery": ("2024-09-27", "2024-10-15"),

    "oct_pre": ("2024-10-01", "2024-10-18"),
    "oct_post": ("2024-10-20", "2024-10-27"),
    "oct_recovery": ("2024-10-28", "2024-11-15"),

    "sept_pre_optical": ("2024-08-20", "2024-09-15"),
    "sept_post_optical": ("2024-09-18", "2024-10-05"),

    "oct_pre_optical": ("2024-09-15", "2024-10-18"),
    "oct_post_optical": ("2024-10-20", "2024-11-10"),

    "seasonal_ref": ("2023-08-20", "2023-09-30"), # used as reference for comparison with previous year
}

    events = {
        "sept": {
            "name": "September Lugo-Bagnacavallo-Traversara",
            "aoi": aoi_sept_lugo_bagnacavallo,
            "pre": "sept_pre",
            "post": "sept_post",
            "recovery": "sept_recovery",
            "center": [44.49, 11.91],
            "zoom": 10,
            "aoi_color": "red",
            "aoi_label": "AOI September Lugo-Bagnacavallo-Traversara",
        },
        "oct": {
            "name": "October Bologna-Pianoro",
            "aoi": aoi_bologna,
            "pre": "oct_pre",
            "post": "oct_post",
            "recovery": "oct_recovery",
            "center": [44.43, 11.35],
            "zoom": 10,
            "aoi_color": "blue",
            "aoi_label": "AOI Bologna/Pianoro",
        },
    }

    return {
        "aoi_emilia_romagna": aoi_emilia_romagna,
        "aoi_romagna": aoi_romagna,
        "aoi_bologna": aoi_bologna,
        "aoi_sept_lugo_bagnacavallo": aoi_sept_lugo_bagnacavallo,
        "aoi_sept_faenza": aoi_sept_faenza,
        "aoi_sept_forli": aoi_sept_forli,
        "places": places,
        "dates": dates,
        "events": events,

    }


SAR_PARAMS = {
    "sept": {
        "delta_threshold": -3.2, # consider flood if VV drops by more than 3.2 dB compared to pre-event
        "ratio_threshold": 0.50, # consider flood if backscatter drops by more than 50% compared to pre-event
        "vv_post_threshold": -17, # additional indicator because in some cases the pre-event backscatter is already very low
        "vh_delta_threshold": -3.0, # additional indicator to avoid false positives floods, sensitivity to more complex surfaces and can understand if there is water under vegetation
        "vh_ratio_threshold": 0.55, # consider flood if backscatter drops by more than 55% compared to pre-event in VH polarization
        "slope_threshold": 5, # exclude zones with slope greater than 5 degrees to avoid false positives in hilly areas, can cause shadowing effects that mimic flood signals
        "min_connected_pixels": 60, # consider flood only if there are at least 60 connected pixels to avoid speckle noise
        "speckle_radius": 30, # speckle noise happens when radar signal bounces multiple times before returning to the sensor
        "confidence_threshold": 3, # final mask must be confirmed by at least 3 of the previous indicators to be considered flood.
        "use_otsu_in_confidence": True, # use otsu threshold as component of confidence
        "orbit_pass_override": "DESCENDING", # same orbit pass for pre and post
        "relative_orbit_override": 95, # same orbit for pre and post
    },
    "oct": {
        "delta_threshold": -3.0,
        "ratio_threshold": 0.50,
        "vv_post_threshold": -17,
        "vh_delta_threshold": -3.0,
        "vh_ratio_threshold": 0.55,
        "slope_threshold": 5,
        "min_connected_pixels": 20,
        "confidence_threshold": 3,
        "speckle_radius": 30,
        "use_otsu_in_confidence": True,
        "orbit_pass_override": "ASCENDING",
        "relative_orbit_override": 117,  # 168 Descending
    },
}

OPTICAL_PARAMS = {
    "sept": {
        "max_cloud_pre": 95, # maximum cloud cover allowed in pre-event optical images
        "max_cloud_post": 60, # maximum cloud cover allowed in post-event optical images
        "optical_index": "MNDWI", # use MNDWI (Modified Normalized Difference Water Index) which is more sensitive to water presence than NDVI, especially in urban areas
        "optical_threshold": 0.0, # consider flood if MNDWI is greater than 0, which indicates presence of water
        "optical_delta_threshold": 0.10, # consider flood if MNDWI increases by more than 0.10 compared to pre-event, to capture new water presence
        "pre_window_override": "sept_pre_optical", 
        "post_window_override": "sept_post_optical",
        "composite_method": "median", # use median composite to reduce cloud effects, as it is more robust to outliers than mean
    },
    "oct": {
        "max_cloud_pre": 50,
        "max_cloud_post": 60,
        "optical_index": "MNDWI",
        "optical_threshold": 0.10,
        "optical_delta_threshold": 0.10,
        "pre_window_override": "oct_pre_optical",
        "post_window_override": "oct_post_optical",
        "composite_method": "median",
    },
}

RISK_PARAMS = {
    "sept": {
        "elevation_percentile": 25, # identify low areas if the elevation is in the lowest 25% of the area
        "slope_threshold": 2, # identify flat areas
        "river_distance_threshold": 1000, # identify areas within 1 km from rivers, which are more likely to flood
        "historical_water_threshold": 10, # identify areas that have been water at least 10% of the time in the past
        "population_threshold": 1,
        "population_weight": 1,
    },
    "oct": {
        "elevation_percentile": 25,
        "slope_threshold": 2,
        "river_distance_threshold": 1000,
        "historical_water_threshold": 10,
        "population_threshold": 1,
        "population_weight": 1,
    },
}

WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland", # agricultural damage
    50: "Built-up", # urban damage
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

EXPOSED_WORLDCOVER_CLASSES = [40, 50] # for risk assessment, we focus on cropland and built-up areas

VULNERABILITY_CLASSES = { # translate class values into labels
    0: "No / very low vulnerability", 
    1: "Low vulnerability",
    2: "Medium vulnerability",
    3: "High vulnerability",
}

RISK_CLASSES = { # translate class values into labels
    0: "No / very low risk",
    1: "Low risk",
    2: "Medium risk",
    3: "High risk",
}

RISK_CHANGE_CLASSES = {
    0: "Unchanged",
    1: "Increased by 1 class",
    2: "Increased by 2 classes",
    3: "Increased by 3 classes",
}
import ee


def get_project_config():
    aoi_emilia_romagna = ee.Geometry.Rectangle([10.7, 43.7, 12.8, 45.1])
    aoi_romagna = ee.Geometry.Rectangle([11.3, 44.0, 12.5, 44.8])
    aoi_bologna = ee.Geometry.Rectangle([11.15, 44.15, 11.65, 44.65])
    aoi_sept_lugo_bagnacavallo = ee.Geometry.Rectangle([11.80, 44.34, 12.08, 44.50])
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
    "sept_post": ("2024-09-18", "2024-09-26"),
    "sept_recovery": ("2024-09-27", "2024-10-15"),

    "oct_pre": ("2024-10-01", "2024-10-18"),
    "oct_post": ("2024-10-20", "2024-10-27"),
    "oct_recovery": ("2024-10-28", "2024-11-15"),

    # Wider optical windows used only for Sentinel-2 compositing
    "sept_pre_optical": ("2024-08-20", "2024-09-15"),
    "sept_post_optical": ("2024-09-18", "2024-10-05"),

    "oct_pre_optical": ("2024-09-15", "2024-10-18"),
    "oct_post_optical": ("2024-10-20", "2024-11-10"),

    "seasonal_ref": ("2023-08-20", "2023-09-30"),
}

    events = {
        "sept": {
            "name": "September Lugo-Bagnacavallo-Traversara",
            "aoi": aoi_sept_lugo_bagnacavallo,
            "pre": "sept_pre",
            "post": "sept_post",
            "recovery": "sept_recovery",
            "center": [44.42, 11.95],
            "zoom": 11,
            "aoi_color": "red",
            "aoi_label": "AOI Lugo-Bagnacavallo-Traversara",
        },
        # "sept": {
        #     "name": "September Faenza",
        #     "aoi": aoi_sept_faenza,
        #     "pre": "sept_pre",
        #     "post": "sept_post",
        #     "recovery": "sept_recovery",
        #     "center": [44.29, 11.88],
        #     "zoom": 11,
        #     "aoi_color": "red",
        #     "aoi_label": "AOI Faenza",
        # },
        # "sept": {
        #     "name": "September Forli",
        #     "aoi": aoi_sept_forli,
        #     "pre": "sept_pre",
        #     "post": "sept_post",
        #     "recovery": "sept_recovery",
        #     "center": [44.22, 12.04],
        #     "zoom": 11,
        #     "aoi_color": "red",
        #     "aoi_label": "AOI Forli",
        # },
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
        "delta_threshold": -2.5,
        "ratio_threshold": 0.7,
        "vv_post_threshold": -17,
        "slope_threshold": 5,
        "min_connected_pixels": 10,
        "orbit_pass_override": "DESCENDING",
        "relative_orbit_override": 95,
        "use_otsu_in_confidence": False,
    },
    "oct": {
        "delta_threshold": -2.5,
        "ratio_threshold": 0.7,
        "vv_post_threshold": -17,
        "slope_threshold": 5,
        "min_connected_pixels": 10,
        "orbit_pass_override": "ASCENDING",
        "relative_orbit_override": 117,  # 168 Descending
        "use_otsu_in_confidence": False,
    },
}

OPTICAL_PARAMS = {
    "sept": {
        "max_cloud_pre": 95,
        "max_cloud_post": 100,
        "optical_index": "MNDWI",
        "optical_threshold": 0.0,
        "pre_window_override": "sept_pre_optical",
        "post_window_override": "sept_post_optical",
        "composite_method": "median",
    },
    "oct": {
        "max_cloud_pre": 95,
        "max_cloud_post": 100,
        "optical_index": "MNDWI",
        "optical_threshold": 0.0,
        "pre_window_override": "oct_pre_optical",
        "post_window_override": "oct_post_optical",
        "composite_method": "median",
    },
}

RISK_PARAMS = {
    "sept": {
        "elevation_percentile": 25,
        "slope_threshold": 2,
        "river_distance_threshold": 1000,
        "historical_water_threshold": 10,
        "roi_buffer_m": 3000,
    },
    "oct": {
        "elevation_percentile": 25,
        "slope_threshold": 2,
        "river_distance_threshold": 1000,
        "historical_water_threshold": 10,
        "roi_buffer_m": 3000,
    },
}
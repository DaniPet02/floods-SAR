# Flood Mapping and Risk Assessment in Emilia-Romagna

## Project Overview

This project implements a Google Earth Engine workflow for mapping flood-affected areas in Emilia-Romagna using Earth Observation data.

The analysis focuses on two flood events in 2024 and combines Sentinel-1 SAR, Sentinel-2 optical imagery and auxiliary geographic datasets to detect possible flooded areas and produce final hazard, vulnerability and risk maps.

The workflow is designed to be simple, reproducible and organized in a clear notebook structure, with diagnostic maps, summary tables and exported results.

## Main Objective

The main objective is to identify areas potentially affected by flooding and interpret them in relation to the characteristics of the territory.

The project does not only map possible flood evidence from satellite data, but also evaluates where this evidence overlaps with vulnerable areas.

In short, the workflow answers two main questions:

- Where does the satellite detect possible flooding?
- Where does this flood signal become more relevant because the territory is vulnerable?

## Events Analyzed

The notebook can analyze two events through the global variable MONTH.

### September Event

The September configuration focuses on the Lugo, Bagnacavallo and Traversara area.

This event mainly represents a flatter agricultural and floodplain context, where Sentinel-1 SAR flood mapping is expected to be more spatially coherent.

### October Event

The October configuration focuses on the Bologna and Pianoro area.

This event includes a more complex and hilly territory, where slope and terrain morphology are more important for interpreting the SAR signal and the final risk map.

Changing MONTH automatically changes the event area, dates, SAR parameters, optical parameters and risk parameters.

## Project Structure

The project is organized into a main notebook and a set of utility files:

    floods-SAR/
    │
    ├── flood_analysis.ipynb
    │
    ├── utils/
    │   ├── config.py
    │   ├── utils_setup.py
    │   ├── utils_SAR.py
    │   ├── utils_optical.py
    │   └── utils_final.py
    │
    └── outputs/
        ├── figures/
        └── tables/

- `flood_analysis.ipynb`: main notebook that runs the complete workflow.
- `utils/config.py`: contains event configuration, dates, AOIs and processing parameters.
- `utils/utils_setup.py`: contains general utilities for loading data, visualization, area calculation and exports.
- `utils/utils_SAR.py`: contains the Sentinel-1 SAR flood mapping pipeline.
- `utils/utils_optical.py`: contains the Sentinel-2 optical comparison pipeline.
- `utils/utils_final.py`: contains the final hazard, vulnerability and risk workflow.
- `outputs/figures/`: stores exported PNG figures.
- `outputs/tables/`: stores exported CSV tables.


## Pipeline Overview

### SAR Pipeline

The SAR pipeline uses Sentinel-1 images before and after the flood event.

It creates pre-event and post-event composites, calculates radar backscatter changes, builds several candidate flood masks and combines them into a SAR confidence map.

The SAR result is the main source for flood detection because Sentinel-1 can observe the surface during both day and night and is not strongly limited by cloud cover.

### Optical Pipeline

The optical pipeline uses Sentinel-2 Surface Reflectance images.

It removes cloudy or invalid pixels, calculates water and vegetation indices such as NDWI, MNDWI and NDVI, creates pre-event and post-event composites and detects optically visible water.

This part is mainly used to support and compare the SAR result, because Sentinel-2 provides useful spectral information but is limited by clouds, shadows and image availability.

### Risk Pipeline

The final pipeline converts the SAR flood evidence into hazard classes and combines them with territorial vulnerability.

Vulnerability is derived from factors such as elevation, slope, distance from rivers, historical water presence and land cover.

The final risk map is obtained by combining hazard and vulnerability, so high risk appears where flood evidence overlaps with vulnerable territory.

## Datasets Used

The project uses the following main datasets:

- Sentinel-1 SAR: main flood detection source.
- Sentinel-2 Surface Reflectance: optical comparison and water indices.
- JRC Global Surface Water: permanent water mask.
- Copernicus DEM: elevation and slope.
- ESA WorldCover: land cover and exposure.
- HydroSHEDS rivers: river proximity and vulnerability.

## Outputs

The project produces both visual and quantitative outputs.

### Saved Images

The workflow can export PNG figures such as:

- SAR change maps
- SAR flood masks
- Sentinel-2 optical water maps
- SAR vs optical comparison maps
- hazard vs risk maps

These images are useful for the final report and presentation.

### Saved Tables

The workflow can also save CSV tables such as:

- number of available images
- SAR area statistics
- optical processing summary
- SAR-optical agreement table
- hazard area table
- vulnerability area table
- risk area table
- land-cover exposure table

These tables help document the results quantitatively.

## Strengths

- The same workflow can be reused for both September and October events.
- Sentinel-1 SAR provides robust flood evidence even with cloud cover.
- Sentinel-2 adds useful optical and spectral interpretation.
- Multiple SAR indicators are combined instead of relying on a single threshold.
- Permanent water, slope and small noisy pixel clusters are removed.
- The final result includes not only flood detection, but also risk interpretation.
- The project produces both maps and numerical tables.

## Limitations

- SAR results can be affected by speckle, terrain effects, vegetation, urban double bounce and threshold choices.
- Sentinel-2 results can be affected by clouds, shadows, haze and date mismatch with the flood peak.
- The risk map is a simplified spatial indicator, not a complete socio-economic risk assessment.
- The final maps should be interpreted as satellite-based estimates, not as ground-truth validation.

## Summary

This project provides a complete Earth Observation workflow for flood analysis in Emilia-Romagna.

The workflow first selects the event, then loads Sentinel-1, Sentinel-2 and auxiliary geographic data. Sentinel-1 is used to detect possible flooded areas, Sentinel-2 is used for optical comparison, and geographic layers are used to build vulnerability.

Finally, hazard and vulnerability are combined to produce a risk map.

The main idea is:

Satellite flood evidence plus territorial vulnerability gives a more meaningful interpretation of flood risk.
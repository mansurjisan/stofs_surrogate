# STOFS2D Offshore Timeseries Generation Workflow

## Overview

This document describes the workflow for generating offshore timeseries plots from STOFS2D fort.63 style NetCDF output files. The plots compare bias-corrected (CWL) vs non-bias-corrected (noanomaly) water level data at offshore locations along the U.S. East and West coasts.

## Input Data

### NetCDF Files
The input files are located in `/mnt/d/STOFS2D-Analysis/MAXELE_PLOTS/FIELD2DPLOTS/20251122/`:

| Cycle | Bias-Corrected File | Non-Bias-Corrected File |
|-------|---------------------|-------------------------|
| 00Z | `stofs_2d_glo.t00z.fields.cwl.nc` | `stofs_2d_glo.t00z.fields.cwl.noanomaly.nc` |
| 06Z | `stofs_2d_glo.t06z.fields.cwl.nc` | `stofs_2d_glo.t06z.fields.cwl.noanomaly.nc` |
| 12Z | `stofs_2d_glo.t12z.fields.cwl.nc` | `stofs_2d_glo.t12z.fields.cwl.noanomaly.nc` |
| 18Z | `stofs_2d_glo.t18z.fields.cwl.nc` | `stofs_2d_glo.t18z.fields.cwl.noanomaly.nc` |

### File Structure
- Each NetCDF file contains ~12.7 million nodes
- Variables: `zeta` (water level), `x` (longitude), `y` (latitude), `time`
- Time dimension: ~186 hourly timesteps (~8 days of forecast)

## Script: fort63_simple_timeseries.py

### Location
`/mnt/d/STOFS2D-Analysis/fort63_simple_timeseries.py`

### Key Features
- Uses KDTree for efficient nearest-neighbor spatial queries
- Finds nearest CO-OPS tide gauge station for each location
- Generates dual-panel plots: timeseries (left) + location map (right)
- Supports both predefined coastal locations and custom locations
- Automatically combines all PNG plots into a single PDF

### Dependencies
```
numpy
xarray
scipy (KDTree)
matplotlib
cartopy
pandas
tqdm
```

### Command Line Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `--cwl` | Path to bias-corrected NetCDF file | Yes |
| `--noanomaly` | Path to non-bias-corrected NetCDF file | Yes |
| `--output-dir` | Output directory for plots | Yes |
| `--coasts` | Which coasts to process: `east`, `west`, or `all` | No |
| `--custom-locations` | Custom locations as "Name:lon,lat" | No |
| `--output-pdf` | Output PDF filename | No |

## Offshore Locations

### 47 Expanded Offshore Locations (Built-in)

**East Coast (23 locations):**
- Offshore Maine 1, 2, 3
- Georges Bank 1, 2
- Offshore Cape Cod 1, 2
- Mid-Atlantic Bight 1, 2, 3
- Delaware Bay Offshore
- Offshore New Jersey
- Offshore Virginia 1, 2
- Cape Hatteras Offshore 1, 2
- Offshore Carolinas 1, 2, 3
- South Atlantic Bight 1, 2
- Offshore Georgia
- Offshore Florida Atlantic

**West Coast (24 locations):**
- Strait of Juan de Fuca 1, 2
- Offshore Washington 1, 2, 3
- Offshore Oregon 1, 2, 3, 4
- Offshore Humboldt 1, 2
- Offshore Mendocino
- Offshore Point Reyes 1, 2
- Monterey Bay Offshore 1, 2
- Offshore Big Sur
- Offshore Morro Bay
- Santa Barbara Channel 1, 2
- San Pedro Channel 1, 2
- Offshore San Diego 1, 2

### 18 Overview Map Locations (Custom)

**East Coast (10 locations):**
| Name | Longitude | Latitude |
|------|-----------|----------|
| Gulf of Maine 50km Offshore | -69.0 | 43.5 |
| Georges Bank | -67.5 | 41.2 |
| Nantucket Shoals | -69.5 | 40.8 |
| NY Bight 60km Offshore | -73.0 | 39.8 |
| Mid-Atlantic Shelf | -74.5 | 38.0 |
| Virginia Shelf Break | -74.5 | 36.8 |
| Cape Hatteras 40km Offshore | -75.0 | 35.2 |
| Onslow Bay NC | -77.0 | 34.0 |
| Charleston Bump Offshore | -79.0 | 32.0 |
| Florida Straits North | -80.0 | 26.5 |

**West Coast (8 locations):**
| Name | Longitude | Latitude |
|------|-----------|----------|
| Juan de Fuca Canyon | -125.0 | 48.3 |
| WA Shelf 50km Offshore | -125.0 | 47.0 |
| Columbia River Plume | -124.5 | 46.2 |
| Oregon Shelf Break | -125.0 | 44.5 |
| Cape Blanco 40km Offshore | -124.8 | 42.8 |
| Point Arena 30km Offshore | -124.0 | 39.0 |
| CA Current Central | -123.0 | 37.0 |
| SoCal Bight 50km Offshore | -118.0 | 33.0 |

## Execution Workflow

### Step 1: Generate 47 Expanded Locations

Run for each cycle (00z, 06z, 18z) - can be run in parallel:

```bash
# 00Z Cycle
/home/mjisan/miniconda3/envs/xesmf_env/bin/python /mnt/d/STOFS2D-Analysis/fort63_simple_timeseries.py \
  --cwl "/mnt/d/STOFS2D-Analysis/MAXELE_PLOTS/FIELD2DPLOTS/20251122/stofs_2d_glo.t00z.fields.cwl.nc" \
  --noanomaly "/mnt/d/STOFS2D-Analysis/MAXELE_PLOTS/FIELD2DPLOTS/20251122/stofs_2d_glo.t00z.fields.cwl.noanomaly.nc" \
  --coasts east west \
  --output-dir "/mnt/d/STOFS2D-Analysis/offshore_timeseries_20251122_00z"

# 06Z Cycle
/home/mjisan/miniconda3/envs/xesmf_env/bin/python /mnt/d/STOFS2D-Analysis/fort63_simple_timeseries.py \
  --cwl "/mnt/d/STOFS2D-Analysis/MAXELE_PLOTS/FIELD2DPLOTS/20251122/stofs_2d_glo.t06z.fields.cwl.nc" \
  --noanomaly "/mnt/d/STOFS2D-Analysis/MAXELE_PLOTS/FIELD2DPLOTS/20251122/stofs_2d_glo.t06z.fields.cwl.noanomaly.nc" \
  --coasts east west \
  --output-dir "/mnt/d/STOFS2D-Analysis/offshore_timeseries_20251122_06z"

# 18Z Cycle
/home/mjisan/miniconda3/envs/xesmf_env/bin/python /mnt/d/STOFS2D-Analysis/fort63_simple_timeseries.py \
  --cwl "/mnt/d/STOFS2D-Analysis/MAXELE_PLOTS/FIELD2DPLOTS/20251122/stofs_2d_glo.t18z.fields.cwl.nc" \
  --noanomaly "/mnt/d/STOFS2D-Analysis/MAXELE_PLOTS/FIELD2DPLOTS/20251122/stofs_2d_glo.t18z.fields.cwl.noanomaly.nc" \
  --coasts east west \
  --output-dir "/mnt/d/STOFS2D-Analysis/offshore_timeseries_20251122_18z"
```

### Step 2: Add 18 Overview Map Locations

Run custom locations for each cycle to add to the same output directories:

```bash
# 00Z Cycle - Custom Locations
/home/mjisan/miniconda3/envs/xesmf_env/bin/python /mnt/d/STOFS2D-Analysis/fort63_simple_timeseries.py \
  --cwl "/mnt/d/STOFS2D-Analysis/MAXELE_PLOTS/FIELD2DPLOTS/20251122/stofs_2d_glo.t00z.fields.cwl.nc" \
  --noanomaly "/mnt/d/STOFS2D-Analysis/MAXELE_PLOTS/FIELD2DPLOTS/20251122/stofs_2d_glo.t00z.fields.cwl.noanomaly.nc" \
  --output-dir "/mnt/d/STOFS2D-Analysis/offshore_timeseries_20251122_00z" \
  --custom-locations \
    "Gulf_of_Maine_50km_Offshore:-69.0,43.5" \
    "Georges_Bank:-67.5,41.2" \
    "Nantucket_Shoals:-69.5,40.8" \
    "NY_Bight_60km_Offshore:-73.0,39.8" \
    "Mid-Atlantic_Shelf:-74.5,38.0" \
    "Virginia_Shelf_Break:-74.5,36.8" \
    "Cape_Hatteras_40km_Offshore:-75.0,35.2" \
    "Onslow_Bay_NC:-77.0,34.0" \
    "Charleston_Bump_Offshore:-79.0,32.0" \
    "Florida_Straits_North:-80.0,26.5" \
    "Juan_de_Fuca_Canyon:-125.0,48.3" \
    "WA_Shelf_50km_Offshore:-125.0,47.0" \
    "Columbia_River_Plume:-124.5,46.2" \
    "Oregon_Shelf_Break:-125.0,44.5" \
    "Cape_Blanco_40km_Offshore:-124.8,42.8" \
    "Point_Arena_30km_Offshore:-124.0,39.0" \
    "CA_Current_Central:-123.0,37.0" \
    "SoCal_Bight_50km_Offshore:-118.0,33.0"
```

Repeat for 06Z and 18Z cycles with appropriate file paths.

### Step 3: Rename PDFs

The script generates `offshore_timeseries.pdf` by default. Rename to requested format:

```bash
mv /mnt/d/STOFS2D-Analysis/offshore_timeseries_20251122_00z/offshore_timeseries.pdf \
   /mnt/d/STOFS2D-Analysis/offshore_timeseries_20251122_00z/offshore_timeseries_20251122_00Z.pdf

mv /mnt/d/STOFS2D-Analysis/offshore_timeseries_20251122_06z/offshore_timeseries.pdf \
   /mnt/d/STOFS2D-Analysis/offshore_timeseries_20251122_06z/offshore_timeseries_20251122_06Z.pdf

mv /mnt/d/STOFS2D-Analysis/offshore_timeseries_20251122_18z/offshore_timeseries.pdf \
   /mnt/d/STOFS2D-Analysis/offshore_timeseries_20251122_18z/offshore_timeseries_20251122_18Z.pdf
```

## Output Structure

### Directory Layout
```
/mnt/d/STOFS2D-Analysis/
├── offshore_timeseries_20251122_00z/
│   ├── offshore_timeseries_20251122_00Z.pdf    # Combined PDF (65 pages)
│   ├── Offshore_Maine_1_timeseries.png
│   ├── Offshore_Maine_2_timeseries.png
│   ├── ... (47 expanded location PNGs)
│   ├── Gulf_of_Maine_50km_Offshore_timeseries.png
│   ├── Georges_Bank_timeseries.png
│   └── ... (18 overview location PNGs)
├── offshore_timeseries_20251122_06z/
│   ├── offshore_timeseries_20251122_06Z.pdf
│   └── ... (65 PNG files)
├── offshore_timeseries_20251122_18z/
│   ├── offshore_timeseries_20251122_18Z.pdf
│   └── ... (65 PNG files)
└── offshore_timeseries_overviewmap_20251122_12z/
    ├── offshore_timeseries.pdf
    └── ... (18 PNG files - original 12z overview)
```

### Output Files Summary

| Cycle | Directory | PNG Count | PDF Size |
|-------|-----------|-----------|----------|
| 00Z | `offshore_timeseries_20251122_00z` | 65 | ~8.3 MB |
| 06Z | `offshore_timeseries_20251122_06z` | 65 | ~8.3 MB |
| 18Z | `offshore_timeseries_20251122_18z` | 65 | ~8.2 MB |

## Plot Description

Each timeseries plot contains:

### Left Panel (Timeseries)
- X-axis: Date/Time (UTC)
- Y-axis: Water Level (meters)
- Red line: Without Bias Correction (noanomaly)
- Blue line: With Bias Correction (CWL)
- Title: Location name with coordinates

### Right Panel (Map)
- Red star: Timeseries extraction location
- Green star: Nearest CO-OPS tide gauge station
- Shows distance to nearest CO-OPS station
- Cartopy coastlines and bathymetry

## Processing Time

Approximate processing times on the current system:
- 47 expanded locations: ~3-4 minutes per cycle
- 18 custom locations: ~2 minutes per cycle
- Total for all 3 cycles (parallel): ~5-6 minutes

## Troubleshooting

### Common Issues

1. **Memory errors**: The script loads ~12.7M nodes into memory. Ensure sufficient RAM (~8GB+).

2. **Missing CO-OPS station**: Some offshore locations may not have a nearby CO-OPS station within 1.5 degrees. The script will show a warning but continue.

3. **File not found**: Verify the input NetCDF file paths are correct and files exist.

### Environment Setup

```bash
# Activate conda environment
conda activate xesmf_env

# Or use full path to Python
/home/mjisan/miniconda3/envs/xesmf_env/bin/python
```

## Related Files

- `fort63_simple_timeseries.py` - Main script
- `offshore_timeseries_locations_east_coast.png` - East coast location map
- `offshore_timeseries_locations_west_coast.png` - West coast location map
- `offshore_timeseries_all_locations_overview.png` - Combined overview map

## Contact

For questions about this workflow, contact the STOFS2D analysis team.

---
*Document created: 2025-12-03*
*Last updated: 2025-12-03*

# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed - `scripts/layers.py`

#### Robustness Hardening (CRS, Error Handling, Magic Numbers)

##### CRS and Spatial Logic
- **`assemble_data`**: Replaced the fragile `if crs == 'EPSG:4326' and res == 30` block
  (which left `res_deg` undefined for any other CRS or resolution) with a robust
  `pyproj.CRS.is_geographic` check.  Geographic CRS inputs now compute the
  arc-degree resolution from the `METRES_PER_DEGREE_AT_45_LAT` constant; projected
  CRS inputs pass the metric resolution directly.  Added `from pyproj import CRS as
  ProjCRS` import.
- **`get_uavsar_incidence`**: Removed the hardcoded `.to_crs('EPSG:4326')` when
  deriving AOI bounds for raster slicing.  Bounds are now taken directly from the
  input `crs` (via `GeoSeries.total_bounds`), which is consistent with the CRS
  assumed for the raster coordinates.
- **`get_topo_layers`**: Replaced the brittle string comparison
  `dem.rio.crs == 'EPSG:4326'` with `ProjCRS.from_user_input(dem.rio.crs).is_geographic`.
  Any geographic CRS (not only EPSG:4326) now triggers the reprojection to
  `DEM_METRIC_CRS` before computing topographic derivatives.

##### Error Handling
- **`assemble_data`**: Wrapped `py3dep.get_dem` in a try/except block; errors are
  logged at ERROR level before re-raising.
- **`get_nlcd_layers`**: Wrapped `gh.nlcd_bygeom` in a try/except block; errors are
  logged before re-raising.
- **`get_snow_climatology`**: Wrapped `xr.open_dataset` in a try/except block; errors
  are logged before re-raising.
- **`get_snow_layers`**: Wrapped `earthaccess.search_data` and `earthaccess.open` in
  separate try/except blocks.  Per-granule processing is now done in individual
  try/except loops so that a single bad SWE or SD granule is skipped with a WARNING
  rather than aborting the whole pipeline.  A `RuntimeError` is raised (with a clear
  message) only if *no* granules could be processed at all.
- **`process_ucla_granule`**: Wrapped `xr.open_dataset` in a try/except block.
  Added an explicit `ValueError` if the Water Year regex fails to match (previously
  this would raise an `AttributeError` on `None`).  Updated the docstring to
  document the new raised exceptions.
- **`get_aorc_layers`**: Wrapped `xr.open_mfdataset` in a try/except block; errors
  are logged before re-raising.

##### Magic Numbers → Named Constants
Added the following module-level constants (with docstrings) to eliminate
all magic numbers from function bodies:

| Constant | Value | Replaces |
|---|---|---|
| `METRES_PER_DEGREE_AT_45_LAT` | `111320 × cos 45°` | `0.000381807117` / 30 |
| `DEM_METRIC_CRS` | `'EPSG:5070'` | hard-coded string in `get_topo_layers` |
| `TOPO_PAD_PIXELS` | `1` | `pad(y=1, x=1)` / `isel(y=slice(1, -1))` |
| `SPATIAL_CHUNK_SIZE` | `25` | `'y': 25, 'x': 25` chunk dict |
| `KELVIN_TO_CELSIUS_OFFSET` | `273.15` | `- 273.15` in temperature conversion |
| `BLOWING_SNOW_WIND_THRESHOLD_MS` | `6.0` | `wind_speed > 6.0` |
| `MAX_FREEZE_THAW_CYCLES` | `100` | `.where(...<= 100, ...)` |
| `MAX_HOURS_BLOWING_SNOW` | `700` | `.where(...<= 700, ...)` |
| `UCLA_POSTERIOR_STATS_INDEX` | `2` | `.isel(Stats=2)` |
| `SNOW_CLIMATOLOGY_FILENAME` | `'SnowClass_NA_300m_...'` | hard-coded path fragment |
| `AORC_S3_BASE_URL` | `'noaa-nws-aorc-v1-1-1km'` | hard-coded string + removed spurious double `s3://` |

##### Other Improvements
- `get_snow_climatology`: Replaced `fp + fname` string concatenation (with a
  leading `/` in `fname`) with `os.path.join(fp, fname)` to avoid double-slash
  path issues.
- `get_topo_layers`: DEM edge-trimming slices now use `TOPO_PAD_PIXELS` as the
  slice boundary instead of the literal `1`.

#### Logging Architecture
- Added `import logging` and instantiated a module-level logger with
  `logger = logging.getLogger(__name__)`.
- Replaced every `print()` statement with an appropriate `logging` call:
  - Pipeline timing/progress messages → `logger.info()`
  - Missing-data warnings → `logger.warning()`
- Added `logger.debug()` calls throughout to track dataset shapes,
  file paths resolved, and intermediate pipeline steps.

#### Import Cleanup
- Removed unused top-level `import shapely` (bare module was never
  referenced; `Polygon` is already imported from `shapely.geometry`).
- Removed unused `import fsspec` (only referenced in dead commented-out
  code blocks).
- Moved all mid-file `import` statements to the top of the module:
  `import xrspatial as xrs`, `import earthaccess`, `import re`,
  `import pandas as pd`, `import s3fs`.
- Removed duplicate local imports inside `get_nlcd_layers`
  (`import pygeohydro as gh`, `import geopandas as gpd`, `import os`)
  that were already present at the module level.
- Added `from datetime import date` and
  `from typing import Dict, List, Optional, Set, Tuple, Union` for type
  annotations.
- Added `import math` and `from pyproj import CRS as ProjCRS` for
  CRS-agnostic resolution computation and geographic CRS detection.

#### Documentation
- Added complete NumPy-style docstrings to all functions that were
  missing them: `get_snow_climatology`, `get_topo_layers`,
  `get_snow_layers`, `get_snow_metrics`, `process_ucla_granule`,
  `get_years`.
- Updated the existing docstrings for `get_uavsar_coherence`,
  `get_uavsar_incidence`, `get_nlcd_layers`, `get_aorc_layers`, and
  `get_aorc_metrics` to use consistent NumPy style with accurate
  parameter and return-type descriptions.
- Updated the `assemble_data` docstring to match the current function
  signature and clarify the `fp_dest=None` behaviour.
- Updated `process_ucla_granule` docstring to document newly raised
  exceptions.

#### Type Hints
- Added full Python type annotations to all function signatures:
  `assemble_data`, `get_uavsar_coherence`, `get_uavsar_incidence`,
  `get_nlcd_layers`, `get_snow_climatology`, `get_topo_layers`,
  `get_snow_layers`, `get_snow_metrics`, `process_ucla_granule`,
  `get_aorc_layers`, `get_aorc_metrics`, `get_years`.

#### Dead Code Removal
- Removed the large commented-out legacy implementation of
  `get_uavsar_coherence` (≈ 46 lines).
- Removed all other dead commented-out code blocks scattered throughout
  the file (debug `print` calls, superseded logic, stale `TODO`
  comments, leftover iteration alternatives).
- Removed the unused `metrics` list in `assemble_data` (was defined
  with a comment "not currently using!" and never passed to any
  function).
- Removed the unused `bounds` variable in `get_snow_layers`.
- Removed the unused `ds = xr.Dataset()` initialisation before the
  `ds_list` loop in `get_snow_layers` and `get_aorc_layers`.

#### Constraints Respected
- No inputs, outputs, or internal math/logic of any existing function
  were changed.

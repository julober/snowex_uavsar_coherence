# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed - `scripts/layers.py`

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

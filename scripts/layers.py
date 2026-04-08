import logging
import math
import os
import re
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import earthaccess
import geopandas as gpd
import numpy as np
import pandas as pd
import py3dep
import pygeohydro as gh
import rioxarray as rxa
import s3fs
import xarray as xr
import xrspatial as xrs
from pyproj import CRS as ProjCRS
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import transform

from validation import validate_aoi, validate_date_pairs, make_reference_grid, validate_alignment

logger = logging.getLogger(__name__)

# =============================================================================
# MODULE-LEVEL CONSTANTS
# =============================================================================

# ── Spatial / CRS ─────────────────────────────────────────────────────────────

METRES_PER_DEGREE_AT_45_LAT: float = 111320.0 * math.cos(math.radians(45.0))
"""Approximate metres per degree, computed at 45° latitude.

Used to convert a metric resolution (metres) to arc-degrees when building a
reference grid in a geographic CRS.  This is an approximation that is accurate
for mid-latitude study areas (~30°–60° N/S) but will over- or under-estimate
the degree resolution at polar or equatorial latitudes respectively.
"""

DEM_METRIC_CRS: str = 'EPSG:5070'
"""Projected CRS (Conus Albers) used for accurate metric-based topographic derivatives."""

TOPO_PAD_PIXELS: int = 1
"""Number of pixels to pad the DEM edges before computing slope/aspect/curvature."""

SPATIAL_CHUNK_SIZE: int = 25
"""Chunk size (pixels) along x and y dimensions for Zarr output."""

# ── Temperature ────────────────────────────────────────────────────────────────

KELVIN_TO_CELSIUS_OFFSET: float = 273.15
"""Offset to convert temperatures from Kelvin to Celsius (K − offset = °C)."""

# ── Wind / precipitation thresholds ───────────────────────────────────────────

BLOWING_SNOW_WIND_THRESHOLD_MS: float = 6.0
"""Wind speed threshold (m s⁻¹) above which blowing snow is assumed to occur."""

MAX_FREEZE_THAW_CYCLES: int = 100
"""Upper clamp for the freeze–thaw cycle count per coherence interval."""

MAX_HOURS_BLOWING_SNOW: int = 700
"""Upper clamp for the hours-blowing-snow count per coherence interval (~29 days × 24 h)."""

# ── Dataset-specific ──────────────────────────────────────────────────────────

UCLA_POSTERIOR_STATS_INDEX: int = 2
"""Index along the Stats dimension of UCLA WUS_UCLA_SR granules selecting the posterior mean."""

SNOW_CLIMATOLOGY_FILENAME: str = 'SnowClass_NA_300m_10.0arcsec_2021_v01.0.nc'
"""Expected filename for the NSIDC-0768 seasonal snow classification dataset."""

AORC_S3_BASE_URL: str = 'noaa-nws-aorc-v1-1-1km'
"""Public S3 bucket name for the NOAA AORC v1.1 1-km dataset."""


def assemble_data(
    aoi: Polygon,
    date_pairs: List[Tuple[date, date]],
    flight_ids: List[str],
    fp_dest: Optional[str],
    fp_coh: str = '../data/coherence/',
    fp_inc: str = '../data/inc_angle/',
    fp_snowclimate: str = '../data/NSIDC-0768/',
    fp_ann: Optional[str] = None,
    crs: str = 'EPSG:4326',
    res: int = 30,
    overwrite: bool = False,
) -> xr.Dataset:
    """
    Assemble all data layers needed for the coherence model.

    Can handle all flights from a particular flight path. The caller is
    responsible for ensuring that ``date_pairs`` corresponds to actual UAVSAR
    acquisition dates.

    The resulting dataset is written to a ``.zarr`` store with the following
    logical layout::

        location_name/
            static/
            dynamic/
                date1_date2/
                date1_date3/
                date2_date3/
            .zmetadata

    Parameters
    ----------
    aoi:
        Area of interest as a Shapely Polygon in the coordinate system
        specified by ``crs``.
    date_pairs:
        List of ``(start_date, end_date)`` tuples defining the UAVSAR
        coherence intervals.
    flight_ids:
        List of UAVSAR flight-heading identifiers (e.g. ``['08508', '26505']``).
    fp_dest:
        Destination path for the output ``.zarr`` store.  Pass ``None`` to
        skip writing to disk and only return the in-memory dataset.
    fp_coh:
        Directory containing per-flight coherence ``.tif`` files.
    fp_inc:
        Directory containing pre-calculated incidence angle ``.tif`` files.
    fp_snowclimate:
        Directory containing the NSIDC-0768 snow-climatology NetCDF file.
    fp_ann:
        Optional directory containing UAVSAR ``.ann`` annotation files.  When
        provided, key flight metadata (e.g. start time, average yaw/pitch) are
        read and stored as global attributes in the output Zarr store to
        improve provenance tracking and CF/ACDD compliance.
    crs:
        Coordinate reference system string (default ``'EPSG:4326'``).
    res:
        Target spatial resolution in metres (default ``30``).
    overwrite:
        When ``True`` an existing ``.zarr`` store at ``fp_dest`` is
        overwritten; when ``False`` the write will fail if the store already
        exists.

    Returns
    -------
    xr.Dataset
        Merged dataset containing all assembled data layers.
    """

    logger.info("Starting data assembly for flights %s", flight_ids)

    # 1. Validate inputs
    aoi = validate_aoi(aoi)
    date_pairs = validate_date_pairs(date_pairs)
    aoi_gdf = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[aoi])
    logger.debug("AOI validated; %d date pairs provided", len(date_pairs))

    # 2. Create reference grid
    crs_obj = ProjCRS.from_user_input(crs)
    if crs_obj.is_geographic:
        # Convert the requested metric resolution to arc-degrees at ~45° latitude.
        res_grid = res / METRES_PER_DEGREE_AT_45_LAT
        ref = make_reference_grid(aoi=aoi, crs=crs, resolution=res_grid)
    else:
        # Projected CRS: resolution is already in metres.
        wgs84 = ProjCRS('EPSG:4326')
        res_grid = float(res)
        ref = make_reference_grid(aoi=aoi, crs=crs, resolution=res_grid)
        aoi_proj = aoi 

        # get aoi in WGS84 for APIs
        project = Transformer.from_crs(crs_obj, wgs84, always_xy=True).transform
        aoi = transform(project, aoi_proj)
        crs = 'EPSG:4326'

    # ref = make_reference_grid(aoi=aoi, crs=crs, resolution=res_grid)
    logger.debug("Reference grid created with resolution %g (CRS units)", res_grid)

    # 3. Get UAVSAR coherence layers
    logger.info("Starting to load Coherence...")
    s = time.time()
    coh = get_uavsar_coherence(
        aoi=aoi,
        date_pairs=date_pairs,
        flight_ids=flight_ids,
        crs=crs,
        ref_grid=ref,
        fp=fp_coh,
    )
    e = time.time()
    logger.info("Loaded Coherence for flights %s in %.3f seconds.", flight_ids, e - s)
    logger.debug("Coherence dataset shape: %s", dict(coh.sizes))

    # 4. Get incidence angle
    logger.info("Starting to load Incidence Angle...")
    s = time.time()
    incidence = get_uavsar_incidence(
        aoi=aoi,
        flight_ids=flight_ids,
        fp_inc=fp_inc,
        ref_grid=ref,
    )
    e = time.time()
    logger.info("Loaded Incidence Angle for flights %s in %.3f seconds.", flight_ids, e - s)

    # 5. Get DEM
    logger.info("Starting to load DEM...")
    s = time.time()
    try:
        dem = py3dep.get_dem(geometry=aoi, resolution=res, crs=crs)
    except Exception as exc:
        logger.error("py3dep.get_dem failed: %s", exc)
        raise
    dem.rio.write_crs(crs)
    dem = dem.rio.reproject_match(ref)
    e = time.time()
    logger.info("Loaded DEM in %.3f seconds.", e - s)

    # 6. Get topographic derivative layers
    logger.info("Starting to load Topographic Layers...")
    s = time.time()
    topo = get_topo_layers(dem=dem, ref_grid=ref)
    e = time.time()
    logger.info("Loaded Topographic Layers: %s in %.3f seconds.", list(topo.keys()), e - s)

    # 7. Get NLCD layers
    logger.info("Starting to load NLCD Layers...")
    s = time.time()
    nlcd = get_nlcd_layers(aoi=aoi, crs=crs, ref_grid=ref)
    e = time.time()
    logger.info("Loaded NLCD Layers %s in %.3f seconds.", list(nlcd.keys()), e - s)

    # 8. Get snow climatology
    logger.info("Starting to load Snow Climatology...")
    s = time.time()
    snow_class = get_snow_climatology(aoi=aoi, crs=crs, fp=fp_snowclimate, ref_grid=ref)
    e = time.time()
    logger.info("Loaded Snow Climatology: %s in %.3f seconds.", list(snow_class.keys()), e - s)

    # 9. Get AORC meteorological layers
    logger.info("Starting to load AORC Meteorological Layers...")
    s = time.time()
    aorc = get_aorc_layers(aoi=aoi, date_pairs=date_pairs, crs=crs, ref_grid=ref)
    e = time.time()
    logger.info("Loaded AORC Meteorological Layers: %s in %.3f seconds.", list(aorc.keys()), e - s)

    # 10. Get UCLA SWE/SD layers
    logger.info("Starting to load UCLA SWE/SD Layers...")
    s = time.time()
    snow = get_snow_layers(aoi=aoi, date_pairs=date_pairs, crs=crs, ref_grid=ref)
    e = time.time()
    logger.info("Loaded UCLA SWE/SD Layers: %s in %.3f seconds.", list(snow.keys()), e - s)

    ds_list = [coh, incidence, dem, topo, nlcd, snow_class, snow, aorc]
    validate_alignment(ds_list)
    logger.debug("All layers validated for spatial alignment")

    # ── Pre-merge "scorched earth" sanitization ─────────────────────────────
    # Remove any CRS/grid-mapping artefact coordinates and clear all per-layer
    # global attributes so that no upstream metadata (e.g. NSIDC paper URLs,
    # NLCD source strings) leaks into the final store's .zattrs.
    _ARTEFACT_COORDS = ['spatial_ref', 'grid_mapping', 'crs', 'band']

    sanitized_ds_list = []
    for layer in ds_list:
        layer = layer.drop_vars(
            [c for c in _ARTEFACT_COORDS if c in layer.coords],
            errors='ignore',
        )
        layer.attrs.clear()
        if hasattr(layer, 'data_vars'):
            for var in layer.data_vars:
                layer[var].attrs.pop('grid_mapping', None)
        sanitized_ds_list.append(layer)
    logger.debug("Sanitized %d layers before merge", len(sanitized_ds_list))

    # ── Strict merge ─────────────────────────────────────────────────────────
    ds = xr.merge(
        sanitized_ds_list,
        join='exact',
        compat='no_conflicts',
        combine_attrs='drop',
    )
    logger.info("Merged %d layers into a single dataset", len(sanitized_ds_list))

    # ── Write CRS once, immediately after merge ───────────────────────────────
    ds = ds.rio.write_crs(crs)

    # ── Per-variable CF metadata applied post-merge ───────────────────────────
    # The sanitization step clears all layer-level attrs, so elevation (which
    # has no individual loading function) must have its attrs applied here.
    if 'elevation' in ds:
        ds['elevation'].attrs.update({
            'units': 'm',
            'long_name': 'elevation above sea level',
            'valid_range': [-50.0, 9000.0],
            'source': '3DEP (USGS 3D Elevation Program)',
        })

    # ── CF-1.8 / ACDD-1.3 global attributes ──────────────────────────────────
    bounds = aoi.bounds  # (minx, miny, maxx, maxy) in the output CRS
    ds.attrs.update({
        'Conventions': 'CF-1.8, ACDD-1.3',
        'title': 'UAVSAR Coherence Data Cube',
        'summary': (
            'Multi-temporal InSAR coherence and co-registered environmental '
            'co-variables derived from NASA UAVSAR Level-2 products.'
        ),
        'source': 'UAVSAR Level-2 coherence products (NASA/JPL)',
        'date_created': pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'geospatial_lat_min': float(bounds[1]),
        'geospatial_lat_max': float(bounds[3]),
        'geospatial_lon_min': float(bounds[0]),
        'geospatial_lon_max': float(bounds[2]),
        'geospatial_bounds_crs': 'EPSG:4326',
        'flight_ids': ', '.join(str(f) for f in flight_ids),
    })

    # ── Optional UAVSAR annotation metadata ──────────────────────────────────
    if fp_ann is not None:
        ann_attrs = _read_uavsar_annotations(fp_ann=fp_ann, flight_ids=flight_ids)
        ds.attrs.update(ann_attrs)

    # Clear original chunking metadata so to_zarr() does not try to reuse
    # stale chunk information and crash.
    for var in ds.variables:
        ds[var].encoding.pop('chunks', None)
        ds[var].encoding.pop('preferred_chunks', None)

    # Define a spatial-only chunking strategy.
    # Using -1 tells Dask to NOT chunk that dimension (keep it as one block).
    chunk_dict = {
        'pair': -1,                  # Keep the entire time series together
        'y': SPATIAL_CHUNK_SIZE,     # Chunk spatially
        'x': SPATIAL_CHUNK_SIZE,
    }

    ds_chunked = ds.chunk(chunk_dict)

    if fp_dest is not None:
        if overwrite:
            ds_chunked.to_zarr(fp_dest, mode='w', consolidated=True)
        else:
            ds_chunked.to_zarr(fp_dest, mode='w-', consolidated=True)
        logger.info("Dataset written to %s", fp_dest)
    
    return ds


def _read_uavsar_annotations(fp_ann: str, flight_ids: List[str]) -> Dict[str, str]:
    """
    Read UAVSAR annotation files and return a flat dict of provenance attributes.

    One ``.ann`` file is processed per flight ID (the first match under
    ``fp_ann``).  The returned dict is suitable for merging directly into
    ``ds.attrs`` and therefore into ``.zattrs`` of the root Zarr group.

    Parameters
    ----------
    fp_ann:
        Directory that contains UAVSAR ``.ann`` annotation files.
    flight_ids:
        List of UAVSAR flight-heading identifiers whose annotation files
        should be parsed.

    Returns
    -------
    Dict[str, str]
        Flat mapping of ``uavsar_{flight_id}_{key}`` → value strings.
        Empty dict if ``uavsar_pytools`` is not installed or no files are
        found.
    """
    try:
        from uavsar_pytools.convert.tiff_conversion import read_annotation
    except ImportError:
        logger.warning(
            "uavsar_pytools is not installed; annotation metadata will be skipped."
        )
        return {}

    # Fields to extract from each annotation file (subset relevant for CF/ACDD).
    _WANTED_FIELDS = [
        'start time of acquisition',
        'stop time of acquisition',
        'global average yaw',
        'global average pitch',
        'global average roll',
        'site description',
        'flight line',
        'url',
    ]

    attrs: Dict[str, str] = {}
    ann_dir = Path(fp_ann)

    for fid in flight_ids:
        ann_files = sorted(ann_dir.glob(f'*{fid}*.ann'))
        if not ann_files:
            logger.warning("No annotation file found for flight %s in %s", fid, ann_dir)
            continue

        ann_path = ann_files[0]
        logger.debug("Reading annotation file: %s", ann_path)

        try:
            ann_raw = read_annotation(ann_path)
        except Exception as exc:
            logger.warning("Failed to read annotation file %s: %s", ann_path, exc)
            continue

        ann_df = pd.DataFrame(ann_raw).T
        for field in _WANTED_FIELDS:
            if field in ann_df.index:
                safe_key = f"uavsar_{fid}_{field.replace(' ', '_')}"
                attrs[safe_key] = str(ann_df.loc[field, 'value'])

    return attrs


def get_uavsar_coherence(
    aoi: Polygon,
    date_pairs: List[Tuple[date, date]],
    flight_ids: List[str],
    crs: str,
    ref_grid: Union[xr.DataArray, xr.Dataset],
    fp: str = '../data/coherence/',
) -> xr.Dataset:
    """
    Load UAVSAR coherence files and assemble a 4-D cube ``(flight_id, pair, y, x)``.

    Files are read from flight-specific sub-directories under ``fp`` and are
    aligned to ``ref_grid`` before concatenation.

    Expected file structure::

        fp / {flight_id} / *{flight_id}*{date1}*{date2}*.coh.tif

    Parameters
    ----------
    aoi:
        Area of interest as a Shapely Polygon.
    date_pairs:
        List of ``(start_date, end_date)`` tuples defining the coherence
        intervals.
    flight_ids:
        List of UAVSAR flight-heading identifiers.
    crs:
        Coordinate reference system string (e.g. ``'EPSG:4326'``).
    ref_grid:
        Reference grid used to spatially align all arrays.
    fp:
        Root directory that contains per-flight sub-directories.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions ``(flight_id, pair, y, x)`` and a single
        ``coherence`` variable.  The ``pair`` dimension carries three auxiliary
        coordinates that share the same dimension:

        * ``time_1`` – start acquisition date as ``datetime64[ns]``.
        * ``time_2`` – end acquisition date as ``datetime64[ns]``.
        * ``delta_t`` – temporal baseline in days (integer).

    Raises
    ------
    FileNotFoundError
        If a flight directory or an expected coherence file is not found.
    """

    fp = Path(fp)
    flight_arrays = []

    for fid in flight_ids:
        logger.debug("Loading coherence for flight %s", fid)
        pair_arrays = []
        flight_dir = fp / str(fid)

        if not flight_dir.exists():
            raise FileNotFoundError(f"Flight directory not found: {flight_dir}")

        for start_date, end_date in date_pairs:
            # Format dates to match standard UAVSAR YYMMDD naming convention.
            date_str1 = start_date.strftime('%y%m%d')
            date_str2 = end_date.strftime('%y%m%d')

            # Create a pair coordinate name (e.g. '210210_210224').
            pair_name = f"{date_str1}_{date_str2}"

            search_pattern = f"*{fid}*{date_str1}*{date_str2}*coh*"
            found_files = list(flight_dir.glob(search_pattern))

            if not found_files:
                raise FileNotFoundError(
                    f"Could not find coherence file for flight {fid} and "
                    f"dates {pair_name} in {flight_dir}"
                )

            if len(found_files) > 1:
                logger.warning(
                    "Multiple coherence files found for flight %s and dates %s; "
                    "using %s",
                    fid, pair_name, found_files[0],
                )

            file_path = found_files[0]
            logger.debug("Found coherence file: %s", file_path)

            # Load raster, write CRS, and rename the data variable.
            da = xr.open_dataset(file_path, chunks={})
            da.rio.write_crs(crs, inplace=True)
            da = da.rename({list(da.data_vars.keys())[0]: 'coherence'})

            # Align to the master reference grid.
            da_matched = da.rio.reproject_match(ref_grid)

            pair_arrays.append(da_matched)

        # Concatenate all date pairs for this flight.
        pair_coord = xr.DataArray(
            [f"{s.strftime('%y%m%d')}_{e.strftime('%y%m%d')}" for s, e in date_pairs],
            dims=['pair'],
            name='pair',
        )
        flight_da = xr.concat(pair_arrays, dim=pair_coord)
        flight_arrays.append(flight_da)

    # Concatenate all flights into the final 4-D array.
    flight_coord = xr.DataArray(flight_ids, dims=['flight_id'], name='flight_id')
    coherence_da = xr.concat(flight_arrays, dim=flight_coord)

    # ── Auxiliary time coordinates on the 'pair' dimension ─────────────────
    # Build datetime64 arrays directly from the date_pairs objects so that
    # downstream code can use sel/groupby on real timestamps instead of
    # parsing the YYMMDD string label.
    time_1_values, time_2_values, delta_t_values = zip(
        *[
            (
                np.datetime64(s.strftime('%Y-%m-%d'), 'ns'),
                np.datetime64(e.strftime('%Y-%m-%d'), 'ns'),
                (e - s).days,
            )
            for s, e in date_pairs
        ]
    )
    time_1_values = np.array(time_1_values, dtype='datetime64[ns]')
    time_2_values = np.array(time_2_values, dtype='datetime64[ns]')
    delta_t_values = np.array(delta_t_values, dtype=np.int64)

    coherence_da = coherence_da.assign_coords(
        time_1=xr.DataArray(
            time_1_values,
            dims=['pair'],
            attrs={
                'long_name': 'start acquisition date',
                'standard_name': 'time',
                'calendar': 'proleptic_gregorian',
            },
        ),
        time_2=xr.DataArray(
            time_2_values,
            dims=['pair'],
            attrs={
                'long_name': 'end acquisition date',
                'standard_name': 'time',
                'calendar': 'proleptic_gregorian',
            },
        ),
        delta_t=xr.DataArray(
            delta_t_values,
            dims=['pair'],
            attrs={
                'long_name': 'temporal baseline',
                'units': 'days',
            },
        ),
    )

    logger.debug(
        "Coherence array assembled with shape: %s", dict(coherence_da.sizes)
    )

    # ── CF variable metadata ──────────────────────────────────────────────────
    # coherence_da is a Dataset (xr.open_dataset → xr.concat path); guard with
    # isinstance to ensure the subscript accessor is safe.
    if isinstance(coherence_da, xr.Dataset) and 'coherence' in coherence_da:
        coherence_da['coherence'].attrs.update({
            'units': '1',
            'long_name': 'InSAR coherence magnitude',
            'valid_range': [0.0, 1.0],
        })

    return coherence_da


def get_uavsar_incidence(
    aoi: Polygon,
    flight_ids: List[str],
    fp_inc: Union[str, Path],
    ref_grid: Union[xr.DataArray, xr.Dataset],
    crs: str = 'EPSG:4326',
) -> xr.DataArray:
    """
    Load pre-calculated UAVSAR incidence angle files and concatenate them along
    a ``flight_id`` dimension.

    Parameters
    ----------
    aoi:
        Area of interest as a Shapely Polygon used to spatially subset the
        incidence angle rasters before reprojection.
    flight_ids:
        List of UAVSAR flight-heading identifiers (e.g. ``['08508', '26505']``).
    fp_inc:
        Directory containing the pre-calculated incidence angle ``.tif`` files.
        Files are matched using the pattern ``*{flight_id}*s2.inc.tif``.
    ref_grid:
        Reference grid used to spatially align all arrays.
    crs:
        Coordinate reference system string (default ``'EPSG:4326'``).

    Returns
    -------
    xr.DataArray
        3-D DataArray with dimensions ``(flight_id, y, x)`` and the name
        ``'incidence_angle'``.

    Raises
    ------
    FileNotFoundError
        If the incidence file for a given flight is not found.
    """
    fp_inc = Path(fp_inc)
    inc_arrays = []

    for fid in flight_ids:
        search_pattern = f"*{fid}*s2.inc.tif"
        found_files = list(fp_inc.glob(search_pattern))

        if not found_files:
            raise FileNotFoundError(
                f"Could not find pre-calculated incidence file for flight {fid} "
                f"matching pattern '{search_pattern}' in {fp_inc}"
            )

        file_path = found_files[0]
        logger.debug("Loading incidence angle file: %s", file_path)

        # Load the file; masked=True converts nodata/fill values to np.nan.
        da = rxa.open_rasterio(file_path, masked=True).squeeze()
        # Derive bounds in the same CRS as the raster (assumed to match `crs`).
        minx, miny, maxx, maxy = gpd.GeoSeries([aoi], crs=crs).total_bounds
        da = da.sel(x=slice(minx, maxx), y=slice(maxy, miny))

        # Write CRS before reprojecting.
        da.rio.write_crs(crs, inplace=True)

        # Align to the reference grid.
        da_matched = da.rio.reproject_match(ref_grid)

        da_matched.name = 'incidence_angle'

        inc_arrays.append(da_matched)

    # Concatenate into a single DataArray with the 'flight_id' dimension.
    flight_coord = xr.DataArray(flight_ids, dims=['flight_id'], name='flight_id')
    incidence_da = xr.concat(inc_arrays, dim=flight_coord)
    logger.debug(
        "Incidence angle array assembled with shape: %s", dict(incidence_da.sizes)
    )

    # ── CF variable metadata ──────────────────────────────────────────────────
    incidence_da.attrs.update({
        'units': 'degree',
        'long_name': 'local incidence angle',
        'valid_range': [0.0, 90.0],
    })

    return incidence_da


# =============================================================================
# FUNCTIONS FOR DOWNLOADING ENVIRONMENTAL DATA
# =============================================================================

def get_nlcd_layers(
    aoi: Polygon,
    crs: str,
    years: Dict[str, List[int]] = {'cover': [2019], 'canopy': [2019]},
    ref_grid: Optional[Union[xr.DataArray, xr.Dataset]] = None,
) -> xr.Dataset:
    """
    Retrieve National Land Cover Database (NLCD) layers for the given AOI.

    Parameters
    ----------
    aoi:
        Area of interest as a Shapely Polygon.
    crs:
        Coordinate reference system of ``aoi``.
    years:
        Dictionary specifying which NLCD layers and years to download.
        Default is ``{'cover': [2019], 'canopy': [2019]}``.
    ref_grid:
        Optional reference grid used to spatially align the output.

    Returns
    -------
    xr.Dataset
        Dataset of NLCD layers aligned to ``ref_grid`` (when provided).
    """
    logger.debug("Fetching NLCD layers for years: %s", years)

    g = gpd.GeoSeries([aoi], crs=crs)

    try:
        ds = gh.nlcd_bygeom(geometry=g, years=years)[0]
    except Exception as exc:
        logger.error("Failed to fetch NLCD data from pygeohydro: %s", exc)
        raise

    if ref_grid is not None:
        ds = ds.rio.reproject_match(ref_grid)

    # ── CF variable metadata ──────────────────────────────────────────────────
    # NOTE: units and valid_range for NLCD variables need manual verification —
    # see the draft PR comment produced by this commit for details.
    for var in ds.data_vars:
        if var.startswith('cover_'):
            ds[var].attrs.setdefault(
                'long_name', f'NLCD land cover class code ({var.split("_")[-1]})'
            )
        elif var.startswith('canopy_'):
            ds[var].attrs.setdefault(
                'long_name', f'NLCD percent tree canopy cover ({var.split("_")[-1]})'
            )

    return ds

def get_snow_climatology(
    aoi: Polygon,
    crs: str,
    fp: str,
    ref_grid: Optional[Union[xr.DataArray, xr.Dataset]] = None,
) -> xr.Dataset:
    """
    Load the NSIDC-0768 seasonal snow classification dataset for the AOI.

    The expected NetCDF file is
    ``SnowClass_NA_300m_10.0arcsec_2021_v01.0.nc``, which must be placed
    directly under ``fp``.  Download instructions are available at
    https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0768_global_seasonal_snow_classification_v01/.

    Parameters
    ----------
    aoi:
        Area of interest as a Shapely Polygon.
    crs:
        Coordinate reference system of ``aoi``.
    fp:
        Directory containing the snow-climatology NetCDF file.
    ref_grid:
        Optional reference grid used to spatially align the output.  When
        ``None`` the dataset is clipped directly to ``aoi``.

    Returns
    -------
    xr.Dataset
        Dataset containing the ``snow_class`` variable, aligned to
        ``ref_grid`` (when provided).

    Raises
    ------
    ValueError
        If the expected NetCDF file cannot be found under ``fp``.
    """
    fname = SNOW_CLIMATOLOGY_FILENAME
    full_path = os.path.join(fp, fname)

    if not os.path.exists(full_path):
        error_msg = (
            f"Could not open file {full_path}. Please download {fname} "
            "from NSIDC at https://daacdata.apps.nsidc.org/pub/DATASETS/"
            "nsidc0768_global_seasonal_snow_classification_v01/."
        )
        raise ValueError(error_msg)

    logger.debug("Loading snow climatology from %s", full_path)

    try:
        ds = xr.open_dataset(full_path, chunks={})
    except Exception as exc:
        logger.error("Failed to open snow climatology file %s: %s", full_path, exc)
        raise
    ds = ds.rename({'lat': 'y', 'lon': 'x', 'SnowClass': 'snow_class'})
    ds = ds.sortby(['x', 'y'])

    # The climatology dataset is always in EPSG:4326; project the AOI bounds
    # to 4326 for slicing, regardless of the input ``crs``.
    g = gpd.GeoSeries([aoi], crs=crs).to_crs('EPSG:4326').total_bounds
    minx, miny, maxx, maxy = g
    logger.debug(
        "Clipping snow climatology to AOI bounds: %.4f, %.4f, %.4f, %.4f",
        minx, miny, maxx, maxy,
    )

    ds = ds.sel(x=slice(minx, maxx), y=slice(miny, maxy))

    ds = ds.rio.write_crs('EPSG:4326')

    if ref_grid is not None:
        ds = ds.rio.reproject_match(ref_grid)
    else:
        g = gpd.GeoSeries([aoi], crs=crs)
        ds = ds.rio.clip(g, crs=crs)

    # ── CF variable metadata ──────────────────────────────────────────────────
    # NOTE: units and valid_range for snow_class need manual verification —
    # see the draft PR comment produced by this commit for details.
    if 'snow_class' in ds:
        ds['snow_class'].attrs.setdefault(
            'long_name', 'NSIDC-0768 seasonal snow classification'
        )

    return ds


def get_topo_layers(
    dem: xr.DataArray,
    ref_grid: Optional[Union[xr.DataArray, xr.Dataset]] = None,
) -> xr.Dataset:
    """
    Derive topographic layers (slope, aspect, curvature) from a DEM.

    The DEM is reprojected to EPSG:5070 (Conus Albers) when the input CRS is
    EPSG:4326 so that metric-based derivatives are accurate.  A 1-pixel NaN
    padding plus linear extrapolation is applied before computing derivatives
    to avoid edge artefacts, then trimmed back to the original extent.

    Parameters
    ----------
    dem:
        Digital Elevation Model as an :class:`xarray.DataArray`.
    ref_grid:
        Optional reference grid used to spatially align the output.

    Returns
    -------
    xr.Dataset
        Dataset with variables ``slope``, ``aspect``, and ``curve``.
    """
    logger.debug("Computing topographic layers from DEM shape %s", dict(dem.sizes))

    ds = xr.Dataset()

    # Reproject to a metric CRS for accurate gradient-based derivatives; any
    # geographic (angular) CRS is unsuitable for slope/aspect computation.
    dem_crs = ProjCRS.from_user_input(dem.rio.crs)
    if dem_crs.is_geographic:
        dem_reproj = dem.rio.reproject(DEM_METRIC_CRS)
    else:
        dem_reproj = dem

    # Pad the DEM with NaNs to extend values to the edge.
    dem_nan_padded = dem_reproj.pad(y=TOPO_PAD_PIXELS, x=TOPO_PAD_PIXELS, constant_values=np.nan)

    # Use use_coordinate=False to bypass the monotonic-coordinate requirement.
    dem_extrapolated = dem_nan_padded.interpolate_na(
        dim='x', method='linear', fill_value="extrapolate", use_coordinate=False
    ).interpolate_na(
        dim='y', method='linear', fill_value="extrapolate", use_coordinate=False
    )

    # Calculate derivatives on the extrapolated, padded array.
    slope_extrap = xrs.slope(dem_extrapolated)
    aspect_extrap = xrs.aspect(dem_extrapolated)
    curve_extrap = xrs.curvature(dem_extrapolated)

    # Trim the padded boundary back to the original DEM extent.
    ds['slope'] = slope_extrap.isel(y=slice(TOPO_PAD_PIXELS, -TOPO_PAD_PIXELS), x=slice(TOPO_PAD_PIXELS, -TOPO_PAD_PIXELS))
    ds['aspect'] = aspect_extrap.isel(y=slice(TOPO_PAD_PIXELS, -TOPO_PAD_PIXELS), x=slice(TOPO_PAD_PIXELS, -TOPO_PAD_PIXELS))
    ds['curve'] = curve_extrap.isel(y=slice(TOPO_PAD_PIXELS, -TOPO_PAD_PIXELS), x=slice(TOPO_PAD_PIXELS, -TOPO_PAD_PIXELS))

    if ref_grid is not None:
        ds = ds.rio.reproject_match(ref_grid)

    # ── CF variable metadata ──────────────────────────────────────────────────
    ds['slope'].attrs.update({
        'units': 'degree',
        'long_name': 'terrain slope angle',
        'valid_range': [0.0, 90.0],
    })
    ds['aspect'].attrs.update({
        'units': 'degree',
        'long_name': 'terrain aspect (clockwise from north)',
        'valid_range': [0.0, 360.0],
    })
    # NOTE: curvature units need manual verification — see the draft PR comment
    # produced by this commit for details.
    ds['curve'].attrs.update({
        'long_name': 'terrain curvature',
    })

    return ds


def get_snow_layers(
    aoi: Polygon,
    crs: str,
    date_pairs: List[Tuple[date, date]],
    metrics: Union[str, List[str]] = 'all',
    ref_grid: Optional[Union[xr.DataArray, xr.Dataset]] = None,
) -> xr.Dataset:
    """
    Retrieve UCLA Western US snow reanalysis (WUS_UCLA_SR) layers for the AOI.

    SWE and snow-depth granules are searched via the Earthaccess API,
    processed into time series, and summarised into per-date-pair metrics.

    Parameters
    ----------
    aoi:
        Area of interest as a Shapely Polygon.
    crs:
        Coordinate reference system of ``aoi``.
    date_pairs:
        List of ``(start_date, end_date)`` tuples defining the intervals
        for which snow metrics are computed.
    metrics:
        Either ``'all'`` or a list of metric names accepted by
        :func:`get_snow_metrics`.
    ref_grid:
        Optional reference grid used to spatially align the output.

    Returns
    -------
    xr.Dataset
        Dataset of snow metrics indexed by a ``pair`` coordinate.
    """
    xx, yy = aoi.exterior.coords.xy
    x = xx.tolist()
    y = yy.tolist()

    g = gpd.GeoSeries([aoi], crs=crs)

    years = get_years(date_pairs)
    logger.debug("Searching WUS_UCLA_SR granules for years: %s", years)

    try:
        grans = earthaccess.search_data(
            short_name='WUS_UCLA_SR',
            cloud_hosted=True,
            temporal=(str(min(years)), str(max(years))),
            polygon=list(zip(x, y)),
        )
    except Exception as exc:
        logger.error("earthaccess.search_data failed for WUS_UCLA_SR: %s", exc)
        raise
    logger.debug("Found %d WUS_UCLA_SR granules", len(grans))

    auth = earthaccess.login()
    if not auth.authenticated:
        auth.login(strategy="interactive", persist=True)

    try:
        fileset = earthaccess.open(grans)
    except Exception as exc:
        logger.error("earthaccess.open failed for WUS_UCLA_SR granules: %s", exc)
        raise

    # Separate granules by type based on naming convention.
    swe_files = [f for f in fileset if "SWE_SCA_POST" in (f.path if hasattr(f, 'path') else str(f))]
    sd_files = [f for f in fileset if "SD_POST" in (f.path if hasattr(f, 'path') else str(f))]

    # Process granules individually so that a single bad file does not abort
    # the whole pipeline.
    processed_swe: List[xr.Dataset] = []
    for f in swe_files:
        try:
            processed_swe.append(process_ucla_granule(f))
        except Exception as exc:
            logger.warning("Skipping SWE granule %s due to error: %s", f, exc)

    processed_sd: List[xr.Dataset] = []
    for f in sd_files:
        try:
            processed_sd.append(process_ucla_granule(f))
        except Exception as exc:
            logger.warning("Skipping SD granule %s due to error: %s", f, exc)

    if not processed_swe or not processed_sd:
        raise RuntimeError(
            "No UCLA WUS_UCLA_SR granules could be processed. "
            "Check authentication and data availability."
        )

    ds_swe = xr.concat(processed_swe, dim='time').sortby('time')
    ds_sd = xr.concat(processed_sd, dim='time').sortby('time')

    # Merge SWE and SD into one data cube.
    ds_full = xr.merge([ds_swe, ds_sd], compat='override')
    ds_full = ds_full.isel(Stats=UCLA_POSTERIOR_STATS_INDEX)
    ds_full = ds_full.rio.write_crs("EPSG:4326")

    ds_clip = ds_full.rio.clip(g, crs=crs)
    ds_clip = ds_clip.transpose('y', 'x', 'time')

    ds_list = []
    for start_date, end_date in date_pairs:
        ds_slice = ds_clip.sel(time=slice(start_date, end_date))

        if len(ds_slice['time']) == 0:
            logger.warning(
                "No WUS_UCLA_SR data found for %s to %s; skipping.",
                start_date, end_date,
            )
            continue

        ds_metrics = get_snow_metrics(ds_slice, metrics=metrics)
        pair_name = start_date.strftime('%y%m%d') + "_" + end_date.strftime('%y%m%d')
        ds_metrics['pair'] = pair_name
        ds_metrics = ds_metrics.set_coords('pair')

        ds_list.append(ds_metrics)

    ds = xr.concat(ds_list, dim='pair')
    ds = ds.sortby('pair')
    if ref_grid is not None:
        ds = ds.rio.reproject_match(ref_grid)
    return ds


def get_snow_metrics(
    ds: xr.Dataset,
    metrics: Union[str, List[str]] = 'all',
) -> xr.Dataset:
    """
    Compute summary snow metrics over an interval.

    Parameters
    ----------
    ds:
        Dataset with ``SWE_Post`` and ``SD_Post`` variables along a ``time``
        dimension.  Expected to cover a single coherence interval.
    metrics:
        Either ``'all'`` (compute all supported metrics) or a list of metric
        names chosen from ``['swe_accum', 'swe_ablate', 'density_change']``.

    Returns
    -------
    xr.Dataset
        Dataset of computed snow metrics (no time dimension).

    Raises
    ------
    ValueError
        If an unrecognised metric name is requested.
    """
    out = xr.Dataset()

    valid_metrics = [
        'swe_accum',
        'swe_ablate',
        'density_change',
    ]

    if metrics == 'all':
        metrics = valid_metrics
    else:
        for metric in metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Valid metrics are: {valid_metrics}")

    # Calculate day-to-day changes.
    swe_diff = ds['SWE_Post'].diff(dim='time')
    sd_diff = ds['SD_Post'].diff(dim='time')

    if 'swe_accum' in metrics:
        # Sum of only the positive daily SWE changes (new snow mass).
        out['swe_accum'] = swe_diff.where(swe_diff > 0, other=0).sum(dim='time')

    if 'swe_ablate' in metrics:
        # Sum of only the negative daily SWE changes (melt/sublimation mass).
        # Absolute value makes this easier to interpret as a positive feature.
        out['swe_ablate'] = abs(swe_diff.where(swe_diff < 0, other=0).sum(dim='time'))

    if 'density_change' in metrics:
        # Bulk Density = SWE / Snow Depth (division by zero where no snow).
        # Reference day density.
        dens_ref = ds['SWE_Post'].isel(time=0) / ds['SD_Post'].isel(time=0).where(ds['SD_Post'].isel(time=0) > 0)

        # Secondary day density.
        dens_sec = ds['SWE_Post'].isel(time=-1) / ds['SD_Post'].isel(time=-1).where(ds['SD_Post'].isel(time=-1) > 0)

        # Change in density. Fill NaNs with 0 (no snow = no density change).
        out['density_change'] = (dens_sec - dens_ref).fillna(0)

    if 'snow_status_change' in metrics:
        # +1 if snow appeared, -1 if snow melted, 0 if unchanged presence.
        has_snow_ref = (ds['SWE_Post'].isel(time=0) > 0).astype(int)
        has_snow_sec = (ds['SWE_Post'].isel(time=-1) > 0).astype(int)

        out['snow_status_change'] = has_snow_sec - has_snow_ref

    # ── CF variable metadata ──────────────────────────────────────────────────
    # NOTE: units and valid_range for SWE/SD-derived metrics need manual
    # verification — see the draft PR comment for details.
    if 'swe_accum' in out:
        out['swe_accum'].attrs.update({
            'long_name': 'accumulated SWE gain over coherence interval',
        })
    if 'swe_ablate' in out:
        out['swe_ablate'].attrs.update({
            'long_name': 'accumulated SWE loss (absolute value) over coherence interval',
        })
    if 'density_change' in out:
        out['density_change'].attrs.update({
            'long_name': 'change in bulk snow density over coherence interval',
        })

    return out


def process_ucla_granule(file_obj) -> xr.Dataset:
    """
    Parse a single UCLA WUS_UCLA_SR granule into a properly indexed dataset.

    The function extracts the Water Year from the filename, constructs a
    daily ``time`` axis starting on October 1st of that year, and renames
    the spatial coordinates to the standard ``y``/``x`` convention used
    throughout this pipeline.

    Parameters
    ----------
    file_obj:
        An open file-like object (e.g. from ``earthaccess.open``) or a path
        string pointing to a UCLA granule NetCDF file.

    Returns
    -------
    xr.Dataset
        Dataset with a ``time`` dimension and ``y``/``x`` spatial coordinates.

    Raises
    ------
    OSError
        If the granule file cannot be opened.
    ValueError
        If the Water Year cannot be parsed from the filename.
    """
    filename = file_obj.path if hasattr(file_obj, 'path') else str(file_obj)

    try:
        ds = xr.open_dataset(file_obj, chunks={})
    except Exception as exc:
        logger.error("Failed to open UCLA granule %s: %s", filename, exc)
        raise

    # Parse the Water Year (WY) from the filename for the time axis.
    match = re.search(r'WY(\d{4})_\d{2}', filename)
    if match is None:
        raise ValueError(
            f"Could not parse Water Year from UCLA granule filename: {filename}"
        )
    start_year = int(match.group(1))
    logger.debug("Processing UCLA granule for WY%d: %s", start_year, filename)

    # Create daily time coordinates starting Oct 1st of the water year.
    num_days = ds.sizes['Day']
    time_coords = pd.date_range(start=f"{start_year}-10-01", periods=num_days, freq='D')

    # Flatten spatial coordinates from (N×1) arrays to 1-D.
    lat_values = ds.Latitude.values.flatten()
    lon_values = ds.Longitude.values.flatten()

    # Rename dimensions to standard names.
    ds = ds.rename({'Day': 'time', 'Latitude': 'y', 'Longitude': 'x'})

    # Assign the new dimensional coordinates.
    ds = ds.assign_coords(
        time=time_coords,
        y=lat_values,
        x=lon_values,
    )

    return ds


def get_aorc_layers(
    aoi: Polygon,
    date_pairs: List[Tuple[date, date]],
    crs: str,
    ref_grid: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    metrics: Union[str, List[str]] = 'all',
) -> xr.Dataset:
    """
    Retrieve AORC (Analysis of Record for Calibration) meteorological layers.

    Data are read directly from the NOAA public S3 bucket and summarised into
    per-date-pair metrics using :func:`get_aorc_metrics`.

    Parameters
    ----------
    aoi:
        Area of interest as a Shapely Polygon.
    date_pairs:
        List of ``(start_date, end_date)`` tuples defining the intervals
        for which AORC metrics are computed.
    crs:
        Coordinate reference system of ``aoi``.
    ref_grid:
        Optional reference grid used to spatially align the output.
    metrics:
        Either ``'all'`` or a list of metric names accepted by
        :func:`get_aorc_metrics`.

    Returns
    -------
    xr.Dataset
        Dataset of AORC metrics indexed by a ``pair`` coordinate.
    """
    years = get_years(date_pairs)
    g = gpd.GeoSeries([aoi], crs=crs)
    logger.debug("Opening AORC zarr stores for years: %s", years)

    s3_out = s3fs.S3FileSystem(anon=True)
    fileset = [
        s3fs.S3Map(root=f"s3://{AORC_S3_BASE_URL}/{yr}.zarr", s3=s3_out, check=False)
        for yr in years
    ]
    try:
        ds_full = xr.open_mfdataset(fileset, engine='zarr')
    except Exception as exc:
        logger.error(
            "Failed to open AORC zarr store(s) from s3://%s: %s",
            AORC_S3_BASE_URL, exc,
        )
        raise

    # Clip to the AOI and rename spatial dimensions.
    ds_clip = ds_full.rio.clip(g.geometry.values, crs=crs)
    ds_clip = ds_clip.rename({'latitude': 'y', 'longitude': 'x'})

    ds_list = []
    for start_date, end_date in date_pairs:
        ds_slice = ds_clip.sel(time=slice(start_date, end_date))

        if len(ds_slice['time']) == 0:
            logger.warning(
                "No AORC data found for %s to %s; skipping.",
                start_date, end_date,
            )
            continue

        ds_metrics = get_aorc_metrics(ds_slice, metrics=metrics)
        pair_name = start_date.strftime('%y%m%d') + "_" + end_date.strftime('%y%m%d')
        ds_metrics['pair'] = pair_name
        ds_metrics = ds_metrics.set_coords('pair')

        ds_list.append(ds_metrics)

    ds = xr.concat(ds_list, dim='pair')
    ds = ds.sortby('pair')

    if ref_grid is not None:
        ds = ds.rio.reproject_match(ref_grid)
    return ds

def get_aorc_metrics(
    aorc_ds: xr.Dataset,
    metrics: Union[str, List[str]] = 'all',
) -> xr.Dataset:
    """
    Compute summary AORC meteorological metrics over a coherence interval.

    Parameters
    ----------
    aorc_ds:
        AORC dataset sliced to a single coherence interval.  Must contain
        the variables ``TMP_2maboveground``, ``APCP_surface``,
        ``UGRD_10maboveground``, and ``VGRD_10maboveground``.
    metrics:
        Either ``'all'`` (compute all supported metrics) or a list of metric
        names chosen from the ``valid_metrics`` list defined inside this
        function.

    Returns
    -------
    xr.Dataset
        Dataset of computed AORC metrics (no time dimension).

    Raises
    ------
    ValueError
        If an unrecognised metric name is requested.
    """
    ds = xr.Dataset()

    valid_metrics = [
        'mean_temp',
        'max_temp',
        'total_posdeg',
        'temp_diff',
        'temp_diff_acq',
        'freeze_thaw_cycles',
        'diurnal_temp_range',
        'total_precip',
        'total_rain',
        'total_snow',
        'acq_day_precip',
        'mean_wind',
        'max_wind',
        'hours_blowing_snow',
    ]

    if metrics == 'all':
        metrics = valid_metrics
    else:
        for metric in metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Valid metrics are: {valid_metrics}")

    # Temperature metrics.
    temp = aorc_ds['TMP_2maboveground'] - KELVIN_TO_CELSIUS_OFFSET  # convert from K to °C
    if 'mean_temp' in metrics:
        ds['mean_temp'] = temp.mean(dim='time')
    if 'max_temp' in metrics:
        ds['max_temp'] = temp.max(dim='time')
    if 'total_posdeg' in metrics:
        ds['total_posdeg'] = (
            temp
            .where(temp > 0, other=0)
            .sum(dim='time')
        )
    if 'temp_diff' in metrics:
        ds['temp_diff'] = temp.isel(time=-1) - temp.isel(time=0)
    if 'temp_diff_acq' in metrics:
        # Absolute temperature difference between reference and secondary acquisitions.
        ds['temp_diff_acq'] = abs(temp.isel(time=0) - temp.isel(time=-1))
    if 'freeze_thaw_cycles' in metrics:
        # Count sign-crossings of the 0 °C threshold over the interval.
        is_above_freezing = temp > 0
        ds['freeze_thaw_cycles'] = (is_above_freezing.astype(int).diff(dim='time') != 0).sum(dim='time')
        # Clamp to physically plausible range.
        ds['freeze_thaw_cycles'] = ds['freeze_thaw_cycles'].where(
            ds['freeze_thaw_cycles'] <= MAX_FREEZE_THAW_CYCLES, other=np.nan
        )
        ds['freeze_thaw_cycles'] = ds['freeze_thaw_cycles'].where(
            ds['freeze_thaw_cycles'] >= 0, other=np.nan
        )
    if 'diurnal_temp_range' in metrics:
        # Average daily temperature range over the interval.
        daily_max = temp.resample(time='1D').max()
        daily_min = temp.resample(time='1D').min()
        ds['diurnal_temp_range'] = (daily_max - daily_min).mean(dim='time')

    # Precipitation metrics.
    precip = aorc_ds['APCP_surface']
    if 'total_precip' in metrics:
        ds['total_precip'] = precip.sum(dim='time')
    if 'total_rain' in metrics:
        # Sum of precipitation only when temperature is above 0 °C.
        ds['total_rain'] = precip.where(temp > 0, other=0).sum(dim='time')
    if 'total_snow' in metrics:
        # Sum of precipitation only when temperature is at or below 0 °C.
        ds['total_snow'] = precip.where(temp <= 0, other=0).sum(dim='time')
    if 'acq_day_precip' in metrics:
        # Total precipitation on the reference and secondary acquisition days.
        ds['acq_day_precip'] = precip.isel(time=0) + precip.isel(time=-1)

    # Wind metrics.
    wind_speed = np.sqrt(aorc_ds['UGRD_10maboveground']**2 + aorc_ds['VGRD_10maboveground']**2)

    if 'hours_blowing_snow' in metrics:
        # Count hourly time steps exceeding the blowing-snow wind threshold.
        ds['hours_blowing_snow'] = (wind_speed > BLOWING_SNOW_WIND_THRESHOLD_MS).sum(dim='time')
        # Clamp to physically plausible range.
        ds['hours_blowing_snow'] = ds['hours_blowing_snow'].where(
            (ds['hours_blowing_snow'] >= 0) & (ds['hours_blowing_snow'] <= MAX_HOURS_BLOWING_SNOW),
            other=np.nan,
        )
    if 'mean_wind' in metrics:
        ds['mean_wind'] = np.sqrt(
            aorc_ds['UGRD_10maboveground']**2 + aorc_ds['VGRD_10maboveground']**2
        ).mean(dim='time')
    if 'max_wind' in metrics:
        ds['max_wind'] = np.sqrt(
            aorc_ds['UGRD_10maboveground']**2 + aorc_ds['VGRD_10maboveground']**2
        ).max(dim='time')

    # ── CF variable metadata ──────────────────────────────────────────────────
    # Temperature metrics (degrees Celsius after K→°C conversion).
    _temp_meta = {
        'mean_temp': 'mean air temperature over coherence interval',
        'max_temp': 'maximum air temperature over coherence interval',
        'temp_diff': 'air temperature difference (end − start) over coherence interval',
        'temp_diff_acq': 'absolute air temperature difference between acquisitions',
        'diurnal_temp_range': 'mean daily temperature range over coherence interval',
    }
    for var, long_name in _temp_meta.items():
        if var in ds:
            ds[var].attrs.update({'units': 'degree_Celsius', 'long_name': long_name})

    # total_posdeg accumulates hourly positive temperatures; the AORC dataset is
    # hourly, so the unit is effectively degree_Celsius × hours.
    # NOTE: confirm whether this should be expressed as degree_Celsius × d
    # (positive degree-days) or degree_Celsius × h — see draft PR comment.
    if 'total_posdeg' in ds:
        ds['total_posdeg'].attrs.update({
            'long_name': 'accumulated positive air temperature over coherence interval',
        })

    if 'freeze_thaw_cycles' in ds:
        ds['freeze_thaw_cycles'].attrs.update({
            'units': '1',
            'long_name': 'number of freeze-thaw cycles over coherence interval',
        })

    # Wind metrics.
    if 'mean_wind' in ds:
        ds['mean_wind'].attrs.update({
            'units': 'm s-1',
            'long_name': 'mean wind speed over coherence interval',
        })
    if 'max_wind' in ds:
        ds['max_wind'].attrs.update({
            'units': 'm s-1',
            'long_name': 'maximum wind speed over coherence interval',
        })
    if 'hours_blowing_snow' in ds:
        ds['hours_blowing_snow'].attrs.update({
            'units': 'h',
            'long_name': (
                f'hours with wind speed exceeding '
                f'{BLOWING_SNOW_WIND_THRESHOLD_MS} m/s (blowing-snow threshold)'
            ),
        })

    # Precipitation metrics.
    # NOTE: AORC APCP_surface accumulation period and units (kg m⁻² or mm)
    # need manual verification — see draft PR comment.
    _precip_meta = {
        'total_precip': 'total precipitation over coherence interval',
        'total_rain': 'total liquid precipitation over coherence interval',
        'total_snow': 'total solid precipitation over coherence interval',
        'acq_day_precip': 'total precipitation on acquisition days',
    }
    for var, long_name in _precip_meta.items():
        if var in ds:
            ds[var].attrs.update({'long_name': long_name})

    return ds


def get_years(
    date_pairs: List[Tuple[date, date]]
) -> Set[int]:
    """
    Extract the unique set of calendar years covered by a list of date pairs.

    Parameters
    ----------
    date_pairs:
        List of ``(start_date, end_date)`` tuples.

    Returns
    -------
    Set[int]
        Set of integer years referenced by ``date_pairs``.
    """
    years: Set[int] = set()
    for start_date, end_date in date_pairs:
        years.update([start_date.year, end_date.year])
    return years
import logging
import math
import os
import re
import sys
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
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
# from scripts.validation import validate_aoi, validate_date_pairs, make_reference_grid, validate_alignment
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

RAIN_SNOW_THRESHOLD: float = 2.0
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
    tile_aoi: Polygon,
    date_pairs: List[Tuple[date, date]],
    flight_ids: List[str],
    fp_coh: str = '../data/coherence/',
    fp_inc: str = '../data/inc_angle/',
    fp_snowclimate: str = '../data/NSIDC-0768/',
    crs: str = 'EPSG:4326',
    res: int = 30,
) -> xr.Dataset:
    """
    Fetch, align, and return all data layers for a single spatial tile.

    This is a **pure function**: it reads from disk and remote APIs, aligns
    everything to a reference grid derived from ``tile_aoi``, and returns the
    merged in-memory :class:`xarray.Dataset`.  No data is written to disk.
    Use :func:`assemble_data_largeaoi` to process a large master AOI by
    splitting it into tiles and writing each tile to a Zarr store.

    The caller is responsible for ensuring that ``date_pairs`` corresponds to
    actual UAVSAR acquisition dates.

    Parameters
    ----------
    tile_aoi:
        Small spatial tile (a Shapely Polygon) representing the chunk of the
        master extent to be processed.  Must be given in the coordinate system
        specified by ``crs``.
    date_pairs:
        List of ``(start_date, end_date)`` tuples defining the UAVSAR
        coherence intervals.
    flight_ids:
        List of UAVSAR flight-heading identifiers (e.g. ``['08508', '26505']``).
    fp_coh:
        Directory containing per-flight coherence ``.tif`` files.
    fp_inc:
        Directory containing pre-calculated incidence angle ``.tif`` files.
    fp_snowclimate:
        Directory containing the NSIDC-0768 snow-climatology NetCDF file.
    crs:
        Coordinate reference system string (default ``'EPSG:4326'``).
    res:
        Target spatial resolution in metres (default ``30``).

    Returns
    -------
    xr.Dataset
        Merged dataset containing all assembled data layers for this tile.
    """

    logger.info("Starting tile data assembly for flights %s", flight_ids)

    # 1. Validate inputs
    tile_aoi = validate_aoi(tile_aoi)
    date_pairs = validate_date_pairs(date_pairs)
    logger.debug("Tile AOI validated; %d date pairs provided", len(date_pairs))

    # 2. Create tile reference grid
    crs_obj = ProjCRS.from_user_input(crs)
    if crs_obj.is_geographic:
        # Convert the requested metric resolution to arc-degrees at ~45° latitude.
        res_grid = res / METRES_PER_DEGREE_AT_45_LAT
        tile_ref = make_reference_grid(aoi=tile_aoi, crs=crs, resolution=res_grid)
    else:
        # Projected CRS: resolution is already in metres.
        wgs84 = ProjCRS('EPSG:4326')
        res_grid = float(res)
        tile_ref = make_reference_grid(aoi=tile_aoi, crs=crs, resolution=res_grid)
        tile_aoi_proj = tile_aoi

        # get tile_aoi in WGS84 for APIs
        project = Transformer.from_crs(crs_obj, wgs84, always_xy=True).transform
        tile_aoi = transform(project, tile_aoi_proj)
        crs = 'EPSG:4326'

    logger.debug("Tile reference grid created with resolution %g (CRS units)", res_grid)

    # 3. Get UAVSAR coherence layers
    s = time.time()
    coh = get_uavsar_coherence(
        tile_aoi=tile_aoi,
        date_pairs=date_pairs,
        flight_ids=flight_ids,
        crs=crs,
        tile_ref_grid=tile_ref,
        fp=fp_coh,
    )
    e = time.time()
    logger.info("Tile: loaded coherence for flights %s in %.3f seconds.", flight_ids, e - s)
    logger.debug("Tile coherence dataset shape: %s", dict(coh.sizes))

    # 4. Get incidence angle
    s = time.time()
    incidence = get_uavsar_incidence(
        tile_aoi=tile_aoi,
        flight_ids=flight_ids,
        fp_inc=fp_inc,
        tile_ref_grid=tile_ref,
        crs=crs,
    )
    e = time.time()
    logger.info("Tile: loaded incidence angles for flights %s in %.3f seconds.", flight_ids, e - s)

    # 5. Get DEM
    s = time.time()
    try:
        dem = py3dep.get_dem(geometry=tile_aoi, resolution=res, crs=crs)
    except Exception as exc:
        logger.error("py3dep.get_dem failed: %s", exc)
        raise
    dem.rio.write_crs(crs)
    dem = dem.rio.reproject_match(tile_ref)
    e = time.time()
    logger.info("Tile: got DEM in %.3f seconds.", e - s)

    s = time.time()
    topo = get_topo_layers(dem=dem, tile_ref_grid=tile_ref)
    e = time.time()
    logger.info("Tile: got topo layers: %s in %.3f seconds.", list(topo.keys()), e - s)

    # 6. Get NLCD layers
    s = time.time()
    nlcd = get_nlcd_layers(tile_aoi=tile_aoi, crs=crs, tile_ref_grid=tile_ref)
    e = time.time()
    logger.info("Tile: got NLCD %s in %.3f seconds.", list(nlcd.keys()), e - s)

    s = time.time()
    snow_class = get_snow_climatology(tile_aoi=tile_aoi, crs=crs, fp=fp_snowclimate, tile_ref_grid=tile_ref)
    e = time.time()
    logger.info("Tile: got snow climatology: %s in %.3f seconds.", list(snow_class.keys()), e - s)

    # 7. Get AORC meteorological layers
    s = time.time()
    aorc = get_aorc_layers(tile_aoi=tile_aoi, date_pairs=date_pairs, crs=crs, tile_ref_grid=tile_ref)
    e = time.time()
    logger.info("Tile: got AORC: %s in %.3f seconds.", list(aorc.keys()), e - s)

    # 8. Get UCLA SWE/SD layers
    s = time.time()
    snow = get_snow_layers(tile_aoi=tile_aoi, date_pairs=date_pairs, crs=crs, tile_ref_grid=tile_ref)
    e = time.time()
    logger.info("Tile: got SWE layers: %s in %.3f seconds.", list(snow.keys()), e - s)

    ds_list = [coh, incidence, dem, topo, nlcd, snow_class, snow, aorc]
    validate_alignment(ds_list)
    logger.debug("Tile: all layers validated for spatial alignment")

    ds = xr.merge(ds_list, join='exact', compat='minimal')
    logger.info("Tile: merged %d layers into a single dataset", len(ds_list))

    # Fix OOM masking: use lazy .isnull() instead of .values which forces eager
    # evaluation of the entire array into RAM.
    nan_mask = ds['coherence'].isnull().all(dim=('flight_id', 'pair'))
    for var in ds.data_vars:
        if 'x' in ds[var].dims and 'y' in ds[var].dims:
            ds[var] = ds[var].where(~nan_mask, drop=False)

    return ds


def assemble_data_largeaoi(
    master_aoi: Polygon,
    date_pairs: List[Tuple[date, date]],
    flight_ids: List[str],
    fp_dest: str,
    fp_coh: str = '../data/coherence/',
    fp_inc: str = '../data/inc_angle/',
    fp_snowclimate: str = '../data/NSIDC-0768/',
    crs: str = 'EPSG:4326',
    res: int = 30,
    tile_size_m: float = 5000.0,
    overwrite: bool = False,
) -> None:
    """
    Assemble all data layers for a large AOI using an out-of-core tile-by-tile
    processing strategy to avoid out-of-memory (OOM) errors.

    The master AOI is divided into a regular grid of smaller spatial tiles
    (default 5 km × 5 km).  Each tile is processed independently by
    :func:`assemble_data`, and the resulting in-memory dataset is appended to
    a pre-initialised Zarr store via ``region``-based writes.

    Parameters
    ----------
    master_aoi:
        Full area of interest as a Shapely Polygon in the coordinate system
        specified by ``crs``.
    date_pairs:
        List of ``(start_date, end_date)`` tuples defining the UAVSAR
        coherence intervals.
    flight_ids:
        List of UAVSAR flight-heading identifiers (e.g. ``['08508', '26505']``).
    fp_dest:
        Destination path for the output ``.zarr`` store.
    fp_coh:
        Directory containing per-flight coherence ``.tif`` files.
    fp_inc:
        Directory containing pre-calculated incidence angle ``.tif`` files.
    fp_snowclimate:
        Directory containing the NSIDC-0768 snow-climatology NetCDF file.
    crs:
        Coordinate reference system string (default ``'EPSG:4326'``).
    res:
        Target spatial resolution in metres (default ``30``).
    tile_size_m:
        Side length of each square spatial tile in metres (default ``5000``).
    overwrite:
        When ``True`` an existing ``.zarr`` store at ``fp_dest`` is
        overwritten; when ``False`` the write will fail if the store already
        exists.

    Returns
    -------
    None
        Data is written directly to the Zarr store at ``fp_dest``.
    """
    logger.info(
        "Starting large-AOI data assembly for flights %s; tile_size_m=%.0f",
        flight_ids, tile_size_m,
    )

    master_aoi = validate_aoi(master_aoi)
    date_pairs = validate_date_pairs(date_pairs)

    # Build a global reference grid covering the full master AOI.
    crs_obj = ProjCRS.from_user_input(crs)
    if crs_obj.is_geographic:
        res_grid = res / METRES_PER_DEGREE_AT_45_LAT
        tile_size_deg = tile_size_m / METRES_PER_DEGREE_AT_45_LAT
    else:
        res_grid = float(res)
        tile_size_deg = tile_size_m  # already in CRS units (metres)

    global_ref = make_reference_grid(aoi=master_aoi, crs=crs, resolution=res_grid)
    logger.debug(
        "Global reference grid shape: %s", dict(global_ref.sizes)
    )

    # Split master_aoi into a regular grid of tiles.
    minx, miny, maxx, maxy = master_aoi.bounds
    x_starts = list(_frange(minx, maxx, tile_size_deg))
    y_starts = list(_frange(miny, maxy, tile_size_deg))
    total_tiles = len(x_starts) * len(y_starts)
    logger.info(
        "Split master AOI into %d tiles (%d cols × %d rows)",
        total_tiles, len(x_starts), len(y_starts),
    )

    # Initialise an empty Zarr store sized to the global grid (no compute).
    # We create a zero-filled template dataset with the correct structure by
    # processing the first tile, then write an empty shell for the full extent.
    logger.info("Initialising empty Zarr store at %s", fp_dest)
    # Build a template using the global grid coordinates but zero-filled data.
    # We defer the actual template creation to after the first tile is processed
    # so that we know the exact data variables and dtypes.
    _zarr_initialized = False
    global_x = global_ref.coords['x'].values
    global_y = global_ref.coords['y'].values

    tile_num = 0
    for tile_y0 in y_starts:
        tile_y1 = min(tile_y0 + tile_size_deg, maxy)
        for tile_x0 in x_starts:
            tile_x1 = min(tile_x0 + tile_size_deg, maxx)
            tile_num += 1

            from shapely.geometry import box as shapely_box
            tile_polygon = shapely_box(tile_x0, tile_y0, tile_x1, tile_y1)

            logger.info(
                "Processing tile %d/%d: bounds=(%.4f, %.4f, %.4f, %.4f)",
                tile_num, total_tiles, tile_x0, tile_y0, tile_x1, tile_y1,
            )

            try:
                tile_ds = assemble_data(
                    tile_aoi=tile_polygon,
                    date_pairs=date_pairs,
                    flight_ids=flight_ids,
                    fp_coh=fp_coh,
                    fp_inc=fp_inc,
                    fp_snowclimate=fp_snowclimate,
                    crs=crs,
                    res=res,
                )
            except Exception as exc:
                logger.error(
                    "Tile %d/%d failed: %s – skipping.", tile_num, total_tiles, exc
                )
                continue

            # Initialise the global Zarr store using the first successfully
            # processed tile as a structural template.
            if not _zarr_initialized:
                logger.info("Initialising Zarr store structure from first tile")
                template_ds = _build_global_template(tile_ds, global_x, global_y)

                # Clear stale encoding before writing.
                for var in template_ds.variables:
                    template_ds[var].encoding.pop('chunks', None)
                    template_ds[var].encoding.pop('preferred_chunks', None)

                chunk_dict = {
                    'pair': -1,
                    'y': SPATIAL_CHUNK_SIZE,
                    'x': SPATIAL_CHUNK_SIZE,
                }
                template_ds = template_ds.chunk(chunk_dict)

                zarr_mode = 'w' if overwrite else 'w-'
                template_ds.to_zarr(fp_dest, mode=zarr_mode, consolidated=True, compute=False)
                logger.info("Empty Zarr store initialised at %s", fp_dest)
                _zarr_initialized = True

            # Compute integer slice indices for this tile within the global grid.
            tile_x_vals = tile_ds.coords['x'].values
            tile_y_vals = tile_ds.coords['y'].values

            x_start_idx = int(np.searchsorted(global_x, tile_x_vals[0]))
            x_end_idx = x_start_idx + len(tile_x_vals)
            # global_y is descending (north→south), so we search in reversed order.
            y_start_idx = int(np.searchsorted(-global_y, -tile_y_vals[0]))
            y_end_idx = y_start_idx + len(tile_y_vals)

            region = {
                'x': slice(x_start_idx, x_end_idx),
                'y': slice(y_start_idx, y_end_idx),
            }

            # Clear encoding before region write.
            for var in tile_ds.variables:
                tile_ds[var].encoding.pop('chunks', None)
                tile_ds[var].encoding.pop('preferred_chunks', None)

            # Drop purely non-spatial 1-D dimension coordinates (e.g., 'flight_id',
            # 'pair').  These variables have no x/y dimension, so zarr region writes
            # require them to be absent.  They were already written into the global
            # template during store initialisation and do not change between tiles.
            _spatial_dims = {'x', 'y'}
            _coords_to_drop = [
                c for c in tile_ds.coords
                if set(tile_ds[c].dims).isdisjoint(_spatial_dims)
            ]
            tile_ds_for_write = tile_ds.drop_vars(_coords_to_drop)
            tile_ds_for_write.to_zarr(fp_dest, region=region)
            logger.info(
                "Tile %d/%d written to Zarr (x=%d:%d, y=%d:%d)",
                tile_num, total_tiles,
                x_start_idx, x_end_idx,
                y_start_idx, y_end_idx,
            )

    logger.info("Large-AOI assembly complete. Output written to %s", fp_dest)


def _frange(start: float, stop: float, step: float):
    """Yield float values from *start* to *stop* (exclusive) in increments of *step*."""
    val = start
    while val < stop:
        yield val
        val += step


def _build_global_template(tile_ds: xr.Dataset, global_x: np.ndarray, global_y: np.ndarray) -> xr.Dataset:
    """
    Build a zero-filled global-extent template dataset matching the structure of *tile_ds*.

    The template has the same data variables, dtypes, and non-spatial coordinates
    as *tile_ds*, but the ``x`` and ``y`` dimensions are replaced with the full
    global coordinate arrays.

    Parameters
    ----------
    tile_ds:
        A representative tile dataset whose structure (variables, dtypes, extra
        dimensions such as ``pair`` and ``flight_id``) defines the template.
    global_x:
        1-D array of global x-coordinates (ascending).
    global_y:
        1-D array of global y-coordinates (descending, north→south).

    Returns
    -------
    xr.Dataset
        Zero-filled dataset with global spatial extent.
    """
    ny = len(global_y)
    nx = len(global_x)
    template_vars: dict = {}

    for var in tile_ds.data_vars:
        da = tile_ds[var]
        # Build the new shape by replacing y/x sizes with global sizes.
        new_dims = []
        new_coords: dict = {}
        new_shape = []

        for dim in da.dims:
            if dim == 'y':
                new_dims.append('y')
                new_coords['y'] = global_y
                new_shape.append(ny)
            elif dim == 'x':
                new_dims.append('x')
                new_coords['x'] = global_x
                new_shape.append(nx)
            else:
                new_dims.append(dim)
                new_coords[dim] = da.coords[dim].values
                new_shape.append(da.sizes[dim])

        template_data = np.zeros(new_shape, dtype=da.dtype)
        template_vars[var] = xr.DataArray(
            template_data,
            dims=new_dims,
            coords=new_coords,
            attrs=da.attrs,
        )

    return xr.Dataset(template_vars)


def get_uavsar_coherence(
    tile_aoi: Polygon,
    date_pairs: List[Tuple[date, date]],
    flight_ids: List[str],
    crs: str,
    tile_ref_grid: Union[xr.DataArray, xr.Dataset],
    fp: str = '../data/coherence/',
) -> xr.Dataset:
    """
    Load UAVSAR coherence files and assemble a 4-D cube ``(flight_id, pair, y, x)``.

    Files are read from flight-specific sub-directories under ``fp``.  Before
    reprojection, each raster is sliced to the bounding box of ``tile_aoi``
    in the raster's native CRS to avoid loading the entire file into memory.
    The sliced array is then aligned to ``tile_ref_grid``.

    Expected file structure::

        fp / {flight_id} / *{flight_id}*{date1}*{date2}*.coh.tif

    Parameters
    ----------
    tile_aoi:
        Spatial tile as a Shapely Polygon (a small chunk of the master extent).
        Must be in the CRS given by ``crs``.
    date_pairs:
        List of ``(start_date, end_date)`` tuples defining the coherence
        intervals.
    flight_ids:
        List of UAVSAR flight-heading identifiers.
    crs:
        Coordinate reference system string of ``tile_aoi`` (e.g. ``'EPSG:4326'``).
    tile_ref_grid:
        Reference grid for this tile used to spatially align all arrays.
    fp:
        Root directory that contains per-flight sub-directories.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions ``(flight_id, pair, y, x)`` and a single
        ``coherence`` variable.

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

            search_pattern = f"*{fid}*{date_str1}*{date_str2}*VV_s2*int_coh*" # TODO - pass in all other parameters needed to uniquely identify the coherence file.
            found_files = list(flight_dir.glob(search_pattern))

            if not found_files:
                logger.warning(f"No coherence file found for {fid} on {date_str1}_{date_str2}. Padding with NaNs.")
                dummy_da = xr.full_like(tile_ref_grid, fill_value=np.nan)
                # Wrap it in a Dataset container with the correct variable name
                dummy_ds = dummy_da.to_dataset(name='coherence')
                pair_arrays.append(dummy_ds)
                continue # Move on to the next date pair

            # if not found_files:
            #     raise FileNotFoundError(
            #         f"Could not find coherence file for flight {fid} and "
            #         f"dates {pair_name} in {flight_dir}"
            #     )

            if len(found_files) > 1:
                logger.warning(
                    "Multiple coherence files found for flight %s and dates %s; "
                    "using %s",
                    fid, pair_name, found_files[0],
                )

            file_path = found_files[0]
            logger.debug("Found coherence file: %s", file_path)

            # Load raster lazily, write CRS, and rename the data variable.
            da = xr.open_dataset(file_path, chunks={})
            da.rio.write_crs(crs, inplace=True)
            da = da.rename({list(da.data_vars.keys())[0]: 'coherence'})

            # Pre-clip to the tile bounding box in the raster's native CRS
            # before calling reproject_match to avoid reading the full raster.
            # Guard: skip pre-clipping when tile_aoi is None (e.g. testing
            # without a spatial filter) or when the native CRS cannot be
            # resolved (e.g. mocked in tests).
            if tile_aoi is not None:
                # try:
                #     native_crs = ProjCRS.from_user_input(da.rio.crs)
                #     tile_aoi_native = gpd.GeoSeries([tile_aoi], crs=crs).to_crs(
                #         native_crs.to_epsg() or native_crs.to_wkt()
                #     )
                #     minx, miny, maxx, maxy = tile_aoi_native.total_bounds

                #     # Handle both ascending and descending y-axes.
                #     y_vals = da.coords['y'].values
                #     if len(y_vals) >= 2 and y_vals[0] > y_vals[-1]:
                #         # Descending y (north→south): slice maxy first, miny last.
                #         da = da.sel(x=slice(minx, maxx), y=slice(maxy, miny))
                #     else:
                #         # Ascending y (south→north).
                #         da = da.sel(x=slice(minx, maxx), y=slice(miny, maxy))
                # except Exception as exc:
                #     logger.debug(
                #         "Native-CRS pre-clip skipped for coherence file %s: %s",
                #         file_path, exc,
                #     )
                try:
                    # Use our new fast function!
                    print(f"Using VRT to read and reproject {file_path}")
                    ds_matched = read_and_reproject_rasterio(
                        filepath=str(file_path), 
                        ref_da=tile_ref_grid, 
                        var_name='coherence',
                        resampling=Resampling.bilinear
                    )
                    pair_arrays.append(ds_matched)
                    
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
                    dummy_da = xr.full_like(tile_ref_grid, fill_value=np.nan)
                    pair_arrays.append(dummy_da.to_dataset(name='coherence'))

            # Align to the tile reference grid.
            # da_matched = da.rio.reproject_match(tile_ref_grid)

            # pair_arrays.append(da_matched)

        # Concatenate all date pairs for this flight.
        pair_coord = xr.DataArray(
            [f"{s.strftime('%y%m%d')}_{e.strftime('%y%m%d')}" for s, e in date_pairs],
            dims=['pair'],
            name='pair',
        )
        print(pair_arrays)
        flight_da = xr.concat(pair_arrays, dim=pair_coord, coords="minimal", compat="override")
        flight_arrays.append(flight_da)

    # Concatenate all flights into the final 4-D array.
    flight_coord = xr.DataArray(flight_ids, dims=['flight_id'], name='flight_id')
    coherence_da = xr.concat(flight_arrays, dim=flight_coord)
    logger.debug(
        "Coherence array assembled with shape: %s", dict(coherence_da.sizes)
    )

    return coherence_da


def get_uavsar_incidence(
    tile_aoi: Polygon,
    flight_ids: List[str],
    fp_inc: Union[str, Path],
    tile_ref_grid: Union[xr.DataArray, xr.Dataset],
    crs: str = 'EPSG:4326',
) -> xr.DataArray:
    """
    Load pre-calculated UAVSAR incidence angle files and concatenate them along
    a ``flight_id`` dimension.

    Before reprojection, each raster is sliced to the bounding box of
    ``tile_aoi`` projected into the raster's native CRS.  This avoids reading
    the entire raster into memory.

    Parameters
    ----------
    tile_aoi:
        Spatial tile as a Shapely Polygon (a small chunk of the master extent).
        Must be in the CRS given by ``crs``.
    flight_ids:
        List of UAVSAR flight-heading identifiers (e.g. ``['08508', '26505']``).
    fp_inc:
        Directory containing the pre-calculated incidence angle ``.tif`` files.
        Files are matched using the pattern ``*{flight_id}*s2.inc.tif``.
    tile_ref_grid:
        Reference grid for this tile used to spatially align all arrays.
    crs:
        Coordinate reference system string of ``tile_aoi`` (default ``'EPSG:4326'``).

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

        # Pre-clip to the tile bounding box in the raster's native CRS to
        # avoid reading the full file into memory before reprojection.
        # Falls back gracefully if the native CRS cannot be resolved.
        try:
            native_crs = ProjCRS.from_user_input(da.rio.crs)
            tile_aoi_native = gpd.GeoSeries([tile_aoi], crs=crs).to_crs(
                native_crs.to_epsg() or native_crs.to_wkt()
            )
            minx, miny, maxx, maxy = tile_aoi_native.total_bounds

            # Handle both ascending and descending y-axes.
            y_vals = da.coords['y'].values
            if len(y_vals) >= 2 and y_vals[0] > y_vals[-1]:
                # Descending y (north→south): slice maxy first, miny last.
                da = da.sel(x=slice(minx, maxx), y=slice(maxy, miny))
            else:
                # Ascending y (south→north).
                da = da.sel(x=slice(minx, maxx), y=slice(miny, maxy))
        except Exception as exc:
            logger.debug(
                "Native-CRS pre-clip skipped for incidence file %s: %s",
                file_path, exc,
            )

        # Write CRS before reprojecting.
        da.rio.write_crs(crs, inplace=True)

        # Align to the tile reference grid.
        da_matched = da.rio.reproject_match(tile_ref_grid)

        da_matched.name = 'incidence_angle'

        inc_arrays.append(da_matched)

    # Concatenate into a single DataArray with the 'flight_id' dimension.
    flight_coord = xr.DataArray(flight_ids, dims=['flight_id'], name='flight_id')
    incidence_da = xr.concat(inc_arrays, dim=flight_coord)
    logger.debug(
        "Incidence angle array assembled with shape: %s", dict(incidence_da.sizes)
    )

    return incidence_da


# =============================================================================
# FUNCTIONS FOR DOWNLOADING ENVIRONMENTAL DATA
# =============================================================================

def get_nlcd_layers(
    tile_aoi: Polygon,
    crs: str,
    years: Dict[str, List[int]] = {'cover': [2019], 'canopy': [2019]},
    tile_ref_grid: Optional[Union[xr.DataArray, xr.Dataset]] = None,
) -> xr.Dataset:
    """
    Retrieve National Land Cover Database (NLCD) layers for the given tile AOI.

    The NLCD API is queried only for the extent of ``tile_aoi``, ensuring that
    only tile-relevant data is transferred over the network.

    Parameters
    ----------
    tile_aoi:
        Spatial tile as a Shapely Polygon (a small chunk of the master extent).
        Must be in the CRS given by ``crs``.
    crs:
        Coordinate reference system of ``tile_aoi``.
    years:
        Dictionary specifying which NLCD layers and years to download.
        Default is ``{'cover': [2019], 'canopy': [2019]}``.
    tile_ref_grid:
        Optional reference grid for this tile used to spatially align the output.

    Returns
    -------
    xr.Dataset
        Dataset of NLCD layers aligned to ``tile_ref_grid`` (when provided).
    """
    logger.debug("Fetching NLCD layers for years: %s", years)

    g = gpd.GeoSeries([tile_aoi], crs=crs)

    try:
        ds = gh.nlcd_bygeom(geometry=g, years=years)[0]
    except Exception as exc:
        logger.error("Failed to fetch NLCD data from pygeohydro: %s", exc)
        raise

    if tile_ref_grid is not None:
        ds = ds.rio.reproject_match(tile_ref_grid)

    return ds

def get_snow_climatology(
    tile_aoi: Polygon,
    crs: str,
    fp: str,
    tile_ref_grid: Optional[Union[xr.DataArray, xr.Dataset]] = None,
) -> xr.Dataset:
    """
    Load the NSIDC-0768 seasonal snow classification dataset for the tile AOI.

    The expected NetCDF file is
    ``SnowClass_NA_300m_10.0arcsec_2021_v01.0.nc``, which must be placed
    directly under ``fp``.  Download instructions are available at
    https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0768_global_seasonal_snow_classification_v01/.

    The dataset is pre-clipped to the bounding box of ``tile_aoi`` before any
    reprojection to avoid loading the full continental dataset into memory.

    Parameters
    ----------
    tile_aoi:
        Spatial tile as a Shapely Polygon (a small chunk of the master extent).
        Must be in the CRS given by ``crs``.
    crs:
        Coordinate reference system of ``tile_aoi``.
    fp:
        Directory containing the snow-climatology NetCDF file.
    tile_ref_grid:
        Optional reference grid for this tile used to spatially align the output.
        When ``None`` the dataset is clipped directly to ``tile_aoi``.

    Returns
    -------
    xr.Dataset
        Dataset containing the ``snow_class`` variable, aligned to
        ``tile_ref_grid`` (when provided).

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

    # The climatology dataset is always in EPSG:4326; project the tile AOI bounds
    # to 4326 for slicing, regardless of the input ``crs``.
    g = gpd.GeoSeries([tile_aoi], crs=crs).to_crs('EPSG:4326').total_bounds
    minx, miny, maxx, maxy = g
    logger.debug(
        "Clipping snow climatology to tile AOI bounds: %.4f, %.4f, %.4f, %.4f",
        minx, miny, maxx, maxy,
    )

    ds = ds.sel(x=slice(minx, maxx), y=slice(miny, maxy))

    ds = ds.rio.write_crs('EPSG:4326')

    if tile_ref_grid is not None:
        ds = ds.rio.reproject_match(tile_ref_grid)
    else:
        g = gpd.GeoSeries([tile_aoi], crs=crs)
        ds = ds.rio.clip(g, crs=crs)

    return ds


def get_topo_layers(
    dem: xr.DataArray,
    tile_ref_grid: Optional[Union[xr.DataArray, xr.Dataset]] = None,
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
    tile_ref_grid:
        Optional reference grid for this tile used to spatially align the output.

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

    if tile_ref_grid is not None:
        ds = ds.rio.reproject_match(tile_ref_grid)

    return ds


def get_snow_layers(
    tile_aoi: Polygon,
    crs: str,
    date_pairs: List[Tuple[date, date]],
    metrics: Union[str, List[str]] = 'all',
    tile_ref_grid: Optional[Union[xr.DataArray, xr.Dataset]] = None,
) -> xr.Dataset:
    """
    Retrieve UCLA Western US snow reanalysis (WUS_UCLA_SR) layers for the tile AOI.

    SWE and snow-depth granules are searched via the Earthaccess API,
    processed into time series, and summarised into per-date-pair metrics.
    Only the spatial extent of ``tile_aoi`` is requested from the API.

    Parameters
    ----------
    tile_aoi:
        Spatial tile as a Shapely Polygon (a small chunk of the master extent).
        Must be in the CRS given by ``crs``.
    crs:
        Coordinate reference system of ``tile_aoi``.
    date_pairs:
        List of ``(start_date, end_date)`` tuples defining the intervals
        for which snow metrics are computed.
    metrics:
        Either ``'all'`` or a list of metric names accepted by
        :func:`get_snow_metrics`.
    tile_ref_grid:
        Optional reference grid for this tile used to spatially align the output.

    Returns
    -------
    xr.Dataset
        Dataset of snow metrics indexed by a ``pair`` coordinate.
    """
    xx, yy = tile_aoi.exterior.coords.xy
    x = xx.tolist()
    y = yy.tolist()

    g = gpd.GeoSeries([tile_aoi], crs=crs)

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
    if tile_ref_grid is not None:
        ds = ds.rio.reproject_match(tile_ref_grid)
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
        'big_accum',
        'snow_status_change',
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
        # Mean of only the positive daily SWE changes (new snow mass).
        out['swe_accum'] = swe_diff.where(swe_diff > 0, other=0).mean(dim='time') 

    if 'big_accum' in metrics:
        # Value of 1 if there was any positive daily SWE change > 1 m (heavy snowfall event), else 0.
        storm = (swe_diff > 1.0).any(dim='time').astype(int)
        out['big_accum'] = storm.clip(0, 1)

    if 'swe_ablate' in metrics:
        # Mean of only the negative daily SWE changes (melt/sublimation mass).
        # Absolute value makes this easier to interpret as a positive feature.
        out['swe_ablate'] = abs(swe_diff.where(swe_diff < 0, other=0).mean(dim='time'))

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

        out['snow_status_change'] = (has_snow_sec - has_snow_ref).clip(-1, 1)

    # ── CF variable metadata ──────────────────────────────────────────────────
    # NOTE: units and valid_range for SWE/SD-derived metrics need manual
    # verification — see the draft PR comment for details.
    if 'swe_accum' in out:
        out['swe_accum'].attrs.update({
            'units': 'm', 
            'long_name': 'accumulated SWE gain over coherence interval',
        })
    if 'swe_ablate' in out:
        out['swe_ablate'].attrs.update({
            'units': 'm',
            'long_name': 'accumulated SWE loss (absolute value) over coherence interval',
        })
    if 'density_change' in out:
        out['density_change'].attrs.update({
            'units': '1', 
            'long_name': 'change in bulk snow density over coherence interval',
        })
    if 'big_storm' in out:
        out['big_storm'].attrs.update({
            'units': '1',
            'long_name': 'indicator of any heavy snowfall event (>1 m daily SWE gain) during coherence interval',
        })
    if 'snow_status_change' in out:
        out['snow_status_change'].attrs.update({
            'units': '1',
            'long_name': 'change in snow presence status over coherence interval (+1=snow appeared, -1=snow disappeared, 0=no change)',
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
    tile_aoi: Polygon,
    date_pairs: List[Tuple[date, date]],
    crs: str,
    tile_ref_grid: Optional[Union[xr.DataArray, xr.Dataset]] = None,
    metrics: Union[str, List[str]] = 'all',
) -> xr.Dataset:
    """
    Retrieve AORC (Analysis of Record for Calibration) meteorological layers.

    Data are read directly from the NOAA public S3 bucket and summarised into
    per-date-pair metrics using :func:`get_aorc_metrics`.  Only the spatial
    extent of ``tile_aoi`` is clipped from the remote store.

    Parameters
    ----------
    tile_aoi:
        Spatial tile as a Shapely Polygon (a small chunk of the master extent).
        Must be in the CRS given by ``crs``.
    date_pairs:
        List of ``(start_date, end_date)`` tuples defining the intervals
        for which AORC metrics are computed.
    crs:
        Coordinate reference system of ``tile_aoi``.
    tile_ref_grid:
        Optional reference grid for this tile used to spatially align the output.
    metrics:
        Either ``'all'`` or a list of metric names accepted by
        :func:`get_aorc_metrics`.

    Returns
    -------
    xr.Dataset
        Dataset of AORC metrics indexed by a ``pair`` coordinate.
    """
    years = get_years(date_pairs)
    g = gpd.GeoSeries([tile_aoi], crs=crs)
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

    # Clip to the tile AOI and rename spatial dimensions.
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

    if tile_ref_grid is not None:
        ds = ds.rio.reproject_match(tile_ref_grid)
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
        'avg_precip',
        'total_rain',
        'avg_rain',
        'total_snow',
        'avg_snow',
        'acq_day_precip',
        'mean_wind',
        'max_wind',
        # 'hours_blowing_snow',
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
            .mean(dim='time')
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
    if 'avg_precip' in metrics:
        ds['avg_precip'] = precip.mean(dim='time')
    if 'total_rain' in metrics:
        # Sum of precipitation only when temperature is above 0 °C.
        ds['total_rain'] = precip.where(temp > RAIN_SNOW_THRESHOLD, other=0).sum(dim='time')
    if 'avg_rain' in metrics:
        ds['avg_rain'] = precip.where(temp > RAIN_SNOW_THRESHOLD, other=0).mean(dim='time')
    if 'total_snow' in metrics:
        # Sum of precipitation only when temperature is at or below 0 °C.
        ds['total_snow'] = precip.where(temp <= RAIN_SNOW_THRESHOLD, other=0).sum(dim='time')
    if 'avg_snow' in metrics:
        ds['avg_snow'] = precip.where(temp <= RAIN_SNOW_THRESHOLD, other=0).mean(dim='time')
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
            'units': 'positive degree-hours', 
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
        'avg_precip': 'average precipitation over coherence interval',
        'total_rain': 'total liquid precipitation over coherence interval',
        'avg_rain': 'average liquid precipitation over coherence interval',
        'total_snow': 'total solid precipitation over coherence interval',
        'avg_snow': 'average solid precipitation over coherence interval',
        'acq_day_precip': 'total precipitation on acquisition days',
    }
    for var, long_name in _precip_meta.items():
        if var in ds:
            ds[var].attrs.update({'long_name': long_name,
                                  'units': 'mm'})

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

import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

def read_and_reproject_rasterio(
    filepath: str, 
    ref_da: xr.DataArray, 
    var_name: str, 
    resampling=Resampling.bilinear
    ) -> xr.Dataset:
    """
    High-performance read and reproject using raw rasterio and WarpedVRT.
    Bypasses Xarray overhead.
    """
    # Extract target grid properties from our reference DataArray
    dst_crs = ref_da.rio.crs
    dst_transform = ref_da.rio.transform()
    dst_width = ref_da.rio.width
    dst_height = ref_da.rio.height
    
    with rasterio.open(filepath) as src:
        # Create a virtual warped dataset in memory
        with WarpedVRT(
            src,
            crs=dst_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            resampling=resampling
        ) as vrt:
            # Read the data into a numpy array (band 1)
            data = vrt.read(1)
            
            # Mask out nodata values (UAVSAR usually uses 0.0 for NoData)
            nodata = src.nodata or 0.0
            data = np.where(data == nodata, np.nan, data)
            
    # Wrap the fast numpy array back into an Xarray Dataset so it plays nice with the rest of your code
    da = xr.DataArray(
        data,
        coords=ref_da.coords,
        dims=ref_da.dims,
        name=var_name
    )
    
    return da.to_dataset()
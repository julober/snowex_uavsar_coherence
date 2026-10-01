import logging
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import asf_search as asf
import earthaccess
import geopandas as gpd
import h5py
import numpy as np
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor on xr.DataArray)
import shapely
import xarray as xr
from pyproj import Transformer
from rasterio.enums import Resampling
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)

# =============================================================================
# MODULE-LEVEL CONSTANTS
# =============================================================================

BASE_GRIDS: str = 'science/LSAR/GUNW/grids/frequencyA'
"""Root group for the NISAR GUNW interferometric grid layers."""

BASE_META: str = 'science/LSAR/GUNW/metadata'
"""Root group for the NISAR GUNW radar-grid metadata (e.g. incidence angle)."""

LAYER_REGISTRY: dict = {
    'unwrapped_phase': {
        'path': 'unwrappedInterferogram/{pol}/unwrappedPhase',
        'coords': 'unwrapped',
    },
    'coherence_unwrapped': {
        'path': 'unwrappedInterferogram/{pol}/coherenceMagnitude',
        'coords': 'unwrapped',
    },
    'wrapped_phase': {
        'path': 'wrappedInterferogram/{pol}/wrappedInterferogram',
        'coords': 'wrapped',
    },
    'coherence_wrapped': {
        'path': 'wrappedInterferogram/{pol}/coherenceMagnitude',
        'coords': 'wrapped',
    },
    'incidence_angle': {
        'dataset': 'incidenceAngle',
        'coords': 'metadata',
    },
    'parallel_baseline': {
        'dataset': 'parallelBaseline',
        'coords': 'metadata',
    },
    'perpendicular_baseline': {
        'dataset': 'perpendicularBaseline',
        'coords': 'metadata',
    },
    'reference_slant_range': {
        'dataset': 'referenceSlantRange',
        'coords': 'metadata',
    },
    'secondary_slant_range': {
        'dataset': 'secondarySlantRange',
        'coords': 'metadata',
    },
}
"""Maps a friendly layer name to its relative HDF5 path template (``{pol}`` is
filled in with the requested polarization) and the coordinate group used to
georeference it. Layers with coords ``'metadata'`` (``incidence_angle``,
``parallel_baseline``, ``perpendicular_baseline``, ``reference_slant_range``,
``secondary_slant_range``) live under the metadata
radar grid rather than a polarized interferogram group; they are identified by
a ``'dataset'`` name under ``radarGrid`` and keep every height level as a band."""

COORD_PATHS: dict = {
    'unwrapped': ('unwrappedInterferogram/{pol}/xCoordinates', 'unwrappedInterferogram/{pol}/yCoordinates'),
    'wrapped': ('wrappedInterferogram/{pol}/xCoordinates', 'wrappedInterferogram/{pol}/yCoordinates'),
    'metadata': ('radarGrid/xCoordinates', 'radarGrid/yCoordinates'),
}
"""Relative x/y coordinate dataset paths for each coordinate group."""


def _get_epsg(hf: h5py.File, polarization: str) -> int:
    """Read the UTM EPSG code from whichever grid group is present in the file."""
    for group in ('unwrappedInterferogram', 'wrappedInterferogram'):
        path = f'{BASE_GRIDS}/{group}/{polarization}/projection'
        if path in hf:
            return int(hf[path][()])
    raise KeyError(
        f"Could not find a 'projection' dataset for polarization '{polarization}' in this granule."
    )


def _read_coords(hf: h5py.File, coord_group: str, polarization: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read the x/y coordinate arrays for a given coordinate group."""
    x_rel, y_rel = COORD_PATHS[coord_group]
    x_rel, y_rel = x_rel.format(pol=polarization), y_rel.format(pol=polarization)
    base = BASE_META if coord_group == 'metadata' else BASE_GRIDS
    x = hf[f'{base}/{x_rel}'][:]
    y = hf[f'{base}/{y_rel}'][:]
    return x, y


def _read_radar_grid_layer(
    hf: h5py.File, dataset: str, name: str, x: np.ndarray, y: np.ndarray, epsg: int
) -> xr.DataArray:
    """Read a radar-grid metadata layer with all height levels as bands."""
    heights = hf[f'{BASE_META}/radarGrid/heightAboveEllipsoid'][:]
    data = hf[f'{BASE_META}/radarGrid/{dataset}'][:]
    if data.dtype != np.float64:  # keep float64 (slant ranges need the precision)
        data = data.astype(np.float32)
    data[data == 0] = np.nan

    da = xr.DataArray(
        data, dims=['band', 'y', 'x'], coords={'band': heights, 'x': x, 'y': y}, name=name
    )
    return da.rio.write_crs(f'EPSG:{epsg}').rio.write_nodata(np.nan)


def _read_layer(hf: h5py.File, layer: str, polarization: str, epsg: int) -> xr.DataArray:
    """Read and mask a single requested layer, returning a georeferenced DataArray."""
    if layer not in LAYER_REGISTRY:
        raise ValueError(
            f"Unknown layer '{layer}'. Valid options are: {sorted(LAYER_REGISTRY)}"
        )

    entry = LAYER_REGISTRY[layer]
    x, y = _read_coords(hf, entry['coords'], polarization)

    if entry['coords'] == 'metadata':
        return _read_radar_grid_layer(hf, entry['dataset'], layer, x, y, epsg)

    ds_path = f"{BASE_GRIDS}/{entry['path'].format(pol=polarization)}"
    ds = hf[ds_path]

    if np.issubdtype(ds.dtype, np.complexfloating):
        data = ds[:]
    else:
        data = ds[:].astype(np.float32)
        fill_value = ds.attrs.get('_FillValue', None)
        if fill_value is not None:
            data = np.where(data == fill_value, np.nan, data)
        data = np.where(data == 0, np.nan, data)

    da = xr.DataArray(data, dims=['y', 'x'], coords={'x': x, 'y': y}, name=layer)
    return da.rio.write_crs(f'EPSG:{epsg}').rio.write_nodata(np.nan)

_CRID_RE = re.compile(r'^[A-Z]\d{5}$')


def _scene_crid(scene_name: str) -> Optional[str]:
    """Return the processing code (CRID, e.g. ``'P05023'``) token of a scene name, if any."""
    for token in scene_name.split('_'):
        if _CRID_RE.match(token):
            return token
    return None


def _filter_by_crid(results: list, crid: Optional[str]) -> list:
    """Keep only search results whose scene name carries the given CRID token.

    ``asf.search`` has no CRID/processing-version keyword for NISAR, so this is
    applied client-side. ``crid=None`` returns ``results`` unchanged.
    """
    if crid is None:
        return list(results)
    return [r for r in results if crid in r.properties['sceneName'].split('_')]


def _warn_mixed_crids(results: list) -> None:
    """Warn when several processing versions exist for the same date pair."""
    by_pair: dict = {}
    for r in results:
        name = r.properties['sceneName']
        try:
            pair = parse_dates(name)
        except IndexError:
            continue
        by_pair.setdefault(pair, set()).add(_scene_crid(name))
    mixed = {pair: sorted(c for c in crids if c) for pair, crids in by_pair.items() if len(crids) > 1}
    if mixed:
        all_crids = sorted({c for crids in mixed.values() for c in crids})
        logger.warning(
            f"{len(mixed)} date pair(s) have multiple processing versions {all_crids}; "
            f"pass crid=... to select one."
        )


def download_nisar(
    track: int,
    frame: int,
    start_date: str,
    end_date: str,
    aoi: Union[str, Path, gpd.GeoDataFrame],
    layers: List[str],
    output_dir: Union[str, Path] = '.',
    polarization: str = 'HH',
    crid: Optional[str] = None,
) -> Tuple[List[str], List[Path]]:
    """
    Search for NISAR GUNW granules over a track/frame and date range, clip a
    set of requested layers to an AOI, and download them as GeoTIFFs.

    Follows the methodology developed in
    ``notebooks/04_nisar_coherence.ipynb``: granules are streamed directly
    from the ASF Earthdata Cloud over HTTPS (no full-file download), the
    requested layers are read out of the HDF5 hierarchy, masked, clipped to
    the AOI, and only the clipped result is written to disk.

    Parameters
    ----------
    track : int
        Relative orbit / track number to search for.
    frame : int
        Frame number to search for.
    start_date : str
        Start of the search window (e.g. ``'2026-02-01'``).
    end_date : str
        End of the search window (e.g. ``'2026-02-20'``).
    aoi : str | Path | geopandas.GeoDataFrame
        Area of interest to clip layers to. Either a path to a vector file
        readable by geopandas (shapefile, GeoJSON, etc.) or an already-loaded
        GeoDataFrame.
    layers : list of str
        Names of the layers to extract for each granule. Valid options are
        the keys of ``LAYER_REGISTRY``: ``'unwrapped_phase'``,
        ``'coherence_unwrapped'``, ``'wrapped_phase'``,
        ``'coherence_wrapped'``, ``'incidence_angle'``,
        ``'parallel_baseline'``, ``'perpendicular_baseline'``,
        ``'reference_slant_range'``, ``'secondary_slant_range'``. The last five
        are radar-grid metadata layers written as multi-band GeoTIFFs with one
        band per height level above the ellipsoid (band descriptions hold the
        height in metres).
    output_dir : str | Path
        Directory to write clipped GeoTIFFs into. Created if it doesn't exist.
    polarization : str, default='HH'
        Polarization channel to read the interferogram layers from.
    crid : str, optional
        Processing code (composite release ID) to keep, e.g. ``'P05023'``.
        ``asf.search`` cannot filter on this, so granules whose scene name
        lacks this token are dropped after the search. If ``None`` (default),
        all processing versions are kept (a warning is logged if a date pair
        exists in more than one version).

    Returns
    -------
    granule_names : list of str
        Scene names of every granule found matching the search criteria.
    downloaded_files : list of Path
        Paths to every clipped GeoTIFF for the requested layers, whether
        freshly written this call or already present in ``output_dir``. Files
        that already exist are left untouched and skipped rather than
        re-downloaded, so re-running with the same arguments is cheap.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    earthaccess.login()

    results = _search_nisar('GUNW', track, frame, start_date, end_date)

    if crid is None:
        _warn_mixed_crids(results)
    else:
        n_found = len(results)
        results = _filter_by_crid(results, crid)
        logger.info(f"Kept {len(results)} of {n_found} granule(s) with processing code {crid}.")

    granule_names = [r.properties['sceneName'] for r in results]
    logger.info(f"Found {len(granule_names)} matching granule(s).")

    if not results:
        return granule_names, []

    boundary = _load_boundary(aoi)

    fs = earthaccess.get_fsspec_https_session()
    downloaded_files: List[Path] = []

    for granule in results:
        scene_name = granule.properties['sceneName']
        out_paths = {layer: output_dir / f'{scene_name}_{layer}.tif' for layer in layers}

        for layer, out_path in out_paths.items():
            if out_path.exists():
                logger.info(f"  {out_path.name} already exists, skipping")
                downloaded_files.append(out_path)

        missing_layers = [layer for layer in layers if not out_paths[layer].exists()]
        if not missing_layers:
            continue

        url = granule.properties['url']
        logger.info(f"Processing granule: {scene_name}")

        with h5py.File(fs.open(url, cache_type='background', block_size=16 * 1024 * 1024), 'r') as hf:
            epsg = _get_epsg(hf, polarization)
            clip_geom = boundary.to_crs(f'EPSG:{epsg}').geometry

            for layer in missing_layers:
                da = _read_layer(hf, layer, polarization, epsg)
                clipped = da.rio.clip(clip_geom, all_touched=True, drop=True)

                out_path = out_paths[layer]
                clipped.rio.to_raster(out_path, bigtiff='YES')
                if 'band' in clipped.dims:
                    with rasterio.open(out_path, 'r+') as dst:
                        for i, h in enumerate(clipped['band'].values, start=1):
                            dst.set_band_description(i, f'height={float(h):g}')
                downloaded_files.append(out_path)
                logger.info(f"  wrote {out_path.name} ({clipped.shape[-1]}x{clipped.shape[-2]})")

    return granule_names, downloaded_files


def _search_nisar(level: str, track: int, frame: int, start_date: str, end_date: str) -> list:
    """Search ASF for NISAR granules of a given processing level over a track/frame."""
    logger.info(f"Searching ASF for NISAR {level} granules: track={track}, frame={frame}, {start_date}–{end_date}")
    return list(asf.search(
        dataset='NISAR',
        processingLevel=level,
        relativeOrbit=track,
        frame=frame,
        start=start_date,
        end=end_date,
    ))


def _load_boundary(aoi: Union[str, Path, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Return the AOI as a GeoDataFrame, reading it from disk if given a path."""
    return aoi if isinstance(aoi, gpd.GeoDataFrame) else gpd.read_file(aoi)


GSLC_BASE_GRIDS: str = 'science/LSAR/GSLC/grids/frequencyA'
"""Root group for the NISAR GSLC backscatter grids."""

GSLC_LAYER_REGISTRY: dict = {
    'backscatter': {
        'group': GSLC_BASE_GRIDS,
        'complex': True,
        'clip': 'pixels',
    },
    'NEBZ': {
        'group': 'science/LSAR/GSLC/metadata/calibrationInformation/frequencyA/noiseEquivalentBackscatter',
        'complex': False,
        'clip': 'cells',
    },
}
"""Maps a GSLC layer name to the HDF5 group holding its per-polarization dataset
(named by the polarization, e.g. ``.../HH``) together with that layer's own
``xCoordinates``/``yCoordinates``/``projection``. ``clip='pixels'`` clips to the
AOI polygon at pixel level (full-resolution backscatter); ``clip='cells'`` keeps
every coarse grid cell overlapping the AOI (noise-equivalent backscatter)."""


def _gslc_window(
    x: np.ndarray, y: np.ndarray, x_spacing: float, y_spacing: float,
    bounds: Tuple[float, float, float, float],
) -> Tuple[slice, slice]:
    """Pixel window (rows, cols) of the grid that overlaps ``bounds``.

    ``x``/``y`` are pixel-centre coordinates (``y`` normally descending);
    ``bounds`` is ``(xmin, ymin, xmax, ymax)`` in the same CRS. Raises
    ``ValueError`` if the bounds do not intersect the grid.
    """
    xmin, ymin, xmax, ymax = bounds
    hx, hy = abs(x_spacing) / 2, abs(y_spacing) / 2
    cols = np.flatnonzero((x + hx > xmin) & (x - hx < xmax))
    rows = np.flatnonzero((y + hy > ymin) & (y - hy < ymax))
    if rows.size == 0 or cols.size == 0:
        raise ValueError("The AOI does not intersect this granule's GSLC grid.")
    return slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)


def _grid_spacing(hf: h5py.File, group: str, name: str, coords: np.ndarray) -> float:
    """Coordinate spacing from ``{name}CoordinateSpacing`` if present, else from the coordinates."""
    path = f'{group}/{name}CoordinateSpacing'
    if path in hf:
        return float(hf[path][()])
    if len(coords) < 2:
        raise ValueError(f"Cannot determine {name} spacing from a single coordinate in {group}.")
    return float(coords[1] - coords[0])


def _read_gslc(
    hf: h5py.File, layer: str, polarization: str, boundary: gpd.GeoDataFrame,
    epsg: Optional[int] = None,
) -> xr.DataArray:
    """Read a GSLC layer for ``polarization``, only over the AOI's pixel window.

    The backscatter raster is tens of GB, so only the h5py slice covering the
    AOI bounding box is read; the caller clips it to the exact AOI geometry.
    Coordinates and projection come from the layer's own group.
    """
    if layer not in GSLC_LAYER_REGISTRY:
        raise ValueError(
            f"Unknown GSLC layer '{layer}'. Valid options are: {sorted(GSLC_LAYER_REGISTRY)}"
        )
    entry = GSLC_LAYER_REGISTRY[layer]
    group = entry['group']

    ds_path = f'{group}/{polarization}'
    if ds_path not in hf:
        available = [k for k, v in hf[group].items() if isinstance(v, h5py.Dataset) and v.ndim == 2]
        raise KeyError(f"Polarization '{polarization}' not found for layer '{layer}'; available: {available}")

    proj_path = f'{group}/projection'
    if proj_path in hf:
        epsg = int(hf[proj_path][()])
    elif epsg is None:
        raise KeyError(f"No 'projection' dataset at {group}; pass epsg=... to specify the granule's CRS.")

    x = hf[f'{group}/xCoordinates'][:]
    y = hf[f'{group}/yCoordinates'][:]
    x_spacing = _grid_spacing(hf, group, 'x', x)
    y_spacing = _grid_spacing(hf, group, 'y', y)

    bounds = tuple(boundary.to_crs(f'EPSG:{epsg}').total_bounds)
    rows, cols = _gslc_window(x, y, x_spacing, y_spacing, bounds)
    ds = hf[ds_path]
    logger.info(f"  reading {layer} {polarization} window {rows.stop - rows.start}x{cols.stop - cols.start} "
                f"of {ds.shape[0]}x{ds.shape[1]}")
    data = ds[rows, cols]

    if not entry['complex']:
        fill_value = ds.attrs.get('_FillValue', None)
        if fill_value is not None:
            data = np.where(data == fill_value, np.nan, data)
        data = np.where(data == 0, np.nan, data)

    da = xr.DataArray(data, dims=['y', 'x'], coords={'x': x[cols], 'y': y[rows]}, name=layer)
    da = da.rio.write_crs(f'EPSG:{epsg}')
    da.attrs['x_spacing'], da.attrs['y_spacing'] = x_spacing, y_spacing
    return da


def _cell_overlap_mask(da: xr.DataArray, geometry, x_spacing: float, y_spacing: float) -> xr.DataArray:
    """Boolean ``(y, x)`` mask of grid cells whose footprint overlaps ``geometry``.

    Cell footprints are the pixel-centre coordinates +/- half the spacing, so a
    cell counts even if its centre lies outside the geometry. Cells that only
    share an edge or corner with the geometry do not count.
    """
    hx, hy = abs(x_spacing) / 2, abs(y_spacing) / 2
    xx, yy = np.meshgrid(da['x'].values, da['y'].values)
    cells = shapely.box(xx - hx, yy - hy, xx + hx, yy + hy)
    aoi = shapely.union_all(list(geometry))
    mask = shapely.intersects(cells, aoi) & ~shapely.touches(cells, aoi)
    return xr.DataArray(mask, dims=('y', 'x'), coords={'y': da['y'], 'x': da['x']})


def _clip_gslc(da: xr.DataArray, layer: str, boundary: gpd.GeoDataFrame) -> xr.DataArray:
    """Clip a windowed GSLC layer to the AOI according to the layer's ``clip`` mode."""
    geom = boundary.to_crs(da.rio.crs).geometry
    if GSLC_LAYER_REGISTRY[layer]['clip'] == 'cells':
        mask = _cell_overlap_mask(da, geom, da.attrs['x_spacing'], da.attrs['y_spacing'])
        rows, cols = np.flatnonzero(mask.any('x').values), np.flatnonzero(mask.any('y').values)
        if rows.size == 0 or cols.size == 0:
            raise ValueError("No grid cells overlap the AOI.")
        clipped = da.where(mask).isel(y=slice(rows[0], rows[-1] + 1), x=slice(cols[0], cols[-1] + 1))
        return clipped.rio.write_nodata(np.nan)
    return da.rio.write_nodata(0).rio.clip(geom, all_touched=True, drop=True)


def download_nisar_gslc(
    track: int,
    frame: int,
    start_date: str,
    end_date: str,
    aoi: Union[str, Path, gpd.GeoDataFrame],
    layers: List[str] = ('backscatter',),
    output_dir: Union[str, Path] = '.',
    polarization: Union[str, List[str]] = 'HH',
    crid: Optional[str] = None,
    epsg: Optional[int] = None,
) -> Tuple[List[str], List[Path]]:
    """
    Search for NISAR GSLC granules over a track/frame and date range, clip the
    requested layers to an AOI, and write them as GeoTIFFs.

    Granules are streamed over HTTPS and only the pixel window covering the
    AOI is read (the full GSLC backscatter raster is ~39 GB), then clipped.

    Parameters
    ----------
    track, frame : int
        Relative orbit / track and frame numbers to search for.
    start_date, end_date : str
        Search window (e.g. ``'2026-02-01'``).
    aoi : str | Path | geopandas.GeoDataFrame
        Area of interest: a vector file path or a loaded GeoDataFrame.
    layers : list of str, default=['backscatter']
        Valid options are the keys of ``GSLC_LAYER_REGISTRY``:
        ``'backscatter'`` (complex64 ``grids/frequencyA/{pol}``, clipped to the
        AOI polygon at pixel level) and ``'NEBZ'`` (noise-equivalent
        backscatter on its coarse calibration grid, keeping every grid cell
        that overlaps the AOI at all, not only cells whose centre is inside).
    output_dir : str | Path
        Directory for the GeoTIFFs. Created if it doesn't exist.
    polarization : str | list of str, default='HH'
        Polarization(s) to extract, e.g. ``'HH'`` or ``['HH', 'HV']``.
    crid : str, optional
        Processing code to keep, e.g. ``'P05023'`` (filtered client-side).
    epsg : int, optional
        CRS to assume for a layer's group if it has no ``projection`` dataset.

    Returns
    -------
    granule_names : list of str
        Scene names of every granule found.
    downloaded_files : list of Path
        ``{scene}_{layer}_{pol}.tif`` paths, new or already present (existing
        files are skipped). Backscatter pixels outside the AOI polygon are 0
        (complex NaN is not a reliable GeoTIFF nodata); NEBZ cells outside the
        overlap and invalid values are NaN.
    """
    layers = [layers] if isinstance(layers, str) else list(layers)
    unknown = [lyr for lyr in layers if lyr not in GSLC_LAYER_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown GSLC layer(s) {unknown}. Valid options are: {sorted(GSLC_LAYER_REGISTRY)}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pols = [polarization] if isinstance(polarization, str) else list(polarization)
    combos = [(layer, pol) for layer in layers for pol in pols]

    earthaccess.login()

    results = _search_nisar('GSLC', track, frame, start_date, end_date)
    if crid is not None:
        n_found = len(results)
        results = _filter_by_crid(results, crid)
        logger.info(f"Kept {len(results)} of {n_found} granule(s) with processing code {crid}.")

    granule_names = [r.properties['sceneName'] for r in results]
    logger.info(f"Found {len(granule_names)} matching granule(s).")
    if not results:
        return granule_names, []

    boundary = _load_boundary(aoi)
    fs = earthaccess.get_fsspec_https_session()
    downloaded_files: List[Path] = []

    for granule in results:
        scene_name = granule.properties['sceneName']
        out_paths = {c: output_dir / f'{scene_name}_{c[0]}_{c[1]}.tif' for c in combos}

        for out_path in out_paths.values():
            if out_path.exists():
                logger.info(f"  {out_path.name} already exists, skipping")
                downloaded_files.append(out_path)

        missing = [c for c in combos if not out_paths[c].exists()]
        if not missing:
            continue

        logger.info(f"Processing granule: {scene_name}")
        url = granule.properties['url']

        with h5py.File(fs.open(url, cache_type='background', block_size=16 * 1024 * 1024), 'r') as hf:
            for layer, pol in missing:
                da = _read_gslc(hf, layer, pol, boundary, epsg)
                clipped = _clip_gslc(da, layer, boundary)

                out_path = out_paths[(layer, pol)]
                clipped.rio.to_raster(out_path, bigtiff='YES')
                downloaded_files.append(out_path)
                logger.info(f"  wrote {out_path.name} ({clipped.shape[-1]}x{clipped.shape[-2]})")

    return granule_names, downloaded_files


def parse_dates(fname: Union[str, Path]) -> Tuple[str, str]:
    """
    Extract the reference and secondary acquisition dates (``YYYYMMDD``) from
    a NISAR GUNW scene name, or one of the GeoTIFF filenames written by
    ``download_nisar`` (which prefixes the layer name with the scene name).

    Parameters
    ----------
    fname : str | Path
        Scene name or filename containing the four GUNW acquisition
        timestamps (e.g. ``..._20260207T124619_20260207T124654_20260219T124619_20260219T124654_...``).

    Returns
    -------
    (reference_date, secondary_date) : tuple of str
        The reference and secondary acquisition dates as ``YYYYMMDD`` strings.
    """
    fname = str(fname)
    d1 = re.findall(r'_(\d{8})T\d{6}_\d{8}T\d{6}_', fname)[0]
    d2 = re.findall(r'_\d{8}T\d{6}_\d{8}T\d{6}_(\d{8})T\d{6}_\d{8}T\d{6}_', fname)[0]
    return d1, d2


def build_cube(files: List[Union[str, Path]], band_name: str) -> xr.DataArray:
    """
    Stack a list of single-band GeoTIFFs sharing a common grid (e.g. the same
    layer across multiple date pairs, as returned by ``download_nisar``) into
    a single 3D ``(pair, y, x)`` DataArray.

    Files sharing the same reference/secondary date pair (as parsed by
    ``parse_dates``) are deduplicated, keeping only the first (sorted by
    filename).

    Parameters
    ----------
    files : list of str | Path
        Paths to single-band GeoTIFFs, all on the same grid.
    band_name : str
        Name to assign to the returned DataArray.

    Returns
    -------
    xarray.DataArray
        Stacked cube with dims ``('pair', 'y', 'x')``, coordinates ``pair``
        (date-pair strings ``'YYYYMMDD_YYYYMMDD'``) and pixel-center
        ``x``/``y`` coordinates. The source CRS is stored in
        ``da.attrs['crs']``.
    """
    arrays, pairs = [], []
    seen = set()

    for f in sorted(files):
        d1, d2 = parse_dates(f)
        pair_id = f"{d1}_{d2}"
        if pair_id in seen:
            continue
        seen.add(pair_id)

        with rasterio.open(f) as src:
            arrays.append(src.read(1))
            transform = src.transform
            crs = src.crs
            height, width = src.height, src.width
        pairs.append(pair_id)

    stack = np.stack(arrays)
    xs = transform.c + (np.arange(width) + 0.5) * transform.a
    ys = transform.f + (np.arange(height) + 0.5) * transform.e

    da = xr.DataArray(
        stack, dims=('pair', 'y', 'x'),
        coords={'pair': pairs, 'y': ys, 'x': xs},
        name=band_name,
    )
    da.attrs['crs'] = str(crs)
    return da.to_dataset()


def _read_heights(path: Union[str, Path]) -> np.ndarray:
    """Parse the per-band ``height=<m>`` descriptions written by ``download_nisar``."""
    with rasterio.open(path) as src:
        descs = src.descriptions
    try:
        return np.array([float(d.split('=', 1)[1]) for d in descs])
    except (AttributeError, IndexError, ValueError):
        raise ValueError(
            f"{path}: band descriptions must look like 'height=<metres>' "
            f"(as written by download_nisar); got {descs}"
        )


def _interp_radar_grid(
    values: np.ndarray, heights: np.ndarray, ys: np.ndarray, xs: np.ndarray, x_pts: np.ndarray,
    y_pts: np.ndarray, z_pts: np.ndarray, chunk_pixels: int = 2_000_000,
    dtype=np.float32,
) -> np.ndarray:
    """Trilinear interpolation of ``values`` (height, y, x) at 2D point arrays.

    ``ys``/``xs`` must be ascending; ``heights`` ascending. Points outside the
    horizontal extent give NaN; elevations outside the height range are clamped.
    """
    interp = RegularGridInterpolator(
        (ys, xs), np.moveaxis(values, 0, -1), bounds_error=False, fill_value=np.nan
    )
    out = np.full(z_pts.shape, np.nan, dtype=dtype)
    rows_per_chunk = max(1, chunk_pixels // z_pts.shape[1])

    for r0 in range(0, z_pts.shape[0], rows_per_chunk):
        sl = slice(r0, r0 + rows_per_chunk)
        z = z_pts[sl]
        pts = np.stack([y_pts[sl].ravel(), x_pts[sl].ravel()], axis=-1)
        lev = interp(pts)  # (n, n_heights)

        zf = np.clip(z.ravel(), heights[0], heights[-1])
        hi = np.clip(np.searchsorted(heights, zf, side='right'), 1, len(heights) - 1)
        lo = hi - 1
        w = (zf - heights[lo]) / (heights[hi] - heights[lo])
        v_lo = np.take_along_axis(lev, lo[:, None], axis=1)[:, 0]
        v_hi = np.take_along_axis(lev, hi[:, None], axis=1)[:, 0]
        res = v_lo * (1 - w) + v_hi * w
        res[np.isnan(z.ravel())] = np.nan
        out[sl] = res.reshape(z.shape)
    return out


def interpolate_radar_grid_to_dem(
    dem: Union[str, Path, xr.DataArray],
    files: List[Union[str, Path]],
    band_name: str = None,
) -> xr.DataArray:
    """
    Interpolate radar-grid metadata layers (incidence angle or baselines, as
    written by ``download_nisar``) onto the grid of a DEM, using each DEM
    pixel's x, y and elevation.

    Each file holds one band per height above the ellipsoid (taken from the
    ``height=<m>`` band descriptions). Values are interpolated linearly in
    x/y and then linearly in height at the DEM elevation. DEM elevations
    outside the layer's height range are clamped to the nearest level, and
    pixels outside the radar-grid extent are NaN.

    Parameters
    ----------
    dem : str | Path | xarray.DataArray
        2D elevation raster (metres above the ellipsoid) with a CRS. If its
        CRS differs from the files', its pixel locations are transformed to
        the files' CRS for sampling. The output stays on the DEM grid.
    files : list of str | Path
        Multi-band GeoTIFFs on the radar grid, one per date pair, all
        sharing the same grid and heights.
    band_name : str, optional
        Name for the returned array. Defaults to ``'radar_grid_layer'``.

    Returns
    -------
    xarray.DataArray
        ``(y, x)`` for a single file, or ``(pair, y, x)`` for several, where
        ``pair`` is ``'YYYYMMDD_YYYYMMDD'`` (see ``parse_dates``). Shares the
        DEM's coordinates and CRS.
    """
    if not isinstance(dem, xr.DataArray):
        dem = rioxarray.open_rasterio(dem, masked=True)
    dem = dem.squeeze(drop=True)
    if dem.rio.crs is None:
        raise ValueError("dem must have a CRS")
    files = [Path(f) for f in files]
    if not files:
        raise ValueError("files must not be empty")

    xx, yy = np.meshgrid(dem['x'].values, dem['y'].values)
    z = dem.values.astype(np.float64)
    if dem.rio.nodata is not None and not np.isnan(dem.rio.nodata):
        z = np.where(z == dem.rio.nodata, np.nan, z)

    arrays, pairs = [], []
    cached_crs = None
    for f in files:
        with rioxarray.open_rasterio(f, masked=True) as src:
            src = src.load()
        if src.rio.crs != cached_crs:
            cached_crs = src.rio.crs
            if cached_crs == dem.rio.crs:
                x_pts, y_pts = xx, yy
            else:
                tf = Transformer.from_crs(dem.rio.crs, cached_crs, always_xy=True)
                x_pts, y_pts = tf.transform(xx, yy)

        heights = _read_heights(f)
        order = np.argsort(heights)
        src = src.isel(band=order).sortby('y')
        arrays.append(_interp_radar_grid(
            src.values.astype(np.float64), heights[order], src['y'].values, src['x'].values,
            x_pts, y_pts, z, dtype=src.dtype,
        ))
        try:
            pairs.append('_'.join(parse_dates(f)))
        except IndexError:
            pairs.append(f.stem)

    name = band_name or 'radar_grid_layer'
    if len(arrays) == 1:
        da = xr.DataArray(arrays[0], dims=('y', 'x'), coords={'y': dem['y'], 'x': dem['x']}, name=name)
    else:
        da = xr.DataArray(
            np.stack(arrays), dims=('pair', 'y', 'x'),
            coords={'pair': pairs, 'y': dem['y'], 'x': dem['x']}, name=name,
        )
    return da.rio.write_crs(dem.rio.crs).rio.write_nodata(np.nan)


# =============================================================================
# SIGNAL-TO-NOISE
# =============================================================================

_GSLC_FILE_RE = re.compile(r'^(?P<scene>.+)_(?P<layer>backscatter|NEBZ)_(?P<pol>[HV]{2})\.tif$')
_TIMESTAMP_RE = re.compile(r'(\d{8}T\d{6})')


def _open_2d(src: Union[str, Path, xr.DataArray]) -> xr.DataArray:
    """Open a single-band raster (or pass a DataArray through) as a 2D ``(y, x)`` array.

    Real-valued nodata is converted to NaN; complex rasters are left untouched.
    """
    if isinstance(src, xr.DataArray):
        da = src
    else:
        da = rioxarray.open_rasterio(src)
    if 'band' in da.dims:
        da = da.squeeze('band', drop=True)
    nodata = da.rio.nodata
    if (not np.issubdtype(da.dtype, np.complexfloating) and nodata is not None
            and not np.isnan(nodata)):
        da = da.where(da != nodata)
    return da


def calculate_snr(
    backscatter: Union[str, Path, xr.DataArray],
    nebz: Union[str, Path, xr.DataArray],
    db: bool = False,
) -> xr.DataArray:
    """
    Signal-to-noise ratio: backscatter power divided by the noise-equivalent
    backscatter (NEBZ).

    Both inputs must already be on exactly the same grid (same shape, ``x``/``y``
    coordinates and CRS); reproject NEBZ onto the backscatter grid first.

    Parameters
    ----------
    backscatter : str | Path | xarray.DataArray
        Backscatter GeoTIFF or array. Complex values (as in GSLC) are converted
        to power, ``|z|**2``; real values are assumed to already be linear power.
    nebz : str | Path | xarray.DataArray
        NEBZ GeoTIFF or array, in the same linear units.
    db : bool, default=False
        Return ``10 * log10(snr)`` instead of the linear ratio.

    Returns
    -------
    xarray.DataArray
        float32 ``(y, x)`` array named ``'snr'`` on the backscatter grid. Pixels
        where backscatter is 0/NaN or NEBZ is NaN or <= 0 are NaN.

    Raises
    ------
    ValueError
        If the two inputs differ in shape, coordinates, or CRS.
    """
    bs = _open_2d(backscatter)
    nz = _open_2d(nebz)

    if bs.dims != nz.dims or bs.shape != nz.shape:
        raise ValueError(
            f"backscatter and NEBZ shapes differ: {bs.dims}{bs.shape} vs {nz.dims}{nz.shape}. "
            "Reproject NEBZ onto the backscatter grid first."
        )
    for dim in ('y', 'x'):
        if not np.array_equal(bs[dim].values, nz[dim].values):
            raise ValueError(
                f"backscatter and NEBZ '{dim}' coordinates differ. "
                "Reproject NEBZ onto the backscatter grid first."
            )
    if bs.rio.crs != nz.rio.crs:
        raise ValueError(f"backscatter and NEBZ CRS differ: {bs.rio.crs} vs {nz.rio.crs}.")

    power = np.abs(bs) ** 2 if np.issubdtype(bs.dtype, np.complexfloating) else bs
    power = power.where(power != 0)
    noise = nz.where(nz > 0)

    snr = (power.astype(np.float64) / noise.astype(np.float64))
    if db:
        snr = 10 * np.log10(snr)
    snr = snr.astype(np.float32)
    snr.name = 'snr'
    snr.attrs = {'units': 'dB' if db else 'linear'}
    return snr.rio.write_crs(bs.rio.crs) if bs.rio.crs is not None else snr


def parse_acquisition_time(fname: Union[str, Path]) -> datetime:
    """
    Acquisition start time from a NISAR scene name or a filename that begins
    with one (the first ``YYYYMMDDTHHMMSS`` token; GSLC names carry start and
    stop times).
    """
    m = _TIMESTAMP_RE.search(Path(str(fname)).name)
    if m is None:
        raise ValueError(f"No YYYYMMDDTHHMMSS timestamp found in '{fname}'.")
    return datetime.strptime(m.group(1), '%Y%m%dT%H%M%S')


def _match_gslc_files(
    backscatter_files: List[Union[str, Path]],
    nebz_files: List[Union[str, Path]],
    polarization: str = 'HH',
) -> dict:
    """
    Pair ``*_backscatter_{pol}.tif`` with ``*_NEBZ_{pol}.tif`` files by scene
    name (which embeds track, frame, times and processing code) and polarization.

    Only files of ``polarization`` are considered. Returns a dict with
    ``matched`` (list of ``(scene, backscatter_path, nebz_path)``),
    ``unmatched_backscatter``, ``unmatched_nebz`` and ``unparsed`` (names that
    don't follow the expected pattern or are in the wrong list).
    """
    def index(files, expected_layer):
        found, unparsed = {}, []
        for f in files:
            m = _GSLC_FILE_RE.match(Path(f).name)
            if m is None or m['layer'] != expected_layer:
                unparsed.append(Path(f))
            elif m['pol'] == polarization:
                found[m['scene']] = Path(f)
        return found, unparsed

    bs, bad_bs = index(backscatter_files, 'backscatter')
    nz, bad_nz = index(nebz_files, 'NEBZ')

    return {
        'matched': [(scene, bs[scene], nz[scene]) for scene in sorted(bs) if scene in nz],
        'unmatched_backscatter': [bs[k] for k in sorted(bs) if k not in nz],
        'unmatched_nebz': [nz[k] for k in sorted(nz) if k not in bs],
        'unparsed': bad_bs + bad_nz,
    }


def build_snr_timeseries(
    backscatter_files: List[Union[str, Path]],
    nebz_files: List[Union[str, Path]],
    polarization: str = 'HH',
    resampling: str = 'bilinear',
    db: bool = False,
) -> xr.DataArray:
    """
    Compute SNR for every matching backscatter/NEBZ file pair and stack the
    results along an overpass-time dimension.

    Files are paired by scene name and polarization (see ``_match_gslc_files``).
    Each NEBZ raster is reprojected onto its backscatter grid with
    ``reproject_match`` and passed to ``calculate_snr``.

    Parameters
    ----------
    backscatter_files, nebz_files : list of str | Path
        Files written by ``download_nisar_gslc``
        (``{scene}_backscatter_{pol}.tif`` / ``{scene}_NEBZ_{pol}.tif``).
    polarization : str, default='HH'
        Only files of this polarization are used.
    resampling : str, default='bilinear'
        ``rasterio.enums.Resampling`` name used to reproject NEBZ
        (``'nearest'``, ``'bilinear'``, ``'cubic'``, ...).
    db : bool, default=False
        Return SNR in dB.

    Returns
    -------
    xarray.DataArray
        ``(date, y, x)`` array named ``'snr'``. ``date`` holds the acquisition
        start time (xarray stores Python datetimes as ``datetime64``), sorted
        ascending, with a ``scene`` coordinate alongside.

    Warns
    -----
    UserWarning
        Listing every backscatter/NEBZ file without a partner (and unparseable
        names), and any duplicate acquisition times.

    Raises
    ------
    ValueError
        If nothing matches, or the matched backscatter files are not all on
        the same grid.
    """
    m = _match_gslc_files(backscatter_files, nebz_files, polarization)

    problems = []
    if m['unmatched_backscatter']:
        problems.append("backscatter without NEBZ: " + ", ".join(p.name for p in m['unmatched_backscatter']))
    if m['unmatched_nebz']:
        problems.append("NEBZ without backscatter: " + ", ".join(p.name for p in m['unmatched_nebz']))
    if m['unparsed']:
        problems.append("unrecognized/misplaced filenames: " + ", ".join(p.name for p in m['unparsed']))
    if problems:
        msg = "Unmatched files (" + polarization + "): " + "; ".join(problems)
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    if not m['matched']:
        raise ValueError(f"No matching backscatter/NEBZ file pairs found for polarization '{polarization}'.")

    items = sorted(((parse_acquisition_time(scene), scene, bs, nz) for scene, bs, nz in m['matched']),
                   key=lambda t: (t[0], t[1]))
    times = [t[0] for t in items]
    if len(set(times)) != len(times):
        dup = sorted({str(t) for t in times if times.count(t) > 1})
        warnings.warn(f"Duplicate acquisition times: {dup}", UserWarning, stacklevel=2)

    layers, ref = [], None
    for when, scene, bs_path, nz_path in items:
        bs = _open_2d(bs_path)
        if ref is None:
            ref = bs
        elif (bs.shape != ref.shape or bs.rio.crs != ref.rio.crs
              or not np.array_equal(bs['x'].values, ref['x'].values)
              or not np.array_equal(bs['y'].values, ref['y'].values)):
            raise ValueError(f"{bs_path.name} is not on the same grid as the other backscatter files.")
        nz = _open_2d(nz_path).rio.reproject_match(bs, resampling=Resampling[resampling])
        layers.append(calculate_snr(bs, nz, db=db).drop_vars('spatial_ref', errors='ignore'))

    out = xr.concat(layers, dim=xr.DataArray(np.array(times, dtype='datetime64[ns]'), dims='date', name='date'))
    out = out.assign_coords(scene=('date', [t[1] for t in items]))
    out.name = 'snr'
    out.attrs = {'units': 'dB' if db else 'linear'}
    return out.rio.write_crs(ref.rio.crs)

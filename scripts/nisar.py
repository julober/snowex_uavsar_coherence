import logging
import re
from pathlib import Path
from typing import List, Tuple, Union

import asf_search as asf
import earthaccess
import geopandas as gpd
import h5py
import numpy as np
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor on xr.DataArray)
import xarray as xr

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
        'path': None,
        'coords': 'metadata',
    },
}
"""Maps a friendly layer name to its relative HDF5 path template (``{pol}`` is
filled in with the requested polarization) and the coordinate group used to
georeference it. ``incidence_angle`` is special-cased since it lives under
the metadata radar grid rather than a polarized interferogram group."""

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


def _read_incidence_angle(hf: h5py.File, x: np.ndarray, y: np.ndarray, epsg: int) -> xr.DataArray:
    """Read the incidence angle at the height level nearest the ellipsoid."""
    heights = hf[f'{BASE_META}/radarGrid/heightAboveEllipsoid'][:]
    height_idx = int(np.argmin(np.abs(heights)))
    data = hf[f'{BASE_META}/radarGrid/incidenceAngle'][height_idx].astype(np.float32)
    data[data == 0] = np.nan

    da = xr.DataArray(data, dims=['y', 'x'], coords={'x': x, 'y': y}, name='incidence_angle')
    return da.rio.write_crs(f'EPSG:{epsg}').rio.write_nodata(np.nan)


def _read_layer(hf: h5py.File, layer: str, polarization: str, epsg: int) -> xr.DataArray:
    """Read and mask a single requested layer, returning a georeferenced DataArray."""
    if layer not in LAYER_REGISTRY:
        raise ValueError(
            f"Unknown layer '{layer}'. Valid options are: {sorted(LAYER_REGISTRY)}"
        )

    entry = LAYER_REGISTRY[layer]
    x, y = _read_coords(hf, entry['coords'], polarization)

    if layer == 'incidence_angle':
        return _read_incidence_angle(hf, x, y, epsg)

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


def download_nisar(
    track: int,
    frame: int,
    start_date: str,
    end_date: str,
    aoi: Union[str, Path, gpd.GeoDataFrame],
    layers: List[str],
    output_dir: Union[str, Path],
    polarization: str = 'HH',
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
        ``'coherence_wrapped'``, ``'incidence_angle'``.
    output_dir : str | Path
        Directory to write clipped GeoTIFFs into. Created if it doesn't exist.
    polarization : str, default='HH'
        Polarization channel to read the interferogram layers from.

    Returns
    -------
    granule_names : list of str
        Scene names of every granule found matching the search criteria.
    downloaded_files : list of Path
        Paths to every clipped GeoTIFF written to ``output_dir``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    earthaccess.login()

    logger.info(f"Searching ASF for NISAR GUNW granules: track={track}, frame={frame}, {start_date}–{end_date}")
    results = asf.search(
        dataset='NISAR',
        processingLevel='GUNW',
        relativeOrbit=track,
        frame=frame,
        start=start_date,
        end=end_date,
    )

    granule_names = [r.properties['sceneName'] for r in results]
    logger.info(f"Found {len(granule_names)} matching granule(s).")

    if not results:
        return granule_names, []

    if isinstance(aoi, gpd.GeoDataFrame):
        boundary = aoi
    else:
        boundary = gpd.read_file(aoi)

    fs = earthaccess.get_fsspec_https_session()
    downloaded_files: List[Path] = []

    for granule in results:
        scene_name = granule.properties['sceneName']
        url = granule.properties['url']
        logger.info(f"Processing granule: {scene_name}")

        with h5py.File(fs.open(url, cache_type='background', block_size=16 * 1024 * 1024), 'r') as hf:
            epsg = _get_epsg(hf, polarization)
            clip_geom = boundary.to_crs(f'EPSG:{epsg}').geometry

            for layer in layers:
                da = _read_layer(hf, layer, polarization, epsg)
                clipped = da.rio.clip(clip_geom, all_touched=True, drop=True)

                out_path = output_dir / f'{scene_name}_{layer}.tif'
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
    return da

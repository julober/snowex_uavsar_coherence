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
from pyproj import Transformer
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
}
"""Maps a friendly layer name to its relative HDF5 path template (``{pol}`` is
filled in with the requested polarization) and the coordinate group used to
georeference it. Layers with coords ``'metadata'`` (``incidence_angle``,
``parallel_baseline``, ``perpendicular_baseline``) live under the metadata
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
    data = hf[f'{BASE_META}/radarGrid/{dataset}'][:].astype(np.float32)
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

def download_nisar(
    track: int,
    frame: int,
    start_date: str,
    end_date: str,
    aoi: Union[str, Path, gpd.GeoDataFrame],
    layers: List[str],
    output_dir: Union[str, Path] = '.',
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
        ``'coherence_wrapped'``, ``'incidence_angle'``,
        ``'parallel_baseline'``, ``'perpendicular_baseline'``. The last three
        are radar-grid metadata layers written as multi-band GeoTIFFs with one
        band per height level above the ellipsoid (band descriptions hold the
        height in metres).
    output_dir : str | Path
        Directory to write clipped GeoTIFFs into. Created if it doesn't exist.
    polarization : str, default='HH'
        Polarization channel to read the interferogram layers from.

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
) -> np.ndarray:
    """Trilinear interpolation of ``values`` (height, y, x) at 2D point arrays.

    ``ys``/``xs`` must be ascending; ``heights`` ascending. Points outside the
    horizontal extent give NaN; elevations outside the height range are clamped.
    """
    interp = RegularGridInterpolator(
        (ys, xs), np.moveaxis(values, 0, -1), bounds_error=False, fill_value=np.nan
    )
    out = np.full(z_pts.shape, np.nan, dtype=np.float32)
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
            x_pts, y_pts, z,
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

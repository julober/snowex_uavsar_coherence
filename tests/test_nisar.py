"""Tests for the radar-grid metadata layers in ``scripts/nisar.py``."""

import h5py
import numpy as np
import pytest

import nisar  # noqa: E402  (conftest puts scripts/ on sys.path)

N_H, NY, NX = 4, 6, 7
LAYERS = {
    'incidence_angle': 'incidenceAngle',
    'parallel_baseline': 'parallelBaseline',
    'perpendicular_baseline': 'perpendicularBaseline',
    'reference_slant_range': 'referenceSlantRange',
    'secondary_slant_range': 'secondarySlantRange',
}
FLOAT64 = {'reference_slant_range', 'secondary_slant_range'}


@pytest.fixture
def hf():
    f = h5py.File('meta.h5', 'w', driver='core', backing_store=False)
    grp = f.create_group(f'{nisar.BASE_META}/radarGrid')
    grp['heightAboveEllipsoid'] = np.array([-500.0, 0.0, 500.0, 1000.0])
    grp['xCoordinates'] = 500000.0 + 100.0 * np.arange(NX)
    grp['yCoordinates'] = 4000000.0 - 100.0 * np.arange(NY)
    for layer, name in LAYERS.items():
        dt = np.float64 if layer in FLOAT64 else np.float32
        grp[name] = np.random.rand(N_H, NY, NX).astype(dt) + 1
    yield f
    f.close()


@pytest.mark.parametrize('layer', list(LAYERS))
def test_metadata_layers_keep_all_heights(hf, layer):
    da = nisar._read_layer(hf, layer, 'HH', 32611)
    assert da.dims == ('band', 'y', 'x')
    assert da.shape == (N_H, NY, NX)
    assert list(da['band'].values) == [-500.0, 0.0, 500.0, 1000.0]
    assert da.rio.crs.to_epsg() == 32611
    assert da.dtype == (np.float64 if layer in FLOAT64 else np.float32)


# ---------------------------------------------------------------------------
# interpolate_radar_grid_to_dem
# ---------------------------------------------------------------------------

import rasterio  # noqa: E402
import xarray as xr  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

A, B, C = 0.001, -0.002, 0.01
HEIGHTS = [0.0, 500.0, 1000.0]
PAIR = 'NISAR_L2_PR_GUNW_001_001_A_001_4000_SHNA_A_20260207T124619_20260207T124654_20260219T124619_20260219T124654_X_0_0_layer.tif'


def _radar_tif(path, crs='EPSG:32611', dtype='float32'):
    xs = 500000.0 + 1000.0 * np.arange(6)
    ys = 4001000.0 - 1000.0 * np.arange(6)
    data = np.stack([A * xs[None, :] + B * ys[:, None] + C * h * np.ones((6, 6)) for h in HEIGHTS]).astype(dtype)
    with rasterio.open(path, 'w', driver='GTiff', height=6, width=6, count=3, dtype=dtype,
                       crs=crs, transform=from_origin(xs[0] - 500, ys[0] + 500, 1000, 1000)) as dst:
        dst.write(data)
        for i, h in enumerate(HEIGHTS, 1):
            dst.set_band_description(i, f'height={h:g}')


def _dem(z):
    xs = 501500.0 + 500.0 * np.arange(4)
    ys = 4000500.0 - 500.0 * np.arange(3)
    da = xr.DataArray(z, dims=('y', 'x'), coords={'y': ys, 'x': xs})
    return da.rio.write_crs('EPSG:32611')


def test_interpolation_matches_linear_field(tmp_path):
    f = tmp_path / PAIR
    _radar_tif(f)
    z = np.array([[0, 250, 700, 1000], [100, 300, 900, 50], [0, 0, 0, 0]], dtype=float)
    dem = _dem(z)
    out = nisar.interpolate_radar_grid_to_dem(dem, [f])
    X, Y = np.meshgrid(dem.x.values, dem.y.values)
    assert out.dims == ('y', 'x')
    np.testing.assert_allclose(out.values, A * X + B * Y + C * z, rtol=1e-5)


def test_clamping_nan_and_pairs(tmp_path):
    f1, f2 = tmp_path / PAIR, tmp_path / PAIR.replace('20260219', '20260303')
    _radar_tif(f1)
    _radar_tif(f2)
    z = np.full((3, 4), 5000.0)
    z[0, 0] = np.nan
    out = nisar.interpolate_radar_grid_to_dem(_dem(z), [f1, f2])
    assert out.dims == ('pair', 'y', 'x')
    assert list(out.pair.values) == ['20260207_20260219', '20260207_20260303']
    assert np.isnan(out.values[:, 0, 0]).all()
    X, Y = np.meshgrid(out.x.values, out.y.values)
    np.testing.assert_allclose(out.values[0, 1, 1], A * X[1, 1] + B * Y[1, 1] + C * 1000, rtol=1e-5)


def test_outside_extent_is_nan_and_crs_reprojected(tmp_path):
    f = tmp_path / PAIR
    _radar_tif(f)
    dem = _dem(np.zeros((3, 4)))
    far = dem.assign_coords(x=dem.x + 100000.0)
    assert np.isnan(nisar.interpolate_radar_grid_to_dem(far, [f]).values).all()

    from pyproj import Transformer
    lon, lat = Transformer.from_crs('EPSG:32611', 'EPSG:4326', always_xy=True).transform(502000.0, 4000000.0)
    dem_ll = xr.DataArray(np.zeros((1, 1)), dims=('y', 'x'), coords={'y': [lat], 'x': [lon]}).rio.write_crs('EPSG:4326')
    out = nisar.interpolate_radar_grid_to_dem(dem_ll, [f])
    np.testing.assert_allclose(out.values[0, 0], A * 502000.0 + B * 4000000.0, rtol=1e-4)


def test_missing_height_descriptions(tmp_path):
    f = tmp_path / PAIR
    _radar_tif(f)
    with rasterio.open(f, 'r+') as dst:
        dst.set_band_description(1, 'nope')
    with pytest.raises(ValueError, match='height='):
        nisar.interpolate_radar_grid_to_dem(_dem(np.zeros((3, 4))), [f])


def test_float64_input_gives_float64_output(tmp_path):
    f = tmp_path / PAIR
    _radar_tif(f, dtype='float64')
    assert nisar.interpolate_radar_grid_to_dem(_dem(np.zeros((3, 4))), [f]).dtype == np.float64


# ---------------------------------------------------------------------------
# CRID filtering
# ---------------------------------------------------------------------------

class _Result:
    def __init__(self, name):
        self.properties = {'sceneName': name}


_BASE = 'NISAR_L2_PR_GUNW_001_001_A_001_4000_SHNA_A_20260207T124619_20260207T124654_20260219T124619_20260219T124654_{}_N_F_J_001'
RESULTS = [_Result(_BASE.format('X05010')), _Result(_BASE.format('P05023'))]


def test_filter_by_crid():
    kept = nisar._filter_by_crid(RESULTS, 'P05023')
    assert [r.properties['sceneName'] for r in kept] == [RESULTS[1].properties['sceneName']]
    assert nisar._filter_by_crid(RESULTS, 'P0502') == []  # token match, not substring
    assert nisar._filter_by_crid(RESULTS, None) == RESULTS


def test_warn_mixed_crids(caplog):
    with caplog.at_level('WARNING'):
        nisar._warn_mixed_crids(RESULTS)
    assert 'P05023' in caplog.text and 'X05010' in caplog.text
    caplog.clear()
    with caplog.at_level('WARNING'):
        nisar._warn_mixed_crids(RESULTS[:1])
    assert caplog.text == ''


# ---------------------------------------------------------------------------
# GSLC windowed read
# ---------------------------------------------------------------------------

import geopandas as gpd  # noqa: E402
from shapely.geometry import box  # noqa: E402

G_NY, G_NX, G_DX = 40, 50, 10.0
G_X = 500000.0 + G_DX * np.arange(G_NX) + G_DX / 2
G_Y = 4001000.0 - G_DX * np.arange(G_NY) - G_DX / 2


class _SpyDataset:
    """Wraps an h5py dataset and records the indices it is sliced with."""

    def __init__(self, ds):
        self._ds, self.keys = ds, []
        self.shape, self.dtype = ds.shape, ds.dtype

    def __getitem__(self, key):
        self.keys.append(key)
        return self._ds[key]


def _gslc_file(with_projection=True, pols=('HH',)):
    f = h5py.File('gslc.h5', 'w', driver='core', backing_store=False)
    g = f.create_group(nisar.GSLC_BASE_GRIDS)
    g['xCoordinates'], g['yCoordinates'] = G_X, G_Y
    g['xCoordinateSpacing'], g['yCoordinateSpacing'] = G_DX, -G_DX
    if with_projection:
        g['projection'] = np.uint32(32611)
    for p in pols:
        g[p] = (np.arange(G_NY * G_NX).reshape(G_NY, G_NX) * (1 + 1j)).astype(np.complex64)
    return f


def _aoi(xmin, ymin, xmax, ymax):
    return gpd.GeoDataFrame(geometry=[box(xmin, ymin, xmax, ymax)], crs='EPSG:32611')


def test_gslc_window_interior_and_edge():
    rows, cols = nisar._gslc_window(G_X, G_Y, G_DX, -G_DX, (500100, 4000600, 500200, 4000800))
    assert (cols.start, cols.stop) == (10, 20)
    assert (rows.start, rows.stop) == (20, 40)
    rows, cols = nisar._gslc_window(G_X, G_Y, G_DX, -G_DX, (499000, 4000900, 500050, 4009999))
    assert cols.start == 0 and rows.start == 0 and cols.stop == 5 and rows.stop == 10


def test_gslc_window_no_overlap():
    with pytest.raises(ValueError, match='does not intersect'):
        nisar._gslc_window(G_X, G_Y, G_DX, -G_DX, (0, 0, 10, 10))


def test_read_gslc_reads_only_window():
    f = _gslc_file()
    spy = _SpyDataset(f[f'{nisar.GSLC_BASE_GRIDS}/HH'])

    class _Proxy:  # route the HH dataset through the spy
        def __contains__(self, k): return k in f
        def __getitem__(self, k): return spy if k.endswith('/HH') else f[k]

    da = nisar._read_gslc(_Proxy(), 'backscatter', 'HH', _aoi(500100, 4000600, 500200, 4000800))
    assert da.shape == (20, 10) and da.dtype == np.complex64
    assert len(spy.keys) == 1 and spy.keys[0] == (slice(20, 40), slice(10, 20))
    assert da.rio.crs.to_epsg() == 32611
    np.testing.assert_array_equal(da['x'].values, G_X[10:20])
    f.close()


def test_read_gslc_errors():
    f = _gslc_file()
    with pytest.raises(KeyError, match='HV'):
        nisar._read_gslc(f, 'backscatter', 'HV', _aoi(500100, 4000600, 500200, 4000800))
    f.close()

    f = _gslc_file(with_projection=False)
    aoi = _aoi(500100, 4000600, 500200, 4000800)
    with pytest.raises(KeyError, match='epsg'):
        nisar._read_gslc(f, 'backscatter', 'HH', aoi)
    assert nisar._read_gslc(f, 'backscatter', 'HH', aoi, epsg=32611).rio.crs.to_epsg() == 32611
    f.close()


def test_gslc_clip_and_complex_geotiff_roundtrip(tmp_path):
    f = _gslc_file()
    aoi = _aoi(500100, 4000600, 500200, 4000800)
    da = nisar._read_gslc(f, 'backscatter', 'HH', aoi).rio.write_nodata(0)
    clipped = da.rio.clip(aoi.to_crs(da.rio.crs).geometry, all_touched=True, drop=True)
    out = tmp_path / 'x_HH.tif'
    clipped.rio.to_raster(out, bigtiff='YES')
    with rasterio.open(out) as src:
        assert src.dtypes[0] == 'complex64'
        np.testing.assert_array_equal(src.read(1), clipped.values)
    f.close()


# ---------------------------------------------------------------------------
# NEBZ layer (own coarse grid, clipped by cell overlap)
# ---------------------------------------------------------------------------

N_NX, N_NY, N_DX = 5, 4, 100.0  # 100 m cells, centres at 500050, 500150, ...
N_X = 500000.0 + N_DX * np.arange(N_NX) + N_DX / 2
N_Y = 4001000.0 - 1000.0 + 400.0 - N_DX * np.arange(N_NY) - N_DX / 2  # 4000350 .. 4000050
_NEB = nisar.GSLC_LAYER_REGISTRY['NEBZ']['group']


def _neb_file(spacing=False):
    f = _gslc_file()
    g = f.create_group(_NEB)
    g['xCoordinates'], g['yCoordinates'] = N_X, N_Y
    g['projection'] = np.uint32(32611)
    if spacing:
        g['xCoordinateSpacing'], g['yCoordinateSpacing'] = N_DX, -N_DX
    data = (1.0 + np.arange(N_NY * N_NX).reshape(N_NY, N_NX)).astype(np.float64)
    data[0, 0] = 0  # invalid
    g['HH'] = data
    return f


def test_nebz_read_own_grid_dtype_and_spacing_fallback():
    f = _neb_file()
    da = nisar._read_gslc(f, 'NEBZ', 'HH', _aoi(500000, 4000000, 500500, 4000400))
    assert da.shape == (N_NY, N_NX) and da.dtype == np.float64
    assert da.rio.crs.to_epsg() == 32611
    assert da.attrs['x_spacing'] == N_DX and da.attrs['y_spacing'] == -N_DX
    assert np.isnan(da.values[0, 0])
    f.close()


def test_unknown_layer_raises():
    f = _neb_file()
    with pytest.raises(ValueError, match=r"\['NEBZ', 'backscatter'\]"):
        nisar._read_gslc(f, 'nope', 'HH', _aoi(500000, 4000000, 500500, 4000400))
    f.close()


def _neb_clip(aoi):
    f = _neb_file(spacing=True)
    da = nisar._read_gslc(f, 'NEBZ', 'HH', aoi)
    out = nisar._clip_gslc(da, 'NEBZ', aoi)
    f.close()
    return out


def test_nebz_keeps_cell_with_only_corner_overlap():
    # covers only the lower-left corner of the cell centred at (500250, 4000150); no centre inside
    out = _neb_clip(_aoi(500205, 4000105, 500215, 4000115))
    assert out.shape == (1, 1)
    assert out['x'].values[0] == 500250 and out['y'].values[0] == 4000150


def test_nebz_keeps_only_overlapping_cells_and_excludes_edge_touch():
    # rectangle over cells in columns 1-2 / rows 1-2; its edges lie exactly on cell boundaries
    out = _neb_clip(_aoi(500100, 4000100, 500300, 4000300))
    assert out.shape == (2, 2)
    assert not np.isnan(out.values).any()


def test_nebz_diagonal_aoi_leaves_nan_cells():
    from shapely.geometry import Polygon
    tri = gpd.GeoDataFrame(geometry=[Polygon([(500010, 4000010), (500490, 4000010), (500490, 4000390)])], crs='EPSG:32611')
    out = _neb_clip(tri)
    assert out.shape == (4, 5)
    assert np.isnan(out.values[0, 0]) and not np.isnan(out.values[3, 0])

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

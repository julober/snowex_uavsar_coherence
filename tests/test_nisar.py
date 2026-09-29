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
}


@pytest.fixture
def hf():
    f = h5py.File('meta.h5', 'w', driver='core', backing_store=False)
    grp = f.create_group(f'{nisar.BASE_META}/radarGrid')
    grp['heightAboveEllipsoid'] = np.array([-500.0, 0.0, 500.0, 1000.0])
    grp['xCoordinates'] = 500000.0 + 100.0 * np.arange(NX)
    grp['yCoordinates'] = 4000000.0 - 100.0 * np.arange(NY)
    for name in LAYERS.values():
        grp[name] = np.random.rand(N_H, NY, NX).astype(np.float32) + 1
    yield f
    f.close()


@pytest.mark.parametrize('layer', list(LAYERS))
def test_metadata_layers_keep_all_heights(hf, layer):
    da = nisar._read_layer(hf, layer, 'HH', 32611)
    assert da.dims == ('band', 'y', 'x')
    assert da.shape == (N_H, NY, NX)
    assert list(da['band'].values) == [-500.0, 0.0, 500.0, 1000.0]
    assert da.rio.crs.to_epsg() == 32611

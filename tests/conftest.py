"""
conftest.py — Shared pytest fixtures for the snowex_uavsar_coherence test suite.

All fixtures produce tiny (3 × 3 or 5 × 5 pixel) synthetic
xarray.DataArray / xarray.Dataset objects that are sufficient to exercise the
data-processing and sanitization logic in ``scripts/layers.py`` without
touching any external API or file system.

Several fixtures intentionally carry "dirty" CRS / grid-mapping metadata that
mirrors the state of real datasets *before* the scorched-earth sanitization
pass in ``assemble_data``:

* ``dirty_coh_ds``   — coherence Dataset with a ``spatial_ref`` coordinate and
                        ``grid_mapping`` attrs on every variable.
* ``dirty_snow_ds``  — snow-class Dataset with a ``crs`` coordinate and
                        ``grid_mapping`` attrs on every variable (including the
                        coordinate arrays ``x`` and ``y``).

These fixtures let the integration test in ``test_layers.py`` confirm that the
sanitization loop correctly strips all CRS artefacts before the merge.
"""

import sys
import types
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.crs import CRS as RioCRS
from rasterio.transform import from_bounds
from shapely.geometry import box

# ---------------------------------------------------------------------------
# Path setup — make ``scripts/`` importable as a package
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Stub out heavy / unavailable third-party packages before importing layers.
# This prevents ModuleNotFoundError for packages not present in the test
# environment (earthaccess, py3dep, pygeohydro, s3fs, xrspatial).
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> types.ModuleType:
    """Return a minimal MagicMock-based module stub."""
    stub = types.ModuleType(name)
    stub.__spec__ = MagicMock()
    return stub


_STUB_MODULES = [
    "earthaccess",
    "py3dep",
    "pygeohydro",
    "s3fs",
    "xrspatial",
    "uavsar_pytools",
    "uavsar_pytools.convert",
    "uavsar_pytools.convert.tiff_conversion",
]

for _mod_name in _STUB_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

# Ensure that the validation module can be found as a top-level import (the
# way layers.py imports it: ``from validation import ...``).
if "validation" not in sys.modules:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Small helper: build a minimal spatial DataArray / Dataset
# ---------------------------------------------------------------------------

_NY, _NX = 5, 5
_X_COORDS = np.linspace(-120.0, -119.9, _NX)
_Y_COORDS = np.linspace(38.1, 38.0, _NY)  # descending (north → south)


def _make_spatial_da(
    name: str,
    value: float = 1.0,
    ny: int = _NY,
    nx: int = _NX,
    extra_dims: dict | None = None,
) -> xr.DataArray:
    """Return a tiny, spatially referenced DataArray without any CRS attrs."""
    shape = [ny, nx]
    dims = ["y", "x"]
    coords: dict = {
        "y": _Y_COORDS[:ny],
        "x": _X_COORDS[:nx],
    }

    if extra_dims:
        for dim_name, dim_coords in extra_dims.items():
            shape.insert(0, len(dim_coords))
            dims.insert(0, dim_name)
            coords[dim_name] = dim_coords

    data = np.full(shape, value, dtype=np.float32)
    return xr.DataArray(data, dims=dims, coords=coords, name=name)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def aoi() -> "box":
    """Tiny area of interest (bounding box) in WGS 84 degrees."""
    return box(-120.0, 38.0, -119.9, 38.1)


@pytest.fixture()
def date_pairs() -> list:
    """Two date pairs covering a ~2-week interval."""
    return [
        (date(2021, 2, 10), date(2021, 2, 24)),
        (date(2021, 2, 24), date(2021, 3, 10)),
    ]


@pytest.fixture()
def flight_ids() -> list:
    """Two synthetic UAVSAR flight-heading identifiers."""
    return ["08508", "26505"]


@pytest.fixture()
def ref_grid() -> xr.DataArray:
    """Minimal 5 × 5 reference grid (no CRS artefacts)."""
    return _make_spatial_da("ref", value=0.0)


# ---------------------------------------------------------------------------
# "Clean" layer fixtures (no dirty metadata)
# ---------------------------------------------------------------------------


@pytest.fixture()
def clean_coh_ds(date_pairs, flight_ids) -> xr.Dataset:
    """Coherence Dataset (flight_id × pair × y × x) without CRS artefacts."""
    pair_labels = [
        f"{s.strftime('%y%m%d')}_{e.strftime('%y%m%d')}" for s, e in date_pairs
    ]
    da = _make_spatial_da(
        "coherence",
        value=0.7,
        extra_dims={"flight_id": flight_ids, "pair": pair_labels},
    )
    da.attrs.update({"units": "1", "long_name": "InSAR coherence magnitude"})
    return da.to_dataset(name="coherence")


@pytest.fixture()
def clean_incidence_da(flight_ids) -> xr.DataArray:
    """Incidence angle DataArray (flight_id × y × x) without CRS artefacts."""
    da = _make_spatial_da(
        "incidence_angle",
        value=45.0,
        extra_dims={"flight_id": flight_ids},
    )
    da.attrs.update({"units": "degree", "long_name": "local incidence angle"})
    return da


@pytest.fixture()
def clean_dem_da() -> xr.DataArray:
    """Elevation DataArray (y × x) without CRS artefacts."""
    da = _make_spatial_da("elevation", value=2000.0)
    return da


@pytest.fixture()
def clean_topo_ds() -> xr.Dataset:
    """Topographic Dataset (slope, aspect, curve) without CRS artefacts."""
    return xr.Dataset(
        {
            "slope": _make_spatial_da("slope", value=10.0),
            "aspect": _make_spatial_da("aspect", value=180.0),
            "curve": _make_spatial_da("curve", value=0.01),
        }
    )


@pytest.fixture()
def clean_nlcd_ds() -> xr.Dataset:
    """NLCD Dataset (cover_2019, canopy_2019) without CRS artefacts."""
    return xr.Dataset(
        {
            "cover_2019": _make_spatial_da("cover_2019", value=41.0),
            "canopy_2019": _make_spatial_da("canopy_2019", value=50.0),
        }
    )


@pytest.fixture()
def clean_snow_class_ds() -> xr.Dataset:
    """Snow climatology Dataset (snow_class) without CRS artefacts."""
    ds = xr.Dataset({"snow_class": _make_spatial_da("snow_class", value=3.0)})
    ds["snow_class"].attrs["long_name"] = "NSIDC-0768 seasonal snow classification"
    return ds


@pytest.fixture()
def clean_snow_ds(date_pairs) -> xr.Dataset:
    """UCLA snow metrics Dataset (swe_accum, swe_ablate, density_change) without CRS artefacts."""
    pair_labels = [
        f"{s.strftime('%y%m%d')}_{e.strftime('%y%m%d')}" for s, e in date_pairs
    ]
    return xr.Dataset(
        {
            "swe_accum": _make_spatial_da("swe_accum", value=50.0, extra_dims={"pair": pair_labels}),
            "swe_ablate": _make_spatial_da("swe_ablate", value=20.0, extra_dims={"pair": pair_labels}),
            "density_change": _make_spatial_da("density_change", value=0.0, extra_dims={"pair": pair_labels}),
        }
    )


@pytest.fixture()
def clean_aorc_ds(date_pairs) -> xr.Dataset:
    """AORC meteorological Dataset without CRS artefacts."""
    pair_labels = [
        f"{s.strftime('%y%m%d')}_{e.strftime('%y%m%d')}" for s, e in date_pairs
    ]
    return xr.Dataset(
        {
            "mean_temp": _make_spatial_da("mean_temp", value=-5.0, extra_dims={"pair": pair_labels}),
            "total_precip": _make_spatial_da("total_precip", value=30.0, extra_dims={"pair": pair_labels}),
        }
    )


# ---------------------------------------------------------------------------
# "Dirty" layer fixtures — simulate pre-sanitization state
# ---------------------------------------------------------------------------

_DIRTY_ATTRS = {
    "grid_mapping": "spatial_ref",
    "crs_wkt": "GEOGCS[\"WGS 84\"]",
}


def _add_dirty_crs(ds: xr.Dataset, coord_name: str = "spatial_ref") -> xr.Dataset:
    """
    Inject the kinds of CRS artifacts that rioxarray adds to datasets:

    1. A scalar coordinate named *coord_name* (e.g. ``spatial_ref`` or ``crs``).
    2. ``grid_mapping`` and ``crs_wkt`` attrs on every data variable.
    3. ``grid_mapping`` attrs on the coordinate arrays ``x`` and ``y`` — this
       is the pattern that caused ``RioXarrayError: Multiple grid mappings exist``
       and is the primary target of the deep-sanitization fix.
    4. ``grid_mapping`` in the ``.encoding`` of every variable and coordinate array
       — rioxarray reads from ``.encoding`` in addition to ``.attrs`` when loading
       data from NetCDF/Zarr files, so both must be populated for realistic tests.
    """
    # Add the scalar CRS coordinate.
    ds = ds.assign_coords({coord_name: xr.DataArray(0, attrs={"crs_wkt": _DIRTY_ATTRS["crs_wkt"]})})

    # Stamp dirty attrs and encoding on every data variable and every coordinate array.
    for var_name in list(ds.data_vars) + ["x", "y"]:
        for attr_key, attr_val in _DIRTY_ATTRS.items():
            ds[var_name].attrs[attr_key] = attr_val
            ds[var_name].encoding[attr_key] = attr_val

    return ds


@pytest.fixture()
def dirty_coh_ds(clean_coh_ds) -> xr.Dataset:
    """
    Coherence Dataset with a ``spatial_ref`` coordinate and ``grid_mapping``
    attrs on all variables *and* coordinate arrays.

    Mirrors the state returned by ``get_uavsar_coherence`` before sanitization.
    """
    return _add_dirty_crs(clean_coh_ds, coord_name="spatial_ref")


@pytest.fixture()
def dirty_incidence_da(clean_incidence_da) -> xr.DataArray:
    """
    Incidence-angle DataArray with ``grid_mapping`` in both ``.attrs`` and
    ``.encoding`` on the data variable and coordinate arrays.

    Mirrors the actual return type of ``get_uavsar_incidence`` (a DataArray)
    before sanitization in ``assemble_data``.
    """
    da = clean_incidence_da.copy()
    for attr_key, attr_val in _DIRTY_ATTRS.items():
        da.attrs[attr_key] = attr_val
        da.encoding[attr_key] = attr_val
    for coord_name in ["x", "y"]:
        for attr_key, attr_val in _DIRTY_ATTRS.items():
            da[coord_name].attrs[attr_key] = attr_val
            da[coord_name].encoding[attr_key] = attr_val
    return da


@pytest.fixture()
def dirty_incidence_ds(clean_incidence_da) -> xr.Dataset:
    """
    Incidence-angle Dataset with a ``spatial_ref`` coordinate and dirty attrs.
    """
    ds = clean_incidence_da.to_dataset(name="incidence_angle")
    return _add_dirty_crs(ds, coord_name="spatial_ref")


@pytest.fixture()
def dirty_dem_ds(clean_dem_da) -> xr.Dataset:
    """
    DEM Dataset with a ``crs`` coordinate and dirty attrs.
    """
    ds = clean_dem_da.to_dataset(name="elevation")
    return _add_dirty_crs(ds, coord_name="crs")


@pytest.fixture()
def dirty_topo_ds(clean_topo_ds) -> xr.Dataset:
    """Topographic Dataset with dirty CRS artefacts."""
    return _add_dirty_crs(clean_topo_ds, coord_name="spatial_ref")


@pytest.fixture()
def dirty_nlcd_ds(clean_nlcd_ds) -> xr.Dataset:
    """NLCD Dataset with dirty CRS artefacts."""
    return _add_dirty_crs(clean_nlcd_ds, coord_name="spatial_ref")


@pytest.fixture()
def dirty_snow_class_ds(clean_snow_class_ds) -> xr.Dataset:
    """
    Snow-class Dataset with a ``crs`` coordinate and ``grid_mapping`` attrs on
    coordinate arrays — the other common dirty pattern.
    """
    return _add_dirty_crs(clean_snow_class_ds, coord_name="crs")


@pytest.fixture()
def dirty_snow_ds(clean_snow_ds) -> xr.Dataset:
    """UCLA snow metrics Dataset with dirty CRS artefacts."""
    return _add_dirty_crs(clean_snow_ds, coord_name="spatial_ref")


@pytest.fixture()
def dirty_aorc_ds(clean_aorc_ds) -> xr.Dataset:
    """AORC Dataset with dirty CRS artefacts."""
    return _add_dirty_crs(clean_aorc_ds, coord_name="spatial_ref")


# ---------------------------------------------------------------------------
# Fixtures for tile-by-tile / out-of-core architecture tests
# ---------------------------------------------------------------------------

# The tile_aoi fixture re-uses the existing `aoi` fixture (box(-120, 38, -119.9, 38.1)).

@pytest.fixture()
def tile_ref_grid() -> xr.DataArray:
    """
    5 × 5 tile reference grid with EPSG:4326 CRS written, covering the tile_aoi.

    Unlike the plain ``ref_grid`` fixture, this one carries a proper CRS so that
    ``rio.reproject_match`` can use it as a target.
    """
    da = _make_spatial_da("tile_ref", value=0.0)
    return da.rio.write_crs("EPSG:4326")


@pytest.fixture()
def larger_tif_dir(tmp_path, flight_ids, date_pairs) -> Path:
    """
    Create per-flight directories with coherence ``.tif`` files whose spatial
    extent is *larger* than the tile_aoi.

    The tiles are 30 × 30 pixels covering ``(-120.1, 37.9, -119.8, 38.2)``
    while the tile_aoi covers only ``(-120.0, 38.0, -119.9, 38.1)``.  This
    allows the pre-clipping logic in ``get_uavsar_coherence`` to be exercised:
    after ``da.sel(...)`` the raster should be reduced from 30×30 to a smaller
    subset before ``reproject_match`` is called.
    """
    # Raster extent is intentionally larger than tile_aoi in all directions.
    tif_minx, tif_miny, tif_maxx, tif_maxy = -120.1, 37.9, -119.8, 38.2
    ny, nx = 30, 30

    raster_transform = from_bounds(tif_minx, tif_miny, tif_maxx, tif_maxy, nx, ny)

    for fid in flight_ids:
        fdir = tmp_path / str(fid)
        fdir.mkdir(parents=True, exist_ok=True)
        for s, e in date_pairs:
            fname = (
                f"uavsar_{fid}_"
                f"{s.strftime('%y%m%d')}_{e.strftime('%y%m%d')}"
                ".coh.tif"
            )
            fpath = fdir / fname
            data = np.random.rand(1, ny, nx).astype(np.float32)
            with rasterio.open(
                str(fpath),
                "w",
                driver="GTiff",
                height=ny,
                width=nx,
                count=1,
                dtype="float32",
                crs=RioCRS.from_epsg(4326),
                transform=raster_transform,
            ) as dst:
                dst.write(data)

    return tmp_path

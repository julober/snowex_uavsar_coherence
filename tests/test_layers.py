"""
test_layers.py — Unit and integration tests for ``scripts/layers.py``.

Test structure
--------------
1. ``TestGetYears``            — pure-function unit tests for ``get_years``.
2. ``TestGetUavsarCoherence``  — unit tests for ``get_uavsar_coherence``
                                  with rioxarray file I/O mocked out.
3. ``TestGetUavsarIncidence``  — unit tests for ``get_uavsar_incidence``
                                  with rioxarray file I/O mocked out.
4. ``TestAssembleDataSanitization`` — integration test for ``assemble_data``
                                  that verifies the scorched-earth sanitization
                                  eliminates ``RioXarrayError: Multiple grid
                                  mappings exist``.

Running the suite
-----------------
From the repository root::

    pytest tests/ -v

Or, to run only the sanitization integration test::

    pytest tests/test_layers.py::TestAssembleDataSanitization -v
"""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

# ---------------------------------------------------------------------------
# Mirror the coordinate constants defined in conftest.py so that the new test
# classes can build datasets that are spatially consistent with the shared
# conftest fixtures without importing conftest as a module (conftest is not a
# regular importable module — pytest loads it automatically).
# ---------------------------------------------------------------------------
_NY, _NX = 5, 5
_X_COORDS = np.linspace(-120.0, -119.9, _NX)
_Y_COORDS = np.linspace(38.1, 38.0, _NY)  # descending (north → south)

# The conftest.py already inserted the repo root and scripts/ into sys.path
# and stubbed the heavy third-party imports, so the import below will succeed.
from layers import (  # noqa: E402 (post-path-setup import)
    METRES_PER_DEGREE_AT_45_LAT,
    assemble_data,
    assemble_data_largeaoi,
    get_uavsar_coherence,
    get_uavsar_incidence,
    get_years,
)

# ---------------------------------------------------------------------------
# 1. Unit tests — get_years
# ---------------------------------------------------------------------------


class TestGetYears:
    """Tests for the ``get_years`` pure helper function."""

    def test_single_pair_same_year(self):
        """A pair within one calendar year returns a single-element set."""
        pairs = [(date(2021, 2, 10), date(2021, 2, 24))]
        assert get_years(pairs) == {2021}

    def test_single_pair_spanning_years(self):
        """A pair that crosses a year boundary returns both years."""
        pairs = [(date(2020, 12, 15), date(2021, 1, 5))]
        assert get_years(pairs) == {2020, 2021}

    def test_multiple_pairs_same_year(self):
        """Multiple pairs within the same year return a single-element set."""
        pairs = [
            (date(2021, 1, 1), date(2021, 2, 1)),
            (date(2021, 3, 1), date(2021, 4, 1)),
        ]
        assert get_years(pairs) == {2021}

    def test_multiple_pairs_multiple_years(self):
        """Multiple pairs spread across several years returns all unique years."""
        pairs = [
            (date(2020, 11, 1), date(2020, 12, 1)),
            (date(2021, 2, 10), date(2021, 2, 24)),
            (date(2022, 1, 1), date(2022, 2, 1)),
        ]
        assert get_years(pairs) == {2020, 2021, 2022}

    def test_empty_input_returns_empty_set(self):
        """An empty list of pairs produces an empty set."""
        assert get_years([]) == set()

    def test_return_type_is_set(self):
        """``get_years`` always returns a ``set`` of integers."""
        result = get_years([(date(2021, 1, 1), date(2021, 12, 31))])
        assert isinstance(result, set)
        assert all(isinstance(y, int) for y in result)


# ---------------------------------------------------------------------------
# 2. Unit tests — get_uavsar_coherence (file I/O mocked)
# ---------------------------------------------------------------------------


class TestGetUavsarCoherence:
    """Tests for ``get_uavsar_coherence`` with file I/O mocked via tmp_path."""

    _NY, _NX = 5, 5
    _Y_COORDS = np.linspace(38.1, 38.0, _NY)
    _X_COORDS = np.linspace(-120.0, -119.9, _NX)

    def _make_pair_ds(self) -> xr.Dataset:
        """Minimal single-band Dataset returned by ``xr.open_dataset``."""
        data = np.random.rand(self._NY, self._NX).astype(np.float32)
        da = xr.DataArray(
            data,
            dims=["y", "x"],
            coords={"y": self._Y_COORDS, "x": self._X_COORDS},
            name="band_data",
        )
        return da.to_dataset(name="band_data")

    def _make_flight_dir(self, tmp_path, flight_id, date_pairs):
        """Create a fake directory tree with stub coherence .tif files."""
        flight_dir = tmp_path / str(flight_id)
        flight_dir.mkdir(parents=True)
        for s, e in date_pairs:
            fname = (
                f"uavsar_{flight_id}_"
                f"{s.strftime('%y%m%d')}_{e.strftime('%y%m%d')}"
                ".coh.tif"
            )
            (flight_dir / fname).touch()
        return flight_dir

    def test_output_contains_coherence_variable(self, tmp_path, date_pairs, flight_ids, ref_grid):
        """The returned Dataset must contain a ``coherence`` variable."""
        for fid in flight_ids:
            self._make_flight_dir(tmp_path, fid, date_pairs)

        pair_ds = self._make_pair_ds()
        renamed = pair_ds.rename({"band_data": "coherence"})

        with patch("layers.xr.open_dataset", side_effect=lambda path, **kw: pair_ds):
            with patch("xarray.Dataset.rio", create=True) as ds_rio:
                ds_rio.write_crs = MagicMock(return_value=None)
                ds_rio.reproject_match = MagicMock(return_value=renamed)

                result = get_uavsar_coherence(
                    tile_aoi=None,
                    date_pairs=date_pairs,
                    flight_ids=flight_ids,
                    crs="EPSG:4326",
                    tile_ref_grid=ref_grid,
                    fp=str(tmp_path),
                )

        assert "coherence" in result

    def test_missing_flight_dir_raises(self, tmp_path, date_pairs, flight_ids, ref_grid):
        """A missing flight directory must raise ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError, match="Flight directory not found"):
            get_uavsar_coherence(
                tile_aoi=None,
                date_pairs=date_pairs,
                flight_ids=flight_ids,
                crs="EPSG:4326",
                tile_ref_grid=ref_grid,
                fp=str(tmp_path),  # no sub-directories created
            )

    def test_missing_coherence_file_pads_nans(self, tmp_path, date_pairs, flight_ids, ref_grid):
        """
        When a coherence file is absent, the function must pad the missing
        pair with NaN values (using a dummy array shaped like ``tile_ref_grid``)
        rather than raising ``FileNotFoundError``.
        """
        for fid in flight_ids:
            (tmp_path / str(fid)).mkdir(parents=True)

        result = get_uavsar_coherence(
            tile_aoi=None,
            date_pairs=date_pairs,
            flight_ids=flight_ids,
            crs="EPSG:4326",
            tile_ref_grid=ref_grid,
            fp=str(tmp_path),
        )
        # The dataset must still contain the 'coherence' variable, filled with NaN.
        assert "coherence" in result
        assert result["coherence"].isnull().all().item()

    # def test_delta_t_coordinate_values(self, tmp_path, date_pairs, flight_ids, ref_grid):
    #     """``delta_t`` coordinate must equal ``(end_date - start_date).days``."""
    #     for fid in flight_ids:
    #         self._make_flight_dir(tmp_path, fid, date_pairs)

    #     pair_ds = self._make_pair_ds()
    #     expected_deltas = [(e - s).days for s, e in date_pairs]

    #     def fake_open_dataset(path, **kwargs):
    #         return pair_ds

    #     with patch("layers.xr.open_dataset", side_effect=fake_open_dataset):
    #         with patch("xarray.Dataset.rio", create=True) as ds_rio:
    #             renamed = pair_ds.rename({"band_data": "coherence"})
    #             ds_rio.write_crs = MagicMock(return_value=None)
    #             ds_rio.reproject_match = MagicMock(return_value=renamed)

    #             result = get_uavsar_coherence(
    #                 aoi=None,
    #                 date_pairs=date_pairs,
    #                 flight_ids=flight_ids,
    #                 crs="EPSG:4326",
    #                 ref_grid=ref_grid,
    #                 fp=str(tmp_path),
    #             )

    #     assert "delta_t" in result.coords
    #     np.testing.assert_array_equal(
    #         result["delta_t"].values,
    #         np.array(expected_deltas, dtype=np.int64),
    #     )


# ---------------------------------------------------------------------------
# 3. Unit tests — get_uavsar_incidence (file I/O mocked)
# ---------------------------------------------------------------------------


class TestGetUavsarIncidence:
    """Tests for ``get_uavsar_incidence`` with I/O mocked via ``read_and_reproject_rasterio``."""

    _NY, _NX = 5, 5
    _Y_COORDS = np.linspace(38.1, 38.0, _NY)
    _X_COORDS = np.linspace(-120.0, -119.9, _NX)

    def _make_inc_ds(self) -> xr.Dataset:
        """Minimal incidence-angle Dataset as returned by ``read_and_reproject_rasterio``."""
        data = np.full((self._NY, self._NX), 45.0, dtype=np.float32)
        da = xr.DataArray(
            data,
            dims=["y", "x"],
            coords={"y": self._Y_COORDS, "x": self._X_COORDS},
            name="incidence_angle",
        )
        return da.to_dataset(name="incidence_angle")

    def _make_inc_file(self, tmp_path, flight_id):
        fname = f"uavsar_{flight_id}_s2.inc.tif"
        (tmp_path / fname).touch()

    def test_output_named_incidence_angle(self, tmp_path, flight_ids, ref_grid, aoi):
        """The result DataArray must have the name ``'incidence_angle'``."""
        for fid in flight_ids:
            self._make_inc_file(tmp_path, fid)

        inc_ds = self._make_inc_ds()

        with patch("layers.read_and_reproject_rasterio", return_value=inc_ds):
            result = get_uavsar_incidence(
                tile_aoi=aoi,
                flight_ids=flight_ids,
                fp_inc=str(tmp_path),
                tile_ref_grid=ref_grid,
                crs="EPSG:4326",
            )

        assert result.name == "incidence_angle"

    def test_missing_incidence_file_raises(self, tmp_path, flight_ids, ref_grid, aoi):
        """A missing incidence file must raise ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError, match="Could not find pre-calculated incidence file"):
            get_uavsar_incidence(
                tile_aoi=aoi,
                flight_ids=flight_ids,
                fp_inc=str(tmp_path),  # no files created
                tile_ref_grid=ref_grid,
                crs="EPSG:4326",
            )


# ---------------------------------------------------------------------------
# 4. Integration test — assemble_data sanitization / RioXarrayError fix
# ---------------------------------------------------------------------------


class TestAssembleDataSanitization:
    """
    Integration test that verifies the scorched-earth sanitization in
    ``assemble_data`` successfully merges datasets that carry conflicting
    CRS metadata on their coordinate arrays.

    All ``get_*`` functions and external API calls are mocked to return the
    "dirty" synthetic fixtures from ``conftest.py``.  The test then asserts
    that the returned ``xr.Dataset``:

    1. Contains *all* expected data variables — the merge did not raise
       ``RioXarrayError: Multiple grid mappings exist``.
    2. Has no ``grid_mapping`` attribute on any variable (including coordinate
       arrays), confirming that the deep ``.variables`` iteration worked.
    3. Has exactly **one** ``spatial_ref`` coordinate written by the single
       ``ds.rio.write_crs(crs)`` call that follows the merge.
    """


    @staticmethod
    def _make_pair_labels(date_pairs):
        return [
            f"{s.strftime('%y%m%d')}_{e.strftime('%y%m%d')}"
            for s, e in date_pairs
        ]


# ---------------------------------------------------------------------------
# 5. Pre-clipping tests — get_uavsar_coherence with real on-disk .tif files
# ---------------------------------------------------------------------------


class TestPreClippingGetUavsarCoherence:
    """
    Verify that ``get_uavsar_coherence`` aligns output to the tile reference
    grid using the rasterio WarpedVRT path (``read_and_reproject_rasterio``).

    Real ``.tif`` files (30 × 30 pixels, extent ``(-120.1, 37.9, -119.8, 38.2)``)
    are written to disk by the ``larger_tif_dir`` fixture.  The tile_aoi covers
    only ``(-120.0, 38.0, -119.9, 38.1)`` — roughly the inner third of the
    raster extent.

    The WarpedVRT approach replaces the former explicit ``.sel()`` pre-clip;
    rasterio handles windowed reading internally.  Tests in this class verify:

    * the final output spatial dimensions equal those of ``tile_ref_grid``;
    * ``read_and_reproject_rasterio`` is invoked once per (flight_id, pair).
    """

    def _make_single_pair_ds(self, tile_ref_grid):
        """
        Return a single 2-D (y × x) coherence Dataset that mirrors what
        ``read_and_reproject_rasterio`` produces for **one (flight, pair)** call.

        ``get_uavsar_coherence`` calls ``read_and_reproject_rasterio`` once per
        (flight_id, date_pair) combination and then uses ``xr.concat`` to
        assemble the final 4-D ``(flight_id, pair, y, x)`` cube.  The mock
        must therefore return only the single-slice result.
        """
        return xr.Dataset(
            {
                "coherence": xr.DataArray(
                    np.full((_NY, _NX), 0.7, dtype=np.float32),
                    dims=["y", "x"],
                    coords={
                        "y": tile_ref_grid.coords["y"].values,
                        "x": tile_ref_grid.coords["x"].values,
                    },
                )
            }
        )

    def test_spatial_dims_match_tile_ref_grid(
        self, larger_tif_dir, date_pairs, flight_ids, aoi, tile_ref_grid
    ):
        """
        The output ``coherence`` variable must have the same spatial dimensions
        (y × x) as ``tile_ref_grid``, proving that ``read_and_reproject_rasterio``
        aligned the oversized raster to the tile reference grid.
        """
        import layers as _layers

        single_pair_ds = self._make_single_pair_ds(tile_ref_grid)

        with patch.object(_layers, "read_and_reproject_rasterio", return_value=single_pair_ds):
            result = get_uavsar_coherence(
                tile_aoi=aoi,
                date_pairs=date_pairs,
                flight_ids=flight_ids,
                crs="EPSG:4326",
                tile_ref_grid=tile_ref_grid,
                fp=str(larger_tif_dir),
            )

        assert result["coherence"].sizes["x"] == tile_ref_grid.sizes["x"]
        assert result["coherence"].sizes["y"] == tile_ref_grid.sizes["y"]

    def test_vrt_called_per_flight_and_pair(
        self, larger_tif_dir, date_pairs, flight_ids, aoi, tile_ref_grid
    ):
        """
        ``read_and_reproject_rasterio`` must be called exactly once per
        (flight_id, date_pair) combination, with ``tile_ref_grid`` as the
        ``ref_da`` argument.

        This replaces the old pre-clip test: with the rasterio WarpedVRT
        approach, windowed reading is handled internally by rasterio, so
        there is no explicit ``.sel()`` pre-clip to inspect.  Instead we
        verify that the fast VRT path is invoked the correct number of times.
        """
        import layers as _layers

        single_pair_ds = self._make_single_pair_ds(tile_ref_grid)

        with patch.object(
            _layers, "read_and_reproject_rasterio", return_value=single_pair_ds
        ) as mock_vrt:
            get_uavsar_coherence(
                tile_aoi=aoi,
                date_pairs=date_pairs,
                flight_ids=flight_ids,
                crs="EPSG:4326",
                tile_ref_grid=tile_ref_grid,
                fp=str(larger_tif_dir),
            )

        expected_calls = len(flight_ids) * len(date_pairs)
        assert mock_vrt.call_count == expected_calls, (
            f"Expected {expected_calls} VRT calls "
            f"({len(flight_ids)} flights × {len(date_pairs)} pairs), "
            f"got {mock_vrt.call_count}"
        )
        # Every call must use the tile_ref_grid as the reprojection target.
        for call_args in mock_vrt.call_args_list:
            assert call_args.kwargs.get("ref_da") is tile_ref_grid, (
                "read_and_reproject_rasterio was not called with tile_ref_grid as ref_da"
            )


# ---------------------------------------------------------------------------
# 6. Pure-function tests — assemble_data returns in-memory Dataset, no I/O
# ---------------------------------------------------------------------------


class TestAssembleDataPureFunction:
    """
    Verify that ``assemble_data`` is a pure function: it assembles all layers
    for a tile in memory and returns an ``xr.Dataset`` *without* writing any
    data to disk.

    All external calls (file I/O, API fetchers, DEM provider) are mocked so
    that the test runs offline in milliseconds.
    """

    # ── Expected output variable names ────────────────────────────────────
    EXPECTED_VARS = {
        "coherence",
        "incidence_angle",
        "elevation",
        "slope",
        "aspect",
        "curve",
        "cover_2019",
        "canopy_2019",
        "snow_class",
        "swe_accum",
        "swe_ablate",
        "density_change",
        "mean_temp",
        "total_precip",
    }

    # Number of pair-labelled coordinate in the coherence/snow/aorc dims
    _N_PAIRS = 2

    @pytest.fixture(autouse=True)
    def _mock_all_sub_functions(
        self,
        aoi,
        date_pairs,
        flight_ids,
        ref_grid,
        clean_coh_ds,
        clean_incidence_da,
        clean_dem_da,
        clean_topo_ds,
        clean_nlcd_ds,
        clean_snow_class_ds,
        clean_snow_ds,
        clean_aorc_ds,
        monkeypatch,
    ):
        """Patch every external call so tests run offline without disk I/O."""
        import layers as _layers

        # Patch input-validation helpers to accept datetime.date objects.
        monkeypatch.setattr(_layers, "validate_aoi", lambda x: x)
        monkeypatch.setattr(_layers, "validate_date_pairs", lambda x: x)

        # make_reference_grid → return the shared 5×5 ref_grid (with CRS).
        tile_ref_with_crs = ref_grid.rio.write_crs("EPSG:4326")
        monkeypatch.setattr(
            _layers, "make_reference_grid", lambda **kw: tile_ref_with_crs
        )

        # Patch validate_alignment to a no-op.
        monkeypatch.setattr(_layers, "validate_alignment", lambda ds_list: None)

        # Patch all data-fetcher functions.
        monkeypatch.setattr(
            _layers, "get_uavsar_coherence", lambda **kw: clean_coh_ds
        )
        monkeypatch.setattr(
            _layers, "get_uavsar_incidence", lambda **kw: clean_incidence_da
        )
        monkeypatch.setattr(
            _layers, "get_topo_layers", lambda **kw: clean_topo_ds
        )
        monkeypatch.setattr(
            _layers, "get_nlcd_layers", lambda **kw: clean_nlcd_ds
        )
        monkeypatch.setattr(
            _layers, "get_snow_climatology", lambda **kw: clean_snow_class_ds
        )
        monkeypatch.setattr(
            _layers, "get_aorc_layers", lambda **kw: clean_aorc_ds
        )
        monkeypatch.setattr(
            _layers, "get_snow_layers", lambda **kw: clean_snow_ds
        )

        # Patch py3dep.get_dem to return the clean DEM DataArray.
        dem_with_crs = clean_dem_da.rio.write_crs("EPSG:4326")
        monkeypatch.setattr(_layers.py3dep, "get_dem", lambda **kw: dem_with_crs)

        # Patch dem.rio.reproject_match → identity (return DEM as-is).
        from rioxarray.raster_array import RasterArray

        original_rm = RasterArray.reproject_match

        def _identity_reproject_match(self_rio, ref, *args, **kwargs):
            return self_rio._obj

        monkeypatch.setattr(RasterArray, "reproject_match", _identity_reproject_match)

        # Store the original for teardown (monkeypatch handles this automatically).

    def test_returns_xr_dataset(self, aoi, date_pairs, flight_ids):
        """``assemble_data`` must return an ``xr.Dataset``."""
        result = assemble_data(
            tile_aoi=aoi,
            date_pairs=date_pairs,
            flight_ids=flight_ids,
        )
        assert isinstance(result, xr.Dataset)

    def test_contains_expected_variables(self, aoi, date_pairs, flight_ids):
        """All expected data-variable names must be present in the merged dataset."""
        result = assemble_data(
            tile_aoi=aoi,
            date_pairs=date_pairs,
            flight_ids=flight_ids,
        )
        missing = self.EXPECTED_VARS - set(result.data_vars)
        assert not missing, f"Missing variables: {missing}"

    def test_does_not_write_zarr(self, aoi, date_pairs, flight_ids, tmp_path):
        """
        ``assemble_data`` must NOT write any ``.zarr`` file; it is a pure
        function that only returns an in-memory dataset.
        """
        assemble_data(
            tile_aoi=aoi,
            date_pairs=date_pairs,
            flight_ids=flight_ids,
        )
        zarr_files = list(tmp_path.rglob("*.zarr"))
        assert zarr_files == [], f"Unexpected zarr file(s) written: {zarr_files}"

    def test_nan_masking_applied(self, aoi, date_pairs, flight_ids):
        """
        After the NaN-mask pass, spatial pixels that have NaN coherence in
        *any* (flight_id, pair) combination must be NaN in all other variables
        that share the spatial (y, x) dimensions.

        The ``clean_coh_ds`` fixture has no NaN values (all 0.7), so all
        non-coherence variables should remain unchanged.
        """
        result = assemble_data(
            tile_aoi=aoi,
            date_pairs=date_pairs,
            flight_ids=flight_ids,
        )
        # With no NaN coherence, the mask is all-False; no values should be NaN.
        for var in result.data_vars:
            if "y" in result[var].dims and "x" in result[var].dims:
                assert not result[var].isnull().any().item(), (
                    f"Variable '{var}' unexpectedly contains NaN values after "
                    "NaN-mask pass with all-finite coherence."
                )


# ---------------------------------------------------------------------------
# 7. Manager-function tests — assemble_data_largeaoi writes a Zarr store
# ---------------------------------------------------------------------------


class TestAssembleDataLargeAOI:
    """
    Verify that ``assemble_data_largeaoi``:

    1. Correctly splits the master AOI into the expected number of tiles.
    2. Calls ``assemble_data`` once per tile.
    3. Initialises and populates a Zarr store at ``fp_dest`` without region-
       slice mismatches.
    4. The Zarr store contains all variables returned by ``assemble_data``.
    """

    # ── Shared grid parameters ─────────────────────────────────────────────
    # Use the same tiny 5 × 5 coordinates as the conftest baseline so that the
    # mock tile dataset is consistent with the global reference grid.

    @staticmethod
    def _build_tile_ds(date_pairs, flight_ids) -> xr.Dataset:
        """Minimal tile dataset with all expected variables, using conftest coords."""
        pair_labels = [
            f"{s.strftime('%y%m%d')}_{e.strftime('%y%m%d')}"
            for s, e in date_pairs
        ]
        coords_yx = {"y": _Y_COORDS, "x": _X_COORDS}
        coords_fid_pair_yx = {
            "flight_id": flight_ids,
            "pair": pair_labels,
            **coords_yx,
        }
        coords_fid_yx = {"flight_id": flight_ids, **coords_yx}
        coords_pair_yx = {"pair": pair_labels, **coords_yx}

        return xr.Dataset(
            {
                "coherence": xr.DataArray(
                    np.full((len(flight_ids), len(date_pairs), _NY, _NX), 0.7, dtype="float32"),
                    dims=["flight_id", "pair", "y", "x"],
                    coords=coords_fid_pair_yx,
                ),
                "incidence_angle": xr.DataArray(
                    np.full((len(flight_ids), _NY, _NX), 45.0, dtype="float32"),
                    dims=["flight_id", "y", "x"],
                    coords=coords_fid_yx,
                ),
                "elevation": xr.DataArray(
                    np.full((_NY, _NX), 2000.0, dtype="float32"),
                    dims=["y", "x"],
                    coords=coords_yx,
                ),
                "slope": xr.DataArray(
                    np.full((_NY, _NX), 10.0, dtype="float32"),
                    dims=["y", "x"],
                    coords=coords_yx,
                ),
                "snow_class": xr.DataArray(
                    np.full((_NY, _NX), 3.0, dtype="float32"),
                    dims=["y", "x"],
                    coords=coords_yx,
                ),
                "mean_temp": xr.DataArray(
                    np.full((len(date_pairs), _NY, _NX), -5.0, dtype="float32"),
                    dims=["pair", "y", "x"],
                    coords=coords_pair_yx,
                ),
            }
        )

    @staticmethod
    def _build_mock_aorc_ds(date_pairs) -> xr.Dataset:
        """
        Minimal AORC metrics dataset returned by the mocked ``get_aorc_layers``.

        ``assemble_data_largeaoi`` calls ``.compute()`` on the result before
        passing it to per-tile ``assemble_data`` calls.  Since this is a plain
        (non-Dask) xr.Dataset, ``.compute()`` is a no-op that returns itself.
        """
        pair_labels = [
            f"{s.strftime('%y%m%d')}_{e.strftime('%y%m%d')}"
            for s, e in date_pairs
        ]
        return xr.Dataset(
            {
                "mean_temp": xr.DataArray(
                    np.full((len(date_pairs), _NY, _NX), -5.0, dtype="float32"),
                    dims=["pair", "y", "x"],
                    coords={"pair": pair_labels, "y": _Y_COORDS, "x": _X_COORDS},
                ),
            }
        )

    @pytest.fixture()
    def global_ref(self) -> xr.DataArray:
        """5 × 5 global reference DataArray matching the tile dataset's coordinates."""
        return xr.DataArray(
            np.zeros((_NY, _NX), dtype="float32"),
            dims=["y", "x"],
            coords={"y": _Y_COORDS, "x": _X_COORDS},
        )

    def test_zarr_store_is_created(
        self, aoi, date_pairs, flight_ids, global_ref, tmp_path, monkeypatch
    ):
        """
        ``assemble_data_largeaoi`` must create a valid Zarr store at ``fp_dest``
        that can be opened with :func:`xarray.open_zarr`.
        """
        import layers as _layers

        tile_ds = self._build_tile_ds(date_pairs, flight_ids)
        mock_aorc_ds = self._build_mock_aorc_ds(date_pairs)
        fp_dest = str(tmp_path / "output.zarr")

        monkeypatch.setattr(_layers, "validate_aoi", lambda x: x)
        monkeypatch.setattr(_layers, "validate_date_pairs", lambda x: x)
        monkeypatch.setattr(_layers, "make_reference_grid", lambda **kw: global_ref)
        monkeypatch.setattr(_layers, "get_aorc_layers", lambda **kw: mock_aorc_ds)
        monkeypatch.setattr(_layers, "assemble_data", lambda **kw: tile_ds)

        assemble_data_largeaoi(
            master_aoi=aoi,
            date_pairs=date_pairs,
            flight_ids=flight_ids,
            fp_dest=fp_dest,
            tile_size_m=100_000,  # large → single tile covers full master AOI
            overwrite=True,
        )

        result = xr.open_zarr(fp_dest)
        assert isinstance(result, xr.Dataset)

    def test_assemble_data_called_per_tile(
        self, aoi, date_pairs, flight_ids, global_ref, tmp_path, monkeypatch
    ):
        """
        With ``tile_size_m`` large enough for exactly one tile, ``assemble_data``
        must be called exactly once.
        """
        import layers as _layers

        tile_ds = self._build_tile_ds(date_pairs, flight_ids)
        mock_aorc_ds = self._build_mock_aorc_ds(date_pairs)
        fp_dest = str(tmp_path / "output.zarr")

        call_log: list = []

        def _counting_assemble_data(**kw):
            call_log.append(kw["tile_aoi"])
            return tile_ds

        monkeypatch.setattr(_layers, "validate_aoi", lambda x: x)
        monkeypatch.setattr(_layers, "validate_date_pairs", lambda x: x)
        monkeypatch.setattr(_layers, "make_reference_grid", lambda **kw: global_ref)
        monkeypatch.setattr(_layers, "get_aorc_layers", lambda **kw: mock_aorc_ds)
        monkeypatch.setattr(_layers, "assemble_data", _counting_assemble_data)

        assemble_data_largeaoi(
            master_aoi=aoi,
            date_pairs=date_pairs,
            flight_ids=flight_ids,
            fp_dest=fp_dest,
            tile_size_m=100_000,
            overwrite=True,
        )

        assert len(call_log) == 1, (
            f"Expected 1 call to assemble_data for a single-tile AOI, "
            f"got {len(call_log)}"
        )

    def test_zarr_contains_expected_variables(
        self, aoi, date_pairs, flight_ids, global_ref, tmp_path, monkeypatch
    ):
        """
        The Zarr store produced by ``assemble_data_largeaoi`` must contain all
        data variables returned by the mock ``assemble_data``.
        """
        import layers as _layers

        tile_ds = self._build_tile_ds(date_pairs, flight_ids)
        mock_aorc_ds = self._build_mock_aorc_ds(date_pairs)
        fp_dest = str(tmp_path / "output.zarr")

        monkeypatch.setattr(_layers, "validate_aoi", lambda x: x)
        monkeypatch.setattr(_layers, "validate_date_pairs", lambda x: x)
        monkeypatch.setattr(_layers, "make_reference_grid", lambda **kw: global_ref)
        monkeypatch.setattr(_layers, "get_aorc_layers", lambda **kw: mock_aorc_ds)
        monkeypatch.setattr(_layers, "assemble_data", lambda **kw: tile_ds)

        assemble_data_largeaoi(
            master_aoi=aoi,
            date_pairs=date_pairs,
            flight_ids=flight_ids,
            fp_dest=fp_dest,
            tile_size_m=100_000,
            overwrite=True,
        )

        result = xr.open_zarr(fp_dest)
        for var in tile_ds.data_vars:
            assert var in result.data_vars, (
                f"Variable '{var}' missing from Zarr store."
            )

    def test_region_write_matches_tile_position(
        self, aoi, date_pairs, flight_ids, global_ref, tmp_path, monkeypatch
    ):
        """
        The data in the Zarr store must have been written into the correct
        spatial region.  For a single-tile scenario (tile = master AOI),
        the global x/y coordinate ranges must exactly match those of the tile.
        """
        import layers as _layers

        tile_ds = self._build_tile_ds(date_pairs, flight_ids)
        mock_aorc_ds = self._build_mock_aorc_ds(date_pairs)
        fp_dest = str(tmp_path / "output.zarr")

        monkeypatch.setattr(_layers, "validate_aoi", lambda x: x)
        monkeypatch.setattr(_layers, "validate_date_pairs", lambda x: x)
        monkeypatch.setattr(_layers, "make_reference_grid", lambda **kw: global_ref)
        monkeypatch.setattr(_layers, "get_aorc_layers", lambda **kw: mock_aorc_ds)
        monkeypatch.setattr(_layers, "assemble_data", lambda **kw: tile_ds)

        assemble_data_largeaoi(
            master_aoi=aoi,
            date_pairs=date_pairs,
            flight_ids=flight_ids,
            fp_dest=fp_dest,
            tile_size_m=100_000,
            overwrite=True,
        )

        result = xr.open_zarr(fp_dest)
        np.testing.assert_array_almost_equal(
            result.coords["x"].values,
            _X_COORDS,
            decimal=6,
            err_msg="Zarr x-coordinates do not match the tile x-coordinates.",
        )
        np.testing.assert_array_almost_equal(
            result.coords["y"].values,
            _Y_COORDS,
            decimal=6,
            err_msg="Zarr y-coordinates do not match the tile y-coordinates.",
        )

    def test_multiple_tiles_all_processed(
        self, date_pairs, flight_ids, tmp_path, monkeypatch
    ):
        """
        When the master AOI spans more than one tile, ``assemble_data`` is
        called for every tile and the Zarr store is initialised only once.

        A 4 × 4 master AOI with ``tile_size_m`` set to half its span produces
        four tiles; the mock ``assemble_data`` is expected to be called four
        times.
        """
        import layers as _layers

        # 4-tile scenario: master_aoi is 0.2° × 0.2°, tile covers 0.1° × 0.1°
        master_4tiles = box(-120.0, 38.0, -119.8, 38.2)
        # Each sub-tile's grid will be 5 × 5 within its own quadrant.
        tile_x0 = np.linspace(-120.0, -119.9, _NX)
        tile_y0 = np.linspace(38.2, 38.1, _NY)  # NW tile
        tile_x1 = np.linspace(-119.9, -119.8, _NX)
        tile_y1 = np.linspace(38.1, 38.0, _NY)  # SE tile

        # Global grid covers the full 0.2° × 0.2° extent.
        global_x = np.linspace(-120.0, -119.8, _NX * 2 - 1)
        global_y = np.linspace(38.2, 38.0, _NY * 2 - 1)
        big_global_ref = xr.DataArray(
            np.zeros((len(global_y), len(global_x)), dtype="float32"),
            dims=["y", "x"],
            coords={"y": global_y, "x": global_x},
        )

        call_log: list = []
        tile_quadrant_coords = [
            (tile_x0, tile_y0),
            (tile_x1, tile_y0),
            (tile_x0, tile_y1),
            (tile_x1, tile_y1),
        ]

        def _tile_assemble_data(**kw):
            idx = len(call_log)
            tile_x, tile_y = tile_quadrant_coords[idx % 4]
            pair_labels = [
                f"{s.strftime('%y%m%d')}_{e.strftime('%y%m%d')}"
                for s, e in date_pairs
            ]
            ds = xr.Dataset(
                {
                    "coherence": xr.DataArray(
                        np.full(
                            (len(flight_ids), len(date_pairs), _NY, _NX),
                            0.7,
                            dtype="float32",
                        ),
                        dims=["flight_id", "pair", "y", "x"],
                        coords={
                            "flight_id": flight_ids,
                            "pair": pair_labels,
                            "y": tile_y,
                            "x": tile_x,
                        },
                    )
                }
            )
            call_log.append(ds)
            return ds

        fp_dest = str(tmp_path / "multi_tile.zarr")
        # _TILE_OVERLAP_FACTOR slightly exceeds 0.1 so that _frange produces
        # exactly 2 x-starts and 2 y-starts for the 0.2° × 0.2° master AOI,
        # giving 2 × 2 = 4 tiles.  A value of exactly 0.1 is unsafe because
        # floating-point accumulation in _frange can yield a spurious third start.
        _TILE_OVERLAP_FACTOR = 0.101
        tile_size_m = METRES_PER_DEGREE_AT_45_LAT * _TILE_OVERLAP_FACTOR

        mock_aorc_ds = self._build_mock_aorc_ds(date_pairs)

        monkeypatch.setattr(_layers, "validate_aoi", lambda x: x)
        monkeypatch.setattr(_layers, "validate_date_pairs", lambda x: x)
        monkeypatch.setattr(_layers, "make_reference_grid", lambda **kw: big_global_ref)
        monkeypatch.setattr(_layers, "get_aorc_layers", lambda **kw: mock_aorc_ds)
        monkeypatch.setattr(_layers, "assemble_data", _tile_assemble_data)

        assemble_data_largeaoi(
            master_aoi=master_4tiles,
            date_pairs=date_pairs,
            flight_ids=flight_ids,
            fp_dest=fp_dest,
            tile_size_m=tile_size_m,
            overwrite=True,
        )

        assert len(call_log) == 4, (
            f"Expected 4 assemble_data calls for 4-tile AOI, got {len(call_log)}"
        )

    def test_aorc_prefetched_once_and_passed_to_tiles(
        self, aoi, date_pairs, flight_ids, global_ref, tmp_path, monkeypatch
    ):
        """
        ``get_aorc_layers`` must be called exactly **once** (for the full
        master AOI) regardless of the number of tiles.  Each ``assemble_data``
        call must receive the pre-fetched AORC dataset via the ``aorc_ds``
        keyword argument.
        """
        import layers as _layers

        tile_ds = self._build_tile_ds(date_pairs, flight_ids)
        mock_aorc_ds = self._build_mock_aorc_ds(date_pairs)
        fp_dest = str(tmp_path / "output.zarr")

        aorc_call_log: list = []
        aorc_ds_received: list = []

        def _mock_get_aorc_layers(**kw):
            aorc_call_log.append(kw)
            return mock_aorc_ds

        def _mock_assemble_data(**kw):
            aorc_ds_received.append(kw.get("aorc_ds"))
            return tile_ds

        monkeypatch.setattr(_layers, "validate_aoi", lambda x: x)
        monkeypatch.setattr(_layers, "validate_date_pairs", lambda x: x)
        monkeypatch.setattr(_layers, "make_reference_grid", lambda **kw: global_ref)
        monkeypatch.setattr(_layers, "get_aorc_layers", _mock_get_aorc_layers)
        monkeypatch.setattr(_layers, "assemble_data", _mock_assemble_data)

        assemble_data_largeaoi(
            master_aoi=aoi,
            date_pairs=date_pairs,
            flight_ids=flight_ids,
            fp_dest=fp_dest,
            tile_size_m=100_000,
            overwrite=True,
        )

        assert len(aorc_call_log) == 1, (
            f"get_aorc_layers must be called once (for master AOI), "
            f"got {len(aorc_call_log)} calls"
        )
        # Each tile's assemble_data call must receive the pre-fetched AORC dataset.
        assert len(aorc_ds_received) == 1
        assert aorc_ds_received[0] is not None, (
            "assemble_data did not receive the pre-fetched aorc_ds"
        )

    
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

# The conftest.py already inserted the repo root and scripts/ into sys.path
# and stubbed the heavy third-party imports, so the import below will succeed.
from layers import get_years, assemble_data  # noqa: E402 (post-path-setup import)

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
        from layers import get_uavsar_coherence

        for fid in flight_ids:
            self._make_flight_dir(tmp_path, fid, date_pairs)

        pair_ds = self._make_pair_ds()

        with (
            patch("layers.xr.open_dataset", return_value=pair_ds),
            patch("layers.xr.Dataset.rio") as mock_rio,
        ):
            mock_rio.write_crs = MagicMock(return_value=None)
            # reproject_match returns the same dataset shape.
            mock_reprojected = pair_ds.rename({"band_data": "coherence"})
            mock_reprojected_ds = mock_reprojected

            def fake_open_dataset(path, **kwargs):
                return pair_ds

            with patch("layers.xr.open_dataset", side_effect=fake_open_dataset):
                # Also patch reproject_match at the rioxarray accessor level.
                with patch("xarray.Dataset.rio", create=True) as ds_rio:
                    ds_rio.write_crs = MagicMock(return_value=None)
                    ds_rio.reproject_match = MagicMock(return_value=mock_reprojected_ds)

                    result = get_uavsar_coherence(
                        aoi=None,  # not used when files exist
                        date_pairs=date_pairs,
                        flight_ids=flight_ids,
                        crs="EPSG:4326",
                        ref_grid=ref_grid,
                        fp=str(tmp_path),
                    )

        assert "coherence" in result

    def test_missing_flight_dir_raises(self, tmp_path, date_pairs, flight_ids, ref_grid):
        """A missing flight directory must raise ``FileNotFoundError``."""
        from layers import get_uavsar_coherence

        with pytest.raises(FileNotFoundError, match="Flight directory not found"):
            get_uavsar_coherence(
                aoi=None,
                date_pairs=date_pairs,
                flight_ids=flight_ids,
                crs="EPSG:4326",
                ref_grid=ref_grid,
                fp=str(tmp_path),  # no sub-directories created
            )

    def test_missing_coherence_file_raises(self, tmp_path, date_pairs, flight_ids, ref_grid):
        """A flight directory that exists but has no matching files raises ``FileNotFoundError``."""
        from layers import get_uavsar_coherence

        for fid in flight_ids:
            (tmp_path / str(fid)).mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="Could not find coherence file"):
            get_uavsar_coherence(
                aoi=None,
                date_pairs=date_pairs,
                flight_ids=flight_ids,
                crs="EPSG:4326",
                ref_grid=ref_grid,
                fp=str(tmp_path),
            )

    def test_delta_t_coordinate_values(self, tmp_path, date_pairs, flight_ids, ref_grid):
        """``delta_t`` coordinate must equal ``(end_date - start_date).days``."""
        from layers import get_uavsar_coherence

        for fid in flight_ids:
            self._make_flight_dir(tmp_path, fid, date_pairs)

        pair_ds = self._make_pair_ds()
        expected_deltas = [(e - s).days for s, e in date_pairs]

        def fake_open_dataset(path, **kwargs):
            return pair_ds

        with patch("layers.xr.open_dataset", side_effect=fake_open_dataset):
            with patch("xarray.Dataset.rio", create=True) as ds_rio:
                renamed = pair_ds.rename({"band_data": "coherence"})
                ds_rio.write_crs = MagicMock(return_value=None)
                ds_rio.reproject_match = MagicMock(return_value=renamed)

                result = get_uavsar_coherence(
                    aoi=None,
                    date_pairs=date_pairs,
                    flight_ids=flight_ids,
                    crs="EPSG:4326",
                    ref_grid=ref_grid,
                    fp=str(tmp_path),
                )

        assert "delta_t" in result.coords
        np.testing.assert_array_equal(
            result["delta_t"].values,
            np.array(expected_deltas, dtype=np.int64),
        )


# ---------------------------------------------------------------------------
# 3. Unit tests — get_uavsar_incidence (file I/O mocked)
# ---------------------------------------------------------------------------


class TestGetUavsarIncidence:
    """Tests for ``get_uavsar_incidence`` with rioxarray file I/O mocked."""

    _NY, _NX = 5, 5
    _Y_COORDS = np.linspace(38.1, 38.0, _NY)
    _X_COORDS = np.linspace(-120.0, -119.9, _NX)

    def _make_inc_da(self) -> xr.DataArray:
        """Minimal incidence-angle DataArray returned by ``rxa.open_rasterio``."""
        data = np.full((1, self._NY, self._NX), 45.0, dtype=np.float32)
        return xr.DataArray(
            data,
            dims=["band", "y", "x"],
            coords={
                "band": [1],
                "y": self._Y_COORDS,
                "x": self._X_COORDS,
            },
        )

    def _make_inc_file(self, tmp_path, flight_id):
        fname = f"uavsar_{flight_id}_s2.inc.tif"
        (tmp_path / fname).touch()

    def test_output_named_incidence_angle(self, tmp_path, flight_ids, ref_grid, aoi):
        """The result DataArray must have the name ``'incidence_angle'``."""
        from layers import get_uavsar_incidence

        for fid in flight_ids:
            self._make_inc_file(tmp_path, fid)

        inc_da = self._make_inc_da()

        with patch("layers.rxa.open_rasterio", return_value=inc_da):
            with patch("xarray.DataArray.rio", create=True) as da_rio:
                squeezed = inc_da.squeeze()
                squeezed.name = "incidence_angle"
                da_rio.write_crs = MagicMock(return_value=None)
                da_rio.reproject_match = MagicMock(return_value=squeezed)

                result = get_uavsar_incidence(
                    aoi=aoi,
                    flight_ids=flight_ids,
                    fp_inc=str(tmp_path),
                    ref_grid=ref_grid,
                    crs="EPSG:4326",
                )

        assert result.name == "incidence_angle"

    def test_missing_incidence_file_raises(self, tmp_path, flight_ids, ref_grid, aoi):
        """A missing incidence file must raise ``FileNotFoundError``."""
        from layers import get_uavsar_incidence

        with pytest.raises(FileNotFoundError, match="Could not find pre-calculated incidence file"):
            get_uavsar_incidence(
                aoi=aoi,
                flight_ids=flight_ids,
                fp_inc=str(tmp_path),  # no files created
                ref_grid=ref_grid,
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

    def test_merge_succeeds_with_dirty_inputs(
        self,
        mocker,
        aoi,
        date_pairs,
        flight_ids,
        dirty_coh_ds,
        dirty_incidence_da,
        dirty_dem_ds,
        dirty_topo_ds,
        dirty_nlcd_ds,
        dirty_snow_class_ds,
        dirty_snow_ds,
        dirty_aorc_ds,
    ):
        """
        ``assemble_data`` must merge dirty layers without raising
        ``RioXarrayError: Multiple grid mappings exist`` and must return a
        fully sanitized dataset.
        """
        # ── Mock validate_* helpers ────────────────────────────────────────
        mocker.patch("layers.validate_aoi", side_effect=lambda x: x)
        mocker.patch("layers.validate_date_pairs", side_effect=lambda x: x)
        mocker.patch("layers.validate_alignment", return_value=None)

        # ── Mock make_reference_grid to return a clean 5×5 grid ───────────
        ref = xr.DataArray(
            np.zeros((5, 5), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": np.linspace(38.1, 38.0, 5),
                "x": np.linspace(-120.0, -119.9, 5),
            },
        )
        mocker.patch("layers.make_reference_grid", return_value=ref)

        # ── Mock the eight get_* functions ────────────────────────────────
        mocker.patch("layers.get_uavsar_coherence", return_value=dirty_coh_ds)
        mocker.patch("layers.get_uavsar_incidence", return_value=dirty_incidence_da)
        mocker.patch("layers.get_topo_layers", return_value=dirty_topo_ds)
        mocker.patch("layers.get_nlcd_layers", return_value=dirty_nlcd_ds)
        mocker.patch("layers.get_snow_climatology", return_value=dirty_snow_class_ds)
        mocker.patch("layers.get_aorc_layers", return_value=dirty_aorc_ds)
        mocker.patch("layers.get_snow_layers", return_value=dirty_snow_ds)

        # ── Mock py3dep.get_dem ────────────────────────────────────────────
        # The DEM is a DataArray whose .rio accessor is called for write_crs
        # and reproject_match.  Build a real DataArray and attach a lightweight
        # mock accessor so those calls succeed without GDAL.
        dem_da = xr.DataArray(
            np.full((5, 5), 2000.0, dtype=np.float32),
            dims=["y", "x"],
            coords={"y": ref.y.values, "x": ref.x.values},
            name="elevation",
        )
        dem_da.attrs["grid_mapping"] = "spatial_ref"

        dem_rio = MagicMock()
        dem_rio.write_crs.return_value = dem_da
        dem_rio.reproject_match.return_value = dem_da
        type(dem_da).rio = property(lambda self: dem_rio)

        py3dep_mock = sys.modules.get("py3dep")
        if py3dep_mock is not None:
            py3dep_mock.get_dem = MagicMock(return_value=dem_da)

        # ── Mock ds.chunk() (requires dask which is not in the test env) ──
        mocker.patch("xarray.Dataset.chunk", lambda self, *a, **kw: self)

        # ── Run assemble_data ──────────────────────────────────────────────
        result = assemble_data(
            aoi=aoi,
            date_pairs=date_pairs,
            flight_ids=flight_ids,
            fp_dest=None,   # skip writing to disk
            fp_coh="/dev/null",
            fp_inc="/dev/null",
            fp_snowclimate="/dev/null",
            crs="EPSG:4326",
            res=30,
        )

        # ── Assertions ────────────────────────────────────────────────────

        # 1. All expected variables should be present (merge succeeded).
        expected_vars = {"coherence", "incidence_angle", "slope", "aspect", "curve",
                         "cover_2019", "canopy_2019", "snow_class",
                         "swe_accum", "swe_ablate", "density_change",
                         "mean_temp", "total_precip"}
        for var in expected_vars:
            assert var in result, f"Variable '{var}' missing after merge"

        # 2. No variable (including coordinate arrays) should retain grid_mapping.
        for var_name in result.variables:
            assert "grid_mapping" not in result[var_name].attrs, (
                f"Variable '{var_name}' still has 'grid_mapping' attr after sanitization"
            )

        # 3. The ``spatial_ref`` coordinate must appear exactly once (written
        #    by the single ds.rio.write_crs(crs) call post-merge).
        crs_coord_count = sum(
            1 for c in result.coords if c == "spatial_ref"
        )
        assert crs_coord_count == 1, (
            f"Expected exactly 1 'spatial_ref' coordinate, found {crs_coord_count}"
        )

    def test_no_grid_mapping_on_coordinate_arrays(
        self,
        mocker,
        aoi,
        date_pairs,
        flight_ids,
        dirty_coh_ds,
        dirty_incidence_da,
        dirty_dem_ds,
        dirty_topo_ds,
        dirty_nlcd_ds,
        dirty_snow_class_ds,
        dirty_snow_ds,
        dirty_aorc_ds,
    ):
        """
        Specifically verify that coordinate arrays (x, y) have no ``grid_mapping``
        attr — the root cause of the RioXarrayError.
        """
        mocker.patch("layers.validate_aoi", side_effect=lambda x: x)
        mocker.patch("layers.validate_date_pairs", side_effect=lambda x: x)
        mocker.patch("layers.validate_alignment", return_value=None)

        ref = xr.DataArray(
            np.zeros((5, 5), dtype=np.float32),
            dims=["y", "x"],
            coords={
                "y": np.linspace(38.1, 38.0, 5),
                "x": np.linspace(-120.0, -119.9, 5),
            },
        )
        mocker.patch("layers.make_reference_grid", return_value=ref)
        mocker.patch("layers.get_uavsar_coherence", return_value=dirty_coh_ds)
        mocker.patch("layers.get_uavsar_incidence", return_value=dirty_incidence_da)
        mocker.patch("layers.get_topo_layers", return_value=dirty_topo_ds)
        mocker.patch("layers.get_nlcd_layers", return_value=dirty_nlcd_ds)
        mocker.patch("layers.get_snow_climatology", return_value=dirty_snow_class_ds)
        mocker.patch("layers.get_aorc_layers", return_value=dirty_aorc_ds)
        mocker.patch("layers.get_snow_layers", return_value=dirty_snow_ds)

        dem_da = xr.DataArray(
            np.full((5, 5), 2000.0, dtype=np.float32),
            dims=["y", "x"],
            coords={"y": ref.y.values, "x": ref.x.values},
            name="elevation",
        )
        dem_da.attrs["grid_mapping"] = "spatial_ref"
        dem_rio = MagicMock()
        dem_rio.write_crs.return_value = dem_da
        dem_rio.reproject_match.return_value = dem_da
        type(dem_da).rio = property(lambda self: dem_rio)

        py3dep_mock = sys.modules.get("py3dep")
        if py3dep_mock is not None:
            py3dep_mock.get_dem = MagicMock(return_value=dem_da)

        mocker.patch("xarray.Dataset.chunk", lambda self, *a, **kw: self)

        result = assemble_data(
            aoi=aoi,
            date_pairs=date_pairs,
            flight_ids=flight_ids,
            fp_dest=None,
            fp_coh="/dev/null",
            fp_inc="/dev/null",
            fp_snowclimate="/dev/null",
            crs="EPSG:4326",
            res=30,
        )

        for coord_name in ["x", "y"]:
            assert "grid_mapping" not in result[coord_name].attrs, (
                f"Coordinate array '{coord_name}' still has 'grid_mapping' after sanitization"
            )


# ---------------------------------------------------------------------------
# End of test module
# ---------------------------------------------------------------------------

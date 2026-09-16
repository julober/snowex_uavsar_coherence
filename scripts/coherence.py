import xarray as xr
import rioxarray as rxa
import rasterio
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter, uniform_filter
import math

def calc_coherence(
    slc1, 
    slc2, 
    filter_type='uniform', 
    window_size=(13, 13), 
    sigma=2
):
    """
    Calculates unweighted SAR coherence with an option for uniform or Gaussian filtering.
    
    Parameters:
    -----------
    slc1, slc2 : numpy.ndarray
        The two co-registered complex Single Look Complex (SLC) images.
    filter_type : str
        'uniform' for a boxcar average, or 'gaussian' for a Gaussian weighted average.
    window_size : int or tuple of ints
        The size of the moving average window for the uniform filter.
    sigma : scalar or sequence of scalars
        Standard deviation for the Gaussian kernel.
        
    Returns:
    --------
    coherence : numpy.ndarray
        The calculated coherence magnitude map (values ranging from 0 to 1).
    """
    # --- 1. Input Parsing Helper ---
    def _parse_input(data):
        """Standardizes input into a 2D numpy array."""
        if isinstance(data, (str, Path)):
            # Load file, squeeze out any extra band dimensions, and grab numpy values
            return rxa.open_rasterio(data).squeeze().values
        elif isinstance(data, xr.DataArray):
            # Extract numpy values and squeeze
            return data.squeeze().values
        elif isinstance(data, np.ndarray):
            # Just ensure there are no dummy dimensions
            return np.squeeze(data)
        else:
            raise TypeError(f"Unsupported input type: {type(data)}. Must be path, xarray, or numpy array.")

    # Parse both inputs into raw 2D numpy arrays
    arr1 = _parse_input(slc1)
    arr2 = _parse_input(slc2)

    # Sanity check to ensure images are perfectly co-registered / same size
    if arr1.shape != arr2.shape:
        raise ValueError(f"Shape mismatch: slc1 is {arr1.shape}, but slc2 is {arr2.shape}. Images must be the exact same dimensions.")
    if filter_type not in ['uniform', 'gaussian']:
        raise ValueError("filter_type must be either 'uniform' or 'gaussian'")

    # numerator
    cross_product = arr1 * np.conj(arr2)
    # denominator
    int1 = np.abs(arr1)**2
    int2 = np.abs(arr2)**2
    
    def apply_filter(data):
        if filter_type == 'uniform':
            return uniform_filter(data, size=window_size)
        elif filter_type == 'gaussian':
            return gaussian_filter(data, sigma=sigma)
            
    # moving average 
    cross_avg = apply_filter(cross_product.real) + 1j * apply_filter(cross_product.imag)
    int1_avg = apply_filter(int1)
    int2_avg = apply_filter(int2)
    
    # --- 3. Compute Final Coherence ---
    # Coherence = |<S1 * S2*>| / sqrt(<|S1|^2> * <|S2|^2>)
    epsilon = 1e-10 
    denominator = np.sqrt(int1_avg * int2_avg) + epsilon
    coherence_mag = np.abs(cross_avg) / denominator
    coherence_mag = np.clip(coherence_mag, 0.0, 1.0)

    fill_value = -9999 + -9999j
    # print(arr1, arr2)
    mask1 = np.isnan(arr1) | np.isclose(arr1, fill_value)
    mask2 = np.isnan(arr2) | np.isclose(arr2, fill_value)
    # print(mask1, mask2)
    nan_mask = mask1 | mask2
    
    coherence_mag[nan_mask] = np.nan
    
    return coherence_mag

def calc_coherence_matrix(coherences, 
                          num_scenes, 
                          method='mean', 
                          threshold=0.3):
    # Accept list or 3D numpy array
    if isinstance(coherences, list):
        arrs = coherences
    elif isinstance(coherences, np.ndarray) and coherences.ndim == 3:
        arrs = [coherences[i] for i in range(coherences.shape[0])]
    else:
        raise ValueError("coherences must be a list or 3D numpy array")

    if math.comb(num_scenes, 2) != len(arrs):
        raise ValueError("Number of coherence arrays does not match number of scenes")

    mtx = np.zeros([num_scenes, num_scenes])
    counter = 0
    for i in range(num_scenes):
        for j in range(i, num_scenes):
            if i == j:
                mtx[i, j] = 1
                continue
            arr_vals = arrs[counter]
            if method == 'mean':
                mtx[i, j] = np.nanmean(arr_vals)
            elif method == 'prop':
                mtx[i, j] = np.sum(arr_vals > threshold) / arr_vals.size
            counter += 1

    mtx = np.where(mtx == 0, np.nan, mtx)
    return mtx



def calculate_coherence(
    file1_path: Path,
    file2_path: Path,
    out_path: Path,
    window_size: tuple[int, int] = (5, 5),
    chunk_size: int = 2048,
) -> None:
    """
    Compute SAR coherence between two geocoded complex TIF files and save the result.

    Designed for large files (5–20 GB): data is never fully loaded into RAM.
    Each file is opened as a chunked Dask-backed DataArray via rioxarray, all
    intermediate operations build a lazy computation graph, and the output is
    written chunk-by-chunk with rioxarray's windowed writer.  Peak memory is
    proportional to ``chunk_size`` rather than file size.

    The moving-average filter is applied with ``xarray.DataArray.rolling``,
    which handles chunk boundaries correctly without any explicit halo padding
    (unlike ``scipy.ndimage.uniform_filter`` applied naively to Dask arrays).

    Parameters
    ----------
    file1_path : Path
        Path to the first input geocoded complex TIF file.
    file2_path : Path
        Path to the second input geocoded complex TIF file.
    out_path : Path
        Path where the output coherence TIF will be written.  Parent
        directories are created if they do not exist.
    window_size : tuple[int, int]
        Rectangular averaging window as ``(row_window_size, col_window_size)``.
    chunk_size : int
        Spatial chunk size in pixels along both x and y.  At complex64 dtype a
        ``chunk_size=2048`` chunk is ~32 MB; two inputs plus intermediates
        stay well under 1 GB per core.  Increase for fewer, larger tasks;
        decrease if per-core memory is constrained.  Should be substantially
        larger than the largest ``window_size`` dimension.
    """
    row_win, col_win = window_size

    # ------------------------------------------------------------------
    # 1. Open files lazily — no pixel data is read yet
    # ------------------------------------------------------------------
    slc1 = rxa.open_rasterio(
        file1_path, chunks={"x": chunk_size, "y": chunk_size}
    ).squeeze()
    slc2 = rxa.open_rasterio(
        file2_path, chunks={"x": chunk_size, "y": chunk_size}
    ).squeeze()

    if slc1.shape != slc2.shape:
        raise ValueError(
            f"Shape mismatch: {file1_path.name} is {slc1.shape}, "
            f"but {file2_path.name} is {slc2.shape}. "
            "Images must have identical dimensions."
        )

    # ------------------------------------------------------------------
    # 2. Nodata mask — lazy elementwise operations
    # ------------------------------------------------------------------
    # Use apply_ufunc so np.isnan dispatches correctly through Dask
    _FILL = -9999.0
    nan1 = xr.apply_ufunc(np.isnan, slc1, dask="parallelized", output_dtypes=[bool])
    nan2 = xr.apply_ufunc(np.isnan, slc2, dask="parallelized", output_dtypes=[bool])
    # Check real and imaginary parts separately to avoid complex isclose issues
    fill1 = (abs(slc1.real - _FILL) < 0.5) & (abs(slc1.imag - _FILL) < 0.5)
    fill2 = (abs(slc2.real - _FILL) < 0.5) & (abs(slc2.imag - _FILL) < 0.5)
    nan_mask = nan1 | nan2 | fill1 | fill2

    # ------------------------------------------------------------------
    # 3. Cross product and intensities — lazy elementwise
    #    slc1 * conj(slc2) = (a+bi)(c-di) = (ac+bd) + (bc-ad)i
    # ------------------------------------------------------------------
    cross_real = slc1.real * slc2.real + slc1.imag * slc2.imag
    cross_imag = slc1.imag * slc2.real - slc1.real * slc2.imag
    int1 = slc1.real ** 2 + slc1.imag ** 2
    int2 = slc2.real ** 2 + slc2.imag ** 2

    # ------------------------------------------------------------------
    # 4. Moving-average filter — lazy, chunk-boundary-safe
    #
    #    xr.DataArray.rolling handles the halo problem internally: values
    #    near chunk edges draw on the correct neighboring pixels before
    #    averaging, so the result is identical to a full-array uniform
    #    filter.  min_periods=1 mirrors scipy's default edge behaviour
    #    (use available pixels rather than producing NaN at boundaries).
    # ------------------------------------------------------------------
    def _roll(da: xr.DataArray) -> xr.DataArray:
        return da.rolling(y=row_win, x=col_win, center=True, min_periods=1).mean()

    cross_real_avg = _roll(cross_real)
    cross_imag_avg = _roll(cross_imag)
    int1_avg = _roll(int1)
    int2_avg = _roll(int2)

    # ------------------------------------------------------------------
    # 5. Coherence magnitude — lazy elementwise
    #    |<S1 S2*>| / sqrt(<|S1|^2> <|S2|^2>)
    # ------------------------------------------------------------------
    epsilon = 1e-10
    numerator = np.sqrt(cross_real_avg ** 2 + cross_imag_avg ** 2)
    denominator = np.sqrt(int1_avg * int2_avg) + epsilon
    coherence_mag = (numerator / denominator).clip(0.0, 1.0)

    # ------------------------------------------------------------------
    # 6. Apply nodata mask — lazy
    # ------------------------------------------------------------------
    coherence_mag = coherence_mag.where(~nan_mask)

    # ------------------------------------------------------------------
    # 7. Write output chunk-by-chunk — this is where computation happens.
    #    windowed=True tells rioxarray to materialise and write one chunk
    #    at a time, so peak RAM stays proportional to chunk_size.
    #    tiled=True + compress='deflate' produce a Cloud-Optimised GeoTIFF
    #    that supports efficient windowed reads in downstream steps.
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (
        coherence_mag
        .astype("float32")
        .rio.write_nodata(np.nan, inplace=True)
        .rio.to_raster(
            out_path,
            tiled=True,
            windowed=True,
            compress="deflate",
        )
    )

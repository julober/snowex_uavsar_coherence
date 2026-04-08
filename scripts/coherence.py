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
    window_size: int = 5,
) -> None:
    """
    Compute SAR coherence between two geocoded complex TIF files and save the result.

    Parameters
    ----------
    file1_path : Path
        Path to the first input complex TIF file.
    file2_path : Path
        Path to the second input complex TIF file.
    out_path : Path
        Path where the output coherence TIF will be written.
    window_size : int
        Side length (in pixels) of the square uniform averaging window.
    """
    coherence_mag = calc_coherence(file1_path, file2_path, window_size=(window_size, window_size))

    with rasterio.open(file1_path) as src:
        profile = src.profile.copy()

    profile.update(dtype="float32", count=1, nodata=np.nan)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(coherence_mag.astype(np.float32), 1)
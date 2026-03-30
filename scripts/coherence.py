import xarray as xr
import rioxarray as rxa
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
import math

def calc_coherence_unweighted(slc1, slc2, window_size=(5, 5)):
    """
    Calculates unweighted SAR coherence by computing pixel-wise products first, 
    then applying a spatial averaging window.
    
    Parameters:
    -----------
    slc1, slc2 : numpy.ndarray
        The two co-registered complex Single Look Complex (SLC) images.
    window_size : int or tuple of ints
        The size of the moving average window (looks_row, looks_col). 
        
    Returns:
    --------
    coherence : numpy.ndarray
        The calculated coherence magnitude map (values ranging from 0 to 1).
    """
    
    # --- 1. The "Dot Product" (Pixel-by-pixel multiplication) ---
    # Multiply slc1 by the complex conjugate of slc2
    cross_product = slc1 * np.conj(slc2)
    
    # Calculate intensities (power) for both images
    int1 = np.abs(slc1)**2
    int2 = np.abs(slc2)**2
    
    # --- 2. Apply the Averaging Window (Multi-looking) ---
    # gaussian_filter calculates the mean inside the moving window.
    cross_avg_real = gaussian_filter(cross_product.real, size=window_size)
    cross_avg_imag = gaussian_filter(cross_product.imag, size=window_size)
    cross_avg = cross_avg_real + 1j * cross_avg_imag
    
    int1_avg = gaussian_filter(int1, size=window_size)
    int2_avg = gaussian_filter(int2, size=window_size)
    
    # --- 3. Compute Final Coherence ---
    # Coherence = |<S1 * S2*>| / sqrt(<|S1|^2> * <|S2|^2>)
    
    # Add a tiny epsilon to the denominator to prevent division-by-zero 
    # in radar shadow regions where intensity drops to 0.
    epsilon = 1e-10 
    denominator = np.sqrt(int1_avg * int2_avg) + epsilon
    
    coherence_mag = np.abs(cross_avg) / denominator
    coherence_mag = np.clip(coherence_mag, 0.0, 1.0)
    
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
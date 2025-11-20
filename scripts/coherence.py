import numpy as np 
import xarray as xr
from scipy.signal import convolve2d 
import math

# Calculate the coherence magnitude over 13x13 pixels 
# img1 and img2 should be the same size
def calc_coherence_unweighted(img1, img2, window=13) :

    # create window over which to do the magnitude calculation 
    kernel = np.ones((window, window), dtype=np.float32)
    kernel /= kernel.sum()

    if (type(img1) != xr.DataArray) or (type(img2) != xr.DataArray):
        print("Input is not Xarray")

    # Numerator: E[u1 * conj(u2)] a.k.a. the cross product of the two complex numbers
    cross_prod = img1 * np.conj(img2.values)

    # print(type(cross_prod))

    # convolve2d does the leg work of moving the window over all pixels. 
    # the 'symm' boundary creates a mirror reflection to fill in areas for pixels at the edges
    num_real = convolve2d(cross_prod.real, kernel, mode='same', boundary='symm')
    num_imag = convolve2d(cross_prod.imag, kernel, mode='same', boundary='symm')
    # put the separate components back together 
    numerator = num_real + 1j * num_imag

    # Denominator: sqrt(E[|u1|^2] * E[|u2|^2])
    power1 = convolve2d(np.abs(img1.values)**2, kernel, mode='same', boundary='symm')
    power2 = convolve2d(np.abs(img2.values)**2, kernel, mode='same', boundary='symm')
    denom = np.sqrt(power1 * power2)
    denom[denom == 0] = 1e-12  # Prevent divide by zero

    # Coherence magnitude
    coherence = np.abs(numerator / denom)
    return img1.copy(data=coherence)

def calc_coherence_matrix(coherences,
                          num_scenes,
                          method = 'mean', 
                          threshold = 0.3) : 
    """
    Calculates a coherence matrix from a list of coherence arrays.

    Parameters: 
    - coherences: list of 2D numpy arrays representing coherence of interferograms
    - method: 'mean' to use mean coherence, 'prop' to use proportion above threshold. default is 'mean'
    - threshold: threshold value for 'prop' method (optional). default is 0.3
    """

    if (method not in ['mean', 'prop']) :
        raise ValueError("Method must be 'mean' or 'prop'")
    if (math.comb(num_scenes, 2) != len(coherences)) :
        raise ValueError("Number of coherence arrays does not match number of scenes")

    mtx = np.zeros([num_scenes, num_scenes])
    counter = 0
    for i in range(num_scenes) : 
        for j in range(i, num_scenes) : 
            if (i == j) : 
                mtx[i, j] = 1
                continue
            if (method == 'mean') :
                mtx[i, j] = np.mean(coherences[counter])
            elif (method == 'prop') :
                mtx[i, j] = np.sum(np.where(coherences[counter] > threshold, 1, 0)) \
                                   / coherences[counter].size
            counter += 1

    # set non-needed values to nan
    return np.where(mtx == 0, np.nan, mtx)
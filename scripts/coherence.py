import numpy as np 
from scipy.signal import convolve2d 

# Calculate the coherence magnitude over 13x13 pixels 
# img1 and img2 should be the same size
def calc_coherence_unweighted(img1, img2, window=13) :

    # create window over which to do the magnitude calculation 
    kernel = np.ones((window, window), dtype=np.float32)
    kernel /= kernel.sum()

    # Numerator: E[u1 * conj(u2)] a.k.a. the cross product of the two complex numbers
    cross_prod = img1 * np.conj(img2)

    # convolve2d does the leg work of moving the window over all pixels. 
    # the 'symm' boundary creates a mirror reflection to fill in areas for pixels at the edges
    num_real = convolve2d(cross_prod.real, kernel, mode='same', boundary='symm')
    num_imag = convolve2d(cross_prod.imag, kernel, mode='same', boundary='symm')
    # put the separate components back together 
    numerator = num_real + 1j * num_imag

    # Denominator: sqrt(E[|u1|^2] * E[|u2|^2])
    power1 = convolve2d(np.abs(img1)**2, kernel, mode='same', boundary='symm')
    power2 = convolve2d(np.abs(img2)**2, kernel, mode='same', boundary='symm')
    denom = np.sqrt(power1 * power2)
    denom[denom == 0] = 1e-12  # Prevent divide by zero

    # Coherence magnitude
    coherence = np.abs(numerator / denom)
    return coherence


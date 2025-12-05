## COPIED FROM https://github.com/ZachHoppinen/uavsar-coherence/

import numpy as np
from scipy.optimize import curve_fit

# Calculate tau and gamma over a 2D grid. Returns grid of fitted parameters and success flags. 
# coh_arr: 3D numpy array (n_coh_pairs, width, height)
# days: 1D numpy array (n_coh_pairs,)
# tau_guess: initial guess for tau
# bounds: bounds for tau 
# xtol, ftol: tolerances for curve fitting
# gamma_inf: if provided, fit with fixed gamma_inf value
def get_taus_spatial(coh_arr, 
                     days, 
                     tau_guess=6, 
                     bounds=(0,30), 
                     xtol=1e-6, 
                     ftol=1e-6, 
                     gamma_inf = 0.17):
    n, width, height = coh_arr.shape


    # fit the gamma infinity parameter if it is set to None
    #if gamma_inf is None:
    tau_grid = np.zeros(shape=(width, height))
    success_grid = np.zeros(shape=(width, height), dtype=bool)
    if gamma_inf is None : gamma_grid = np.zeros(shape=(width, height))

    # calculate tau for each pixel 
    for w in range(width) : 
        for h in range(height) : 
            if np.any(np.isnan(coh_arr[:, w, h])): # skip over any nans - e.g., corner areas
                tau_grid[w,h] = np.nan
                success_grid[w,h] = False
                if gamma_inf is None: gamma_grid[w,h] = np.nan 
                continue
            cohs = coh_arr[:, w, h]
            try : 
                if gamma_inf is None: # fit tau and gamma
                    gamma_inf_fit, tau, pcov = fit_coh_decay_model(cohs, days, tau_guess, bounds, xtol, ftol, gamma_inf = None)
                else : # use a fixed gamma_inf
                    tau, pcov = fit_coh_decay_model(cohs, days, tau_guess, bounds, xtol, ftol, gamma_inf)
                success_grid[w, h] = True
            except Exception : # report a failed fit
                success_grid[w, h] = False
                continue
            if gamma_inf is None: gamma_grid[w,h] = gamma_inf_fit
            tau_grid[w,h] = tau

    # return values vary based on whether gamma_inf was fit or fixed
    if gamma_inf is None : return tau_grid, gamma_grid, success_grid
    else : return tau_grid, success_grid

# Fit coherence decay model to data. Returns fitted parameters and covariances.
# cohs: 1D array of coherence values
# days: 1D array of time differences
# tau_guess: initial guess for tau
# bounds: bounds for tau
# xtol, ftol: tolerances for curve fitting
# gamma_inf: if provided, fit with fixed gamma_inf value
def fit_coh_decay_model(cohs, days, tau_guess, bounds, xtol, ftol, gamma_inf = None):
    if gamma_inf is not None:
        params, pcov = curve_fit(lambda t, tau: gamma_inf + (1 - gamma_inf) * np.exp(- t / tau), days, cohs, p0=(tau_guess),
            bounds = bounds, ftol = ftol, xtol = xtol)
        tau = params
        return tau, pcov
    else : 
        # Initial guesses
        gamma_inf_guess = 0.3
        params, pcov = curve_fit(lambda t, gamma_inf, tau: gamma_inf + (1 - gamma_inf) * np.exp(- t / tau), days, cohs, p0=(gamma_inf_guess, tau_guess),\
            bounds = bounds, ftol = ftol, xtol = xtol)
        gamma_inf, tau = params
        return gamma_inf, tau, pcov

# Decorrelation temporal model. Returns coherence based on time difference and model parameters.
# t: time difference
# gamma_inf: asymptotic coherence value
# tau: decorrelation time constant
def decorrelation_temporal_model(t, gamma_inf, tau):
    coherence =  gamma_inf + (1 - gamma_inf) * np.exp(- t / tau)
    return coherence
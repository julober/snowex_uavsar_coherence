## COPIED FROM https://github.com/ZachHoppinen/uavsar-coherence/

import numpy as np
from scipy.optimize import curve_fit

# Calculate tau and gamma over a 2D grid
# coh_arr: 3D numpy array (n_coh_pairs, width, height)
# days: 1D numpy array (n_coh_pairs,)
# tau_guess: initial guess for tau
# bounds: bounds for tau fitting
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
    for w in range(width) : 
        for h in range(height) : 
            if np.any(np.isnan(coh_arr[:, w, h])): 
                # print("found a nan, skipping")
                tau_grid[:,w,h] = np.nan
                continue
            cohs = coh_arr[:, w, h]
            try : 
                if gamma_inf is None:
                    gamma_inf_fit, tau, pcov = fit_coh_decay_model(cohs, days, tau_guess, bounds, xtol, ftol, gamma_inf = None)
                else :
                    tau, pcov = fit_coh_decay_model(cohs, days, tau_guess, bounds, xtol, ftol, gamma_inf)
                success_grid[w, h] = True
            except Exception :
                success_grid[w, h] = False
                continue
            if gamma_inf is None: gamma_grid[w,h] = gamma_inf_fit
            tau_grid[w,h] = tau
    if gamma_inf is None : return tau_grid, gamma_grid, success_grid
    else : return tau_grid, success_grid
    # else : # fit with fixed gamma_inf
    #     tau_grid = np.zeros(shape=(width, height))
    #     for w in range(width) : 
    #         for h in range(height) : 
    #             if np.any(np.isnan(coh_arr[:, w, h])): 
    #                 # print("found a nan, skipping")
    #                 tau_grid[w,h] = np.nan
    #                 continue
    #             cohs = coh_arr[:, w, h]
    #             tau, pcov = fit_coh_decay_model(cohs, days, tau_guess, bounds, xtol, ftol, gamma_inf)
    #             tau_grid[w,h] = tau
    #     return tau_grid

def fit_coh_decay_model(cohs, days, tau_guess, bounds, xtol, ftol, gamma_inf = None):
    # https://rowannicholls.github.io/python/curve_fitting/exponential.html

    if gamma_inf is not None:
        params, pcov = curve_fit(lambda t, tau: gamma_inf + (1 - gamma_inf) * np.exp(- t / tau), days, cohs, p0=(tau_guess),\
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
    # Fit the function a * np.exp(b * t) + c to x and y

    # gamma_inf, tau = params

    # return gamma_inf, tau, pcov

def decorrelation_temporal_model(t, gamma_inf, tau):
    coherence =  gamma_inf + (1 - gamma_inf) * np.exp(- t / tau)
    return coherence
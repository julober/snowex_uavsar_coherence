## COPIED FROM https://github.com/ZachHoppinen/uavsar-coherence/

import numpy as np
from scipy.optimize import curve_fit

def fit_coh_decay_model(cohs, days, tau_guess, bounds, xtol, ftol, gamma_inf_guess = 0.3):
    # https://rowannicholls.github.io/python/curve_fitting/exponential.html

    # Fit the function a * np.exp(b * t) + c to x and y
    params, pcov = curve_fit(lambda t, gamma_inf, tau: gamma_inf + (1 - gamma_inf) * np.exp(- t / tau), days, cohs, p0=(gamma_inf_guess, tau_guess),\
        bounds = bounds, ftol = ftol, xtol = xtol)

    gamma_inf, tau = params

    return gamma_inf, tau, pcov

def decorrelation_temporal_model(t, gamma_inf, tau):
    coherence =  gamma_inf + (1 - gamma_inf) * np.exp(- t / tau)
    return coherence
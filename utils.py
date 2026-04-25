import numpy as np
from scipy.integrate import trapezoid

def L_inf_error(x_num, x_ref):
    """Maximum absolute error"""
    return np.max(np.abs(x_num - x_ref))

def L2_error(x_num, x_ref, t):
    """L2 norm of error"""
    return np.sqrt(trapezoid((x_num - x_ref)**2, t))

def convergence_rates(errors, N_vals):
    """Estimate convergence rate from log-log slope"""
    logN = np.log(N_vals)
    logE = np.log(errors)
    slope = np.polyfit(logN, logE, 1)[0]
    return -slope  # positive rate

import numpy as np

def p_t(t):
    """Policy coefficient p(t) = 1 + 0.1*t"""
    return 1.0 + 0.1 * t

def g_func(t):
    """Baseline business cycle component"""
    return 0.05 * np.sin(2.0 * np.pi * t)

def delta_radius(xprime, D05):
    """State-dependent uncertainty radius"""
    return 0.02 * (1.0 + np.abs(xprime) + np.abs(D05))

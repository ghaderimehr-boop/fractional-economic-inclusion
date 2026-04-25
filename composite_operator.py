import numpy as np
from .l1_derivative import l1_caputo_derivative

def composite_fractional_operator(x, t, sigma, omega, lam, p_func):
    """
    Compute D^σ( (1/p(t)) D^ω x + λ x )
    
    Parameters:
    x : array, x(t) values
    t : array, time grid
    sigma, omega : fractional orders
    lam : float, λ
    p_func : callable, p(t)
    
    Returns:
    result : array, LHS
    """
    # Inner D^ω x
    Domega_x = l1_caputo_derivative(x, t, omega)
    
    # (1/p(t)) * Dωx + λ*x
    inner = np.zeros_like(x)
    for i, ti in enumerate(t):
        inner[i] = Domega_x[i] / p_func(ti) + lam * x[i]
    
    # Outer D^σ (inner)
    result = l1_caputo_derivative(inner, t, sigma)
    return result

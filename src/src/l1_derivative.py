import numpy as np
from scipy.special import gamma

def l1_caputo_derivative(y, t, alpha):
    """
    Caputo derivative of order alpha (0<alpha<1) using L1 scheme.
    
    Parameters:
    y : array, function values at grid points (length n+1)
    t : array, uniform time grid
    alpha : float, fractional order in (0,1)
    
    Returns:
    D : array (same length as y), derivative (D[0]=0)
    """
    n = len(y) - 1
    h = t[1] - t[0]
    D = np.zeros(n+1)
    
    # Precompute weights
    w = np.zeros(n+2)
    for k in range(1, n+1):
        w[k] = (k+1)**(1-alpha) - 2*k**(1-alpha) + (k-1)**(1-alpha)
    
    for i in range(1, n+1):
        s = 0.0
        for k in range(1, i):
            s += w[k] * y[i - k]
        D[i] = (h**(-alpha) / gamma(2-alpha)) * (y[i] - y[i-1] + s)
    
    return D

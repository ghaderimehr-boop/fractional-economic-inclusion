#!/usr/bin/env python3
# complete_economic_analysis_final.py
# Final version with separated plots and clean styling

import numpy as np
import math
from scipy.special import gamma
from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import time

# Set global style for beautiful plots
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 3.0
plt.rcParams['lines.markersize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['legend.framealpha'] = 0.9
plt.rcParams['legend.fancybox'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.grid'] = True

# ---------------------------
# User parameters
# ---------------------------
T = 1.0
N = 400
h = T / N
t = np.linspace(0.0, T, N+1)

sigma = 0.7
omega = 1.8
inner_half = 0.5
lambda_ = 0.05
eta = 0.5

eps = 1e-14
clip_val = 1e12
fp_max_iter = 30
fp_tol = 1e-8

# ---------------------------
# Enhanced Color Scheme
# ---------------------------
colors = {
    'pess': '#E74C3C',  # Vibrant red
    'base': '#3498DB',  # Vibrant blue  
    'opt': '#2ECC71',   # Vibrant green
    'uncertainty': '#9B59B6',  # Purple
    'sensitivity': '#F39C12'   # Orange
}

markers = {
    'pess': 'o',
    'base': 's', 
    'opt': '^',
    'convergence': 'D'
}

# ---------------------------
# Model functions
# ---------------------------
def p_t(t_arr):
    return 1.0 + 0.1 * t_arr

def g_func(t_arr):
    return 0.05 * np.sin(2.0 * np.pi * t_arr)

def delta_radius(xprime_arr, D05_arr):
    return 0.02 * (1.0 + np.abs(xprime_arr) + np.abs(D05_arr))

# ---------------------------
# L1 weights precomputation
# ---------------------------
def L1_weights(alpha, M):
    m = np.arange(1, M+1, dtype=float)
    a = np.maximum(m, eps)
    b = np.maximum(m-1.0, eps)
    w = a**(1.0 - alpha) - b**(1.0 - alpha)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    return w

# ---------------------------
# Caputo L1 operator
# ---------------------------
def caputo_L1(alpha, x, h_local):
    n = len(x)
    if n <= 1:
        return np.zeros(n)
    d = np.diff(x)
    M = len(d)
    if M == 0:
        return np.zeros(n)
    if alpha <= 1.0:
        w = L1_weights(alpha, M)
        conv = fftconvolve(d, w, mode='full')[:M]
        D = np.zeros(n)
        factor = 1.0 / (h_local**alpha * gamma(2.0 - alpha))
        D[1:] = conv * factor
        D = np.nan_to_num(D, nan=0.0, posinf=clip_val, neginf=-clip_val)
        return D
    else:
        beta = alpha - 1.0
        dx = np.zeros_like(x)
        dx[0] = (x[1] - x[0]) / h_local
        dx[-1] = (x[-1] - x[-2]) / h_local
        if n > 2:
            dx[1:-1] = (x[2:] - x[:-2]) / (2.0 * h_local)
        ddx = np.diff(dx)
        Mdx = len(ddx)
        if Mdx == 0:
            return np.zeros(n)
        w = L1_weights(beta, Mdx)
        conv = fftconvolve(ddx, w, mode='full')[:Mdx]
        D = np.zeros(n)
        factor = 1.0 / (h_local**beta * gamma(2.0 - beta))
        D[1:] = conv * factor
        D = np.nan_to_num(D, nan=0.0, posinf=clip_val, neginf=-clip_val)
        return D

# ---------------------------
# Build discrete Caputo matrix
# ---------------------------
def build_caputo_matrix(alpha, N_local, h_local):
    mat = np.zeros((N_local+1, N_local+1))
    for j in range(N_local+1):
        e = np.zeros(N_local+1)
        e[j] = 1.0
        D = caputo_L1(alpha, e, h_local)
        mat[:, j] = D
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat

# ---------------------------
# Assemble composite operator
# ---------------------------
def assemble_composite_matrix(C_sigma, B_omega, pvec, lam):
    invp = 1.0 / np.maximum(pvec, 1e-12)
    M = (invp[:, None] * B_omega) + lam * np.eye(B_omega.shape[0])
    A = C_sigma @ M
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    return A

# ---------------------------
# Solve linear system with slope
# ---------------------------
def solve_linear_with_slope(A, f_vec, s, h_local):
    Np = A.shape[0]
    if Np < 3:
        x = np.zeros(Np)
        x[0] = 0.0
        if Np > 1:
            x[1] = s * h_local
        return x
    known = np.zeros(Np)
    known[0] = 0.0
    known[1] = s * h_local
    unknown_idx = np.arange(2, Np)
    A_rr = A[np.ix_(unknown_idx, unknown_idx)]
    A_rk = A[np.ix_(unknown_idx, [0,1])]
    f_r = f_vec[unknown_idx]
    rhs = f_r - A_rk @ known[[0,1]]
    x_r = np.linalg.solve(A_rr, rhs)
    x = np.zeros(Np)
    x[0] = known[0]
    x[1] = known[1]
    x[2:] = x_r
    x = np.nan_to_num(x, nan=0.0, posinf=clip_val, neginf=-clip_val)
    return x

# ---------------------------
# Optimize slope for boundary condition
# ---------------------------
def find_s_for_f(A, f_vec, h_local, t_grid, eta_local):
    k = int(round(eta_local / (t_grid[1] - t_grid[0])))
    if k < 1:
        k = 1
    def obj(s):
        x_candidate = solve_linear_with_slope(A, f_vec, s, h_local)
        integral = trapezoid(x_candidate[:k+1], t_grid[:k+1])
        return abs(s - integral)
    res = minimize_scalar(obj, bounds=(-10.0, 10.0), method='bounded', options={'xatol':1e-8})
    return res.x, res.fun

# ---------------------------
# Fixed-point iteration
# ---------------------------
def run_fixed_point_selection(selection_name, selection_type, A, t_grid, h_local, eta_local,
                              max_fp_iter=fp_max_iter, tol=fp_tol, verbose=True):
    Np = A.shape[0]
    f_vec = g_func(t_grid).copy()
    x_old = np.zeros(Np)
    s_old = 0.0

    for it in range(max_fp_iter):
        s_star, objval = find_s_for_f(A, f_vec, h_local, t_grid, eta_local)
        x_new = solve_linear_with_slope(A, f_vec, s_star, h_local)

        D05 = caputo_L1(inner_half, x_new, h_local)
        xprime = np.zeros_like(x_new)
        xprime[0] = (x_new[1] - x_new[0]) / h_local
        xprime[1:-1] = (x_new[2:] - x_new[:-2]) / (2.0 * h_local)
        xprime[-1] = (x_new[-1] - x_new[-2]) / h_local

        if selection_type == 'pess':
            f_new = g_func(t_grid) - delta_radius(xprime, D05)
        elif selection_type == 'base':
            f_new = g_func(t_grid)
        elif selection_type == 'opt':
            f_new = g_func(t_grid) + delta_radius(xprime, D05)
        else:
            raise ValueError("Unknown selection type")

        diff_norm = np.linalg.norm(x_new - x_old, ord=np.inf)
        f_norm = np.linalg.norm(f_new - f_vec, ord=np.inf)
        if verbose:
            print(f"[{selection_name}] FP iter {it:2d}: ||dx||_inf={diff_norm:.3e}, ||df||_inf={f_norm:.3e}, slope={s_star:.6e}, obj={objval:.3e}")
        
        x_old = x_new.copy()
        f_vec = f_new.copy()
        s_old = s_star
        
        if diff_norm < tol and f_norm < tol:
            if verbose:
                print(f"[{selection_name}] Fixed-point converged in {it+1} iters.")
            break
    else:
        if verbose:
            print(f"[{selection_name}] Fixed-point did NOT converge in {max_fp_iter} iters (last diff={diff_norm:.3e}).")

    return x_new, s_old, f_vec

# ---------------------------
# Separated Plotting Functions
# ---------------------------
def plot_economic_scenarios(results, t):
    """Plot 1: Economic scenarios solutions (separated)"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for typ, (x_sol, slope, f_final) in results.items():
        ax.plot(t, x_sol, color=colors[typ], 
                label=f'{typ.capitalize()} (Final: {x_sol[-1]:.3f})', 
                linewidth=3.5,
                alpha=0.9)
    
    ax.set_xlabel('Time $t$', fontsize=14, fontweight='bold')
    ax.set_ylabel('$x(t)$', fontsize=14, fontweight='bold')
    ax.set_title('Economic Scenarios Solutions', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper left', frameon=True, shadow=True)
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_absolute_differences(results, t):
    """Plot 2: Absolute differences from baseline (separated)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    base_solution = results['base'][0]
    
    ax.plot(t, np.abs(results['pess'][0] - base_solution), 
             color=colors['pess'], 
             label='|Pessimistic - Baseline|', 
             linewidth=3.0,
             alpha=0.9)
    
    ax.plot(t, np.abs(results['opt'][0] - base_solution), 
             color=colors['opt'], 
             label='|Optimistic - Baseline|', 
             linewidth=3.0,
             alpha=0.9)
    
    ax.set_xlabel('Time $t$', fontsize=14, fontweight='bold')
    ax.set_ylabel('Absolute Difference', fontsize=14, fontweight='bold')
    ax.set_title('Scenario Deviations from Baseline', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_convergence_norms(N_values, results_by_N):
    """Plot 3: Convergence norms (L∞ and L2 together)"""
    t_ref, ref_results = results_by_N[800]
    
    L_inf_errors = {'pess': [], 'base': [], 'opt': []}
    L2_errors = {'pess': [], 'base': [], 'opt': []}
    
    for N in N_values[:-1]:
        t_current, current_results = results_by_N[N]
        for scenario in ['pess', 'base', 'opt']:
            x_current_interp = np.interp(t_ref, t_current, current_results[scenario])
            error = np.abs(x_current_interp - ref_results[scenario])
            L_inf = np.max(error)
            L2 = np.sqrt(trapezoid(error**2, t_ref))
            L_inf_errors[scenario].append(L_inf)
            L2_errors[scenario].append(L2)
    
    # Convergence plots together
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for scenario in ['pess', 'base', 'opt']:
        # L∞ errors
        ax1.loglog(N_values[:-1], L_inf_errors[scenario], 
                  marker=markers[scenario], 
                  color=colors[scenario], 
                  label=f'{scenario.capitalize()}',
                  linewidth=2.5,
                  markersize=10,
                  markerfacecolor='white',
                  markeredgewidth=2,
                  markeredgecolor=colors[scenario])
        
        # L2 errors  
        ax2.loglog(N_values[:-1], L2_errors[scenario],
                  marker=markers[scenario], 
                  color=colors[scenario],
                  label=f'{scenario.capitalize()}',
                  linewidth=2.5,
                  markersize=10,
                  markerfacecolor='white',
                  markeredgewidth=2,
                  markeredgecolor=colors[scenario])
    
    ax1.set_xlabel('Number of Grid Points (N)', fontsize=13, fontweight='bold')
    ax1.set_ylabel(r'$L_\infty$ Error', fontsize=13, fontweight='bold')
    ax1.set_title('Convergence - L∞ Norm', fontsize=15, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2.set_xlabel('Number of Grid Points (N)', fontsize=13, fontweight='bold')
    ax2.set_ylabel(r'$L_2$ Error', fontsize=13, fontweight='bold')
    ax2.set_title('Convergence - L2 Norm', fontsize=15, fontweight='bold', pad=15)
    ax2.legend(fontsize=11, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return L_inf_errors, L2_errors

def plot_derivatives(results, t, h):
    """Plot 4: First derivatives and fractional derivatives together"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # First derivatives
    for typ, (x_sol, slope, f_final) in results.items():
        x_prime = np.gradient(x_sol, t)
        ax1.plot(t, x_prime, color=colors[typ], 
                label=f'{typ.capitalize()} (max: {np.max(np.abs(x_prime)):.3f})', 
                linewidth=3.0,
                alpha=0.9)
    
    ax1.set_xlabel('Time $t$', fontsize=13, fontweight='bold')
    ax1.set_ylabel("$x'(t)$", fontsize=13, fontweight='bold')
    ax1.set_title('First Derivatives', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Fractional derivatives
    for typ, (x_sol, slope, f_final) in results.items():
        D05 = caputo_L1(0.5, x_sol, h)
        ax2.plot(t, D05, color=colors[typ], 
                label=f'{typ.capitalize()} (max: {np.max(np.abs(D05)):.3f})', 
                linewidth=3.0,
                alpha=0.9)
    
    ax2.set_xlabel('Time $t$', fontsize=13, fontweight='bold')
    ax2.set_ylabel('$D^{0.5}x(t)$', fontsize=13, fontweight='bold')
    ax2.set_title('Fractional Derivatives (α=0.5)', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_sensitivity_analysis(results, t, h):
    """Plot 5: Sensitivity analysis (separated)"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sigma_values = [0.5, 0.7, 0.9]
    sensitivity_colors = ['#E67E22', '#3498DB', '#9B59B6']
    
    base_results = results['base']
    ax.plot(t, base_results[0], color=sensitivity_colors[1], 
            linewidth=3.5, label='Base (σ=0.7)', alpha=0.9)
    
    for i, sigma_val in enumerate(sigma_values):
        if sigma_val != 0.7:
            C_sigma_temp = build_caputo_matrix(sigma_val, N, h)
            B_omega_temp = build_caputo_matrix(omega, N, h)
            pvec_temp = p_t(t)
            A_temp = assemble_composite_matrix(C_sigma_temp, B_omega_temp, pvec_temp, lambda_)
            x_temp, _, _ = run_fixed_point_selection(
                f'Sigma={sigma_val}', 'base', A_temp, t, h, eta, 
                max_fp_iter=10, tol=1e-6, verbose=False)
            ax.plot(t, x_temp, '--', color=sensitivity_colors[i], 
                    linewidth=2.5, label=f'σ={sigma_val}', alpha=0.9)
    
    ax.set_xlabel('Time $t$', fontsize=14, fontweight='bold')
    ax.set_ylabel('$x(t)$', fontsize=14, fontweight='bold')
    ax.set_title('Sensitivity to Fractional Order σ', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_uncertainty_radius(results, t, h):
    """Plot 6: Uncertainty radius (separated)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    base_x = results['base'][0]
    x_prime_base = np.gradient(base_x, t)
    D05_base = caputo_L1(0.5, base_x, h)
    radius = delta_radius(x_prime_base, D05_base)
    
    ax.plot(t, radius, color=colors['uncertainty'], 
            linewidth=3.0, label='Uncertainty Radius', alpha=0.9)
    
    ax.set_xlabel('Time $t$', fontsize=14, fontweight='bold')
    ax.set_ylabel('Radius', fontsize=14, fontweight='bold')
    ax.set_title('Uncertainty Radius Evolution', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# ---------------------------
# Convergence Table
# ---------------------------
def create_convergence_table(N_values, L_inf_errors, L2_errors):
    """Create convergence table"""
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS TABLE")
    print("="*80)
    print(f"{'Scenario':<12} | {'N':<6} | {'L∞ Error':<12} | {'L2 Error':<12} | {'Rate L∞':<10} | {'Rate L2':<10}")
    print("-"*80)
    
    scenarios = ['pess', 'base', 'opt']
    scenario_names = {'pess': 'Pessimistic', 'base': 'Baseline', 'opt': 'Optimistic'}
    
    for scenario in scenarios:
        print(f"{scenario_names[scenario]:<12} | {'---':<6} | {'---':<12} | {'---':<12} | {'---':<10} | {'---':<10}")
        
        for i, N in enumerate(N_values[:-1]):
            L_inf = L_inf_errors[scenario][i]
            L2 = L2_errors[scenario][i]
            
            if i > 0:
                rate_L_inf = np.log(L_inf_errors[scenario][i-1] / L_inf) / np.log(2)
                rate_L2 = np.log(L2_errors[scenario][i-1] / L2) / np.log(2)
                rate_L_inf_str = f"{rate_L_inf:.3f}"
                rate_L2_str = f"{rate_L2:.3f}"
            else:
                rate_L_inf_str = "---"
                rate_L2_str = "---"
                
            print(f"{'':<12} | {N:<6} | {L_inf:<12.2e} | {L2:<12.2e} | {rate_L_inf_str:<10} | {rate_L2_str:<10}")

# ---------------------------
# Main execution with separated plots
# ---------------------------
def run_main_solution():
    """Run the main solution"""
    print("Building Caputo matrices...")
    t0 = time.perf_counter()
    
    C_sigma = build_caputo_matrix(sigma, N, h)
    B_omega = build_caputo_matrix(omega, N, h)
    t1 = time.perf_counter()
    print(f"Built matrices in {t1-t0:.2f}s")

    pvec = p_t(t)
    A = assemble_composite_matrix(C_sigma, B_omega, pvec, lambda_)
    build_time = time.perf_counter()
    print(f"Composite matrix assembled in {build_time - t1:.2f}s")

    scenarios = [('Pessimistic','pess'), ('Baseline','base'), ('Optimistic','opt')]
    results = {}
    for name, typ in scenarios:
        print(f"\n--- Running scenario: {name} ---")
        t_run = time.time()
        x_sol, slope_star, f_final = run_fixed_point_selection(name, typ, A, t, h, eta,
                                                               verbose=True)
        dt_run = time.time() - t_run
        print(f"Scenario {name} done in {dt_run:.2f}s; slope*={slope_star:.6e}")
        results[typ] = (x_sol, slope_star, f_final)

    return results, A

def run_convergence_analysis():
    """Run convergence analysis"""
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS")
    print("="*60)
    
    N_values = [100, 200, 400, 800]
    results_by_N = {}
    
    for i, N_val in enumerate(N_values):
        print(f"\n--- Running convergence test for N={N_val} ({i+1}/{len(N_values)}) ---")
        h_local = T / N_val
        t_local = np.linspace(0.0, T, N_val+1)
        
        C_sigma = build_caputo_matrix(sigma, N_val, h_local)
        B_omega = build_caputo_matrix(omega, N_val, h_local)
        pvec = p_t(t_local)
        A_local = assemble_composite_matrix(C_sigma, B_omega, pvec, lambda_)
        
        scenario_results = {}
        for name, typ in [('Pessimistic','pess'), ('Baseline','base'), ('Optimistic','opt')]:
            x_sol, slope_star, f_final = run_fixed_point_selection(
                name, typ, A_local, t_local, h_local, eta, 
                max_fp_iter=15, tol=1e-6, verbose=False)
            scenario_results[typ] = x_sol
        
        results_by_N[N_val] = (t_local, scenario_results)
    
    return N_values, results_by_N

# ---------------------------
# Final Main execution
# ---------------------------
if __name__ == "__main__":
    start_all = time.perf_counter()
    
    print("="*70)
    print("FRACTIONAL DIFFERENTIAL INCLUSION ANALYSIS")
    print("="*70)
    
    # Run main solution
    results, A = run_main_solution()
    
    # Create separated plots
    print("\nCreating separated plots...")
    
    # Plot 1: Economic scenarios (separated)
    print("1. Plotting economic scenarios...")
    plot_economic_scenarios(results, t)
    
    # Plot 2: Absolute differences (separated)
    print("2. Plotting absolute differences...")
    plot_absolute_differences(results, t)
    
    # Run convergence analysis and create plots
    N_values, results_by_N = run_convergence_analysis()
    
    # Plot 3: Convergence norms (together)
    print("3. Plotting convergence norms...")
    L_inf_errors, L2_errors = plot_convergence_norms(N_values, results_by_N)
    
    # Plot 4: Derivatives (together)
    print("4. Plotting derivatives...")
    plot_derivatives(results, t, h)
    
    # Plot 5: Sensitivity analysis (separated)
    print("5. Plotting sensitivity analysis...")
    plot_sensitivity_analysis(results, t, h)
    
    # Plot 6: Uncertainty radius (separated)
    print("6. Plotting uncertainty radius...")
    plot_uncertainty_radius(results, t, h)
    
    # Create convergence table
    create_convergence_table(N_values, L_inf_errors, L2_errors)
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print("{:<12} | {:<12} | {:<12} | {:<12}".format('Scenario', 'Final x(T)', 'Max |x\'|', 'Max |D^0.5x|'))
    print("-"*60)
    
    for typ, (x_sol, slope, f_final) in results.items():
        x_prime = np.gradient(x_sol, t)
        D05 = caputo_L1(0.5, x_sol, h)
        print(f"{typ.capitalize():<12} | {x_sol[-1]:<12.4f} | {np.max(np.abs(x_prime)):<12.4f} | "
              f"{np.max(np.abs(D05)):<12.4f}")
    
    total_time = time.perf_counter() - start_all
    print(f"\n✅ Analysis completed in {total_time:.2f}s!")
    print("📊 All plots created successfully with separated layouts!")

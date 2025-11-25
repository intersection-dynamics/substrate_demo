#!/usr/bin/env python3
"""
parameter_sweep.py

Automated parameter sweep for scalar field + defrag gravity.

Explores (g_defrag, lambda_param) space to find optimal parameters
for structure formation from uniform noise.

Key metrics:
- Variance growth rate ω (from var_rho ~ exp(2ωt))
- Maximum density growth factor
- Final binding energy
- Fit quality (R²)

Usage:
    python parameter_sweep.py --mode coarse    # Quick 16-run exploration
    python parameter_sweep.py --mode fine      # Detailed 25-run refinement
    python parameter_sweep.py --mode custom    # Custom ranges (edit script)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time
from scalar_field_defrag_gpu import ScalarFieldDefragGPU

def compute_variance_growth_rate(df, t_start=None, t_end=None):
    """
    Fit var_rho ~ A * exp(2*ω*t) to extract growth rate ω.
    
    Args:
        df: Diagnostics dataframe
        t_start: Start time for fit (default: first 20%)
        t_end: End time for fit (default: last 80%)
    
    Returns:
        omega: Growth rate (positive = instability, negative = damping)
        R2: Fit quality (1.0 = perfect)
        A: Amplitude prefactor
    """
    if t_start is None:
        t_start = df['time'].iloc[int(0.2 * len(df))]
    if t_end is None:
        t_end = df['time'].iloc[-1]
    
    mask = (df['time'] >= t_start) & (df['time'] <= t_end)
    t = df['time'][mask].values
    var = df['var_rho'][mask].values
    
    # Fit log(var) = log(A) + 2*ω*t
    if np.all(var > 0) and len(t) > 2:
        log_var = np.log(var)
        
        # Linear fit
        coeffs = np.polyfit(t, log_var, 1)
        omega = coeffs[0] / 2.0  # Factor of 2!
        log_A = coeffs[1]
        A = np.exp(log_A)
        
        # R² for quality
        fit = coeffs[0]*t + coeffs[1]
        residuals = log_var - fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_var - np.mean(log_var))**2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        return omega, R2, A
    else:
        return 0.0, 0.0, 0.0


def run_single_config(g_defrag, lambda_param, output_dir, 
                     L=64, n_steps=500, save_snapshots=False):
    """
    Run single parameter configuration.
    
    Returns summary dict with key metrics.
    """
    print(f"\n{'='*70}")
    print(f"Running: g={g_defrag:.2f}, λ={lambda_param:.2f}")
    print(f"{'='*70}")
    
    # Create simulator
    sim = ScalarFieldDefragGPU(
        L=L,
        dx=1.0,
        dt=0.005,
        g_defrag=g_defrag,
        v=1.0,
        lambda_param=lambda_param
    )
    
    # Initial state: uniform + noise
    psi_init = sim.create_uniform_noise(mean=1.0, noise_amp=0.1, seed=42)
    
    # Run evolution
    start_time = time.time()
    df = sim.run_evolution(
        psi_init,
        n_steps=n_steps,
        snapshot_interval=max(n_steps // 10, 1),
        output_dir=output_dir,
        save_snapshots=save_snapshots
    )
    elapsed = time.time() - start_time
    
    # Compute growth rate
    omega, R2, A = compute_variance_growth_rate(df)
    
    # Extract key metrics
    initial_max_rho = df['max_rho'].iloc[0]
    final_max_rho = df['max_rho'].iloc[-1]
    max_rho_growth = final_max_rho / initial_max_rho
    
    initial_var = df['var_rho'].iloc[0]
    final_var = df['var_rho'].iloc[-1]
    var_growth = final_var / initial_var
    
    final_E_defrag = df['E_defrag'].iloc[-1]
    
    # Summary
    summary = {
        'g_defrag': g_defrag,
        'lambda_param': lambda_param,
        'omega': omega,
        'R2': R2,
        'A': A,
        'max_rho_growth': max_rho_growth,
        'var_growth': var_growth,
        'final_max_rho': final_max_rho,
        'final_E_defrag': final_E_defrag,
        'elapsed_time': elapsed
    }
    
    print(f"\nResults:")
    print(f"  Growth rate ω: {omega:+.6f} (R²={R2:.4f})")
    print(f"  Max ρ growth: {max_rho_growth:.4f}x")
    print(f"  Var growth: {var_growth:.4f}x")
    print(f"  Final E_defrag: {final_E_defrag:.6e}")
    print(f"  Time: {elapsed:.2f} sec")
    
    return summary, df


def run_parameter_sweep(g_values, lambda_values, output_dir, 
                       L=64, n_steps=500, save_best=True):
    """
    Run full parameter sweep over (g, λ) grid.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / 'raw_data'
    raw_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("PARAMETER SWEEP: SCALAR FIELD + DEFRAG GRAVITY")
    print("="*70)
    print(f"\nGrid:")
    print(f"  g_defrag: {g_values}")
    print(f"  lambda_param: {lambda_values}")
    print(f"  Total runs: {len(g_values) * len(lambda_values)}")
    print(f"  Steps per run: {n_steps}")
    print(f"  Grid size: {L}×{L}")
    
    results = []
    best_omega = -np.inf
    best_config = None
    best_df = None
    
    total_runs = len(g_values) * len(lambda_values)
    run_idx = 0
    
    for g in g_values:
        for lam in lambda_values:
            run_idx += 1
            print(f"\n{'='*70}")
            print(f"RUN {run_idx}/{total_runs}")
            print(f"{'='*70}")
            
            # Output directory for this run
            run_dir = raw_dir / f'g{g:.3f}_lambda{lam:.3f}'
            
            # Run simulation
            summary, df = run_single_config(
                g_defrag=g,
                lambda_param=lam,
                output_dir=run_dir,
                L=L,
                n_steps=n_steps,
                save_snapshots=False  # Save space unless it's the best
            )
            
            results.append(summary)
            
            # Track best
            if summary['omega'] > best_omega:
                best_omega = summary['omega']
                best_config = (g, lam)
                best_df = df
    
    # Convert to dataframe
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'sweep_summary.csv', index=False)
    print(f"\n{'='*70}")
    print(f"Saved summary to {output_dir / 'sweep_summary.csv'}")
    
    # Save best run with full snapshots
    if save_best and best_config is not None:
        print(f"\n{'='*70}")
        print(f"BEST CONFIGURATION:")
        print(f"  g_defrag = {best_config[0]:.3f}")
        print(f"  lambda_param = {best_config[1]:.3f}")
        print(f"  Growth rate ω = {best_omega:+.6f}")
        print(f"{'='*70}")
        
        best_dir = output_dir / 'best_run'
        best_dir.mkdir(exist_ok=True)
        
        # Re-run with snapshots
        print("\nRe-running best configuration with snapshots...")
        sim_best = ScalarFieldDefragGPU(
            L=L, dx=1.0, dt=0.005,
            g_defrag=best_config[0],
            v=1.0,
            lambda_param=best_config[1]
        )
        psi_init = sim_best.create_uniform_noise(mean=1.0, noise_amp=0.1, seed=42)
        df_best = sim_best.run_evolution(
            psi_init,
            n_steps=n_steps,
            snapshot_interval=max(n_steps // 20, 1),
            output_dir=best_dir,
            save_snapshots=True
        )
    
    return results_df


def plot_sweep_results(results_df, output_dir):
    """
    Create heatmaps and analysis plots from sweep results.
    """
    output_dir = Path(output_dir)
    
    print(f"\n{'='*70}")
    print("CREATING ANALYSIS PLOTS")
    print(f"{'='*70}")
    
    # Pivot for heatmaps
    g_values = sorted(results_df['g_defrag'].unique())
    lambda_values = sorted(results_df['lambda_param'].unique())
    
    # Create meshgrids
    omega_grid = results_df.pivot(
        index='lambda_param', 
        columns='g_defrag', 
        values='omega'
    )
    
    max_rho_grid = results_df.pivot(
        index='lambda_param',
        columns='g_defrag',
        values='max_rho_growth'
    )
    
    R2_grid = results_df.pivot(
        index='lambda_param',
        columns='g_defrag',
        values='R2'
    )
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Heatmap 1: Growth rate ω
    im1 = axes[0, 0].imshow(
        omega_grid, 
        cmap='RdYlGn', 
        origin='lower',
        extent=[min(g_values), max(g_values), 
                min(lambda_values), max(lambda_values)],
        aspect='auto',
        vmin=-0.1, vmax=0.2
    )
    axes[0, 0].set_xlabel('g_defrag')
    axes[0, 0].set_ylabel('λ')
    axes[0, 0].set_title('Growth Rate ω (green = instability)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Add contour at ω=0
    axes[0, 0].contour(
        g_values, lambda_values, omega_grid,
        levels=[0], colors='black', linewidths=2
    )
    
    # Heatmap 2: Max ρ growth
    im2 = axes[0, 1].imshow(
        max_rho_grid,
        cmap='hot',
        origin='lower',
        extent=[min(g_values), max(g_values),
                min(lambda_values), max(lambda_values)],
        aspect='auto'
    )
    axes[0, 1].set_xlabel('g_defrag')
    axes[0, 1].set_ylabel('λ')
    axes[0, 1].set_title('Max Density Growth Factor')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Add contour at 3× growth (success threshold)
    axes[0, 1].contour(
        g_values, lambda_values, max_rho_grid,
        levels=[3.0], colors='cyan', linewidths=2,
        linestyles='--'
    )
    
    # Heatmap 3: R² fit quality
    im3 = axes[1, 0].imshow(
        R2_grid,
        cmap='viridis',
        origin='lower',
        extent=[min(g_values), max(g_values),
                min(lambda_values), max(lambda_values)],
        aspect='auto',
        vmin=0, vmax=1
    )
    axes[1, 0].set_xlabel('g_defrag')
    axes[1, 0].set_ylabel('λ')
    axes[1, 0].set_title('Fit Quality R² (1 = perfect exponential)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Scatter plot: ω vs max_rho_growth
    axes[1, 1].scatter(
        results_df['omega'],
        results_df['max_rho_growth'],
        c=results_df['lambda_param'],
        cmap='coolwarm',
        s=100,
        alpha=0.7,
        edgecolors='black'
    )
    axes[1, 1].axhline(y=3.0, color='red', linestyle='--', 
                       alpha=0.5, label='3× threshold')
    axes[1, 1].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    axes[1, 1].set_xlabel('Growth Rate ω')
    axes[1, 1].set_ylabel('Max Density Growth Factor')
    axes[1, 1].set_title('Structure Formation Map')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sweep_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmaps to {output_dir / 'sweep_heatmaps.png'}")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SWEEP STATISTICS")
    print(f"{'='*70}")
    print(f"\nGrowth rate ω:")
    print(f"  Min: {results_df['omega'].min():+.6f}")
    print(f"  Max: {results_df['omega'].max():+.6f}")
    print(f"  Mean: {results_df['omega'].mean():+.6f}")
    print(f"  Std: {results_df['omega'].std():.6f}")
    
    print(f"\nMax density growth:")
    print(f"  Min: {results_df['max_rho_growth'].min():.4f}x")
    print(f"  Max: {results_df['max_rho_growth'].max():.4f}x")
    print(f"  Mean: {results_df['max_rho_growth'].mean():.4f}x")
    
    print(f"\nConfigurations with ω > 0.05: {np.sum(results_df['omega'] > 0.05)}")
    print(f"Configurations with ω > 0.1: {np.sum(results_df['omega'] > 0.1)}")
    print(f"Configurations with >3× growth: {np.sum(results_df['max_rho_growth'] > 3.0)}")
    
    # Top 5 configurations
    print(f"\n{'='*70}")
    print("TOP 5 CONFIGURATIONS (by growth rate ω)")
    print(f"{'='*70}")
    top5 = results_df.nlargest(5, 'omega')
    print(top5[['g_defrag', 'lambda_param', 'omega', 'R2', 'max_rho_growth']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Parameter sweep for scalar field + defrag')
    parser.add_argument('--mode', type=str, default='coarse',
                       choices=['coarse', 'fine', 'custom'],
                       help='Sweep mode: coarse (4×4), fine (5×5), or custom')
    parser.add_argument('--output', type=str, default='sweep_results',
                       help='Output directory')
    parser.add_argument('--steps', type=int, default=500,
                       help='Evolution steps per run')
    parser.add_argument('--grid_size', type=int, default=64,
                       help='Spatial grid size (L×L)')
    
    args = parser.parse_args()
    
    # Define parameter ranges
    if args.mode == 'coarse':
        g_values = [0.3, 0.5, 1.0, 2.0]
        lambda_values = [0.0, 0.1, 0.3, 0.5]
    elif args.mode == 'fine':
        g_values = np.linspace(0.5, 1.5, 5)
        lambda_values = np.linspace(0.0, 0.3, 5)
    else:  # custom
        # Edit these for custom ranges
        g_values = [1.5]
        lambda_values = [0.0]
    
    # Run sweep
    results_df = run_parameter_sweep(
        g_values=g_values,
        lambda_values=lambda_values,
        output_dir=args.output,
        L=args.grid_size,
        n_steps=args.steps,
        save_best=True
    )
    
    # Create analysis plots
    plot_sweep_results(results_df, args.output)
    
    print(f"\n{'='*70}")
    print("PARAMETER SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"\nAll results saved to: {args.output}/")
    print(f"  - sweep_summary.csv: Full results table")
    print(f"  - sweep_heatmaps.png: Visualization")
    print(f"  - best_run/: Best configuration with snapshots")
    print(f"  - raw_data/: Individual run outputs")


if __name__ == '__main__':
    main()
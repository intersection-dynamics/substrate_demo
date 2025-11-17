#!/usr/bin/env python3
"""
investigate_energy_scaling.py

Investigate why E_bind/site increases with system size.

For an extensive quantity: E_bind should scale ~ L^2 → E_bind/L^2 = constant
For interface energy: E_bind should scale ~ L (perimeter) → E_bind/L = constant

What we observe: E_bind/L^2 increases with L

Possible causes:
1. FFT normalization issue
2. Defrag potential has non-local bulk contribution
3. Stripe width changes with L (shouldn't, but check)
4. Implementation bug

This script:
- Runs careful energy analysis at different L
- Checks FFT normalization
- Separates interface vs bulk contributions
- Tests different scalings
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

try:
    from ising_defrag_gpu import IsingDefragGPU
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    import numpy as cp
    print("WARNING: GPU not available, using CPU")


def run_detailed_energy_analysis(L, n_sweeps=2000):
    """Run simulation and track energy components."""
    
    print(f"\n{'='*70}")
    print(f"ANALYZING L={L}")
    print(f"{'='*70}")
    
    sim = IsingDefragGPU(L=L, T=1.0, g_defrag=0.5, coarse_grain_size=1)
    spins = sim.create_noise_spins(flip_prob=0.2, seed=42)
    
    # Evolve to equilibrium
    print(f"Evolving to equilibrium...")
    for sweep in range(n_sweeps):
        if sweep % 500 == 0:
            print(f"  Sweep {sweep}/{n_sweeps}")
        Phi = sim.solve_defrag_potential(spins)
        spins = sim.metropolis_sweep(spins, Phi)
    
    # Get final state
    Phi = sim.solve_defrag_potential(spins)
    
    # Move to CPU for analysis
    if GPU_AVAILABLE:
        spins_cpu = cp.asnumpy(spins)
        Phi_cpu = cp.asnumpy(Phi)
    else:
        spins_cpu = spins
        Phi_cpu = Phi
    
    # Count walls
    walls = 0
    for i in range(L):
        for j in range(L):
            # Horizontal walls
            if spins_cpu[i,j] != spins_cpu[(i+1)%L, j]:
                walls += 1
            # Vertical walls (count only one direction to avoid double-counting)
            if spins_cpu[i,j] != spins_cpu[i, (j+1)%L]:
                walls += 1
    walls = walls // 2  # Each wall counted twice
    
    # Compute magnetization field for defrag
    if sim.coarse_grain_size > 1:
        cg = sim.coarse_grain_size
        L_cg = L // cg
        M_field = np.zeros((L_cg, L_cg))
        for i in range(L_cg):
            for j in range(L_cg):
                block = spins_cpu[i*cg:(i+1)*cg, j*cg:(j+1)*cg]
                M_field[i,j] = np.mean(block)
    else:
        M_field = spins_cpu.astype(float)
    
    # Get density fluctuation
    s = M_field - np.mean(M_field)
    
    # Compute defrag energy components
    # E_defrag = -(g/2) * sum_x s(x) Phi(x)
    # where Phi solves: nabla^2 Phi = s
    
    # Total defrag energy
    E_defrag_total = -0.5 * sim.g_defrag * np.sum(s * Phi_cpu)
    
    # Compute gradient of Phi (interface contribution)
    grad_Phi_x = np.gradient(Phi_cpu, axis=0)
    grad_Phi_y = np.gradient(Phi_cpu, axis=1)
    grad_Phi_mag = np.sqrt(grad_Phi_x**2 + grad_Phi_y**2)
    
    # Ising energy
    E_ising = 0.0
    J = sim.J
    for i in range(L):
        for j in range(L):
            # Only count each pair once
            E_ising -= J * spins_cpu[i,j] * spins_cpu[(i+1)%L, j]
            E_ising -= J * spins_cpu[i,j] * spins_cpu[i, (j+1)%L]
    
    # Magnetization
    M = np.mean(spins_cpu)
    
    # Collect results
    results = {
        'L': L,
        'N_sites': L*L,
        'walls': walls,
        'M': M,
        'E_ising': E_ising,
        'E_defrag_total': E_defrag_total,
        'E_total': E_ising + E_defrag_total,
        # Normalized quantities
        'E_defrag_per_site': E_defrag_total / (L*L),
        'E_defrag_per_L': E_defrag_total / L,
        'E_defrag_per_wall': E_defrag_total / walls if walls > 0 else 0,
        # Potential statistics
        'Phi_mean': np.mean(Phi_cpu),
        'Phi_std': np.std(Phi_cpu),
        'Phi_max': np.max(np.abs(Phi_cpu)),
        's_std': np.std(s),
        'grad_Phi_mean': np.mean(grad_Phi_mag),
        'grad_Phi_max': np.max(grad_Phi_mag),
    }
    
    print(f"\nResults for L={L}:")
    print(f"  Walls: {walls}")
    print(f"  E_defrag: {E_defrag_total:.2e}")
    print(f"  E_defrag/site: {results['E_defrag_per_site']:.2f}")
    print(f"  E_defrag/L: {results['E_defrag_per_L']:.2f}")
    print(f"  E_defrag/wall: {results['E_defrag_per_wall']:.2f}")
    print(f"  Phi_max: {results['Phi_max']:.2e}")
    print(f"  grad_Phi_max: {results['grad_Phi_max']:.2e}")
    
    return results, spins_cpu, Phi_cpu


def test_fft_normalization():
    """Test if FFT Poisson solver has correct normalization."""
    
    print("\n" + "="*70)
    print("FFT NORMALIZATION TEST")
    print("="*70)
    
    # Test with simple source: s(x,y) = sin(2π x/L) sin(2π y/L)
    # Analytical solution: Phi = -s / (2*(2π/L)^2) = -s*L^2/(8π^2)
    
    for L in [32, 64, 128]:
        print(f"\nTesting L={L}")
        
        x = np.arange(L)
        y = np.arange(L)
        X, Y = np.meshgrid(x, y)
        
        # Source term
        s = np.sin(2*np.pi*X/L) * np.sin(2*np.pi*Y/L)
        s = s - np.mean(s)  # Zero mean
        
        # Analytical solution
        k = 2*np.pi/L
        Phi_analytical = -s / (2*k**2)
        
        # Numerical solution using FFT
        s_fft = np.fft.fft2(s)
        
        kx = np.fft.fftfreq(L, d=1.0) * 2 * np.pi
        ky = np.fft.fftfreq(L, d=1.0) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        K2[0,0] = 1.0  # Avoid division by zero
        
        Phi_fft = -s_fft / K2
        Phi_fft[0,0] = 0.0  # Zero mean
        Phi_numerical = np.fft.ifft2(Phi_fft).real
        
        # Compare
        error = np.abs(Phi_numerical - Phi_analytical).max()
        relative_error = error / np.abs(Phi_analytical).max()
        
        print(f"  Max absolute error: {error:.2e}")
        print(f"  Relative error: {relative_error:.2%}")
        
        if relative_error < 0.01:
            print(f"  [PASS] FFT normalization correct for L={L}")
        else:
            print(f"  [FAIL] FFT normalization incorrect for L={L}")


def plot_energy_scaling(results_list):
    """Create comprehensive energy scaling plots."""
    
    df = pd.DataFrame(results_list)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: E_defrag vs L^2 (should be linear if extensive)
    ax = axes[0, 0]
    ax.plot(df['N_sites'], df['E_defrag_total'], 'o-', markersize=10, linewidth=2)
    ax.set_xlabel('N = L²')
    ax.set_ylabel('E_defrag')
    ax.set_title('Total Defrag Energy vs System Size')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: E_defrag/site vs L (should be constant if extensive)
    ax = axes[0, 1]
    ax.plot(df['L'], df['E_defrag_per_site'], 'o-', markersize=10, linewidth=2, color='red')
    ax.set_xlabel('L')
    ax.set_ylabel('E_defrag / N')
    ax.set_title('Energy per Site (should be constant)')
    ax.axhline(df['E_defrag_per_site'].mean(), linestyle='--', color='gray', 
               label='Mean')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: E_defrag/L vs L (interface energy scaling)
    ax = axes[0, 2]
    ax.plot(df['L'], df['E_defrag_per_L'], 'o-', markersize=10, linewidth=2, color='green')
    ax.set_xlabel('L')
    ax.set_ylabel('E_defrag / L')
    ax.set_title('Energy per Length (interface scaling)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: E_defrag/wall vs L
    ax = axes[1, 0]
    ax.plot(df['L'], df['E_defrag_per_wall'], 'o-', markersize=10, linewidth=2, color='purple')
    ax.set_xlabel('L')
    ax.set_ylabel('E_defrag / wall')
    ax.set_title('Energy per Wall (should be constant)')
    ax.axhline(df['E_defrag_per_wall'].mean(), linestyle='--', color='gray')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Phi_max vs L
    ax = axes[1, 1]
    ax.plot(df['L'], df['Phi_max'], 'o-', markersize=10, linewidth=2, color='orange')
    ax.set_xlabel('L')
    ax.set_ylabel('max|Phi|')
    ax.set_title('Maximum Potential (normalization check)')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Different scaling hypotheses
    ax = axes[1, 2]
    
    # Test different scalings
    L_arr = df['L'].values
    E_arr = np.abs(df['E_defrag_total'].values)
    
    # Normalize to L=48 value
    E_48 = E_arr[df['L']==48].item()
    L_48 = 48
    
    # Expected scalings
    scaling_L = E_48 * (L_arr / L_48)
    scaling_L2 = E_48 * (L_arr / L_48)**2
    scaling_L3 = E_48 * (L_arr / L_48)**3
    
    ax.plot(L_arr, E_arr, 'ko-', markersize=10, linewidth=2, label='Observed')
    ax.plot(L_arr, scaling_L, '--', alpha=0.7, label='~ L (interface)')
    ax.plot(L_arr, scaling_L2, '--', alpha=0.7, label='~ L² (extensive)')
    ax.plot(L_arr, scaling_L3, '--', alpha=0.7, label='~ L³')
    
    ax.set_xlabel('L')
    ax.set_ylabel('|E_defrag|')
    ax.set_title('Observed vs Expected Scaling')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    output_dir = Path("energy_scaling_analysis")
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'energy_scaling_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[PASS] Saved plots to {output_dir / 'energy_scaling_plots.png'}")


def main():
    """Run energy scaling investigation."""
    
    print("="*70)
    print("ENERGY SCALING INVESTIGATION")
    print("="*70)
    print("\nInvestigating why E_bind/site increases with L")
    print("\nRunning simulations at L = 32, 48, 64, 96")
    print("This will take ~30 minutes...")
    print("="*70)
    
    # Test FFT normalization first
    test_fft_normalization()
    
    # Run simulations at different sizes
    L_values = [32, 48, 64, 96]
    results_list = []
    
    for L in L_values:
        results, spins, Phi = run_detailed_energy_analysis(L, n_sweeps=2000)
        results_list.append(results)
    
    # Create DataFrame
    df = pd.DataFrame(results_list)
    
    # Save results
    output_dir = Path("energy_scaling_analysis")
    output_dir.mkdir(exist_ok=True)
    
    df.to_csv(output_dir / 'energy_scaling_results.csv', index=False)
    print(f"\n[PASS] Saved results to {output_dir / 'energy_scaling_results.csv'}")
    
    # Analysis
    print("\n" + "="*70)
    print("SCALING ANALYSIS")
    print("="*70)
    
    print("\nRaw data:")
    print(df[['L', 'walls', 'E_defrag_total', 'E_defrag_per_site', 'E_defrag_per_L', 'E_defrag_per_wall']].to_string(index=False))
    
    # Test which scaling fits best
    print("\n" + "-"*70)
    print("Testing scaling hypotheses:")
    print("-"*70)
    
    from scipy import stats
    
    # Remove L=32 if it doesn't have stripes
    df_valid = df[df['walls'] > 10].copy()
    
    if len(df_valid) < 3:
        print("\nWARNING: Not enough valid data points (need L with stripes)")
        df_valid = df
    
    # Test E ~ L
    log_L = np.log(df_valid['L'].values)
    log_E = np.log(np.abs(df_valid['E_defrag_total'].values))
    slope_L, intercept, r_value, _, _ = stats.linregress(log_L, log_E)
    
    print(f"\nLog-log fit: log(E) = {slope_L:.2f} * log(L) + {intercept:.2f}")
    print(f"  R² = {r_value**2:.4f}")
    print(f"  Interpretation: E ~ L^{slope_L:.2f}")
    
    if 0.9 < slope_L < 1.1:
        print("  → Scales like L (INTERFACE energy)")
    elif 1.9 < slope_L < 2.1:
        print("  → Scales like L² (EXTENSIVE/BULK energy)")
    elif 2.5 < slope_L < 3.5:
        print("  → Scales like L³ (SUPER-EXTENSIVE - unexpected!)")
    else:
        print(f"  → Unusual scaling exponent {slope_L:.2f}")
    
    # Check if E/wall is constant
    print("\n" + "-"*70)
    print("Energy per wall analysis:")
    print("-"*70)
    
    E_per_wall_mean = df_valid['E_defrag_per_wall'].mean()
    E_per_wall_std = df_valid['E_defrag_per_wall'].std()
    E_per_wall_cv = E_per_wall_std / abs(E_per_wall_mean)
    
    print(f"  Mean: {E_per_wall_mean:.2f}")
    print(f"  Std: {E_per_wall_std:.2f}")
    print(f"  CV: {E_per_wall_cv:.2%}")
    
    if E_per_wall_cv < 0.2:
        print("  [PASS] E/wall approximately constant - energy is from walls")
    else:
        print("  [FAIL] E/wall varies significantly - not pure interface energy")
    
    # Create plots
    plot_energy_scaling(results_list)
    
    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if 0.9 < slope_L < 1.1:
        print("\nEnergy scales as E ~ L (interface/perimeter)")
        print("This is EXPECTED for stripe domain walls.")
        print("The increasing E/site is because interface/bulk ratio ~ 1/L")
    elif 1.9 < slope_L < 2.1:
        print("\nEnergy scales as E ~ L² (bulk/extensive)")
        print("This is EXPECTED for uniform field energy.")
        print("The E/site should be constant (it's not - needs investigation)")
    elif 2.5 < slope_L < 3.5:
        print("\nEnergy scales as E ~ L³ (super-extensive)")
        print("This is UNEXPECTED and suggests:")
        print("  1. FFT normalization issue")
        print("  2. Non-local bulk contribution from defrag")
        print("  3. Implementation bug")
        print("\nRECOMMENDATION: Check FFT Poisson solver implementation")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
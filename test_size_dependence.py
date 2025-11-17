#!/usr/bin/env python3
"""
test_size_dependence.py

Test if stripe phase depends on system size L.

CRITICAL TEST: If stripes are real physics (not grid artifact):
- Wall count should scale ~ 2L (more walls in bigger systems)
- Stripe width should be constant (~ 30 sites regardless of L)
- Energy per site should be constant

If it's a grid artifact:
- Wall count would be fixed or scale strangely
- Structure would change with L

Runs: L = 32, 48, 64, 96
Same parameters: T=1.0, g=0.5, cg=1, seed=42
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

try:
    from ising_defrag_gpu import IsingDefragGPU
    print("✓ Imported ising_defrag_gpu")
except ImportError:
    print("ERROR: Cannot import ising_defrag_gpu.py")
    print("Make sure ising_defrag_gpu.py is in the same directory")
    sys.exit(1)


def run_size_test(L, T=1.0, g_defrag=0.5, n_sweeps=2000, seed=42):
    """Run single size test."""
    print(f"\n{'='*70}")
    print(f"TESTING L={L}")
    print(f"{'='*70}")
    
    output_dir = f"size_test_L{L}"
    
    # Create simulator
    sim = IsingDefragGPU(
        L=L,
        T=T,
        g_defrag=g_defrag,
        coarse_grain_size=1,
    )
    
    # Initial condition (same relative noise)
    spins = sim.create_noise_spins(flip_prob=0.2, seed=seed)
    
    # Run evolution
    print(f"Running {n_sweeps} sweeps...")
    df = sim.run_evolution(
        spins,
        n_sweeps=n_sweeps,
        snapshot_interval=500,
        output_dir=output_dir
    )
    
    # Get final state
    final = df.iloc[-1]
    
    result = {
        'L': L,
        'N_sites': L*L,
        'final_M': final['M'],
        'final_M_abs': final['M_abs'],
        'final_walls': final['n_walls'],
        'final_E_bind': final['E_bind'],
        'E_bind_per_site': final['E_bind'] / (L*L),
        'walls_per_L': final['n_walls'] / L,
        'stripe_width_estimate': L / (final['n_walls'] / 2) if final['n_walls'] > 0 else 0
    }
    
    print(f"\nResults for L={L}:")
    print(f"  Walls: {result['final_walls']:.0f}")
    print(f"  Walls/L: {result['walls_per_L']:.2f}")
    print(f"  E_bind/site: {result['E_bind_per_site']:.2f}")
    print(f"  Stripe width: {result['stripe_width_estimate']:.1f} sites")
    print(f"  |M|: {result['final_M_abs']:.6f}")
    
    return result, df


def main():
    """Run all size tests and create comparison."""
    
    print("="*70)
    print("SYSTEM SIZE DEPENDENCE TEST")
    print("="*70)
    print("\nTesting if stripe phase is real physics or grid artifact")
    print("\nExpected if REAL:")
    print("  - Walls scale ~ 2L")
    print("  - Stripe width constant")
    print("  - Energy/site constant")
    print("\nExpected if ARTIFACT:")
    print("  - Walls don't scale properly")
    print("  - Structure changes with L")
    print("="*70)
    
    # Test sizes
    L_values = [32, 48, 64, 96]
    
    results = []
    all_df = {}
    
    # Run each size
    for L in L_values:
        result, df = run_size_test(L)
        results.append(result)
        all_df[L] = df
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_dir = Path("size_dependence_test")
    output_dir.mkdir(exist_ok=True)
    
    results_df.to_csv(output_dir / "size_dependence_results.csv", index=False)
    print(f"\n✓ Saved results to {output_dir / 'size_dependence_results.csv'}")
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Analysis
    print("\n" + "="*70)
    print("SCALING ANALYSIS")
    print("="*70)
    
    # Check wall scaling
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        results_df['L'], 
        results_df['final_walls']
    )
    
    print(f"\nWall count vs L:")
    print(f"  Linear fit: walls = {slope:.2f}*L + {intercept:.1f}")
    print(f"  R² = {r_value**2:.4f}")
    print(f"  Expected if real: slope ≈ 2.0")
    print(f"  Actual slope: {slope:.2f}")
    
    if 1.8 < slope < 2.2:
        print("  ✓ PASSES: Walls scale ~ 2L (real physics!)")
    else:
        print("  ✗ FAILS: Unusual scaling (possible artifact)")
    
    # Check stripe width
    mean_width = results_df['stripe_width_estimate'].mean()
    std_width = results_df['stripe_width_estimate'].std()
    
    print(f"\nStripe width:")
    print(f"  Mean: {mean_width:.1f} ± {std_width:.1f} sites")
    print(f"  Relative variation: {std_width/mean_width*100:.1f}%")
    
    if std_width/mean_width < 0.2:
        print("  ✓ PASSES: Width approximately constant (real physics!)")
    else:
        print("  ✗ FAILS: Width varies significantly (possible artifact)")
    
    # Check energy per site
    mean_E_per_site = results_df['E_bind_per_site'].mean()
    std_E_per_site = results_df['E_bind_per_site'].std()
    
    print(f"\nEnergy per site:")
    print(f"  Mean: {mean_E_per_site:.2f} ± {std_E_per_site:.2f}")
    print(f"  Relative variation: {std_E_per_site/abs(mean_E_per_site)*100:.1f}%")
    
    if std_E_per_site/abs(mean_E_per_site) < 0.1:
        print("  ✓ PASSES: Energy/site constant (extensive property!)")
    else:
        print("  ✗ FAILS: Energy/site varies (non-extensive)")
    
    # Create plots
    create_plots(results_df, all_df, output_dir)
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if (1.8 < slope < 2.2 and 
        std_width/mean_width < 0.2 and 
        std_E_per_site/abs(mean_E_per_site) < 0.1):
        print("✓✓✓ STRIPE PHASE IS REAL PHYSICS ✓✓✓")
        print("\nEvidence:")
        print("  - Walls scale linearly with L")
        print("  - Stripe width is system-size independent")
        print("  - Energy is extensive (scales with N)")
        print("\nThis is NOT a grid artifact!")
    else:
        print("⚠ RESULTS INCONCLUSIVE OR CONCERNING")
        print("\nSome scaling properties don't match expected behavior.")
        print("May need further investigation.")
    
    print("\n" + "="*70)


def create_plots(results_df, all_df, output_dir):
    """Create comparison plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Walls vs L
    ax = axes[0, 0]
    ax.scatter(results_df['L'], results_df['final_walls'], s=100, color='blue')
    
    # Fit line
    from scipy import stats
    slope, intercept, r_value, _, _ = stats.linregress(
        results_df['L'], 
        results_df['final_walls']
    )
    L_fit = np.array([30, 100])
    ax.plot(L_fit, slope*L_fit + intercept, 'r--', 
            label=f'Fit: {slope:.2f}L + {intercept:.1f}\nR²={r_value**2:.3f}')
    
    # Expected 2L line
    ax.plot(L_fit, 2*L_fit, 'g--', alpha=0.5, label='Expected: 2L')
    
    ax.set_xlabel('System Size L')
    ax.set_ylabel('Number of Walls')
    ax.set_title('Wall Scaling Test')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Stripe width vs L
    ax = axes[0, 1]
    ax.scatter(results_df['L'], results_df['stripe_width_estimate'], 
               s=100, color='green')
    mean_width = results_df['stripe_width_estimate'].mean()
    ax.axhline(mean_width, color='red', linestyle='--', 
               label=f'Mean: {mean_width:.1f} sites')
    ax.set_xlabel('System Size L')
    ax.set_ylabel('Stripe Width (sites)')
    ax.set_title('Stripe Width Independence Test')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Energy per site vs L
    ax = axes[0, 2]
    ax.scatter(results_df['L'], results_df['E_bind_per_site'], 
               s=100, color='purple')
    mean_E = results_df['E_bind_per_site'].mean()
    ax.axhline(mean_E, color='red', linestyle='--',
               label=f'Mean: {mean_E:.2f}')
    ax.set_xlabel('System Size L')
    ax.set_ylabel('E_bind per site')
    ax.set_title('Energy Extensivity Test')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Magnetization evolution for all L
    ax = axes[1, 0]
    for L in all_df.keys():
        df = all_df[L]
        ax.plot(df['sweep'], df['M_abs'], label=f'L={L}', alpha=0.7)
    ax.set_xlabel('Sweep')
    ax.set_ylabel('|M|')
    ax.set_title('Magnetization Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Walls evolution for all L
    ax = axes[1, 1]
    for L in all_df.keys():
        df = all_df[L]
        ax.plot(df['sweep'], df['n_walls'], label=f'L={L}', alpha=0.7)
    ax.set_xlabel('Sweep')
    ax.set_ylabel('Number of Walls')
    ax.set_title('Wall Count Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Binding energy evolution (normalized)
    ax = axes[1, 2]
    for L in all_df.keys():
        df = all_df[L]
        N = L*L
        ax.plot(df['sweep'], df['E_bind']/N, label=f'L={L}', alpha=0.7)
    ax.set_xlabel('Sweep')
    ax.set_ylabel('E_bind per site')
    ax.set_title('Normalized Binding Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'size_dependence_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved plots to {output_dir / 'size_dependence_analysis.png'}")


if __name__ == '__main__':
    main()
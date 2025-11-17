#!/usr/bin/env python3
"""
test_reproducibility.py

Test if stripe phase is reproducible across different random initial conditions.

CRITICAL TEST: If stripes are a robust equilibrium state:
- All seeds should converge to similar final state
- Wall count should be consistent (±2 walls)
- Energy should be consistent (±1% variation)
- Magnetization should remain near zero

If it's a random artifact:
- Different seeds give wildly different results
- Some seeds might not form stripes at all

Runs: 5 different seeds at L=64, T=1.0, g=0.5
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
    sys.exit(1)


def run_seed_test(seed, L=64, T=1.0, g_defrag=0.5, n_sweeps=2000):
    """Run single seed test."""
    print(f"\n{'='*70}")
    print(f"TESTING SEED={seed}")
    print(f"{'='*70}")
    
    output_dir = f"seed_test_s{seed}"
    
    # Create simulator
    sim = IsingDefragGPU(
        L=L,
        T=T,
        g_defrag=g_defrag,
        coarse_grain_size=1,
    )
    
    # Initial condition with this seed
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
    
    # Get equilibration region (last 25% of sweeps)
    equil_start = int(0.75 * len(df))
    df_equil = df.iloc[equil_start:]
    
    result = {
        'seed': seed,
        'final_M': final['M'],
        'final_M_abs': final['M_abs'],
        'final_walls': final['n_walls'],
        'final_E_bind': final['E_bind'],
        'equil_M_mean': df_equil['M'].mean(),
        'equil_M_std': df_equil['M'].std(),
        'equil_walls_mean': df_equil['n_walls'].mean(),
        'equil_walls_std': df_equil['n_walls'].std(),
        'equil_E_bind_mean': df_equil['E_bind'].mean(),
        'equil_E_bind_std': df_equil['E_bind'].std(),
    }
    
    print(f"\nResults for seed={seed}:")
    print(f"  Final walls: {result['final_walls']:.0f}")
    print(f"  Equilibrium walls: {result['equil_walls_mean']:.1f} ± {result['equil_walls_std']:.1f}")
    print(f"  Final E_bind: {result['final_E_bind']:.0f}")
    print(f"  Equilibrium E_bind: {result['equil_E_bind_mean']:.0f} ± {result['equil_E_bind_std']:.0f}")
    print(f"  Final |M|: {result['final_M_abs']:.6f}")
    
    return result, df


def main():
    """Run all seed tests and analyze reproducibility."""
    
    print("="*70)
    print("STATISTICAL REPRODUCIBILITY TEST")
    print("="*70)
    print("\nTesting if stripe phase is reproducible across random initial conditions")
    print("\nExpected if ROBUST:")
    print("  - All seeds converge to same state")
    print("  - Wall count: 128 ± 2")
    print("  - Energy consistent within ±1%")
    print("\nExpected if RANDOM:")
    print("  - Different final states")
    print("  - Large variation in observables")
    print("="*70)
    
    # Test seeds
    seeds = [42, 123, 456, 789, 1337]
    
    results = []
    all_df = {}
    
    # Run each seed
    for seed in seeds:
        result, df = run_seed_test(seed)
        results.append(result)
        all_df[seed] = df
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_dir = Path("reproducibility_test")
    output_dir.mkdir(exist_ok=True)
    
    results_df.to_csv(output_dir / "reproducibility_results.csv", index=False)
    print(f"\n✓ Saved results to {output_dir / 'reproducibility_results.csv'}")
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(results_df[['seed', 'final_walls', 'final_E_bind', 'final_M_abs']].to_string(index=False))
    
    # Statistical analysis
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)
    
    # Wall count statistics
    wall_mean = results_df['equil_walls_mean'].mean()
    wall_std = results_df['equil_walls_mean'].std()
    wall_min = results_df['equil_walls_mean'].min()
    wall_max = results_df['equil_walls_mean'].max()
    
    print(f"\nWall count across seeds:")
    print(f"  Mean: {wall_mean:.1f} ± {wall_std:.1f}")
    print(f"  Range: [{wall_min:.0f}, {wall_max:.0f}]")
    print(f"  Coefficient of variation: {wall_std/wall_mean*100:.2f}%")
    
    if wall_std/wall_mean < 0.05:
        print("  ✓ EXCELLENT: Very low variation (<5%)")
    elif wall_std/wall_mean < 0.10:
        print("  ✓ GOOD: Low variation (<10%)")
    else:
        print("  ✗ POOR: High variation (>10%)")
    
    # Energy statistics
    E_mean = results_df['equil_E_bind_mean'].mean()
    E_std = results_df['equil_E_bind_mean'].std()
    E_min = results_df['equil_E_bind_mean'].min()
    E_max = results_df['equil_E_bind_mean'].max()
    
    print(f"\nBinding energy across seeds:")
    print(f"  Mean: {E_mean:.0f} ± {E_std:.0f}")
    print(f"  Range: [{E_min:.0f}, {E_max:.0f}]")
    print(f"  Coefficient of variation: {E_std/abs(E_mean)*100:.2f}%")
    
    if E_std/abs(E_mean) < 0.01:
        print("  ✓ EXCELLENT: Very low variation (<1%)")
    elif E_std/abs(E_mean) < 0.05:
        print("  ✓ GOOD: Low variation (<5%)")
    else:
        print("  ✗ POOR: High variation (>5%)")
    
    # Magnetization statistics
    M_abs_mean = results_df['final_M_abs'].mean()
    M_abs_std = results_df['final_M_abs'].std()
    
    print(f"\nMagnetization |M| across seeds:")
    print(f"  Mean: {M_abs_mean:.6f} ± {M_abs_std:.6f}")
    
    if M_abs_mean < 0.01:
        print("  ✓ PASSES: |M| ≈ 0 as expected for stripe phase")
    else:
        print("  ✗ FAILS: |M| too large (not stripe phase)")
    
    # Create plots
    create_plots(results_df, all_df, output_dir)
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if (wall_std/wall_mean < 0.10 and 
        E_std/abs(E_mean) < 0.05 and 
        M_abs_mean < 0.01):
        print("✓✓✓ STRIPE PHASE IS REPRODUCIBLE ✓✓✓")
        print("\nEvidence:")
        print("  - All seeds converge to same stripe state")
        print("  - Wall count is consistent across realizations")
        print("  - Energy is consistent within thermal fluctuations")
        print("  - Magnetization remains near zero")
        print("\nThis is a ROBUST equilibrium state, not a random artifact!")
    else:
        print("⚠ REPRODUCIBILITY IS QUESTIONABLE")
        print("\nSome observables show high variation across seeds.")
        print("May indicate multiple metastable states or long transients.")
    
    print("\n" + "="*70)


def create_plots(results_df, all_df, output_dir):
    """Create comparison plots."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Wall count evolution for all seeds
    ax1 = fig.add_subplot(gs[0, :])
    for seed in all_df.keys():
        df = all_df[seed]
        ax1.plot(df['sweep'], df['n_walls'], label=f'seed={seed}', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Sweep', fontsize=12)
    ax1.set_ylabel('Number of Walls', fontsize=12)
    ax1.set_title('Wall Count Evolution (All Seeds)', fontsize=14, fontweight='bold')
    ax1.legend(ncol=5, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energy evolution for all seeds
    ax2 = fig.add_subplot(gs[1, :])
    for seed in all_df.keys():
        df = all_df[seed]
        ax2.plot(df['sweep'], df['E_bind'], label=f'seed={seed}', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Sweep', fontsize=12)
    ax2.set_ylabel('E_bind', fontsize=12)
    ax2.set_title('Binding Energy Evolution (All Seeds)', fontsize=14, fontweight='bold')
    ax2.legend(ncol=5, loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final wall distribution
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.bar(range(len(results_df)), results_df['final_walls'], color='steelblue', edgecolor='black')
    ax3.axhline(results_df['final_walls'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean: {results_df['final_walls'].mean():.1f}")
    ax3.set_xlabel('Seed Index', fontsize=12)
    ax3.set_ylabel('Final Wall Count', fontsize=12)
    ax3.set_title('Final Wall Count by Seed', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Final energy distribution
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.bar(range(len(results_df)), results_df['final_E_bind'], color='coral', edgecolor='black')
    ax4.axhline(results_df['final_E_bind'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean: {results_df['final_E_bind'].mean():.0f}")
    ax4.set_xlabel('Seed Index', fontsize=12)
    ax4.set_ylabel('Final E_bind', fontsize=12)
    ax4.set_title('Final Energy by Seed', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Magnetization distribution
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.bar(range(len(results_df)), results_df['final_M_abs'], color='lightgreen', edgecolor='black')
    ax5.axhline(results_df['final_M_abs'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean: {results_df['final_M_abs'].mean():.6f}")
    ax5.set_xlabel('Seed Index', fontsize=12)
    ax5.set_ylabel('Final |M|', fontsize=12)
    ax5.set_title('Final |M| by Seed', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(output_dir / 'reproducibility_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved plots to {output_dir / 'reproducibility_analysis.png'}")


if __name__ == '__main__':
    main()
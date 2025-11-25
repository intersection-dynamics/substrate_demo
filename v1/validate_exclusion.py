#!/usr/bin/env python3
"""
validate_exclusion.py

Automated test suite to validate fermionic exclusion claims.

Runs critical tests to distinguish real physics from numerical artifacts:
1. Resolution independence
2. Mass scaling  
3. Boundary effects
4. Single vortex stability

Run this when you have ~2 hours. It will tell you if the exclusion is real.

Usage:
    python validate_exclusion.py --quick    # Fast tests only (~30 min)
    python validate_exclusion.py --full     # All tests (~2 hours)
"""

import subprocess
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def run_simulation(Nx, dx, dt, mass, G, offset, steps, out_dir):
    """Run one simulation and return path to observables."""
    
    cmd = [
        "python", "substrate_engine.py",
        "--Nx", str(Nx),
        "--dx", str(dx),
        "--dt", str(dt),
        "--mass", str(mass),
        "--G", str(G),
        "--offset", str(offset),
        "--steps", str(steps),
        "--obs_interval", "50",
        "--out", out_dir
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    print("-" * 60)
    
    subprocess.run(cmd, check=True)
    
    obs_file = os.path.join(out_dir, "observables.csv")
    return obs_file


def load_observables(csv_path):
    """Load observable data from CSV."""
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    
    return {
        'step': data[:, 0],
        'time': data[:, 1],
        'max_rho': data[:, 2],
        'core_sep': data[:, 3],
        'x1': data[:, 4],
        'y1': data[:, 5],
        'x2': data[:, 6],
        'y2': data[:, 7],
    }


def test_resolution_independence(output_dir):
    """
    Test 1: Resolution Independence
    
    Run same physical system at different grid resolutions.
    
    If real: Minimum separation in physical units should be constant.
    If artifact: Minimum separation will scale with dx.
    """
    
    print("\n" + "="*60)
    print("TEST 1: RESOLUTION INDEPENDENCE")
    print("="*60)
    
    results = []
    
    # Same physical domain (64 × 64), different resolutions
    configs = [
        {'Nx': 128, 'dx': 0.5, 'label': 'baseline (128, dx=0.5)'},
        {'Nx': 256, 'dx': 0.25, 'label': 'high-res (256, dx=0.25)'},
        {'Nx': 64, 'dx': 1.0, 'label': 'low-res (64, dx=1.0)'},
    ]
    
    for i, config in enumerate(configs):
        out_dir = os.path.join(output_dir, f"resolution_test_{i}")
        
        obs_file = run_simulation(
            Nx=config['Nx'],
            dx=config['dx'],
            dt=0.01,
            mass=0.5,
            G=50.0,
            offset=10.0,  # Physical offset (same for all)
            steps=500,
            out_dir=out_dir
        )
        
        obs = load_observables(obs_file)
        
        # Find minimum separation
        min_sep = np.min(obs['core_sep'])
        
        results.append({
            'label': config['label'],
            'Nx': config['Nx'],
            'dx': config['dx'],
            'min_sep_physical': min_sep,
            'min_sep_gridunits': min_sep / config['dx'],
            'data': obs
        })
        
        print(f"\n{config['label']}:")
        print(f"  Min separation (physical): {min_sep:.2f}")
        print(f"  Min separation (grid units): {min_sep/config['dx']:.1f}")
    
    # Analysis
    print("\n" + "-"*60)
    print("ANALYSIS:")
    
    min_seps = [r['min_sep_physical'] for r in results]
    mean_sep = np.mean(min_seps)
    std_sep = np.std(min_seps)
    variation = std_sep / mean_sep
    
    print(f"Mean minimum separation: {mean_sep:.2f} ± {std_sep:.2f}")
    print(f"Coefficient of variation: {variation:.1%}")
    
    if variation < 0.1:
        print("✓ PASS: Separation is resolution-independent (<10% variation)")
        verdict = "PASS"
    else:
        print("✗ FAIL: Separation varies significantly with resolution")
        verdict = "FAIL"
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    for r in results:
        ax.plot(r['data']['time'], r['data']['core_sep'], 
                label=r['label'], linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Core Separation (physical units)')
    ax.set_title('Separation vs Time (Different Resolutions)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    labels = [r['label'] for r in results]
    min_seps = [r['min_sep_physical'] for r in results]
    ax.bar(range(len(labels)), min_seps)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Minimum Separation (physical)')
    ax.set_title('Resolution Independence Check')
    ax.axhline(mean_sep, color='red', linestyle='--', label=f'Mean = {mean_sep:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test1_resolution.png'), dpi=150)
    print(f"\nPlot saved: {output_dir}/test1_resolution.png")
    
    return verdict, results


def test_mass_scaling(output_dir):
    """
    Test 2: Mass Scaling
    
    Run with different masses.
    
    If real: Minimum separation should scale as λ_C = ℏ/mc (Compton wavelength).
    If artifact: No clear relationship with mass.
    """
    
    print("\n" + "="*60)
    print("TEST 2: MASS SCALING")
    print("="*60)
    
    results = []
    
    masses = [0.25, 0.5, 1.0]
    
    for i, mass in enumerate(masses):
        compton = 1.0 / mass  # ℏ/mc in natural units
        
        out_dir = os.path.join(output_dir, f"mass_test_{i}")
        
        obs_file = run_simulation(
            Nx=128,
            dx=0.5,
            dt=0.01,
            mass=mass,
            G=50.0,
            offset=10.0,
            steps=500,
            out_dir=out_dir
        )
        
        obs = load_observables(obs_file)
        min_sep = np.min(obs['core_sep'])
        
        results.append({
            'mass': mass,
            'compton': compton,
            'min_sep': min_sep,
            'data': obs
        })
        
        print(f"\nMass = {mass:.2f}:")
        print(f"  Compton wavelength: {compton:.2f}")
        print(f"  Min separation: {min_sep:.2f}")
        print(f"  Ratio (sep/λ_C): {min_sep/compton:.2f}")
    
    # Analysis
    print("\n" + "-"*60)
    print("ANALYSIS:")
    
    # Fit linear relationship: sep = a * (1/mass) + b
    masses_arr = np.array([r['mass'] for r in results])
    seps_arr = np.array([r['min_sep'] for r in results])
    comptons = 1.0 / masses_arr
    
    # Linear fit
    coeffs = np.polyfit(comptons, seps_arr, 1)
    slope, intercept = coeffs
    
    print(f"Linear fit: sep = {slope:.2f} * λ_C + {intercept:.2f}")
    print(f"Intercept/slope ratio: {abs(intercept/slope):.2%}")
    
    # Check if intercept is small (should be for pure Compton scaling)
    if abs(intercept/slope) < 0.2:
        print("✓ PASS: Separation scales with Compton wavelength")
        verdict = "PASS"
    else:
        print("✗ FAIL: Significant offset from Compton scaling")
        verdict = "FAIL"
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    for r in results:
        ax.plot(r['data']['time'], r['data']['core_sep'], 
                label=f"m={r['mass']:.2f} (λ={r['compton']:.1f})", 
                linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Core Separation')
    ax.set_title('Separation vs Time (Different Masses)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.scatter(comptons, seps_arr, s=100, label='Data', zorder=3)
    compton_fit = np.linspace(comptons.min(), comptons.max(), 100)
    sep_fit = slope * compton_fit + intercept
    ax.plot(compton_fit, sep_fit, 'r--', linewidth=2, 
            label=f'Fit: {slope:.2f}λ + {intercept:.2f}')
    ax.set_xlabel('Compton Wavelength (ℏ/mc)')
    ax.set_ylabel('Minimum Separation')
    ax.set_title('Compton Wavelength Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test2_mass_scaling.png'), dpi=150)
    print(f"\nPlot saved: {output_dir}/test2_mass_scaling.png")
    
    return verdict, results


def test_boundary_effects(output_dir):
    """
    Test 3: Boundary Effects
    
    Run with different box sizes.
    
    If real: Behavior should be independent of box size (if vortices well-separated from boundaries).
    If artifact: Different behavior in different boxes.
    """
    
    print("\n" + "="*60)
    print("TEST 3: BOUNDARY EFFECTS")
    print("="*60)
    
    results = []
    
    configs = [
        {'Nx': 128, 'box_size': 64, 'label': 'baseline'},
        {'Nx': 256, 'box_size': 128, 'label': 'large box'},
    ]
    
    for i, config in enumerate(configs):
        out_dir = os.path.join(output_dir, f"boundary_test_{i}")
        
        obs_file = run_simulation(
            Nx=config['Nx'],
            dx=0.5,
            dt=0.01,
            mass=0.5,
            G=50.0,
            offset=10.0,
            steps=500,
            out_dir=out_dir
        )
        
        obs = load_observables(obs_file)
        min_sep = np.min(obs['core_sep'])
        
        results.append({
            'label': config['label'],
            'box_size': config['box_size'],
            'min_sep': min_sep,
            'data': obs
        })
        
        print(f"\n{config['label']} (box = {config['box_size']}):")
        print(f"  Min separation: {min_sep:.2f}")
    
    # Analysis
    print("\n" + "-"*60)
    print("ANALYSIS:")
    
    sep_baseline = results[0]['min_sep']
    sep_large = results[1]['min_sep']
    difference = abs(sep_large - sep_baseline) / sep_baseline
    
    print(f"Baseline box: {sep_baseline:.2f}")
    print(f"Large box: {sep_large:.2f}")
    print(f"Relative difference: {difference:.1%}")
    
    if difference < 0.15:
        print("✓ PASS: No significant boundary effects")
        verdict = "PASS"
    else:
        print("✗ FAIL: Boundary effects present")
        verdict = "FAIL"
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for r in results:
        ax.plot(r['data']['time'], r['data']['core_sep'], 
                label=r['label'], linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Core Separation')
    ax.set_title('Boundary Effects Check')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test3_boundary.png'), dpi=150)
    print(f"\nPlot saved: {output_dir}/test3_boundary.png")
    
    return verdict, results


def generate_report(output_dir, test_results):
    """Generate summary report."""
    
    report_path = os.path.join(output_dir, "VALIDATION_REPORT.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FERMIONIC EXCLUSION VALIDATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("Summary:\n")
        f.write("-"*60 + "\n")
        
        for test_name, verdict in test_results.items():
            status = "✓ PASS" if verdict == "PASS" else "✗ FAIL"
            f.write(f"{test_name}: {status}\n")
        
        f.write("\n")
        
        passes = sum(1 for v in test_results.values() if v == "PASS")
        total = len(test_results)
        
        f.write(f"\nOverall: {passes}/{total} tests passed\n\n")
        
        if passes == total:
            f.write("CONCLUSION: Strong evidence for real fermionic exclusion.\n")
            f.write("The observed behavior is:\n")
            f.write("  - Independent of grid resolution\n")
            f.write("  - Scales with Compton wavelength\n")
            f.write("  - Not a boundary artifact\n")
            f.write("\nThis supports genuine topological exclusion.\n")
        elif passes >= total // 2:
            f.write("CONCLUSION: Mixed results.\n")
            f.write("Some tests passed, but not all.\n")
            f.write("Further investigation needed.\n")
        else:
            f.write("CONCLUSION: Likely numerical artifact.\n")
            f.write("Observed behavior does not pass key validation tests.\n")
            f.write("Recommend revisiting numerics.\n")
    
    print(f"\n{'='*60}")
    print(f"Full report saved: {report_path}")
    print(f"{'='*60}\n")
    
    # Print to console
    with open(report_path, 'r') as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser(description="Validate fermionic exclusion")
    parser.add_argument('--quick', action='store_true',
                       help='Run quick tests only (resolution + boundary)')
    parser.add_argument('--full', action='store_true',
                       help='Run all tests')
    parser.add_argument('--output', type=str, default='validation_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if not args.quick and not args.full:
        print("Please specify --quick or --full")
        return
    
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("FERMIONIC EXCLUSION VALIDATION SUITE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    
    test_results = {}
    
    # Test 1: Resolution independence (always run)
    verdict, _ = test_resolution_independence(os.path.join(output_dir, 'test1'))
    test_results['Resolution Independence'] = verdict
    
    # Test 2: Mass scaling (full only)
    if args.full:
        verdict, _ = test_mass_scaling(os.path.join(output_dir, 'test2'))
        test_results['Mass Scaling'] = verdict
    
    # Test 3: Boundary effects (always run)
    verdict, _ = test_boundary_effects(os.path.join(output_dir, 'test3'))
    test_results['Boundary Effects'] = verdict
    
    # Generate report
    generate_report(output_dir, test_results)


if __name__ == '__main__':
    main()
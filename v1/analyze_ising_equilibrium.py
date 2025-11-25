#!/usr/bin/env python3
"""
analyze_ising_equilibrium.py

Analyze whether Ising + defrag system reaches dynamic equilibrium
or is kinetically arrested (true glass).

Key distinctions:
- Dynamic equilibrium: Bounded fluctuations, thermal sampling, stable statistics
- Kinetic arrest: Frozen, no fluctuations, incomplete relaxation

Usage:
    python analyze_ising_equilibrium.py <path_to_diagnostics.csv>
    python analyze_ising_equilibrium.py exp4_coarse_grain/cg2/diagnostics.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy import stats
from scipy.signal import welch


def load_diagnostics(csv_path):
    """Load diagnostics CSV."""
    df = pd.read_csv(csv_path)
    return df


def detect_equilibration(df, observable='E_bind', window=100, threshold=0.1):
    """
    Detect when system equilibrates using running statistics.
    
    Returns the sweep index where equilibration is reached.
    Uses criterion: when running mean stabilizes (std < threshold).
    """
    values = df[observable].values
    n = len(values)
    
    running_means = []
    for i in range(window, n):
        running_means.append(np.mean(values[i-window:i]))
    
    running_means = np.array(running_means)
    
    # Find where running mean stabilizes
    running_std = []
    for i in range(window, len(running_means)):
        running_std.append(np.std(running_means[i-window:i]))
    
    running_std = np.array(running_std)
    
    # Find first point where std < threshold * mean
    if len(running_std) > 0:
        mean_val = np.abs(np.mean(running_means))
        relative_std = running_std / mean_val
        
        equilibrated = np.where(relative_std < threshold)[0]
        if len(equilibrated) > 0:
            equil_idx = equilibrated[0] + 2 * window  # Account for windowing
            return min(equil_idx, n - window)
    
    return window  # Default to after initial window


def analyze_fluctuations(df, equil_sweep, observables=['M', 'M_abs', 'n_walls', 'E_bind']):
    """
    Analyze fluctuation statistics after equilibration.
    
    Returns dict with statistics for each observable.
    """
    # Extract equilibrated region
    df_equil = df[df['sweep'] >= equil_sweep].copy()
    
    results = {}
    
    for obs in observables:
        values = df_equil[obs].values
        
        results[obs] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'relative_std': np.std(values) / (np.abs(np.mean(values)) + 1e-10),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'n_samples': len(values)
        }
        
        # Test for stationarity (mean shouldn't drift)
        # Split into two halves and compare means
        mid = len(values) // 2
        mean1 = np.mean(values[:mid])
        mean2 = np.mean(values[mid:])
        drift = np.abs(mean2 - mean1) / (np.abs(mean1) + 1e-10)
        results[obs]['drift'] = drift
        
        # Autocorrelation time (measure of decorrelation)
        if len(values) > 10:
            autocorr = np.correlate(values - np.mean(values), 
                                   values - np.mean(values), 
                                   mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Find where autocorr drops below 1/e
            try:
                tau_idx = np.where(autocorr < 1/np.e)[0][0]
                results[obs]['autocorr_time'] = tau_idx
            except:
                results[obs]['autocorr_time'] = len(values)
    
    return results


def test_thermal_distribution(df, equil_sweep, observable='E_bind', kT=1.0):
    """
    Test if fluctuations are consistent with thermal distribution.
    
    For canonical ensemble, energy fluctuations should have:
        <(E - <E>)^2> = kT^2 * C_v
    where C_v is heat capacity.
    
    Returns bool indicating if distribution looks thermal.
    """
    df_equil = df[df['sweep'] >= equil_sweep].copy()
    values = df_equil[observable].values
    
    mean_val = np.mean(values)
    variance = np.var(values)
    
    # For large system, thermal fluctuations scale with sqrt(N)
    # Check if fluctuations are of thermal scale
    thermal_scale = np.sqrt(np.abs(mean_val) * kT)
    observed_scale = np.sqrt(variance)
    
    ratio = observed_scale / thermal_scale
    
    # Also check if distribution is roughly Gaussian (thermal)
    _, p_value = stats.normaltest(values)
    
    return {
        'thermal_ratio': ratio,
        'gaussian_p_value': p_value,
        'is_thermal': 0.1 < ratio < 10 and p_value > 0.01,
        'variance': variance,
        'thermal_scale': thermal_scale
    }


def compute_power_spectrum(df, equil_sweep, observable='M'):
    """
    Compute power spectrum of fluctuations.
    
    White noise → flat spectrum (equilibrium)
    1/f noise → power-law (non-equilibrium, aging)
    """
    df_equil = df[df['sweep'] >= equil_sweep].copy()
    values = df_equil[observable].values
    
    # Detrend
    values = values - np.mean(values)
    
    # Compute power spectrum
    freqs, power = welch(values, fs=1.0, nperseg=min(256, len(values)//4))
    
    # Fit power law: P(f) ~ f^alpha
    # alpha ~ 0: white noise (equilibrium)
    # alpha ~ -1: pink noise (1/f, non-equilibrium)
    log_f = np.log10(freqs[1:])  # Skip DC component
    log_p = np.log10(power[1:])
    
    slope, intercept, r_value, _, _ = stats.linregress(log_f, log_p)
    
    return {
        'freqs': freqs,
        'power': power,
        'slope': slope,
        'r_squared': r_value**2,
        'is_white': np.abs(slope) < 0.5  # Flat spectrum
    }


def analyze_interface_dynamics(df, equil_sweep):
    """
    Analyze domain wall dynamics specifically.
    
    In equilibrium: walls fluctuate but don't systematically grow/shrink
    In non-equilibrium: walls drift (coarsening)
    """
    df_equil = df[df['sweep'] >= equil_sweep].copy()
    walls = df_equil['n_walls'].values
    
    # Test for drift (coarsening would show decay)
    sweeps = df_equil['sweep'].values
    slope, intercept, r_value, p_value, _ = stats.linregress(sweeps, walls)
    
    # Normalized slope (change per 1000 sweeps)
    slope_normalized = slope * 1000
    
    # If walls are essentially constant, that's perfectly stable
    # (std=0 causes NaN p-value from regression, which shouldn't mean unstable)
    if np.std(walls) < 0.5:
        is_stable = True
    else:
        is_stable = np.abs(slope_normalized) < 1.0 and p_value > 0.05
    
    return {
        'initial_walls': walls[0],
        'final_walls': walls[-1],
        'mean_walls': np.mean(walls),
        'std_walls': np.std(walls),
        'drift_rate': slope_normalized,
        'drift_p_value': p_value,
        'is_stable': is_stable
    }


def classify_state(results_dict):
    """
    Classify system state based on all analyses.
    
    Returns: 'equilibrium', 'kinetic_arrest', or 'non_equilibrium'
    """
    # Extract key indicators
    energy_drift = results_dict['fluctuations']['E_bind']['drift']
    walls_stable = results_dict['interface']['is_stable']
    is_thermal = results_dict['thermal']['is_thermal']
    is_white = results_dict['spectrum']['is_white']
    
    # Decision tree
    if energy_drift < 0.05 and walls_stable and is_thermal:
        return 'dynamic_equilibrium'
    elif energy_drift < 0.01 and results_dict['fluctuations']['M']['std'] < 0.01:
        return 'kinetic_arrest'  # Frozen
    else:
        return 'non_equilibrium'  # Still relaxing


def create_diagnostic_plots(df, equil_sweep, results, output_dir):
    """Create comprehensive diagnostic plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Time series with equilibration marker
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Magnetization
    axes[0, 0].plot(df['sweep'], df['M'], linewidth=0.5, alpha=0.7)
    axes[0, 0].axvline(equil_sweep, color='red', linestyle='--', 
                       label=f'Equilibration (sweep {equil_sweep})')
    axes[0, 0].set_xlabel('Sweep')
    axes[0, 0].set_ylabel('Magnetization M')
    axes[0, 0].set_title('Magnetization Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
    
    # Domain walls
    axes[0, 1].plot(df['sweep'], df['n_walls'], linewidth=1, alpha=0.7)
    axes[0, 1].axvline(equil_sweep, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Sweep')
    axes[0, 1].set_ylabel('Number of walls')
    axes[0, 1].set_title('Domain Wall Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Binding energy
    axes[1, 0].plot(df['sweep'], df['E_bind'], linewidth=0.5, alpha=0.7)
    axes[1, 0].axvline(equil_sweep, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Sweep')
    axes[1, 0].set_ylabel('E_bind')
    axes[1, 0].set_title('Defrag Binding Energy')
    axes[1, 0].grid(True, alpha=0.3)
    
    # |M| with fluctuations highlighted
    df_equil = df[df['sweep'] >= equil_sweep]
    axes[1, 1].plot(df['sweep'], df['M_abs'], linewidth=0.5, alpha=0.7, label='|M|')
    axes[1, 1].axvline(equil_sweep, color='red', linestyle='--')
    axes[1, 1].axhline(results['fluctuations']['M_abs']['mean'], 
                       color='green', linestyle='--', label='Mean (equilibrium)')
    axes[1, 1].fill_between(df_equil['sweep'],
                            results['fluctuations']['M_abs']['mean'] - 
                            results['fluctuations']['M_abs']['std'],
                            results['fluctuations']['M_abs']['mean'] + 
                            results['fluctuations']['M_abs']['std'],
                            alpha=0.3, color='green', label='±1σ')
    axes[1, 1].set_xlabel('Sweep')
    axes[1, 1].set_ylabel('|M|')
    axes[1, 1].set_title('Order Parameter Fluctuations')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'time_series_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'time_series_analysis.png'}")
    
    # Figure 2: Fluctuation statistics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    df_equil = df[df['sweep'] >= equil_sweep]
    
    # Histogram of M (should be centered at 0 for stripes)
    axes[0, 0].hist(df_equil['M'], bins=30, density=True, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', label='M=0 (stripe state)')
    axes[0, 0].set_xlabel('Magnetization M')
    axes[0, 0].set_ylabel('Probability density')
    axes[0, 0].set_title('Magnetization Distribution (Equilibrium)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of walls (should be peaked around 128)
    axes[0, 1].hist(df_equil['n_walls'], bins=20, density=True, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(results['fluctuations']['n_walls']['mean'], 
                       color='red', linestyle='--', label=f"Mean = {results['fluctuations']['n_walls']['mean']:.1f}")
    axes[0, 1].set_xlabel('Number of walls')
    axes[0, 1].set_ylabel('Probability density')
    axes[0, 1].set_title('Wall Count Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Energy distribution
    axes[1, 0].hist(df_equil['E_bind'], bins=30, density=True, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(results['fluctuations']['E_bind']['mean'],
                       color='red', linestyle='--', label=f"Mean = {results['fluctuations']['E_bind']['mean']:.2e}")
    axes[1, 0].set_xlabel('E_bind')
    axes[1, 0].set_ylabel('Probability density')
    axes[1, 0].set_title('Binding Energy Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot (test for Gaussian distribution)
    stats.probplot(df_equil['E_bind'], dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot: E_bind vs Normal Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fluctuation_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'fluctuation_distributions.png'}")
    
    # Figure 3: Power spectrum
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # M power spectrum
    freqs = results['spectrum']['freqs']
    power = results['spectrum']['power']
    
    axes[0].loglog(freqs[1:], power[1:], 'o-', markersize=4, alpha=0.7)
    axes[0].set_xlabel('Frequency (1/sweep)')
    axes[0].set_ylabel('Power')
    axes[0].set_title(f"M Power Spectrum (slope={results['spectrum']['slope']:.2f})")
    axes[0].grid(True, alpha=0.3, which='both')
    
    # Add reference lines
    f_range = freqs[1:]
    axes[0].plot(f_range, power[1] * (f_range/f_range[0])**(-1), 
                'r--', alpha=0.5, label='1/f (pink noise)')
    axes[0].plot(f_range, power[1] * np.ones_like(f_range), 
                'g--', alpha=0.5, label='flat (white noise)')
    axes[0].legend()
    
    # Autocorrelation function
    values = df_equil['M'].values - np.mean(df_equil['M'].values)
    autocorr = np.correlate(values, values, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]
    
    lags = np.arange(len(autocorr))
    axes[1].plot(lags[:min(200, len(lags))], 
                autocorr[:min(200, len(autocorr))], linewidth=2)
    axes[1].axhline(1/np.e, color='red', linestyle='--', 
                    label=f"τ = {results['fluctuations']['M']['autocorr_time']} sweeps")
    axes[1].set_xlabel('Lag (sweeps)')
    axes[1].set_ylabel('Autocorrelation')
    axes[1].set_title('Magnetization Autocorrelation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'power_spectrum_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'power_spectrum_analysis.png'}")


def print_summary(results, classification):
    """Print human-readable summary of analysis."""
    print("\n" + "="*70)
    print("EQUILIBRIUM ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nSYSTEM CLASSIFICATION: {classification.upper()}")
    
    print("\n--- Equilibration ---")
    print(f"Equilibration sweep: {results['equil_sweep']}")
    
    print("\n--- Fluctuation Statistics (Post-Equilibration) ---")
    for obs in ['M', 'M_abs', 'n_walls', 'E_bind']:
        stats = results['fluctuations'][obs]
        print(f"\n{obs}:")
        print(f"  Mean:         {stats['mean']:+.6f}")
        print(f"  Std:          {stats['std']:.6f}")
        print(f"  Relative std: {stats['relative_std']:.4%}")
        print(f"  Range:        [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"  Drift:        {stats['drift']:.4%}")
        print(f"  τ_corr:       {stats['autocorr_time']} sweeps")
    
    print("\n--- Thermal Distribution Test ---")
    thermal = results['thermal']
    print(f"Thermal ratio:     {thermal['thermal_ratio']:.3f}")
    print(f"Gaussian p-value:  {thermal['gaussian_p_value']:.4f}")
    print(f"Is thermal?        {thermal['is_thermal']}")
    
    print("\n--- Power Spectrum ---")
    spectrum = results['spectrum']
    print(f"Spectral slope:    {spectrum['slope']:.3f}")
    print(f"Is white noise?    {spectrum['is_white']}")
    
    print("\n--- Interface Dynamics ---")
    interface = results['interface']
    print(f"Mean walls:        {interface['mean_walls']:.1f} ± {interface['std_walls']:.1f}")
    print(f"Drift rate:        {interface['drift_rate']:.3f} walls/1000 sweeps")
    print(f"Drift p-value:     {interface['drift_p_value']:.4f}")
    print(f"Is stable?         {interface['is_stable']}")
    
    print("\n--- Interpretation ---")
    if classification == 'dynamic_equilibrium':
        print("✓ System has reached DYNAMIC EQUILIBRIUM")
        print("  - Energy and structure fluctuate but don't drift")
        print("  - Fluctuations consistent with thermal distribution")
        print("  - Interface is stable (no coarsening)")
        print("  - This is a thermodynamically stable phase!")
    elif classification == 'kinetic_arrest':
        print("✓ System is KINETICALLY ARRESTED (true glass)")
        print("  - Fluctuations are negligible")
        print("  - System is frozen in metastable state")
        print("  - Not thermally equilibrated")
    else:
        print("⚠ System is still in NON-EQUILIBRIUM relaxation")
        print("  - Energy or structure still drifting")
        print("  - Need longer simulation time")
    
    print("\n" + "="*70)


def save_results_json(results, classification, output_file):
    """Save results to JSON for programmatic access."""
    import json
    
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        else:
            return obj
    
    results_serializable = convert(results)
    results_serializable['classification'] = classification
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Saved results to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Ising + defrag equilibrium dynamics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_ising_equilibrium.py exp4_coarse_grain/cg2/diagnostics.csv
  python analyze_ising_equilibrium.py exp4_coarse_grain/cg2/diagnostics.csv --output analysis_cg2
        """
    )
    parser.add_argument('diagnostics', type=str, help='Path to diagnostics.csv')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output directory (default: same as input)')
    parser.add_argument('--kT', type=float, default=1.0, 
                       help='Temperature for thermal analysis')
    
    args = parser.parse_args()
    
    # Load data
    df = load_diagnostics(args.diagnostics)
    
    # Determine output directory
    if args.output is None:
        output_dir = Path(args.diagnostics).parent / 'equilibrium_analysis'
    else:
        output_dir = Path(args.output)
    
    print(f"Analyzing: {args.diagnostics}")
    print(f"Output to: {output_dir}")
    
    # Run analyses
    print("\nDetecting equilibration...")
    equil_sweep = detect_equilibration(df, observable='E_bind', window=50, threshold=0.02)
    
    print(f"Equilibration detected at sweep {equil_sweep}")
    print(f"Analyzing {len(df[df['sweep'] >= equil_sweep])} post-equilibration points...")
    
    results = {}
    results['equil_sweep'] = equil_sweep
    
    print("\nComputing fluctuation statistics...")
    results['fluctuations'] = analyze_fluctuations(df, equil_sweep)
    
    print("Testing thermal distribution...")
    results['thermal'] = test_thermal_distribution(df, equil_sweep, kT=args.kT)
    
    print("Computing power spectrum...")
    results['spectrum'] = compute_power_spectrum(df, equil_sweep, observable='M')
    
    print("Analyzing interface dynamics...")
    results['interface'] = analyze_interface_dynamics(df, equil_sweep)
    
    print("\nClassifying system state...")
    classification = classify_state(results)
    
    # Print summary
    print_summary(results, classification)
    
    # Create plots
    print("\nGenerating diagnostic plots...")
    create_diagnostic_plots(df, equil_sweep, results, output_dir)
    
    # Save results
    save_results_json(results, classification, output_dir / 'equilibrium_results.json')
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
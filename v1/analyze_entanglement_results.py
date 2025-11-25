#!/usr/bin/env python
"""
analyze_entanglement_results.py

Analyze and visualize entanglement measurement results from quantum substrate simulations.
"""

import numpy as np
import json
import argparse
import os
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARNING] Matplotlib not available - plotting disabled")


def load_evolution_results(results_dir):
    """Load entanglement evolution data from JSON."""
    json_file = os.path.join(results_dir, 'entanglement_evolution.json')
    
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Results file not found: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return data


def plot_entropy_evolution(data, output_file=None):
    """Plot entanglement entropy vs time."""
    if not MATPLOTLIB_AVAILABLE:
        print("[ERROR] Matplotlib required for plotting")
        return
    
    times = np.array(data['times'])
    entropies = np.array(data['entropies'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(times, entropies, 'b-', linewidth=2, label='S(pattern)')
    ax.axhline(entropies[0], color='gray', linestyle='--', alpha=0.5, label=f'Initial: {entropies[0]:.3f}')
    ax.axhline(entropies[-1], color='red', linestyle='--', alpha=0.5, label=f'Final: {entropies[-1]:.3f}')
    
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Entanglement Entropy S', fontsize=14)
    ax.set_title('Pattern-Environment Entanglement Evolution', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add parameters as text
    params = data.get('parameters', {})
    param_text = f"Nx={params.get('Nx', '?')}×{params.get('Ny', '?')}, "
    param_text += f"w={params.get('winding', '?')}, "
    param_text += f"J_nn={params.get('J_nn', '?')}, J_defrag={params.get('J_defrag', '?')}"
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"[SAVED] Plot → {output_file}")
    else:
        plt.show()
    
    plt.close()


def analyze_entanglement_growth(data):
    """Analyze how entanglement grows over time."""
    times = np.array(data['times'])
    entropies = np.array(data['entropies'])
    
    # Initial and final entropy
    S_initial = entropies[0]
    S_final = entropies[-1]
    delta_S = S_final - S_initial
    
    # Growth rate (linear fit to middle section)
    mid_start = len(entropies) // 4
    mid_end = 3 * len(entropies) // 4
    if mid_end > mid_start:
        t_mid = times[mid_start:mid_end]
        S_mid = entropies[mid_start:mid_end]
        growth_rate = np.polyfit(t_mid, S_mid, 1)[0]  # Slope
    else:
        growth_rate = 0.0
    
    # Saturation: has entropy plateaued?
    if len(entropies) > 10:
        last_10_var = np.var(entropies[-10:])
        saturated = last_10_var < 0.01
    else:
        saturated = False
    
    # Monogamy estimate: if S_final is high, pattern is highly entangled
    # For a pattern with n sites and local_dim=2, max entropy is n*log(2)
    params = data.get('parameters', {})
    n_pattern_sites = len(data.get('pattern_sites', []))
    if n_pattern_sites > 0:
        S_max = n_pattern_sites * np.log(2)
        entanglement_fraction = S_final / S_max if S_max > 0 else 0
    else:
        S_max = None
        entanglement_fraction = None
    
    analysis = {
        'S_initial': S_initial,
        'S_final': S_final,
        'delta_S': delta_S,
        'growth_rate': growth_rate,
        'saturated': saturated,
        'S_max_theoretical': S_max,
        'entanglement_fraction': entanglement_fraction
    }
    
    return analysis


def compare_winding_numbers(results_dirs, winding_numbers, output_file=None):
    """
    Compare entanglement evolution for different winding numbers.
    
    Args:
        results_dirs: List of directories containing results
        winding_numbers: List of corresponding winding numbers
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[ERROR] Matplotlib required for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_dirs)))
    
    for i, (results_dir, w) in enumerate(zip(results_dirs, winding_numbers)):
        try:
            data = load_evolution_results(results_dir)
            times = np.array(data['times'])
            entropies = np.array(data['entropies'])
            
            ax.plot(times, entropies, color=colors[i], linewidth=2,
                   label=f'w={w} (final S={entropies[-1]:.3f})')
        except FileNotFoundError:
            print(f"[WARNING] Results not found for w={w} in {results_dir}")
            continue
    
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Entanglement Entropy S', fontsize=14)
    ax.set_title('Entanglement Evolution: Comparison of Winding Numbers', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"[SAVED] Comparison plot → {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze quantum entanglement results")
    parser.add_argument("results_dir", type=str, help="Directory with evolution results")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--compare", type=str, nargs='+', help="Compare multiple results directories")
    parser.add_argument("--windings", type=int, nargs='+', help="Winding numbers for comparison")
    parser.add_argument("--output", type=str, help="Output file for plots")
    
    args = parser.parse_args()
    
    if args.compare:
        # Comparison mode
        if not args.windings or len(args.windings) != len(args.compare):
            print("[ERROR] Must provide same number of winding numbers as directories")
            return
        
        print("\n=== Comparing Results Across Winding Numbers ===\n")
        compare_winding_numbers(args.compare, args.windings, args.output)
        
        # Print individual analyses
        for results_dir, w in zip(args.compare, args.windings):
            try:
                data = load_evolution_results(results_dir)
                analysis = analyze_entanglement_growth(data)
                
                print(f"\n--- Winding w={w} ---")
                print(f"Initial entropy: S(0) = {analysis['S_initial']:.6f}")
                print(f"Final entropy:   S(T) = {analysis['S_final']:.6f}")
                print(f"Change: ΔS = {analysis['delta_S']:.6f}")
                print(f"Growth rate: dS/dt ≈ {analysis['growth_rate']:.6f}")
                print(f"Saturated: {analysis['saturated']}")
                if analysis['entanglement_fraction'] is not None:
                    print(f"Entanglement fraction: {analysis['entanglement_fraction']:.2%}")
            except Exception as e:
                print(f"[ERROR] Failed to analyze w={w}: {e}")
    
    else:
        # Single result analysis
        print(f"\n=== Analyzing Results from {args.results_dir} ===\n")
        
        try:
            data = load_evolution_results(args.results_dir)
            analysis = analyze_entanglement_growth(data)
            
            print("Parameters:")
            params = data.get('parameters', {})
            for key, val in params.items():
                print(f"  {key}: {val}")
            
            print("\nEntanglement Analysis:")
            print(f"  Initial entropy: S(0) = {analysis['S_initial']:.6f}")
            print(f"  Final entropy:   S(T) = {analysis['S_final']:.6f}")
            print(f"  Change: ΔS = {analysis['delta_S']:.6f}")
            print(f"  Growth rate: dS/dt ≈ {analysis['growth_rate']:.6f}")
            print(f"  Saturated: {analysis['saturated']}")
            
            if analysis['S_max_theoretical'] is not None:
                print(f"  Maximum possible entropy: S_max = {analysis['S_max_theoretical']:.6f}")
                print(f"  Entanglement fraction: {analysis['entanglement_fraction']:.2%}")
            
            # Monogamy assessment
            print("\nMonogamy Assessment:")
            if analysis['entanglement_fraction'] and analysis['entanglement_fraction'] > 0.7:
                print("  ✓ High entanglement - pattern strongly entangled with environment")
                print("  → Likely monogamous if entanglement is with single environment region")
            elif analysis['entanglement_fraction'] and analysis['entanglement_fraction'] > 0.3:
                print("  ⚠ Moderate entanglement - partial entanglement with environment")
            else:
                print("  ✗ Low entanglement - pattern not strongly entangled")
                print("  → Unlikely to exhibit monogamous structure")
            
            if args.plot:
                output_file = args.output if args.output else os.path.join(args.results_dir, "entropy_evolution.png")
                plot_entropy_evolution(data, output_file)
        
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== Analysis Complete ===\n")


if __name__ == "__main__":
    main()
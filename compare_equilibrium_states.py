#!/usr/bin/env python3
"""
compare_equilibrium_states.py

Compare equilibrium analysis across multiple Ising + defrag runs.
Creates summary tables and comparison plots.

Usage:
    python compare_equilibrium_states.py exp4_coarse_grain/*/diagnostics.csv
    python compare_equilibrium_states.py exp2_temperature/*/diagnostics.csv --output temp_comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import glob

# Import analysis functions
from analyze_ising_equilibrium import (
    load_diagnostics, detect_equilibration, analyze_fluctuations,
    test_thermal_distribution, compute_power_spectrum,
    analyze_interface_dynamics, classify_state
)


def extract_condition_from_path(path):
    """
    Extract experimental condition from directory path.
    
    Examples:
        exp4_coarse_grain/cg2/diagnostics.csv -> {'experiment': 'coarse_grain', 'cg': 2}
        exp2_temperature/T1.5/diagnostics.csv -> {'experiment': 'temperature', 'T': 1.5}
    """
    path = Path(path)
    parts = path.parts
    
    condition = {}
    
    # Extract experiment type
    for part in parts:
        if 'coarse_grain' in part:
            condition['experiment'] = 'coarse_grain'
        elif 'temperature' in part:
            condition['experiment'] = 'temperature'
        elif 'strength' in part:
            condition['experiment'] = 'defrag_strength'
        elif 'universality' in part:
            condition['experiment'] = 'universality'
    
    # Extract parameter values
    for part in parts:
        if part.startswith('cg'):
            condition['cg'] = int(part[2:])
        elif part.startswith('T'):
            try:
                condition['T'] = float(part[1:])
            except:
                pass
        elif part.startswith('g'):
            try:
                condition['g'] = float(part[1:])
            except:
                pass
        elif part.startswith('seed'):
            condition['seed'] = int(part[4:])
    
    condition['path'] = str(path.parent)
    
    return condition


def analyze_single_run(csv_path, kT=1.0):
    """Run full equilibrium analysis on single run."""
    df = load_diagnostics(csv_path)
    
    # Detect equilibration
    equil_sweep = detect_equilibration(df, observable='E_bind', window=50, threshold=0.02)
    
    # Run analyses
    results = {
        'equil_sweep': equil_sweep,
        'fluctuations': analyze_fluctuations(df, equil_sweep),
        'thermal': test_thermal_distribution(df, equil_sweep, kT=kT),
        'spectrum': compute_power_spectrum(df, equil_sweep),
        'interface': analyze_interface_dynamics(df, equil_sweep)
    }
    
    # Classify
    classification = classify_state(results)
    results['classification'] = classification
    
    return results


def create_summary_table(all_results):
    """Create DataFrame summarizing all runs."""
    rows = []
    
    for result in all_results:
        row = {
            'path': result['condition']['path'],
            'experiment': result['condition'].get('experiment', 'unknown'),
        }
        
        # Add condition parameters
        for key in ['cg', 'T', 'g', 'seed']:
            if key in result['condition']:
                row[key] = result['condition'][key]
        
        # Add classification
        row['classification'] = result['analysis']['classification']
        
        # Add key metrics
        row['equil_sweep'] = result['analysis']['equil_sweep']
        row['final_M_abs'] = result['analysis']['fluctuations']['M_abs']['mean']
        row['sigma_M'] = result['analysis']['fluctuations']['M']['std']
        row['final_walls'] = result['analysis']['fluctuations']['n_walls']['mean']
        row['sigma_walls'] = result['analysis']['fluctuations']['n_walls']['std']
        row['E_bind'] = result['analysis']['fluctuations']['E_bind']['mean']
        row['sigma_E_bind'] = result['analysis']['fluctuations']['E_bind']['std']
        row['E_drift'] = result['analysis']['fluctuations']['E_bind']['drift']
        row['wall_drift_rate'] = result['analysis']['interface']['drift_rate']
        row['thermal_ratio'] = result['analysis']['thermal']['thermal_ratio']
        row['spectral_slope'] = result['analysis']['spectrum']['slope']
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def plot_comparison_by_parameter(df, parameter, output_dir):
    """
    Create comparison plots for runs varying a single parameter.
    
    Args:
        df: Summary DataFrame
        parameter: 'cg', 'T', 'g', etc.
        output_dir: Where to save plots
    """
    if parameter not in df.columns:
        print(f"Warning: Parameter '{parameter}' not found in data")
        return
    
    df_sorted = df.sort_values(parameter)
    param_values = df_sorted[parameter].values
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Color by classification
    colors = {'dynamic_equilibrium': 'green', 
              'kinetic_arrest': 'red', 
              'non_equilibrium': 'orange'}
    point_colors = [colors.get(c, 'gray') for c in df_sorted['classification']]
    
    # Final magnetization
    axes[0, 0].scatter(param_values, df_sorted['final_M_abs'], 
                      c=point_colors, s=100, alpha=0.7, edgecolors='black')
    axes[0, 0].set_xlabel(parameter)
    axes[0, 0].set_ylabel('Final |M|')
    axes[0, 0].set_title('Order Parameter')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Wall count
    axes[0, 1].scatter(param_values, df_sorted['final_walls'],
                      c=point_colors, s=100, alpha=0.7, edgecolors='black')
    axes[0, 1].set_xlabel(parameter)
    axes[0, 1].set_ylabel('Number of walls')
    axes[0, 1].set_title('Domain Structure')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Binding energy
    axes[0, 2].scatter(param_values, df_sorted['E_bind'],
                      c=point_colors, s=100, alpha=0.7, edgecolors='black')
    axes[0, 2].set_xlabel(parameter)
    axes[0, 2].set_ylabel('E_bind')
    axes[0, 2].set_title('Defrag Binding Energy')
    axes[0, 2].set_yscale('symlog')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Fluctuation magnitude
    axes[1, 0].scatter(param_values, df_sorted['sigma_M'],
                      c=point_colors, s=100, alpha=0.7, edgecolors='black')
    axes[1, 0].set_xlabel(parameter)
    axes[1, 0].set_ylabel('σ(M)')
    axes[1, 0].set_title('Magnetization Fluctuations')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Thermal ratio
    axes[1, 1].scatter(param_values, df_sorted['thermal_ratio'],
                      c=point_colors, s=100, alpha=0.7, edgecolors='black')
    axes[1, 1].axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Thermal')
    axes[1, 1].set_xlabel(parameter)
    axes[1, 1].set_ylabel('Thermal ratio')
    axes[1, 1].set_title('Thermal Distribution Test')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Classification
    class_map = {'dynamic_equilibrium': 2, 'kinetic_arrest': 1, 'non_equilibrium': 0}
    class_values = [class_map.get(c, -1) for c in df_sorted['classification']]
    axes[1, 2].scatter(param_values, class_values,
                      c=point_colors, s=150, alpha=0.7, edgecolors='black')
    axes[1, 2].set_xlabel(parameter)
    axes[1, 2].set_ylabel('State')
    axes[1, 2].set_yticks([0, 1, 2])
    axes[1, 2].set_yticklabels(['Non-eq', 'Arrested', 'Equilibrium'])
    axes[1, 2].set_title('System Classification')
    axes[1, 2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'comparison_vs_{parameter}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / f'comparison_vs_{parameter}.png'}")


def plot_phase_diagram_2d(df, param1, param2, output_dir):
    """
    Create 2D phase diagram if two parameters are varied.
    
    Args:
        df: Summary DataFrame
        param1, param2: Parameters to plot (e.g., 'T', 'g')
        output_dir: Where to save
    """
    if param1 not in df.columns or param2 not in df.columns:
        print(f"Cannot create 2D phase diagram: missing {param1} or {param2}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color by classification
    colors = {'dynamic_equilibrium': 'green', 
              'kinetic_arrest': 'red', 
              'non_equilibrium': 'orange'}
    point_colors = [colors.get(c, 'gray') for c in df['classification']]
    
    # Phase diagram
    scatter = axes[0].scatter(df[param1], df[param2], 
                             c=point_colors, s=150, alpha=0.7, 
                             edgecolors='black', linewidths=2)
    axes[0].set_xlabel(param1)
    axes[0].set_ylabel(param2)
    axes[0].set_title('Phase Diagram')
    axes[0].grid(True, alpha=0.3)
    
    # Add legend
    for class_name, color in colors.items():
        axes[0].scatter([], [], c=color, s=100, label=class_name, edgecolors='black')
    axes[0].legend()
    
    # Order parameter magnitude
    scatter2 = axes[1].scatter(df[param1], df[param2],
                              c=df['final_M_abs'], s=150, alpha=0.7,
                              cmap='RdBu_r', edgecolors='black', linewidths=2,
                              vmin=0, vmax=1)
    axes[1].set_xlabel(param1)
    axes[1].set_ylabel(param2)
    axes[1].set_title('Order Parameter |M|')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='|M|')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'phase_diagram_{param1}_vs_{param2}.png', 
               dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / f'phase_diagram_{param1}_vs_{param2}.png'}")


def print_comparison_summary(df):
    """Print human-readable comparison summary."""
    print("\n" + "="*70)
    print("EQUILIBRIUM COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\nTotal runs analyzed: {len(df)}")
    
    # Count classifications
    for classification in ['dynamic_equilibrium', 'kinetic_arrest', 'non_equilibrium']:
        count = sum(df['classification'] == classification)
        pct = 100 * count / len(df)
        print(f"  {classification}: {count} ({pct:.1f}%)")
    
    print("\n" + "-"*70)
    print("Key Metrics Summary:")
    print("-"*70)
    
    print(f"\n{'Condition':<30} {'Class':<15} {'|M|':<10} {'Walls':<10} {'E_bind':<12}")
    print("-"*70)
    
    for _, row in df.iterrows():
        # Create condition string
        cond_parts = []
        for key in ['cg', 'T', 'g', 'seed']:
            if key in row and pd.notna(row[key]):
                if key in ['T', 'g']:
                    cond_parts.append(f"{key}={row[key]:.2f}")
                else:
                    cond_parts.append(f"{key}={row[key]:.0f}")
        condition_str = ", ".join(cond_parts) if cond_parts else row['path']
        
        class_short = row['classification'][:8]  # Abbreviate
        
        print(f"{condition_str:<30} {class_short:<15} "
              f"{row['final_M_abs']:<10.4f} {row['final_walls']:<10.1f} "
              f"{row['E_bind']:<12.2e}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Compare equilibrium analyses across multiple runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare coarse-graining runs
  python compare_equilibrium_states.py exp4_coarse_grain/*/diagnostics.csv
  
  # Compare temperature runs
  python compare_equilibrium_states.py exp2_temperature/*/diagnostics.csv
  
  # Compare all runs matching pattern
  python compare_equilibrium_states.py exp*/*/diagnostics.csv --output comparison_all
        """
    )
    parser.add_argument('files', type=str, nargs='+',
                       help='Paths to diagnostics.csv files (can use wildcards)')
    parser.add_argument('--output', type=str, default='comparison',
                       help='Output directory')
    parser.add_argument('--kT', type=float, default=1.0,
                       help='Temperature for thermal analysis')
    
    args = parser.parse_args()
    
    # Expand wildcards
    all_files = []
    for pattern in args.files:
        all_files.extend(glob.glob(pattern))
    
    if len(all_files) == 0:
        print("ERROR: No files found matching pattern(s)")
        return
    
    print(f"Found {len(all_files)} diagnostic files to analyze")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze all runs
    all_results = []
    
    for i, csv_path in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] Analyzing: {csv_path}")
        
        try:
            condition = extract_condition_from_path(csv_path)
            analysis = analyze_single_run(csv_path, kT=args.kT)
            
            all_results.append({
                'condition': condition,
                'analysis': analysis
            })
            
            print(f"  → Classification: {analysis['classification']}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    if len(all_results) == 0:
        print("\nNo successful analyses!")
        return
    
    print(f"\nSuccessfully analyzed {len(all_results)}/{len(all_files)} runs")
    
    # Create summary table
    print("\nCreating summary table...")
    summary_df = create_summary_table(all_results)
    
    # Save table
    summary_df.to_csv(output_dir / 'comparison_summary.csv', index=False)
    print(f"Saved: {output_dir / 'comparison_summary.csv'}")
    
    # Print summary
    print_comparison_summary(summary_df)
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    
    # Determine which parameters vary
    varying_params = []
    for param in ['cg', 'T', 'g', 'seed']:
        if param in summary_df.columns and summary_df[param].nunique() > 1:
            varying_params.append(param)
    
    print(f"Varying parameters: {varying_params}")
    
    # Plot vs each varying parameter
    for param in varying_params:
        plot_comparison_by_parameter(summary_df, param, output_dir)
    
    # If exactly 2 parameters vary, make 2D phase diagram
    if len(varying_params) == 2:
        print(f"\nCreating 2D phase diagram: {varying_params[0]} vs {varying_params[1]}")
        plot_phase_diagram_2d(summary_df, varying_params[0], varying_params[1], output_dir)
    
    # Save detailed results
    results_file = output_dir / 'detailed_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved detailed results: {results_file}")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
analyze_sweep.py

Post-processing and analysis of parameter sweep results.

Provides detailed analysis beyond the automated heatmaps:
- Variance growth analysis with fits
- Power spectrum evolution
- Phase diagrams
- Detailed diagnostics for selected runs

Usage:
    python analyze_sweep.py sweep_results/
    python analyze_sweep.py sweep_results/ --analyze-best
    python analyze_sweep.py sweep_results/ --compare g1.0_lambda0.1 g2.0_lambda0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def analyze_variance_growth(df, title="Variance Growth Analysis"):
    """
    Detailed analysis of variance growth with exponential fit.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Linear plot
    axes[0].plot(df['time'], df['var_rho'], 'o-', label='Data', markersize=4)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Var(ρ)')
    axes[0].set_title(f'{title}\nVariance Growth')
    axes[0].grid(True, alpha=0.3)
    
    # Exponential fit
    t = df['time'].values
    var = df['var_rho'].values
    
    # Fit region: skip initial transient (first 20%)
    fit_start_idx = int(0.2 * len(t))
    t_fit = t[fit_start_idx:]
    var_fit = var[fit_start_idx:]
    
    if np.all(var_fit > 0):
        log_var = np.log(var_fit)
        coeffs = np.polyfit(t_fit, log_var, 1)
        omega = coeffs[0] / 2.0
        log_A = coeffs[1]
        
        # Plot fit
        t_dense = np.linspace(t[0], t[-1], 200)
        var_fit_curve = np.exp(log_A + 2*omega*t_dense)
        axes[0].plot(t_dense, var_fit_curve, '--', 
                    label=f'Fit: ω={omega:+.4f}', linewidth=2)
        axes[0].axvline(t[fit_start_idx], color='gray', 
                       linestyle=':', alpha=0.5, label='Fit region')
        axes[0].legend()
        
        # Log plot
        axes[1].semilogy(df['time'], df['var_rho'], 'o-', 
                        label='Data', markersize=4)
        axes[1].semilogy(t_dense, var_fit_curve, '--', 
                        label=f'Fit: ω={omega:+.4f}', linewidth=2)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Var(ρ)')
        axes[1].set_title('Log Scale (exponential = straight line)')
        axes[1].axvline(t[fit_start_idx], color='gray', 
                       linestyle=':', alpha=0.5)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Add text with fit quality
        residuals = log_var - (coeffs[0]*t_fit + coeffs[1])
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_var - np.mean(log_var))**2)
        R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        
        axes[1].text(0.05, 0.95, f'R² = {R2:.4f}',
                    transform=axes[1].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def analyze_structure_formation(df, title="Structure Formation Analysis"):
    """
    Multi-panel diagnostic plot for structure formation.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 1. Max density
    axes[0, 0].plot(df['time'], df['max_rho'], linewidth=2)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('max(ρ)')
    axes[0, 0].set_title('Maximum Density')
    axes[0, 0].grid(True, alpha=0.3)
    
    growth = df['max_rho'].iloc[-1] / df['max_rho'].iloc[0]
    axes[0, 0].text(0.05, 0.95, f'{growth:.2f}× growth',
                   transform=axes[0, 0].transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # 2. Variance (log scale)
    axes[0, 1].semilogy(df['time'], df['var_rho'], linewidth=2)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Var(ρ)')
    axes[0, 1].set_title('Variance (exponential growth?)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Binding energy
    axes[0, 2].plot(df['time'], df['E_defrag'], linewidth=2)
    axes[0, 2].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 2].set_xlabel('Time')
    axes[0, 2].set_ylabel('E_defrag')
    axes[0, 2].set_title('Defrag Binding Energy')
    axes[0, 2].grid(True, alpha=0.3)
    
    if df['E_defrag'].iloc[-1] < 0:
        axes[0, 2].text(0.05, 0.95, 'BOUND',
                       transform=axes[0, 2].transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 4. Participation ratio
    axes[1, 0].plot(df['time'], df['participation_ratio'], linewidth=2)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('PR')
    axes[1, 0].set_title('Participation Ratio (localization?)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Energy budget
    axes[1, 1].plot(df['time'], df['E_substrate'], label='E_substrate')
    axes[1, 1].plot(df['time'], df['E_defrag'], label='E_defrag')
    axes[1, 1].plot(df['time'], df['E_total'], label='E_total', 
                   linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Energy')
    axes[1, 1].set_title('Energy Components')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Norm conservation
    axes[1, 2].plot(df['time'], df['norm'], linewidth=2)
    axes[1, 2].axhline(df['norm'].iloc[0], color='r', 
                      linestyle='--', alpha=0.5, label='Initial')
    axes[1, 2].set_xlabel('Time')
    axes[1, 2].set_ylabel('∫|ψ|²')
    axes[1, 2].set_title('Norm Conservation')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def compare_runs(run_dirs, labels, output_file='comparison.png'):
    """
    Compare multiple runs on the same plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_dirs)))
    
    for run_dir, label, color in zip(run_dirs, labels, colors):
        df_path = Path(run_dir) / 'diagnostics.csv'
        if not df_path.exists():
            print(f"Warning: {df_path} not found, skipping")
            continue
        
        df = pd.read_csv(df_path)
        
        # Max density
        axes[0, 0].plot(df['time'], df['max_rho'], 
                       label=label, color=color, linewidth=2)
        
        # Variance (log)
        axes[0, 1].semilogy(df['time'], df['var_rho'], 
                           label=label, color=color, linewidth=2)
        
        # Binding energy
        axes[1, 0].plot(df['time'], df['E_defrag'], 
                       label=label, color=color, linewidth=2)
        
        # Participation ratio
        axes[1, 1].plot(df['time'], df['participation_ratio'], 
                       label=label, color=color, linewidth=2)
    
    # Formatting
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('max(ρ)')
    axes[0, 0].set_title('Maximum Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Var(ρ)')
    axes[0, 1].set_title('Variance (log scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('E_defrag')
    axes[1, 0].set_title('Binding Energy')
    axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('PR')
    axes[1, 1].set_title('Participation Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Run Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison to {output_file}")


def create_phase_diagram(summary_df, output_file='phase_diagram.png'):
    """
    Create phase diagram showing regions of parameter space.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Phase diagram 1: Color by omega
    scatter1 = axes[0].scatter(
        summary_df['g_defrag'],
        summary_df['lambda_param'],
        c=summary_df['omega'],
        s=200,
        cmap='RdYlGn',
        vmin=-0.1, vmax=0.2,
        edgecolors='black',
        linewidths=1.5,
        alpha=0.8
    )
    axes[0].set_xlabel('g_defrag (coupling strength)', fontsize=12)
    axes[0].set_ylabel('λ (self-interaction)', fontsize=12)
    axes[0].set_title('Phase Diagram: Growth Rate ω', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='ω (growth rate)')
    
    # Add regions
    axes[0].axhline(0.3, color='red', linestyle='--', alpha=0.3, linewidth=2)
    axes[0].text(0.5, 0.32, 'High λ: Stable', 
                transform=axes[0].get_xaxis_transform(),
                ha='center', fontsize=10, color='red')
    
    # Phase diagram 2: Color by max_rho_growth
    scatter2 = axes[1].scatter(
        summary_df['g_defrag'],
        summary_df['lambda_param'],
        c=summary_df['max_rho_growth'],
        s=200,
        cmap='hot',
        edgecolors='black',
        linewidths=1.5,
        alpha=0.8
    )
    axes[1].set_xlabel('g_defrag (coupling strength)', fontsize=12)
    axes[1].set_ylabel('λ (self-interaction)', fontsize=12)
    axes[1].set_title('Phase Diagram: Density Growth', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='max(ρ) growth factor')
    
    # Mark 3× threshold
    for idx, row in summary_df.iterrows():
        if row['max_rho_growth'] > 3.0:
            axes[1].plot(row['g_defrag'], row['lambda_param'], 
                        'c*', markersize=15, markeredgecolor='blue')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved phase diagram to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze parameter sweep results')
    parser.add_argument('sweep_dir', type=str, 
                       help='Directory containing sweep results')
    parser.add_argument('--analyze-best', action='store_true',
                       help='Create detailed analysis of best run')
    parser.add_argument('--compare', nargs='+', type=str,
                       help='Compare specific runs (e.g., g1.0_lambda0.1 g2.0_lambda0.0)')
    parser.add_argument('--phase-diagram', action='store_true',
                       help='Create phase diagram')
    
    args = parser.parse_args()
    
    sweep_dir = Path(args.sweep_dir)
    
    # Load summary
    summary_path = sweep_dir / 'sweep_summary.csv'
    if not summary_path.exists():
        print(f"Error: {summary_path} not found")
        return
    
    summary_df = pd.read_csv(summary_path)
    print(f"\nLoaded sweep summary: {len(summary_df)} runs")
    
    # Analyze best run
    if args.analyze_best:
        print("\n" + "="*70)
        print("ANALYZING BEST RUN")
        print("="*70)
        
        best_dir = sweep_dir / 'best_run'
        if best_dir.exists():
            df_best = pd.read_csv(best_dir / 'diagnostics.csv')
            
            best_row = summary_df.loc[summary_df['omega'].idxmax()]
            title = (f"Best Run: g={best_row['g_defrag']:.3f}, "
                    f"λ={best_row['lambda_param']:.3f}")
            
            # Variance growth analysis
            fig1 = analyze_variance_growth(df_best, title)
            fig1.savefig(sweep_dir / 'best_variance_analysis.png', 
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved variance analysis to {sweep_dir / 'best_variance_analysis.png'}")
            
            # Structure formation analysis
            fig2 = analyze_structure_formation(df_best, title)
            fig2.savefig(sweep_dir / 'best_structure_analysis.png',
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved structure analysis to {sweep_dir / 'best_structure_analysis.png'}")
        else:
            print(f"Warning: {best_dir} not found")
    
    # Compare specific runs
    if args.compare:
        print("\n" + "="*70)
        print("COMPARING RUNS")
        print("="*70)
        
        run_dirs = [sweep_dir / 'raw_data' / run for run in args.compare]
        labels = args.compare
        
        compare_runs(run_dirs, labels, sweep_dir / 'run_comparison.png')
    
    # Create phase diagram
    if args.phase_diagram:
        print("\n" + "="*70)
        print("CREATING PHASE DIAGRAM")
        print("="*70)
        
        create_phase_diagram(summary_df, sweep_dir / 'phase_diagram.png')
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nTotal runs: {len(summary_df)}")
    print(f"\nGrowth rate ω:")
    print(f"  Range: [{summary_df['omega'].min():+.6f}, {summary_df['omega'].max():+.6f}]")
    print(f"  Mean: {summary_df['omega'].mean():+.6f} ± {summary_df['omega'].std():.6f}")
    
    print(f"\nRuns with ω > 0 (unstable): {np.sum(summary_df['omega'] > 0)} ({100*np.sum(summary_df['omega'] > 0)/len(summary_df):.1f}%)")
    print(f"Runs with ω > 0.1 (strong instability): {np.sum(summary_df['omega'] > 0.1)}")
    
    print(f"\nMax density growth:")
    print(f"  Range: [{summary_df['max_rho_growth'].min():.4f}x, {summary_df['max_rho_growth'].max():.4f}x]")
    print(f"  Runs with >3× growth: {np.sum(summary_df['max_rho_growth'] > 3.0)}")
    
    # Best configurations
    print("\n" + "="*70)
    print("TOP 3 CONFIGURATIONS (by ω)")
    print("="*70)
    top3 = summary_df.nlargest(3, 'omega')
    for idx, row in top3.iterrows():
        print(f"\n{idx+1}. g={row['g_defrag']:.3f}, λ={row['lambda_param']:.3f}")
        print(f"   ω = {row['omega']:+.6f} (R²={row['R2']:.4f})")
        print(f"   max_rho_growth = {row['max_rho_growth']:.4f}x")
        print(f"   final E_defrag = {row['final_E_defrag']:.6e}")


if __name__ == '__main__':
    main()
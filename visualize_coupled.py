#!/usr/bin/env python3
"""
visualize_coupled.py

Visualize snapshots from coupled scalar-gauge simulation.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def load_snapshot(npz_path):
    """Load a snapshot NPZ file."""
    data = np.load(npz_path)
    return {
        'psi': data['psi'],
        'ax': data['ax'],
        'ay': data['ay'],
        'rho': data['rho'],
    }

def visualize_snapshot(npz_path, out_path=None, vmin_rho=None, vmax_rho=None):
    """Create visualization of a single snapshot."""
    data = load_snapshot(npz_path)
    
    rho = data['rho']
    ax = data['ax']
    ay = data['ay']
    
    # Gauge field amplitude
    gauge_amp = np.sqrt(ax**2 + ay**2)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Scalar density
    im0 = axes[0].imshow(rho.T, origin='lower', cmap='viridis')
    if vmin_rho is not None and vmax_rho is not None:
        im0.set_clim(vmin_rho, vmax_rho)
    axes[0].set_title(f'Scalar density ρ = |ψ|²\nmax={rho.max():.3e}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0])
    
    # Gauge potential amplitude
    im1 = axes[1].imshow(gauge_amp.T, origin='lower', cmap='plasma')
    axes[1].set_title(f'Gauge amplitude √(ax²+ay²)\nmax={gauge_amp.max():.3e}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1])
    
    # Gauge field ax (just x-component)
    im2 = axes[2].imshow(ax.T, origin='lower', cmap='RdBu_r', 
                         vmin=-np.abs(ax).max(), vmax=np.abs(ax).max())
    axes[2].set_title(f'Gauge ax\nmax={np.abs(ax).max():.3e}')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[2])
    
    plt.suptitle(f'Snapshot: {npz_path.name}', fontsize=14)
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
    else:
        plt.show()
    
    plt.close()
    
    return rho.max(), gauge_amp.max()

def compare_snapshots(npz_paths, out_path=None):
    """Compare multiple snapshots side by side."""
    n_snaps = len(npz_paths)
    fig, axes = plt.subplots(2, n_snaps, figsize=(5*n_snaps, 8))
    
    if n_snaps == 1:
        axes = axes.reshape(2, 1)
    
    for i, npz_path in enumerate(npz_paths):
        data = load_snapshot(npz_path)
        rho = data['rho']
        ax = data['ax']
        ay = data['ay']
        gauge_amp = np.sqrt(ax**2 + ay**2)
        
        # Scalar density
        im0 = axes[0, i].imshow(rho.T, origin='lower', cmap='viridis')
        axes[0, i].set_title(f'{npz_path.stem}\nρ max={rho.max():.3e}')
        axes[0, i].set_xlabel('x')
        if i == 0:
            axes[0, i].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0, i], fraction=0.046)
        
        # Gauge amplitude
        im1 = axes[1, i].imshow(gauge_amp.T, origin='lower', cmap='plasma')
        axes[1, i].set_title(f'Gauge max={gauge_amp.max():.3e}')
        axes[1, i].set_xlabel('x')
        if i == 0:
            axes[1, i].set_ylabel('y')
        plt.colorbar(im1, ax=axes[1, i], fraction=0.046)
    
    plt.suptitle('Snapshot Comparison', fontsize=16)
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
    else:
        plt.show()
    
    plt.close()

def plot_evolution(output_dir, prefix='coupled', out_path=None):
    """Plot time evolution of max values."""
    output_dir = Path(output_dir)
    
    # Find all snapshots
    snaps = sorted(output_dir.glob(f"{prefix}_snap_*.npz"))
    
    if not snaps:
        print(f"No snapshots found in {output_dir}")
        return
    
    times = []
    max_rho = []
    max_gauge = []
    
    for snap_path in snaps:
        # Extract step number from filename
        step_str = snap_path.stem.split('_')[-1]
        step = int(step_str)
        
        data = load_snapshot(snap_path)
        rho = data['rho']
        ax = data['ax']
        ay = data['ay']
        gauge_amp = np.sqrt(ax**2 + ay**2)
        
        times.append(step)
        max_rho.append(rho.max())
        max_gauge.append(gauge_amp.max())
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(times, max_rho, 'o-', label='max(ρ)')
    axes[0].set_ylabel('Max scalar density')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(times, max_gauge, 'o-', color='C1', label='max(gauge)')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Max gauge amplitude')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.suptitle('Field Evolution Over Time')
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize coupled scalar-gauge snapshots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        choices=['single', 'compare', 'evolution'],
        default='single',
        help='Visualization mode'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input NPZ file(s) or directory. For compare mode, use comma-separated list.'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='coupled_output',
        help='Output directory for evolution mode'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='coupled',
        help='Prefix for snapshot files'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output PNG file path (optional, will display if not provided)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.input:
            print("Error: --input required for single mode")
            return
        npz_path = Path(args.input)
        visualize_snapshot(npz_path, out_path=args.output)
    
    elif args.mode == 'compare':
        if not args.input:
            print("Error: --input required for compare mode")
            return
        npz_paths = [Path(p.strip()) for p in args.input.split(',')]
        compare_snapshots(npz_paths, out_path=args.output)
    
    elif args.mode == 'evolution':
        plot_evolution(args.output_dir, prefix=args.prefix, out_path=args.output)

if __name__ == "__main__":
    main()
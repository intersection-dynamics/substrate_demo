#!/usr/bin/env python3
"""
vorticity_analyzer.py

Vorticity and topological charge analysis for coupled scalar-gauge substrate.

Functions to:
- Find lump centers from density peaks
- Compute winding numbers around vortices
- Visualize circulation patterns
- Measure spin topology
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass, maximum_filter, gaussian_filter


# ======================================================================
# VORTICITY ANALYSIS FUNCTIONS
# ======================================================================

def compute_circulation_field(psi, dx):
    """
    Compute circulation vector field J = Im(psi* grad psi)
    This is the quantum current/probability current.
    
    Returns: (jx, jy) arrays of same shape as psi
    """
    # Gradient of psi using centered differences
    dpsi_dx = (np.roll(psi, -1, axis=0) - np.roll(psi, +1, axis=0)) / (2.0 * dx)
    dpsi_dy = (np.roll(psi, -1, axis=1) - np.roll(psi, +1, axis=1)) / (2.0 * dx)
    
    # J = Im(psi* grad psi)
    jx = np.imag(np.conj(psi) * dpsi_dx)
    jy = np.imag(np.conj(psi) * dpsi_dy)
    
    return jx, jy


def find_lump_centers(rho, threshold_factor=2.0, min_separation=5):
    """
    Find centers of density lumps using local maxima detection.
    
    Parameters:
    -----------
    rho : 2D array
        Density field |psi|^2
    threshold_factor : float
        Lumps must have density > threshold_factor * mean(rho)
    min_separation : int
        Minimum distance between lump centers (in grid units)
    
    Returns:
    --------
    centers : list of (x, y) tuples
        Coordinates of lump centers
    """
    # Smooth slightly to avoid noise
    rho_smooth = gaussian_filter(rho, sigma=1.0)
    
    # Find local maxima
    local_max = maximum_filter(rho_smooth, size=min_separation) == rho_smooth
    
    # Threshold: must be significantly above mean
    threshold = threshold_factor * np.mean(rho)
    peaks = local_max & (rho_smooth > threshold)
    
    # Get coordinates
    labeled, num_features = label(peaks)
    centers = center_of_mass(peaks, labeled, range(1, num_features + 1))
    
    # Convert to integer coordinates
    centers = [(int(x), int(y)) for x, y in centers]
    
    return centers


def compute_winding_number(psi, center, radius, n_points=100):
    """
    Compute topological winding number around a point.
    
    Integrates phase change around a circular contour:
    n = (1/2π) ∮ dθ
    
    Parameters:
    -----------
    psi : 2D complex array
        Complex scalar field
    center : (x, y) tuple
        Center point to compute winding around
    radius : float
        Radius of integration contour
    n_points : int
        Number of points on contour
    
    Returns:
    --------
    winding_number : float
        Topological charge (should be integer for true vortices)
    """
    cx, cy = center
    L = psi.shape[0]
    
    # Points around circle
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    phase_changes = []
    prev_phase = None
    
    for angle in angles:
        # Position on circle (with periodic boundaries)
        x = int(np.round(cx + radius * np.cos(angle))) % L
        y = int(np.round(cy + radius * np.sin(angle))) % L
        
        # Phase at this point
        phase = np.angle(psi[x, y])
        
        if prev_phase is not None:
            # Compute phase difference (unwrapped)
            dphase = phase - prev_phase
            # Unwrap: keep dphase in [-π, π]
            while dphase > np.pi:
                dphase -= 2*np.pi
            while dphase < -np.pi:
                dphase += 2*np.pi
            phase_changes.append(dphase)
        
        prev_phase = phase
    
    # Total phase change around loop
    total_phase_change = np.sum(phase_changes)
    
    # Winding number
    winding_number = total_phase_change / (2*np.pi)
    
    return winding_number


def analyze_vorticity(psi, dx, threshold_factor=2.0, min_separation=8, radius_fraction=0.4):
    """
    Complete vorticity analysis of a scalar field snapshot.
    
    Parameters:
    -----------
    psi : 2D complex array
        Complex scalar field
    dx : float
        Grid spacing
    threshold_factor : float
        Density threshold for lump detection
    min_separation : int
        Minimum separation between lumps
    radius_fraction : float
        Fraction of min_separation to use as integration radius
    
    Returns:
    --------
    results : dict containing:
        - 'centers': list of lump centers
        - 'winding_numbers': list of winding numbers for each lump
        - 'circulation': (jx, jy) circulation field
        - 'statistics': summary statistics
    """
    # Density
    rho = np.abs(psi)**2
    
    # Find lump centers
    centers = find_lump_centers(rho, threshold_factor, min_separation)
    
    # Compute winding number for each lump
    integration_radius = radius_fraction * min_separation
    winding_numbers = []
    
    for center in centers:
        w = compute_winding_number(psi, center, integration_radius)
        winding_numbers.append(w)
    
    # Compute circulation field
    jx, jy = compute_circulation_field(psi, dx)
    
    # Statistics
    winding_numbers = np.array(winding_numbers)
    stats = {
        'n_lumps': len(centers),
        'mean_winding': np.mean(np.abs(winding_numbers)) if len(winding_numbers) > 0 else 0,
        'winding_histogram': np.histogram(winding_numbers, bins=np.arange(-2.5, 2.6, 0.25)),
    }
    
    results = {
        'centers': centers,
        'winding_numbers': winding_numbers,
        'circulation': (jx, jy),
        'density': rho,
        'statistics': stats,
    }
    
    return results


# ======================================================================
# VISUALIZATION FUNCTIONS
# ======================================================================

def plot_vorticity_analysis(psi, dx, results, save_path=None):
    """
    Create comprehensive visualization of vorticity analysis.
    
    Parameters:
    -----------
    psi : 2D complex array
        Complex scalar field
    dx : float
        Grid spacing
    results : dict
        Output from analyze_vorticity()
    save_path : str, optional
        If provided, save figure to this path
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    rho = results['density']
    jx, jy = results['circulation']
    centers = results['centers']
    winding_numbers = results['winding_numbers']
    
    L = psi.shape[0]
    extent = [0, L*dx, 0, L*dx]
    
    # 1. Density with marked centers
    ax = axes[0, 0]
    im = ax.imshow(rho.T, origin='lower', extent=extent, cmap='hot')
    for (cx, cy), w in zip(centers, winding_numbers):
        color = 'cyan' if w > 0 else 'lime'
        ax.plot(cx*dx, cy*dx, 'o', color=color, markersize=10, 
                markeredgecolor='white', markeredgewidth=2)
        ax.text(cx*dx, cy*dx, f'{w:.2f}', color='white', 
                ha='center', va='center', fontsize=8, weight='bold')
    ax.set_title('Density |ψ|² with Winding Numbers\n(Cyan=+, Lime=-)', fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='ρ')
    
    # 2. Phase
    ax = axes[0, 1]
    phase = np.angle(psi)
    im = ax.imshow(phase.T, origin='lower', extent=extent, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    for (cx, cy) in centers:
        ax.plot(cx*dx, cy*dx, 'w+', markersize=15, markeredgewidth=2)
    ax.set_title('Phase arg(ψ)', fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='θ (radians)')
    
    # 3. Circulation magnitude
    ax = axes[0, 2]
    j_mag = np.sqrt(jx**2 + jy**2)
    im = ax.imshow(j_mag.T, origin='lower', extent=extent, cmap='viridis')
    ax.set_title('Circulation Magnitude |J|', fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='|J|')
    
    # 4. Circulation streamlines (zoomed on a lump if available)
    ax = axes[1, 0]
    if len(centers) > 0:
        # Pick first lump
        cx, cy = centers[0]
        zoom_size = 20
        x_slice = slice(max(0, cx-zoom_size), min(L, cx+zoom_size))
        y_slice = slice(max(0, cy-zoom_size), min(L, cy+zoom_size))
        
        rho_zoom = rho[x_slice, y_slice]
        jx_zoom = jx[x_slice, y_slice]
        jy_zoom = jy[x_slice, y_slice]
        
        x_zoom = np.arange(rho_zoom.shape[0]) + max(0, cx-zoom_size)
        y_zoom = np.arange(rho_zoom.shape[1]) + max(0, cy-zoom_size)
        X, Y = np.meshgrid(x_zoom * dx, y_zoom * dx, indexing='ij')
        
        ax.imshow(rho_zoom.T, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()], 
                  cmap='gray', alpha=0.5)
        ax.streamplot(X.T, Y.T, jx_zoom.T, jy_zoom.T, color='cyan', linewidth=1.5, density=1.5)
        ax.plot(cx*dx, cy*dx, 'r*', markersize=20)
        ax.set_title(f'Circulation Streamlines (Lump at ({cx},{cy}))\nw={winding_numbers[0]:.2f}', fontsize=14)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    else:
        ax.text(0.5, 0.5, 'No lumps found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Circulation Streamlines', fontsize=14)
    
    # 5. Winding number histogram
    ax = axes[1, 1]
    if len(winding_numbers) > 0:
        bins = np.arange(-2.5, 2.6, 0.25)
        ax.hist(winding_numbers, bins=bins, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.axvline(0.5, color='blue', linestyle='--', linewidth=1, label='±1/2')
        ax.axvline(-0.5, color='blue', linestyle='--', linewidth=1)
        ax.axvline(1.0, color='green', linestyle='--', linewidth=1, label='±1')
        ax.axvline(-1.0, color='green', linestyle='--', linewidth=1)
        ax.set_xlabel('Winding Number', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Winding Number Distribution\n(n={len(winding_numbers)} lumps)', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No lumps found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Winding Number Distribution', fontsize=14)
    
    # 6. Statistics summary
    ax = axes[1, 2]
    ax.axis('off')
    
    stats = results['statistics']
    summary_text = f"""
    VORTICITY ANALYSIS SUMMARY
    ═══════════════════════════
    
    Number of lumps found: {stats['n_lumps']}
    
    Mean |winding number|: {stats['mean_winding']:.3f}
    
    """
    
    if len(winding_numbers) > 0:
        summary_text += f"""
    Winding number range: [{winding_numbers.min():.2f}, {winding_numbers.max():.2f}]
    
    Spin classification:
    """
        # Count by ranges
        scalar = np.sum(np.abs(winding_numbers) < 0.1)
        half_integer = np.sum((np.abs(winding_numbers) > 0.3) & (np.abs(winding_numbers) < 0.7))
        integer = np.sum(np.abs(winding_numbers) > 0.9)
        
        summary_text += f"""
      Scalar-like (|w| < 0.1):  {scalar} lumps
      Half-integer (|w| ≈ 0.5): {half_integer} lumps  
      Integer (|w| ≈ 1.0):      {integer} lumps
      Other:                    {len(winding_numbers) - scalar - half_integer - integer} lumps
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VORTICITY] Saved plot -> {save_path}")
    
    return fig


def analyze_snapshot_file(npz_path, dx=1.0, save_plot=True):
    """
    Load a snapshot NPZ file and run complete vorticity analysis.
    
    Parameters:
    -----------
    npz_path : str
        Path to NPZ file containing 'psi' field
    dx : float
        Grid spacing
    save_plot : bool
        Whether to save visualization
    
    Returns:
    --------
    results : dict
        Vorticity analysis results
    """
    # Load data
    data = np.load(npz_path)
    psi = data['psi']
    
    print(f"\n{'='*60}")
    print(f"VORTICITY ANALYSIS: {npz_path}")
    print(f"{'='*60}")
    print(f"Grid size: {psi.shape}")
    print(f"dx = {dx}")
    
    # Run analysis
    results = analyze_vorticity(psi, dx)
    
    # Print results
    stats = results['statistics']
    print(f"\nFound {stats['n_lumps']} lumps")
    print(f"Mean |winding number|: {stats['mean_winding']:.3f}")
    
    if len(results['winding_numbers']) > 0:
        print(f"\nWinding numbers:")
        for i, (center, w) in enumerate(zip(results['centers'], results['winding_numbers'])):
            print(f"  Lump {i+1} at {center}: w = {w:+.3f}")
    
    # Plot
    if save_plot:
        plot_path = npz_path.replace('.npz', '_vorticity.png')
        plot_vorticity_analysis(psi, dx, results, save_path=plot_path)
        plt.close()
    else:
        plot_vorticity_analysis(psi, dx, results)
        plt.show()
    
    return results


# ======================================================================
# COMMAND LINE INTERFACE
# ======================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vorticity_analyzer.py <snapshot.npz> [--dx DX]")
        print("\nExample:")
        print("  python vorticity_analyzer.py sim_output/sim_snap_002000.npz --dx 1.0")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    
    # Parse dx if provided
    dx = 1.0
    if '--dx' in sys.argv:
        dx_idx = sys.argv.index('--dx')
        dx = float(sys.argv[dx_idx + 1])
    
    # Run analysis
    results = analyze_snapshot_file(npz_path, dx=dx, save_plot=True)
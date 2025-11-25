#!/usr/bin/env python3
"""
test_spatial_structure.py

Compare spatial structure of g=0 (no defrag) vs g=0.5 (with defrag).

CRITICAL TEST: Both have M≈0, but are they actually different?

For g=0 (no defrag):
- Random domain configuration
- Short-range correlations
- Isotropic power spectrum
- No preferred orientation

For g=0.5 (with defrag):
- Organized stripe structure
- Long-range correlations
- Anisotropic power spectrum (stripe peaks)
- Clear stripe orientation

This proves the stripe phase is STRUCTURED, not just "M=0 by accident".
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

try:
    from ising_defrag_gpu import IsingDefragGPU
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("WARNING: GPU not available, using CPU (slower)")


def compute_2d_correlation(spins):
    """
    Compute 2D spatial correlation function.
    
    C(r) = <s(x) s(x+r)> - <s>²
    
    Returns correlation as function of displacement.
    """
    if GPU_AVAILABLE and isinstance(spins, cp.ndarray):
        spins = cp.asnumpy(spins)
    
    L = spins.shape[0]
    
    # Compute 2D FFT
    ft = np.fft.fft2(spins)
    power = np.abs(ft)**2
    
    # Inverse FFT to get correlation
    corr_2d = np.fft.ifft2(power).real
    corr_2d = np.fft.fftshift(corr_2d)
    
    # Normalize
    corr_2d = corr_2d / corr_2d[L//2, L//2]
    
    return corr_2d


def compute_radial_average(array_2d):
    """Compute radially averaged profile from 2D array."""
    L = array_2d.shape[0]
    center = L // 2
    
    # Create distance array
    y, x = np.ogrid[:L, :L]
    r = np.sqrt((x - center)**2 + (y - center)**2).astype(int)
    
    # Radial bins
    r_bins = np.arange(0, L//2)
    radial_profile = np.zeros(len(r_bins))
    
    for i, r_val in enumerate(r_bins):
        mask = (r == r_val)
        if mask.sum() > 0:
            radial_profile[i] = array_2d[mask].mean()
    
    return r_bins, radial_profile


def compute_structure_factor(spins):
    """
    Compute 2D structure factor (Fourier transform of correlations).
    
    S(k) = <|s(k)|²>
    """
    if GPU_AVAILABLE and isinstance(spins, cp.ndarray):
        spins = cp.asnumpy(spins)
    
    ft = np.fft.fft2(spins)
    S_k = np.abs(ft)**2
    S_k = np.fft.fftshift(S_k)
    
    return S_k


def analyze_anisotropy(S_k):
    """
    Measure anisotropy of structure factor.
    
    For stripes: strong peaks at specific angles
    For random: isotropic (all angles similar)
    """
    L = S_k.shape[0]
    center = L // 2
    
    # Sample at fixed radius
    radius = L // 4
    angles = np.linspace(0, 2*np.pi, 360)
    
    intensities = []
    for angle in angles:
        x = int(center + radius * np.cos(angle))
        y = int(center + radius * np.sin(angle))
        if 0 <= x < L and 0 <= y < L:
            intensities.append(S_k[y, x])
    
    intensities = np.array(intensities)
    
    # Anisotropy measure: std/mean
    anisotropy = np.std(intensities) / np.mean(intensities)
    
    return angles, intensities, anisotropy


def run_structure_analysis(g_defrag, label, L=64, T=1.0, n_sweeps=2000, seed=42):
    """Run simulation and analyze spatial structure."""
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {label} (g={g_defrag})")
    print(f"{'='*70}")
    
    # Create simulator
    sim = IsingDefragGPU(
        L=L,
        T=T,
        g_defrag=g_defrag,
        coarse_grain_size=1,
    )
    
    # Run simulation
    spins = sim.create_noise_spins(flip_prob=0.2, seed=seed)
    
    print(f"Running {n_sweeps} sweeps...")
    for sweep in range(n_sweeps):
        if sweep % 500 == 0:
            print(f"  Sweep {sweep}/{n_sweeps}")
        
        Phi = sim.solve_defrag_potential(spins)
        spins = sim.metropolis_sweep(spins, Phi)
    
    # Get final state on CPU
    if GPU_AVAILABLE:
        spins_cpu = cp.asnumpy(spins)
    else:
        spins_cpu = spins
    
    print("\nComputing spatial statistics...")
    
    # Compute correlations
    corr_2d = compute_2d_correlation(spins_cpu)
    r_bins, corr_radial = compute_radial_average(corr_2d)
    
    # Compute structure factor
    S_k = compute_structure_factor(spins_cpu)
    k_bins, S_k_radial = compute_radial_average(S_k)
    
    # Analyze anisotropy
    angles, intensities, anisotropy = analyze_anisotropy(S_k)
    
    # Compute correlation length
    # Find where correlation drops to 1/e
    threshold = 1/np.e
    try:
        xi_idx = np.where(corr_radial < threshold)[0][0]
        xi = r_bins[xi_idx]
    except:
        xi = L // 2  # If doesn't decay, set to max
    
    print(f"\nResults:")
    print(f"  Correlation length: {xi:.1f} sites")
    print(f"  Anisotropy measure: {anisotropy:.3f}")
    
    if anisotropy > 1.0:
        print(f"  → ANISOTROPIC (organized structure)")
    else:
        print(f"  → ISOTROPIC (random structure)")
    
    results = {
        'spins': spins_cpu,
        'corr_2d': corr_2d,
        'corr_radial': (r_bins, corr_radial),
        'S_k': S_k,
        'S_k_radial': (k_bins, S_k_radial),
        'anisotropy': (angles, intensities, anisotropy),
        'xi': xi,
    }
    
    return results


def create_comparison_plots(results_g0, results_g05, output_dir):
    """Create detailed comparison plots."""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: Spin configurations
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(results_g0['spins'], cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_title('g=0: Random Domains', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(results_g05['spins'], cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_title('g=0.5: Organized Stripes', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Row 1: 2D correlations
    ax3 = fig.add_subplot(gs[0, 2])
    im1 = ax3.imshow(results_g0['corr_2d'], cmap='seismic', vmin=-0.5, vmax=1.0)
    ax3.set_title('g=0: 2D Correlation', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im1, ax=ax3, fraction=0.046)
    
    ax4 = fig.add_subplot(gs[0, 3])
    im2 = ax4.imshow(results_g05['corr_2d'], cmap='seismic', vmin=-0.5, vmax=1.0)
    ax4.set_title('g=0.5: 2D Correlation', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im2, ax=ax4, fraction=0.046)
    
    # Row 2: Radial correlation
    ax5 = fig.add_subplot(gs[1, :2])
    r0, c0 = results_g0['corr_radial']
    r05, c05 = results_g05['corr_radial']
    ax5.plot(r0, c0, 'o-', label='g=0 (random)', linewidth=2, markersize=3)
    ax5.plot(r05, c05, 's-', label='g=0.5 (stripes)', linewidth=2, markersize=3)
    ax5.axhline(1/np.e, color='red', linestyle='--', label='1/e decay')
    ax5.axvline(results_g0['xi'], color='blue', linestyle='--', alpha=0.5, 
                label=f"ξ(g=0)={results_g0['xi']:.1f}")
    ax5.axvline(results_g05['xi'], color='green', linestyle='--', alpha=0.5,
                label=f"ξ(g=0.5)={results_g05['xi']:.1f}")
    ax5.set_xlabel('Distance r (sites)', fontsize=12)
    ax5.set_ylabel('Correlation C(r)', fontsize=12)
    ax5.set_title('Radial Correlation Function', fontsize=13, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 32)
    
    # Row 2: Structure factors
    ax6 = fig.add_subplot(gs[1, 2])
    im3 = ax6.imshow(np.log10(results_g0['S_k'] + 1), cmap='hot')
    ax6.set_title('g=0: log S(k)', fontsize=12, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im3, ax=ax6, fraction=0.046)
    
    ax7 = fig.add_subplot(gs[1, 3])
    im4 = ax7.imshow(np.log10(results_g05['S_k'] + 1), cmap='hot')
    ax7.set_title('g=0.5: log S(k)', fontsize=12, fontweight='bold')
    ax7.axis('off')
    plt.colorbar(im4, ax=ax7, fraction=0.046)
    
    # Row 3: Angular intensity profiles
    ax8 = fig.add_subplot(gs[2, :2])
    angles0, int0, aniso0 = results_g0['anisotropy']
    angles05, int05, aniso05 = results_g05['anisotropy']
    ax8.plot(np.degrees(angles0), int0, '-', label=f'g=0 (anisotropy={aniso0:.3f})', 
             linewidth=2, alpha=0.7)
    ax8.plot(np.degrees(angles05), int05, '-', label=f'g=0.5 (anisotropy={aniso05:.3f})', 
             linewidth=2, alpha=0.7)
    ax8.set_xlabel('Angle (degrees)', fontsize=12)
    ax8.set_ylabel('S(k) Intensity', fontsize=12)
    ax8.set_title('Angular Distribution of Structure Factor', fontsize=13, fontweight='bold')
    ax8.legend(fontsize=11)
    ax8.grid(True, alpha=0.3)
    
    # Row 3: Radial structure factor
    ax9 = fig.add_subplot(gs[2, 2:])
    k0, S0 = results_g0['S_k_radial']
    k05, S05 = results_g05['S_k_radial']
    ax9.semilogy(k0[1:], S0[1:], 'o-', label='g=0 (random)', linewidth=2, markersize=4)
    ax9.semilogy(k05[1:], S05[1:], 's-', label='g=0.5 (stripes)', linewidth=2, markersize=4)
    ax9.set_xlabel('Wavenumber k', fontsize=12)
    ax9.set_ylabel('S(k)', fontsize=12)
    ax9.set_title('Radial Structure Factor', fontsize=13, fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3, which='both')
    
    plt.savefig(output_dir / 'spatial_structure_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved comparison plots to {output_dir / 'spatial_structure_comparison.png'}")


def main():
    """Run spatial structure comparison."""
    
    print("="*70)
    print("SPATIAL STRUCTURE ANALYSIS")
    print("="*70)
    print("\nComparing g=0 (no defrag) vs g=0.5 (with defrag)")
    print("\nIf structures are DIFFERENT:")
    print("  - g=0 should show short-range, isotropic correlations")
    print("  - g=0.5 should show long-range, anisotropic correlations")
    print("  - Structure factors will look qualitatively different")
    print("\nIf structures are SAME:")
    print("  - Similar correlation lengths")
    print("  - Similar anisotropy measures")
    print("  - Wall count difference is meaningless")
    print("="*70)
    
    # Run both analyses
    results_g0 = run_structure_analysis(
        g_defrag=0.0,
        label="No Defrag (Control)",
        seed=42
    )
    
    results_g05 = run_structure_analysis(
        g_defrag=0.5,
        label="With Defrag (Stripes)",
        seed=42
    )
    
    # Create output directory
    output_dir = Path("spatial_structure_test")
    output_dir.mkdir(exist_ok=True)
    
    # Create comparison plots
    create_comparison_plots(results_g0, results_g05, output_dir)
    
    # Print comparison
    print("\n" + "="*70)
    print("QUANTITATIVE COMPARISON")
    print("="*70)
    
    print(f"\nCorrelation length:")
    print(f"  g=0.0: ξ = {results_g0['xi']:.1f} sites")
    print(f"  g=0.5: ξ = {results_g05['xi']:.1f} sites")
    print(f"  Ratio: {results_g05['xi']/results_g0['xi']:.2f}×")
    
    if results_g05['xi'] > 2 * results_g0['xi']:
        print("  ✓ g=0.5 has MUCH longer-range correlations")
    
    print(f"\nAnisotropy measure:")
    print(f"  g=0.0: {results_g0['anisotropy'][2]:.3f}")
    print(f"  g=0.5: {results_g05['anisotropy'][2]:.3f}")
    print(f"  Ratio: {results_g05['anisotropy'][2]/results_g0['anisotropy'][2]:.2f}×")
    
    if results_g05['anisotropy'][2] > 2 * results_g0['anisotropy'][2]:
        print("  ✓ g=0.5 is MUCH more anisotropic (directional structure)")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if (results_g05['xi'] > 2 * results_g0['xi'] and
        results_g05['anisotropy'][2] > 2 * results_g0['anisotropy'][2]):
        print("✓✓✓ STRUCTURES ARE FUNDAMENTALLY DIFFERENT ✓✓✓")
        print("\nEvidence:")
        print("  - g=0.5 has much longer correlation length")
        print("  - g=0.5 has directional anisotropy (stripes!)")
        print("  - Spatial organization is qualitatively different")
        print("\nThe stripe phase is REAL STRUCTURE, not just 'M=0 by accident'!")
    else:
        print("⚠ STRUCTURES MAY BE SIMILAR")
        print("\nCorrelations and anisotropy are not dramatically different.")
        print("Wall count difference might not reflect real structural organization.")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
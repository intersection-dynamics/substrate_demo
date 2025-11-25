#!/usr/bin/env python3
"""
lump_interaction_potential.py

Measure the static interaction potential between two topological defects (vortices)
in the coupled substrate system.

Strategy:
1. Initialize two vortices (w=+1 each) at controlled separation
2. Let system relax briefly (short equilibration)
3. Measure total energy as function of separation
4. Plot V(d) = E(d) - E(∞)

Usage:
    python lump_interaction_potential.py --L 128 --n_separations 15
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Try GPU
try:
    import cupy as cp
    xp = cp
    GPU_AVAILABLE = True
except ImportError:
    xp = np
    GPU_AVAILABLE = False


def create_vortex_pair(L, dx, separation, winding1=1, winding2=1, core_size=3.0):
    """
    Create initial field with two vortices.
    
    Vortex structure:
        ψ(x,y) = f(r) * exp(i*w*θ)
    where:
        θ = atan2(y - y0, x - x0)
        f(r) = tanh(r / core_size)  (amplitude profile)
        w = winding number
    
    Args:
        L: Grid size
        dx: Lattice spacing
        separation: Distance between vortex centers (in grid units)
        winding1: Winding number of first vortex
        winding2: Winding number of second vortex
        core_size: Size of vortex core (in grid units)
    
    Returns:
        psi: Complex field with two vortices
    """
    # Grid coordinates
    i = xp.arange(L)
    j = xp.arange(L)
    X, Y = xp.meshgrid(i, j, indexing='ij')
    
    # Vortex positions (centered, separated along x-axis)
    x1 = L/2 - separation/2
    y1 = L/2
    
    x2 = L/2 + separation/2
    y2 = L/2
    
    # Distance and angle from each vortex
    r1 = xp.sqrt((X - x1)**2 + (Y - y1)**2)
    theta1 = xp.arctan2(Y - y1, X - x1)
    
    r2 = xp.sqrt((X - x2)**2 + (Y - y2)**2)
    theta2 = xp.arctan2(Y - y2, X - x2)
    
    # Amplitude profile (goes to 0 at core, 1 far away)
    amp1 = xp.tanh(r1 / core_size)
    amp2 = xp.tanh(r2 / core_size)
    
    # Phase winding
    phase1 = winding1 * theta1
    phase2 = winding2 * theta2
    
    # Construct vortices
    vortex1 = amp1 * xp.exp(1j * phase1)
    vortex2 = amp2 * xp.exp(1j * phase2)
    
    # Combine: multiply the two fields
    # This is a simple approach; more sophisticated would solve for the 
    # proper two-vortex solution, but this should work for initial state
    psi = vortex1 * vortex2
    
    # Normalize to reasonable amplitude
    psi = psi / xp.max(xp.abs(psi))
    
    return psi


def laplacian(field, dx):
    """2D Laplacian with periodic BCs."""
    return (
        xp.roll(field, +1, axis=0)
        + xp.roll(field, -1, axis=0)
        + xp.roll(field, +1, axis=1)
        + xp.roll(field, -1, axis=1)
        - 4.0 * field
    ) / (dx * dx)


def compute_defrag_energy(psi, g_defrag, dx):
    """
    Compute defrag potential energy.
    
    H_defrag = (g_defrag/2) ∫ ρ(x) Φ(x) dx
    
    where ∇²Φ = -ρ and ρ = |ψ|²
    """
    L = psi.shape[0]
    rho = xp.abs(psi)**2
    
    # FFT-based Poisson solve
    kx = xp.fft.fftfreq(L, d=dx) * 2.0 * np.pi
    ky = xp.fft.fftfreq(L, d=dx) * 2.0 * np.pi
    KX, KY = xp.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0  # avoid division by zero
    
    rho_k = xp.fft.fftn(rho)
    phi_k = -rho_k / K2
    phi_k[0, 0] = 0.0  # set DC component to zero
    phi = xp.fft.ifftn(phi_k).real
    
    E_defrag = 0.5 * g_defrag * xp.sum(rho * phi) * dx * dx
    
    if GPU_AVAILABLE:
        return float(cp.asnumpy(E_defrag))
    else:
        return float(E_defrag)


def compute_gradient_energy(psi, dx):
    """
    Compute kinetic/gradient energy.
    
    E_grad = (1/2) ∫ |∇ψ|² dx
    """
    dpsi_dx = (xp.roll(psi, -1, axis=0) - xp.roll(psi, +1, axis=0)) / (2.0 * dx)
    dpsi_dy = (xp.roll(psi, -1, axis=1) - xp.roll(psi, +1, axis=1)) / (2.0 * dx)
    
    grad_sq = xp.abs(dpsi_dx)**2 + xp.abs(dpsi_dy)**2
    
    E_grad = 0.5 * xp.sum(grad_sq) * dx * dx
    
    if GPU_AVAILABLE:
        return float(cp.asnumpy(E_grad))
    else:
        return float(E_grad)


def relax_configuration(psi, g_defrag, dx, dt, n_steps):
    """
    Relax the configuration by evolving for a short time.
    
    Simple gradient descent: ψ → ψ - dt * (δH/δψ*)
    
    This is NOT the full Schrödinger evolution, just a quick relaxation
    to get to a local energy minimum.
    """
    for step in range(n_steps):
        # Compute forces
        rho = xp.abs(psi)**2
        
        # Defrag potential
        L = psi.shape[0]
        kx = xp.fft.fftfreq(L, d=dx) * 2.0 * np.pi
        ky = xp.fft.fftfreq(L, d=dx) * 2.0 * np.pi
        KX, KY = xp.meshgrid(kx, ky, indexing='ij')
        K2 = KX**2 + KY**2
        K2[0, 0] = 1.0
        
        rho_k = xp.fft.fftn(rho)
        phi_k = -rho_k / K2
        phi_k[0, 0] = 0.0
        phi = xp.fft.ifftn(phi_k).real
        
        V_defrag = g_defrag * phi
        
        # Gradient energy force: -∇²ψ
        lap_psi = laplacian(psi, dx)
        
        # Total force: δH/δψ* = -∇²ψ + V*ψ
        force = -lap_psi + V_defrag * psi
        
        # Gradient descent step
        psi = psi - dt * force
        
        # Optional: renormalize to prevent runaway
        psi = psi / (xp.max(xp.abs(psi)) + 1e-10)
    
    return psi


def measure_interaction_potential(args):
    """
    Main measurement function.
    """
    print("=" * 60)
    print("LUMP INTERACTION POTENTIAL MEASUREMENT")
    print("=" * 60)
    print(f"Grid size: {args.L}x{args.L}")
    print(f"dx = {args.dx}")
    print(f"g_defrag = {args.g_defrag}")
    print(f"Separations: {args.n_separations} points from {args.d_min} to {args.d_max}")
    print(f"GPU: {GPU_AVAILABLE}")
    print()
    
    # Separation distances to test
    separations = np.linspace(args.d_min, args.d_max, args.n_separations)
    
    energies_total = []
    energies_defrag = []
    energies_grad = []
    
    for i, sep in enumerate(separations):
        print(f"[{i+1}/{args.n_separations}] Separation = {sep:.2f} dx", end=" ... ")
        
        # Create initial state with two vortices
        psi = create_vortex_pair(
            L=args.L,
            dx=args.dx,
            separation=sep,
            winding1=+1,
            winding2=+1,
            core_size=args.core_size
        )
        
        # Relax configuration
        if args.relax_steps > 0:
            psi = relax_configuration(
                psi=psi,
                g_defrag=args.g_defrag,
                dx=args.dx,
                dt=args.relax_dt,
                n_steps=args.relax_steps
            )
        
        # Measure energies
        E_defrag = compute_defrag_energy(psi, args.g_defrag, args.dx)
        E_grad = compute_gradient_energy(psi, args.dx)
        E_total = E_defrag + E_grad
        
        energies_total.append(E_total)
        energies_defrag.append(E_defrag)
        energies_grad.append(E_grad)
        
        print(f"E_total = {E_total:.4f}, E_defrag = {E_defrag:.4f}, E_grad = {E_grad:.4f}")
    
    # Convert to numpy arrays
    separations = np.array(separations)
    energies_total = np.array(energies_total)
    energies_defrag = np.array(energies_defrag)
    energies_grad = np.array(energies_grad)
    
    # Compute interaction potential (subtract infinite separation energy)
    E_inf = energies_total[-1]  # Energy at largest separation
    V_interaction = energies_total - E_inf
    
    # Save data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    np.savetxt(
        output_dir / "interaction_potential.csv",
        np.column_stack([separations, energies_total, energies_defrag, energies_grad, V_interaction]),
        delimiter=',',
        header='separation,E_total,E_defrag,E_grad,V_interaction',
        comments=''
    )
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Energy at d_min = {separations[0]:.2f}: E = {energies_total[0]:.4f}")
    print(f"Energy at d_max = {separations[-1]:.2f}: E = {energies_total[-1]:.4f}")
    print(f"ΔE = {energies_total[0] - energies_total[-1]:.4f}")
    print()
    
    # Check for repulsion
    if V_interaction[0] > 0:
        print("✓ REPULSIVE interaction detected (V > 0 at small separation)")
    elif V_interaction[0] < 0:
        print("✗ ATTRACTIVE interaction (V < 0 at small separation)")
    else:
        print("? No clear interaction")
    
    print(f"\nData saved to {output_dir / 'interaction_potential.csv'}")
    
    # Plot results
    plot_results(separations, energies_total, energies_defrag, energies_grad, V_interaction, args)
    
    return separations, energies_total, V_interaction


def plot_results(separations, E_total, E_defrag, E_grad, V_interaction, args):
    """
    Create plots of the interaction potential.
    """
    output_dir = Path(args.output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Total energy vs separation
    ax = axes[0, 0]
    ax.plot(separations, E_total, 'o-', color='black', linewidth=2, markersize=6)
    ax.set_xlabel('Separation (dx)', fontsize=12)
    ax.set_ylabel('Total Energy', fontsize=12)
    ax.set_title('Total Energy vs Separation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Energy components
    ax = axes[0, 1]
    ax.plot(separations, E_defrag, 'o-', label='Defrag', linewidth=2, markersize=6)
    ax.plot(separations, E_grad, 's-', label='Gradient', linewidth=2, markersize=6)
    ax.set_xlabel('Separation (dx)', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title('Energy Components', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Interaction potential V(d)
    ax = axes[1, 0]
    ax.plot(separations, V_interaction, 'o-', color='red', linewidth=2, markersize=6)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Separation (dx)', fontsize=12)
    ax.set_ylabel('V(d) = E(d) - E(∞)', fontsize=12)
    ax.set_title('Interaction Potential', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add text annotation
    if V_interaction[0] > 0:
        ax.text(0.95, 0.95, 'REPULSIVE', transform=ax.transAxes,
                fontsize=14, fontweight='bold', color='red',
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    elif V_interaction[0] < 0:
        ax.text(0.95, 0.95, 'ATTRACTIVE', transform=ax.transAxes,
                fontsize=14, fontweight='bold', color='blue',
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 4: Log-log plot (check for power law)
    ax = axes[1, 1]
    # Only plot positive values for log-log
    mask = (separations > 0) & (np.abs(V_interaction) > 1e-10)
    if np.sum(mask) > 2:
        ax.loglog(separations[mask], np.abs(V_interaction[mask]), 'o-', 
                 linewidth=2, markersize=6, label='|V(d)|')
        
        # Try to fit power law: V ~ d^n
        if np.sum(mask) >= 3:
            log_d = np.log(separations[mask])
            log_V = np.log(np.abs(V_interaction[mask]))
            coeffs = np.polyfit(log_d, log_V, 1)
            power = coeffs[0]
            
            # Plot fit
            d_fit = np.logspace(np.log10(separations[mask][0]), 
                               np.log10(separations[mask][-1]), 50)
            V_fit = np.exp(coeffs[1]) * d_fit**power
            ax.loglog(d_fit, V_fit, '--', color='gray', linewidth=2,
                     label=f'Fit: d^{power:.2f}')
            
            ax.text(0.05, 0.95, f'Power law: V ∝ d^{power:.2f}',
                   transform=ax.transAxes, fontsize=12,
                   va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Separation (dx)', fontsize=12)
    ax.set_ylabel('|V(d)|', fontsize=12)
    ax.set_title('Power Law Analysis (log-log)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'interaction_potential.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_dir / 'interaction_potential.png'}")
    
    if not args.no_display:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Measure interaction potential between topological lumps"
    )
    
    # Grid parameters
    parser.add_argument('--L', type=int, default=128,
                       help='Grid size (LxL)')
    parser.add_argument('--dx', type=float, default=1.0,
                       help='Lattice spacing')
    
    # Physics parameters
    parser.add_argument('--g_defrag', type=float, default=1.5,
                       help='Defrag coupling strength')
    parser.add_argument('--core_size', type=float, default=3.0,
                       help='Vortex core size (in grid units)')
    
    # Separation range
    parser.add_argument('--d_min', type=float, default=5.0,
                       help='Minimum separation (dx units)')
    parser.add_argument('--d_max', type=float, default=50.0,
                       help='Maximum separation (dx units)')
    parser.add_argument('--n_separations', type=int, default=15,
                       help='Number of separation points to measure')
    
    # Relaxation parameters
    parser.add_argument('--relax_steps', type=int, default=100,
                       help='Number of relaxation steps (0 = no relaxation)')
    parser.add_argument('--relax_dt', type=float, default=0.01,
                       help='Timestep for relaxation')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='interaction_results',
                       help='Output directory for results')
    parser.add_argument('--no_display', action='store_true',
                       help='Do not display plot (save only)')
    
    args = parser.parse_args()
    
    # Run measurement
    measure_interaction_potential(args)


if __name__ == '__main__':
    main()
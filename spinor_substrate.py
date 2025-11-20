#!/usr/bin/env python3
"""
spinor_substrate.py

Spinor Substrate Framework: Coupled scalar-gauge system with internal DOFs

Upgrades single complex field ψ(x,y) to spinor field Ψ(x,y) ∈ ℂ²:
    Ψ = [ψ₁(x,y)]
        [ψ₂(x,y)]

This gives each point internal structure - the minimal addition needed for:
- Spin degrees of freedom
- Skyrmion topological defects
- Fermionic exclusion (potentially)
- Richer entanglement structure

Key equations:
    H = H_kinetic + H_gauge + H_defrag
    
    H_kinetic = (1/2) ∫ |D_μ Ψ|² dx
    where D_μ = ∂_μ - iq A_μ (acts on both components)
    
    H_gauge = (1/2) ∫ B² dx
    where B = ∇ × A
    
    H_defrag = (g/2) ∫ ρ(x) Φ(x) dx
    where ∇²Φ = -ρ and ρ = |ψ₁|² + |ψ₂|²

Evolution:
    i ∂_t Ψ = H Ψ (Schrödinger)
    Using symplectic integration for energy conservation

Topological structures:
    - Skyrmions: topological charge B = (1/4π) ∫ Ψ†(∂ᵢΨ × ∂ⱼΨ)·Ψ dx
    - Spin texture: local spin direction s = Ψ†σΨ
    
Physics tests:
    - Do skyrmions with same topological charge repel? (fermionic)
    - Do they have stable spin structure?
    - What statistics do they exhibit under exchange?
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Try GPU
try:
    import cupy as cp
    xp = cp
    GPU_AVAILABLE = True
    print("GPU (CuPy) available")
except ImportError:
    xp = np
    GPU_AVAILABLE = False
    print("GPU not available, using CPU (NumPy)")


class SpinorSubstrate:
    """
    Spinor substrate with internal SU(2) structure.
    
    Fields:
        psi1, psi2: Complex scalar fields (spinor components)
        Ax, Ay: Gauge field components
    
    Parameters:
        L: Grid size
        dx: Lattice spacing
        dt: Time step
        q: Gauge coupling strength
        g_defrag: Defrag potential strength
        m_gauge: Gauge field mass (if nonzero)
    """
    
    def __init__(self, L, dx, dt, q=1.0, g_defrag=1.5, m_gauge=0.0):
        self.L = L
        self.dx = dx
        self.dt = dt
        self.q = q
        self.g_defrag = g_defrag
        self.m_gauge = m_gauge
        
        # Initialize fields
        self.psi1 = xp.zeros((L, L), dtype=xp.complex128)
        self.psi2 = xp.zeros((L, L), dtype=xp.complex128)
        self.Ax = xp.zeros((L, L), dtype=xp.float64)
        self.Ay = xp.zeros((L, L), dtype=xp.float64)
        
        # Momentum fields (for symplectic integration)
        self.Pi1 = xp.zeros((L, L), dtype=xp.complex128)
        self.Pi2 = xp.zeros((L, L), dtype=xp.complex128)
        self.Px = xp.zeros((L, L), dtype=xp.float64)
        self.Py = xp.zeros((L, L), dtype=xp.float64)
        
        # For defrag potential (FFT-based Poisson solver)
        kx = xp.fft.fftfreq(L, d=dx) * 2.0 * np.pi
        ky = xp.fft.fftfreq(L, d=dx) * 2.0 * np.pi
        KX, KY = xp.meshgrid(kx, ky, indexing='ij')
        self.K2 = KX**2 + KY**2
        self.K2[0, 0] = 1.0  # Avoid division by zero
        
        # Statistics
        self.time = 0.0
        self.step_count = 0
        
    def initialize_noise(self, amplitude=0.1, seed=None):
        """Initialize with random noise."""
        if seed is not None:
            if GPU_AVAILABLE:
                xp.random.seed(seed)
            else:
                np.random.seed(seed)
        
        # Random complex fields
        self.psi1 = amplitude * (xp.random.randn(self.L, self.L) + 
                                 1j * xp.random.randn(self.L, self.L))
        self.psi2 = amplitude * (xp.random.randn(self.L, self.L) + 
                                 1j * xp.random.randn(self.L, self.L))
        
        # Normalize (optional)
        norm = xp.sqrt(xp.abs(self.psi1)**2 + xp.abs(self.psi2)**2)
        self.psi1 /= (norm + 1e-10)
        self.psi2 /= (norm + 1e-10)
        
        # Random gauge field
        self.Ax = 0.01 * xp.random.randn(self.L, self.L)
        self.Ay = 0.01 * xp.random.randn(self.L, self.L)
        
    def initialize_skyrmion(self, center, charge=1, spin_up=True):
        """
        Initialize a single skyrmion.
        
        Args:
            center: (x, y) position
            charge: Topological charge (±1)
            spin_up: If True, spin points up at center
        """
        i = xp.arange(self.L)
        j = xp.arange(self.L)
        X, Y = xp.meshgrid(i, j, indexing='ij')
        
        cx, cy = center
        r = xp.sqrt((X - cx)**2 + (Y - cy)**2)
        theta = xp.arctan2(Y - cy, X - cx)
        
        # Skyrmion profile
        # f(r) goes from π at r=0 to 0 at r=∞
        R = 5.0  # characteristic size
        f = xp.pi * xp.exp(-r / R)
        
        # Spinor configuration for skyrmion
        if spin_up:
            self.psi1 = xp.cos(f/2) * xp.exp(1j * charge * theta)
            self.psi2 = xp.sin(f/2) * xp.exp(-1j * charge * theta)
        else:
            self.psi1 = xp.sin(f/2) * xp.exp(1j * charge * theta)
            self.psi2 = xp.cos(f/2) * xp.exp(-1j * charge * theta)
    
    def initialize_two_skyrmions(self, separation, charge1=1, charge2=1):
        """Initialize two skyrmions for interaction studies."""
        
        i = xp.arange(self.L)
        j = xp.arange(self.L)
        X, Y = xp.meshgrid(i, j, indexing='ij')
        
        # Centers
        cx1 = self.L/2 - separation/2
        cy1 = self.L/2
        cx2 = self.L/2 + separation/2
        cy2 = self.L/2
        
        # Distances and angles
        r1 = xp.sqrt((X - cx1)**2 + (Y - cy1)**2)
        theta1 = xp.arctan2(Y - cy1, X - cx1)
        
        r2 = xp.sqrt((X - cx2)**2 + (Y - cy2)**2)
        theta2 = xp.arctan2(Y - cy2, X - cx2)
        
        # Skyrmion size
        R = 3.0
        
        # Profile functions (0 to π)
        f1 = xp.pi * xp.exp(-r1 / R)
        f2 = xp.pi * xp.exp(-r2 / R)
        
        # Weight functions (to avoid overlap artifacts)
        w1 = xp.exp(-r1**2 / (4*R**2))
        w2 = xp.exp(-r2**2 / (4*R**2))
        w_norm = w1 + w2 + 1e-10
        w1 = w1 / w_norm
        w2 = w2 / w_norm
        
        # Spinor for skyrmion 1
        psi1_a = xp.cos(f1/2) * xp.exp(1j * charge1 * theta1)
        psi2_a = xp.sin(f1/2) * xp.exp(-1j * charge1 * theta1)
        
        # Spinor for skyrmion 2
        psi1_b = xp.cos(f2/2) * xp.exp(1j * charge2 * theta2)
        psi2_b = xp.sin(f2/2) * xp.exp(-1j * charge2 * theta2)
        
        # Weighted combination (smooth transition between them)
        self.psi1 = w1 * psi1_a + w2 * psi1_b
        self.psi2 = w1 * psi2_a + w2 * psi2_b
        
        # Normalize
        norm = xp.sqrt(xp.abs(self.psi1)**2 + xp.abs(self.psi2)**2)
        self.psi1 /= (norm + 1e-10)
        self.psi2 /= (norm + 1e-10)
        
        # Scale down initial amplitude to avoid instability
        self.psi1 *= 0.5
        self.psi2 *= 0.5
        
    def gradient(self, field, axis):
        """Compute gradient with periodic BC."""
        if axis == 0:  # x-direction
            return (xp.roll(field, -1, axis=0) - xp.roll(field, +1, axis=0)) / (2.0 * self.dx)
        else:  # y-direction
            return (xp.roll(field, -1, axis=1) - xp.roll(field, +1, axis=1)) / (2.0 * self.dx)
    
    def laplacian(self, field):
        """Compute Laplacian with periodic BC."""
        return (xp.roll(field, +1, axis=0) + xp.roll(field, -1, axis=0) +
                xp.roll(field, +1, axis=1) + xp.roll(field, -1, axis=1) - 
                4.0 * field) / (self.dx * self.dx)
    
    def covariant_derivative(self, psi, axis):
        """
        Covariant derivative D_μ ψ = ∂_μ ψ - iq A_μ ψ
        
        Acts identically on both spinor components.
        """
        grad_psi = self.gradient(psi, axis)
        if axis == 0:
            A = self.Ax
        else:
            A = self.Ay
        return grad_psi - 1j * self.q * A * psi
    
    def magnetic_field(self):
        """Compute B = ∇ × A."""
        return self.gradient(self.Ay, 0) - self.gradient(self.Ax, 1)
    
    def compute_defrag_potential(self):
        """
        Solve for defrag potential: ∇²Φ = -ρ
        where ρ = |ψ₁|² + |ψ₂|²
        """
        rho = xp.abs(self.psi1)**2 + xp.abs(self.psi2)**2
        
        # FFT-based Poisson solve
        rho_k = xp.fft.fftn(rho)
        phi_k = -rho_k / self.K2
        phi_k[0, 0] = 0.0  # Set DC component to zero
        phi = xp.fft.ifftn(phi_k).real
        
        return phi
    
    def compute_forces(self):
        """
        Compute forces (time derivatives) for all fields.
        
        Returns:
            dpsi1_dt, dpsi2_dt, dAx_dt, dAy_dt
        """
        # Scalar kinetic energy: -(1/2) ∇²ψ
        lap_psi1 = self.laplacian(self.psi1)
        lap_psi2 = self.laplacian(self.psi2)
        
        # Gauge coupling: iq A·∇ψ + (q²/2) A²ψ
        Dx_psi1 = self.covariant_derivative(self.psi1, 0)
        Dy_psi1 = self.covariant_derivative(self.psi1, 1)
        Dx_psi2 = self.covariant_derivative(self.psi2, 0)
        Dy_psi2 = self.covariant_derivative(self.psi2, 1)
        
        # Defrag potential
        phi = self.compute_defrag_potential()
        V_defrag = self.g_defrag * phi
        
        # Force on psi1
        F_psi1 = (-0.5 * lap_psi1 + 
                  1j * self.q * (self.Ax * self.gradient(self.psi1, 0) + 
                                 self.Ay * self.gradient(self.psi1, 1)) -
                  0.5 * self.q**2 * (self.Ax**2 + self.Ay**2) * self.psi1 +
                  V_defrag * self.psi1)
        
        # Force on psi2 (same structure)
        F_psi2 = (-0.5 * lap_psi2 + 
                  1j * self.q * (self.Ax * self.gradient(self.psi2, 0) + 
                                 self.Ay * self.gradient(self.psi2, 1)) -
                  0.5 * self.q**2 * (self.Ax**2 + self.Ay**2) * self.psi2 +
                  V_defrag * self.psi2)
        
        # Current from spinor: j = q Im(Ψ† D Ψ) = q Im(ψ₁* D_μ ψ₁ + ψ₂* D_μ ψ₂)
        jx = self.q * (xp.imag(xp.conj(self.psi1) * Dx_psi1) + 
                       xp.imag(xp.conj(self.psi2) * Dx_psi2))
        jy = self.q * (xp.imag(xp.conj(self.psi1) * Dy_psi1) + 
                       xp.imag(xp.conj(self.psi2) * Dy_psi2))
        
        # Gauge field equations: ∂_t A = -∇B - j - m² A
        B = self.magnetic_field()
        
        F_Ax = -self.gradient(B, 1) - jx - self.m_gauge**2 * self.Ax
        F_Ay = +self.gradient(B, 0) - jy - self.m_gauge**2 * self.Ay
        
        return F_psi1, F_psi2, F_Ax, F_Ay
    
    def step_symplectic(self):
        """
        Symplectic integration step (velocity Verlet / leapfrog).
        
        Split Hamiltonian: H = T(p) + V(q)
        - Update positions with half-step momenta
        - Update momenta with full-step forces
        """
        # Half step for positions using current momenta
        self.psi1 += 0.5 * self.dt * (-1j * self.Pi1)
        self.psi2 += 0.5 * self.dt * (-1j * self.Pi2)
        self.Ax += 0.5 * self.dt * self.Px
        self.Ay += 0.5 * self.dt * self.Py
        
        # Full step for momenta using forces at updated positions
        F_psi1, F_psi2, F_Ax, F_Ay = self.compute_forces()
        
        self.Pi1 += self.dt * (-1j * F_psi1)
        self.Pi2 += self.dt * (-1j * F_psi2)
        self.Px += self.dt * F_Ax
        self.Py += self.dt * F_Ay
        
        # Half step for positions using updated momenta
        self.psi1 += 0.5 * self.dt * (-1j * self.Pi1)
        self.psi2 += 0.5 * self.dt * (-1j * self.Pi2)
        self.Ax += 0.5 * self.dt * self.Px
        self.Ay += 0.5 * self.dt * self.Py
        
        self.time += self.dt
        self.step_count += 1
    
    def energy_kinetic(self):
        """Kinetic energy: (1/2) ∫ |D_μ Ψ|² dx"""
        Dx_psi1 = self.covariant_derivative(self.psi1, 0)
        Dy_psi1 = self.covariant_derivative(self.psi1, 1)
        Dx_psi2 = self.covariant_derivative(self.psi2, 0)
        Dy_psi2 = self.covariant_derivative(self.psi2, 1)
        
        E = 0.5 * xp.sum(xp.abs(Dx_psi1)**2 + xp.abs(Dy_psi1)**2 +
                         xp.abs(Dx_psi2)**2 + xp.abs(Dy_psi2)**2) * self.dx**2
        
        if GPU_AVAILABLE:
            return float(cp.asnumpy(E))
        return float(E)
    
    def energy_gauge(self):
        """Gauge energy: (1/2) ∫ B² dx"""
        B = self.magnetic_field()
        E = 0.5 * xp.sum(B**2) * self.dx**2
        
        if GPU_AVAILABLE:
            return float(cp.asnumpy(E))
        return float(E)
    
    def energy_defrag(self):
        """Defrag energy: (g/2) ∫ ρ Φ dx"""
        rho = xp.abs(self.psi1)**2 + xp.abs(self.psi2)**2
        phi = self.compute_defrag_potential()
        E = 0.5 * self.g_defrag * xp.sum(rho * phi) * self.dx**2
        
        if GPU_AVAILABLE:
            return float(cp.asnumpy(E))
        return float(E)
    
    def energy_total(self):
        """Total energy."""
        return self.energy_kinetic() + self.energy_gauge() + self.energy_defrag()
    
    def density(self):
        """Spinor density: ρ = |ψ₁|² + |ψ₂|²"""
        rho = xp.abs(self.psi1)**2 + xp.abs(self.psi2)**2
        if GPU_AVAILABLE:
            return cp.asnumpy(rho)
        return rho
    
    def spin_texture(self):
        """
        Compute local spin vector: s = Ψ† σ Ψ
        where σ = (σₓ, σᵧ, σᵣ) are Pauli matrices
        
        Returns:
            sx, sy, sz: Spin components
        """
        # Pauli matrices act on spinor (ψ₁, ψ₂)
        # sx = Ψ† σₓ Ψ = ψ₁* ψ₂ + ψ₂* ψ₁
        # sy = Ψ† σᵧ Ψ = i(ψ₂* ψ₁ - ψ₁* ψ₂)
        # sz = Ψ† σᵣ Ψ = |ψ₁|² - |ψ₂|²
        
        sx = 2.0 * xp.real(xp.conj(self.psi1) * self.psi2)
        sy = 2.0 * xp.imag(xp.conj(self.psi1) * self.psi2)
        sz = xp.abs(self.psi1)**2 - xp.abs(self.psi2)**2
        
        if GPU_AVAILABLE:
            return cp.asnumpy(sx), cp.asnumpy(sy), cp.asnumpy(sz)
        return sx, sy, sz
    
    def skyrmion_density(self):
        """
        Compute topological charge density.
        
        B = (1/4π) Ψ†(∂ᵢΨ × ∂ⱼΨ)·Ψ
        
        This is a 2D version - full 3D skyrmion number requires 3D.
        """
        # Gradients
        d1_psi1 = self.gradient(self.psi1, 0)
        d2_psi1 = self.gradient(self.psi1, 1)
        d1_psi2 = self.gradient(self.psi2, 0)
        d2_psi2 = self.gradient(self.psi2, 1)
        
        # Cross product in spinor space (ψ† (∂ᵢψ × ∂ⱼψ))
        # This is simplified - proper formula involves all components
        
        # Approximate topological density using winding
        phase1 = xp.angle(self.psi1)
        phase2 = xp.angle(self.psi2)
        
        winding1 = (self.gradient(phase1, 0) * self.gradient(xp.abs(self.psi1)**2, 1) -
                    self.gradient(phase1, 1) * self.gradient(xp.abs(self.psi1)**2, 0))
        
        winding2 = (self.gradient(phase2, 0) * self.gradient(xp.abs(self.psi2)**2, 1) -
                    self.gradient(phase2, 1) * self.gradient(xp.abs(self.psi2)**2, 0))
        
        B = (winding1 + winding2) / (2.0 * xp.pi)
        
        if GPU_AVAILABLE:
            return cp.asnumpy(B)
        return B


def run_simulation(args):
    """Run the spinor substrate simulation."""
    
    print("="*60)
    print("SPINOR SUBSTRATE SIMULATION")
    print("="*60)
    print(f"Grid: {args.L}x{args.L}")
    print(f"dx = {args.dx}, dt = {args.dt}")
    print(f"q = {args.q}, g_defrag = {args.g_defrag}")
    print(f"Steps: {args.n_steps}, save every {args.save_every}")
    print()
    
    # Initialize
    substrate = SpinorSubstrate(
        L=args.L,
        dx=args.dx,
        dt=args.dt,
        q=args.q,
        g_defrag=args.g_defrag,
        m_gauge=args.m_gauge
    )
    
    # Initial condition
    if args.init_mode == 'noise':
        print("Initializing with random noise...")
        substrate.initialize_noise(amplitude=args.init_amplitude, seed=args.seed)
    elif args.init_mode == 'skyrmion':
        print("Initializing with single skyrmion...")
        substrate.initialize_skyrmion((args.L/2, args.L/2), charge=1)
    elif args.init_mode == 'two_skyrmions':
        print(f"Initializing with two skyrmions (separation={args.separation})...")
        substrate.initialize_two_skyrmions(separation=args.separation)
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Storage
    times = []
    energies = []
    
    # Initial diagnostics
    E0 = substrate.energy_total()
    print(f"Initial energy: {E0:.6f}")
    print()
    
    # Evolution
    print("Evolving...")
    t_start = time.time()
    
    for step in range(args.n_steps):
        substrate.step_symplectic()
        
        if step % args.save_every == 0:
            E = substrate.energy_total()
            
            # Check for NaN
            if np.isnan(E) or np.isinf(E):
                print(f"\n*** INSTABILITY DETECTED at step {step} ***")
                print(f"Energy = {E}")
                print("Simulation unstable. Try:")
                print("  - Smaller dt (current: {args.dt})")
                print("  - Larger separation")
                print("  - Weaker g_defrag")
                break
            
            times.append(substrate.time)
            energies.append(E)
            
            dE = abs(E - E0) / abs(E0) if E0 != 0 else 0
            
            # Warn if energy change is large
            if dE > 0.1 and step > 0:
                print(f"Step {step:6d}, t={substrate.time:8.3f}, "
                      f"E={E:12.6f}, ΔE/E={dE:.2e} ⚠️  LARGE DRIFT")
            else:
                print(f"Step {step:6d}, t={substrate.time:8.3f}, "
                      f"E={E:12.6f}, ΔE/E={dE:.2e}")
            
            # Save snapshot
            if step % (args.save_every * 10) == 0:
                save_snapshot(substrate, output_dir / f"snapshot_{step:06d}.npz")
    
    t_end = time.time()
    print()
    print(f"Simulation complete in {t_end - t_start:.2f} seconds")
    print(f"Final energy: {energies[-1]:.6f}")
    print(f"Energy drift: {abs(energies[-1] - energies[0])/abs(energies[0]):.2e}")
    
    # Save final state
    save_snapshot(substrate, output_dir / "final_state.npz")
    
    # Save time series
    np.savez(output_dir / "time_series.npz",
             times=times,
             energies=energies)
    
    # Create plots
    plot_results(substrate, times, energies, output_dir)
    
    return substrate


def save_snapshot(substrate, filename):
    """Save complete state."""
    if GPU_AVAILABLE:
        np.savez(filename,
                 psi1=cp.asnumpy(substrate.psi1),
                 psi2=cp.asnumpy(substrate.psi2),
                 Ax=cp.asnumpy(substrate.Ax),
                 Ay=cp.asnumpy(substrate.Ay),
                 time=substrate.time,
                 step=substrate.step_count)
    else:
        np.savez(filename,
                 psi1=substrate.psi1,
                 psi2=substrate.psi2,
                 Ax=substrate.Ax,
                 Ay=substrate.Ay,
                 time=substrate.time,
                 step=substrate.step_count)


def plot_results(substrate, times, energies, output_dir):
    """Create diagnostic plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Density
    ax = axes[0, 0]
    rho = substrate.density()
    im = ax.imshow(rho.T, origin='lower', cmap='viridis')
    ax.set_title('Density ρ = |ψ₁|² + |ψ₂|²')
    plt.colorbar(im, ax=ax)
    
    # Spin texture (sz component)
    ax = axes[0, 1]
    sx, sy, sz = substrate.spin_texture()
    im = ax.imshow(sz.T, origin='lower', cmap='RdBu', vmin=-1, vmax=1)
    ax.set_title('Spin sz = |ψ₁|² - |ψ₂|²')
    plt.colorbar(im, ax=ax)
    
    # Magnetic field
    ax = axes[0, 2]
    if GPU_AVAILABLE:
        B = cp.asnumpy(substrate.magnetic_field())
    else:
        B = substrate.magnetic_field()
    im = ax.imshow(B.T, origin='lower', cmap='RdBu')
    ax.set_title('Magnetic Field B')
    plt.colorbar(im, ax=ax)
    
    # Skyrmion density
    ax = axes[1, 0]
    B_sky = substrate.skyrmion_density()
    im = ax.imshow(B_sky.T, origin='lower', cmap='RdBu')
    ax.set_title('Topological Charge Density')
    plt.colorbar(im, ax=ax)
    
    # Phase of psi1
    ax = axes[1, 1]
    if GPU_AVAILABLE:
        phase = cp.asnumpy(xp.angle(substrate.psi1))
    else:
        phase = xp.angle(substrate.psi1)
    im = ax.imshow(phase.T, origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Phase(ψ₁)')
    plt.colorbar(im, ax=ax)
    
    # Energy conservation
    ax = axes[1, 2]
    ax.plot(times, energies, 'b-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Energy')
    ax.set_title('Energy Conservation')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"Plots saved to {output_dir / 'diagnostics.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Spinor Substrate Simulation")
    
    # Grid parameters
    parser.add_argument('--L', type=int, default=128,
                       help='Grid size')
    parser.add_argument('--dx', type=float, default=1.0,
                       help='Lattice spacing')
    parser.add_argument('--dt', type=float, default=0.001,
                       help='Time step (0.001 recommended for stability)')
    
    # Physics parameters
    parser.add_argument('--q', type=float, default=1.0,
                       help='Gauge coupling')
    parser.add_argument('--g_defrag', type=float, default=1.5,
                       help='Defrag strength')
    parser.add_argument('--m_gauge', type=float, default=0.0,
                       help='Gauge field mass')
    
    # Evolution
    parser.add_argument('--n_steps', type=int, default=1000,
                       help='Number of steps')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save frequency')
    
    # Initial conditions
    parser.add_argument('--init_mode', type=str, default='noise',
                       choices=['noise', 'skyrmion', 'two_skyrmions'],
                       help='Initialization mode')
    parser.add_argument('--init_amplitude', type=float, default=0.1,
                       help='Initial noise amplitude')
    parser.add_argument('--separation', type=float, default=20.0,
                       help='Separation for two skyrmions')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='spinor_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    run_simulation(args)


if __name__ == '__main__':
    main()
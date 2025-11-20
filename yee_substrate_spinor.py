#!/usr/bin/env python3
"""
yee_substrate_spinor.py

SPINOR UPGRADE of yee_substrate_coupled.py

Changes from original:
  - Single complex scalar ψ → Two-component spinor Ψ = [ψ₁, ψ₂]
  - Adds spin texture tracking: s = Ψ† σ Ψ
  - Adds skyrmion initialization
  - Preserves ALL stability features from original:
    * Quartic self-interaction (repulsive pressure)
    * RK2 midpoint integration
    * CFL checking
    * Yee Maxwell scheme

Everything else identical to working code.

Usage:
    python yee_substrate_spinor.py --mode substrate --L 128 --dt 0.001 \
        --g_defrag 0.5 --n_steps 5000 --init_mode skyrmion
"""

import argparse
import os
from typing import Tuple

import numpy as np

# Try GPU
try:
    import cupy as cp
    xp = cp
    GPU_AVAILABLE = True
except ImportError:
    xp = np
    GPU_AVAILABLE = False


# ======================================================================
# Basic finite-difference operators (unchanged from original)
# ======================================================================

def laplacian(field, dx: float):
    """2D Laplacian with periodic BCs on a cell-centered grid."""
    return (
        xp.roll(field, +1, axis=0)
        + xp.roll(field, -1, axis=0)
        + xp.roll(field, +1, axis=1)
        + xp.roll(field, -1, axis=1)
        - 4.0 * field
    ) / (dx * dx)


def deriv_x(field, dx: float):
    """Central derivative in x with periodic BCs."""
    return (xp.roll(field, -1, axis=0) - xp.roll(field, +1, axis=0)) / (2.0 * dx)


def deriv_y(field, dx: float):
    """Central derivative in y with periodic BCs."""
    return (xp.roll(field, -1, axis=1) - xp.roll(field, +1, axis=1)) / (2.0 * dx)


def smooth_density(rho):
    """
    Simple 3x3 box smoothing of a scalar field.
    This plays the role of a coarse-grained density for symmetry selection.
    """
    acc = rho.copy()
    # axial neighbours
    for sx in (-1, 1):
        acc += xp.roll(rho, sx, 0)
    for sy in (-1, 1):
        acc += xp.roll(rho, sy, 1)
    # diagonal neighbours
    for sx in (-1, 1):
        for sy in (-1, 1):
            acc += xp.roll(xp.roll(rho, sx, 0), sy, 1)
    return acc / 9.0


# ======================================================================
# Covariant gradient for SPINOR (both components)
# ======================================================================

def covariant_gradient_sq_spinor(psi1, psi2, ax_center, ay_center, dx: float, q: float):
    """
    |D Ψ|² = |D ψ₁|² + |D ψ₂|²
    where D = ∂ - iq A (acts identically on both components)
    """
    # Component 1
    dpsi1_x = deriv_x(psi1, dx)
    dpsi1_y = deriv_y(psi1, dx)
    Dpsi1_x = dpsi1_x - 1j * q * ax_center * psi1
    Dpsi1_y = dpsi1_y - 1j * q * ay_center * psi1
    
    # Component 2
    dpsi2_x = deriv_x(psi2, dx)
    dpsi2_y = deriv_y(psi2, dx)
    Dpsi2_x = dpsi2_x - 1j * q * ax_center * psi2
    Dpsi2_y = dpsi2_y - 1j * q * ay_center * psi2
    
    return (xp.abs(Dpsi1_x)**2 + xp.abs(Dpsi1_y)**2 + 
            xp.abs(Dpsi2_x)**2 + xp.abs(Dpsi2_y)**2)


def gauge_current_spinor(psi1, psi2, ax_center, ay_center, dx: float, q: float):
    """
    Gauge current j = q Im(Ψ† D Ψ) = q Im(ψ₁* D ψ₁ + ψ₂* D ψ₂)
    """
    # Component 1
    dpsi1_x = deriv_x(psi1, dx)
    dpsi1_y = deriv_y(psi1, dx)
    Dpsi1_x = dpsi1_x - 1j * q * ax_center * psi1
    Dpsi1_y = dpsi1_y - 1j * q * ay_center * psi1
    
    # Component 2
    dpsi2_x = deriv_x(psi2, dx)
    dpsi2_y = deriv_y(psi2, dx)
    Dpsi2_x = dpsi2_x - 1j * q * ax_center * psi2
    Dpsi2_y = dpsi2_y - 1j * q * ay_center * psi2
    
    jx = q * (xp.imag(xp.conjugate(psi1) * Dpsi1_x) + 
              xp.imag(xp.conjugate(psi2) * Dpsi2_x))
    jy = q * (xp.imag(xp.conjugate(psi1) * Dpsi1_y) + 
              xp.imag(xp.conjugate(psi2) * Dpsi2_y))
    
    return jx, jy


def symmetry_potential(rho, mode: str, lambda_F: float, alpha_B: float, beta_B: float):
    """
    Build symmetry selector potential V_sym from a smoothed density rho_smooth.
    (Unchanged from original - still works with spinor density)
    """
    if mode == "none":
        return xp.zeros_like(rho), rho

    rho_smooth = smooth_density(rho)

    if mode == "fermion":
        V_sym = lambda_F * rho_smooth
    elif mode == "boson":
        V_sym = -alpha_B * rho_smooth + beta_B * rho_smooth**2
    else:
        V_sym = xp.zeros_like(rho)

    return V_sym, rho_smooth


# ======================================================================
# NEW: Spin texture calculation
# ======================================================================

def spin_texture(psi1, psi2):
    """
    Compute local spin vector: s = Ψ† σ Ψ
    
    Returns:
        sx, sy, sz: Three components of spin
    """
    # Pauli matrices:
    # sx = Ψ† σₓ Ψ = ψ₁* ψ₂ + ψ₂* ψ₁ = 2 Re(ψ₁* ψ₂)
    # sy = Ψ† σᵧ Ψ = i(ψ₂* ψ₁ - ψ₁* ψ₂) = 2 Im(ψ₁* ψ₂)
    # sz = Ψ† σᵣ Ψ = |ψ₁|² - |ψ₂|²
    
    sx = 2.0 * xp.real(xp.conjugate(psi1) * psi2)
    sy = 2.0 * xp.imag(xp.conjugate(psi1) * psi2)
    sz = xp.abs(psi1)**2 - xp.abs(psi2)**2
    
    return sx, sy, sz


# ======================================================================
# Spinor Substrate Engine (upgraded from original)
# ======================================================================

class YeeSubstrateSpinor2D:
    """
    Spinor substrate engine - upgraded from YeeSubstrateCoupled2D.
    
    Changes:
      - psi → (psi1, psi2) spinor field
      - All stability features preserved
      - Adds spin texture tracking
    """

    def __init__(
        self,
        L: int,
        dx: float,
        dt: float,
        c: float,
        q: float,
        g_defrag: float,
        n_steps: int,
        sample_interval: int,
        symmetry_mode: str = "none",
        lambda_F: float = 0.0,
        alpha_B: float = 0.0,
        beta_B: float = 0.0,
    ):
        self.L = L
        self.dx = dx
        self.dy = dx
        self.dt = dt
        self.c = c
        self.q = q
        self.g_defrag = g_defrag
        self.n_steps = n_steps
        self.sample_interval = sample_interval

        self.symmetry_mode = symmetry_mode
        self.lambda_F = lambda_F
        self.alpha_B = alpha_B
        self.beta_B = beta_B

        shape = (L, L)

        # SPINOR field (two components)
        self.psi1 = xp.zeros(shape, dtype=xp.complex128)
        self.psi2 = xp.zeros(shape, dtype=xp.complex128)

        # Yee EM fields (unchanged)
        self.Ex = xp.zeros(shape, dtype=xp.float64)
        self.Ey = xp.zeros(shape, dtype=xp.float64)
        self.Bz = xp.zeros(shape, dtype=xp.float64)

        # Vector potential (unchanged)
        self.Ax = xp.zeros(shape, dtype=xp.float64)
        self.Ay = xp.zeros(shape, dtype=xp.float64)

        # FFT machinery for defrag potential (unchanged)
        kx = xp.fft.fftfreq(L, d=dx) * 2.0 * np.pi
        ky = xp.fft.fftfreq(L, d=dx) * 2.0 * np.pi
        self.KX, self.KY = xp.meshgrid(kx, ky, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0, 0] = 1.0

        # CFL check (unchanged)
        cfl_limit = dx / (np.sqrt(2.0) * c)
        if dt > cfl_limit:
            print(
                f"[WARN] dt={dt:g} exceeds CFL limit ~{cfl_limit:g} "
                f"for c={c:g}, dx={dx:g} (expect instability)."
            )
        else:
            print(
                f"[INFO] CFL OK: dt={dt:g} <= {cfl_limit:g} "
                f"(for c={c:g}, dx={dx:g})."
            )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def init_scalar_noise(self, amp: float):
        """Random noise in both spinor components."""
        rs = xp.random.RandomState(1234)
        self.psi1 = amp * (rs.randn(self.L, self.L) + 1j * rs.randn(self.L, self.L))
        self.psi2 = amp * (rs.randn(self.L, self.L) + 1j * rs.randn(self.L, self.L))
        
        # Normalize
        rho = xp.abs(self.psi1)**2 + xp.abs(self.psi2)**2
        norm = xp.sqrt(rho + 1e-10)
        self.psi1 /= norm
        self.psi2 /= norm
        
        print(f"[INIT] Spinor noise amplitude = {amp:g}")

    def init_skyrmion(self, center_x, center_y, charge=1, R=5.0):
        """
        Initialize a single skyrmion centered at (center_x, center_y).
        
        Skyrmion ansatz:
            Ψ = [cos(f/2) e^(i*charge*θ)]
                [sin(f/2) e^(-i*charge*θ)]
        
        where f(r) = π * exp(-r/R) goes from π at center to 0 at infinity.
        """
        i = xp.arange(self.L)
        j = xp.arange(self.L)
        X, Y = xp.meshgrid(i, j, indexing='ij')
        
        r = xp.sqrt((X - center_x)**2 + (Y - center_y)**2)
        theta = xp.arctan2(Y - center_y, X - center_x)
        
        f = xp.pi * xp.exp(-r / R)
        
        self.psi1 = xp.cos(f/2) * xp.exp(1j * charge * theta)
        self.psi2 = xp.sin(f/2) * xp.exp(-1j * charge * theta)
        
        print(f"[INIT] Skyrmion at ({center_x:.1f}, {center_y:.1f}), "
              f"charge={charge}, R={R:.1f}")

    def init_two_skyrmions(self, separation, charge1=1, charge2=1, R=5.0):
        """Initialize two skyrmions separated along x-axis."""
        cx1 = self.L/2 - separation/2
        cy1 = self.L/2
        cx2 = self.L/2 + separation/2
        cy2 = self.L/2
        
        # Create grids
        i = xp.arange(self.L)
        j = xp.arange(self.L)
        X, Y = xp.meshgrid(i, j, indexing='ij')
        
        # Skyrmion 1
        r1 = xp.sqrt((X - cx1)**2 + (Y - cy1)**2)
        theta1 = xp.arctan2(Y - cy1, X - cx1)
        f1 = xp.pi * xp.exp(-r1 / R)
        
        psi1_a = xp.cos(f1/2) * xp.exp(1j * charge1 * theta1)
        psi2_a = xp.sin(f1/2) * xp.exp(-1j * charge1 * theta1)
        
        # Skyrmion 2
        r2 = xp.sqrt((X - cx2)**2 + (Y - cy2)**2)
        theta2 = xp.arctan2(Y - cy2, X - cx2)
        f2 = xp.pi * xp.exp(-r2 / R)
        
        psi1_b = xp.cos(f2/2) * xp.exp(1j * charge2 * theta2)
        psi2_b = xp.sin(f2/2) * xp.exp(-1j * charge2 * theta2)
        
        # Weight functions (smooth blend)
        w1 = xp.exp(-r1**2 / (4*R**2))
        w2 = xp.exp(-r2**2 / (4*R**2))
        w_norm = w1 + w2 + 1e-10
        
        # Combine
        self.psi1 = (w1 * psi1_a + w2 * psi1_b) / w_norm
        self.psi2 = (w1 * psi2_a + w2 * psi2_b) / w_norm
        
        # Renormalize
        rho = xp.abs(self.psi1)**2 + xp.abs(self.psi2)**2
        norm = xp.sqrt(rho + 1e-10)
        self.psi1 /= norm
        self.psi2 /= norm
        
        print(f"[INIT] Two skyrmions, separation={separation:.1f}, "
              f"charges=({charge1}, {charge2})")

    def init_gauge_noise(self, amp: float):
        """Random gauge field (unchanged from original)."""
        rs = xp.random.RandomState(5678)
        self.Ex = xp.zeros_like(self.Ex)
        self.Ey = xp.zeros_like(self.Ey)
        self.Bz = xp.zeros_like(self.Bz)
        self.Ax = amp * rs.randn(self.L, self.L)
        self.Ay = amp * rs.randn(self.L, self.L)
        print(f"[INIT] Gauge A noise amplitude = {amp:g}")

    # ------------------------------------------------------------------
    # Defrag potential (unchanged - works with spinor density)
    # ------------------------------------------------------------------

    def defrag_potential(self, rho):
        """
        Solve ∇²φ = -ρ for defrag potential.
        Now rho = |ψ₁|² + |ψ₂|² (spinor density).
        """
        rho_k = xp.fft.fftn(rho)
        phi_k = -rho_k / self.K2
        phi_k[0, 0] = 0.0
        phi = xp.fft.ifftn(phi_k).real
        return self.g_defrag * phi

    # ------------------------------------------------------------------
    # RHS for spinor (both components)
    # ------------------------------------------------------------------

    def rhs_psi(self, psi1, psi2):
        """
        d Ψ / dt = -i H Ψ
        
        H = -½ ∇² + V_eff
        
        where V_eff = V_defrag + V_sym + 0.5 * rho
        
        CRITICAL: The +0.5*rho term provides repulsive pressure
        that prevents gravitational collapse.
        """
        
        # Spinor density
        rho = xp.abs(psi1)**2 + xp.abs(psi2)**2
        
        # Interpolate gauge field to centers
        Ax_c = 0.5 * (self.Ax + xp.roll(self.Ax, +1, axis=0))
        Ay_c = 0.5 * (self.Ay + xp.roll(self.Ay, +1, axis=1))
        
        # Defrag potential
        V_def = 0.0
        if self.g_defrag != 0.0:
            V_def = self.defrag_potential(rho)
        
        # Symmetry selector
        V_sym, _rho_smooth = symmetry_potential(
            rho,
            self.symmetry_mode,
            self.lambda_F,
            self.alpha_B,
            self.beta_B,
        )
        
        # CRITICAL: Quartic self-interaction (REPULSIVE PRESSURE)
        # This is what prevents collapse!
        V_eff = V_def + V_sym + 0.5 * rho
        
        # Kinetic terms (Laplacian on each component)
        lap_psi1 = laplacian(psi1, self.dx)
        lap_psi2 = laplacian(psi2, self.dx)
        
        # Hamiltonian applied to each component
        # H ψ₁ = -½ ∇²ψ₁ + V_eff ψ₁
        # H ψ₂ = -½ ∇²ψ₂ + V_eff ψ₂
        Hpsi1 = -0.5 * lap_psi1 + V_eff * psi1
        Hpsi2 = -0.5 * lap_psi2 + V_eff * psi2
        
        # Schrödinger evolution: i ∂ₜ Ψ = H Ψ
        return -1j * Hpsi1, -1j * Hpsi2

    # ------------------------------------------------------------------
    # Time step (RK2 midpoint - STABLE METHOD from original)
    # ------------------------------------------------------------------

    def step(self):
        """
        One full timestep:
        1. Maxwell update with spinor current
        2. Integrate A from E
        3. RK2 midpoint for spinor
        """
        dt = self.dt
        dx = self.dx
        dy = self.dy
        c = self.c

        # --- 1) Gauge current from spinor
        Ax_c = 0.5 * (self.Ax + xp.roll(self.Ax, +1, axis=0))
        Ay_c = 0.5 * (self.Ay + xp.roll(self.Ay, +1, axis=1))
        jx_c, jy_c = gauge_current_spinor(
            self.psi1, self.psi2, Ax_c, Ay_c, dx, self.q
        )
        
        Jx = jx_c
        Jy = jy_c

        # --- 2) Maxwell update (Yee TEz)
        # Bz update
        dEx_dy = (xp.roll(self.Ex, -1, axis=1) - self.Ex) / dy
        dEy_dx = (xp.roll(self.Ey, -1, axis=0) - self.Ey) / dx
        self.Bz = self.Bz + dt * (dEx_dy - dEy_dx)

        # Ex update
        dBz_dy = (self.Bz - xp.roll(self.Bz, +1, axis=1)) / dy
        self.Ex = self.Ex + dt * (c**2 * dBz_dy - Jx)

        # Ey update
        dBz_dx = (self.Bz - xp.roll(self.Bz, +1, axis=0)) / dx
        self.Ey = self.Ey - dt * (c**2 * dBz_dx + Jy)

        # --- 3) Update A (temporal gauge)
        self.Ax = self.Ax - dt * self.Ex
        self.Ay = self.Ay - dt * self.Ey

        # --- 4) RK2 MIDPOINT for spinor (STABLE!)
        psi1_0 = self.psi1
        psi2_0 = self.psi2
        
        # First RHS evaluation
        k1_1, k1_2 = self.rhs_psi(psi1_0, psi2_0)
        
        # Midpoint
        psi1_mid = psi1_0 + 0.5 * dt * k1_1
        psi2_mid = psi2_0 + 0.5 * dt * k1_2
        
        # Second RHS evaluation at midpoint
        k2_1, k2_2 = self.rhs_psi(psi1_mid, psi2_mid)
        
        # Final update using midpoint derivative
        self.psi1 = psi1_0 + dt * k2_1
        self.psi2 = psi2_0 + dt * k2_2

    # ------------------------------------------------------------------
    # Energy diagnostic
    # ------------------------------------------------------------------

    def total_energy(self) -> float:
        """
        Total energy (modified for spinor).
        """
        dx = self.dx
        dy = self.dy

        rho = xp.abs(self.psi1)**2 + xp.abs(self.psi2)**2
        Ax_c = 0.5 * (self.Ax + xp.roll(self.Ax, +1, axis=0))
        Ay_c = 0.5 * (self.Ay + xp.roll(self.Ay, +1, axis=1))

        grad2 = covariant_gradient_sq_spinor(
            self.psi1, self.psi2, Ax_c, Ay_c, dx, self.q
        )
        
        # Scalar energy includes quartic term
        E_scalar = 0.5 * xp.sum(grad2 + rho)

        E_defrag = 0.0
        if self.g_defrag != 0.0:
            rho_k = xp.fft.fftn(rho)
            phi_k = -rho_k / self.K2
            phi_k[0, 0] = 0.0
            phi = xp.fft.ifftn(phi_k).real
            E_defrag = 0.5 * self.g_defrag * xp.sum(rho * phi)

        E_em = 0.5 * xp.sum(self.Ex**2 + self.Ey**2 + (self.c**2) * self.Bz**2)

        E_tot = E_scalar + E_defrag + E_em

        if GPU_AVAILABLE:
            return float(cp.asnumpy(E_tot)) * dx * dy
        else:
            return float(E_tot) * dx * dy

    # ------------------------------------------------------------------
    # Convert to NumPy for saving
    # ------------------------------------------------------------------

    def to_numpy_fields(self):
        """Return all fields as numpy arrays."""
        if GPU_AVAILABLE:
            return (
                cp.asnumpy(self.psi1),
                cp.asnumpy(self.psi2),
                cp.asnumpy(self.Ax),
                cp.asnumpy(self.Ay),
                cp.asnumpy(self.Ex),
                cp.asnumpy(self.Ey),
                cp.asnumpy(self.Bz),
            )
        else:
            return (
                self.psi1,
                self.psi2,
                self.Ax,
                self.Ay,
                self.Ex,
                self.Ey,
                self.Bz,
            )


# ======================================================================
# Driver
# ======================================================================

def run_sim(args):
    if GPU_AVAILABLE:
        print("✓ GPU (CuPy) detected - spinor substrate engine")
    else:
        print("CPU mode (NumPy) - spinor substrate engine")

    sim = YeeSubstrateSpinor2D(
        L=args.L,
        dx=args.dx,
        dt=args.dt,
        c=args.c,
        q=args.q,
        g_defrag=args.g_defrag,
        n_steps=args.n_steps,
        sample_interval=args.sample_interval,
        symmetry_mode=args.symmetry,
        lambda_F=args.lambda_F,
        alpha_B=args.alpha_B,
        beta_B=args.beta_B,
    )

    out_dir = f"{args.out_prefix}_output"
    os.makedirs(out_dir, exist_ok=True)

    # Initialize
    if args.init_mode == "noise":
        sim.init_scalar_noise(args.scalar_amp)
    elif args.init_mode == "skyrmion":
        sim.init_skyrmion(args.L/2, args.L/2, charge=1)
    elif args.init_mode == "two_skyrmions":
        sim.init_two_skyrmions(args.separation)
    
    if args.gauge_amp > 0:
        sim.init_gauge_noise(args.gauge_amp)

    # Energy tracking
    energies_file = os.path.join(out_dir, f"{args.out_prefix}_energies.csv")
    with open(energies_file, "w") as fout:
        fout.write("step,time,energy\n")

    E0 = sim.total_energy()
    print(f"[START] Initial energy = {E0:.6f}")
    print()
    print("Step       Time          Energy            ΔE/E")
    print("-" * 60)

    # Main loop
    for step in range(args.n_steps + 1):
        E = sim.total_energy()
        
        # Check for instability
        if np.isnan(E) or np.isinf(E):
            print(f"\n*** INSTABILITY at step {step} ***")
            print(f"Energy = {E}")
            break
        
        # Save snapshot
        if step % args.sample_interval == 0:
            drift = abs(E - E0) / abs(E0) if E0 != 0 else 0
            
            status = ""
            if drift > 0.1:
                status = " ⚠️  LARGE DRIFT"
            
            print(f"{step:6d}  {step*sim.dt:10.4f}  {E:16.6f}  {drift:10.2e}{status}")
            
            psi1, psi2, ax, ay, ex, ey, bz = sim.to_numpy_fields()
            sx, sy, sz = spin_texture(sim.psi1, sim.psi2)
            
            if GPU_AVAILABLE:
                sx = cp.asnumpy(sx)
                sy = cp.asnumpy(sy)
                sz = cp.asnumpy(sz)
            
            snap_file = os.path.join(
                out_dir, f"{args.out_prefix}_snap_{step:06d}.npz"
            )
            np.savez(
                snap_file,
                psi1=psi1,
                psi2=psi2,
                ax=ax,
                ay=ay,
                ex=ex,
                ey=ey,
                bz=bz,
                sx=sx,
                sy=sy,
                sz=sz,
                time=step * sim.dt,
                step=step,
            )
            
            with open(energies_file, "a") as fout:
                fout.write(f"{step},{step*sim.dt},{E}\n")
        
        if step < args.n_steps:
            sim.step()

    print()
    print(f"[DONE] Final energy = {E:.6f}")
    print(f"[DONE] Energy drift = {abs(E-E0)/abs(E0):.2e}")
    print(f"[DONE] Output saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Spinor substrate simulation (upgraded from yee_substrate_coupled)"
    )
    
    # Grid
    parser.add_argument("--L", type=int, default=128, help="Grid size")
    parser.add_argument("--dx", type=float, default=1.0, help="Spatial step")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step")
    
    # Physics
    parser.add_argument("--c", type=float, default=1.0, help="Speed of light")
    parser.add_argument("--q", type=float, default=1.0, help="Gauge coupling")
    parser.add_argument("--g_defrag", type=float, default=0.5, help="Defrag strength")
    
    # Evolution
    parser.add_argument("--n_steps", type=int, default=5000, help="Number of steps")
    parser.add_argument("--sample_interval", type=int, default=100, 
                       help="Save every N steps")
    
    # Initialization
    parser.add_argument("--init_mode", type=str, default="noise",
                       choices=["noise", "skyrmion", "two_skyrmions"],
                       help="Initialization mode")
    parser.add_argument("--scalar_amp", type=float, default=0.1,
                       help="Scalar field amplitude")
    parser.add_argument("--gauge_amp", type=float, default=0.0,
                       help="Gauge field amplitude")
    parser.add_argument("--separation", type=float, default=40.0,
                       help="Separation for two skyrmions")
    
    # Symmetry selector (from original)
    parser.add_argument("--symmetry", type=str, default="none",
                       choices=["none", "fermion", "boson"],
                       help="Symmetry selection mode")
    parser.add_argument("--lambda_F", type=float, default=0.0,
                       help="Fermion repulsion strength")
    parser.add_argument("--alpha_B", type=float, default=0.0,
                       help="Boson attraction strength")
    parser.add_argument("--beta_B", type=float, default=0.0,
                       help="Boson saturation strength")
    
    # Output
    parser.add_argument("--out_prefix", type=str, default="spinor",
                       help="Output prefix")
    
    args = parser.parse_args()
    run_sim(args)


if __name__ == "__main__":
    main()
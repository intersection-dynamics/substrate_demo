#!/usr/bin/env python3
"""
coupled_photon_capable.py

2D coupled scalar + gauge toy substrate with:

- Scalar field psi (complex)
- Gauge field (ax, ay) with electric fields (ex, ey)
- Defrag self-interaction via Poisson potential
- Gauge currents from covariant derivative D psi
- Symplectic-style integrator:
    * kick–drift–kick for gauge sector
    * 2nd-order midpoint update for psi

Modes:
  --photon_test 1  : "free photon" test (psi = 0, q=0, defrag=0, gauge noise)
  --photon_test 0  : full substrate mode (scalar noise, defrag on, usual q)
"""

import argparse
import os

import numpy as np

try:
    import cupy as cp
    xp = cp
    GPU_AVAILABLE = True
except ImportError:
    xp = np
    GPU_AVAILABLE = False


# ======================================================================
# Low-level operators
# ======================================================================

def laplacian(field, dx):
    """
    2D Laplacian with periodic BCs.
    """
    return (
        xp.roll(field, +1, axis=0)
        + xp.roll(field, -1, axis=0)
        + xp.roll(field, +1, axis=1)
        + xp.roll(field, -1, axis=1)
        - 4.0 * field
    ) / (dx * dx)


def covariant_gradient_sq(psi, ax, ay, dx, q):
    """
    |D psi|^2 = |(∂x - i q ax) psi|^2 + |(∂y - i q ay) psi|^2
    """
    dpsi_x = (xp.roll(psi, -1, axis=0) - xp.roll(psi, +1, axis=0)) / (2.0 * dx)
    dpsi_y = (xp.roll(psi, -1, axis=1) - xp.roll(psi, +1, axis=1)) / (2.0 * dx)
    Dpsi_x = dpsi_x - 1j * q * ax * psi
    Dpsi_y = dpsi_y - 1j * q * ay * psi
    return xp.abs(Dpsi_x) ** 2 + xp.abs(Dpsi_y) ** 2


def gauge_current(psi, ax, ay, dx, q):
    """
    Gauge current j = q Im(psi* D psi).
    """
    dpsi_x = (xp.roll(psi, -1, axis=0) - xp.roll(psi, +1, axis=0)) / (2.0 * dx)
    dpsi_y = (xp.roll(psi, -1, axis=1) - xp.roll(psi, +1, axis=1)) / (2.0 * dx)
    Dpsi_x = dpsi_x - 1j * q * ax * psi
    Dpsi_y = dpsi_y - 1j * q * ay * psi

    jx = q * xp.imag(xp.conjugate(psi) * Dpsi_x)
    jy = q * xp.imag(xp.conjugate(psi) * Dpsi_y)
    return jx, jy


# ======================================================================
# Main simulation class
# ======================================================================

class CoupledScalarGauge2D:
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
    ):
        self.L = L
        self.dx = dx
        self.dt = dt
        self.c = c
        self.q = q
        self.g_defrag = g_defrag
        self.n_steps = n_steps
        self.sample_interval = sample_interval

        shape = (L, L)

        # Fields
        self.psi = xp.zeros(shape, dtype=xp.complex128)
        self.ax = xp.zeros(shape, dtype=xp.float64)
        self.ay = xp.zeros(shape, dtype=xp.float64)
        self.ex = xp.zeros(shape, dtype=xp.float64)
        self.ey = xp.zeros(shape, dtype=xp.float64)

        # FFT machinery for Poisson solver
        kx = xp.fft.fftfreq(L, d=dx) * 2.0 * np.pi
        ky = xp.fft.fftfreq(L, d=dx) * 2.0 * np.pi
        self.KX, self.KY = xp.meshgrid(kx, ky, indexing="ij")
        self.K2 = self.KX ** 2 + self.KY ** 2
        self.K2[0, 0] = 1.0  # avoid division by zero

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_scalar_noise(self, amp: float):
        rs = xp.random.RandomState(1234)
        self.psi = amp * (rs.randn(self.L, self.L) + 1j * rs.randn(self.L, self.L))
        print(f"[INIT] Scalar noise amplitude = {amp:g}")

    def init_gauge_noise(self, amp: float):
        rs = xp.random.RandomState(5678)
        self.ax = amp * rs.randn(self.L, self.L)
        self.ay = amp * rs.randn(self.L, self.L)
        self.ex[...] = 0.0
        self.ey[...] = 0.0
        print(f"[INIT] Gauge noise amplitude  = {amp:g}")

    # ------------------------------------------------------------------
    # Defrag potential: solve ∇² φ = -ρ
    # ------------------------------------------------------------------

    def defrag_potential(self, rho):
        rho_k = xp.fft.fftn(rho)
        phi_k = -rho_k / self.K2
        phi_k[0, 0] = 0.0
        phi = xp.fft.ifftn(phi_k).real
        return self.g_defrag * phi

    # ------------------------------------------------------------------
    # Psi RHS for midpoint integrator
    # ------------------------------------------------------------------

    def rhs_psi(self, psi):
        """
        RHS for psi: d psi / dt = -i [ -½ ∇² psi + V_defrag(psi) psi ]
        Gauge coupling in psi is intentionally omitted here for now;
        gauge back-reaction only enters through currents into (ax, ay).
        """
        lap_psi = laplacian(psi, self.dx)
        rho = xp.abs(psi) ** 2
        V_def = self.defrag_potential(rho)
        H_psi = -0.5 * lap_psi + V_def * psi
        return -1j * H_psi

    # ------------------------------------------------------------------
    # Energy diagnostic
    # ------------------------------------------------------------------

    def total_energy(self) -> float:
        rho = xp.abs(self.psi) ** 2
        grad2 = covariant_gradient_sq(self.psi, self.ax, self.ay, self.dx, self.q)

        # scalar gradient + "mass" energy (toy)
        E_scalar = xp.sum(0.5 * grad2 + 0.5 * rho)

        # defrag energy from Poisson potential
        rho_k = xp.fft.fftn(rho)
        phi_k = -rho_k / self.K2
        phi_k[0, 0] = 0.0
        phi = xp.fft.ifftn(phi_k).real
        E_defrag = 0.5 * self.g_defrag * xp.sum(rho * phi)

        # gauge electric + "magnetic" energy
        E_g_el = 0.5 * xp.sum(self.ex ** 2 + self.ey ** 2)
        grad_ax = (xp.roll(self.ax, -1, axis=0) - xp.roll(self.ax, +1, axis=0)) / (
            2.0 * self.dx
        )
        grad_ay = (xp.roll(self.ay, -1, axis=1) - xp.roll(self.ay, +1, axis=1)) / (
            2.0 * self.dx
        )
        E_g_mag = 0.5 * (self.c ** 2) * xp.sum(grad_ax ** 2 + grad_ay ** 2)

        E_tot = E_scalar + E_defrag + E_g_el + E_g_mag
        if GPU_AVAILABLE:
            return float(cp.asnumpy(E_tot))
        else:
            return float(E_tot)

    # ------------------------------------------------------------------
    # One symplectic step: kick–drift–kick + midpoint psi
    # ------------------------------------------------------------------

    def step_symplectic(self):
        dt = self.dt

        # ---- 1) First half-kick for gauge (ex, ey) ----
        jx, jy = gauge_current(self.psi, self.ax, self.ay, self.dx, self.q)
        self.ex += 0.5 * dt * (self.c ** 2 * laplacian(self.ax, self.dx) - jx)
        self.ey += 0.5 * dt * (self.c ** 2 * laplacian(self.ay, self.dx) - jy)

        # ---- 2) Drift for gauge coordinates (ax, ay) ----
        self.ax += dt * self.ex
        self.ay += dt * self.ey

        # ---- 3) Midpoint integrator for psi (Strang-like) ----
        psi0 = self.psi

        # k1 at psi0
        k1 = self.rhs_psi(psi0)
        psi_mid = psi0 + 0.5 * dt * k1

        # k2 at midpoint
        k2 = self.rhs_psi(psi_mid)
        psi_new = psi0 + dt * k2

        self.psi = psi_new

        # ---- 4) Second half-kick for gauge (ex, ey) using updated psi ----
        jx, jy = gauge_current(self.psi, self.ax, self.ay, self.dx, self.q)
        self.ex += 0.5 * dt * (self.c ** 2 * laplacian(self.ax, self.dx) - jx)
        self.ey += 0.5 * dt * (self.c ** 2 * laplacian(self.ay, self.dx) - jy)

    # ------------------------------------------------------------------
    # Helper to export NumPy versions of fields
    # ------------------------------------------------------------------

    def to_numpy_fields(self):
        if GPU_AVAILABLE:
            psi = cp.asnumpy(self.psi)
            ax = cp.asnumpy(self.ax)
            ay = cp.asnumpy(self.ay)
            ex = cp.asnumpy(self.ex)
            ey = cp.asnumpy(self.ey)
        else:
            psi, ax, ay, ex, ey = self.psi, self.ax, self.ay, self.ex, self.ey
        return psi, ax, ay, ex, ey


# ======================================================================
# Driver
# ======================================================================

def run_sim(args):
    if GPU_AVAILABLE:
        print("✓ GPU (CuPy) detected")
    else:
        print("CPU mode (NumPy)")

    sim = CoupledScalarGauge2D(
        L=args.L,
        dx=args.dx,
        dt=args.dt,
        c=args.c,
        q=args.q,
        g_defrag=args.g_defrag,
        n_steps=args.n_steps,
        sample_interval=args.sample_interval,
    )

    out_dir = f"{args.out_prefix}_output"
    os.makedirs(out_dir, exist_ok=True)

    if args.photon_test == 1:
        print("=== PHOTON TEST MODE (psi=0, q=0, g_defrag=0) ===")
        sim.init_scalar_noise(0.0)
        sim.init_gauge_noise(args.amp_gauge)
        sim.q = 0.0
        sim.g_defrag = 0.0
    else:
        print("=== SUBSTRATE MODE (scalar noise bootstrap) ===")
        sim.init_scalar_noise(args.amp_scalar)
        sim.init_gauge_noise(args.amp_gauge)

    energies = []

    for step in range(sim.n_steps + 1):
        t = step * sim.dt

        if step % sim.sample_interval == 0:
            E = sim.total_energy()
            energies.append((t, E))
            psi, ax, ay, ex, ey = sim.to_numpy_fields()
            snap_path = os.path.join(out_dir, f"{args.out_prefix}_snap_{step:06d}.npz")
            np.savez(
                snap_path,
                psi=psi,
                ax=ax,
                ay=ay,
                ex=ex,
                ey=ey,
                time=t,
            )
            print(f"[DIAG] step={step:6d}, t={t:8.3f}, E={E:12.6f}")

        if step < sim.n_steps:
            sim.step_symplectic()

    energies = np.array(energies)
    csv_path = os.path.join(out_dir, f"{args.out_prefix}_energies.csv")
    np.savetxt(
        csv_path,
        energies,
        delimiter=",",
        header="time,energy",
        comments="",
    )
    print(f"[RESULT] Saved energies -> {csv_path}")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Coupled scalar-gauge substrate with photon-test mode."
    )
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=0.05)
    parser.add_argument("--g_defrag", type=float, default=0.5)

    parser.add_argument("--amp_scalar", type=float, default=0.05)
    parser.add_argument("--amp_gauge", type=float, default=0.0)

    parser.add_argument("--n_steps", type=int, default=4000)
    parser.add_argument("--sample_interval", type=int, default=20)
    parser.add_argument("--out_prefix", type=str, default="sim")

    parser.add_argument("--photon_test", type=int, default=0)

    args = parser.parse_args()
    run_sim(args)


if __name__ == "__main__":
    main()

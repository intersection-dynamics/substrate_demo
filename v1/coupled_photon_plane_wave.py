#!/usr/bin/env python3
"""
coupled_photon_plane_wave.py

2D coupled scalar + gauge substrate with three modes:

  1) SUBSTRATE mode (default):
       - Scalar psi seeded with complex noise.
       - Gauge fields optionally seeded with small noise.
       - Defrag self-interaction enabled.
       - q, g_defrag set by CLI.

  2) PHOTON_NOISE mode (--mode photon_noise):
       - psi = 0, q = 0, g_defrag = 0.
       - Gauge potentials seeded with random noise.
       - Free-photon vacuum soup (what you already ran).

  3) PHOTON_PLANE mode (--mode photon_plane):
       - psi = 0, q = 0, g_defrag = 0.
       - Gauge potentials seeded with a single plane wave:
           ax(x,y,0) = amp_plane * cos(kx x + ky y)
         with kx, ky determined by integer mode indices (nx, ny).
       - ex = ey = 0.
       - Clean single-k photon, ideal for dispersion tests.

All modes use the same symplectic-like integrator:
  - kick–drift–kick for gauge sector
  - midpoint (2nd-order) update for psi (though psi=0 in photon modes)

Outputs:
  out_prefix_output/
    <prefix>_energies.csv
    <prefix>_snap_XXXXXX.npz

which can be analyzed by your existing scripts (analyze_photons.py,
photon_dispersion.py, etc.).
"""

import argparse
import os
from typing import Tuple

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

def laplacian(field, dx: float):
    """2D Laplacian with periodic BCs."""
    return (
        xp.roll(field, +1, axis=0)
        + xp.roll(field, -1, axis=0)
        + xp.roll(field, +1, axis=1)
        + xp.roll(field, -1, axis=1)
        - 4.0 * field
    ) / (dx * dx)


def covariant_gradient_sq(psi, ax, ay, dx: float, q: float):
    """
    |D psi|^2 = |(∂x - i q ax) psi|^2 + |(∂y - i q ay) psi|^2
    using central differences.
    """
    dpsi_x = (xp.roll(psi, -1, axis=0) - xp.roll(psi, +1, axis=0)) / (2.0 * dx)
    dpsi_y = (xp.roll(psi, -1, axis=1) - xp.roll(psi, +1, axis=1)) / (2.0 * dx)
    Dpsi_x = dpsi_x - 1j * q * ax * psi
    Dpsi_y = dpsi_y - 1j * q * ay * psi
    return xp.abs(Dpsi_x) ** 2 + xp.abs(Dpsi_y) ** 2


def gauge_current(psi, ax, ay, dx: float, q: float):
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
        self.K2[0, 0] = 1.0  # avoid division by zero at k=0

    # ------------------------------------------------------------------
    # Initialization helpers
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

    def init_plane_wave(self, amp: float, nx: int, ny: int):
        """
        Seed a single plane wave in ax:

            ax(x,y,0) = amp * cos(kx x + ky y)
            ay = 0
            ex = ey = 0

        where kx = 2π nx / (L dx), ky = 2π ny / (L dx).
        """
        L = self.L
        dx = self.dx

        # Construct coordinates
        i = xp.arange(L)
        j = xp.arange(L)
        X, Y = xp.meshgrid(i, j, indexing="ij")
        x = X * dx
        y = Y * dx

        kx = 2.0 * np.pi * nx / (L * dx)
        ky = 2.0 * np.pi * ny / (L * dx)

        phase = kx * x + ky * y

        self.ax = amp * xp.cos(phase)
        self.ay[...] = 0.0
        self.ex[...] = 0.0
        self.ey[...] = 0.0
        self.psi[...] = 0.0 + 0.0j

        print(f"[INIT] Plane-wave photon: amp={amp:g}, nx={nx}, ny={ny}")
        print(f"       kx={kx:.6f}, ky={ky:.6f}, |k|={np.sqrt(kx**2 + ky**2):.6f}")

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
        RHS for psi: d psi / dt = -i [ -½ ∇² psi + V_defrag(psi) psi ].
        Gauge coupling in psi is not included here (gauge mainly
        enters via currents), which keeps the scalar sector simple.
        In photon modes psi=0, so this is effectively zero.
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

        # Scalar energy (toy: gradient + "mass" ½|psi|^2)
        E_scalar = xp.sum(0.5 * grad2 + 0.5 * rho)

        # Defrag energy
        rho_k = xp.fft.fftn(rho)
        phi_k = -rho_k / self.K2
        phi_k[0, 0] = 0.0
        phi = xp.fft.ifftn(phi_k).real
        E_defrag = 0.5 * self.g_defrag * xp.sum(rho * phi)

        # Gauge electric + "magnetic" energy
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

        # 1) First half-kick for gauge fields
        jx, jy = gauge_current(self.psi, self.ax, self.ay, self.dx, self.q)
        self.ex += 0.5 * dt * (self.c ** 2 * laplacian(self.ax, self.dx) - jx)
        self.ey += 0.5 * dt * (self.c ** 2 * laplacian(self.ay, self.dx) - jy)

        # 2) Drift for gauge coordinates
        self.ax += dt * self.ex
        self.ay += dt * self.ey

        # 3) Midpoint update for psi
        psi0 = self.psi
        k1 = self.rhs_psi(psi0)
        psi_mid = psi0 + 0.5 * dt * k1
        k2 = self.rhs_psi(psi_mid)
        self.psi = psi0 + dt * k2

        # 4) Second half-kick for gauge fields
        jx, jy = gauge_current(self.psi, self.ax, self.ay, self.dx, self.q)
        self.ex += 0.5 * dt * (self.c ** 2 * laplacian(self.ax, self.dx) - jx)
        self.ey += 0.5 * dt * (self.c ** 2 * laplacian(self.ay, self.dx) - jy)

    # ------------------------------------------------------------------
    # Convert fields to NumPy for saving
    # ------------------------------------------------------------------

    def to_numpy_fields(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    mode = args.mode.lower()
    print(f"[MODE] {mode}")

    if mode == "substrate":
        # Full substrate: scalar noise + optional gauge noise
        sim.init_scalar_noise(args.amp_scalar)
        sim.init_gauge_noise(args.amp_gauge)

    elif mode == "photon_noise":
        # Random gauge photons, no matter
        sim.q = 0.0
        sim.g_defrag = 0.0
        sim.init_scalar_noise(0.0)
        sim.init_gauge_noise(args.amp_gauge)

    elif mode == "photon_plane":
        # Clean plane-wave photon, no matter
        sim.q = 0.0
        sim.g_defrag = 0.0
        sim.init_plane_wave(args.amp_plane, args.nx, args.ny)

    else:
        raise ValueError(f"Unknown mode '{args.mode}'. Use substrate, photon_noise, or photon_plane.")

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
        description="Coupled scalar-gauge substrate with photon plane-wave mode."
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

    parser.add_argument(
        "--mode",
        type=str,
        default="substrate",
        help="substrate | photon_noise | photon_plane",
    )

    # Plane-wave parameters (used when mode=photon_plane)
    parser.add_argument("--amp_plane", type=float, default=0.01,
                        help="Amplitude of plane-wave ax.")
    parser.add_argument("--nx", type=int, default=1,
                        help="Integer mode index in x for plane wave.")
    parser.add_argument("--ny", type=int, default=0,
                        help="Integer mode index in y for plane wave.")

    args = parser.parse_args()
    run_sim(args)


if __name__ == "__main__":
    main()

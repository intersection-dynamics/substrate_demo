#!/usr/bin/env python3
"""
maxwell_yee_plane.py

2D free Maxwell solver using a Yee-style FDTD scheme (TEz: Ex, Ey, Hz)
on a periodic square lattice. Designed as a clean photon testbed for
dispersion analysis.

Features
--------
- Yee-like staggering in space (TEz), leapfrogged in time.
- Periodic boundary conditions.
- Plane-wave photon injection with selectable (nx, ny) Fourier mode.
- GPU acceleration via CuPy if available.
- Outputs:
    <prefix>_output/
        <prefix>_energies.csv
        <prefix>_snap_XXXXXX.npz

The .npz files include:
    ex, ey, bz, ax, ay, psi, time

This makes them compatible with the existing photon_dispersion.py,
which can read 'ex' or 'ey' as usual.
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
# Yee TEz Maxwell solver
# ======================================================================

class MaxwellYee2D:
    """
    2D TEz Maxwell system with fields:
        Ex(x, y, t), Ey(x, y, t), Hz(x, y, t)

    Discrete update (periodic BCs, Yee-style):

      Hz^{n+1/2}(i,j) = Hz^{n-1/2}(i,j)
                        + dt * [ (Ex^n(i,j+1) - Ex^n(i,j))/dy
                                 - (Ey^n(i+1,j) - Ey^n(i,j))/dx ]

      Ex^{n+1}(i,j)   = Ex^n(i,j)
                        + dt * ( Hz^{n+1/2}(i,j) - Hz^{n+1/2}(i,j-1) ) / dy

      Ey^{n+1}(i,j)   = Ey^n(i,j)
                        - dt * ( Hz^{n+1/2}(i,j) - Hz^{n+1/2}(i-1,j) ) / dx

    Here we take c = 1 in the update (you can rescale time if desired).
    """

    def __init__(
        self,
        L: int,
        dx: float,
        dt: float,
        n_steps: int,
        sample_interval: int,
        c: float = 1.0,
    ):
        self.L = L
        self.dx = dx
        self.dy = dx  # square lattice
        self.dt = dt
        self.n_steps = n_steps
        self.sample_interval = sample_interval
        self.c = c

        shape = (L, L)
        # Fields (NumPy or CuPy arrays)
        self.Ex = xp.zeros(shape, dtype=xp.float64)
        self.Ey = xp.zeros(shape, dtype=xp.float64)
        self.Hz = xp.zeros(shape, dtype=xp.float64)

        # Check CFL stability condition (for info only)
        # For 2D Yee on a square grid: c dt <= dx / sqrt(2)
        cfl_limit = dx / (np.sqrt(2.0) * c)
        if dt > cfl_limit:
            print(
                f"[WARN] dt={dt:g} exceeds CFL limit ~{cfl_limit:g} "
                f"for c={c:g}, dx={dx:g}. Expect instability."
            )
        else:
            print(
                f"[INFO] CFL OK: dt={dt:g} <= {cfl_limit:g} "
                f"(for c={c:g}, dx={dx:g})"
            )

    # ------------------------------------------------------------------
    # Initialization: plane-wave photon
    # ------------------------------------------------------------------

    def init_plane_wave(self, amp: float, nx: int, ny: int):
        """
        Initialize a TEz plane wave:

          Ey(x,y,0) = amp * cos(k · r)
          Hz(x,y,0) = amp * cos(k · r)
          Ex(x,y,0) = 0

        with periodic boundary conditions and wavenumber:

          kx = 2π nx / (L dx)
          ky = 2π ny / (L dx)

        This is not an exact Yee eigenmode at t=0, but it rapidly settles.
        """

        if nx == 0 and ny == 0:
            raise ValueError("Plane wave requires nonzero (nx, ny).")

        L = self.L
        dx = self.dx

        i = xp.arange(L)
        j = xp.arange(L)
        X, Y = xp.meshgrid(i, j, indexing="ij")
        x = X * dx
        y = Y * dx

        kx = 2.0 * np.pi * nx / (L * dx)
        ky = 2.0 * np.pi * ny / (L * dx)
        k_mag = float(np.sqrt(kx**2 + ky**2))

        phase = kx * x + ky * y
        cos_phase = xp.cos(phase)

        self.Ex[...] = 0.0
        self.Ey[...] = amp * cos_phase
        self.Hz[...] = amp * cos_phase

        print(f"[INIT] Yee plane wave: amp={amp:g}, nx={nx}, ny={ny}")
        print(f"       kx={kx:.6f}, ky={ky:.6f}, |k|={k_mag:.6f}")

    # ------------------------------------------------------------------
    # One Yee update step
    # ------------------------------------------------------------------

    def step(self):
        dx = self.dx
        dy = self.dy
        dt = self.dt

        Ex = self.Ex
        Ey = self.Ey
        Hz = self.Hz

        # 1) Update Hz (curl E)   (Hz^n -> Hz^{n+1/2} conceptually)
        dEx_dy = (xp.roll(Ex, -1, axis=1) - Ex) / dy
        dEy_dx = (xp.roll(Ey, -1, axis=0) - Ey) / dx
        Hz += dt * (dEx_dy - dEy_dx)

        # 2) Update Ex (curl H)   (Ex^n -> Ex^{n+1})
        dHz_dy = (Hz - xp.roll(Hz, +1, axis=1)) / dy
        Ex += dt * dHz_dy

        # 3) Update Ey (curl H)   (Ey^n -> Ey^{n+1})
        dHz_dx = (Hz - xp.roll(Hz, +1, axis=0)) / dx
        Ey -= dt * dHz_dx

        # Store back (not strictly needed but explicit)
        self.Ex = Ex
        self.Ey = Ey
        self.Hz = Hz

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def total_energy(self) -> float:
        """
        Discrete EM energy ~ 1/2 ∑ (Ex^2 + Ey^2 + Hz^2) dx dy.
        Overall factor is irrelevant; we include dx*dy for completeness.
        """
        energy_density = 0.5 * (self.Ex ** 2 + self.Ey ** 2 + self.Hz ** 2)
        if GPU_AVAILABLE:
            total = float(cp.asnumpy(energy_density.sum())) * self.dx * self.dy
        else:
            total = float(energy_density.sum()) * self.dx * self.dy
        return total

    def to_numpy_fields(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if GPU_AVAILABLE:
            Ex = cp.asnumpy(self.Ex)
            Ey = cp.asnumpy(self.Ey)
            Hz = cp.asnumpy(self.Hz)
        else:
            Ex, Ey, Hz = self.Ex, self.Ey, self.Hz
        return Ex, Ey, Hz


# ======================================================================
# Driver
# ======================================================================

def run_sim(args):
    if GPU_AVAILABLE:
        print("✓ GPU (CuPy) detected")
    else:
        print("CPU mode (NumPy)")

    sim = MaxwellYee2D(
        L=args.L,
        dx=args.dx,
        dt=args.dt,
        n_steps=args.n_steps,
        sample_interval=args.sample_interval,
        c=args.c,
    )

    out_dir = f"{args.out_prefix}_output"
    os.makedirs(out_dir, exist_ok=True)

    # Initialize plane-wave photon
    sim.init_plane_wave(args.amp_plane, args.nx, args.ny)

    energies = []

    for step in range(sim.n_steps + 1):
        t = step * sim.dt

        if step % sim.sample_interval == 0:
            E = sim.total_energy()
            energies.append((t, E))
            Ex, Ey, Hz = sim.to_numpy_fields()

            # Save a snapshot with keys compatible with photon_dispersion.py
            snap_path = os.path.join(
                out_dir, f"{args.out_prefix}_snap_{step:06d}.npz"
            )
            # ax, ay, psi are included as zeros to keep the interface uniform
            ax = np.zeros_like(Ex)
            ay = np.zeros_like(Ex)
            psi = np.zeros_like(Ex, dtype=np.complex128)

            np.savez(
                snap_path,
                ex=Ex,
                ey=Ey,
                bz=Hz,
                ax=ax,
                ay=ay,
                psi=psi,
                time=t,
            )
            print(f"[DIAG] step={step:6d}, t={t:8.3f}, E={E:12.6e}")

        if step < sim.n_steps:
            sim.step()

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
        description="2D Yee FDTD Maxwell solver with plane-wave injection."
    )
    parser.add_argument("--L", type=int, default=64, help="Grid size (LxL).")
    parser.add_argument("--dx", type=float, default=1.0, help="Lattice spacing.")
    parser.add_argument(
        "--dt",
        type=float,
        default=0.005,
        help="Time step (must satisfy CFL for stability).",
    )
    parser.add_argument("--c", type=float, default=1.0, help="Wave speed (normally 1).")

    parser.add_argument("--n_steps", type=int, default=4000)
    parser.add_argument("--sample_interval", type=int, default=20)
    parser.add_argument("--out_prefix", type=str, default="yee_plane")

    parser.add_argument("--amp_plane", type=float, default=0.01,
                        help="Amplitude of initial plane wave.")
    parser.add_argument("--nx", type=int, default=4,
                        help="Integer mode index in x for plane wave.")
    parser.add_argument("--ny", type=int, default=0,
                        help="Integer mode index in y for plane wave.")

    args = parser.parse_args()
    run_sim(args)


if __name__ == "__main__":
    main()

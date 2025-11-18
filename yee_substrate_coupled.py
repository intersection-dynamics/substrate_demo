#!/usr/bin/env python3
"""
yee_substrate_coupled.py

Unified 2D substrate engine:

  - Complex scalar field psi(x,y,t) on a periodic grid.
  - Maxwell gauge sector using a Yee-style FDTD TEz scheme:
        Ex(x,y,t), Ey(x,y,t), Bz(x,y,t)
  - Vector potential A = (Ax, Ay) obtained by integrating E:
        E = -∂A/∂t  (temporal gauge Phi=0).
  - Minimal-ish gauge coupling between psi and A via covariant gradients.
  - Defrag potential V_def[psi] from Poisson solve on rho = |psi|^2.

Modes:
  1) substrate     : psi noise + gauge noise + defrag + coupling.
  2) photon_noise  : psi=0, defrag off, random Maxwell photons.
  3) photon_plane  : psi=0, defrag off, single plane-wave photon.

Outputs:
  <out_prefix>_output/
      <out_prefix>_energies.csv
      <out_prefix>_snap_XXXXXX.npz

Each snapshot contains:
    psi, ax, ay, ex, ey, bz, time
so it's compatible with your existing photon_dispersion.py (ex/ey).
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
# Basic finite-difference operators for psi / A (cell-centered fields)
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


# ======================================================================
# Covariant gradient, current, defrag
# ======================================================================

def covariant_gradient_sq(psi, ax_center, ay_center, dx: float, q: float):
    """
    |D psi|^2 = |(∂x - i q Ax) psi|^2 + |(∂y - i q Ay) psi|^2
    psi, Ax_center, Ay_center defined at cell centers.
    """
    dpsi_x = deriv_x(psi, dx)
    dpsi_y = deriv_y(psi, dx)
    Dpsi_x = dpsi_x - 1j * q * ax_center * psi
    Dpsi_y = dpsi_y - 1j * q * ay_center * psi
    return xp.abs(Dpsi_x)**2 + xp.abs(Dpsi_y)**2


def gauge_current(psi, ax_center, ay_center, dx: float, q: float):
    """
    Gauge current j = q Im(psi* D psi) at cell centers.
    """
    dpsi_x = deriv_x(psi, dx)
    dpsi_y = deriv_y(psi, dx)
    Dpsi_x = dpsi_x - 1j * q * ax_center * psi
    Dpsi_y = dpsi_y - 1j * q * ay_center * psi

    jx = q * xp.imag(xp.conjugate(psi) * Dpsi_x)
    jy = q * xp.imag(xp.conjugate(psi) * Dpsi_y)
    return jx, jy


# ======================================================================
# Unified Yee-based substrate engine
# ======================================================================

class YeeSubstrateCoupled2D:
    """
    Unified substrate engine:

      - psi: complex scalar at cell centers.
      - Ex, Ey, Bz: Yee-style EM fields on periodic grid.
      - Ax, Ay: vector potential (same grid as E) integrated from E.

    Time stepping:

      1) Maxwell (Yee TEz) with matter current sources Jx,Jy.
      2) Integrate A from E (E = -∂A/∂t).
      3) Update psi using midpoint RK2 with covariant gradients and defrag.
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

        shape = (L, L)

        # Scalar field
        self.psi = xp.zeros(shape, dtype=xp.complex128)

        # Yee EM fields
        self.Ex = xp.zeros(shape, dtype=xp.float64)
        self.Ey = xp.zeros(shape, dtype=xp.float64)
        self.Bz = xp.zeros(shape, dtype=xp.float64)  # Hz

        # Vector potential at same grid as E
        self.Ax = xp.zeros(shape, dtype=xp.float64)
        self.Ay = xp.zeros(shape, dtype=xp.float64)

        # FFT machinery for defrag potential (Poisson solve)
        kx = xp.fft.fftfreq(L, d=dx) * 2.0 * np.pi
        ky = xp.fft.fftfreq(L, d=dx) * 2.0 * np.pi
        self.KX, self.KY = xp.meshgrid(kx, ky, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0, 0] = 1.0  # avoid division by zero

        # CFL info
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
        rs = xp.random.RandomState(1234)
        self.psi = amp * (rs.randn(self.L, self.L) + 1j * rs.randn(self.L, self.L))
        print(f"[INIT] Scalar noise amplitude = {amp:g}")

    def init_gauge_noise(self, amp: float):
        rs = xp.random.RandomState(5678)
        self.Ex = xp.zeros_like(self.Ex)
        self.Ey = xp.zeros_like(self.Ey)
        self.Bz = xp.zeros_like(self.Bz)
        self.Ax = amp * rs.randn(self.L, self.L)
        self.Ay = amp * rs.randn(self.L, self.L)
        print(f"[INIT] Gauge A noise amplitude = {amp:g}")

    def init_plane_photon(self, amp: float, nx: int, ny: int):
        """
        Initialize a simple EY/Bz plane wave for photon tests:

            Ey(x,y,0) = amp * cos(k·r)
            Bz(x,y,0) = amp * cos(k·r)
            Ex(x,y,0) = 0

        And set A to be consistent with Ey via E = -∂A/∂t at t=0 (approx),
        starting with Ax=0, Ay=0 is fine for short-time dispersion tests.
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
        self.Bz[...] = amp * cos_phase

        self.Ax[...] = 0.0
        self.Ay[...] = 0.0

        self.psi[...] = 0.0 + 0.0j

        print(f"[INIT] Photon plane wave: amp={amp:g}, nx={nx}, ny={ny}")
        print(f"       kx={kx:.6f}, ky={ky:.6f}, |k|={k_mag:.6f}")

    # ------------------------------------------------------------------
    # Defrag potential
    # ------------------------------------------------------------------

    def defrag_potential(self, rho):
        """
        Solve ∇² phi = -rho  (periodic) then V_def = g_defrag * phi.
        """
        rho_k = xp.fft.fftn(rho)
        phi_k = -rho_k / self.K2
        phi_k[0, 0] = 0.0
        phi = xp.fft.ifftn(phi_k).real
        return self.g_defrag * phi

    # ------------------------------------------------------------------
    # RHS for psi (scalar sector)
    # ------------------------------------------------------------------

    def rhs_psi(self, psi):
        """
        d psi / dt = -i [  -½ ∇² psi + V_def * psi + ½ |psi|^2 psi ]
        with gauge-covariant gradients included in the kinetic term via Ax,Ay.

        We treat the self-interaction term ½|psi|^2 as a simple local nonlinearity.
        """

        rho = xp.abs(psi)**2

        # Interpolate Ax,Ay (edge) to cell centers (simple average in x,y).
        Ax_c = 0.5 * (self.Ax + xp.roll(self.Ax, +1, axis=0))
        Ay_c = 0.5 * (self.Ay + xp.roll(self.Ay, +1, axis=1))

        grad2 = covariant_gradient_sq(psi, Ax_c, Ay_c, self.dx, self.q)

        V_def = 0.0
        if self.g_defrag != 0.0:
            V_def = self.defrag_potential(rho)

        # Effective Hamiltonian density term (we construct in a toy way):
        # H_psi = ½|D psi|^2 + ½|psi|^2 + V_def |psi|^2
        # For the Schrödinger-like equation, we mimic
        #  -½ ∇² psi + V_def psi + ½|psi|^2 psi
        lap_psi = laplacian(psi, self.dx)
        Hpsi = -0.5 * lap_psi + V_def * psi + 0.5 * rho * psi

        return -1j * Hpsi

    # ------------------------------------------------------------------
    # One full step: Yee Maxwell + A update + psi update
    # ------------------------------------------------------------------

    def step(self):
        dt = self.dt
        dx = self.dx
        dy = self.dy
        c = self.c

        Ex = self.Ex
        Ey = self.Ey
        Bz = self.Bz
        Ax = self.Ax
        Ay = self.Ay
        psi = self.psi

        # --- 1) Compute Ax,Ay at centers and gauge current jx,jy
        Ax_c = 0.5 * (Ax + xp.roll(Ax, +1, axis=0))
        Ay_c = 0.5 * (Ay + xp.roll(Ay, +1, axis=1))
        jx_c, jy_c = gauge_current(psi, Ax_c, Ay_c, dx, self.q)

        # For simplicity, collocate J with E (no extra averaging).
        Jx = jx_c
        Jy = jy_c

        # --- 2) Maxwell update (Yee TEz with sources)
        # Bz_{n+1/2} = Bz_n + dt * (curl E)
        dEx_dy = (xp.roll(Ex, -1, axis=1) - Ex) / dy
        dEy_dx = (xp.roll(Ey, -1, axis=0) - Ey) / dx
        Bz = Bz + dt * (dEx_dy - dEy_dx)

        # Ex_{n+1} = Ex_n + dt * [ c^2 (dBz/dy) - Jx ]
        dBz_dy = (Bz - xp.roll(Bz, +1, axis=1)) / dy
        Ex = Ex + dt * (c**2 * dBz_dy - Jx)

        # Ey_{n+1} = Ey_n - dt * [ c^2 (dBz/dx) + Jy ]
        dBz_dx = (Bz - xp.roll(Bz, +1, axis=0)) / dx
        Ey = Ey - dt * (c**2 * dBz_dx + Jy)

        # --- 3) Update A using E = -∂A/∂t  (temporal gauge)
        Ax = Ax - dt * Ex
        Ay = Ay - dt * Ey

        # --- 4) Update psi with midpoint RK2
        psi0 = psi
        k1 = self.rhs_psi(psi0)
        psi_mid = psi0 + 0.5 * dt * k1
        k2 = self.rhs_psi(psi_mid)
        psi = psi0 + dt * k2

        # Store back
        self.Ex = Ex
        self.Ey = Ey
        self.Bz = Bz
        self.Ax = Ax
        self.Ay = Ay
        self.psi = psi

    # ------------------------------------------------------------------
    # Energy diagnostic
    # ------------------------------------------------------------------

    def total_energy(self) -> float:
        """
        Total toy energy:
          E_scalar = sum( ½|D psi|^2 + ½|psi|^2 )
          E_defrag = ½ g_defrag ∫ rho phi
          E_EM     = ½ ∫ (Ex^2 + Ey^2 + c^2 Bz^2)
        """
        dx = self.dx
        dy = self.dy

        rho = xp.abs(self.psi)**2
        Ax_c = 0.5 * (self.Ax + xp.roll(self.Ax, +1, axis=0))
        Ay_c = 0.5 * (self.Ay + xp.roll(self.Ay, +1, axis=1))

        grad2 = covariant_gradient_sq(self.psi, Ax_c, Ay_c, dx, self.q)
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

    def to_numpy_fields(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray, np.ndarray]:
        if GPU_AVAILABLE:
            psi = cp.asnumpy(self.psi)
            ax = cp.asnumpy(self.Ax)
            ay = cp.asnumpy(self.Ay)
            ex = cp.asnumpy(self.Ex)
            ey = cp.asnumpy(self.Ey)
            bz = cp.asnumpy(self.Bz)
        else:
            psi, ax, ay, ex, ey, bz = (
                self.psi,
                self.Ax,
                self.Ay,
                self.Ex,
                self.Ey,
                self.Bz,
            )
        return psi, ax, ay, ex, ey, bz


# ======================================================================
# Driver
# ======================================================================

def run_sim(args):
    if GPU_AVAILABLE:
        print("✓ GPU (CuPy) detected - unified Yee substrate engine")
    else:
        print("CPU mode (NumPy) - unified Yee substrate engine")

    sim = YeeSubstrateCoupled2D(
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
        # Full substrate: scalar noise + gauge A noise + defrag + coupling
        sim.init_scalar_noise(args.amp_scalar)
        sim.init_gauge_noise(args.amp_gauge)

    elif mode == "photon_noise":
        # Free photons only: psi=0, no defrag, no coupling
        sim.psi[...] = 0.0 + 0.0j
        sim.q = 0.0
        sim.g_defrag = 0.0
        sim.init_gauge_noise(0.0)
        # Add random EY/Bz noise instead of A noise
        rs = xp.random.RandomState(9876)
        sim.Ey = args.amp_plane * rs.randn(sim.L, sim.L)
        sim.Bz = args.amp_plane * rs.randn(sim.L, sim.L)
        sim.Ex[...] = 0.0
        print(f"[INIT] Photon noise mode, amp={args.amp_plane:g}")

    elif mode == "photon_plane":
        # Single plane-wave photon: psi=0, no defrag, no coupling
        sim.psi[...] = 0.0 + 0.0j
        sim.q = 0.0
        sim.g_defrag = 0.0
        sim.init_plane_photon(args.amp_plane, args.nx, args.ny)

    else:
        raise ValueError(f"Unknown mode '{args.mode}'. Use substrate, photon_noise, photon_plane.")

    energies = []

    for step in range(sim.n_steps + 1):
        t = step * sim.dt

        if step % sim.sample_interval == 0:
            E = sim.total_energy()
            energies.append((t, E))
            psi, ax, ay, ex, ey, bz = sim.to_numpy_fields()

            snap_path = os.path.join(
                out_dir, f"{args.out_prefix}_snap_{step:06d}.npz"
            )
            np.savez(
                snap_path,
                psi=psi,
                ax=ax,
                ay=ay,
                ex=ex,
                ey=ey,
                bz=bz,
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
        description="Unified 2D Yee-Maxwell substrate engine with matter coupling."
    )
    parser.add_argument("--L", type=int, default=64, help="Grid size (LxL).")
    parser.add_argument("--dx", type=float, default=1.0, help="Lattice spacing.")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step.")
    parser.add_argument("--c", type=float, default=1.0, help="Wave speed.")
    parser.add_argument("--q", type=float, default=0.05, help="Gauge charge of psi.")
    parser.add_argument("--g_defrag", type=float, default=0.5, help="Defrag coupling strength.")

    parser.add_argument("--amp_scalar", type=float, default=0.05,
                        help="Scalar noise amplitude (substrate mode).")
    parser.add_argument("--amp_gauge", type=float, default=0.01,
                        help="Initial A noise amplitude (substrate mode).")

    parser.add_argument("--n_steps", type=int, default=4000)
    parser.add_argument("--sample_interval", type=int, default=20)
    parser.add_argument("--out_prefix", type=str, default="yee_sim")

    parser.add_argument(
        "--mode",
        type=str,
        default="substrate",
        help="substrate | photon_noise | photon_plane",
    )

    # Plane-wave parameters (photon_plane mode)
    parser.add_argument("--amp_plane", type=float, default=0.01,
                        help="Amplitude for photon_noise / photon_plane.")
    parser.add_argument("--nx", type=int, default=4,
                        help="Integer mode index in x for plane wave.")
    parser.add_argument("--ny", type=int, default=0,
                        help="Integer mode index in y for plane wave.")

    args = parser.parse_args()
    run_sim(args)


if __name__ == "__main__":
    main()

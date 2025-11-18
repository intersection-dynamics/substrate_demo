#!/usr/bin/env python3
"""
gauge_scalar_substrate_gpu.py

Minimal gauge-capable substrate:
- Complex scalar field psi on sites (LxL).
- Real gauge fields ax, ay on links.
- Local, gauge-covariant kinetic term.
- Mexican-hat potential for psi.
- Static gauge sector for now (ax, ay fixed).

This is a first prototype substrate that:
- Is local and kinetic.
- Has site + link structure (gauge-flavored).
- Is ready for future dynamical gauge extensions (EM-like sector).

We do *not* claim emergent electromagnetism here.
We simply build a substrate that can *host* gauge-like structure.

Author: Ben (with AI co-pilot)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Try CuPy for GPU
try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore
    CUPY_AVAILABLE = False


def get_xp(use_gpu: bool):
    """Return (xp, is_gpu) where xp is numpy or cupy."""
    if use_gpu and CUPY_AVAILABLE:
        print("✓ GPU (CuPy) detected - using GPU acceleration")
        return cp, True
    else:
        if use_gpu and not CUPY_AVAILABLE:
            print("⚠ Requested GPU but CuPy not available, falling back to NumPy")
        else:
            print("✓ Using NumPy (CPU)")
        return np, False


class GaugeScalar2DGPU:
    """
    Gauge-covariant complex scalar field in 2D with static U(1)-like gauge links.

    Fields:
      psi: complex (L, L)
      ax: real (L, L) - link from (x,y) -> (x+1,y)
      ay: real (L, L) - link from (x,y) -> (x,y+1)

    We implement:
      - covariant forward differences D_x psi, D_y psi
      - energy diagnostic (matter + gauge)
      - RK4 evolution for psi via a Schrödinger-like equation:
            i dpsi/dt = -kappa * (D†D psi) + dV/dpsi*
    """

    def __init__(
        self,
        L: int,
        dx: float = 1.0,
        dt: float = 0.005,
        kappa: float = 1.0,
        v: float = 1.0,
        lambda_param: float = 0.5,
        g_gauge: float = 1.0,
        use_gpu: bool = True,
    ):
        self.L = L
        self.dx = dx
        self.dt = dt
        self.kappa = kappa
        self.v = v
        self.lambda_param = lambda_param
        self.g_gauge = g_gauge

        self.xp, self.is_gpu = get_xp(use_gpu)

        # Allocate fields
        self.psi = self.xp.zeros((L, L), dtype=self.xp.complex128)
        # Gauge links (static for now)
        self.ax = self.xp.zeros((L, L), dtype=self.xp.float64)
        self.ay = self.xp.zeros((L, L), dtype=self.xp.float64)

    # ----------------- initialization helpers -----------------

    def init_psi_uniform_with_noise(self, amp_noise: float = 0.01, seed: int | None = None):
        """Initialize psi ≈ v * exp(i small phase noise)."""
        if seed is not None:
            if self.is_gpu:
                self.xp.random.seed(seed)
            else:
                np.random.seed(seed)

        phase_noise = amp_noise * (2.0 * self.xp.random.rand(self.L, self.L) - 1.0)
        self.psi = self.v * self.xp.exp(1j * phase_noise)

    def init_gauge_zero(self):
        """Set ax = ay = 0 (no gauge twist)."""
        self.ax[...] = 0.0
        self.ay[...] = 0.0

    def init_gauge_vortex(self, strength: float = 0.1):
        """
        Example static gauge pattern: a simple vortex-like twist around center.
        This is just a toy pattern to test gauge-covariant structure.
        """
        L = self.L
        xs = self.xp.arange(L) - L / 2
        ys = self.xp.arange(L) - L / 2
        X, Y = self.xp.meshgrid(xs, ys, indexing="xy")
        # polar angle
        theta = self.xp.arctan2(Y, X)
        # approximate gradient of theta as gauge field
        # small factor `strength` to keep it gentle
        self.ax = strength * self.xp.diff(theta, axis=0, append=theta[0:1, :])
        self.ay = strength * self.xp.diff(theta, axis=1, append=theta[:, 0:1])

    # ----------------- covariant derivatives -----------------

    def D_forward(self, psi: np.ndarray | "cp.ndarray"):
        """
        Compute forward covariant derivatives D_x psi, D_y psi with periodic BCs.

        D_x psi(x,y) = exp(i ax(x,y)) psi(x+1,y) - psi(x,y)
        D_y psi(x,y) = exp(i ay(x,y)) psi(x,y+1) - psi(x,y)
        """
        xp = self.xp
        psi = psi

        # roll for neighbors with periodic BCs
        psi_xp = xp.roll(psi, shift=-1, axis=0)
        psi_yp = xp.roll(psi, shift=-1, axis=1)

        D_x = xp.exp(1j * self.ax) * psi_xp - psi
        D_y = xp.exp(1j * self.ay) * psi_yp - psi

        return D_x, D_y

    def D_daggerD(self, psi: np.ndarray | "cp.ndarray"):
        """
        Compute the gauge-covariant Laplacian D†D psi.

        We approximate D†D by summing backward covariant differences of D_x, D_y:

        (D†D psi)(x,y) ~ -[D_x psi(x,y) - e^{-i ax(x-1,y)} D_x psi(x-1,y)]
                         -[D_y psi(x,y) - e^{-i ay(x,y-1)} D_y psi(x,y-1)]

        This is not a precise lattice gauge theory operator, but it captures
        the spirit: gauge-covariant second differences with local couplings.
        """
        xp = self.xp
        L = self.L

        D_x, D_y = self.D_forward(psi)

        # backward neighbors with periodic BCs
        D_xm = xp.roll(D_x, shift=1, axis=0)
        D_ym = xp.roll(D_y, shift=1, axis=1)

        axm = xp.roll(self.ax, shift=1, axis=0)
        aym = xp.roll(self.ay, shift=1, axis=1)

        # backward covariant differences
        # ∇_x^† D_x ~ D_x - e^{-i a_x(x-1,y)} D_x(x-1,y)
        back_x = D_x - xp.exp(-1j * axm) * D_xm
        back_y = D_y - xp.exp(-1j * aym) * D_ym

        # minus sign to mimic Laplacian-like operator
        DdagD = -(back_x + back_y)
        return DdagD

    # ----------------- energies -----------------

    def compute_matter_energy_density(self, psi: np.ndarray | "cp.ndarray"):
        """Compute local matter energy density: |D psi|^2 + V(|psi|^2)."""
        xp = self.xp
        D_x, D_y = self.D_forward(psi)
        kin = (xp.abs(D_x) ** 2 + xp.abs(D_y) ** 2) / (self.dx ** 2)

        # Mexican hat potential: lambda/2 (|psi|^2 - v^2)^2
        mod2 = xp.abs(psi) ** 2
        V = 0.5 * self.lambda_param * (mod2 - self.v ** 2) ** 2

        return kin + V

    def compute_gauge_energy_density(self):
        """
        Compute a simple gauge energy from plaquettes:

          theta_p(x,y) = ax(x,y) + ay(x+1,y) - ax(x,y+1) - ay(x,y)

        E_gauge ~ (1/(2g^2)) theta_p^2
        """
        xp = self.xp
        ax = self.ax
        ay = self.ay

        ax_xp = xp.roll(ax, shift=-1, axis=0)
        ay_yp = xp.roll(ay, shift=-1, axis=1)

        theta_p = ax + ax_xp * 0.0  # placeholder to get shape
        # careful: build plaquette correctly
        # theta_p(x,y) = ax(x,y) + ay(x+1,y) - ax(x,y+1) - ay(x,y)
        ax_xy = ax
        ay_xp_y = xp.roll(ay, shift=-1, axis=0)
        ax_x_y1 = xp.roll(ax, shift=-1, axis=1)
        ay_xy = ay

        theta_p = ax_xy + ay_xp_y - ax_x_y1 - ay_xy

        E_g = 0.5 * (theta_p ** 2) / (self.g_gauge ** 2)
        return E_g

    def total_energy(self):
        """Return (E_total, E_matter, E_gauge) as Python floats."""
        xp = self.xp
        psi = self.psi
        e_m = self.compute_matter_energy_density(psi)
        e_g = self.compute_gauge_energy_density()
        e_tot = xp.sum(e_m + e_g)
        e_m_tot = xp.sum(e_m)
        e_g_tot = xp.sum(e_g)

        if self.is_gpu:
            e_tot = float(e_tot.get())
            e_m_tot = float(e_m_tot.get())
            e_g_tot = float(e_g_tot.get())
        else:
            e_tot = float(e_tot)
            e_m_tot = float(e_m_tot)
            e_g_tot = float(e_g_tot)
        return e_tot, e_m_tot, e_g_tot

    # ----------------- time evolution -----------------

    def rhs(self, psi: np.ndarray | "cp.ndarray"):
        """
        Right-hand side of i dpsi/dt = -kappa * D†D psi + dV/dpsi*.

        dV/dpsi* = lambda (|psi|^2 - v^2) psi
        """
        xp = self.xp
        DdagD_psi = self.D_daggerD(psi)
        mod2 = xp.abs(psi) ** 2
        dV = self.lambda_param * (mod2 - self.v ** 2) * psi
        rhs = -self.kappa * DdagD_psi + dV
        return rhs

    def evolve_step_rk4(self):
        """
        One RK4 step for psi:

          i dpsi/dt = RHS(psi)
          => dpsi/dt = -i RHS(psi)

        Gauge fields ax, ay are static in this prototype.
        """
        xp = self.xp
        dt = self.dt

        psi = self.psi

        def f(psi_local):
            return -1j * self.rhs(psi_local)

        k1 = f(psi)
        k2 = f(psi + 0.5 * dt * k1)
        k3 = f(psi + 0.5 * dt * k2)
        k4 = f(psi + dt * k3)

        self.psi = psi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # ----------------- snapshot helpers -----------------

    def psi_to_numpy(self):
        """Return psi as a NumPy array for plotting/saving."""
        if self.is_gpu:
            return cp.asnumpy(self.psi)
        else:
            return np.array(self.psi)

    def gauge_to_numpy(self):
        """Return (ax, ay) as NumPy arrays."""
        if self.is_gpu:
            return cp.asnumpy(self.ax), cp.asnumpy(self.ay)
        else:
            return np.array(self.ax), np.array(self.ay)


# ----------------- CLI / main -----------------

def run_simulation(
    L: int,
    dx: float,
    dt: float,
    kappa: float,
    v: float,
    lambda_param: float,
    g_gauge: float,
    n_steps: int,
    sample_interval: int,
    gauge_pattern: str,
    amp_noise: float,
    out_prefix: str,
    use_gpu: bool,
):
    out_dir = Path(f"{out_prefix}_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = GaugeScalar2DGPU(
        L=L,
        dx=dx,
        dt=dt,
        kappa=kappa,
        v=v,
        lambda_param=lambda_param,
        g_gauge=g_gauge,
        use_gpu=use_gpu,
    )

    print("=" * 72)
    print(" gauge_scalar_substrate_gpu.py ")
    print("=" * 72)
    print(f"L={L}, dx={dx}, dt={dt}")
    print(f"kappa={kappa}, v={v}, lambda={lambda_param}, g_gauge={g_gauge}")
    print(f"n_steps={n_steps}, sample_interval={sample_interval}")
    print(f"gauge_pattern={gauge_pattern}")
    print(f"output_dir={out_dir}")
    print("=" * 72)

    # Init psi
    sim.init_psi_uniform_with_noise(amp_noise=amp_noise, seed=1234)

    # Init gauge
    if gauge_pattern == "zero":
        sim.init_gauge_zero()
    elif gauge_pattern == "vortex":
        sim.init_gauge_vortex(strength=0.2)
    else:
        sim.init_gauge_zero()

    # Initial energy
    E0, Em0, Eg0 = sim.total_energy()
    print(f"[INIT] E_tot={E0:.6e}, E_matter={Em0:.6e}, E_gauge={Eg0:.6e}")

    times = []
    Etot_list = []
    Em_list = []
    Eg_list = []

    for step in range(n_steps + 1):
        t = step * dt

        if step % sample_interval == 0 or step == n_steps:
            Etot, Em, Eg = sim.total_energy()
            times.append(t)
            Etot_list.append(Etot)
            Em_list.append(Em)
            Eg_list.append(Eg)
            print(
                f"[DIAG] step={step:6d}, t={t:8.4f}, "
                f"E_tot={Etot:.6e}, E_m={Em:.6e}, E_g={Eg:.6e}"
            )

            # Save a snapshot of |psi| and arg(psi)
            psi_np = sim.psi_to_numpy()
            amp = np.abs(psi_np)
            phase = np.angle(psi_np)

            np.savez_compressed(
                out_dir / f"{out_prefix}_snap_{step:06d}.npz",
                psi=psi_np.astype(np.complex128),
                amp=amp.astype(np.float32),
                phase=phase.astype(np.float32),
            )

        if step < n_steps:
            sim.evolve_step_rk4()

    # Save energy diagnostics
    import pandas as pd

    df = pd.DataFrame(
        {
            "time": times,
            "E_tot": Etot_list,
            "E_matter": Em_list,
            "E_gauge": Eg_list,
        }
    )
    csv_path = out_dir / f"{out_prefix}_energies.csv"
    df.to_csv(csv_path, index=False)
    print(f"[RESULT] Saved energy diagnostics -> {csv_path}")

    # Quick energy plot
    plt.figure(figsize=(8, 4))
    plt.plot(times, Etot_list, label="E_tot")
    plt.plot(times, Em_list, label="E_matter")
    plt.plot(times, Eg_list, label="E_gauge")
    plt.xlabel("time")
    plt.ylabel("Energy")
    plt.title("Gauge-Scalar Substrate Energies")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = out_dir / f"{out_prefix}_energies.png"
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"[RESULT] Saved energy plot -> {png_path}")

    print("=" * 72)
    print(" Simulation complete ")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Gauge-capable scalar substrate (2D, static gauge links).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--L", type=int, default=128, help="Grid size (LxL).")
    parser.add_argument("--dx", type=float, default=1.0, help="Lattice spacing.")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step.")
    parser.add_argument("--kappa", type=float, default=1.0, help="Kinetic coefficient.")
    parser.add_argument("--v", type=float, default=1.0, help="VEV / background amplitude.")
    parser.add_argument(
        "--lambda_param", type=float, default=0.5, help="Self-interaction strength."
    )
    parser.add_argument(
        "--g_gauge", type=float, default=1.0, help="Gauge stiffness (smaller = softer)."
    )
    parser.add_argument("--n_steps", type=int, default=4000, help="Number of time steps.")
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=50,
        help="Sample diagnostics every this many steps.",
    )
    parser.add_argument(
        "--gauge_pattern",
        type=str,
        default="zero",
        choices=["zero", "vortex"],
        help="Static gauge pattern to use.",
    )
    parser.add_argument(
        "--amp_noise",
        type=float,
        default=0.01,
        help="Amplitude of initial random phase noise in psi.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="gauge_scalar",
        help="Prefix for output directory and files.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (NumPy) even if GPU is available.",
    )

    args = parser.parse_args()
    use_gpu = not args.cpu

    run_simulation(
        L=args.L,
        dx=args.dx,
        dt=args.dt,
        kappa=args.kappa,
        v=args.v,
        lambda_param=args.lambda_param,
        g_gauge=args.g_gauge,
        n_steps=args.n_steps,
        sample_interval=args.sample_interval,
        gauge_pattern=args.gauge_pattern,
        amp_noise=args.amp_noise,
        out_prefix=args.out_prefix,
        use_gpu=use_gpu,
    )


if __name__ == "__main__":
    main()

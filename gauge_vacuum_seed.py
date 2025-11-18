#!/usr/bin/env python3
"""
gauge_vacuum_seed.py

Prototype: pure gauge "vacuum" sector on a 2D lattice with local wave dynamics.

- Fields live on links:
    ax(x,y): component along x (link from (x,y) -> (x+1,y))
    ay(x,y): component along y (link from (x,y) -> (x,y+1))

- Conjugate "electric" fields:
    ex(x,y), ey(x,y)

- Dynamics (toy, not a full lattice gauge theory):
    d ax / dt = ex
    d ay / dt = ey
    d ex / dt = c^2 ∇^2 ax
    d ey / dt = c^2 ∇^2 ay

  So each component obeys a local wave equation:
    d^2 ax / dt^2 = c^2 ∇^2 ax   (and same for ay)

- Energy density (per site):
    E = 1/2 [ ex^2 + ey^2 + c^2 (|∇ax|^2 + |∇ay|^2) ]

We seed the vacuum with small random ax, ay and ex=ey=0,
then watch whether the energy stays finite and waves propagate.

This is a *toy gauge wave substrate*:
- local, kinetic, supports light-like propagation,
- but we are NOT claiming full U(1) lattice gauge theory here yet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore
    CUPY_AVAILABLE = False


def get_xp(use_gpu: bool):
    if use_gpu and CUPY_AVAILABLE:
        print("✓ GPU (CuPy) detected - using GPU for gauge vacuum")
        return cp, True
    else:
        if use_gpu and not CUPY_AVAILABLE:
            print("⚠ Requested GPU, but CuPy not available. Falling back to NumPy.")
        else:
            print("✓ Using NumPy (CPU) for gauge vacuum")
        return np, False


class GaugeVacuum2D:
    """
    Pure gauge vacuum sector with wave-like dynamics on a 2D lattice.

    Fields:
      ax, ay : gauge potentials (real)
      ex, ey : conjugate "electric" fields (real)

    Dynamics:
      d ax / dt = ex
      d ay / dt = ey
      d ex / dt = c^2 ∇^2 ax
      d ey / dt = c^2 ∇^2 ay
    """

    def __init__(
        self,
        L: int,
        dx: float = 1.0,
        dt: float = 0.05,
        c: float = 1.0,
        use_gpu: bool = True,
    ):
        self.L = L
        self.dx = dx
        self.dt = dt
        self.c = c

        self.xp, self.is_gpu = get_xp(use_gpu)

        # gauge potentials and electric fields
        self.ax = self.xp.zeros((L, L), dtype=self.xp.float64)
        self.ay = self.xp.zeros((L, L), dtype=self.xp.float64)
        self.ex = self.xp.zeros((L, L), dtype=self.xp.float64)
        self.ey = self.xp.zeros((L, L), dtype=self.xp.float64)

    # ---------- initialization ----------

    def seed_vacuum(self, amp_A: float = 1e-2, seed: int | None = None):
        """Small random gauge fluctuations, zero initial electric field."""
        if seed is not None:
            if self.is_gpu:
                self.xp.random.seed(seed)
            else:
                np.random.seed(seed)

        # Random potentials with small amplitude
        self.ax = amp_A * (2.0 * self.xp.random.rand(self.L, self.L) - 1.0)
        self.ay = amp_A * (2.0 * self.xp.random.rand(self.L, self.L) - 1.0)

        # Start with no electric field (pure "potential" disturbance)
        self.ex[...] = 0.0
        self.ey[...] = 0.0

    # ---------- spatial operators ----------

    def laplacian(self, field):
        """2D periodic Laplacian with spacing dx."""
        xp = self.xp
        dx2 = self.dx ** 2

        f_ip = xp.roll(field, shift=-1, axis=0)
        f_im = xp.roll(field, shift=+1, axis=0)
        f_jp = xp.roll(field, shift=-1, axis=1)
        f_jm = xp.roll(field, shift=+1, axis=1)

        lap = (f_ip + f_im + f_jp + f_jm - 4.0 * field) / dx2
        return lap

    def grad_sq(self, field):
        """
        |∇field|^2 using central differences (for energy density).
        """
        xp = self.xp
        dx = self.dx

        f_ip = xp.roll(field, shift=-1, axis=0)
        f_im = xp.roll(field, shift=+1, axis=0)
        f_jp = xp.roll(field, shift=-1, axis=1)
        f_jm = xp.roll(field, shift=+1, axis=1)

        dfdx = (f_ip - f_im) / (2.0 * dx)
        dfdy = (f_jp - f_jm) / (2.0 * dx)

        return dfdx ** 2 + dfdy ** 2

    # ---------- energy diagnostics ----------

    def energy_density(self):
        """
        Local energy density:

          e = 1/2 [ ex^2 + ey^2 + c^2 (|∇ax|^2 + |∇ay|^2) ]
        """
        xp = self.xp
        grad_ax_sq = self.grad_sq(self.ax)
        grad_ay_sq = self.grad_sq(self.ay)

        e = 0.5 * (
            self.ex ** 2
            + self.ey ** 2
            + (self.c ** 2) * (grad_ax_sq + grad_ay_sq)
        )
        return e

    def total_energy(self) -> float:
        xp = self.xp
        e = self.energy_density()
        E_tot = xp.sum(e)
        if self.is_gpu:
            E_tot = float(E_tot.get())
        else:
            E_tot = float(E_tot)
        return E_tot

    # ---------- time evolution (RK4) ----------

    def rhs(self, ax, ay, ex, ey):
        """
        Right-hand side for first-order system:

          d ax / dt = ex
          d ay / dt = ey
          d ex / dt = c^2 ∇^2 ax
          d ey / dt = c^2 ∇^2 ay
        """
        lap_ax = self.laplacian(ax)
        lap_ay = self.laplacian(ay)

        dax_dt = ex
        day_dt = ey
        dex_dt = (self.c ** 2) * lap_ax
        dey_dt = (self.c ** 2) * lap_ay

        return dax_dt, day_dt, dex_dt, dey_dt

    def step_rk4(self):
        xp = self.xp
        dt = self.dt

        ax0 = self.ax
        ay0 = self.ay
        ex0 = self.ex
        ey0 = self.ey

        def f(ax, ay, ex, ey):
            return self.rhs(ax, ay, ex, ey)

        k1_ax, k1_ay, k1_ex, k1_ey = f(ax0, ay0, ex0, ey0)

        k2_ax, k2_ay, k2_ex, k2_ey = f(
            ax0 + 0.5 * dt * k1_ax,
            ay0 + 0.5 * dt * k1_ay,
            ex0 + 0.5 * dt * k1_ex,
            ey0 + 0.5 * dt * k1_ey,
        )

        k3_ax, k3_ay, k3_ex, k3_ey = f(
            ax0 + 0.5 * dt * k2_ax,
            ay0 + 0.5 * dt * k2_ay,
            ex0 + 0.5 * dt * k2_ex,
            ey0 + 0.5 * dt * k2_ey,
        )

        k4_ax, k4_ay, k4_ex, k4_ey = f(
            ax0 + dt * k3_ax,
            ay0 + dt * k3_ay,
            ex0 + dt * k3_ex,
            ey0 + dt * k3_ey,
        )

        self.ax = ax0 + (dt / 6.0) * (k1_ax + 2 * k2_ax + 2 * k3_ax + k4_ax)
        self.ay = ay0 + (dt / 6.0) * (k1_ay + 2 * k2_ay + 2 * k3_ay + k4_ay)
        self.ex = ex0 + (dt / 6.0) * (k1_ex + 2 * k2_ex + 2 * k3_ex + k4_ex)
        self.ey = ey0 + (dt / 6.0) * (k1_ey + 2 * k2_ey + 2 * k3_ey + k4_ey)

    # ---------- helpers for saving ----------

    def to_numpy(self):
        if self.is_gpu:
            return (
                cp.asnumpy(self.ax),
                cp.asnumpy(self.ay),
                cp.asnumpy(self.ex),
                cp.asnumpy(self.ey),
            )
        else:
            return (
                np.array(self.ax),
                np.array(self.ay),
                np.array(self.ex),
                np.array(self.ey),
            )


def run_gauge_vacuum(
    L: int,
    dx: float,
    dt: float,
    c: float,
    amp_A: float,
    n_steps: int,
    sample_interval: int,
    out_prefix: str,
    use_gpu: bool,
):
    out_dir = Path(f"{out_prefix}_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = GaugeVacuum2D(L=L, dx=dx, dt=dt, c=c, use_gpu=use_gpu)

    print("=" * 72)
    print(" gauge_vacuum_seed.py ")
    print("=" * 72)
    print(f"L={L}, dx={dx}, dt={dt}, c={c}")
    print(f"amp_A={amp_A}, n_steps={n_steps}, sample_interval={sample_interval}")
    print(f"output_dir={out_dir}")
    print("=" * 72)

    sim.seed_vacuum(amp_A=amp_A, seed=1234)
    E0 = sim.total_energy()
    print(f"[INIT] Total gauge energy E0 = {E0:.6e}")

    times = []
    energies = []

    # For a quick "movie-able" snapshot: track |A| = sqrt(ax^2 + ay^2)
    for step in range(n_steps + 1):
        t = step * dt

        if step % sample_interval == 0 or step == n_steps:
            Etot = sim.total_energy()
            times.append(t)
            energies.append(Etot)
            print(f"[DIAG] step={step:6d}, t={t:8.4f}, E_tot={Etot:.6e}")

            ax_np, ay_np, ex_np, ey_np = sim.to_numpy()
            ampA = np.sqrt(ax_np ** 2 + ay_np ** 2)

            np.savez_compressed(
                out_dir / f"{out_prefix}_snap_{step:06d}.npz",
                ax=ax_np.astype(np.float32),
                ay=ay_np.astype(np.float32),
                ex=ex_np.astype(np.float32),
                ey=ey_np.astype(np.float32),
                ampA=ampA.astype(np.float32),
            )

        if step < n_steps:
            sim.step_rk4()

    # Save energy vs time
    import pandas as pd

    df = pd.DataFrame({"time": times, "E_tot": energies})
    csv_path = out_dir / f"{out_prefix}_energies.csv"
    df.to_csv(csv_path, index=False)
    print(f"[RESULT] Saved energy diagnostics -> {csv_path}")

    # Plot energy vs time
    plt.figure(figsize=(8, 4))
    plt.plot(times, energies, label="E_tot")
    plt.xlabel("time")
    plt.ylabel("Gauge energy")
    plt.title("Gauge Vacuum Energy vs Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = out_dir / f"{out_prefix}_energies.png"
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"[RESULT] Saved energy plot -> {png_path}")

    print("=" * 72)
    print(" Gauge vacuum run complete ")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Gauge vacuum seed: wave-like dynamics of link fields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--L", type=int, default=128, help="Grid size (LxL).")
    parser.add_argument("--dx", type=float, default=1.0, help="Lattice spacing.")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step.")
    parser.add_argument(
        "--c", type=float, default=1.0, help="Wave speed for gauge sector."
    )
    parser.add_argument(
        "--amp_A",
        type=float,
        default=1e-2,
        help="Initial random amplitude for gauge potentials.",
    )
    parser.add_argument(
        "--n_steps", type=int, default=2000, help="Number of time steps."
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=20,
        help="Sample diagnostics every this many steps.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="gauge_vacuum",
        help="Prefix for output directory and files.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (NumPy) even if GPU is available.",
    )

    args = parser.parse_args()
    use_gpu = not args.cpu

    run_gauge_vacuum(
        L=args.L,
        dx=args.dx,
        dt=args.dt,
        c=args.c,
        amp_A=args.amp_A,
        n_steps=args.n_steps,
        sample_interval=args.sample_interval,
        out_prefix=args.out_prefix,
        use_gpu=use_gpu,
    )


if __name__ == "__main__":
    main()

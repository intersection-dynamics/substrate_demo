#!/usr/bin/env python3
"""
gauge_lightprobe.py

Probe dispersion of a single gauge mode in a 2D wave-like gauge vacuum.

Model:
  - Real gauge potentials on links:
        ax(x,y), ay(x,y)
    (here we only excite ax for simplicity)

  - Conjugate "electric" fields:
        ex(x,y), ey(x,y)

  - Dynamics (toy, local wave equation per component):
        d ax / dt = ex
        d ay / dt = ey
        d ex / dt = c^2 ∇^2 ax
        d ey / dt = c^2 ∇^2 ay

  so that each component obeys:
        d^2 ax / dt^2 = c^2 ∇^2 ax, etc.

We excite a single sinusoidal mode in ax:

    ax(x,y, t=0) = A * cos(k · x)

with k = (2π nx / L, 2π ny / L), and ex=ey=ay=0 initially.

We then:
  - evolve the system,
  - measure the complex Fourier amplitude a_k(t) of ax,
  - FFT the time series to estimate the dominant frequency ω(k),
  - save CSV + plots + a summary text file.

This is a "light probe" for the gauge sector: if the dispersion is
approximately ω ~ c |k| (up to lattice corrections), the substrate
supports light-like gauge waves.
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
        print("✓ GPU (CuPy) detected - using GPU for gauge probe")
        return cp, True
    else:
        if use_gpu and not CUPY_AVAILABLE:
            print("⚠ Requested GPU, but CuPy not available. Falling back to NumPy.")
        else:
            print("✓ Using NumPy (CPU) for gauge probe")
        return np, False


class GaugeVacuum2D:
    """
    Same gauge vacuum model as in gauge_vacuum_seed.py:

      Fields:
        ax, ay : gauge potentials (real)
        ex, ey : electric fields (real)

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

        self.ax = self.xp.zeros((L, L), dtype=self.xp.float64)
        self.ay = self.xp.zeros((L, L), dtype=self.xp.float64)
        self.ex = self.xp.zeros((L, L), dtype=self.xp.float64)
        self.ey = self.xp.zeros((L, L), dtype=self.xp.float64)

    # ---------- spatial operators ----------

    def laplacian(self, field):
        xp = self.xp
        dx2 = self.dx ** 2

        f_ip = xp.roll(field, shift=-1, axis=0)
        f_im = xp.roll(field, shift=+1, axis=0)
        f_jp = xp.roll(field, shift=-1, axis=1)
        f_jm = xp.roll(field, shift=+1, axis=1)

        lap = (f_ip + f_im + f_jp + f_jm - 4.0 * field) / dx2
        return lap

    # ---------- time evolution ----------

    def rhs(self, ax, ay, ex, ey):
        lap_ax = self.laplacian(ax)
        lap_ay = self.laplacian(ay)

        dax_dt = ex
        day_dt = ey
        dex_dt = (self.c ** 2) * lap_ax
        dey_dt = (self.c ** 2) * lap_ay

        return dax_dt, day_dt, dex_dt, dey_dt

    def step_rk4(self):
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

    def to_numpy_ax(self):
        if self.is_gpu:
            return np.array(cp.asnumpy(self.ax))
        else:
            return np.array(self.ax)


# ---------- Fourier-mode machinery ----------

def initialize_single_mode(sim: GaugeVacuum2D, nx: int, ny: int, amp_mode: float):
    """
    Initialize ax(x,y,0) = amp_mode * cos(k·x), ay=0, ex=ey=0.

    kx = 2π nx / L, ky = 2π ny / L.
    """
    L = sim.L
    xp = sim.xp

    xs = xp.arange(L)
    ys = xp.arange(L)
    X, Y = xp.meshgrid(xs, ys, indexing="ij")  # shape (L, L)

    kx = 2.0 * np.pi * nx / float(L)
    ky = 2.0 * np.pi * ny / float(L)

    phase = kx * X + ky * Y
    sim.ax = amp_mode * xp.cos(phase)
    sim.ay[...] = 0.0
    sim.ex[...] = 0.0
    sim.ey[...] = 0.0

    print(
        "Initialized single mode: nx=%d, ny=%d, amp_mode=%.3e"
        % (nx, ny, amp_mode)
    )


def measure_mode_amplitude(ax_np: np.ndarray, nx: int, ny: int) -> complex:
    """
    Compute complex Fourier amplitude of ax at mode (nx, ny).

    We use:
      a_k = (1/N) sum_x,y ax(x,y) * exp(-i 2π (nx x + ny y)/L )

    where N = L^2.
    """
    L = ax_np.shape[0]
    xs = np.arange(L)
    ys = np.arange(L)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    kx = 2.0 * np.pi * nx / float(L)
    ky = 2.0 * np.pi * ny / float(L)
    phase = kx * X + ky * Y

    N = float(L * L)
    a_k = np.sum(ax_np * np.exp(-1j * phase)) / N
    return a_k


def run_gauge_lightprobe(
    L: int,
    dx: float,
    dt: float,
    c: float,
    nx: int,
    ny: int,
    amp_mode: float,
    n_steps: int,
    sample_interval: int,
    out_prefix: str,
    use_gpu: bool,
):
    out_dir = Path(f"{out_prefix}_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = GaugeVacuum2D(L=L, dx=dx, dt=dt, c=c, use_gpu=use_gpu)

    print("=" * 72)
    print(" gauge_lightprobe.py ")
    print("=" * 72)
    print("L=%d, dx=%.3f, dt=%.4f, c=%.3f" % (L, dx, dt, c))
    print("nx=%d, ny=%d, amp_mode=%.3e" % (nx, ny, amp_mode))
    print("n_steps=%d, sample_interval=%d" % (n_steps, sample_interval))
    print("output_dir=%s" % str(out_dir))
    print("=" * 72)

    initialize_single_mode(sim, nx=nx, ny=ny, amp_mode=amp_mode)

    times = []
    a_real = []
    a_imag = []

    # main loop
    for step in range(n_steps + 1):
        t = step * dt

        if step % sample_interval == 0 or step == n_steps:
            ax_np = sim.to_numpy_ax()
            a_k = measure_mode_amplitude(ax_np, nx=nx, ny=ny)
            times.append(t)
            a_real.append(a_k.real)
            a_imag.append(a_k.imag)

            print(
                "[DIAG] step=%6d, t=%8.4f, Re(a_k)= %+ .4e, Im(a_k)= %+ .4e"
                % (step, t, a_k.real, a_k.imag)
            )

        if step < n_steps:
            sim.step_rk4()

    times = np.array(times)
    a_real = np.array(a_real)
    a_imag = np.array(a_imag)
    a_abs = np.sqrt(a_real ** 2 + a_imag ** 2)

    # Save time series CSV
    import pandas as pd

    df = pd.DataFrame(
        {
            "time": times,
            "Re_a_k": a_real,
            "Im_a_k": a_imag,
            "abs_a_k": a_abs,
        }
    )
    csv_path = out_dir / f"{out_prefix}_mode_timeseries.csv"
    df.to_csv(csv_path, index=False)
    print("[RESULT] Saved mode time series -> %s" % csv_path)

    # ---------- FFT to estimate dispersion ----------

    # Use Re(a_k) for FFT; subtract mean to remove DC component
    y = a_real - np.mean(a_real)
    dt_eff = times[1] - times[0] if len(times) > 1 else dt

    n_samples = len(times)
    freqs = np.fft.rfftfreq(n_samples, d=dt_eff)
    fft_vals = np.fft.rfft(y)
    power = np.abs(fft_vals) ** 2

    # ignore zero-frequency bin when searching peak
    if len(power) > 1:
        idx0 = 1
    else:
        idx0 = 0
    peak_idx = idx0 + np.argmax(power[idx0:])
    f_peak = freqs[peak_idx]
    omega_peak = 2.0 * np.pi * f_peak

    print(
        "[RESULT] Dominant frequency: f_peak = %.6f, omega_peak = %.6f"
        % (f_peak, omega_peak)
    )

    # ---------- plots ----------

    # |a_k|(t)
    plt.figure(figsize=(8, 4))
    plt.plot(times, a_abs)
    plt.xlabel("time")
    plt.ylabel("|a_k|")
    plt.title("|a_k(t)|, mode (nx=%d, ny=%d)" % (nx, ny))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png1 = out_dir / f"{out_prefix}_mode_abs_time_series.png"
    plt.savefig(png1, dpi=200)
    plt.close()
    print("[RESULT] Saved |a_k| plot -> %s" % png1)

    # Re and Im vs time
    plt.figure(figsize=(10, 4))
    plt.plot(times, a_real, label="Re(a_k)")
    plt.plot(times, a_imag, label="Im(a_k)")
    plt.xlabel("time")
    plt.ylabel("a_k")
    plt.title("Mode amplitude a_k(t), (nx=%d, ny=%d)" % (nx, ny))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png2 = out_dir / f"{out_prefix}_mode_reim_time_series.png"
    plt.savefig(png2, dpi=200)
    plt.close()
    print("[RESULT] Saved Re/Im(a_k) plot -> %s" % png2)

    # Power spectrum
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, power)
    plt.axvline(f_peak, linestyle="--")
    plt.xlabel("frequency f")
    plt.ylabel("|FFT(Re(a_k))|^2")
    plt.title("Mode power spectrum, (nx=%d, ny=%d)" % (nx, ny))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png3 = out_dir / f"{out_prefix}_mode_spectrum.png"
    plt.savefig(png3, dpi=200)
    plt.close()
    print("[RESULT] Saved power spectrum plot -> %s" % png3)

    # ---------- summary text (ASCII only) ----------

    summary_path = out_dir / f"{out_prefix}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("gauge_lightprobe summary\n")
        f.write("-----------------------\n")
        f.write("L = %d\n" % L)
        f.write("dx = %.6f\n" % dx)
        f.write("dt = %.6f\n" % dt)
        f.write("c = %.6f\n" % c)
        f.write("nx = %d\n" % nx)
        f.write("ny = %d\n" % ny)
        f.write("amp_mode = %.6e\n" % amp_mode)
        f.write("n_steps = %d\n" % n_steps)
        f.write("sample_interval = %d\n" % sample_interval)
        f.write("\n")
        f.write("f_peak = %.8f\n" % f_peak)
        f.write("omega_peak = %.8f\n" % omega_peak)
        f.write("\n")
        f.write("Note: Expect continuum dispersion omega ~ c * k\n")
        f.write("with k = 2*pi*sqrt(nx^2 + ny^2)/L (in lattice units).\n")

    print("[RESULT] Saved summary -> %s" % summary_path)

    print("=" * 72)
    print(" Gauge light probe run complete ")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Gauge light probe: dispersion of a single gauge mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--L", type=int, default=128, help="Grid size (LxL).")
    parser.add_argument("--dx", type=float, default=1.0, help="Lattice spacing.")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step.")
    parser.add_argument(
        "--c", type=float, default=1.0, help="Wave speed for gauge sector."
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=1,
        help="Mode index in x (k_x = 2*pi*nx/L).",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=0,
        help="Mode index in y (k_y = 2*pi*ny/L).",
    )
    parser.add_argument(
        "--amp_mode",
        type=float,
        default=1e-2,
        help="Initial amplitude of the gauge mode.",
    )
    parser.add_argument(
        "--n_steps", type=int, default=4000, help="Number of time steps."
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=1,
        help="Sample the mode every this many steps.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="gauge_light_k10",
        help="Prefix for output directory and files.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (NumPy) even if GPU is available.",
    )

    args = parser.parse_args()
    use_gpu = not args.cpu

    run_gauge_lightprobe(
        L=args.L,
        dx=args.dx,
        dt=args.dt,
        c=args.c,
        nx=args.nx,
        ny=args.ny,
        amp_mode=args.amp_mode,
        n_steps=args.n_steps,
        sample_interval=args.sample_interval,
        out_prefix=args.out_prefix,
        use_gpu=use_gpu,
    )


if __name__ == "__main__":
    main()

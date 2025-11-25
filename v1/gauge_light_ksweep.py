#!/usr/bin/env python3
"""
gauge_light_ksweep.py

Sweep over several gauge modes (nx, ny) in the 2D gauge vacuum
and measure their dispersion relation omega(k).

Uses the same toy gauge model as gauge_lightprobe.py:

  Fields:
    ax, ay : gauge potentials (real)
    ex, ey : electric fields (real)

  Dynamics:
    d ax / dt = ex
    d ay / dt = ey
    d ex / dt = c^2 ∇^2 ax
    d ey / dt = c^2 ∇^2 ay

We excite each mode with:

    ax(x,y,0) = amp_mode * cos(2*pi*(nx*x + ny*y)/L)

and ay = ex = ey = 0 initially.

For each (nx, ny) we:
  - evolve the gauge field,
  - track the complex Fourier amplitude a_k(t) of that mode,
  - FFT the time series to estimate the dominant frequency,
  - record k_mod, f_peak, omega_peak.

Outputs:
  - CSV with columns: nx, ny, k_mod, f_peak, omega_peak
  - omega_vs_k plot
  - text summary (ASCII only).
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
        print("✓ GPU (CuPy) detected - using GPU for k-sweep")
        return cp, True
    else:
        if use_gpu and not CUPY_AVAILABLE:
            print("⚠ Requested GPU, but CuPy not available. Falling back to NumPy.")
        else:
            print("✓ Using NumPy (CPU) for k-sweep")
        return np, False


class GaugeVacuum2D:
    """
    Pure gauge vacuum with wave-like dynamics:

      d ax / dt = ex
      d ay / dt = ey
      d ex / dt = c^2 ∇^2 ax
      d ey / dt = c^2 ∇^2 ay
    """

    def __init__(self, L: int, dx: float = 1.0, dt: float = 0.05,
                 c: float = 1.0, use_gpu: bool = True):
        self.L = L
        self.dx = dx
        self.dt = dt
        self.c = c

        self.xp, self.is_gpu = get_xp(use_gpu)

        self.ax = self.xp.zeros((L, L), dtype=self.xp.float64)
        self.ay = self.xp.zeros((L, L), dtype=self.xp.float64)
        self.ex = self.xp.zeros((L, L), dtype=self.xp.float64)
        self.ey = self.xp.zeros((L, L), dtype=self.xp.float64)

    # ----- spatial operators -----

    def laplacian(self, field):
        xp = self.xp
        dx2 = self.dx ** 2

        f_ip = xp.roll(field, shift=-1, axis=0)
        f_im = xp.roll(field, shift=+1, axis=0)
        f_jp = xp.roll(field, shift=-1, axis=1)
        f_jm = xp.roll(field, shift=+1, axis=1)

        lap = (f_ip + f_im + f_jp + f_jm - 4.0 * field) / dx2
        return lap

    # ----- time evolution -----

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


# ----- mode helpers -----

def initialize_single_mode(sim: GaugeVacuum2D, nx: int, ny: int, amp_mode: float):
    """
    ax(x,y,0) = amp_mode * cos(2*pi*(nx*x + ny*y)/L), ay=ex=ey=0.
    """
    L = sim.L
    xp = sim.xp

    xs = xp.arange(L)
    ys = xp.arange(L)
    X, Y = xp.meshgrid(xs, ys, indexing="ij")

    kx = 2.0 * np.pi * nx / float(L)
    ky = 2.0 * np.pi * ny / float(L)

    phase = kx * X + ky * Y
    sim.ax = amp_mode * xp.cos(phase)
    sim.ay[...] = 0.0
    sim.ex[...] = 0.0
    sim.ey[...] = 0.0

    print("  Initialized mode nx=%d, ny=%d, amp_mode=%.3e" % (nx, ny, amp_mode))


def measure_mode_amplitude(ax_np: np.ndarray, nx: int, ny: int) -> complex:
    """
    a_k = (1/N) sum_{x,y} ax(x,y) * exp(-i 2π (nx x + ny y)/L).
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


def parse_k_list(k_list_str: str):
    """
    Parse a string like "1,0;2,0;3,0;1,1;2,1" into a list of (nx, ny).
    """
    pairs = []
    for chunk in k_list_str.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(",")
        if len(parts) != 2:
            raise ValueError("Bad k-list entry: %r" % chunk)
        nx = int(parts[0])
        ny = int(parts[1])
        pairs.append((nx, ny))
    return pairs


def measure_frequency(times: np.ndarray, a_real: np.ndarray):
    """
    Given times and Re(a_k), subtract DC component, FFT, and
    find the dominant frequency and angular frequency.
    """
    y = a_real - np.mean(a_real)
    if len(times) < 2:
        # fallback
        return 0.0, 0.0

    dt_eff = times[1] - times[0]
    n_samples = len(times)

    freqs = np.fft.rfftfreq(n_samples, d=dt_eff)
    fft_vals = np.fft.rfft(y)
    power = np.abs(fft_vals) ** 2

    if len(power) > 1:
        idx0 = 1
    else:
        idx0 = 0
    peak_idx = idx0 + np.argmax(power[idx0:])
    f_peak = float(freqs[peak_idx])
    omega_peak = 2.0 * np.pi * f_peak
    return f_peak, omega_peak


def run_k_sweep(
    L: int,
    dx: float,
    dt: float,
    c: float,
    modes: list[tuple[int, int]],
    amp_mode: float,
    n_steps: int,
    sample_interval: int,
    out_prefix: str,
    use_gpu: bool,
):
    out_dir = Path(f"{out_prefix}_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" gauge_light_ksweep.py ")
    print("=" * 72)
    print("L=%d, dx=%.3f, dt=%.4f, c=%.3f" % (L, dx, dt, c))
    print("n_steps=%d, sample_interval=%d" % (n_steps, sample_interval))
    print("modes:", modes)
    print("amp_mode=%.3e" % amp_mode)
    print("output_dir=%s" % str(out_dir))
    print("=" * 72)

    results = []

    for (nx, ny) in modes:
        print("\n--- Probing mode (nx=%d, ny=%d) ---" % (nx, ny))
        # new sim for each mode
        sim = GaugeVacuum2D(L=L, dx=dx, dt=dt, c=c, use_gpu=use_gpu)
        initialize_single_mode(sim, nx, ny, amp_mode)

        times = []
        a_real = []
        a_imag = []

        for step in range(n_steps + 1):
            t = step * dt

            if step % sample_interval == 0 or step == n_steps:
                ax_np = sim.to_numpy_ax()
                a_k = measure_mode_amplitude(ax_np, nx=nx, ny=ny)
                times.append(t)
                a_real.append(a_k.real)
                a_imag.append(a_k.imag)

            if step < n_steps:
                sim.step_rk4()

        times = np.array(times)
        a_real = np.array(a_real)
        a_imag = np.array(a_imag)

        f_peak, omega_peak = measure_frequency(times, a_real)

        k_mod = 2.0 * np.pi * np.sqrt(nx * nx + ny * ny) / float(L)

        print(
            "  Mode (nx=%d, ny=%d): k_mod=%.6f, f_peak=%.6f, omega_peak=%.6f"
            % (nx, ny, k_mod, f_peak, omega_peak)
        )

        results.append(
            {
                "nx": nx,
                "ny": ny,
                "k_mod": k_mod,
                "f_peak": f_peak,
                "omega_peak": omega_peak,
            }
        )

    # Save CSV
    import pandas as pd

    df = pd.DataFrame(results)
    csv_path = out_dir / f"{out_prefix}_dispersion.csv"
    df.to_csv(csv_path, index=False)
    print("\n[RESULT] Saved dispersion data -> %s" % csv_path)

    # Plot omega vs k
    k_vals = df["k_mod"].to_numpy()
    omega_vals = df["omega_peak"].to_numpy()

    plt.figure(figsize=(6, 5))
    plt.scatter(k_vals, omega_vals, label="measured omega(k)")
    # Fit a line through the origin for visual slope
    if len(k_vals) >= 1:
        slope = np.sum(k_vals * omega_vals) / np.sum(k_vals ** 2)
        k_line = np.linspace(0.0, k_vals.max() * 1.1, 100)
        omega_line = slope * k_line
        plt.plot(k_line, omega_line, "--", label="linear fit, slope=%.3f" % slope)
    plt.xlabel("k_mod")
    plt.ylabel("omega_peak")
    plt.title("Gauge-mode dispersion: omega vs k")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    png_path = out_dir / f"{out_prefix}_dispersion.png"
    plt.savefig(png_path, dpi=200)
    plt.close()
    print("[RESULT] Saved dispersion plot -> %s" % png_path)

    # ASCII summary
    summary_path = out_dir / f"{out_prefix}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("gauge_light_ksweep summary\n")
        f.write("--------------------------\n")
        f.write("L = %d\n" % L)
        f.write("dx = %.6f\n" % dx)
        f.write("dt = %.6f\n" % dt)
        f.write("c = %.6f\n" % c)
        f.write("n_steps = %d\n" % n_steps)
        f.write("sample_interval = %d\n" % sample_interval)
        f.write("amp_mode = %.6e\n" % amp_mode)
        f.write("\n")
        for r in results:
            f.write(
                "mode nx=%d ny=%d  k_mod=%.8f  f_peak=%.8f  omega_peak=%.8f\n"
                % (
                    r["nx"],
                    r["ny"],
                    r["k_mod"],
                    r["f_peak"],
                    r["omega_peak"],
                )
            )
        if len(k_vals) >= 1:
            slope = np.sum(k_vals * omega_vals) / np.sum(k_vals ** 2)
            f.write("\nApprox linear slope omega/k = %.8f\n" % slope)
            f.write("This is the effective speed of light in lattice units.\n")

    print("[RESULT] Saved summary -> %s" % summary_path)
    print("\nSweep complete.")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Gauge light k-sweep: measure omega(k) for several modes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--L", type=int, default=128, help="Grid size (LxL).")
    parser.add_argument("--dx", type=float, default=1.0, help="Lattice spacing.")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step.")
    parser.add_argument(
        "--c", type=float, default=1.0, help="Wave speed for the gauge sector."
    )
    parser.add_argument(
        "--amp_mode",
        type=float,
        default=5e-3,
        help="Initial amplitude of gauge mode.",
    )
    parser.add_argument(
        "--n_steps", type=int, default=4000, help="Number of time steps."
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=1,
        help="Sample mode amplitude every this many steps.",
    )
    parser.add_argument(
        "--k_list",
        type=str,
        default="1,0;2,0;3,0;1,1;2,1",
        help='List of modes as "nx,ny;nx,ny;...".',
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="gauge_light_ksweep",
        help="Prefix for output directory and files.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (NumPy) even if GPU is available.",
    )

    args = parser.parse_args()
    use_gpu = not args.cpu
    modes = parse_k_list(args.k_list)

    run_k_sweep(
        L=args.L,
        dx=args.dx,
        dt=args.dt,
        c=args.c,
        modes=modes,
        amp_mode=args.amp_mode,
        n_steps=args.n_steps,
        sample_interval=args.sample_interval,
        out_prefix=args.out_prefix,
        use_gpu=use_gpu,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
berry_on_rays_analyzer.py

Compute a Berry-like geometric phase along a trajectory of substrate states,
treating each saved snapshot as a *ray* in projective Hilbert space.

We:
  - load snapshots:  <prefix>_snap_XXXXXX.npz  from a directory
  - extract psi(x,y) and flatten to a big Hilbert vector |psi_n>
  - normalize each:        |psi_n> -> |psi_n>/||psi_n||
  - treat global phase as gauge: work only with *rays* [psi_n]
  - compute overlaps:      c_n = <psi_{n+1} | psi_n>
  - compute phase increments:   Δφ_n = arg(c_n)
  - compute cumulative phase:   Φ_k = sum_{n<k} Δφ_n
  - total Berry-like phase:     Φ_total = Φ_{N-1} (optionally wrapped to [-π, π])

This is a ray-based, projective-space view of the evolution: we are really
tracking a loop in projective Hilbert space and its U(1) holonomy.

Outputs:
  - <out_prefix>_berry_output/
       berry_phases.csv        (step_idx, time, phase_increment, phase_cumulative)
       berry_phase_vs_time.png (Φ(t))
       berry_phase_increment_hist.png (histogram of Δφ)

Assumptions:
  - snapshots named "<prefix>_snap_XXXXXX.npz"
  - each npz contains at least "psi" and "time".
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt


def find_snapshots(input_dir, prefix):
    """
    Find all snapshot files matching prefix_snap_XXXXXX.npz in input_dir.
    Returns a list of (step_idx, full_path) sorted by step_idx.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}_snap_(\d+)\.npz$")
    snaps = []

    for fname in os.listdir(input_dir):
        m = pattern.match(fname)
        if m:
            step = int(m.group(1))
            path = os.path.join(input_dir, fname)
            snaps.append((step, path))

    snaps.sort(key=lambda t: t[0])
    return snaps


def load_psi_ray(path):
    """
    Load psi from snapshot and return:
        time (float),
        normalized flattened psi (1D complex128 numpy array)

    We treat |psi> as a Hilbert vector and normalize to unit norm, so it
    represents a *ray* in projective Hilbert space.
    """
    data = np.load(path)
    psi = data["psi"]
    t = float(data["time"])

    vec = psi.astype(np.complex128).ravel()
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        raise ValueError(f"Zero-norm psi in file {path}")

    vec /= norm
    return t, vec


def compute_berry_on_rays(times, rays):
    """
    Given:
        times: list/array of times t_n
        rays:  list/array of normalized vectors |psi_n>, shape (N, D)

    We compute:
        c_n      = <psi_{n+1} | psi_n>  (complex overlaps)
        Δφ_n     = arg(c_n)
        Φ_k      = cumulative sum of Δφ up to step k

    Returns:
        phase_increments: array of length N-1
        phase_cumulative: array of length N   (Φ_0 = 0, Φ_{k>0} = sum_{n<k} Δφ_n)
        total_phase_raw:  Φ_{N-1} (float)
        total_phase_wrapped: Φ_{N-1} wrapped to [-π, π]
    """
    N = len(rays)
    if N < 2:
        raise ValueError("Need at least 2 snapshots to define a Berry phase.")

    rays_arr = np.stack(rays, axis=0)  # (N, D)

    # overlaps c_n = <psi_{n+1} | psi_n>
    overlaps = np.empty(N - 1, dtype=np.complex128)
    for n in range(N - 1):
        # np.vdot does conjugate(psi_{n+1}) * psi_n and sums
        overlaps[n] = np.vdot(rays_arr[n + 1], rays_arr[n])

    # phase increments Δφ_n
    phase_increments = np.angle(overlaps)

    # cumulative phase Φ_k
    phase_cumulative = np.zeros(N, dtype=np.float64)
    phase_cumulative[1:] = np.cumsum(phase_increments)

    total_phase_raw = float(phase_cumulative[-1])
    total_phase_wrapped = float(np.arctan2(np.sin(total_phase_raw), np.cos(total_phase_raw)))

    return phase_increments, phase_cumulative, total_phase_raw, total_phase_wrapped


def save_results(out_dir, out_prefix, steps, times, phase_inc, phase_cum,
                 total_raw, total_wrapped):
    """
    Save CSV + plots.
    """
    os.makedirs(out_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(out_dir, f"{out_prefix}_berry_phases.csv")
    arr = np.column_stack([steps, times, phase_inc_with_pad(phase_inc), phase_cum])
    header = "step_idx,time,phase_increment,phase_cumulative"
    np.savetxt(csv_path, arr, delimiter=",", header=header, comments="")
    print(f"[SAVE] {csv_path}")

    # Text summary
    summary_path = os.path.join(out_dir, f"{out_prefix}_berry_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Berry-on-rays analysis\n")
        f.write("======================\n\n")
        f.write(f"Number of snapshots: {len(steps)}\n")
        f.write(f"First step: {steps[0]}, last step: {steps[-1]}\n")
        f.write(f"First time: {times[0]:.6f}, last time: {times[-1]:.6f}\n")
        f.write("\n")
        f.write(f"Total Berry-like phase (raw):     {total_raw:.6f} rad\n")
        f.write(f"Total Berry-like phase (wrapped): {total_wrapped:.6f} rad\n")
    print(f"[SAVE] {summary_path}")

    # Plot: cumulative phase vs time
    plt.figure(figsize=(7, 4))
    plt.plot(times, phase_cum, linestyle="-")
    plt.xlabel("time")
    plt.ylabel("cumulative phase Φ(t) [rad]")
    plt.title("Berry-like cumulative phase vs time")
    plt.tight_layout()
    png_path = os.path.join(out_dir, f"{out_prefix}_berry_phase_vs_time.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"[SAVE] {png_path}")

    # Plot: histogram of phase increments
    plt.figure(figsize=(6, 4))
    plt.hist(phase_inc, bins=60, density=True)
    plt.xlabel("phase increment Δφ [rad]")
    plt.ylabel("PDF")
    plt.title("Berry-like phase increments Δφ")
    plt.tight_layout()
    png_path = os.path.join(out_dir, f"{out_prefix}_berry_phase_increment_hist.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"[SAVE] {png_path}")


def phase_inc_with_pad(phase_inc):
    """
    For CSV: we want phase_increment aligned with phase_cumulative length.
    We set phase_increment[0] = 0 for the first snapshot (no previous step).
    """
    N = len(phase_inc) + 1
    out = np.zeros(N, dtype=np.float64)
    out[1:] = phase_inc
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Berry-on-rays analyzer: compute Berry-like phase along a trajectory of substrate snapshots."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="yee_sim_output",
        help="Directory containing <prefix>_snap_XXXXXX.npz files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="yee_sim",
        help="Snapshot prefix (before '_snap_XXXXXX.npz').",
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=None,
        help="Optional: minimum step index to include (e.g. 0).",
    )
    parser.add_argument(
        "--end_step",
        type=int,
        default=None,
        help="Optional: maximum step index to include (inclusive).",
    )
    parser.add_argument(
        "--step_stride",
        type=int,
        default=1,
        help="Use every 'step_stride'-th snapshot.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="yee_sim",
        help="Prefix for output files (summary, CSV, plots).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="berry_output",
        help="Directory to write Berry analysis outputs.",
    )

    args = parser.parse_args()

    snaps = find_snapshots(args.input_dir, args.prefix)
    if not snaps:
        raise SystemExit(
            f"No snapshots found in '{args.input_dir}' with prefix '{args.prefix}_snap_XXXXXX.npz'"
        )

    # Filter by step range
    steps_all = np.array([s for (s, _) in snaps], dtype=int)
    paths_all = np.array([p for (_, p) in snaps])

    mask = np.ones_like(steps_all, dtype=bool)
    if args.start_step is not None:
        mask &= (steps_all >= args.start_step)
    if args.end_step is not None:
        mask &= (steps_all <= args.end_step)

    steps = steps_all[mask]
    paths = paths_all[mask]

    if len(steps) == 0:
        raise SystemExit("No snapshots left after applying start_step/end_step filter.")

    # Apply stride
    steps = steps[::args.step_stride]
    paths = paths[::args.step_stride]

    print(f"[INFO] Using {len(steps)} snapshots from '{args.input_dir}'")
    print(f"       steps {steps[0]} .. {steps[-1]} (stride={args.step_stride})")

    # Load rays
    times = []
    rays = []
    for step, path in zip(steps, paths):
        t, ray = load_psi_ray(path)
        times.append(t)
        rays.append(ray)
        # simple progress print for long runs
        if (step == steps[0]) or (step == steps[-1]) or (step % (10 * args.step_stride) == 0):
            print(f"[LOAD] step={step:6d}, time={t:10.4f}, norm=1.000")

    times = np.array(times, dtype=np.float64)

    # Compute Berry-like phase on rays
    phase_inc, phase_cum, total_raw, total_wrapped = compute_berry_on_rays(times, rays)

    print("\n[RESULT] Berry-on-rays:")
    print(f"  Total phase (raw):     {total_raw:.6f} rad")
    print(f"  Total phase (wrapped): {total_wrapped:.6f} rad")

    # Save CSV + plots
    save_results(
        out_dir=args.out_dir,
        out_prefix=args.out_prefix,
        steps=steps,
        times=times,
        phase_inc=phase_inc,
        phase_cum=phase_cum,
        total_raw=total_raw,
        total_wrapped=total_wrapped,
    )


if __name__ == "__main__":
    main()

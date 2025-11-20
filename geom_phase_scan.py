#!/usr/bin/env python3
"""
geom_phase_scan.py

Scan geometric phases for many defect trajectories in a substrate simulation.

Inputs:
-------
1) Snapshot files: <input_dir>/<prefix>_snap_*.npz
   Each contains:
     - psi : complex field, shape (L, L)
     - time: scalar float

2) Trajectory CSV: <traj_csv>
   Produced by defect_braid_analysis_gpu.py, with header:
     traj_id,charge,time,x,y

What it does:
-------------
- Rebuilds trajectories from the CSV.
- For each trajectory:
    * if length >= min_len AND |charge| in allowed_charges:
        - for each point (t, x, y), find the corresponding snapshot
        - sample the phase of psi on a small ring around (x, y)
        - angle-mean the samples on that ring
        - unwrap the phase along the trajectory
        - compute total geometric phase (end - start)
- Saves a CSV with per-trajectory phases:
    traj_id, charge, length, total_phase, phase_cycles
- Optionally makes a histogram PNG of phase_cycles and/or total_phase.

Usage example:
--------------
python geom_phase_scan.py ^
  --input_dir yee_coupled_output ^
  --prefix yee_coupled ^
  --traj_csv yee_coupled_defects_trajectories.csv ^
  --min_len 80 ^
  --allowed_charges 1 -1 ^
  --ring_radius 1.5 ^
  --ring_samples 16 ^
  --out_prefix yee_coupled_geom

"""

import os
import glob
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Snapshot loading
# ---------------------------------------------------------------------

def load_snapshots(input_dir: str, prefix: str):
    """
    Load all snapshots of the form:
      input_dir / f"{prefix}_snap_*.npz"

    Returns:
      snaps: list of dicts with keys: "psi", "time"
      times: np.ndarray of times
      time_to_idx: dict mapping rounded time -> index
    """
    pattern = os.path.join(input_dir, f"{prefix}_snap_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No snapshot files matching {pattern}")

    snaps = []
    times = []

    for fname in files:
        data = np.load(fname)
        if "psi" not in data or "time" not in data:
            raise KeyError(f"{fname} missing 'psi' or 'time' arrays.")
        t = float(data["time"])
        snaps.append({
            "psi": data["psi"],
            "time": t,
        })
        times.append(t)

    times = np.array(times, dtype=float)

    # Build a lookup from rounded time to index
    time_to_idx: Dict[float, int] = {}
    for i, t in enumerate(times):
        key = round(t, 10)  # robust against tiny float noise
        time_to_idx[key] = i

    return snaps, times, time_to_idx


# ---------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------

def load_trajectories_from_csv(traj_csv: str):
    """
    Load trajectories from CSV with header:
      traj_id,charge,time,x,y

    Returns:
      trajectories: dict[traj_id] = {
          "id": int,
          "charge": int,
          "points": list of (t, x, y)
      }
    """
    data = np.genfromtxt(traj_csv, delimiter=",", names=True)
    # Ensure structured array fields exist
    required = ["traj_id", "charge", "time", "x", "y"]
    for name in required:
        if name not in data.dtype.names:
            raise KeyError(f"CSV {traj_csv} missing required column '{name}'")

    traj_ids = data["traj_id"]
    charges = data["charge"]
    times = data["time"]
    xs = data["x"]
    ys = data["y"]

    trajectories: Dict[int, Dict] = {}
    for tid, q, t, x, y in zip(traj_ids, charges, times, xs, ys):
        tid_int = int(tid)
        q_int = int(q)
        if tid_int not in trajectories:
            trajectories[tid_int] = {
                "id": tid_int,
                "charge": q_int,
                "points": [],
            }
        trajectories[tid_int]["points"].append((float(t), float(x), float(y)))

    # Sort points in time for each trajectory
    for tr in trajectories.values():
        tr["points"].sort(key=lambda p: p[0])

    return trajectories


# ---------------------------------------------------------------------
# Phase sampling on a ring
# ---------------------------------------------------------------------

def phase_at_ring(psi: np.ndarray, x: float, y: float, r: float = 1.5, n_samples: int = 16):
    """
    Sample phase of psi on a small ring around (x,y) with radius r.
    Returns average phase using a proper angle-mean (vector average).

    psi: complex field, shape (Lx, Ly)
    x,y: position in index units (0..L-1)
    r: radius in index units
    n_samples: number of samples on the ring

    Output:
      avg_phase in (-π, π]
    """
    Lx, Ly = psi.shape
    angles = []

    for k in range(n_samples):
        theta = 2.0 * np.pi * k / n_samples
        xx = x + r * np.cos(theta)
        yy = y + r * np.sin(theta)
        ix = int(np.floor(xx)) % Lx
        iy = int(np.floor(yy)) % Ly
        val = psi[ix, iy]
        ang = np.angle(val)
        angles.append(ang)

    angles = np.array(angles)
    # vector mean of angles
    z = np.exp(1j * angles).mean()
    return float(np.angle(z))


def compute_geometric_phase_for_trajectory(
    traj,
    snaps,
    time_to_idx: Dict[float, int],
    ring_radius: float = 1.5,
    ring_samples: int = 16,
):
    """
    For a given trajectory, compute a simple "geometric phase" by:
      - sampling the average phase on a ring around the defect at each time
        (using the snapshot at that time)
      - unwrapping the phase along the trajectory

    traj: dict with keys "id", "charge", "points" [(t,x,y)...]
    snaps: list of {"psi": array, "time": float}
    time_to_idx: mapping from rounded time->snapshot index
    ring_radius: radius for ring sampling (in grid units)
    ring_samples: samples on ring

    Returns:
      total_phase (float), phases_over_time (list of (t, phase))
    """
    points = traj["points"]
    times = [p[0] for p in points]
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]

    phases = []

    for t, x, y in zip(times, xs, ys):
        key = round(t, 10)
        if key in time_to_idx:
            idx = time_to_idx[key]
        else:
            # fallback: nearest snapshot in time
            snapshot_times = np.array([s["time"] for s in snaps])
            idx = int(np.argmin(np.abs(snapshot_times - t)))

        psi = snaps[idx]["psi"]
        avg_phase = phase_at_ring(psi, x, y, r=ring_radius, n_samples=ring_samples)
        phases.append(avg_phase)

    phases = np.array(phases)
    unwrapped = np.unwrap(phases)
    total_phase = float(unwrapped[-1] - unwrapped[0])
    phases_over_time = list(zip(times, phases))
    return total_phase, phases_over_time


# ---------------------------------------------------------------------
# Histogram plotting
# ---------------------------------------------------------------------

def plot_histogram(values, out_png: str, title: str, xlabel: str):
    """
    Simple 1D histogram of values.
    """
    if len(values) == 0:
        print(f"[WARN] No values to plot for {out_png}")
        return

    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=40, edgecolor="black", alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[RESULT] Saved histogram -> {out_png}")


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scan geometric phases for defect trajectories."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing <prefix>_snap_*.npz files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Snapshot prefix (e.g. yee_coupled).",
    )
    parser.add_argument(
        "--traj_csv",
        type=str,
        required=True,
        help="Trajectory CSV produced by defect_braid_analysis_gpu.py.",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=80,
        help="Minimum number of points in a trajectory to include.",
    )
    parser.add_argument(
        "--allowed_charges",
        type=int,
        nargs="+",
        default=[1, -1],
        help="Allowed charges (winding numbers) to include, e.g. 1 -1.",
    )
    parser.add_argument(
        "--ring_radius",
        type=float,
        default=1.5,
        help="Ring radius (grid units) for phase sampling.",
    )
    parser.add_argument(
        "--ring_samples",
        type=int,
        default=16,
        help="Number of samples on ring for phase sampling.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="geom_phase",
        help="Prefix for output CSV and PNGs.",
    )
    args = parser.parse_args()

    print(f"[INFO] Loading snapshots from {args.input_dir} with prefix {args.prefix}...")
    snaps, snap_times, time_to_idx = load_snapshots(args.input_dir, args.prefix)
    print(f"[INFO] Loaded {len(snaps)} snapshots.")

    print(f"[INFO] Loading trajectories from {args.traj_csv}...")
    trajectories = load_trajectories_from_csv(args.traj_csv)
    print(f"[INFO] Loaded {len(trajectories)} trajectories from CSV.")

    allowed_charges = set(args.allowed_charges)

    rows = []
    phases_rad = []
    phases_cycles = []

    print("[INFO] Computing geometric phases for trajectories...")
    for tid, tr in trajectories.items():
        length = len(tr["points"])
        charge = int(tr["charge"])
        if length < args.min_len:
            continue
        if charge not in allowed_charges:
            continue

        total_phase, phases_over_time = compute_geometric_phase_for_trajectory(
            tr,
            snaps,
            time_to_idx,
            ring_radius=args.ring_radius,
            ring_samples=args.ring_samples,
        )
        cycles = total_phase / (2.0 * np.pi)

        rows.append((tid, charge, length, total_phase, cycles))
        phases_rad.append(total_phase)
        phases_cycles.append(cycles)

    rows = np.array(rows, dtype=float) if rows else np.zeros((0, 5), dtype=float)
    out_csv = f"{args.out_prefix}_phases.csv"
    header = "traj_id,charge,length,total_phase_rad,total_phase_cycles"
    np.savetxt(out_csv, rows, delimiter=",", header=header, comments="")
    print(f"[RESULT] Saved per-trajectory phases -> {out_csv}")
    print(f"[INFO] Included {rows.shape[0]} trajectories with length >= {args.min_len} and charge in {sorted(allowed_charges)}.")

    # Histograms
    out_hist_rad = f"{args.out_prefix}_phases_rad_hist.png"
    out_hist_cycles = f"{args.out_prefix}_phases_cycles_hist.png"
    plot_histogram(
        phases_rad,
        out_hist_rad,
        title="Total geometric phase (radians)",
        xlabel="Total phase [rad]",
    )
    plot_histogram(
        phases_cycles,
        out_hist_cycles,
        title="Total geometric phase (cycles = phase / 2π)",
        xlabel="Total phase [cycles]",
    )

    print("[DONE] Geometric phase scan complete.")


if __name__ == "__main__":
    main()

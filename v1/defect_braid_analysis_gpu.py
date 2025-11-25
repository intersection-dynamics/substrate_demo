#!/usr/bin/env python3
"""
defect_braid_analysis_gpu.py

Fast defect and braid analysis for substrate simulations, with GPU acceleration
via CuPy for vortex detection and efficient CPU-side tracking.

Changes vs previous version:
- Still uses CuPy (if available) to detect vortices via vectorized phase winding.
- Replaces Hungarian / full distance matrices with a fast grid-based nearest-neighbor
  tracker (O(N)ish per frame instead of O(N^3)).
- Limits exchange search to the top N longest trajectories (configurable) to avoid
  O(N^2) blow-up over ~20k trajectories.

Pipeline:
---------
1. Load all .npz snapshots from a given output directory.
2. For each snapshot:
   - Compute the phase of psi.
   - Detect vortices using phase winding on plaquettes (vectorized, GPU-backed).
   - Record defect positions and charges (w = ±1, ±2, ...).
3. Track defects over time using a grid-based nearest-neighbor match.
4. Identify candidate exchanges/braids only among the top K longest trajectories.
5. Optionally compute a simple geometric phase along a few long trajectories.

Outputs:
--------
- <out_prefix>_defects.csv
- <out_prefix>_trajectories.csv
- <out_prefix>_exchanges.csv
"""

import os
import glob
import argparse
from typing import List, Dict, Tuple

import numpy as np

# Try GPU (CuPy) for vortex detection
try:
    import cupy as cp
    xp = cp
    GPU_AVAILABLE = True
    print("✓ CuPy detected – using GPU for vortex detection.")
except ImportError:
    xp = np
    GPU_AVAILABLE = False
    print("CuPy not found – running purely on CPU (NumPy).")


# ---------------------------------------------------------------------
# Utility: load snapshots
# ---------------------------------------------------------------------

def load_snapshots(input_dir: str, prefix: str):
    """
    Load all snapshots of the form:
      input_dir / f"{prefix}_snap_*.npz"
    Returns:
      snaps: list of dicts with keys: "psi", "time"
      indices: list of integer snapshot indices
    """
    pattern = os.path.join(input_dir, f"{prefix}_snap_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No snapshot files matching {pattern}")

    snaps = []
    indices = []

    for fname in files:
        base = os.path.basename(fname)
        # expect: prefix_snap_XXXXXX.npz
        try:
            idx_str = base.split("_snap_")[1].split(".")[0]
            idx = int(idx_str)
        except Exception:
            idx = len(indices)

        data = np.load(fname)
        if "psi" not in data or "time" not in data:
            raise KeyError(f"{fname} missing 'psi' or 'time' arrays.")

        snaps.append({
            "psi": data["psi"],  # NumPy array from disk
            "time": float(data["time"]),
        })
        indices.append(idx)

    return snaps, np.array(indices)


# ---------------------------------------------------------------------
# Vortex / defect detection via phase winding (vectorized, GPU-capable)
# ---------------------------------------------------------------------

def detect_vortices(psi_np: np.ndarray, dx: float, winding_threshold: float = 0.5):
    """
    Detect vortices using phase winding around plaquettes.

    psi_np: complex field, shape (L, L) as NumPy array.
    dx    : lattice spacing (used only for physical scaling if needed).
    winding_threshold: minimum |w_int| to consider a plaquette a defect.

    This implementation is fully vectorized and uses CuPy if available.
    """
    # Move to GPU (or just use NumPy) as xp.array
    psi = xp.asarray(psi_np)
    Lx, Ly = psi.shape

    # Phase of psi
    theta = xp.angle(psi)

    # Helper: wrapped phase difference in (-π, π]
    def dtheta(a, b):
        d = b - a
        d = (d + xp.pi) % (2.0 * xp.pi) - xp.pi
        return d

    # Four corners of each plaquette:
    # t00: [0..L-2, 0..L-2]
    # t10: [1..L-1, 0..L-2]
    # t11: [1..L-1, 1..L-1]
    # t01: [0..L-2, 1..L-1]
    t00 = theta[:-1, :-1]
    t10 = theta[1:,  :-1]
    t11 = theta[1:,  1:]
    t01 = theta[:-1, 1:]

    d1 = dtheta(t00, t10)
    d2 = dtheta(t10, t11)
    d3 = dtheta(t11, t01)
    d4 = dtheta(t01, t00)

    total = d1 + d2 + d3 + d4
    w = total / (2.0 * xp.pi)
    w_int = xp.rint(w).astype(xp.int32)

    # Mask of plaquettes with nontrivial winding
    mask = xp.abs(w_int) >= winding_threshold

    # Get indices of defects
    idx_i, idx_j = xp.nonzero(mask)

    # Bring indices and winding back to CPU explicitly
    if GPU_AVAILABLE:
        idx_i = cp.asnumpy(idx_i)
        idx_j = cp.asnumpy(idx_j)
        w_int_cpu_full = cp.asnumpy(w_int)
    else:
        idx_i = np.asarray(idx_i)
        idx_j = np.asarray(idx_j)
        w_int_cpu_full = np.asarray(w_int)

    w_int_cpu = w_int_cpu_full[idx_i, idx_j]

    defects = []
    # Position at plaquette center (i+0.5, j+0.5)
    for i, j, win in zip(idx_i, idx_j, w_int_cpu):
        x = float(i) + 0.5
        y = float(j) + 0.5
        defects.append((x, y, int(win)))

    return defects


# ---------------------------------------------------------------------
# Tracking defects over time (grid-based nearest neighbor)
# ---------------------------------------------------------------------

def _build_spatial_grid(defects: List[Tuple[float, float, int]], cell_size: float):
    """
    Build a simple spatial hash grid for defects.

    defects: list of (x, y, w)
    cell_size: linear dimension of each grid cell.

    Returns:
      grid: dict[(cx, cy)] -> list of indices into defects
    """
    grid = {}
    for idx, (x, y, w) in enumerate(defects):
        cx = int(x // cell_size)
        cy = int(y // cell_size)
        key = (cx, cy)
        if key not in grid:
            grid[key] = []
        grid[key].append(idx)
    return grid


def match_defects_grid(
    prev_defects: List[Tuple[float, float, int]],
    curr_defects: List[Tuple[float, float, int]],
    max_dist: float = 2.5,
):
    """
    Match defects between two frames using a grid-based nearest-neighbor heuristic.

    prev_defects, curr_defects: lists of (x, y, w)
    max_dist: maximum allowed distance for a match. Beyond this, treat
              as new birth / death.

    Returns:
      matches: list of (prev_idx, curr_idx)
      births: list of curr_idx that are unmatched (new defects)
      deaths: list of prev_idx that are unmatched (disappeared)

    This avoids computing full O(N^2) distance matrices and is much faster
    for ~1000+ defects per frame.
    """
    if not prev_defects or not curr_defects:
        return [], list(range(len(curr_defects))), list(range(len(prev_defects)))

    cell_size = max_dist
    grid = _build_spatial_grid(curr_defects, cell_size=cell_size)

    used_curr = set()
    matches = []

    max_dist_sq = max_dist * max_dist

    for pi, (x0, y0, w0) in enumerate(prev_defects):
        cx = int(x0 // cell_size)
        cy = int(y0 // cell_size)

        best_j = None
        best_d2 = None

        # Search in this cell and neighbors (3x3 block)
        for dx_cell in (-1, 0, 1):
            for dy_cell in (-1, 0, 1):
                key = (cx + dx_cell, cy + dy_cell)
                if key not in grid:
                    continue
                for j in grid[key]:
                    if j in used_curr:
                        continue
                    x1, y1, w1 = curr_defects[j]
                    dx = x1 - x0
                    dy = y1 - y0
                    d2 = dx * dx + dy * dy
                    if d2 <= max_dist_sq:
                        if best_j is None or d2 < best_d2:
                            best_j = j
                            best_d2 = d2

        if best_j is not None:
            used_curr.add(best_j)
            matches.append((pi, best_j))

    all_curr = set(range(len(curr_defects)))
    births = [j for j in all_curr if j not in used_curr]
    matched_prev = {pi for (pi, _) in matches}
    all_prev = set(range(len(prev_defects)))
    deaths = [i for i in all_prev if i not in matched_prev]

    return matches, births, deaths


def build_trajectories(
    all_defects: List[List[Tuple[float, float, int]]],
    times: np.ndarray,
    max_dist: float = 2.5,
):
    """
    Build trajectories by matching defects across frames.

    all_defects[k] is list of defects at frame k: (x,y,w)
    times[k] is the time for that frame.

    Returns:
      trajectories: dict[id] = {
          "id": int,
          "charge": int (w),
          "points": list of (t, x, y)
      }
    """
    trajectories: Dict[int, Dict] = {}
    next_id = 0

    if not all_defects:
        return trajectories

    prev_defects = all_defects[0]
    prev_ids = []
    for d in prev_defects:
        tid = next_id
        next_id += 1
        trajectories[tid] = {
            "id": tid,
            "charge": d[2],
            "points": [(float(times[0]), float(d[0]), float(d[1]))],
        }
        prev_ids.append(tid)

    for k in range(1, len(all_defects)):
        curr_defects = all_defects[k]
        matches, births, deaths = match_defects_grid(prev_defects, curr_defects, max_dist=max_dist)
        curr_ids = [None] * len(curr_defects)

        # Matches
        for pi, ci in matches:
            tid = prev_ids[pi]
            curr_ids[ci] = tid
            t = float(times[k])
            x, y, w = curr_defects[ci]
            trajectories[tid]["points"].append((t, float(x), float(y)))

        # Births
        for ci in births:
            x, y, w = curr_defects[ci]
            tid = next_id
            next_id += 1
            trajectories[tid] = {
                "id": tid,
                "charge": int(w),
                "points": [(float(times[k]), float(x), float(y))],
            }
            curr_ids[ci] = tid

        # Deaths: trajectories just stop growing

        prev_defects = curr_defects
        prev_ids = curr_ids

    return trajectories


# ---------------------------------------------------------------------
# Geometric phase along a defect trajectory
# ---------------------------------------------------------------------

def phase_at_ring(psi: np.ndarray, x: float, y: float, r: float = 1.5, n_samples: int = 16):
    """
    Sample phase of psi on a small ring around (x,y) with radius r.
    Returns average phase using a proper angle-mean (vector average).

    This uses NumPy only; cost is negligible compared to vortex detection.
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
    z = np.exp(1j * angles).mean()
    return float(np.angle(z))


def compute_geometric_phase_for_trajectory(traj, snaps, r: float = 1.5, n_samples: int = 16):
    """
    For a given trajectory, compute a simple "geometric phase" by:
      - sampling the average phase on a ring around the defect at each time
      - integrating the incremental change in that averaged phase over time,
        with angle unwrapping.

    Returns:
      total_phase (float), phases_over_time (list of (t, phase))
    """
    times = [p[0] for p in traj["points"]]
    xs = [p[1] for p in traj["points"]]
    ys = [p[2] for p in traj["points"]]

    snap_times = np.array([s["time"] for s in snaps])
    phases = []

    for t, x, y in zip(times, xs, ys):
        idx = int(np.argmin(np.abs(snap_times - t)))
        psi = snaps[idx]["psi"]
        avg_phase = phase_at_ring(psi, x, y, r=r, n_samples=n_samples)
        phases.append(avg_phase)

    phases = np.array(phases)
    unwrapped = np.unwrap(phases)
    total_phase = float(unwrapped[-1] - unwrapped[0])
    phases_over_time = list(zip(times, phases))
    return total_phase, phases_over_time


# ---------------------------------------------------------------------
# Simple exchange / braid finder (on top K longest trajectories)
# ---------------------------------------------------------------------

def find_candidate_exchanges(
    trajectories,
    min_overlap_time: float = 0.0,
    max_sep: float = 5.0,
    max_trajs_for_exchange: int = 200,
):
    """
    Heuristic to detect candidate "exchanges" between pairs of trajectories:
      - Restrict to the top `max_trajs_for_exchange` longest trajectories.
      - For each pair (i,j), find overlap in time.
      - Sample separation over that overlap.
      - If they approach within max_sep, mark as a candidate exchange.

    Returns:
      exchanges: list of dict with keys:
        "id1", "id2", "t_start", "t_end", "min_sep"
    """
    # Sort trajectories by length (longest first)
    traj_items = sorted(
        trajectories.items(),
        key=lambda kv: len(kv[1]["points"]),
        reverse=True,
    )
    if max_trajs_for_exchange is not None and len(traj_items) > max_trajs_for_exchange:
        traj_items = traj_items[:max_trajs_for_exchange]

    traj_list = [tr for (tid, tr) in traj_items]
    exchanges = []

    for a in range(len(traj_list)):
        for b in range(a + 1, len(traj_list)):
            tr1 = traj_list[a]
            tr2 = traj_list[b]

            t1 = np.array([p[0] for p in tr1["points"]])
            x1 = np.array([p[1] for p in tr1["points"]])
            y1 = np.array([p[2] for p in tr1["points"]])

            t2 = np.array([p[0] for p in tr2["points"]])
            x2 = np.array([p[1] for p in tr2["points"]])
            y2 = np.array([p[2] for p in tr2["points"]])

            t_min = max(t1[0], t2[0])
            t_max = min(t1[-1], t2[-1])
            if t_max <= t_min or (t_max - t_min) < min_overlap_time:
                continue

            ts = np.linspace(t_min, t_max, num=64)
            x1_s = np.interp(ts, t1, x1)
            y1_s = np.interp(ts, t1, y1)
            x2_s = np.interp(ts, t2, x2)
            y2_s = np.interp(ts, t2, y2)

            sep = np.sqrt((x1_s - x2_s) ** 2 + (y1_s - y2_s) ** 2)
            min_sep = float(sep.min())
            if min_sep <= max_sep:
                exchanges.append({
                    "id1": tr1["id"],
                    "id2": tr2["id"],
                    "t_start": float(t_min),
                    "t_end": float(t_max),
                    "min_sep": min_sep,
                })

    return exchanges


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------

def save_defects_csv(all_defects, times, out_path: str):
    """
    Save per-frame defects as:
      frame_index, time, x, y, w
    """
    rows = []
    for k, defects in enumerate(all_defects):
        t = float(times[k])
        for (x, y, w) in defects:
            rows.append((k, t, x, y, w))
    if not rows:
        print("[WARN] No defects found – defects CSV will be empty.")
        rows_arr = np.zeros((0, 5), dtype=float)
    else:
        rows_arr = np.array(rows, dtype=float)
    header = "frame_idx,time,x,y,w"
    np.savetxt(out_path, rows_arr, delimiter=",", header=header, comments="")
    print(f"[RESULT] Saved defects -> {out_path}")


def save_trajectories_csv(trajectories, out_path: str):
    """
    Save trajectories as:
      traj_id, charge, t, x, y
    """
    rows = []
    for tid, tr in trajectories.items():
        charge = tr["charge"]
        for (t, x, y) in tr["points"]:
            rows.append((tid, charge, t, x, y))
    if not rows:
        print("[WARN] No trajectories found – trajectories CSV will be empty.")
        rows_arr = np.zeros((0, 5), dtype=float)
    else:
        rows_arr = np.array(rows, dtype=float)
    header = "traj_id,charge,time,x,y"
    np.savetxt(out_path, rows_arr, delimiter=",", header=header, comments="")
    print(f"[RESULT] Saved trajectories -> {out_path}")


def save_exchanges_csv(exchanges, out_path: str):
    """
    Save candidate exchanges as:
      id1,id2,t_start,t_end,min_sep
    """
    rows = []
    for ex in exchanges:
        rows.append((
            ex["id1"],
            ex["id2"],
            ex["t_start"],
            ex["t_end"],
            ex["min_sep"],
        ))
    if not rows:
        print("[WARN] No exchanges found – exchanges CSV will be empty.")
        rows_arr = np.zeros((0, 5), dtype=float)
    else:
        rows_arr = np.array(rows, dtype=float)
    header = "id1,id2,t_start,t_end,min_sep"
    np.savetxt(out_path, rows_arr, delimiter=",", header=header, comments="")
    print(f"[RESULT] Saved exchanges -> {out_path}")


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated defect and braid analysis for substrate simulations."
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
        "--dx",
        type=float,
        default=1.0,
        help="Lattice spacing (for physical units; not critical here).",
    )
    parser.add_argument(
        "--max_track_dist",
        type=float,
        default=2.5,
        help="Maximum distance to match defects between frames.",
    )
    parser.add_argument(
        "--max_exchange_sep",
        type=float,
        default=5.0,
        help="Max separation for candidate exchange events.",
    )
    parser.add_argument(
        "--max_exchange_trajs",
        type=int,
        default=200,
        help="Max number of longest trajectories to consider in exchange search.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="defect_analysis",
        help="Prefix for output CSV files.",
    )
    args = parser.parse_args()

    snaps, indices = load_snapshots(args.input_dir, args.prefix)
    times = np.array([s["time"] for s in snaps])
    print(f"[INFO] Loaded {len(snaps)} snapshots from {args.input_dir}")

    # 1) Detect defects per frame
    all_defects = []
    for k, snap in enumerate(snaps):
        psi_np = snap["psi"]
        defects = detect_vortices(psi_np, dx=args.dx, winding_threshold=0.5)
        all_defects.append(defects)
        print(f"[DEFECT] frame={k:4d}, t={snap['time']:8.3f}, count={len(defects)}")

    # 2) Build trajectories (grid-based tracking)
    trajectories = build_trajectories(
        all_defects,
        times,
        max_dist=args.max_track_dist,
    )
    print(f"[INFO] Built {len(trajectories)} trajectories.")

    # 3) Find candidate exchanges (only among top K longest trajectories)
    exchanges = find_candidate_exchanges(
        trajectories,
        min_overlap_time=0.0,
        max_sep=args.max_exchange_sep,
        max_trajs_for_exchange=args.max_exchange_trajs,
    )
    print(f"[INFO] Found {len(exchanges)} candidate exchanges (among top {args.max_exchange_trajs} trajectories).")

    # 4) Save results
    out_defects = f"{args.out_prefix}_defects.csv"
    out_trajs = f"{args.out_prefix}_trajectories.csv"
    out_exch = f"{args.out_prefix}_exchanges.csv"

    save_defects_csv(all_defects, times, out_defects)
    save_trajectories_csv(trajectories, out_trajs)
    save_exchanges_csv(exchanges, out_exch)

    # 5) Example: geometric phase for a few longest trajectories
    traj_lengths = [
        (tid, len(tr["points"])) for tid, tr in trajectories.items()
    ]
    traj_lengths.sort(key=lambda x: x[1], reverse=True)
    top = traj_lengths[:5]

    print("[INFO] Geometric phase for top 5 longest trajectories:")
    for tid, length in top:
        tr = trajectories[tid]
        total_phase, phases_over_time = compute_geometric_phase_for_trajectory(
            tr, snaps, r=1.5, n_samples=16
        )
        print(
            f"  traj_id={tid}, length={length}, charge={tr['charge']}, "
            f"total_geom_phase={total_phase:.3f} rad"
        )


if __name__ == "__main__":
    main()

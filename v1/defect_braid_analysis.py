#!/usr/bin/env python3
"""
defect_braid_analysis.py

Analyze phase singularities ("defects" / vortices) in substrate simulations.

Pipeline:
---------
1. Load all .npz snapshots from a given output directory.
2. For each snapshot:
   - Compute the phase of psi.
   - Detect vortices using winding-number on plaquettes.
   - Record defect positions and charges (w = ±1, ±2, ...).
3. Track defects over time using a simple assignment (Hungarian algorithm)
   with distance threshold, building worldlines / trajectories.
4. Identify candidate "exchange" events (braid-like interactions) between
   pairs of defects whose trajectories approach and then separate.
5. For each defect trajectory, compute a simple geometric phase:
   integrate the average phase on a small circle around the defect over time.

Outputs:
--------
- <out_prefix>_defects.csv
- <out_prefix>_trajectories.csv
- <out_prefix>_exchanges.csv

Each is a CSV so you can poke at them in Python, Excel, or your plotting stack.
"""

import os
import glob
import argparse
from typing import List, Dict, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARN] SciPy not found. Tracking will fall back to greedy matching.")


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
            "psi": data["psi"],
            "time": float(data["time"]),
        })
        indices.append(idx)

    return snaps, np.array(indices)


# ---------------------------------------------------------------------
# Vortex / defect detection via phase winding
# ---------------------------------------------------------------------

def detect_vortices(psi: np.ndarray, dx: float, winding_threshold: float = 0.5):
    """
    Detect vortices using phase winding around plaquettes.

    psi: complex field, shape (L, L)
    dx : lattice spacing (currently used only if you want to convert to
         physical coordinates; we just return indices for now).

    Algorithm:
      - Get phase theta = arg(psi).
      - For each plaquette (i,j) (lower-left corner), compute
          Δθ around the four edges with branch cuts fixed to (-π, π].
      - Total winding w = round(Δθ / (2π)).
      - If |w| >= winding_threshold, record a defect at plaquette center.

    Returns:
      defects: list of (x, y, w) in lattice coordinates (float positions).
    """
    Lx, Ly = psi.shape
    theta = np.angle(psi)

    # Helper to compute wrapped phase differences in (-π, π]
    def dtheta(a, b):
        d = b - a
        d = (d + np.pi) % (2.0 * np.pi) - np.pi
        return d

    # Compute differences along edges of each plaquette
    # We'll loop; it's only O(L^2) and Python can handle 256^2 fine.
    defects = []
    for i in range(Lx - 1):
        for j in range(Ly - 1):
            t00 = theta[i, j]
            t10 = theta[i + 1, j]
            t11 = theta[i + 1, j + 1]
            t01 = theta[i, j + 1]

            d1 = dtheta(t00, t10)
            d2 = dtheta(t10, t11)
            d3 = dtheta(t11, t01)
            d4 = dtheta(t01, t00)

            total = d1 + d2 + d3 + d4
            w = total / (2.0 * np.pi)
            w_int = int(np.round(w))

            if abs(w_int) >= winding_threshold:
                # Position at plaquette center (i+0.5, j+0.5)
                x = i + 0.5
                y = j + 0.5
                defects.append((x, y, w_int))

    return defects


# ---------------------------------------------------------------------
# Tracking defects over time
# ---------------------------------------------------------------------

def match_defects(
    prev_defects: List[Tuple[float, float, int]],
    curr_defects: List[Tuple[float, float, int]],
    max_dist: float = 2.5,
):
    """
    Match defects between two frames.

    prev_defects, curr_defects: lists of (x, y, w)
    max_dist: maximum allowed distance for a match. Beyond this, treat
              as new birth / death.

    Returns:
      matches: list of (prev_idx, curr_idx)
      births: list of curr_idx that are unmatched (new defects)
      deaths: list of prev_idx that are unmatched (disappeared)
    """
    if not prev_defects or not curr_defects:
        return [], list(range(len(curr_defects))), list(range(len(prev_defects)))

    prev_xy = np.array([[d[0], d[1]] for d in prev_defects])
    curr_xy = np.array([[d[0], d[1]] for d in curr_defects])

    # Compute distance matrix
    dists = np.linalg.norm(prev_xy[:, None, :] - curr_xy[None, :, :], axis=-1)

    if SCIPY_AVAILABLE:
        row_ind, col_ind = linear_sum_assignment(dists)
        matches = []
        used_prev = set()
        used_curr = set()
        for r, c in zip(row_ind, col_ind):
            if dists[r, c] <= max_dist:
                matches.append((r, c))
                used_prev.add(r)
                used_curr.add(c)
        births = [j for j in range(len(curr_defects)) if j not in used_curr]
        deaths = [i for i in range(len(prev_defects)) if i not in used_prev]
    else:
        # Greedy fallback
        matches = []
        used_prev = set()
        used_curr = set()
        flat_indices = np.argsort(dists.ravel())
        for idx in flat_indices:
            i = idx // dists.shape[1]
            j = idx % dists.shape[1]
            if i in used_prev or j in used_curr:
                continue
            if dists[i, j] <= max_dist:
                matches.append((i, j))
                used_prev.add(i)
                used_curr.add(j)
        births = [j for j in range(len(curr_defects)) if j not in used_curr]
        deaths = [i for i in range(len(prev_defects)) if i not in used_prev]

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

    # At k=0, every defect starts a new trajectory
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

    # Now propagate
    for k in range(1, len(all_defects)):
        curr_defects = all_defects[k]
        matches, births, deaths = match_defects(prev_defects, curr_defects, max_dist=max_dist)

        curr_ids = [None] * len(curr_defects)

        # Handle matches
        for pi, ci in matches:
            tid = prev_ids[pi]
            curr_ids[ci] = tid
            t = float(times[k])
            x, y, w = curr_defects[ci]
            trajectories[tid]["points"].append((t, float(x), float(y)))

        # Handle births
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

        # Deaths: we just stop extending those trajectories
        # (they already have all their points)

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
    """
    Lx, Ly = psi.shape
    angles = []
    for k in range(n_samples):
        theta = 2.0 * np.pi * k / n_samples
        xx = x + r * np.cos(theta)
        yy = y + r * np.sin(theta)
        # Periodic wrap
        ix = int(np.floor(xx)) % Lx
        iy = int(np.floor(yy)) % Ly
        val = psi[ix, iy]
        ang = np.angle(val)
        angles.append(ang)

    angles = np.array(angles)
    # Vector-mean of angles
    z = np.exp(1j * angles).mean()
    return np.angle(z)


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

    # Map time to nearest snapshot index
    all_times = np.array([s["time"] for s in snaps])
    phases = []

    for t, x, y in zip(times, xs, ys):
        idx = int(np.argmin(np.abs(all_times - t)))
        psi = snaps[idx]["psi"]
        avg_phase = phase_at_ring(psi, x, y, r=r, n_samples=n_samples)
        phases.append(avg_phase)

    phases = np.array(phases)
    # unwrap and integrate
    unwrapped = np.unwrap(phases)
    total_phase = float(unwrapped[-1] - unwrapped[0])

    phases_over_time = list(zip(times, phases))
    return total_phase, phases_over_time


# ---------------------------------------------------------------------
# Simple exchange / braid finder
# ---------------------------------------------------------------------

def find_candidate_exchanges(trajectories, min_overlap_time: float = 0.0, max_sep: float = 5.0):
    """
    Very simple heuristic to detect candidate "exchanges" between pairs
    of trajectories:
      - For each pair of trajectories (i,j), find the time window where
        both exist.
      - Sample their separation over that overlap window.
      - If they approach within max_sep and then separate again, count
        this as a candidate exchange event.

    Returns:
      exchanges: list of dict with keys:
        "id1", "id2", "t_start", "t_end", "min_sep"
    """
    traj_list = list(trajectories.values())
    exchanges = []

    for a in range(len(traj_list)):
        for b in range(a + 1, len(traj_list)):
            tr1 = traj_list[a]
            tr2 = traj_list[b]

            # Extract time series
            t1 = np.array([p[0] for p in tr1["points"]])
            x1 = np.array([p[1] for p in tr1["points"]])
            y1 = np.array([p[2] for p in tr1["points"]])

            t2 = np.array([p[0] for p in tr2["points"]])
            x2 = np.array([p[1] for p in tr2["points"]])
            y2 = np.array([p[2] for p in tr2["points"]])

            # Overlap in time
            t_min = max(t1[0], t2[0])
            t_max = min(t1[-1], t2[-1])
            if t_max <= t_min or (t_max - t_min) < min_overlap_time:
                continue

            # Sample some times in overlap
            ts = np.linspace(t_min, t_max, num=64)
            # Interpolate positions
            x1_s = np.interp(ts, t1, x1)
            y1_s = np.interp(ts, t1, y1)
            x2_s = np.interp(ts, t2, x2)
            y2_s = np.interp(ts, t2, y2)

            sep = np.sqrt((x1_s - x2_s)**2 + (y1_s - y2_s)**2)
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
    arr = np.array(rows, dtype=float)
    header = "frame_idx,time,x,y,w"
    np.savetxt(out_path, arr, delimiter=",", header=header, comments="")
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
    arr = np.array(rows, dtype=float)
    header = "traj_id,charge,time,x,y"
    np.savetxt(out_path, arr, delimiter=",", header=header, comments="")
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
    arr = np.array(rows, dtype=float)
    header = "id1,id2,t_start,t_end,min_sep"
    np.savetxt(out_path, arr, delimiter=",", header=header, comments="")
    print(f"[RESULT] Saved exchanges -> {out_path}")


# ---------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Defect and braid analysis for substrate simulations."
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
        psi = snap["psi"]
        defects = detect_vortices(psi, dx=args.dx, winding_threshold=0.5)
        all_defects.append(defects)
        print(f"[DEFECT] frame={k:4d}, t={snap['time']:8.3f}, count={len(defects)}")

    # 2) Build trajectories
    trajectories = build_trajectories(
        all_defects,
        times,
        max_dist=args.max_track_dist,
    )
    print(f"[INFO] Built {len(trajectories)} trajectories.")

    # 3) Find candidate exchanges
    exchanges = find_candidate_exchanges(
        trajectories,
        min_overlap_time=0.0,
        max_sep=args.max_exchange_sep,
    )
    print(f"[INFO] Found {len(exchanges)} candidate exchanges.")

    # 4) Save results
    out_defects = f"{args.out_prefix}_defects.csv"
    out_trajs = f"{args.out_prefix}_trajectories.csv"
    out_exch = f"{args.out_prefix}_exchanges.csv"

    save_defects_csv(all_defects, times, out_defects)
    save_trajectories_csv(trajectories, out_trajs)
    save_exchanges_csv(exchanges, out_exch)

    # (Optional) Example: compute geometric phase for a few longest trajectories
    # so you can inspect them manually.
    traj_lengths = [
        (tid, len(tr["points"])) for tid, tr in trajectories.items()
    ]
    traj_lengths.sort(key=lambda x: x[1], reverse=True)
    top = traj_lengths[:5]

    print("[INFO] Computing geometric phase for top 5 longest trajectories:")
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

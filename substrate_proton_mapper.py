#!/usr/bin/env python3
"""
substrate_proton_mapper.py

Tools for detecting and "graphing" proton-like lumps in a 3D substrate simulation.

Assumptions:
- field is stored as a numpy array, shape (Nx, Ny, Nz, C) or (Nx, Ny, Nz).
- You have a notion of "vacuum" value; default is 0.0 for all components.
- Protons are localized blobs of high |field - vacuum| and/or high energy density.

Features:
- Load .npz snapshots (single file, or numbered sequence).
- Compute a crude energy density and "lumpiness" scalar.
- Detect connected high-lump regions -> proton candidates.
- Track candidates over time with nearest-neighbor matching.
- Save CSV summary and basic matplotlib plots.

You can:
- Run as a CLI tool on saved snapshots.
- Import and call `find_protons_in_field` from your solver.
"""

import argparse
import glob
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProtonCandidate:
    """A single proton-like lump in one snapshot."""
    frame_index: int            # integer index in time sequence
    time: float                 # physical time if known, else frame_index
    id_local: int               # id within this frame
    x: float                    # centroid x (physical units)
    y: float                    # centroid y
    z: float                    # centroid z
    radius: float               # approx radius (rms in physical units)
    energy: float               # integrated energy in lump
    q_top: float                # integrated topological density in lump (if available; else 0)
    volume: float               # volume of lump in physical units
    label: int                  # integer label from connected components


@dataclass
class TrackedProton:
    """A proton tracked across multiple frames."""
    track_id: int
    frames: List[int]
    times: List[float]
    xs: List[float]
    ys: List[float]
    zs: List[float]
    energies: List[float]
    q_tops: List[float]
    radii: List[float]


# ---------------------------------------------------------------------------
# Diagnostics: energy & topological density
# ---------------------------------------------------------------------------

def compute_lumpiness(field: np.ndarray, vacuum: float = 0.0) -> np.ndarray:
    """
    Very crude 'lumpiness' measure: |field - vacuum|^2 summed over components.
    """
    f = field
    if f.ndim == 3:
        diff = f - vacuum
        lump = diff * diff
    elif f.ndim == 4:
        # last axis is components
        diff = f - vacuum
        lump = np.sum(diff * diff, axis=-1)
    else:
        raise ValueError(f"Unsupported field shape {f.shape}; expected 3D or 4D array.")
    return lump


def compute_energy_density(field: np.ndarray, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> np.ndarray:
    """
    Crude energy density ~ (grad field)^2 + mass_term*field^2.
    You can replace this with your actual Hamiltonian density.

    For now:
        ε = 0.5 * Σ_i |∂_i field|^2 + 0.5 * |field|^2
    """
    if field.ndim == 3:
        # add fake "component" axis for unified handling
        f = field[..., None]
    elif field.ndim == 4:
        f = field
    else:
        raise ValueError(f"Unsupported field shape {field.shape}; expected 3D or 4D array.")

    # spatial derivatives via central differences
    # axis 0 -> x, 1 -> y, 2 -> z
    dfdx = np.gradient(f, dx, axis=0)
    dfdy = np.gradient(f, dy, axis=1)
    dfdz = np.gradient(f, dz, axis=2)

    grad_sq = dfdx * dfdx + dfdy * dfdy + dfdz * dfdz  # shape (..., C)
    grad_sq = np.sum(grad_sq, axis=-1)  # sum over components

    field_sq = np.sum(f * f, axis=-1)   # ||field||^2

    energy = 0.5 * grad_sq + 0.5 * field_sq
    return energy


def compute_topological_density_placeholder(field: np.ndarray) -> np.ndarray:
    """
    Placeholder for topological density (e.g. baryon number density / Hopf charge density).

    For now, returns zeros with same spatial shape as field[...].
    Replace this with your own Q-density routine and plug it in.
    """
    if field.ndim == 3:
        shape = field.shape
    elif field.ndim == 4:
        shape = field.shape[:3]
    else:
        raise ValueError(f"Unsupported field shape {field.shape}; expected 3D or 4D array.")
    return np.zeros(shape, dtype=np.float32)


# ---------------------------------------------------------------------------
# Proton detection in a single snapshot
# ---------------------------------------------------------------------------

def find_protons_in_field(
    field: np.ndarray,
    frame_index: int,
    time: float,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    lumpiness_threshold: float = None,
    min_voxels: int = 10,
    vacuum: float = 0.0,
    use_energy: bool = True,
    topological_density: Optional[np.ndarray] = None,
) -> List[ProtonCandidate]:
    """
    Detect proton-like blobs in a single 3D field snapshot.

    Steps:
    1. Compute lumpiness = |field - vacuum|^2.
    2. Optionally weight by energy density.
    3. Threshold, connected-components.
    4. Measure each component.

    Returns list of ProtonCandidate.
    """
    lump = compute_lumpiness(field, vacuum=vacuum)
    energy = compute_energy_density(field, dx=dx, dy=dy, dz=dz) if use_energy else lump

    if topological_density is None:
        q_density = compute_topological_density_placeholder(field)
    else:
        q_density = topological_density

    # composite scalar to threshold on
    # here: lump_weight = lump + energy
    lump_weight = lump + energy

    if lumpiness_threshold is None:
        # auto-threshold: mean + N * std
        mu = np.mean(lump_weight)
        sigma = np.std(lump_weight)
        lumpiness_threshold = mu + 2.0 * sigma  # you can tune this

    mask = lump_weight > lumpiness_threshold

    # connected components (6-connectivity in 3D by default)
    labeled, num_labels = ndimage.label(mask)
    candidates: List[ProtonCandidate] = []

    # voxel volume in physical units
    voxel_volume = dx * dy * dz

    for label in range(1, num_labels + 1):
        region_mask = (labeled == label)
        voxel_indices = np.argwhere(region_mask)
        if voxel_indices.shape[0] < min_voxels:
            continue

        # positions in grid indices
        xs_idx = voxel_indices[:, 0]
        ys_idx = voxel_indices[:, 1]
        zs_idx = voxel_indices[:, 2]

        # centroid in index space
        x_idx_mean = xs_idx.mean()
        y_idx_mean = ys_idx.mean()
        z_idx_mean = zs_idx.mean()

        # map to physical coordinates (you can shift origin if desired)
        x_phys = x_idx_mean * dx
        y_phys = y_idx_mean * dy
        z_phys = z_idx_mean * dz

        # rms radius
        dxs = (xs_idx - x_idx_mean) * dx
        dys = (ys_idx - y_idx_mean) * dy
        dzs = (zs_idx - z_idx_mean) * dz
        r2 = dxs * dxs + dys * dys + dzs * dzs
        r_rms = float(np.sqrt(r2.mean()))

        # integrated energy and topological charge
        energy_lump = float(energy[region_mask].sum() * voxel_volume)
        q_lump = float(q_density[region_mask].sum() * voxel_volume)

        volume_lump = voxel_indices.shape[0] * voxel_volume

        candidate = ProtonCandidate(
            frame_index=frame_index,
            time=time,
            id_local=label,
            x=float(x_phys),
            y=float(y_phys),
            z=float(z_phys),
            radius=r_rms,
            energy=energy_lump,
            q_top=q_lump,
            volume=volume_lump,
            label=label,
        )
        candidates.append(candidate)

    return candidates


# ---------------------------------------------------------------------------
# Tracking protons across time
# ---------------------------------------------------------------------------

def track_protons_over_time(
    all_candidates: List[List[ProtonCandidate]],
    max_link_distance: float = 3.0,
) -> List[TrackedProton]:
    """
    Given a list of candidate lists (one per frame), build tracks by nearest-neighbor matching.

    Simple greedy algorithm:
    - Start tracks from first frame.
    - For each next frame, assign candidates to existing tracks if within max_link_distance
      of the last known position; otherwise start new tracks.
    """
    tracks: List[TrackedProton] = []
    next_track_id = 0

    # initialize with first frame
    if not all_candidates:
        return tracks

    for cand in all_candidates[0]:
        tracks.append(TrackedProton(
            track_id=next_track_id,
            frames=[cand.frame_index],
            times=[cand.time],
            xs=[cand.x],
            ys=[cand.y],
            zs=[cand.z],
            energies=[cand.energy],
            q_tops=[cand.q_top],
            radii=[cand.radius],
        ))
        next_track_id += 1

    # process subsequent frames
    for frame_idx in range(1, len(all_candidates)):
        frame_cands = all_candidates[frame_idx]
        # track assignment bookkeeping
        used = set()  # used candidate indices
        # For convenience, precompute candidate positions
        cand_positions = np.array([[c.x, c.y, c.z] for c in frame_cands])

        # Try to extend existing tracks
        for track in tracks:
            # last known position
            last_pos = np.array([track.xs[-1], track.ys[-1], track.zs[-1]])

            if cand_positions.shape[0] == 0:
                continue

            # distances to all candidates
            d2 = np.sum((cand_positions - last_pos) ** 2, axis=1)
            idx_sorted = np.argsort(d2)
            assigned = False
            for idx in idx_sorted:
                if idx in used:
                    continue
                dist = np.sqrt(d2[idx])
                if dist <= max_link_distance:
                    # assign this candidate
                    c = frame_cands[idx]
                    track.frames.append(c.frame_index)
                    track.times.append(c.time)
                    track.xs.append(c.x)
                    track.ys.append(c.y)
                    track.zs.append(c.z)
                    track.energies.append(c.energy)
                    track.q_tops.append(c.q_top)
                    track.radii.append(c.radius)
                    used.add(idx)
                    assigned = True
                    break
            # if none within max_link_distance, we just let the track end here

        # Start new tracks from unassigned candidates
        for idx, c in enumerate(frame_cands):
            if idx in used:
                continue
            tracks.append(TrackedProton(
                track_id=next_track_id,
                frames=[c.frame_index],
                times=[c.time],
                xs=[c.x],
                ys=[c.y],
                zs=[c.z],
                energies=[c.energy],
                q_tops=[c.q_top],
                radii=[c.radius],
            ))
            next_track_id += 1

    return tracks


# ---------------------------------------------------------------------------
# Plotting / exporting
# ---------------------------------------------------------------------------

def save_candidates_csv(all_candidates: List[List[ProtonCandidate]], out_path: str) -> None:
    """
    Flatten all candidates into one CSV table.
    """
    import csv
    rows = []
    for frame_cands in all_candidates:
        for c in frame_cands:
            row = asdict(c)
            rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else [
        "frame_index", "time", "id_local", "x", "y", "z", "radius", "energy", "q_top", "volume", "label"
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Wrote {len(rows)} candidates to {out_path}")


def save_tracks_csv(tracks: List[TrackedProton], out_path: str) -> None:
    """
    Save tracks in a long table: one row per (track, frame).
    """
    import csv
    rows = []
    for tr in tracks:
        for i in range(len(tr.frames)):
            rows.append({
                "track_id": tr.track_id,
                "frame_index": tr.frames[i],
                "time": tr.times[i],
                "x": tr.xs[i],
                "y": tr.ys[i],
                "z": tr.zs[i],
                "energy": tr.energies[i],
                "q_top": tr.q_tops[i],
                "radius": tr.radii[i],
            })

    fieldnames = ["track_id", "frame_index", "time", "x", "y", "z", "energy", "q_top", "radius"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Wrote {len(rows)} track points to {out_path}")


def plot_tracks_3d(tracks: List[TrackedProton], out_path: str) -> None:
    """
    Simple 3D plot of all proton tracks.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for tr in tracks:
        ax.plot(tr.xs, tr.ys, tr.zs, marker="o", linestyle="-", alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Proton Tracks")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved 3D tracks plot to {out_path}")


def plot_energy_vs_time(tracks: List[TrackedProton], out_path: str, min_length: int = 2) -> None:
    """
    Plot per-track energy vs time for tracks longer than min_length.
    """
    fig, ax = plt.subplots()
    for tr in tracks:
        if len(tr.times) < min_length:
            continue
        ax.plot(tr.times, tr.energies, alpha=0.7)
    ax.set_xlabel("time")
    ax.set_ylabel("energy")
    ax.set_title("Proton Energy vs Time")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved energy-vs-time plot to {out_path}")


# ---------------------------------------------------------------------------
# I/O helpers for snapshots
# ---------------------------------------------------------------------------

def load_snapshot_npz(path: str) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Load a single snapshot NPZ file.

    Expected keys:
        field: 3D or 4D array
        t (optional): time (float)
        dx, dy, dz (optional): lattice spacings

    Returns: field, t, dx, dy, dz
    """
    data = np.load(path)
    field = data["field"]
    t = float(data["t"]) if "t" in data else 0.0
    dx = float(data["dx"]) if "dx" in data else 1.0
    dy = float(data["dy"]) if "dy" in data else 1.0
    dz = float(data["dz"]) if "dz" in data else 1.0
    return field, t, dx, dy, dz


def discover_snapshot_paths(pattern: str) -> List[str]:
    """
    Glob-expand a file pattern into a sorted list of paths.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No snapshot files match pattern: {pattern}")
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Detect and graph proton-like lumps in substrate snapshots.")
    parser.add_argument(
        "--pattern",
        type=str,
        required=True,
        help="Glob pattern for snapshot npz files (e.g. out/snap_*.npz).",
    )
    parser.add_argument(
        "--vacuum",
        type=float,
        default=0.0,
        help="Vacuum field value (used for lumpiness measure).",
    )
    parser.add_argument(
        "--min_voxels",
        type=int,
        default=10,
        help="Minimum number of voxels in a lump to be considered a proton candidate.",
    )
    parser.add_argument(
        "--max_link_distance",
        type=float,
        default=3.0,
        help="Maximum distance to link blobs across frames when tracking.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="proton_map",
        help="Prefix for output CSV and plot files.",
    )
    parser.add_argument(
        "--threshold_sigma",
        type=float,
        default=2.0,
        help="Auto-threshold is mean + threshold_sigma * std of lumpiness.",
    )
    args = parser.parse_args()

    paths = discover_snapshot_paths(args.pattern)
    print(f"[INFO] Found {len(paths)} snapshot files.")

    all_candidates: List[List[ProtonCandidate]] = []

    for frame_index, path in enumerate(paths):
        field, t, dx, dy, dz = load_snapshot_npz(path)
        print(f"[INFO] Processing frame {frame_index} ({path}), t={t}")

        # We could pass a custom topological density here if you have it.
        # For now we use the placeholder.
        candidates = find_protons_in_field(
            field=field,
            frame_index=frame_index,
            time=t if t is not None else float(frame_index),
            dx=dx,
            dy=dy,
            dz=dz,
            lumpiness_threshold=None,   # we’ll override inside using threshold_sigma
            min_voxels=args.min_voxels,
            vacuum=args.vacuum,
        )
        # Adjust threshold if you want to use threshold_sigma externally; current code computes inside.
        all_candidates.append(candidates)
        print(f"[INFO] Found {len(candidates)} proton candidates in frame {frame_index}")

    # Track across time
    tracks = track_protons_over_time(
        all_candidates=all_candidates,
        max_link_distance=args.max_link_distance,
    )
    print(f"[INFO] Built {len(tracks)} tracks in total.")

    # Outputs
    out_candidates_csv = f"{args.out_prefix}_candidates.csv"
    out_tracks_csv = f"{args.out_prefix}_tracks.csv"
    out_tracks_plot = f"{args.out_prefix}_tracks_3d.png"
    out_energy_plot = f"{args.out_prefix}_energy_vs_time.png"

    save_candidates_csv(all_candidates, out_candidates_csv)
    save_tracks_csv(tracks, out_tracks_csv)
    plot_tracks_3d(tracks, out_tracks_plot)
    plot_energy_vs_time(tracks, out_energy_plot)


if __name__ == "__main__":
    main()

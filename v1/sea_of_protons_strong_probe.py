#!/usr/bin/env python3
"""
sea_of_protons_strong_probe.py

Analyze a "sea of protons" run for strong-force-like behavior by
computing the radial pair-correlation function g(r) from lumps_all.csv.

What it does:
  - loads lumps_all.csv from a sea_of_protons_output directory,
  - filters to times t >= t_min to focus on the equilibrium regime,
  - for each selected frame:
      * takes lump positions (x, y),
      * computes all pairwise distances using minimal-image convention
        on an L x L periodic box,
  - accumulates a histogram of pair distances,
  - normalizes it to an ideal-gas reference to obtain g(r),
  - plots g(r) vs r and prints a few summary stats.

Interpretation:
  - g(r) ~ 0 at small r, peak at some finite r  => short-range repulsion +
                                                   intermediate attraction
  - g(r) ~ 1 everywhere                          => essentially non-interacting
  - g(r) >> 1 at small r                         => clustering / collapse

This script does *no* new physics — it just measures what the sea-of-protons
simulation already produced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def minimal_image_distances(positions: np.ndarray, L: float) -> np.ndarray:
    """
    Compute all unique pairwise distances with minimal-image convention
    for a square box of side L (periodic BCs).

    positions: (N, 2) array of (x, y)
    returns:   1D array of length N_pairs = N*(N-1)/2
    """
    N = positions.shape[0]
    if N < 2:
        return np.array([], dtype=float)

    # pairwise differences via broadcasting
    dx = positions[:, 0][:, None] - positions[:, 0][None, :]
    dy = positions[:, 1][:, None] - positions[:, 1][None, :]

    # minimal-image: shift differences into [-L/2, L/2)
    dx = (dx + L / 2.0) % L - L / 2.0
    dy = (dy + L / 2.0) % L - L / 2.0

    dist = np.sqrt(dx**2 + dy**2)

    # take upper triangle (i < j) to avoid double-counting/self
    iu = np.triu_indices(N, k=1)
    return dist[iu]


def compute_g_of_r(
    df_lumps: pd.DataFrame,
    L: float,
    t_min: float,
    dr: float,
    r_max: float | None = None,
):
    """
    Compute g(r) from lump positions in df_lumps for times t >= t_min.

    Returns:
      r_centers : 1D array of bin centers
      g_r       : 1D array of g(r) values
    """
    # restrict to equilibrium-ish times
    df_eq = df_lumps[df_lumps["time"] >= t_min].copy()
    if df_eq.empty:
        raise ValueError(f"No lumps found with time >= {t_min}.")

    # collect distances frame by frame
    all_dists = []
    frame_groups = df_eq.groupby("frame_index")

    total_pairs = 0
    total_lumps = 0
    for fid, group in frame_groups:
        positions = group[["x", "y"]].to_numpy()
        N = positions.shape[0]
        if N < 2:
            continue
        dists = minimal_image_distances(positions, L=L)
        if dists.size == 0:
            continue
        all_dists.append(dists)
        total_pairs += dists.size
        total_lumps += N

    if not all_dists:
        raise ValueError("No pair distances computed; not enough lumps per frame?")

    all_dists = np.concatenate(all_dists)
    print(f"[INFO] Used {len(frame_groups)} frames, total pairs={total_pairs}, "
          f"avg lumps/frame≈{total_lumps / max(len(frame_groups), 1):.1f}")

    if r_max is None:
        r_max = L / 2.0  # max meaningful distance in periodic 2D box

    # histogram distances
    bins = np.arange(0.0, r_max + dr, dr)
    hist, edges = np.histogram(all_dists, bins=bins)
    r_centers = 0.5 * (edges[:-1] + edges[1:])

    # normalize to ideal-gas reference
    # In 2D, expected number of pairs in shell [r, r+dr]:
    #   dN_ideal(r) = 2π r dr * ρ * N_total_frames / 2   (approx up to constants)
    # We'll normalize such that g(r) -> 1 for a uniform distribution.
    area = L * L
    avg_N = total_lumps / max(len(frame_groups), 1)
    rho = avg_N / area  # lump number density per frame

    shell_area = 2.0 * np.pi * r_centers * dr
    # expected pairs per frame per shell for an ideal gas:
    # N * rho * shell_area / 2  (each pair counted twice)
    expected_per_frame = avg_N * rho * shell_area / 2.0
    # multiply by number of frames to compare with hist
    n_frames = len(frame_groups)
    expected_total = expected_per_frame * n_frames

    # avoid division by zero at r=0
    g_r = np.zeros_like(r_centers)
    mask = expected_total > 0
    g_r[mask] = hist[mask] / expected_total[mask]

    return r_centers, g_r


def main():
    parser = argparse.ArgumentParser(
        description="Probe sea-of-protons data for strong-force-like behavior "
                    "by computing g(r) from lumps_all.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="sea_of_protons_output",
        help="Directory containing lumps_all.csv from sea_of_protons_sim.py.",
    )
    parser.add_argument(
        "--L",
        type=float,
        default=256.0,
        help="Box size (physical units; for dx=1, this is the grid size).",
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=10.0,
        help="Minimum time to include (focus on equilibrium regime).",
    )
    parser.add_argument(
        "--dr",
        type=float,
        default=1.0,
        help="Radial bin width for g(r).",
    )
    parser.add_argument(
        "--r-max",
        type=float,
        default=None,
        help="Maximum radius to consider (default: L/2).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="strong_probe",
        help="Prefix for output plot filename.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    lumps_path = input_dir / "lumps_all.csv"
    if not lumps_path.exists():
        raise FileNotFoundError(f"Could not find {lumps_path}")

    print(f"[INFO] Loading lumps from {lumps_path}")
    df_lumps = pd.read_csv(lumps_path)

    r_centers, g_r = compute_g_of_r(
        df_lumps=df_lumps,
        L=args.L,
        t_min=args.t_min,
        dr=args.dr,
        r_max=args.r_max,
    )

    # Simple summary: where is g(r) suppressed / enhanced?
    peak_idx = int(np.argmax(g_r))
    print(f"[RESULT] Max g(r) ≈ {g_r[peak_idx]:.2f} at r ≈ {r_centers[peak_idx]:.1f}")

    # Save plot
    plt.figure(figsize=(6, 4))
    plt.plot(r_centers, g_r, "-o", markersize=3)
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="g(r)=1 (ideal gas)")
    plt.xlabel("r (physical units)")
    plt.ylabel("g(r)")
    plt.title("Radial pair-correlation g(r) (sea of protons)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = input_dir / f"{args.output_prefix}_gofr.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[RESULT] Saved g(r) plot -> {out_png}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
symmetry_gr_occupancy_compare.py

Compare "fermion" vs "boson" symmetry-selector runs using defect catalogs
from defect_braid_analysis_gpu.py.

Input files (expected in CWD by default):
    fermion_defects_defects.csv
    boson_defects_defects.csv

Each defects CSV should have columns:
    frame_idx, time, x, y, w

Coordinates x,y are half-integers on a periodic LxL grid:
    x, y ∈ {0.5, 1.5, ..., L-0.5}
We infer L from max(x,y).

The script computes, for each symmetry sector separately (fermion/boson):

    1) Occupancy histograms:
        - all defects
        - w = +1 defects
        - w = -1 defects

    2) Pair correlation functions g(r):
        - all defects
        - w = +1
        - w = -1

and writes plots to symmetry_analysis_output/.

Example:
    python symmetry_gr_occupancy_compare.py ^
        --fermion_prefix fermion_defects ^
        --boson_prefix boson_defects ^
        --dx 1.0 ^
        --dr 0.5 ^
        --frame_stride 4 ^
        --max_per_frame 400
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def infer_L_from_defects(df):
    """
    Given a defects dataframe with x,y in half-integer cell centers,
    infer L such that x,y in {0.5, 1.5, ..., L-0.5}.

    We take:
        L ~ round(max(x) + 0.5)
    """
    max_x = df["x"].max()
    max_y = df["y"].max()
    Lx = int(round(max_x + 0.5))
    Ly = int(round(max_y + 0.5))
    if Lx != Ly:
        print(f"[WARN] Non-square domain inferred: Lx={Lx}, Ly={Ly}. Using L={Lx}.")
    return Lx


def coords_to_indices(x, y):
    """
    Map half-integer coordinates (0.5, 1.5, ...) to integer cell indices (0,...,L-1).
    i = round(x - 0.5)
    """
    i = np.rint(x - 0.5).astype(int)
    j = np.rint(y - 0.5).astype(int)
    return i, j


def compute_occupancy(df, L):
    """
    Compute occupancy arrays for:
        - all defects
        - w=+1 only
        - w=-1 only

    Returns:
        occ_all, occ_plus, occ_minus (each LxL array)
    """
    occ_all = np.zeros((L, L), dtype=np.int64)
    occ_plus = np.zeros((L, L), dtype=np.int64)
    occ_minus = np.zeros((L, L), dtype=np.int64)

    x = df["x"].values
    y = df["y"].values
    w = df["w"].values

    i, j = coords_to_indices(x, y)

    # All defects
    np.add.at(occ_all, (i, j), 1)

    # Plus defects
    mask_plus = (w > 0)
    np.add.at(occ_plus, (i[mask_plus], j[mask_plus]), 1)

    # Minus defects
    mask_minus = (w < 0)
    np.add.at(occ_minus, (i[mask_minus], j[mask_minus]), 1)

    return occ_all, occ_plus, occ_minus


def minimal_image_displacements(pos, L_phys):
    """
    For pos: (N,2) array on periodic [0,L_phys) x [0,L_phys),
    return pairwise displacements with minimal image convention.

    Returns:
        dx, dy: (N,N) arrays
    """
    x = pos[:, 0][:, None] - pos[:, 0][None, :]
    y = pos[:, 1][:, None] - pos[:, 1][None, :]

    x = x - np.rint(x / L_phys) * L_phys
    y = y - np.rint(y / L_phys) * L_phys

    return x, y


def accumulate_gr_for_subset(pos, L_phys, r_edges, counts):
    """
    Given positions pos (N,2), accumulate pair distances into counts (len nbins)
    using radial bins r_edges, with periodic boundaries.

    counts is modified in place and also returned.
    """
    N = pos.shape[0]
    if N < 2:
        return counts

    dx, dy = minimal_image_displacements(pos, L_phys)
    r = np.sqrt(dx ** 2 + dy ** 2)

    # Use upper triangle (i < j) to avoid double counting and self pairs
    iu, ju = np.triu_indices(N, k=1)
    r_pairs = r[iu, ju]

    hist, _ = np.histogram(r_pairs, bins=r_edges)
    counts += hist
    return counts


def compute_gr(df, L, dx, dr, frame_stride=1, max_per_frame=None, seed=1234):
    """
    Compute pair correlation histograms for:
        - all defects
        - w=+1
        - w=-1

    using subsampling for efficiency if desired.

    Parameters
    ----------
    df : DataFrame
        defects table with columns frame_idx, x, y, w
    L : int
        grid size
    dx : float
        lattice spacing
    dr : float
        radial bin width
    frame_stride : int
        use every 'frame_stride'-th frame
    max_per_frame : int or None
        if not None, randomly downsample defects per frame to this many
    seed : int
        RNG seed for reproducibility

    Returns
    -------
    r_centers, g_all_counts, g_plus_counts, g_minus_counts
    (raw pair counts; you can optionally normalize these to true g(r))

    Note: we are computing something proportional to g(r), but not
    performing the final normalization by 2πrρ etc. For relative
    comparisons between symmetry sectors this is usually fine.
    """
    rng = np.random.default_rng(seed)

    L_phys = L * dx
    r_max = L_phys * np.sqrt(2.0) / 2.0  # up to half-diagonal
    r_edges = np.arange(0.0, r_max + dr, dr)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    nbins = len(r_centers)

    g_all = np.zeros(nbins, dtype=np.float64)
    g_plus = np.zeros(nbins, dtype=np.float64)
    g_minus = np.zeros(nbins, dtype=np.float64)

    frames = np.sort(df["frame_idx"].unique())
    frames = frames[::frame_stride]

    print(f"[g(r)] Using {len(frames)} frames (stride={frame_stride}).")

    for f in frames:
        sub = df[df["frame_idx"] == f]
        if len(sub) < 2:
            continue

        # Positions in physical units
        x = sub["x"].values * dx
        y = sub["y"].values * dx
        w = sub["w"].values

        # Optional downsample for speed
        idx_all = np.arange(len(sub))
        if (max_per_frame is not None) and (len(sub) > max_per_frame):
            idx_all = rng.choice(idx_all, size=max_per_frame, replace=False)

        # All defects
        pos_all = np.stack([x[idx_all], y[idx_all]], axis=1)
        g_all = accumulate_gr_for_subset(pos_all, L_phys, r_edges, g_all)

        # Plus defects
        mask_plus = (w > 0)
        idx_plus = idx_all[mask_plus[idx_all]]
        if len(idx_plus) >= 2:
            pos_plus = np.stack([x[idx_plus], y[idx_plus]], axis=1)
            g_plus = accumulate_gr_for_subset(pos_plus, L_phys, r_edges, g_plus)

        # Minus defects
        mask_minus = (w < 0)
        idx_minus = idx_all[mask_minus[idx_all]]
        if len(idx_minus) >= 2:
            pos_minus = np.stack([x[idx_minus], y[idx_minus]], axis=1)
            g_minus = accumulate_gr_for_subset(pos_minus, L_phys, r_edges, g_minus)

    return r_centers, g_all, g_plus, g_minus


def plot_occupancy_hist(occ, title, out_path):
    """
    Plot a 1D histogram of occupancy counts (flattened grid).
    """
    counts = occ.ravel()
    plt.figure(figsize=(6, 4))
    plt.hist(counts, bins=50, density=True)
    plt.xlabel("Occupancy per cell")
    plt.ylabel("Probability density")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVE] {out_path}")


def plot_gr(r, g, title, out_path):
    """
    Plot raw pair-count g(r) curve.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(r, g, marker="", linestyle="-")
    plt.xlabel("r")
    plt.ylabel("pair count (arb. units)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVE] {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare fermion vs boson symmetry runs using defect catalogs."
    )
    parser.add_argument(
        "--fermion_prefix",
        type=str,
        default="fermion_defects",
        help="Prefix for fermion run (expects <prefix>_defects_defects.csv).",
    )
    parser.add_argument(
        "--boson_prefix",
        type=str,
        default="boson_defects",
        help="Prefix for boson run (expects <prefix>_defects_defects.csv).",
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=1.0,
        help="Lattice spacing.",
    )
    parser.add_argument(
        "--dr",
        type=float,
        default=0.5,
        help="Radial bin size for g(r).",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=4,
        help="Use every frame_stride-th frame for g(r).",
    )
    parser.add_argument(
        "--max_per_frame",
        type=int,
        default=400,
        help="Max defects per frame for g(r) (downsample for speed).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="symmetry_analysis_output",
        help="Directory to write plots.",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load defects ---
    f_def_path = f"{args.fermion_prefix}_defects_defects.csv"
    b_def_path = f"{args.boson_prefix}_defects_defects.csv"

    print(f"[LOAD] Fermion defects from {f_def_path}")
    df_f = pd.read_csv(f_def_path)

    print(f"[LOAD] Boson defects from {b_def_path}")
    df_b = pd.read_csv(b_def_path)

    # Sanity: infer L from fermion run
    L_f = infer_L_from_defects(df_f)
    L_b = infer_L_from_defects(df_b)
    if L_f != L_b:
        print(f"[WARN] Fermion L={L_f}, Boson L={L_b}, using L={L_f}")
    L = L_f

    print(f"[INFO] Inferred grid size L={L}, dx={args.dx}")

    # ------------------------------
    # Fermion run: occupancy & g(r)
    # ------------------------------
    print("[FERMION] Computing occupancy...")
    occ_f_all, occ_f_plus, occ_f_minus = compute_occupancy(df_f, L)

    plot_occupancy_hist(
        occ_f_all,
        title="Fermion run: occupancy (all defects)",
        out_path=os.path.join(args.out_dir, "occupancy_hist_fermion_all.png"),
    )
    plot_occupancy_hist(
        occ_f_plus,
        title="Fermion run: occupancy (w=+1)",
        out_path=os.path.join(args.out_dir, "occupancy_hist_fermion_plus.png"),
    )
    plot_occupancy_hist(
        occ_f_minus,
        title="Fermion run: occupancy (w=-1)",
        out_path=os.path.join(args.out_dir, "occupancy_hist_fermion_minus.png"),
    )

    print("[FERMION] Computing g(r)...")
    r_f, g_f_all, g_f_plus, g_f_minus = compute_gr(
        df_f,
        L=L,
        dx=args.dx,
        dr=args.dr,
        frame_stride=args.frame_stride,
        max_per_frame=args.max_per_frame,
        seed=1234,
    )

    plot_gr(
        r_f,
        g_f_all,
        title="Fermion run: g(r) (all defects)",
        out_path=os.path.join(args.out_dir, "g_r_fermion_all.png"),
    )
    plot_gr(
        r_f,
        g_f_plus,
        title="Fermion run: g(r) (w=+1)",
        out_path=os.path.join(args.out_dir, "g_r_fermion_plus.png"),
    )
    plot_gr(
        r_f,
        g_f_minus,
        title="Fermion run: g(r) (w=-1)",
        out_path=os.path.join(args.out_dir, "g_r_fermion_minus.png"),
    )

    # ------------------------------
    # Boson run: occupancy & g(r)
    # ------------------------------
    print("[BOSON] Computing occupancy...")
    occ_b_all, occ_b_plus, occ_b_minus = compute_occupancy(df_b, L)

    plot_occupancy_hist(
        occ_b_all,
        title="Boson run: occupancy (all defects)",
        out_path=os.path.join(args.out_dir, "occupancy_hist_boson_all.png"),
    )
    plot_occupancy_hist(
        occ_b_plus,
        title="Boson run: occupancy (w=+1)",
        out_path=os.path.join(args.out_dir, "occupancy_hist_boson_plus.png"),
    )
    plot_occupancy_hist(
        occ_b_minus,
        title="Boson run: occupancy (w=-1)",
        out_path=os.path.join(args.out_dir, "occupancy_hist_boson_minus.png"),
    )

    print("[BOSON] Computing g(r)...")
    r_b, g_b_all, g_b_plus, g_b_minus = compute_gr(
        df_b,
        L=L,
        dx=args.dx,
        dr=args.dr,
        frame_stride=args.frame_stride,
        max_per_frame=args.max_per_frame,
        seed=5678,
    )

    plot_gr(
        r_b,
        g_b_all,
        title="Boson run: g(r) (all defects)",
        out_path=os.path.join(args.out_dir, "g_r_boson_all.png"),
    )
    plot_gr(
        r_b,
        g_b_plus,
        title="Boson run: g(r) (w=+1)",
        out_path=os.path.join(args.out_dir, "g_r_boson_plus.png"),
    )
    plot_gr(
        r_b,
        g_b_minus,
        title="Boson run: g(r) (w=-1)",
        out_path=os.path.join(args.out_dir, "g_r_boson_minus.png"),
    )

    print("[DONE] Plots written to", args.out_dir)


if __name__ == "__main__":
    main()

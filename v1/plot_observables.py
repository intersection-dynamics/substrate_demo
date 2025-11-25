#!/usr/bin/env python
"""
plot_observables.py

Utility script for the Substrate Framework Exclusion Engine.

- Loads observables.csv from a run directory
- Plots:
    1) Core separation vs time
    2) Max density vs time
    3) Raw core positions (x1,x2,y1,y2) vs time
    4) Tracked worldlines (xA,xB,yA,yB) with identity continuity

Usage (from c:\\GitHub\\substrate_demo):

    python plot_observables.py --run_dir output_exclusion

You can point --run_dir at w1_run, w0_run, etc.
"""

import os
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_observables(run_dir: str) -> pd.DataFrame:
    """
    Load observables.csv from the given run directory.
    """
    obs_path = os.path.join(run_dir, "observables.csv")
    if not os.path.isfile(obs_path):
        raise FileNotFoundError(f"Could not find observables.csv in: {run_dir}")
    
    df = pd.read_csv(obs_path)
    return df


def plot_separation(df: pd.DataFrame, run_dir: str):
    """
    Plot core separation vs time and save to PNG.
    """
    t = df["time"].values
    sep = df["core_sep"].values

    plt.figure()
    plt.plot(t, sep, marker="o")
    plt.xlabel("time")
    plt.ylabel("core separation")
    plt.title("Core separation vs time")
    plt.grid(True)

    out_path = os.path.join(run_dir, "core_separation.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Saved separation plot to {out_path}")


def plot_max_rho(df: pd.DataFrame, run_dir: str):
    """
    Plot max density vs time and save to PNG.
    """
    t = df["time"].values
    max_rho = df["max_rho"].values

    plt.figure()
    plt.plot(t, max_rho, marker="o")
    plt.xlabel("time")
    plt.ylabel("max rho")
    plt.title("Peak density vs time")
    plt.grid(True)

    out_path = os.path.join(run_dir, "max_rho.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Saved max_rho plot to {out_path}")


def plot_raw_worldlines(df: pd.DataFrame, run_dir: str):
    """
    Plot raw core worldlines x1(t), x2(t), y1(t), y2(t).
    """
    t = df["time"].values
    x1 = df["x1"].values
    y1 = df["y1"].values
    x2 = df["x2"].values
    y2 = df["y2"].values

    # X positions vs time
    plt.figure()
    plt.plot(t, x1, marker="o", label="x1 (raw)")
    plt.plot(t, x2, marker="s", label="x2 (raw)")
    plt.xlabel("time")
    plt.ylabel("x position")
    plt.title("Core x-positions vs time (raw)")
    plt.legend()
    plt.grid(True)
    out_path_x = os.path.join(run_dir, "worldlines_x_raw.png")
    plt.tight_layout()
    plt.savefig(out_path_x, dpi=150)
    plt.close()
    print(f"[plot] Saved raw x-worldlines to {out_path_x}")

    # Y positions vs time
    plt.figure()
    plt.plot(t, y1, marker="o", label="y1 (raw)")
    plt.plot(t, y2, marker="s", label="y2 (raw)")
    plt.xlabel("time")
    plt.ylabel("y position")
    plt.title("Core y-positions vs time (raw)")
    plt.legend()
    plt.grid(True)
    out_path_y = os.path.join(run_dir, "worldlines_y_raw.png")
    plt.tight_layout()
    plt.savefig(out_path_y, dpi=150)
    plt.close()
    print(f"[plot] Saved raw y-worldlines to {out_path_y}")


def track_worldlines(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build "tracked" worldlines (xA,yA,xB,yB) from raw (x1,y1,x2,y2) by enforcing
    identity continuity: at each timestep, choose the assignment that minimizes
    the total displacement from the previous A/B positions.

    Returns
    -------
    xA, yA, xB, yB : np.ndarray
        Tracked positions for core A and core B.
    """
    n = len(x1)
    if not (len(y1) == len(x2) == len(y2) == n):
        raise ValueError("Input arrays must have the same length")

    xA = np.zeros_like(x1, dtype=float)
    yA = np.zeros_like(y1, dtype=float)
    xB = np.zeros_like(x2, dtype=float)
    yB = np.zeros_like(y2, dtype=float)

    # Initialize: at t0, just take raw labels as A/B
    xA[0], yA[0] = x1[0], y1[0]
    xB[0], yB[0] = x2[0], y2[0]

    for k in range(1, n):
        # Positions at previous step (A,B) and current raw (1,2)
        xA_prev, yA_prev = xA[k - 1], yA[k - 1]
        xB_prev, yB_prev = xB[k - 1], yB[k - 1]

        x1_curr, y1_curr = x1[k], y1[k]
        x2_curr, y2_curr = x2[k], y2[k]

        # Option 1: keep (1->A, 2->B)
        dA1 = np.hypot(x1_curr - xA_prev, y1_curr - yA_prev)
        dB2 = np.hypot(x2_curr - xB_prev, y2_curr - yB_prev)
        cost_keep = dA1 + dB2

        # Option 2: swap (2->A, 1->B)
        dA2 = np.hypot(x2_curr - xA_prev, y2_curr - yA_prev)
        dB1 = np.hypot(x1_curr - xB_prev, y1_curr - yB_prev)
        cost_swap = dA2 + dB1

        if cost_keep <= cost_swap:
            # Keep labels
            xA[k], yA[k] = x1_curr, y1_curr
            xB[k], yB[k] = x2_curr, y2_curr
        else:
            # Swap labels
            xA[k], yA[k] = x2_curr, y2_curr
            xB[k], yB[k] = x1_curr, y1_curr

    return xA, yA, xB, yB


def plot_tracked_worldlines(df: pd.DataFrame, run_dir: str):
    """
    Plot tracked worldlines A/B to remove label-jumping artefacts.
    """
    t = df["time"].values
    x1 = df["x1"].values
    y1 = df["y1"].values
    x2 = df["x2"].values
    y2 = df["y2"].values

    xA, yA, xB, yB = track_worldlines(x1, y1, x2, y2)

    # X positions vs time (tracked)
    plt.figure()
    plt.plot(t, xA, marker="o", label="xA (tracked)")
    plt.plot(t, xB, marker="s", label="xB (tracked)")
    plt.xlabel("time")
    plt.ylabel("x position")
    plt.title("Core x-positions vs time (tracked)")
    plt.legend()
    plt.grid(True)
    out_path_x = os.path.join(run_dir, "worldlines_x_tracked.png")
    plt.tight_layout()
    plt.savefig(out_path_x, dpi=150)
    plt.close()
    print(f"[plot] Saved tracked x-worldlines to {out_path_x}")

    # Y positions vs time (tracked)
    plt.figure()
    plt.plot(t, yA, marker="o", label="yA (tracked)")
    plt.plot(t, yB, marker="s", label="yB (tracked)")
    plt.xlabel("time")
    plt.ylabel("y position")
    plt.title("Core y-positions vs time (tracked)")
    plt.legend()
    plt.grid(True)
    out_path_y = os.path.join(run_dir, "worldlines_y_tracked.png")
    plt.tight_layout()
    plt.savefig(out_path_y, dpi=150)
    plt.close()
    print(f"[plot] Saved tracked y-worldlines to {out_path_y}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot observables from Substrate Exclusion Engine runs."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="output_exclusion",
        help="Directory containing observables.csv (e.g., output_exclusion, w1_run, w0_run).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    print(f"[info] Loading observables from: {run_dir}")

    df = load_observables(run_dir)

    print("[info] Data summary:")
    print(df.describe())

    plot_separation(df, run_dir)
    plot_max_rho(df, run_dir)
    plot_raw_worldlines(df, run_dir)
    plot_tracked_worldlines(df, run_dir)

    print("[done] All plots generated.")


if __name__ == "__main__":
    main()

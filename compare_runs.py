#!/usr/bin/env python
"""
compare_runs.py

Compare two Substrate Exclusion Engine runs (e.g., fermion-like w=1 vs boson-like w=0).

- Each run directory must contain an observables.csv produced by substrate_engine.py
- This script loads both and overlays:
    1) core separation vs time
    2) max density vs time

Usage (from c:\\GitHub\\substrate_demo):

    python compare_runs.py --run1 w1_run --run2 w0_run --label1 "w=1" --label2 "w=0"

Defaults assume:
    run1 = "w1_run"
    run2 = "w0_run"
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_observables(run_dir: str) -> pd.DataFrame:
    """
    Load observables.csv from a given run directory.
    """
    obs_path = os.path.join(run_dir, "observables.csv")
    if not os.path.isfile(obs_path):
        raise FileNotFoundError(f"Could not find observables.csv in: {run_dir}")
    df = pd.read_csv(obs_path)
    return df


def compare_separation(df1: pd.DataFrame,
                       df2: pd.DataFrame,
                       label1: str,
                       label2: str,
                       out_dir: str):
    """
    Plot core separation vs time for two runs on the same axes.
    """
    t1 = df1["time"].values
    sep1 = df1["core_sep"].values

    t2 = df2["time"].values
    sep2 = df2["core_sep"].values

    plt.figure()
    plt.plot(t1, sep1, marker="o", label=f"{label1} separation")
    plt.plot(t2, sep2, marker="s", label=f"{label2} separation")

    plt.xlabel("time")
    plt.ylabel("core separation")
    plt.title("Core separation vs time (comparison)")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(out_dir, "compare_separation.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[compare] Saved separation comparison to {out_path}")


def compare_max_rho(df1: pd.DataFrame,
                    df2: pd.DataFrame,
                    label1: str,
                    label2: str,
                    out_dir: str):
    """
    Plot max_rho vs time for two runs on the same axes.
    """
    t1 = df1["time"].values
    rho1 = df1["max_rho"].values

    t2 = df2["time"].values
    rho2 = df2["max_rho"].values

    plt.figure()
    plt.plot(t1, rho1, marker="o", label=f"{label1} max rho")
    plt.plot(t2, rho2, marker="s", label=f"{label2} max rho")

    plt.xlabel("time")
    plt.ylabel("max rho")
    plt.title("Peak density vs time (comparison)")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(out_dir, "compare_max_rho.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[compare] Saved max_rho comparison to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two Substrate Exclusion Engine runs."
    )
    parser.add_argument(
        "--run1",
        type=str,
        default="w1_run",
        help="First run directory (e.g., fermion-like w=1)."
    )
    parser.add_argument(
        "--run2",
        type=str,
        default="w0_run",
        help="Second run directory (e.g., boson-like w=0)."
    )
    parser.add_argument(
        "--label1",
        type=str,
        default="w=1",
        help="Label for first run in plots."
    )
    parser.add_argument(
        "--label2",
        type=str,
        default="w=0",
        help="Label for second run in plots."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="comparison",
        help="Output directory for comparison plots."
    )

    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    print(f"[info] Loading observables from '{args.run1}' and '{args.run2}'")

    df1 = load_observables(args.run1)
    df2 = load_observables(args.run2)

    print("[info] Summary for run1:")
    print(df1.describe())
    print("\n[info] Summary for run2:")
    print(df2.describe())

    compare_separation(df1, df2, args.label1, args.label2, args.out)
    compare_max_rho(df1, df2, args.label1, args.label2, args.out)

    print("[done] Comparison plots generated.")


if __name__ == "__main__":
    main()

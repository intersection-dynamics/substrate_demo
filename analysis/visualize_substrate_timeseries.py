"""
analysis/visualize_substrate_timeseries.py

Visualize timeseries diagnostics from a single substrate_pointer_exp run.

It expects a run directory structure like:

  outputs/substrate_pointer_exp/<run_id>/
    data/timeseries.npz
    summary.json
    ...

This script:
  - Loads data/timeseries.npz.
  - Plots:
      * energy vs step
      * fermion_lump_count vs step
      * fermion COM components vs step
      * COM xy-trajectory
  - Writes figures to:
      <run_dir>/analysis_figures/

Usage (from repo root):

  # Point to a specific run directory
  python analysis\\visualize_substrate_timeseries.py --run-dir outputs\\substrate_pointer_exp\\20251124_120000_dt0p010_m2f-1p0_s0

  # Or let it pick the latest run under outputs\\substrate_pointer_exp
  python analysis\\visualize_substrate_timeseries.py --latest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize substrate_pointer_exp timeseries for a single run."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Path to a specific run directory (e.g. outputs/substrate_pointer_exp/<run_id>).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="If set, ignore --run-dir and select the most recently modified run under outputs/substrate_pointer_exp.",
    )
    return parser.parse_args()


def find_latest_run(base_root: Path) -> Optional[Path]:
    if not base_root.exists():
        return None

    run_dirs = [d for d in base_root.iterdir() if d.is_dir()]
    if not run_dirs:
        return None

    # Sort by modification time, newest last
    run_dirs.sort(key=lambda p: p.stat().st_mtime)
    return run_dirs[-1]


def load_timeseries(run_dir: Path) -> dict:
    data_path = run_dir / "data" / "timeseries.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"timeseries.npz not found under {data_path}")

    arrs = np.load(data_path)
    # Convert to regular dict of numpy arrays
    return {k: arrs[k] for k in arrs.files}


def safe_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main() -> None:
    args = parse_args()

    if args.latest:
        base_root = REPO_ROOT / "outputs" / "substrate_pointer_exp"
        run_dir = find_latest_run(base_root)
        if run_dir is None:
            print(f"[ERROR] No run directories found under {base_root}")
            sys.exit(1)
    else:
        if not args.run_dir:
            print("[ERROR] You must supply --run-dir or use --latest.")
            sys.exit(1)
        run_dir = Path(args.run_dir).resolve()

    if not run_dir.exists():
        print(f"[ERROR] Run directory does not exist: {run_dir}")
        sys.exit(1)

    print(f"[INFO] Using run directory: {run_dir}")

    ts = load_timeseries(run_dir)
    summary = safe_summary(run_dir)

    step = ts.get("step")
    energy = ts.get("energy")
    norm = ts.get("norm")
    fermion_lump_count = ts.get("fermion_lump_count")
    fermion_com = ts.get("fermion_com")

    if step is None:
        print("[ERROR] 'step' not found in timeseries.")
        sys.exit(1)

    # Prepare output directory
    analysis_fig_dir = run_dir / "analysis_figures"
    analysis_fig_dir.mkdir(parents=True, exist_ok=True)

    # Optional title suffix from params
    params = summary.get("params", {})
    dt = params.get("dt", None)
    mass2_fermion = params.get("mass2_fermion", None)
    tag_parts = []
    if dt is not None:
        tag_parts.append(f"dt={dt}")
    if mass2_fermion is not None:
        tag_parts.append(f"m2f={mass2_fermion}")
    title_suffix = ", ".join(tag_parts)

    # 1) Energy vs step
    if energy is not None:
        plt.figure()
        plt.plot(step, energy)
        plt.xlabel("step")
        plt.ylabel("mean energy density")
        plt.title("Energy vs step" + (f" ({title_suffix})" if title_suffix else ""))
        plt.tight_layout()
        plt.savefig(analysis_fig_dir / "energy_vs_step.png", dpi=150)
        plt.close()

    # 2) Norm vs step
    if norm is not None:
        plt.figure()
        plt.plot(step, norm)
        plt.xlabel("step")
        plt.ylabel("norm")
        plt.title("Norm vs step" + (f" ({title_suffix})" if title_suffix else ""))
        plt.tight_layout()
        plt.savefig(analysis_fig_dir / "norm_vs_step.png", dpi=150)
        plt.close()

    # 3) Fermion lump count vs step
    if fermion_lump_count is not None and fermion_lump_count.size == step.size:
        plt.figure()
        plt.plot(step, fermion_lump_count, drawstyle="steps-post")
        plt.xlabel("step")
        plt.ylabel("fermion_lump_count")
        plt.title("Fermion lump count vs step" + (f" ({title_suffix})" if title_suffix else ""))
        plt.tight_layout()
        plt.savefig(analysis_fig_dir / "fermion_lump_count_vs_step.png", dpi=150)
        plt.close()

    # 4) Fermion COM components vs step
    if fermion_com is not None and fermion_com.ndim == 2 and fermion_com.shape[0] == step.size:
        plt.figure()
        plt.plot(step, fermion_com[:, 0], label="x")
        if fermion_com.shape[1] > 1:
            plt.plot(step, fermion_com[:, 1], label="y")
        if fermion_com.shape[1] > 2:
            plt.plot(step, fermion_com[:, 2], label="z")
        plt.xlabel("step")
        plt.ylabel("COM (lattice units)")
        plt.title("Fermion COM components vs step" + (f" ({title_suffix})" if title_suffix else ""))
        plt.legend()
        plt.tight_layout()
        plt.savefig(analysis_fig_dir / "fermion_com_components_vs_step.png", dpi=150)
        plt.close()

        # 5) Fermion COM XY-trajectory
        if fermion_com.shape[1] >= 2:
            plt.figure()
            plt.plot(fermion_com[:, 0], fermion_com[:, 1], marker=".", linestyle="-", linewidth=0.8, markersize=3)
            plt.xlabel("COM_x")
            plt.ylabel("COM_y")
            plt.title("Fermion COM xy-trajectory" + (f" ({title_suffix})" if title_suffix else ""))
            plt.tight_layout()
            plt.savefig(analysis_fig_dir / "fermion_com_xy_trajectory.png", dpi=150)
            plt.close()

    print(f"[INFO] Saved figures under {analysis_fig_dir}")


if __name__ == "__main__":
    main()

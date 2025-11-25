"""
analysis/visualize_hilbert_substrate.py

Visualization tools for hilbert_substrate_exp runs.

Loads:
    outputs/hilbert_substrate_exp/<run_id>/data/snapshots.npz

Expected contents:
    collapse_step: (T,)
    configs: (T, n_sites) with entries in {0,1,2}

Produces:
    field_figures/occupancy_raster.png  (site vs collapse index, color = occupancy)
    field_figures/double_count_vs_time.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Hilbert-substrate occupancy patterns for a single run."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Path to a specific run directory, e.g. outputs/hilbert_substrate_exp/<run_id>",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="If set, pick the most recently modified run under outputs/hilbert_substrate_exp.",
    )
    return parser.parse_args()


def find_latest_run(base_root: Path) -> Optional[Path]:
    if not base_root.exists():
        return None
    run_dirs = [d for d in base_root.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda p: p.stat().st_mtime)
    return run_dirs[-1]


def main() -> None:
    args = parse_args()

    if args.latest:
        base_root = REPO_ROOT / "outputs" / "hilbert_substrate_exp"
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

    snap_path = run_dir / "data" / "snapshots.npz"
    if not snap_path.exists():
        print(f"[ERROR] snapshots.npz not found at {snap_path}")
        sys.exit(1)

    snaps = np.load(snap_path)
    collapse_step = snaps["collapse_step"]
    configs = snaps["configs"]

    if configs.ndim != 2:
        print(f"[ERROR] Expected configs with shape (T, n_sites), got {configs.shape}")
        sys.exit(1)

    T, n_sites = configs.shape
    print(f"[INFO] Loaded {T} collapse snapshots for {n_sites} sites.")

    out_dir = run_dir / "field_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Occupancy raster: y = collapse index, x = site, value = n_i in {0,1,2}
    plt.figure()
    # imshow expects (rows, cols) → (T, n_sites); origin lower so time increases upward
    plt.imshow(configs, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(label="occupancy (n_i in {0,1,2})")
    plt.xlabel("site index")
    plt.ylabel("collapse index")
    plt.title("Occupancy raster over collapse events")
    plt.tight_layout()
    plt.savefig(out_dir / "occupancy_raster.png", dpi=150)
    plt.close()

    # 2) Double-occupancy count vs collapse
    double_counts = np.count_nonzero(configs == 2, axis=1)
    plt.figure()
    plt.plot(collapse_step, double_counts, marker="o")
    plt.xlabel("step")
    plt.ylabel("number of sites with n_i = 2")
    plt.title("Double occupancy count vs collapse")
    plt.tight_layout()
    plt.savefig(out_dir / "double_count_vs_time.png", dpi=150)
    plt.close()

    print(f"[INFO] Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()

"""
analysis/visualize_substrate_field.py

Visualize field density snapshots from a substrate_pointer_exp run.

Expects:
  <run_dir>/data/snapshots.npz with:
    step: (T,)
    rho_F: (T, g, g, g)
    rho_B: (T, g, g, g)

Creates:
  <run_dir>/field_figures/fermion_density_tXXXXX.png
  <run_dir>/field_figures/boson_density_tXXXXX.png
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
        description="Visualize boson/fermion field density snapshots for a single run."
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

    run_dirs.sort(key=lambda p: p.stat().st_mtime)
    return run_dirs[-1]


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

    snap_path = run_dir / "data" / "snapshots.npz"
    if not snap_path.exists():
        print(f"[ERROR] snapshots.npz not found at {snap_path}")
        sys.exit(1)

    snaps = np.load(snap_path)
    step = snaps["step"]
    rho_F = snaps["rho_F"]
    rho_B = snaps["rho_B"]

    if rho_F.ndim != 4 or rho_B.ndim != 4:
        print("[ERROR] Expected rho_F and rho_B to have shape (T, g, g, g).")
        sys.exit(1)

    T, g, _, _ = rho_F.shape
    print(f"[INFO] Loaded {T} snapshots on a {g}^3 lattice.")

    out_dir = run_dir / "field_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    z_mid = g // 2

    for idx in range(T):
        s = int(step[idx])

        # Fermion density slice
        ferm_slice = rho_F[idx, :, :, z_mid]
        plt.figure()
        plt.imshow(ferm_slice.T, origin="lower", interpolation="nearest")
        plt.colorbar(label="fermion density (mid-z)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Fermion density, step={s}, z={z_mid}")
        plt.tight_layout()
        plt.savefig(out_dir / f"fermion_density_t{s:05d}.png", dpi=150)
        plt.close()

        # Boson density slice
        boson_slice = rho_B[idx, :, :, z_mid]
        plt.figure()
        plt.imshow(boson_slice.T, origin="lower", interpolation="nearest")
        plt.colorbar(label="boson density (mid-z)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Boson density, step={s}, z={z_mid}")
        plt.tight_layout()
        plt.savefig(out_dir / f"boson_density_t{s:05d}.png", dpi=150)
        plt.close()

    print(f"[INFO] Saved field slice figures under {out_dir}")


if __name__ == "__main__":
    main()

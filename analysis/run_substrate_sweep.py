"""
analysis/run_substrate_sweep.py

Parameter-sweep driver for experiments/substrate_pointer_exp.py.

This script:
  - Builds a grid of parameter combinations (dt, mass2_fermion, seed).
  - Calls the experiment script once per combo via subprocess.
  - Tags each run based on parameters for easy identification.

Usage (from repo root):

  python analysis\\run_substrate_sweep.py ^
    --dt-values 0.01 0.005 ^
    --mass2-fermion-values -1.0 -0.5 ^
    --seeds 0 1 ^
    --quiet-experiments

(Spaces between values, no commas.)
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


@dataclass
class SweepConfig:
    output_root: str
    dt_values: List[float]
    mass2_fermion_values: List[float]
    seeds: List[int]
    grid_size: int
    steps: int
    n_boson: int
    n_color: int
    n_spin: int
    mass2_boson: float
    lambda4_boson: float
    lambda4_fermion: float
    init_boson_amp: float
    init_fermion_amp: float
    g_bf: float
    lump_sigma_threshold: float
    lump_min_voxels: int
    dry_run: bool
    quiet_experiments: bool


def parse_args() -> SweepConfig:
    parser = argparse.ArgumentParser(
        description="Run parameter sweeps for substrate_pointer_exp."
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root directory where experiment outputs are written.",
    )

    parser.add_argument(
        "--dt-values",
        type=float,
        nargs="+",
        default=[0.01, 0.005],
        help="Space-separated list of dt values, e.g. --dt-values 0.01 0.005",
    )
    parser.add_argument(
        "--mass2-fermion-values",
        type=float,
        nargs="+",
        default=[-1.0],
        help="Space-separated list of fermion mass^2 values, e.g. --mass2-fermion-values -1.0 -0.5",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Space-separated list of integer seeds, e.g. --seeds 0 1 2",
    )

    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)

    parser.add_argument("--n-boson", type=int, default=1)
    parser.add_argument("--n-color", type=int, default=3)
    parser.add_argument("--n-spin", type=int, default=1)

    parser.add_argument("--mass2-boson", type=float, default=1.0)
    parser.add_argument("--lambda4-boson", type=float, default=1.0)
    parser.add_argument("--lambda4-fermion", type=float, default=2.0)

    parser.add_argument("--init-boson-amp", type=float, default=0.05)
    parser.add_argument("--init-fermion-amp", type=float, default=0.05)

    parser.add_argument(
        "--g-bf",
        type=float,
        default=0.0,
        help="Boson–fermion coupling strength.",
    )

    parser.add_argument("--lump-sigma-threshold", type=float, default=2.0)
    parser.add_argument("--lump-min-voxels", type=int, default=4)

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run, but do not execute.",
    )

    parser.add_argument(
        "--quiet-experiments",
        action="store_true",
        help="Pass --quiet to substrate_pointer_exp.py to reduce terminal spam.",
    )

    args = parser.parse_args()

    cfg = SweepConfig(
        output_root=args.output_root,
        dt_values=args.dt_values,
        mass2_fermion_values=args.mass2_fermion_values,
        seeds=args.seeds,
        grid_size=args.grid_size,
        steps=args.steps,
        n_boson=args.n_boson,
        n_color=args.n_color,
        n_spin=args.n_spin,
        mass2_boson=args.mass2_boson,
        lambda4_boson=args.lambda4_boson,
        lambda4_fermion=args.lambda4_fermion,
        init_boson_amp=args.init_boson_amp,
        init_fermion_amp=args.init_fermion_amp,
        g_bf=args.g_bf,
        lump_sigma_threshold=args.lump_sigma_threshold,
        lump_min_voxels=args.lump_min_voxels,
        dry_run=args.dry_run,
        quiet_experiments=args.quiet_experiments,
    )
    return cfg


def build_tag(dt: float, m2f: float, seed: int) -> str:
    """
    Build a compact tag like 'dt0p010_m2f-1p0_s0'.
    """
    def fmt(x: float) -> str:
        s = f"{x:.3g}"
        s = s.replace(".", "p")
        if not s.startswith(("+", "-")):
            s = "+" + s
        return s

    return f"dt{fmt(dt)}_m2f{fmt(m2f)}_s{seed}"


def run_single(cfg: SweepConfig, dt: float, m2f: float, seed: int) -> None:
    script_path = REPO_ROOT / "experiments" / "substrate_pointer_exp.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Experiment script not found: {script_path}")

    tag = build_tag(dt, m2f, seed)

    cmd = [
        sys.executable,
        str(script_path),
        "--output-root",
        cfg.output_root,
        "--tag",
        tag,
        "--grid-size",
        str(cfg.grid_size),
        "--dt",
        str(dt),
        "--steps",
        str(cfg.steps),
        "--n-boson",
        str(cfg.n_boson),
        "--n-color",
        str(cfg.n_color),
        "--n-spin",
        str(cfg.n_spin),
        "--mass2-boson",
        str(cfg.mass2_boson),
        "--mass2-fermion",
        str(m2f),
        "--lambda4-boson",
        str(cfg.lambda4_boson),
        "--lambda4-fermion",
        str(cfg.lambda4_fermion),
        "--init-boson-amp",
        str(cfg.init_boson_amp),
        "--init-fermion-amp",
        str(cfg.init_fermion_amp),
        "--g-bf",
        str(cfg.g_bf),
        "--lump-sigma-threshold",
        str(cfg.lump_sigma_threshold),
        "--lump-min-voxels",
        str(cfg.lump_min_voxels),
        "--seed",
        str(seed),
    ]

    if cfg.quiet_experiments:
        cmd.append("--quiet")

    print("--------------------------------------------------------------------------------")
    print(f"Running dt={dt}, mass2_fermion={m2f}, seed={seed}")
    print("Command:", " ".join(cmd))

    if cfg.dry_run:
        print("[DRY RUN] Not executing.")
        return

    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    print("Done.")


def run_sweep(cfg: SweepConfig) -> None:
    combos = list(itertools.product(cfg.dt_values, cfg.mass2_fermion_values, cfg.seeds))
    if not combos:
        print("No parameter combinations to run.")
        return

    print(f"Total runs to execute: {len(combos)}")

    for dt, m2f, seed in combos:
        run_single(cfg, dt, m2f, seed)

    print("All runs completed.")


def main() -> None:
    cfg = parse_args()
    run_sweep(cfg)


if __name__ == "__main__":
    main()

"""
experiments/substrate_pointer_exp.py

Single-run experiment harness for engines/substrate_engine.py.

Supports:
  - Optional field snapshots controlled by --snapshot-every (passed to engine).
  - Saves timeseries.npz and snapshots.npz (if present).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import the engine
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engines import substrate_engine  # type: ignore[import]


FRAMEWORK_VERSION = "substrate_engine_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Substrate pointer experiment (boson+fermion substrate engine)."
    )
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--tag", type=str, default="")

    # Lattice and evolution
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=200)

    # Bosonic sector
    parser.add_argument("--n-boson", type=int, default=1)
    parser.add_argument("--mass2-boson", type=float, default=1.0)
    parser.add_argument("--lambda4-boson", type=float, default=1.0)
    parser.add_argument("--init-boson-amp", type=float, default=0.05)

    # Fermionic sector
    parser.add_argument("--n-color", type=int, default=3)
    parser.add_argument("--n-spin", type=int, default=1)
    parser.add_argument("--mass2-fermion", type=float, default=-1.0)
    parser.add_argument("--lambda4-fermion", type=float, default=2.0)
    parser.add_argument("--init-fermion-amp", type=float, default=0.05)

    # Coupling
    parser.add_argument("--g-bf", type=float, default=0.0)

    # Lump detection
    parser.add_argument("--lump-sigma-threshold", type=float, default=2.0)
    parser.add_argument("--lump-min-voxels", type=int, default=4)

    # Snapshots
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="If > 0, record boson/fermion density snapshots every this many steps.",
    )

    # RNG
    parser.add_argument("--seed", type=int, default=1)

    # Verbosity
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce terminal output; still logs to logs/run.log.",
    )

    return parser.parse_args()


def generate_run_id(tag: str = "") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if tag:
        return f"{ts}_{tag}"
    return ts


def create_run_dirs(output_root: str, experiment_name: str, run_id: str) -> Dict[str, Path]:
    base = Path(output_root) / experiment_name / run_id
    if base.exists():
        raise RuntimeError(f"Run directory already exists: {base}")
    data_dir = base / "data"
    figures_dir = base / "figures"
    logs_dir = base / "logs"
    for d in (base, data_dir, figures_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=False)
    return {"base": base, "data": data_dir, "figures": figures_dir, "logs": logs_dir}


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    experiment_name = Path(__file__).stem  # "substrate_pointer_exp"
    run_id = generate_run_id(args.tag)

    dirs = create_run_dirs(args.output_root, experiment_name, run_id)
    base_dir = dirs["base"]
    data_dir = dirs["data"]
    figures_dir = dirs["figures"]
    logs_dir = dirs["logs"]

    log_path = logs_dir / "run.log"
    quiet = args.quiet

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        if not quiet:
            print(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log("==============================================")
    log(f"Experiment: {experiment_name}")
    log(f"Run ID:     {run_id}")
    log(f"Output dir: {base_dir}")
    log("==============================================")
    log(f"CLI args: {vars(args)}")

    # Prepare engine params dict
    engine_params: Dict[str, Any] = {
        "grid_size": args.grid_size,
        "dt": args.dt,
        "steps": args.steps,
        "n_boson": args.n_boson,
        "n_color": args.n_color,
        "n_spin": args.n_spin,
        "mass2_boson": args.mass2_boson,
        "mass2_fermion": args.mass2_fermion,
        "lambda4_boson": args.lambda4_boson,
        "lambda4_fermion": args.lambda4_fermion,
        "g_bf": args.g_bf,
        "init_boson_amp": args.init_boson_amp,
        "init_fermion_amp": args.init_fermion_amp,
        "lump_sigma_threshold": args.lump_sigma_threshold,
        "lump_min_voxels": args.lump_min_voxels,
        "snapshot_every": args.snapshot_every,
        "seed": args.seed,
    }

    metadata = {
        "framework_version": FRAMEWORK_VERSION,
        "script": f"{experiment_name}.py",
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "working_dir": str(os.getcwd()),
        "output_root": str(args.output_root),
        "seed": args.seed,
        "cli_args": vars(args),
        "engine_params": engine_params,
        "git": {
            "commit": None,
            "branch": None,
            "dirty": None,
        },
    }

    write_json(base_dir / "metadata.json", metadata)
    write_json(base_dir / "params.json", engine_params)

    # Run engine
    log("Starting substrate_engine.run_experiment(...)")
    try:
        results = substrate_engine.run_experiment(engine_params)
        error = None
        log("Engine run completed successfully.")
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
        log(f"Engine run FAILED: {error}")
        results = {
            "params": engine_params,
            "metrics": {},
            "diagnostics": {"error": error},
            "verdicts": {},
            "timeseries": {},
        }

    # Save summary.json
    summary = {
        "framework_version": FRAMEWORK_VERSION,
        "script": f"{experiment_name}.py",
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "params": results.get("params", engine_params),
        "metrics": results.get("metrics", {}),
        "diagnostics": results.get("diagnostics", {}),
        "verdicts": results.get("verdicts", {}),
    }
    write_json(base_dir / "summary.json", summary)

    # Save timeseries to NPZ
    ts = results.get("timeseries", {})
    step = np.asarray(ts.get("step", []), dtype=np.int32)
    energy = np.asarray(ts.get("energy", []), dtype=np.float64)
    norm = np.asarray(ts.get("norm", []), dtype=np.float64)
    fermion_phi_rms = np.asarray(ts.get("fermion_phi_rms", []), dtype=np.float64)
    fermion_phi_max = np.asarray(ts.get("fermion_phi_max", []), dtype=np.float64)
    fermion_lump_count = np.asarray(ts.get("fermion_lump_count", []), dtype=np.float64)
    fermion_com = np.asarray(ts.get("fermion_com", []), dtype=np.float64)

    np.savez_compressed(
        data_dir / "timeseries.npz",
        step=step,
        energy=energy,
        norm=norm,
        fermion_phi_rms=fermion_phi_rms,
        fermion_phi_max=fermion_phi_max,
        fermion_lump_count=fermion_lump_count,
        fermion_com=fermion_com,
    )

    # Save snapshots if present
    snapshots = results.get("snapshots", None)
    if snapshots is not None:
        snap_step = np.asarray(snapshots.get("step", []), dtype=np.int32)
        rho_F = np.asarray(snapshots.get("rho_F", []), dtype=np.float64)
        rho_B = np.asarray(snapshots.get("rho_B", []), dtype=np.float64)
        np.savez_compressed(
            data_dir / "snapshots.npz",
            step=snap_step,
            rho_F=rho_F,
            rho_B=rho_B,
        )

    # Quick-look figures if we have data
    if step.size > 1:
        log("Saving quick-look figures...")

        # Energy vs step
        plt.figure()
        plt.plot(step, energy)
        plt.xlabel("step")
        plt.ylabel("mean energy density")
        plt.title("Energy vs step")
        plt.tight_layout()
        plt.savefig(figures_dir / "energy_vs_step.png", dpi=150)
        plt.close()

        # Fermion lump count vs step
        if fermion_lump_count.size == step.size:
            plt.figure()
            plt.plot(step, fermion_lump_count, drawstyle="steps-post")
            plt.xlabel("step")
            plt.ylabel("fermion_lump_count")
            plt.title("Fermion lump count vs step")
            plt.tight_layout()
            plt.savefig(figures_dir / "fermion_lump_count_vs_step.png", dpi=150)
            plt.close()

        # Fermion COM_x vs step
        if fermion_com.ndim == 2 and fermion_com.shape[0] == step.size:
            plt.figure()
            plt.plot(step, fermion_com[:, 0])
            plt.xlabel("step")
            plt.ylabel("fermion COM_x (lattice units)")
            plt.title("Fermion COM_x vs step")
            plt.tight_layout()
            plt.savefig(figures_dir / "fermion_com_x_vs_step.png", dpi=150)
            plt.close()

    log("==== Run Summary ====")
    log(f"Metrics: {summary.get('metrics', {})}")
    if error is not None:
        log("Run finished with ERRORS (see diagnostics).")
    else:
        log("Run finished successfully.")
    log("=====================")


if __name__ == "__main__":
    main()

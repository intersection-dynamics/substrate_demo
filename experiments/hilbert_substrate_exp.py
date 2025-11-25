"""
experiments/hilbert_substrate_exp.py

Experiment harness for engines/hilbert_substrate_engine.py.

Runs a single Hilbert-substrate experiment and saves:
  - metadata.json
  - summary.json
  - data/timeseries.npz
  - data/snapshots.npz
  - figures/double_fraction_vs_collapse.png
  - figures/mean_occupancy_vs_collapse.png

This is intentionally lightweight. It delegates all physics to the engine.
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

# Make sure we can import the engine from the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engines import hilbert_substrate_engine  # type: ignore[import]


FRAMEWORK_VERSION = "hilbert_substrate_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hilbert substrate experiment (0/1 vs fragile 2-occupancy)."
    )

    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--tag", type=str, default="")

    # Lattice / time
    parser.add_argument("--n-sites", type=int, default=6)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=400)

    # Generator couplings
    parser.add_argument("--J-hop", type=float, default=1.0)
    parser.add_argument("--J-split", type=float, default=1.0)
    parser.add_argument("--J-merge", type=float, default=0.3)
    parser.add_argument("--U-double", type=float, default=2.0)

    # Decoherence cadence
    parser.add_argument("--decoherence-every", type=int, default=10)

    # Initial local probabilities
    parser.add_argument("--p0", type=float, default=0.6)
    parser.add_argument("--p1", type=float, default=0.3)
    parser.add_argument("--p2", type=float, default=0.1)

    # RNG
    parser.add_argument("--seed", type=int, default=1)

    # Verbosity
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output; still logs to logs/run.log.",
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
    experiment_name = Path(__file__).stem  # "hilbert_substrate_exp"
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

    # Map CLI args -> engine params
    engine_params: Dict[str, Any] = {
        "n_sites": args.n_sites,
        "dt": args.dt,
        "steps": args.steps,
        "J_hop": args.J_hop,
        "J_split": args.J_split,
        "J_merge": args.J_merge,
        "U_double": args.U_double,
        "decoherence_every": args.decoherence_every,
        "p0": args.p0,
        "p1": args.p1,
        "p2": args.p2,
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
    log("Starting hilbert_substrate_engine.run_experiment(...)")
    try:
        results = hilbert_substrate_engine.run_experiment(engine_params)
        error = None
        log("Engine run completed successfully.")
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
        log(f"Engine run FAILED: {error}")
        results = {
            "params": engine_params,
            "metrics": {},
            "timeseries": {},
            "snapshots": {},
            "diagnostics": {"error": error},
            "verdicts": {},
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

    # Save timeseries
    ts = results.get("timeseries", {})
    collapse_step = np.asarray(ts.get("collapse_step", []), dtype=np.int32)
    mean_occupancy = np.asarray(ts.get("mean_occupancy", []), dtype=np.float64)
    double_fraction = np.asarray(ts.get("double_fraction", []), dtype=np.float64)

    np.savez_compressed(
        data_dir / "timeseries.npz",
        collapse_step=collapse_step,
        mean_occupancy=mean_occupancy,
        double_fraction=double_fraction,
    )

    # Save snapshots
    snaps = results.get("snapshots", {})
    collapse_step_snap = np.asarray(snaps.get("collapse_step", []), dtype=np.int32)
    configs = snaps.get("configs", None)
    if isinstance(configs, np.ndarray):
        configs_arr = configs.astype(np.int8, copy=False)
    else:
        configs_arr = np.zeros((0, args.n_sites), dtype=np.int8)

    np.savez_compressed(
        data_dir / "snapshots.npz",
        collapse_step=collapse_step_snap,
        configs=configs_arr,
    )

    # Quick-look figures
    if collapse_step.size > 0:
        log("Saving quick-look figures...")

        # Double fraction vs collapse index
        plt.figure()
        plt.plot(collapse_step, double_fraction, marker="o")
        plt.xlabel("step")
        plt.ylabel("double_fraction")
        plt.title("Fraction of sites with n_i = 2 at each collapse")
        plt.tight_layout()
        plt.savefig(figures_dir / "double_fraction_vs_collapse.png", dpi=150)
        plt.close()

        # Mean occupancy vs collapse index
        plt.figure()
        plt.plot(collapse_step, mean_occupancy, marker="o")
        plt.xlabel("step")
        plt.ylabel("mean occupancy per site")
        plt.title("Mean occupancy at each collapse")
        plt.tight_layout()
        plt.savefig(figures_dir / "mean_occupancy_vs_collapse.png", dpi=150)
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

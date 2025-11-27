"""
experiments/chsh_substrate_exp.py

CHSH test experiment on the Hilbert substrate.

This script:
  - Parses CLI arguments (output-root, tag, seed, state/noise params),
  - Creates an output run directory:
        outputs/chsh_substrate_exp/<run_id>/
  - Writes:
        params.json
        summary.json
        metadata.json
        logs/run.log
        data/chsh_results.npz
        figures/chsh_correlators.png
        figures/chsh_S_summary.png
  - Calls engines/chsh_substrate_engine.py to compute S_CHSH etc.

Usage (from repo root):

  python experiments\\chsh_substrate_exp.py --seed 1 --tag baseline

You can also choose different Bell states or noise models:

  python experiments\\chsh_substrate_exp.py ^
      --seed 1 ^
      --state-type singlet ^
      --noise-type depolarizing ^
      --noise-strength 0.1 ^
      --tag noisy_test
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Ensure repo root is on sys.path so we can import engines
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engines import chsh_substrate_engine as engine  # noqa: E402


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CHSH test on the Hilbert substrate."
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root directory for all experiment outputs.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag to append to run_id.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (recorded for reproducibility).",
    )

    parser.add_argument(
        "--state-type",
        type=str,
        default="singlet",
        choices=["singlet", "phi_plus"],
        help="Which Bell-like state to test.",
    )
    parser.add_argument(
        "--noise-type",
        type=str,
        default="none",
        choices=["none", "depolarizing"],
        help="Noise model for the CHSH test.",
    )
    parser.add_argument(
        "--noise-strength",
        type=float,
        default=0.0,
        help="Noise strength parameter (e.g., depolarizing p in [0,1]).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------


class TeeLogger:
    """Write messages to both stdout and a log file."""

    def __init__(self, logfile_path: Path):
        self.logfile_path = logfile_path
        self.logfile = logfile_path.open("w", encoding="utf-8")

    def write(self, msg: str) -> None:
        msg_str = str(msg)
        # Console
        sys.stdout.write(msg_str)
        sys.stdout.flush()
        # File
        self.logfile.write(msg_str)
        self.logfile.flush()

    def close(self) -> None:
        self.logfile.close()


# ---------------------------------------------------------------------
# Output directory / metadata helpers
# ---------------------------------------------------------------------


def make_run_dir(output_root: str, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{tag}" if tag else ts
    base = Path(output_root) / "chsh_substrate_exp" / run_id
    (base / "data").mkdir(parents=True, exist_ok=False)
    (base / "figures").mkdir(parents=True, exist_ok=False)
    (base / "logs").mkdir(parents=True, exist_ok=False)
    return base


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def plot_correlators(correlators, labels, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(labels))
    ax.bar(x, correlators, color="steelblue")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("E(a, b)")
    ax.set_title("CHSH Correlators")

    for i, val in enumerate(correlators):
        ax.text(i, val + 0.05 * np.sign(val if val != 0 else 1),
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_S_summary(S: float, classical_bound: float, tsirelson_bound: float,
                   out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.axhline(classical_bound, color="gray", linestyle="--", label="Classical bound (2)")
    ax.axhline(tsirelson_bound, color="red", linestyle="--", label="Tsirelson bound (2√2)")

    ax.bar([0], [S], width=0.4, color="purple", label="Substrate S_CHSH")

    ax.set_xticks([0])
    ax.set_xticklabels(["S_CHSH"])
    ax.set_ylabel("S")
    ax.set_title("CHSH S-parameter vs bounds")
    ax.set_ylim(0, max(tsirelson_bound * 1.1, abs(S) * 1.1))

    ax.text(0, S + 0.05, f"{S:.3f}", ha="center", va="bottom", fontsize=10)

    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    run_dir = make_run_dir(args.output_root, args.tag)
    data_dir = run_dir / "data"
    fig_dir = run_dir / "figures"
    logs_dir = run_dir / "logs"

    # Setup logging
    logger = TeeLogger(logs_dir / "run.log")
    logger.write("==============================================\n")
    logger.write("Experiment: chsh_substrate_exp\n")
    logger.write(f"Run dir:    {run_dir}\n")
    logger.write("==============================================\n")

    # Record parameters
    engine_params: Dict[str, Any] = {
        "seed": args.seed,
        "state_type": args.state_type,
        "noise_type": args.noise_type,
        "noise_strength": float(args.noise_strength),
    }

    logger.write("Engine parameters:\n")
    for k, v in engine_params.items():
        logger.write(f"  {k}: {v}\n")
    logger.write("----------------------------------------------\n")

    # Run engine
    logger.write("Running engine...\n")
    results = engine.run_experiment(engine_params)
    logger.write("Engine run completed.\n")
    logger.write("----------------------------------------------\n")

    # Build metadata / params / summary
    framework_version = "0.1.0"
    script_name = "chsh_substrate_exp.py"
    ts_iso = datetime.now().isoformat()

    metadata = {
        "framework_version": framework_version,
        "script": script_name,
        "run_id": run_dir.name,
        "timestamp": ts_iso,
        "seed": args.seed,
    }

    params_obj = {
        "cli": {
            "output_root": args.output_root,
            "tag": args.tag,
            "seed": args.seed,
            "state_type": args.state_type,
            "noise_type": args.noise_type,
            "noise_strength": args.noise_strength,
        },
        "engine": results["params"],
    }

    summary = {
        "framework_version": framework_version,
        "script": script_name,
        "run_id": run_dir.name,
        "timestamp": ts_iso,
        "params": params_obj,
        "metrics": results["metrics"],
        "diagnostics": results["diagnostics"],
        "verdicts": results["verdicts"],
    }

    # Write JSON files
    write_json(run_dir / "params.json", params_obj)
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "summary.json", summary)

    # Save raw data arrays
    data = results["data"]
    np.savez_compressed(
        data_dir / "chsh_results.npz",
        settings_labels=np.array(data["settings_labels"], dtype=object),
        correlators=np.array(data["correlators"], dtype=float),
        rho_real=np.array(data["rho_real"], dtype=float),
        rho_imag=np.array(data["rho_imag"], dtype=float),
    )

    # Print human-readable summary to log + console
    m = results["metrics"]
    d = results["diagnostics"]
    v = results["verdicts"]

    logger.write("==== CHSH Summary ====\n")
    logger.write(f"S_CHSH:          {m['S_CHSH']:.6f}\n")
    logger.write(f"classical_bound: {m['classical_bound']:.6f}\n")
    logger.write(f"tsirelson_bound: {m['tsirelson_bound']:.6f}\n")
    logger.write(f"E(A,B):          {m['E_AB']:.6f}\n")
    logger.write(f"E(A,B'):         {m['E_ABp']:.6f}\n")
    logger.write(f"E(A',B):         {m['E_ApB']:.6f}\n")
    logger.write(f"E(A',B'):        {m['E_ApBp']:.6f}\n")
    logger.write("\n---- Diagnostics ----\n")
    logger.write(f"within_tsirelson:      {d['within_tsirelson']}\n")
    logger.write(f"warnings:              {d['warnings']}\n")
    logger.write("\n---- Verdicts ----\n")
    logger.write(f"has_bell_violation:    {v['has_bell_violation']}\n")
    logger.write(f"violates_classical_bound: {v['violates_classical_bound']}\n")
    logger.write("==============================================\n")

    # Plots
    correlators = np.array(data["correlators"], dtype=float)
    labels = list(data["settings_labels"])

    plot_correlators(correlators, labels, fig_dir / "chsh_correlators.png")
    plot_S_summary(
        m["S_CHSH"],
        m["classical_bound"],
        m["tsirelson_bound"],
        fig_dir / "chsh_S_summary.png",
    )

    logger.write(f"Outputs written under: {run_dir}\n")
    logger.close()


if __name__ == "__main__":
    main()

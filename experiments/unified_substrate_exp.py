"""
experiments/unified_substrate_exp.py

Unified Hilbert Substrate Experiment Driver

This script wraps engines/substrate_engine.py, exposing different
experiment modes via a single CLI:

  --experiment chsh
  --experiment patterns
  --experiment metric

For each run it:

  - Creates an output directory:

        outputs/unified_substrate_exp/<run_id>/

    where <run_id> encodes a timestamp and optional tag.

  - Writes:
        params.json    # CLI + engine params
        metadata.json  # script name, timestamp, run_id, seed
        summary.json   # metrics + diagnostics + verdicts

  - Writes logs:
        logs/run.log

  - Writes raw data (mode-dependent):
        data/chsh_results.npz
        or data/patterns_results.npz
        or data/metric_results.npz

  - Writes figures (mode-dependent) into:
        figures/

Usage examples (from repo root):

  # CHSH Bell test, default singlet, no noise
  python experiments\\unified_substrate_exp.py --experiment chsh --tag chsh_baseline

  # CHSH with depolarizing noise p=0.2
  python experiments\\unified_substrate_exp.py ^
      --experiment chsh ^
      --noise-type depolarizing ^
      --noise-strength 0.2 ^
      --tag chsh_noisy

  # Information-pattern copyability test
  python experiments\\unified_substrate_exp.py ^
      --experiment patterns ^
      --pattern-type antisymmetric ^
      --n-detections 12 ^
      --tag patterns_antisym

  # Emergent metric on a chain graph
  python experiments\\unified_substrate_exp.py ^
      --experiment metric ^
      --graph-type chain ^
      --n-modes 8 ^
      --tag metric_chain
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

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

from engines import substrate_engine as engine  # noqa: E402


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Hilbert substrate experiment driver."
    )

    # Common options
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["chsh", "patterns", "metric"],
        help="Which experiment to run: chsh | patterns | metric",
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
        help="Random seed (recorded + passed to engine when relevant).",
    )

    # -----------------------------
    # CHSH-specific options
    # -----------------------------
    parser.add_argument(
        "--state-type",
        type=str,
        default="singlet",
        choices=["singlet", "phi_plus"],
        help="CHSH: Bell-like state type.",
    )
    parser.add_argument(
        "--noise-type",
        type=str,
        default="none",
        choices=["none", "depolarizing"],
        help="CHSH: noise model.",
    )
    parser.add_argument(
        "--noise-strength",
        type=float,
        default=0.0,
        help="CHSH: noise strength parameter (e.g., depolarizing p in [0,1]).",
    )
    parser.add_argument(
        "--alice-qubit",
        type=int,
        default=0,
        help="CHSH: substrate qubit index used as Alice.",
    )
    parser.add_argument(
        "--bob-qubit",
        type=int,
        default=1,
        help="CHSH: substrate qubit index used as Bob.",
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=None,
        help="Substrate: total number of qubits (optional; if omitted, inferred).",
    )
    parser.add_argument(
        "--graph-type",
        type=str,
        default="chain",
        choices=["chain", "complete"],
        help="Substrate: graph topology (used for metric / propagation).",
    )
    parser.add_argument(
        "--coupling",
        type=float,
        default=1.0,
        help="Substrate: Hamiltonian coupling strength (metric diagnostics).",
    )

    # -----------------------------
    # Patterns-specific options
    # -----------------------------
    parser.add_argument(
        "--pattern-type",
        type=str,
        default="symmetric",
        choices=["local", "symmetric", "antisymmetric", "bell"],
        help="Patterns: information pattern type.",
    )
    parser.add_argument(
        "--n-sys-qubits",
        type=int,
        default=2,
        help="Patterns: number of system qubits.",
    )
    parser.add_argument(
        "--n-env-qubits",
        type=int,
        default=3,
        help="Patterns: number of environment qubits.",
    )
    parser.add_argument(
        "--n-detections",
        type=int,
        default=8,
        help="Patterns: number of repeated detection operations.",
    )
    parser.add_argument(
        "--sys-qubit-index",
        type=int,
        default=0,
        help="Patterns: index of system qubit used as CNOT control.",
    )
    parser.add_argument(
        "--fidelity-threshold",
        type=float,
        default=0.8,
        help="Patterns: threshold for final fidelity (bosonic-like vs fermionic-like).",
    )
    parser.add_argument(
        "--purity-threshold",
        type=float,
        default=0.8,
        help="Patterns: threshold for final purity (bosonic-like vs fermionic-like).",
    )

    # -----------------------------
    # Metric-specific options
    # -----------------------------
    parser.add_argument(
        "--n-modes",
        type=int,
        default=6,
        help="Metric: number of modes (graph vertices).",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=10.0,
        help="Metric: total evolution time.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=200,
        help="Metric: number of time steps.",
    )
    parser.add_argument(
        "--occupancy-threshold",
        type=float,
        default=0.1,
        help="Metric: occupancy threshold for 'arrival time'.",
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
    base = Path(output_root) / "unified_substrate_exp" / run_id
    (base / "data").mkdir(parents=True, exist_ok=False)
    (base / "figures").mkdir(parents=True, exist_ok=False)
    (base / "logs").mkdir(parents=True, exist_ok=False)
    return base


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------
# Plotters for each experiment type
# ---------------------------------------------------------------------


def plot_chsh(data: Dict[str, Any], metrics: Dict[str, Any], fig_dir: Path) -> None:
    correlators = np.array(data["correlators"], dtype=float)
    labels = list(data["settings_labels"])
    S = metrics["S_CHSH"]
    classical_bound = metrics["classical_bound"]
    tsirelson_bound = metrics["tsirelson_bound"]

    # Correlators bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x, correlators)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("E(a, b)")
    ax.set_title("CHSH Correlators")
    for i, val in enumerate(correlators):
        ax.text(
            i,
            val + 0.05 * np.sign(val if val != 0 else 1),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "chsh_correlators.png", dpi=150)
    plt.close()

    # S vs bounds
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(classical_bound, color="gray", linestyle="--", label="Classical bound (2)")
    ax.axhline(tsirelson_bound, color="red", linestyle="--", label="Tsirelson bound (2√2)")
    ax.bar([0], [S], width=0.4, label="Substrate S_CHSH")
    ax.set_xticks([0])
    ax.set_xticklabels(["S_CHSH"])
    ax.set_ylabel("S")
    ax.set_title("CHSH S-parameter vs bounds")
    ax.set_ylim(0, max(tsirelson_bound * 1.1, abs(S) * 1.1))
    ax.text(0, S + 0.05, f"{S:.3f}", ha="center", va="bottom", fontsize=10)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "chsh_S_summary.png", dpi=150)
    plt.close()


def plot_patterns(data: Dict[str, Any], fig_dir: Path) -> None:
    purity = np.array(data["purity_trace"], dtype=float)
    fidelity = np.array(data["fidelity_trace"], dtype=float)
    steps = np.array(data["steps"], dtype=int)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, purity, marker="o", label="Purity")
    ax.plot(steps, fidelity, marker="s", label="Fidelity to initial")
    ax.set_xlabel("Detection step")
    ax.set_ylabel("Value")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Pattern dynamics ({data.get('pattern_type', 'unknown')})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "patterns_purity_fidelity.png", dpi=150)
    plt.close()

    # Also visualize initial ρ_sys magnitude as a heatmap
    rho0 = np.array(data["rho_sys_initial"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(np.abs(rho0), cmap="viridis", interpolation="nearest")
    ax.set_title("|ρ_sys_initial|")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(fig_dir / "patterns_rho_initial.png", dpi=150)
    plt.close()


def plot_metric(data: Dict[str, Any], fig_dir: Path) -> None:
    D = np.array(data["distance_matrix"], dtype=float)
    A = np.array(data["adjacency"], dtype=float)

    # Distance matrix heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(D, cmap="viridis", interpolation="nearest")
    ax.set_title("Emergent distance matrix")
    ax.set_xlabel("Mode j")
    ax.set_ylabel("Mode i")
    plt.colorbar(im, ax=ax, label="Arrival time")
    plt.tight_layout()
    plt.savefig(fig_dir / "metric_distance_matrix.png", dpi=150)
    plt.close()

    # Adjacency heatmap
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(A, cmap="Greys", interpolation="nearest")
    ax.set_title("Graph adjacency")
    ax.set_xlabel("Mode j")
    ax.set_ylabel("Mode i")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(fig_dir / "metric_adjacency.png", dpi=150)
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

    logger = TeeLogger(logs_dir / "run.log")
    logger.write("==============================================\n")
    logger.write("Experiment: unified_substrate_exp\n")
    logger.write(f"Run dir:    {run_dir}\n")
    logger.write(f"Experiment mode: {args.experiment}\n")
    logger.write("==============================================\n")

    # Build engine params based on experiment type
    engine_params: Dict[str, Any] = {
        "experiment": args.experiment,
        "seed": args.seed,
        "graph_type": args.graph_type,
        "coupling": float(args.coupling),
    }

    # N selection:
    # - if user gives --n-qubits, we pass it through
    # - otherwise:
    #   - for chsh: inferred from alice/bob indices
    #   - for patterns: n_sys + n_env
    #   - for metric: n_modes
    if args.n_qubits is not None:
        engine_params["n_qubits"] = args.n_qubits
    else:
        if args.experiment == "metric":
            engine_params["n_modes"] = args.n_modes
        elif args.experiment == "patterns":
            engine_params["n_sys_qubits"] = args.n_sys_qubits
            engine_params["n_env_qubits"] = args.n_env_qubits
        elif args.experiment == "chsh":
            engine_params["alice_qubit"] = args.alice_qubit
            engine_params["bob_qubit"] = args.bob_qubit

    if args.experiment == "chsh":
        engine_params.update(
            {
                "state_type": args.state_type,
                "noise_type": args.noise_type,
                "noise_strength": float(args.noise_strength),
                "alice_qubit": args.alice_qubit,
                "bob_qubit": args.bob_qubit,
            }
        )
    elif args.experiment == "patterns":
        engine_params.update(
            {
                "pattern_type": args.pattern_type,
                "n_sys_qubits": args.n_sys_qubits,
                "n_env_qubits": args.n_env_qubits,
                "n_detections": args.n_detections,
                "sys_qubit_index": args.sys_qubit_index,
                "fidelity_threshold": float(args.fidelity_threshold),
                "purity_threshold": float(args.purity_threshold),
            }
        )
    elif args.experiment == "metric":
        engine_params.update(
            {
                "n_modes": args.n_modes,
                "t_max": float(args.t_max),
                "n_steps": args.n_steps,
                "occupancy_threshold": float(args.occupancy_threshold),
            }
        )
    else:
        logger.write(f"Unknown experiment mode: {args.experiment}\n")
        logger.close()
        raise SystemExit(1)

    logger.write("Engine parameters:\n")
    for k, v in engine_params.items():
        logger.write(f"  {k}: {v}\n")
    logger.write("----------------------------------------------\n")

    # Run engine
    logger.write("Running engine...\n")
    results = engine.run_experiment(engine_params)
    logger.write("Engine run completed.\n")
    logger.write("----------------------------------------------\n")

    # Metadata / params / summary
    framework_version = "0.2.0"
    script_name = "unified_substrate_exp.py"
    ts_iso = datetime.now().isoformat()

    metadata = {
        "framework_version": framework_version,
        "script": script_name,
        "run_id": run_dir.name,
        "timestamp": ts_iso,
        "seed": args.seed,
        "experiment": args.experiment,
    }

    params_obj = {
        "cli": {
            "experiment": args.experiment,
            "output_root": args.output_root,
            "tag": args.tag,
            "seed": args.seed,
            "state_type": args.state_type,
            "noise_type": args.noise_type,
            "noise_strength": args.noise_strength,
            "alice_qubit": args.alice_qubit,
            "bob_qubit": args.bob_qubit,
            "n_qubits": args.n_qubits,
            "graph_type": args.graph_type,
            "coupling": args.coupling,
            "pattern_type": args.pattern_type,
            "n_sys_qubits": args.n_sys_qubits,
            "n_env_qubits": args.n_env_qubits,
            "n_detections": args.n_detections,
            "sys_qubit_index": args.sys_qubit_index,
            "fidelity_threshold": args.fidelity_threshold,
            "purity_threshold": args.purity_threshold,
            "n_modes": args.n_modes,
            "t_max": args.t_max,
            "n_steps": args.n_steps,
            "occupancy_threshold": args.occupancy_threshold,
        },
        "engine": results["params"],
    }

    summary = {
        "framework_version": framework_version,
        "script": script_name,
        "run_id": run_dir.name,
        "timestamp": ts_iso,
        "experiment": args.experiment,
        "params": params_obj,
        "metrics": results["metrics"],
        "diagnostics": results["diagnostics"],
        "verdicts": results["verdicts"],
    }

    write_json(run_dir / "params.json", params_obj)
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "summary.json", summary)

    # Save raw data
    data = results["data"]
    if args.experiment == "chsh":
        # Backward/forward compatible: handle either "rho_real" or "rho_pair_real"
        rho_real = data.get("rho_real", data.get("rho_pair_real"))
        rho_imag = data.get("rho_imag", data.get("rho_pair_imag"))

        np.savez_compressed(
            data_dir / "chsh_results.npz",
            settings_labels=np.array(data["settings_labels"], dtype=object),
            correlators=np.array(data["correlators"], dtype=float),
            rho_real=np.array(rho_real, dtype=float),
            rho_imag=np.array(rho_imag, dtype=float),
        )
    elif args.experiment == "patterns":
        np.savez_compressed(
            data_dir / "patterns_results.npz",
            purity_trace=np.array(data["purity_trace"], dtype=float),
            fidelity_trace=np.array(data["fidelity_trace"], dtype=float),
            steps=np.array(data["steps"], dtype=int),
            rho_sys_initial=np.array(data["rho_sys_initial"], dtype=complex),
            pattern_type=np.array(data["pattern_type"]),
        )
    elif args.experiment == "metric":
        np.savez_compressed(
            data_dir / "metric_results.npz",
            distance_matrix=np.array(data["distance_matrix"], dtype=float),
            adjacency=np.array(data["adjacency"], dtype=float),
        )

    # Human-readable summary
    m = results["metrics"]
    d = results["diagnostics"]
    v = results["verdicts"]

    logger.write("==== Summary ====\n")
    logger.write(f"Experiment mode:    {args.experiment}\n")

    # Print some mode-dependent key metrics
    if args.experiment == "chsh":
        logger.write(f"S_CHSH:             {m['S_CHSH']:.6f}\n")
        logger.write(f"classical_bound:    {m['classical_bound']:.6f}\n")
        logger.write(f"tsirelson_bound:    {m['tsirelson_bound']:.6f}\n")
        logger.write(f"E(A,B):             {m['E_AB']:.6f}\n")
        logger.write(f"E(A,B'):            {m['E_ABp']:.6f}\n")
        logger.write(f"E(A',B):            {m['E_ApB']:.6f}\n")
        logger.write(f"E(A',B'):           {m['E_ApBp']:.6f}\n")

    elif args.experiment == "patterns":
        logger.write(f"initial_purity:     {m['initial_purity']:.6f}\n")
        logger.write(f"final_purity:       {m['final_purity']:.6f}\n")
        logger.write(f"initial_fidelity:   {m['initial_fidelity']:.6f}\n")
        logger.write(f"final_fidelity:     {m['final_fidelity']:.6f}\n")

    elif args.experiment == "metric":
        logger.write(f"n_modes:            {m['n_modes']}\n")
        logger.write(f"violation_rate:     {m['violation_rate']:.6f}\n")

    logger.write("\n---- Diagnostics ----\n")
    for key, val in d.items():
        logger.write(f"{key}: {val}\n")

    logger.write("\n---- Verdicts ----\n")
    for key, val in v.items():
        logger.write(f"{key}: {val}\n")

    logger.write("==============================================\n")

    # Figures
    if args.experiment == "chsh":
        plot_chsh(data, m, fig_dir)
    elif args.experiment == "patterns":
        plot_patterns(data, fig_dir)
    elif args.experiment == "metric":
        plot_metric(data, fig_dir)

    logger.write(f"Outputs written under: {run_dir}\n")
    logger.close()


if __name__ == "__main__":
    main()

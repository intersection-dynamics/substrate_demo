"""
experiments/basin_stability_exp.py

Basin-of-stability scanner for the unified Hilbert substrate engine.

This script probes how robust an experiment's "verdict" is under
random parameter perturbations around a chosen baseline.

It wraps engines/substrate_engine.py (the unified engine) and supports
all three experiment modes:

    --experiment chsh
    --experiment patterns
    --experiment metric

For each run:

  1. Run a baseline experiment with the given CLI parameters.
  2. Sample N random perturbations of selected parameters within
     ±variation_frac (e.g. ±25%).
  3. For each perturbed parameter set, call engine.run_experiment(...)
     and check whether the verdict matches the baseline verdict.
  4. Report the fraction of perturbed runs that keep the same verdict.
     This is a crude estimate of the "basin of stability" around the
     chosen point in parameter space.

Outputs (per run):

    outputs/basin_stability_exp/<run_id>/
      params.json
      metadata.json
      summary.json
      logs/run.log
      data/basin_results.npz
      figures/basin_plot_*.png

Usage examples (from repo root):

  # CHSH basin around singlet with depolarizing noise p=0.1
  python experiments\\basin_stability_exp.py ^
      --experiment chsh ^
      --noise-type depolarizing ^
      --noise-strength 0.1 ^
      --variation-frac 0.25 ^
      --n-samples 50 ^
      --tag chsh_basin

  # Patterns basin around symmetric pattern
  python experiments\\basin_stability_exp.py ^
      --experiment patterns ^
      --pattern-type symmetric ^
      --n-detections 8 ^
      --variation-frac 0.25 ^
      --n-samples 50 ^
      --tag patterns_basin

  # Metric basin around a chain graph
  python experiments\\basin_stability_exp.py ^
      --experiment metric ^
      --graph-type chain ^
      --n-modes 8 ^
      --variation-frac 0.25 ^
      --n-samples 50 ^
      --tag metric_basin
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List

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
        description="Basin-of-stability scanner for unified substrate engine."
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
        help="Random seed for perturbation sampling (passed as base seed too).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of random perturbation samples to probe.",
    )
    parser.add_argument(
        "--variation-frac",
        type=float,
        default=0.25,
        help="Fractional variation ±f for continuous parameters (e.g. 0.25 = ±25%).",
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
        help="Patterns: threshold for final fidelity.",
    )
    parser.add_argument(
        "--purity-threshold",
        type=float,
        default=0.8,
        help="Patterns: threshold for final purity.",
    )

    # -----------------------------
    # Metric-specific options
    # -----------------------------
    parser.add_argument(
        "--graph-type",
        type=str,
        default="chain",
        choices=["chain", "complete"],
        help="Metric: graph topology.",
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        default=6,
        help="Metric: number of modes (graph vertices).",
    )
    parser.add_argument(
        "--coupling",
        type=float,
        default=1.0,
        help="Metric: Hamiltonian coupling strength.",
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
# Output / JSON helpers
# ---------------------------------------------------------------------


def make_run_dir(output_root: str, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{tag}" if tag else ts
    base = Path(output_root) / "basin_stability_exp" / run_id
    (base / "data").mkdir(parents=True, exist_ok=False)
    (base / "figures").mkdir(parents=True, exist_ok=False)
    (base / "logs").mkdir(parents=True, exist_ok=False)
    return base


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------
# Parameter perturbation helpers
# ---------------------------------------------------------------------


def _perturb_float(base: float, frac: float, rng: np.random.Generator,
                   low_clip: float | None = None,
                   high_clip: float | None = None) -> float:
    """
    Perturb a float parameter around 'base' by ±frac in a multiplicative way.

        new = base * (1 + u * frac),  u ~ Uniform[-1, 1].

    If base == 0, fall back to:
        new = u * frac

    Optionally clip to [low_clip, high_clip].
    """
    u = rng.uniform(-1.0, 1.0)
    if base == 0.0:
        new = u * frac
    else:
        new = base * (1.0 + u * frac)
    if low_clip is not None:
        new = max(low_clip, new)
    if high_clip is not None:
        new = min(high_clip, new)
    return float(new)


def _perturb_int(base: int, frac: float, rng: np.random.Generator,
                 min_val: int = 1) -> int:
    """
    Perturb an integer parameter by an amount proportional to base*frac,
    but at least ±1 where possible.
    """
    if base <= 0:
        return max(min_val, int(round(rng.integers(min_val, min_val + 3))))
    max_step = max(1, int(round(abs(base) * frac)))
    step = int(rng.integers(-max_step, max_step + 1))
    new = base + step
    return max(min_val, new)


# ---------------------------------------------------------------------
# Stability comparison helpers
# ---------------------------------------------------------------------


def _extract_verdict_key(experiment: str, results: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    Extract the minimal "verdict signature" we want to preserve when
    probing the basin of stability.

    For each experiment we choose:
      - chsh:     (has_bell_violation, within_tsirelson)
      - patterns: (classification,)
      - metric:   (approximately_metric,)
    """
    v = results["verdicts"]
    d = results["diagnostics"]

    if experiment == "chsh":
        return (bool(v["has_bell_violation"]), bool(d["within_tsirelson"]))
    elif experiment == "patterns":
        return (str(v["classification"]),)
    elif experiment == "metric":
        return (bool(v["approximately_metric"]),)
    else:
        raise ValueError(f"Unknown experiment '{experiment}' in verdict key.")


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------


def plot_chsh_basin(noise_samples: np.ndarray,
                    S_samples: np.ndarray,
                    stable_flags: np.ndarray,
                    m_base: Dict[str, Any],
                    fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    # Color-stable vs unstable differently
    stable = stable_flags.astype(bool)
    ax.scatter(
        noise_samples[stable],
        S_samples[stable],
        marker="o",
        label="stable verdict",
    )
    ax.scatter(
        noise_samples[~stable],
        S_samples[~stable],
        marker="x",
        label="verdict changed",
    )

    ax.axhline(m_base["classical_bound"], color="gray", linestyle="--",
               label="Classical bound (2)")
    ax.axhline(m_base["tsirelson_bound"], color="red", linestyle="--",
               label="Tsirelson bound (2√2)")

    ax.set_xlabel("noise_strength")
    ax.set_ylabel("S_CHSH")
    ax.set_title("CHSH basin of stability (noise-strength vs S)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "basin_chsh.png", dpi=150)
    plt.close()


def plot_patterns_basin(final_fidelity: np.ndarray,
                        final_purity: np.ndarray,
                        stable_flags: np.ndarray,
                        m_base: Dict[str, Any],
                        fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    stable = stable_flags.astype(bool)
    ax.scatter(
        final_fidelity[stable],
        final_purity[stable],
        marker="o",
        label="stable verdict",
    )
    ax.scatter(
        final_fidelity[~stable],
        final_purity[~stable],
        marker="x",
        label="verdict changed",
    )

    ax.axvline(m_base["fidelity_threshold"], color="gray", linestyle="--",
               label="fidelity_threshold")
    ax.axhline(m_base["purity_threshold"], color="red", linestyle="--",
               label="purity_threshold")

    ax.set_xlabel("final_fidelity")
    ax.set_ylabel("final_purity")
    ax.set_title("Patterns basin of stability")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "basin_patterns.png", dpi=150)
    plt.close()


def plot_metric_basin(occ_samples: np.ndarray,
                      viol_samples: np.ndarray,
                      stable_flags: np.ndarray,
                      fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    stable = stable_flags.astype(bool)
    ax.scatter(
        occ_samples[stable],
        viol_samples[stable],
        marker="o",
        label="stable verdict",
    )
    ax.scatter(
        occ_samples[~stable],
        viol_samples[~stable],
        marker="x",
        label="verdict changed",
    )

    ax.set_xlabel("occupancy_threshold")
    ax.set_ylabel("violation_rate")
    ax.set_title("Metric basin of stability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "basin_metric.png", dpi=150)
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
    logger.write("Experiment: basin_stability_exp\n")
    logger.write(f"Run dir:        {run_dir}\n")
    logger.write(f"Experiment mode: {args.experiment}\n")
    logger.write("==============================================\n")

    rng = np.random.default_rng(args.seed)

    # Build baseline engine params
    engine_params_base: Dict[str, Any] = {
        "experiment": args.experiment,
        "seed": args.seed,
    }

    if args.experiment == "chsh":
        engine_params_base.update(
            {
                "state_type": args.state_type,
                "noise_type": args.noise_type,
                "noise_strength": float(args.noise_strength),
            }
        )
    elif args.experiment == "patterns":
        engine_params_base.update(
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
        engine_params_base.update(
            {
                "graph_type": args.graph_type,
                "n_modes": args.n_modes,
                "coupling": float(args.coupling),
                "t_max": float(args.t_max),
                "n_steps": args.n_steps,
                "occupancy_threshold": float(args.occupancy_threshold),
            }
        )

    logger.write("Baseline engine parameters:\n")
    for k, v in engine_params_base.items():
        logger.write(f"  {k}: {v}\n")
    logger.write("----------------------------------------------\n")

    # Baseline run
    logger.write("Running baseline experiment...\n")
    baseline_results = engine.run_experiment(engine_params_base)
    logger.write("Baseline run completed.\n")
    logger.write("----------------------------------------------\n")

    base_verdict_key = _extract_verdict_key(args.experiment, baseline_results)
    m_base = baseline_results["metrics"]
    d_base = baseline_results["diagnostics"]
    v_base = baseline_results["verdicts"]

    logger.write("Baseline metrics (key summary):\n")
    if args.experiment == "chsh":
        logger.write(f"S_CHSH:          {m_base['S_CHSH']:.6f}\n")
        logger.write(f"has_bell_violation: {v_base['has_bell_violation']}\n")
        logger.write(f"within_tsirelson:   {d_base['within_tsirelson']}\n")
    elif args.experiment == "patterns":
        logger.write(f"final_purity:    {m_base['final_purity']:.6f}\n")
        logger.write(f"final_fidelity:  {m_base['final_fidelity']:.6f}\n")
        logger.write(f"classification:  {v_base['classification']}\n")
    elif args.experiment == "metric":
        logger.write(f"violation_rate:  {m_base['violation_rate']:.6f}\n")
        logger.write(f"approximately_metric: {v_base['approximately_metric']}\n")
    logger.write("----------------------------------------------\n")

    # Perturbation loop
    n_samples = args.n_samples
    variation_frac = args.variation_frac

    logger.write(f"Sampling {n_samples} perturbed parameter sets...\n")

    stable_count = 0

    # Storage for plots
    chsh_noise_samples: List[float] = []
    chsh_S_samples: List[float] = []
    chsh_stable_flags: List[bool] = []

    patterns_final_purity: List[float] = []
    patterns_final_fidelity: List[float] = []
    patterns_stable_flags: List[bool] = []

    metric_occ_samples: List[float] = []
    metric_violation_samples: List[float] = []
    metric_stable_flags: List[bool] = []

    for i in range(n_samples):
        params_pert = dict(engine_params_base)  # shallow copy

        if args.experiment == "chsh":
            # Perturb noise_strength only (bounded in [0,1])
            base_noise = engine_params_base["noise_strength"]
            new_noise = _perturb_float(base_noise, variation_frac, rng,
                                       low_clip=0.0, high_clip=1.0)
            params_pert["noise_strength"] = new_noise

        elif args.experiment == "patterns":
            # Perturb thresholds; keep structure (n_sys, n_env, etc.) fixed
            base_fid = engine_params_base["fidelity_threshold"]
            base_pur = engine_params_base["purity_threshold"]
            params_pert["fidelity_threshold"] = _perturb_float(
                base_fid, variation_frac, rng, low_clip=0.0, high_clip=1.0
            )
            params_pert["purity_threshold"] = _perturb_float(
                base_pur, variation_frac, rng, low_clip=0.0, high_clip=1.0
            )

        elif args.experiment == "metric":
            # Perturb coupling, t_max, and occupancy_threshold
            base_coup = engine_params_base["coupling"]
            base_tmax = engine_params_base["t_max"]
            base_occ = engine_params_base["occupancy_threshold"]

            params_pert["coupling"] = _perturb_float(
                base_coup, variation_frac, rng, low_clip=1e-6
            )
            params_pert["t_max"] = _perturb_float(
                base_tmax, variation_frac, rng, low_clip=1e-6
            )
            params_pert["occupancy_threshold"] = _perturb_float(
                base_occ, variation_frac, rng, low_clip=1e-6, high_clip=1.0
            )

        # Run perturbed experiment
        results_pert = engine.run_experiment(params_pert)
        verdict_key_pert = _extract_verdict_key(args.experiment, results_pert)
        is_stable = (verdict_key_pert == base_verdict_key)
        stable_count += int(is_stable)

        # Collect per-experiment data
        if args.experiment == "chsh":
            chsh_noise_samples.append(params_pert["noise_strength"])
            chsh_S_samples.append(results_pert["metrics"]["S_CHSH"])
            chsh_stable_flags.append(is_stable)
        elif args.experiment == "patterns":
            patterns_final_purity.append(results_pert["metrics"]["final_purity"])
            patterns_final_fidelity.append(results_pert["metrics"]["final_fidelity"])
            patterns_stable_flags.append(is_stable)
        elif args.experiment == "metric":
            metric_occ_samples.append(params_pert["occupancy_threshold"])
            metric_violation_samples.append(results_pert["metrics"]["violation_rate"])
            metric_stable_flags.append(is_stable)

    stability_fraction = stable_count / max(1, n_samples)

    logger.write("----------------------------------------------\n")
    logger.write("Basin-of-stability estimate:\n")
    logger.write(f"  stable_count:       {stable_count} / {n_samples}\n")
    logger.write(f"  stability_fraction: {stability_fraction:.3f}\n")
    logger.write("==============================================\n")

    # Metadata / params / summary
    framework_version = "0.1.0"
    script_name = "basin_stability_exp.py"
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
            "n_samples": args.n_samples,
            "variation_frac": args.variation_frac,
            "state_type": args.state_type,
            "noise_type": args.noise_type,
            "noise_strength": args.noise_strength,
            "pattern_type": args.pattern_type,
            "n_sys_qubits": args.n_sys_qubits,
            "n_env_qubits": args.n_env_qubits,
            "n_detections": args.n_detections,
            "sys_qubit_index": args.sys_qubit_index,
            "fidelity_threshold": args.fidelity_threshold,
            "purity_threshold": args.purity_threshold,
            "graph_type": args.graph_type,
            "n_modes": args.n_modes,
            "coupling": args.coupling,
            "t_max": args.t_max,
            "n_steps": args.n_steps,
            "occupancy_threshold": args.occupancy_threshold,
        },
        "engine_baseline": baseline_results["params"],
    }

    summary = {
        "framework_version": framework_version,
        "script": script_name,
        "run_id": run_dir.name,
        "timestamp": ts_iso,
        "experiment": args.experiment,
        "params": params_obj,
        "baseline_metrics": baseline_results["metrics"],
        "baseline_diagnostics": baseline_results["diagnostics"],
        "baseline_verdicts": baseline_results["verdicts"],
        "stability": {
            "n_samples": n_samples,
            "stable_count": stable_count,
            "stability_fraction": stability_fraction,
        },
    }

    write_json(run_dir / "params.json", params_obj)
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "summary.json", summary)

    # Save raw data
    if args.experiment == "chsh":
        np.savez_compressed(
            data_dir / "basin_results.npz",
            noise_strength=np.array(chsh_noise_samples, dtype=float),
            S_CHSH=np.array(chsh_S_samples, dtype=float),
            stable=np.array(chsh_stable_flags, dtype=bool),
        )
    elif args.experiment == "patterns":
        np.savez_compressed(
            data_dir / "basin_results.npz",
            final_purity=np.array(patterns_final_purity, dtype=float),
            final_fidelity=np.array(patterns_final_fidelity, dtype=float),
            stable=np.array(patterns_stable_flags, dtype=bool),
        )
    elif args.experiment == "metric":
        np.savez_compressed(
            data_dir / "basin_results.npz",
            occupancy_threshold=np.array(metric_occ_samples, dtype=float),
            violation_rate=np.array(metric_violation_samples, dtype=float),
            stable=np.array(metric_stable_flags, dtype=bool),
        )

    # Plot
    if args.experiment == "chsh":
        plot_chsh_basin(
            np.array(chsh_noise_samples, dtype=float),
            np.array(chsh_S_samples, dtype=float),
            np.array(chsh_stable_flags, dtype=bool),
            m_base,
            fig_dir,
        )
    elif args.experiment == "patterns":
        plot_patterns_basin(
            np.array(patterns_final_fidelity, dtype=float),
            np.array(patterns_final_purity, dtype=float),
            np.array(patterns_stable_flags, dtype=bool),
            m_base,
            fig_dir,
        )
    elif args.experiment == "metric":
        plot_metric_basin(
            np.array(metric_occ_samples, dtype=float),
            np.array(metric_violation_samples, dtype=float),
            np.array(metric_stable_flags, dtype=bool),
            fig_dir,
        )

    logger.write(f"Outputs written under: {run_dir}\n")
    logger.close()


if __name__ == "__main__":
    main()

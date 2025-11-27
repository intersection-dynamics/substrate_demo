"""
experiments/lightcone_exp.py

Lieb–Robinson style light-cone diagnostic on a single-excitation substrate.

We treat an N-mode graph with adjacency A_ij and Hamiltonian

    H = g * sum_{i != j, A_ij != 0} |i><j|

in the single-excitation sector.

For each source site i, we:
  - Localize an excitation at i at t=0.
  - Evolve in time up to t_max in n_steps with exact diagonalization.
  - Record occupancy p_i->j(t) for all j.

We then:
  - Compute graph distances d_graph(i,j) as unweighted shortest-path length.
  - Define arrival times τ_ij as the earliest t where p_i->j(t) >= occupancy_threshold.
  - Compute effective velocities v_ij = d_graph(i,j) / τ_ij for all pairs
    with d_graph(i,j) > 0 and τ_ij < t_max.

We summarize:
  - v_min, v_max, v_mean over all valid (i,j),
  - number of pairs used,
  - distance matrix and arrival-time matrix,
  - velocity matrix (with NaN where undefined),
  - and write:
        outputs/lightcone_exp/<run_id>/
            metadata.json
            summary.json
            data/lightcone_results.npz
            figures/distances.png
            figures/velocities.png
            figures/arrival_vs_distance.png
            logs/run.log

Usage example (Windows, from repo root):

  python experiments\\lightcone_exp.py ^
      --graph-type chain ^
      --n-modes 6 ^
      --t-max 10.0 ^
      --n-steps 200 ^
      --occupancy-threshold 0.1 ^
      --coupling 1.0 ^
      --tag lr_chain_6
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Graph + Hamiltonian helpers
# ---------------------------------------------------------------------


def build_graph_adjacency(n_modes: int, graph_type: str = "chain") -> np.ndarray:
    """
    Build a simple unweighted adjacency matrix for n_modes sites.

    graph_type:
      - "chain"    -> 1D chain with nearest-neighbor edges
      - "complete" -> fully connected graph
    """
    A = np.zeros((n_modes, n_modes), dtype=float)
    gt = graph_type.lower()

    if gt == "chain":
        for i in range(n_modes - 1):
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0
    elif gt == "complete":
        A = np.ones((n_modes, n_modes), dtype=float) - np.eye(n_modes, dtype=float)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    return A


def build_single_excitation_hamiltonian(
    n_modes: int,
    adjacency: np.ndarray,
    coupling: float,
) -> np.ndarray:
    """
    Build a single-excitation Hamiltonian:

        H = g * sum_{i != j, A_ij != 0} |i><j|

    This is an adjacency-based hopping Hamiltonian on N modes.
    """
    g = float(coupling)
    H = np.zeros((n_modes, n_modes), dtype=complex)
    for i in range(n_modes):
        for j in range(n_modes):
            if i != j and adjacency[i, j] != 0.0:
                H[i, j] += g
    return H


def all_pairs_shortest_paths(adjacency: np.ndarray) -> np.ndarray:
    """
    Compute unweighted all-pairs shortest-path distances using a Floyd–Warshall
    style dynamic programming.

    Returns dist[i,j] = minimum number of edges from i to j (float, inf if no path).
    """
    n = adjacency.shape[0]
    inf = 1e9
    dist = np.full((n, n), inf, dtype=float)

    # Initialize
    for i in range(n):
        dist[i, i] = 0.0
        for j in range(n):
            if adjacency[i, j] != 0.0 and i != j:
                dist[i, j] = 1.0

    # Floyd–Warshall
    for k in range(n):
        for i in range(n):
            dik = dist[i, k]
            if dik == inf:
                continue
            for j in range(n):
                alt = dik + dist[k, j]
                if alt < dist[i, j]:
                    dist[i, j] = alt

    return dist


# ---------------------------------------------------------------------
# Time propagation in single-excitation sector
# ---------------------------------------------------------------------


def propagate_single_source(
    H: np.ndarray,
    source: int,
    t_max: float,
    n_steps: int,
) -> np.ndarray:
    """
    Propagate a single excitation initially localized at 'source' in
    the single-excitation sector described by Hamiltonian H.

    Returns:
      occupancy[t_idx, mode] with t_idx in [0..n_steps] (including t=0).
    """
    n = H.shape[0]
    evals, evecs = np.linalg.eigh(H)
    evecs_dag = evecs.conj().T

    def U_dt(dt: float) -> np.ndarray:
        phase = np.exp(-1j * evals * dt)
        return evecs @ np.diag(phase) @ evecs_dag

    psi = np.zeros(n, dtype=complex)
    psi[source] = 1.0

    times = np.linspace(0.0, t_max, n_steps + 1)
    occupancy = np.zeros((n_steps + 1, n), dtype=float)
    occupancy[0] = np.abs(psi) ** 2

    for k in range(1, n_steps + 1):
        dt = times[k] - times[k - 1]
        U = U_dt(dt)
        psi = U @ psi
        occupancy[k] = np.abs(psi) ** 2

    return occupancy


def compute_arrival_times(
    occupancy: np.ndarray,
    t_max: float,
    threshold: float,
) -> np.ndarray:
    """
    Given occupancy[t, mode], return an array arrival[mode] of earliest
    times when occupancy >= threshold (or t_max if never reaches).
    """
    n_steps = occupancy.shape[0] - 1
    n_modes = occupancy.shape[1]
    times = np.linspace(0.0, t_max, n_steps + 1)
    arrivals = np.full(n_modes, t_max, dtype=float)

    for mode in range(n_modes):
        above = np.where(occupancy[:, mode] >= threshold)[0]
        if above.size > 0:
            arrivals[mode] = float(times[above[0]])

    return arrivals


# ---------------------------------------------------------------------
# Logging + JSON helpers
# ---------------------------------------------------------------------


class TeeLogger:
    """Write messages to both stdout and a log file."""

    def __init__(self, logfile_path: Path):
        self.logfile_path = logfile_path
        self.logfile = logfile_path.open("w", encoding="utf-8")

    def write(self, msg: str) -> None:
        msg_str = str(msg)
        sys.stdout.write(msg_str)
        sys.stdout.flush()
        self.logfile.write(msg_str)
        self.logfile.flush()

    def close(self) -> None:
        self.logfile.close()


def make_run_dir(output_root: str, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{tag}" if tag else ts
    base = Path(output_root) / "lightcone_exp" / run_id
    (base / "data").mkdir(parents=True, exist_ok=False)
    (base / "figures").mkdir(parents=True, exist_ok=False)
    (base / "logs").mkdir(parents=True, exist_ok=False)
    return base


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------


def plot_distance_matrix(D: np.ndarray, fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(D, cmap="viridis", interpolation="nearest")
    ax.set_title("Graph distance matrix")
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("d_graph(i,j)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def plot_velocity_matrix(V: np.ndarray, fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(V, cmap="magma", interpolation="nearest")
    ax.set_title("Effective velocity v_ij = d_ij / τ_ij")
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("v_ij (hops / time)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def plot_arrival_vs_distance(
    D: np.ndarray,
    T: np.ndarray,
    t_max: float,
    fig_path: Path,
) -> None:
    """
    Scatter plot of arrival times τ_ij vs graph distances d_ij
    for all pairs with finite distance and arrival < t_max.
    """
    n = D.shape[0]
    d_list = []
    t_list = []

    for i in range(n):
        for j in range(n):
            d = D[i, j]
            tau = T[i, j]
            if i == j:
                continue
            if d >= 1e8:
                continue  # disconnected
            if tau >= t_max:
                continue  # never reached threshold
            d_list.append(d)
            t_list.append(tau)

    if not d_list:
        return

    d_arr = np.array(d_list, dtype=float)
    t_arr = np.array(t_list, dtype=float)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(d_arr, t_arr, s=15, alpha=0.7)
    ax.set_xlabel("graph distance d_ij (hops)")
    ax.set_ylabel("arrival time τ_ij")
    ax.set_title("Arrival times vs graph distance")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lieb–Robinson style light-cone diagnostic in single-excitation sector."
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag for this run.",
    )
    parser.add_argument(
        "--graph-type",
        type=str,
        default="chain",
        choices=["chain", "complete"],
        help="Graph topology for the substrate modes.",
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        default=6,
        help="Number of modes/sites (single-excitation dimension).",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=10.0,
        help="Total evolution time.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=200,
        help="Number of time steps (resolution).",
    )
    parser.add_argument(
        "--occupancy-threshold",
        type=float,
        default=0.1,
        help="Threshold for defining 'arrival' at a site.",
    )
    parser.add_argument(
        "--coupling",
        type=float,
        default=1.0,
        help="Hopping strength g in the single-excitation Hamiltonian.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed (currently unused, reserved for future randomness).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = make_run_dir(args.output_root, args.tag)
    data_dir = run_dir / "data"
    fig_dir = run_dir / "figures"
    logs_dir = run_dir / "logs"

    logger = TeeLogger(logs_dir / "run.log")
    logger.write("==============================================\n")
    logger.write("Experiment: lightcone_exp\n")
    logger.write(f"Run dir:    {run_dir}\n")
    logger.write("==============================================\n")

    logger.write("Parameters:\n")
    logger.write(f"  graph_type:          {args.graph_type}\n")
    logger.write(f"  n_modes:             {args.n_modes}\n")
    logger.write(f"  t_max:               {args.t_max}\n")
    logger.write(f"  n_steps:             {args.n_steps}\n")
    logger.write(f"  occupancy_threshold: {args.occupancy_threshold}\n")
    logger.write(f"  coupling:            {args.coupling}\n")
    logger.write(f"  seed:                {args.seed}\n")
    logger.write("----------------------------------------------\n")

    # Build graph + Hamiltonian
    n = args.n_modes
    A = build_graph_adjacency(n, args.graph_type)
    H = build_single_excitation_hamiltonian(n, A, args.coupling)
    D_graph = all_pairs_shortest_paths(A)

    # Propagate from each source and collect arrival times
    arrival_times = np.zeros((n, n), dtype=float)

    logger.write("Computing arrival times from each source...\n")
    for i in range(n):
        occupancy = propagate_single_source(H, i, args.t_max, args.n_steps)
        tau = compute_arrival_times(occupancy, args.t_max, args.occupancy_threshold)
        arrival_times[i, :] = tau

    # Compute effective velocities v_ij = d_ij / τ_ij
    inf = 1e9
    velocity_matrix = np.full((n, n), np.nan, dtype=float)
    v_list: List[float] = []

    for i in range(n):
        for j in range(n):
            d = D_graph[i, j]
            tau = arrival_times[i, j]
            if i == j:
                continue
            if d >= inf:
                continue  # disconnected
            if tau >= args.t_max:
                continue  # never reached threshold
            if tau <= 0.0:
                # arrival at t=0 for nonzero d would be unphysical; skip
                continue
            v = d / tau
            velocity_matrix[i, j] = v
            v_list.append(v)

    if v_list:
        v_arr = np.array(v_list, dtype=float)
        v_min = float(np.min(v_arr))
        v_max = float(np.max(v_arr))
        v_mean = float(np.mean(v_arr))
        v_median = float(np.median(v_arr))
        n_pairs = int(len(v_arr))
        has_finite_lightcone = True
    else:
        v_min = float("nan")
        v_max = float("nan")
        v_mean = float("nan")
        v_median = float("nan")
        n_pairs = 0
        has_finite_lightcone = False

    metrics = {
        "n_modes": n,
        "v_min": v_min,
        "v_max": v_max,
        "v_mean": v_mean,
        "v_median": v_median,
        "n_pairs_used": n_pairs,
    }

    diagnostics = {
        "graph_type": args.graph_type,
        "t_max": args.t_max,
        "n_steps": args.n_steps,
        "occupancy_threshold": args.occupancy_threshold,
        "coupling": args.coupling,
    }

    verdicts = {
        "has_finite_lightcone": has_finite_lightcone,
        "suggested_LR_velocity": v_max,
    }

    metadata = {
        "script": "lightcone_exp.py",
        "run_id": run_dir.name,
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
    }

    summary = {
        "script": "lightcone_exp.py",
        "run_id": run_dir.name,
        "timestamp": metadata["timestamp"],
        "params": {
            "cli": {
                "output_root": args.output_root,
                "tag": args.tag,
                "graph_type": args.graph_type,
                "n_modes": args.n_modes,
                "t_max": args.t_max,
                "n_steps": args.n_steps,
                "occupancy_threshold": args.occupancy_threshold,
                "coupling": args.coupling,
                "seed": args.seed,
            }
        },
        "metrics": metrics,
        "diagnostics": diagnostics,
        "verdicts": verdicts,
    }

    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "summary.json", summary)

    # Save raw data
    np.savez_compressed(
        data_dir / "lightcone_results.npz",
        distance_matrix=D_graph,
        arrival_times=arrival_times,
        velocity_matrix=velocity_matrix,
    )

    # Human-readable summary
    logger.write("==== Lieb–Robinson Light-Cone Summary ====\n")
    logger.write(f"n_modes:            {metrics['n_modes']}\n")
    logger.write(f"n_pairs_used:       {metrics['n_pairs_used']}\n")
    logger.write(f"v_min:              {metrics['v_min']}\n")
    logger.write(f"v_max:              {metrics['v_max']}\n")
    logger.write(f"v_mean:             {metrics['v_mean']}\n")
    logger.write(f"v_median:           {metrics['v_median']}\n")
    logger.write("\n---- Verdicts ----\n")
    for k, v in verdicts.items():
        logger.write(f"{k}: {v}\n")
    logger.write("==============================================\n")

    # Figures
    plot_distance_matrix(D_graph, fig_dir / "distances.png")
    plot_velocity_matrix(velocity_matrix, fig_dir / "velocities.png")
    plot_arrival_vs_distance(
        D_graph,
        arrival_times,
        args.t_max,
        fig_dir / "arrival_vs_distance.png",
    )

    logger.write(f"Outputs written under: {run_dir}\n")
    logger.close()


if __name__ == "__main__":
    main()

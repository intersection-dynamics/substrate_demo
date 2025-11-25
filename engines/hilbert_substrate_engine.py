"""
engines/hilbert_substrate_engine.py

Hilbert Substrate Engine v1

Concept:
    A 1D lattice of "slots", each with a 3-level local Hilbert space:

        |0>  = empty
        |1>  = single excitation
        |2>  = double excitation (allowed but dynamically fragile)

We evolve a global pure state |psi> over the 3^N basis states in the
occupation number basis (n_0, n_1, ..., n_{N-1}), with n_i in {0,1,2}.

Dynamics has two key pieces:

    1) Local "Hamiltonian-like" generator G that:
        - penalizes double occupancy (energy cost U_double)
        - allows hopping of single excitations between neighbors (J_hop)
        - allows double occupancy to split into two singles (J_split)
        - optionally the reverse process: two singles merge (J_merge)

       This is NOT required to be strictly Hermitian; we treat it as a
       toy generator that pushes amplitude out of multi-occupancy states.

    2) Decoherence / "measurement" events every decoherence_every steps:
        - We compute probabilities |psi_s|^2 in the occupation basis
        - We sample one configuration s according to these probabilities
        - We collapse |psi> to that basis state (pointer-like event)

The design goal is:

    "Show me a substrate where multi-occupancy states are systematically
     fragile, and 0/1 occupancy patterns are the only ones that look like
     classical records."

In this toy model:

    - Multi-occupancy (n_i = 2) is allowed in principle.
    - The generator pushes amplitude out of those configurations into
      configurations with separated single excitations.
    - Decoherence collapses the state into occupation-number basis
      configurations (pointer basis).
    - Over time, the long-lived classical records are dominated by
      patterns with n_i in {0,1}, while n_i = 2 appears rarely and
      does not persist.

Interface:

    run_experiment(raw_params: dict) -> dict

There is NO file I/O in this engine. An experiment script is responsible
for calling this engine and saving / plotting the results.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class HilbertSubstrateParams:
    # Lattice / topology
    n_sites: int = 6  # modest by default; dim = 3^n_sites

    # Time evolution
    dt: float = 0.05
    steps: int = 400

    # Generator parameters (toy "Hamiltonian-like" couplings)
    J_hop: float = 1.0     # single-excitation hopping between neighbors
    J_split: float = 1.0   # double -> two singles on neighboring sites
    J_merge: float = 0.3   # two singles -> double (reverse of split)
    U_double: float = 2.0  # onsite penalty for n_i = 2

    # Decoherence: pointer-state collapse in the occupation basis
    decoherence_every: int = 10  # apply collapse every N steps

    # Initial local occupation probabilities (per site)
    # These define a product state |psi(0)> = ⊗_i (α|0> + β|1> + γ|2>),
    # with probabilities p0, p1, p2 and random phases.
    p0: float = 0.6
    p1: float = 0.3
    p2: float = 0.1

    # RNG
    seed: int = 1


def _make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _build_basis(n_sites: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the occupation-number basis for n_sites with local dim = 3.

    Returns:
        index_to_config: ndarray of shape (dim, n_sites) with entries in {0,1,2},
                         where dim = 3**n_sites.
        pow3: ndarray of shape (n_sites,), where pow3[i] = 3**i.

    The index mapping is:

        index = sum_{i=0}^{n_sites-1} n_i * 3**i

    (Little-endian in site index; this choice is arbitrary but convenient.)
    """
    n = n_sites
    pow3 = np.array([3**i for i in range(n)], dtype=np.int64)
    dim = 3**n

    index_to_config = np.zeros((dim, n), dtype=np.int8)
    for idx in range(dim):
        x = idx
        for i in range(n):
            index_to_config[idx, i] = x % 3
            x //= 3

    return index_to_config, pow3


def _config_to_index(config: np.ndarray, pow3: np.ndarray) -> int:
    """
    Convert a configuration (n_sites,) with entries 0,1,2 into an index 0..dim-1.
    """
    # config[i] * 3**i, summed over i
    return int(np.dot(config.astype(np.int64), pow3))


def _initial_state(params: HilbertSubstrateParams,
                   index_to_config: np.ndarray,
                   pow3: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Build the initial global state |psi(0)> as a product state over sites:

        |psi_i> = sqrt(p0) e^{i φ0_i} |0> + sqrt(p1) e^{i φ1_i} |1> + sqrt(p2) e^{i φ2_i} |2>

    The phases are random, and p0+p1+p2 should be 1.0 (we renormalize if not).
    """
    p0, p1, p2 = params.p0, params.p1, params.p2
    total_p = max(1e-12, p0 + p1 + p2)
    p0 /= total_p
    p1 /= total_p
    p2 /= total_p

    amps_local = np.zeros((params.n_sites, 3), dtype=np.complex128)
    for i in range(params.n_sites):
        phi0, phi1, phi2 = rng.uniform(0.0, 2.0 * np.pi, size=3)
        amps_local[i, 0] = np.sqrt(p0) * np.exp(1j * phi0)
        amps_local[i, 1] = np.sqrt(p1) * np.exp(1j * phi1)
        amps_local[i, 2] = np.sqrt(p2) * np.exp(1j * phi2)

    dim = index_to_config.shape[0]
    psi = np.zeros(dim, dtype=np.complex128)

    # Build the product state explicitly in the occupation basis
    for idx in range(dim):
        config = index_to_config[idx]
        amp = 1.0 + 0.0j
        for site in range(params.n_sites):
            occ = config[site]
            amp *= amps_local[site, occ]
        psi[idx] = amp

    # Normalize
    norm = np.sqrt(np.sum(np.abs(psi)**2))
    if norm > 0:
        psi /= norm
    else:
        # Fallback: if something went wrong, pick a random basis state
        k = rng.integers(0, dim)
        psi[:] = 0.0
        psi[k] = 1.0 + 0.0j

    return psi


def _apply_generator(
    psi: np.ndarray,
    index_to_config: np.ndarray,
    pow3: np.ndarray,
    params: HilbertSubstrateParams,
) -> np.ndarray:
    """
    Compute hpsi = G psi, where G is a toy "generator" that:

        - penalizes double occupancy via U_double
        - allows hopping of single excitations between neighbors (J_hop)
        - allows double occupancy to split into two singles (J_split)
        - allows two singles to merge into a double (J_merge)

    This is intentionally simple and not guaranteed Hermitian.
    It is just a mechanism to push amplitude out of multi-occupancy states.
    """
    dim, n_sites = index_to_config.shape
    hpsi = np.zeros_like(psi, dtype=np.complex128)

    J_hop = params.J_hop
    J_split = params.J_split
    J_merge = params.J_merge
    U_double = params.U_double

    for idx in range(dim):
        amp = psi[idx]
        if amp == 0.0:
            continue

        config = index_to_config[idx]

        # Diagonal term: energy penalty for double occupancy
        n_double = np.count_nonzero(config == 2)
        if n_double > 0:
            hpsi[idx] += U_double * n_double * amp

        # Off-diagonal: neighbor interactions
        # We'll implement:
        #   - hop: 10 <-> 01
        #   - split: 20 or 02 -> 11
        #   - merge: 11 -> 20 or 02
        for site in range(n_sites - 1):
            a = config[site]
            b = config[site + 1]

            # Single hop: 10 -> 01
            if a == 1 and b == 0:
                cfg2 = config.copy()
                cfg2[site] = 0
                cfg2[site + 1] = 1
                idx2 = _config_to_index(cfg2, pow3)
                hpsi[idx2] += J_hop * amp

            # Single hop: 01 -> 10
            if a == 0 and b == 1:
                cfg2 = config.copy()
                cfg2[site] = 1
                cfg2[site + 1] = 0
                idx2 = _config_to_index(cfg2, pow3)
                hpsi[idx2] += J_hop * amp

            # Split: 20 -> 11
            if a == 2 and b == 0:
                cfg2 = config.copy()
                cfg2[site] = 1
                cfg2[site + 1] = 1
                idx2 = _config_to_index(cfg2, pow3)
                hpsi[idx2] += J_split * amp

            # Split: 02 -> 11
            if a == 0 and b == 2:
                cfg2 = config.copy()
                cfg2[site] = 1
                cfg2[site + 1] = 1
                idx2 = _config_to_index(cfg2, pow3)
                hpsi[idx2] += J_split * amp

            # Merge: 11 -> 20
            if a == 1 and b == 1:
                # 11 -> 20
                cfg2 = config.copy()
                cfg2[site] = 2
                cfg2[site + 1] = 0
                idx2 = _config_to_index(cfg2, pow3)
                hpsi[idx2] += J_merge * amp

                # 11 -> 02
                cfg3 = config.copy()
                cfg3[site] = 0
                cfg3[site + 1] = 2
                idx3 = _config_to_index(cfg3, pow3)
                hpsi[idx3] += J_merge * amp

    return hpsi


def _collapse_to_basis_state(
    psi: np.ndarray,
    index_to_config: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a projective measurement in the occupation-number basis:

        - Compute probabilities p_k = |psi_k|^2
        - Sample an index k ~ p
        - Set psi' = |k>, i.e. psi'_k = 1, psi'_{j!=k} = 0

    Returns:
        psi_new: collapsed state (dim,)
        config_k: the chosen configuration, shape (n_sites,)
    """
    dim = psi.shape[0]
    probs = np.abs(psi)**2
    total = probs.sum()
    if total <= 0.0:
        # Degenerate case: pick uniformly
        k = rng.integers(0, dim)
    else:
        probs /= total
        # Sample with replacement= False equivalent
        k = rng.choice(dim, p=probs)

    psi_new = np.zeros_like(psi, dtype=np.complex128)
    psi_new[k] = 1.0 + 0.0j
    config_k = index_to_config[k].copy()
    return psi_new, config_k


def run_experiment(raw_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a Hilbert-substrate experiment with the given parameters (as a dict).

    The returned dict has the structure:

        {
          "params": { ... },
          "metrics": {
             "mean_double_fraction": float,
             "max_double_fraction": float,
             "n_collapses": int,
          },
          "timeseries": {
             "collapse_step": [ ... ],
             "mean_occupancy": [ ... ],
             "double_fraction": [ ... ],
          },
          "snapshots": {
             "collapse_step": [ ... ],
             "configs": ndarray of shape (T, n_sites), int8
          },
          "diagnostics": {
             "notes": [ ... ],
          },
          "verdicts": {
             "multi_occupancy_fragile": bool,
             "zero_one_dominant": bool,
          },
        }

    Where:
        - "collapse_step" records the timestep index (1..steps) at each
          decoherence event.
        - "mean_occupancy" is the average n_i across the chain at each
          collapse.
        - "double_fraction" is the fraction of sites with n_i == 2 at
          each collapse.
    """
    # Merge user parameters with defaults
    base = HilbertSubstrateParams()
    for k, v in raw_params.items():
        if hasattr(base, k):
            setattr(base, k, v)
    params = base

    rng = _make_rng(params.seed)

    # Build basis (index <-> occupancy configuration)
    index_to_config, pow3 = _build_basis(params.n_sites)
    dim = index_to_config.shape[0]

    # Initialize global state
    psi = _initial_state(params, index_to_config, pow3, rng)

    # Time-series diagnostics (at collapse events only)
    collapse_steps: List[int] = []
    mean_occupancies: List[float] = []
    double_fractions: List[float] = []
    snapshot_configs: List[np.ndarray] = []

    # Simple note log
    notes: List[str] = []

    # Time evolution loop
    dt = params.dt
    deco_every = max(1, params.decoherence_every)

    for step in range(params.steps):
        # Apply generator: psi <- psi + dt * G psi, then renormalize
        hpsi = _apply_generator(psi, index_to_config, pow3, params)
        psi = psi + dt * hpsi

        # Renormalize
        norm = np.sqrt(np.sum(np.abs(psi)**2))
        if norm > 0:
            psi /= norm
        else:
            # If something went wrong numerically, reinitialize to a random basis state
            k = rng.integers(0, dim)
            psi[:] = 0.0
            psi[k] = 1.0 + 0.0j
            notes.append(f"Renormalization underflow at step {step}, reinitialized to basis state.")

        # Decoherence / pointer-state collapse
        if (step + 1) % deco_every == 0 or step == params.steps - 1:
            psi, config_k = _collapse_to_basis_state(psi, index_to_config, rng)

            # Record diagnostics
            collapse_steps.append(step + 1)
            mean_occ = float(np.mean(config_k))
            double_frac = float(np.count_nonzero(config_k == 2) / params.n_sites)
            mean_occupancies.append(mean_occ)
            double_fractions.append(double_frac)
            snapshot_configs.append(config_k)

    # Convert snapshots to ndarray
    if snapshot_configs:
        snapshot_array = np.stack(snapshot_configs, axis=0)
    else:
        snapshot_array = np.zeros((0, params.n_sites), dtype=np.int8)

    # Metrics summary
    if double_fractions:
        mean_double_fraction = float(np.mean(double_fractions))
        max_double_fraction = float(np.max(double_fractions))
    else:
        mean_double_fraction = 0.0
        max_double_fraction = 0.0

    # Simple verdicts:
    #   multi_occupancy_fragile: true if the typical double fraction is small
    #   zero_one_dominant: true if 0/1 patterns dominate (double_frac << 1)
    multi_occupancy_fragile = mean_double_fraction < 0.2 and max_double_fraction < 0.6
    zero_one_dominant = mean_double_fraction < 0.1

    metrics = {
        "mean_double_fraction": mean_double_fraction,
        "max_double_fraction": max_double_fraction,
        "n_collapses": len(collapse_steps),
    }

    timeseries = {
        "collapse_step": collapse_steps,
        "mean_occupancy": mean_occupancies,
        "double_fraction": double_fractions,
    }

    snapshots = {
        "collapse_step": collapse_steps,
        "configs": snapshot_array,
    }

    diagnostics = {
        "notes": notes,
    }

    verdicts = {
        "multi_occupancy_fragile": multi_occupancy_fragile,
        "zero_one_dominant": zero_one_dominant,
    }

    results: Dict[str, Any] = {
        "params": asdict(params),
        "metrics": metrics,
        "timeseries": timeseries,
        "snapshots": snapshots,
        "diagnostics": diagnostics,
        "verdicts": verdicts,
    }

    return results


if __name__ == "__main__":
    # Simple sanity check when run directly:
    #   python engines/hilbert_substrate_engine.py
    default_params = {}
    out = run_experiment(default_params)
    print("Params:", out["params"])
    print("Metrics:", out["metrics"])
    print("Verdicts:", out["verdicts"])

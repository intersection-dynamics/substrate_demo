"""
engines/chsh_substrate_engine.py

CHSH test on a minimal Hilbert substrate.

We treat two substrate modes as "Alice" and "Bob" qubits and evaluate
the CHSH S-parameter for a chosen entangled state, optionally with noise.

Axioms: Hilbert Space Realism, Unitary Evolution.
Constraint: Quantum Consistency (Bell/CHSH up to Tsirelson bound).

Public entrypoint:
    run_experiment(params: dict) -> dict
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any

import numpy as np


# ---------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------


@dataclass
class CHSHParams:
    # Randomness (currently only for possible future sampling; we still record it)
    seed: int = 1

    # Which Bell-like state to use
    #   'singlet'   -> |ψ-> = (|01⟩ - |10⟩)/√2
    #   'phi_plus'  -> |Φ+⟩ = (|00⟩ + |11⟩)/√2
    state_type: str = "singlet"

    # Noise model
    #   'none'          -> pure state
    #   'depolarizing'  -> ρ' = (1-p) ρ + p * I/4
    noise_type: str = "none"
    noise_strength: float = 0.0  # p in [0,1]

    # Numerical tolerances
    classical_bound: float = 2.0
    tsirelson_bound: float = 2.0 * np.sqrt(2.0)
    bound_tolerance: float = 1e-6  # slack for comparing with Tsirelson


# ---------------------------------------------------------------------
# Core linear algebra helpers
# ---------------------------------------------------------------------


def _pauli_matrices() -> Dict[str, np.ndarray]:
    """Return Pauli matrices and identity for a single qubit."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def _tensor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Kronecker product."""
    return np.kron(a, b)


# ---------------------------------------------------------------------
# States and noise
# ---------------------------------------------------------------------


def _build_bell_state(params: CHSHParams) -> np.ndarray:
    """
    Build a 2-qubit Bell-like state |ψ⟩ as a length-4 vector.

    Computational basis ordering: |00⟩, |01⟩, |10⟩, |11⟩.
    """
    psi = np.zeros(4, dtype=complex)

    if params.state_type.lower() == "singlet":
        # |ψ-> = (|01⟩ - |10⟩)/√2
        psi[1] = 1.0 / np.sqrt(2.0)
        psi[2] = -1.0 / np.sqrt(2.0)

    elif params.state_type.lower() == "phi_plus":
        # |Φ+⟩ = (|00⟩ + |11⟩)/√2
        psi[0] = 1.0 / np.sqrt(2.0)
        psi[3] = 1.0 / np.sqrt(2.0)

    else:
        raise ValueError(f"Unknown state_type: {params.state_type}")

    # Normalize defensively
    psi = psi / np.linalg.norm(psi)
    return psi


def _apply_noise(rho: np.ndarray, params: CHSHParams) -> np.ndarray:
    """
    Apply the chosen noise model to density matrix rho (4x4).
    """
    if params.noise_type.lower() == "none":
        return rho

    if params.noise_type.lower() == "depolarizing":
        p = float(params.noise_strength)
        p = max(0.0, min(1.0, p))
        dim = rho.shape[0]
        I = np.eye(dim, dtype=complex)
        return (1.0 - p) * rho + p * I / dim

    # Fallback: unknown noise -> no change, but record it in diagnostics later
    return rho


# ---------------------------------------------------------------------
# Measurement settings and CHSH computation
# ---------------------------------------------------------------------


def _build_measurement_operators() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build CHSH measurement operators for Alice and Bob.

    We use the standard choice that achieves Tsirelson bound on the singlet:

      A  = σ_z
      A' = σ_x
      B  = (σ_z + σ_x)/√2
      B' = (σ_z - σ_x)/√2

    These are ±1-valued observables.
    """
    pauli = _pauli_matrices()
    X = pauli["X"]
    Z = pauli["Z"]

    A = Z
    A_prime = X
    B = (Z + X) / np.sqrt(2.0)
    B_prime = (Z - X) / np.sqrt(2.0)

    return {
        "Alice": {
            "A": A,
            "A_prime": A_prime,
        },
        "Bob": {
            "B": B,
            "B_prime": B_prime,
        },
    }


def _expectation(rho: np.ndarray, op: np.ndarray) -> float:
    """⟨op⟩ = Tr(ρ op) (real for Hermitian op)."""
    val = np.trace(rho @ op)
    return float(np.real(val))


def _compute_chsh(rho: np.ndarray, params: CHSHParams) -> Dict[str, Any]:
    """
    Compute CHSH correlators and S for given 2-qubit density matrix ρ.
    """
    meas = _build_measurement_operators()
    A = meas["Alice"]["A"]
    A_prime = meas["Alice"]["A_prime"]
    B = meas["Bob"]["B"]
    B_prime = meas["Bob"]["B_prime"]

    # Tensor to two-qubit space
    A_B = _tensor(A, B)
    A_Bp = _tensor(A, B_prime)
    Ap_B = _tensor(A_prime, B)
    Ap_Bp = _tensor(A_prime, B_prime)

    E_AB = _expectation(rho, A_B)
    E_ABp = _expectation(rho, A_Bp)
    E_ApB = _expectation(rho, Ap_B)
    E_ApBp = _expectation(rho, Ap_Bp)

    S = E_AB + E_ABp + E_ApB - E_ApBp

    return {
        "E_AB": E_AB,
        "E_ABp": E_ABp,
        "E_ApB": E_ApB,
        "E_ApBp": E_ApBp,
        "S_CHSH": S,
    }


# ---------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------


def run_experiment(raw_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a CHSH test on the substrate.

    raw_params:
        Dict overriding CHSHParams defaults, e.g.:
          {
            "seed": 1,
            "state_type": "singlet",
            "noise_type": "depolarizing",
            "noise_strength": 0.1
          }

    Returns:
        {
          "params": {...},
          "metrics": {...},
          "diagnostics": {...},
          "verdicts": {...},
          "data": {...}
        }
    """
    # Merge defaults with overrides
    base = CHSHParams()
    for k, v in raw_params.items():
        if hasattr(base, k):
            setattr(base, k, v)
    params = base

    # Build state and density matrix
    _ = np.random.default_rng(params.seed)  # reserved for future sampling
    psi = _build_bell_state(params)
    rho_pure = np.outer(psi, np.conj(psi))

    # Apply noise if any
    rho = _apply_noise(rho_pure, params)

    # Compute CHSH correlators and S
    chsh = _compute_chsh(rho, params)
    S = float(chsh["S_CHSH"])

    # Bounds and diagnostics
    classical_bound = float(params.classical_bound)
    tsirelson_bound = float(params.tsirelson_bound)
    tol = float(params.bound_tolerance)

    violates_classical = abs(S) > (classical_bound + 1e-6)
    within_tsirelson = abs(S) <= (tsirelson_bound + tol)

    warnings = []
    if not within_tsirelson:
        warnings.append(
            f"S_CHSH={S:.6f} exceeds Tsirelson bound {tsirelson_bound:.6f} "
            f"beyond tolerance {tol:.1e}"
        )
    if params.noise_type.lower() not in ("none", "depolarizing"):
        warnings.append(f"Unknown noise_type: {params.noise_type}")

    metrics: Dict[str, Any] = {
        "S_CHSH": S,
        "E_AB": chsh["E_AB"],
        "E_ABp": chsh["E_ABp"],
        "E_ApB": chsh["E_ApB"],
        "E_ApBp": chsh["E_ApBp"],
        "classical_bound": classical_bound,
        "tsirelson_bound": tsirelson_bound,
    }

    diagnostics: Dict[str, Any] = {
        "within_tsirelson": within_tsirelson,
        "warnings": warnings,
    }

    verdicts: Dict[str, Any] = {
        "has_bell_violation": violates_classical,
        "violates_classical_bound": violates_classical,
    }

    data: Dict[str, Any] = {
        "settings_labels": ["A,B", "A,B'", "A',B", "A',B'"],
        "correlators": [
            chsh["E_AB"],
            chsh["E_ABp"],
            chsh["E_ApB"],
            chsh["E_ApBp"],
        ],
        "rho_real": np.real(rho),
        "rho_imag": np.imag(rho),
    }

    return {
        "params": asdict(params),
        "metrics": metrics,
        "diagnostics": diagnostics,
        "verdicts": verdicts,
        "data": data,
    }

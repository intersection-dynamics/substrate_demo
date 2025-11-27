from __future__ import annotations

"""
engines/substrate_engine.py

One-Hilbert-Space Substrate Engine (v2.1, with upgraded patterns experiment)

We define a single finite-dimensional Hilbert substrate:

    H_substrate = (C^2)^{⊗ N}

i.e. an N-qubit chain/graph.

All diagnostics are *views* on that same substrate:

  - CHSH:
        Pick two qubits (alice_qubit, bob_qubit) inside [0..N-1].
        Prepare a Bell-like pattern on that pair (rest in |0...0>).
        Compute CHSH S-parameter on that subsystem.
        Noise acts on the 2-qubit reduced state.

  - Patterns (UPGRADED):
        Use qubits [0..n_sys_qubits-1] as "system",
        and [n_sys_qubits..n_sys_qubits+n_env_qubits-1] as "environment".
        (We enforce N = n_sys_qubits + n_env_qubits for this experiment.)

        Detection scheme (new):
          - detection_mode = "phase_sensitive" (default):
              For the toy N=3, n_sys=2, n_env=1 case:
                * symmetric pattern remains pure (bosonic_like),
                * antisymmetric pattern decoheres (fermionic_like).
              For other sizes, falls back automatically.

          - detection_mode = "population":
              Original scheme: plain CNOTs from one system qubit into env.

  - Metric:
        Use all N modes as a graph for propagation (single-excitation sector
        of the same substrate). Propagate a single excitation and define
        an emergent distance matrix from arrival times.

Public entrypoint:

    run_experiment(raw_params: dict) -> dict

raw_params must contain:

    "experiment": one of {"chsh", "patterns", "metric"}

and can include other keys depending on the experiment.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import numpy as np


# =====================================================================
# Common helpers
# =====================================================================


def _pauli_matrices() -> Dict[str, np.ndarray]:
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def _tensor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.kron(a, b)


def _purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


def _fidelity_pure_target(rho: np.ndarray, psi_target: np.ndarray) -> float:
    """
    Fidelity between mixed state ρ and pure state |ψ><ψ|:

        F = <ψ| ρ |ψ>.
    """
    v = psi_target.conj().T @ (rho @ psi_target)
    return float(np.real(v))


def _partial_trace_env(
    rho_full: np.ndarray,
    n_sys_qubits: int,
    n_env_qubits: int,
) -> np.ndarray:
    """
    Trace out env qubits from a full density matrix ρ_full.

    Convention here:

      - Total N = n_sys_qubits + n_env_qubits qubits.
      - Qubit ordering is [sys_0, ..., sys_{n_sys-1}, env_0, ..., env_{n_env-1}].

    Return ρ_sys with shape (2^n_sys, 2^n_sys).
    """
    d_sys = 2**n_sys_qubits
    d_env = 2**n_env_qubits
    rho_reshaped = rho_full.reshape(d_sys, d_env, d_sys, d_env)
    rho_sys = np.trace(rho_reshaped, axis1=1, axis2=3)
    return rho_sys


def _expectation(rho: np.ndarray, op: np.ndarray) -> float:
    """
    Expectation value Tr(ρ op) for Hermitian op (real).
    """
    val = np.trace(rho @ op)
    return float(np.real(val))


def _build_graph_adjacency(n_modes: int, graph_type: str = "chain") -> np.ndarray:
    """
    Simple graph adjacency for N modes.

    graph_type:
        "chain"   -> 1D chain
        "complete"-> fully connected
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


def _build_single_excitation_hamiltonian(
    n_modes: int,
    adjacency: np.ndarray,
    coupling: float,
) -> np.ndarray:
    """
    Single-excitation Hamiltonian on an n-mode graph:

        H = g ∑_{i≠j, A_ij != 0} |i><j|

    (Hermitian because A is symmetric and off-diagonals are mirrored.)
    """
    g = float(coupling)
    H = np.zeros((n_modes, n_modes), dtype=complex)
    for i in range(n_modes):
        for j in range(n_modes):
            if i != j and adjacency[i, j] != 0.0:
                H[i, j] += g
    return H


def _build_cnot_unitary_full(
    n_qubits: int,
    control_index: int,
    target_index: int,
) -> np.ndarray:
    """
    Build a full N-qubit CNOT (control, target) as a permutation matrix.

    Qubit ordering: [q0, q1, ..., q_{N-1}], big-endian in the integer index:

        basis index x corresponds to bits:
            bits[0] = q0 (most significant),
            ...
            bits[N-1] = q_{N-1} (least significant).

    control_index, target_index are in [0..N-1].
    """
    N = n_qubits
    dim = 2**N
    U = np.zeros((dim, dim), dtype=complex)

    for x in range(dim):
        # Extract bits of x
        bits = [(x >> k) & 1 for k in reversed(range(N))]
        control_bit = bits[control_index]
        if control_bit == 1:
            bits[target_index] ^= 1
        # Convert bits back to integer y
        y = 0
        for b in bits:
            y = (y << 1) | b
        U[y, x] = 1.0

    return U


# =====================================================================
# Substrate class
# =====================================================================


@dataclass
class SubstrateConfig:
    n_qubits: int = 4
    graph_type: str = "chain"
    coupling: float = 1.0
    seed: int = 1


class Substrate:
    """
    One Hilbert substrate: N-qubit chain/graph.

    - Hilbert space: (C^2)^{⊗ N}
    - Graph: adjacency A (N×N)
    - Single-excitation Hamiltonian H_single on the same N modes.

    We don't maintain a single persistent psi here because different
    diagnostics prepare different initial patterns, but all diagnostics
    *assume* they live in (and/or are subspaces of) this N-qubit space.
    """

    def __init__(self, config: SubstrateConfig):
        self.config = config
        self.n_qubits = int(config.n_qubits)
        self.graph_type = str(config.graph_type)
        self.coupling = float(config.coupling)
        self.seed = int(config.seed)

        if self.n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")

        self.rng = np.random.default_rng(self.seed)

        # Graph + single-excitation Hamiltonian for metric-style propagation
        self.adjacency = _build_graph_adjacency(self.n_qubits, self.graph_type)
        self.H_single = _build_single_excitation_hamiltonian(
            self.n_qubits,
            self.adjacency,
            self.coupling,
        )

    # -----------------------------------------------------------------
    # CHSH / Bell diagnostic
    # -----------------------------------------------------------------

    @dataclass
    class CHSHParams:
        # Which qubits act as Alice and Bob
        alice_qubit: int = 0
        bob_qubit: int = 1

        # Bell-like state type
        state_type: str = "singlet"   # "singlet" or "phi_plus"

        # Noise model
        noise_type: str = "none"      # "none" or "depolarizing"
        noise_strength: float = 0.0   # p in [0,1] for depolarizing

        # Bounds and tolerances
        classical_bound: float = 2.0
        tsirelson_bound: float = 2.0 * np.sqrt(2.0)
        bound_tolerance: float = 1e-6

        seed: int = 1  # reserved

    def _build_bell_state_pair(self, params: "Substrate.CHSHParams") -> np.ndarray:
        """
        Build a 4-vector Bell-like state for the *pair* only.

        Basis order: |00>, |01>, |10>, |11>.
        """
        psi_pair = np.zeros(4, dtype=complex)
        st = params.state_type.lower()

        if st == "singlet":
            # |ψ-> = (|01⟩ - |10⟩)/√2
            psi_pair[1] = 1.0 / np.sqrt(2.0)
            psi_pair[2] = -1.0 / np.sqrt(2.0)
        elif st == "phi_plus":
            # |Φ+⟩ = (|00⟩ + |11⟩)/√2
            psi_pair[0] = 1.0 / np.sqrt(2.0)
            psi_pair[3] = 1.0 / np.sqrt(2.0)
        else:
            raise ValueError(f"Unknown state_type: {params.state_type}")

        psi_pair = psi_pair / np.linalg.norm(psi_pair)
        return psi_pair

    @staticmethod
    def _apply_chsh_noise_pair(
        rho_pair: np.ndarray,
        params: "Substrate.CHSHParams",
    ) -> np.ndarray:
        """
        Apply noise to a 4×4 2-qubit density matrix.
        """
        nt = params.noise_type.lower()
        if nt == "none":
            return rho_pair

        if nt == "depolarizing":
            p = max(0.0, min(1.0, float(params.noise_strength)))
            dim = rho_pair.shape[0]
            I = np.eye(dim, dtype=complex)
            return (1.0 - p) * rho_pair + p * I / dim

        # Unknown noise -> no change; flagged later
        return rho_pair

    @staticmethod
    def _build_chsh_measurements_pair() -> Dict[str, Dict[str, np.ndarray]]:
        """
        Standard CHSH settings on a 2-qubit system (pair only):

            A  = σ_z
            A' = σ_x
            B  = (σ_z + σ_x)/√2
            B' = (σ_z - σ_x)/√2
        """
        pauli = _pauli_matrices()
        X = pauli["X"]
        Z = pauli["Z"]

        A = Z
        A_prime = X
        B = (Z + X) / np.sqrt(2.0)
        B_prime = (Z - X) / np.sqrt(2.0)

        return {
            "Alice": {"A": A, "A_prime": A_prime},
            "Bob": {"B": B, "B_prime": B_prime},
        }

    def run_chsh(self, raw_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        CHSH diagnostic on a **pair of qubits** inside this substrate.

        We conceptually embed the pair into the N-qubit substrate with
        all other qubits in |0>. For the CHSH computation itself, only
        the 2-qubit reduced state matters, so we do the algebra in that
        4-dimensional subspace.
        """
        # Resolve parameters with defaults
        base = self.CHSHParams()
        for k, v in raw_params.items():
            if hasattr(base, k):
                setattr(base, k, v)
        params = base

        # Basic sanity on qubit indices
        if self.n_qubits < 2:
            raise ValueError("CHSH requires at least 2 qubits in the substrate.")
        if not (0 <= params.alice_qubit < self.n_qubits):
            raise ValueError("alice_qubit out of range for substrate size.")
        if not (0 <= params.bob_qubit < self.n_qubits):
            raise ValueError("bob_qubit out of range for substrate size.")
        if params.alice_qubit == params.bob_qubit:
            raise ValueError("alice_qubit and bob_qubit must be different.")

        # Build Bell state on the pair (2-qubit space)
        self.rng = np.random.default_rng(params.seed)
        psi_pair = self._build_bell_state_pair(params)
        rho_pair_pure = np.outer(psi_pair, psi_pair.conj())

        # Apply noise on the pair
        rho_pair = self._apply_chsh_noise_pair(rho_pair_pure, params)

        # Build measurement operators for the pair
        meas = self._build_chsh_measurements_pair()
        A = meas["Alice"]["A"]
        A_p = meas["Alice"]["A_prime"]
        B = meas["Bob"]["B"]
        B_p = meas["Bob"]["B_prime"]

        # CHSH on pair-space
        A_B = _tensor(A, B)
        A_Bp = _tensor(A, B_p)
        A_pB = _tensor(A_p, B)
        A_pBp = _tensor(A_p, B_p)

        E_AB = _expectation(rho_pair, A_B)
        E_ABp = _expectation(rho_pair, A_Bp)
        E_ApB = _expectation(rho_pair, A_pB)
        E_ApBp = _expectation(rho_pair, A_pBp)

        S = E_AB + E_ABp + E_ApB - E_ApBp

        classical_bound = float(params.classical_bound)
        tsirelson_bound = float(params.tsirelson_bound)
        tol = float(params.bound_tolerance)

        violates_classical = abs(S) > (classical_bound + 1e-6)
        within_tsirelson = abs(S) <= (tsirelson_bound + tol)

        warnings: List[str] = []
        if not within_tsirelson:
            warnings.append(
                f"S_CHSH={S:.6f} exceeds Tsirelson bound {tsirelson_bound:.6f} "
                f"beyond tolerance {tol:.1e}"
            )
        if params.noise_type.lower() not in ("none", "depolarizing"):
            warnings.append(f"Unknown noise_type: {params.noise_type}")

        metrics = {
            "S_CHSH": float(S),
            "E_AB": E_AB,
            "E_ABp": E_ABp,
            "E_ApB": E_ApB,
            "E_ApBp": E_ApBp,
            "classical_bound": classical_bound,
            "tsirelson_bound": tsirelson_bound,
        }

        diagnostics = {
            "within_tsirelson": within_tsirelson,
            "warnings": warnings,
            "substrate_n_qubits": self.n_qubits,
            "alice_qubit": params.alice_qubit,
            "bob_qubit": params.bob_qubit,
        }

        verdicts = {
            "has_bell_violation": violates_classical,
            "violates_classical_bound": violates_classical,
        }

        data = {
            "settings_labels": ["A,B", "A,B'", "A',B", "A',B'"],
            "correlators": [E_AB, E_ABp, E_ApB, E_ApBp],
            "rho_pair_real": np.real(rho_pair),
            "rho_pair_imag": np.imag(rho_pair),
        }

        params_dict = {
            "substrate": asdict(self.config),
            "experiment": asdict(params),
        }

        return {
            "experiment": "chsh",
            "params": params_dict,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "verdicts": verdicts,
            "data": data,
        }

    # -----------------------------------------------------------------
    # Patterns / copyability diagnostic (UPGRADED)
    # -----------------------------------------------------------------

    @dataclass
    class PatternParams:
        # System + environment layout
        n_sys_qubits: int = 2
        n_env_qubits: int = 3

        # Pattern choice (on system only):
        #   "local"        -> |10...0>
        #   "symmetric"    -> (|10...0> + |01...0>)/√2
        #   "antisymmetric"-> (|10...0> - |01...0>)/√2
        #   "bell"         -> (|00...0> + |11...1>)/√2
        pattern_type: str = "symmetric"

        # Detection protocol
        n_detections: int = 8
        sys_qubit_index: int = 0  # index inside [0..n_sys_qubits-1]

        # NEW: detection mode
        #   "phase_sensitive" (default):
        #       For N=3, n_sys=2, n_env=1:
        #           symmetric stays pure,
        #           antisymmetric decoheres (purity≈0.5, fidelity≈0.5).
        #   "population":
        #       Original CNOT-based scheme.
        detection_mode: str = "phase_sensitive"

        # Thresholds for classification
        fidelity_threshold: float = 0.8
        purity_threshold: float = 0.8

        seed: int = 1

    @staticmethod
    def _build_sys_pattern(params: "Substrate.PatternParams") -> np.ndarray:
        """
        Construct a 2^n_sys_qubits pure state vector |ψ_sys⟩.
        """
        n = params.n_sys_qubits
        dim = 2**n
        psi = np.zeros(dim, dtype=complex)
        pt = params.pattern_type.lower()

        def basis_index(bits: List[int]) -> int:
            idx = 0
            for b in bits:
                idx = (idx << 1) | (b & 1)
            return idx

        if pt == "local":
            bits = [1] + [0] * (n - 1)
            psi[basis_index(bits)] = 1.0

        elif pt == "symmetric":
            if n < 2:
                raise ValueError("symmetric pattern requires n_sys_qubits >= 2")
            bits1 = [1, 0] + [0] * (n - 2)
            bits2 = [0, 1] + [0] * (n - 2)
            psi[basis_index(bits1)] = 1.0 / np.sqrt(2.0)
            psi[basis_index(bits2)] = 1.0 / np.sqrt(2.0)

        elif pt == "antisymmetric":
            if n < 2:
                raise ValueError("antisymmetric pattern requires n_sys_qubits >= 2")
            bits1 = [1, 0] + [0] * (n - 2)
            bits2 = [0, 1] + [0] * (n - 2)
            psi[basis_index(bits1)] = 1.0 / np.sqrt(2.0)
            psi[basis_index(bits2)] = -1.0 / np.sqrt(2.0)

        elif pt == "bell":
            bits1 = [0] * n
            bits2 = [1] * n
            psi[basis_index(bits1)] = 1.0 / np.sqrt(2.0)
            psi[basis_index(bits2)] = 1.0 / np.sqrt(2.0)

        else:
            raise ValueError(f"Unknown pattern_type: {params.pattern_type}")

        psi = psi / np.linalg.norm(psi)
        return psi

    # --- NEW helper: phase-sensitive 3-qubit detection unitary ---

    @staticmethod
    def _build_phase_sensitive_detection_unitary_3qubit() -> np.ndarray:
        """
        Build an 8x8 unitary U acting on 3 qubits (sys0, sys1, env0) such that:

          - For symmetric pattern:
                |ψ+> = (|10> + |01>)/√2, env=|0>
                U (|ψ+>|0>) = |ψ+>|0>        (unchanged, pure)

          - For antisymmetric pattern:
                |ψ-> = (|10> - |01>)/√2, env=|0>
                U (|ψ->>|0>) = (|ψ->>|0> + |ψ+>|1>)/√2
                             → reduced ρ_sys = 1/2(|ψ->><ψ-| + |ψ+><ψ+|)
                               (mixed, purity=0.5, fidelity=0.5).

        Outside the subspace spanned by these states, U acts as identity.
        """
        dim = 8
        I = np.eye(dim, dtype=complex)

        # computational basis |a b c> with index 4a + 2b + c
        basis = [I[:, i] for i in range(dim)]

        # |10 0> = index 4, |01 0> = index 2
        psi_plus_env0 = (basis[4] + basis[2]) / np.sqrt(2.0)
        psi_minus_env0 = (basis[4] - basis[2]) / np.sqrt(2.0)
        # |10 1> = 5, |01 1> = 3
        psi_plus_env1 = (basis[5] + basis[3]) / np.sqrt(2.0)

        # Orthonormal vectors v1, v2
        v1 = psi_minus_env0 / np.linalg.norm(psi_minus_env0)
        v2 = psi_plus_env1 / np.linalg.norm(psi_plus_env1)

        # Gram-Schmidt to complete basis {v1, v2, ...}
        def gramschmidt(vecs: List[np.ndarray]) -> List[np.ndarray]:
            ortho: List[np.ndarray] = []
            for v in vecs:
                w = v.astype(complex)
                for u in ortho:
                    w = w - np.vdot(u, w) * u
                nrm = np.linalg.norm(w)
                if nrm > 1e-10:
                    ortho.append(w / nrm)
            return ortho

        comp = [np.eye(dim, dtype=complex)[:, i] for i in range(dim)]
        B = gramschmidt([v1, v2] + comp)  # domain basis
        # B[0] = v1, B[1] = v2

        # Target basis W: w1,w2,...:
        # w1 = (v1 + v2)/√2, w2 = (v2 - v1)/√2, rest unchanged.
        w1 = (v1 + v2) / np.sqrt(2.0)
        w2 = (v2 - v1) / np.sqrt(2.0)
        W = [w1, w2] + B[2:]

        # Build U = Σ_i |W_i><B_i|
        U = np.zeros((dim, dim), dtype=complex)
        for wi, bi in zip(W, B):
            U += np.outer(wi, bi.conj())

        # Small sanity: U should be unitary to numerical precision
        # (we won't assert here, but this was checked in dev).
        return U

    def run_patterns(self, raw_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Patterns / copyability diagnostic on this substrate.

        We enforce that the substrate's N matches:

            N = n_sys_qubits + n_env_qubits

        System qubits: indices [0..n_sys_qubits-1]
        Environment qubits: indices [n_sys_qubits..N-1]

        Detection modes:

          - "phase_sensitive" (default):
                If (n_sys_qubits=2, n_env_qubits=1, N=3):
                    use a phase-sensitive 3-qubit unitary U that:
                      * preserves symmetric pattern,
                      * decoheres antisymmetric pattern.
                Otherwise, fall back to "population".

          - "population":
                Original scheme: CNOT from sys_qubit_index to each env qubit.
        """
        base = self.PatternParams()
        for k, v in raw_params.items():
            if hasattr(base, k):
                setattr(base, k, v)
        params = base

        n_sys = params.n_sys_qubits
        n_env = params.n_env_qubits
        N = self.n_qubits

        if N != n_sys + n_env:
            raise ValueError(
                f"Substrate n_qubits={N} must equal n_sys_qubits+n_env_qubits={n_sys+n_env} "
                "for the patterns diagnostic."
            )

        if not (0 <= params.sys_qubit_index < n_sys):
            raise ValueError("sys_qubit_index must be within [0..n_sys_qubits-1].")

        self.rng = np.random.default_rng(params.seed)

        # System pattern |ψ_sys⟩
        psi_sys_0 = self._build_sys_pattern(params)

        # Env initial state |0...0>_env
        dim_env = 2**n_env
        psi_env_0 = np.zeros(dim_env, dtype=complex)
        psi_env_0[0] = 1.0

        # Full substrate state |ψ_full⟩ = |ψ_sys⟩ ⊗ |0...0>_env
        psi_full = np.kron(psi_sys_0, psi_env_0)
        psi_full = psi_full / np.linalg.norm(psi_full)
        rho_full = np.outer(psi_full, psi_full.conj())

        # Initial reduced system state
        rho_sys_0 = _partial_trace_env(rho_full, n_sys, n_env)
        purity_0 = _purity(rho_sys_0)

        # Detection mode selection
        requested_mode = params.detection_mode.lower()
        detection_mode_used = requested_mode
        fallback_reason = ""

        # Phase-sensitive detection only fully supported for the minimal toy:
        #   n_sys_qubits=2, n_env_qubits=1, N=3
        U_phase_full = None
        if requested_mode == "phase_sensitive":
            if n_sys == 2 and n_env == 1 and N == 3:
                # Build 3-qubit U and treat it as full substrate unitary
                U3 = self._build_phase_sensitive_detection_unitary_3qubit()
                U_phase_full = U3
            else:
                detection_mode_used = "population"
                fallback_reason = (
                    "phase_sensitive mode currently only implemented for "
                    "n_sys_qubits=2, n_env_qubits=1, N=3; "
                    "falling back to population mode."
                )

        # Build unitaries for population mode (CNOTs)
        population_unitaries: List[np.ndarray] = []
        if detection_mode_used == "population":
            control_global = params.sys_qubit_index
            for env_local_idx in range(n_env):
                target_global = n_sys + env_local_idx
                U = _build_cnot_unitary_full(N, control_global, target_global)
                population_unitaries.append(U)

        purity_trace: List[float] = []
        fidelity_trace: List[float] = []

        # Step 0 (no detection yet)
        rho_sys = _partial_trace_env(rho_full, n_sys, n_env)
        purity_trace.append(_purity(rho_sys))
        fidelity_trace.append(_fidelity_pure_target(rho_sys, psi_sys_0))

        # Repeated detections
        for step in range(1, params.n_detections + 1):
            if detection_mode_used == "phase_sensitive" and U_phase_full is not None:
                # Same phase-sensitive unitary each step
                rho_full = U_phase_full @ rho_full @ U_phase_full.conj().T
            else:
                # Population-mode CNOTs
                env_idx = (step - 1) % n_env
                U = population_unitaries[env_idx]
                rho_full = U @ rho_full @ U.conj().T

            rho_sys = _partial_trace_env(rho_full, n_sys, n_env)
            purity_trace.append(_purity(rho_sys))
            fidelity_trace.append(_fidelity_pure_target(rho_sys, psi_sys_0))

        final_purity = purity_trace[-1]
        final_fidelity = fidelity_trace[-1]

        is_bosonic_like = (
            final_fidelity >= params.fidelity_threshold
            and final_purity >= params.purity_threshold
        )
        classification = "bosonic_like" if is_bosonic_like else "fermionic_like"

        metrics = {
            "initial_purity": purity_trace[0],
            "final_purity": final_purity,
            "initial_fidelity": fidelity_trace[0],
            "final_fidelity": final_fidelity,
            "fidelity_threshold": params.fidelity_threshold,
            "purity_threshold": params.purity_threshold,
        }

        diagnostics = {
            "seed": params.seed,
            "n_sys_qubits": n_sys,
            "n_env_qubits": n_env,
            "n_detections": params.n_detections,
            "initial_purity_check": purity_0,
            "substrate_n_qubits": self.n_qubits,
            "detection_mode_requested": requested_mode,
            "detection_mode_used": detection_mode_used,
            "detection_fallback_reason": fallback_reason,
        }

        verdicts = {
            "classification": classification,
            "bosonic_like": is_bosonic_like,
            "fermionic_like": not is_bosonic_like,
        }

        data = {
            "purity_trace": np.array(purity_trace, dtype=float),
            "fidelity_trace": np.array(fidelity_trace, dtype=float),
            "steps": np.arange(len(purity_trace), dtype=int),
            "rho_sys_initial": rho_sys_0,
            "pattern_type": params.pattern_type,
            "detection_mode_used": detection_mode_used,
        }

        params_dict = {
            "substrate": asdict(self.config),
            "experiment": asdict(params),
        }

        return {
            "experiment": "patterns",
            "params": params_dict,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "verdicts": verdicts,
            "data": data,
        }

    # -----------------------------------------------------------------
    # Metric / propagation diagnostic
    # -----------------------------------------------------------------

    @dataclass
    class MetricParams:
        t_max: float = 10.0
        n_steps: int = 200
        occupancy_threshold: float = 0.1
        seed: int = 1

    def _propagate_single_source(
        self,
        source: int,
        t_max: float,
        n_steps: int,
    ) -> np.ndarray:
        """
        Propagate a single excitation initially localized at 'source'
        in the single-excitation sector of the substrate.

        Returns occupancy[t, mode] with t in [0..n_steps] (including t=0).
        """
        n = self.n_qubits
        H = self.H_single

        # Diagonalize H once
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

    @staticmethod
    def _compute_arrival_times(
        occupancy: np.ndarray,
        t_max: float,
        threshold: float,
    ) -> np.ndarray:
        """
        Given occupancy[t, mode], return an array arrival[mode] of
        earliest times when occupancy >= threshold (or t_max if never).
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

    def run_metric(self, raw_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Emergent metric diagnostic:

          - Uses ALL N modes of the substrate.
          - For each source i, propagate in the single-excitation sector.
          - Compute arrival times to each mode j.
          - Distance matrix D_ij = arrival_time(i→j).
          - Compute triangle-inequality violation rate.
        """
        base = self.MetricParams()
        for k, v in raw_params.items():
            if hasattr(base, k):
                setattr(base, k, v)
        params = base

        if self.n_qubits < 2:
            raise ValueError("Metric diagnostic requires at least 2 qubits/modes.")

        n = self.n_qubits
        D = np.zeros((n, n), dtype=float)

        for i in range(n):
            occupancy = self._propagate_single_source(i, params.t_max, params.n_steps)
            arrivals = self._compute_arrival_times(
                occupancy, params.t_max, params.occupancy_threshold
            )
            D[i, :] = arrivals

        # Triangle inequality: D(i,k) <= D(i,j) + D(j,k) + eps
        eps = 1e-9
        violations = 0
        total_triples = 0
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                for k in range(n):
                    if k == j:
                        continue
                    total_triples += 1
                    if D[i, k] > D[i, j] + D[j, k] + eps:
                        violations += 1

        violation_rate = violations / max(1, total_triples)

        metrics = {
            "violation_rate": violation_rate,
            "n_modes": n,
        }

        diagnostics = {
            "graph_type": self.graph_type,
            "occupancy_threshold": params.occupancy_threshold,
            "n_triples_checked": total_triples,
            "substrate_n_qubits": self.n_qubits,
        }

        verdicts = {
            "approximately_metric": violation_rate < 0.1,  # toy threshold
        }

        data = {
            "distance_matrix": D,
            "adjacency": self.adjacency,
        }

        params_dict = {
            "substrate": asdict(self.config),
            "experiment": asdict(params),
        }

        return {
            "experiment": "metric",
            "params": params_dict,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "verdicts": verdicts,
            "data": data,
        }


# =====================================================================
# Unified public entrypoint
# =====================================================================


def run_experiment(raw_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified entrypoint for the one-Hilbert-space substrate engine.

    raw_params MUST contain:
        "experiment": one of {"chsh", "patterns", "metric"}

    In addition, it may contain:

      Substrate-level:
        - "n_qubits"   (int)
        - "graph_type" (str; for metric/propagation)
        - "coupling"   (float; for metric/propagation)
        - "seed"       (int)

      CHSH-specific:
        - "alice_qubit", "bob_qubit"
        - "state_type"
        - "noise_type"
        - "noise_strength"
        - "classical_bound", "tsirelson_bound", "bound_tolerance"

      Patterns-specific:
        - "n_sys_qubits", "n_env_qubits"
        - "pattern_type"
        - "n_detections"
        - "sys_qubit_index"
        - "detection_mode"
        - "fidelity_threshold", "purity_threshold"

      Metric-specific:
        - "t_max", "n_steps", "occupancy_threshold"

    Returns:
        {
          "experiment": "...",
          "params": {...},
          "metrics": {...},
          "diagnostics": {...},
          "verdicts": {...},
          "data": {...}
        }
    """
    if "experiment" not in raw_params:
        raise ValueError(
            "raw_params must include an 'experiment' key: "
            "'chsh', 'patterns', or 'metric'."
        )

    exp = str(raw_params["experiment"]).lower()

    # -----------------------------
    # Decide substrate size N
    # -----------------------------
    if "n_qubits" in raw_params:
        n_qubits = int(raw_params["n_qubits"])
    else:
        if exp == "chsh":
            alice_q = int(raw_params.get("alice_qubit", 0))
            bob_q = int(raw_params.get("bob_qubit", 1))
            n_qubits = max(alice_q, bob_q) + 1
            n_qubits = max(n_qubits, 2)
        elif exp == "patterns":
            n_sys = int(raw_params.get("n_sys_qubits", 2))
            n_env = int(raw_params.get("n_env_qubits", 3))
            n_qubits = n_sys + n_env
        elif exp == "metric":
            # Use n_modes as n_qubits so propagation sees all modes
            n_qubits = int(raw_params.get("n_modes", 6))
        else:
            raise ValueError(f"Unknown experiment type: {raw_params['experiment']}")

    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1.")

    graph_type = str(raw_params.get("graph_type", "chain"))
    coupling = float(raw_params.get("coupling", 1.0))
    seed = int(raw_params.get("seed", 1))

    config = SubstrateConfig(
        n_qubits=n_qubits,
        graph_type=graph_type,
        coupling=coupling,
        seed=seed,
    )
    substrate = Substrate(config)

    # -----------------------------
    # Dispatch to appropriate diagnostic
    # -----------------------------
    if exp == "chsh":
        # Only pass CHSH-relevant keys down
        chsh_keys = {
            "alice_qubit",
            "bob_qubit",
            "state_type",
            "noise_type",
            "noise_strength",
            "classical_bound",
            "tsirelson_bound",
            "bound_tolerance",
            "seed",
        }
        chsh_params = {k: v for k, v in raw_params.items() if k in chsh_keys}
        return substrate.run_chsh(chsh_params)

    elif exp == "patterns":
        pattern_keys = {
            "n_sys_qubits",
            "n_env_qubits",
            "pattern_type",
            "n_detections",
            "sys_qubit_index",
            "detection_mode",
            "fidelity_threshold",
            "purity_threshold",
            "seed",
        }
        pattern_params = {k: v for k, v in raw_params.items() if k in pattern_keys}
        return substrate.run_patterns(pattern_params)

    elif exp == "metric":
        metric_keys = {
            "t_max",
            "n_steps",
            "occupancy_threshold",
            "seed",
        }
        metric_params = {k: v for k, v in raw_params.items() if k in metric_keys}
        # Ensure substrate.n_qubits matches n_modes for metric
        # (already handled above by choosing n_qubits from n_modes).
        return substrate.run_metric(metric_params)

    else:
        raise ValueError(f"Unknown experiment type: {raw_params['experiment']}")

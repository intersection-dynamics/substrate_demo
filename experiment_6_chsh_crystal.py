#!/usr/bin/env python3
"""
experiment_6_chsh_crystal_realistic.py

CHSH experiment with a more realistic simulated SPDC crystal + detectors:

- Pulsed laser: N_pulses pump pulses.
- Each pulse hits a nonlinear crystal.
- With probability p_pair, the crystal emits *one* entangled pair.
  Otherwise it emits no pair (low-gain SPDC approximation).

- The emitted pair is not perfectly entangled: we use a simple
  "visibility" model (Werner-like mixture):
    with probability V: sample from an ideal Bell state |Phi+>
    with probability 1-V: sample completely random, uncorrelated outcomes.

- Detectors on Alice/Bob arms have:
    detection efficiency eta_A, eta_B
    dark count probability dark_A, dark_B
  We keep only events where both sides register exactly one click
  (coincidence events).

- For all coincidence events, we:
    * compute correlations E(Ai,Bj),
    * compute CHSH S,
    * report correlated vs anticorrelated counts.

Conventions:
- |0> = |H>, |1> = |V> (polarization basis)
- Settings: A0, A1 for Alice; B0, B1 for Bob.
- Outcomes: ±1 (dichotomic).
"""

import numpy as np

# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def normalize(state):
    """Normalize a state vector (returns 1D vector)."""
    state = np.asarray(state, dtype=complex).reshape(-1)
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("Zero-norm state.")
    return state / norm


def kron(*ops):
    """
    Kronecker product of many operators/vectors.

    Starts from a 1D scalar so that:
      kron(H, H) -> shape (4,)
    rather than (1,4).
    """
    M = np.array([1.0+0j])   # 1D scalar
    for op in ops:
        M = np.kron(M, op)
    return M


# ---------------------------------------------------------------------
# Polarization basis and Bell state
# ---------------------------------------------------------------------

# |0> = |H>, |1> = |V>
H_vec = np.array([1.0, 0.0], dtype=complex)
V_vec = np.array([0.0, 1.0], dtype=complex)

# Two-photon computational basis: |HH>, |HV>, |VH>, |VV>
HH = kron(H_vec, H_vec)
HV = kron(H_vec, V_vec)
VH = kron(V_vec, H_vec)
VV = kron(V_vec, V_vec)

def bell_state_phi_plus():
    """
    Ideal polarization Bell state:
      |Phi+> = (|HH> + |VV>)/sqrt(2).
    """
    state = (HH + VV) / np.sqrt(2.0)
    return normalize(state)


# ---------------------------------------------------------------------
# Laser + crystal: emission model
# ---------------------------------------------------------------------

def simulate_crystal_pulses(N_pulses, p_pair, rng):
    """
    Low-gain SPDC-like emission model:

    Each pulse:
      with probability p_pair: emits exactly one entangled pair,
      otherwise: no pair.

    Returns:
      list of pulse indices that emitted a pair.
    """
    emitted = []
    for pulse_idx in range(N_pulses):
        if rng.random() < p_pair:
            emitted.append(pulse_idx)
    return emitted


# ---------------------------------------------------------------------
# Polarization measurement and outcome sampling
# ---------------------------------------------------------------------

def polarization_eigenstates(theta):
    """
    Return the + and - eigenstates for a linear polarizer at angle theta.

    |theta>      = cos(theta)|H> + sin(theta)|V>  -> outcome +1
    |theta_perp> = -sin(theta)|H> + cos(theta)|V> -> outcome -1
    """
    c = np.cos(theta)
    s = np.sin(theta)
    ket_plus  = c * H_vec + s * V_vec
    ket_minus = -s * H_vec + c * V_vec
    ket_plus  = normalize(ket_plus)
    ket_minus = normalize(ket_minus)
    return ket_plus, ket_minus


def sample_entangled_pair_outcomes(state, theta_A, theta_B, rng):
    """
    Sample one outcome pair (a,b) from an ideal entangled state `state`
    for measurement settings (theta_A, theta_B).

    Outcomes a,b ∈ {+1, -1}.
    """
    state = normalize(state)

    # Local eigenstates
    kA_plus, kA_minus = polarization_eigenstates(theta_A)
    kB_plus, kB_minus = polarization_eigenstates(theta_B)

    # Two-photon basis kets for each outcome combination
    k_pp = kron(kA_plus,  kB_plus)
    k_pm = kron(kA_plus,  kB_minus)
    k_mp = kron(kA_minus, kB_plus)
    k_mm = kron(kA_minus, kB_minus)

    kets = [k_pp, k_pm, k_mp, k_mm]
    outcomes = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]

    probs = np.array([np.abs(np.vdot(k, state))**2 for k in kets], dtype=float)
    probs /= probs.sum()

    idx = rng.choice(4, p=probs)
    return outcomes[idx]


def sample_pair_with_visibility(state, theta_A, theta_B, V, rng):
    """
    Visibility model (Werner-like):

      with probability V:
         sample outcomes from the ideal entangled state `state`,
      with probability 1-V:
         sample completely random, uncorrelated outcomes (a,b).

    This effectively implements a Werner state mixture:
      ρ = V |Phi+><Phi+| + (1-V) (I/4).
    """
    if rng.random() < V:
        # Ideal entangled sample
        return sample_entangled_pair_outcomes(state, theta_A, theta_B, rng)
    else:
        # Completely random, uncorrelated ±1
        a = rng.choice([-1, +1])
        b = rng.choice([-1, +1])
        return a, b


# ---------------------------------------------------------------------
# Detector model
# ---------------------------------------------------------------------

def apply_detectors(a_true, b_true, eta_A, eta_B, dark_A, dark_B, rng):
    """
    Apply detector efficiency and dark counts.

    For each side:

      If a photon is present with true outcome (±1):

        With prob eta: register that true outcome.
        With prob (1-eta):
            with prob dark: register a dark-count outcome (random ±1).
            else: no click.

    We keep only events where BOTH sides registered exactly one click.

    Returns:
      (has_event, a_obs, b_obs)
      where has_event is a bool; if False, a_obs and b_obs are None.
    """
    # Alice
    click_A = False
    a_obs = None
    if rng.random() < eta_A:
        click_A = True
        a_obs = a_true
    else:
        # No true detection; maybe dark count
        if rng.random() < dark_A:
            click_A = True
            a_obs = rng.choice([-1, +1])

    # Bob
    click_B = False
    b_obs = None
    if rng.random() < eta_B:
        click_B = True
        b_obs = b_true
    else:
        if rng.random() < dark_B:
            click_B = True
            b_obs = rng.choice([-1, +1])

    if click_A and click_B:
        return True, a_obs, b_obs
    else:
        return False, None, None


# ---------------------------------------------------------------------
# CHSH from event data
# ---------------------------------------------------------------------

def compute_chsh_from_events(events):
    """
    Given a list of coincidence events (each a dict with keys:
      'setting_A', 'setting_B', 'a', 'b'),
    compute:

      E(A0,B0), E(A0,B1), E(A1,B0), E(A1,B1), and CHSH S.

    Returns:
      S, E_dict, counts_dict

      where E_dict[(sA,sB)] = E,
            counts_dict[(sA,sB)] = N_events for that setting.
    """
    groups = {
        ("A0", "B0"): [],
        ("A0", "B1"): [],
        ("A1", "B0"): [],
        ("A1", "B1"): [],
    }

    for ev in events:
        key = (ev["setting_A"], ev["setting_B"])
        if key in groups:
            groups[key].append(ev)

    E = {}
    counts = {}
    for key, data in groups.items():
        N = len(data)
        counts[key] = N
        if N == 0:
            E[key] = np.nan
            continue
        acc = 0.0
        for ev in data:
            acc += ev["a"] * ev["b"]
        E[key] = acc / N

    E00 = E[("A0", "B0")]
    E01 = E[("A0", "B1")]
    E10 = E[("A1", "B0")]
    E11 = E[("A1", "B1")]

    S = E00 + E01 + E10 - E11
    return S, E, counts


def summarize_correlations(events):
    """
    Summarize correlated vs anticorrelated outcomes per setting and overall.
    """
    groups = {
        ("A0", "B0"): [],
        ("A0", "B1"): [],
        ("A1", "B0"): [],
        ("A1", "B1"): [],
    }

    for ev in events:
        key = (ev["setting_A"], ev["setting_B"])
        if key in groups:
            groups[key].append(ev)

    per_setting = {}
    total_N = 0
    total_corr = 0
    total_anticorr = 0

    for key, data in groups.items():
        N = len(data)
        N_corr = sum(1 for ev in data if ev["a"] == ev["b"])
        N_anticorr = N - N_corr

        per_setting[key] = (N, N_corr, N_anticorr)

        total_N += N
        total_corr += N_corr
        total_anticorr += N_anticorr

    totals = (total_N, total_corr, total_anticorr)
    return per_setting, totals


# ---------------------------------------------------------------------
# Theoretical CHSH for Werner state
# ---------------------------------------------------------------------

def theoretical_CHSH_for_Werner(V):
    """
    For a Werner state built from |Phi+> with visibility V, the CHSH
    maximum (with the right angles) is:

      S_max = 2 * sqrt(2) * V
    """
    return 2.0 * np.sqrt(2.0) * V


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    rng = np.random.default_rng(123)

    # Experiment parameters
    N_pulses = 500_000      # number of pump pulses
    p_pair   = 0.05         # per-pulse probability to emit a pair

    # Imperfections
    V       = 0.95          # visibility (1.0 = perfect Bell; <1 mixes in noise)
    eta_A   = 0.85          # detector efficiency Alice
    eta_B   = 0.85          # detector efficiency Bob
    dark_A  = 1e-3          # dark count probability per pulse (Alice)
    dark_B  = 1e-3          # dark count probability per pulse (Bob)

    # CHSH angles (radians) that maximize S for |Phi+>
    angles = {
        "A0": 0.0,
        "A1": np.pi / 4.0,
        "B0": np.pi / 8.0,
        "B1": -np.pi / 8.0,
    }

    print("=== CHSH experiment with more realistic SPDC + detectors ===\n")
    print(f"Number of pump pulses      : {N_pulses}")
    print(f"Per-pulse pair probability : p_pair = {p_pair}")
    print(f"Visibility (Werner mix)    : V = {V}")
    print(f"Detector efficiency        : eta_A = {eta_A}, eta_B = {eta_B}")
    print(f"Dark count probabilities   : dark_A = {dark_A}, dark_B = {dark_B}")
    print()
    print("Analyzer angles (degrees):")
    print(f"  Alice: A0 = {angles['A0']*180/np.pi:6.2f}°,  A1 = {angles['A1']*180/np.pi:6.2f}°")
    print(f"  Bob:   B0 = {angles['B0']*180/np.pi:6.2f}°,  B1 = {angles['B1']*180/np.pi:6.2f}°")
    print()

    # Pulses that actually emit pairs
    emitted_pulses = simulate_crystal_pulses(N_pulses, p_pair, rng)
    n_pairs = len(emitted_pulses)
    print(f"Pulses that emitted a pair : {n_pairs}")
    print(f"Effective emission rate    : {n_pairs / N_pulses:.3f} per pulse")
    print()

    # Ideal Bell state produced when emission happens
    pair_state = bell_state_phi_plus()

    settings_A = ["A0", "A1"]
    settings_B = ["B0", "B1"]

    # Coincidence events (both detectors clicked)
    events = []

    for pulse_idx in emitted_pulses:
        # Random settings chosen per pulse (like real CHSH experiment)
        sA = rng.integers(0, 2)
        sB = rng.integers(0, 2)
        key_A = settings_A[sA]
        key_B = settings_B[sB]

        theta_A = angles[key_A]
        theta_B = angles[key_B]

        # Underlying "true" outcomes from Werner-like visibility model
        a_true, b_true = sample_pair_with_visibility(
            pair_state, theta_A, theta_B, V, rng
        )

        # Pass through detectors with inefficiency + dark counts
        has_event, a_obs, b_obs = apply_detectors(
            a_true, b_true, eta_A, eta_B, dark_A, dark_B, rng
        )

        if has_event:
            events.append({
                "pulse": pulse_idx,
                "setting_A": key_A,
                "setting_B": key_B,
                "a": a_obs,
                "b": b_obs,
            })

    n_events = len(events)
    print(f"Recorded coincidence events : {n_events}")
    if n_pairs > 0:
        print(f"Coincidence fraction (events / emitted pairs) = {n_events / n_pairs:.3f}")
    print()

    # CHSH from data
    S_data, E_data, counts = compute_chsh_from_events(events)
    print("Correlations estimated from coincidence data:")
    print(f"  E(A0,B0) ≈ {E_data[('A0','B0')]:.6f}  (N = {counts[('A0','B0')]})")
    print(f"  E(A0,B1) ≈ {E_data[('A0','B1')]:.6f}  (N = {counts[('A0','B1')]})")
    print(f"  E(A1,B0) ≈ {E_data[('A1','B0')]:.6f}  (N = {counts[('A1','B0')]})")
    print(f"  E(A1,B1) ≈ {E_data[('A1','B1')]:.6f}  (N = {counts[('A1','B1')]})")
    print()
    print(f"CHSH S from data ≈ {S_data:.6f}")
    print("  Local realism bound      : |S| <= 2")
    S_theory = theoretical_CHSH_for_Werner(V)
    print(f"  Ideal Werner CHSH (this V): S_max ≈ {S_theory:.6f}")
    print()

    # Correlation vs anticorrelation stats
    per_setting, totals = summarize_correlations(events)
    print("Event statistics per setting (correlated vs anticorrelated):")
    for key in [("A0", "B0"), ("A0", "B1"), ("A1", "B0"), ("A1", "B1")]:
        N, N_corr, N_anticorr = per_setting[key]
        if N == 0:
            frac_corr = 0.0
            frac_anticorr = 0.0
        else:
            frac_corr = N_corr / N
            frac_anticorr = N_anticorr / N
        print(f"  {key[0]},{key[1]}: N = {N:6d}, "
              f"corr = {N_corr:6d} ({frac_corr:.3f}), "
              f"anticorr = {N_anticorr:6d} ({frac_anticorr:.3f})")

    total_N, total_corr, total_anticorr = totals
    print("\nOverall coincidence statistics:")
    print(f"  Total coincidences      = {total_N}")
    if total_N > 0:
        print(f"  Total correlated        = {total_corr} ({total_corr/total_N:.3f})")
        print(f"  Total anticorrelated    = {total_anticorr} ({total_anticorr/total_N:.3f})")


if __name__ == "__main__":
    main()

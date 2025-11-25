#!/usr/bin/env python3
"""
twofermion3d_blob_classifier.py

Run the 3D two-excitation substrate engine and classify the emergent
"blob" using simple spatial + spin diagnostics:

- Spatial:
    * center-of-mass (x_cm, y_cm, z_cm)
    * effective radius r_eff
    * peak fraction (max rho_tot / sum rho_tot)
    * number of blobs (local maxima above a threshold)
    * normalized Shannon entropy of rho_tot

- Spin:
    * antisymmetry score
    * singlet fraction at overlap
    * CHSH S value

Requires: substrate_engine_3d.py in the same directory or on PYTHONPATH.
"""

import argparse
import math
from dataclasses import asdict
from typing import Dict, Any, Tuple, List

import numpy as np

# Import your engine
import substrate_engine_3d as se3d


# ---------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------

def compute_com_and_reff(
    rho_tot: np.ndarray,
    params: se3d.TwoFermion3DParams
) -> Tuple[Tuple[float, float, float], float]:
    """
    Compute center of mass (x_cm, y_cm, z_cm) and effective radius r_eff
    from a 3D site density rho_tot[r], r = 0..Ns-1.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz

    assert rho_tot.shape[0] == Ns, "rho_tot length must match Lx*Ly*Lz"

    # Coordinates on the lattice
    xs = np.empty(Ns, dtype=float)
    ys = np.empty(Ns, dtype=float)
    zs = np.empty(Ns, dtype=float)
    for r in range(Ns):
        x, y, z = se3d.site_coords_3d(r, Lx, Ly, Lz)
        xs[r] = x
        ys[r] = y
        zs[r] = z

    total = float(rho_tot.sum())
    if total <= 0.0:
        return (0.0, 0.0, 0.0), 0.0

    # Normalize
    p = rho_tot / total

    x_cm = float(np.sum(p * xs))
    y_cm = float(np.sum(p * ys))
    z_cm = float(np.sum(p * zs))

    # Effective radius (RMS distance to COM)
    dx2 = (xs - x_cm) ** 2
    dy2 = (ys - y_cm) ** 2
    dz2 = (zs - z_cm) ** 2
    r2 = dx2 + dy2 + dz2
    r_eff = float(np.sqrt(np.sum(p * r2)))

    return (x_cm, y_cm, z_cm), r_eff


def density_entropy(rho_tot: np.ndarray) -> float:
    """
    Normalized Shannon entropy of the site distribution.

    H_norm = -sum_i p_i log p_i / log(Ns), with p_i = rho_tot_i / sum rho_tot_i.
    """
    total = float(rho_tot.sum())
    if total <= 0.0:
        return 0.0

    p = rho_tot / total
    # avoid log(0)
    p_nonzero = p[p > 0.0]
    H = -float(np.sum(p_nonzero * np.log(p_nonzero)))
    Ns = rho_tot.size
    if Ns <= 1:
        return 0.0
    H_norm = H / math.log(Ns)
    return H_norm


def count_local_maxima(
    rho_tot: np.ndarray,
    params: se3d.TwoFermion3DParams,
    frac_threshold: float = 0.05
) -> int:
    """
    Count "blobs" as local maxima in rho_tot above a fractional threshold
    of the global maximum.

    - frac_threshold: sites with rho < frac_threshold * max_rho are ignored.
    - Neighborhood: 6 nearest neighbors in the 3D periodic lattice.

    This is a very simple heuristic but good enough to distinguish:
    * single concentrated blob vs multiple separated lobes vs noisy clutter.
    """
    Lx, Ly, Lz = params.Lx, params.Ly, params.Lz
    Ns = Lx * Ly * Lz
    assert rho_tot.shape[0] == Ns

    if Ns == 0:
        return 0

    max_rho = float(rho_tot.max())
    if max_rho <= 0.0:
        return 0

    threshold = frac_threshold * max_rho
    blobs = 0

    def neighbors(x: int, y: int, z: int) -> List[Tuple[int, int, int]]:
        xp = (x + 1) % Lx
        xm = (x - 1) % Lx
        yp = (y + 1) % Ly
        ym = (y - 1) % Ly
        zp = (z + 1) % Lz
        zm = (z - 1) % Lz
        return [
            (xp, y, z), (xm, y, z),
            (x, yp, z), (x, ym, z),
            (x, y, zp), (x, y, zm),
        ]

    for r in range(Ns):
        val = rho_tot[r]
        if val < threshold:
            continue

        x, y, z = se3d.site_coords_3d(r, Lx, Ly, Lz)

        is_max = True
        for xn, yn, zn in neighbors(x, y, z):
            rn = se3d.site_index_3d(xn, yn, zn, Lx, Ly, Lz)
            if rho_tot[rn] > val:
                is_max = False
                break

        if is_max:
            blobs += 1

    return blobs


# ---------------------------------------------------------------------
# Spin + blob classification
# ---------------------------------------------------------------------

def classify_spatial_blob(
    rho_tot: np.ndarray,
    params: se3d.TwoFermion3DParams
) -> Dict[str, Any]:
    """
    Use simple rules to classify the spatial structure of rho_tot as:

    - "single_central_blob"
    - "double_blob"
    - "multi_blob"
    - "spread_out"

    plus report quantitative features.
    """
    (x_cm, y_cm, z_cm), r_eff = compute_com_and_reff(rho_tot, params)
    H_norm = density_entropy(rho_tot)

    total = float(rho_tot.sum())
    peak = float(rho_tot.max())
    peak_fraction = peak / total if total > 0.0 else 0.0

    n_blobs = count_local_maxima(rho_tot, params, frac_threshold=0.05)

    # Simple rule-based classification (tunable)
    if n_blobs <= 0:
        blob_type = "empty"
    elif n_blobs == 1 and peak_fraction > 0.5 and H_norm < 0.5:
        blob_type = "single_central_blob"
    elif n_blobs == 2 and peak_fraction < 0.7:
        blob_type = "double_blob"
    elif n_blobs > 2 and H_norm < 0.8:
        blob_type = "multi_blob"
    else:
        blob_type = "spread_out"

    return {
        "blob_type": blob_type,
        "x_cm": x_cm,
        "y_cm": y_cm,
        "z_cm": z_cm,
        "r_eff": r_eff,
        "peak_fraction": peak_fraction,
        "entropy_norm": H_norm,
        "n_blobs": n_blobs,
    }


def classify_spin_sector(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify the spin / exchange behavior using antisymmetry + overlap +
    CHSH S.

    Rough categories:

    - "fermionic_singlet_bell":
        antisym_score > 0.95, singlet_fraction > 0.95, |S_chsh| > 2.0

    - "fermionic_singlet":
        antisym_score > 0.95, singlet_fraction > 0.95, |S_chsh| <= 2.0

    - "mixed_spin_entangled":
        singlet_fraction between 0.5 and 0.95, |S_chsh| > 2.0

    - "mixed_spin":
        0.1 < singlet_fraction < 0.9, |S_chsh| <= 2.0

    - "triplet_like":
        singlet_fraction < 0.1

    - "other":
        fallback.
    """
    anti = results["antisymmetry"]
    overlap = results["overlap"]
    S_chsh = float(results["S_chsh"])

    antisym_score = float(anti["antisym_score"])
    singlet_fraction = float(overlap["singlet_fraction"])
    abs_S = abs(S_chsh)

    if antisym_score > 0.95 and singlet_fraction > 0.95 and abs_S > 2.0:
        spin_type = "fermionic_singlet_bell"
    elif antisym_score > 0.95 and singlet_fraction > 0.95:
        spin_type = "fermionic_singlet"
    elif 0.5 <= singlet_fraction < 0.95 and abs_S > 2.0:
        spin_type = "mixed_spin_entangled"
    elif 0.1 < singlet_fraction < 0.9 and abs_S <= 2.0:
        spin_type = "mixed_spin"
    elif singlet_fraction < 0.1:
        spin_type = "triplet_like"
    else:
        spin_type = "other"

    return {
        "spin_type": spin_type,
        "antisym_score": antisym_score,
        "sym_score": float(anti["sym_score"]),
        "singlet_fraction": singlet_fraction,
        "overlap_prob": float(overlap["overlap_prob"]),
        "S_chsh": S_chsh,
        "abs_S_chsh": abs_S,
    }


def classify_twofermion3d_blob(
    params: se3d.TwoFermion3DParams
) -> Dict[str, Any]:
    """
    Run the two-fermion 3D experiment and classify the emergent "blob".
    """
    # Run the engine
    results = se3d.run_twofermion3d_experiment(params)

    rho_tot = results["rho_tot"]
    spatial = classify_spatial_blob(rho_tot, params)
    spin = classify_spin_sector(results)

    return {
        "params": asdict(params),
        "E0": float(results["E0"]),
        "E_gauss": float(results["E_gauss"]),
        "spatial": spatial,
        "spin": spin,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Classify emergent blobs in the 3D two-excitation substrate."
    )
    p.add_argument("--Lx", type=int, default=2)
    p.add_argument("--Ly", type=int, default=2)
    p.add_argument("--Lz", type=int, default=2)

    p.add_argument("--J-hop", type=float, default=1.0, dest="J_hop")
    p.add_argument("--mass", type=float, default=0.1)

    p.add_argument("--g-defrag", type=float, default=1.0)
    p.add_argument("--sigma-defrag", type=float, default=1.0)

    p.add_argument("--lambda-G", type=float, default=5.0, dest="lambda_G")
    p.add_argument("--lambda-S", type=float, default=-1.0, dest="lambda_S")
    p.add_argument("--lambda-T", type=float, default=0.0, dest="lambda_T")
    p.add_argument("--J-exch", type=float, default=1.0, dest="J_exch")

    p.add_argument("--max-eigsh-iter", type=int, default=5000)
    p.add_argument("--k-eigs", type=int, default=1)

    return p.parse_args()


def main():
    args = parse_args()

    params = se3d.TwoFermion3DParams(
        Lx=args.Lx,
        Ly=args.Ly,
        Lz=args.Lz,
        J_hop=args.J_hop,
        m=args.mass,
        g_defrag=args.g_defrag,
        sigma_defrag=args.sigma_defrag,
        lambda_G=args.lambda_G,
        lambda_S=args.lambda_S,
        lambda_T=args.lambda_T,
        J_exch=args.J_exch,
        max_eigsh_iter=args.max_eigsh_iter,
        k_eigs=args.k_eigs,
    )

    classification = classify_twofermion3d_blob(params)

    print("======================================================================")
    print("3D TWO-EXCITATION BLOB CLASSIFICATION")
    print("======================================================================")
    print("Parameters:")
    for k, v in classification["params"].items():
        print(f"  {k:15s} = {v}")
    print("----------------------------------------------------------------------")
    print(f"Ground state energy E0       = {classification['E0']:.6f}")
    print(f"Gauss-like energy <H_gauss>  = {classification['E_gauss']:.6f}")
    print("----------------------------------------------------------------------")

    spatial = classification["spatial"]
    print("Spatial blob classification:")
    print(f"  blob_type       = {spatial['blob_type']}")
    print(f"  COM (x,y,z)     = ({spatial['x_cm']:.3f}, "
          f"{spatial['y_cm']:.3f}, {spatial['z_cm']:.3f})")
    print(f"  r_eff           = {spatial['r_eff']:.3f}")
    print(f"  peak_fraction   = {spatial['peak_fraction']:.3f}")
    print(f"  entropy_norm    = {spatial['entropy_norm']:.3f}")
    print(f"  n_blobs (maxima)= {spatial['n_blobs']}")
    print("----------------------------------------------------------------------")

    spin = classification["spin"]
    print("Spin / exchange classification:")
    print(f"  spin_type       = {spin['spin_type']}")
    print(f"  antisym_score   = {spin['antisym_score']:.6f}")
    print(f"  singlet_fraction= {spin['singlet_fraction']:.6f}")
    print(f"  overlap_prob    = {spin['overlap_prob']:.6f}")
    print(f"  CHSH S          = {spin['S_chsh']:.6f}")
    print(f"  |S|             = {spin['abs_S_chsh']:.6f}")
    print("======================================================================")


if __name__ == "__main__":
    main()

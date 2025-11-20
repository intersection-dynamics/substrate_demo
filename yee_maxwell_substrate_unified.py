#!/usr/bin/env python
"""
Unified Yee–Maxwell Substrate Engine with Symmetry Selection (corrected EM update)

- Complex substrate field psi(x,y,t)
- 2D EM fields Ex, Ey, Bz (c = 1 units)
- Matter → EM coupling via charge currents Jx, Jy
- Defrag potential + optional symmetry selector:
    * symmetry = 'none'   → V_sym = 0
    * symmetry = 'fermion'→ V_sym = lambda_F * rho_smooth
    * symmetry = 'boson'  → V_sym = -alpha_B * rho_smooth + beta_B * rho_smooth^2

Outputs:
- npz snapshots:  yee_unified_snap_XXXXXX.npz
- energies CSV:   yee_unified_energies.csv

Example:
    python yee_maxwell_substrate_unified.py ^
        --Nx 64 --Ny 64 --dt 0.01 --steps 2000 ^
        --symmetry fermion --lambda_F 2.0 ^
        --g_defrag 0.3 --rho0 0.1 ^
        --out_dir yee_unified_output
"""

import os
import argparse
import time

# ------------------------------------------------------------
# Backend selection: CuPy if available, otherwise NumPy
# ------------------------------------------------------------
try:
    import cupy as cp  # type: ignore
    xp = cp
    USE_CUPY = True
except Exception:
    import numpy as np  # type: ignore
    xp = np
    USE_CUPY = False


# ------------------------------------------------------------
# Utility finite-difference helpers
# ------------------------------------------------------------
def roll(arr, shift, axis):
    """xp.roll wrapper (periodic boundary)"""
    return xp.roll(arr, shift, axis=axis)


def laplacian_2d(field, dx):
    """Periodic 2D Laplacian with second-order finite differences."""
    return (
        roll(field, 1, 0) + roll(field, -1, 0) +
        roll(field, 1, 1) + roll(field, -1, 1) -
        4.0 * field
    ) / (dx * dx)


def grad_2d(field, dx):
    """Return gradients (df/dx, df/dy) with central differences, periodic."""
    dfdx = (roll(field, -1, 0) - roll(field, 1, 0)) / (2.0 * dx)
    dfdy = (roll(field, -1, 1) - roll(field, 1, 1)) / (2.0 * dx)
    return dfdx, dfdy


def smooth_density(rho):
    """
    Simple 3x3 box smoothing of a scalar field.
    This plays the role of a coarse-grained density for symmetry selection.
    """
    acc = rho.copy()
    # axial neighbours
    for sx in (-1, 1):
        acc += roll(rho, sx, 0)
    for sy in (-1, 1):
        acc += roll(rho, sy, 1)
    # diagonal neighbours
    for sx in (-1, 1):
        for sy in (-1, 1):
            acc += roll(roll(rho, sx, 0), sy, 1)
    return acc / 9.0


# ------------------------------------------------------------
# Matter → EM coupling: currents
# ------------------------------------------------------------
def compute_currents(psi, dx, mass, charge):
    """
    Compute charge currents (Jx, Jy) from complex scalar field psi.

    Probability current for Schrödinger field:
        j = (1/m) Im(psi* ∇ psi)
    Charge current:
        J = q * j
    """
    dpsi_dx, dpsi_dy = grad_2d(psi, dx)
    psi_conj = xp.conj(psi)

    jx = (1.0 / mass) * xp.imag(psi_conj * dpsi_dx)
    jy = (1.0 / mass) * xp.imag(psi_conj * dpsi_dy)

    Jx = charge * jx
    Jy = charge * jy
    return Jx, Jy


# ------------------------------------------------------------
# Maxwell update (collocated grid, corrected signs)
# ------------------------------------------------------------
def update_maxwell(Ex, Ey, Bz, dt, dx, Jx, Jy):
    """
    Simple 2D FDTD-like Maxwell update with periodic boundaries.

    Units: c = 1, mu0 = eps0 = 1.

    Equations (TEz):
        ∂t Bz = - (∂x Ey - ∂y Ex)
        ∂t Ex =   (∂y Bz - Jx)
        ∂t Ey = - (∂x Bz + Jy)
    """
    # Update magnetic field Bz using curl E
    dEy_dx = (roll(Ey, -1, 0) - Ey) / dx
    dEx_dy = (roll(Ex, -1, 1) - Ex) / dx
    curlE = dEy_dx - dEx_dy
    Bz_new = Bz - dt * curlE

    # Update electric fields using curl Bz and currents
    dBz_dy = (roll(Bz_new, -1, 1) - Bz_new) / dx
    dBz_dx = (roll(Bz_new, -1, 0) - Bz_new) / dx

    # Corrected signs:
    Ex_new = Ex + dt * (dBz_dy - Jx)
    Ey_new = Ey - dt * (dBz_dx + Jy)

    return Ex_new, Ey_new, Bz_new


# ------------------------------------------------------------
# Symmetry selector potential
# ------------------------------------------------------------
def symmetry_potential(psi, mode, params):
    """
    Compute symmetry-selector potential V_sym(x) based on local density.

    mode:
        'none'   → V_sym = 0
        'fermion'→ V_sym = lambda_F * rho_smooth (linear repulsion)
        'boson'  → V_sym = -alpha_B * rho_smooth + beta_B * rho_smooth^2
    """
    rho = xp.abs(psi) ** 2
    rho_tilde = smooth_density(rho)

    if mode == "fermion":
        lam = params.get("lambda_F", 0.0)
        V = lam * rho_tilde
    elif mode == "boson":
        alpha = params.get("alpha_B", 0.0)
        beta = params.get("beta_B", 0.0)
        V = -alpha * rho_tilde + beta * rho_tilde ** 2
    else:
        V = xp.zeros_like(rho)

    return V, rho, rho_tilde


# ------------------------------------------------------------
# Energy bookkeeping
# ------------------------------------------------------------
def compute_energies(psi, Ex, Ey, Bz, dx, mass, g_defrag, rho0,
                     symm_mode, symm_params):
    """Compute total energy of matter + EM + potentials."""
    # Matter kinetic energy ~ |grad psi|^2 / (2m)
    dpsi_dx, dpsi_dy = grad_2d(psi, dx)
    grad_sq = xp.abs(dpsi_dx) ** 2 + xp.abs(dpsi_dy) ** 2
    E_kin = xp.sum(grad_sq) / (2.0 * mass) * dx * dx

    # Symmetry potential + density
    V_sym, rho, _ = symmetry_potential(psi, symm_mode, symm_params)

    # Defrag potential energy ~ 0.5 g (rho - rho0)^2
    E_defrag = 0.5 * g_defrag * xp.sum((rho - rho0) ** 2) * dx * dx

    # Symmetry energy: ∫ V_sym * rho d^2x
    E_symm = xp.sum(V_sym * rho) * dx * dx

    # EM energy density = 0.5 (E^2 + B^2)
    E_EM = 0.5 * xp.sum(Ex ** 2 + Ey ** 2 + Bz ** 2) * dx * dx

    E_total = float(E_kin + E_defrag + E_symm + E_EM)

    return E_total, float(E_kin), float(E_defrag), float(E_symm), float(E_EM)


# ------------------------------------------------------------
# Main simulation
# ------------------------------------------------------------
def run_sim(params):
    Nx = params["Nx"]
    Ny = params["Ny"]
    dx = params["dx"]
    dt = params["dt"]
    steps = params["steps"]
    out_every = params["out_every"]
    out_dir = params["out_dir"]

    mass = params["mass"]
    charge = params["charge"]
    g_defrag = params["g_defrag"]
    rho0 = params["rho0"]
    symm_mode = params["symm_mode"]
    symm_params = params["symm_params"]

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    shape = (Nx, Ny)

    # Initial substrate: small complex noise around zero
    psi = 0.1 * (xp.random.rand(*shape) + 1j * xp.random.rand(*shape))
    psi = psi - xp.mean(psi)

    # EM fields start at zero
    Ex = xp.zeros(shape, dtype=xp.float64)
    Ey = xp.zeros(shape, dtype=xp.float64)
    Bz = xp.zeros(shape, dtype=xp.float64)

    energies = []

    t0 = time.time()
    for n in range(steps + 1):
        t = n * dt

        # Diagnostics + snapshots
        if n % out_every == 0:
            E_tot, E_kin, E_defrag, E_symm, E_EM = compute_energies(
                psi, Ex, Ey, Bz, dx, mass, g_defrag, rho0,
                symm_mode, symm_params
            )
            energies.append((t, E_tot, E_kin, E_defrag, E_symm, E_EM))
            print(f"[STEP {n:6d}] t={t:8.3f}  "
                  f"E_tot={E_tot:.6e}  E_EM={E_EM:.6e}")

            # simple safety check: if it’s obviously blowing, stop
            if not (xp.isfinite(E_tot) and xp.isfinite(E_EM)):
                print("[WARN] Non-finite energy detected, stopping early.")
                break

            import numpy as np  # local import
            snap = {
                "t": float(t),
                "psi": xp.asnumpy(psi) if USE_CUPY else np.array(psi),
                "Ex": xp.asnumpy(Ex) if USE_CUPY else np.array(Ex),
                "Ey": xp.asnumpy(Ey) if USE_CUPY else np.array(Ey),
                "Bz": xp.asnumpy(Bz) if USE_CUPY else np.array(Bz),
            }
            np.savez(
                os.path.join(out_dir, f"yee_unified_snap_{n:06d}.npz"),
                **snap
            )

        if n == steps:
            break

        # -------- Matter update (explicit Schr-like) --------
        V_sym, rho, _ = symmetry_potential(psi, symm_mode, symm_params)
        V_defrag = g_defrag * (rho - rho0)
        V_eff = V_defrag + V_sym

        lap_psi = laplacian_2d(psi, dx)

        # i ∂t ψ = -(1/2m) ∇²ψ + V_eff ψ
        rhs = (1j / (2.0 * mass)) * lap_psi - 1j * V_eff * psi
        psi = psi + dt * rhs

        # -------- Maxwell update --------
        Jx, Jy = compute_currents(psi, dx, mass, charge)
        Ex, Ey, Bz = update_maxwell(Ex, Ey, Bz, dt, dx, Jx, Jy)

    # Save energies
    import numpy as np
    energies = np.array(energies)
    np.savetxt(
        os.path.join(out_dir, "yee_unified_energies.csv"),
        energies,
        delimiter=",",
        header="t,E_total,E_kin,E_defrag,E_symm,E_EM",
        comments=""
    )

    print(f"[DONE] steps={len(energies)-1}, dt={dt}, "
          f"runtime={time.time() - t0:.2f} s. "
          f"Energies → {out_dir}/yee_unified_energies.csv")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unified Yee-Maxwell Substrate Engine with Symmetry Selection"
    )
    parser.add_argument("--Nx", type=int, default=64, help="Grid size in x")
    parser.add_argument("--Ny", type=int, default=64, help="Grid size in y")
    parser.add_argument("--dx", type=float, default=1.0, help="Lattice spacing")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    parser.add_argument("--steps", type=int, default=4000,
                        help="Number of time steps")
    parser.add_argument("--out_every", type=int, default=200,
                        help="Diagnostics / snapshot interval")

    parser.add_argument("--mass", type=float, default=1.0,
                        help="Matter mass parameter")
    parser.add_argument("--charge", type=float, default=1.0,
                        help="Matter charge")
    parser.add_argument("--g_defrag", type=float, default=0.1,
                        help="Defrag potential strength")
    parser.add_argument("--rho0", type=float, default=0.1,
                        help="Preferred density for defrag potential")

    parser.add_argument("--symmetry", type=str, default="none",
                        choices=["none", "fermion", "boson"],
                        help="Symmetry selector mode")
    parser.add_argument("--lambda_F", type=float, default=0.0,
                        help="Fermion-like repulsion strength")
    parser.add_argument("--alpha_B", type=float, default=0.0,
                        help="Boson-like attraction parameter")
    parser.add_argument("--beta_B", type=float, default=0.0,
                        help="Boson-like saturation parameter")

    parser.add_argument("--out_dir", type=str, default="yee_unified_output",
                        help="Output directory")

    args = parser.parse_args()

    symm_params = {
        "lambda_F": args.lambda_F,
        "alpha_B": args.alpha_B,
        "beta_B": args.beta_B,
    }

    params = {
        "Nx": args.Nx,
        "Ny": args.Ny,
        "dx": args.dx,
        "dt": args.dt,
        "steps": args.steps,
        "out_every": args.out_every,
        "mass": args.mass,
        "charge": args.charge,
        "g_defrag": args.g_defrag,
        "rho0": args.rho0,
        "symm_mode": args.symmetry,
        "symm_params": symm_params,
        "out_dir": args.out_dir,
    }

    backend = "CuPy (GPU)" if USE_CUPY else "NumPy (CPU)"
    print("[INIT] Unified Yee-Maxwell Substrate Engine")
    print(f"[INIT] Backend: {backend}")
    print(f"[INIT] Grid: {args.Nx} x {args.Ny}, dx={args.dx}, dt={args.dt}")
    print(f"[INIT] Symmetry mode: {args.symmetry}, params={symm_params}")

    run_sim(params)


if __name__ == "__main__":
    main()

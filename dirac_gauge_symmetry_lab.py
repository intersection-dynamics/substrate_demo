#!/usr/bin/env python3
"""
dirac_gauge_symmetry_lab.py

Gauge-symmetry lab built on top of yee_dirac_substrate.py.

Goal:
  - Use the existing 2+1D Dirac+Maxwell (Yee) engine as a base.
  - Turn on EM backreaction (vector potential A).
  - Evolve the system for a short time.
  - Take a snapshot (Psi, Ax, Ay) and construct a local gauge phase χ(x,y).
  - Apply a U(1) gauge transformation:
        Psi' = exp(i q χ) Psi
        Ax'  = Ax + ∂_x χ
        Ay'  = Ay + ∂_y χ
  - Compute the Dirac energy expectation using the covariant derivative
    before and after the gauge transform, and compare.

If the discrete covariant derivative and minimal coupling are implemented
correctly, the Dirac energy should be (up to numerical error)
invariant under this local gauge transformation.

This script is NOT about emergent gauge symmetry; it's about making
the existing explicit gauge structure concrete and testable.
"""

import argparse
import os
import time

import numpy as np

import yee_dirac_substrate as yds


def parse_args():
    p = argparse.ArgumentParser(
        description="Gauge-symmetry check on the 2+1D Dirac–Maxwell substrate."
    )

    # Grid and evolution parameters
    p.add_argument("--Nx", type=int, default=32, help="Grid size in x")
    p.add_argument("--Ny", type=int, default=32, help="Grid size in y")
    p.add_argument("--dx", type=float, default=1.0, help="Spatial grid spacing")
    p.add_argument("--dt", type=float, default=0.01, help="Time step")
    p.add_argument("--steps", type=int, default=200, help="Number of warmup steps")

    # Physical parameters
    p.add_argument("--c", type=float, default=1.0, help="Signal speed (c=1 units)")
    p.add_argument("--q", type=float, default=1.0, help="Charge of Dirac field")
    p.add_argument("--m", type=float, default=0.1, help="Dirac mass")

    # Defrag options (we can leave off by default; not needed for gauge test)
    p.add_argument("--defrag", action="store_true",
                   help="Enable defrag scalar field phi.")
    p.add_argument("--defrag-kappa", type=float, default=0.1,
                   help="Defrag relaxation rate.")
    p.add_argument("--defrag-lambda", type=float, default=1.0,
                   help="Defrag Laplacian weight.")
    p.add_argument("--defrag-g", type=float, default=0.5,
                   help="Coupling strength of defrag field to mass term.")

    # Gauge phase parameters
    p.add_argument("--chi-amp", type=float, default=0.5,
                   help="Amplitude of the gauge phase χ(x,y).")
    p.add_argument("--chi-mode-x", type=int, default=1,
                   help="Mode number in x for χ(x,y) = amp * sin(2π n_x x / Lx).")
    p.add_argument("--chi-mode-y", type=int, default=1,
                   help="Mode number in y for χ(x,y) = amp * sin(2π n_y y / Ly).")

    # Output options
    p.add_argument("--out-dir", type=str, default="gauge_lab_output",
                   help="Directory for small diagnostic outputs.")

    return p.parse_args()


def build_chi_field(Ny, Nx, dx, amp, mode_x, mode_y, xp):
    """
    Build a smooth gauge phase field χ(x,y) = amp * sin(2π n_x x/Lx) * sin(2π n_y y/Ly).

    Returns χ as an array of shape (Ny, Nx) in the xp backend (numpy or cupy).
    """
    Lx = Nx * dx
    Ly = Ny * dx

    # Coordinates (in xp backend)
    x = xp.arange(Nx) * dx
    y = xp.arange(Ny) * dx
    X, Y = xp.meshgrid(x, y)

    kx = 2.0 * np.pi * mode_x / Lx
    ky = 2.0 * np.pi * mode_y / Ly

    chi = amp * xp.sin(kx * X) * xp.sin(ky * Y)
    return chi


def dirac_energy_covariant(Psi, dx, c, m, alpha1, alpha2, beta,
                           Ax, Ay, q, phi, g_phi, xp):
    """
    Compute the Dirac energy expectation with covariant derivative,
    using the same Dirac Hamiltonian as in yee_dirac_substrate, but now
    including the vector potential via minimal coupling.

    Approximate:
        E_D = Re ∫ Ψ† H Ψ d^2x

    where H Ψ is given by dirac_hamiltonian_action().
    """
    Ny, Nx = Psi.shape[1], Psi.shape[2]
    dA = dx * dx

    H_Psi = yds.dirac_hamiltonian_action(
        Psi, dx, c, m, alpha1, alpha2, beta, xp,
        Ax=Ax, Ay=Ay, q=q,
        phi=phi, g_phi=g_phi
    )

    Psi_flat = Psi.reshape(2, Ny * Nx)
    H_Psi_flat = H_Psi.reshape(2, Ny * Nx)

    Psi_dag_flat = xp.conjugate(Psi_flat).T  # (Ny*Nx,2)
    # inner product Ψ† H Ψ over spinor components
    density_flat = xp.sum(Psi_dag_flat * H_Psi_flat.T, axis=1)
    E_complex = xp.sum(density_flat) * dA

    return float(E_complex.real)


def run_gauge_symmetry_lab(args):
    xp = yds.xp
    XP_BACKEND = yds.XP_BACKEND

    Nx = args.Nx
    Ny = args.Ny
    dx = args.dx
    dt = args.dt
    steps = args.steps
    c = args.c
    q = args.q
    m = args.m

    defrag = args.defrag
    defrag_kappa = args.defrag_kappa
    defrag_lambda = args.defrag_lambda
    defrag_g = args.defrag_g if defrag else 0.0

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("======================================================================")
    print("DIRAC GAUGE SYMMETRY LAB")
    print("======================================================================")
    print(f"[INIT] Backend      : {XP_BACKEND}")
    print(f"[INIT] Grid         : Nx={Nx}, Ny={Ny}, dx={dx}, dt={dt}")
    print(f"[INIT] Steps (warm) : {steps}")
    print(f"[INIT] Parameters   : c={c}, q={q}, m={m}")
    if defrag:
        print(f"[INIT] Defrag ON    : kappa={defrag_kappa}, lambda={defrag_lambda}, g_phi={defrag_g}")
    else:
        print("[INIT] Defrag OFF")
    print(f"[INIT] Gauge phase  : chi_amp={args.chi_amp}, modes=({args.chi_mode_x},{args.chi_mode_y})")
    print("======================================================================")

    # --- Build Dirac matrices
    sigma_x, sigma_y, sigma_z, gamma0, gamma1, gamma2, alpha1, alpha2, beta = \
        yds.build_dirac_matrices(xp)

    # --- Initialize Dirac spinor and EM fields
    Psi_n = yds.init_spinor(Ny, Nx, dx, xp, m)     # Ψ^0
    Ex, Ey, Bz = yds.init_em_fields(Ny, Nx, xp)    # EM fields

    # Vector potential A for EM->Dirac backreaction (temporal gauge: dA/dt = -E)
    Ax = xp.zeros_like(Ex)
    Ay = xp.zeros_like(Ey)

    # Defrag scalar field (optional)
    if defrag:
        phi = xp.zeros((Ny, Nx), dtype=xp.float64)
    else:
        phi = None

    # One-step Euler to get Ψ^1 for leapfrog
    H_Psi = yds.dirac_hamiltonian_action(
        Psi_n, dx, c, m, alpha1, alpha2, beta, xp,
        Ax=Ax, Ay=Ay, q=q,
        phi=phi, g_phi=defrag_g
    )
    Psi_np1 = Psi_n - 1j * dt * H_Psi

    # Normalize (we care about rays)
    norm = xp.sqrt(xp.sum(xp.abs(Psi_np1) ** 2))
    if norm != 0:
        Psi_np1 /= norm

    Psi_nm1 = Psi_n
    Psi_n = Psi_np1

    # --- Time loop: warm up the system
    eps0 = 1.0
    print("[INFO] Warming up system with coupled Dirac+EM evolution...")
    t0 = time.time()
    for step in range(1, steps + 1):
        t = step * dt

        # Current from Psi_n
        rho, Jx, Jy = yds.compute_spinor_current(Psi_n, alpha1, alpha2, q, c, xp)

        # Update EM fields
        Ex, Ey, Bz = yds.update_em_fields(Ex, Ey, Bz, Jx, Jy, dx, dt, c, eps0, xp)

        # Update vector potential (temporal gauge: dA/dt = -E)
        Ax = Ax - dt * Ex
        Ay = Ay - dt * Ey

        # Defrag scalar evolution
        if defrag:
            rho_sm = rho + defrag_lambda * yds.laplacian(rho, dx)
            phi = phi + dt * (-defrag_kappa * (phi - rho_sm))

        # Dirac Hamiltonian at Psi_n (with A and optional phi)
        H_Psi_n = yds.dirac_hamiltonian_action(
            Psi_n, dx, c, m, alpha1, alpha2, beta, xp,
            Ax=Ax, Ay=Ay, q=q,
            phi=phi, g_phi=defrag_g
        )

        # Leapfrog step
        Psi_new = Psi_nm1 - 2j * dt * H_Psi_n

        # Normalize
        norm = xp.sqrt(xp.sum(xp.abs(Psi_new) ** 2))
        if norm != 0:
            Psi_new /= norm

        Psi_nm1, Psi_n = Psi_n, Psi_new

        if step % max(1, steps // 5) == 0:
            print(f"[WARMUP] step={step:6d} t={t:7.3f}")

    t1 = time.time()
    print(f"[INFO] Warmup complete in {t1 - t0:.2f} s")

    # --- Snapshot before gauge transform
    print("[INFO] Computing Dirac covariant energy before gauge transform...")
    E_D_before = dirac_energy_covariant(
        Psi_n, dx, c, m, alpha1, alpha2, beta,
        Ax, Ay, q, phi, defrag_g, xp
    )
    norm_before = float(np.sqrt(float(yds.xp.sum(yds.xp.abs(Psi_n) ** 2))))

    # --- Build gauge phase χ(x,y)
    chi = build_chi_field(Ny, Nx, dx,
                          amp=args.chi_amp,
                          mode_x=args.chi_mode_x,
                          mode_y=args.chi_mode_y,
                          xp=xp)

    # --- Compute gradient of χ for A-transform
    dchi_dx = yds.central_diff_x(chi, dx)
    dchi_dy = yds.central_diff_y(chi, dx)

    # --- Apply gauge transform:
    #       Psi' = exp(i q χ) Psi
    #       Ax'  = Ax + ∂_x χ
    #       Ay'  = Ay + ∂_y χ
    print("[INFO] Applying local U(1) gauge transformation...")
    phase = xp.exp(1j * q * chi)
    Psi_prime = Psi_n * phase  # broadcast over spinor components

    Ax_prime = Ax + dchi_dx
    Ay_prime = Ay + dchi_dy

    # --- Compute Dirac covariant energy after gauge transform
    print("[INFO] Computing Dirac covariant energy after gauge transform...")
    E_D_after = dirac_energy_covariant(
        Psi_prime, dx, c, m, alpha1, alpha2, beta,
        Ax_prime, Ay_prime, q, phi, defrag_g, xp
    )
    norm_after = float(np.sqrt(float(yds.xp.sum(yds.xp.abs(Psi_prime) ** 2))))

    # --- Diagnostics
    dE = E_D_after - E_D_before
    rel_err = dE / E_D_before if E_D_before != 0.0 else float("nan")

    print("======================================================================")
    print("GAUGE SYMMETRY NUMERICAL CHECK (COVARIANT DIRAC ENERGY)")
    print("======================================================================")
    print(f"Dirac energy before gauge transform : E_D = {E_D_before:.10e}")
    print(f"Dirac energy after  gauge transform : E_D' = {E_D_after:.10e}")
    print(f"Difference ΔE = E_D' - E_D           : ΔE  = {dE:.10e}")
    print(f"Relative error ΔE / E_D              :     = {rel_err:.10e}")
    print()
    print(f"Norm ||Psi|| before = {norm_before:.10e}")
    print(f"Norm ||Psi'|| after = {norm_after:.10e}")
    print("======================================================================")
    print("Interpretation:")
    print("  - In the continuum, exact gauge invariance means E_D' = E_D exactly.")
    print("  - On a discrete grid, small ΔE is expected from finite-difference and")
    print("    leapfrog approximations.")
    print("  - If ΔE is numerically very small (e.g. ≪ 1), this supports that")
    print("    the Dirac+EM minimal coupling is correctly implementing local")
    print("    U(1) gauge symmetry on this lattice.")
    print("======================================================================")


def main():
    args = parse_args()
    run_gauge_symmetry_lab(args)


if __name__ == "__main__":
    main()

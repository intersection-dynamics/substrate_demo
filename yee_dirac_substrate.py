#!/usr/bin/env python
"""
yee_dirac_substrate.py

2+1D Dirac spinor substrate coupled to 2D Maxwell fields on a Yee-like grid,
with optional defrag scalar potential and controllable spinor renormalization.

- Dirac spinor Ψ(x, y, t) has 2 components (2+1D Dirac representation):
    gamma^0 = sigma_z
    gamma^1 = i sigma_y
    gamma^2 = -i sigma_x

- Hamiltonian (D2 choice):
    i ∂_t Ψ = [ -i c (alpha^1 ∂_x + alpha^2 ∂_y) + m_eff(x,y,t) c^2 beta ] Ψ
    with alpha^1 = gamma^0 gamma^1 = -sigma_x
         alpha^2 = gamma^0 gamma^2 =  sigma_y
         beta    = gamma^0          =  sigma_z

  Here m_eff(x,y,t) = m + g_phi * phi(x,y,t) if defrag is enabled, else m.

- Maxwell fields on a 2D Yee-like grid (TE-like):
    Fields: Ex, Ey, Bz
    Update:
        ∂_t B_z = - (∂_x E_y - ∂_y E_x)
        ∂_t E_x =  c^2 ∂_y B_z - J_x / eps0
        ∂_t E_y = -c^2 ∂_x B_z - J_y / eps0

- Spinor current:
    rho  = q Ψ† Ψ
    J_x  = q c Ψ† alpha^1 Ψ
    J_y  = q c Ψ† alpha^2 Ψ

Optional extras:
  - Vector potential A (Ax, Ay) for minimal EM->Dirac backreaction (temporal gauge).
  - Defrag scalar field φ with simple relaxation dynamics:
        ∂_t φ = -kappa * (φ - (ρ + λ ∇²ρ))

Time stepping:
  - Dirac: leapfrog for the spinor (second-order, time-symmetric).
  - EM: explicit FDTD-like scheme (Yee-like, central differences).
  - φ: simple forward Euler relaxation.

Outputs:
  - dirac_energies.csv with columns:
      step,time,norm,E_total,E_dirac,E_em
  - dirac_snap_XXXXXX.npz with fields:
      t, density, spin_z, Ex, Ey, Bz,
      and optionally: phi, density_block, spin_z_block

"""

import os
import time
import argparse

# Try to use CuPy if available; otherwise fall back to NumPy.
try:
    import cupy as xp
    XP_BACKEND = "cupy"
except Exception:
    import numpy as xp
    XP_BACKEND = "numpy"


def parse_args():
    p = argparse.ArgumentParser(description="2+1D Dirac–Maxwell substrate (D2 Dirac choice, m=0.1).")
    p.add_argument("--Nx", type=int, default=64, help="Grid size in x")
    p.add_argument("--Ny", type=int, default=64, help="Grid size in y")
    p.add_argument("--dx", type=float, default=1.0, help="Spatial grid spacing")
    p.add_argument("--dt", type=float, default=0.01, help="Time step")
    p.add_argument("--steps", type=int, default=2000, help="Number of time steps")
    p.add_argument("--out_every", type=int, default=200, help="Snapshot/diagnostic interval")
    p.add_argument("--c", type=float, default=1.0, help="Signal speed (set c=1 units)")
    p.add_argument("--q", type=float, default=1.0, help="Charge of Dirac field")
    p.add_argument("--m", type=float, default=0.1, help="Dirac mass (in c=1 units)")
    p.add_argument("--out_dir", type=str, default="dirac_output", help="Output directory")

    # EM->Dirac backreaction via vector potential
    p.add_argument("--em_backreaction", action="store_true",
                   help="Include EM->Dirac minimal coupling via a vector potential A (temporal gauge).")

    # Spinor renormalization control
    p.add_argument("--no_renorm", action="store_true",
                   help="Disable global spinor renormalization each time step.")

    # Defrag scalar potential options
    p.add_argument("--defrag", action="store_true",
                   help="Enable defrag scalar field phi coupled into the mass term.")
    p.add_argument("--defrag_kappa", type=float, default=0.1,
                   help="Relaxation rate for defrag scalar field phi.")
    p.add_argument("--defrag_lambda", type=float, default=1.0,
                   help="Laplacian smoothing weight for density in defrag.")
    p.add_argument("--defrag_g", type=float, default=0.5,
                   help="Coupling strength of defrag potential into mass term (m_eff = m + g_phi * phi).")

    # Coarse-graining
    p.add_argument("--block_size", type=int, default=4,
                   help="Block size for coarse-grained density/spin maps in snapshots.")

    return p.parse_args()


def central_diff_x(f, dx):
    """
    Central difference derivative in x with periodic BC.
    f: array(..., Ny, Nx)
    """
    return (xp.roll(f, -1, axis=-1) - xp.roll(f, 1, axis=-1)) / (2.0 * dx)


def central_diff_y(f, dx):
    """
    Central difference derivative in y with periodic BC.
    f: array(..., Ny, Nx)
    """
    return (xp.roll(f, -1, axis=-2) - xp.roll(f, 1, axis=-2)) / (2.0 * dx)


def laplacian(f, dx):
    """
    2D Laplacian with periodic BC:
        ∇² f ≈ (f_{i+1,j} + f_{i-1,j} + f_{i,j+1} + f_{i,j-1} - 4 f_{i,j}) / dx^2
    """
    return (
        xp.roll(f, 1, axis=-2)
        + xp.roll(f, -1, axis=-2)
        + xp.roll(f, 1, axis=-1)
        + xp.roll(f, -1, axis=-1)
        - 4.0 * f
    ) / (dx * dx)


def build_dirac_matrices(xp_mod):
    """
    Build 2x2 Pauli matrices and Dirac matrices for 2+1D representation:

        gamma^0 = sigma_z
        gamma^1 = i sigma_y
        gamma^2 = -i sigma_x

    Then alpha^i = gamma^0 gamma^i:
        alpha^1 = -sigma_x
        alpha^2 =  sigma_y
        beta    =  gamma^0 = sigma_z
    """
    sigma_x = xp_mod.array([[0, 1],
                            [1, 0]], dtype=xp_mod.complex128)
    sigma_y = xp_mod.array([[0, -1j],
                            [1j, 0]], dtype=xp_mod.complex128)
    sigma_z = xp_mod.array([[1, 0],
                            [0, -1]], dtype=xp_mod.complex128)

    gamma0 = sigma_z
    gamma1 = 1j * sigma_y
    gamma2 = -1j * sigma_x

    alpha1 = gamma0 @ gamma1
    alpha2 = gamma0 @ gamma2
    beta = gamma0

    return sigma_x, sigma_y, sigma_z, gamma0, gamma1, gamma2, alpha1, alpha2, beta


def init_spinor(Ny, Nx, dx, xp_mod, m):
    """
    Initialize a localized Gaussian spinor packet in the upper component:

        Ψ = [ψ_up, ψ_down]^T

    Take ψ_up as a Gaussian, ψ_down = 0 initially.
    """
    y = xp_mod.arange(Ny) * dx
    x = xp_mod.arange(Nx) * dx
    X, Y = xp_mod.meshgrid(x, y)

    x0 = 0.5 * Nx * dx
    y0 = 0.5 * Ny * dx
    sigma = 4.0 * dx

    psi_up = xp_mod.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * sigma ** 2))
    psi_down = xp_mod.zeros_like(psi_up)

    Psi = xp_mod.stack([psi_up, psi_down], axis=0)  # shape (2, Ny, Nx)

    # Global normalization to unit norm
    norm_sq = xp_mod.sum(xp_mod.abs(Psi) ** 2)
    if norm_sq > 0:
        Psi /= xp_mod.sqrt(norm_sq)

    return Psi


def init_em_fields(Ny, Nx, xp_mod):
    """
    Initialize EM fields to zero.
    """
    Ex = xp_mod.zeros((Ny, Nx), dtype=xp_mod.float64)
    Ey = xp_mod.zeros((Ny, Nx), dtype=xp_mod.float64)
    Bz = xp_mod.zeros((Ny, Nx), dtype=xp_mod.float64)
    return Ex, Ey, Bz


def dirac_hamiltonian_action(
    Psi,
    dx,
    c,
    m,
    alpha1,
    alpha2,
    beta,
    xp_mod,
    Ax=None,
    Ay=None,
    q=0.0,
    phi=None,
    g_phi=0.0,
):
    """
    Compute H Ψ for the Dirac Hamiltonian:
        H = -i c (alpha^1 ∂_x + alpha^2 ∂_y) + m_eff c^2 beta

    where m_eff(x,y) = m + g_phi * phi(x,y) if phi is provided, else m.

    Psi shape: (2, Ny, Nx)
    Returns H_Psi shape: (2, Ny, Nx).
    """
    Ny, Nx = Psi.shape[1], Psi.shape[2]

    # Derivatives of each component
    dPsi_dx = central_diff_x(Psi, dx)
    dPsi_dy = central_diff_y(Psi, dx)

    # Optional minimal coupling to a vector potential A via covariant derivative:
    #   ∂_i -> ∂_i - i q A_i
    if Ax is not None and Ay is not None and q != 0.0:
        dPsi_dx = dPsi_dx - 1j * q * Ax * Psi
        dPsi_dy = dPsi_dy - 1j * q * Ay * Psi

    # Flatten spinor components for matrix operations: (2, Ny*Nx)
    Psi_flat = Psi.reshape(2, Ny * Nx)
    dPsi_dx_flat = dPsi_dx.reshape(2, Ny * Nx)
    dPsi_dy_flat = dPsi_dy.reshape(2, Ny * Nx)

    # alpha1 dPsi_dx + alpha2 dPsi_dy
    term_flat = alpha1 @ dPsi_dx_flat + alpha2 @ dPsi_dy_flat

    kinetic_flat = -1j * c * term_flat

    # Mass term: m_eff c^2 beta Ψ
    beta_Psi_flat = beta @ Psi_flat

    if phi is not None and g_phi != 0.0:
        phi_flat = phi.reshape(Ny * Nx)
        m_eff_flat = m + g_phi * phi_flat
        mass_flat = (c ** 2) * (beta_Psi_flat * m_eff_flat[xp_mod.newaxis, :])
    else:
        mass_flat = m * (c ** 2) * beta_Psi_flat

    H_flat = kinetic_flat + mass_flat
    H_Psi = H_flat.reshape(2, Ny, Nx)

    return H_Psi


def compute_spinor_current(Psi, alpha1, alpha2, q, c, xp_mod):
    """
    Compute charge density rho and currents Jx, Jy from spinor Ψ:

        rho  = q Ψ† Ψ
        J_x  = q c Ψ† alpha1 Ψ
        J_y  = q c Ψ† alpha2 Ψ

    Psi shape: (2, Ny, Nx)
    Returns rho, Jx, Jy (all real arrays shape (Ny, Nx))
    """
    Ny, Nx = Psi.shape[1], Psi.shape[2]
    Psi_flat = Psi.reshape(2, Ny * Nx)
    Psi_dag_flat = xp_mod.conjugate(Psi_flat).T  # shape (Ny*Nx, 2)

    # rho = q Ψ† Ψ
    density_flat = xp_mod.sum(Psi_dag_flat * Psi_flat.T, axis=1).real
    rho = (q * density_flat).reshape(Ny, Nx)

    # Currents: J_i = q c Ψ† alpha^i Ψ
    alpha1_Psi_flat = alpha1 @ Psi_flat
    alpha2_Psi_flat = alpha2 @ Psi_flat

    Jx_flat = xp_mod.sum(Psi_dag_flat * alpha1_Psi_flat.T, axis=1).real
    Jy_flat = xp_mod.sum(Psi_dag_flat * alpha2_Psi_flat.T, axis=1).real

    Jx = (q * c * Jx_flat).reshape(Ny, Nx)
    Jy = (q * c * Jy_flat).reshape(Ny, Nx)

    return rho, Jx, Jy


def update_em_fields(Ex, Ey, Bz, Jx, Jy, dx, dt, c, eps0, xp_mod):
    """
    Update EM fields using a simple FDTD-like scheme:

        ∂_t B_z = - (∂_x E_y - ∂_y E_x)
        ∂_t E_x =  c^2 ∂_y B_z - J_x / eps0
        ∂_t E_y = -c^2 ∂_x B_z - J_y / eps0

    All fields shape: (Ny, Nx)
    """
    curlE = central_diff_x(Ey, dx) - central_diff_y(Ex, dx)
    Bz_new = Bz - dt * curlE

    dBz_dy = central_diff_y(Bz_new, dx)
    dBz_dx = central_diff_x(Bz_new, dx)

    Ex_new = Ex + dt * (c ** 2 * dBz_dy - Jx / eps0)
    Ey_new = Ey + dt * (-c ** 2 * dBz_dx - Jy / eps0)

    return Ex_new, Ey_new, Bz_new


def compute_energies(Psi, Ex, Ey, Bz, dx, m, c, beta, xp_mod):
    """
    Compute approximate energies:

    - Dirac "mass" energy-like quantity:
        E_Dirac = ∫ m c^2 Ψ† beta Ψ d^2x
      (note: this ignores kinetic pieces; it's a diagnostic, not a full Hamiltonian)

    - EM energy:
        E_EM = ∫ 0.5 (E^2 + B^2) d^2x

    Returns E_total = E_Dirac + E_EM, E_Dirac, E_EM (all Python floats).
    """
    Ny, Nx = Psi.shape[1], Psi.shape[2]
    Psi_flat = Psi.reshape(2, Ny * Nx)
    Psi_dag_flat = xp_mod.conjugate(Psi_flat).T

    beta_Psi_flat = beta @ Psi_flat
    mass_density_flat = xp_mod.sum(Psi_dag_flat * beta_Psi_flat.T, axis=1).real

    dA = dx * dx
    E_dirac = m * (c ** 2) * xp_mod.sum(mass_density_flat) * dA

    E2 = Ex ** 2 + Ey ** 2
    B2 = Bz ** 2
    E_em = 0.5 * xp_mod.sum(E2 + B2) * dA

    E_total = E_dirac + E_em

    return float(E_total), float(E_dirac), float(E_em)


def save_snapshot(step, t, Psi, Ex, Ey, Bz, phi, out_dir, xp_mod, block_size=4):
    """
    Save a snapshot of the system as a compressed .npz file with:
      - density = |Ψ|^2
      - spin_z  = Ψ† sigma_z Ψ
      - Ex, Ey, Bz
      - phi (if provided)
      - block-averaged density and spin_z (if block_size > 1 divides Nx, Ny)
    """
    os.makedirs(out_dir, exist_ok=True)
    Ny, Nx = Psi.shape[1], Psi.shape[2]

    # Compute density and spin_z
    density = xp_mod.sum(xp_mod.abs(Psi) ** 2, axis=0)

    sigma_z = xp_mod.array([[1, 0],
                            [0, -1]], dtype=xp_mod.complex128)
    Psi_flat = Psi.reshape(2, Ny * Nx)
    Psi_dag_flat = xp_mod.conjugate(Psi_flat).T
    sigma_z_Psi_flat = sigma_z.dot(Psi_flat)
    spin_z_flat = xp_mod.sum(Psi_dag_flat * sigma_z_Psi_flat.T, axis=1).real
    spin_z = spin_z_flat.reshape(Ny, Nx)

    # Block-averaged density and spin_z
    density_block = None
    spin_z_block = None
    if block_size is not None and block_size > 1 and (Ny % block_size == 0) and (Nx % block_size == 0):
        by = Ny // block_size
        bx = Nx // block_size
        density_block = density.reshape(by, block_size, bx, block_size).mean(axis=(1, 3))
        spin_z_block = spin_z.reshape(by, block_size, bx, block_size).mean(axis=(1, 3))

    # Move to CPU for saving if on GPU
    if XP_BACKEND == "cupy":
        import numpy as np
        density_np = xp_mod.asnumpy(density)
        spin_z_np = xp_mod.asnumpy(spin_z)
        Ex_np = xp_mod.asnumpy(Ex)
        Ey_np = xp_mod.asnumpy(Ey)
        Bz_np = xp_mod.asnumpy(Bz)
        phi_np = xp_mod.asnumpy(phi) if phi is not None else None
        density_block_np = xp_mod.asnumpy(density_block) if density_block is not None else None
        spin_z_block_np = xp_mod.asnumpy(spin_z_block) if spin_z_block is not None else None
    else:
        import numpy as np
        density_np = density
        spin_z_np = spin_z
        Ex_np = Ex
        Ey_np = Ey
        Bz_np = Bz
        phi_np = phi
        density_block_np = density_block
        spin_z_block_np = spin_z_block

    data = dict(
        t=t,
        density=density_np,
        spin_z=spin_z_np,
        Ex=Ex_np,
        Ey=Ey_np,
        Bz=Bz_np,
    )
    if phi is not None:
        data["phi"] = phi_np
    if density_block_np is not None:
        data["density_block"] = density_block_np
    if spin_z_block_np is not None:
        data["spin_z_block"] = spin_z_block_np

    fname = os.path.join(out_dir, f"dirac_snap_{step:06d}.npz")
    np.savez_compressed(fname, **data)


def main():
    args = parse_args()

    Nx = args.Nx
    Ny = args.Ny
    dx = args.dx
    dt = args.dt
    steps = args.steps
    out_every = args.out_every
    c = args.c
    q = args.q
    m = args.m
    out_dir = args.out_dir
    em_backreaction = args.em_backreaction
    do_renorm = not args.no_renorm

    defrag = args.defrag
    defrag_kappa = args.defrag_kappa
    defrag_lambda = args.defrag_lambda
    defrag_g = args.defrag_g
    block_size = args.block_size

    eps0 = 1.0  # set eps0 = 1 in code units

    print("[INIT] 2+1D Dirac–Maxwell Substrate Engine")
    print(f"[INIT] Backend: {XP_BACKEND}")
    print(f"[INIT] Grid: {Nx} x {Ny}, dx={dx}, dt={dt}")
    print(f"[INIT] Parameters: c={c}, q={q}, m={m}")
    if defrag:
        print(f"[INIT] Defrag ON: kappa={defrag_kappa}, lambda={defrag_lambda}, g_phi={defrag_g}")
    if em_backreaction:
        print("[INIT] EM backreaction ON (vector potential A).")
    if not do_renorm:
        print("[INIT] Global spinor renormalization DISABLED.")

    os.makedirs(out_dir, exist_ok=True)

    # Build Dirac matrices
    sigma_x, sigma_y, sigma_z, gamma0, gamma1, gamma2, alpha1, alpha2, beta = build_dirac_matrices(xp)

    # Initialize fields
    Psi_n = init_spinor(Ny, Nx, dx, xp, m)   # Ψ^0 initial spinor
    Ex, Ey, Bz = init_em_fields(Ny, Nx, xp)

    # Optional vector potential for EM->Dirac backreaction (temporal gauge: dA/dt = -E)
    if em_backreaction:
        Ax = xp.zeros_like(Ex)
        Ay = xp.zeros_like(Ey)
    else:
        Ax = None
        Ay = None

    # Defrag scalar field phi
    if defrag:
        phi = xp.zeros((Ny, Nx), dtype=xp.float64)
    else:
        phi = None

    # One-step Euler to get Ψ^(n+1) for leapfrog:
    H_Psi = dirac_hamiltonian_action(
        Psi_n, dx, c, m, alpha1, alpha2, beta, xp,
        Ax=Ax, Ay=Ay, q=q,
        phi=phi, g_phi=defrag_g
    )
    Psi_np1 = Psi_n - 1j * dt * H_Psi  # Ψ^(1) ≈ Ψ^0 - i dt H Ψ^0

    # Optional normalization (global ray representative)
    if do_renorm:
        norm = xp.sqrt(xp.sum(xp.abs(Psi_np1) ** 2))
        if norm != 0:
            Psi_np1 /= norm

    # Set up leapfrog: Psi_nm1 = Ψ^0, Psi_n = Ψ^1
    Psi_nm1 = Psi_n
    Psi_n = Psi_np1

    # Time loop (leapfrog for spinor, explicit for EM & phi)
    t0 = time.time()
    energies_path = os.path.join(out_dir, "dirac_energies.csv")
    with open(energies_path, "w") as f:
        f.write("step,time,norm,E_total,E_dirac,E_em\n")

    print_interval = max(1, out_every)

    for step in range(1, steps + 1):
        t = step * dt

        # Current from Psi_n (centered)
        rho, Jx, Jy = compute_spinor_current(Psi_n, alpha1, alpha2, q, c, xp)

        # Update EM fields
        Ex, Ey, Bz = update_em_fields(Ex, Ey, Bz, Jx, Jy, dx, dt, c, eps0, xp)

        # Optional update of vector potential for EM->Dirac backreaction
        if em_backreaction:
            Ax = Ax - dt * Ex
            Ay = Ay - dt * Ey

        # Defrag scalar update: ∂_t φ = -kappa * (φ - (ρ + λ ∇²ρ))
        if defrag:
            rho_sm = rho + defrag_lambda * laplacian(rho, dx)
            phi = phi + dt * (-defrag_kappa * (phi - rho_sm))

        # Dirac Hamiltonian at Psi_n
        H_Psi_n = dirac_hamiltonian_action(
            Psi_n, dx, c, m, alpha1, alpha2, beta, xp,
            Ax=Ax, Ay=Ay, q=q,
            phi=phi, g_phi=defrag_g if defrag else 0.0
        )

        # Leapfrog step:
        #   Ψ^(n+1) = Ψ^(n-1) - 2 i dt H Ψ^n
        Psi_new = Psi_nm1 - 2j * dt * H_Psi_n  # true leapfrog: use Ψ at n-1 and n

        # Optional global norm renormalization (we care about rays)
        if do_renorm:
            norm = xp.sqrt(xp.sum(xp.abs(Psi_new) ** 2))
            if norm != 0:
                Psi_new /= norm

        # Rotate spinor states: (Psi_nm1, Psi_n) <- (Psi_n, Psi_new)
        Psi_nm1, Psi_n = Psi_n, Psi_new

        # Diagnostics
        if step % out_every == 0:
            norm_val = float(xp.sqrt(xp.sum(xp.abs(Psi_n) ** 2)))
            E_tot, E_dirac, E_em = compute_energies(Psi_n, Ex, Ey, Bz, dx, m, c, beta, xp)
            with open(energies_path, "a") as f:
                f.write(f"{step},{t:.6f},{norm_val:.16e},{E_tot:.16e},{E_dirac:.16e},{E_em:.16e}\n")
            print(f"[STEP {step:6d}] t={t:7.3f}  |Psi|={norm_val:.6e}  E_tot={E_tot:.6e}  E_Dirac={E_dirac:.6e}  E_EM={E_em:.6e}")
            save_snapshot(step, t, Psi_n, Ex, Ey, Bz, phi, out_dir, xp, block_size=block_size)

    t1 = time.time()
    print(f"[DONE] steps={steps}, dt={dt}, runtime={t1 - t0:.2f} s.")
    print(f"[DONE] Energies → {energies_path}")
    print(f"[DONE] Snapshots → {out_dir}/dirac_snap_*.npz")


if __name__ == "__main__":
    main()

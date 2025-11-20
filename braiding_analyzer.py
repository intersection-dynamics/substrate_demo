#!/usr/bin/env python
"""
braiding_analyzer.py - Final, Corrected Version

Analyzes the topological braiding of Substrate vortices.
Calculates TOTAL, DYNAMICAL, and GEOMETRIC phases by integrating the FULL Hamiltonian
(including the Defrag potential) to accurately test for Fermionic statistics.
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

# --- Global Constants for Hamiltonian/Potential (matching simulation) ---
# NOTE: Interaction_epsilon was hardcoded to 1.0 in the engine
INTERACTION_EPSILON = 1.0 

# --- Utility Functions for Hamiltonian Reconstruction (using NumPy) ---

def build_dirac_matrices():
    """Build 2x2 Dirac matrices for 2+1D (D2 choice) in NumPy."""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    gamma0 = sigma_z
    alpha1 = gamma0.dot(1j * sigma_y)
    alpha2 = gamma0.dot(-1j * sigma_x)
    beta = gamma0
    return alpha1, alpha2, beta

def central_diff_x(f, dx):
    """Central difference derivative in x with periodic BC."""
    return (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2.0 * dx)

def central_diff_y(f, dx):
    """Central difference derivative in y with periodic BC."""
    return (np.roll(f, -1, axis=-2) - np.roll(f, 1, axis=-2)) / (2.0 * dx)

def compute_defrag_potential(rho, G, dx):
    """
    Reconstructs the Defrag potential V(x) = rho * Kernel via FFT.
    This must exactly match the implementation in substrate_engine.py.
    """
    if G == 0:
        return np.zeros_like(rho)

    Ny, Nx = rho.shape
    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dx
    X, Y = np.meshgrid(x, y)
    
    # Centered coordinates for kernel construction
    cx = Nx * dx / 2
    cy = Ny * dx / 2
    
    # Distance squared from center (periodic approximation)
    r2 = (X - cx)**2 + (Y - cy)**2
    
    # Defrag Kernel: -1 / (r^2 + epsilon)
    kernel = -1.0 / (r2 + INTERACTION_EPSILON)
    
    # Normalization (Crucial!)
    kernel /= (np.sum(np.abs(kernel)) * dx * dx)

    # FFT of the kernel
    kernel_hat = np.fft.fft2(kernel)

    # Convolution
    rho_hat = np.fft.fft2(rho)
    pot_hat = rho_hat * kernel_hat
    V = np.fft.ifft2(pot_hat).real
    
    # Shift to center FFT result (undoing what numpy.fft.fft2 does)
    V = np.fft.fftshift(V)
    
    return G * V

def calculate_E_expectation(Psi, dx, c, m, G):
    """
    Compute E = <Psi|H_FULL|Psi> = Sum(Psi* . H Psi) * dx^2
    H_FULL = H_Dirac (Kinetic + Mass) + V_Defrag
    """
    alpha1, alpha2, beta = build_dirac_matrices()

    # --- 1. Compute H_Dirac * Psi ---
    # Derivatives
    dPsi_dx = central_diff_x(Psi, dx)
    dPsi_dy = central_diff_y(Psi, dx)

    Ny, Nx = Psi.shape[1], Psi.shape[2]
    Psi_flat = Psi.reshape(2, Ny * Nx)
    dPsi_dx_flat = dPsi_dx.reshape(2, Ny * Nx)
    dPsi_dy_flat = dPsi_dy.reshape(2, Ny * Nx)

    # Kinetic term: -i c (alpha1 ∂_x + alpha2 ∂_y) Ψ
    kinetic_flat = -1j * c * (alpha1.dot(dPsi_dx_flat) + alpha2.dot(dPsi_dy_flat))

    # Mass term: m c^2 beta Ψ
    mass_flat = (m * c * c) * (beta.dot(Psi_flat))
    
    H_Dirac_Psi = (kinetic_flat + mass_flat).reshape(2, Ny, Nx)

    # --- 2. Compute V_Defrag * Psi (Local Term) ---
    # Density for V_Defrag calculation
    rho = np.sum(np.abs(Psi)**2, axis=0).real 
    V_defrag = compute_defrag_potential(rho, G, dx)
    
    # V_defrag * Psi (potential energy term is local, acting on the field)
    V_Psi = V_defrag[np.newaxis, :, :] * Psi 
    
    # --- 3. Compute Expectation Value <E> ---
    # E = <Psi | (H_Dirac + V_Defrag) | Psi>
    
    # Contribution from H_Dirac
    E_dirac_density = np.sum(np.conjugate(Psi) * H_Dirac_Psi, axis=0)
    
    # Contribution from V_Defrag (Note: <Psi|V_Defrag|Psi> = V_Defrag * <Psi|Psi> = V_Defrag * rho)
    E_pot_density = V_defrag * rho
    
    # Sum densities and integrate spatially
    E_total_density = E_dirac_density + E_pot_density
    
    E_exp_value = np.sum(E_total_density).real * (dx ** 2)
    
    return E_exp_value

# --- Rest of the Analysis and Main Function (Fixed Plotting) ---

def find_peaks(rho, dx, threshold=0.001):
    """Finds the (x, y) indices of the two vortex centers."""
    neighborhood_size = 10
    local_max = maximum_filter(rho, size=neighborhood_size)
    peaks_mask = (rho == local_max) & (rho > threshold)
    indices = np.argwhere(peaks_mask)
    if len(indices) < 2: return None
    densities = rho[indices[:, 0], indices[:, 1]]
    sorted_idx = np.argsort(densities)[::-1]
    indices = indices[sorted_idx]
    p1 = indices[0]
    p2 = None
    min_dist_sq = (5.0/dx)**2
    for i in range(1, len(indices)):
        curr = indices[i]
        dist_sq = (curr[0]-p1[0])**2 + (curr[1]-p1[1])**2
        if dist_sq > min_dist_sq:
            p2 = curr
            break
    if p2 is None: return None
    return p1, p2

def load_snapshots(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "substrate_*.npz")))
    if not files:
        print(f"[ERROR] No data found in {data_dir}")
        return []
    return files

def analyze_braiding(files, dx, dt, c, m, G):
    
    trajectory_angle = []
    accumulated_phase_total = []
    energy_expectation = [] # To calculate Dynamical Phase

    # 1. Get initial total phase (reference frame)
    with np.load(files[0]) as data_0:
        Psi_0 = data_0['Psi']
        initial_phase = np.angle(np.sum(Psi_0))
        
    current_phase_total = initial_phase
    
    print(f"[INFO] Analyzing {len(files)} frames for all phase components...")
    
    for i, fname in enumerate(files):
        with np.load(fname) as data:
            rho = data['rho']
            t = data['t']
            Psi_curr = data['Psi']

            # --- Orbital Analysis ---
            peaks = find_peaks(rho, dx)
            if peaks is None: continue
            p1, p2 = peaks
            dy = (p2[0] - p1[0]) * dx
            dx_val = (p2[1] - p1[1]) * dx
            theta = np.arctan2(dy, dx_val)
            trajectory_angle.append((t, theta))

            # --- Total Quantum Phase Analysis (Geometric + Dynamical) ---
            if i > 0:
                prev_Psi = np.load(files[i-1])['Psi']
                overlap = np.sum(np.conjugate(Psi_curr) * prev_Psi)
                delta_phase = np.angle(overlap)
                current_phase_total += delta_phase

            accumulated_phase_total.append((t, current_phase_total))

            # --- Energy Expectation for Dynamical Phase (NOW CORRECTED) ---
            E_exp = calculate_E_expectation(Psi_curr, dx, c, m, G)
            energy_expectation.append((t, E_exp))

    # Unwrap angle and center phase
    trajectory_angle = np.array(trajectory_angle)
    if len(trajectory_angle) > 0:
        trajectory_angle[:, 1] = np.unwrap(trajectory_angle[:, 1])

    accumulated_phase_total = np.array(accumulated_phase_total)
    if len(accumulated_phase_total) > 0:
        accumulated_phase_total[:, 1] -= accumulated_phase_total[0, 1]

    # --- Dynamical Phase Integration ---
    energy_expectation = np.array(energy_expectation)
    E_t = energy_expectation[:, 0]
    E_exp = energy_expectation[:, 1]
    
    # Dynamical Phase: Phi_dyn = - Integral <E> dt 
    Phi_dyn = - np.cumsum(E_exp) * dt
    Phi_dyn -= Phi_dyn[0]
    
    accumulated_phase_dyn = np.stack((E_t, Phi_dyn), axis=1)

    # --- Geometric Phase Calculation ---
    # Phi_geom = Phi_total - Phi_dyn
    phi_total_interp = np.interp(E_t, accumulated_phase_total[:, 0], accumulated_phase_total[:, 1])
    Phi_geom = phi_total_interp - Phi_dyn
    
    accumulated_phase_geom = np.stack((E_t, Phi_geom), axis=1)

    return trajectory_angle, accumulated_phase_total, accumulated_phase_dyn, accumulated_phase_geom

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="output_fermions_close", help="Directory for Fermion data")
    parser.add_argument("--dx", type=float, default=0.5, help="Spatial grid spacing")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation time step (Original engine was 0.01)")
    parser.add_argument("--c", type=float, default=1.0, help="Speed of light parameter")
    parser.add_argument("--m", type=float, default=0.5, help="Dirac mass parameter")
    # NEW ARGUMENT: Defrag potential strength
    parser.add_argument("--G", type=float, default=50.0, help="Defrag potential strength (Must match simulation run)")
    args = parser.parse_args()
    
    files = load_snapshots(args.dir)
    if not files: return

    traj, phi_total, phi_dyn, phi_geom = analyze_braiding(files, args.dx, args.dt, args.c, args.m, args.G)
    
    if len(traj) == 0:
        print("Could not track distinct particles.")
        return
        
    # --- Final Results ---
    t = traj[:, 0]
    theta = traj[:, 1]
    
    total_rot_rad = theta[-1] - theta[0]
    total_rot_deg = np.degrees(total_rot_rad)
    
    final_phi_total = phi_total[-1, 1]
    final_phi_dyn = phi_dyn[-1, 1]
    final_phi_geom = phi_geom[-1, 1]

    num_exchanges = total_rot_deg / 180.0
    expected_phi_geom = num_exchanges * np.pi
    
    print(f"\n--- Rigorous Quantum Braiding Analysis ---")
    print(f"Total Orbital Angle: {total_rot_deg:.2f} degrees ({num_exchanges:.2f} exchanges)")
    print(f"----------------------------------------")
    print(f"Total Phase (Observed):    {final_phi_total:.4f} radians")
    print(f"Dynamical Phase (Calculated): {final_phi_dyn:.4f} radians")
    print(f"Geometric Phase (Result: Total - Dynamical): {final_phi_geom:.4f} radians")
    print(f"----------------------------------------")
    print(f"Expected Geometric Phase (Fermion: {num_exchanges:.2f} * pi): {expected_phi_geom:.4f} radians")
    print(f"Ratio Geometric Phase / Expected: {final_phi_geom / expected_phi_geom:.4f}")

    # --- Plotting (NameError Fixed) ---
    fig, ax1 = plt.subplots(figsize=(12, 7))
    fig.suptitle("Emergent Fermionic Statistics: Dynamical vs. Geometric Phase", fontsize=14)

    color = 'tab:blue'
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Orbital Angle (rad)", color=color)
    ax1.plot(t, theta, color=color, label='Orbital Angle')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    
    ax2.plot(phi_geom[:, 0], phi_geom[:, 1], color='red', linewidth=2.5, label='Geometric Phase $\Phi_{geom}$')
    ax2.plot(phi_dyn[:, 0], phi_dyn[:, 1], color='orange', linestyle=':', label='Dynamical Phase $\Phi_{dyn}$')
    ax2.plot(phi_total[:, 0], phi_total[:, 1], color='gray', linestyle='--', label='Total Phase $\Phi_{total}$')

    ax2.set_ylabel("Accumulated Quantum Phase $\Phi$ (rad)", color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    ax2.axhline(np.pi, color='green', linestyle=':', label='Fermionic Phase ($\pi$)')
    # Fixed the NameError by using a simple string label
    ax2.axhline(expected_phi_geom, color='purple', linestyle='--', alpha=0.6, label=f'Expected Geom Phase ({num_exchanges:.2f}$\pi$)')

    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("geometric_phase_plot_corrected.png")
    print("Combined plot saved to geometric_phase_plot_corrected.png")

if __name__ == "__main__":
    main()
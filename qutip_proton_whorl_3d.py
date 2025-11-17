#!/usr/bin/env python3
"""
qutip_proton_whorl_3d.py

Toy-universe "proton + electron whorl" state in QuTiP, exported as:

- 2D density and phase images (PNG)
- 3D OBJ surface mesh (z ~ |psi|^2, normalized and scaled)

This version normalizes the density before scaling so the 3D mesh
isn't a nearly-flat square.

Requirements:
    pip install qutip matplotlib
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from qutip import Qobj, ket2dm

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------

L = 64                 # grid size (L x L)
dx = 1.0               # lattice spacing (arbitrary units)
sigma = 8.0            # Gaussian width for the proton core
vortex_charge = 1      # winding number m: psi ~ e^{i m theta}

# This is height in 3D units after normalizing rho to [0,1]
z_scale = 20.0         # try 10–50 for good visual relief

output_dir = Path("qutip_proton_whorl_output")
output_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Build a 2D grid and a vortex-like proton state
# ----------------------------------------------------------------------

coords = np.arange(L) - (L - 1) / 2.0
X, Y = np.meshgrid(coords * dx, coords * dx, indexing="xy")

R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)

# Amplitude: Gaussian proton core
amp = np.exp(-R**2 / (2.0 * sigma**2))

# Phase: vortex with winding m
phase = vortex_charge * Theta
psi_grid = amp * np.exp(1j * phase)

# Normalize state
norm = np.sqrt(np.sum(np.abs(psi_grid) ** 2))
psi_grid /= norm

psi_flat = psi_grid.ravel()
N = psi_flat.size

psi_qutip = Qobj(psi_flat, dims=[[N], [1]])

# Basic checks for your "decoherence microscope"
norm2 = psi_qutip.norm() ** 2
rho_dm = ket2dm(psi_qutip)
purity = (rho_dm * rho_dm).tr().real

print("QuTiP proton+whorl state built:")
print(f"  Hilbert dim: {N}")
print(f"  <psi|psi>   = {norm2:.6f}")
print(f"  Purity Tr(rho^2) = {purity:.6f} (1.0 for pure state)")

# ----------------------------------------------------------------------
# Save 2D density and phase images
# ----------------------------------------------------------------------

rho_2d = np.abs(psi_grid) ** 2
phase_wrapped = np.angle(psi_grid)

fig1, ax1 = plt.subplots(1, 2, figsize=(10, 4))

im0 = ax1[0].imshow(rho_2d, origin="lower", cmap="magma")
ax1[0].set_title("Density |psi|^2 (toy proton core)")
ax1[0].set_xticks([]); ax1[0].set_yticks([])
plt.colorbar(im0, ax=ax1[0], fraction=0.046)

im1 = ax1[1].imshow(phase_wrapped, origin="lower", cmap="twilight")
ax1[1].set_title("Phase arg(psi) (whorl)")
ax1[1].set_xticks([]); ax1[1].set_yticks([])
plt.colorbar(im1, ax=ax1[1], fraction=0.046)

plt.tight_layout()
img_path = output_dir / "proton_whorl_2d.png"
plt.savefig(img_path, dpi=200)
plt.close(fig1)

print(f"[OUT] Saved 2D density+phase image to {img_path}")

# ----------------------------------------------------------------------
# Export a 3D OBJ surface with normalized height
# ----------------------------------------------------------------------

def save_heightfield_obj(
    rho_2d: np.ndarray,
    dx: float,
    z_scale: float,
    out_path: Path,
):
    """
    Save a quad-mesh OBJ:
        x = i * dx
        y = j * dx
        z = (rho_norm[i,j]) * z_scale

    where rho_norm is rho_2d normalized to [0,1].
    """
    ny, nx = rho_2d.shape

    rho_min = float(rho_2d.min())
    rho_max = float(rho_2d.max())
    if rho_max <= rho_min:
        rho_norm = np.zeros_like(rho_2d)
    else:
        rho_norm = (rho_2d - rho_min) / (rho_max - rho_min)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Toy proton whorl heightfield OBJ (normalized)\n")

        # vertices
        for j in range(ny):
            for i in range(nx):
                x = i * dx
                y = j * dx
                z = float(rho_norm[j, i]) * z_scale
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

        # helper to map grid index -> vertex index
        def vert_index(ix, iy):
            return iy * nx + ix + 1  # OBJ is 1-based

        # faces
        for j in range(ny - 1):
            for i in range(nx - 1):
                v1 = vert_index(i, j)
                v2 = vert_index(i + 1, j)
                v3 = vert_index(i + 1, j + 1)
                v4 = vert_index(i, j + 1)
                f.write(f"f {v1} {v2} {v3}\n")
                f.write(f"f {v1} {v3} {v4}\n")

obj_path = output_dir / "proton_whorl_surface.obj"
save_heightfield_obj(rho_2d, dx=dx, z_scale=z_scale, out_path=obj_path)
print(f"[OUT] Saved 3D OBJ mesh to {obj_path}")

print("Done. In your 3D viewer:")
print("  - Import proton_whorl_surface.obj")
print("  - You should now see a clear hill (proton core) instead of a flat plate.")

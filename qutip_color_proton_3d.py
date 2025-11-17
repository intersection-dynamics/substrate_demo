#!/usr/bin/env python3
"""
qutip_color_proton_3d.py  (light version)

Higher-resolution toy proton with internal "color charges", but without
building a gigantic density matrix (so it won't murder your PC).

We build three complex fields on an LxL grid:

  psi_R(x,y) = Gaussian core * e^{+i theta}          (red whorl)
  psi_G(x,y) = Gaussian core * e^{-i theta}          (green counter-whorl)
  psi_B(x,y) = Gaussian core * radial oscillation    (blue breathing mode)

Total density:
  rho_total = |psi_R|^2 + |psi_G|^2 + |psi_B|^2

We:

  - Pack the state into a QuTiP ket with dims [[L*L, 3],[1,1]]
  - Save:
      * 2D total-density + color-charge image (PNG)
      * 3D OBJ surface with z ~ total density (normalized and scaled)

Requirements:
    pip install qutip matplotlib
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------

L = 64                # grid size (L x L). Start here; you can try 96 later.
dx = 1.0              # lattice spacing
sigma = 8.0           # Gaussian width for the proton core
vortex_charge_R = 1   # red winding number
vortex_charge_G = -1  # green winding number (opposite)
radial_k_B = 5.0      # sets radial oscillation scale for blue component

z_scale = 20.0        # height of the proton hill after normalization

output_dir = Path("qutip_color_proton_output")
output_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Build 2D grid and color components
# ----------------------------------------------------------------------

coords = np.arange(L) - (L - 1) / 2.0
X, Y = np.meshgrid(coords * dx, coords * dx, indexing="xy")
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)

# Shared Gaussian envelope for the proton core
envelope = np.exp(-R**2 / (2.0 * sigma**2))

# Red: vortex with +1 winding
psi_R = envelope * np.exp(1j * vortex_charge_R * Theta)

# Green: vortex with -1 winding (counter-rotating)
psi_G = envelope * np.exp(1j * vortex_charge_G * Theta)

# Blue: radial breathing mode (no net angular winding, but shells)
radial_mod = 0.5 * (1.0 + np.cos(R / radial_k_B))
psi_B = envelope * radial_mod

# Stack into 3-color field and normalize total state
rho_R = np.abs(psi_R) ** 2
rho_G = np.abs(psi_G) ** 2
rho_B = np.abs(psi_B) ** 2
rho_total = rho_R + rho_G + rho_B

norm = np.sqrt(np.sum(rho_total))
psi_R /= norm
psi_G /= norm
psi_B /= norm

rho_R = np.abs(psi_R) ** 2
rho_G = np.abs(psi_G) ** 2
rho_B = np.abs(psi_B) ** 2
rho_total = rho_R + rho_G + rho_B

# ----------------------------------------------------------------------
# Build QuTiP state with position ⊗ color, but DON'T build rho = ket2dm(psi)
# ----------------------------------------------------------------------

N_pos = L * L
psi_flat = np.concatenate([
    psi_R.ravel(),
    psi_G.ravel(),
    psi_B.ravel(),
])

psi_qutip = Qobj(psi_flat, dims=[[N_pos, 3], [1, 1]])

norm2 = float(psi_qutip.norm() ** 2)
print("QuTiP color-proton state built (light version):")
print(f"  Position dim: {N_pos}, color dim: 3  → total dim {3 * N_pos}")
print(f"  <psi|psi>     = {norm2:.6f}")
print("  (Skipping density matrix / purity to avoid huge memory use)")

# ----------------------------------------------------------------------
# 2D visualization: total density + color charge map
# ----------------------------------------------------------------------

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

im0 = ax[0].imshow(rho_total, origin="lower", cmap="magma")
ax[0].set_title("Total density |psi|^2 (toy proton core)")
ax[0].set_xticks([]); ax[0].set_yticks([])
plt.colorbar(im0, ax=ax[0], fraction=0.046)

# Color charge map: RGB from normalized component densities
rho_rgb = np.stack([rho_R, rho_G, rho_B], axis=-1)  # (L, L, 3)
eps = 1e-12
max_per_pixel = rho_rgb.max(axis=-1, keepdims=True)
max_per_pixel[max_per_pixel < eps] = 1.0
rgb_norm = rho_rgb / max_per_pixel

ax[1].imshow(rgb_norm, origin="lower")
ax[1].set_title("Color charges (R,G,B components)")
ax[1].set_xticks([]); ax[1].set_yticks([])

plt.tight_layout()
img_path = output_dir / "color_proton_2d.png"
plt.savefig(img_path, dpi=200)
plt.close(fig)

print(f"[OUT] Saved 2D total-density + color-charge image to {img_path}")

# ----------------------------------------------------------------------
# 3D OBJ: heightfield from total density
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
        z = rho_norm[i,j] * z_scale

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
        f.write("# Color proton heightfield OBJ (total density, normalized)\n")

        for j in range(ny):
            for i in range(nx):
                x = i * dx
                y = j * dx
                z = float(rho_norm[j, i]) * z_scale
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

        def vid(ix, iy):
            return iy * nx + ix + 1  # OBJ indices are 1-based

        for j in range(ny - 1):
            for i in range(nx - 1):
                v1 = vid(i, j)
                v2 = vid(i + 1, j)
                v3 = vid(i + 1, j + 1)
                v4 = vid(i, j + 1)
                f.write(f"f {v1} {v2} {v3}\n")
                f.write(f"f {v1} {v3} {v4}\n")

obj_path = output_dir / "color_proton_surface.obj"
save_heightfield_obj(rho_total, dx=dx, z_scale=z_scale, out_path=obj_path)
print(f"[OUT] Saved 3D OBJ mesh to {obj_path}")

print("Done. You should be able to:")
print("  - View color_proton_2d.png for the internal color structure.")
print("  - Import color_proton_surface.obj into Blender/MeshLab.")
print("If you want higher spatial resolution, try increasing L to 96—but")
print("keep in mind the Hilbert dimension grows as 3 * L^2.")

#!/usr/bin/env python3
"""
QCD Tornado Proton — Decoherence Microscope (memory-safe edition)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ----------------------------------------------------------------------
# Same proton as before
# ----------------------------------------------------------------------
L = 256
coords = np.arange(L) - (L-1)/2.0
X, Y = np.meshgrid(coords, coords, indexing='xy')
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)
sigma = 9.0
angular_offset = np.pi / 1.7

envelope_R = np.exp(-R**2 / (2*(sigma*0.8)**2))
envelope_G = np.exp(-R**2 / (2*(sigma*0.85)**2))
envelope_B = np.exp(-R**2 / (2*(sigma*1.2)**2))

psi_R = envelope_R * np.exp(1j*(Theta + angular_offset))
psi_G = envelope_G * np.exp(1j*(-Theta - angular_offset))
psi_B = envelope_B * (0.5 + 0.5*np.cos(R/4))

rho_R = np.abs(psi_R)**2
rho_G = np.abs(psi_G)**2
rho_B = np.abs(psi_B)**2
rho_total = rho_R + rho_G + rho_B

norm = np.sqrt(np.sum(rho_total))
psi_R /= norm; psi_G /= norm; psi_B /= norm
rho_R = np.abs(psi_R)**2; rho_G = np.abs(psi_G)**2; rho_B = np.abs(psi_B)**2
rho_total = rho_R + rho_G + rho_B

# ----------------------------------------------------------------------
# Decoherence rate map Γ(x,y) — high on color vortices, low elsewhere
# ----------------------------------------------------------------------
Gamma = 12.0 * (rho_R + rho_G) / (rho_total + 1e-12)   # fast on red/green arms
Gamma += 1.0 * rho_B / (rho_total + 1e-12)            # slow on blue breathing
Gamma += 0.1                                          # tiny global background

# ----------------------------------------------------------------------
# Simple exponential decoherence of OFF-DIAGONAL color terms
# We completely avoid building the full density matrix
# ----------------------------------------------------------------------
tlist = np.linspace(0, 1.2, 70)

frames = []
for t in tlist:
    # Coherence between R and G decays fast where Gamma is high
    coherence_RG = np.exp(-t * Gamma) * psi_R * np.conj(psi_G)
    coherence_RB = np.exp(-t * Gamma * 0.5) * psi_R * np.conj(psi_B)
    coherence_GB = np.exp(-t * Gamma * 0.5) * psi_G * np.conj(psi_B)

    # Reconstructed color densities (diagonals never decay)
    R_t = rho_R + 2*np.abs(coherence_RG + coherence_RB)
    G_t = rho_G + 2*np.abs(coherence_RG + coherence_GB)
    B_t = rho_B + 2*np.abs(coherence_RB + coherence_GB)

    rgb = np.stack([R_t, G_t, B_t], axis=-1)
    rgb /= rgb.max() + 1e-12
    rgb = np.clip(rgb, 0, 1)

    frames.append(rgb)

# ----------------------------------------------------------------------
# Make the GIF
# ----------------------------------------------------------------------
output_dir = Path("qutip_color_proton_output")
output_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(1, 2, figsize=(14,7), facecolor='black')
im1 = ax[0].imshow(frames[0])
im2 = ax[1].imshow(rho_total, cmap='magma')
ax[0].set_title("Color Structure (decohering)", color='white', fontsize=18)
ax[1].set_title("Total Density (stable)", color='white', fontsize=18)
for a in ax: a.axis('off')

def update(i):
    im1.set_array(frames[i])
    ax[0].set_title(f"Decoherence Microscope — t = {tlist[i]:.2f}", color='white', fontsize=18)
    return im1,

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100)
ani.save(output_dir / "decoherence_microscope_fixed.gif", writer='pillow', fps=12, dpi=180)
print("Done! → decoherence_microscope_fixed.gif")
print("Watch the color vortices wash out while the proton stays perfectly sharp.")
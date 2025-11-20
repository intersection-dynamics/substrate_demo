import numpy as np
import matplotlib.pyplot as plt

snap = np.load("dirac_output/dirac_snap_002000.npz")  # last snapshot

density       = snap["density"]
phi           = snap["phi"]
density_block = snap["density_block"]
spin_block    = snap["spin_z_block"]

print(density.shape, phi.shape, density_block.shape)

# Quick look at density vs phi
plt.figure()
plt.imshow(density, origin="lower")
plt.title("Density |Psi|^2")
plt.colorbar()

plt.figure()
plt.imshow(phi, origin="lower")
plt.title("Defrag field phi")
plt.colorbar()

plt.figure()
plt.imshow(density_block, origin="lower")
plt.title("Block-averaged density")
plt.colorbar()

plt.show()

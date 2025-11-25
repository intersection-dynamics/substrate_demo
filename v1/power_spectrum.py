from scalar_field_defrag_gpu import ScalarFieldDefragGPU
import numpy as np
import matplotlib.pyplot as plt

# Load final state (you need to save psi_final from your run)
# For now, recreate
sim = ScalarFieldDefragGPU(L=128, g_defrag=1.5, lambda_param=0.0)
psi = sim.create_uniform_noise(mean=1.0, noise_amp=0.1, seed=42)

# Evolve to final state
for _ in range(5000):
    psi, _ = sim.evolve_step_rk4(psi)

# Compute power spectrum
k_radial, P_radial = sim.compute_power_spectrum(psi)

plt.loglog(k_radial, P_radial, 'o-')
plt.xlabel('k')
plt.ylabel('P(k)')
plt.title('Power Spectrum - Final State')
plt.savefig('power_spectrum.png')
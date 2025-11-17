from ising_defrag_gpu import IsingDefragGPU

# NO DEFRAG - this is the control
sim = IsingDefragGPU(L=64, T=1.0, g_defrag=0.0, coarse_grain_size=1)

# Same initial condition as before
spins = sim.create_noise_spins(flip_prob=0.2, seed=42)

# Run it
df = sim.run_evolution(spins, n_sweeps=2000, output_dir='control_g0')
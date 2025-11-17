#!/usr/bin/env python3
"""
Quick experimental protocols for Ising + defrag
"""
from ising_defrag_gpu import IsingDefragGPU
import numpy as np

# EXPERIMENT 1: Stripe Universality (different seeds)
print("="*70)
print("EXPERIMENT 1: STRIPE UNIVERSALITY - Testing 5 seeds")
print("="*70)

for seed in [42, 100, 200, 300, 400]:
    print(f"\n--- Seed {seed} ---")
    sim = IsingDefragGPU(L=64, T=1.0, g_defrag=0.5, coarse_grain_size=1)
    spins = sim.create_noise_spins(flip_prob=0.2, seed=seed)
    df = sim.run_evolution(spins, n_sweeps=1000, 
                          output_dir=f'exp1_universality/seed{seed}')
    print(f"Final: |M|={df['M_abs'].iloc[-1]:.4f}, walls={df['n_walls'].iloc[-1]}")

# EXPERIMENT 2: Temperature Scan
print("\n" + "="*70)
print("EXPERIMENT 2: TEMPERATURE SCAN")
print("="*70)

for T in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    print(f"\n--- T={T:.1f} ---")
    sim = IsingDefragGPU(L=64, T=T, g_defrag=0.5, coarse_grain_size=1)
    spins = sim.create_noise_spins(flip_prob=0.2, seed=42)
    df = sim.run_evolution(spins, n_sweeps=2000,
                          output_dir=f'exp2_temperature/T{T:.1f}')
    print(f"Final: |M|={df['M_abs'].iloc[-1]:.4f}, walls={df['n_walls'].iloc[-1]}")

# EXPERIMENT 3: Defrag Strength
print("\n" + "="*70)
print("EXPERIMENT 3: DEFRAG STRENGTH")
print("="*70)

for g in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
    print(f"\n--- g={g:.1f} ---")
    sim = IsingDefragGPU(L=64, T=1.0, g_defrag=g, coarse_grain_size=1)
    spins = sim.create_noise_spins(flip_prob=0.2, seed=42)
    df = sim.run_evolution(spins, n_sweeps=1000,
                          output_dir=f'exp3_strength/g{g:.1f}')
    print(f"Final: |M|={df['M_abs'].iloc[-1]:.4f}, walls={df['n_walls'].iloc[-1]}")

# EXPERIMENT 4: Coarse-Graining
print("\n" + "="*70)
print("EXPERIMENT 4: COARSE-GRAINING TEST")
print("="*70)

for cg in [1, 2, 4, 8]:
    print(f"\n--- Coarse-grain size={cg} ---")
    sim = IsingDefragGPU(L=64, T=1.0, g_defrag=0.5, coarse_grain_size=cg)
    spins = sim.create_noise_spins(flip_prob=0.2, seed=42)
    df = sim.run_evolution(spins, n_sweeps=2000,
                          output_dir=f'exp4_coarse_grain/cg{cg}')
    print(f"Final: |M|={df['M_abs'].iloc[-1]:.4f}, walls={df['n_walls'].iloc[-1]}")

print("\n" + "="*70)
print("ALL EXPERIMENTS COMPLETE!")
print("="*70)
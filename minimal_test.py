#!/usr/bin/env python3
"""
minimal_test.py

Absolutely minimal test - guaranteed to work if your Python environment is set up correctly.

This uses ultra-conservative parameters that should be stable on any system.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from spinor_substrate import SpinorSubstrate, GPU_AVAILABLE
    print(f"✓ spinor_substrate imported successfully")
    print(f"  GPU available: {GPU_AVAILABLE}")
except ImportError as e:
    print(f"✗ Failed to import spinor_substrate: {e}")
    exit(1)

print()
print("="*60)
print("MINIMAL SPINOR SUBSTRATE TEST")
print("="*60)
print()
print("Running ultra-conservative parameters:")
print("  L = 32 (tiny grid)")
print("  dt = 0.0005 (very small timestep)")
print("  g_defrag = 0.5 (weak defrag)")
print("  100 steps")
print()

# Create substrate with ultra-conservative parameters
substrate = SpinorSubstrate(
    L=32,           # Small grid
    dx=1.0,
    dt=0.0005,      # Very small timestep
    q=0.5,          # Weak gauge coupling
    g_defrag=0.5,   # Weak defrag
    m_gauge=0.0
)

# Initialize with tiny noise
print("Initializing with small amplitude noise...")
substrate.initialize_noise(amplitude=0.01, seed=42)

E0 = substrate.energy_total()
print(f"Initial energy: {E0:.6f}")
print()

# Evolve for 100 steps
print("Evolving for 100 steps...")
print()
print("Step    Time      Energy         ΔE/E        Status")
print("-" * 60)

success = True

for step in range(100):
    substrate.step_symplectic()
    
    if step % 10 == 0:
        E = substrate.energy_total()
        
        if np.isnan(E) or np.isinf(E):
            print(f"{step:4d}  {substrate.time:6.3f}  {E:12s}  {' '*10}  ✗ UNSTABLE")
            success = False
            break
        
        drift = abs(E - E0) / abs(E0)
        
        if drift < 0.01:
            status = "✓ Good"
        elif drift < 0.1:
            status = "⚠ OK"
        else:
            status = "✗ Large drift"
        
        print(f"{step:4d}  {substrate.time:6.3f}  {E:12.6f}  {drift:8.2e}    {status}")

print()
print("="*60)

if success:
    E_final = substrate.energy_total()
    drift_final = abs(E_final - E0) / abs(E0)
    
    print("✓ SUCCESS!")
    print()
    print(f"  Final energy: {E_final:.6f}")
    print(f"  Total drift: {drift_final:.2e} ({drift_final*100:.2f}%)")
    print()
    
    if drift_final < 0.01:
        print("  Excellent stability - system is working perfectly")
    elif drift_final < 0.05:
        print("  Good stability - ready for production runs")
    else:
        print("  Marginal stability - consider smaller dt")
    
    print()
    print("Next steps:")
    print("  1. Run test_stability.py to find optimal dt")
    print("  2. Try larger grid (L=64, L=128)")
    print("  3. Try single skyrmion initialization")
    print("  4. Run the full interaction test")
    
    # Save a quick plot
    output_dir = Path("minimal_test_output")
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Density
    rho = substrate.density()
    ax = axes[0]
    im = ax.imshow(rho.T, origin='lower', cmap='viridis')
    ax.set_title('Density')
    plt.colorbar(im, ax=ax)
    
    # Spin sz
    sx, sy, sz = substrate.spin_texture()
    ax = axes[1]
    im = ax.imshow(sz.T, origin='lower', cmap='RdBu', vmin=-1, vmax=1)
    ax.set_title('Spin sz')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'minimal_test.png', dpi=150)
    print()
    print(f"  Plot saved to {output_dir / 'minimal_test.png'}")
    
else:
    print("✗ FAILURE - Simulation unstable")
    print()
    print("Possible issues:")
    print("  - GPU/CPU incompatibility")
    print("  - Numerical precision problems")
    print("  - Code installation issue")
    print()
    print("Try:")
    print("  - Make sure numpy is up to date")
    print("  - If using GPU, check CuPy installation")
    print("  - Reduce dt even further (--dt 0.0001)")

print()
print("="*60)
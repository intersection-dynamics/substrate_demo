#!/usr/bin/env python3
"""
test_stability.py

Find stable timestep and parameters for spinor substrate.

Tests different dt values to find what's stable.
"""

import numpy as np
import sys
from pathlib import Path

try:
    from spinor_substrate import SpinorSubstrate, GPU_AVAILABLE
except ImportError:
    print("Error: spinor_substrate.py must be in same directory")
    sys.exit(1)


def test_timestep(L=64, dt=0.001, n_steps=100, g_defrag=1.5, init_mode='noise'):
    """
    Test if system is stable with given parameters.
    
    Returns:
        stable: True if no NaN/Inf
        max_drift: Maximum energy drift
    """
    
    substrate = SpinorSubstrate(L=L, dx=1.0, dt=dt, g_defrag=g_defrag)
    
    if init_mode == 'noise':
        substrate.initialize_noise(amplitude=0.01)
    elif init_mode == 'skyrmion':
        substrate.initialize_skyrmion((L/2, L/2), charge=1)
    
    E0 = substrate.energy_total()
    
    if np.isnan(E0) or np.isinf(E0):
        return False, np.inf
    
    max_drift = 0.0
    
    for step in range(n_steps):
        substrate.step_symplectic()
        
        if step % 10 == 0:
            E = substrate.energy_total()
            
            if np.isnan(E) or np.isinf(E):
                return False, np.inf
            
            drift = abs(E - E0) / abs(E0) if E0 != 0 else 0
            max_drift = max(max_drift, drift)
    
    return True, max_drift


def find_stable_dt():
    """Find maximum stable timestep."""
    
    print("="*60)
    print("STABILITY TEST: Finding stable timestep")
    print("="*60)
    print(f"GPU: {GPU_AVAILABLE}")
    print()
    
    # Test range of timesteps
    dt_values = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    
    print("Testing noise initialization:")
    print()
    
    for dt in dt_values:
        print(f"dt = {dt:7.4f} ... ", end='', flush=True)
        
        stable, drift = test_timestep(L=64, dt=dt, n_steps=100, init_mode='noise')
        
        if stable:
            print(f"✓ STABLE (drift = {drift:.2e})")
        else:
            print(f"✗ UNSTABLE (NaN)")
    
    print()
    print("Testing single skyrmion:")
    print()
    
    for dt in dt_values:
        print(f"dt = {dt:7.4f} ... ", end='', flush=True)
        
        stable, drift = test_timestep(L=64, dt=dt, n_steps=100, init_mode='skyrmion')
        
        if stable:
            print(f"✓ STABLE (drift = {drift:.2e})")
        else:
            print(f"✗ UNSTABLE (NaN)")
    
    print()
    print("="*60)
    print("RECOMMENDATION:")
    print("="*60)
    print()
    print("Use dt = 0.001 or smaller for production runs")
    print("Use dt = 0.0001 if you see any instability")
    print()


def test_defrag_strength():
    """Test different defrag strengths."""
    
    print("="*60)
    print("STABILITY TEST: Defrag strength")
    print("="*60)
    print()
    
    g_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    dt = 0.001
    
    for g in g_values:
        print(f"g_defrag = {g:.1f} ... ", end='', flush=True)
        
        stable, drift = test_timestep(L=64, dt=dt, n_steps=200, 
                                     g_defrag=g, init_mode='noise')
        
        if stable:
            print(f"✓ STABLE (drift = {drift:.2e})")
        else:
            print(f"✗ UNSTABLE")
    
    print()


def quick_evolution_test():
    """Run short evolution and show energy."""
    
    print("="*60)
    print("QUICK EVOLUTION TEST")
    print("="*60)
    print()
    print("Running 1000 steps with recommended parameters...")
    print("dt = 0.001, L = 64, g_defrag = 1.5")
    print()
    
    substrate = SpinorSubstrate(L=64, dx=1.0, dt=0.001, g_defrag=1.5)
    substrate.initialize_noise(amplitude=0.01, seed=42)
    
    E0 = substrate.energy_total()
    print(f"Initial energy: {E0:.6f}")
    print()
    
    print("Step      Time        Energy          ΔE/E")
    print("-" * 55)
    
    for step in range(1000):
        substrate.step_symplectic()
        
        if step % 100 == 0:
            E = substrate.energy_total()
            drift = abs(E - E0) / abs(E0)
            print(f"{step:4d}   {substrate.time:8.3f}   {E:12.6f}   {drift:.2e}")
    
    print()
    
    if drift < 0.01:
        print("✓ System is stable!")
        print("  Energy drift < 1%")
        print("  Ready for production runs")
    elif drift < 0.1:
        print("⚠️  System is marginally stable")
        print(f"  Energy drift = {drift:.1%}")
        print("  Consider smaller dt or weaker g_defrag")
    else:
        print("✗ System is unstable")
        print(f"  Energy drift = {drift:.1%}")
        print("  Use smaller dt or different parameters")
    
    print()


if __name__ == '__main__':
    find_stable_dt()
    print()
    test_defrag_strength()
    print()
    quick_evolution_test()
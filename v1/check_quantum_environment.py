#!/usr/bin/env python
"""
check_quantum_environment.py

Diagnostic script to test if quantum simulation environment is properly set up.
Run this BEFORE trying the full simulation to catch installation issues.
"""

import sys

print("=" * 60)
print("Quantum Environment Diagnostics")
print("=" * 60)
print()

# Test 1: NumPy
print("[1/5] Testing NumPy...")
try:
    import numpy as np
    print(f"  ✓ NumPy {np.__version__} installed")
except ImportError as e:
    print(f"  ✗ NumPy not found: {e}")
    print("  Install with: pip install numpy")
    sys.exit(1)

# Test 2: CuPy (optional)
print("\n[2/5] Testing CuPy (GPU acceleration - optional)...")
try:
    import cupy as cp
    print(f"  ✓ CuPy {cp.__version__} installed")
    print("  → GPU acceleration available")
except ImportError:
    print("  ○ CuPy not found (optional)")
    print("  Install with: pip install cupy (requires CUDA)")

# Test 3: QuTiP (required)
print("\n[3/5] Testing QuTiP (required)...")
try:
    import qutip
    print(f"  ✓ QuTiP {qutip.__version__} installed")
except ImportError as e:
    print(f"  ✗ QuTiP not found: {e}")
    print("  Install with: pip install qutip")
    print("  This is REQUIRED for quantum evolution")
    sys.exit(1)

# Test 4: Matplotlib (for plotting)
print("\n[4/5] Testing Matplotlib (plotting - optional)...")
try:
    import matplotlib
    print(f"  ✓ Matplotlib {matplotlib.__version__} installed")
except ImportError:
    print("  ○ Matplotlib not found (optional)")
    print("  Install with: pip install matplotlib")

# Test 5: Basic QuTiP operations
print("\n[5/5] Testing basic QuTiP operations...")
try:
    from qutip import basis, tensor, qeye, sigmax, mesolve
    
    # Create a simple 2-qubit system
    psi0 = tensor(basis(2, 0), basis(2, 0))
    
    # Simple Hamiltonian
    H = tensor(sigmax(), qeye(2))
    
    # Try a short evolution
    times = [0, 0.1]
    result = mesolve(H, psi0, times, [], [])
    
    if len(result.states) == 2:
        print("  ✓ Basic quantum evolution works")
        print(f"  → Can simulate {psi0.shape[0]}-dimensional Hilbert space")
    else:
        print("  ✗ Evolution produced unexpected results")
        sys.exit(1)
        
except Exception as e:
    print(f"  ✗ QuTiP operations failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Memory estimate
print("\n" + "=" * 60)
print("System Capabilities")
print("=" * 60)

try:
    import psutil
    mem_gb = psutil.virtual_memory().total / (1024**3)
    print(f"\nTotal RAM: {mem_gb:.1f} GB")
    
    # Estimates for different lattice sizes
    print("\nHilbert space sizes (qubits, local_dim=2):")
    for nx in [2, 3, 4, 5, 6]:
        n_sites = nx * nx
        dim = 2 ** n_sites
        mem_per_state_mb = dim * 16 / (1024**2)  # Complex128 = 16 bytes
        
        status = "✓" if mem_per_state_mb < mem_gb * 100 else "✗"
        print(f"  {status} {nx}x{nx}: dimension = 2^{n_sites} = {dim:,}, ~{mem_per_state_mb:.1f} MB per state")
        
        if mem_per_state_mb > mem_gb * 1000:
            print(f"     (Would need {mem_per_state_mb/1000:.0f} GB - impractical)")
            break
            
except ImportError:
    print("\nCouldn't check memory (psutil not installed)")
    print("Recommended lattice sizes:")
    print("  ✓ 3x3: Easy (~4 MB)")
    print("  ✓ 4x4: OK (~65 MB)")  
    print("  ⚠ 5x5: Challenging (~34 GB)")

# Recommendations
print("\n" + "=" * 60)
print("Recommendations")
print("=" * 60)
print()

all_good = True
try:
    import qutip
except:
    all_good = False

if all_good:
    print("✓ Environment is properly configured!")
    print()
    print("Recommended starting point:")
    print("  Lattice size: 3x3 (512-dimensional Hilbert space)")
    print("  Local dimension: 2 (qubits)")
    print("  Evolution time: 2-5 time units")
    print()
    print("Next step:")
    print("  bash test_single_vortex.sh")
else:
    print("✗ Environment has issues")
    print()
    print("Required installation:")
    print("  pip install qutip numpy")
    print()
    print("Optional but recommended:")
    print("  pip install matplotlib cupy")

print()
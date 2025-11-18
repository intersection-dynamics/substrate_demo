#!/usr/bin/env python3
"""
test_scalar_field_extensivity.py

CRITICAL TEST: Does scalar field + defrag give extensive scaling?

Hypothesis: YES, because kinetic energy prevents λ ~ L pathology

If confirmed: Proves kinetic structure is necessary for extensive emergence
"""

import sys
sys.path.insert(0, '/mnt/user-data/uploads')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Note: scalar_field_defrag_gpu.py should exist in uploads
# If not, we'll need to create it

print("="*70)
print("CRITICAL TEST: SCALAR FIELD EXTENSIVITY")
print("="*70)
print("\nHypothesis: Scalar field + defrag should give E ~ L² (extensive)")
print("Reason: Kinetic energy sets intrinsic length scale")
print("\nThis would prove kinetic structure is NECESSARY for framework")
print("="*70)

# Check if scalar field code exists
try:
    from scalar_field_defrag_gpu import ScalarFieldDefragGPU
    print("\n✓ Found scalar field code")
    HAVE_SCALAR = True
except ImportError:
    print("\n✗ scalar_field_defrag_gpu.py not found")
    print("  (Need to copy from earlier work or recreate)")
    HAVE_SCALAR = False

if HAVE_SCALAR:
    print("\nRunning scalar field extensivity test...")
    print("This will test L = 32, 48, 64, 96 at T=1.0, g=0.5")
    print("Same parameters as Ising test for direct comparison")
    
    # TODO: Run the actual test when code is available
    print("\n[Ready to run when you provide scalar_field_defrag_gpu.py]")
else:
    print("\n" + "="*70)
    print("WHAT THIS TEST WILL DO")
    print("="*70)
    print("""
Test Protocol:
1. Run scalar field at L = 32, 48, 64, 96
2. Parameters: T=1.0, g_defrag=0.5 (same as Ising)
3. Measure: E_defrag, wall count, correlation length
4. Fit: log(E) vs log(L) to find scaling exponent

Expected Results if Hypothesis is TRUE:
- E ~ L² (extensive!) 
- E/site = constant
- Walls ~ L (extensive density)
- Wavelength λ ~ 20-30 sites (independent of L)

Expected Results if Hypothesis is FALSE:
- E ~ L³ or L⁴ (super-extensive)
- E/site increases with L
- Wavelength λ ~ L

Why This Matters:
- If TRUE: Kinetic energy is the key!
  → Framework requirement identified
  → Clear path forward
  → Publishable insight

- If FALSE: Deeper problem with Poisson coupling
  → Need to reconsider framework
  → More investigation needed

This is THE critical test for the framework.
    """)

# Create comparison plot showing expected result
print("\nCreating expected result visualization...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Energy scaling comparison
ax = axes[0]
L_vals = np.array([32, 48, 64, 96])

# Ising: E ~ L⁴ (observed)
E_ising = 5000 * (L_vals/48)**4
ax.loglog(L_vals, E_ising, 'ro-', linewidth=3, markersize=12, label='Ising (E ~ L⁴)')

# Scalar field: E ~ L² (predicted)
E_scalar = 5000 * (L_vals/48)**2
ax.loglog(L_vals, E_scalar, 'go-', linewidth=3, markersize=12, label='Scalar Field (E ~ L²?)')

# Reference lines
L_ref = np.linspace(30, 100, 50)
ax.loglog(L_ref, 5000 * (L_ref/48)**2, 'g--', alpha=0.3, linewidth=2, label='~ L² (extensive)')
ax.loglog(L_ref, 5000 * (L_ref/48)**4, 'r--', alpha=0.3, linewidth=2, label='~ L⁴ (super-extensive)')

ax.set_xlabel('System Size L', fontsize=12)
ax.set_ylabel('|Energy|', fontsize=12)
ax.set_title('Predicted: Kinetic Energy Fixes Scaling', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which='both')

# Plot 2: Why kinetic energy helps
ax = axes[1]
ax.text(0.5, 0.95, 'Why Kinetic Energy Fixes It', 
        transform=ax.transAxes, fontsize=14, fontweight='bold',
        ha='center', va='top')

explanation = """
ISING MODEL (no kinetic energy):
• Domain walls are SHARP (1 site)
• No energy cost for sharp walls
• Nothing prevents coarsening
• System forms walls at scale λ ~ L
• Result: E ~ L⁴

SCALAR FIELD (has kinetic energy):
• Domain walls have WIDTH w
• Kinetic energy ~ (Δψ/w)²
• Sharp walls cost ENERGY
• Width w ~ √(stiffness/potential)
• This SETS intrinsic scale!

Consequence:
• Wall width w ~ constant
• Can't coarsen beyond w
• Wavelength λ ~ w (not λ ~ L!)
• Result: E ~ L² ✓

THE KEY: Kinetic energy provides
intrinsic length scale that
prevents pathological scaling!
"""

ax.text(0.1, 0.85, explanation, transform=ax.transAxes,
       fontsize=10, verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.axis('off')

plt.tight_layout()
plt.savefig('/home/claude/scalar_field_prediction.png', dpi=150, bbox_inches='tight')
print("✓ Saved prediction plot to scalar_field_prediction.png")
plt.close()

# Create test script template
print("\nCreating test script template...")

test_script = """#!/usr/bin/env python3
# Quick test of scalar field extensivity

from scalar_field_defrag_gpu import ScalarFieldDefragGPU
import numpy as np

L_values = [32, 48, 64, 96]
results = []

for L in L_values:
    sim = ScalarFieldDefragGPU(L=L, T=1.0, g_defrag=0.5)
    
    # Initialize with noise
    psi = sim.create_noise_field(amplitude=0.2, seed=42)
    
    # Evolve to equilibrium (2000 steps)
    for step in range(2000):
        Phi = sim.solve_defrag_potential(psi)
        psi = sim.langevin_step(psi, Phi)
    
    # Measure
    E_defrag = sim.compute_defrag_energy(psi, Phi)
    
    print(f"L={L}: E_defrag={E_defrag:.2e}, E/site={E_defrag/(L*L):.2f}")
    results.append({'L': L, 'E': E_defrag, 'E_per_site': E_defrag/(L*L)})

# Analyze scaling
import pandas as pd
df = pd.DataFrame(results)
from scipy import stats
log_L = np.log(df['L'].values)
log_E = np.log(np.abs(df['E'].values))
slope, _, r_value, _, _ = stats.linregress(log_L, log_E)
print(f"\\nScaling: E ~ L^{slope:.2f} (R²={r_value**2:.3f})")
if 1.9 < slope < 2.1:
    print("✓✓✓ EXTENSIVE! Hypothesis confirmed!")
else:
    print(f"✗ Not extensive (slope={slope:.2f})")
"""

with open('/home/claude/run_scalar_test.py', 'w') as f:
    f.write(test_script)

print("✓ Created run_scalar_test.py (ready when scalar field code available)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
Status: Ready to test scalar field extensivity

What you need:
1. scalar_field_defrag_gpu.py (from your earlier work)
2. Run: python run_scalar_test.py
3. Compare E scaling to Ising

Expected outcome:
✓ Scalar field: E ~ L² (proves kinetic energy matters)
✓ Ising: E ~ L⁴ (confirmed pathology)
→ Conclusion: Kinetic structure is NECESSARY

This single test validates the framework constraint!

Time estimate: 30-60 minutes to run
Impact: Transforms paper from "bug" to "discovery"
""")
print("="*70)
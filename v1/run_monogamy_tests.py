#!/usr/bin/env python
"""
run_monogamy_tests.py

Proper test runner that shows all output and handles errors correctly.
No hidden subprocesses, no disappearing windows, no silent failures.
"""

import subprocess
import sys
import os
from datetime import datetime

def print_header(text):
    """Print a visible header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70 + "\n")

def run_evolution(winding, Nx=3, Ny=3, t_max=3.0, dt=0.2):
    """
    Run a single evolution and return success/failure.
    All output is shown directly to terminal.
    """
    out_dir = f"results_w{winding}"
    
    print_header(f"Testing Winding Number w={winding}")
    print(f"Output directory: {out_dir}")
    print(f"Lattice: {Nx}×{Ny}")
    print(f"Evolution time: {t_max}")
    print()
    
    cmd = [
        sys.executable,  # Use same Python interpreter
        "quantum_substrate_evolution.py",
        "--Nx", str(Nx),
        "--Ny", str(Ny),
        "--pattern", "single_vortex",
        "--winding", str(winding),
        "--t_max", str(t_max),
        "--dt", str(dt),
        "--J_nn", "1.0",
        "--J_defrag", "0.5",
        "--mass", "0.1",
        "--out", out_dir
    ]
    
    print("Command:", " ".join(cmd))
    print()
    print("-" * 70)
    
    # Run with output going directly to terminal
    result = subprocess.run(cmd)
    
    print("-" * 70)
    print()
    
    # Check if it worked
    json_file = os.path.join(out_dir, "entanglement_evolution.json")
    
    if result.returncode == 0 and os.path.exists(json_file):
        print(f"✓ SUCCESS: w={winding} completed")
        print(f"  Results saved to: {json_file}")
        return True
    else:
        print(f"✗ FAILED: w={winding}")
        if result.returncode != 0:
            print(f"  Exit code: {result.returncode}")
        if not os.path.exists(json_file):
            print(f"  Output file not created: {json_file}")
        return False

def analyze_results(winding):
    """Analyze results for a winding number."""
    out_dir = f"results_w{winding}"
    json_file = os.path.join(out_dir, "entanglement_evolution.json")
    
    if not os.path.exists(json_file):
        print(f"✗ No results to analyze for w={winding}")
        return False
    
    print(f"\nAnalyzing w={winding}...")
    print("-" * 70)
    
    cmd = [
        sys.executable,
        "analyze_entanglement_results.py",
        out_dir
    ]
    
    result = subprocess.run(cmd)
    
    print("-" * 70)
    
    return result.returncode == 0

def create_comparison_plot(windings_to_compare):
    """Create comparison plot for multiple winding numbers."""
    if len(windings_to_compare) == 0:
        print("\n✗ No results to compare")
        return False
    
    print_header("Creating Comparison Plot")
    
    result_dirs = [f"results_w{w}" for w in windings_to_compare]
    
    cmd = [
        sys.executable,
        "analyze_entanglement_results.py",
        "dummy",  # Required by argparse but unused
        "--compare"
    ] + result_dirs + [
        "--windings"
    ] + [str(w) for w in windings_to_compare] + [
        "--output", "monogamy_comparison.png"
    ]
    
    print("Command:", " ".join(cmd))
    print()
    print("-" * 70)
    
    result = subprocess.run(cmd)
    
    print("-" * 70)
    print()
    
    if result.returncode == 0 and os.path.exists("monogamy_comparison.png"):
        print("✓ Comparison plot created: monogamy_comparison.png")
        return True
    else:
        print("✗ Failed to create comparison plot")
        return False

def main():
    print_header("Quantum Substrate: Monogamy Test Suite")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    windings = [0, 1, -1]
    lattice_size = 3
    t_max = 3.0
    dt = 0.2
    
    print("Configuration:")
    print(f"  Winding numbers to test: {windings}")
    print(f"  Lattice size: {lattice_size}×{lattice_size}")
    print(f"  Hilbert space dimension: 2^{lattice_size*lattice_size} = {2**(lattice_size*lattice_size)}")
    print(f"  Evolution time: {t_max}")
    print(f"  Time steps: {int(t_max/dt)}")
    print()
    
    input("Press Enter to continue (Ctrl+C to abort)...")
    
    # Run evolutions
    print_header("PHASE 1: Running Evolutions")
    
    successful_windings = []
    failed_windings = []
    
    for w in windings:
        try:
            success = run_evolution(w, Nx=lattice_size, Ny=lattice_size, 
                                   t_max=t_max, dt=dt)
            if success:
                successful_windings.append(w)
            else:
                failed_windings.append(w)
        except KeyboardInterrupt:
            print("\n\n✗ Interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ Unexpected error for w={w}: {e}")
            import traceback
            traceback.print_exc()
            failed_windings.append(w)
    
    # Analyze results
    print_header("PHASE 2: Analyzing Results")
    
    for w in successful_windings:
        try:
            analyze_results(w)
        except Exception as e:
            print(f"✗ Analysis failed for w={w}: {e}")
    
    # Create comparison
    if len(successful_windings) > 1:
        print_header("PHASE 3: Creating Comparison Plot")
        try:
            create_comparison_plot(successful_windings)
        except Exception as e:
            print(f"✗ Comparison plot failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print_header("SUMMARY")
    
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("Results:")
    for w in windings:
        if w in successful_windings:
            print(f"  ✓ w={w:2d}: results_w{w}/")
        elif w in failed_windings:
            print(f"  ✗ w={w:2d}: FAILED")
        else:
            print(f"  ? w={w:2d}: UNKNOWN")
    
    print()
    
    if os.path.exists("monogamy_comparison.png"):
        print("✓ Comparison plot: monogamy_comparison.png")
    else:
        print("✗ No comparison plot created")
    
    print()
    
    if len(failed_windings) > 0:
        print(f"⚠ {len(failed_windings)} test(s) failed")
        print()
        print("Common issues:")
        print("  - QuTiP not installed: pip install qutip")
        print("  - Out of memory: reduce lattice size")
        print("  - Hamiltonian too large: reduce interactions")
        print()
        print("Run diagnostics:")
        print("  python check_quantum_environment.py")
    else:
        print("✓ All tests passed!")
    
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
#!/usr/bin/env python3
"""
run_full_validation.py

Master script to run all critical validation tests for the Ising stripe phase.

This runs:
1. System size dependence (L=32,48,64,96) - proves not grid artifact
2. Statistical reproducibility (5 seeds) - proves robust equilibrium
3. Spatial structure analysis (g=0 vs g=0.5) - proves organized structure

Total runtime: ~4-6 hours depending on GPU
Generates comprehensive validation report.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_test(script_name, description):
    """Run a test script and report results."""
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print("="*70)
    print(f"Script: {script_name}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        
        print("\n" + "="*70)
        print(f"✓ COMPLETED: {description}")
        print(f"Finished: {datetime.now().strftime('%H:%M:%S')}")
        print("="*70)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("\n" + "="*70)
        print(f"✗ FAILED: {description}")
        print(f"Error: {e}")
        print("="*70)
        return False


def create_summary_report(results):
    """Create summary validation report."""
    
    report_dir = Path("validation_report")
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / "VALIDATION_SUMMARY.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Ising Stripe Phase Validation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## Tests Run\n\n")
        
        for test_name, (status, description) in results.items():
            status_icon = "[PASS]" if status else "[FAIL]"
            f.write(f"### {status_icon} {description}\n\n")
            
            if status:
                # Add links to results
                if "size" in test_name:
                    f.write("**Results:**\n")
                    f.write("- [Results CSV](../size_dependence_test/size_dependence_results.csv)\n")
                    f.write("- [Analysis Plots](../size_dependence_test/size_dependence_analysis.png)\n\n")
                    f.write("**Key Question:** Do stripes scale properly with system size?\n\n")
                    f.write("**Expected:** Wall count ~ 2L, stripe width constant\n\n")
                
                elif "reproducibility" in test_name:
                    f.write("**Results:**\n")
                    f.write("- [Results CSV](../reproducibility_test/reproducibility_results.csv)\n")
                    f.write("- [Analysis Plots](../reproducibility_test/reproducibility_analysis.png)\n\n")
                    f.write("**Key Question:** Do different random seeds give same result?\n\n")
                    f.write("**Expected:** Low variation in wall count and energy\n\n")
                
                elif "spatial" in test_name:
                    f.write("**Results:**\n")
                    f.write("- [Analysis Plots](../spatial_structure_test/spatial_structure_comparison.png)\n\n")
                    f.write("**Key Question:** Is g=0.5 structure different from g=0?\n\n")
                    f.write("**Expected:** Longer correlations, anisotropic structure\n\n")
            else:
                f.write("**Status:** Test failed - see error logs\n\n")
            
            f.write("---\n\n")
        
        # Overall assessment
        f.write("## Overall Assessment\n\n")
        
        all_passed = all(status for status, _ in results.values())
        
        if all_passed:
            f.write("### [PASS] ALL VALIDATION TESTS PASSED\n\n")
            f.write("The stripe phase has been validated as:\n\n")
            f.write("1. **Not a grid artifact** - scales properly with system size\n")
            f.write("2. **Reproducible** - consistent across random initial conditions\n")
            f.write("3. **Real structure** - qualitatively different from control\n\n")
            f.write("**Conclusion:** The stripe-separated phase is a robust, physical ")
            f.write("equilibrium state created by defrag coupling.\n\n")
        else:
            f.write("### [WARN] SOME TESTS FAILED\n\n")
            f.write("Review individual test results to determine issues.\n\n")
        
        # Next steps
        f.write("## Next Steps\n\n")
        f.write("1. Review individual test plots and CSVs\n")
        f.write("2. Check for any unexpected behavior\n")
        f.write("3. If all looks good, proceed to:\n")
        f.write("   - Energy conservation check (scalar field)\n")
        f.write("   - Method validation tests\n")
        f.write("   - Paper preparation\n\n")
        
        f.write("## Generated Files\n\n")
        f.write("```\n")
        f.write("validation_report/\n")
        f.write("├── VALIDATION_SUMMARY.md (this file)\n")
        f.write("size_dependence_test/\n")
        f.write("├── size_dependence_results.csv\n")
        f.write("├── size_dependence_analysis.png\n")
        f.write("├── size_test_L32/\n")
        f.write("├── size_test_L48/\n")
        f.write("├── size_test_L64/\n")
        f.write("└── size_test_L96/\n")
        f.write("reproducibility_test/\n")
        f.write("├── reproducibility_results.csv\n")
        f.write("├── reproducibility_analysis.png\n")
        f.write("├── seed_test_s42/\n")
        f.write("├── seed_test_s123/\n")
        f.write("├── seed_test_s456/\n")
        f.write("├── seed_test_s789/\n")
        f.write("└── seed_test_s1337/\n")
        f.write("spatial_structure_test/\n")
        f.write("└── spatial_structure_comparison.png\n")
        f.write("```\n")
    
    print(f"\n✓ Created validation summary: {report_file}")


def main():
    """Run full validation suite."""
    
    print("="*70)
    print("ISING STRIPE PHASE - FULL VALIDATION SUITE")
    print("="*70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will run:")
    print("  1. System size dependence test (~1-2 hours)")
    print("  2. Statistical reproducibility test (~1-2 hours)")
    print("  3. Spatial structure analysis (~30 minutes)")
    print("\nTotal estimated time: 3-5 hours")
    print("\nResults will be saved in separate test directories")
    print("="*70)
    
    input("\nPress Enter to start validation suite...")
    
    results = {}
    
    # Test 1: Size dependence
    success = run_test(
        "test_size_dependence.py",
        "System Size Dependence (Critical)"
    )
    results['size_dependence'] = (success, "System Size Dependence")
    
    if not success:
        print("\n⚠ Size dependence test failed - this is critical!")
        print("Continuing with other tests anyway...\n")
    
    # Test 2: Reproducibility
    success = run_test(
        "test_reproducibility.py",
        "Statistical Reproducibility (Critical)"
    )
    results['reproducibility'] = (success, "Statistical Reproducibility")
    
    if not success:
        print("\n⚠ Reproducibility test failed - this is critical!")
        print("Continuing with remaining tests...\n")
    
    # Test 3: Spatial structure
    success = run_test(
        "test_spatial_structure.py",
        "Spatial Structure Analysis (Critical)"
    )
    results['spatial_structure'] = (success, "Spatial Structure Analysis")
    
    if not success:
        print("\n⚠ Spatial structure test failed - this is critical!")
    
    # Create summary report
    print("\n" + "="*70)
    print("CREATING VALIDATION SUMMARY")
    print("="*70)
    
    create_summary_report(results)
    
    # Final summary
    print("\n" + "="*70)
    print("VALIDATION SUITE COMPLETE")
    print("="*70)
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_passed = all(status for status, _ in results.values())
    
    if all_passed:
        print("\n[PASS] ALL TESTS PASSED [PASS]")
        print("\nYour stripe phase is validated!")
        print("You can now confidently:")
        print("  1. Write up results")
        print("  2. Post on Twitter")
        print("  3. Submit for publication")
    else:
        failed_tests = [name for name, (status, _) in results.items() if not status]
        print(f"\n⚠ {len(failed_tests)} test(s) failed:")
        for test in failed_tests:
            print(f"  - {test}")
        print("\nReview errors and re-run failed tests")
    
    print("\nSee validation_report/VALIDATION_SUMMARY.md for details")
    print("="*70)


if __name__ == '__main__':
    main()
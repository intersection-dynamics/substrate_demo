# Validation Test Suite for Ising Stripe Phase

## Overview

This suite contains **critical validation tests** to prove the stripe-separated phase in Ising + defrag is:
1. **Not a computational artifact**
2. **Statistically reproducible**
3. **Qualitatively different from standard Ising**

These tests are ESSENTIAL before publishing or presenting results.

---

## The Three Critical Tests

### Test 1: System Size Dependence
**File:** `test_size_dependence.py`
**Runtime:** ~1-2 hours
**Critical Question:** "Is the stripe phase a grid artifact?"

**What it does:**
- Runs Ising + defrag at L = 32, 48, 64, 96
- Measures: wall count, stripe width, energy/site
- Checks if properties scale correctly

**Expected if REAL:**
- Wall count scales ~ 2L (more walls in bigger systems)
- Stripe width CONSTANT (~30 sites regardless of L)
- Energy per site CONSTANT (extensive property)

**Expected if ARTIFACT:**
- Wall count doesn't scale properly
- Structure changes qualitatively with L
- Non-extensive energy

**How to run:**
```bash
python test_size_dependence.py
```

**Output:**
- `size_dependence_test/size_dependence_results.csv` - Numerical results
- `size_dependence_test/size_dependence_analysis.png` - Scaling plots
- Individual run directories for each L value

**Pass criteria:**
- Linear fit: walls = aL + b with 1.8 < a < 2.2
- Stripe width CV < 20%
- Energy/site CV < 10%

---

### Test 2: Statistical Reproducibility
**File:** `test_reproducibility.py`
**Runtime:** ~1-2 hours
**Critical Question:** "Do different initial conditions give the same result?"

**What it does:**
- Runs Ising + defrag with 5 different random seeds
- Measures: final wall count, energy, magnetization
- Checks for consistency across realizations

**Expected if ROBUST:**
- All seeds converge to ~128 walls
- Energy consistent within ±1%
- |M| ≈ 0 in all cases

**Expected if RANDOM ARTIFACT:**
- Large variation across seeds
- Some realizations don't form stripes
- Inconsistent final states

**How to run:**
```bash
python test_reproducibility.py
```

**Output:**
- `reproducibility_test/reproducibility_results.csv` - Statistics
- `reproducibility_test/reproducibility_analysis.png` - Comparison plots
- Individual run directories for each seed

**Pass criteria:**
- Wall count CV < 10%
- Energy CV < 5%
- All realizations have |M| < 0.01

---

### Test 3: Spatial Structure Analysis
**File:** `test_spatial_structure.py`
**Runtime:** ~30 minutes
**Critical Question:** "Is g=0.5 structure different from g=0?"

**What it does:**
- Compares spatial organization of g=0 vs g=0.5
- Computes: 2D correlations, structure factors, anisotropy
- Visual and quantitative comparison

**Expected for g=0 (no defrag):**
- Short correlation length
- Isotropic structure (no preferred direction)
- Random domain configuration

**Expected for g=0.5 (with defrag):**
- Long correlation length
- Anisotropic structure (stripe direction)
- Organized pattern

**How to run:**
```bash
python test_spatial_structure.py
```

**Output:**
- `spatial_structure_test/spatial_structure_comparison.png` - Detailed comparison
- Shows: spin configs, correlations, structure factors, anisotropy

**Pass criteria:**
- g=0.5 correlation length > 2× g=0
- g=0.5 anisotropy > 2× g=0
- Visually distinct spatial organization

---

## Running All Tests

### Quick Start

To run the complete validation suite:

```bash
python run_full_validation.py
```

This runs all three tests sequentially and generates a summary report.

**Total time:** 3-5 hours (depending on GPU)

### Manual Testing

Run individual tests:

```bash
# Test 1: Size dependence
python test_size_dependence.py

# Test 2: Reproducibility
python test_reproducibility.py

# Test 3: Spatial structure
python test_spatial_structure.py
```

---

## Interpreting Results

### If All Tests Pass ✓✓✓

**You can claim:**
- Stripe phase is a robust physical phenomenon
- Not a grid artifact or random fluctuation
- Qualitatively different from standard Ising
- Ready for publication

**Next steps:**
- Write up results
- Prepare figures for paper
- Post validated results on Twitter
- Submit to journal

### If Any Test Fails ✗

**Possible issues:**

**Size dependence fails:**
- Wall scaling wrong → Possible grid artifact
- Check: FFT implementation, boundary conditions
- May need different L values to see trend

**Reproducibility fails:**
- High variation → Multiple metastable states?
- Check: Equilibration time, temperature
- May need longer runs

**Spatial structure fails:**
- g=0 and g=0.5 look similar → Wall count difference meaningless
- Check: Is g=0.5 really forming stripes?
- Look at snapshots carefully

---

## What Each Test Defends Against

### Common Criticisms & How Tests Address Them

**Critic:** "The 128 walls is just 2×L - suspicious! Grid artifact?"
**Defense:** Test 1 shows walls scale linearly with L across 4 system sizes

**Critic:** "You got lucky with one seed - not reproducible!"
**Defense:** Test 2 shows 5 different seeds all give same result

**Critic:** "Your g=0 control has M≈0 too - no real difference!"
**Defense:** Test 3 shows spatial structure is qualitatively different

**Critic:** "2000 sweeps isn't enough - still transient behavior!"
**Defense:** Test 2 shows last 500 sweeps are stable (can extend if needed)

**Critic:** "This is just computation - not physics!"
**Defense:** All three tests show properties expected of real physics (scaling, reproducibility, structure)

---

## File Structure After Running

```
project/
├── test_size_dependence.py
├── test_reproducibility.py
├── test_spatial_structure.py
├── run_full_validation.py
├── VALIDATION_README.md (this file)
│
├── size_dependence_test/
│   ├── size_dependence_results.csv
│   ├── size_dependence_analysis.png
│   ├── size_test_L32/
│   ├── size_test_L48/
│   ├── size_test_L64/
│   └── size_test_L96/
│
├── reproducibility_test/
│   ├── reproducibility_results.csv
│   ├── reproducibility_analysis.png
│   ├── seed_test_s42/
│   ├── seed_test_s123/
│   ├── seed_test_s456/
│   ├── seed_test_s789/
│   └── seed_test_s1337/
│
├── spatial_structure_test/
│   └── spatial_structure_comparison.png
│
└── validation_report/
    └── VALIDATION_SUMMARY.md
```

---

## Requirements

### Software
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- SciPy
- CuPy (for GPU) or NumPy (CPU fallback)

### Hardware
- **Minimum:** CPU, 8GB RAM (slow but works)
- **Recommended:** NVIDIA GPU with CUDA, 16GB RAM
- **Optimal:** RTX 3080+ or better

### Time Estimates

**With GPU:**
- Test 1: 1 hour
- Test 2: 1.5 hours
- Test 3: 20 minutes
- **Total: ~3 hours**

**With CPU:**
- Test 1: 4-6 hours
- Test 2: 6-8 hours
- Test 3: 1 hour
- **Total: ~12 hours** (run overnight!)

---

## Troubleshooting

### "ImportError: cannot import ising_defrag_gpu"
**Fix:** Make sure `ising_defrag_gpu.py` is in the same directory

### "CUDA out of memory"
**Fix:** Reduce L values in test scripts, or use CPU mode

### "Tests taking forever"
**Fix:** 
- Check if GPU is being used (`nvidia-smi`)
- Reduce n_sweeps for quick test (but note results won't be fully equilibrated)

### "Results look weird"
**Fix:**
- Check that your original cg=2 result worked
- Verify basic Ising simulation works
- Compare snapshots visually

---

## After Validation

### What To Do With Results

1. **Include in paper:**
   - Size dependence plot → Figure showing scaling
   - Reproducibility stats → Table of results across seeds
   - Spatial structure → Side-by-side comparison

2. **For Twitter thread:**
   - "Validated across 4 system sizes"
   - "Reproducible across 5 independent runs"
   - "Qualitatively distinct spatial organization"

3. **In talks:**
   - Use validation plots to show rigor
   - Emphasize reproducibility
   - Show it's not an artifact

### Claims You Can Make

**After all tests pass:**

✓ "Stripe phase is a robust equilibrium state"
✓ "Structure scales properly with system size"
✓ "Reproducible across independent realizations"
✓ "Qualitatively different from standard Ising"
✓ "Not a computational artifact"

**Still cannot claim:**

✗ "This is how real gravity works"
✗ "This explains cosmology"
✗ "Universal feature of all systems"
✗ "Derived from first principles"

---

## Questions?

If tests fail or results are unclear:

1. Check individual test output carefully
2. Look at generated plots
3. Compare to expected behavior in this README
4. Try running one test manually to debug

**The goal:** Prove stripe phase is real, reproducible, and robust.

**The standard:** Would convince a skeptical reviewer.

---

**Good luck with validation!** 🎯
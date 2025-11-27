# Hilbert Substrate Framework — Prototype

This repository contains a research prototype for the **Substrate Framework**: a universal-Hilbert-space model where:

- Causal structure (light cones)
- Metric geometry
- Boson/fermion–like statistics gaps

are studied as **emergent features** of unitary dynamics plus information-theoretic constraints.

The code here implements:

- A core **substrate engine** (Hilbert-space dynamics, local operators, detection).
- Several **numerical experiments**:
  - Noise robustness
  - Stability basins
  - Universality across lattices / Hamiltonians / detection channels

> ⚠️ **Current status:**  
> This is a working research sandbox. Scripts live in the repo root and write their
> output into simple folders like `noise_analysis/`, `stability_results/`, and
> `universality_results/` next to the scripts. A more polished `outputs/…` layout
> is planned but not implemented yet.

---

## Files and Experiments

### Core engine

- `substrate.py`  
  Core Hilbert-space substrate:
  - lattice generators (chain, ring, square, honeycomb, random, full)
  - hopping / Heisenberg / XY / random Hamiltonians
  - time evolution (CPU or GPU if available)
  - reduced density matrices and copyability tests
  - CNOT-like detection unitaries
  - basic plotting helpers

### Experiments

All of these are standalone scripts you run from the repo root.

#### 1. Noise robustness

- **File:** `noise_analysis.py`  
- **Output folder (created automatically):** `noise_analysis/`  
- **What it does:**
  - Tests which **types of noise** break the emergent structure.
  - Checks how the **noise threshold** depends on system size.
  - Compares different **network structures** (chain, ring, ladder, etc.).
  - Breaks down contributions from:
    - light-cone quality
    - metric quality
    - statistics gap

- **Generates plots:**
  - `noise_analysis/noise_types.png`
  - `noise_analysis/size_scaling.png`
  - `noise_analysis/structures.png`
  - `noise_analysis/metric_breakdown.png`

- **How to run (Windows, from repo root):**
  ```bash
  python noise_analysis.py
  ```

#### 2. Stability basin

- **File:** `stability_analysis.py`  
- **Output folder:** `stability_results/`  
- **What it does:**
  - Fixes a substrate and sweeps:
    - coupling strength
    - disorder strength
    - locality parameter
    - added noise
  - For each sweep, measures a stability score built from:
    - light-cone quality
    - metric quality
    - statistics gap

- **Generates plots:**
  - `stability_results/sweep_coupling.png`
  - `stability_results/sweep_disorder.png`
  - `stability_results/sweep_locality.png`
  - `stability_results/sweep_noise.png`

- **How to run:**
  ```bash
  python stability_analysis.py
  ```

#### 3. Universality test suite

- **File:** `universality_test.py`  
- **Output folder:** `universality_results/`  
- **What it does:**
  - Scans a grid of configurations:
    - local dimension: d = 2, 3
    - lattices: chain, ring, square, honeycomb, random, full
    - Hamiltonians: hopping, Heisenberg, XY, random-local
    - detection channels: CNOT-like, SWAP-like, ZZ-like, random
  - For each configuration, computes:
    - **LC** – light-cone quality
    - **M** – metric quality
    - **S** – statistics gap
  - Labels a configuration “universal” if (LC, M, S) are all above a threshold.
  - Prints aggregate stats (e.g., universality by lattice / Hamiltonian / detection).

- **Generates:**
  - `universality_results/heatmap.png`
  - `universality_results/distributions.png`
  - `universality_results/invariance.png`
  - `universality_results/results.json` (raw metrics per configuration)

- **How to run:**
  ```bash
  python universality_test.py
  ```

---

## Dependencies

This prototype uses:

- Python 3.10+
- NumPy, SciPy, Matplotlib
- Optional: CuPy for GPU acceleration (if installed and CUDA available)

Install core dependencies with:

```bash
pip install numpy scipy matplotlib
```

and optionally CuPy if you have a CUDA-capable GPU.

---

## Framework References

For the conceptual background (axioms, constraints, and emergent structure), see the framework document.

For the planned software layout (engines vs experiments vs outputs), see the Software Guide.

At the moment, this repo reflects an **in-progress** implementation rather than the final polished layout.

---

## Roadmap

Planned improvements:

- Move `substrate.py` into `engines/` and experiments into `experiments/`.
- Adopt a proper `outputs/<experiment>/<run_id>/…` layout:
  - JSON summaries
  - raw arrays (NPZ)
  - figures
  - logs
- Add command-line options:
  - `--output-root`
  - `--tag`
  - `--seed`
- Container / Colab setup for one-click reproducibility.

For now, the scripts are meant as **research tools**: run them, inspect the plots, and iterate on the physics.


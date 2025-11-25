# Skyrmion Overlap vs Separation: Scaling Test (λ_G = 5.0)

This note records the current scaling test for the gauge–constrained skyrmion toy model.  
The goal is **purely diagnostic**:

> Check whether the *penalty for overlapping two skyrmions*, relative to separating them, grows with system size \(L\) under a fixed Gauss-law penalty \(\lambda_G\).

We do **not** claim to have realized true fermionic exchange statistics. This is strictly about how a local Gauss constraint and onsite repulsion shape the energy cost of overlap in this finite model.

---

## Setup

- Lattice: 2D square, linear size \(L = 2,\dots,6\).
- Gauge sector: \(\mathbb{Z}_2\) gauge field with Gauss-law penalty \(\lambda_G = 5.0\).
- Hamiltonian parameters (same across all \(L\)):
  - \(J_\mathrm{hop} = 1.0\)
  - \(J_\mathrm{spin} = 0.5\)
  - \(m = 0.1\)
  - \(U_\mathrm{onsite} = 1.0\)
  - \(g_\mathrm{gauge} = 0.5\)
  - \(\lambda_G = 5.0\)
- States:
  - **One skyrmion**: single localized texture.
  - **Two skyrmions, overlapping**: two textures centered at the same location.
  - **Two skyrmions, separated**: two textures centered far apart and combined as a product-state superposition.

For each configuration we compute **expectation values in the chosen product states**:
- \(E_1, E_\mathrm{ov}, E_\mathrm{sep}\): energy expectations.
- \(V_1, V_\mathrm{ov}, V_\mathrm{sep}\): Gauss-law “violation” expectations (the Gauss penalty term).
- \(\Delta E = E_\mathrm{ov} - E_\mathrm{sep}\): direct energy cost of overlap vs separation.
- \(\Delta V = V_\mathrm{ov} - V_\mathrm{sep}\): excess Gauss-law penalty for overlap vs separation.

These are **not** eigenvalues; they are expectation values of \(H\) and the Gauss term in fixed trial states.

---

## Raw results (λ_G = 5.0)

From `skyrmion_scaling.py` and the exported CSV (`scaling_lambda5.csv`):

| L |   E₁      |  E_ov     |  E_sep    | ΔE = E_ov − E_sep |   V₁     |  V_ov    |  V_sep   | ΔV = V_ov − V_sep |
|---|-----------|-----------|-----------|--------------------|----------|----------|----------|-------------------|
| 2 | 29.118145 | 56.981598 | 55.185957 |  1.795642          | 6.040651 | 11.329998| 10.993418| 0.336580          |
| 3 | 36.873738 | 89.277122 | 80.338675 |  8.938447          | 8.210309 | 18.345675| 16.674259| 1.671415          |
| 4 | 34.176589 | 91.964022 | 73.362037 | 18.601984          | 8.643904 | 19.920507| 16.323418| 3.597089          |
| 5 | 28.628570 | 87.832695 | 70.969060 | 16.863634          | 8.717626 | 20.299123| 17.089584| 3.209539          |
| 6 | 20.859221 | 79.249688 | 55.816778 | 23.432910          | 8.698511 | 20.138058| 15.653023| 4.485035          |

For convenience, the summary view the script prints:

- \(L = 2\): \(\Delta E = 1.795642,\;\Delta V = 0.336580,\;E_1 = 29.118145,\;V_1 = 6.040651\)
- \(L = 3\): \(\Delta E = 8.938447,\;\Delta V = 1.671415,\;E_1 = 36.873738,\;V_1 = 8.210309\)
- \(L = 4\): \(\Delta E = 18.601984,\;\Delta V = 3.597089,\;E_1 = 34.176589,\;V_1 = 8.643904\)
- \(L = 5\): \(\Delta E = 16.863634,\;\Delta V = 3.209539,\;E_1 = 28.628570,\;V_1 = 8.717626\)
- \(L = 6\): \(\Delta E = 23.432910,\;\Delta V = 4.485035,\;E_1 = 20.859221,\;V_1 = 8.698511\)

---

## Observations

### 1. Overlap cost grows with L

The direct overlap penalty
\[
\Delta E(L) = E_\mathrm{ov}(L) - E_\mathrm{sep}(L)
\]
increases from \(\sim 1.8\) at \(L = 2\) to \(\sim 23.4\) at \(L = 6\). The growth is not strictly monotone (L=4 vs L=5), but overall:

- \(\Delta E(2) \ll \Delta E(6)\).

Within this toy, that supports the statement:

> In this gauge-constrained product-state model, overlapping two skyrmion textures becomes increasingly disfavoured relative to separating them as the system size grows.

This is **only** about this Hamiltonian and these trial states. We do not generalize beyond that.

### 2. Gauss-law penalty tracks the energy cost

The Gauss violation excess
\[
\Delta V(L) = V_\mathrm{ov}(L) - V_\mathrm{sep}(L)
\]
also grows with \(L\), from \(\sim 0.34\) to \(\sim 4.49\).

The ratio \(\Delta E / \Delta V\) is numerically close to \(\lambda_G = 5.0\) for all \(L\) in this test:

- \(L = 2: \Delta E/\Delta V \approx 5.33\)
- \(L = 3: \Delta E/\Delta V \approx 5.35\)
- \(L = 4: \Delta E/\Delta V \approx 5.17\)
- \(L = 5: \Delta E/\Delta V \approx 5.25\)
- \(L = 6: \Delta E/\Delta V \approx 5.22\)

So, within numerical noise and the crudeness of the trial states:

> The overlap penalty is dominated by the Gauss-law term, with \(\Delta E \approx \lambda_G \,\Delta V\).

This is consistent with the idea that the constraint sector (“Gauss is the data-protection system”) is the main mechanism that suppresses overlap in this toy.

### 3. Single-skyrmion Gauss violation saturates

The single-skyrmion Gauss expectation \(V_1(L)\) appears to saturate for \(L \ge 3\) at \(\sim 8.6\text{–}8.7\), while \(E_1(L)\) decreases as \(L\) grows:

- \(V_1(3) \approx 8.21\)
- \(V_1(4) \approx 8.64\)
- \(V_1(5) \approx 8.72\)
- \(V_1(6) \approx 8.70\)

This is consistent with a **localized Gauss “cloud”** around the texture: enlarging the lattice doesn’t significantly change the local constraint cost once the texture fits comfortably inside.

Again, we keep this interpretation strictly inside the model.

---

## What we **are not** claiming

- We are **not** claiming to have realized true fermionic exchange statistics.
- We are **not** claiming a derivation of the Pauli principle from this setup.
- We are **not** extrapolating to continuum field theory or real materials.

All we can honestly say from this dataset is:

> In this finite, gauge-constrained lattice model, the energy penalty for overlapping two skyrmion-like product states grows with system size and is largely controlled by the Gauss-law term. The constraint sector behaves like a local “overlap suppression mechanism” that scales with \(L\) in a nontrivial way.

If the data do not show more than this, we do not say more than this.

---

## Reproducibility

- Script: `skyrmion_scaling.py`
- Parameters: `--L-min 2 --L-max 6 --lambda-G 5.0`
- Output CSV: `scaling_lambda5.csv`

Running:

```bash
python skyrmion_scaling.py --L-min 2 --L-max 6 --lambda-G 5.0

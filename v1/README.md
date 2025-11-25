# Substrate Framework — Exploring Emergent Structure from Quantum Substrates

**Computational toy models investigating how classical-like behavior can arise from finite Hilbert spaces with local Hamiltonians and constraint operators.**

## Philosophy

This research explores a minimal set of assumptions:

1. **Hilbert Space Realism**: Physical states are vectors in a Hilbert space
2. **Unitary Evolution**: Dynamics arise from local Hermitian Hamiltonians  
3. **Emergent Structure**: Classical-like patterns emerge as stable configurations

The goal is to investigate how far these assumptions can take us in understanding structured behavior in finite quantum systems.

**Critical caveat**: These are toy models on small lattices (2×2 to 8×8). No claims are made about real particle physics, the Standard Model, or quantum gravity. All statements about "emergence" refer solely to patterns observed within these finite models.

## Recent Results

### Constraint-Driven Exclusion (November 2025)

In a 2D lattice model with Z₂ gauge fields and Gauss-law constraints:

- **Observation**: Overlapping skyrmion-like patterns exhibit higher constraint violations and energies than spatially separated patterns
- **Scaling**: The energy penalty ΔE grows ~13× from L=2 to L=6, suggesting the mechanism is not purely a finite-size artifact
- **Mechanism**: The effect is dominated by Gauss-law violation penalties (ΔE ≈ λ_G × ΔV)

**What this shows**: Local gauge-like constraints can produce pattern-dependent energy penalties that favor spatial separation over overlap.

**What this does NOT show**: True fermionic exchange statistics, Pauli exclusion principle, or antisymmetric wave functions under particle exchange.

See `docs/skyrmion_overlap_scaling.md` for detailed analysis.

### Other Explorations

- **Maxwell Emergence**: Demonstration that gauge field dynamics consistent with Maxwell's equations can arise from substrate constraints ("Mr. Magnetic")
- **Bell Correlations**: CHSH violation in substrate models without non-local connections
- **Structure Formation**: Hierarchical clustering in "defrag gravity" simulations

Each of these is documented as what was observed in specific toy models, without extrapolation to real physics.

## Code Structure

### Main Simulations

- `src/skyrmion_scaling.py` → Product-state scaling experiments (L=2 to L=8)
- `src/gauge_exclusion_with_gauss.py` → Gauss-constraint exclusion tests
- `src/mr_magnetic_demo.py` → Maxwell equation emergence
- `src/bell_violation_substrate.py` → CHSH correlation tests

All scripts run in seconds to minutes on standard laptops.

### Data

- `data/scaling_lambda5.csv` → Raw scaling data for λ_G = 5.0
- Additional datasets documented in respective directories

## Methodology

**Product-state expectations**: Most results use expectation values computed on fixed product states, not full many-body eigenstate calculations. This allows scaling to larger lattices but limits interpretation.

**Finite systems**: Exact quantum simulation hits exponential walls. Current results are from L×L lattices with L ≤ 8. Claims about "emergence" must be interpreted within these scale limitations.

**Reproducibility**: All parameters, initial conditions, and measurement procedures are documented. Scripts include command-line arguments for parameter sweeps.

## Current Limitations

1. **Scale**: Cannot demonstrate thermodynamic limit or continuum behavior
2. **Exact simulation**: Limited to ~10⁶ dimensional Hilbert spaces
3. **Product states**: Not exploring full many-body entanglement
4. **Toy models**: Hamiltonians are chosen for tractability, not physical realism

## Documents

- Framework overview: `docs/Substrate_Physics.pdf` (November 21, 2025)
- Scaling analysis: `docs/skyrmion_overlap_scaling.md`
- Earlier versions available in `docs/archive/`

## Principle

**If the data do not show it, we do not say it.**

All results are presented with explicit acknowledgment of what is observed versus what is speculated. Interpretive hypotheses are clearly labeled as such.

## License

- Code: MIT License
- Documentation: CC-BY-4.0

## Contact

Benjamin Bray  
Independent Researcher  
Wildwood, Florida

Questions, criticisms, and feedback welcome.

---

*This is exploratory computational physics on finite systems. Results are shared for transparency and potential collaborative development, not as claims about fundamental physics.*
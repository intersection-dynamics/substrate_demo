# Substrate Demo

This repository contains a collection of computational toy models exploring
quantum mechanics from a "substrate" perspective. These models are not intended
as new physics, but as numerical experiments for understanding:

- Decoherence structures
- Gauge-like consistency constraints
- Internal graph-like Hilbert-space decompositions
- Effective dynamics arising from structured environments
- Benchmarks against standard quantum predictions (e.g. CHSH)

The code here is exploratory, iterative, and openly documented so that each idea
can be tested transparently and compared against known quantum behaviors.

## Example Components

### CHSH Benchmark (Realistic SPDC Model)
`experiment_6_chsh_crystal.py` simulates:

- A pulsed SPDC-like photon source
- Finite entanglement visibility
- Detector inefficiency and dark counts
- Random analyzer angles and coincidence detection
- Computation of CHSH correlations and S-values

### Other Demos
Additional experiments explore:

- Emergent gauge invariance (U(1), SU(2), SU(3) toy models)
- Decoherence landscapes
- Substrate–observable decompositions
- Simple geometric or topological effects in toy Hilbert spaces

These are meant as computational playgrounds rather than finished theories.

## How to Run
Requires Python 3 and NumPy.


#!/usr/bin/env python3
"""
config.py

Central configuration for the 3D substrate research bundle.

This file is meant to be the **single place** you tweak:
- Lattice and engine parameters
- Sweep ranges (lambda_G, J_exch)
- Perturbation counts/strengths
- Which experiments to run
- Output directory and RNG seed

Other scripts (run_all_experiments.py, sweeps/tests) import from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# =============================================================================
# Basic model / lattice / engine config
# =============================================================================

@dataclass
class LatticeConfig:
    Lx: int = 2
    Ly: int = 2
    Lz: int = 2


@dataclass
class EngineConfig:
    """
    Shared "background" parameters for TwoFermion3DParams.
    The parts that change between experiments (lambda_G, J_exch)
    are handled in the experiment-specific configs below.
    """

    J_hop: float = 1.0
    m: float = 0.1
    g_defrag: float = 1.0
    sigma_defrag: float = 1.0

    # Spin/Gauss sector defaults (can be overridden per experiment):
    lambda_S: float = -1.0
    lambda_T: float = 0.0

    max_eigsh_iter: int = 10000
    k_eigs: int = 1


# =============================================================================
# Experiment-specific configs
# =============================================================================

@dataclass
class GaussSweepConfig:
    """
    Sweep over lambda_G (Gauss constraint strength).
    These values are passed into gauss_sweep.run_gauss_sweep().
    """
    lambda_G_start: float = 0.0
    lambda_G_stop: float = 10.0
    n_points: int = 21


@dataclass
class ExchangeSweepConfig:
    """
    Sweep over J_exch (exchange interaction strength), with lambda_G fixed.
    J_exch_values are passed into exchange_sweep.run_exchange_sweep().
    """
    J_exch_start: float = 0.0
    J_exch_stop: float = 2.0
    n_points: int = 21

    # Gauss constraint used in that script (for your own bookkeeping)
    lambda_G_fixed: float = 5.0


@dataclass
class SequentialPerturbationConfig:
    """
    Sequential perturbation test:
    - Redundancy vs energy/structure.
    """
    n_perturbations: int = 15
    perturbation_strength: float = 0.2

    # Effective Hamiltonian “sector”:
    lambda_G: float = 5.0
    J_exch: float = 1.0


@dataclass
class InfoExtractionConfig:
    """
    Information extraction test:
    - Substrate entropy vs classical mutual information.
    """
    n_perturbations: int = 20
    perturbation_strength: float = 0.15

    lambda_G: float = 5.0
    J_exch: float = 1.0


# =============================================================================
# Paths, RNG, and run flags
# =============================================================================

@dataclass
class PathsConfig:
    """
    Paths are relative to this file by default.
    """
    base_dir: Path = Path(__file__).resolve().parent
    outputs_dir: Path = Path(__file__).resolve().parent / "outputs"

    def ensure_dirs(self) -> "PathsConfig":
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class RunFlags:
    """
    Toggle which experiments to run from run_all_experiments.py
    without editing that script.
    """
    run_gauss_sweep: bool = True
    run_exchange_sweep: bool = True
    run_sequential_perturbation: bool = True
    run_information_extraction: bool = True


@dataclass
class ExperimentConfig:
    lattice: LatticeConfig
    engine: EngineConfig
    gauss_sweep: GaussSweepConfig
    exchange_sweep: ExchangeSweepConfig
    sequential: SequentialPerturbationConfig
    info_extraction: InfoExtractionConfig
    paths: PathsConfig
    run_flags: RunFlags
    rng_seed: int = 42  # master RNG seed


# =============================================================================
# Global config instance
# =============================================================================

paths = PathsConfig().ensure_dirs()

experiment_config = ExperimentConfig(
    lattice=LatticeConfig(),
    engine=EngineConfig(),
    gauss_sweep=GaussSweepConfig(),
    exchange_sweep=ExchangeSweepConfig(),
    sequential=SequentialPerturbationConfig(),
    info_extraction=InfoExtractionConfig(),
    paths=paths,
    run_flags=RunFlags(),
    rng_seed=42,
)


# =============================================================================
# Convenience helper functions for sweeps
# =============================================================================

def lambda_G_array(cfg: ExperimentConfig = experiment_config) -> np.ndarray:
    """Return the lambda_G sweep array."""
    g = cfg.gauss_sweep
    return np.linspace(g.lambda_G_start, g.lambda_G_stop, g.n_points)


def J_exch_array(cfg: ExperimentConfig = experiment_config) -> np.ndarray:
    """Return the J_exch sweep array."""
    e = cfg.exchange_sweep
    return np.linspace(e.J_exch_start, e.J_exch_stop, e.n_points)

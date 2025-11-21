#!/usr/bin/env python3
"""
two_particle_config_space.py

A small visualization tool for the configuration space of
two indistinguishable particles in 1D, and for symmetric
(bosonic-like) vs antisymmetric (fermionic-like) wavefunctions
defined on that space.

Conceptual goals:
- Show that the *configuration space* for two particles in 1D
  is a square (x1, x2) with x1, x2 in [0, 1].
- Highlight the diagonal x1 = x2 as a "collision" set, which
  can be treated as a removed set / singular line.
- Visualize a symmetric wavefunction that is nonzero on the diagonal.
- Visualize an antisymmetric wavefunction that vanishes on the diagonal,
  embodying a Pauli-like exclusion.

This is a toy, not a full quantum solver.
"""

import numpy as np
import matplotlib.pyplot as plt


def make_grid(n_points: int = 200):
    """
    Build a uniform grid for x1, x2 in [0, 1].

    Parameters
    ----------
    n_points : int
        Number of grid points along each axis.

    Returns
    -------
    x : ndarray, shape (n_points,)
        1D grid.
    X1, X2 : ndarray, shape (n_points, n_points)
        2D meshgrids for x1, x2.
    """
    x = np.linspace(0.0, 1.0, n_points)
    X1, X2 = np.meshgrid(x, x, indexing="ij")
    return x, X1, X2


def base_orbital(x: np.ndarray, mode: int = 1):
    """
    A simple 1D "orbital" on [0,1] with hard-wall-ish boundaries:
    phi_n(x) ~ sin(n * pi * x).

    Parameters
    ----------
    x : ndarray
        Positions x in [0,1].
    mode : int
        Mode index n.

    Returns
    -------
    phi : ndarray
        Orbital values at each x.
    """
    return np.sin(mode * np.pi * x)


def build_two_particle_states(X1: np.ndarray,
                              X2: np.ndarray,
                              n1: int = 1,
                              n2: int = 2):
    """
    Construct symmetric and antisymmetric two-particle wavefunctions
    from simple product orbitals phi_n(x).

    For two orbitals phi_n1, phi_n2 we define:

        psi_sym(x1,x2)  = ( phi_n1(x1) phi_n2(x2) +
                            phi_n2(x1) phi_n1(x2) ) / sqrt(2)

        psi_asym(x1,x2) = ( phi_n1(x1) phi_n2(x2) -
                            phi_n2(x1) phi_n1(x2) ) / sqrt(2)

    The antisymmetric combination vanishes on x1 = x2.

    Parameters
    ----------
    X1, X2 : ndarray
        Meshgrids for x1, x2.
    n1, n2 : int
        Mode indices for the two orbitals.

    Returns
    -------
    psi_sym : ndarray
        Symmetric wavefunction.
    psi_asym : ndarray
        Antisymmetric wavefunction.
    """
    x1 = X1[:, 0]
    x2 = X2[0, :]

    phi_n1_x1 = base_orbital(x1, mode=n1)[:, None]  # shape (N,1)
    phi_n2_x1 = base_orbital(x1, mode=n2)[:, None]
    phi_n1_x2 = base_orbital(x2, mode=n1)[None, :]  # shape (1,N)
    phi_n2_x2 = base_orbital(x2, mode=n2)[None, :]

    # Product states
    phi_n1x1_n2x2 = phi_n1_x1 * phi_n2_x2
    phi_n2x1_n1x2 = phi_n2_x1 * phi_n1_x2

    # Symmetric / antisymmetric combos
    norm = np.sqrt(2.0)
    psi_sym = (phi_n1x1_n2x2 + phi_n2x1_n1x2) / norm
    psi_asym = (phi_n1x1_n2x2 - phi_n2x1_n1x2) / norm

    return psi_sym, psi_asym


def plot_configuration_space(X1: np.ndarray,
                             X2: np.ndarray,
                             ax=None):
    """
    Plot the configuration space as a square (x1,x2) with the diagonal x1=x2
    highlighted as the "collision" / coincidence set.

    Parameters
    ----------
    X1, X2 : ndarray
        Meshgrids for x1, x2 in [0,1].
    ax : matplotlib.axes.Axes or None
        Axis to plot into. If None, a new figure+axis is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Light background
    ax.set_title("Two-particle configuration space in 1D\n(x1, x2) ∈ [0,1] × [0,1]")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

    # Plot the diagonal x1 = x2
    x = np.linspace(0.0, 1.0, 400)
    ax.plot(x, x, 'k--', linewidth=1.5, label="coincidence set x₁ = x₂")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")

    return ax


def plot_wavefunction(X1: np.ndarray,
                      X2: np.ndarray,
                      psi: np.ndarray,
                      title: str,
                      ax=None,
                      vmin=None,
                      vmax=None):
    """
    Plot a 2D wavefunction amplitude (real-valued) over configuration space.

    Parameters
    ----------
    X1, X2 : ndarray
        Meshgrids for x1, x2.
    psi : ndarray
        Real-valued wavefunction values psi(x1,x2).
    title : str
        Title for the subplot.
    ax : matplotlib.axes.Axes or None
        Axis to plot into.
    vmin, vmax : float or None
        Color scale limits. If None, derived from psi.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    if vmin is None or vmax is None:
        absmax = np.max(np.abs(psi))
        vmin = -absmax
        vmax = absmax

    im = ax.imshow(
        psi.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        cmap="coolwarm",
    )
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    # Build grid and basic configuration space visualization
    x, X1, X2 = make_grid(n_points=200)

    # Build symmetric and antisymmetric two-particle states
    psi_sym, psi_asym = build_two_particle_states(X1, X2, n1=1, n2=2)

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))

    ax_conf = fig.add_subplot(1, 3, 1)
    ax_sym = fig.add_subplot(1, 3, 2)
    ax_asym = fig.add_subplot(1, 3, 3)

    # 1) Configuration space + diagonal
    plot_configuration_space(X1, X2, ax=ax_conf)

    # 2) Symmetric wavefunction
    plot_wavefunction(
        X1,
        X2,
        psi_sym,
        title="Symmetric ψₛᵧₘ(x₁,x₂)\n(bosonic-like)",
        ax=ax_sym,
    )

    # 3) Antisymmetric wavefunction
    plot_wavefunction(
        X1,
        X2,
        psi_asym,
        title="Antisymmetric ψₐₛᵧₘ(x₁,x₂)\n(fermionic-like, zero on x₁=x₂)",
        ax=ax_asym,
    )

    fig.suptitle(
        "Two-particle configuration space in 1D and symmetric/antisymmetric states",
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0.0, 1, 0.95])

    # Print some explanation to the console
    print("=" * 80)
    print("TWO-PARTICLE CONFIGURATION SPACE TOY")
    print("=" * 80)
    print("""
We consider two particles moving in 1D, with positions x₁, x₂ ∈ [0,1].

- The *configuration space* is the set of all (x₁, x₂) pairs.
  We visualize this as a square [0,1] × [0,1].

- The diagonal x₁ = x₂ is the "coincidence set" where the two
  particles occupy the same position. In more advanced treatments
  of indistinguishable particles, this diagonal can be treated as
  a removed set / singular line, which gives configuration space
  a nontrivial topology (a 'hole').

- ψₛᵧₘ(x₁,x₂) is a symmetric two-particle wavefunction built from
  simple 1D orbitals φₙ(x) ~ sin(nπx). It is nonzero on the diagonal.

- ψₐₛᵧₘ(x₁,x₂) is the antisymmetric combination. Swapping x₁ and x₂
  flips its sign, and it vanishes on x₁ = x₂. This is a toy version
  of fermionic antisymmetry and Pauli exclusion: no two fermions can
  occupy the same one-particle state, which shows up here as ψ = 0
  on the coincidence line.
""")
    print("=" * 80)
    print("Close the figure window to exit.")
    print("=" * 80)

    plt.show()


if __name__ == "__main__":
    main()

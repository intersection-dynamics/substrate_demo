#!/usr/bin/env python3
"""
gauge_scalar_coupled_probe.py

Toy 2D coupled gauge + scalar (matter) substrate with a Hamiltonian-style
gauge coupling using covariant derivatives.

Fields
------
  - psi(x,y) : complex scalar "matter" field on sites
  - ax, ay   : real gauge potentials on sites
  - ex, ey   : real electric fields on sites

Dynamics (toy model)
--------------------
Gauge sector (wave-like, sourced by matter currents):

    d ax / dt = ex
    d ay / dt = ey

    B = ∂x ay - ∂y ax

    d ex / dt =  c^2 (∂y B) - jx
    d ey / dt = -c^2 (∂x B) - jy

where (jx, jy) are lattice gauge currents computed from psi using
a covariant derivative.

Matter sector (Schrödinger/Gross–Pitaevskii-like, minimally coupled):

    Define covariant derivatives (site-based approximation):

        D_x psi = ∂x psi - i q ax psi
        D_y psi = ∂y psi - i q ay psi

    The toy Hamiltonian density is:

        H = κ |Dψ|^2 + m^2 |ψ|^2 + λ |ψ|^4
            + ½ (Ex^2 + Ey^2 + c^2 B^2)

    and the equation of motion is:

        i d psi / dt = δH / δψ*  ≈  -κ (D_x^† D_x + D_y^† D_y) ψ
                                   + m^2 ψ
                                   + λ |ψ|^2 ψ

We track:
  - E_matter(t)
  - E_gauge(t)
  - E_total(t)

and write diagnostics to CSV and a plot.

We support two time integration schemes:

  --scheme symplectic (default)
      Symplectic-style "kick–drift–kick" for (A, E) with a midpoint
      update for ψ in the middle of the step.

  --scheme rk4
      Original 4th-order Runge–Kutta for the full coupled system
      (kept mostly for comparison / debugging).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore
    CUPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Utility: choose backend
# ---------------------------------------------------------------------------

def get_xp(use_gpu: bool):
    if use_gpu and CUPY_AVAILABLE:
        print("✓ GPU (CuPy) detected - using GPU")
        return cp, True
    else:
        if use_gpu and not CUPY_AVAILABLE:
            print("⚠ Requested GPU, but CuPy not available. Falling back to NumPy.")
        else:
            print("✓ Using NumPy (CPU)")
        return np, False


# ---------------------------------------------------------------------------
# Gauge + scalar substrate
# ---------------------------------------------------------------------------

class GaugeScalar2D:
    """
    Toy 2D gauge + scalar substrate with local Hamiltonian-style dynamics.
    """

    def __init__(
        self,
        L: int,
        dx: float = 1.0,
        dt: float = 0.02,
        c: float = 1.0,
        kappa: float = 1.0,
        m2: float = 0.0,
        lam: float = 0.0,
        g_coup: float = 0.5,   # used as "charge" q in the covariant derivative
        use_gpu: bool = True,
    ):
        self.L = L
        self.dx = dx
        self.dt = dt
        self.c = c
        self.kappa = kappa
        self.m2 = m2
        self.lam = lam
        self.g_coup = g_coup  # interpret as q

        self.xp, self.is_gpu = get_xp(use_gpu)

        # scalar matter field
        self.psi = self.xp.zeros((L, L), dtype=self.xp.complex128)

        # gauge fields
        self.ax = self.xp.zeros((L, L), dtype=self.xp.float64)
        self.ay = self.xp.zeros((L, L), dtype=self.xp.float64)
        self.ex = self.xp.zeros((L, L), dtype=self.xp.float64)
        self.ey = self.xp.zeros((L, L), dtype=self.xp.float64)

    # ---------- spatial operators ----------

    def roll(self, arr, shift, axis):
        return self.xp.roll(arr, shift=shift, axis=axis)

    def grad_x(self, field):
        # central difference, periodic
        return (self.roll(field, -1, axis=0) - self.roll(field, +1, axis=0)) / (
            2.0 * self.dx
        )

    def grad_y(self, field):
        return (self.roll(field, -1, axis=1) - self.roll(field, +1, axis=1)) / (
            2.0 * self.dx
        )

    def laplacian(self, field):
        xp = self.xp
        dx2 = self.dx ** 2
        f_ip = self.roll(field, -1, axis=0)
        f_im = self.roll(field, +1, axis=0)
        f_jp = self.roll(field, -1, axis=1)
        f_jm = self.roll(field, +1, axis=1)
        return (f_ip + f_im + f_jp + f_jm - 4.0 * field) / dx2

    # ---------- physical fields ----------

    def magnetic_field(self):
        """
        B_z = ∂x ay - ∂y ax
        """
        return self.grad_x(self.ay) - self.grad_y(self.ax)

    # ---------- covariant derivatives & currents ----------

    def covariant_derivatives(self):
        """
        Compute covariant derivatives D_x psi, D_y psi using a site-based
        minimal coupling:

          D_x psi = ∂x psi - i q ax psi
          D_y psi = ∂y psi - i q ay psi

        where q is taken from self.g_coup.
        """
        xp = self.xp
        psi = self.psi

        dpsi_dx = self.grad_x(psi)
        dpsi_dy = self.grad_y(psi)

        q = self.g_coup

        Dpsi_x = dpsi_dx - 1j * q * self.ax * psi
        Dpsi_y = dpsi_dy - 1j * q * self.ay * psi

        return Dpsi_x, Dpsi_y

    def matter_currents(self):
        """
        Gauge currents derived from the covariant derivative:

          j_i = q * Im(psi* D_i psi)

        This is consistent with a kinetic term ~ |D psi|^2 in the Hamiltonian.
        """
        xp = self.xp
        psi = self.psi
        psi_conj = xp.conjugate(psi)

        Dpsi_x, Dpsi_y = self.covariant_derivatives()

        q = self.g_coup
        jx = q * xp.imag(psi_conj * Dpsi_x)
        jy = q * xp.imag(psi_conj * Dpsi_y)
        return jx, jy

    # ---------- RHS for coupled system ----------

    def rhs_gauge(self):
        """
        Right-hand side for gauge fields:
          d ax / dt = ex
          d ay / dt = ey
          d ex / dt =  c^2 (∂y B) - jx
          d ey / dt = -c^2 (∂x B) - jy

        with currents from matter_currents().
        """
        xp = self.xp
        ax, ay, ex, ey = self.ax, self.ay, self.ex, self.ey

        B = self.magnetic_field()
        dB_dx = self.grad_x(B)
        dB_dy = self.grad_y(B)

        jx, jy = self.matter_currents()

        dax_dt = ex
        day_dt = ey
        dex_dt = (self.c ** 2) * dB_dy - jx
        dey_dt = -(self.c ** 2) * dB_dx - jy

        return dax_dt, day_dt, dex_dt, dey_dt

    def rhs_matter(self):
        """
        Right-hand side for scalar field using covariant derivatives:

          H_scalar = κ |Dψ|^2 + m^2 |ψ|^2 + λ |ψ|^4

        The (approximate) equation of motion is:

          i d psi / dt ≈ -κ (D_x^† D_x + D_y^† D_y) ψ
                         + m^2 ψ
                         + λ |ψ|^2 ψ

        We implement D_x^† D_x + D_y^† D_y via a discrete divergence of the
        covariant derivatives:

          cov_lap psi ≈ ∂x(D_x psi) + ∂y(D_y psi)

        Then:

          d psi / dt = -i [ -κ cov_lap psi
                            + m^2 psi
                            + λ |psi|^2 psi ]
        """
        xp = self.xp
        psi = self.psi

        # Covariant derivatives
        Dpsi_x, Dpsi_y = self.covariant_derivatives()

        # Divergence of covariant derivative (covariant "Laplacian")
        Dpsi_x_dx = self.grad_x(Dpsi_x)
        Dpsi_y_dy = self.grad_y(Dpsi_y)
        cov_lap = Dpsi_x_dx + Dpsi_y_dy

        abs2 = xp.abs(psi) ** 2

        rhs = (
            -self.kappa * cov_lap
            + self.m2 * psi
            + self.lam * abs2 * psi
        )
        dpsi_dt = -1j * rhs
        return dpsi_dt

    # ---------- time integration (RK4) ----------

    def step_rk4(self):
        """
        4th-order Runge–Kutta step for the full coupled system.
        Kept here for comparison / debugging.
        """
        dt = self.dt

        psi0 = self.psi
        ax0, ay0 = self.ax, self.ay
        ex0, ey0 = self.ex, self.ey

        # Stage 1
        dpsi1 = self.rhs_matter()
        dax1, day1, dex1, dey1 = self.rhs_gauge()

        # Stage 2
        self.psi = psi0 + 0.5 * dt * dpsi1
        self.ax = ax0 + 0.5 * dt * dax1
        self.ay = ay0 + 0.5 * dt * day1
        self.ex = ex0 + 0.5 * dt * dex1
        self.ey = ey0 + 0.5 * dt * dey1

        dpsi2 = self.rhs_matter()
        dax2, day2, dex2, dey2 = self.rhs_gauge()

        # Stage 3
        self.psi = psi0 + 0.5 * dt * dpsi2
        self.ax = ax0 + 0.5 * dt * dax2
        self.ay = ay0 + 0.5 * dt * day2
        self.ex = ex0 + 0.5 * dt * dex2
        self.ey = ey0 + 0.5 * dt * dey2

        dpsi3 = self.rhs_matter()
        dax3, day3, dex3, dey3 = self.rhs_gauge()

        # Stage 4
        self.psi = psi0 + dt * dpsi3
        self.ax = ax0 + dt * dax3
        self.ay = ay0 + dt * day3
        self.ex = ex0 + dt * dex3
        self.ey = ey0 + dt * dey3

        dpsi4 = self.rhs_matter()
        dax4, day4, dex4, dey4 = self.rhs_gauge()

        # Combine stages
        self.psi = psi0 + (dt / 6.0) * (dpsi1 + 2 * dpsi2 + 2 * dpsi3 + dpsi4)
        self.ax = ax0 + (dt / 6.0) * (dax1 + 2 * dax2 + 2 * dax3 + dax4)
        self.ay = ay0 + (dt / 6.0) * (day1 + 2 * day2 + 2 * day3 + day4)
        self.ex = ex0 + (dt / 6.0) * (dex1 + 2 * dex2 + 2 * dex3 + dex4)
        self.ey = ey0 + (dt / 6.0) * (dey1 + 2 * dey2 + 2 * dey3 + dey4)

    # ---------- time integration (symplectic-style leapfrog) ----------

    def step_symplectic(self):
        """
        Symplectic-style time step using a kick–drift–kick scheme.

        - Gauge sector (ax, ay, ex, ey) is updated with a velocity-Verlet /
          leapfrog integrator.

        - Matter sector psi is advanced in the middle using a 2nd-order
          midpoint method (time-reversible) in the updated gauge background.
        """
        dt = self.dt

        # --- 1) Kick: half-step update of gauge momenta (ex, ey) ---
        _, _, dex_dt0, dey_dt0 = self.rhs_gauge()
        self.ex += 0.5 * dt * dex_dt0
        self.ey += 0.5 * dt * dey_dt0

        # --- 2) Drift: full-step update of gauge coordinates (ax, ay) ---
        self.ax += dt * self.ex
        self.ay += dt * self.ey

        # --- 3) Matter evolution: 2nd-order midpoint for psi ---
        psi0 = self.psi

        # Stage 1: evaluate dpsi/dt at psi0, using updated (ax, ay)
        dpsi1 = self.rhs_matter()

        # Temporary mid-step psi
        self.psi = psi0 + 0.5 * dt * dpsi1

        # Stage 2: evaluate dpsi/dt at mid-step
        dpsi2 = self.rhs_matter()

        # Full update for psi
        self.psi = psi0 + dt * dpsi2

        # --- 4) Kick: second half-step update of (ex, ey) ---
        _, _, dex_dt1, dey_dt1 = self.rhs_gauge()
        self.ex += 0.5 * dt * dex_dt1
        self.ey += 0.5 * dt * dey_dt1

    # ---------- energies ----------

    def energies(self):
        """
        Compute E_matter, E_gauge, E_total (toy expressions) from the
        Hamiltonian density:

          H = κ |Dψ|^2 + m^2 |ψ|^2 + λ |ψ|^4
              + ½ (Ex^2 + Ey^2 + c^2 B^2)
        """
        xp = self.xp
        dx2 = self.dx ** 2

        # matter energy density using covariant derivatives
        Dpsi_x, Dpsi_y = self.covariant_derivatives()
        grad2 = xp.abs(Dpsi_x) ** 2 + xp.abs(Dpsi_y) ** 2

        abs2 = xp.abs(self.psi) ** 2
        E_matter_density = (
            self.kappa * grad2
            + 0.5 * self.m2 * abs2
            + 0.25 * self.lam * abs2 ** 2
        )

        # gauge energy density
        B = self.magnetic_field()
        E_gauge_density = 0.5 * (
            self.ex ** 2 + self.ey ** 2 + (self.c ** 2) * B ** 2
        )

        E_matter = float(dx2 * xp.sum(E_matter_density))
        E_gauge = float(dx2 * xp.sum(E_gauge_density))
        E_total = E_matter + E_gauge

        return E_matter, E_gauge, E_total

    # ---------- host copies ----------

    def psi_numpy(self):
        if self.is_gpu:
            return np.array(cp.asnumpy(self.psi))
        else:
            return np.array(self.psi)


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------

def initialize_gaussian_lump(sim: GaugeScalar2D,
                             amp: float = 1.0,
                             sigma: float = 8.0,
                             momentum: float = 0.0):
    """
    Initialize psi as a single Gaussian lump in the center, possibly with
    a plane-wave phase e^{i p x}.

      psi(x,y) = amp * exp(-r^2/(2 sigma^2)) * exp(i p x)
    """
    L = sim.L
    xp = sim.xp

    xs = xp.arange(L)
    ys = xp.arange(L)
    X, Y = xp.meshgrid(xs, ys, indexing="ij")

    cx = (L // 2)
    cy = (L // 2)
    dx = X - cx
    dy = Y - cy
    r2 = dx * dx + dy * dy

    envelope = amp * xp.exp(-r2 / (2.0 * sigma ** 2))
    phase = xp.exp(1j * momentum * dx)
    sim.psi = envelope * phase

    print(
        "Initialized Gaussian lump: amp=%.3f, sigma=%.1f, momentum=%.3f"
        % (amp, sigma, momentum)
    )


def initialize_small_gauge_noise(sim: GaugeScalar2D, level: float = 0.0):
    """
    Optional: seed small random gauge field; by default level=0.0 (zero).
    """
    if level <= 0.0:
        print("Gauge fields initialized to zero.")
        return

    xp = sim.xp
    rs = xp.random.RandomState(1234)
    sim.ax = level * (rs.rand(sim.L, sim.L) - 0.5)
    sim.ay = level * (rs.rand(sim.L, sim.L) - 0.5)
    sim.ex[...] = 0.0
    sim.ey[...] = 0.0
    print("Gauge fields initialized with small random noise, level=%.3e" % level)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_gauge_scalar_coupled(
    L: int,
    dx: float,
    dt: float,
    c: float,
    kappa: float,
    m2: float,
    lam: float,
    g_coup: float,
    amp_lump: float,
    sigma_lump: float,
    momentum_lump: float,
    gauge_noise: float,
    n_steps: int,
    analysis_interval: int,
    out_prefix: str,
    use_gpu: bool,
    scheme: str = "symplectic",
):
    out_dir = Path(f"{out_prefix}_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" gauge_scalar_coupled_probe.py ")
    print("=" * 72)
    print("Integrator scheme   : %s" % scheme)
    print("L=%d, dx=%.3f, dt=%.4f, c=%.3f" % (L, dx, dt, c))
    print("kappa=%.3f, m2=%.3f, lam=%.3f, g_coup(q)=%.3f" % (kappa, m2, lam, g_coup))
    print("amp_lump=%.3f, sigma_lump=%.1f, momentum=%.3f" %
          (amp_lump, sigma_lump, momentum_lump))
    print("gauge_noise=%.3e" % gauge_noise)
    print("n_steps=%d, analysis_interval=%d" % (n_steps, analysis_interval))
    print("output_dir=%s" % out_dir)
    print("=" * 72)

    sim = GaugeScalar2D(
        L=L,
        dx=dx,
        dt=dt,
        c=c,
        kappa=kappa,
        m2=m2,
        lam=lam,
        g_coup=g_coup,
        use_gpu=use_gpu,
    )

    initialize_gaussian_lump(sim, amp=amp_lump,
                             sigma=sigma_lump,
                             momentum=momentum_lump)
    initialize_small_gauge_noise(sim, level=gauge_noise)

    # Choose integrator
    if scheme.lower() == "rk4":
        stepper = sim.step_rk4
    else:
        stepper = sim.step_symplectic

    times = []
    E_m_list = []
    E_g_list = []
    E_tot_list = []

    # initial energies
    E_m, E_g, E_tot = sim.energies()
    times.append(0.0)
    E_m_list.append(E_m)
    E_g_list.append(E_g)
    E_tot_list.append(E_tot)

    print(
        "[INIT] E_matter=%.6e, E_gauge=%.6e, E_total=%.6e"
        % (E_m, E_g, E_tot)
    )

    # time loop
    for step in range(1, n_steps + 1):
        stepper()

        if step % analysis_interval == 0 or step == n_steps:
            t = step * dt
            E_m, E_g, E_tot = sim.energies()
            times.append(t)
            E_m_list.append(E_m)
            E_g_list.append(E_g)
            E_tot_list.append(E_tot)

            print(
                "[ANALYSIS] step=%6d, t=%8.4f, "
                "E_m=%.6e, E_g=%.6e, E_tot=%.6e"
                % (step, t, E_m, E_g, E_tot)
            )

    times = np.array(times)
    E_m_list = np.array(E_m_list)
    E_g_list = np.array(E_g_list)
    E_tot_list = np.array(E_tot_list)

    # Save CSV
    import pandas as pd

    df = pd.DataFrame(
        {
            "time": times,
            "E_matter": E_m_list,
            "E_gauge": E_g_list,
            "E_total": E_tot_list,
        }
    )
    csv_path = out_dir / f"{out_prefix}_energies.csv"
    df.to_csv(csv_path, index=False)
    print("[RESULT] Saved energies -> %s" % csv_path)

    # Plot energies vs time
    plt.figure(figsize=(8, 4))
    plt.plot(times, E_tot_list, label="E_total")
    plt.plot(times, E_m_list, label="E_matter")
    plt.plot(times, E_g_list, label="E_gauge")
    plt.xlabel("time")
    plt.ylabel("Energy")
    plt.title("Gauge–Matter Energies vs Time (covariant, symplectic)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = out_dir / f"{out_prefix}_energies.png"
    plt.savefig(png_path, dpi=200)
    plt.close()
    print("[RESULT] Saved energy plot -> %s" % png_path)

    # Summary text
    summary_path = out_dir / f"{out_prefix}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("gauge_scalar_coupled_probe summary\n")
        f.write("---------------------------------\n")
        f.write("L = %d\n" % L)
        f.write("dx = %.6f\n" % dx)
        f.write("dt = %.6f\n" % dt)
        f.write("c = %.6f\n" % c)
        f.write("kappa = %.6f\n" % kappa)
        f.write("m2 = %.6f\n" % m2)
        f.write("lam = %.6f\n" % lam)
        f.write("g_coup(q) = %.6f\n" % g_coup)
        f.write("n_steps = %d\n" % n_steps)
        f.write("analysis_interval = %d\n" % analysis_interval)
        f.write("scheme = %s\n" % scheme)
        f.write("\n")
        f.write("Initial E_matter = %.8e\n" % E_m_list[0])
        f.write("Initial E_gauge  = %.8e\n" % E_g_list[0])
        f.write("Initial E_total  = %.8e\n" % E_tot_list[0])
        f.write("Final   E_matter = %.8e\n" % E_m_list[-1])
        f.write("Final   E_gauge  = %.8e\n" % E_g_list[-1])
        f.write("Final   E_total  = %.8e\n" % E_tot_list[-1])
        f.write("\n")
        f.write("Total energy drift (final - initial) = %.8e\n"
                % (E_tot_list[-1] - E_tot_list[0]))
    print("[RESULT] Saved summary -> %s" % summary_path)

    print("=" * 72)
    print(" Run complete. ")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Toy coupled gauge + scalar probe (covariant, symplectic).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--L", type=int, default=128, help="Grid size LxL.")
    parser.add_argument("--dx", type=float, default=1.0, help="Lattice spacing.")
    parser.add_argument("--dt", type=float, default=0.02, help="Time step.")
    parser.add_argument(
        "--c", type=float, default=1.0, help="Gauge wave speed."
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=1.0,
        help="Matter kinetic coefficient in front of |Dψ|^2.",
    )
    parser.add_argument(
        "--m2",
        type=float,
        default=0.0,
        help="Mass-squared term for matter field.",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.1,
        help="Quartic self-coupling for matter.",
    )
    parser.add_argument(
        "--g_coup",
        type=float,
        default=0.5,
        help="Gauge coupling / charge q in the covariant derivative.",
    )
    parser.add_argument(
        "--amp_lump",
        type=float,
        default=1.0,
        help="Amplitude of initial Gaussian lump.",
    )
    parser.add_argument(
        "--sigma_lump",
        type=float,
        default=8.0,
        help="Width (sigma) of initial Gaussian lump.",
    )
    parser.add_argument(
        "--momentum_lump",
        type=float,
        default=0.15,
        help="Plane-wave momentum in x for the lump.",
    )
    parser.add_argument(
        "--gauge_noise",
        type=float,
        default=0.0,
        help="Initial small random gauge noise level.",
    )
    parser.add_argument(
        "--n_steps", type=int, default=4000, help="Number of time steps."
    )
    parser.add_argument(
        "--analysis_interval",
        type=int,
        default=20,
        help="How often to record energies.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="gauge_scalar_coupled",
        help="Prefix for output directory and files.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if GPU is available.",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="symplectic",
        choices=["rk4", "symplectic"],
        help="Time integration scheme.",
    )

    args = parser.parse_args()
    use_gpu = not args.cpu

    run_gauge_scalar_coupled(
        L=args.L,
        dx=args.dx,
        dt=args.dt,
        c=args.c,
        kappa=args.kappa,
        m2=args.m2,
        lam=args.lam,
        g_coup=args.g_coup,
        amp_lump=args.amp_lump,
        sigma_lump=args.sigma_lump,
        momentum_lump=args.momentum_lump,
        gauge_noise=args.gauge_noise,
        n_steps=args.n_steps,
        analysis_interval=args.analysis_interval,
        out_prefix=args.out_prefix,
        use_gpu=use_gpu,
        scheme=args.scheme,
    )


if __name__ == "__main__":
    main()

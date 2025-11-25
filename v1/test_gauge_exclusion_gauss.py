#!/usr/bin/env python
"""
test_gauge_exclusion_gauss.py

Z2 GAUGE + GAUSS LAW + SKYRMION EXCLUSION

Goal:
- Move from "soft" energy penalties to "hard-ish" constraints by
  explicitly enforcing a Z2 Gauss law on a spinor lattice with
  skyrmion-like matter configurations.

Setup:
- 2x2 lattice of matter sites.
- Each site: 4-level (2 field x 2 spin).
- Each nearest-neighbor link: 2-level Z2 gauge "qubit".
- Hopping is gauge-covariant: a_i^† U_ij a_j with U_ij = σ_z on link.
- Gauge field has kinetic term: -g_gauge * σ_x on each link.
- Gauss law at each site s:
    G_s = (-1)^{n_s} * Π_{links touching s} σ_x(link)
  Physical (gauge-invariant) states satisfy G_s |ψ> = +|ψ>.

We then compare:
- One skyrmion.
- Two skyrmions overlapping (same location).
- Two skyrmions separated.

Under:
- No Gauss-law enforcement (lambda_G = 0).
- Strong Gauss-law enforcement (lambda_G > 0).

The quantity of interest is:
    ΔE = E_two_overlap - E_two_separated

If Gauss+gauge structure acts like a hard constraint against
overlapping "fermionic information", ΔE should become large and
positive when lambda_G is big.
"""

import numpy as np
from qutip import Qobj, tensor, qeye, basis, expect


class GaugeLatticeGauss:
    def __init__(self, Nx=2, Ny=2):
        self.Nx = Nx
        self.Ny = Ny
        self.n_sites = Nx * Ny

        # Matter sites: 4 states (2 field x 2 spin)
        self.site_dim = 4

        # Gauge links: 2 states (Z2 gauge qubit)
        self.link_dim = 2

        # Link counting (open boundary)
        self.n_horizontal = Nx * (Ny - 1)      # (i,j) -> (i,j+1)
        self.n_vertical = (Nx - 1) * Ny        # (i,j) -> (i+1,j)
        self.n_links = self.n_horizontal + self.n_vertical

        self.total_dim = (self.site_dim ** self.n_sites) * \
                         (self.link_dim ** self.n_links)

        print(f"[INIT] Gauge lattice: {Nx} x {Ny}")
        print(f"[INIT] Sites: {self.n_sites} (4^{self.n_sites})")
        print(f"[INIT] Links: {self.n_links} (2^{self.n_links})")
        print(f"[INIT] Total Hilbert dim: {self.total_dim}")

        self.H = None
        self.state = None

    # -----------------------
    # Geometry
    # -----------------------
    def site_index(self, i, j):
        return i * self.Ny + j

    def site_coords(self, s):
        i = s // self.Ny
        j = s % self.Ny
        return i, j

    def link_index_horizontal(self, i, j):
        """
        Horizontal link at (i,j) -> (i,j+1).
        Only valid for j in [0, Ny-2].
        """
        return i * (self.Ny - 1) + j

    def link_index_vertical(self, i, j):
        """
        Vertical link at (i,j) -> (i+1,j).
        Only valid for i in [0, Nx-2].
        """
        return self.n_horizontal + i * self.Ny + j

    def star_links(self, site):
        """
        Links touching a given site (the "star" of s) for Gauss law.
        """
        i, j = self.site_coords(site)
        links = []

        # Horizontal left: (i,j-1) -> (i,j)
        if j - 1 >= 0:
            links.append(self.link_index_horizontal(i, j - 1))

        # Horizontal right: (i,j) -> (i,j+1)
        if j + 1 < self.Ny:
            links.append(self.link_index_horizontal(i, j))

        # Vertical up: (i-1,j) -> (i,j)
        if i - 1 >= 0:
            links.append(self.link_index_vertical(i - 1, j))

        # Vertical down: (i,j) -> (i+1,j)
        if i + 1 < self.Nx:
            links.append(self.link_index_vertical(i, j))

        return links

    # -----------------------
    # Local operators
    # -----------------------
    def build_operators(self):
        """
        Build single-site matter & single-link gauge operators.
        Returns:
            n_field, sx, sy, sz, a, a_dag, gauge_x, gauge_z, n_parity
        """
        # Matter number operator: excited states (2 and 3)
        n_field = Qobj([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        # Pauli-like spin ops acting on both vac and excited spin
        sx = Qobj([[0, 1, 0, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]]) / 2

        sy = Qobj([[0, -1j, 0, 0],
                   [1j, 0, 0, 0],
                   [0, 0, 0, -1j],
                   [0, 0, 1j, 0]]) / 2

        sz = Qobj([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, -1]]) / 2

        # Field ladder operators (vac <-> excited)
        a = Qobj([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [1, 0, 0, 0],
                  [0, 1, 0, 0]])
        a_dag = a.dag()

        # Z2 gauge link ops
        gauge_x = Qobj([[0, 1],
                        [1, 0]])
        gauge_z = Qobj([[1, 0],
                        [0, -1]])

        # Parity operator: (-1)^{n_field}
        # n_field eigenvalues: 0,0,1,1 -> parity: 1,1,-1,-1
        n_parity = Qobj([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, -1]])

        return n_field, sx, sy, sz, a, a_dag, gauge_x, gauge_z, n_parity

    # -----------------------
    # Embedding helpers
    # -----------------------
    def matter_op(self, op, site):
        """
        Embed a site-local matter operator 'op' at given site.
        """
        ops = []
        for s in range(self.n_sites):
            if s == site:
                ops.append(op)
            else:
                ops.append(qeye(self.site_dim))
        # Gauge factor: identity on all links
        for _ in range(self.n_links):
            ops.append(qeye(self.link_dim))
        return tensor(ops)

    def gauge_op(self, op, link):
        """
        Embed a link-local gauge operator 'op' at given link.
        """
        ops = []
        # Matter sites: identity
        for _ in range(self.n_sites):
            ops.append(qeye(self.site_dim))
        # Gauge links
        for l in range(self.n_links):
            if l == link:
                ops.append(op)
            else:
                ops.append(qeye(self.link_dim))
        return tensor(ops)

    def gauge_covariant_hop(self, a_dag, site1, a, site2, link):
        """
        Gauge-covariant hopping term a†_i U_ij a_j with U_ij = σ_z(link).
        """
        n_field, sx, sy, sz, a_local, a_dag_local, gauge_x, gauge_z, n_parity = \
            self.build_operators()

        ops = []
        # Matter part (sites)
        for s in range(self.n_sites):
            if s == site1:
                ops.append(a_dag_local)
            elif s == site2:
                ops.append(a_local)
            else:
                ops.append(qeye(self.site_dim))

        # Gauge part (links)
        for l in range(self.n_links):
            if l == link:
                ops.append(gauge_z)
            else:
                ops.append(qeye(self.link_dim))

        return tensor(ops)

    # -----------------------
    # Gauss-law operator
    # -----------------------
    def gauss_operator(self, site):
        """
        Z2 Gauss-law operator at a site:

            G_s = (-1)^{n_s} * Π_{links in star(s)} σ_x(link)

        Physical states satisfy G_s |ψ> = +|ψ>.
        """
        n_field, sx, sy, sz, a_local, a_dag_local, gauge_x, gauge_z, n_parity = \
            self.build_operators()

        ops = []

        # Matter parity at this site
        for s in range(self.n_sites):
            if s == site:
                ops.append(n_parity)
            else:
                ops.append(qeye(self.site_dim))

        # Gauge σ_x on star links, identity elsewhere
        star = set(self.star_links(site))
        for l in range(self.n_links):
            if l in star:
                ops.append(gauge_x)
            else:
                ops.append(qeye(self.link_dim))

        return tensor(ops)

    # -----------------------
    # Hamiltonian construction
    # -----------------------
    def build_hamiltonian(self, J_hop=1.0, J_spin=0.5,
                          mass=0.1, U_onsite=1.0, g_gauge=0.5,
                          lambda_G=0.0):
        """
        Build Hamiltonian:

        H = H_hop + H_spin + H_mass + H_onsite + H_gauge_kin + H_Gauss

        where:
        - H_hop: gauge-covariant hopping
        - H_spin: simple sz-sz nearest-neighbor interactions
        - H_mass: mass * n_field
        - H_onsite: U_onsite * n_field^2 (stiffness)
        - H_gauge_kin: -g_gauge * Σ σ_x(link)
        - H_Gauss: λ_G Σ_s (I - G_s)^2
        """
        print("[BUILD] Constructing Hamiltonian...")
        n_field, sx, sy, sz, a_local, a_dag_local, gauge_x, gauge_z, n_parity = \
            self.build_operators()

        # Identity with correct tensor dims
        id_sites = [qeye(self.site_dim) for _ in range(self.n_sites)]
        id_links = [qeye(self.link_dim) for _ in range(self.n_links)]
        I_full = tensor(id_sites + id_links)

        # Initialize H with correct dims
        H = 0 * I_full

        # Hopping (matter-gauge coupling)
        print("[BUILD] - Hopping (gauge-covariant)")
        for i in range(self.Nx):
            for j in range(self.Ny):
                site = self.site_index(i, j)

                # Right neighbor (horizontal)
                if j + 1 < self.Ny:
                    site_right = self.site_index(i, j + 1)
                    link = self.link_index_horizontal(i, j)
                    hop_term = self.gauge_covariant_hop(
                        a_dag=a_local, site1=site,
                        a=a_local, site2=site_right,
                        link=link
                    )
                    H -= J_hop * (hop_term + hop_term.dag())

                # Down neighbor (vertical)
                if i + 1 < self.Nx:
                    site_down = self.site_index(i + 1, j)
                    link = self.link_index_vertical(i, j)
                    hop_term = self.gauge_covariant_hop(
                        a_dag=a_local, site1=site,
                        a=a_local, site2=site_down,
                        link=link
                    )
                    H -= J_hop * (hop_term + hop_term.dag())

        # Spin interactions (simple sz-sz)
        print("[BUILD] - Spin interactions")
        for i in range(self.Nx):
            for j in range(self.Ny):
                site = self.site_index(i, j)

                # Right neighbor
                if j + 1 < self.Ny:
                    site_right = self.site_index(i, j + 1)
                    ops_left = []
                    ops_right = []
                    for s in range(self.n_sites):
                        if s == site:
                            ops_left.append(sz)
                            ops_right.append(qeye(self.site_dim))
                        elif s == site_right:
                            ops_left.append(qeye(self.site_dim))
                            ops_right.append(sz)
                        else:
                            ops_left.append(qeye(self.site_dim))
                            ops_right.append(qeye(self.site_dim))
                    # gauge identity
                    for _ in range(self.n_links):
                        ops_left.append(qeye(self.link_dim))
                        ops_right.append(qeye(self.link_dim))

                    H += J_spin * tensor(ops_left) * tensor(ops_right)

                # Down neighbor
                if i + 1 < self.Nx:
                    site_down = self.site_index(i + 1, j)
                    ops_left = []
                    ops_right = []
                    for s in range(self.n_sites):
                        if s == site:
                            ops_left.append(sz)
                            ops_right.append(qeye(self.site_dim))
                        elif s == site_down:
                            ops_left.append(qeye(self.site_dim))
                            ops_right.append(sz)
                        else:
                            ops_left.append(qeye(self.site_dim))
                            ops_right.append(qeye(self.site_dim))
                    for _ in range(self.n_links):
                        ops_left.append(qeye(self.link_dim))
                        ops_right.append(qeye(self.link_dim))

                    H += J_spin * tensor(ops_left) * tensor(ops_right)

        # Mass + onsite repulsion
        print("[BUILD] - Mass + onsite repulsion")
        for site in range(self.n_sites):
            n_site = self.matter_op(n_field, site)
            H += mass * n_site
            H += U_onsite * n_site * n_site  # stiffness for occupation

        # Gauge kinetic term
        print(f"[BUILD] - Gauge kinetic (g_gauge = {g_gauge})")
        for link in range(self.n_links):
            H -= g_gauge * self.gauge_op(gauge_x, link)

        # Gauss-law penalty
        if lambda_G > 0.0:
            print(f"[BUILD] - Gauss-law penalty (lambda_G = {lambda_G})")
            for s in range(self.n_sites):
                Gs = self.gauss_operator(s)
                # (I - Gs)^2 = 2(I - Gs) since Gs^2 = I, but explicit is fine
                H += lambda_G * (I_full - Gs) * (I_full - Gs)

        self.H = H
        print(f"[BUILD] Done. H dim = {H.shape}")
        return H

    # -----------------------
    # State prep
    # -----------------------
    def init_skyrmion_state(self, x0, y0, amplitude=1.0, mult=1.0):
        """
        Build a skyrmion-like matter pattern (single or "double" amplitude)
        and tensor with a gauge configuration where each link is in |+>
        (eigenstate of σ_x with eigenvalue +1). That way, Gauss-law
        penalty depends on matter parity instead of being a constant.
        """
        local_matter_states = []

        for i in range(self.Nx):
            for j in range(self.Ny):
                dx = i - x0
                dy = j - y0
                r = np.sqrt(dx**2 + dy**2)
                theta = np.arctan2(dy, dx)

                phase_spatial = np.exp(1j * theta)
                amp = amplitude * mult * np.exp(-r**2 / 2.0)

                spin_angle = theta
                spin_up = np.cos(spin_angle / 2)
                spin_down = np.exp(1j * spin_angle) * np.sin(spin_angle / 2)

                c_1up = amp * phase_spatial * spin_up
                c_1down = amp * phase_spatial * spin_down

                # Local 4-level spinor: (vac_up, vac_down, 1_up, 1_down)
                psi_site = Qobj([[1.0],
                                 [0.0],
                                 [c_1up],
                                 [c_1down]]).unit()
                local_matter_states.append(psi_site)

        # Gauge initialized to |+> on each link: eigenstate of σ_x with eigenvalue +1
        plus = (basis(self.link_dim, 0) + basis(self.link_dim, 1)).unit()
        gauge_plus = [plus for _ in range(self.n_links)]

        full_state = tensor(local_matter_states + gauge_plus)
        self.state = full_state
        return full_state

    # -----------------------
    # Measurements
    # -----------------------
    def measure_energy(self, state):
        E = expect(self.H, state)
        return float(np.real(E))

    def measure_total_occupation(self, state):
        n_field, sx, sy, sz, a_local, a_dag_local, gauge_x, gauge_z, n_parity = \
            self.build_operators()
        N_total = 0.0
        for site in range(self.n_sites):
            N_total += expect(self.matter_op(n_field, site), state)
        return float(np.real(N_total))

    def measure_gauss_violation(self, state):
        """
        Compute <(I - G_s)^2> for each site s and return a list.
        This is the local Gauss-law violation measure that the
        penalty term is using.
        """
        n_field, sx, sy, sz, a_local, a_dag_local, gauge_x, gauge_z, n_parity = \
            self.build_operators()

        # Identity with correct dims
        id_sites = [qeye(self.site_dim) for _ in range(self.n_sites)]
        id_links = [qeye(self.link_dim) for _ in range(self.n_links)]
        I_full = tensor(id_sites + id_links)

        violations = []
        for s in range(self.n_sites):
            Gs = self.gauss_operator(s)
            op = (I_full - Gs) * (I_full - Gs)
            violations.append(float(np.real(expect(op, state))))
        return violations


# ---------------------------
# Exclusion test
# ---------------------------
def run_exclusion_test(Nx=2, Ny=2, U_onsite=1.0, g_gauge=0.5, lambda_G=0.0):
    print("\n" + "="*70)
    print(f"GAUGE EXCLUSION TEST (lambda_G = {lambda_G})")
    print("="*70)

    lat = GaugeLatticeGauss(Nx, Ny)
    lat.build_hamiltonian(J_hop=1.0, J_spin=0.5,
                          mass=0.1, U_onsite=U_onsite,
                          g_gauge=g_gauge, lambda_G=lambda_G)

    # One skyrmion centered in the middle
    print("\n[CONFIG] One skyrmion")
    lat.init_skyrmion_state(x0=Nx/2, y0=Ny/2, amplitude=1.0, mult=1.0)
    E1 = lat.measure_energy(lat.state)
    N1 = lat.measure_total_occupation(lat.state)
    viol1 = lat.measure_gauss_violation(lat.state)
    print(f"  E1 = {E1:.6f}, N1 = {N1:.6f}, E1/N1 = {E1/N1:.6f}")
    print(f"  Gauss violation per site: {viol1}")

    # Two skyrmions OVERLAPPING: just double amplitude at same location
    print("\n[CONFIG] Two skyrmions OVERLAPPING")
    lat.init_skyrmion_state(x0=Nx/2, y0=Ny/2, amplitude=1.0, mult=2.0)
    E2_overlap = lat.measure_energy(lat.state)
    N2_overlap = lat.measure_total_occupation(lat.state)
    viol_overlap = lat.measure_gauss_violation(lat.state)
    print(f"  E2_overlap = {E2_overlap:.6f}, N2_overlap = {N2_overlap:.6f}, "
          f"E2_overlap/N2_overlap = {E2_overlap/N2_overlap:.6f}")
    print(f"  Gauss violation per site: {viol_overlap}")

    # Two skyrmions SEPARATED: superposition of left- and right-centered skyrmions
    print("\n[CONFIG] Two skyrmions SEPARATED (superposition)")
    lat_left = GaugeLatticeGauss(Nx, Ny)
    lat_left.build_hamiltonian(J_hop=1.0, J_spin=0.5,
                               mass=0.1, U_onsite=U_onsite,
                               g_gauge=g_gauge, lambda_G=lambda_G)
    state_left = lat_left.init_skyrmion_state(x0=0.5, y0=Ny/2, amplitude=0.8, mult=1.0)

    lat_right = GaugeLatticeGauss(Nx, Ny)
    lat_right.build_hamiltonian(J_hop=1.0, J_spin=0.5,
                                mass=0.1, U_onsite=U_onsite,
                                g_gauge=g_gauge, lambda_G=lambda_G)
    state_right = lat_right.init_skyrmion_state(x0=Nx-0.5, y0=Ny/2, amplitude=0.8, mult=1.0)

    # Superposition |ψ> ∝ |left> + |right>
    state_sep = (state_left + state_right).unit()
    E2_sep = lat.measure_energy(state_sep)
    N2_sep = lat.measure_total_occupation(state_sep)
    viol_sep = lat.measure_gauss_violation(state_sep)
    print(f"  E2_sep = {E2_sep:.6f}, N2_sep = {N2_sep:.6f}, "
          f"E2_sep/N2_sep = {E2_sep/N2_sep:.6f}")
    print(f"  Gauss violation per site: {viol_sep}")

    # Exclusion diagnostics
    print("\n[ANALYSIS]")
    E_overlap_excess = E2_overlap - 2*E1
    E_sep_excess = E2_sep - 2*E1
    energy_cost = E2_overlap - E2_sep

    print(f"  Overlap excess vs 2x single:  ΔE_overlap = {E_overlap_excess:.6f}")
    print(f"  Separated excess vs 2x single: ΔE_sep     = {E_sep_excess:.6f}")
    print(f"  Direct cost (overlap - sep):   ΔE         = {energy_cost:.6f}")

    if energy_cost > 1.0:
        verdict = "FERMIONIC-LIKE (strong exclusion)"
    elif energy_cost < -1.0:
        verdict = "BOSONIC-LIKE (overlap favored)"
    else:
        verdict = "UNCLEAR / WEAK EFFECT"

    print(f"\n[VERDICT] {verdict}")
    return {
        "lambda_G": lambda_G,
        "E1": E1,
        "N1": N1,
        "E2_overlap": E2_overlap,
        "N2_overlap": N2_overlap,
        "E2_sep": E2_sep,
        "N2_sep": N2_sep,
        "E_overlap_excess": E_overlap_excess,
        "E_sep_excess": E_sep_excess,
        "energy_cost": energy_cost,
        "verdict": verdict,
        "gauss_one": viol1,
        "gauss_overlap": viol_overlap,
        "gauss_sep": viol_sep,
    }


def main():
    print("="*70)
    print("GAUGE + GAUSS-LAW SKYRMION EXCLUSION TEST")
    print("="*70)

    # Soft case: no Gauss enforcement
    res_soft = run_exclusion_test(lambda_G=0.0)

    # Harder case: strong Gauss-law penalty
    res_hard = run_exclusion_test(lambda_G=5.0)

    print("\n" + "="*70)
    print("COMPARISON (SOFT vs HARD GAUGE CONSTRAINT)")
    print("="*70)
    print(f"Soft  (λ_G=0.0): ΔE = {res_soft['energy_cost']:.6f}, verdict = {res_soft['verdict']}")
    print(f"Hard  (λ_G=5.0): ΔE = {res_hard['energy_cost']:.6f}, verdict = {res_hard['verdict']}")
    print(f"Soft  Gauss (one/overlap/sep): {res_soft['gauss_one']}, "
          f"{res_soft['gauss_overlap']}, {res_soft['gauss_sep']}")
    print(f"Hard  Gauss (one/overlap/sep): {res_hard['gauss_one']}, "
          f"{res_hard['gauss_overlap']}, {res_hard['gauss_sep']}")


if __name__ == "__main__":
    main()

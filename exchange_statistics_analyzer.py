#!/usr/bin/env python3
"""
exchange_statistics_analyzer.py

Test the composite fermion hypothesis by analyzing:
1. Phase singularity tracking over time
2. Exchange/braiding events between singularities
3. Berry phase accumulation during exchanges
4. Gauge flux per singularity
5. Correlation between N_singularities (odd/even) and spin statistics

Author: Ben (with Claude assistance)
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os
from tqdm import tqdm

# ======================================================================
# PHASE SINGULARITY TRACKING
# ======================================================================

def find_phase_singularities(psi, threshold=0.05):
    """
    Find phase singularities (vortex cores) in complex field.
    
    A phase singularity occurs where |psi| ≈ 0 and phase is undefined.
    
    Parameters:
    -----------
    psi : 2D complex array
        Complex scalar field
    threshold : float
        Density threshold (fraction of mean) for singularity detection
        
    Returns:
    --------
    singularities : Nx2 array
        Positions of singularities [[x1,y1], [x2,y2], ...]
    """
    from scipy.ndimage import minimum_filter
    
    rho = np.abs(psi)**2
    mean_rho = np.mean(rho)
    
    # Local minima in density
    local_min = minimum_filter(rho, size=3) == rho
    
    # Must be very low density
    singularities_mask = local_min & (rho < threshold * mean_rho)
    
    # Get coordinates
    singularities = np.argwhere(singularities_mask)
    
    return singularities


def track_singularities_over_time(snapshots, dx=1.0, max_distance=10.0):
    """
    Track individual phase singularities across multiple time snapshots.
    
    Uses Hungarian algorithm to match singularities between frames based on
    minimum total displacement.
    
    Parameters:
    -----------
    snapshots : list of 2D complex arrays
        Time series of psi field
    dx : float
        Grid spacing
    max_distance : float
        Maximum distance a singularity can move between frames
        
    Returns:
    --------
    trajectories : dict
        {singularity_id: [(t, x, y), ...]}
    events : list
        Detected topological events (creation, annihilation, braiding)
    """
    print("\n" + "="*70)
    print("TRACKING PHASE SINGULARITIES OVER TIME")
    print("="*70)
    
    # Find singularities at each timestep
    all_singularities = []
    for i, psi in enumerate(tqdm(snapshots, desc="Finding singularities")):
        sings = find_phase_singularities(psi)
        all_singularities.append(sings)
        print(f"  Frame {i}: {len(sings)} singularities")
    
    # Track singularities using Hungarian algorithm
    trajectories = {}
    next_id = 0
    events = []
    
    # Initialize with first frame
    for pos in all_singularities[0]:
        trajectories[next_id] = [(0, pos[0], pos[1])]
        next_id += 1
    
    # Match singularities between consecutive frames
    for t in tqdm(range(1, len(all_singularities)), desc="Tracking"):
        prev_sings = all_singularities[t-1]
        curr_sings = all_singularities[t]
        
        if len(prev_sings) == 0 or len(curr_sings) == 0:
            continue
        
        # Compute distance matrix
        dist_matrix = cdist(prev_sings, curr_sings)
        
        # Apply maximum distance constraint
        dist_matrix[dist_matrix > max_distance] = 1e10
        
        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        
        # Track which current singularities are matched
        matched_curr = set()
        
        # Update existing trajectories
        active_ids = [tid for tid, traj in trajectories.items() 
                      if traj[-1][0] == t-1]
        
        for i, (prev_idx, curr_idx) in enumerate(zip(row_ind, col_ind)):
            if dist_matrix[prev_idx, curr_idx] < max_distance:
                # Find which trajectory this corresponds to
                tid = active_ids[prev_idx]
                pos = curr_sings[curr_idx]
                trajectories[tid].append((t, pos[0], pos[1]))
                matched_curr.add(curr_idx)
        
        # Handle creation events (unmatched current singularities)
        for curr_idx in range(len(curr_sings)):
            if curr_idx not in matched_curr:
                pos = curr_sings[curr_idx]
                trajectories[next_id] = [(t, pos[0], pos[1])]
                events.append({
                    'type': 'creation',
                    'time': t,
                    'position': pos,
                    'id': next_id
                })
                next_id += 1
        
        # Handle annihilation events (unmatched previous singularities)
        for prev_idx in range(len(prev_sings)):
            if prev_idx not in row_ind:
                tid = active_ids[prev_idx]
                events.append({
                    'type': 'annihilation',
                    'time': t,
                    'position': prev_sings[prev_idx],
                    'id': tid
                })
    
    print(f"\n✓ Tracked {len(trajectories)} singularities")
    print(f"✓ Detected {len(events)} topological events")
    
    return trajectories, events


# ======================================================================
# BRAIDING DETECTION
# ======================================================================

def detect_braiding_events(trajectories, min_duration=5):
    """
    Detect when two singularities exchange positions (braid around each other).
    
    A braiding event occurs when:
    1. Two singularities come close
    2. They circle around each other
    3. They separate having exchanged relative positions
    
    Parameters:
    -----------
    trajectories : dict
        Output from track_singularities_over_time
    min_duration : int
        Minimum number of frames for valid braid
        
    Returns:
    --------
    braids : list of dict
        Each braid contains: {id1, id2, t_start, t_end, winding_angle}
    """
    print("\n" + "="*70)
    print("DETECTING BRAIDING EVENTS")
    print("="*70)
    
    braids = []
    
    # Get all pairs of long-lived singularities
    long_traj = {tid: traj for tid, traj in trajectories.items() 
                 if len(traj) >= min_duration * 2}
    
    ids = list(long_traj.keys())
    
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            id1, id2 = ids[i], ids[j]
            traj1 = long_traj[id1]
            traj2 = long_traj[id2]
            
            # Find overlapping time window
            t1_set = set(t for t, x, y in traj1)
            t2_set = set(t for t, x, y in traj2)
            overlap = sorted(t1_set & t2_set)
            
            if len(overlap) < min_duration * 2:
                continue
            
            # Extract positions during overlap
            pos1 = np.array([[x, y] for t, x, y in traj1 if t in overlap])
            pos2 = np.array([[x, y] for t, x, y in traj2 if t in overlap])
            
            # Compute relative position vector
            rel_pos = pos2 - pos1
            
            # Compute relative angle over time
            angles = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])
            
            # Unwrap angles
            angles_unwrapped = np.unwrap(angles)
            
            # Total winding
            total_winding = angles_unwrapped[-1] - angles_unwrapped[0]
            
            # Significant braiding if |winding| > π/2
            if abs(total_winding) > np.pi/2:
                braids.append({
                    'id1': id1,
                    'id2': id2,
                    't_start': overlap[0],
                    't_end': overlap[-1],
                    'winding_angle': total_winding,
                    'n_winds': total_winding / (2*np.pi),
                })
    
    print(f"\n✓ Found {len(braids)} braiding events")
    
    # Print summary
    if braids:
        print("\nSignificant braids:")
        for braid in braids[:10]:  # Show first 10
            print(f"  IDs {braid['id1']}-{braid['id2']}: "
                  f"θ = {braid['winding_angle']:.3f} rad "
                  f"({braid['n_winds']:.3f} winds) "
                  f"over t={braid['t_start']}-{braid['t_end']}")
    
    return braids


# ======================================================================
# EXCHANGE STATISTICS MEASUREMENT
# ======================================================================

def compute_exchange_phase(psi_initial, psi_final, region_mask):
    """
    Compute Berry phase accumulated during adiabatic exchange.
    
    For two singularities exchanging positions, the wavefunction picks up
    a geometric phase:
    
    θ = arg(⟨ψ_initial|ψ_final⟩) in the exchange region
    
    If θ = π → fermionic statistics
    If θ = 2π → bosonic statistics
    
    Parameters:
    -----------
    psi_initial : 2D complex array
        Wavefunction before exchange
    psi_final : 2D complex array
        Wavefunction after exchange
    region_mask : 2D bool array
        Region where exchange occurred
        
    Returns:
    --------
    theta : float
        Exchange phase in radians
    """
    # Overlap integral in exchange region
    overlap = np.sum(np.conj(psi_initial[region_mask]) * psi_final[region_mask])
    
    # Berry phase
    theta = np.angle(overlap)
    
    return theta


def analyze_exchange_statistics(snapshots, braids, dx=1.0):
    """
    Compute exchange phase for all detected braiding events.
    
    Parameters:
    -----------
    snapshots : list of 2D complex arrays
        Time series
    braids : list
        Output from detect_braiding_events
    dx : float
        Grid spacing
        
    Returns:
    --------
    statistics : dict
        Statistical analysis of exchange phases
    """
    print("\n" + "="*70)
    print("ANALYZING EXCHANGE STATISTICS")
    print("="*70)
    
    exchange_phases = []
    
    for braid in tqdm(braids, desc="Computing exchange phases"):
        t_start = braid['t_start']
        t_end = braid['t_end']
        
        if t_end >= len(snapshots):
            continue
        
        psi_initial = snapshots[t_start]
        psi_final = snapshots[t_end]
        
        # Create region mask around the two singularities
        # (simplified: use entire field for now)
        region_mask = np.ones(psi_initial.shape, dtype=bool)
        
        theta = compute_exchange_phase(psi_initial, psi_final, region_mask)
        
        exchange_phases.append({
            'braid': braid,
            'theta': theta,
            'theta_normalized': theta % (2*np.pi),
        })
    
    if not exchange_phases:
        print("\n⚠ No exchange events to analyze")
        return {'n_events': 0}
    
    # Statistical analysis
    thetas = [e['theta'] for e in exchange_phases]
    thetas_norm = [e['theta_normalized'] for e in exchange_phases]
    
    statistics = {
        'n_events': len(exchange_phases),
        'mean_theta': np.mean(thetas),
        'std_theta': np.std(thetas),
        'mean_theta_normalized': np.mean(thetas_norm),
        'events': exchange_phases,
    }
    
    # Check for fermionic vs bosonic signature
    near_pi = sum(1 for t in thetas_norm if 0.8*np.pi < t < 1.2*np.pi)
    near_2pi = sum(1 for t in thetas_norm if 1.8*np.pi < t < 2.2*np.pi)
    near_0 = sum(1 for t in thetas_norm if t < 0.2*np.pi or t > 1.8*np.pi)
    
    print(f"\nExchange phase distribution:")
    print(f"  Near 0/2π (bosonic):   {near_0 + near_2pi} events")
    print(f"  Near π (fermionic):    {near_pi} events")
    print(f"  Other:                 {len(exchange_phases) - near_pi - near_0 - near_2pi} events")
    
    if near_pi > near_0 + near_2pi:
        print("\n✓ FERMIONIC SIGNATURE DETECTED!")
    elif near_0 + near_2pi > near_pi:
        print("\n✓ BOSONIC SIGNATURE DETECTED")
    else:
        print("\n⚠ Mixed or inconclusive statistics")
    
    return statistics


# ======================================================================
# FLUX QUANTIZATION ANALYSIS
# ======================================================================

def compute_flux_per_singularity(psi, ax, ay, singularities, dx=1.0, radius=3):
    """
    Compute gauge flux through loops around each singularity.
    
    Flux = ∮ A·dl = ∫∫ B·dA
    
    For U(1) gauge theory:
    - Integer flux → bosonic
    - Half-integer flux → fermionic
    
    Parameters:
    -----------
    psi : 2D complex array
        Scalar field
    ax, ay : 2D arrays
        Gauge potential components
    singularities : Nx2 array
        Singularity positions
    dx : float
        Grid spacing
    radius : float
        Integration radius around singularity
        
    Returns:
    --------
    fluxes : array
        Flux through each singularity
    """
    fluxes = []
    
    for sing_pos in singularities:
        cx, cy = int(sing_pos[0]), int(sing_pos[1])
        
        # Compute flux through circle around singularity
        # Flux = ∮ A·dl
        n_points = 50
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        
        flux = 0
        for i in range(n_points):
            angle = angles[i]
            x = int(cx + radius * np.cos(angle)) % psi.shape[0]
            y = int(cy + radius * np.sin(angle)) % psi.shape[1]
            
            # Tangent vector
            tx = -np.sin(angle)
            ty = np.cos(angle)
            
            # A·dl
            flux += (ax[x, y] * tx + ay[x, y] * ty) * radius * (2*np.pi / n_points)
        
        fluxes.append(flux)
    
    return np.array(fluxes)


def analyze_flux_quantization(snapshots, ax_snapshots, ay_snapshots, dx=1.0):
    """
    Analyze flux quantization pattern across all singularities.
    
    Parameters:
    -----------
    snapshots : list of 2D complex arrays
        Psi field over time
    ax_snapshots, ay_snapshots : lists of 2D arrays
        Gauge potential over time
    dx : float
        Grid spacing
        
    Returns:
    --------
    flux_stats : dict
        Statistical analysis of flux distribution
    """
    print("\n" + "="*70)
    print("ANALYZING FLUX QUANTIZATION")
    print("="*70)
    
    all_fluxes = []
    
    for i, psi in enumerate(tqdm(snapshots, desc="Computing fluxes")):
        singularities = find_phase_singularities(psi)
        
        if len(singularities) == 0:
            continue
        
        ax = ax_snapshots[i]
        ay = ay_snapshots[i]
        
        fluxes = compute_flux_per_singularity(psi, ax, ay, singularities, dx)
        all_fluxes.extend(fluxes)
    
    if len(all_fluxes) == 0:
        print("⚠ No singularities found")
        return {'n_singularities': 0}
    
    all_fluxes = np.array(all_fluxes)
    
    # Normalize by flux quantum (2π for U(1))
    flux_quantum = 2 * np.pi
    normalized_fluxes = all_fluxes / flux_quantum
    
    # Statistics
    flux_stats = {
        'n_singularities': len(all_fluxes),
        'mean_flux': np.mean(all_fluxes),
        'std_flux': np.std(all_fluxes),
        'mean_normalized': np.mean(normalized_fluxes),
        'fluxes': all_fluxes,
        'normalized_fluxes': normalized_fluxes,
    }
    
    # Check quantization
    near_integer = np.sum(np.abs(normalized_fluxes - np.round(normalized_fluxes)) < 0.1)
    near_half = np.sum(np.abs(normalized_fluxes - (np.round(normalized_fluxes*2)/2)) < 0.1)
    
    print(f"\nFlux quantization:")
    print(f"  Near integer flux quantum:      {near_integer} ({100*near_integer/len(all_fluxes):.1f}%)")
    print(f"  Near half-integer flux quantum: {near_half - near_integer} ({100*(near_half-near_integer)/len(all_fluxes):.1f}%)")
    print(f"  Mean flux: {flux_stats['mean_normalized']:.3f} × Φ₀")
    
    return flux_stats


# ======================================================================
# ODD/EVEN N CORRELATION
# ======================================================================

def analyze_parity_correlation(snapshots, dx=1.0):
    """
    Test if lumps with ODD number of singularities behave differently
    from those with EVEN number (composite fermion hypothesis).
    
    Parameters:
    -----------
    snapshots : list of 2D complex arrays
        Time series
    dx : float
        Grid spacing
        
    Returns:
    --------
    parity_stats : dict
        Comparison of odd vs even N lumps
    """
    print("\n" + "="*70)
    print("ANALYZING ODD/EVEN SINGULARITY PARITY")
    print("="*70)
    
    from vorticity_analyzer import find_lump_centers, compute_winding_number
    
    odd_lumps = []
    even_lumps = []
    
    for psi in tqdm(snapshots, desc="Analyzing lumps"):
        rho = np.abs(psi)**2
        centers = find_lump_centers(rho, threshold_factor=2.5, min_separation=10)
        
        for center in centers:
            # Extract region around lump
            radius = 15
            cx, cy = center
            L = psi.shape[0]
            
            region = np.zeros((2*radius, 2*radius), dtype=psi.dtype)
            for i in range(2*radius):
                for j in range(2*radius):
                    x = (cx - radius + i) % L
                    y = (cy - radius + j) % L
                    region[i, j] = psi[x, y]
            
            # Count singularities in region
            singularities = find_phase_singularities(region)
            N = len(singularities)
            
            # Compute total winding number
            w = compute_winding_number(psi, center, radius=radius*0.7)
            
            lump_data = {
                'center': center,
                'N_singularities': N,
                'winding': w,
                'density': np.mean(rho[max(0,cx-5):cx+5, max(0,cy-5):cy+5]),
            }
            
            if N % 2 == 1:
                odd_lumps.append(lump_data)
            else:
                even_lumps.append(lump_data)
    
    print(f"\nFound:")
    print(f"  {len(odd_lumps)} lumps with ODD N_singularities")
    print(f"  {len(even_lumps)} lumps with EVEN N_singularities")
    
    if len(odd_lumps) > 0 and len(even_lumps) > 0:
        # Compare winding numbers
        odd_w = [l['winding'] for l in odd_lumps]
        even_w = [l['winding'] for l in even_lumps]
        
        print(f"\nWinding number statistics:")
        print(f"  ODD N:  mean |w| = {np.mean(np.abs(odd_w)):.3f} ± {np.std(np.abs(odd_w)):.3f}")
        print(f"  EVEN N: mean |w| = {np.mean(np.abs(even_w)):.3f} ± {np.std(np.abs(even_w)):.3f}")
        
        # Check for half-integer vs integer preference
        odd_half_int = sum(1 for w in odd_w if 0.4 < abs(w) < 0.6)
        even_half_int = sum(1 for w in even_w if 0.4 < abs(w) < 0.6)
        odd_int = sum(1 for w in odd_w if 0.9 < abs(w) < 1.1)
        even_int = sum(1 for w in even_w if 0.9 < abs(w) < 1.1)
        
        print(f"\nSpin character:")
        print(f"  ODD N:  {odd_half_int}/{len(odd_lumps)} half-integer, {odd_int}/{len(odd_lumps)} integer")
        print(f"  EVEN N: {even_half_int}/{len(even_lumps)} half-integer, {even_int}/{len(even_lumps)} integer")
        
        if odd_half_int > even_half_int * 1.5:
            print("\n✓ ODD N LUMPS PREFER HALF-INTEGER WINDING (FERMION-LIKE)!")
        elif even_int > odd_int * 1.5:
            print("\n✓ EVEN N LUMPS PREFER INTEGER WINDING (BOSON-LIKE)!")
    
    return {
        'odd_lumps': odd_lumps,
        'even_lumps': even_lumps,
    }


# ======================================================================
# MAIN ANALYSIS PIPELINE
# ======================================================================

def run_complete_analysis(data_dir, output_dir="exchange_analysis"):
    """
    Run complete exchange statistics analysis pipeline.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing simulation snapshots (*.npz files)
    output_dir : str
        Directory to save results
    """
    import glob
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("EXCHANGE STATISTICS & FLUX ANALYSIS")
    print("="*70)
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load all snapshots
    snapshot_files = sorted(glob.glob(os.path.join(data_dir, "*_snap_*.npz")))
    
    if not snapshot_files:
        print(f"\nERROR: No snapshot files found in {data_dir}")
        print("Looking for files matching pattern: *_snap_*.npz")
        return
    
    print(f"\nLoading {len(snapshot_files)} snapshots...")
    
    snapshots = []
    ax_snapshots = []
    ay_snapshots = []
    times = []
    
    for f in tqdm(snapshot_files, desc="Loading"):
        data = np.load(f)
        snapshots.append(data['psi'])
        ax_snapshots.append(data['ax'])
        ay_snapshots.append(data['ay'])
        times.append(data['time'])
    
    print(f"✓ Loaded {len(snapshots)} frames")
    print(f"  Time range: {times[0]:.2f} - {times[-1]:.2f}")
    print(f"  Grid size: {snapshots[0].shape}")
    
    # Run analyses
    results = {}
    
    # 1. Track singularities
    trajectories, events = track_singularities_over_time(snapshots)
    results['trajectories'] = trajectories
    results['events'] = events
    
    # 2. Detect braiding
    braids = detect_braiding_events(trajectories)
    results['braids'] = braids
    
    # 3. Exchange statistics
    exchange_stats = analyze_exchange_statistics(snapshots, braids)
    results['exchange_statistics'] = exchange_stats
    
    # 4. Flux quantization
    flux_stats = analyze_flux_quantization(snapshots, ax_snapshots, ay_snapshots)
    results['flux_statistics'] = flux_stats
    
    # 5. Odd/even parity correlation
    parity_stats = analyze_parity_correlation(snapshots)
    results['parity_statistics'] = parity_stats
    
    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    # Save numerical results
    np.savez(
        os.path.join(output_dir, "exchange_analysis_results.npz"),
        trajectories=trajectories,
        events=events,
        braids=braids,
        exchange_statistics=exchange_stats,
        flux_statistics=flux_stats,
        parity_statistics=parity_stats,
    )
    print(f"✓ Saved: exchange_analysis_results.npz")
    
    # Generate plots
    plot_results(results, output_dir)
    
    # Summary report
    generate_summary_report(results, output_dir)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\nResults saved in: {output_dir}/")
    
    return results


def plot_results(results, output_dir):
    """Generate comprehensive visualization of results."""
    
    print("\nGenerating plots...")
    
    # Plot 1: Singularity trajectories
    fig, ax = plt.subplots(figsize=(12, 12))
    trajectories = results['trajectories']
    
    for tid, traj in list(trajectories.items())[:100]:  # Plot first 100
        if len(traj) < 5:
            continue
        positions = np.array([[x, y] for t, x, y in traj])
        ax.plot(positions[:, 0], positions[:, 1], alpha=0.5, linewidth=0.5)
        ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=3)
        ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=3)
    
    ax.set_title('Phase Singularity Trajectories', fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'singularity_trajectories.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ singularity_trajectories.png")
    
    # Plot 2: Exchange phase histogram
    if results['exchange_statistics']['n_events'] > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        thetas = [e['theta_normalized'] for e in results['exchange_statistics']['events']]
        ax.hist(thetas, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.pi, color='red', linestyle='--', linewidth=2, label='π (fermion)')
        ax.axvline(2*np.pi, color='blue', linestyle='--', linewidth=2, label='2π (boson)')
        ax.set_xlabel('Exchange Phase θ (radians)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Exchange Phase Distribution', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'exchange_phase_histogram.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ exchange_phase_histogram.png")
    
    # Plot 3: Flux quantization
    if results['flux_statistics']['n_singularities'] > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        fluxes = results['flux_statistics']['normalized_fluxes']
        ax.hist(fluxes, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='±1/2 Φ₀')
        ax.axvline(1.0, color='blue', linestyle='--', linewidth=2, label='±1 Φ₀')
        ax.set_xlabel('Flux (units of Φ₀)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Flux Quantization per Singularity', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'flux_quantization.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ flux_quantization.png")
    
    # Plot 4: Odd vs Even N comparison
    parity = results['parity_statistics']
    if len(parity['odd_lumps']) > 0 and len(parity['even_lumps']) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Winding numbers
        ax = axes[0]
        odd_w = [abs(l['winding']) for l in parity['odd_lumps']]
        even_w = [abs(l['winding']) for l in parity['even_lumps']]
        ax.hist([odd_w, even_w], bins=20, alpha=0.7, label=['ODD N', 'EVEN N'], edgecolor='black')
        ax.axvline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.axvline(1.0, color='blue', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_xlabel('|Winding Number|', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Winding Number: ODD vs EVEN N', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # N singularities distribution
        ax = axes[1]
        odd_N = [l['N_singularities'] for l in parity['odd_lumps']]
        even_N = [l['N_singularities'] for l in parity['even_lumps']]
        ax.hist([odd_N, even_N], bins=20, alpha=0.7, label=['ODD N', 'EVEN N'], edgecolor='black')
        ax.set_xlabel('Number of Singularities', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('N Distribution', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'odd_even_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ odd_even_comparison.png")


def generate_summary_report(results, output_dir):
    """Generate text summary report."""
    
    report_path = os.path.join(output_dir, "ANALYSIS_SUMMARY.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("EXCHANGE STATISTICS & COMPOSITE FERMION ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        # Singularity tracking
        f.write("PHASE SINGULARITY TRACKING:\n")
        f.write(f"  Total singularities tracked: {len(results['trajectories'])}\n")
        f.write(f"  Topological events detected: {len(results['events'])}\n")
        
        # Count event types
        events = results['events']
        n_creation = sum(1 for e in events if e['type'] == 'creation')
        n_annihilation = sum(1 for e in events if e['type'] == 'annihilation')
        f.write(f"    - Creation events: {n_creation}\n")
        f.write(f"    - Annihilation events: {n_annihilation}\n\n")
        
        # Braiding
        f.write("BRAIDING EVENTS:\n")
        f.write(f"  Detected braids: {len(results['braids'])}\n")
        if results['braids']:
            winds = [b['n_winds'] for b in results['braids']]
            f.write(f"  Mean winding: {np.mean(winds):.3f} ± {np.std(winds):.3f} turns\n\n")
        
        # Exchange statistics
        f.write("EXCHANGE STATISTICS:\n")
        exch = results['exchange_statistics']
        f.write(f"  Exchange events analyzed: {exch['n_events']}\n")
        if exch['n_events'] > 0:
            f.write(f"  Mean exchange phase: {exch['mean_theta_normalized']:.3f} rad\n")
            
            thetas = [e['theta_normalized'] for e in exch['events']]
            near_pi = sum(1 for t in thetas if 0.8*np.pi < t < 1.2*np.pi)
            near_2pi = sum(1 for t in thetas if 1.8*np.pi < t or t < 0.2*np.pi)
            
            f.write(f"  Near π (fermionic): {near_pi}/{len(thetas)} = {100*near_pi/len(thetas):.1f}%\n")
            f.write(f"  Near 2π (bosonic): {near_2pi}/{len(thetas)} = {100*near_2pi/len(thetas):.1f}%\n")
            
            if near_pi > near_2pi:
                f.write("\n  ✓ FERMIONIC SIGNATURE DOMINANT\n\n")
            else:
                f.write("\n  ✓ BOSONIC SIGNATURE DOMINANT\n\n")
        
        # Flux
        f.write("FLUX QUANTIZATION:\n")
        flux = results['flux_statistics']
        f.write(f"  Singularities analyzed: {flux['n_singularities']}\n")
        if flux['n_singularities'] > 0:
            f.write(f"  Mean flux: {flux['mean_normalized']:.3f} Φ₀\n\n")
        
        # Parity
        f.write("ODD/EVEN N CORRELATION:\n")
        parity = results['parity_statistics']
        f.write(f"  ODD N lumps: {len(parity['odd_lumps'])}\n")
        f.write(f"  EVEN N lumps: {len(parity['even_lumps'])}\n")
        
        if parity['odd_lumps'] and parity['even_lumps']:
            odd_w = [abs(l['winding']) for l in parity['odd_lumps']]
            even_w = [abs(l['winding']) for l in parity['even_lumps']]
            f.write(f"  ODD N: mean |w| = {np.mean(odd_w):.3f}\n")
            f.write(f"  EVEN N: mean |w| = {np.mean(even_w):.3f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("CONCLUSION:\n")
        f.write("="*70 + "\n\n")
        
        # Determine overall conclusion
        fermion_evidence = 0
        boson_evidence = 0
        
        if exch['n_events'] > 0:
            thetas = [e['theta_normalized'] for e in exch['events']]
            near_pi = sum(1 for t in thetas if 0.8*np.pi < t < 1.2*np.pi)
            near_2pi = sum(1 for t in thetas if 1.8*np.pi < t or t < 0.2*np.pi)
            if near_pi > near_2pi:
                fermion_evidence += 1
            else:
                boson_evidence += 1
        
        if fermion_evidence > boson_evidence:
            f.write("Evidence suggests FERMIONIC EXCHANGE STATISTICS in phase bundle composites.\n")
            f.write("The ~70-80 phase singularities may be forming fermion-like bound states.\n")
        elif boson_evidence > fermion_evidence:
            f.write("Evidence suggests BOSONIC EXCHANGE STATISTICS dominate.\n")
            f.write("Phase bundles form integer-spin composite structures.\n")
        else:
            f.write("Mixed evidence. Further analysis needed.\n")
    
    print(f"  ✓ ANALYSIS_SUMMARY.txt")
    print(f"\n✓ Full report saved: {report_path}")


# ======================================================================
# COMMAND LINE INTERFACE
# ======================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage: python exchange_statistics_analyzer.py <data_directory> [output_directory]")
        print("\nExample:")
        print("  python exchange_statistics_analyzer.py vortex_output/ exchange_results/")
        print("\nThis will analyze all *_snap_*.npz files in the data directory.")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "exchange_analysis"
    
    results = run_complete_analysis(data_dir, output_dir)
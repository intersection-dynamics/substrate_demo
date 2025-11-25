#!/usr/bin/env python
"""
visualize_exclusion.py

Visualizes the output of the Substrate Engine.
Creates a side-by-side animated GIF comparing Fermionic (w=1) and Bosonic (w=0) behavior.
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def load_snapshots(data_dir):
    """
    Loads all .npz snapshots from a directory and sorts them by step.
    Returns a list of (t, rho) tuples.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "substrate_*.npz")))
    if not files:
        print(f"[WARNING] No .npz files found in {data_dir}")
        return []

    data = []
    print(f"[INFO] Loading {len(files)} snapshots from {data_dir}...")
    for f in files:
        try:
            with np.load(f) as loaded:
                # Extract time and density
                t = loaded['t']
                rho = loaded['rho']
                data.append((t, rho))
        except Exception as e:
            print(f"[ERROR] Failed to read {f}: {e}")
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Visualize Substrate Exclusion Experiment")
    parser.add_argument("--fermions", type=str, default="output_fermions", help="Directory for Fermion data")
    parser.add_argument("--bosons", type=str, default="output_bosons", help="Directory for Boson data")
    parser.add_argument("--out", type=str, default="exclusion_comparison.gif", help="Output GIF filename")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    args = parser.parse_args()

    # Load Data
    fermion_data = load_snapshots(args.fermions)
    boson_data = load_snapshots(args.bosons)

    if not fermion_data and not boson_data:
        print("No data found. Exiting.")
        return

    # Determine max density for consistent color scaling
    # We check a few frames to get a sense of the scale, or use a fixed safe max
    # Bosons spike high, Fermions stay low. We want a scale that shows the Boson spike but doesn't hide Fermions.
    # Let's use a dynamic scaling or a fixed reasonable one.
    # The logs showed Bosons hitting ~0.1 and Fermions ~0.01. 
    # A log scale or a cap at 0.05 might be best visually. Let's stick to linear but cap it.
    vmax = 0.05 

    # Setup Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Emergent Statistics: Topological Exclusion vs. Defrag Attraction", fontsize=16)

    # Initial empty plots
    img_f = axes[0].imshow(np.zeros((10, 10)), cmap='inferno', vmin=0, vmax=vmax, origin='lower')
    axes[0].set_title(f"Fermions (w=1)\nRepulsion / Orbit", color='cyan', fontweight='bold')
    axes[0].axis('off')

    img_b = axes[1].imshow(np.zeros((10, 10)), cmap='inferno', vmin=0, vmax=vmax, origin='lower')
    axes[1].set_title(f"Bosons (w=0)\nAttraction / Merge", color='orange', fontweight='bold')
    axes[1].axis('off')

    # Text annotations for time
    time_text = fig.text(0.5, 0.05, '', ha='center', fontsize=12)

    # Number of frames is the minimum of the two datasets
    n_frames = min(len(fermion_data), len(boson_data))
    if n_frames == 0:
        # Handle case where only one exists
        n_frames = max(len(fermion_data), len(boson_data))

    def update(frame_idx):
        # Update Fermion Plot
        if frame_idx < len(fermion_data):
            t_f, rho_f = fermion_data[frame_idx]
            img_f.set_data(rho_f)
            # Adjust extent if needed, or just assume grid is static
        
        # Update Boson Plot
        if frame_idx < len(boson_data):
            t_b, rho_b = boson_data[frame_idx]
            img_b.set_data(rho_b)

        current_t = fermion_data[frame_idx][0] if frame_idx < len(fermion_data) else boson_data[frame_idx][0]
        time_text.set_text(f"Simulation Time: {current_t:.2f}")
        
        return img_f, img_b, time_text

    print(f"[INFO] Generating animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/args.fps, blit=True)

    print(f"[INFO] Saving to {args.out} (this may take a moment)...")
    # Use PillowWriter for .gif
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    print(f"[DONE] Animation saved to {args.out}")

if __name__ == "__main__":
    main()
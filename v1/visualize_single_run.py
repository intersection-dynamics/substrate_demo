import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import glob
import os
import argparse

def visualize_single_run(data_dir, output_gif):
    files = sorted(glob.glob(os.path.join(data_dir, "substrate_*.npz")))
    if not files:
        print(f"Error: No .npz files found in {data_dir}")
        return

    frames = []
    
    print(f"Generating GIF from {len(files)} snapshots...")

    # Load first file to set up plot
    with np.load(files[0]) as data:
        rho = data['rho']
        
    vmin = 0.0
    # Determine max density across all files for a consistent color scale
    vmax = np.max([np.max(np.load(f)['rho']) for f in files]) * 1.05 

    fig, ax = plt.subplots(figsize=(6, 6))
    
    for i, fname in enumerate(files):
        with np.load(fname) as data:
            rho = data['rho']
            t = data['t']
        
        ax.clear()

        # Plot the density
        im = ax.imshow(rho, cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
        
        # Add labels and title
        ax.set_title(f"Stable Fermion Density (w=1)\nTime: t={t:.2f}")
        ax.set_xlabel("X Grid Index")
        ax.set_ylabel("Y Grid Index")
        
        # Add a colorbar on the first frame
        if i == 0:
            plt.colorbar(im, ax=ax, label='Probability Density $\\rho$')

        # Save the current frame to a temporary file
        plt.tight_layout()
        plt.savefig('temp_frame.png')
        
        frames.append(imageio.imread('temp_frame.png'))
        
        # Clean up the temporary file
        os.remove('temp_frame.png')

    plt.close(fig)
    
    # Save the GIF
    imageio.mimsave(output_gif, frames, fps=15)
    print(f"Successfully saved GIF to {output_gif}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a single Substrate simulation run.")
    parser.add_argument("--dir", type=str, required=True, help="Input directory containing substrate_*.npz files.")
    parser.add_argument("--out", type=str, default="fermion_single_run.gif", help="Output GIF file name.")
    args = parser.parse_args()

    visualize_single_run(args.dir, args.out)
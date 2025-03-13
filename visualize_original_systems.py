import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import imageio
import imageio.v2 as imageio
import shutil

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

def load_dataset(folder):
    """Load dataset from a given folder."""
    u = np.load(os.path.join(folder, "u.npy"))
    x = np.load(os.path.join(folder, "x.npy"))
    t = np.load(os.path.join(folder, "t.npy"))
    y = np.load(os.path.join(folder, "y.npy")) if os.path.exists(os.path.join(folder, "y.npy")) else None
    v = np.load(os.path.join(folder, "v.npy")) if os.path.exists(os.path.join(folder, "v.npy")) else None
    return u.squeeze(), v.squeeze() if v is not None else None, x.squeeze(), y.squeeze() if y is not None else None, t.squeeze()


def plot_1d_heatmap(data, times, X, variable_name, results_dir, colormap):
    """Generate and save a heatmap image for 1D data (systems 1 and 2)."""
    plt.figure(figsize=(10, 6))
    plt.imshow(data.T, aspect='auto', cmap=colormap, 
               extent=[X.min(), X.max(), times.min(), times.max()], origin='lower')
    plt.colorbar()
    plt.xlabel(r'$x$', fontsize=14)
    plt.ylabel(r'$t$', fontsize=14)
    plt.title(r'$u(x,t)$', fontsize=16, pad=10)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(ticks=np.linspace(X.min(), X.max(), 5),
               labels=[rf"${val:.2f}$" for val in np.linspace(X.min(), X.max(), 5)])
    plt.yticks(ticks=np.linspace(times.min(), times.max(), 5),
               labels=[rf"${val:.2f}$" for val in np.linspace(times.min(), times.max(), 5)])


    plt.tight_layout() 
    
    image_path = os.path.join(results_dir, f'{variable_name}_heatmap.png')
    
    plt.savefig(image_path, dpi=300)
    plt.close()
    print(f"Heatmap saved as '{image_path}'")

    
def plot_1d_heatmap_anim(data, times, X, variable_name, results_dir, colormap):
    """Generate and save animation for 1D data (systems 1 and 2) using a colormap and colorbar."""
    x_min, x_max = X.min(), X.max()
    u_min, u_max = data.min(), data.max()  # Normalize u values for colormap
    snapshots = data.shape[-1]
    
    frames_dir = os.path.join(results_dir, f"frames_{variable_name}")
    os.makedirs(frames_dir, exist_ok=True)

    # Get colormap and normalize it
    cmap = plt.get_cmap(colormap)  # FIX: Updated to avoid deprecation warning
    norm = mcolors.Normalize(vmin=u_min, vmax=u_max)

    plt.figure(figsize=(8, 6))
    for i in range(snapshots):
        plt.clf()
        
        # Get color based on the value of u at this time step
        colors = [cmap(norm(u)) for u in data[:, i]]

        # Plot each segment with the corresponding color
        for j in range(len(X) - 1):
            plt.plot(X[j:j+2], data[j:j+2, i], lw=3, color=colors[j])

        plt.xlabel(r'$x$', fontsize=14)
        plt.ylabel(r'$u$', fontsize=14)
        plt.title(rf'$u(x, t={i}/{snapshots - 1})$', fontsize=16, pad=10)
        plt.ylim(u_min, u_max)
        
        plt.xticks(ticks=np.linspace(X.min(), X.max(), 5),
                   labels=[rf"${val:.2f}$" for val in np.linspace(X.min(), X.max(), 5)])
        plt.yticks(ticks=np.linspace(u_min, u_max, 5),
                   labels=[rf"${val:.2f}$" for val in np.linspace(u_min, u_max, 5)])

        plt.grid()

        # FIX: Create ScalarMappable and properly link it to plt.colorbar()
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # Required for colorbar to work
        cbar = plt.colorbar(sm, ax=plt.gca())  # FIX: Explicitly link colorbar to axes
        cbar.set_label(r'$u(x,t)$', fontsize=14)

        plt.savefig(f'{frames_dir}/frame_{i:04d}.png', dpi=300)
    
    plt.close()
    
    # Create GIF
    gif_path = os.path.join(results_dir, f'{variable_name}_animation.gif')
    with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
        for i in range(snapshots):
            frame = imageio.imread(f'{frames_dir}/frame_{i:04d}.png')
            writer.append_data(frame)

    shutil.rmtree(frames_dir)
    print(f"GIF saved as '{gif_path}'")



def plot_2d_heatmap_anim(data, times, X, Y, variable_name, results_dir, colormap):
    """Generate and save animation for 2D data (system 3)."""
    x_min, x_max = X.min(), X.max()
    y_min, y_max = (Y.min(), Y.max()) if Y is not None else (0, 1)
    
    if data.ndim == 2:
        data = np.expand_dims(data, axis=-1)
    
    snapshots = data.shape[-1]

    frames_dir = os.path.join(results_dir, f"frames_{variable_name}")
    os.makedirs(frames_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    for i in range(snapshots):
        plt.clf()
        plt.imshow(data[:, :, i], extent=[x_min, x_max, y_min, y_max],
                   origin='lower', cmap=colormap)
        plt.colorbar()
        plt.xlabel(r'$x$', fontsize=14)
        plt.ylabel(r'$y$', fontsize=14)
        plt.title(rf'$u(x, y, t = {i})$', fontsize=16, pad=10)
        
        plt.xticks(ticks=np.linspace(x_min, x_max, 5),
           labels=[rf"${val:.2f}$" for val in np.linspace(x_min, x_max, 5)])
        plt.yticks(ticks=np.linspace(y_min, y_max, 5),
           labels=[rf"${val:.2f}$" for val in np.linspace(y_min, y_max, 5)])

        plt.savefig(f'{frames_dir}/frame_{i:04d}.png')
    
    plt.close()
    
    gif_path = os.path.join(results_dir, f'{variable_name}_animation.gif')
    with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
        for i in range(snapshots):
            frame = imageio.imread(f'{frames_dir}/frame_{i:04d}.png')
            writer.append_data(frame)

    shutil.rmtree(frames_dir)
    print(f"GIF saved as '{gif_path}'")
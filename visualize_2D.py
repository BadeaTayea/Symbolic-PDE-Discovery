import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math
import imageio
import shutil
import matplotlib.ticker as ticker

import matplotlib.image as mpimg
import imageio.v2 as imageio


def load_dataset(folder):
    """Load dataset from a given folder."""
    u = np.load(os.path.join(folder, "u.npy"))
    x = np.load(os.path.join(folder, "x.npy"))
    t = np.load(os.path.join(folder, "t.npy"))
    y = np.load(os.path.join(folder, "y.npy")) if os.path.exists(os.path.join(folder, "y.npy")) else None
    v = np.load(os.path.join(folder, "v.npy")) if os.path.exists(os.path.join(folder, "v.npy")) else None
    return u.squeeze(), v.squeeze() if v is not None else None, x.squeeze(), y.squeeze() if y is not None else None, t.squeeze()

def animate_comparison_heatmaps_2D(data_true, data_pred, x, y, times, variable_name, results_dir, colormap):
    """
    Create an animation with three subplots (side-by-side) for each time frame:
      - Left: Original field (e.g. true u_t or v_t)
      - Middle: Predicted field
      - Right: Absolute error (with relative L2 error metric in the title)
      
    Parameters:
        data_true (ndarray): True field; 3D array of shape (nx, ny, nt).
        data_pred (ndarray): Predicted field; 3D array of shape (nx, ny, nt).
        x (ndarray): 1D spatial grid for x.
        y (ndarray): 1D spatial grid for y.
        times (ndarray): 1D array of time values (length nt).
        variable_name (str): Name of the variable (e.g. "u_t" or "v_t").
        results_dir (str): Directory in which to save the animation.
        colormap (str): Colormap to use (e.g. "jet").
    """
    # Ensure data are real
    data_true = np.real(data_true).astype(float)
    data_pred = np.real(data_pred).astype(float)
    
    # Define spatial extent from x and y (assuming x and y are 1D)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    nt = data_true.shape[-1]
    
    # Create a temporary directory to store frames
    frames_dir = os.path.join(results_dir, f"frames_{variable_name}")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Loop over time snapshots to generate frames
    for i in range(nt):
        plt.figure(figsize=(18, 6))
        
        # Left subplot: original field
        ax1 = plt.subplot(1, 3, 1)
        im1 = ax1.imshow(data_true[:, :, i].T, aspect='auto', origin='lower',
                         extent=[x_min, x_max, y_min, y_max], cmap=colormap)
        ax1.set_title(f"Original ${variable_name}$ $|$ Downsampled", fontsize=14)
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("$y$")
        plt.colorbar(im1, ax=ax1)
        
        # Middle subplot: predicted field
        ax2 = plt.subplot(1, 3, 2)
        im2 = ax2.imshow(data_pred[:, :, i].T, aspect='auto', origin='lower',
                         extent=[x_min, x_max, y_min, y_max], cmap=colormap)
        ax2.set_title(f"Predicted ${variable_name}$", fontsize=14)
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("$y$")
        plt.colorbar(im2, ax=ax2)
        
        # Right subplot: absolute error
        err = np.abs(data_true[:, :, i] - data_pred[:, :, i])
        # Compute relative L2 error for the current frame
        norm_true = np.linalg.norm(data_true[:, :, i])
        rel_err = np.linalg.norm(err) / norm_true if norm_true != 0 else 0.0
        
        ax3 = plt.subplot(1, 3, 3)
        im3 = ax3.imshow(err.T, aspect='auto', origin='lower',
                         extent=[x_min, x_max, y_min, y_max], cmap=colormap)
        ax3.set_title(f"Absolute Error | {rel_err:.2e}", fontsize=14)
        ax3.set_xlabel("$x$")
        ax3.set_ylabel("$y$")
        plt.colorbar(im3, ax=ax3)
        
        # Overall figure title with the current time snapshot
        plt.suptitle(f"$ {variable_name}(x,y,t) $ at $t = {times[i]:.2f}$", fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=300)
        plt.close()
    
    # Create GIF animation from saved frames
    gif_path = os.path.join(results_dir, f"{variable_name}_animation.gif")
    with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
        for i in range(nt):
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            image = imageio.imread(frame_path)
            writer.append_data(image)
    
    shutil.rmtree(frames_dir)
    print(f"Animation saved as '{gif_path}'")


def plot_residual_over_time_2D(ut, u_t_pred, t, save_path, variable_label="u_t"):
    """
    Plot the residual (L2 norm over the spatial dimensions x and y) over time.
    
    Parameters:
        ut (ndarray): True time derivative array of shape (nx, ny, nt).
        u_t_pred (ndarray): Predicted time derivative array of shape (nx, ny, nt).
        t (ndarray): 1D temporal grid (length nt).
        save_path (str): Path to save the resulting plot.
        variable_label (str): Label for the variable (e.g. "u_t" or "v_t").
    """
    # Ensure the data are real-valued floats
    ut = np.real(ut).astype(float)
    u_t_pred = np.real(u_t_pred).astype(float)
    
    # Initialize an array to hold the error metric at each time step
    res_norm = np.zeros(t.size)
    
    # For each time step, compute the L2 norm of the difference over the spatial domain.
    for i in range(t.size):
        residual = ut[:, :, i] - u_t_pred[:, :, i]
        res_norm[i] = np.linalg.norm(residual)
    
    # Create the plot with markers (white-filled circles with blue edge)
    plt.figure(figsize=(10, 6))
    plt.plot(t, res_norm, 'b-', lw=2)
    plt.xlabel(r"$t$", fontsize=14)
    plt.ylabel(rf"$\|{variable_label} - \widehat{{{variable_label}}}\|_2$", fontsize=14)
    plt.title(rf"$\textrm{{Residual over Time for }} {variable_label}$", fontsize=16)
    plt.grid(True, linestyle='--', color='gray', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Residual over time plot for {variable_label} saved as '{save_path}'")


def animate_comparison_2D_slice(x, t, u_true, u_pred, y_index, save_path, variable_label, duration=0.1):
    """
    Create an animation where, for each time step, the true and approximated 
    {variable_label} (extracted from a fixed y-slice of a 3D array, shape (nx, ny, nt))
    are plotted as functions of x.
    
    The legend shows:
      - "Original {variable_label}"
      - "Approximated {variable_label}"
    
    The title updates with the current frame number, the relative L2 error (computed over x at the fixed y),
    and indicates the fixed y-value.
    
    Parameters:
        x (ndarray): 1D spatial grid for x (length nx).
        t (ndarray): 1D temporal grid (length nt).
        u_true (ndarray): True field (shape: nx x ny x nt).
        u_pred (ndarray): Approximated field (shape: nx x ny x nt).
        y_index (int): Index in the y dimension to fix.
        save_path (str): Path to save the animation (GIF).
        variable_label (str): Label for the variable (e.g. "u_t" or "v_t").
        duration (float): Duration between frames (used to set fps).
    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Ensure data are real-valued floats
    u_true = np.real(u_true).astype(float)
    u_pred = np.real(u_pred).astype(float)
    
    nx = x.size
    nt = t.size
    
    # Determine the fixed y value if a 1D y grid exists.
    # Here, if not provided, we simply use the index.
    y_value = y_index  # you can modify this if a 1D y grid is available
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create plot lines with markers for both curves.
    line_true, = ax.plot([], [], color='b', linestyle='-', lw=2, marker='o',
                           label=f"Original ${variable_label}$")
    line_pred, = ax.plot([], [], color='r', linestyle='--', lw=2, marker='o',
                           label=f"Approximated ${variable_label}$")
    
    ax.set_xlim(x.min(), x.max())
    overall_min = min(u_true.min(), u_pred.min())
    overall_max = max(u_true.max(), u_pred.max())
    ax.set_ylim(overall_min, overall_max)
    ax.set_xlabel("$x$", fontsize=14)
    ax.set_ylabel(f"${variable_label}$", fontsize=14)
    ax.grid(True, linestyle='--', color='gray', alpha=0.7)
    
    # Fix the legend position once.
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    def init():
        line_true.set_data([], [])
        line_pred.set_data([], [])
        # Compute initial relative L2 error over x at the fixed y for frame 0.
        true_slice = u_true[:, y_index, 0]
        pred_slice = u_pred[:, y_index, 0]
        rel_err = np.linalg.norm(true_slice - pred_slice) / np.linalg.norm(true_slice)
        ax.set_title(rf"${variable_label}(x, t={1}/{nt}) \;|\; y={y_value} \;|\; \textrm{{Relative }}L_2\textrm{{ Error: }} {rel_err:.2e}$")
        return line_true, line_pred

    def update(frame):
        true_slice = u_true[:, y_index, frame]
        pred_slice = u_pred[:, y_index, frame]
        line_true.set_data(x, true_slice)
        line_pred.set_data(x, pred_slice)
        rel_err = np.linalg.norm(true_slice - pred_slice) / np.linalg.norm(true_slice)
        ax.set_title(rf"${variable_label}(x, t={frame+1}/{nt}) \;|\; y={y_value} \;|\; \textrm{{Relative }}L_2\textrm{{ Error: }} {rel_err:.2e}$")
        return line_true, line_pred

    ani = animation.FuncAnimation(fig, update, frames=nt, init_func=init, blit=False)
    
    ani.save(save_path, writer='imagemagick', fps=int(1/duration))
    plt.close()
    print(f"Animation saved as '{save_path}'")




def plot_pde_terms_multioutput(rhs_description, Xi, save_path_u, save_path_v):
    """
    Plot bar charts for the PDE term coefficients for a multi-output (coupled) PDE.
    
    Parameters:
        rhs_description (list): List of strings describing each candidate term.
        Xi (ndarray): Coefficient matrix of shape (n_terms, 2).
            The first column corresponds to the u_t equation and the second to the v_t equation.
        save_path_u (str): File path to save the bar chart for u_t.
        save_path_v (str): File path to save the bar chart for v_t.
    """
    import matplotlib.pyplot as plt
    # Convert coefficients to real-valued 1D arrays
    xi_u = np.real(Xi[:, 0]).flatten()
    xi_v = np.real(Xi[:, 1]).flatten()
    # Wrap each term in math mode (empty term is the constant, displayed as "1")
    terms = [f"${term}$" if term != "" else "$1$" for term in rhs_description]
    
    # Plot for u_t
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(xi_u)), xi_u, color='blue')
    ax.set_xticks(range(len(xi_u)))
    ax.set_xticklabels(terms, rotation=45, ha='right')
    ax.set_xlabel(r"$\textrm{PDE\ Term}$", fontsize=14)
    ax.set_ylabel(r"$\textrm{Coefficient\ Value}$", fontsize=14)
    ax.set_title(r"$\textrm{PDE\ Term\ Coefficients for } u_t$", fontsize=16)
    ax.grid(True, which='major', axis='y', linestyle='--', color='gray', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path_u, dpi=300)
    plt.close()
    print(f"Term coefficients for u_t plot saved as '{save_path_u}'")
    
    # Plot for v_t
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(xi_v)), xi_v, color='blue')
    ax.set_xticks(range(len(xi_v)))
    ax.set_xticklabels(terms, rotation=45, ha='right')
    ax.set_xlabel(r"$\textrm{PDE\ Term}$", fontsize=14)
    ax.set_ylabel(r"$\textrm{Coefficient\ Value}$", fontsize=14)
    ax.set_title(r"$\textrm{PDE\ Term\ Coefficients for } v_t$", fontsize=16)
    ax.grid(True, which='major', axis='y', linestyle='--', color='gray', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path_v, dpi=300)
    plt.close()
    print(f"Term coefficients for v_t plot saved as '{save_path_v}'")


def plot_candidate_terms_PDE3(Theta, rhs_description, nx, ny, nt, x, t, y_index, save_path):
    """
    Plot each candidate term (each column of Theta) as a heatmap over a fixed y-slice
    (i.e., as a function of x and t) for a coupled 2+1D PDE.
    
    Parameters:
        Theta (ndarray): Candidate feature matrix of shape (nx*ny*nt, n_terms).
        rhs_description (list): List of strings describing each candidate term.
        nx (int): Number of x points.
        ny (int): Number of y points.
        nt (int): Number of time points.
        x (ndarray): 1D spatial grid for x.
        t (ndarray): 1D temporal grid.
        y_index (int): The index in the y dimension to fix (e.g. middle of the domain).
        save_path (str): Path to save the resulting figure.
    """
    n_terms = Theta.shape[1]
    # Reshape Theta back to a 4D array with dimensions (nx, ny, nt, n_terms)
    Theta_reshaped = Theta.reshape((nx, ny, nt, n_terms), order='F')
    
    # Choose a grid for subplots (for example, a square grid)
    cols = int(np.ceil(np.sqrt(n_terms)))
    rows = int(np.ceil(n_terms / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).flatten()
    
    for i in range(n_terms):
        # Extract a 2D slice at the fixed y-index (shape: (nx, nt))
        term_data = np.real(Theta_reshaped[:, y_index, :, i]).astype(float)
        ax = axes[i]
        im = ax.imshow(term_data.T, aspect='auto', origin='lower',
                       extent=[x.min(), x.max(), t.min(), t.max()],
                       cmap='jet')
        # Use the candidate term description; if empty, show "1"
        title = rhs_description[i] if rhs_description[i] != "" else "1"
        ax.set_title(f"${title}$", fontsize=12)
        ax.set_xlabel(r"$x$", fontsize=10)
        ax.set_ylabel(r"$t$", fontsize=10)
        fig.colorbar(im, ax=ax)
    
    # Remove any unused subplots
    for j in range(n_terms, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Candidate terms plot saved as '{save_path}'")


def plot_candidate_terms_PDE3(Theta, rhs_description, nx, ny, nt, x, t, y_index, save_path):
    """
    Plot each candidate term (each column of Theta) as a heatmap over a fixed y-slice
    (i.e., as a function of x and t) for a coupled 2+1D PDE using the full data.
    
    Parameters:
        Theta (ndarray): Candidate feature matrix of shape (nx*ny*nt, n_terms).
        rhs_description (list): List of strings describing each candidate term.
        nx (int): Number of x points.
        ny (int): Number of y points.
        nt (int): Number of time points.
        x (ndarray): 1D spatial grid for x.
        t (ndarray): 1D temporal grid.
        y_index (int): The index in the y dimension to fix (e.g. ny//2).
        save_path (str): Path to save the resulting figure.
    """
    import matplotlib.pyplot as plt
    n_terms = Theta.shape[1]
    # Reshape Theta back to a 4D array with dimensions (nx, ny, nt, n_terms) using Fortran order.
    Theta_reshaped = Theta.reshape((nx, ny, nt, n_terms), order='F')
    
    # Set up a grid of subplots; here we arrange them roughly square.
    cols = int(np.ceil(np.sqrt(n_terms)))
    rows = int(np.ceil(n_terms / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).flatten()
    
    for i in range(n_terms):
        # Extract the (nx x nt) slice at the fixed y-index.
        term_data = np.real(Theta_reshaped[:, y_index, :, i]).astype(float)
        ax = axes[i]
        im = ax.imshow(term_data.T, aspect='auto', origin='lower',
                       extent=[x.min(), x.max(), t.min(), t.max()],
                       cmap='jet')
        # If the candidate description is empty (constant term), display "1".
        title = rhs_description[i] if rhs_description[i] != "" else "1"
        ax.set_title(f"${title}$", fontsize=12)
        ax.set_xlabel(r"$x$", fontsize=10)
        ax.set_ylabel(r"$t$", fontsize=10)
        fig.colorbar(im, ax=ax)
    
    # Remove any extra subplots.
    for j in range(n_terms, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Candidate terms plot saved as '{save_path}'")


def plot_candidate_terms_spatial(Theta, rhs_description, nx, ny, nt, x, y, t_index, save_path):
    """
    Plot each candidate term (each column of Theta) as a spatial heatmap at a fixed time.
    
    Parameters:
        Theta (ndarray): Candidate feature matrix of shape (nx*ny*nt, n_terms).
        rhs_description (list): List of strings describing each candidate term.
        nx (int): Number of x points.
        ny (int): Number of y points.
        nt (int): Number of time points.
        x (ndarray): 1D spatial grid for x.
        y (ndarray): 1D spatial grid for y.
        t_index (int): The time index to use for the snapshot.
        save_path (str): Path to save the resulting plot.
    """
    import matplotlib.pyplot as plt
    n_terms = Theta.shape[1]
    # Reshape Theta back to a 4D array: (nx, ny, nt, n_terms)
    Theta_reshaped = Theta.reshape((nx, ny, nt, n_terms), order='F')
    
    # Set up a grid for subplots.
    cols = int(np.ceil(np.sqrt(n_terms)))
    rows = int(np.ceil(n_terms / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).flatten()
    
    for i in range(n_terms):
        # Extract the 2D spatial slice at the chosen time index
        term_data = np.real(Theta_reshaped[:, :, t_index, i]).astype(float)
        ax = axes[i]
        im = ax.imshow(term_data.T, aspect='auto', origin='lower',
                       extent=[x.min(), x.max(), y.min(), y.max()],
                       cmap='jet')
        # If candidate term is empty (constant term), show "1"
        title = rhs_description[i] if rhs_description[i] != "" else "1"
        ax.set_title(f"${title}$", fontsize=12)
        ax.set_xlabel(r"$x$", fontsize=10)
        ax.set_ylabel(r"$y$", fontsize=10)
        fig.colorbar(im, ax=ax)
    
    # Remove any unused subplots
    for j in range(n_terms, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Spatial candidate terms plot saved as '{save_path}'")


def scatter_true_vs_pred_with_footnote(true, pred, xlabel, ylabel, title, file_name,
                                       minor_tick_length=4, footnote="* All spatial and temporal points pooled"):
    """
    Generates a scatter plot comparing true and reconstructed values,
    including a perfect-fit reference line, minor ticks, grid lines, and a footnote.
    
    Parameters:
        true (array-like): True values.
        pred (array-like): Reconstructed (predicted) values.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Plot title.
        file_name (str): Path to save the resulting plot.
        minor_tick_length (int, optional): Length of minor tick marks (default is 4).
        footnote (str, optional): Footnote text to be displayed (default is "* All spatial and temporal points pooled").
    """
    plt.figure(figsize=(8, 6))
    
    # Scatter plot for true vs. predicted values (flattened)
    plt.scatter(true.flatten(), pred.flatten(), alpha=1, s=10, color="blue", label="Downsampled Data*")
    
    # Plot a perfect-fit line
    plt.plot([true.min(), true.max()], [true.min(), true.max()], 'r--', label="Perfect Fit")
    
    # Set labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    
    # Enable minor ticks and set them automatically
    plt.minorticks_on()
    plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    # Adjust tick parameters for minor ticks
    plt.tick_params(which='minor', length=minor_tick_length, width=1, color='gray')
    
    # Add grid lines for both major and minor ticks
    plt.grid(which='major', linestyle='-', linewidth=0.5, color='gray')
    plt.grid(which='minor', linestyle=':', linewidth=0.5, color='gray')
    
    # Add footnote text centered at the bottom of the figure
    plt.figtext(0.5, 0.01, footnote, ha="center", fontsize=10, color='gray')
    
    # Save the figure and display it
    plt.savefig(file_name, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Scatter plot over time plot saved as '{file_name}'")



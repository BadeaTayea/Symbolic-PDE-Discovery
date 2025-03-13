import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math
import imageio
import shutil
import matplotlib.ticker as ticker



def plot_comparison_heatmaps(ut, u_pred, error, x, t, save_path):
    """
    Generate three side-by-side heatmaps for:
      - Original u_t,
      - Approximated u_t,
      - Absolute error, with a metric in the title.
      
    The metric (global relative L2 error) is computed as:
         ||ut - u_pred|| / ||ut||.
    
    Parameters:
        ut (ndarray): Original time derivative (nx x nt).
        u_pred (ndarray): Approximated time derivative (nx x nt).
        error (ndarray): Absolute error (nx x nt).
        x (ndarray): Spatial grid (1D array).
        t (ndarray): Temporal grid (1D array).
        save_path (str): Path to save the resulting plot.
    """
    # Convert to real floats
    ut = np.real(ut).astype(float)
    u_pred = np.real(u_pred).astype(float)
    error = np.real(error).astype(float)
    
    # Compute global relative L2 error metric
    rel_error = np.linalg.norm(ut - u_pred) / np.linalg.norm(ut)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original derivative heatmap
    im0 = axes[0].imshow(ut.T, aspect='auto', origin='lower',
                          extent=[x.min(), x.max(), t.min(), t.max()],
                          cmap='jet')
    axes[0].set_title('Original $u_t$')
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$t$')
    fig.colorbar(im0, ax=axes[0])
    
    # Approximated derivative heatmap
    im1 = axes[1].imshow(u_pred.T, aspect='auto', origin='lower',
                          extent=[x.min(), x.max(), t.min(), t.max()],
                          cmap='jet')
    axes[1].set_title('Approximated $u_t$')
    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel('$t$')
    fig.colorbar(im1, ax=axes[1])
    
    # Absolute error heatmap with metric in title
    im2 = axes[2].imshow(error.T, aspect='auto', origin='lower',
                          extent=[x.min(), x.max(), t.min(), t.max()],
                          cmap='jet')
    axes[2].set_title(f"Absolute Error | {rel_error:.2e}")
    axes[2].set_xlabel('$x$')
    axes[2].set_ylabel('$t$')
    fig.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Heatmap comparison saved as '{save_path}'")




def animate_comparison(x, t, ut, u_pred, pde_string, save_path, duration=0.1):
    """
    Create an animation where, for each time step, the original and approximated u_t 
    are plotted as functions of x. The legend is fixed at a given location, with:
      - "Original u_t"
      - "Approximated u_t = <pde_string>"
    The title updates with the frame number and the relative L2 error.
    
    Parameters:
        x (ndarray): Spatial grid (1D, length nx).
        t (ndarray): Temporal grid (1D, length nt).
        ut (ndarray): Original u_t (shape: nx x nt).
        u_pred (ndarray): Approximated u_t (shape: nx x nt).
        pde_string (str): LaTeX string for the approximated PDE.
        save_path (str): Path to save the animation (GIF).
        duration (float): Duration between frames (fps is set in save).
    """
    import matplotlib.animation as animation
    
    # Ensure data are real-valued floats
    ut = np.real(ut).astype(float)
    u_pred = np.real(u_pred).astype(float)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create plot lines with markers (with smaller marker size)
    line_orig, = ax.plot([], [], color='b', linestyle='-', lw=2, label="Original $u_t$")
    line_pred, = ax.plot([], [], color='r', linestyle='--', lw=2, label=f"Approximated $u_t = {pde_string}$")
    
    ax.set_xlim(x.min(), x.max())
    y_min = min(ut.min(), u_pred.min())
    y_max = max(ut.max(), u_pred.max())
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u_t$')
    ax.grid(True, linestyle='--', color='gray', alpha=0.7)
    
    # Fix the legend position (set it once and do not update it in each frame)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    def init():
        line_orig.set_data([], [])
        line_pred.set_data([], [])
        initial_error = np.linalg.norm(ut[:, 0] - u_pred[:, 0]) / np.linalg.norm(ut[:, 0])
        ax.set_title(rf"$u_t(x, t = {1}/{ut.shape[1]}) \;|\; \textrm{{Relative }}L_2\textrm{{ Error: }} {initial_error:.2e}$")
        return line_orig, line_pred

    def update(frame):
        y_orig = ut[:, frame]
        y_pred = u_pred[:, frame]
        line_orig.set_data(x, y_orig)
        line_pred.set_data(x, y_pred)
        error_frame = np.linalg.norm(ut[:, frame] - u_pred[:, frame]) / np.linalg.norm(ut[:, frame])
        ax.set_title(rf"$u_t(x, t = {frame+1}/{ut.shape[1]}) \;|\; \textrm{{Relative }}L_2\textrm{{ Error: }} {error_frame:.2e}$")
        return line_orig, line_pred

    ani = animation.FuncAnimation(fig, update, frames=ut.shape[1],
                                  init_func=init, blit=False)
    
    ani.save(save_path, writer='imagemagick', fps=10)
    plt.close()
    print(f"Animation saved as '{save_path}'")


def plot_pde_terms(rhs_description, xi, save_path):
    """
    Plot a bar chart of the PDE term coefficients with LaTeX-formatted title and axes labels.
    
    Parameters:
        rhs_description (list): List of strings describing each term.
        xi (ndarray): Coefficient vector (shape: (n_terms, 1)).
        save_path (str): Path to save the resulting plot.
    """
    # Convert xi to a real-valued 1D array
    xi = np.real(xi).flatten()
    # Wrap each term in math mode; use '$1$' for the empty string (constant term)
    terms = [f"${term}$" if term != "" else "$1$" for term in rhs_description]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(xi)), xi, color='blue')
    ax.set_xticks(range(len(xi)))
    ax.set_xticklabels(terms, rotation=45, ha='right')
    
    # Use \textrm instead of \text to render plain text
    ax.set_xlabel(r"$\textrm{PDE\ Term}$", fontsize=14)
    ax.set_ylabel(r"$\textrm{Coefficient\ Value}$", fontsize=14)
    ax.set_title(r"$\textrm{PDE\ Term\ Coefficients}$", fontsize=16)

    ax.grid(True, which='major', axis='y', linestyle='--', color='gray', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Term coefficients plot saved as '{save_path}'")


def plot_candidate_terms(Theta, rhs_description, n2, m2, x, t, save_path):
    """
    Plot each candidate term (column of Theta) as a heatmap over the spatiotemporal domain.
    
    Parameters:
        Theta (ndarray): Feature matrix of shape (n2*m2, n_terms).
        rhs_description (list): List of strings describing each term.
        n2 (int): Reduced spatial dimension.
        m2 (int): Reduced temporal dimension.
        x (ndarray): Spatial grid (1D array).
        t (ndarray): Temporal grid (1D array).
        save_path (str): Path to save the resulting plot image.
    """
    n_terms = Theta.shape[1]
    # Choose a grid for subplots (for example, a square grid)
    cols = int(np.ceil(np.sqrt(n_terms)))
    rows = int(np.ceil(n_terms / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).flatten()
        
    for i in range(n_terms):
        # Reshape the i-th column and convert to real-valued floats
        term_data = np.real(Theta[:, i].reshape((n2, m2), order='F')).astype(float)
        ax = axes[i]
        im = ax.imshow(term_data.T, aspect='auto', origin='lower',
                       extent=[x.min(), x.max(), t.min(), t.max()],
                       cmap='jet')
        # If the description is empty (the constant term), show "1"
        title = rhs_description[i] if rhs_description[i] != "" else "1"
        ax.set_title(f"${title}$", fontsize=12)
        ax.set_xlabel(r"$x$", fontsize=10)
        ax.set_ylabel(r"$t$", fontsize=10)
        fig.colorbar(im, ax=ax)
    
        
    # Remove any unused axes
    for j in range(n_terms, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Candidate terms plot saved as '{save_path}'")


def plot_residual_over_time(ut, u_t_pred, t, save_path):
    """
    Plot the residual (L2 norm over space) over time.
    
    Parameters:
        ut (ndarray): Original time derivative array of shape (nx, nt).
        u_t_pred (ndarray): Approximated time derivative array of shape (nx, nt).
        t (ndarray): Temporal grid (1D array, length nt).
        save_path (str): Path to save the plot.
    """
    # Ensure the data are real-valued
    ut = np.real(ut).astype(float)
    u_t_pred = np.real(u_t_pred).astype(float)
    
    # Compute residual at each time step
    residual = ut - u_t_pred
    # Compute the L2 norm along the spatial dimension for each time step
    res_norm = np.linalg.norm(residual, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, res_norm, 'b-', lw=2)
    plt.xlabel(r"$t$", fontsize=14)
    plt.ylabel(r"$\|u_t - \hat{u}_t\|_2$", fontsize=14)
    plt.title(r"$\textrm{Residual over Time}$", fontsize=16)
    plt.grid(True, linestyle='--', color='gray', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Residual over time plot saved as '{save_path}'")



def scatter_true_vs_pred(true, pred, file_name, minor_tick_length=2):
    """
    Generates a scatter plot comparing true and reconstructed values,
    including a perfect-fit reference line, minor ticks, and grid lines.
    
    Parameters:
        true (array-like): True values.
        pred (array-like): Reconstructed (predicted) values.
        file_name (str): Path to save the resulting plot.
        minor_tick_length (int, optional): Length of minor tick marks (default is 2).
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot for true vs. predicted values
    ax.scatter(true, pred, alpha=1, s=10, color="blue")
    
    # Plot a perfect-fit line
    ax.plot([true.min(), true.max()], [true.min(), true.max()], 'r--', label="Perfect Fit")
    
    # Set labels and title
    ax.set_xlabel("True $u_t$")
    ax.set_ylabel("Reconstructed $u_t$")
    ax.set_title("True vs Reconstructed $u_t$")
    ax.legend()
    
    # Enable minor ticks and set them automatically
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    # Adjust tick parameters for minor ticks
    ax.tick_params(which='minor', length=minor_tick_length, width=1, color='gray')
    
    # Add grid lines for both major and minor ticks
    ax.grid(which='major', linestyle='-', linewidth=0.5, color='gray')
    ax.grid(which='minor', linestyle=':', linewidth=0.5, color='gray')
    
    # Save the figure
    plt.savefig(file_name, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Scatter plot over time plot saved as '{file_name}'")


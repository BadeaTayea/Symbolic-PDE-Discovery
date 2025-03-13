import numpy as np
import scipy.sparse as sparse
import itertools
import operator

from sparse_regression_solvers import TrainSTRidge

def TrainSTRidge_multioutput(Theta, U_t, lam, d_tol, maxit=25, STR_iters=10, normalize=2, split=0.8, print_best_tol=False):
    """
    Perform sparse regression for a multi-output problem.
    U_t is of shape (N, 2). This function applies STRidge to each column independently.
    Returns a coefficient matrix Xi of shape (n_terms, 2).
    """
    Xi = np.zeros((Theta.shape[1], U_t.shape[1]), dtype=Theta.dtype)
    for i in range(U_t.shape[1]):
        Xi[:, i:i+1] = TrainSTRidge(Theta, U_t[:, i:i+1], lam, d_tol, maxit=maxit, STR_iters=STR_iters, normalize=normalize, split=split, print_best_tol=print_best_tol)
    return Xi


def build_linear_system_2D(u, v, dt, dx, dy, D=3, P=3, time_diff='FD', space_diff='FD'):
    """
    Build the linear system for a coupled 2+1D PDE.
    
    Parameters:=
      u, v: 3D arrays of shape (nx, ny, nt) for the two components.
      dt, dx, dy: Grid spacings.
      Other parameters: (D, P, methods) -- for this example we ignore D and P and use a simple candidate library.
      
    Returns:
      U_t: (N, 2) array with first column u_t and second v_t.
      Theta: (N, n_terms) candidate feature matrix.
      rhs_description: List of strings describing each candidate term.
      
    Here we compute derivatives using np.gradient.
    We include candidate terms: constant, u, v, u^2, v^2, uv, u_x, u_y, v_x, v_y.
    """
    nx, ny, nt = u.shape
    # Compute time derivatives (along axis 2)
    u_t = np.gradient(u, dt, axis=2)
    v_t = np.gradient(v, dt, axis=2)
    
    # Compute spatial derivatives (using central differences approximated by np.gradient)
    u_x = np.gradient(u, dx, axis=0)
    u_y = np.gradient(u, dy, axis=1)
    v_x = np.gradient(v, dx, axis=0)
    v_y = np.gradient(v, dy, axis=1)
    
    # For regression, we flatten all data.
    # We will flatten spatial dimensions and time together such that each sample is one (x,y,t) point.
    N = nx * ny * nt  # total number of samples
    
    # Helper to flatten a field to (N, )
    def flatten_field(field):
        return field.reshape(-1, order='F')
    
    # Flatten time derivatives for u and v.
    u_t_flat = flatten_field(u_t)
    v_t_flat = flatten_field(v_t)
    # Form U_t as an (N,2) array.
    U_t = np.hstack([u_t_flat[:, None], v_t_flat[:, None]])
    
    # Flatten the original fields and spatial derivatives.
    u_flat   = flatten_field(u)
    v_flat   = flatten_field(v)
    u_x_flat = flatten_field(u_x)
    u_y_flat = flatten_field(u_y)
    v_x_flat = flatten_field(v_x)
    v_y_flat = flatten_field(v_y)
    
    # Build candidate terms.
    constant = np.ones_like(u_flat)
    # Candidate library: constant, u, v, u^2, v^2, uv, u_x, u_y, v_x, v_y.
    Theta = np.column_stack([
        constant,
        u_flat,
        v_flat,
        u_flat**2,
        v_flat**2,
        u_flat * v_flat,
        u_x_flat,
        u_y_flat,
        v_x_flat,
        v_y_flat
    ])
    rhs_description = ["", "u", "v", "u^2", "v^2", "uv", "u_x", "u_y", "v_x", "v_y"]
    
    return U_t, Theta, rhs_description


def downsample_3d(array, factor):
    """
    Downsample a 3D array by taking every 'factor'-th element in each dimension.
    
    Parameters:
        array (ndarray): Input 3D array.
        factor (int): Downsampling factor.
        
    Returns:
        ndarray: Downsampled array.
    """
    return array[::factor, ::factor, ::factor]

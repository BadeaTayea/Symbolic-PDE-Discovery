import numpy as np
import scipy.sparse as sparse

def TikhonovDiff(f, dx, lam, d = 1):
    """
    Perform Tikhonov regularization to compute derivatives of a given function.

    This method solves the optimization problem:
        argmin_g ||Ag - f||_2^2 + λ||Dg||_2^2
    where:
        - A represents trapezoidal integration.
        - D represents finite differences for the first derivative.
    
    Parameters:
        f (array-like): Input function values to differentiate.
        dx (float): Spacing between grid points.
        lam (float): Regularization parameter controlling smoothness.
        d (int, optional): Desired order of differentiation (default: 1).

    Returns:
        ndarray: The estimated derivative of order `d`.

    Notes:
        - For d > 1, finite differences are applied after smoothing.
        - This method works well for ODEs but may introduce bias for PDEs.
        - For noisy data, consider using polynomial-based differentiation.
    """

    # Initialize a few things    
    n = len(f)
    f = np.matrix(f - f[0]).reshape((n,1))

    # Get a trapezoidal approximation to an integral
    A = np.zeros((n,n))
    for i in range(1, n):
        A[i,i] = dx/2
        A[i,0] = dx/2
        for j in range(1,i): A[i,j] = dx
    
    e = np.ones(n-1)
    D = sparse.diags([e, -e], [1, 0], shape=(n-1, n)).todense() / dx
    
    # Invert to find derivative
    g = np.squeeze(np.asarray(np.linalg.lstsq(A.T.dot(A) + lam*D.T.dot(D),A.T.dot(f),rcond=None)[0]))
    
    if d == 1: return g

    # For higher order derivatives, we can use finite differences
    else: return FiniteDiff(g, dx, d-1)
    

#####################################################################################################
#####################################################################################################


def FiniteDiff(u, dx, d):
    """
    Compute the d-th derivative of data using the 2nd-order finite difference method.

    This method supports up to third-order derivatives directly, with reduced accuracy for d > 3 
    (computed recursively).

    Parameters:
        u (array-like): Input data to differentiate.
        dx (float): Grid spacing, assumes uniform spacing.
        d (int): Order of the derivative (e.g., 1 for first derivative).

    Returns:
        ndarray: The d-th derivative of the input data.

    Notes:
        - For d = 1: Computes the first derivative with forward/backward finite differences at boundaries.
        - For d = 2: Computes the second derivative with forward/backward differences at boundaries.
        - For d = 3: Computes the third derivative using forward/backward differences at boundaries.
        - For d > 3: Computes recursively, using d = 3 as the base case.
        - Accuracy decreases for d > 3; consider alternative methods for high-order derivatives.
    """
    
    n = u.size
    ux = np.zeros(n, dtype=np.complex64)
    
    if d == 1:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-u[i-1]) / (2*dx)
        
        ux[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dx
        ux[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dx
        return ux
    
    if d == 2:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-2*u[i]+u[i-1]) / dx**2
        
        ux[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / dx**2
        ux[n-1] = (2*u[n-1] - 5*u[n-2] + 4*u[n-3] - u[n-4]) / dx**2
        return ux
    
    if d == 3:
        for i in range(2,n-2):
            ux[i] = (u[i+2]/2-u[i+1]+u[i-1]-u[i-2]/2) / dx**3
        
        ux[0] = (-2.5*u[0]+9*u[1]-12*u[2]+7*u[3]-1.5*u[4]) / dx**3
        ux[1] = (-2.5*u[1]+9*u[2]-12*u[3]+7*u[4]-1.5*u[5]) / dx**3
        ux[n-1] = (2.5*u[n-1]-9*u[n-2]+12*u[n-3]-7*u[n-4]+1.5*u[n-5]) / dx**3
        ux[n-2] = (2.5*u[n-2]-9*u[n-3]+12*u[n-4]-7*u[n-5]+1.5*u[n-6]) / dx**3
        return ux
    
    if d > 3:
        return FiniteDiff(FiniteDiff(u,dx,3), dx, d-3)
    

#####################################################################################################
#####################################################################################################


def ConvSmoother(x, p, sigma):
    """
    Apply a Gaussian convolutional smoother to a one-dimensional series.

    This function smooths noisy data using a Gaussian kernel, accounting for boundary effects.

    Parameters:
        x (array-like): One-dimensional input data to be smoothed.
        p (int): Half-width of the smoothing kernel (number of neighboring points to consider).
        sigma (float): Standard deviation of the Gaussian smoothing kernel.

    Returns:
        ndarray: Smoothed version of the input data.
    """
    
    n = len(x)
    y = np.zeros(n, dtype=np.complex64)
    g = np.exp(-np.power(np.linspace(-p,p,2*p),2)/(2.0*sigma**2))

    for i in range(n):
        a = max([i-p,0])
        b = min([i+p,n])
        c = max([0, p-i])
        d = min([2*p,p+n-i])
        y[i] = np.sum(np.multiply(x[a:b], g[c:d]))/np.sum(g[c:d])
        
    return y


#####################################################################################################
#####################################################################################################


def PolyDiff(u, x, deg = 3, diff = 1, width = 5):
    
    """
    Compute derivatives of a function using polynomial fitting.

    This function approximates the derivatives of a function by fitting 
    a polynomial of specified degree over a moving window of points. 
    The derivatives are evaluated at the center of the window, excluding 
    edges where fitting becomes less accurate.

    Parameters:
        u (array-like): Values of the function to differentiate (1D array).
        x (array-like): Corresponding x-coordinates where the function values are known (1D array).
        deg (int, optional): Degree of the polynomial to fit (default: 3).
        diff (int, optional): Maximum order of derivatives to compute (default: 1).
        width (int, optional): Half-width of the moving window used for polynomial fitting 
                               (default: 5, resulting in a total window size of `2*width + 1`).

    Returns:
        ndarray: A 2D array of shape `(n - 2 * width, diff)`, where each column corresponds 
                 to a derivative order (1st, 2nd, ..., `diff`) and rows correspond to 
                 the central points where derivatives are computed.
    """

    u = u.flatten()
    x = x.flatten()

    n = len(x)
    du = np.zeros((n - 2*width,diff))

    # Take the derivatives in the center of the domain
    for j in range(width, n-width):

        # Note code originally used an even number of points here.
        # This is an oversight in the original code fixed in 2022.
        points = np.arange(j - width, j + width + 1)

        # Fit to a polynomial
        poly = np.polynomial.chebyshev.Chebyshev.fit(x[points],u[points],deg)

        # Take derivatives
        for d in range(1,diff+1):
            du[j-width, d-1] = poly.deriv(m=d)(x[j])

    return du


#####################################################################################################
#####################################################################################################


def PolyDiffPoint(u, x, deg = 3, diff = 1, index = None):
    
    """
    Compute derivatives of a function at a single point using polynomial fitting.

    This function fits a polynomial of a specified degree to a set of function 
    values and their corresponding coordinates. It computes the derivatives 
    at a specified index or at the midpoint of the data if no index is provided.

    Parameters:
        u (array-like): Values of the function to differentiate (1D array).
        x (array-like): Corresponding x-coordinates where the function values are known (1D array).
        deg (int, optional): Degree of the polynomial to fit (default: 3).
        diff (int, optional): Maximum order of derivatives to compute (default: 1).
        index (int, optional): Index of the point at which derivatives are evaluated. 
                               If not provided, the midpoint of the data is used 
                               (default: `(len(x) - 1) // 2`).

    Returns:
        list: A list of length `diff` containing the derivatives of the function 
              at the specified point, from the 1st derivative up to the `diff`-th derivative.
    """
    
    n = len(x)
    if index == None: index = (n-1)//2

    # Fit to a polynomial
    poly = np.polynomial.chebyshev.Chebyshev.fit(x,u,deg)
    
    # Take derivatives
    derivatives = []
    for d in range(1,diff+1):
        derivatives.append(poly.deriv(m=d)(x[index]))
        
    return derivatives
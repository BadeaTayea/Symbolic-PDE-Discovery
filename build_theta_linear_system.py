import numpy as np
import scipy.sparse as sparse
import itertools
import operator

from derivative_approximators import FiniteDiff, ConvSmoother, PolyDiff, TikhonovDiff


def build_Theta(data, derivatives, derivatives_description, P, data_description = None):
    """
    Constructs a feature matrix, Theta, with columns representing polynomials up to degree P 
    of variables in `data` and their derivatives.

    Parameters:
        data (ndarray): Array where column 0 is U (solution data) and subsequent columns are optional inputs (Q).
        derivatives (ndarray): Array of derivatives of U (and optionally Q), starting with a column of ones.
        derivatives_description (list): Descriptions of each column in `derivatives`.
        P (int): Maximum polynomial degree for terms in Theta.
        data_description (list, optional): Descriptions of columns in `data`.

    Returns:
        Theta (ndarray): Feature matrix containing all polynomial and derivative combinations.
        descr (list): Descriptions of all columns in Theta.

    Raises:
        Exception: If dimensions of `data` and `derivatives` do not match, or if `data_description` 
                   has inconsistent length with `data`.
    """
    
    n,d = data.shape
    m, d2 = derivatives.shape
    if n != m: raise Exception('dimension error')
    if data_description is not None: 
        if len(data_description) != d: raise Exception('data descrption error')
    
    # Create a list of all polynomials in d variables up to degree P
    rhs_functions = {}
    f = lambda x, y : np.prod(np.power(list(x), list(y)))
    powers = []            
    for p in range(1,P+1):
            size = d + p - 1
            for indices in itertools.combinations(range(size), d-1):
                starts = [0] + [index+1 for index in indices]
                stops = indices + (size,)
                powers.append(tuple(map(operator.sub, stops, starts)))
    for power in powers: rhs_functions[power] = [lambda x, y = power: f(x,y), power]

    # First column of Theta is just ones.
    Theta = np.ones((n,1), dtype=np.complex64)
    descr = ['']
    
    # Add the derivaitves onto Theta
    for D in range(1,derivatives.shape[1]):
        Theta = np.hstack([Theta, derivatives[:,D].reshape(n,1)])
        descr.append(derivatives_description[D])
        
    # Add on derivatives times polynomials
    for D in range(derivatives.shape[1]):
        for k in rhs_functions.keys():
            func = rhs_functions[k][0]
            new_column = np.zeros((n,1), dtype=np.complex64)
            for i in range(n):
                new_column[i] = func(data[i,:])*derivatives[i,D]
            Theta = np.hstack([Theta, new_column])
            if data_description is None: descr.append(str(rhs_functions[k][1]) + derivatives_description[D])
            else:
                function_description = ''
                for j in range(d):
                    if rhs_functions[k][1][j] != 0:
                        if rhs_functions[k][1][j] == 1:
                            function_description = function_description + data_description[j]
                        else:
                            function_description = function_description + data_description[j] + '^' + str(rhs_functions[k][1][j])
                descr.append(function_description + derivatives_description[D])

    return Theta, descr


#####################################################################################################
#####################################################################################################


def build_linear_system(u, dt, dx, D = 3, P = 3,time_diff = 'poly',
                        space_diff = 'poly',lam_t = None,lam_x = None,
                        width_x = None,width_t = None, deg_x = 5,deg_t = None,sigma = 2):
    """
    Constructs the linear system for PDE regression by computing time derivatives, spatial derivatives,
    and assembling the feature matrix.

    Parameters:
        u (ndarray): Data to fit the PDE (2D array with dimensions: space x time).
        dt (float): Temporal grid spacing.
        dx (float): Spatial grid spacing.
        D (int, optional): Maximum derivative order to include in the feature matrix (default: 3).
        P (int, optional): Maximum polynomial degree of the solution to include (default: 3).
        time_diff (str, optional): Method to compute time derivatives. Options:
                                   'poly' (default), 'FD', 'FDconv', 'Tik'.
        space_diff (str, optional): Method to compute spatial derivatives. Options:
                                    'poly' (default), 'FD', 'FDconv', 'Fourier', 'Tik'.
        lam_t (float, optional): Regularization for time derivatives (default: 1/m).
        lam_x (float, optional): Regularization for spatial derivatives (default: 1/n).
        width_x (int, optional): Width of the polynomial interpolation or smoothing in the spatial domain.
        width_t (int, optional): Width of the polynomial interpolation in the temporal domain.
        deg_x (int, optional): Polynomial degree for spatial derivatives (default: 5).
        deg_t (int, optional): Polynomial degree for time derivatives (default: deg_x).
        sigma (float, optional): Standard deviation for Gaussian smoothing (applies to 'FDconv').

    Returns:
        ut (ndarray): Time derivative vector reshaped as a column.
        Theta (ndarray): Feature matrix containing all combinations of derivatives and polynomial terms.
        rhs_description (list): Descriptions of the terms in Theta.

    Raises:
        ValueError: If the dimensions of `u` do not match the expected sizes.

    Notes:
        - The `ut` is calculated based on the specified `time_diff` method.
        - Spatial derivatives up to order `D` are computed using the `space_diff` method.
        - The `Theta` matrix contains polynomial combinations of `u` and its derivatives.
    """


    n, m = u.shape

    if width_x == None: width_x = n//10
    if width_t == None: width_t = m//10
    if deg_t == None: deg_t = deg_x

    if time_diff == 'poly': 
        m2 = m-2*width_t
        offset_t = width_t
    else: 
        m2 = m
        offset_t = 0
    if space_diff == 'poly': 
        n2 = n-2*width_x
        offset_x = width_x
    else: 
        n2 = n
        offset_x = 0

    if lam_t == None: lam_t = 1.0/m
    if lam_x == None: lam_x = 1.0/n

    ########################
    # Time derivaitve for the LHS of the equation
    ########################
    ut = np.zeros((n2,m2), dtype=np.complex64)

    if time_diff == 'FDconv':
        Usmooth = np.zeros((n,m), dtype=np.complex64)
        # Smooth across x cross-sections
        for j in range(m):
            Usmooth[:,j] = ConvSmoother(u[:,j],width_t,sigma)
        # Now take finite differences
        for i in range(n2):
            ut[i,:] = FiniteDiff(Usmooth[i + offset_x,:],dt,1)

    elif time_diff == 'poly':
        T= np.linspace(0,(m-1)*dt,m)
        for i in range(n2):
            ut[i,:] = PolyDiff(u[i+offset_x,:],T,diff=1,width=width_t,deg=deg_t)[:,0]

    elif time_diff == 'Tik':
        for i in range(n2):
            ut[i,:] = TikhonovDiff(u[i + offset_x,:], dt, lam_t)

    else:
        for i in range(n2):
            ut[i,:] = FiniteDiff(u[i + offset_x,:],dt,1)
    
    ut = np.reshape(ut, (n2*m2,1), order='F')

    ########################
    # Form the rhs one column at a time, and record each one 
    ########################

    u2 = u[offset_x:n-offset_x,offset_t:m-offset_t]
    Theta = np.zeros((n2*m2, (D+1)*(P+1)), dtype=np.complex64)
    ux = np.zeros((n2,m2), dtype=np.complex64)
    rhs_description = ['' for i in range((D+1)*(P+1))]

    if space_diff == 'poly': 
        Du = {}
        for i in range(m2):
            Du[i] = PolyDiff(u[:,i+offset_t],np.linspace(0,(n-1)*dx,n),diff=D,width=width_x,deg=deg_x)
    if space_diff == 'Fourier': ik = 1j*np.fft.fftfreq(n)*n
        
    for d in range(D+1):

        if d > 0:
            for i in range(m2):
                if space_diff == 'Tik': ux[:,i] = TikhonovDiff(u[:,i+offset_t], dx, lam_x, d=d)
                elif space_diff == 'FDconv':
                    Usmooth = ConvSmoother(u[:,i+offset_t],width_x,sigma)
                    ux[:,i] = FiniteDiff(Usmooth,dx,d)
                elif space_diff == 'FD': ux[:,i] = FiniteDiff(u[:,i+offset_t],dx,d)
                elif space_diff == 'poly': ux[:,i] = Du[i][:,d-1]
                elif space_diff == 'Fourier': ux[:,i] = np.fft.ifft(ik**d*np.fft.fft(ux[:,i]))
        else: ux = np.ones((n2,m2), dtype=np.complex64) 
            
        for p in range(P+1):
            Theta[:, d*(P+1)+p] = np.reshape(np.multiply(ux, np.power(u2,p)), (n2*m2), order='F')

            if p == 1: rhs_description[d*(P+1)+p] = rhs_description[d*(P+1)+p]+'u'
            elif p>1: rhs_description[d*(P+1)+p] = rhs_description[d*(P+1)+p]+'u^' + str(p)
            if d > 0: rhs_description[d*(P+1)+p] = rhs_description[d*(P+1)+p]+\
                                                   'u_{' + ''.join(['x' for _ in range(d)]) + '}'

    return ut, Theta, rhs_description

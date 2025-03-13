import numpy as np


def TrainSTRidge(R, Ut, lam, d_tol, maxit = 25, STR_iters = 10, l0_penalty = None, normalize = 2, split = 0.8, print_best_tol = True):
    """
    Trains a sparse predictor using Sequential Threshold Ridge Regression (STRidge).

    Parameters:
        R (ndarray): Feature matrix (n_samples x n_features).
        Ut (ndarray): Target values (n_samples x 1).
        lam (float): Ridge regression regularization parameter.
        d_tol (float): Initial tolerance for feature selection.
        maxit (int, optional): Maximum number of iterations for tolerance adjustment (default: 25).
        STR_iters (int, optional): Number of STRidge iterations per tolerance adjustment (default: 10).
        l0_penalty (float, optional): Penalty for the L0 norm to encourage sparsity (default: 0.001 * condition number of R).
        normalize (int, optional): Normalization option for STRidge (default: 2).
        split (float, optional): Fraction of data to use for training (default: 0.8).
        print_best_tol (bool, optional): Whether to print the optimal tolerance value (default: False).

    Returns:
        ndarray: Optimal coefficient vector with sparse non-zero entries.
    """

    # Split data into 80% training and 20% test, then search for the best tolderance.
    np.random.seed(0) # for consistancy
    n,_ = R.shape
    train = np.random.choice(n, int(n*split), replace = False)
    test = [i for i in np.arange(n) if i not in train]
    TrainR = R[train,:]
    TestR = R[test,:]
    TrainY = Ut[train,:]
    TestY = Ut[test,:]
    D = TrainR.shape[1]       

    # Set up the initial tolerance and l0 penalty
    d_tol = float(d_tol)
    tol = d_tol
    if l0_penalty == None: l0_penalty = 0.001*np.linalg.cond(R)

    # Get the standard least squares estimator
    w = np.zeros((D,1))
    w_best = np.linalg.lstsq(TrainR, TrainY,rcond=None)[0]
    err_best = np.linalg.norm(TestY - TestR.dot(w_best), 2) + l0_penalty*np.count_nonzero(w_best)
    tol_best = 0

    # Now increase tolerance until test performance decreases
    for iter in range(maxit):

        # Get a set of coefficients and error
        w = STRidge(TrainR,TrainY,lam,STR_iters,tol,normalize = normalize)
        err = np.linalg.norm(TestY - TestR.dot(w), 2) + l0_penalty*np.count_nonzero(w)

        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            w_best = w
            tol_best = tol
            tol = tol + d_tol

        else:
            tol = max([0,tol - 2*d_tol])
            d_tol  = 2*d_tol / (maxit - iter)
            tol = tol + d_tol

    if print_best_tol: print("Optimal tolerance:", tol_best)

    return w_best


def STRidge(X0, y, lam, maxit, tol, normalize=2, print_results=True):
    """
    Sequential Threshold Ridge Regression algorithm.
    """
    n, d = X0.shape
    X = np.zeros((n, d), dtype=np.complex64)

    # Normalize data
    if normalize != 0:
        Mreg = np.zeros((d, 1))
        for i in range(d):
            Mreg[i] = 1.0 / (np.linalg.norm(X0[:, i], normalize))
            X[:, i] = Mreg[i] * X0[:, i]
    else:
        X = X0

    # Initial Ridge Estimate
    if lam != 0:
        w = np.linalg.lstsq(X.T.dot(X) + lam * np.eye(d), X.T.dot(y), rcond=None)[0]
    else:
        w = np.linalg.lstsq(X, y, rcond=None)[0]

    num_relevant = d
    biginds = np.where(np.abs(w) > tol)[0]

    for j in range(maxit):
        # Identify small coefficients to be removed
        smallinds = np.where(np.abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]

        # Stop if no change in sparsity pattern
        if len(new_biginds) == num_relevant:
            break
        else:
            num_relevant = len(new_biginds)

        # Handle case where all coefficients are eliminated
        if len(new_biginds) == 0:
            if j == 0:
                if print_results:
                    print("Tolerance too high; all coefficients eliminated.")
                return np.zeros_like(w)
            else:
                break

        biginds = new_biginds

        # Update coefficients for non-zero terms
        w[smallinds] = 0
        if len(biginds) > 0:
            if lam != 0:
                w[biginds] = np.linalg.lstsq(
                    X[:, biginds].T.dot(X[:, biginds]) + lam * np.eye(len(biginds)),
                    X[:, biginds].T.dot(y),
                    rcond=None,
                )[0]
            else:
                w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]

    if normalize != 0:
        return np.multiply(Mreg, w)
    else:
        return w
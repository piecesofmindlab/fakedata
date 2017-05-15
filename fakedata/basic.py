"""Basic module of data generation functions."""
import numpy as _np
from scipy.stats import zscore
from scipy import linalg as la
import os

boilerplate_file = os.path.join(os.path.split(__file__)[0], 'boilerplate.py')

# Useful: http://docs.scipy.org/doc/scipy/reference/tutorial/linalg.html

def from_x(X, wts, signal_corr=1.0, do_zscore=True):
    """Make fake data from some design matrix + weights on each channel in it.

    Builds a fake data matrix from some source matrices. Computes `X.dot(wts)`
    and adds noise such that the output (`Y`) is correlated with X.dot(wts) by
    (approximately) `signal_corr`.

    Size, std, sparsity of output are all determined by X, wts.

    Parameters
    ----------
    X : array-like, t x chan
        Design matrix for fake data. Each column in `X` is a feature / factor / channel that
        influences the final data according to the weight assigned to that channel by `wts`
    wts : array-like, chan x n
        Weights for each of <chan> columns in  X. If wts is 1D, returns a single Y;
        if n > 1, returns fake data for n samples (/ voxels / repeats)
    signal_corr : scalar or vector | 1.0-> SCALAR ONLY FOR NOW??
        Desired signal level (correlation with perfect X.dot(w) matrix) If a vector is provided,
        it should be the same length as n (2nd dimension of wts) to specify
    do_zscore : bool | True

    Returns
    -------
    Y : array-like, t x n

    Notes
    -----
    TO DO:
    - Return class instance instead of just matrix...
    - store original X, w, signal_corr, sparsity, etc (???)

    """
    ## -- Input handling -- ##
    if not isinstance(wts,_np.ndarray):
        wts = _np.array(wts)
    if wts.ndim==1:
        wts = wts[:,None] # make into a 2D array
    t,nCh1 = X.shape
    nCh2,nVox = wts.shape
    ## -- Add noise -- ##
    signal = X.dot(wts) # if signal_corr is a vector, does this work??
    sig_std = _np.std(signal,axis=0)
    noise = _np.random.randn(t,nVox)*sig_std # if signal_corr is a vector, does this work??
    # From http://www.sitmo.com/article/generating-correlated-random-numbers/
    dat = signal*signal_corr+noise*_np.sqrt(1-signal_corr**2)
    ## -- Optionally zscore -- ##
    if do_zscore:
        dat = zscore(dat)
    return dat

def from_matrix(data,sz,signal_corr=1.0,corr_falloff=None,do_zscore=True):
    """Generate fake data such that columns share variance w/ input matrix

    Useful for generating dependent design matrices for fake regressions.

    Parameters
    ----------
    data : array-like, m x n
        Input matrix with which to correlate generated fake data

    sz : tuple (x,y)
        (x,y) - number of columns in x and y variables
        x should always be >= y
    nt : scalar
        length of 1st dim (time), same for both
    corr_falloff : vector (1D array) of monotonically decreasing values (1->0)
        allows some channels to be less correlated than others. Does not have
        to go all the way to zero. Default = all ones.
    do_zscore = bool | True
        whether to z score all columns of output matrix

    Notes
    -----
    []
    """
    ## -- Input handling -- ##
    m_in, n_in = data.shape
    m_out, n_out = sz
    if n_out > n_in or m_out > m_in:
        raise Exception('Output shape must be <= shape of input data in all dimensions')
    if corr_falloff is None:
        corr_falloff = _np.ones((n_out))
    # Singular value decomposition to get underlying structure of input data matrix
    U, S, Vt = la.svd(data)
    sn = len(S)
    # U is m_in x m_in (= m_out x m_out)
    # Sm is sn x xn (sn = min(m_in,n_in))
    # Vt is n_in x n_in (!= n_out x n_out) -> Needs to be n_out x n_out
    # Shuffle columns
    Vtx = _np.random.randn(n_out, n_out)
    Sm = _np.diag(S)
    if sn < m_out:
        Sm = _np.vstack((Sm, _np.zeros((m_out-sn, sn))))
    if sn < n_out:
        Sm = _np.hstack((Sm, _np.zeros((sn, n_out-sn))))
    # Create base signal
    signal = U.dot(Sm).dot(Vtx)
    # Add noise
    sig_std = _np.std(signal,axis=0)
    noise = _np.random.randn(m_out,n_out)*sig_std 
    # if signal_corr is a vector, does this work??
    # From http://www.sitmo.com/article/generating-correlated-random-numbers/
    X = signal*signal_corr+noise*_np.sqrt(1-signal_corr**2)
    ## -- Optionally zscore -- ##
    if do_zscore:
        X = zscore(X)
    return X


def from_svd(S,m,n,U=None,Vt=None,do_zscore=True):
    """Generate matrix based on U,S,Vt matrices as in Singular Value Decomposition

    This is a useful way to generate a random matrix with some degree of internal structure (i.e.,
    some amount of correlation between columns or rows). See
    https://en.wikipedia.org/wiki/Singular_value_decomposition for useful discussion of SVD.

    Parameters
    ----------
    S : Singular values for matrix
    m : scalar
        number of rows of matrix
    n : scalar
        number of columns of matrix
    U : array-like | None
        Left singular vectors, or eigenvectors for row covariance for matrix. Should be a unitary matrix.
        If `U` is None, a random unitary matrix is generated.
    Vt : array-like | None
        Right singular vectors, or eigenvectors for column covariance for matrix. Should be a unitary matrix.
        If `Vt` is None, a random unitary matrix is generated.

    Returns
    -------
    dat : array-like
        m x n array (U*S*Vt')

    Notes
    -----
    For more sophisticated random matrix generation, check out QuTiP (Quantum Toolbox in Python, https://github.com/qutip)
    Are there ways to add more constraints here?? (Specific sparsity? from_x like behavior?)
    Possibly use this function in from_matrix??
    """
    S = _np.diag(S)
    if m>n:
        S = _np.pad(S,[(0,m-n),(0,0)],mode='constant')
    elif n>m:
        S = _np.pad(S,[(0,0),(0,n-m)],mode='constant')
    if U is None:
        U = rand_unitary_matrix(m)
    if Vt is None:
        Vt = rand_unitary_matrix(n)
    M = U.dot(S).dot(Vt.T)
    if do_zscore:
        M = zscore(M)
    return M

def get_correlated_signal(x,signal_corr=0.5,noise=None,do_zscore=False):
    """Generate a vector or matrix with the columns correlated with the input by (approximately) `signal_corr`

    Method taken from http://www.sitmo.com/article/generating-correlated-random-numbers/

    Parameters
    ----------
    x : vector or array
        Starting point for signal generation. output will be correlated with the columns of x.
    signal_corr : scalar or vector
        The desired correlation between output and x. If a vector of values is provided, its 
        length should match the number of columns in x. 
    noise : vector or array
        Specific noise matrix to use to reduce correlations between x and output. Useful if you 
        want to add correlated noise to different signals. 
    do_zscore : bool
        Whether to z-score the output or not

    Returns
    -------
    y : vector or array
        data the same size as x that is (approximately) correlated as `signal_corr` specifies.
    """
    if noise is None:
        sig_std = _np.nanstd(x,axis=0)
        noise = _np.random.randn(*x.shape)*sig_std 
    # if signal_corr is a vector, does this work??
    y = x*signal_corr+noise*_np.sqrt(1-signal_corr**2)
    if do_zscore:
        y = zscore(y)
    return y

def rand_unitary_matrix(N):
    """Get a random unitary matrix

    A unitary matrix is an orthogonal matrix, such that U.T.dot(U) = U.dot(U.T) = Identity 
    """
    # Start with random
    U = _np.random.randn(N,N)
    # Enforce 
    U = U.dot(la.inv(la.sqrtm(U.T.dot(U))))
    # This works. Off-diagonal are <10e-14, diagonal values are ~1 . To show:
    # print("U orthogonality test:") 
    # print(U.T.dot(U))
    return U

def add_noise_cols(X,n):
    """Add noise columns to a matrix.

    Add `n` columns of noise channels to the right of array `X`
    
    Parameters
    ----------
    X : array-like
        Original data
    n : scalar
        Number of noise columns to add
    
    TO DO:
    Add parameters to allow addition columns correlated with extant values of X
    Look into _np.pad (with padding function) to do this...
    """
    return _np.hstack((X,_np.random.randn(X.shape[0],n)))


"""
Module to allow easy creation of fake data with desired parameters.

Created by ML 2014.08
"""

from datagen import datagen
import numpy as _np
from scipy.stats import zscore as _zs
from scipy import linalg as la
import os
#import fmri #? 
#import stats #? 

boilerplate_file = os.path.join(os.path.split(__file__)[0],'boilerplate.py')

def from_x(X,wts,signal_corr=1.0,do_zscore=True):
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
		wts.shape += (1,) # make into a 2D array
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
		dat = _zs(dat)
	return dat

def from_matrix(data,sz,signal_corr=None,shuf_channels=False,do_zscore=True,corr_falloff=None):
	"""Generate fake data such that some columns share variance w/ input matrix
	
	<Blah>

	Parameters
	----------
	data : array-like, m x n
		Input matrix with which to correlate generated fake data

	ncols : tuple of ints | (1,1)
		(x,y) - number of columns in x and y variables
		x should always be >= y
	nt : scalar
		length of 1st dim (time), same for both
	rho : float from 0-1
		(mean) correlation btw x and y
		Y = rho*Y + sqrt(1-rho**2)*noise
	corr_falloff : vector (1D array) of monotonically decreasing values (1->0)
		allows some channels to be less correlated than others. Does not have
		to go all the way to zero. Default = all ones.
	do_zscore = bool | True
		whether to z score all columns of output matrix
	Notes
	-----
	Currently broken. WIP.
	"""
	## -- Input handling -- ##
	# un-tuple (?)
	m_in,n_in = data.shape
	m_out,n_out = sz
	if n_out>n_in or m_out>m_in:
		raise Exception('Output shape must be <= shape of input data in all dimensions')
	raise NotImplementedError('This is currently borked right here. Need to define size of matrix wrt inputs.')
	if corr_falloff is None:
		corr_falloff = _np.ones((n))
	# Make common data
	common = _np.random.randn(nt,n)
	# Create X
	X = add_noise_cols(common,m-n)# + m
	# Optionally Shuffle columns (important?)
	if shuf_channels:
		idx = arange(X.shape[1])
		_np.random.shuffle(idx)
		X = X[:,idx]
	# Create Y, combine w/ max signal fraction = sig_fract, falloff as defined
	n1 = _np.random.randn(nt,m) # noise by channel (controlled by corr_falloff)
	n2 = _np.random.randn(nt,n) # noise for whole matrix (overall SNR)
	# Channel-specific correlation reduction with by-channel noise
	Y = common.dot(_np.diag(corr_falloff)) + n1.dot(_np.diag(1-corr_falloff))
	# Whole matrix signal reduction w/ general noise
	Y = Y*rho+n2*_np.sqrt(1-rho)
	# Re-center
	Y -= Y.mean(0)
	X -= X.mean(0)
	Y /= Y.std(0)
	X /= X.std(0)
	return X,Y

def from_svd(S,m,n,U=None,V=None,do_zscore=True):
	"""Generate matrix based on U,S,V matrices as in Singular Value Decomposition

	This is a useful way to generate a random matrix with some degree of internal structure (i.e., 
	some amount of correlation between columns or rows). 

	Parameters
	----------
	S : Singular values for matrix
	m : scalar
		number of rows of matrix
	n : scalar
		number of columns of matrix
	U : array-like | None
		Eigenvectors for row covariance for matrix. If `U`=None, a random ( N~(0,1) ) matrix is used.
	V : array-like | None
		Eigenvectors for column covariance for matrix. If `V`=None, a random ( N~(0,1) ) matrix is used.

	Returns
	-------
	dat : array-like
		m x n array (U*S*V')

	Notes
	-----
	Are there ways to add more constraints here?? (Specific sparsity? from_x like behavior?)
	Possibly use this function in from_matrix??
	"""
	S = _np.diag(S)
	if m>n:
		S = _np.pad(S,[(0,m-n),(0,0)],mode='constant')
	elif n>m:
		S = _np.pad(S,[(0,0),(0,n-m)],mode='constant')
	if U is None:
		# Need orthogonal matrix, such that U.T.dot(U) = U.dot(U.T) = Identity   sqrtm, inv
		U = _np.random.randn(m,m)
		U = U.dot(la.inv(la.sqrtm(U.T.dot(U))))
		#print("U orthogonality test:") # works. off-diagonal are <10e-14, diagonal values are ~1
		#print(U.T.dot(U))
	if V is None:
		V = _np.random.randn(n,n)
		V = V.dot(la.inv(la.sqrtm(V.T.dot(V))))
		#print("V orthogonality test:")
		#print(V.T.dot(V))
	M = U.dot(S).dot(V.T)
	if do_zscore:
		M = _zs(M)
	return M

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


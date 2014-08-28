"""
Class to generate fake data

IS THIS EVEN NECESSARY AS A CLASS?? PROBABLY NOT.

dg = datagen('randn')
dg = datagen.from_x(some_design_matrix,weights)
# Store dg.x, dg.w, dg.sz, dg.signal_corr
"""
import numpy as np # avoid importing me a bunch of times? 
from .stats import zs

class datagen(object): # inherit from np.ndarray? Seems complicated.
	def __init__(self,d='randn',sz=(100,30),std=None,means=None,sparsity=None):
		# Define matrix .d based on inputs
		# without inputs, simply make random data
		# optionally fix mean, std, whatever? 
		if d.lower() in ('randn','rand','randint'):
			rfn = np.random.__getattribute__(d.lower())
			self.d = rfn(*sz) # Mean? STD? Other? 
		else:
			# check for numeric d
			self.d = d
		#if not std is None:

	@classmethod
	def from_x(cls,X,wts,signal_corr=1.0,do_zscore=True):
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
		if not isinstance(wts,np.ndarray):
			wts = np.array(wts)
		if wts.ndims==1:
			wts.shape += (1,) # make into a 2D array
		t,nCh1 = X.shape
		nCh2,nVox = wts.shape
		## -- Add noise -- ##
		signal = X.dot(wts)*signal_corr # if signal_corr is a vector, does this work??
		noise = np.random.randn(t,nVox)*np.sqrt(1-signal_corr) # if signal_corr is a vector, does this work??
		dat = signal+noise
		## -- Optionally zscore -- ##
		if do_zscore:
			dat = zs(dat)
		# Call __init__? Call __new__?
		return dat

	@classmethod
	def from_matrix(cls,data,signal_corr=None):
		"""Generate fake data based on another whole matrix.
		"""

		#def make_data_pairs2(nt,ncols=(1,1),rho=.5,corr_falloff=None,shuf_channels=False):
		"""Generates fake data w/ some columns sharing common variance (same data + noise)
		
		Parameters
		----------
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
		
		"""
		## -- Input handling -- ##
		# un-tuple (?)
		nx,ny = ncols
		if ny>nx:
			raise Exception('NOOOOO! x must be bigger or both must be equal!')
		if corr_falloff is None:
			corr_falloff = np.ones((ny))
		# Make common data
		common = np.random.randn(nt,ny)
		# Create X
		X = add_noise_chan(common,nx-ny)# + nX
		# Optionally Shuffle columns (important?)
		if shuf_channels:
			idx = arange(X.shape[1])
			np.random.shuffle(idx)
			X = X[:,idx]
		# Create Y, combine w/ max signal fraction = sig_fract, falloff as defined
		nY1 = np.random.randn(nt,nx) # noise by channel (controlled by corr_falloff)
		nY2 = np.random.randn(nt,ny) # noise for whole matrix (overall SNR)
		# Channel-specific correlation reduction with by-channel noise
		Y = common.dot(np.diag(corr_falloff)) + nY1.dot(np.diag(1-corr_falloff))
		# Whole matrix signal reduction w/ general noise
		Y = Y*rho+nY2*np.sqrt(1-rho)
		# Re-center
		Y -= Y.mean(0)
		X -= X.mean(0)
		Y /= Y.std(0)
		X /= X.std(0)
		return X,Y

#t = np.linspace(1,2,np.minimum(mt,n))
#S = t**-30. # Enough to have some visible structure
def from_svd(S,m,n,U=None,V=None):
	"""Generate matrix based on U,S,V matrices as in Singular Value Decomposition

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
	S = np.diag(S)
	if m>n:
		S = np.pad(S,[(0,m-n),(0,0)],mode='constant')
	elif n>m:
		S = np.pad(S,[(0,0),(0,n-m)],mode='constant')
	if U is None:
		U = randn(m,m)
	if V is None:
		V = randn(n,n)
	M = (U.dot(S)).dot(V.T)
	return M

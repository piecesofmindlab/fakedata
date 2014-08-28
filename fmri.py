"""
Data generation useful for neuroscience & psychology experimental protocols 
(Originally intended for simulating fMRI research designs.)

Mostly very simple stuff.

TO DO: 

- HRF function
- generate design matrices for n conditions with minimization of correlations between condition onsets
- generate design matrices w/ jitter of onsets? 

(All the matlab shit that went into previous work...)

"""

	


def rep_n(X,n):
	return np.tile(X,(n,1)).T.flatten()


def rand_design_matrix(sz,ct=None,do_ones=True):
	"""Makes non-overlapping binary design matrix for two conditions.
	
	Parameters
	----------
	sz : tuple, (nTimePoints,nConds)
		Size of design matrix, time x conditions
	ct : count of trials per condition 
		Optionally, specify how many trials per condition. Otherwise, divide
		the number of time points evenly into all conditions (plus a blank condition)
		
	"""
	t,c = sz
	if ct is None:
		ct = np.floor(t/float(c+1))
		ct = [ct]*c
	cnums = np.hstack([np.ones(n)*cnum for n,cnum in zip(ct,np.arange(1,c+1))]+[np.zeros(t-np.sum(ct))])
	np.random.shuffle(cnums)
	DM = np.array([cnums==cond for cond in np.arange(1,c+1)]).T
	DM = np.hstack((np.ones((DM.shape[0],1)),np.cast['float'](DM)))
	return DM

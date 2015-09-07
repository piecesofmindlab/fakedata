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
import numpy as np
import scipy.stats
import warnings

def hrf(shape='twogamma', sf=1, pttp=5, nttp=15, pnr=6, ons=0, pdsp=1, ndsp=1, s=None, d=0):
	"""Create canonical hemodynamic response filter

	Parameters
	----------
	shape : 
		HRF general shape {'twogamma' [, 'boynton']}
	sf : 
		HRF sample frequency (default: 1s/16, OK: [1e-3 .. 5])
	pttp : 
		time to positive (response) peak (default: 5 secs)
	nttp : 
		time to negative (undershoot) peak (default: 15 secs)
	pnr : 
		pos-to-neg ratio (default: 6, OK: [1 .. Inf])
	ons : 
		onset of the HRF (default: 0 secs, OK: [-5 .. 5])
	pdsp : 
		dispersion of positive gamma PDF (default: 1)
	ndsp : 
		dispersion of negative gamma PDF (default: 1)
	s : 
		sampling range (default: [0, ons + 2 * (nttp + 1)])
	d : 
		derivatives (default: 0)
	
	Returns
	-------
	h : HRF function given within [0 .. onset + 2*nttp]
	s : HRF sample points
	
	Notes
	-----
	The pttp and nttp parameters are increased by 1 before given
	as parameters into the scipy.stats.gamma.pdf function (which is a property
	of the gamma PDF!)

	Converted to python from BVQXtools 
	Version:  v0.7f
	Build:    8110521
	Date:     Nov-05 2008, 9:00 PM CET
	Author:   Jochen Weber, SCAN Unit, Columbia University, NYC, NY, USA
	URL/Info: http://wiki.brainvoyager.com/BVQXtools
	"""

	# Input checks
	if sf > 5:
	    sf = 1/16
	elif sf < 0.001:
	    sf = 0.001
	if not shape.lower() in ('twogamma', 'boynton'):
		warnings.warn('Shape can only be "twogamma" or "boynton"')
		shape = 'twogamma'
	if s is None:
	    s = np.arange(0,(ons + 2 * (nttp + 1)), sf) - ons
	else:
	    s = np.arange(np.min(s),np.max(s),sf) - ons;

	# computation (according to shape)
	h = np.zeros((len(s), d + 1));
	if shape.lower()=='boynton':
	    # boynton (single-gamma) HRF
		h[:,0] = scipy.stats.gamma.pdf(s, pttp + 1, pdsp)
		if d > 0:
			raise NotImplementedError('Still WIP - code just needs porting.')
			"""# Matlab code, partly translated:
			h[:, 1] = h[:, 1] - scipy.stats.gamma.pdf(s + 1, pttp + 1, pdsp);
			hi = find(h[:, 1] ~= 0);
			h[hi, 1] = h[hi, 1] - ((pinv(h[hi, 1]' * h[hi, 1]) * h[hi, 1]' * h[hi, 1])' * h[hi, 1]')';
			if d > 1:
				h[:,2] = h[:, 1] - scipy.stats.gamma.pdf(s, pttp + 1, pdsp / 1.01);
				hi = find(h[:,2] ~= 0);
				h[hi,2] = h[hi,2] - ((pinv(h[hi, [1, 2]).T * h[hi, [1, 2])) * h[hi, [1, 2]).T * h[hi,2]).T * h[hi, [1, 2]).T).T;"""
	elif shape.lower()=='twogamma':
		gpos = scipy.stats.gamma.pdf(s, pttp + 1, pdsp)
		gneg = scipy.stats.gamma.pdf(s, nttp + 1, ndsp) / pnr
		h[:,0] =  gpos-gneg	            
		if d > 0:
			raise NotImplementedError('Still WIP. Sorting through matlab multiplications is annoying.')
			"""gpos = scipy.stats.gamma.pdf(s-1, pttp + 1, pdsp)
			gneg = scipy.stats.gamma.pdf(s-1, nttp + 1, ndsp) / pnr
			h[:, 1] = h[:, 0] - gpos - gneg
			hi = h[:, 1] != 0
			h[hi, 1] = h[hi, 0] - ((np.linalg.pinv(h[hi, 0].T * h[hi, 0]) * h[hi, 0].T * h[hi, 0]).T * h[hi, 1].T).T
			if d > 1:
				h[:,2] = (h[:, 1] - (scipy.stats.gamma.pdf(s, (pttp + 1) / 1.01, pdsp / 1.01) - scipy.stats.gamma.pdf(s, nttp + 1, ndsp) / pnr)) / 0.01;
				hi = h[:,2] != 0
				h[hi,2] = h[hi,2] - ((pinv(h[hi, [1, 2]).T * h[hi, [1, 2])) * h[hi, [1, 2]).T * h[hi,2]).T * h[hi, [1, 2]).T).T;
			"""
	# normalize for convolution
	if d < 1:
		h /= np.sum(h)
	else:
		h /= np.tile(np.sqrt(np.sum(h**2)), h.shape[0], 1)
	return s,h

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

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

def hrf(shape='twogamma', sf=1, pttp=5, nttp=15, pnr=6, ons=0, pdsp=1, ndsp=1, s=None, d=None):
	"""Return canonical hrf function. 

	Essentially copied from Jochen Weber's hrf.m from NeuroElf
	(http://neuroelf.net/) See below for copyright

	Parameters
	----------
	shape : {'twogamma' | 'boynton'}
		HRF general shape; default: 'twogamma'
	sf : scalar
		HRF sample frequency (default: 1 hz, min=1e-3,max=5)
	pttp : scalar 
		time to positive (response) peak (default: 5 secs)
	nttp : scalar
		time to negative (undershoot) peak (default: 15 secs)
	pnr : scalar
		pos-to-neg ratio (default: 6, OK: [1 .. Inf])
	ons : scalar
		onset of the HRF (default: 0 secs, OK: [-5 .. 5])
	pdsp : scalar 
		dispersion of positive gamma PDF (default: 1)
	ndsp : scalar
		dispersion of negative gamma PDF (default: 1)
	s : None | tuple
		sampling range, (start, end); default (None) gives
		(0, ons + 2 * (nttp + 1))
	d : scalar
		derivatives (default: 0) [NOT IMPLEMENTED YET]
	
	Returns
	--------
	h : 1D array
		HRF function given within [0 .. onset + 2*nttp]
	s : 1D array
		HRF sample points
	
	Notes
	-----
	The pttp and nttp parameters are increased by 1 before given
	as parameters into the gammapdf function (which is a property
	of the gamma PDF!)
	
	Derivatives (d) are NOT implemented yet
	From matlab function, copyright stuff:
	Version:  v0.9a
	Build:	10051710
	Date:	 May-17 2010, 10:48 AM EST
	Author:   Jochen Weber, SCAN Unit, Columbia University, NYC, NY, USA
	URL/Info: http://neuroelf.net/

	Copyright (c) 2010, Jochen Weber
	All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are met:
		* Redistributions of source code must retain the above copyright
		  notice, this list of conditions and the following disclaimer.
		* Redistributions in binary form must reproduce the above copyright
		  notice, this list of conditions and the following disclaimer in the
		  documentation and/or other materials provided with the distribution.
		* Neither the name of Columbia University nor the
		  names of its contributors may be used to endorse or promote products
		  derived from this software without specific prior written permission.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
	DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
	"""
	from scipy.stats import gamma
	# (Deleted most argument checks in hrf.m, mostly unnecessary with defaults)
	shape = shape.lower()
	if not shape in ('twogamma','boynton'):
		raise ValueError("shape input should be 'twogamma' or 'boynton'")
	if s is None:
		s = np.arange(0,ons+2*(nttp+1),sf)
	elif isinstance(s,(list,tuple)):
		s = np.arange(s[0],s[1],sf)-ons
	if not d is None:
		raise NotImplementedError("HRF derivatives are not implemented yet.")
	# computation (according to shape)
	#h = np.zeros((len(s), d + 1)) # for derivatives, eventually
	if shape=='boynton':
		# boynton (single-gamma) HRF
		h[:,0] = gamma.pdf(s,pttp+1,pdsp)
		# if d > 0:
		# 	#h(:, 2) = h(:, 1) - gammapdf(s(:) + 1, pttp + 1, pdsp);
		# 	h[:,1] = h[:, 0] - gamma.pdf(s + 1, pttp + 1, pdsp);
		# 	hi = find(h(:, 2) ~= 0);
		# 	h(hi, 2) = h(hi, 2) - ((pinv(h(hi, 1)' * h(hi, 1)) * h(hi, 1)' * h(hi, 2))' * h(hi, 1)')';
		# 	if d > 1
		# 		h(:, 3) = h(:, 1) - gammapdf(s(:), pttp + 1, pdsp / 1.01);
		# 		hi = find(h(:, 3) ~= 0);
		# 		h(hi, 3) = h(hi, 3) - ((pinv(h(hi, [1, 2])' * h(hi, [1, 2])) * ...
		# 			h(hi, [1, 2])' * h(hi, 3))' * h(hi, [1, 2])')';

	elif shape=='twogamma':
		# two-gamma HRF
		h = gamma.pdf(s, pttp + 1, pdsp) - gamma.pdf(s, nttp + 1, ndsp) / pnr
		#h(:, 1) = gammapdf(s(:), pttp + 1, pdsp) - ...
		#	gammapdf(s(:), nttp + 1, ndsp) / pnr;
		# if d > 0
		# 	h(:, 2) = h(:, 1) - (gammapdf(s(:) - 1, pttp + 1, pdsp) - ...
		# 		gammapdf(s(:) - 1, nttp + 1, ndsp) / pnr);
		# 	hi = find(h(:, 2) ~= 0);
		# 	h(hi, 2) = h(hi, 2) - ((pinv(h(hi, 1)' * h(hi, 1)) * h(hi, 1)' * h(hi, 2))' * h(hi, 1)')';
		# 	if d > 1
		# 		h(:, 3) = (h(:, 1) - (gammapdf(s(:), (pttp + 1) / 1.01, pdsp / 1.01) - ...
		# 			gammapdf(s(:), nttp + 1, ndsp) / pnr)) ./ 0.01;
		# 		hi = find(h(:, 3) ~= 0);
		# 		h(hi, 3) = h(hi, 3) - ((pinv(h(hi, [1, 2])' * h(hi, [1, 2])) * ...
		# 			h(hi, [1, 2])' * h(hi, 3))' * h(hi, [1, 2])')';
	# normalize for convolution
	#if d < 1:
	#	h = h ./ sum(h);
	#else:
	#	h = h ./ repmat(sqrt(sum(h .* h)), size(h, 1), 1);
	h /= np.sum(h)
	return h,s
"""
Boilerplate fake data creation code.
"""


####################################################################################
## -- 						Basics for data set creation 					   -- ##
####################################################################################
# Parameters
n_wts = n = 100
n_tps_trn = mt = 200
n_tps_val = mv = 90
n_vox = 10 # voxels or repeats 
snr_t = 0.8 # np.random.randn(n_vox)*.1+0.5 # Correlation between true 
snr_v = 0.8 # np.random.randn(n_vox)*.1+0.7 # Correlation between true 
# True influences on signal - independent
Xtrn = np.random.randn(n_tps_trn,n_wts)
Xval = np.random.randn(n_tps_val,n_wts)
# True weights
w = np.random.randn(n_wts,n_vox) 
# Signals
Ytrn = fd.from_x(Xtrn,w,snr_t,do_zscore=True) 
Yval = fd.from_x(Xval,w,snr_v,do_zscore=True)

########################################################################################
## -- Parameters for creation of fake data based on partly-collinear design matrix -- ##
########################################################################################
ccdist = np.random.randn(n_vox)*.1+.5 # distribution of signal over voxels
# Parameters for creation of partly-collinear design matrix
n_wts_dm = 1
cc_dm = np.random.randn(n_wts)*.1+0.5**0.5
# Create a correlated design matrix
x = zscore(np.random.randn(n_tps_t,n_wts_dm),axis=0)
xw = np.ones((n_wts_dm,n_wts))
Xtrn = fd.from_x(x,xw,signal_corr=cc_dm,do_zscore=True)

# Ytrn = ...
# Yval = ...

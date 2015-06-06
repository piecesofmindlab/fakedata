"""
Boilerplate fake data creation code.
"""

########################################################################################
## -- Parameters for creation of fake data based on partly-collinear design matrix -- ##
########################################################################################
n_wts = 300
n_tps_t = 350
n_tps_v = 90
n_vox = 100
ccdist = np.random.randn(n_vox)*.1+.5 # distribution of signal over voxels
# Parameters for creation of partly-collinear design matrix
n_wts_dm = 1
cc_dm = np.random.randn(n_wts)*.1+0.5**0.5
# Create a correlated design matrix
x = zscore(np.random.randn(n_tps_t,n_wts_dm),axis=0)
xw = np.ones((n_wts_dm,n_wts))
Xt = fd.from_x(x,xw,signal_corr=cc_dm,do_zscore=True)

####################################################################################
## -- Define a data set of random noise, with more regressors than data points -- ##
####################################################################################
# Parameters
n_regressors = n = 100
n_data_points_trn = mt = 200
n_data_points_val = mv = 90
n_vox = 10 # voxels or repeats - for now, 1
snr_t = .9 # np.random.randn(n_vox)*.1+0.5 # Correlation between true 
snr_v = 1 # np.random.randn(n_vox)*.1+0.7 # Correlation between true 
# True influences on signal - independent
Xtrn = np.random.randn(n_data_points_trn,n_regressors)
Xval = np.random.randn(n_data_points_val,n_regressors)
# True weights
w = np.random.randn(n_regressors,n_vox) 
# Signals
Ytrn = fd.from_x(Xtrn,w,snr_t,do_zscore=False) 
Yval = fd.from_x(Xval,w,snr_v,do_zscore=False)
print('Training/Validation SNR (r) = %0.02f/%0.02f'%(snr_t,snr_v))
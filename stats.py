"""
Useful statistics functions for fake data.

MOVE ME TO AN INDEPENDENT MODULE TO BE SHARED ACROSS MULTIPLE MODULES.

"""

"""
NAME SHIT IN HERE CONSISTENTLY W/ OTHER MODULES (mlSTRF, regression, etc)

OR BETTER STILL: Make one goddamn stats addition module that is general to ALL THE FUKCING MODULES.
"""

#def zs(X):
#    return (X-ml.nanmean(X,axis=0))/ml.nanstd(X,axis=0)
from scipy.stats import zscore as zs

def ccMatrix(x,y,dof=0): 
    """
    Efficient computation of correlation between all COLUMNS 
    of two matrices x and y
    """
    x = (x-np.mean(x,0))/np.std(x,0,ddof=dof)
    y = (y-np.mean(y,0))/np.std(y,0,ddof=dof)
    return np.mean(x*y,0)
    #return np.nansum(xy,0)/nancount(xy,0)

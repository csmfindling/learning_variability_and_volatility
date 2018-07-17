# from rpy2.robjects.packages import importr
# utils = importr("utils")
# utils.install_packages('randtoolbox')

from rpy2.robjects.packages import importr
import numpy as np

rtb = importr("randtoolbox")

N     = 10
ndim  = 2
sobol = np.array(rtb.sobol(N,dim=ndim))

import pickle
pickle.dump(sobol, open('sobol_{0}_{1}.pkl'.format(N, ndim), 'wb'))
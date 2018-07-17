import cython
import numpy as np
cimport numpy as np

cdef extern from "smc.hpp" namespace "smc":
    double get_marg_lkd(int nS, int nC, int* stim_seq, int* act_seq, int* rewards, int* act_corr, int* trap_trials, double* params, int nA, int nTrials, double* output_probabilities)
    
@cython.boundscheck(False)
@cython.wraparound(False)

def get_marg_lkd_c(int nS, int nC, np.ndarray[int, ndim=1, mode="c"] stim_seq not None, np.ndarray[int, ndim=1, mode="c"] act_seq not None, \
                    np.ndarray[int, ndim=1, mode="c"] rewards not None, np.ndarray[int, ndim=1, mode="c"] act_corr not None, \
                    np.ndarray[int, ndim=1, mode="c"] trap_trials not None,  np.ndarray[double, ndim=1, mode="c"] params not None, int nA, int nTrials, np.ndarray[double, ndim=2, mode="c"] output_probabilities not None):
    return get_marg_lkd(nS, nC, &stim_seq[0], &act_seq[0], &rewards[0], &act_corr[0], &trap_trials[0], &params[0], nA, nTrials, &output_probabilities[0,0])
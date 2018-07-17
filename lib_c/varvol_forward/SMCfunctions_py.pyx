import cython
import numpy as np
cimport numpy as np

cdef extern from "SMCfunctions.hpp" namespace "smc":

    void bootstrapUpdateStep(int* currentStateRes, double* logThetaWeights, double* currentTauRes, double* gammaList, double* betaList, double* nuList, double tauDefault, int t,\
                             int* ancestorsStateList, double* ancestorsTauList, double* weightsList, int* mapping, int currentStimulus, int action, int reward, \
                                    int numberOfThetaSamples, int numberOfStateSamples, int K, int numberOfStimuli);

@cython.boundscheck(False)
@cython.wraparound(False)


def bootstrapUpdateStep_c(np.ndarray[int, ndim=2, mode="c"] currentStateRes not None, np.ndarray[double, ndim=1, mode="c"] logThetaWeights not None, np.ndarray[double, ndim=2, mode="c"] currentTauRes not None, np.ndarray[double, ndim=2, mode="c"] gammaList not None, \
                np.ndarray[double, ndim=1, mode="c"] betaList not None, np.ndarray[double, ndim=1, mode="c"] nuList not None, double tauDefault, int t, np.ndarray[int, ndim=2, mode="c"] ancestorsStateList not None, \
                            np.ndarray[double, ndim=2, mode="c"] ancestorsTauList not None, np.ndarray[double, ndim=2, mode="c"] weightsList not None, np.ndarray[int, ndim=2, mode="c"] mapping not None, int currentStimulus, int action, int reward):
    cdef int K, numberOfStimuli, numberOfStateSamples, numberOfThetaSamples
    K, numberOfStimuli, numberOfStateSamples, numberOfThetaSamples = gammaList.shape[1], mapping.shape[0], currentStateRes.shape[1], currentStateRes.shape[0]
    bootstrapUpdateStep(&currentStateRes[0,0], &logThetaWeights[0], &currentTauRes[0,0], &gammaList[0,0], &betaList[0], &nuList[0], tauDefault, t, &ancestorsStateList[0,0], &ancestorsTauList[0,0], &weightsList[0,0], \
                                                                                        &mapping[0,0], currentStimulus, action, reward, numberOfThetaSamples, numberOfStateSamples, K, numberOfStimuli)
    return;
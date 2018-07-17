import cython
import numpy as np
cimport numpy as np

cdef extern from "SMCfunctions.hpp" namespace "smc":
    double guidedSMC(int* stateRes, double* weightsRes, double* gamma, int K, double beta, double tau, int* mapping, int numberOfStimuli, int* stimuli, int* rewards, int T, \
                                                            int* actions, int numberOfSamples);
    
    void guidedUpdateStep(double* logApproxLikelihood, double* logThetaWeights, int* currentRes, double* gammaList, double* betaList, double* tauList, int t, int* ancestorsList, double* weightsList, int* mapping, \
                                        int previousStimulus, int currentStimulus, int reward, int action, int numberOfThetaSamples, int numberOfStateSamples, int K, int numberOfStimuli);
    
    void bootstrapUpdateStep(int* currentRes, double* gammaList, double* betaList, double* tauList, int t, int* ancestorsList, double* weightsList, \
                                                int* mapping, int previousStimulus, int numberOfThetaSamples, int numberOfStateSamples, int K, int numberOfStimuli);


@cython.boundscheck(False)
@cython.wraparound(False)

def guidedSmc_c(np.ndarray[int, ndim=1, mode="c"] stateRes not None, np.ndarray[double, ndim=1, mode="c"] weightsRes not None, np.ndarray[double, ndim=1, mode="c"] gamma not None, \
 				double beta, double tau, np.ndarray[int, ndim=2, mode="c"] mapping not None, np.ndarray[int, ndim=1, mode="c"] stimuli not None, \
 				np.ndarray[int, ndim=1, mode="c"] rewards not None, np.ndarray[int, ndim=1, mode="c"] actions not None, int numberOfSamples):
    cdef int K, numberOfStimuli, T
    cdef double approxLkd
    K, numberOfStimuli, T = gamma.shape[0], mapping.shape[0], stimuli.shape[0]
    approxLkd = guidedSMC(&stateRes[0], &weightsRes[0], &gamma[0], K, beta, tau, &mapping[0,0], numberOfStimuli, &stimuli[0], &rewards[0], T, &actions[0], numberOfSamples)
    return approxLkd

def guidedUpdateStep_c(np.ndarray[double, ndim=1, mode="c"] logApproxLikelihood not None, np.ndarray[double, ndim=1, mode="c"] logThetaWeights not None, np.ndarray[int, ndim=2, mode="c"] currentRes not None, np.ndarray[double, ndim=2, mode="c"] gammaList not None, \
                np.ndarray[double, ndim=1, mode="c"] betaList not None, np.ndarray[double, ndim=1, mode="c"] tauList not None, int t, np.ndarray[int, ndim=2, mode="c"] ancestorsList not None, \
                np.ndarray[double, ndim=2, mode="c"] weightsList not None, np.ndarray[int, ndim=2, mode="c"] mapping not None, int previousStimulus, int currentStimulus, int reward, int action):
    cdef int K, numberOfStimuli, numberOfStateSamples, numberOfThetaSamples
    K, numberOfStimuli, numberOfStateSamples, numberOfThetaSamples = gammaList.shape[1], mapping.shape[0], currentRes.shape[1], currentRes.shape[0]
    guidedUpdateStep(&logApproxLikelihood[0], &logThetaWeights[0], &currentRes[0,0], &gammaList[0,0], &betaList[0], &tauList[0], t, &ancestorsList[0,0], &weightsList[0,0], &mapping[0,0], previousStimulus, currentStimulus, reward, action, \
                                                                                                                    numberOfThetaSamples, numberOfStateSamples, K , numberOfStimuli)
    return;

def bootstrapUpdateStep_c(np.ndarray[int, ndim=2, mode="c"] currentRes not None, np.ndarray[double, ndim=2, mode="c"] gammaList not None, \
                np.ndarray[double, ndim=1, mode="c"] betaList not None, np.ndarray[double, ndim=1, mode="c"] tauList not None, int t, np.ndarray[int, ndim=2, mode="c"] ancestorsList not None, \
                np.ndarray[double, ndim=2, mode="c"] weightsList not None, np.ndarray[int, ndim=2, mode="c"] mapping not None, int previousStimulus):
    cdef int K, numberOfStimuli, numberOfStateSamples, numberOfThetaSamples
    K, numberOfStimuli, numberOfStateSamples, numberOfThetaSamples = gammaList.shape[1], mapping.shape[0], currentRes.shape[1], currentRes.shape[0]
    bootstrapUpdateStep(&currentRes[0,0], &gammaList[0,0], &betaList[0], &tauList[0], t, &ancestorsList[0,0], &weightsList[0,0], &mapping[0,0], previousStimulus, numberOfThetaSamples, numberOfStateSamples, K, numberOfStimuli)
    return;


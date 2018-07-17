import cython
import numpy as np
cimport numpy as np

cdef extern from "SMCfunctions.hpp" namespace "smc":
    double guidedSMC(int* stateRes, double* tauRes, double* weightsRes, double* gamma, int K, double beta, double nu, double tauDefault, int* mapping, int numberOfStimuli, int* stimuli, int* rewards, \
                                                     int T, int* actions, int numberOfSamples);

    void guidedUpdateStep(double* logApproxLikelihood, double* logThetaWeights, int* currentStateRes, double* currentTauRes, double* gammaList, double* betaList, double* nuList, double tauDefault, int t, \
                                                    int* ancestorsStateList, double* ancestorsTauList, double* weightsList, int* mapping, int previousStimulus, int currentStimulus, int reward, int action, \
                                                                                                                                int numberOfThetaSamples, int numberOfStateSamples, int K, int numberOfStimuli);

    void bootstrapUpdateStep(int* currentStateRes, double* currentTauRes, double* gammaList, double* betaList, double* nuList, double tauDefault, int t, int* ancestorsStateList, double* ancestorsTauList, \
                                    double* weightsList, int* mapping, int previousStimulus, int numberOfThetaSamples, int numberOfStateSamples, int K, int numberOfStimuli);

@cython.boundscheck(False)
@cython.wraparound(False)

def guidedSmc_c(np.ndarray[int, ndim=1, mode="c"] stateRes not None, np.ndarray[double, ndim=1, mode="c"] tauRes not None, np.ndarray[double, ndim=1, mode="c"] weightsRes not None, np.ndarray[double, ndim=1, mode="c"] gamma not None, \
 				double beta, double nu, double tauDefault, np.ndarray[int, ndim=2, mode="c"] mapping not None, np.ndarray[int, ndim=1, mode="c"] stimuli not None, \
 				np.ndarray[int, ndim=1, mode="c"] rewards not None, np.ndarray[int, ndim=1, mode="c"] actions not None, int numberOfSamples):
    cdef int K, numberOfStimuli, T
    cdef double approxLkd
    K, numberOfStimuli, T = gamma.shape[0], mapping.shape[0], stimuli.shape[0]
    approxLkd = guidedSMC(&stateRes[0], &tauRes[0], &weightsRes[0], &gamma[0], K, beta, nu, tauDefault, &mapping[0,0], numberOfStimuli, &stimuli[0], &rewards[0], T, &actions[0], numberOfSamples)
    return approxLkd

def guidedUpdateStep_c(np.ndarray[double, ndim=1, mode="c"] logApproxLikelihood not None, np.ndarray[double, ndim=1, mode="c"] logThetaWeights not None, np.ndarray[int, ndim=2, mode="c"] currentStateRes not None, \
                np.ndarray[double, ndim=2, mode="c"] currentTauRes not None, np.ndarray[double, ndim=2, mode="c"] gammaList not None, np.ndarray[double, ndim=1, mode="c"] betaList not None, \
                np.ndarray[double, ndim=1, mode="c"] nuList not None, double tauDefault, int t, np.ndarray[int, ndim=2, mode="c"] ancestorsStateList not None, np.ndarray[double, ndim=2, mode="c"] ancestorsTauList not None, \
                                                    np.ndarray[double, ndim=2, mode="c"] weightsList not None, np.ndarray[int, ndim=2, mode="c"] mapping not None, int previousStimulus, int currentStimulus, int reward, int action):
    cdef int K, numberOfStimuli, numberOfStateSamples, numberOfThetaSamples
    K, numberOfStimuli, numberOfStateSamples, numberOfThetaSamples = gammaList.shape[1], mapping.shape[0], currentStateRes.shape[1], currentStateRes.shape[0]
    guidedUpdateStep(&logApproxLikelihood[0], &logThetaWeights[0], &currentStateRes[0,0], &currentTauRes[0,0], &gammaList[0,0], &betaList[0], &nuList[0], tauDefault, t, &ancestorsStateList[0,0], &ancestorsTauList[0,0], \
                                        &weightsList[0,0], &mapping[0,0], previousStimulus, currentStimulus, reward, action, numberOfThetaSamples, numberOfStateSamples, K , numberOfStimuli)
    return;

def bootstrapUpdateStep_c(np.ndarray[int, ndim=2, mode="c"] currentStateRes not None, np.ndarray[double, ndim=2, mode="c"] currentTauRes not None, np.ndarray[double, ndim=2, mode="c"] gammaList not None, \
                np.ndarray[double, ndim=1, mode="c"] betaList not None, np.ndarray[double, ndim=1, mode="c"] nuList not None, double tauDefault, int t, np.ndarray[int, ndim=2, mode="c"] ancestorsStateList not None, \
                            np.ndarray[double, ndim=2, mode="c"] ancestorsTauList not None, np.ndarray[double, ndim=2, mode="c"] weightsList not None, np.ndarray[int, ndim=2, mode="c"] mapping not None, int previousStimulus):
    cdef int K, numberOfStimuli, numberOfStateSamples, numberOfThetaSamples
    K, numberOfStimuli, numberOfStateSamples, numberOfThetaSamples = gammaList.shape[1], mapping.shape[0], currentStateRes.shape[1], currentStateRes.shape[0]
    bootstrapUpdateStep(&currentStateRes[0,0], &currentTauRes[0,0], &gammaList[0,0], &betaList[0], &nuList[0], tauDefault, t, &ancestorsStateList[0,0], &ancestorsTauList[0,0], &weightsList[0,0], \
                                                                                        &mapping[0,0], previousStimulus, numberOfThetaSamples, numberOfStateSamples, K, numberOfStimuli)
    return;
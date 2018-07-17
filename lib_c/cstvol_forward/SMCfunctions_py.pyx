import cython
import numpy as np
cimport numpy as np

cdef extern from "SMCfunctions.hpp" namespace "smc":

    void bootstrapUpdateStep(int* currentRes, double* logThetaWeights, double* gammaList, double* betaList, double* tauList, int  t, int* ancestorsList, double* weightsList, int* mapping,
                                         int  currentStimulus, int  action, int  reward, int  numberOfThetaSamples, int  numberOfStateSamples, int  K, int  numberOfStimuli,
                                          int* ancestorsIndexes, double* gammaAdaptedProba, double* likelihoods, int* positiveStates, int positiveStatesProcessed);



    double bootstrap_guided(int* currentRes, double* logThetaWeights, double* gammaList, double* betaList, double* tauList, int  t, int* ancestorsList, double* weightsList, int* mapping,
                                         int currentStimulus, int action, int nextStim, int nextAction, int  reward, int  numberOfThetaSamples, int  numberOfStateSamples, int  K, int  numberOfStimuli,
                                         int* ancestorsIndexes, double* gammaAdaptedProba, double* likelihoods, int* positiveStates, int positiveStatesProcessed, double beta_softmax, double C);



@cython.boundscheck(False)
@cython.wraparound(False)

def bootstrapUpdateStep_c(np.ndarray[int, ndim=2, mode="c"] currentRes not None, np.ndarray[double, ndim=1, mode="c"] logThetaWeights not None, np.ndarray[double, ndim=2, mode="c"] gammaList not None, \
                np.ndarray[double, ndim=1, mode="c"] betaList not None, np.ndarray[double, ndim=1, mode="c"] tauList not None, int t, np.ndarray[int, ndim=2, mode="c"] ancestorsList not None, \
                np.ndarray[double, ndim=2, mode="c"] weightsList not None, np.ndarray[int, ndim=2, mode="c"] mapping not None, int currentStimulus, int action, int reward,\
                np.ndarray[int, ndim=1, mode="c"] ancestorsIndexes not None, np.ndarray[double, ndim=1, mode="c"] gammaAdaptedProba not None, np.ndarray[double, ndim=1, mode="c"] likelihoods not None, \
                np.ndarray[int, ndim=1, mode="c"] positiveStates not None, int positiveStatesProcessed):
    cdef int K, numberOfStimuli, numberOfStateSamples, numberOfThetaSamples
    K, numberOfStimuli, numberOfStateSamples, numberOfThetaSamples = gammaList.shape[1], mapping.shape[0], currentRes.shape[1], currentRes.shape[0]
    bootstrapUpdateStep(&currentRes[0,0], &logThetaWeights[0], &gammaList[0,0], &betaList[0], &tauList[0], t, &ancestorsList[0,0], &weightsList[0,0], &mapping[0,0], currentStimulus, action, reward, \
                        numberOfThetaSamples, numberOfStateSamples, K, numberOfStimuli, &ancestorsIndexes[0], &gammaAdaptedProba[0], &likelihoods[0], &positiveStates[0], positiveStatesProcessed)
    return;

def guided_bootstrap_c(np.ndarray[int, ndim=2, mode="c"] currentRes not None, np.ndarray[double, ndim=1, mode="c"] logThetaWeights not None, np.ndarray[double, ndim=2, mode="c"] gammaList not None, \
                np.ndarray[double, ndim=1, mode="c"] betaList not None, np.ndarray[double, ndim=1, mode="c"] tauList not None, int t, np.ndarray[int, ndim=2, mode="c"] ancestorsList not None, \
                np.ndarray[double, ndim=2, mode="c"] weightsList not None, np.ndarray[int, ndim=2, mode="c"] mapping not None, int currentStimulus, int action, int nextStim, int nextAction, \
                int reward, double beta, double C, np.ndarray[int, ndim=1, mode="c"] ancestorsIndexes not None, np.ndarray[double, ndim=1, mode="c"] gammaAdaptedProba not None, \
                np.ndarray[double, ndim=1, mode="c"] likelihoods not None, np.ndarray[int, ndim=1, mode="c"] positiveStates not None, int positiveStatesProcessed):
    cdef int K, numberOfStimuli, numberOfStateSamples, numberOfThetaSamples
    K, numberOfStimuli, numberOfStateSamples, numberOfThetaSamples = gammaList.shape[1], mapping.shape[0], currentRes.shape[1], currentRes.shape[0]
    return bootstrap_guided(&currentRes[0,0], &logThetaWeights[0], &gammaList[0,0], &betaList[0], &tauList[0], t, &ancestorsList[0,0], &weightsList[0,0], &mapping[0,0], currentStimulus, \
        action, nextStim, nextAction, reward, numberOfThetaSamples, numberOfStateSamples, K, numberOfStimuli, &ancestorsIndexes[0], &gammaAdaptedProba[0], \
        &likelihoods[0], &positiveStates[0], positiveStatesProcessed, beta, C);

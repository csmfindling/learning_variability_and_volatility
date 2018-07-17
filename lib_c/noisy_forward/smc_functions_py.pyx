import cython
import numpy as np
cimport numpy as np

cdef extern from "smc_functions.hpp" namespace "smc":

    double bootstrap_smc_step(double* logParamWeights, double* distances, int* currentTaskSetSamples, int* ancestorTaskSetSamples, double* weightsList, double* paramDirichletSamples, 
                                    double* paramBetaSamples, double lambdaa, double eta, double noise_inertie, int* mapping, int currentStimulus, int reward, int action,
                                        int numberOfParamSamples, int numberOfLatentSamples, int K, int t, int numberOfStimuli, double* likelihoods, int* positiveStates, 
                                        double* ante_proba_local, double* post_proba_local, int* ancestorsIndexes, double* gammaAdaptedProba, double* sum_weightsList, double* currentNoises, double temperature);

@cython.boundscheck(False)
@cython.wraparound(False)

def bootstrap_smc_step_c(np.ndarray[double, ndim=1, mode="c"] logParamWeights not None, np.ndarray[double, ndim=2, mode="c"] distances not None, np.ndarray[double, ndim=1, mode="c"] paramBetaSamples not None,  
                     double lambdaa, double eta, double noise_inertie, np.ndarray[double, ndim=2, mode="c"] paramDirichletSamples not None, np.ndarray[int, ndim=2, mode="c"] currentTaskSetSamples not None, np.ndarray[int, ndim=2, mode="c"] ancestorTaskSetSamples not None,
                     np.ndarray[double, ndim=2, mode="c"] weightsList not None, np.ndarray[int, ndim=2, mode="c"] mapping not None, int currentStimulus, int reward, int action, int t,
                     np.ndarray[double, ndim=1, mode="c"] likelihoods not None, np.ndarray[int, ndim=1, mode="c"] positiveStates not None, np.ndarray[double, ndim=1, mode="c"] ante_proba_local not None,
                     np.ndarray[double, ndim=1, mode="c"] post_proba_local not None, np.ndarray[int, ndim=1, mode="c"] ancestorsIndexes not None, np.ndarray[double, ndim=1, mode="c"] gammaAdaptedProba not None,
                     np.ndarray[double, ndim=1, mode="c"] sum_weightsList not None, np.ndarray[double, ndim=2, mode="c"] currentNoises not None, double temperature):
    cdef int K, numberOfStimuli, numberOfLatentSamples, numberOfParamSamples
    numberOfParamSamples, numberOfLatentSamples = currentTaskSetSamples.shape[0], currentTaskSetSamples.shape[1]
    K = paramDirichletSamples.shape[1]
    numberOfStimuli = mapping.shape[0]
    return bootstrap_smc_step(&logParamWeights[0], &distances[0,0], &currentTaskSetSamples[0,0], &ancestorTaskSetSamples[0,0], &weightsList[0,0],
        &paramDirichletSamples[0,0], &paramBetaSamples[0], lambdaa, eta, noise_inertie, &mapping[0,0], currentStimulus, reward, action, numberOfParamSamples, numberOfLatentSamples, K, t, numberOfStimuli,
        &likelihoods[0], &positiveStates[0], &ante_proba_local[0], &post_proba_local[0], &ancestorsIndexes[0], &gammaAdaptedProba[0], &sum_weightsList[0], &currentNoises[0,0], temperature)

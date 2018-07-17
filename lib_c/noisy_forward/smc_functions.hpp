//
//  smc_functions.hpp
//  smc_functions
//
//  Created by Charles Findling on 3/16/16.
//  Copyright © 2016 Charles Findling. All rights reserved.
//

#ifndef smc_functions_hpp
#define smc_functions_hpp

#include <stdio.h>

namespace smc {

    double bootstrap_smc_step(double* logParamWeights, double* distances, int* currentTaskSetSamples, int* ancestorTaskSetSamples, double* weightsList, double* const paramDirichletSamples, 
                                    double* const paramBetaSamples, double lambda, double eta, double noise_inertie, int* mapping, int const currentStimulus, int const reward, int const action,
                                        int numberOfParamSamples, int numberOfLatentSamples, int const K, int t, int const numberOfStimuli, double* likelihoods, int* positiveStates, 
                                        double* ante_proba_local, double* post_proba_local, int* ancestorsIndexes, double* gammaAdaptedProba, double* sum_weightsList, double* currentNoises, double temperature);
}

#endif /* smc_functions_hpp */

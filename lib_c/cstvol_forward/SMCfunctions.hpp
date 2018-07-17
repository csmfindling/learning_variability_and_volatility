//
//  SMCfunctions.hpp
//  CreateTask
//
//  Created by Charles Findling on 21/01/2016.
//  Copyright © 2016 Charles Findling. All rights reserved.
//

#ifndef SMCfunctions_hpp
#define SMCfunctions_hpp

#include <stdio.h>

namespace smc {
    
    void bootstrapUpdateStep(int* currentRes, double* logThetaWeights, double* gammaList, double* betaList, double* tauList, int const t, int* ancestorsList, double* weightsList, int* mapping,
                                         int const currentStimulus, int const action, int const reward, int const numberOfThetaSamples, int const numberOfStateSamples, int const K, int const numberOfStimuli,
                                         int* ancestorsIndexes, double* gammaAdaptedProba, double* likelihoods, int* positiveStates, int positiveStatesProcessed);


    double bootstrap_guided(int* currentRes, double* logThetaWeights, double* gammaList, double* betaList, double* tauList, int const t, int* ancestorsList, double* weightsList, int* mapping,
                                         int const currentStimulus, int const action, int nextStim, int nextAction, int const reward, int const numberOfThetaSamples, int const numberOfStateSamples, int const K, int const numberOfStimuli,
                                         int* ancestorsIndexes, double* gammaAdaptedProba, double* likelihoods, int* positiveStates, int positiveStatesProcessed, double beta_softmax, double C);    
}

#endif /* SMCfunctions_hpp */

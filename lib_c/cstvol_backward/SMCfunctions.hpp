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
    
    void bootstrapUpdateStep(int* currentRes, double* gammaList, double* betaList, double* tauList, int const t, int* ancestorsList, double* weightsList, int* mapping, int const previousStimulus, int const numberOfThetaSamples, int const numberOfStateSamples, int const K, int const numberOfStimuli);
    
    void guidedUpdateStep(double* logApproxLikelihood, double* logThetaWeights, int* currentRes, double* gammaList, double* betaList, double* tauList, int const t, int* ancestorsList, double* weightsList, int* mapping, int const previousStimulus, int const currentStimulus, int const reward, int const action, int const numberOfThetaSamples, int const numberOfStateSamples, int const K, int const numberOfStimuli);
    
    double guidedSMC(int* stateRes, double* weightsRes, double* const gamma, int const K, double const beta, double const tau, int* const mapping, int const numberOfStimuli, int* const stimuli, int* const rewards, int const T, int* const actions, int const numberOfSamples);
}

#endif /* SMCfunctions_hpp */

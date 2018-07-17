//
//  SMCfunctions.hpp
//  CreateTask
//
//  Created by Charles Findling on 23/01/2016.
//  Copyright © 2016 Charles Findling. All rights reserved.
//

#ifndef SMCfunctions_hpp
#define SMCfunctions_hpp

#include <stdio.h>

namespace smc {
    
    void bootstrapUpdateStep(int* currentStateRes, double* currentTauRes, double* const gammaList, double* const betaList, double* const nuList, double const tauDefault, int const t, int* const ancestorsStateList, double* const ancestorsTauList, double* weightsList, int* mapping, int const previousStimulus, int numberOfThetaSamples, int numberOfStateSamples, int const K, int const numberOfStimuli);
    
    void guidedUpdateStep(double* logApproxLikelihood, double* logThetaWeights, int* currentStateRes, double* currentTauRes, double* const gammaList, double* const betaList, double* const nuList, double const tauDefault, int const t, int* const ancestorsStateList, double* const ancestorsTauList, double* weightsList,  int* mapping, int const previousStimulus, int const currentStimulus, int const reward, int const action, int numberOfThetaSamples, int numberOfStateSamples, int const K, int const numberOfStimuli);
    
    double guidedSMC(int* stateRes, double* tauRes, double* weightsRes, double* const gamma, int const K, double const beta, double const nu, double const tauDefault, int* const mapping, int const numberOfStimuli, int* const stimuli, int* const rewards, int const T, int* const actions, int const numberOfSamples);
}
#endif /* SMCfunctions_hpp */

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
    
    void bootstrapUpdateStep(int* currentStateRes, double* logThetaWeights, double* currentTauRes, double* const gammaList, double* const betaList, double* const nuList, double const tauDefault, int const t, int* const ancestorsStateList, double* const ancestorsTauList, double* weightsList, int* mapping, int const currentStimulus, int const action, int const reward, int numberOfThetaSamples, int numberOfStateSamples, int const K, int const numberOfStimuli);
      
}
#endif /* SMCfunctions_hpp */

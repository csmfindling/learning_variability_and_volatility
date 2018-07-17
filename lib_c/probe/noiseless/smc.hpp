//
//  smc.hpp
//  smc_probe_1
//
//  Created by Charles Findling on 9/7/16.
//  Copyright © 2016 csmfindling. All rights reserved.
//

#ifndef smc_hpp
#define smc_hpp

#include <stdio.h>

namespace smc {
       
    double get_marg_lkd(int nS, int nC, int* stim_seq, int* act_seq, int* rewards, int* act_corr, int* trap_trials, double* params, int nA, int nTrials, double* output_probabilities);
}

#endif /* smc_hpp */

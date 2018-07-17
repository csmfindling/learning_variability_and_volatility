//
//  SMCfunctions.cpp
//  CreateTask
//
//  Created by Charles Findling on 23/01/2016.
//  Copyright © 2016 Charles Findling. All rights reserved.
//

#include "SMCfunctions.hpp"
#include "usefulFunctions.hpp"
#include "usefulFunctions.cpp"
#include <boost/random/mersenne_twister.hpp>


namespace smc {
    
    boost::mt19937 generator(static_cast<unsigned int>(time(0)));
    boost::uniform_real<> distribution(0,1);
    boost::normal_distribution<> distribution_normal(0, 1);

    void bootstrapUpdateStep(int* currentStateRes, double* logThetaWeights, double* currentTauRes, double* const gammaList, double* const betaList, double* const nuList, double const tauDefault, int const t, int* const ancestorsStateList, double* const ancestorsTauList, double* weightsList, int* mapping, int const currentStimulus, int const action, int const reward, int numberOfThetaSamples, int numberOfStateSamples, int const K, int const numberOfStimuli)
    {
        std::vector<int> ancestorsIndexes(numberOfStateSamples);
        std::vector<double> gammaAdaptedProba(K);
        std::vector<bool> positiveStates(K,1);
        std::vector<double> likelihoods(K);
        std::vector<double> sum_weightsList(numberOfThetaSamples);
        double sum_gammaAdaptedProba;
        double cumSum;
        int index;
        double uniform_sample = 0.;
        int ancestorState;
        double ancestorTau;
        double tauSampleTmp;

        if (t > 0)
        {
            positiveStates = isEqual_and_adapted_logical_xor(mapping, currentStimulus, numberOfStimuli, K, action, reward);

            for (int param_idx = 0; param_idx < numberOfThetaSamples; ++param_idx)
            {
                
                for (int k = 0; k < K; ++k)
                {
                    likelihoods[k] = positiveStates[k] * (*(betaList + param_idx)) + (1 - positiveStates[k]) * (1 - (*(betaList + param_idx)));
                }

                double sum_post_weights = 0;

                for (int traj_idx = 0; traj_idx < numberOfStateSamples; ++traj_idx)
                {
                    *(weightsList + param_idx * numberOfStateSamples + traj_idx) = likelihoods[*(ancestorsStateList + param_idx * numberOfStateSamples + traj_idx)];
                    sum_post_weights += *(weightsList + param_idx * numberOfStateSamples + traj_idx);
                }

                sum_weightsList[param_idx] = sum_post_weights;
                *(logThetaWeights + param_idx) += log(sum_post_weights / numberOfStateSamples);
            }
        }        

        // Loop of theta samples
        for (int theta_idx = 0; theta_idx < numberOfThetaSamples; ++theta_idx) {
            
            stratified_resampling(distribution(generator), weightsList + theta_idx * numberOfStateSamples, numberOfStateSamples, &ancestorsIndexes[0], sum_weightsList[theta_idx]);

            // Sample descendants for theta sample theta_idx
            if (t > 0) {
                for (int traj_idx = 0; traj_idx < numberOfStateSamples; ++traj_idx) {
                    ancestorState                                                    = *(ancestorsStateList + theta_idx * numberOfStateSamples + ancestorsIndexes[traj_idx]);
                    ancestorTau                                                      = *(ancestorsTauList + theta_idx * numberOfStateSamples + ancestorsIndexes[traj_idx]);
                    tauSampleTmp                                                     = ancestorTau + distribution_normal(generator) * sqrt(nuList[theta_idx]); //Sample_Normal_Distribution(generator, ancestorTau, sqrt(nuList[theta_idx]));
                    *(currentTauRes + theta_idx * numberOfStateSamples + traj_idx)   = (tauSampleTmp > 0) * (tauSampleTmp < 1) * tauSampleTmp + (tauSampleTmp > 1);
                    if (distribution(generator) > *(currentTauRes + theta_idx * numberOfStateSamples + traj_idx))
                    {
                        *(currentStateRes + theta_idx * numberOfStateSamples + traj_idx) = ancestorState;
                    }
                    else
                    {
                        sum_gammaAdaptedProba = isNotEqual_timesVector_notnormalise(mapping, currentStimulus, numberOfStimuli, K, mapping[currentStimulus * K + ancestorState], gammaList + theta_idx * K, &gammaAdaptedProba[0]);
                        
                        index          = 0;
                        cumSum         = gammaAdaptedProba[0]/sum_gammaAdaptedProba;
                        uniform_sample = distribution(generator);

                        while (cumSum < uniform_sample){
                            index  += 1;
                            cumSum += gammaAdaptedProba[index]/sum_gammaAdaptedProba;
                        }

                        *(currentStateRes + theta_idx * numberOfStateSamples + traj_idx) = index;
                    }
                }
            }
            else
            {
                generator.discard(70000);
                for (int traj_idx = 0; traj_idx < numberOfStateSamples; ++traj_idx) {
                    *(currentStateRes + theta_idx * numberOfStateSamples + traj_idx) = Sample_Discrete_Distribution(generator, gammaList + theta_idx * K, K);
                    *(currentTauRes + theta_idx * numberOfStateSamples + traj_idx)   = tauDefault;
                }
            }
        }
        return;
    }
}

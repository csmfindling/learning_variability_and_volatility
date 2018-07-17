//
//  smc_functions.cpp
//  smc_functions
//
//  Created by Charles Findling on 3/16/16.
//  Copyright © 2016 Charles Findling. All rights reserved.
//

#include "smc_functions.hpp"
#include "usefulFunctions.hpp"
#include "usefulFunctions.cpp"
#include <boost/random/mersenne_twister.hpp>

namespace smc {
    
    using namespace std;
    
    boost::mt19937 generator(static_cast<unsigned int>(time(0)));
    boost::uniform_real<> distribution(0,1);
    boost::normal_distribution<> distribution_normal(0, 1);

    double bootstrap_smc_step(double* logParamWeights, double* distances, int* currentTaskSetSamples, int* ancestorTaskSetSamples, double* weightsList, double* const paramDirichletSamples, 
                                    double* const paramBetaSamples, double lambda, double eta, double noise_inertie, int* mapping, int const currentStimulus, int const reward, int const action,
                                        int numberOfParamSamples, int numberOfLatentSamples, int const K, int t, int const numberOfStimuli, double* likelihoods, int* positiveStates, 
                                        double* ante_proba_local, double* post_proba_local, int* ancestorsIndexes, double* gammaAdaptedProba, double* sum_weightsList, double* currentNoises, double temperature)
    {
        double partMargLikelihood = 0.;
        double cumSum         = 0.;
        int index             = 0;
        double uniform_sample = 0.;
        double sum_gammaAdaptedProba = 0.;
        double distance           = 0.;
        std::vector<double> ante_proba(K, 0.);
        std::vector<double> post_proba(K, 0.);
        int ancestor;
        double local_temp = 0.5;
        std::vector<double> noisy_post_proba(K, 0.);

        if (t > 0)
        {
            isEqual_and_adapted_logical_xor(mapping, currentStimulus, numberOfStimuli, K, action, reward, positiveStates);

            double maxParamW   = *max_element(logParamWeights, logParamWeights + numberOfParamSamples); // weight rescaling with maximum of weights at time t-1
            double sumParamW_ante = 0;
            double sumParamW_post = 0;

            for (int param_idx = 0; param_idx < numberOfParamSamples; ++param_idx)
            {

                for (int k = 0; k < K; ++k) 
                {
                    likelihoods[k]      = positiveStates[k] * (*(paramBetaSamples + param_idx)) + (1 - positiveStates[k]) * (1 - (*(paramBetaSamples + param_idx)));
                    ante_proba_local[k] = 0.;
                    post_proba_local[k] = 0.;
                }

                double sum_post_weights_local = 0;
                for (int traj_idx = 0; traj_idx < numberOfLatentSamples; ++traj_idx)
                {
                    ante_proba_local[*(ancestorTaskSetSamples + numberOfLatentSamples * param_idx + traj_idx)] += 1;
                    post_proba_local[*(ancestorTaskSetSamples + numberOfLatentSamples * param_idx + traj_idx)] += likelihoods[*(ancestorTaskSetSamples + numberOfLatentSamples * param_idx + traj_idx)];
                    sum_post_weights_local += likelihoods[*(ancestorTaskSetSamples + numberOfLatentSamples * param_idx + traj_idx)];

                    *(weightsList + param_idx * numberOfLatentSamples + traj_idx) = likelihoods[*(ancestorTaskSetSamples + numberOfLatentSamples * param_idx + traj_idx)];
                }

                sum_weightsList[param_idx] = sum_post_weights_local;

                *(distances + param_idx) = 0.;
                for (int k = 0; k < K; ++k)
                {
                    ante_proba_local[k] /= (1. * numberOfLatentSamples);
                    ante_proba[k]       += exp(*(logParamWeights + param_idx) - maxParamW) * ante_proba_local[k];
                    *(distances + param_idx) += lambda * abs(ante_proba_local[k] - post_proba_local[k]);
                }

                *(distances + param_idx) += eta;

                sumParamW_ante                 += exp(*(logParamWeights + param_idx) - maxParamW);
                partMargLikelihood              = sum_post_weights_local/numberOfLatentSamples;
                *(logParamWeights + param_idx) += log(partMargLikelihood);
                sumParamW_post                 += exp(*(logParamWeights + param_idx) - maxParamW);

                for (int k = 0; k < K; ++k)
                {
                    post_proba_local[k] /= sum_post_weights_local;
                    post_proba[k] += exp(*(logParamWeights + param_idx) - maxParamW) * post_proba_local[k];
                }
            }

            distance = 0.;
            for (int k = 0; k < K; ++k)
            {
                if (post_proba[k] > 0)
                {                
                    distance     += abs(ante_proba[k]/sumParamW_ante - post_proba[k]/sumParamW_post); //- (post_proba[k]/sumParamW_post) * log((post_proba[k]/sumParamW_post)); ///(ante_proba[k]/sumParamW_ante)); //   //                
                    post_proba[k] = post_proba[k]/sumParamW_post;
                }
            }
            distance = lambda * distance + eta; 

            local_temp = distance/2.   
                                    
        }        

    

        for (int param_idx = 0; param_idx < numberOfParamSamples; ++param_idx) 
        {
            if (t > 0)
            {   
                stratified_resampling(distribution(generator), weightsList + param_idx * numberOfLatentSamples, numberOfLatentSamples, &ancestorsIndexes[0], sum_weightsList[param_idx]);

                for (int traj_idx = 0; traj_idx < numberOfLatentSamples; ++traj_idx)
                {
                    if (K > 0)
                    {
                        ancestor   = *(ancestorTaskSetSamples + param_idx * numberOfLatentSamples + ancestorsIndexes[traj_idx]);                        
                        *(currentNoises + param_idx * numberOfLatentSamples + traj_idx) = local_temp;
                        if (distribution(generator) > local_temp)
                        {
                            *(currentTaskSetSamples + param_idx * numberOfLatentSamples + traj_idx) = ancestor;
                        }
                        else
                        {
                            if (K > 2)
                            {
                                sum_gammaAdaptedProba = isNotEqual_timesVector_notnormalise(mapping, currentStimulus, numberOfStimuli, K, mapping[currentStimulus * K + ancestor], paramDirichletSamples + param_idx * K, &gammaAdaptedProba[0]);
                                index          = 0;
                                cumSum         = gammaAdaptedProba[0]/sum_gammaAdaptedProba;
                                uniform_sample = distribution(generator);

                                while (cumSum < uniform_sample){
                                    index  += 1;
                                    cumSum += gammaAdaptedProba[index]/sum_gammaAdaptedProba;
                                }
                                *(currentTaskSetSamples + param_idx * numberOfLatentSamples + traj_idx) = index;
                            }
                            else
                            {
                                *(currentTaskSetSamples + param_idx * numberOfLatentSamples + traj_idx) = 1 - ancestor;
                            }
                        }
                    }
                    else
                    {
                        *(currentNoises + param_idx * numberOfLatentSamples + traj_idx)         = temperature;
                        *(currentTaskSetSamples + param_idx * numberOfLatentSamples + traj_idx) = Sample_Discrete_Distribution(generator, &noisy_post_proba[0], K);
                    }

                }
            }
            else
            {
                generator.discard(70000);

                for (int traj_idx = 0; traj_idx < numberOfLatentSamples; ++traj_idx) 
                {
                    *(currentTaskSetSamples + numberOfLatentSamples * param_idx + traj_idx) = Sample_Discrete_Distribution(generator, paramDirichletSamples + param_idx * K, K);
                }
            }
        }
        return distance;
    };
}
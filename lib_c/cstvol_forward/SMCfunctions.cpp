//
//  SMCfunctions.cpp
//  CreateTask
//
//  Created by Charles Findling on 21/01/2016.
//  Copyright © 2016 Charles Findling. All rights reserved.
//

#include "SMCfunctions.hpp"
#include "usefulFunctions.hpp"
#include "usefulFunctions.cpp"
#include <boost/random/mersenne_twister.hpp>

namespace smc {
    
    boost::mt19937 generator(static_cast<unsigned int>(time(0)));
    boost::uniform_real<> distribution(0,1);
    
    void bootstrapUpdateStep(int* currentRes, double* logThetaWeights, double* gammaList, double* betaList, double* tauList, int const t, int* ancestorsList, double* weightsList, int* mapping,
                                         int const currentStimulus, int const action, int const reward, int const numberOfThetaSamples, int const numberOfStateSamples, int const K, int const numberOfStimuli,
                                         int* ancestorsIndexes, double* gammaAdaptedProba, double* likelihoods, int* positiveStates, int positiveStatesProcessed)
    {
        int ancestor;
        double cumSum         = 0.;
        int index             = 0;
        double uniform_sample = 0.;
        double sum_gammaAdaptedProba = 0.;
        std::vector<double> sum_weightsList(numberOfThetaSamples);
        if (t > 0)
        {
            if (positiveStatesProcessed==0)
            {
                isEqual_and_adapted_logical_xor(mapping, currentStimulus, numberOfStimuli, K, action, reward, positiveStates);
            }

            for (int param_idx = 0; param_idx < numberOfThetaSamples; ++param_idx)
            {
                for (int k = 0; k < K; ++k)
                {
                    likelihoods[k] = positiveStates[k] * (*(betaList + param_idx)) + (1 - positiveStates[k]) * (1 - (*(betaList + param_idx)));
                }

                double sum_post_weights = 0;

                for (int traj_idx = 0; traj_idx < numberOfStateSamples; ++traj_idx)
                {
                    *(weightsList + param_idx * numberOfStateSamples + traj_idx) = likelihoods[*(ancestorsList + param_idx * numberOfStateSamples + traj_idx)];
                    sum_post_weights += *(weightsList + param_idx * numberOfStateSamples + traj_idx);
                }

                sum_weightsList[param_idx] = sum_post_weights;
                *(logThetaWeights + param_idx) += log(sum_post_weights / numberOfStateSamples);
            }
        }

        // Loop of theta samples
        for (int theta_idx = 0; theta_idx < numberOfThetaSamples; ++theta_idx) {

            // Samples descendants for theta sample theta_idx
            if (t > 0) {

                // Ancestors for theta sample theta_idx
                stratified_resampling(distribution(generator), weightsList + theta_idx * numberOfStateSamples, numberOfStateSamples, ancestorsIndexes, sum_weightsList[theta_idx]);

                for (int traj_idx = 0; traj_idx < numberOfStateSamples; ++traj_idx) {
                    ancestor = *(ancestorsList + theta_idx * numberOfStateSamples + ancestorsIndexes[traj_idx]);
                    if (distribution(generator) > tauList[theta_idx])
                    {
                        *(currentRes + theta_idx * numberOfStateSamples + traj_idx) = ancestor;
                    }
                    else
                    {
                        sum_gammaAdaptedProba = isNotEqual_timesVector_notnormalise(mapping, currentStimulus, numberOfStimuli, K, mapping[currentStimulus * K + ancestor], gammaList + theta_idx * K, gammaAdaptedProba);
                        index          = 0;
                        cumSum         = gammaAdaptedProba[0]/sum_gammaAdaptedProba;
                        uniform_sample = distribution(generator);

                        while (cumSum < uniform_sample){
                            index  += 1;
                            cumSum += gammaAdaptedProba[index]/sum_gammaAdaptedProba;
                        }
                        *(currentRes + theta_idx * numberOfStateSamples + traj_idx) = index;
                    }
                }
            }
            else
            {
                // Assign gamma vector
                generator.discard(70000);
                for (int traj_idx = 0; traj_idx < numberOfStateSamples; ++traj_idx) {
                    *(currentRes + theta_idx * numberOfStateSamples + traj_idx) = Sample_Discrete_Distribution(generator, gammaList + theta_idx * K, K);
                }
            }
        }
        
        return;
    }

    double bootstrap_guided(int* currentRes, double* logThetaWeights, double* gammaList, double* betaList, double* tauList, int const t, int* ancestorsList, double* weightsList, int* mapping,
                                         int const currentStimulus, int const action, int nextStim, int nextAction, int const reward, int const numberOfThetaSamples, int const numberOfStateSamples, int const K, int const numberOfStimuli,
                                         int* ancestorsIndexes, double* gammaAdaptedProba, double* likelihoods, int* positiveStates, int positiveStatesProcessed, double beta_softmax, double C)
    {
        int ancestor;
        double cumSum         = 0.;
        int index             = 0;
        double uniform_sample = 0.;
        double sum_gammaProposal = 0.;
        double log_proba         = 0.;
        std::vector<double> sum_weightsList(numberOfThetaSamples);
        std::vector<double> actionAdaptedProba(4);
        std::vector<double> gamma_proposal(24);
        if (t > 0)
        {
            if (positiveStatesProcessed==0)
            {
                isEqual_and_adapted_logical_xor(mapping, currentStimulus, numberOfStimuli, K, action, reward, positiveStates);
            }

            for (int param_idx = 0; param_idx < numberOfThetaSamples; ++param_idx)
            {
                for (int k = 0; k < K; ++k)
                {
                    likelihoods[k] = positiveStates[k] * (*(betaList + param_idx)) + (1 - positiveStates[k]) * (1 - (*(betaList + param_idx)));
                }

                double sum_post_weights = 0;

                for (int traj_idx = 0; traj_idx < numberOfStateSamples; ++traj_idx)
                {
                    *(weightsList + param_idx * numberOfStateSamples + traj_idx) = likelihoods[*(ancestorsList + param_idx * numberOfStateSamples + traj_idx)];
                    sum_post_weights += *(weightsList + param_idx * numberOfStateSamples + traj_idx);
                }

                sum_weightsList[param_idx] = sum_post_weights;
                *(logThetaWeights + param_idx) += log(sum_post_weights / numberOfStateSamples);
            }
        }

        // Loop of theta samples
        for (int theta_idx = 0; theta_idx < numberOfThetaSamples; ++theta_idx) {

            // Samples descendants for theta sample theta_idx
            if (t > 0) {

                // Ancestors for theta sample theta_idx
                stratified_resampling(distribution(generator), weightsList + theta_idx * numberOfStateSamples, numberOfStateSamples, ancestorsIndexes, sum_weightsList[theta_idx]);

                for (int traj_idx = 0; traj_idx < numberOfStateSamples; ++traj_idx) 
                {
                    ancestor = *(ancestorsList + theta_idx * numberOfStateSamples + ancestorsIndexes[traj_idx]);

                    actionAdaptedProba.assign(4,0.);
                    sum_gammaProposal = isNotEqual_timesVector_normalise(mapping, currentStimulus, numberOfStimuli, K, mapping[currentStimulus * K + ancestor], gammaList + theta_idx * K, &gamma_proposal[0], *(tauList + theta_idx), ancestor, nextStim, nextAction, C, beta_softmax, gammaAdaptedProba, &actionAdaptedProba[0]);

                    index          = 0;
                    cumSum         = gamma_proposal[0]/sum_gammaProposal;
                    uniform_sample = distribution(generator);
                    gamma_proposal[0] = gamma_proposal[0]/sum_gammaProposal;

                    while (cumSum < uniform_sample)
                    {
                        index  += 1;
                        cumSum += gamma_proposal[index]/sum_gammaProposal;
                        gamma_proposal[index] = gamma_proposal[index]/sum_gammaProposal;
                    }
                    *(currentRes + theta_idx * numberOfStateSamples + traj_idx) = index;

                    log_proba += std::log(gammaAdaptedProba[index]) - std::log(gamma_proposal[index]); //+ std::log(sum_gammaProposal);
                }
            }
            else
            {
                // Assign gamma vector
                generator.discard(70000);
                for (int traj_idx = 0; traj_idx < numberOfStateSamples; ++traj_idx) {
                    *(currentRes + theta_idx * numberOfStateSamples + traj_idx) = Sample_Discrete_Distribution(generator, gammaList + theta_idx * K, K);
                }
            }
        }
        
        return log_proba;
    }
}
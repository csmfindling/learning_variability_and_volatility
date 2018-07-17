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
    
    void bootstrapUpdateStep(int* currentRes, double* gammaList, double* betaList, double* tauList, int const t, int* ancestorsList, double* weightsList, int* mapping, int const previousStimulus, int const numberOfThetaSamples, int const numberOfStateSamples, int const K, int const numberOfStimuli)
    {
        std::vector<double> gamma(K);
        std::vector<int> ancestorsIndexes(numberOfStateSamples);
        std::vector<double> gammaAdaptedProba(K);
        std::vector<double> transitionProba(K);
        std::vector<double> likelihoods(K);
        int ancestor;
        
        // Loop of theta samples
        for (int theta_idx = 0; theta_idx < numberOfThetaSamples; ++theta_idx) {
            
            // Assign gamma vector
            gamma.assign(gammaList + theta_idx * K, gammaList + (theta_idx + 1) * K);
            
            // Ancestors for theta sample theta_idx
            ancestorsIndexes = stratified_resampling(generator, weightsList + theta_idx * numberOfStateSamples, numberOfStateSamples);

            // Samples descendants for theta sample theta_idx
            if (t > 0) {
                for (int traj_idx = 0; traj_idx < numberOfStateSamples; ++traj_idx) {
                    ancestor                  = *(ancestorsList + theta_idx * numberOfStateSamples + ancestorsIndexes[traj_idx]);
                    gammaAdaptedProba         = isNotEqual(mapping, previousStimulus, numberOfStimuli, K, mapping[previousStimulus * K + ancestor]) * gamma;
                    gammaAdaptedProba         = gammaAdaptedProba/sum(gammaAdaptedProba);
                    transitionProba           = gammaAdaptedProba * tauList[theta_idx];
                    transitionProba[ancestor] = 1 - tauList[theta_idx];
                    *(currentRes + theta_idx * numberOfStateSamples + traj_idx) = Sample_Discrete_Distribution(generator, transitionProba);
                    
                }
            }
            else
            {
                for (int traj_idx = 0; traj_idx < numberOfStateSamples; ++traj_idx) {
                    *(currentRes + theta_idx * numberOfStateSamples + traj_idx) = Sample_Discrete_Distribution(generator, gamma);
                }
            }
        }
        
        return;
    }
    
    
    void guidedUpdateStep(double* logApproxLikelihood, double* logThetaWeights, int* currentRes, double* gammaList, double* betaList, double* tauList, int const t, int* ancestorsList, double* weightsList, int* mapping, int const previousStimulus, int const currentStimulus, int const reward, int const action, int const numberOfThetaSamples, int const numberOfStateSamples, int const K, int const numberOfStimuli)
    {
        assert(t > 0);
        std::vector<double> gamma(K);
        std::vector<int> ancestorsIndexes(numberOfStateSamples);
        std::vector<double> gammaAdaptedProba(K);
        std::vector<double> transitionProba(K);
        std::vector<double> likelihoods(K);
        int ancestor;
        double weightsSum;
        
        std::vector<bool> previousStates = isEqual(mapping, currentStimulus, numberOfStimuli, K, action);
        std::vector<bool> positiveStates = adapted_logical_xor(previousStates, reward);
        
        // Loop of theta samples
        for (int theta_idx = 0; theta_idx < numberOfThetaSamples; ++theta_idx) {
            
            // Assign gamma vector
            gamma.assign(gammaList + theta_idx * K, gammaList + (theta_idx + 1) * K);
            
            // Ancestors for theta sample i
            ancestorsIndexes = stratified_resampling(generator, weightsList + theta_idx * numberOfStateSamples, numberOfStateSamples);
            
            // Likelihoods for theta sample i
            for (int k = 0; k < K; ++k) { likelihoods[k] = positiveStates[k] * betaList[theta_idx] + (1 - positiveStates[k]) * (1 - betaList[theta_idx]);}
            
            // Sample Descendants for theta sample i
            if (t > 1)
            {
                for (int traj_idx = 0; traj_idx < numberOfStateSamples; ++traj_idx)
                {
                    ancestor                                                     = *(ancestorsList + theta_idx * numberOfStateSamples + ancestorsIndexes[traj_idx]);
                    gammaAdaptedProba                                            = isNotEqual(mapping, previousStimulus, numberOfStimuli, K, mapping[previousStimulus * K + ancestor]) * gamma;
                    gammaAdaptedProba                                            = gammaAdaptedProba/sum(gammaAdaptedProba);
                    transitionProba                                              = gammaAdaptedProba * tauList[theta_idx];
                    transitionProba[ancestor]                                    = (1 - tauList[theta_idx]);
                    transitionProba                                              = transitionProba * likelihoods;
                    *(currentRes + theta_idx * numberOfStateSamples + traj_idx)  = Sample_Discrete_Distribution(generator, transitionProba);
                    *(weightsList + theta_idx * numberOfStateSamples + traj_idx) = sum(transitionProba);
                }
                weightsSum                          = sum(weightsList + theta_idx * numberOfStateSamples, numberOfStateSamples);
                *(logApproxLikelihood + theta_idx) += log(weightsSum/numberOfStateSamples);
                *(logThetaWeights + theta_idx)     += log(weightsSum/numberOfStateSamples);
                divide(weightsList + theta_idx * numberOfStateSamples, weightsSum, numberOfStateSamples);
            }
            else
            {
                transitionProba                    = gamma * likelihoods;
                *(logApproxLikelihood + theta_idx) = log(sum(transitionProba));
                *(logThetaWeights + theta_idx)     = log(sum(transitionProba));
                for (int traj_idx = 0; traj_idx < numberOfStateSamples; ++ traj_idx)
                {
                    *(currentRes + theta_idx * numberOfStateSamples + traj_idx)  = Sample_Discrete_Distribution(generator, transitionProba);
                    *(weightsList + theta_idx * numberOfStateSamples + traj_idx) = 1./numberOfStateSamples;
                }
            }
        }
        
        return ;
    }
    
    /* Launch guided SMC when the parameters do not represent the actual parameters anymore, backward inference is performed to find an approximation of the likelihood for the M-H acceptance ratio */
    
    double guidedSMC(int* stateRes, double* weightsRes, double* const gamma, int const K, double const beta, double const tau, int* const mapping, int const numberOfStimuli, int* const stimuli, int* const rewards, int const T, int* const actions, int const numberOfSamples)
    {
        // Instantiate output vector
        double logApproximateLikelihood = 0;
        
        // Initialise vectors
        std::vector<bool> previousStates(K);
        std::vector<bool> positiveStates(K);
        std::vector<double> likelihoods(K);
        std::vector<double> transitionProba(K);
        std::vector<double> gammaAdaptedProba(K);
        std::vector<int> ancestors_indexes(numberOfSamples);
        std::vector<int> ancestors(numberOfSamples);
        double weightsSum;
        
        // Loop over the time
        for (int t = 0; t < T; t++) {
            // Precompute statistics
            previousStates = isEqual(mapping, stimuli[t], numberOfStimuli, K, actions[t]);
            positiveStates = adapted_logical_xor(previousStates, rewards[t]);
            
            // Likelihoods
            for (int i = 0; i < K; ++i) { likelihoods[i] = positiveStates[i] * beta + (1 - positiveStates[i]) * (1 - beta);}
            // Sample descendants
            if (t > 0)
            {
                ancestors.assign(&stateRes[0], &stateRes[0] + numberOfSamples);
                ancestors_indexes = stratified_resampling(generator, weightsRes, numberOfSamples);
                for (int i = 0; i < numberOfSamples; ++i) {
                    int ancestor              = ancestors[ancestors_indexes[i]];
                    gammaAdaptedProba         = isNotEqual(mapping,stimuli[t-1], numberOfStimuli, K, mapping[stimuli[t-1]*K + ancestor]) * gamma;
                    gammaAdaptedProba         = gammaAdaptedProba/sum(gammaAdaptedProba);
                    transitionProba           = gammaAdaptedProba * tau;
                    transitionProba[ancestor] = (1 - tau);
                    transitionProba           = transitionProba * likelihoods;
                    *(stateRes + i)           = Sample_Discrete_Distribution(generator, transitionProba);
                    *(weightsRes + i)         = sum(transitionProba);
                }
                weightsSum                = sum(weightsRes,numberOfSamples);
                logApproximateLikelihood += log(weightsSum/numberOfSamples);
                divide(weightsRes, weightsSum, numberOfSamples);
            }
            else
            {
                transitionProba           = (gamma * likelihoods);
                logApproximateLikelihood  = log(sum(transitionProba));
                for (int i = 0; i < numberOfSamples; ++i)
                {
                    *(stateRes + i)   = Sample_Discrete_Distribution(generator, transitionProba);
                    *(weightsRes + i) = 1./numberOfSamples;
                }
            }
        }
        
        return logApproximateLikelihood;
    }
    
}
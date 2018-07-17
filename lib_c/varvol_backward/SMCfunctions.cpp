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
    
    void bootstrapUpdateStep(int* currentStateRes, double* currentTauRes, double* const gammaList, double* const betaList, double* const nuList, double const tauDefault, int const t, int* const ancestorsStateList, double* const ancestorsTauList, double* weightsList, int* mapping, int const previousStimulus, int numberOfThetaSamples, int numberOfStateSamples, int const K, int const numberOfStimuli)
    {
        std::vector<double> gamma(K);
        std::vector<int> ancestorsIndexes(numberOfStateSamples);
        std::vector<double> gammaAdaptedProba(K);
        std::vector<double> transitionProba(K);
        int ancestorState;
        double ancestorTau;
        double tauSampleTmp;
        
        // Loop of theta samples
        for (int theta_idx = 0; theta_idx < numberOfThetaSamples; ++theta_idx) {
            
            // Assign gamma vector
            gamma.assign(gammaList + theta_idx * K, gammaList + (theta_idx + 1) * K);
            
            // Ancestor for theta sample theta_idx
            ancestorsIndexes = stratified_resampling(generator, weightsList + theta_idx * numberOfStateSamples, numberOfStateSamples);
            
            // Sample descendants for theta sample theta_idx
            if (t > 0) {
                for (int traj_idx = 0; traj_idx < numberOfStateSamples; ++traj_idx) {
                    ancestorState                                                    = *(ancestorsStateList + theta_idx * numberOfStateSamples + ancestorsIndexes[traj_idx]);
                    ancestorTau                                                      = *(ancestorsTauList + theta_idx * numberOfStateSamples + ancestorsIndexes[traj_idx]);
                    tauSampleTmp                                                     = Sample_Normal_Distribution(generator, ancestorTau, sqrt(nuList[theta_idx]));
                    *(currentTauRes + theta_idx * numberOfStateSamples + traj_idx)   = (tauSampleTmp > 0) * (tauSampleTmp < 1) * tauSampleTmp + (tauSampleTmp > 1);
                    gammaAdaptedProba                                                = isNotEqual(mapping, previousStimulus, numberOfStimuli, K, mapping[previousStimulus * K + ancestorState]) * gamma;
                    gammaAdaptedProba                                                = gammaAdaptedProba/sum(gammaAdaptedProba);
                    transitionProba                                                  = gammaAdaptedProba * (*(currentTauRes + theta_idx * numberOfStateSamples + traj_idx));
                    transitionProba[ancestorState]                                   = 1 - (*(currentTauRes + theta_idx * numberOfStateSamples + traj_idx));
                    *(currentStateRes + theta_idx * numberOfStateSamples + traj_idx) = Sample_Discrete_Distribution(generator, transitionProba);
                }
            }
            else
            {
                for (int traj_idx = 0; traj_idx < numberOfStateSamples; ++traj_idx) {
                    *(currentStateRes + theta_idx * numberOfStateSamples + traj_idx) = Sample_Discrete_Distribution(generator, gamma);
                    *(currentTauRes + theta_idx * numberOfStateSamples + traj_idx)   = tauDefault;
                }
            }
        }
        
        return;
    }
    
    
    void guidedUpdateStep(double* logApproxLikelihood, double* logThetaWeights, int* currentStateRes, double* currentTauRes, double* const gammaList, double* const betaList, double* const nuList, double const tauDefault, int const t, int* const ancestorsStateList, double* const ancestorsTauList, double* weightsList, int* mapping, int const previousStimulus, int const currentStimulus, int const reward, int const action, int numberOfThetaSamples, int numberOfStateSamples, int const K, int const numberOfStimuli)
    {
        assert(t > 0);
        std::vector<double> gamma(K);
        std::vector<int> ancestorsIndexes(numberOfStateSamples);
        std::vector<double> gammaAdaptedProba(K);
        std::vector<double> transitionProba(K);
        std::vector<double> likelihoods(K);
        int ancestorState;
        double ancestorTau;
        double tauSampleTmp;
        double weightsSum;
        
        std::vector<bool> previousStates = isEqual(mapping, currentStimulus, numberOfStimuli, K, action);
        std::vector<bool> positiveStates = adapted_logical_xor(previousStates, reward);
        
        // Loop of theta samples
        for (int theta_idx = 0; theta_idx < numberOfThetaSamples; ++theta_idx) {
            
            // Assign gamma vector
            gamma.assign(gammaList + theta_idx * K, gammaList + (theta_idx + 1) * K);
            
            // Ancestor for theta sample theta_idx
            ancestorsIndexes = stratified_resampling(generator, weightsList + theta_idx * numberOfStateSamples, numberOfStateSamples);
            
            // Likelihood for theta sample theta_idx
            for (int k = 0; k < K; ++k) {  likelihoods[k] = positiveStates[k] * betaList[theta_idx] + (1 - positiveStates[k]) * (1 - betaList[theta_idx]) ;}
            
            // Sample Descendants for theta sample theta_idx
            if (t > 1) {
                for (int traj_idx = 0; traj_idx < numberOfStateSamples; ++traj_idx) {
                    ancestorState                                                    = *(ancestorsStateList + numberOfStateSamples * theta_idx + ancestorsIndexes[traj_idx]);
                    ancestorTau                                                      = *(ancestorsTauList + numberOfStateSamples * theta_idx + ancestorsIndexes[traj_idx]);
                    tauSampleTmp                                                     = Sample_Normal_Distribution(generator, ancestorTau, sqrt(nuList[theta_idx]));
                    *(currentTauRes + theta_idx * numberOfStateSamples + traj_idx)   = (tauSampleTmp > 0) * (tauSampleTmp < 1) * tauSampleTmp + (tauSampleTmp > 1);
                    gammaAdaptedProba                                                = isNotEqual(mapping, previousStimulus, numberOfStimuli, K, mapping[previousStimulus * K + ancestorState]) * gamma;
                    gammaAdaptedProba                                                = gammaAdaptedProba/sum(gammaAdaptedProba);
                    transitionProba                                                  = gammaAdaptedProba * (*(currentTauRes + theta_idx * numberOfStateSamples + traj_idx));
                    transitionProba[ancestorState]                                   = 1 - (*(currentTauRes + theta_idx * numberOfStateSamples + traj_idx));
                    transitionProba                                                  = transitionProba * likelihoods;
                    *(currentStateRes + theta_idx * numberOfStateSamples + traj_idx) = Sample_Discrete_Distribution(generator, transitionProba);
                    *(weightsList + theta_idx * numberOfStateSamples + traj_idx)     = sum(transitionProba);
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
                for (int traj_idx = 0; traj_idx < numberOfStateSamples; ++traj_idx) {
                    *(currentStateRes + theta_idx * numberOfStateSamples + traj_idx) = Sample_Discrete_Distribution(generator, transitionProba);
                    *(weightsList + theta_idx * numberOfStateSamples + traj_idx)     = 1./numberOfStateSamples;
                    *(currentTauRes + theta_idx * numberOfStateSamples + traj_idx)   = tauDefault;
                }
            }
        }
        return;
    }
    
    
    /* Launch guided SMC when the parameters do not represent the actual parameters anymore, backward inference is performed to find an approximation of the likelihood for the M-H acceptance ratio */
    
    double guidedSMC(int* stateRes, double* tauRes, double* weightsRes, double* const gamma, int const K, double const beta, double const nu, double const tauDefault, int* const mapping, int const numberOfStimuli, int* const stimuli, int* const rewards, int const T, int* const actions, int const numberOfSamples)
    {
        // Instantiate output vector
        double logApproximateLikelihood = 0;
        
        // Initialise vectors
        std::vector<bool> previousStates(K);
        std::vector<bool> positiveStates(K);
        std::vector<double> likelihoods(K);
        std::vector<double> gammaAdaptedProba(K);
        std::vector<double> transitionProba(K);
        std::vector<int> ancestors_indexes(numberOfSamples);
        std::vector<int> ancestorsStates(numberOfSamples);
        std::vector<double> ancestorsTau(numberOfSamples);
        double weightsSum;
        int ancestorState;
        double ancestorTau;
        double tauSampleTmp;
        
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
                ancestorsStates.assign(&stateRes[0], &stateRes[0] + numberOfSamples);
                ancestorsTau.assign(&tauRes[0], &tauRes[0] + numberOfSamples);
                ancestors_indexes = stratified_resampling(generator, weightsRes, numberOfSamples);
                for (int traj_idx = 0; traj_idx < numberOfSamples; ++traj_idx) {
                    ancestorTau                    = ancestorsTau[ancestors_indexes[traj_idx]];
                    ancestorState                  = ancestorsStates[ancestors_indexes[traj_idx]];
                    tauSampleTmp                   = Sample_Normal_Distribution(generator, ancestorTau, sqrt(nu));
                    *(tauRes + traj_idx)           = (tauSampleTmp>0)*(tauSampleTmp<1)*tauSampleTmp + (tauSampleTmp>1);
                    gammaAdaptedProba              = isNotEqual(mapping,stimuli[t-1], numberOfStimuli, K, mapping[stimuli[t-1]*K + ancestorState]) * gamma;
                    gammaAdaptedProba              = gammaAdaptedProba/sum(gammaAdaptedProba);
                    transitionProba                = gammaAdaptedProba * (*(tauRes + traj_idx));
                    transitionProba[ancestorState] = (1 - *(tauRes + traj_idx));
                    transitionProba                = transitionProba * likelihoods;
                    *(stateRes + traj_idx)         = Sample_Discrete_Distribution(generator, transitionProba);
                    *(weightsRes + traj_idx)       = sum(transitionProba);
                }
                weightsSum                = sum(weightsRes, numberOfSamples);
                logApproximateLikelihood += log(weightsSum/numberOfSamples);
                divide(weightsRes, weightsSum, numberOfSamples);
            }
            else
            {
                transitionProba           = (gamma * likelihoods);
                logApproximateLikelihood  = log(sum(transitionProba));
                for (int traj_idx = 0; traj_idx < numberOfSamples; ++traj_idx) {
                    *(stateRes + traj_idx)   = Sample_Discrete_Distribution(generator, transitionProba);
                    *(weightsRes + traj_idx) = 1./numberOfSamples;
                    *(tauRes + traj_idx)     = tauDefault;
                }
            }
        }
        return logApproximateLikelihood;
    }
    
}
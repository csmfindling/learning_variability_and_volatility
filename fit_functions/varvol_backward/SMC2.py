##### Model with constant volatility with C++ and python

#Libraries
import numpy as np
import sys
sys.path.append('../../utils/')
sys.path.append('../../lib_c/')
from varvol_backward import smc_c
import get_mapping
import useful_functions
from scipy.stats import beta as betalib
from scipy.stats import norm as normlib
import matplotlib.pyplot as plt
import time
import numpy

def SMC2(td, show_progress=False, numberOfStateSamples=1000, numberOfThetaSamples=1000, coefficient = .5):

    print('\n')
    print('Varying Volatility Model'); print('\n')

    #Start timer
    start_time_multi = time.time()

    # Extract parameters from task description
    stimuli         = td['S']                                                   # Sequence of Stimuli
    Z_true          = td['Z']                                                   # Sequence of Task Sets
    numberOfActions = td['action_num']                                          # Number of Actions possible
    numberOfStimuli = td['state_num']                                           # Number of states or stimuli
    K               = np.prod(np.arange(numberOfActions+1)[-numberOfStimuli:])  # Number of possible Task Sets
    numberOfTrials  = len(Z_true)                                               # Number of Trials
    
    # Sampling and prior settings
    betaPrior  = np.array([1, 1])                   # Prior on Beta, the feedback noise parameter
    nuPrior    = np.array([3, 1e-3])                # Prior on Nu, the variance on the projected gaussian random walk
    gammaPrior = numpy.ones(K)                      # Prior on Gamma, the Dirichlet parameter
    try:
        tauDefault  = td['tau'][0]
    except:
        tauDefault  = td['tau']
    
    # Mapping from task set to correct action per stimulus
    mapping = get_mapping.Get_TaskSet_Stimulus_Mapping(state_num=numberOfStimuli, action_num=numberOfActions).T
    actions          = np.zeros(numberOfTrials) - 1
    rewards          = np.zeros(numberOfTrials, dtype=bool)

    # Keep track of probability correct/exploration after switches
    countPerformance      = np.zeros(numberOfTrials)       # Number of correct actions after i trials
    countExploration      = np.zeros(numberOfTrials)       # Number of exploratory actions after i trials
    correct_before_switch = np.empty(0)                    # The correct task set before switch
    tsProbability         = np.zeros([numberOfTrials, K])
    volTracking           = np.zeros(numberOfTrials)       # Volatility with time
    volStdTracking        = np.zeros(numberOfTrials)
    nuTracking            = np.zeros(numberOfTrials)
    nuStdTracking         = np.zeros(numberOfTrials)
    betaTracking          = np.zeros(numberOfTrials)
    betaStdTracking       = np.zeros(numberOfTrials)
    acceptanceProba       = 0.                             # Acceptance proba
    acceptance_list       = [1.]
    time_list             = [start_time_multi]

    # SMC particles initialisation
    betaSamples                   = np.random.beta(betaPrior[0], betaPrior[1], numberOfThetaSamples)
    nuSamples                     = useful_functions.sample_inv_gamma(nuPrior[0], nuPrior[1], numberOfThetaSamples)
    gammaSamples                  = np.random.dirichlet(gammaPrior, numberOfThetaSamples)
    logThetaWeights               = np.zeros(numberOfThetaSamples)
    logThetaLks                   = np.zeros(numberOfThetaSamples)
    currentStateSamples           = np.zeros([numberOfThetaSamples, numberOfStateSamples], dtype=np.intc)
    currentTauSamples             = np.zeros([numberOfThetaSamples, numberOfStateSamples], dtype=np.double)
    ancestorStateSamples          = np.zeros([numberOfThetaSamples, numberOfStateSamples], dtype=np.intc)
    ancestorTauSamples            = np.zeros([numberOfThetaSamples, numberOfStateSamples], dtype=np.double)
    ancestorsWeights              = np.ones([numberOfThetaSamples, numberOfStateSamples])/numberOfStateSamples
    unnormalisedAncestorsWeights  = np.ones([numberOfThetaSamples, numberOfStateSamples])
    essList                       = np.zeros(numberOfTrials)
    tasksetLikelihood             = np.zeros(K)

    # Guided SMC variables
    betaSamplesNew               = np.zeros(numberOfThetaSamples)
    nuSamplesNew                 = np.zeros(numberOfThetaSamples)
    gammaSamplesNew              = np.zeros([numberOfThetaSamples, K])
    stateSamplesNew              = np.zeros([numberOfThetaSamples, numberOfStateSamples], dtype=np.intc)
    tauSamplesNew                = np.zeros([numberOfThetaSamples, numberOfStateSamples])
    weightsSamplesNew            = np.zeros([numberOfThetaSamples, numberOfStateSamples])
    logThetaLksNew               = np.zeros(numberOfThetaSamples)
    dirichletParamCandidates     = np.zeros(K)
    stateSamplesCandidates       = np.zeros(numberOfStateSamples, dtype=np.intc)
    tauSamplesCandidates         = np.zeros(numberOfStateSamples, dtype =np.double)
    weightsSamplesCandidates     = np.zeros(numberOfStateSamples)
    idxTrajectories              = np.zeros(numberOfThetaSamples)


    # Plot progress
    if show_progress : plt.figure(figsize=(12,9))

    # Loop over trials
    for T in range(numberOfTrials):

        # Print progress
        if (T+1) % 10 == 0 : sys.stdout.write(' ' + str(T+1));sys.stdout.flush(); time_list.append(time.time() - start_time_multi);
        if (T+1) % 100 == 0: print ('\n')

        if T > 0:
            # Update theta weights
            smc_c.guidedUpdateStep_c(logThetaLks, logThetaWeights, np.ascontiguousarray(currentStateSamples), currentTauSamples, gammaSamples, betaSamples/2. + 1./2, nuSamples, tauDefault, T, np.ascontiguousarray(ancestorStateSamples),\
                                        ancestorTauSamples, ancestorsWeights, np.ascontiguousarray(mapping), stimuli[T-2], stimuli[T-1], rewards[T-1], actions[T-1])
            ancestorTauSamples   = np.array(currentTauSamples)
            ancestorStateSamples = np.array(currentStateSamples)

        # Degeneray criterion
        logEss     = 2 * useful_functions.log_sum(logThetaWeights) - useful_functions.log_sum(2 * logThetaWeights)
        essList[T] = np.exp(logEss)

        # Move step
        normalisedThetaWeights = useful_functions.to_normalized_weights(logThetaWeights)
        if (essList[T] < coefficient * numberOfThetaSamples) and (acceptance_list[-1] > 0.05):
            acceptanceProba          = 0.
            betaMu                   = np.sum(normalisedThetaWeights*betaSamples)
            betaVar                  = np.sum(normalisedThetaWeights * (betaSamples - betaMu)**2)
            betaAlpha                = ((1 - betaMu)/betaVar - 1/betaMu) * betaMu**2
            betaBeta                 = betaAlpha * (1/betaMu - 1)
            assert(betaAlpha > 0); assert(betaBeta > 0)
            nuMu                     = np.sum(normalisedThetaWeights * nuSamples)
            nuVar                    = np.sum(normalisedThetaWeights * (nuSamples - nuMu)**2)
            nuAlpha                  = nuMu**2/nuVar + 2
            nuBeta                   = nuMu * (nuAlpha - 1)
            assert(nuAlpha > 0); assert(nuBeta > 0)
            dirichletMeans           = np.sum(normalisedThetaWeights*gammaSamples.T, axis=1)
            dirichletVar             = np.sum(normalisedThetaWeights*(gammaSamples**2).T, axis=1) - dirichletMeans**2
            dirichletPrecision       = np.sum(dirichletMeans - dirichletMeans**2)/(np.sum(dirichletVar)) - 1
            dirichletParamCandidates = dirichletMeans * dirichletPrecision
            assert((dirichletParamCandidates>0).all())
            
            idxTrajectories          = useful_functions.stratified_resampling(normalisedThetaWeights)

            for theta_idx in range(numberOfThetaSamples):
                nuCandidate                    = useful_functions.sample_inv_gamma(nuAlpha, nuBeta)
                betaCandidate                  = np.random.beta(betaAlpha, betaBeta)
                gammaCandidate                 = np.random.dirichlet(dirichletParamCandidates)

                # Launch guidedSMC
                logLksCandidate                = smc_c.guidedSmc_c(np.ascontiguousarray(stateSamplesCandidates), tauSamplesCandidates, weightsSamplesCandidates, gammaCandidate, betaCandidate/2. + 1./2, nuCandidate, tauDefault, np.ascontiguousarray(mapping), \
                                                            np.ascontiguousarray(stimuli[:T], dtype=np.intc), np.ascontiguousarray(rewards[:T], dtype=np.intc), np.ascontiguousarray(actions[:T], dtype=np.intc), numberOfStateSamples)

                # Update a trajectory
                idx_traj                       = idxTrajectories[theta_idx]
                priorsLogRatio                 = useful_functions.log_invgamma_pdf(nuCandidate, nuPrior[0], nuPrior[1]) + useful_functions.log_dirichlet_pdf(gammaCandidate, gammaPrior) - \
                                                    useful_functions.log_invgamma_pdf(nuSamples[idx_traj], nuPrior[0], nuPrior[1]) - useful_functions.log_dirichlet_pdf(gammaSamples[idx_traj], gammaPrior)

                transLogRatio                  = useful_functions.log_invgamma_pdf(nuSamples[idx_traj], nuAlpha, nuBeta) + useful_functions.log_beta_pdf(betaSamples[idx_traj], betaAlpha, betaBeta) + useful_functions.log_dirichlet_pdf(gammaSamples[idx_traj], dirichletParamCandidates) - \
                                                    useful_functions.log_invgamma_pdf(nuCandidate, nuAlpha, nuBeta) - useful_functions.log_beta_pdf(betaCandidate, betaAlpha, betaBeta) - useful_functions.log_dirichlet_pdf(gammaCandidate, dirichletParamCandidates)

                logLkdRatio                    = logLksCandidate - logThetaLks[idx_traj]
                logAlpha                       = min(0, priorsLogRatio + transLogRatio + logLkdRatio)

                U                              = np.random.rand()

                # Accept or Reject
                if np.log(U) < logAlpha:
                    acceptanceProba             += 1.
                    betaSamplesNew[theta_idx]    = betaCandidate
                    nuSamplesNew[theta_idx]      = nuCandidate
                    gammaSamplesNew[theta_idx]   = gammaCandidate
                    stateSamplesNew[theta_idx]   = stateSamplesCandidates
                    tauSamplesNew[theta_idx]     = tauSamplesCandidates
                    weightsSamplesNew[theta_idx] = weightsSamplesCandidates
                    logThetaLksNew[theta_idx]    = logLksCandidate

                else:
                    betaSamplesNew[theta_idx]    = betaSamples[idx_traj]
                    nuSamplesNew[theta_idx]      = nuSamples[idx_traj]
                    gammaSamplesNew[theta_idx]   = gammaSamples[idx_traj]
                    stateSamplesNew[theta_idx]   = ancestorStateSamples[idx_traj]
                    tauSamplesNew[theta_idx]     = ancestorTauSamples[idx_traj]
                    weightsSamplesNew[theta_idx] = ancestorsWeights[idx_traj]
                    logThetaLksNew[theta_idx]    = logThetaLks[idx_traj]

            print ('\n')
            print ('acceptance ratio is ')
            print (acceptanceProba/numberOfThetaSamples)
            print ('\n')
            acceptance_list.append(acceptanceProba/numberOfThetaSamples)

            ancestorsWeights              = np.array(weightsSamplesNew)
            logThetaLks                   = np.array(logThetaLksNew)
            logThetaWeights               = np.zeros(numberOfThetaSamples)
            ancestorStateSamples          = np.array(stateSamplesNew)
            ancestorTauSamples            = np.array(tauSamplesNew)
            betaSamples                   = np.array(betaSamplesNew)
            nuSamples                     = np.array(nuSamplesNew)
            gammaSamples                  = np.array(gammaSamplesNew)
            normalisedThetaWeights        = useful_functions.to_normalized_weights(logThetaWeights)

        # Launch bootstrap update
        smc_c.bootstrapUpdateStep_c(currentStateSamples, currentTauSamples, gammaSamples, betaSamples/2. + 1./2, nuSamples, tauDefault, T, \
                                            np.ascontiguousarray(ancestorStateSamples, dtype=np.intc), ancestorTauSamples, ancestorsWeights, np.ascontiguousarray(mapping), stimuli[T-1])

        # Take decision
        for ts_idx in range(K):
            tsProbability[T, ts_idx] = np.sum(normalisedThetaWeights * np.sum((currentStateSamples == ts_idx), axis = 1)) # Todo : change!!! take out currentAncestorsWeights

        volTracking[T]     = np.sum(normalisedThetaWeights * (np.sum(currentTauSamples, axis=1)/numberOfStateSamples))
        volStdTracking[T]  = np.sum(normalisedThetaWeights * (np.sum(currentTauSamples**2, axis=1)/numberOfStateSamples)) - volTracking[T]**2
        nuTracking[T]      = np.sum(normalisedThetaWeights * nuSamples)
        nuStdTracking[T]   = np.sum(normalisedThetaWeights * (nuSamples - nuTracking[T])**2)

        betaTracking[T]    = np.sum(normalisedThetaWeights * betaSamples)
        betaStdTracking[T] = np.sum(normalisedThetaWeights * (betaSamples - betaTracking[T])**2)
        rewards[T]            = td['reward'][T]
        actions[T]            = td['A_chosen'][T]

        if show_progress:
            plt.subplot(3,2,1)
            plt.imshow(tsProbability[:T].T, aspect='auto'); plt.hold(True)
            plt.plot(Z_true[:T], 'w--')
            plt.axis([0, T-1, 0, K-1]) 
            plt.xlabel('trials')
            plt.ylabel('p(TS|past) at current time')

            plt.subplot(3,2,2)
            plt.plot(volTracking[:T], 'b'); plt.hold(True)
            plt.fill_between(np.arange(T),volTracking[:T]-volStdTracking[:T], volTracking[:T]+volStdTracking[:T],facecolor=[.5,.5,1], color=[.5,.5,1]); 
            plt.plot(td['tau'], 'b--', linewidth=2)
            plt.axis([0, T-1, 0, .5])
            plt.hold(False)
            plt.xlabel('trials')
            plt.ylabel('Volatility')

            plt.subplot(3,2,3)
            x = np.linspace(0.01,.99,100)
            plt.plot(x, normlib.pdf(x, nuTracking[T], nuStdTracking[T]), 'b'); plt.hold(True)
            plt.plot([nuTracking[T], nuTracking[T]], plt.gca().get_ylim(),'b', linewidth=2)
            plt.plot(x, normlib.pdf(x, betaTracking[T], betaStdTracking[T]), 'r')
            plt.plot([betaTracking[T], betaTracking[T]], plt.gca().get_ylim(),'r', linewidth=2)
            plt.plot([td['beta'], td['beta']], plt.gca().get_ylim(), 'r--', linewidth=2)
            plt.hold(False)
            plt.xlabel('Parameters')
            plt.ylabel('Gaussian pdf')

            plt.subplot(3,2,4)
            plt.plot(np.arange(T)+1, essList[:T], 'g', linewidth=2); plt.hold(True)
            plt.plot(plt.gca().get_xlim(), [coefficient*numberOfThetaSamples,coefficient*numberOfThetaSamples], 'g--', linewidth=2);
            plt.axis([0,T-1,0,numberOfThetaSamples]);
            plt.hold(False);
            plt.xlabel('trials');
            plt.ylabel('ESS');

            plt.subplot(3,2,5);
            plt.plot(np.divide(countPerformance[:T], np.arange(T)+1), 'k--', linewidth=2); plt.hold(True)
            plt.axis([0,T-1,0,1]);
            plt.hold(False);
            plt.xlabel('Trials');
            plt.ylabel('Performance');

            plt.draw()

    elapsed_time = time.time() - start_time_multi

    return [td, nuSamples, nuTracking, nuStdTracking, volTracking, volTracking, betaSamples, betaTracking, tsProbability, betaStdTracking, gammaSamples, countPerformance, actions, acceptance_list, essList, time_list, elapsed_time]









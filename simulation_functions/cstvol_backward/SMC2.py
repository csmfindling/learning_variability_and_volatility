##### Model with constant volatility with C++ and python

#Libraries
import sys
sys.path.append('../../utils/')
sys.path.append('../../lib_c/')
from cstvol_backward import smc_c
import numpy as np
import get_mapping
import useful_functions
import sys
from scipy.stats import beta as betalib
from scipy.stats import norm as normlib
import matplotlib.pyplot as plt
import time
import numpy

def SMC2(td, show_progress=True, numberOfStateSamples=1000, numberOfThetaSamples=1000, coefficient = .5, beta_softmax=None):

    print('Constant Volatility Model'); print('\n')

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
    betaPrior   = np.array([1, 1])                 # Prior on Beta, the feedback noise parameter
    tauPrior    = np.array([1, 1])                 # Prior on Tau, the switch parameter (the volatility)
    gammaPrior  = numpy.ones(K)                    # Prior on Gamma, the Dirichlet parameter
    
    # Mapping from task set to correct action per stimulus
    mapping = get_mapping.Get_TaskSet_Stimulus_Mapping(state_num=numberOfStimuli, action_num=numberOfActions).T

    # Probabilities of every actions updated at every time step -> Used to take the decision
    actionLikelihood = np.zeros(numberOfActions)  # For 1 observation, likelihood of the action. Requires a marginalisation over all task sets
    actions          = np.zeros(numberOfTrials) - 1
    rewards          = np.zeros(numberOfTrials, dtype=bool)

    # Keep track of probability correct/exploration after switches
    countPerformance      = np.zeros(numberOfTrials)    # Number of correct actions after i trials
    countExploration      = np.zeros(numberOfTrials)    # Number of exploratory actions after i trials
    correct_before_switch = np.empty(0)                 # The correct task set before switch
    tsProbability         = np.zeros([numberOfTrials, K])
    acceptanceProba       = 0.
    volTracking           = np.zeros(numberOfTrials)
    volStdTracking        = np.zeros(numberOfTrials)
    betaTracking          = np.zeros(numberOfTrials)
    betaStdTracking       = np.zeros(numberOfTrials)
    acceptance_list       = [1.]
    time_list             = [start_time_multi]

    # SMC particles initialisation
    betaSamples                  = np.random.beta(betaPrior[0], betaPrior[1], numberOfThetaSamples)
    tauSamples                   = np.random.beta(tauPrior[0], tauPrior[1], numberOfThetaSamples)
    gammaSamples                 = np.random.dirichlet(gammaPrior, numberOfThetaSamples)
    logThetaWeights              = np.zeros(numberOfThetaSamples)
    logThetaLks                  = np.zeros(numberOfThetaSamples)
    currentSamples               = np.zeros([numberOfThetaSamples, numberOfStateSamples], dtype=np.intc)
    ancestorSamples              = np.zeros([numberOfThetaSamples, numberOfStateSamples], dtype=np.intc)
    ancestorsWeights             = np.ones([numberOfThetaSamples, numberOfStateSamples])/numberOfStateSamples
    unnormalisedAncestorsWeights = np.ones([numberOfThetaSamples, numberOfStateSamples])
    essList                      = np.zeros(numberOfTrials)
    tasksetLikelihood            = np.zeros(K)

    # Guided SMC variables
    betaSamplesNew               = np.zeros(numberOfThetaSamples)
    tauSamplesNew                = np.zeros(numberOfThetaSamples)
    gammaSamplesNew              = np.zeros([numberOfThetaSamples, K])
    stateSamplesNew              = np.zeros([numberOfThetaSamples, numberOfStateSamples], dtype=np.intc)
    weightsSamplesNew            = np.zeros([numberOfThetaSamples, numberOfStateSamples])
    logThetaLksNew               = np.zeros(numberOfThetaSamples)
    dirichletParamCandidates     = np.zeros(K)
    stateSamplesCandidates       = np.zeros(numberOfStateSamples, dtype=np.intc)
    weightsSamplesCandidates     = np.zeros(numberOfStateSamples)
    idxTrajectories              = np.zeros(numberOfThetaSamples)

    # Plot progress
    if show_progress : plt.figure(figsize=(12,9)); plt.ion();

    # Loop over trials
    for T in range(numberOfTrials):

        # Print progress
        if (T+1) % 10 == 0 : sys.stdout.write(' ' + str(T+1));sys.stdout.flush(); time_list.append(time.time() - start_time_multi);
        if (T+1) % 100 == 0: print ('\n')

        if T > 0:
            smc_c.guidedUpdateStep_c(logThetaLks, logThetaWeights, np.ascontiguousarray(currentSamples), gammaSamples, betaSamples/2. + 1/2., tauSamples/2., T, np.ascontiguousarray(ancestorSamples), ancestorsWeights, \
                                                np.ascontiguousarray(mapping), stimuli[T-2], stimuli[T-1], rewards[T-1], actions[T-1])
            ancestorSamples = np.array(currentSamples)       

        # Degeneray criterion
        logEss     = 2 * useful_functions.log_sum(logThetaWeights) - useful_functions.log_sum(2 * logThetaWeights)
        essList[T] = np.exp(logEss)

        # Move step
        normalisedThetaWeights = useful_functions.to_normalized_weights(logThetaWeights)
        if (essList[T] < coefficient * numberOfThetaSamples) and (acceptance_list[-1] > 0.05):
            acceptanceProba          = 0.
            tauMu                    = np.sum(normalisedThetaWeights * tauSamples)
            tauVar                   = np.sum(normalisedThetaWeights * (tauSamples - tauMu)**2)
            tauAlpha                 = ((1 - tauMu)/tauVar - 1/tauMu) * tauMu**2
            tauBeta                  = tauAlpha * (1/tauMu - 1)
            assert(tauAlpha > 0); assert(tauBeta > 0)
            betaMu                   = np.sum(normalisedThetaWeights*betaSamples)
            betaVar                  = np.sum(normalisedThetaWeights * (betaSamples - betaMu)**2)
            betaAlpha                = ((1 - betaMu)/betaVar - 1/betaMu) * betaMu**2
            betaBeta                 = betaAlpha * (1/betaMu - 1)
            assert(betaAlpha > 0); assert(betaBeta > 0)
            dirichletMeans           = np.sum(normalisedThetaWeights*gammaSamples.T, axis=1)
            dirichletVar             = np.sum(normalisedThetaWeights*(gammaSamples**2).T, axis=1) - dirichletMeans**2
            dirichletPrecision       = np.sum(dirichletMeans - dirichletMeans**2)/(np.sum(dirichletVar)) - 1
            dirichletParamCandidates = dirichletMeans * dirichletPrecision
            assert((dirichletParamCandidates>0).all())
            
            idxTrajectories          = useful_functions.stratified_resampling(normalisedThetaWeights)

            for theta_idx in range(numberOfThetaSamples):
                tauCandidate                   = np.random.beta(tauAlpha, tauBeta)
                betaCandidate                  = np.random.beta(betaAlpha, betaBeta)
                gammaCandidate                 = np.random.dirichlet(dirichletParamCandidates)

                # Launch guidedSMC
                logLksCandidate                = smc_c.guidedSmc_c(np.ascontiguousarray(stateSamplesCandidates), weightsSamplesCandidates, gammaCandidate, betaCandidate/2. + 1/2., tauCandidate/2., np.ascontiguousarray(mapping), \
                                                            np.ascontiguousarray(stimuli[:T], dtype=np.intc), np.ascontiguousarray(rewards[:T], dtype=np.intc), np.ascontiguousarray(actions[:T], dtype=np.intc), numberOfStateSamples)

                # Update a trajectory
                idx_traj                       = idxTrajectories[theta_idx]

                priorsLogRatio                 = useful_functions.log_dirichlet_pdf(gammaCandidate, gammaPrior) - useful_functions.log_dirichlet_pdf(gammaSamples[idx_traj], gammaPrior) 

                transLogRatio                  = useful_functions.log_beta_pdf(tauSamples[idx_traj], tauAlpha, tauBeta) + useful_functions.log_beta_pdf(betaSamples[idx_traj], betaAlpha, betaBeta) + useful_functions.log_dirichlet_pdf(gammaSamples[idx_traj], dirichletParamCandidates) - \
                                                        useful_functions.log_beta_pdf(tauCandidate, tauAlpha, tauBeta) - useful_functions.log_beta_pdf(betaCandidate, betaAlpha, betaBeta) - useful_functions.log_dirichlet_pdf(gammaCandidate, dirichletParamCandidates)

                logLkdRatio                    = logLksCandidate - logThetaLks[idx_traj]

                logAlpha                       = min(0, priorsLogRatio + transLogRatio + logLkdRatio)

                U                              = np.random.rand()

                # Accept or Reject
                if np.log(U) < logAlpha:
                    acceptanceProba             += 1.
                    betaSamplesNew[theta_idx]    = betaCandidate
                    tauSamplesNew[theta_idx]     = tauCandidate
                    gammaSamplesNew[theta_idx]   = gammaCandidate
                    stateSamplesNew[theta_idx]   = stateSamplesCandidates
                    logThetaLksNew[theta_idx]    = logLksCandidate
                    weightsSamplesNew[theta_idx] = weightsSamplesCandidates
                else:
                    betaSamplesNew[theta_idx]    = betaSamples[idx_traj]
                    tauSamplesNew[theta_idx]     = tauSamples[idx_traj]
                    gammaSamplesNew[theta_idx]   = gammaSamples[idx_traj]
                    stateSamplesNew[theta_idx]   = ancestorSamples[idx_traj]
                    logThetaLksNew[theta_idx]    = logThetaLks[idx_traj]
                    weightsSamplesNew[theta_idx] = ancestorsWeights[idx_traj]

            print ('\n')
            print ('acceptance ratio is ')
            print (acceptanceProba/numberOfThetaSamples)
            print ('\n')
            acceptance_list.append(acceptanceProba/numberOfThetaSamples)

            ancestorsWeights              = np.array(weightsSamplesNew)
            logThetaLks                   = np.array(logThetaLksNew)
            logThetaWeights               = np.zeros(numberOfThetaSamples)
            ancestorSamples               = np.array(stateSamplesNew)
            betaSamples                   = np.array(betaSamplesNew)
            tauSamples                    = np.array(tauSamplesNew)
            gammaSamples                  = np.array(gammaSamplesNew)
            normalisedThetaWeights        = useful_functions.to_normalized_weights(logThetaWeights)

        # Launch bootstrap update
        smc_c.bootstrapUpdateStep_c(np.ascontiguousarray(currentSamples), gammaSamples, betaSamples/2. + 1/2., tauSamples/2., T, np.ascontiguousarray(ancestorSamples), ancestorsWeights, np.ascontiguousarray(mapping), stimuli[T-1])

        # Take decision
        for ts_idx in range(K):
            tsProbability[T, ts_idx] = np.sum(normalisedThetaWeights * np.sum((currentSamples == ts_idx), axis = 1))


        if beta_softmax is None:
            # Compute action likelihood
            for action_idx in range(numberOfActions):
                actionLikelihood[action_idx] = np.sum(tsProbability[T, mapping[stimuli[T]] == action_idx])

            # Select action
            actions[T] = np.argmax(actionLikelihood)

        else:
            # Compute action likelihood
            tsProbability[T] /= sum(tsProbability[T])
            
            for action_idx in range(numberOfActions):
                actionLikelihood[action_idx] = np.exp(np.log(np.sum(tsProbability[T, mapping[stimuli[T].astype(int)] == action_idx])) * beta_softmax)

            actionLikelihood /= sum(actionLikelihood)

            # Select action
            actions[T]        = np.where(np.random.multinomial(1, actionLikelihood, size=1)[0])[0][0]

        # Select action and compute vol
        volTracking[T]    = np.sum(normalisedThetaWeights * tauSamples)
        volStdTracking[T] = np.sum(normalisedThetaWeights * (tauSamples - volTracking[T])**2)

        betaTracking[T]    = np.sum(normalisedThetaWeights * betaSamples)
        betaStdTracking[T] = np.sum(normalisedThetaWeights * (betaSamples - betaTracking[T])**2)

        if K == 2:
            assert(mapping[stimuli[T].astype(int), Z_true[T].astype(int)] == Z_true[T])
        if (K == 2) and (actions[T] == mapping[stimuli[T].astype(int), Z_true[T].astype(int)]):
            rewards[T]            = not td['trap'][T]
            countPerformance[T:] += 1
        elif (K == 24) and (actions[T] == td['A_correct'][T]):
            rewards[T]            = not td['trap'][T]
            countPerformance[T:] += 1        
        else:
            rewards[T]            = td['trap'][T]


        if show_progress:
            plt.subplot(3,2,1)
            plt.imshow(tsProbability[:T].T, aspect='auto'); plt.hold(True)
            plt.plot(Z_true[:T], 'w--')
            plt.axis([0, T-1, 0, K-1])
            plt.hold(False)
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
            plt.plot(x, normlib.pdf(x, betaTracking[T], betaStdTracking[T]), 'r'); plt.hold(True)
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

            plt.subplot(3, 2, 5);
            plt.plot(np.divide(countPerformance[:T], np.arange(T)+1), 'k--', linewidth=2); plt.hold(True)
            plt.axis([0,T-1,0,1]);
            plt.hold(False);
            plt.xlabel('Trials');
            plt.ylabel('Performance');

            plt.draw()
            plt.show()
            plt.pause(0.1)

    elapsed_time = time.time() - start_time_multi

    return [td, tauSamples, volTracking, volStdTracking, betaSamples, betaTracking, betaStdTracking, gammaSamples, tsProbability, countPerformance, actions, acceptance_list, essList, time_list, elapsed_time]









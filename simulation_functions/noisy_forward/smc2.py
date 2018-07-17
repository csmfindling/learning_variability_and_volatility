##### Model with noisy with C++ and python

#Libraries
import sys
sys.path.append('../../utils/')
sys.path.append('../../lib_c/')
from noisy_forward import smc_c
import numpy as np
import get_mapping
import useful_functions
import sys
from scipy.stats import beta as betalib
from scipy.stats import norm as normlib
import matplotlib.pyplot as plt
import time
import numpy
import math
from scipy.stats import entropy

def SMC2(td, show_progress=True, lambdaa=.9, eta=0., inertie_noise=0., numberOfStateSamples=2000, numberOfThetaSamples=1000, coefficient = .5, beta_softmax = None , espilon_softmax=0.):

    print('precision model with lambda = {0} and eta = {1}, epsilon= {4}, inertie_noise={5}. Number of state samples : {2} and number of theta samples : {3}'.format(lambdaa, eta, numberOfStateSamples, numberOfThetaSamples, espilon_softmax, inertie_noise))

    #Start timer
    start_time_multi = time.time()

    # Extract parameters from task description
    stimuli         = np.ascontiguousarray(td['S'], dtype=np.intc)                                        # Sequence of Stimuli
    Z               = td['Z']                                             # Sequence of Task Sets
    numberOfActions = td['action_num']                                          # Number of Actions possible
    numberOfStimuli = td['state_num']                                           # Number of states or stimuli
    K               = np.prod(np.arange(numberOfActions+1)[-numberOfStimuli:])  # Number of possible Task Sets
    numberOfTrials  = len(Z)                                               # Number of Trials

        
    # Sampling and prior settings
    betaPrior      = np.array([1, 1])              # Prior on Beta, the feedback noise parameter
    dirichletPrior = np.ones(K)
    
    # Mapping from task set to correct action per stimulus
    mapping = np.ascontiguousarray(get_mapping.Get_TaskSet_Stimulus_Mapping(state_num=numberOfStimuli, action_num=numberOfActions).T, dtype=np.intc)
    Z_true  = Z;

    # Probabilities of every actions updated at every time step -> Used to take the decision
    actionLikelihood = np.zeros(numberOfActions)  # For 1 observation, likelihood of the action. Requires a marginalisation over all task sets
    actions          = np.ascontiguousarray(np.zeros(numberOfTrials) - 1 , dtype=np.intc) 
    rewards          = np.ascontiguousarray(np.zeros(numberOfTrials), dtype=np.intc) 

    # Keep track of probability correct/exploration after switches
    countPerformance       = np.zeros(numberOfTrials)    # Number of correct actions after i trials
    countExploration       = np.zeros(numberOfTrials)    # Number of exploratory actions after i trials
    correct_before_switch  = np.empty(0)                 # The correct task set before switch
    tsProbability          = np.zeros([numberOfTrials, K])
    acceptanceProba        = 0.
    betaTracking           = np.zeros(numberOfTrials)
    betaStdTracking        = np.zeros(numberOfTrials)
    temperatureTracking    = np.zeros(numberOfTrials)
    temperatureStdTracking = np.zeros(numberOfTrials)
    acceptance_list        = [1.]
    transitionProba        = np.zeros([numberOfThetaSamples, K, K])

    # SMC particles initialisation
    betaSamples                  = np.random.beta(betaPrior[0], betaPrior[1], numberOfThetaSamples)
    gammaSamples                 = np.random.dirichlet(dirichletPrior, numberOfThetaSamples)
    logThetaWeights              = np.zeros(numberOfThetaSamples)
    logThetaLks                  = np.zeros(numberOfThetaSamples)
    currentTaskSetSamples        = np.zeros([numberOfThetaSamples,numberOfStateSamples], dtype=np.intc)
    ancestorTaskSetSamples       = np.zeros([numberOfThetaSamples,numberOfStateSamples], dtype=np.intc)
    weightsList                  = np.zeros([numberOfThetaSamples,numberOfStateSamples])
    essList                      = np.zeros(numberOfTrials)
    tasksetLikelihood            = np.zeros(K)
    currentTemperatures          = np.zeros(numberOfTrials)
    entropies                    = np.zeros(numberOfTrials)
    temperature                  = 0.5

    # variables for speed-up

    ante_proba_local             = np.zeros(K)
    post_proba_local             = np.zeros(K)
    sum_weightsList              = np.zeros(numberOfThetaSamples)
    ancestorsIndexes             = np.zeros(numberOfStateSamples, dtype=np.intc)
    gammaAdaptedProba            = np.zeros(K)
    likelihoods                  = np.zeros(K)
    positiveStates               = np.zeros(K, dtype=np.intc)
    distances                    = np.zeros([numberOfThetaSamples, 1])
    currentNoises                = np.zeros([numberOfThetaSamples, numberOfStateSamples])
    noise_amount                 = np.zeros(numberOfTrials)

    # Plot progress
    if show_progress : plt.figure(figsize=(12,9)); plt.ion();

    # Loop over trials
    for T in range(numberOfTrials):

        # Print progress
        if (T+1) % 10 == 0 : sys.stdout.write(' ' + str(T+1));sys.stdout.flush();
        if (T+1) % 100 == 0: print ('\n')

        noise_amount[T] = smc_c.bootstrap_smc_step_c(logThetaWeights, distances, betaSamples/2. + 1/2., lambdaa, eta, inertie_noise, gammaSamples, currentTaskSetSamples, ancestorTaskSetSamples, weightsList, \
                    mapping, stimuli[T-1], rewards[T-1], actions[T-1], T, likelihoods, positiveStates, ante_proba_local,\
                                            post_proba_local, ancestorsIndexes, gammaAdaptedProba, sum_weightsList, currentNoises, float(temperature))
        
        if temperature is None:
            assert(False)

        entropies[T]              = entropy(np.asarray([np.sum(currentTaskSetSamples==i) for i in range(K)])*1./(numberOfThetaSamples*numberOfStateSamples))
        ancestorTaskSetSamples[:] = currentTaskSetSamples

        # Degeneray criterion
        logEss     = 2 * useful_functions.log_sum(logThetaWeights) - useful_functions.log_sum(2 * logThetaWeights)
        essList[T] = np.exp(logEss)

        # Move step
        normalisedThetaWeights = useful_functions.to_normalized_weights(logThetaWeights)
        if essList[T] < coefficient * numberOfThetaSamples and acceptance_list[-1] > 0.05:
             acceptanceProba          = 0.
             betaMu                   = np.sum(normalisedThetaWeights * betaSamples)
             betaVar                  = np.sum(normalisedThetaWeights * (betaSamples - betaMu)**2)
             betaAlpha                = np.maximum(((1 - betaMu)/betaVar - 1/betaMu) * betaMu**2, 1)
             betaBeta                 = np.maximum(betaAlpha * (1/betaMu - 1), 1.)
             assert(betaAlpha > 0); assert(betaBeta > 0)
             dirichletMeans           = np.sum(normalisedThetaWeights*gammaSamples.T, axis=1)
             dirichletVar             = np.sum(normalisedThetaWeights*(gammaSamples**2).T, axis=1) - dirichletMeans**2
             dirichletPrecision       = np.sum(dirichletMeans - dirichletMeans**2)/(np.sum(dirichletVar)) - 1
             dirichletParamCandidates = dirichletMeans * dirichletPrecision
             dirichletParamCandidates = np.maximum(dirichletParamCandidates, 1.)
             assert((dirichletParamCandidates>0).all())

             logThetaWeights[:]            = 0
             betaSamples                   = np.random.beta(betaAlpha, betaBeta, numberOfThetaSamples)
             gammaSamples                  = np.random.dirichlet(dirichletParamCandidates, numberOfThetaSamples)
             normalisedThetaWeights        = useful_functions.to_normalized_weights(logThetaWeights)

        # Take decision
        for ts_idx in range(K):
            tsProbability[T, ts_idx] = np.sum(normalisedThetaWeights * np.sum((currentTaskSetSamples == ts_idx), axis = 1))


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

            actionLikelihood  = actionLikelihood * (1 - espilon_softmax)  + espilon_softmax/K
            # Select action
            actions[T]        = np.where(np.random.multinomial(1, actionLikelihood, size=1)[0])[0][0]

        betaTracking[T]           = np.sum(normalisedThetaWeights * betaSamples)
        betaStdTracking[T]        = np.sum(normalisedThetaWeights * (betaSamples - betaTracking[T])**2)
        temperatureTracking[T]    = np.mean(currentNoises)
        temperatureStdTracking[T] = np.std(currentNoises)

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
            plt.plot(temperatureTracking[:T])
            plt.fill_between(np.arange(T),temperatureTracking[:T]-temperatureStdTracking[:T], temperatureTracking[:T]+temperatureStdTracking[:T],facecolor=[.5,.5,1], color=[.5,.5,1]); 
            plt.hold(False)
            plt.xlabel('trials')
            plt.ylabel('Temperature')

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

    return [td, noise_amount, lambdaa, eta, betaSamples, betaTracking, betaStdTracking, currentTemperatures, temperatureTracking, temperatureStdTracking, gammaSamples, tsProbability, countPerformance, actions, acceptance_list, elapsed_time]









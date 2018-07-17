##### Model with constant volatility with C++ and python

#Libraries
import numpy as np
import sys
sys.path.append('../../utils/')
sys.path.append('../../lib_c/')
from cstvol_forward import smc_c
import get_mapping
import useful_functions
from scipy.stats import uniform 
from scipy.stats import beta as betalib
from scipy.stats import norm as normlib
from scipy.stats import gamma as gammalib 
import time
import numpy
import pickle
from scipy.misc import logsumexp
import warnings
try:
    import mcerp
except:
    warnings.warn('failed to import latin hypercube sampling library')

def SMC2(td, beta_softmax=1., numberOfStateSamples=200, numberOfThetaSamples=200, numberOfBetaSamples=20, coefficient = .5, latin_hyp_sampling=True):

    print('\n')
    print('Forward Constant Volatility Model')
    print('number of theta samples ' + str(numberOfThetaSamples)); print('\n')

    #Start timer
    start_time_multi = time.time()

    # uniform distribution
    if latin_hyp_sampling:
        d0 = uniform()
        print('latin hypercube sampling')
    else:
        print('sobolev sampling')        

    # Extract parameters from task description
    stimuli         = td['S']                                                   # Sequence of Stimuli
    numberOfActions = td['action_num']                                          # Number of Actions possible
    numberOfStimuli = td['state_num']                                           # Number of states or stimuli
    rewards         = td['reward']
    actions         = td['A_chosen']
    K               = np.prod(np.arange(numberOfActions+1)[-numberOfStimuli:])  # Number of possible Task Sets
    numberOfTrials  = len(stimuli)                                               # Number of Trials

    # verification
    if K==2:
        if latin_hyp_sampling == False:
            raise ValueError('Why did you change the latin_hyp_sampling? By default, it is True and has no influence when K=2.')

    # Sampling and prior settings
    betaPrior   = np.array([1, 1]) # Prior on Beta, the feedback noise parameter
    tauPrior    = np.array([1, 1])
    gammaPrior  = np.ones(K)       # Prior on Gamma, the Dirichlet parameter
    log_proba   = 0.
    log_proba_  = 0.
    # Mapping from task set to correct action per stimulus
    mapping = get_mapping.Get_TaskSet_Stimulus_Mapping(state_num=numberOfStimuli, action_num=numberOfActions).T

    betaWeights      = np.zeros(numberOfBetaSamples)
    betaLog          = np.zeros(numberOfBetaSamples)
    logbetaWeights   = np.zeros(numberOfBetaSamples)
    betaAncestors    = np.arange(numberOfBetaSamples)

    # Probabilities of every actions updated at every time step -> Used to take the decision
    actionLikelihood = np.zeros([numberOfBetaSamples, numberOfActions])
    sum_actionLik    = np.zeros(numberOfBetaSamples)
    filt_actionLkd   = np.zeros([numberOfTrials, numberOfBetaSamples, numberOfActions])

    # Keep track of probability correct/exploration after switches
    tsProbability         = np.zeros([numberOfBetaSamples, K])
    sum_tsProbability     = np.zeros(numberOfBetaSamples)
    dirichletParamCandidates = np.zeros(K)
    
    # SMC particles initialisation
    muSamples         = np.zeros([numberOfBetaSamples, numberOfThetaSamples]) #np.random.beta(betaPrior[0], betaPrior[1], [numberOfBetaSamples, numberOfThetaSamples])
    gammaSamples      = np.zeros([numberOfBetaSamples, numberOfThetaSamples, K])
    tauSamples        = np.zeros([numberOfBetaSamples, numberOfThetaSamples]) 

    if K==24:
        try:
            latin_hyp_samples = pickle.load(open('../../utils/sobol_200_26.pkl','rb'))
        except:
            latin_hyp_samples = pickle.load(open('../../models/utils/sobol_200_26.pkl','rb'))
        for beta_idx in range(numberOfBetaSamples):
            if latin_hyp_sampling:
                latin_hyp_samples   = mcerp.lhd(dist = d0, size = numberOfThetaSamples, dims= K + 2)
            muSamples[beta_idx]     = betalib.ppf(latin_hyp_samples[:,0], betaPrior[0], betaPrior[1])
            tauSamples[beta_idx]    = betalib.ppf(latin_hyp_samples[:,1], tauPrior[0], tauPrior[1])
            gammaSamples[beta_idx]  = gammalib.ppf(latin_hyp_samples[:,2:], gammaPrior)
            gammaSamples[beta_idx]  = np.transpose(gammaSamples[beta_idx].T/np.sum(gammaSamples[beta_idx], axis=1))
    elif K==2:
        muSamples                    = np.random.beta(betaPrior[0], betaPrior[1], [numberOfBetaSamples, numberOfThetaSamples])
        tauSamples                   = np.random.beta(tauPrior[0], tauPrior[1], [numberOfBetaSamples, numberOfThetaSamples])
        gammaSamples                 = np.random.dirichlet(gammaPrior, [numberOfBetaSamples, numberOfThetaSamples])
    else:
        raise IndexError('Wrong number of task sets') 

    logThetaWeights              = np.zeros([numberOfBetaSamples, numberOfThetaSamples])
    currentSamples               = np.zeros([numberOfBetaSamples, numberOfThetaSamples, numberOfStateSamples], dtype=np.intc)
    ancestorSamples              = np.zeros([numberOfBetaSamples, numberOfThetaSamples, numberOfStateSamples], dtype=np.intc)
    weightsList                  = np.ones([numberOfThetaSamples, numberOfStateSamples])/numberOfStateSamples
    
    log_proba_corr               = 0.
    ancestorsIndexes             = np.zeros(numberOfStateSamples, dtype=np.intc)
    gammaAdaptedProba            = np.zeros(K)
    likelihoods                  = np.zeros(K)
    positiveStates               = np.zeros(K, dtype=np.intc)

    # Guided SMC variables
    muSamplesNew                  = np.zeros([numberOfBetaSamples, numberOfThetaSamples])
    tauSamplesNew                 = np.zeros([numberOfBetaSamples, numberOfThetaSamples])
    gammaSamplesNew               = np.zeros([numberOfBetaSamples, numberOfThetaSamples, K])
    logThetaWeightsNew            = np.zeros([numberOfBetaSamples, numberOfThetaSamples])
    normalisedThetaWeights        = np.zeros([numberOfBetaSamples, numberOfThetaSamples])


    # Loop over trials
    for T in range(numberOfTrials):

        # Print progress
        if (T+1) % 10 == 0 : sys.stdout.write(' ' + str(T+1)); sys.stdout.flush()
        if (T+1) % 100 == 0: print ('\n')

        for beta_idx in range(numberOfBetaSamples):

            ances          = betaAncestors[beta_idx]

            smc_c.bootstrapUpdateStep_c(currentSamples[beta_idx], logThetaWeights[beta_idx], gammaSamples[ances], muSamples[ances]/2. + 1./2, tauSamples[ances]/2., T, ancestorSamples[ances], weightsList, \
                                            np.ascontiguousarray(mapping), stimuli[T-1], actions[T-1], rewards[T-1], ancestorsIndexes, gammaAdaptedProba, likelihoods, positiveStates, 0)

            # Move step
            normalisedThetaWeights[beta_idx] = useful_functions.to_normalized_weights(logThetaWeights[beta_idx])
            ess                              = 1./np.sum(normalisedThetaWeights[beta_idx]**2)

            if (ess < coefficient * numberOfThetaSamples):
                acceptanceProba          = 0.
                tauMu                    = np.sum(normalisedThetaWeights[beta_idx] * tauSamples[ances])
                tauVar                   = np.sum(normalisedThetaWeights[beta_idx] * (tauSamples[ances] - tauMu)**2)
                tauAlpha                 = ((1 - tauMu)/tauVar - 1/tauMu) * tauMu**2
                tauBeta                  = tauAlpha * (1/tauMu - 1)
                assert(tauAlpha > 0); assert(tauBeta > 0)
                betaMu                   = np.sum(normalisedThetaWeights[beta_idx]*muSamples[ances])
                betaVar                  = np.sum(normalisedThetaWeights[beta_idx] * (muSamples[ances] - betaMu)**2)
                betaAlpha                = ((1 - betaMu)/betaVar - 1/betaMu) * betaMu**2
                betaBeta                 = betaAlpha * (1/betaMu - 1)
                assert(betaAlpha > 0); assert(betaBeta > 0)
                dirichletMeans           = np.sum(normalisedThetaWeights[beta_idx]*gammaSamples[ances].T, axis=1)
                dirichletVar             = np.sum(normalisedThetaWeights[beta_idx]*(gammaSamples[ances]**2).T, axis=1) - dirichletMeans**2
                dirichletPrecision       = np.sum(dirichletMeans - dirichletMeans**2)/(np.sum(dirichletVar)) - 1
                dirichletParamCandidates[:] = np.maximum(dirichletMeans * dirichletPrecision, 1.)
                assert((dirichletParamCandidates>0).all())

                if K==2:
                    tauSamplesNew[beta_idx]         = np.random.beta(tauAlpha, tauBeta, numberOfThetaSamples)
                    muSamplesNew[beta_idx]          = np.random.beta(betaAlpha, betaBeta, numberOfThetaSamples)
                    gammaSamplesNew[beta_idx]       = np.random.dirichlet(dirichletParamCandidates, numberOfThetaSamples)
                elif K==24:
                    if latin_hyp_sampling:
                        latin_hyp_samples       = mcerp.lhd(dist = d0, size = numberOfThetaSamples, dims= K + 2)
                    muSamplesNew[beta_idx]     = betalib.ppf(latin_hyp_samples[:,0], betaAlpha, betaBeta)
                    tauSamplesNew[beta_idx]    = betalib.ppf(latin_hyp_samples[:,1], tauAlpha, tauBeta)
                    gammaSamplesNew[beta_idx]  = gammalib.ppf(latin_hyp_samples[:,2:], dirichletParamCandidates)
                    gammaSamplesNew[beta_idx]  = np.transpose(gammaSamplesNew[beta_idx].T/np.sum(gammaSamplesNew[beta_idx], axis=1))                    

                logThetaWeightsNew[beta_idx]     = 0.
                normalisedThetaWeights[beta_idx] = 1./numberOfThetaSamples
            else:
                tauSamplesNew[beta_idx]         = tauSamples[ances]
                muSamplesNew[beta_idx]          = muSamples[ances]
                gammaSamplesNew[beta_idx]       = gammaSamples[ances]
                logThetaWeightsNew[beta_idx]    = logThetaWeights[beta_idx]
 

        # task set probability
        sum_tsProbability[:] = 0.
        for ts_idx in range(K):
            tsProbability[:, ts_idx] = np.sum(normalisedThetaWeights * np.sum((currentSamples == ts_idx), axis = 2), axis=1)
            sum_tsProbability       += tsProbability[:,ts_idx]

        tsProbability[:] = np.transpose(tsProbability.T/sum_tsProbability)

        # Compute action likelihood
        sum_actionLik[:] = 0.
        for action_idx in range(numberOfActions):
            actionLikelihood[:, action_idx] = np.exp(np.log(np.sum(tsProbability[:,mapping[stimuli[T].astype(int)] == action_idx], axis=1)) * beta_softmax)
            sum_actionLik                  += actionLikelihood[:, action_idx]

        rewards[T]               = td['reward'][T]
        actions[T]               = td['A_chosen'][T]

        actionLikelihood[:]      = np.transpose(actionLikelihood.T/sum_actionLik)
        betaWeights[:]           = actionLikelihood[:,actions[T].astype(int)]
            
        filt_actionLkd[T]        = actionLikelihood

        log_proba_              += np.log(sum(betaWeights)/numberOfBetaSamples)
        betaWeights              = betaWeights/sum(betaWeights)

        betaAncestors[:]         = useful_functions.stratified_resampling(betaWeights)

        # update particles
        muSamples[:]             = muSamplesNew
        gammaSamples[:]          = gammaSamplesNew
        tauSamples[:]            = tauSamplesNew
        logThetaWeights[:]       = logThetaWeightsNew[betaAncestors]
        ancestorSamples[:]       = currentSamples

    elapsed_time = time.time() - start_time_multi

    return log_proba_, filt_actionLkd










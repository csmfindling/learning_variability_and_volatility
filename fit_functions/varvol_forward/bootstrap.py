##### Model with constant volatility with C++ and python

#Libraries
import numpy as np
import sys
sys.path.append('../../utils/')
sys.path.append('../../lib_c/')
from varvol_forward import smc_c
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

def SMC2(td, beta_softmax=1., numberOfStateSamples=200, numberOfThetaSamples=200, numberOfBetaSamples=50, coefficient = .5, latin_hyp_sampling=True):

    print('\n')
    print('Forward Varying Volatility Model')
    print('number of theta samples ' + str(numberOfThetaSamples)); print('\n')

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
    betaPrior   = np.array([1, 1])                   # Prior on Beta, the feedback noise parameter
    nuPrior     = np.array([3, 1e-3])                # Prior on Nu, the variance on the projected gaussian random walk
    gammaPrior  = numpy.ones(K)                      # Prior on Gamma, the Dirichlet parameter
    try:
        tauDefault  = td['tau'][0]
    except:
        tauDefault  = td['tau']
    log_proba_  = 0.

    # Mapping from task set to correct action per stimulus
    mapping = get_mapping.Get_TaskSet_Stimulus_Mapping(state_num=numberOfStimuli, action_num=numberOfActions).T

    betaWeights      = np.zeros(numberOfBetaSamples)
    betaAncestors    = np.arange(numberOfBetaSamples)

    # Probabilities of every actions updated at every time step -> Used to take the decision
    actionLikelihood = np.zeros([numberOfBetaSamples, numberOfActions])
    sum_actionLik    = np.zeros(numberOfBetaSamples)
    filt_actionLkd   = np.zeros([numberOfTrials, numberOfBetaSamples, numberOfActions])

    # Keep track of probability correct/exploration after switches
    tsProbability         = np.zeros([numberOfBetaSamples, K])
    sum_tsProbability     = np.zeros(numberOfBetaSamples)

    # SMC particles initialisation
    muSamples                     = np.zeros([numberOfBetaSamples, numberOfThetaSamples]) 
    nuSamples                     = np.zeros([numberOfBetaSamples, numberOfThetaSamples]) 
    gammaSamples                  = np.zeros([numberOfBetaSamples, numberOfThetaSamples, K])

    if K==24:
        try:
            latin_hyp_samples = pickle.load(open('../../utils/sobol_200_26.pkl','rb'))
        except:
            latin_hyp_samples = pickle.load(open('../../models/utils/sobol_200_26.pkl','rb'))
        for beta_idx in range(numberOfBetaSamples):
            if latin_hyp_sampling:
                latin_hyp_samples       = mcerp.lhd(dist = d0, size = numberOfThetaSamples, dims= K + 2)
            muSamples[beta_idx]     = betalib.ppf(latin_hyp_samples[:,0], betaPrior[0], betaPrior[1])
            nuSamples[beta_idx]     = useful_functions.ppf_inv_gamma(latin_hyp_samples[:,1], nuPrior[0], nuPrior[1])
            gammaSamples[beta_idx]  = gammalib.ppf(latin_hyp_samples[:,2:], gammaPrior)
            gammaSamples[beta_idx]  = np.transpose(gammaSamples[beta_idx].T/np.sum(gammaSamples[beta_idx], axis=1))
    elif K==2:
        muSamples                    = np.random.beta(betaPrior[0], betaPrior[1], [numberOfBetaSamples, numberOfThetaSamples])
        nuSamples                    = useful_functions.sample_inv_gamma(nuPrior[0], nuPrior[1], [numberOfBetaSamples, numberOfThetaSamples])
        gammaSamples                 = np.random.dirichlet(gammaPrior, [numberOfBetaSamples, numberOfThetaSamples])
    else:
        raise IndexError('Wrong number of task sets') 

    muSamplesNew                  = np.zeros([numberOfBetaSamples, numberOfThetaSamples])
    nuSamplesNew                  = np.zeros([numberOfBetaSamples, numberOfThetaSamples])
    gammaSamplesNew               = np.zeros([numberOfBetaSamples, numberOfThetaSamples, K])
    logThetaWeightsNew            = np.zeros([numberOfBetaSamples, numberOfThetaSamples])
    normalisedThetaWeights        = np.zeros([numberOfBetaSamples, numberOfThetaSamples])

    logThetaWeights               = np.zeros([numberOfBetaSamples, numberOfThetaSamples])
    currentStateSamples           = np.zeros([numberOfBetaSamples, numberOfThetaSamples, numberOfStateSamples], dtype=np.intc)
    currentTauSamples             = np.zeros([numberOfBetaSamples, numberOfThetaSamples, numberOfStateSamples], dtype=np.double)
    ancestorStateSamples          = np.zeros([numberOfBetaSamples, numberOfThetaSamples, numberOfStateSamples], dtype=np.intc)
    ancestorTauSamples            = np.zeros([numberOfBetaSamples, numberOfThetaSamples, numberOfStateSamples], dtype=np.double)
    ancestorsWeights              = np.ones([numberOfThetaSamples, numberOfStateSamples])/numberOfStateSamples
    essList                       = np.zeros(numberOfTrials)

    # Guided SMC variables
    dirichletParamCandidates     = np.zeros(K)

    # Loop over trials
    for T in range(numberOfTrials):

        # Print progress
        if (T+1) % 10 == 0 : sys.stdout.write(' ' + str(T+1));sys.stdout.flush()
        if (T+1) % 100 == 0: print ('\n')

        for beta_idx in range(numberOfBetaSamples):

            ances = betaAncestors[beta_idx]
            # Update theta weights
            smc_c.bootstrapUpdateStep_c(currentStateSamples[beta_idx], logThetaWeights[beta_idx], currentTauSamples[beta_idx], gammaSamples[ances], muSamples[ances]/2. + 1./2, nuSamples[ances], tauDefault, T, \
                                            np.ascontiguousarray(ancestorStateSamples[ances], dtype=np.intc), ancestorTauSamples[ances], ancestorsWeights, np.ascontiguousarray(mapping), stimuli[T-1], actions[T-1], rewards[T-1])

            # Degeneray criterion
            logEss     = 2 * useful_functions.log_sum(logThetaWeights[beta_idx]) - useful_functions.log_sum(2 * logThetaWeights[beta_idx])
            essList[T] = np.exp(logEss)

            # Move step
            normalisedThetaWeights[beta_idx] = useful_functions.to_normalized_weights(logThetaWeights[beta_idx])
            if (essList[T] < coefficient * numberOfThetaSamples):
                betaMu                   = np.sum(normalisedThetaWeights[beta_idx] * muSamples[ances])
                betaVar                  = np.sum(normalisedThetaWeights[beta_idx] * (muSamples[ances] - betaMu)**2)
                betaAlpha                = ((1 - betaMu)/betaVar - 1/betaMu) * betaMu**2
                betaBeta                 = betaAlpha * (1/betaMu - 1)
                assert(betaAlpha > 0); assert(betaBeta > 0)
                nuMu                     = np.sum(normalisedThetaWeights[beta_idx] * nuSamples[ances])
                nuVar                    = np.sum(normalisedThetaWeights[beta_idx] * (nuSamples[ances] - nuMu)**2)
                nuAlpha                  = nuMu**2/nuVar + 2
                nuBeta                   = nuMu * (nuAlpha - 1)
                assert(nuAlpha > 0); assert(nuBeta > 0)
                dirichletMeans              = np.sum(normalisedThetaWeights[beta_idx]*gammaSamples[ances].T, axis=1)
                dirichletVar                = np.sum(normalisedThetaWeights[beta_idx]*(gammaSamples[ances]**2).T, axis=1) - dirichletMeans**2
                dirichletPrecision          = np.sum(dirichletMeans - dirichletMeans**2)/(np.sum(dirichletVar)) - 1
                dirichletParamCandidates[:] = np.maximum(dirichletMeans * dirichletPrecision, 1.)
                assert((dirichletParamCandidates>0).all())
                if K==2:
                    nuSamplesNew[beta_idx]       = useful_functions.sample_inv_gamma(nuAlpha, nuBeta, numberOfThetaSamples)
                    muSamplesNew[beta_idx]       = np.random.beta(betaAlpha, betaBeta, numberOfThetaSamples)
                    gammaSamplesNew[beta_idx]    = np.random.dirichlet(dirichletParamCandidates, numberOfThetaSamples)
                elif K==24:
                    if latin_hyp_sampling:
                        latin_hyp_samples      = mcerp.lhd(dist = d0, size = numberOfThetaSamples, dims= K + 2)
                    muSamplesNew[beta_idx]     = betalib.ppf(latin_hyp_samples[:,0], betaAlpha, betaBeta)
                    nuSamplesNew[beta_idx]     = useful_functions.ppf_inv_gamma(latin_hyp_samples[:,1], nuAlpha, nuBeta)
                    gammaSamplesNew[beta_idx]  = gammalib.ppf(latin_hyp_samples[:,2:], dirichletParamCandidates)
                    gammaSamplesNew[beta_idx]  = np.transpose(gammaSamplesNew[beta_idx].T/np.sum(gammaSamplesNew[beta_idx], axis=1))                    

                logThetaWeightsNew[beta_idx]     = 0.
                normalisedThetaWeights[beta_idx] = 1./numberOfThetaSamples

            else:
                muSamplesNew[beta_idx]       = muSamples[ances]
                gammaSamplesNew[beta_idx]    = gammaSamples[ances]
                nuSamplesNew[beta_idx]       = nuSamples[ances]
                logThetaWeightsNew[beta_idx] = logThetaWeights[beta_idx]

        # task set probability
        sum_tsProbability[:] = 0.
        for ts_idx in range(K):
            tsProbability[:, ts_idx] = np.sum(normalisedThetaWeights * np.sum((currentStateSamples == ts_idx), axis = 2), axis=1)
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

        log_proba_       += np.log(sum(betaWeights)/numberOfBetaSamples)
        betaWeights       = betaWeights/sum(betaWeights)

        betaAncestors[:]  = useful_functions.stratified_resampling(betaWeights)

        # update particles
        muSamples[:]            = muSamplesNew
        gammaSamples[:]         = gammaSamplesNew
        nuSamples[:]            = nuSamplesNew
        logThetaWeights[:]      = logThetaWeightsNew[betaAncestors]
        ancestorTauSamples[:]   = currentTauSamples
        ancestorStateSamples[:] = currentStateSamples

    elapsed_time = time.time() - start_time_multi

    return log_proba_, filt_actionLkd









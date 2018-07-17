##### Useful functions used in the script #####

# Libraries
import numpy as np
import math
import operator
import functools
from scipy.stats import beta as betalib
from scipy.special import gammaln
from scipy.stats import norm
from scipy.stats import moment
from scipy.stats import invgamma
import warnings


# Return dictionnary of addresses for each variable; takes an input a list of variables
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def return_all_addresses(*args):
    warnings.warn("Careful in the order of the variables for the return_addresses function")
    assert(len(args) == 19)
    dictionary = {}
    name_variables = ['logThetaLks', 'logThetaWeights', 'betaSamples', 'etaSamples', 'currentTaskSetSamples', 'currentLatentSamples' , 'currentTemperatureSamples' , \
                        'ancestorLatentSamples', 'ancestorTemperatureSamples', 'logLatentWeights', 'latentWeights' , 'candidateLatentSamples', 'candidateTemperatureSamples', 'candidateLogWeights', 'candidateWeights', \
                        'mapping', 'stimuli' , 'rewards' , 'actions']
    for i in range(len(args)):
        dictionary[name_variables[i]] = hex(id(args[i]))
    return dictionary

def return_address_var(arg):
    return hex(id(arg))

def stratified_resampling(logThetaLks, logThetaWeights, betaSamples, etaSamples, currentTaskSetSamples, currentLatentSamples , currentTemperatureSamples , ancestorLatentSamples, \
                                            ancestorTemperatureSamples, latentWeights , candidateLatentSamples, candidateTemperatureSamples, candidateWeights, mapping, \
                                            stimuli , rewards , actions):
    N = len(w)
    v = np.cumsum(w) * N
    s = np.random.uniform()
    o = np.zeros(N, dtype=np.int)
    m = 0
    for i in range(N):
        while v[m] < s : m = m + 1
        o[i] = m; s = s + 1
    return o


# Stratified resample:
def stratified_resampling(w):
    N = len(w)
    v = np.cumsum(w) * N
    s = np.random.uniform()
    o = np.zeros(N, dtype=np.int)
    m = 0
    for i in range(N):
        while v[m] < s : m = m + 1
        o[i] = m; s = s + 1
    return o

# Multinomial un-normalized pick function; the un-normalized probability distribution is p.
def random_pick(p):
	return np.random.choice(len(p), p = np.divide(p,np.sum(p)))

def random_pick_list(p,n):
    return np.random.choice(len(p), size = n, p = np.divide(p,np.sum(p)))

def truncated_normal(mu, std, mini, numberOfSamples):
    res = np.zeros(numberOfSamples)
    for i in range(numberOfSamples):
        sample = np.random.normal(mu,std)
        while (sample < mini):
            sample = np.random.normal(mu,std)
        res[i] = sample
    return res

def sample_inv_gamma(a, b, size=1):
    sample = invgamma.rvs(a=a, size=size)
    return b*sample

def ppf_inv_gamma(x, a, b):
    return b * invgamma.ppf(x, a)
    # b/gammainccinv(a, r[10])

def random_nRW(m, var, mini, maxi):
    std   = np.sqrt(var) 
    s     = np.random.normal(m, std)
    s     = s *(s > mini)*(s < maxi) + (s > maxi)
    return s

def random_nRW_vec(m, var, mini, maxi, nbOfSamples):
    std   = np.sqrt(var)
    s     = np.random.normal(m,std, size = nbOfSamples)
    s     = s *(s > mini)*(s < maxi) + (s > maxi)
    return s

def estimate_param_truncated_normal(data, mini):
    moment_1 = np.mean(data)
    moment_2 = np.mean(data**2)
    moment_3 = np.mean(data**3)
    mu       = mini + (2 * moment_1 * moment_2 - moment_3)/(2 * moment_1**2 - moment_2)
    sigma2   = (moment_1 * moment_3 - moment_2**2)/(2 * moment_1**2 - moment_2)
    return mu, np.sqrt(sigma2)

def dirichlet_pdf(x, alpha):
  return (math.gamma(sum(alpha)) / 
          functools.reduce(operator.mul, [math.gamma(a) for a in alpha]) *
          functools.reduce(operator.mul, [x[i]**(alpha[i]-1.0) for i in range(len(alpha))]))

def log_dirichlet_pdf(x,alpha):
    return (gammaln(sum(alpha)) - functools.reduce(operator.add, [gammaln(a) for a in alpha]) + functools.reduce(operator.add, [(alpha[i] - 1)*np.log(x[i]) for i in range(len(alpha))]))

def log_beta_pdf(x, a, b):
    return gammaln(a + b) - gammaln(a) - gammaln(b) + (a - 1)*np.log(x) + (b - 1)*np.log(1 - x)

def log_truncated_normal_pdf(x, mu, std, mini):
    assert(mini == 0)
    return np.log(norm.pdf((x-mu)/std)) - np.log(1 - norm.cdf(-mu/std)) - np.log(std)

def log_invgamma_pdf(x, a, b):
    return a * np.log(b) - gammaln(a) - (a + 1)*np.log(x) - b/x

def log_sum(logvector):
    b = np.max(logvector)
    return b + np.log(functools.reduce(operator.add, [np.exp(logw - b) for logw in logvector]))

def to_normalized_weights(logWeights):
    b = np.max(logWeights)
    weights = [np.exp(logw - b) for logw in logWeights]
    return weights/sum(weights)

def autocorrelation(x):
    x_norm = np.divide(x - np.mean(x),np.std(x))
    result = np.correlate(np.divide(x_norm, len(x)), x_norm, mode='full')
    return result[result.size/2:]

def plot_results(td, Z_prob, last_Z_prob, tau_params, beta_params, gamma_params, m_h_tau, m_h_gamma, A_corr_count, tau_autocorrelation):
    import matplotlib.pyplot as plt

    [trial_num, K] = gamma_params.shape; Z_true = td['Z'];
    sample_num     = len(tau_autocorrelation);
    plt.figure(figsize=(12, 9));

    # Plot beta
    plt.subplot(2,2,1);
    beta_mean = np.divide(beta_params[:,0], np.sum(beta_params,axis=1));
    beta_std  = np.sqrt(np.divide(np.multiply(beta_mean, beta_params[:,1]), np.multiply(np.sum(beta_params, axis=1), np.sum(beta_params, axis=1)+1)))
    plt.plot(beta_mean, 'r-');plt.hold(True); plt.fill_between(np.arange(trial_num),beta_mean-beta_std, beta_mean+beta_std,facecolor=[1,.5,.5], color=[1,.5,.5]); 
    # Mark switch and trap trials
    plt.axis([0,trial_num-1, 0, 1 ]);
    switch_trials = np.where(td['B'])[0];
    plt.plot([0, trial_num], [td['beta'], td['beta']], 'r--', linewidth=2);
    plt.hold(False);
    plt.ylabel('Estimated beta parameters'); 

    # Plot tau
    plt.subplot(2,2,3);
    tau_mean = np.divide(tau_params[:,0], np.sum(tau_params, axis=1));
    tau_std  = np.sqrt(np.divide(np.multiply(tau_mean, tau_params[:,1]), np.multiply(np.sum(tau_params, axis=1), np.sum(tau_params, axis=1)+1)));
    plt.plot(tau_mean, 'b-');plt.hold(True); plt.fill_between(np.arange(trial_num), tau_mean - tau_std, tau_mean+tau_std, facecolor=[.5,.5,1],color = [.5,.5,1]); 
    # Mark switch and trap trials
    plt.axis([0, trial_num-1, 0, 1]);
    plt.plot([0, trial_num], [td['tau'], td['tau']], 'b--', linewidth=2);
    plt.hold(False);
    plt.ylabel('Estimated tau paramaters'); 

    # Plot gamma paramaters
    plt.subplot(2,2,4);
    plt.imshow(gamma_params.T); plt.hold(True);
    plt.plot(Z_true, 'k--', linewidth=1);
    plt.axis([0,trial_num-1, 0, K-1]);
    plt.xlabel('trials');
    plt.hold(False);
    plt.ylabel('Estimated gamma parameters');

    # Plot state probability
    plt.subplot(2,2,2);
    plt.imshow(Z_prob.T); plt.hold(True);
    plt.plot(Z_true, 'w--');
    plt.axis([0, trial_num-1, 0, K-1]);
    plt.xlabel('trials');
    plt.hold(False);
    plt.ylabel('p(TS|past) at decision time');
    plt.draw();

    # Plot performances
    plt.figure(figsize=(12,9));

    #plot final performance
    plt.subplot(2,2,1)
    plt.plot(np.divide(A_corr_count, np.arange(trial_num)+1), 'k-', linewidth=2); plt.hold(True);
    plt.axis([0,trial_num-1,0,1]);
    plt.hold(False)
    plt.xlabel('trials');
    plt.ylabel('proportion correct answers');

    #plot final Z estimated
    plt.subplot(2,2,3);
    plt.imshow(last_Z_prob.T); plt.hold(True);
    plt.plot(Z_true, 'w--');
    plt.axis([0, trial_num-1, 0, K-1]);
    plt.hold(False);
    plt.xlabel('trials');
    plt.ylabel('p(TS|past) at current time');

    #plot gamma metropolis-hasting acceptance rate
    plt.subplot(2,2,2);
    plt.plot(m_h_gamma, 'g-');plt.hold(True);
    plt.plot(m_h_tau, 'b-');
    plt.axis([0,trial_num-1, 0,1]); 
    plt.hold(False);
    plt.xlabel('trials');
    plt.ylabel('gamma(green)/tau(blue) acceptance rates');

    #plot gibbs autocorrelation function
    plt.subplot(2,2,4);
    plt.plot(tau_autocorrelation, 'k-'); plt.hold(True);
    plt.axis([0,sample_num-1, 0,1]); 
    plt.hold(False);
    plt.xlabel('trials');
    plt.ylabel('Gibbs sampler autocorrelation');

    plt.draw();

    return 'plot ok'
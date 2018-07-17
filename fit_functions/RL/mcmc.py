import numpy as np
from scipy.stats import gamma, norm, truncnorm, multivariate_normal
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as multi_norm
from scipy.misc import logsumexp
import sys
sys.path.append("../useful_functions/")
import useful_functions as uf

# actions = td['A_chosen']; tau=td['tau']; rewards = td['reward']; subj_idx=subj_idx; show_progress = True; beta_softmax=-1; apply_rep_bias=0; temperature=1; n_alpha_model=True
def ibis(actions, rewards, tau, subj_idx, apply_rep_bias, show_progress = True, temperature = True, n_alpha_model=False):

	actions    = np.asarray(actions, dtype=np.intc)
	rewards    = np.ascontiguousarray(rewards)
	nb_samples = 1000
	T          = actions.shape[0]
	upp_bound_eta = 10.

	nb_acceptance  = 0
	# sample initialisation
	if n_alpha_model:
		n_alpha = 6
		tau_unique  = np.unique(tau)
		x_coor      = np.array([np.where(tau_unique == t)[0][0] for t in tau])
	else:
		n_alpha = 1
		x_coor  = np.zeros(len(tau), dtype=np.int8)

	if apply_rep_bias:
		samples                = np.random.rand(n_alpha + 2)
		if temperature:
			upp_bound_beta     = np.sqrt(6)/(np.pi * 5)
		else:
			upp_bound_beta     = 2.
		n_index_beta              = n_alpha
		samples[n_index_beta]     = upp_bound_beta/2.
		samples[n_index_beta + 1] = upp_bound_eta * (np.random.rand() * 2. - 1.)
	else:
		samples                = np.zeros(n_alpha + 1) + .5
		if temperature:
			upp_bound_beta     = np.sqrt(6)/(np.pi * 5)
		else:
			upp_bound_beta     = 2.
		n_index_beta      = n_alpha
		samples[-1]    = upp_bound_beta/2.
	
	all_samples    = np.zeros([nb_samples, len(samples)])
	all_samples[0] = samples
	lkd            = get_loglikelihood(samples, x_coor, rewards, actions, T, apply_rep_bias, temperature)[0]

	# loop
	for n_idx in range(nb_samples):
		Sigma_p  = 1e-2 * np.eye(len(samples))
		Sigma_p[-1][-1] = 1e-3
		while True:
			sample_p = multi_norm(samples, Sigma_p)
			if not apply_rep_bias:
				if np.all(sample_p[:n_alpha] > 0) and np.all(sample_p[:n_alpha] < 1) and sample_p[n_alpha] > 0 and sample_p[n_alpha] <= upp_bound_beta:
					break
			else:
				if np.all(sample_p[:n_alpha] > 0) and np.all(sample_p[:n_alpha] < 1) and sample_p[n_alpha] > 0 and sample_p[n_alpha] <= upp_bound_beta and sample_p[n_alpha + 1] > -upp_bound_eta and sample_p[n_alpha + 1] < upp_bound_eta:
					break

		[loglkd_prop, Q_prop, prev_action_prop] = get_loglikelihood(sample_p, x_coor, rewards, actions, T, apply_rep_bias, temperature) 
		log_ratio                               = loglkd_prop - lkd 

		log_ratio = np.minimum(log_ratio, 0)
		if (np.log(np.random.rand()) < log_ratio):
			nb_acceptance          += 1.
			all_samples[n_idx]     = sample_p
			lkd                    = loglkd_prop
			samples                = sample_p
		else:
			all_samples[n_idx]     = samples

	print('acception ratio is {0}'.format(nb_acceptance/nb_samples))

	return [samples, Q_samples, mean_Q, esslist, acceptance_l, log_weights, p_loglkd, marg_loglkd_l]

def get_logtruncnorm(sample, mu, sigma):
	return multivariate_normal.logpdf(sample, mu, sigma)

def get_logprior(sample, alpha_prior, beta_prior):
	a_alpha     = - alpha_prior[0]/alpha_prior[1]
	b_alpha     = (1 - alpha_prior[0])/alpha_prior[1]
	a_beta      = - beta_prior[0]/beta_prior[1]
	b_beta      = np.inf
	return truncnorm.logpdf(sample[0], a_alpha, b_alpha, alpha_prior[0], alpha_prior[1]) \
								+ truncnorm.logpdf(sample[1], a_alpha, b_alpha, alpha_prior[0], alpha_prior[1]) \
								+ truncnorm.logpdf(sample[2], a_beta, b_beta, beta_prior[0], beta_prior[1])

def get_loglikelihood(sample, x_coor, rewards, actions, T, apply_rep_bias, temperature=True):
	
	if len(np.unique(x_coor)) == 1:
		n_index_beta = 1
	else:
		n_index_beta = 6
	if temperature:
		beta     = 1./sample[n_index_beta]
	else:
		beta     = 10**sample[n_index_beta]
	if apply_rep_bias:
		eta = sample[n_index_beta + 1]
	prev_action    = -1
	log_proba      = 0
	Q0             = .5
	Q1             = .5

	for t_idx in range(T):
		action     = actions[t_idx]
		if len(np.unique(x_coor)) == 1:
			alpha  = sample[0]
		else:
			alpha  = sample[x_coor[t_idx]]

		if prev_action != -1 and apply_rep_bias:
			value       = 1./(1. + np.exp(beta * (Q0 - Q1) - np.sign(prev_action - .5) * eta))
			log_proba  += np.log((value**action) * (1 - value)**(1 - action))
			prev_action = actions[t_idx]
		else:
			value       = 1./(1. + np.exp(beta * (Q0 - Q1)))
			prev_action = actions[t_idx]
			log_proba  += np.log((value**action) * (1 - value)**(1 - action))
		
		if actions[t_idx] == 0:
			Q0 = (1 - alpha) * Q0 + alpha * rewards[t_idx]
			Q1 = (1 - alpha) * Q1 + alpha * (1 - rewards[t_idx])
		else:
			Q1 = (1 - alpha) * Q1 + alpha * rewards[t_idx]
			Q0 = (1 - alpha) * Q0 + alpha * (1 - rewards[t_idx])
			
	return [log_proba, np.array([Q0, Q1]), prev_action]


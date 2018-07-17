import numpy as np
from scipy.stats import gamma, norm, truncnorm, multivariate_normal
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as multi_norm
from scipy.misc import logsumexp
import sys
sys.path.append("../useful_functions/")
import useful_functions as uf

def ibis(actions, rewards, tau, subj_idx, apply_rep_bias, show_progress = True, temperature = True, model_id = 0):

	'''
	model_id = 0 : 1 alpha, 1 beta
	model_id = 1 : n alpha, 1 beta
	model_id = 2 : n alpha, n beta 
	'''

	actions       = np.asarray(actions, dtype=np.intc)
	rewards       = np.ascontiguousarray(rewards)
	nb_samples    = 1000
	T             = actions.shape[0]
	upp_bound_eta = 10.

	# sample initialisation
	if model_id == 2:
		n_alpha    = 6
		n_beta     = 6
		tau_unique = np.unique(tau)
		x_coor_a   = np.array([np.where(tau_unique == t)[0][0] for t in tau])	
		x_coor_b   = np.array([np.where(tau_unique == t)[0][0] for t in tau]) + n_alpha		
	elif model_id == 1:
		n_alpha    = 6
		n_beta     = 1
		tau_unique = np.unique(tau)
		x_coor_a   = np.array([np.where(tau_unique == t)[0][0] for t in tau])
		x_coor_b   = np.zeros(len(tau), dtype=np.int8) + n_alpha
	else:
		n_alpha    = 1
		n_beta     = 1
		x_coor_a   = np.zeros(len(tau), dtype=np.int8)
		x_coor_b   = np.zeros(len(tau), dtype=np.int8) + n_alpha

	n_theta = n_alpha + n_beta

	if apply_rep_bias:
		n_theta += 1

	samples                = np.random.rand(nb_samples, n_theta)
	if temperature:
		upp_bound_beta     = .6
	else:
		upp_bound_beta     = 2.

	samples[:, n_alpha:(n_beta + n_alpha)] = np.random.rand(nb_samples, n_beta) * upp_bound_beta

	if apply_rep_bias:
		samples[:, -1] = upp_bound_eta * (np.random.rand(nb_samples) * 2. - 1.)

	
	Q_samples   = np.zeros([nb_samples, 2]) + .5
	prev_action = np.zeros(nb_samples) - 1

	# ibis param
	esslist       = np.zeros(T)
	log_weights   = np.zeros(nb_samples)
	weights_a     = np.zeros(nb_samples)
	p_loglkd      = np.zeros(nb_samples)
	loglkd        = np.zeros(nb_samples)
	marg_loglkd   = 0
	coefficient   = .5
	marg_loglkd_l = np.zeros(T)
	acceptance_l  = []

	# move step param
	move_samples = np.zeros([nb_samples, n_theta])
	move_p_loglkd    = np.zeros(nb_samples)
	Q_samples_move   = np.zeros([nb_samples, 2])
	prev_action_move = np.zeros(nb_samples)
	mean_Q           = np.zeros([T, 2])

	if show_progress : plt.figure(figsize=(15,9)); plt.suptitle("noiseless rl", fontsize=14); plt.ion()

	# loop
	for t_idx in range(T):
		#print t_idx

		if (t_idx+1) % 10 == 0 : sys.stdout.write(' ' + str(t_idx+1) + ' '); print 'marg_loglkd ' + str(marg_loglkd); 
		if (t_idx+1) % 100 == 0: print ('\n')
		# epsilon
		assert(len(np.unique(prev_action)) == 1)
		# update step
		weights_a[:] = log_weights

		for n_idx in range(nb_samples):
			alpha                  = samples[n_idx, x_coor_a[t_idx]]
			if temperature:
				beta                     = 1./samples[n_idx, x_coor_b[t_idx]]
			else:
				beta                     = 10**samples[n_idx, x_coor_b[t_idx]]
			if apply_rep_bias:
				eta                  = samples[n_idx, -1]
			if prev_action[n_idx] != -1 and apply_rep_bias:
				value              = 1./(1. + np.exp(beta * (Q_samples[n_idx, 0] - Q_samples[n_idx, 1]) - np.sign(prev_action[n_idx] - .5) * eta))
				loglkd[n_idx]      = np.log(((value)**actions[t_idx]) * (1 - value)**((1 - actions[t_idx])))
				prev_action[n_idx] = actions[t_idx]
			else:
				value              = 1./(1. + np.exp(beta * (Q_samples[n_idx, 0] - Q_samples[n_idx, 1])))
				loglkd[n_idx]      = np.log(((value)**actions[t_idx]) * (1 - value)**((1 - actions[t_idx])))
				prev_action[n_idx] = actions[t_idx]	
			
			if np.isnan(loglkd[n_idx]):
				print t_idx
				print n_idx
				print beta
				print value
				raise Exception

			p_loglkd[n_idx]          = p_loglkd[n_idx] + loglkd[n_idx]
			log_weights[n_idx]       = log_weights[n_idx] + loglkd[n_idx]

			if actions[t_idx] == 0:
				Q_samples[n_idx, 0]          = (1 - alpha) * Q_samples[n_idx, 0] + alpha * rewards[t_idx]
				Q_samples[n_idx, 1]          = (1 - alpha) * Q_samples[n_idx, 1] + alpha * (1 - rewards[t_idx])
			else:
				Q_samples[n_idx, 0]          = (1 - alpha) * Q_samples[n_idx, 0] + alpha * (1 - rewards[t_idx])
				Q_samples[n_idx, 1]          = (1 - alpha) * Q_samples[n_idx, 1] + alpha * rewards[t_idx]

		marg_loglkd         += logsumexp(weights_a + loglkd) - logsumexp(weights_a)
		marg_loglkd_l[t_idx] = marg_loglkd
		ess                  = np.exp(2 * logsumexp(log_weights) - logsumexp(2 * log_weights))
		esslist[t_idx]       = ess

		weights_a[:]         = uf.to_normalized_weights(log_weights)
		mean_Q[t_idx]        = np.sum((Q_samples.T * weights_a).T, axis=0)

		# move step
		if ess < coefficient * nb_samples:
			idxTrajectories = uf.stratified_resampling(weights_a)
			mu_p            = np.sum(samples.T * weights_a, axis=1)
			Sigma_p         = np.dot((samples - mu_p).T * weights_a, (samples - mu_p))
			nb_acceptance   = 0.

			for n_idx in range(nb_samples):
				idx_traj = idxTrajectories[n_idx]
				while True:
					sample_p = multi_norm(mu_p, Sigma_p)
					if not apply_rep_bias:
						if np.all(sample_p[:n_alpha] > 0) and np.all(sample_p[:n_alpha] < 1) and np.all(sample_p[n_alpha:(n_beta + n_alpha)] > 0) and np.all(sample_p[n_alpha:(n_beta + n_alpha)] <= upp_bound_beta):
							break
					else:
						if np.all(sample_p[:n_alpha] > 0) and np.all(sample_p[:n_alpha] < 1) and np.all(sample_p[n_alpha:(n_beta + n_alpha)] > 0) and np.all(sample_p[n_alpha:(n_beta + n_alpha)] <= upp_bound_beta) and sample_p[-1] > -upp_bound_eta and sample_p[-1] < upp_bound_eta:
							break
				[loglkd_prop, Q_prop, prev_action_prop] = get_loglikelihood(sample_p, x_coor_a, x_coor_b, rewards, actions, t_idx + 1, apply_rep_bias, temperature) 
				log_ratio                               = loglkd_prop - p_loglkd[idx_traj] \
													         + get_logtruncnorm(samples[idx_traj], mu_p, Sigma_p) - get_logtruncnorm(sample_p, mu_p, Sigma_p)

				log_ratio = np.minimum(log_ratio, 0)
				if (np.log(np.random.rand()) < log_ratio):
					nb_acceptance          += 1.
					move_samples[n_idx]     = sample_p
					move_p_loglkd[n_idx]    = loglkd_prop
					Q_samples_move[n_idx]   = Q_prop
				else:
					move_samples[n_idx]     = samples[idx_traj]
					move_p_loglkd[n_idx]    = p_loglkd[idx_traj]
					Q_samples_move[n_idx]   = Q_samples[idx_traj]

			print 'acceptance ratio %s'%str(nb_acceptance/nb_samples)
			assert(prev_action_prop == prev_action[0])

			acceptance_l.append(nb_acceptance/nb_samples)
			# move samples
			samples[:]     = move_samples
			p_loglkd[:]    = move_p_loglkd
			log_weights[:] = 0.
			Q_samples[:]   = Q_samples_move

		if show_progress and t_idx%10==0 :
			weights_a[:]    = uf.to_normalized_weights(log_weights)

			plt.subplot(3,2,1)
			plt.plot(range(t_idx), mean_Q[:t_idx], 'm', linewidth=2);
			plt.hold(False)
			plt.xlabel('trials')
			plt.ylabel('Q values')

			if apply_rep_bias == 1:
				mean_rep = np.sum(weights_a * samples[:,2])
				std_rep  = np.sqrt(np.sum(weights_a * samples[:,2]**2) - mean_rep**2)
				plt.subplot(3,2,2)
				x = np.linspace(-2.,2.,5000)
				plt.plot(x, norm.pdf(x, mean_rep, std_rep), 'g'); plt.hold(True)
				plt.plot([mean_rep, mean_rep], plt.gca().get_ylim(),'g', linewidth=2)
				plt.hold(False)
				plt.xlabel('trials')
				plt.ylabel('rep param')

			if temperature:
				mean_beta = np.sum(weights_a * 1./samples[:, 1])
				std_beta  = np.sqrt(np.sum(weights_a * ((1./samples[:,1])**2)) - mean_beta**2)
			else:
				mean_beta = np.sum(weights_a * 10**samples[:, 1])
				std_beta  = np.sqrt(np.sum(weights_a * ((10**samples[:,1])**2)) - mean_beta**2)
			plt.subplot(3,2,3)
			x = np.linspace(0.01,200.,5000)
			plt.plot(x, norm.pdf(x, mean_beta, std_beta), 'g', linewidth=2); plt.hold(True)
			plt.plot([mean_beta, mean_beta], plt.gca().get_ylim(), 'g', linewidth=2)
			plt.hold(False)
			plt.xlabel('beta softmax')
			plt.ylabel('pdf')

			mean_alpha_0 = np.sum(weights_a * samples[:, 0])
			std_alpha_0  = np.sqrt(np.sum(weights_a * (samples[:, 0]**2)) - mean_alpha_0**2)
			plt.subplot(3,2,4)
			x = np.linspace(0.,1.,5000)
			plt.plot(x, norm.pdf(x, mean_alpha_0, std_alpha_0), 'm', linewidth=2); plt.hold(True)
			plt.plot([mean_alpha_0, mean_alpha_0], plt.gca().get_ylim(), 'm', linewidth=2)
			plt.hold(False)
			plt.xlabel('learning rate (majenta)')
			plt.ylabel('pdf')

			plt.subplot(3,2,5)
			plt.plot(range(t_idx), esslist[:t_idx], 'b', linewidth=2); plt.hold(True)
			plt.plot(plt.gca().get_xlim(), [nb_samples/2,  nb_samples/2],'b--', linewidth=2)
			plt.axis([0, t_idx-1, 0, nb_samples]) # For speed
			plt.hold(False)
			plt.xlabel('trials')
			plt.ylabel('ess')

			plt.draw()
			plt.show()
			plt.pause(0.05)
		
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

def get_loglikelihood(sample, x_coor_a, x_coor_b, rewards, actions, T, apply_rep_bias, temperature=True):
	
	if apply_rep_bias:
		eta = sample[-1]
	prev_action    = -1
	log_proba      = 0
	Q0             = .5
	Q1             = .5

	for t_idx in range(T):
		action = actions[t_idx]
		alpha  = sample[x_coor_a[t_idx]]

		if temperature:
			beta     = 1./sample[x_coor_b[t_idx]]
		else:
			beta     = 10**sample[x_coor_b[t_idx]]

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


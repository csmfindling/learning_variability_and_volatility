import numpy as np
import smc_c
import pickle
from scipy.stats import norm, truncnorm, beta, multivariate_normal
import matplotlib.pyplot as plt
import sys
import math
import bayesopt

def format_params(inferred_params, mon_size = 3, ctxt_prior = 1, ctxt_decay = 0, bias_ctxt = 0, repulsion = 0, rl_decay = 0, bay_decay = 0):
	return np.array([mon_size, inferred_params[0], rl_decay, inferred_params[1], bay_decay, inferred_params[2], inferred_params[3], inferred_params[4], inferred_params[5], inferred_params[6], ctxt_prior, ctxt_decay, bias_ctxt, repulsion])

def compute_marginals(point):
	# general parameters
	info     = pickle.load(open('data/open_data_case_' + str(0) + '.pkl', 'rb'))
	stim_seq = np.array(info['S'], dtype = np.intc)
	act_seq  = np.array(info['A_chosen'], dtype = np.intc)
	rewards  = np.array(info['reward'], dtype = np.intc)
	nS       = 3
	nC       = 0
	nA       = 4
	nTrials  = len(rewards)
	mon_size = 3
	nSamples = 5000

	rl_alpha      = point[0]
	bay_prior     = 10**((point[1]) * 2 + 1.)
	softmax_beta  = point[2] * 5. + 5.
	softmax_eps   = point[3] / 10.
	volatility    = point[4]
	bias_conf     = (point[5]) * 4 - 2.
	bias_ini      = point[6]

	# parameters for smc and names
	inferred_params       = np.array([rl_alpha, bay_prior, softmax_beta, softmax_eps, volatility, bias_conf, bias_ini])

	# P-MCMC
	all_param   = format_params(inferred_params, mon_size)
	lkd         = -smc_c.bootstrap_smc_c(nSamples, nS, nC, stim_seq, act_seq, rewards, all_param, nA, nTrials)
	#lkd         = -smc_c.get_marg_lkd(nS, nC, stim_seq, act_seq, rewards, all_param, nA, nTrials)

	return lkd



params = {} #bayesopt.initialize_params()

# We decided to change some of them
params['n_iterations']   = 200
params['n_init_samples'] = 5
params['n_iter_relearn'] = 5
# params['l_type'] = "mcmc"
params['noise'] = 1
params['kernel_name'] = "kMaternARD5"
# params['kernel_hp_mean'] = [1]
# params['kernel_hp_std'] = [5]
# params['surr_name'] = "sStudentTProcessNIG"
# params['surr_name'] = "sStudentTProcessNIG"
params['load_save_flag'] = 2

dim = 7
lb = np.ones((dim,))*0.
ub = np.ones((dim,))*1.

mvalue, x_out, error = bayesopt.optimize(compute_marginals, dim, lb, ub, params)




params = np.array([3.0000, 0.0911, 0, 21.9868, 0, 22.3006, 0.2053, 0.7538, -1.4208, 0.8877, 1.0000, 0, 0, 0]);
params = np.array([0.0911, 21.9868, 22.3006, 0.2053, 0.7538, -1.4208, 0.8877 ]);




#python /Users/csmfindling/spearmint/spearmint/spearmint/main.py --driver=local --method=GPEIOptChooser config.pb
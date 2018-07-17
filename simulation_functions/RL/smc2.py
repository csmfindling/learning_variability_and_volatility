import numpy as np
from scipy.stats import gamma, norm
import matplotlib.pyplot as plt
import sys
sys.path.append("../useful_functions/")
import useful_functions as uf
import warnings

# td=td_list[0]; sample=theta; forced_actions = None;  beta_softmax=-1; apply_guided=0; apply_weber=1;apply_cst_noise=1; apply_inertie=1;nb_traj=1
def simulate_noiseless_rl(td, rewards, sample, tau, nb_traj=1):
	assert(nb_traj==1)
	T                  = rewards.shape[-1]
	noisy_trajectories = np.zeros([T, 2])
	actions_simul      = np.zeros(T) - 1
	prev_act		   = - 1
	beta_softmax       = sample[-1]
	rew_sim            = np.zeros(T)
	vol_unique         = np.unique(tau)
	performance        = np.zeros(T)
	for t_idx in range(T):
		if t_idx > 0.:
			alpha    = sample[0] #sample[np.where(td['tau'][t_idx - 1] == vol_unique)[0][0]]
			prev_rew = rewards[:,t_idx - 1]
				
			if actions_simul[t_idx - 1] == 0:
				mu0 = (1 - alpha) * noisy_trajectories[t_idx - 1, 0] + alpha * prev_rew[0]
				mu1 = (1 - alpha) * noisy_trajectories[t_idx - 1, 1] + alpha * prev_rew[1]
			else:
				mu0 = (1 - alpha) * noisy_trajectories[t_idx - 1, 0] + alpha * prev_rew[0]
				mu1 = (1 - alpha) * noisy_trajectories[t_idx - 1, 1] + alpha * prev_rew[1]

			noisy_trajectories[t_idx, 0] = mu0;
			noisy_trajectories[t_idx, 1] = mu1;

			# probability to choose 1
			proba_1 = 1./(1. + np.exp(beta_softmax * (noisy_trajectories[t_idx, 0] - noisy_trajectories[t_idx, 1])))

			# simulate action
			if np.random.rand() < proba_1:
				actions_simul[t_idx] = 1
			else:
				actions_simul[t_idx] = 0

			rew_sim[t_idx] = rewards[actions_simul[t_idx].astype(int), t_idx]
			# prev action
			prev_act = actions_simul[t_idx]

		else:
			noisy_trajectories[t_idx]   = 0.5
			prev_act                    = -1
			if np.random.rand() < .5:
				actions_simul[t_idx] = 1
			else:
				actions_simul[t_idx] = 0

			rew_sim[t_idx] = rewards[actions_simul[t_idx].astype(int), t_idx]
			# previous action
			prev_act = actions_simul[t_idx]
		if actions_simul[t_idx] == td['A_correct'][t_idx]:
			performance[t_idx:] += 1
		
	return td, noisy_trajectories, actions_simul, rew_sim, performance

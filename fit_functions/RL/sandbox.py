


for i in range(nb_samples):
	sample_p = samples[i]
	[loglkd_prop, Q_prop, prev_action_prop] = get_loglikelihood(sample_p, x_coor_a, x_coor_b, rewards, actions, t_idx + 1, apply_rep_bias) 
	print loglkd_prop
	print p_loglkd[i]
	print '\n'


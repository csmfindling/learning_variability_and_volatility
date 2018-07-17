//
//  smc.cpp
//  smc_probe_1
//
//  Created by Charles Findling on 9/7/16.
//  Copyright © 2016 csmfindling. All rights reserved.
//

#include "smc.hpp"
#include <boost/random/mersenne_twister.hpp>
#include <math.h>
#include <stdexcept>
#include "particle.hpp"
#include "usefulFunctions.hpp"
#include "particle.cpp"
#include "usefulFunctions.cpp"

namespace smc {
    
    using namespace std;
    
    boost::mt19937 generator(static_cast<unsigned int>(time(0)));
    
    double get_marg_lkd(int nS, int nC, int* stim_seq, int* act_seq, int* rewards, int* act_corr, int* trap_trials, double* params, int nA, int nTrials, double* output_probabilities)
    {
        std::cout << "noiseless model" << std::endl;
        generator.discard(70000);
        bool f_rl_a_norm        = false;
        bool f_rl_no_act_decay  = false;
        bool f_rl_prediction    = false;
        bool f_bay_norm         = true;
        bool f_bay_no_act_decay = false;
        bool f_bay_selection    = false;
        bool f_rl_s_norm        = true; //true;
        bool f_ctxt_nonbay      = false;
        
        // mimum allows probability (used in some limiting cases)
        double p_min  = 1e-60;
        int N_samples = 1;
        
        // model parameters
        int mon_size        = *(params);
        double rl_alpha     = *(params + 1);
        double rl_decay     = *(params + 2);
        double bay_prior    = *(params + 3);
        double bay_decay    = *(params + 4);
        double softmax_beta = *(params + 5);
        double softmax_eps  = *(params + 6);
        double volatility   = *(params + 7);
        double bias_conf    = *(params + 8);
        double bias_ini     = *(params + 9);
        double ctxt_prior   = *(params + 10);
        double ctxt_decay   = *(params + 11);
        double bias_ctxt    = *(params + 12);
        double repulsion    = *(params + 13);
        
        // verif
        if ((bay_decay != 0)||(f_bay_norm != 1)||(nC > 0)||(f_ctxt_nonbay != 0)||(repulsion!=0))
        {
            throw std::invalid_argument( "received invalid flag value" );
        }

        // 'previous' context, needed for LTM initialisation
        // priors for rl and bayes
        vector<double> bay_p_prior(nA * nS, bay_prior/nA);
        vector<double> rl_p_prior(nA * nS, 1./nA);
        
        double mu_0 = ( 1. / (1-2*volatility) ) * ( ( 1 - volatility ) * 0.3 - volatility * 0.7 );
        double mu_1 = ( 1. / (1-2*volatility) ) * ( ( 1 - volatility ) * 0.7 - volatility * 0.3 );
        
        vector<particle> descendants;
        vector<particle> ancestors;

        for (int n_sample = 0; n_sample < N_samples; ++n_sample) {
            descendants.push_back(particle(nA, nS, rl_p_prior, bay_p_prior, mu_0, mu_1));
            ancestors.push_back(particle(nA, nS, rl_p_prior, bay_p_prior, mu_0, mu_1));
        }

        //print(descendants[0].get_bay_p());

        std::vector<double> act_val(nA, 1.);
        vector<double> act_prob(nA, 1.);
        vector<double> w_traj(N_samples, 1.);
        vector<double> w_traj_norm(N_samples, 1.);
        vector<int> ancestors_traj(N_samples, 1);
        double log_marglkd = 0;
        vector<double> reward_lh;
        vector<double> gamma_sample;
        vector<double> coef_dirichlet;
        
        for (int t = 0; t < nTrials; ++t) {
            //std::cout << t << std::endl;
            int cur_stim = *(stim_seq + t);
            int cur_act  = *(act_seq + t);
            
            if (t > 0)
            {
                w_traj_norm    = w_traj / sum(w_traj);
                ancestors_traj = stratified_resampling(generator, w_traj_norm);
                for (int n_sample = 0; n_sample < N_samples; ++n_sample)
                {
                    ancestors[n_sample].set_particle(descendants[n_sample]);
                }
            }
            
            for (int n_sample = 0; n_sample < N_samples; ++n_sample) {
                
                particle& descendant = descendants[n_sample];
                
                if (t > 0)
                {
                    descendant.set_particle(ancestors[ancestors_traj[n_sample]]);
                    descendant.verification();
                    descendant.update_l_deterministic(volatility);
                }
                descendant.verification();
                // ---------- (2) control: stay/switch ----------
                if (descendant.get_state() == 1)
                { // currently exploiting
                    if (descendant.get_l(descendant.get_actorTS()) > 0.5)
                    { // current task set has confidence >= 0.5 - keep using it
                        descendant.set_state(1);
                    }
                    else
                    { 
                        // actor task set no longer fits - check currently best task set
                        std::pair<double, int> max_l_monTS = descendant.get_max_l_monTS();
                        double lmax = max_l_monTS.first; int imax = max_l_monTS.second;
                        
                        if ((lmax > 0.5)&&(imax != 0))
                        { // task set other than dummy has confidence > 0.5 - switch
                            descendant.set_actorTS(descendant.get_monTS(imax));
                            descendant.set_state(1);
                            
                          // move new actor task set to end of monitoring buffer
                            descendant.erase_monTS(imax); descendant.push_back_monTS(descendant.get_actorTS());
                        }
                        else
                        { // no task set has confidence > 0.5 - build probe
                            descendant.increment_nTS(); //explore with newly created task set
                            int prev_actorTS = descendant.get_actorTS(); // store previous actor TS
                            descendant.set_actorTS(descendant.get_nTS() - 1);
                            descendant.set_state(2);
                            
                          // new ex-ante confidence, based on entropy
                            double entropy = descendant.get_entropy(p_min); // entropy
                            double lopt    = 1 / (1 + exp(entropy)); // entropy-maximising confidence
                            double lprob   = min(1., max(0., bias_conf * 0.5 + (1 - bias_conf) * lopt));
                            descendant.multiply_l_monTS(1 - lprob);
                            descendant.push_back_l(lprob, descendant.get_actorTS());
                            descendant.push_back_mu(descendant.get_actorTS());
                            
                          // avg Q-values and predictive model from long-term memory
                            std::vector<double> avg_rl  = descendant.get_ltm_avg_rl();
                            std::vector<double> avg_bay = descendant.get_ltm_avg_bay();
                            std::vector<double> prev_bay_p = descendant.get_bay_p_ts(prev_actorTS);
                            descendant.push_back_bay_p(avg_bay, bay_p_prior, bay_prior, bias_ini, descendant.get_actorTS());
                            descendant.push_back_rl_p(avg_rl, rl_p_prior, bias_ini, descendant.get_actorTS());
                            descendant.push_back_monTS(descendant.get_actorTS());
                        }
                    }
                }
                else
                { // currently exploring
                    std::pair<double, int> max_l_monTS = descendant.get_max_l_monTS();
                    double lmax = max_l_monTS.first; int imax = max_l_monTS.second;
                    int emergeTS = descendant.get_monTS(imax);
                 
                    if ((lmax > 0.5) && (imax != 0)) {
                        // some task set has confidence > 0.5 - establish and exploit
                        if (emergeTS != descendant.get_actorTS()) {
                            //  new task set is not probe set - remove probe
                            descendant.pop_back_monTS(); // remove probe from buffer
                            descendant.set_last_l((descendant.get_nTS() - 1), 0.); // probe confidence = 0
                            descendant.normalise_l(); // re-normalise
                            // move active TS to end
                            descendant.erase_monTS(imax);
                            descendant.push_back_monTS(emergeTS);
                        }
                        else
                        { // probe set becomes active task set
                            descendant.push_back_ltmTS(descendant.get_actorTS()); // register in long-term mem
                            if (descendant.get_monTS_size() >= mon_size + 2) {
                                // out of memory - remove oldest task set
                                descendant.set_l(descendant.get_monTS(1), 0.);
                                descendant.normalise_l();
                                descendant.erase_monTS(1);
                            }
                        }
                        descendant.set_state(1);
                        descendant.set_actorTS(emergeTS);
                    }
                    else
                    { // no task set has confidence > 0.5 - keep exploring
                        descendant.set_state(2);
                    }
                }
                descendant.verification();
                // ---------- (3) action selection & feedback ----------
                if (f_bay_selection) {
                    act_val = descendant.get_bay_act_val(descendant.get_actorTS(), cur_stim);
                }
                else{
                    act_val = descendant.get_rl_act_val(descendant.get_actorTS(), cur_stim);
                }


                int cur_rew      = -1;
                if (cur_act >= 0) {
                    // compute action probability by softmax
                    act_prob = descendant.get_act_prob(act_val, softmax_beta, softmax_eps);
                    w_traj[n_sample] = act_prob[cur_act];
                    cur_rew = *(rewards + t);
                }
                else{
                    act_prob         = descendant.get_act_prob(act_val, softmax_beta, softmax_eps);        
                    cur_act          = Sample_Discrete_Distribution(generator, act_prob);
                    *(act_seq + t)   = cur_act;
                    w_traj[n_sample] = act_prob[cur_act];
                    cur_rew          = -1;
                    if (cur_act == *(act_corr + t))
                    {
                        if (*(trap_trials + t) == 0)
                        {
                            cur_rew = 1;
                        }
                        else
                        {
                            cur_rew = 0;
                        }
                    }
                    else
                    {
                        if (*(trap_trials + t) == 0)
                        {
                            cur_rew = 0;
                        }
                        else
                        {
                            cur_rew = 1;
                        }
                    }
                    *(rewards + t) = cur_rew;
                }

                for (int action_idx = 0; action_idx < nA; ++action_idx)
                {
                    *(output_probabilities + t * nA + action_idx) = act_prob[action_idx];
                }

                if (cur_rew == -1)
                {
                    throw std::invalid_argument( "something is wrong, reward is -1" );
                }

                // ---------- (4) updated confidence based on reward ----------
                reward_lh = descendant.get_reward_lh(cur_act, cur_stim, cur_rew);
                
                descendant.update_mu(reward_lh);

                // ---------- (5) update of internal models ----------
                // update RL model by standard TD learning + ev. normalization
                
                descendant.set_rl_p(descendant.get_actorTS() * nA * nS + cur_stim * nA + cur_act, (1 - rl_alpha) * descendant.get_rl_p(descendant.get_actorTS() * nA * nS + cur_stim * nA + cur_act) + rl_alpha * cur_rew);
                
                if (f_rl_a_norm) {
                    // if requested, action-normalise values
                    descendant.normalise_rl_act(cur_stim, descendant.get_actorTS());
                }
                if (f_rl_s_norm) {
                    // if requested, stimulus-normalise values
                    descendant.normalise_rl_stim(cur_act, cur_stim, cur_rew, descendant.get_actorTS(), rl_alpha);
                }
                
                // update Bayesian model
                descendant.update_bay(cur_stim, cur_rew, cur_act, descendant.get_actorTS());
                descendant.verification();
            }
            log_marglkd += log(sum(w_traj)/N_samples);
        }
        return log_marglkd;
    };

}






















//
//  particle.hpp
//  smc_probe_1
//
//  Created by Charles Findling on 9/7/16.
//  Copyright © 2016 csmfindling. All rights reserved.
//

#ifndef particle_hpp
#define particle_hpp

#include <stdio.h>
#include <vector>
#include <boost/random/mersenne_twister.hpp>

class particle
{
public:
    particle(int n_action, int n_stim, std::vector<double> rl_p_prior, std::vector<double> bay_p_prior, double mu_0, double mu_1);
    
    void set_nTS(int value);
    void set_state(int value);
    void set_actorTS(int value);
    void set_rl_p(int idx, double value);
    void set_bay_p(int idx, double value);
    void set_monTS(int idx, int value);
    void set_ltmTS(int idx, int value);
    void set_l(int idx, double value);
    void set_mu(int idx, double value);
    void set_last_l(int idx, double value);

    void set_rl_p(std::vector<double> value);
    void set_bay_p(std::vector<double> value);
    void set_monTS(std::vector<int> value);
    void set_ltmTS(std::vector<int> value);
    void set_l(std::vector<double> value);
    void set_mu(std::vector<double> value);
    void set_l_monTS(std::vector<double> value);
    void set_l_monTS_mu();
    
    void push_back_bay_p(std::vector<double> avg_bay, std::vector<double> bay_p_prior, double bay_prior, double bias_ini, int ts);
    void push_back_rl_p(std::vector<double> avg_rl, std::vector<double> rl_p_prior, double bias_ini, int ts);
    void push_back_monTS(int value);
    void pop_back_monTS();
    void push_back_ltmTS(int value);
    void push_back_l(double value, int TS);
    void push_back_mu(int TS);
    void normalise_l();
    
    void verification();
    
    void set_particle(particle value);
    
    int get_nTS();
    int get_state();
    int get_actorTS();
    
    double get_rl_p(int idx);
    double get_bay_p(int idx);
    int get_monTS(int idx);
    int get_ltmTS(int idx);
    double get_l(int idx);
    double get_mu(int idx);
    int get_monTS_size();
    
    void erase_monTS(int idx);
    void increment_nTS();
    double get_entropy(double p_min);
    void multiply_l(int idx, double multiplier);
    void multiply_l_monTS(double multiplier);
    void update_mu(std::vector<double> reward_lh);
    void normalise_rl_act(int stim, int ts);
    void normalise_rl_stim(int act, int stim, int rew, int ts, double rl_alpha);
    void update_bay(int stim, int rew, int act, int ts);
    void update_l_deterministic(double volatility);
    
    std::vector<double> get_sampled_l(double err, boost::mt19937 &generator);
    std::vector<double> get_ltm_avg_rl();
    std::vector<double> get_ltm_avg_bay();
    std::vector<double> get_bay_p_ts(int ts);
    std::vector<double> get_bay_act_val(int ts, int stim);
    std::vector<double> get_rl_act_val(int ts, int stim);
    std::vector<double> get_act_prob(std::vector<double> act_val, double softmax, double softmax_eps);
    std::vector<double> get_reward_lh(int cur_act, int cur_stim, int cur_reward);
    std::pair<double,int> get_max_l_monTS();
    
    std::vector<double> get_rl_p();
    std::vector<double> get_bay_p();
    std::vector<int> get_monTS();
    std::vector<int> get_ltmTS();
    std::vector<double> get_l();
    std::vector<double> get_mu();
    
    double get_err(double volatility);
    
private:
    int nTS;
    int state;
    int actorTS;
    std::vector<double> rl_p;
    std::vector<double> bay_p;
    std::vector<int> monTS;
    std::vector<int> ltmTS;
    std::vector<double> l;
    std::vector<double> mu;
    int nS;
    int nA;
};

#endif /* particle_hpp */

//
//  particle.cpp
//  smc_probe_1
//
//  Created by Charles Findling on 9/7/16.
//  Copyright © 2016 csmfindling. All rights reserved.
//

#include "particle.hpp"
#include "usefulFunctions.hpp"
#include <iostream>
#include <stdexcept>
#include <boost/random/mersenne_twister.hpp>


particle::particle(int n_action, int n_stim, std::vector<double> rl_p_prior, std::vector<double> bay_p_prior, double mu_0, double mu_1)
{
    nTS = 2;
    actorTS = 1;
    state = 1;
    
    nA = n_action;
    nS = n_stim;
    
    rl_p.resize(2 * nA * nS);
    copy(rl_p_prior.begin(), rl_p_prior.end(), rl_p.begin());
    copy(rl_p_prior.begin(), rl_p_prior.end(), rl_p.begin() + nA * nS);
    
    bay_p.resize(2 * nA * nS);
    copy(bay_p_prior.begin(), bay_p_prior.end(), bay_p.begin());
    copy(bay_p_prior.begin(), bay_p_prior.end(), bay_p.begin() + nA * nS);

    monTS.resize(2);
    monTS[0] = 0; monTS[1] = 1;
    
    ltmTS.resize(1);
    ltmTS[0] = 1;
    
    l.resize(2);
    l[0] = 0.3; l[1] = 0.7;
    
    mu.resize(2);
    mu[0] = mu_0; mu[1] = mu_1;
}

void particle::set_particle(particle value)
{
    set_actorTS(value.get_actorTS());
    set_state(value.get_state());
    set_nTS(value.get_nTS());
    set_bay_p(value.get_bay_p());
    set_rl_p(value.get_rl_p());
    set_monTS(value.get_monTS());
    set_ltmTS(value.get_ltmTS());
    set_l(value.get_l());
    set_mu(value.get_mu());
}

std::pair<double,int> particle::get_max_l_monTS()
{
    int imax; double lmax = -1;
    for (int idx = 0; idx < get_monTS_size(); ++idx) {
        if (l[monTS[idx]] > lmax) {
            lmax = l[monTS[idx]];
            imax = idx;
        }
    }
    return std::pair<double, int> (lmax, imax);
}

void particle::set_l_monTS(std::vector<double> value)
{
    if (value.size() != get_monTS_size()) {
        throw std::invalid_argument( "value doest not have right size" );
    }
    for (int monTS_idx = 0; monTS_idx < get_monTS_size(); ++monTS_idx) {
        l[monTS[monTS_idx]] = value[monTS_idx];
    }
}

void particle::update_l_deterministic(double volatility)
{
    for (int monTS_idx = 0; monTS_idx < get_monTS_size(); ++monTS_idx) {
        l[monTS[monTS_idx]] = mu[monTS[monTS_idx]] * (1. - volatility) + volatility / (get_monTS_size() - 1.) * (1. - mu[monTS[monTS_idx]]);
    }
}

void particle::set_l_monTS_mu()
{
    if (mu.size() != l.size()) {
        throw std::invalid_argument( "value doest not have right size" );
    }
    for (int monTS_idx = 0; monTS_idx < get_monTS_size(); ++monTS_idx)
    {
        l[monTS[monTS_idx]] = mu[monTS[monTS_idx]];
    }
}

double particle::get_err(double volatility)
{
    if ((l.size() != nTS)||(mu.size() != nTS))
    {
        throw std::invalid_argument( "l or mu doest not have right size" );
    }
    if ((std::isnan(l[0]))||(std::isnan(mu[0])))
    {
        throw std::invalid_argument( "l or mu nan" );
    }
    double err = 0;
    for (int idx = 0; idx < get_monTS_size(); ++idx) {
        err += std::abs(mu[monTS[idx]] - l[monTS[idx]]);
    }
    return err/volatility;
}

std::vector<double> particle::get_sampled_l(double err, boost::mt19937 &generator)
{
    std::vector<double> coeff_dirichlet(get_monTS_size(), 0.);
    for (int monTS_idx = 0; monTS_idx < get_monTS_size(); ++monTS_idx) {
        coeff_dirichlet[monTS_idx] = std::max(mu[monTS[monTS_idx]]/err, 1.);
    }
    std::vector<double> g_sample = Sample_Dirichlet_Distribution(generator, &coeff_dirichlet[0], (int)coeff_dirichlet.size());
    
    return g_sample;
}

std::vector<double> particle::get_bay_act_val(int ts, int stim)
{
    std::vector<double> output(nA, 0);
    double sum_output = 0.;
    for (int act_idx = 0; act_idx < nA; ++act_idx) {
        output[act_idx] = bay_p[ts * nA * nS + stim * nA + act_idx];
        sum_output += output[act_idx];
    }
    if (sum_output == 0) {
        throw std::invalid_argument( "sum_output nul" );
    }
    return output/sum_output;
}

std::vector<double> particle::get_rl_act_val(int ts, int stim)
{
    std::vector<double> output(nA, 0);
    for (int act_idx = 0; act_idx < nA; ++act_idx) {
        output[act_idx] = rl_p[ts * nA * nS + stim * nA + act_idx];
    }
    return output;
}

void particle::normalise_rl_stim(int act, int stim, int rew, int ts, double rl_alpha)
{
    for (int stim_idx = 0; stim_idx < nS; ++stim_idx) {
        if (stim_idx != stim) {
            rl_p[ts * nA * nS + stim_idx * nA + act] = (1. - rl_alpha) * rl_p[ts * nA * nS + stim_idx * nA + act] + rl_alpha * (1. - rew)/(nS - 1);
        }
    }
}

void particle::update_bay(int stim, int rew, int act, int ts)
{
    for (int act_idx = 0; act_idx < nA; ++act_idx) {
        if (act_idx != act) {
            bay_p[ts * nA * nS + stim * nA + act_idx] = bay_p[ts * nA * nS + stim * nA + act_idx] + (1. - rew)/(nA - 1);
        }
        else
        {
            bay_p[ts * nA * nS + stim * nA + act_idx] = bay_p[ts * nA * nS + stim * nA + act_idx] + (1. - rew)/(nA - 1) + rew * nA/(nA - 1.) - 1./(nA - 1);
        }
    }
}

void particle::update_mu(std::vector<double> reward_lh)
{
    if ((reward_lh.size() != get_monTS_size())||(bay_p.size()/12 != l.size())||(mu.size()!=get_nTS())) {
        throw std::invalid_argument( "reward_lh or mu doest not have right size" );
    }

    mu.assign(get_nTS(), 0.);

    for (int monTS_idx = 0; monTS_idx < get_monTS_size(); ++monTS_idx) {
        mu[monTS[monTS_idx]] = reward_lh[monTS_idx] * l[monTS[monTS_idx]];
    }
    if (sum(mu) == 0) {
        print(mu);
        std::cout << get_monTS_size() << std::endl;
        print(l);
        print(reward_lh);
        print(monTS);
        throw std::invalid_argument( "sum_output nul" );
    }
    mu = mu/sum(mu);
}


std::vector<double> particle::get_reward_lh(int cur_act, int cur_stim, int cur_reward)
{
    if (l.size() != mu.size()||(bay_p.size()/12 != l.size())) {
        throw std::invalid_argument( "l or monTS doest not have right size" );
    }
    
    std::vector<double> output(get_monTS_size());
    double sum_output = 0;
    
    for (int monTS_idx = 0; monTS_idx < get_monTS_size(); ++monTS_idx) {
        output[monTS_idx] = bay_p[monTS[monTS_idx] * nA * nS + cur_stim * nA + cur_act];
        sum_output = 0;
        for (int act_idx = 0; act_idx < nA; ++act_idx) {
            sum_output += bay_p[monTS[monTS_idx] * nA * nS + cur_stim * nA + act_idx];
        }
        if (sum_output == 0) {
            throw std::invalid_argument( "sum_output nul" );
        }
        output[monTS_idx] /= sum_output;
    }
    
    for (int monTS_idx = 0; monTS_idx < get_monTS_size(); ++monTS_idx) {
        if (cur_reward == 1) {
            output[monTS_idx] = output[monTS_idx];
        }
        else if (cur_reward == 0)
        {
            output[monTS_idx] = 1. - output[monTS_idx];
        }
        else
        {
            throw std::invalid_argument( "problem with rewards" );
        }
    }
    return  output;
}

void particle::normalise_rl_act(int stim, int ts)
{
    double sum_normalise_rl = 0.;
    for (int act_idx = 0; act_idx < nA; ++act_idx) {
        sum_normalise_rl += rl_p[ts * nA * nS + stim * nA + act_idx];
    }
    if (sum_normalise_rl == 0) {
        throw std::invalid_argument( "sum_output nul" );
    }
    for (int act_idx = 0; act_idx < nA; ++act_idx) {
        rl_p[ts * nA * nS + stim * nA + act_idx] /= sum_normalise_rl;
    }
}

std::vector<double> particle::get_act_prob(std::vector<double> act_val, double softmax, double softmax_eps)
{
    std::vector<double> output(nA, 0);
    if (act_val.size() != nA) {
        throw std::invalid_argument( "act_val doest not have right size" );
    }
    double max_elem = *std::max_element(act_val.begin(), act_val.end());
    double sum_output = 0.;
    for (int act_idx = 0; act_idx < nA ; ++act_idx) {
        output[act_idx] = exp(softmax * (act_val[act_idx] - max_elem));
        sum_output     += output[act_idx];
    }
    for (int act_idx = 0; act_idx < nA; ++act_idx) {
        output[act_idx] = softmax_eps / nA + (1. - softmax_eps) * output[act_idx] / sum_output;
    }
    return output;
}

void particle::erase_monTS(int idx)
{
    monTS.erase(monTS.begin() + idx);
}

void particle::verification()
{
    if ((std::isnan(l[0]))||(std::isnan(l[1]))||(std::isnan(mu[1]))||(std::isnan(mu[0])))
    {
        throw std::invalid_argument( "l or mu nan" );
    }
}

void particle::push_back_monTS(int value)
{
    monTS.push_back(value);
}

void particle::push_back_ltmTS(int value)
{
    ltmTS.push_back(value);
}

void particle::pop_back_monTS()
{
    monTS.pop_back();
}

void particle::push_back_l(double value, int TS)
{
    if (l.size() >= (TS + 1)) {
        throw std::invalid_argument( "l does not have correct size or TS not new" );
    }
    l.push_back(value);
}

void particle::push_back_mu(int TS)
{
    if (mu.size() >= (TS + 1)) {
        throw std::invalid_argument( "mu does not have correct size or TS not new" );
    }
    mu.push_back(0.);
}

void particle::set_last_l(int idx, double value)
{
    if ((actorTS != l.size() - 1)||(idx != actorTS)) {
        throw std::invalid_argument( "wrong actorTS or wrong size of l" );
    }
    l[actorTS] = value;
}

void particle::normalise_l()
{
    if (l.size() != nTS) {
        throw std::invalid_argument( "wrong size of l" );
    }
    if (sum(l) == 0) {
        throw std::invalid_argument( "sum_output nul" );
    }
    l = l/sum(l);
}

void particle::increment_nTS()
{
    nTS = nTS + 1;
}

double particle::get_entropy(double p_min)
{
    double entropy = 0;
    for (int idx = 0; idx < get_monTS_size(); ++idx) {
        entropy -= log(std::max(p_min, l[monTS[idx]])) * l[monTS[idx]];
    }
    return entropy;
}

void particle::multiply_l(int idx, double multiplier)
{
    l[idx] *= multiplier;
}

void particle::multiply_l_monTS(double multiplier)
{
    for (int idx = 0; idx < get_monTS_size(); ++idx) {
        multiply_l(monTS[idx], multiplier);
    }
}

std::vector<double> particle::get_ltm_avg_rl()
{
    std::vector<double> ltm_avg_rl(nA * nS);
    int size_ltm = (int)ltmTS.size();
    
    for (int act_idx = 0; act_idx < nA; ++act_idx)
    {
        for (int stim_idx = 0; stim_idx < nS ; ++stim_idx)
        {
            for (int idx = 0; idx < size_ltm; ++idx)
            {
                int idx_ts = ltmTS[idx];
                ltm_avg_rl[stim_idx * nA + act_idx] += rl_p[idx_ts * nA * nS + stim_idx * nA + act_idx]/size_ltm;
            }
            
        }
    }
    return ltm_avg_rl;
}

std::vector<double> particle::get_ltm_avg_bay()
{
    std::vector<double> ltm_avg_bay(nA * nS);
    int size_ltm = (int)ltmTS.size();
    std::vector<double> sum_avg_bay(nS * size_ltm);


    for (int act_idx = 0; act_idx < nA; ++act_idx)
    {
        for (int stim_idx = 0; stim_idx < nS ; ++stim_idx)
        {
            for (int idx = 0; idx < size_ltm; ++idx)
            {
                int idx_ts = ltmTS[idx];
                sum_avg_bay[idx * nS + stim_idx] += bay_p[idx_ts * nA * nS + stim_idx * nA + act_idx] * size_ltm;
            }
        }
    }
    for (int act_idx = 0; act_idx < nA; ++act_idx)
    {
        for (int stim_idx = 0; stim_idx < nS ; ++stim_idx)
        {
            for (int idx = 0; idx < size_ltm; ++idx)
            {
                int idx_ts = ltmTS[idx];
                ltm_avg_bay[stim_idx * nA + act_idx] += bay_p[idx_ts * nA * nS + stim_idx * nA + act_idx] / sum_avg_bay[idx * nS + stim_idx];
            }
        }
    }
    return ltm_avg_bay;
}

std::vector<double> particle::get_bay_p_ts(int ts)
{
    std::vector<double> output(nA * nS);
    copy(bay_p.begin() + nA * nS * ts, bay_p.begin() + nA * nS * (ts + 1), output.begin());
    return output;
}

void particle::push_back_bay_p(std::vector<double> avg_bay, std::vector<double> bay_p_prior, double bay_prior, double bias_ini, int ts)
{
    if (bay_p.size()/12. != ts) {
        throw std::invalid_argument( "bay_p does not have correct size" );
    }
    
    for (int stim_idx = 0; stim_idx < nS; ++stim_idx) {
        for (int act_idx = 0; act_idx < nA; ++act_idx) {
            bay_p.push_back(bias_ini * bay_prior * avg_bay[stim_idx * nA + act_idx] + (1. - bias_ini) * bay_p_prior[stim_idx * nA + act_idx]);
        }
    }
}

void particle::push_back_rl_p(std::vector<double> avg_rl, std::vector<double> rl_p_prior, double bias_ini, int ts)
{
    if (rl_p.size()/12. != ts) {
        throw std::invalid_argument( "rl_p does not have correct size" );
    }
    
    for (int stim_idx = 0; stim_idx < nS; ++stim_idx) {
        for (int act_idx = 0; act_idx < nA; ++act_idx) {
            rl_p.push_back(bias_ini * avg_rl[stim_idx * nA + act_idx] + (1. - bias_ini) * rl_p_prior[stim_idx * nA + act_idx]);
        }
    }
}

int particle::get_monTS_size()
{
    return (int)get_monTS().size();
}

void particle::set_nTS(int value)
{
    nTS = value;
}

void particle::set_state(int value)
{
    state = value;
}

void particle::set_actorTS(int value)
{
    actorTS = value;
}

void particle::set_rl_p(int idx, double value)
{
    rl_p[idx] = value;
}

void particle::set_bay_p(int idx, double value)
{
    bay_p[idx] = value;
}

void particle::set_monTS(int idx, int value)
{
    monTS[idx] = value;
}

void particle::set_ltmTS(int idx, int value)
{
    ltmTS[idx] = value;
}

void particle::set_l(int idx, double value)
{
    l[idx] = value;
}

void particle::set_mu(int idx, double value)
{
    mu[idx] = value;
}

void particle::set_rl_p(std::vector<double> value)
{
    rl_p.resize(value.size());
    rl_p = value;
}

void particle::set_bay_p(std::vector<double> value)
{
    bay_p.resize(value.size());
    bay_p = value;
}

void particle::set_monTS(std::vector<int> value)
{
    monTS.resize(value.size());
    monTS = value;
}

void particle::set_ltmTS(std::vector<int> value)
{
    ltmTS.resize(value.size());
    ltmTS = value;
}

void particle::set_l(std::vector<double> value)
{
    l.resize(value.size());
    l = value;
}

void particle::set_mu(std::vector<double> value)
{
    mu.resize(value.size());
    mu = value;
}

int particle::get_nTS()
{
    return nTS;
}

int particle::get_state()
{
    return state;
}

int particle::get_actorTS()
{
    return actorTS;
}

double particle::get_rl_p(int idx)
{
    return rl_p[idx];
}

double particle::get_bay_p(int idx)
{
    return bay_p[idx];
}

int particle::get_monTS(int idx)
{
    return monTS[idx];
}

int particle::get_ltmTS(int idx)
{
    return ltmTS[idx];
}

double particle::get_l(int idx)
{
    return l[idx];
}

double particle::get_mu(int idx)
{
    return mu[idx];
}

std::vector<double> particle::get_rl_p()
{
    return  rl_p;
}

std::vector<double> particle::get_bay_p()
{
    return bay_p;
}

std::vector<int> particle::get_monTS()
{
    return monTS;
}

std::vector<int> particle::get_ltmTS()
{
    return ltmTS;
}

std::vector<double> particle::get_l()
{
    return l;
}

std::vector<double> particle::get_mu()
{
    return  mu;
}
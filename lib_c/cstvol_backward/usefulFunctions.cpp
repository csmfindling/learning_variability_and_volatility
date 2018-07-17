//
//  usefulFunctions.cpp
//  CreateTask
//
//  Created by Charles Findling on 22/09/2015.
//  Copyright © 2015 Charles Findling. All rights reserved.
//

#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <numeric>
#include <cmath>
#include "usefulFunctions.hpp"


using namespace std;

double Sample_Uniform_Distribution(boost::mt19937 &generator)
{
    // Define a uniform real number distribution of values between 0 and 1 and sample
    boost::uniform_real<> distribution(0,1);
    return distribution(generator);
}

vector<double> Sample_Uniform_Distribution(boost::mt19937 generator, int numberOfSamples)
{
    vector<double> res(numberOfSamples);
    vector<double>::iterator it;
    
    // Define a uniform real number distribution of values between 0 and 1.
    
    typedef boost::uniform_real<> distribution_type;
    typedef boost::variate_generator<boost::mt19937&, distribution_type> gen_type;
    gen_type gen(generator, distribution_type(0, 1));
    
    // If you want to use an STL iterator interface, use iterator_adaptors.hpp.
    boost::generator_iterator<gen_type> sample(&gen);
    for(it = res.begin() ; it != res.end(); ++it)
    {
        *it = *sample++;
    }
    return res;
}

vector<int> Sample_Discrete_Distribution(boost::mt19937 &generator, int min, int max, int numberOfSamples)
{
    vector<int> res(numberOfSamples);
    vector<int>::iterator it;
    
    // Define a uniform distribution
    
    typedef boost::random::uniform_int_distribution<>  distribution_type;
    typedef boost::variate_generator<boost::mt19937&, distribution_type> gen_type;
    gen_type gen(generator, distribution_type(min, max));

    boost::generator_iterator<gen_type> sample(&gen);
    for(it = res.begin() ; it != res.end(); ++it)
    {
        *it = *sample++;
    }
    
    return res;
}

int Sample_Discrete_Distribution(boost::mt19937 &generator, const std::vector<double>& probabilities)
{
    // Define discrete distribution
    boost::random::discrete_distribution<>  distribution(probabilities);
    return distribution(generator);
}


vector<int> Sample_Discrete_Distribution(boost::mt19937 &generator, const vector<double> &probabilities, int numberOfSamples)
{
    vector<int> res(numberOfSamples);
    vector<int>::iterator it;
    
    // Define discrete distribution
    typedef boost::random::discrete_distribution<>  distribution_type;
    typedef boost::variate_generator<boost::mt19937&, distribution_type> gen_type;
    gen_type gen(generator, distribution_type(probabilities));
    boost::generator_iterator<gen_type> sample(&gen);
    
    for(it = res.begin() ; it != res.end(); ++it)
    {
        *it = *sample++;
    }
    return res;
}

vector<double> Sample_Beta_Distribution(boost::mt19937 &generator, double a, double b, int numberOfSamples)
{
    vector<double> res(numberOfSamples);
    vector<double>::iterator it;
    
    // Define a beta distribution
    
    typedef boost::random::beta_distribution<> distribution_type;
    typedef boost::variate_generator<boost::mt19937&, distribution_type> gen_type;
    gen_type gen(generator, distribution_type(a, b));
    
    // If you want to use an STL iterator interface, use iterator_adaptors.hpp.
    boost::generator_iterator<gen_type> sample(&gen);
    for(it = res.begin() ; it != res.end(); ++it)
    {
        *it = *sample++;
    }
    return res;
}

vector<double> Sample_Normal_Distribution(boost::mt19937 &generator, double mu, double std, int numberOfSamples)
{
    vector<double> res(numberOfSamples);
    vector<double>::iterator it;
    
    // Define normal distribution
    typedef boost::normal_distribution<> distribution_type;
    typedef boost::variate_generator<boost::mt19937&, distribution_type> gen_type;
    gen_type gen(generator, distribution_type(mu, std));
    
    // Sample
    boost::generator_iterator<gen_type> sample(&gen);
    for (it = res.begin(); it != res.end(); ++it) {
        *it = *sample++;
    }
    return res;
}

vector<double> Sample_Gamma_Distribution(boost::mt19937 &generator, double k, double theta, int numberOfSamples)
{
    vector<double> res(numberOfSamples);
    vector<double>::iterator it;
    
    // Define gamma distribution
    typedef boost::gamma_distribution<> distribution_type;
    typedef boost::variate_generator<boost::mt19937&, distribution_type> gen_type;
    gen_type gen(generator, distribution_type(k, theta));
    
    // Sample
    boost::generator_iterator<gen_type> sample(&gen);
    for (it = res.begin(); it != res.end(); ++it) {
        *it = *sample++;
    }
    return res;
}

std::vector<double> Sample_Dirichlet_Distribution(boost::mt19937 &generator, double* dirichletParam, int dim)
{
    vector<double> res(dim);
    
    for (int i = 0; i < dim; ++i) {
        res[i] = Sample_Gamma_Distribution(generator, dirichletParam[i], 1, 1)[0];
    }
    res = res/sum(res);
    return res;
}

double log_beta_pdf(double x, double a, double b)
{
    double res = lgamma(a + b) - lgamma(a) - lgamma(b) + (a - 1)*log(x) + (b - 1)*log(1 - x);
    return res;
}

double log_dirichlet_pdf(double* sample, double* dirichletParam, int K)
{
    double res = lgamma(sum(dirichletParam, K));
    for (int i = 0; i < K; ++i) {
        res += -lgamma(dirichletParam[i]) + (dirichletParam[i] - 1) * log(sample[i]);
    }
    return res;
}

vector<bool> adapted_logical_xor(vector<bool> const& states, bool const& rew)
{
    unsigned long dim = states.size();
    vector<bool> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = (!states[i] != rew);
    }
    return res;
}

vector<int> stratified_resampling(boost::mt19937 &generator, vector<double> &weights)
{
    unsigned long dim = weights.size();
    vector<int> res(dim);
    vector<double> cumSum(dim);
    partial_sum(weights.begin(), weights.end(), cumSum.begin());
    cumSum               = cumSum * (double)dim;
    double uniformSample = Sample_Uniform_Distribution(generator);
    int index            = 0;
    for (int i = 0; i < dim; ++i) {
        while (cumSum[index] < uniformSample) { ++index;}
        res[i] = index; ++uniformSample;
    }
    return res;
}

vector<int> stratified_resampling(boost::mt19937 &generator, double weights[], int dim)
{
    vector<int> res(dim);
    vector<double> cumSum(dim);
    partial_sum(&weights[0], &weights[0] + dim, cumSum.begin());
    cumSum               = cumSum * (double)dim;
    double uniformSample = Sample_Uniform_Distribution(generator);
    int index            = 0;
    for (int i = 0; i < dim; ++i) {
        while (cumSum[index] < uniformSample) { ++index;}
        res[i] = index; ++uniformSample;
    }
    return res;
}

 
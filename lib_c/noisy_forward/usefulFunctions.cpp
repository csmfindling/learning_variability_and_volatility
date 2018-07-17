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
#include <boost/random/linear_congruential.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <numeric>
#include <cmath>
#include "usefulFunctions.hpp"

using namespace std;


void stratified_resampling(double uniformSample, double weights[], int dim, int* ancestorsIndexes, double sum_weights)
{
    double cumSum = weights[0] * (double)dim / sum_weights;
    int index            = 0;
    for (int i = 0; i < dim; ++i) {
        while (cumSum < uniformSample) { ++index; cumSum += (weights[index] * (double)dim / sum_weights);}
        *(ancestorsIndexes + i) = index; ++uniformSample;
    }
    return ;
}

void stratified_resampling(boost::mt19937 &generator, double uniformSample, double weights[], int dim, int* ancestorsIndexes, double sum_weights, double temperature)
{
    boost::normal_distribution<> distribution(0., 1.);
    double cumSum = (weights[0] + distribution(generator) * temperature) * (double)dim / sum_weights;
    int index            = 0;
    for (int i = 0; i < dim; ++i) {
        while (cumSum < uniformSample) { ++index; cumSum += ((weights[index] + distribution(generator) * temperature) * (double)dim / sum_weights);}
        *(ancestorsIndexes + i) = index; ++uniformSample;
    }
    return ;
}


double exp_log(double sum_so_far, double x)
{
    return sum_so_far + exp(x);
}

double log_sum(vector<double> logvector){
    double b          = *max_element(logvector.begin(), logvector.end());
    double res        = 0;
    unsigned long numberOfElements = logvector.size();
    for (int i = 0; i != numberOfElements; ++i) {
        res = res + exp(logvector[i] - b);
    }
    return b + log(res);
}

double log_sum(double* logvector, int numberOfElements){
    double b          = *max_element(logvector, logvector + numberOfElements);
    double res        = 0;
    for (int i = 0; i!=numberOfElements; ++i) {
        res = res + exp(logvector[i] - b);
    }
    return b + log(res);
}

vector<double> to_normalized_weights(vector<double> logvector){
    vector<double> res(logvector.size());
    double b          = *max_element(logvector.begin(), logvector.end());
    for (int i = 0; i != logvector.size(); ++i) {
        res[i] = exp(logvector[i] - b);
    }
    res = res/sum(res);
    return res;
}

template<typename T>
std::vector<T> maximum(std::vector<T> &vect, T element)
{
    unsigned long dim = vect.size();
    std::vector<T> res(dim);
    for (int i = 0; i != dim; ++i){
        res[i] = std::max(vect[i], element);
    }
    return res;
}

double beta_function(std::vector<double> alpha)
{
    return exp(log_beta_function(alpha));
}

double beta_function(double a, double b){
    return exp(lgamma(a) + lgamma(b) - lgamma(a + b));
}

double log_beta_function(double a, double b)
{
    return lgamma(a) + lgamma(b) - lgamma(a + b);
}

double log_beta_function(std::vector<double> alpha)
{
    double result = 0;
    unsigned long dim = alpha.size();
    for (int i = 0 ; i != dim ; ++i) {
        result += lgamma(alpha[i]);
    }
    result -= lgamma(sum(alpha));
    return result;
}

vector<double> Sample_Uniform_Distribution(boost::mt19937 &generator, int numberOfSamples)
{
    vector<double> res(numberOfSamples);
    vector<double>::iterator it;
    
    // Define a uniform real number distribution of integer values between 0 and 1.
    
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

double Sample_Uniform_Distribution(boost::mt19937 &generator)
{
    // Define a uniform real number distribution of values between 0 and 1 and sample
    boost::uniform_real<> distribution(0,1);
    return distribution(generator);
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

vector<int> Sample_Discrete_Distribution(boost::mt19937 &generator, double* probabilities, int dimension, int numberOfSamples)
{
    vector<int> res(numberOfSamples);
    vector<int>::iterator it;
    
    // Define discrete distribution
    typedef boost::random::discrete_distribution<>  distribution_type;
    typedef boost::variate_generator<boost::mt19937&, distribution_type> gen_type;
    gen_type gen(generator, distribution_type(probabilities, probabilities + dimension));
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

int Sample_Discrete_Distribution(boost::mt19937 &generator, double* proba_pointer, int dimension)
{
    // Define discrete distribution
    boost::random::discrete_distribution<>  distribution(proba_pointer, proba_pointer + dimension);
    return distribution(generator);
}

vector<double> Sample_Beta_Distribution(boost::mt19937 &generator, double a, double b, int numberOfSamples)
{
    vector<double> res(numberOfSamples);
    vector<double>::iterator it;
    double sample_0;
    // Define a beta distribution
    
    typedef boost::random::beta_distribution<> distribution_type;
    typedef boost::variate_generator<boost::mt19937&, distribution_type> gen_type;
    gen_type gen(generator, distribution_type(a, b));
    
    // If you want to use an STL iterator interface, use iterator_adaptors.hpp.
    boost::generator_iterator<gen_type> sample(&gen);
    for(it = res.begin() ; it != res.end(); ++it)
    { 
        do
        {
            sample_0 = *sample++;
        } while ((sample_0<=0)||(sample_0>=1));

        *it = sample_0;
    }
    return res;
}

double Sample_Beta_Distribution(boost::mt19937 &generator, double a, double b)
{
    // Define beta distribution
    boost::random::beta_distribution<> distribution(a,b);
    return distribution(generator);
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

double Sample_Normal_Distribution(boost::mt19937 &generator, double mu, double std)
{
    // Define normal distribution
    boost::normal_distribution<> distribution(mu, std);
    return distribution(generator);
}

double Sample_Truncated_Normal_Distribution(boost::mt19937 &generator, double mu, double std, double min)
{
    // Define normal distribution
    boost::normal_distribution<> distribution(mu, std);
    double answer;
    do
    {
        answer = distribution(generator);
    }
    while(answer < min);
    return answer;
}

double Sample_Gamma_Distribution(boost::mt19937 &generator, double k, double theta)
{
    // Define gamma distribution
    boost::gamma_distribution<> distribution(k, theta);
    return distribution(generator);
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

void Sample_Dirichlet_Distribution(boost::mt19937 &generator, double* dirichletParam, double sum_param, int dim, double inv_noise, double* result)
{
    vector<double> res(dim);
    double sum_ = 0.;

    for (int i = 0; i < dim; ++i) 
    {
        res[i] = Sample_Gamma_Distribution(generator, std::max(dirichletParam[i] * inv_noise / sum_param, 1.), 1, 1)[0];
        sum_  += res[i];
    }

    for (int i = 0; i < dim; ++i)
    {
        *(result + i) = res[i]/sum_;
    }

    return;
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

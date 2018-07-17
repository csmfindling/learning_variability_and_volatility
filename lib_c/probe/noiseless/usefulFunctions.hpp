//
//  usefulFunctions.hpp
//  CreateTask
//
//  Created by Charles Findling on 22/09/2015.
//  Copyright © 2015 Charles Findling. All rights reserved.
//

#ifndef usefulFunctions_hpp
#define usefulFunctions_hpp

#include <stdio.h>
#include <vector>
#include <boost/random/mersenne_twister.hpp>
#include <iostream>

std::vector<int> Sample_Discrete_Distribution(boost::mt19937 &generator, int min, int max, int numberOfSamples = 1);

std::vector<int> Sample_Discrete_Distribution(boost::mt19937 &generator, const std::vector<double>& probabilities, int numberOfSamples);

int Sample_Discrete_Distribution(boost::mt19937 &generator, const std::vector<double>& probabilities);

int Sample_Discrete_Distribution(boost::mt19937 &generator, double* proba_pointer, int dimension);

std::vector<double> Sample_Uniform_Distribution(boost::mt19937 &generator, int numberOfSamples);

double Sample_Uniform_Distribution(boost::mt19937 &generator);

double Sample_Truncated_Normal_Distribution(boost::mt19937 &generator, double mu, double std, double min);

std::vector<double> Sample_Beta_Distribution(boost::mt19937 &generator, double a, double b, int numberOfSamples);

double Sample_Beta_Distribution(boost::mt19937 &generator, double a, double b);

std::vector<double> Sample_Normal_Distribution(boost::mt19937 &generator, double mu, double std, int numberOfSamples);

double Sample_Normal_Distribution(boost::mt19937 &generator, double mu = 0., double std = 1.);

std::vector<double> Sample_Gamma_Distribution(boost::mt19937 &generator, double k, double theta, int numberOfSamples);

double Sample_Gamma_Distribution(boost::mt19937 &generator, double k, double theta);

std::vector<double> Sample_Dirichlet_Distribution(boost::mt19937 &generator, double* dirichletParam, int dim);

double log_beta_pdf(double x, double a, double b);

double log_dirichlet_pdf(double* sample, double* dirichletParam, int K);

std::vector<bool> adapted_logical_xor(std::vector<bool> const& states, bool const& rew);

std::vector<int> stratified_resampling(boost::mt19937 &generator, std::vector<double> &weights);

std::vector<int> stratified_resampling(boost::mt19937 &generator, double weights[], int dim);

double beta_function(std::vector<double> alpha);

double beta_function(double a, double b);

double log_beta_function(std::vector<double> alpha);

double log_beta_function(double a, double b);

double exp_log(double sum_so_far, double x);

double log_sum(std::vector<double> logvector);

double log_sum(double* logvector, int numberOfElements);

std::vector<double> to_normalized_weights(std::vector<double> logvector);

template<typename T>
T sum(std::vector<T> &vect)
{
    T res = 0;
    typename std::vector<T>::iterator it;
    for (it = vect.begin(); it != vect.end(); ++it){
        res = res + (*it);
    }
    return res;
}

template<typename T>
T sum_multiply(std::vector<T> &vect, std::vector<T> &likelihoods)
{
    T res = 0;
    for (int i = 0; i != vect.size(); ++i){
        res = res + vect[i] * likelihoods[i];
    }
    return res;
}

template<typename T>
T sum(T* array, int dim)
{
    T res = 0;
    for (int i = 0; i < dim; ++i){
        res += array[i];
    }
    return res;
}


template<typename T>
T prod(std::vector<T> &vect)
{
    T res = 1;
    typename std::vector<T>::iterator it;
    for (it = vect.begin(); it != vect.end(); ++it) {
        res = res * (*it);
    }
    return res;
}

template<typename T>
std::vector<bool> operator<=(std::vector<T> const& a, std::vector<T> const& b)
{
    assert(a.size() == b.size());
    unsigned long dim = a.size();
    std::vector<bool> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i] <= b[i];
    }
    return res;
}

template<typename T>
std::vector<bool> operator==(std::vector<T> const& a, std::vector<T> const& b)
{
    assert(a.size() == b.size());
    unsigned long dim = a.size();
    std::vector<bool> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i] == b[i];
    }
    return res;
}

template<typename T>
std::vector<bool> operator<=(std::vector<T> const &a, T const& b)
{
    unsigned long dim = a.size();
    std::vector<bool> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i] <= b;
    }
    return res;
}

template<typename T>
void divide(double* array, T divider, int dim)
{
    for (int i = 0; i < dim; ++i) {
        array[i] = array[i]*1./divider;
    }
    return;
}

template<typename T>
std::vector<bool> operator==(std::vector<T> const &a, T const& b)
{
    unsigned long dim = a.size();
    std::vector<bool> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i] == b;
    }
    return res;
}

template<typename T>
std::vector<double> operator+(std::vector<T> const a, int b)
{
    unsigned long dim = a.size();
    std::vector<double> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i] + b;
    }
    return res;
}

template<typename T>
std::vector<bool> isEqual(T* const &a, int const& x, int const& xdim, int const& ydim, T const& b)
{
    assert(x < xdim);
    std::vector<bool> res(ydim);
    for (int i = 0; i < ydim; ++i) {
        res[i] = a[ydim * x + i] == b;
    }
    return res;
}

template<typename T>
std::vector<bool> isEqual_and_adapted_logical_xor(T* const &a, int const& x, int const& xdim, int const& ydim, T const& b, bool const& rew)
{
    assert(x < xdim);
    std::vector<bool> res(ydim);
    for (int i = 0; i < ydim; ++i) {
        res[i] = (!(a[ydim * x + i] == b) != rew);
    }
    return res;
}

template<typename T>
std::vector<bool> operator!=(std::vector<T> const &a, T const& b)
{
    unsigned long dim = a.size();
    std::vector<bool> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = (a[i] != b);
    }
    return res;
}

template<typename T>
std::vector<bool> isNotEqual(T* const &a, int const& x, int const& xdim, int const& ydim, T const& b)
{
    assert(x < xdim);
    std::vector<bool> res(ydim);
    for (int i = 0; i < ydim; ++i) {
        res[i] = a[ydim * x + i] != b;
    }
    return res;
}

template<typename T>
std::vector<double> isNotEqual_timesVector_normalise(T* const &a, int const& x, int const& xdim, int const& ydim, T const& b, double* gamma_pointer)
{
    assert(x < xdim);
    std::vector<double> res(ydim);
    double sum_vector = 0;
    for (int i = 0; i < ydim; ++i) {
        res[i] = (a[ydim * x + i] != b) * (*(gamma_pointer + i));
        sum_vector += res[i];
    }
    for (int i = 0; i < ydim; ++i) {
        res[i] = res[i] / sum_vector;
    }
    return res;
}

template<typename T>
void isNotEqual_timesVector_normalise(double* transProb, T* const &a, int const& x, int const& xdim, int const& ydim, T const& b, double* gamma_pointer)
{
    assert(x < xdim);
    double sum_vector = 0;
    for (int i = 0; i < ydim; ++i) {
        *(transProb + i) = (a[ydim * x + i] != b) * (*(gamma_pointer + i));
        sum_vector += *(transProb + i);
    }
    divide(transProb, sum_vector, ydim);
}

template<typename T>
std::vector<double> isNotEqual_timesVector_normalise_timesLikelihood(T* const &a, int const& x, int const& xdim, int const& ydim, T const& b, double* gamma_pointer, std::vector<double> const& likelihoods)
{
    assert(x < xdim);
    std::vector<double> res(ydim);
    double sum_vector = 0;
    for (int i = 0; i < ydim; ++i) {
        res[i] = (a[ydim * x + i] != b) * (*(gamma_pointer + i));
        sum_vector += res[i];
    }
    for (int i = 0; i < ydim; ++i) {
        res[i] = res[i] * likelihoods[i] / sum_vector;
    }
    return res;
}

template<typename T>
std::vector<T> operator*(std::vector<bool> const &a, std::vector<T> const& b)
{
    assert(a.size() == b.size());
    unsigned long dim = a.size();
    std::vector<T> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i]*b[i];
    }
    return res;
}

template<typename T>
std::vector<T> abs(std::vector<T> &vect)
{
    std::vector<T> res(vect.size());
    for (int i = 0; i != vect.size(); ++i){
        *(&res[0] + i)  = std::abs(*(&vect[0] + i));
    }
    return res;
}

template<typename T>
std::vector<T> operator-(std::vector<T> const &a, std::vector<T> const& b)
{
    assert(a.size() == b.size());
    unsigned long dim = a.size();
    std::vector<T> res(dim);
    for (int i = 0; i != dim; ++i) {
        res[i] = a[i]-b[i];
    }
    return res;
}

template<typename T>
std::vector<T> operator-(std::vector<T> const &a, T b)
{
    unsigned long dim = a.size();
    std::vector<T> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i]-b;
    }
    return res;
}

template<typename T>
std::vector<T> operator*(std::vector<bool> const &a, T* const& b)
{
    unsigned long dim = a.size();
    std::vector<T> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i]*b[i];
    }
    return res;
}

template<typename T>
std::vector<T> operator*(std::vector<T> const &a, std::vector<T> const& b)
{
    assert(a.size() == b.size());
    unsigned long dim = a.size();
    std::vector<T> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i]*b[i];
    }
    return res;
}

template<typename T>
std::vector<T> operator*(T* const &a, std::vector<T> const& b)
{
    unsigned long dim = b.size();
    std::vector<T> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i]*b[i];
    }
    return res;
}

template<typename T>
std::vector<T> operator*(std::vector<T> const &a, T const& b)
{
    unsigned long dim = a.size();
    std::vector<T> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i]*b;
    }
    return res;
}

template<typename T>
std::vector<T> operator/(std::vector<T> const &a, T const& b)
{
    unsigned long dim = a.size();
    std::vector<T> res(dim);
    for (int i = 0; i < dim; ++i) {
        res[i] = a[i]/b;
    }
    return res;
}

template<typename T>
std::vector<T> createRange(T const first, T const last, int const increment)
{
    int dim = (int)abs((last - first)/increment);
    std::vector<T> range(dim);
    for (int i = 0; i < dim; i++) {
        range[i] = first + i * increment;
    }
    return range;
}

template<typename T>
void print(std::vector<T> vect){
    typename std::vector<T>::iterator it;
    for (it = vect.begin(); it != vect.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << " " << std::endl;
}


#endif /* usefulFunctions_hpp */

/* Implementation math operations on vectors and matrices */

#ifndef MATH_HPP
#define MATH_HPP

#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>

using std::vector;

typedef vector<double> dim1;
typedef vector<dim1>   dim2;
typedef vector<dim2>   dim3;

dim1 cumsum(dim1 &a)
{
    /*
    cumulative sum of a 1D vector
    */
    dim1 result;
    result.resize(a.size());
    result[0] = a[0];
    for (size_t i = 1; i < a.size(); ++i)
    {
        result[i] = result[i - 1] + a[i];
    }

    return result;
}

dim1 flipud(dim1 &a)
{
    /*
    flip a 1D vector upside down
    */
    dim1 result;
    result.resize(a.size());
    for (size_t i = 0; i < a.size(); ++i)
    {
        result[i] = a[a.size() - i - 1];
    }

    return result;
}

dim1 zeros(size_t n)
{
    /*
    create a 1D vector of zeros
    */
    dim1 result;
    result.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
        result[i] = 0;
    }

    return result;
}

dim2 zeros(size_t m, size_t n)
{
    /*
    create a 2D vector of zeros
    */
    dim2 result;
    result.resize(m);
    for (size_t i = 0; i < m; ++i)
    {
        result[i].resize(n);
        for (size_t j = 0; j < n; ++j)
        {
            result[i][j] = 0;
        }
    }

    return result;
}


dim1 ones(size_t n)
{
    /*
    create a 1D vector of ones
    */
    dim1 result;
    result.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
        result[i] = 1;
    }

    return result;
}

dim2 ones(size_t m, size_t n)
{
    /*
    create a 2D vector of ones
    */
    dim2 result;
    result.resize(m);
    for (size_t i = 0; i < m; ++i)
    {
        result[i].resize(n);
        for (size_t j = 0; j < n; ++j)
        {
            result[i][j] = 1;
        }
    }

    return result;
}

double mean(dim1 &a)
{
    /*
    calculate the mean of a 1D vector
    */
    double result = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        result += a[i];
    }
    result /= a.size();

    return result;
}

double sum(dim1 &a)
{
    /*
    calculate the sum of a 1D vector
    */
    double result = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        result += a[i];
    }

    return result;
}

dim1 dot(dim2 &a, dim1 &b)
{
    /*
    dot product of a 2D vector and a 1D vector
    */
    assert(a[0].size() == b.size());
    dim1 result;
    result.resize(a.size());
    for (size_t i = 0; i < a.size(); ++i)
    {
        result[i] = 0;
        for (size_t j = 0; j < a[i].size(); ++j)
        {
            result[i] += a[i][j] * b[j];
        }
    }

    return result;
}

dim1 exp(dim1 &a)
{
    /*
    element-wise exponential of a 1D vector
    */
    dim1 result;
    result.resize(a.size());
    for (size_t i = 0; i < a.size(); ++i)
    {
        result[i] = std::exp(a[i]);
    }

    return result;
}



#endif
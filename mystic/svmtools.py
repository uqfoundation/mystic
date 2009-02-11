#!/usr/bin/env python
# 
# Patrick Hung.

"""
Simple utility functions for SV-Regressions
"""

from numpy import zeros, multiply, ndarray, vectorize, array

def KernelMatrix(X, k):
    n = X.size
    Q = zeros((n,n))
    for i in range(n):
       for j in range(n):
           # dumb, but how else to do outer products of arbitrary functions
           # without going through ufunc C-api ?
           Q[i,j] = k(X[i],X[j])
    return Q

def SupportVectors(alpha, eps=0):
    # return index of nonzero alphas (at a tolerance of epsilon)
    return (abs(alpha)>eps).nonzero()[0]

def StandardInnerProduct(i1, i2):
    return i1 * i2

def LinearKernel(i1, i2):
    return 1. + i1 * i2

def Bias(x, y, alpha, eps, kernel=StandardInnerProduct):
    """ Compute regression bias for epsilon insensitive loss regression """
    N = len(alpha)/2
    ap, am = alpha[:N],  alpha[N:]
    sv = SupportVectors(alpha)[0]
    # functionally: b = eps + y[sv] + sum( (ap-am) * map(lambda xx: kernel(xx, x[sv]), x) )
    b = eps + y[sv] + sum( (ap-am) * multiply.outer(x, x[sv]) )
    return b

def RegressionFunction(x, y, alpha, eps, kernel=StandardInnerProduct):
    """ The Support Vector expansion. f(x) = Sum (ap - am) K(xi, x) + b """
    bias = Bias(x, y, alpha, eps, kernel)
    N = len(alpha)/2
    ap, am = alpha[:N],  alpha[N:]
    ad = ap-am
    def _(x_in):
        a = array([kernel(xx, x_in) for xx in x])
        return bias - sum(ad *a)
    def f(x_in):
        if type(x_in) == ndarray:
            return vectorize(_)(x_in)
        return _(x_in)
    return f

# end of file

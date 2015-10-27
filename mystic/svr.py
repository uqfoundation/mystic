#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
Simple utility functions for SV-Regressions
"""

from numpy import multiply, ndarray, vectorize, array, asarray

def InnerProduct(i1, i2): # numpy.multiply(x,x)
    return i1 * i2

def LinearKernel(i1, i2):
    return 1. + i1 * i2

def KernelMatrix(X, k=multiply):
    "outer product of X with self, using k as elementwise product function"
    X = asarray(X).ravel()
    return k(X[:,None], X[None,:])


def SupportVectors(alpha, eps=0):
    """indicies of nonzero alphas (at tolerance eps)"""
    return (abs(alpha)>eps).nonzero()[0]

def Bias(x, y, alpha, eps, kernel=InnerProduct):
    """ Compute regression bias for epsilon insensitive loss regression """
    N = len(alpha)/2
    ap, am = alpha[:N],  alpha[N:]
    sv = SupportVectors(alpha)[0]
    # functionally: b = eps + y[sv] + sum( (ap-am) * map(lambda xx: kernel(xx, x[sv]), x) )
    b = eps + y[sv] + sum( (ap-am) * multiply.outer(x, x[sv]) )
    return b

def RegressionFunction(x, y, alpha, eps, kernel=InnerProduct):
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

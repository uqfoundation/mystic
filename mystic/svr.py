#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2017 The Uncertainty Quantification Foundation.
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


def SupportVectors(alpha, epsilon=0):
    """indices of nonzero alphas (at tolerance epsilon)"""
    return (abs(alpha)>epsilon).nonzero()[0]

def Bias(x, y, alpha, epsilon, kernel=InnerProduct):
    """ Compute regression bias for epsilon insensitive loss regression """
    N = len(alpha)//2
    ap, am = alpha[:N],  alpha[N:]
    sv = SupportVectors(alpha)[0]
    # functionally: b = epsilon + y[sv] + sum( (ap-am) * map(lambda xx: kernel(xx, x[sv]), x) )
    b = epsilon + y[sv] + sum( (ap-am) * multiply.outer(x, x[sv]) )
    return b

def RegressionFunction(x, y, alpha, epsilon, kernel=InnerProduct):
    """ The Support Vector expansion. f(x) = Sum (ap - am) K(xi, x) + b """
    bias = Bias(x, y, alpha, epsilon, kernel)
    N = len(alpha)//2
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

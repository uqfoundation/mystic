#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2018 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Simple utility functions for SV-Regressions
"""
#FIXME: fix/improve this module and mystic.math.distance
#       (1) better with in-place operations
#       (2) correct and reuse distance metrics/definitions

import numpy as np

__all__ = ['InnerProduct','LinearKernel','PolynomialKernel','GaussianKernel', \
           'KernelMatrix','SupportVectors','Bias','RegressionFunction']

def InnerProduct(i1, i2=None): # np.multiply(x,x)
    i2 = i1 if i2 is None else i2
    return i1 * i2 #XXX: assumes misaligned i2?

def LinearKernel(i1, i2=None, coeff=1.):
    return coeff + InnerProduct(i1,i2)

def PolynomialKernel(i1, i2=None, degree=3, coeff=1.): #XXX: arg order?
    if type(degree) is int and degree > 0:
        return LinearKernel(i1,i2,coeff)**degree
    raise ValueError('degree = %s is not an int > 0' % degree)

def GaussianKernel(i1, i2=None, gamma=None, const=1.): #XXX: arg names?
    if gamma is None:
        gamma = np.asarray(i1).shape
        gamma = 1./gamma[-1] if gamma else 1.
    elif gamma <= 0:
        raise ValueError('gamma = %s is not > 0' % gamma)
    i2 = i1 if i2 is None else np.transpose(i2) #FIXME: assumes misaligned i2
    # pairwise (or euclidean distance)
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*np.dot(x,y.T)
    i1i2 = np.dot(i1,np.transpose(i2))
    i1i2 *= -2
    i1i1 = np.sum(i1**2, axis=-1) #NOTE: np.einsum('ij,ij->i',i1,i1)
    i2i2 = i1i1 if i1 is i2 else np.sum(i2**2, axis=-1)
    i1i2 += i1i1
    i1i2 += i2i2
    return const * np.exp(-gamma * i1i2) #XXX: tolist?


def KernelMatrix(X, k=np.multiply): #XXX: flatten? svc.KernelMatrix?
    "outer product of X with self, using k as elementwise product function"
    X = np.asarray(X).ravel()
    return k(X[:,None], X[None,:])


def SupportVectors(alpha, epsilon=0):
    """indices of nonzero alphas (at tolerance epsilon)"""
    return (np.abs(alpha)>epsilon).nonzero()[0]

def Bias(x, y, alpha, epsilon, kernel=InnerProduct): #FIXME: kernel ignored!
    """ Compute regression bias for epsilon insensitive loss regression """
    N = len(alpha)//2
    ap, am = alpha[:N], alpha[N:]
    sv = SupportVectors(alpha)[0]
    # functionally: b = epsilon + y[sv] + sum( (ap-am) * map(lambda xx: kernel(xx, x[sv]), x) )
    b = epsilon + y[sv] + sum( (ap-am) * np.multiply.outer(x, x[sv]) )
    return b

def RegressionFunction(x, y, alpha, epsilon, kernel=InnerProduct):
    """ The Support Vector expansion. f(x) = Sum (ap - am) K(xi, x) + b """
    bias = Bias(x, y, alpha, epsilon, kernel)
    N = len(alpha)//2
    ap, am = alpha[:N], alpha[N:]
    ad = ap-am
    def _(x_in):
        a = np.array([kernel(xx, x_in) for xx in x]) #XXX: inefficient
        return bias - sum(ad *a)
    def f(x_in):
        if type(x_in) == np.ndarray:
            return np.vectorize(_)(x_in)
        return _(x_in)
    return f

# end of file

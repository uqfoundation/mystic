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

__all__ = ['LinearKernel','PolynomialKernel','GaussianKernel', \
           'KernelMatrix','SupportVectors','Bias','RegressionFunction']

def _ensure_arrays(i1,i2=None):
    if i2 is None:
        i1 = np.asarray(i1, dtype=float)
        i2 = i1
    else:
        i1 = np.asarray(i1, dtype=float)
        i2 = np.asarray(i2, dtype=float)
    return i1,i2

def _ensure_gamma(i1, gamma=None):
    if gamma is None:
        gamma = i1.shape #NOTE: assumes numpy array
        gamma = 1./gamma[-1] if gamma else 1.
    elif gamma <= 0:
        raise ValueError('gamma = %s is not > 0' % gamma)
    return gamma

def _row_norm(i1, squared=False):
    i1i1 = np.sum(i1*i1, axis=-1) #NOTE: i1 = np.einsum('ij,ij->i',i1,i1)
    return i1i1 if squared else np.sqrt(i1i1)

def _pairwise_distance(i1, i2=None, squared=False):
    i1,i2 = _ensure_arrays(i1,i2)
    # pairwise (or euclidean distance)
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*np.dot(x,y.T)
    i1i2 = LinearKernel(i1,i2)
    i1i2 *= -2
    i1i1 = _row_norm(i1, squared=True)
    i2i2 = i1i1 if i1 is i2 else _row_norm(i2, squared=True)
    i1i2 += i1i1[:, None] if i1i1.shape else i1i1
    i1i2 += i2i2[None, :] if i2i2.shape else i2i2
    if i1 is i2:
        i1i2.flat[::i1i2.shape[0] + 1] = 0.0
    return i1i2 if squared else np.sqrt(i1i2)


def LinearKernel(i1, i2=None):
    '''linear kernel for i1 and i2

    dot(i1,i2.T), where i2=i1 if i2 is not provided
    '''
    i1,i2 = _ensure_arrays(i1,i2)
    return np.dot(i1,i2.T)

def PolynomialKernel(i1, i2=None, degree=3, gamma=None, coeff=1):
    '''polynomial kernel for i1 and i2

    (coeff + gamma * dot(i1,i2.T))**degree, where i2=i1 if i2 is not provided

    coeff is a float, default of 1.
    gamma is a float, default of 1./i1.shape(1)
    degree is an int, default of 3
    '''
    i1,i2 = _ensure_arrays(i1,i2)
    gamma = _ensure_gamma(i1,gamma)
    i1i2 = LinearKernel(i1,i2)
    i1i2 *= gamma
    i1i2 += coeff
    i1i2 **= degree
    return i1i2

def GaussianKernel(i1, i2=None, gamma=None): #XXX: arg names?
    '''gaussian kernel for i1 and i2

    exp(-gamma * pairwise_distance(i1,i2)), where i2=i1 if i2 is not provided

    gamma is a float, default of 1./i1.shape(1)
    '''
    i1,i2 = _ensure_arrays(i1,i2)
    gamma = _ensure_gamma(i1,gamma)
    i1i2 = _pairwise_distance(i1,i2)
    i1i2 *= -gamma
    return np.exp(i1i2)

def KernelMatrix(X, k=np.dot): #XXX: ravel? svc.KernelMatrix? np.outer?
    "product of X with self, using k as elementwise product function"
    X = _ensure_arrays(X)[0].ravel()
    return k(X[:,None], X[None,:].T)


def SupportVectors(alpha, epsilon=0):
    """indices of nonzero alphas (at tolerance epsilon)"""
    return (np.abs(alpha)>epsilon).nonzero()[0]

def Bias(x, y, alpha, epsilon, kernel=LinearKernel): #FIXME: kernel ignored!
    """ Compute regression bias for epsilon insensitive loss regression """
    N = len(alpha)//2
    ap, am = alpha[:N], alpha[N:]
    sv = SupportVectors(alpha)[0]
    # functionally: b = epsilon + y[sv] + sum( (ap-am) * map(lambda xx: kernel(xx, x[sv]), x) )
    b = epsilon + y[sv] + sum( (ap-am) * np.multiply.outer(x, x[sv]) )
    return b

def RegressionFunction(x, y, alpha, epsilon, kernel=LinearKernel):
    """ The Support Vector expansion. f(x) = Sum (ap - am) K(xi, x) + b """
    bias = Bias(x, y, alpha, epsilon, kernel)
    N = len(alpha)//2
    ap, am = alpha[:N], alpha[N:]
    ad = ap-am
    def _(x_in):
        a = np.array([kernel(xx, x_in) for xx in x]) #XXX: inefficient
        return bias - sum(ad * a)
    def f(x_in):
        if type(x_in) == np.ndarray:
            return np.vectorize(_)(x_in)
        return _(x_in)
    return f

# end of file

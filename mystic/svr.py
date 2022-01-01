#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Simple utility functions for SV-Regressions
"""
import numpy as np
import mystic.math.distance as _distance

__all__ = ['LinearKernel','PolynomialKernel','SigmoidKernel', \
           'LaplacianKernel','GaussianKernel','CosineKernel', \
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

def _ensure_scale(i1):
    if np.isscalar(i1):
        return 1. if i1 == .0 else i1
    elif isinstance(i1, np.ndarray):
        i1[i1 == 0.0] = 1.0
        return i1

def _row_norm(i1, squared=False):
    i1i1 = np.sum(i1*i1, axis=-1) #NOTE: i1 = np.einsum('ij,ij->i',i1,i1)
    return i1i1 if squared else np.sqrt(i1i1)

def _manhattan_distance(i1, i2=None, pair=False):
    i1,i2 = _ensure_arrays(i1, i2)
    s1,s2 = i1.shape,i2.shape
    axis = int(pair) if s1 or s2 else None
    return _distance.manhattan(np.atleast_1d(i1), i2, pair=pair, axis=axis)

def _euclidean_distance(i1, i2=None, pair=False, squared=False):
    i1,i2 = _ensure_arrays(i1,i2)
    s1,s2 = i1.shape,i2.shape
    axis = int(pair) if s1 or s2 else None
    i1i2 = _distance.euclidean(np.atleast_1d(i1), i2, pair=paie, axis=axis)
    return i1i2*i1i2 if squared else i1i2

# alternate: assumes 2D
def __euclidean_distance(i1, i2=None, squared=False):
    i1,i2 = _ensure_arrays(i1,i2)
    # pairwise (or euclidean distance or L2)
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

def CosineKernel(i1, i2=None):
    '''cosine kernel for i1 and i2

    dot(i1,i2.T)/(||i1||*||i2||), where i2=i1 if i2 is not provided,
    and ||i|| is defined as L2-normalized i
    '''
    i1,i2 = _ensure_arrays(i1,i2)
    i1n = i1/_ensure_scale(_distance.Lnorm(i1,2,axis=-1)) #XXX: def normalize
    i2n = i1n if i1 is i2 else i2/_ensure_scale(_distance.Lnorm(i2,2,axis=-1))
    return np.dot(i1n,i2n.T)

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

def SigmoidKernel(i1, i2=None, gamma=None, coeff=1):
    '''sigmoid kernel for i1 and i2

    tanh(coeff + gamma * dot(i1,i2.T)), where i2=i1 if i2 is not provided

    coeff is a float, default of 1.
    gamma is a float, default of 1./i1.shape(1)
    '''
    i1,i2 = _ensure_arrays(i1,i2)
    gamma = _ensure_gamma(i1,gamma)
    i1i2 = LinearKernel(i1,i2)
    i1i2 *= gamma
    i1i2 += coeff
    return np.tanh(i1i2)

def GaussianKernel(i1, i2=None, gamma=None): #XXX: arg names?
    '''gaussian kernel for i1 and i2

    exp(-gamma * euclidean_distance(i1,i2)**2), where i2=i1 if i2 is not provided

    gamma is a float, default of 1./i1.shape(1)
    '''
    i1,i2 = _ensure_arrays(i1,i2)
    gamma = _ensure_gamma(i1,gamma)
    i1i2 = _euclidean_distance(i1,i2,squared=True)
    i1i2 *= -gamma
    return np.exp(i1i2)

def LaplacianKernel(i1, i2=None, gamma=None): #XXX: arg names?
    '''laplacian kernel for i1 and i2

    exp(-gamma * manhattan_distance(i1,i2)), where i2=i1 if i2 is not provided

    gamma is a float, default of 1./i1.shape(1)
    '''
    i1,i2 = _ensure_arrays(i1,i2)
    gamma = _ensure_gamma(i1,gamma)
    i1i2 = _manhattan_distance(i1,i2,pair=False)
    i1i2 *= -gamma
    return np.exp(i1i2)

def KernelMatrix(X, Y=None, kernel=LinearKernel): #XXX: svc.KernelMatrix?
    "outer product, using kernel as elementwise product function"
    X,Y = _ensure_arrays(X,Y)
    return kernel(X.ravel()[:,None], Y.ravel()[None,:].T)
    #FIXME: if X,Y is 2D, return is correct; if 1D, then return XXX.ravel()

def SupportVectors(alpha, epsilon=0):
    """indices of nonzero alphas (at tolerance epsilon)"""
    return (np.abs(alpha)>epsilon).nonzero()[0]

def Bias(x, y, alpha, epsilon, kernel=LinearKernel):
    """ Compute regression bias for epsilon insensitive loss regression """
    N = len(alpha)//2
    ap, am = alpha[:N], alpha[N:]
    sv = SupportVectors(alpha)[0]
    b = epsilon + y[sv] + sum((ap-am) * KernelMatrix(x,x[sv],kernel).ravel())
    return b

def RegressionFunction(x, y, alpha, epsilon, kernel=LinearKernel):
    """ The Support Vector expansion. f(x) = Sum (ap - am) K(xi, x) + b """
    bias = Bias(x, y, alpha, epsilon, kernel)
    N = len(alpha)//2
    ap, am = alpha[:N], alpha[N:]
    ad = ap-am
    def _(x_in):
        return bias - sum(ad * KernelMatrix(x, x_in, kernel).ravel())
        #return bias - sum(ad * np.array([kernel(xx, x_in) for xx in x]))
    def f(x_in):
        if type(x_in) == np.ndarray:
            return np.vectorize(_)(x_in) #XXX: need vectorize w/ KernelMatrix?
        return _(x_in)
    return f

# end of file

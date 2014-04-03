#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
Simple utility functions for SV-classifications
"""

from numpy import zeros, multiply, ndarray, vectorize, array, dot, transpose, diag, sum

def KernelMatrix(X, k=dot):
    n = X.shape[0]
    Q = zeros((n,n))
    for i in range(n):
       for j in range(i, n):
           Q[i,j] = k(X[i,:],X[j,:])
    return Q + transpose(Q) - diag(Q.diagonal())

def WeightVector(alpha, X, y):
    ay = (alpha * y).flatten()
    aXy = transpose(ay * transpose(X))
    return sum(aXy, 0)


def SupportVectors(alpha, y=None, eps = 0):
    import mystic.svmtools
    sv = svmtools.SupportVectors(alpha,eps)
    if y == None:
        return sv
    else:
        class1 = set((y>0).nonzero()[1])
        class2 = set((y<0).nonzero()[1])
        sv1 = class1.intersection(sv)
        sv2 = class2.intersection(sv)
        return list(sv1), list(sv2)

def Bias(alpha, X, y, kernel=dot):
    """Compute classification bias. """
    sv1, sv2 = SupportVectors(alpha, y,eps=1e-6)
    pt1, pt2 = X[sv1[0],:], X[sv2[0],:]
    k1, k2 = kernel(X, pt1), kernel(X,pt2)
    return -0.5 * (sum(alpha*y*k1) + sum(alpha*y*k2))

# end of file

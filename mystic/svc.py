#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Simple utility functions for SV-classifications
"""

from numpy import multiply, asarray, dot, transpose, sum

def KernelMatrix(X, k=dot): #XXX: generalized dot/inner product?  K(X,Y)?
    "inner product of X with self, using k as elementwise product function"
    # the following is tensordot(X,X,axes=(-1,-1)), with dot --> k
    # 3-clause BSD (see: v1.7.2 http://docs.scipy.org/doc/numpy/license.html)
    X = asarray(X)
    Xs = X.shape
    ndX = len(X.shape)
    nX = 1
    axes = [ndX - 1]

    # Move the axes to sum over to the end of "a"
    # and to the front of "b" in inner(a,b)
    notin = [_ for _ in range(ndX) if _ not in axes]
    newaxes_a = notin + axes
    N2 = 1
    for axis in axes: N2 *= Xs[axis]
    newshape_a = (-1, N2)
    olda = [Xs[axis] for axis in notin]

    newaxes_b = axes + notin
    N2 = 1
    for axis in axes: N2 *= Xs[axis]
    newshape_b = (N2, -1)
    oldb = [Xs[axis] for axis in notin]

    at = X.transpose(newaxes_a).reshape(newshape_a)
    bt = X.transpose(newaxes_b).reshape(newshape_b)
    return k(at,bt).reshape(olda + oldb)


def WeightVector(alpha, X, y):
    ay = (alpha * y).flatten()
    aXy = transpose(ay * transpose(X))
    return sum(aXy, 0)


def SupportVectors(alpha, y=None, epsilon=0):
    """indices of nonzero alphas (at tolerance epsilon)

If labels y are provided, then group indices by label
    """
    import mystic.svr as svr
    sv = svr.SupportVectors(alpha,epsilon)
    if y is None:
        return sv
    else:
        class1 = set((y>0).nonzero()[1])
        class2 = set((y<0).nonzero()[1])
        sv1 = class1.intersection(sv)
        sv2 = class2.intersection(sv)
        return list(sv1), list(sv2)

def Bias(alpha, X, y, kernel=dot):
    """Compute classification bias. """
    sv1, sv2 = SupportVectors(alpha, y, epsilon=1e-6)
    pt1, pt2 = X[sv1[0],:], X[sv2[0],:]
    k1, k2 = kernel(X, pt1), kernel(X,pt2)
    return -0.5 * (sum(alpha*y*k1) + sum(alpha*y*k2))

# end of file

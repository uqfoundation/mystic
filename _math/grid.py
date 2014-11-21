#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
tools for generating points on a grid
"""

def gridpts(q):
    """
takes a list of lists of arbitrary length q = [[1,2],[3,4]]
and produces a list of gridpoints g = [[1,3],[1,4],[2,3],[2,4]]
    """
    w = [[] for i in range(len(q[-1]))]
    for j in range(len(q)-1,-1,-1):
      for k in range(len(q[j])):
        for l in range(k*len(w)/len(q[j]), (k+1)*len(w)/len(q[j])):
          w[l].append(q[j][k])
      if j: w += [i[:] for i in w[:]*(len(q[j-1])-1)]
    return [list(reversed(w[i])) for i in range(len(w))]


def samplepts(lb,ub,npts):
    """
takes lower and upper bounds (e.g. lb = [0,3], ub = [2,4])
produces a list of sample points s = [[1,3],[1,4],[2,3],[2,4]]

Inputs:
    lb  --  a list of the lower bounds
    ub  --  a list of the upper bounds
    npts  --  number of sample points
    """
    from mystic.math.samples import random_samples
    q = random_samples(lb,ub,npts)
    q = [list(i) for i in q]
    q = zip(*q)
    return [list(i) for i in q]


def randomly_bin(N, ndim=None, ones=True, exact=True):
    """
generate N bins randomly gridded across ndim dimensions

Inputs:
    N  --  integer number of bins, where N = prod(bins)
    ndim  --  integer length of bins, thus ndim = len(bins)
    ones  --  if False, prevent bins from containing "1s", wherever possible
    exact  --  if False, find N-1 bins for prime numbers
    """
    if ndim == 0: return []
    if N == 0: return [0] if ndim else [0]*ndim
    from itertools import chain
    from mystic.tools import random_state
    random = random_state().random
    def factors(n):
        result = list()
        for i in chain([2],xrange(3,n+1,2)):
            s = 0
            while n%i == 0:
                n /= i
                s += 1
            result.extend([i]*s)
            if n == 1:
                return result
    result = factors(N)
    dim = nfact = len(result)
    prime = nfact == 1
    if ndim: result += [1] * (ndim - (nfact / ndim));  dim = ndim
    elif ones: result += [1] # add some 'randomness' by adding a "1"
    # if ones, mix in the 1s; otherwise, only use 1s when ndim < len(result)
    if ones: result = sorted(result, key=lambda v: random())
    else: result[:nfact] = sorted(result[:nfact], key=lambda v: random())
    from numpy import product
    result = [product(result[i::dim]) for i in range(dim)]
    # if not ones, now needs a full sort to sort in the 1s
    if not ones: result = sorted(result, key=lambda v: random())
    elif not ndim and 1 in result: result.remove(1) # remove the added "1"
    # if it's a prime, then do N-1 if exact=False
    if not exact and N > 3 and prime:
        result = randomly_bin(N-1, ndim, ones)
    return result


#######################################################################
if __name__ == '__main__':

  nbins = 2
  lower = [0, 3, 6]
  upper = [2, 5, 8]

  # generate arrays of points defining a grid in parameter space
  grid_dimensions = len(lower)
  bins = []
  for i in range(grid_dimensions):  #XXX: different nbins for each direction?
    step = abs(upper[i] - lower[i])/nbins
    bins.append( [lower[i] + (j+0.5)*step for j in range(nbins)] )

  # build a grid of starting points
  initial_values = gridpts(bins)
  print "grid: %s" % initial_values

  npts = 10
  lower = [0.0, 3.0, 6.0]
  upper = [2.0, 5.0, 8.0]

  # generate a set of random starting points
  initial_values = samplepts(lower,upper,npts)
  print "scatter: %s" % initial_values


# EOF

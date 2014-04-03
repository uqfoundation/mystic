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
takes upper and lower bounds (e.g. ub = [2,4], lb = [0,3])
produces a list of sample points s = [[1,3],[1,4],[2,3],[2,4]]

Inputs:
    lower bounds  --  a list of the lower bounds
    upper bounds  --  a list of the upper bounds
    npts  --  number of sample points
    """
    from mystic.math.samples import random_samples
    q = random_samples(lb,ub,npts)
    q = [list(i) for i in q]
    q = zip(*q)
    return [list(i) for i in q]


def randomly_bin(nbins, ndim):
    """generate n bins randomly gridded across ndim dimensions"""
    from itertools import chain
    from random import random
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
    result = factors(nbins)
    result += [1] * (ndim - (len(result) / ndim))
    result = sorted(result, key=lambda v: random())
    from numpy import prod
    return [prod(result[i::ndim]) for i in range(ndim)]


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

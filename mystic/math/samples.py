#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2024 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
tools related to sampling
"""
# These functions are consolidated and adapted from samples.py and cut.py in 
# branches/UQ/math/ for general use.
# sample() is from cut.py, and samplepts() is from examples2/seesaw2d.py, and 
# everything else is from samples.py

# SAMPLING #
def _random_samples(lb, ub, npts=10000):
  """
generate npts random samples between given lb & ub

Inputs:
    lb -- a list of the lower bounds
    ub -- a list of the upper bounds
    npts -- number of sample points [default = 10000]
"""
  from mystic.tools import random_state
  dim = len(lb)
  pts = random_state(module='numpy.random').rand(dim,npts)
  for i in range(dim): #XXX: use array operations?
    lbi = lb[i]
    ubi = ub[i]
    pts[i] = (pts[i] * abs(ubi - lbi)) + lbi
  return pts  #XXX: returns a numpy.array
 #return [list(i) for i in pts]


def random_samples(lb, ub, npts=10000, dist=None, clip=False):
  """
generate npts samples from the given distribution between given lb & ub

Inputs:
    lb -- a list of the lower bounds
    ub -- a list of the upper bounds
    npts -- number of sample points [default = 10000]
    dist -- a mystic.tools.Distribution instance (or list of Distributions)
    clip -- if True, clip at bounds, else resample [default = False]
"""
  if dist is None:
    return _random_samples(lb,ub, npts)
  import numpy as np
  if hasattr(dist, '__len__'): #FIXME: isiterable
    pts = np.array(tuple(di(npts) for di in dist)).T
  else:
    pts = dist((npts,len(lb))) # transpose of desired shape
    dist = (dist,)*len(lb)
  pts = np.clip(pts, lb, ub).T
  if clip: return pts  #XXX: returns a numpy.array
  bad = ((pts.T == lb) + (pts.T == ub)).T
  new = bad.sum(-1)
  _n, n = 1, 1000 #FIXME: fixed number of max tries
  while any(new):
    if _n == n: #XXX: slows the while loop...
      raise RuntimeError('bounds could not be applied in %s iterations' % n)
    for i,inew in enumerate(new): #XXX: slows... but enables iterable dist
      if inew: pts[i][bad[i]] = dist[i](inew)
    pts = np.clip(pts.T, lb, ub).T
    bad = ((pts.T == lb) + (pts.T == ub)).T
    new = bad.sum(-1)
    _n += 1
  return pts  #XXX: returns a numpy.array


def sample(f, lb, ub, npts=10000, map=None):
  """
return number of failures and successes for some boolean function f

Inputs:
    f -- a function that returns True for 'success' and False for 'failure'
    lb -- a list of lower bounds
    ub -- a list of upper bounds
    npts -- the number of points to sample [Default is npts=10000]
    map -- the mapping function [Default is builtins.map]
"""
  if map is None:
    from builtins import map
  from numpy import transpose, atleast_2d
  pts = _random_samples(lb, ub, npts)

  results = list(map(f, atleast_2d(transpose(pts)).tolist()))
  failure = results.count(False)
  success = len(results) - failure
  return failure, success


# STATISTICS #
def sampled_mean(f, lb, ub, npts=10000, map=None):
  """
use random sampling to calculate the mean of a function

Inputs:
    f -- a function that takes a list and returns a number
    lb -- a list of lower bounds
    ub -- a list of upper bounds
    npts -- the number of points to sample [Default is npts=10000]
    map -- the mapping function [Default is builtins.map]
"""
  pts = _random_samples(lb, ub, npts)
  return _expectation_given_samples(f, pts, map)

# if map is None:
#   from builtins import map
# from numpy import inf, transpose, atleast_2d
# from mystic.tools import wrap_bounds
# pts = _random_samples(lb, ub, npts)
# f = wrap_bounds(f,lb,ub)

# if len(lb) != 1:
#   results = map(f, atleast_2d(transpose(pts)).tolist())
# else: #FIXME: fails for len(lb) == 1 (and the above works!)
#   _f = lambda x: f(float(x))
#   results = map(_f, transpose(pts))
# #ave = 0; count = 0
# #for Fx in results:
# #  if Fx != inf: # outside of bounds evaluates to inf
# #    ave += Fx
# #    count += 1
# results = list(filter(lambda Fx: Fx != inf, results))
# count = len(results)
# if not count: return None  #XXX: define 0/0 = None
# ave = sum(results)
# ave = float(ave) / float(count)
# return ave


def sampled_variance(f, lb, ub, npts=10000, map=None): #XXX: could be improved
  """
use random sampling to calculate the variance of a function

Inputs:
    f -- a function that takes a list and returns a number
    lb -- a list of lower bounds
    ub -- a list of upper bounds
    npts -- the number of points to sample [Default is npts=10000]
    map -- the mapping function [Default is builtins.map]
"""
  pts = _random_samples(lb, ub, npts)
  return _variance_given_samples(f, pts, map)


def sampled_pof(f, lb, ub, npts=10000, map=None):
  """
use random sampling to calculate probability of failure for a function

Inputs:
    f -- a function that returns True for 'success' and False for 'failure'
    lb -- a list of lower bounds
    ub -- a list of upper bounds
    npts -- the number of points to sample [Default is npts=10000]
    map -- the mapping function [Default is builtins.map]
"""
  pts = _random_samples(lb, ub, npts)
  return _pof_given_samples(f, pts, map)


# ALTERNATE: GIVEN SAMPLE POINTS #
def _pof_given_samples(f, pts, map=None):
  """
use given sample pts to calculate probability of failure for function f

Inputs:
    f -- a function that returns True for 'success' and False for 'failure'
    pts -- a list of sample points
    map -- the mapping function [Default is builtins.map]
"""
  if map is None:
    from builtins import map
  from numpy import transpose, atleast_2d
  results = list(map(f, atleast_2d(transpose(pts)).tolist()))
  pof = float(results.count(False)) / float(len(results))
  return pof


def _minimum_given_samples(f, pts, map=None):
  """
use given sample pts to calculate minimum for function f

Inputs:
    f -- a function that returns a single value, given a list of inputs
    pts -- a list of sample points
    map -- the mapping function [Default is builtins.map]
"""
  if map is None:
    from builtins import map
  from numpy import transpose, atleast_2d
  return min(list(map(f, atleast_2d(transpose(pts)).tolist())))


def _expectation_given_samples(f, pts, map=None):
  """
use given sample pts to calculate expected value for function f

Inputs:
    f -- a function that returns a single value, given a list of inputs
    pts -- a list of sample points
    map -- the mapping function [Default is builtins.map]
"""
  if map is None:
    from builtins import map
  from numpy import transpose, mean, atleast_2d
  return mean(list(map(f, atleast_2d(transpose(pts)).tolist())))


def _variance_given_samples(f, pts, map=None):
  """
use given sample pts to calculate expected variance for function f

Inputs:
    f -- a function that returns a single value, given a list of inputs
    pts -- a list of sample points
    map -- the mapping function [Default is builtins.map]
"""
  if map is None:
    from builtins import map
  from numpy import transpose, var, atleast_2d
  return var(list(map(f, atleast_2d(transpose(pts)).tolist())))


def _maximum_given_samples(f, pts, map=None):
  """
use given sample pts to calculate maximum for function f

Inputs:
    f -- a function that returns a single value, given a list of inputs
    pts -- a list of sample points
    map -- the mapping function [Default is builtins.map]
"""
  if map is None:
    from builtins import map
  from numpy import transpose, atleast_2d
  return max(list(map(f, atleast_2d(transpose(pts)).tolist())))


def _ptp_given_samples(f, pts, map=None):
  """
use given sample pts to calculate spread for function f

Inputs:
    f -- a function that returns a single value, given a list of inputs
    pts -- a list of sample points
    map -- the mapping function [Default is builtins.map]
"""
  if map is None:
    from builtins import map
  from numpy import transpose, ptp, atleast_2d
  return ptp(list(map(f, atleast_2d(transpose(pts)).tolist())))


def sampled_pts(pts, lb, ub, map=None):
  """
determine the number of sample points inside the given bounds

Inputs:
    pts -- a list of sample points
    lb -- a list of lower bounds
    ub -- a list of upper bounds
    map -- the mapping function [Default is builtins.map]
"""
  if map is None:
    from builtins import map
  from numpy import inf, transpose, atleast_2d
  from mystic.tools import wrap_bounds
  def zero(x):
    return 0
  f = wrap_bounds(zero,lb,ub)
  results = map(f, atleast_2d(transpose(pts)).tolist())
  #npts = len(list(filter(lambda Fx: Fx != inf, results)))
  npts = list(results).count(0) # outside of bounds evaluates to inf
  return npts

def sampled_prob(pts, lb, ub, map=None):
  """
calculates probability by sampling if points are inside the given bounds

Inputs:
    pts -- a list of sample points
    lb -- a list of lower bounds
    ub -- a list of upper bounds
    map -- the mapping function [Default is builtins.map]
"""
  prob = float(sampled_pts(pts,lb,ub,map)) / float(len(pts[0]))
  return prob






#def minF(x):
#  return model(x)

#def maxF(x):
#  return -model(x)

def alpha(n, diameter, epsilon=0.01):
  from math import log
 #return diameter * n**(-0.5) * (-log(epsilon))**(0.5)
  return diameter * (-log(epsilon) / (2.0 * n))**(0.5)


#######################################################################


if __name__ == '__main__':

  def __test1():
    # From branches/UQ/math/samples.py
    num_sample_points = 5
    lower = [10.0, 0.0, 2.1]
    upper = [100.0, 30.0, 2.8]

    pts = _random_samples(lower,upper,num_sample_points)
    print("randomly sampled points\nbetween %s and %s" % (lower, upper))
    print(pts)

  def __test2():
    # From branches/UQ/math/cut.py
    from mystic.tools import random_seed
    random_seed(123)
    lower = [-60.0, -10.0, -50.0]
    upper = [105.0, 30.0, 75.0]

    def model(x):
      x1,x2,x3 = x
      if x1 > (x2 + x3): return x1*x2 - x3
      return 0.0

    failure,success = sample(model,lower,upper)
    pof = float(failure) / float(failure + success)
    print("PoF using method 1: %s" % pof)
    random_seed(123)
    print("PoF using method 2: %s" % sampled_pof(model,lower,upper))

  # run the tests
  __test1()
  __test2()


# EOF

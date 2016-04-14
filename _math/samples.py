#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
tools related to sampling
"""
from __future__ import division
from __future__ import print_function
from builtins import range
from past.utils import old_div
# These functions are consolidated and adapted from samples.py and cut.py in 
# branches/UQ/math/ for general use.
# sample() is from cut.py, and samplepts() is from examples2/seesaw2d.py, and 
# everything else is from samples.py

# SAMPLING #
def random_samples(lb,ub,npts=10000):
  """
generate npts random samples between given lb & ub

Inputs:
    lower bounds  --  a list of the lower bounds
    upper bounds  --  a list of the upper bounds
    npts  --  number of sample points [default = 10000]
"""
  from mystic.tools import random_state
  dim = len(lb)
  pts = random_state(module='numpy.random').rand(dim,npts)
  for i in range(dim):
    pts[i] = (pts[i] * abs(ub[i] - lb[i])) + lb[i]
  return pts  #XXX: returns a numpy.array
 #return [list(i) for i in pts]


def sample(f,lb,ub,npts=10000):
  """
return number of failures and successes for some boolean function f

Inputs:
    f -- a function that returns True for 'success' and False for 'failure'
    lb -- a list of lower bounds
    ub -- a list of upper bounds
    npts -- the number of points to sample [Default is npts=10000]
"""
  from numpy import transpose
  pts = random_samples(lb, ub, npts)

  failure = 0; success = 0
  for i in range(npts):
    xvector = transpose(pts)[i]
    if f(list(xvector)):
      success += 1
    else:
      failure += 1
  return failure,success


# STATISTICS #
def sampled_mean(f, lb,ub, npts=10000):
  """
use random sampling to calculate the mean of a function

Inputs:
    f -- a function that takes a list and returns a number
    lb -- a list of lower bounds
    ub -- a list of upper bounds
    npts -- the number of points to sample [Default is npts=10000]
"""
  from numpy import inf, transpose
  from mystic.tools import wrap_bounds
  pts = random_samples(lb, ub, npts)
  f = wrap_bounds(f,lb,ub)
  ave = 0; count = 0
  for i in range(len(pts[0])):
    if len(lb) != 1:
      xvector = transpose(pts)[i]
      Fx = f(list(xvector))
    else:
      Fx = f(float(transpose(pts)[i]))
    if Fx != -inf: # outside of bounds evaluates to -inf
      ave += Fx
      count += 1
  if not count: return None  #XXX: define 0/0 = None
  ave = old_div(float(ave), float(count))
  return ave


def sampled_variance(f, lb, ub, npts=10000): #XXX: this could be improved
  """
use random sampling to calculate the variance of a function

Inputs:
    f -- a function that takes a list and returns a number
    lb -- a list of lower bounds
    ub -- a list of upper bounds
    npts -- the number of points to sample [Default is npts=10000]
"""
  m = sampled_mean(f,lb,ub)
  def g(x):
    return abs(f(x) - m)**2
  return sampled_mean(g,lb,ub)


def sampled_pof(f, lb, ub, npts=10000):
  """
use random sampling to calculate probability of failure for a function

Inputs:
    f -- a function that returns True for 'success' and False for 'failure'
    lb -- a list of lower bounds
    ub -- a list of upper bounds
    npts -- the number of points to sample [Default is npts=10000]
"""
  pts = random_samples(lb, ub, npts)
  return _pof_given_samples(f, pts)


# ALTERNATE: GIVEN SAMPLE POINTS #
def _pof_given_samples(f, pts):
  """
use given sample pts to calculate probability of failure for function f

Inputs:
    f -- a function that returns True for 'success' and False for 'failure'
    pts -- a list of sample points
"""
  from numpy import transpose
  failure = 0
  npts = len(pts[0]) #XXX: fails when pts = []; also assumes a nested list
  for i in range(npts):
    xvector = transpose(pts)[i]
    if not f(list(xvector)):
      failure += 1
  pof = old_div(float(failure), float(npts))
  return pof


def sampled_pts(pts,lb,ub):
  """
determine the number of sample points inside the given bounds

Inputs:
    pts -- a list of sample points
    lb -- a list of lower bounds
    ub -- a list of upper bounds
"""
  from numpy import inf, transpose
  def identity(x):
    return x
  from mystic.tools import wrap_bounds
  f = wrap_bounds(identity,lb,ub)
  npts = 0
  for i in range(len(pts[0])):
    xvector = transpose(pts)[i]
    Fx = f(list(xvector))
    if Fx != -inf: # outside of bounds evaluates to -inf
      npts += 1
  return npts

def sampled_prob(pts,lb,ub):
  """
calculates probability by sampling if points are inside the given bounds

Inputs:
    pts -- a list of sample points
    lb -- a list of lower bounds
    ub -- a list of upper bounds
"""
  prob = old_div(float(sampled_pts(pts,lb,ub)), float(len(pts[0])))
  return prob






#def minF(x):
#  return model(x)

#def maxF(x):
#  return -model(x)

def alpha(n,diameter,epsilon=0.01):
  from math import log
 #return diameter * n**(-0.5) * (-log(epsilon))**(0.5)
  return diameter * (old_div(-log(epsilon), (2.0 * n)))**(0.5)


#######################################################################


if __name__ == '__main__':

  def __test1():
    # From branches/UQ/math/samples.py
    num_sample_points = 5
    lower = [10.0, 0.0, 2.1]
    upper = [100.0, 30.0, 2.8]

    pts = random_samples(lower,upper,num_sample_points)
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
    pof = old_div(float(failure), float(failure + success))
    print("PoF using method 1: %s" % pof)
    random_seed(123)
    print("PoF using method 2: %s" % sampled_pof(model,lower,upper))

    # run the tests
    __test1()
    __test2()


# EOF

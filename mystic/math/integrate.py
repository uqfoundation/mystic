#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
math tools related to integration
"""

# INTEGRATE #
def integrate(f, lb, ub):
  """
Returns the integral of an n-dimensional function f from lb to ub

Inputs:
    f -- a function that takes a list and returns a number
    lb -- a list of lower bounds
    ub -- a list of upper bounds

If scipy is installed, and number of dimensions is 3 or less, scipy.integrate
is used. Otherwise, use mystic's n-dimensional Monte Carlo integrator."""
  # Try to use scipy if possible for problems whose dimension is 1, 2, or 3
  # Otherwise, use n-dimensional Monte Carlo integrator
  if len(lb) <= 3:
    scipy = True
    try: 
      import imp
      imp.find_module('scipy')
    except ImportError:
      scipy = False
    if scipy:
      expectation = _scipy_integrate(f, lb, ub)
    else:
      expectation = monte_carlo_integrate(f, lb, ub)
  else:
    expectation = monte_carlo_integrate(f, lb, ub)
  return expectation


# STATISTICS #
def integrated_mean(f, lb, ub):
  """
calculate the integrated mean of a function f

Inputs:
    f -- a function that takes a list and returns a number
    lb -- a list of lower bounds
    ub -- a list of upper bounds
"""
  expectation = integrate(f, lb, ub)
  from mystic.math.stats import mean,volume
  vol = volume(lb, ub)
  return mean(expectation, vol)

def integrated_variance(f,lb,ub):
  """
calculate the integrated variance of a function f

Inputs:
    f -- a function that takes a list and returns a number
    lb -- a list of lower bounds
    ub -- a list of upper bounds
"""
  m = integrated_mean(f,lb,ub)
  def g(x):
    return abs(f(x) - m)**2
  return integrated_mean(g,lb,ub)

# def integrated_pof(f, lb, ub):


# ALTERNATE: INTEGRATION METHODS #
def _scipy_integrate(f,lb,ub):  
  """
Returns the integral of an n-dimensional function f from lb to ub
(where n = 1, 2, or 3), using scipy.integrate

Inputs:
    f -- a function that takes a list and returns a number.
    lb -- a list of lower bounds
    ub -- a list of upper bounds
"""
  from scipy.integrate import quad, dblquad, tplquad
  if len(lb) == 3:
    def func(z,y,x): return f([x,y,z])
    def qf(x,y): return lb[2]
    def rf(x,y): return ub[2]
    def gf(x): return lb[1]
    def hf(x): return ub[1]
    expectation,confidence = tplquad(func,lb[0],ub[0],gf,hf,qf,rf)
    return expectation
  if len(lb) == 2:
    def func(y,x): return f([x,y])
    def gf(x): return lb[1]
    def hf(x): return ub[1]
    expectation,confidence = dblquad(func,lb[0],ub[0],gf,hf)
    return expectation 
  if len(lb) == 1:
    expectation,confidence = quad(f,lb[0],ub[0])
    return expectation 


def monte_carlo_integrate(f, lb, ub, n=10000): #XXX ok default for n?
  """
Returns the integral of an m-dimensional function f from lb to ub
using a Monte Carlo integration of n points

Inputs:
    f -- a function that takes a list and returns a number.
    lb -- a list of lower bounds
    ub -- a list of upper bounds
    n -- the number of points to sample [Default is n=10000]

References:
    1. "A Primer on Scientific Programming with Python", by Hans Petter
       Langtangen, page 443-445, 2014.
    2. http://en.wikipedia.org/wiki/Monte_Carlo_integration
    3. http://math.fullerton.edu/mathews/n2003/MonteCarloMod.html
"""
  from mystic.math.stats import volume
  from mystic.math.samples import random_samples
  vol = volume(lb, ub)
  x = [random_samples(lb, ub, npts=1) for k in range(1, n+1)]
  r = map(f, x)  #FIXME: , nnodes=nnodes, launcher=launcher)
  s = sum(r)[0]
  I = (vol/n)*s
  return float(I)


# ALTERNATE: STATISTICS SPECIAL CASES #
def __uniform_integrated_mean(lb,ub):
  """use integration of cumulative function to calculate mean (in 1D)

same as: mean = (lb + ub) / 2.0
"""
  lb = float(lb); ub = float(ub)
  def g(x): return x
  return integrated_mean(g,[lb],[ub])

def __uniform_integrated_variance(lb,ub):
  """use integration of cumulative function to calculate variance (in 1D)

same as: variance = (ub - lb)**2 / 12.0
"""
  lb = float(lb); ub = float(ub)
  def g(x): return x
  return integrated_variance(g,[lb],[ub])

#----------------------------------------------------------------------------
# Tests

def __test_integrator1():
  def f(x):
    return x[0]/2. + x[1]**3 + 3.*x[2]
  lb = [1., 1., 1.]
  ub = [2., 2., 2.]
  print("monte_carlo_integrate says: %s" % monte_carlo_integrate(f, lb, ub))
  print("_scipy_integrate says: %s" % _scipy_integrate(f, lb, ub))
  return

def __test_mean():
  def f(x):
    return x
  lb = [0.]
  ub = [1.]
  from mystic.math.samples import sampled_mean
  print("Sampled mean says: %s" % sampled_mean(f, lb, ub))
  print("Integrated mean says: %s" % integrated_mean(f, lb, ub))

if __name__ == '__main__':
  __test_integrator1()
  __test_mean()

# EOF

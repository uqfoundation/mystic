#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Methods to support discrete measures
"""
# what about sample_unsafe ?
from mystic.math.stats import *
from mystic.math.samples import *
from mystic.math.integrate import *
from mystic.symbolic import generate_solvers, generate_constraint, solve
from mystic.symbolic import generate_conditions, generate_penalty
from mystic.math import almostEqual
from mystic.math.distance import Lnorm

__all__ = ['weighted_select','spread','norm','maximum','ess_maximum',\
          'minimum','ess_minimum','ptp','ess_ptp','expectation',\
          'expected_variance','expected_std','_expected_moment','mean',\
          'support_index','support','moment','standard_moment','variance',\
          'std','skewness','kurtosis','impose_mean','impose_variance',\
          'impose_std','impose_moment','impose_spread','impose_expectation',\
          '_impose_expected_moment','impose_expected_variance',\
          'impose_expected_std','impose_expected_mean_and_variance',\
          'impose_weight_norm','normalize','impose_reweighted_mean',\
          'impose_reweighted_variance','impose_reweighted_std','_sort',\
          'median','mad','impose_median','impose_mad','_k','tmean',\
          'tvariance','tstd','impose_tmean','impose_tvariance','impose_tstd',\
          'impose_support','impose_unweighted','impose_collapse',\
          'impose_sum','impose_product','_pack','_unpack','_flat',\
          '_nested','_nested_split','split_param']

def weighted_select(samples, weights, mass=1.0):
  """randomly select a sample from weighted set of samples

Args:
    samples (list): a list of sample points
    weights (list): a list of sample weights
    mass (float, default=1.0): sum of normalized weights

Returns:
    a randomly selected sample point
"""
  from numpy import sum, array
  from mystic.tools import random_state
  rand = random_state().random
  # generate a list representing the weighted distribution
  wts = normalize(weights, mass)
  wts = array([sum(wts[:i+1]) for i in range(len(wts))])
  # correct for any rounding error
  wts[-1] = mass
  # generate a random weight
  w = mass * rand()
  # select samples that corresponds to randomly selected weight
  selected = len(wts[ wts <= w ])
  return samples[selected]

##### calculate methods #####
def spread(samples):
  """calculate the range for a list of points 

``spread(x) = max(x) - min(x)``

Args:
    samples (list): a list of sample points

Returns:
    the range of the samples
"""
  return max(samples) - min(samples)

def norm(weights):
  """calculate the norm of a list of points

``norm(x) = mean(x)``

Args:
    weights (list): a list of sample weights

Returns:
    the mean of the weights
"""
  return mean(weights)

def maximum(f, samples):
  """calculate the max of function for the given list of points

``maximum(f,x) = max(f(x))``

Args:
    f (func): a function that takes a list and returns a number
    samples (list): a list of sample points

Returns:
    the maximum output value for a function at the given inputs
"""
  y = [f(x) for x in samples] #XXX: parallel map?
  return max(y)

def ess_maximum(f, samples, weights=None, tol=0.):
  """calculate the max of function for support on the given list of points

``ess_maximum(f,x,w) = max(f(support(x,w)))``

Args:
    f (func): a function that takes a list and returns a number
    samples (list): a list of sample points
    weights (list, default=None): a list of sample weights
    tol (float, default=0.0): a tolerance, where any ``weight <= tol`` is zero

Returns:
    the maximum output value for a function at the given support points
"""
  if weights is None:
    return maximum(f, samples)
  return maximum(f, support(samples, weights, tol))

def minimum(f, samples):
  """calculate the min of function for the given list of points

``minimum(f,x) = min(f(x))``

Args:
    f (func): a function that takes a list and returns a number
    samples (list): a list of sample points

Returns:
    the minimum output value for a function at the given inputs
"""
  y = [f(x) for x in samples] #XXX: parallel map?
  return min(y)

def ess_minimum(f, samples, weights=None, tol=0.):
  """calculate the min of function for support on the given list of points

``ess_minimum(f,x,w) = min(f(support(x,w)))``

Args:
    f (func): a function that takes a list and returns a number
    samples (list): a list of sample points
    weights (list, default=None): a list of sample weights
    tol (float, default=0.0): a tolerance, where any ``weight <= tol`` is zero

Returns:
    the minimum output value for a function at the given support points
"""
  if weights is None:
    return minimum(f, samples)
  return minimum(f, support(samples, weights, tol))

def ptp(f, samples):
  """calculate the spread of function for the given list of points

``minimum(f,x) = max(f(x)) - min(f(x))``

Args:
    f (func): a function that takes a list and returns a number
    samples (list): a list of sample points

Returns:
    the spread in output value for a function at the given inputs
"""
  y = [f(x) for x in samples] #XXX: parallel map?
  return max(y) - min(y)

def ess_ptp(f, samples, weights=None, tol=0.):
  """calculate the spread of function for support on the given list of points

``ess_minimum(f,x,w) = max(f(support(x,w))) - min(f(support(x,w)))``

Args:
    f (func): a function that takes a list and returns a number
    samples (list): a list of sample points
    weights (list, default=None): a list of sample weights
    tol (float, default=0.0): a tolerance, where any ``weight <= tol`` is zero

Returns:
    the spread in output value for a function at the given support points
"""
  if weights is None:
    return ptp(f, samples)
  return ptp(f, support(samples, weights, tol))

def expectation(f, samples, weights=None, tol=0.0):
  """calculate the (weighted) expectation of a function for a list of points

Args:
    f (func): a function that takes a list and returns a number
    samples (list): a list of sample points
    weights (list, default=None): a list of sample weights
    tol (float, default=0.0): a tolerance, where any ``weight <= tol`` is zero

Returns:
    the weighted expectation for a list of sample points
"""
  if weights is None:
    y = [f(x) for x in samples] #XXX: parallel map?
    return mean(y, weights)
  # contributed by TJS #
  # to prevent function evaluation if weight is "too small":
  # skip evaluation of f(x) if the corresponding weight <= tol
  # import itertools as it   #XXX: assumes is faster than zip
  #weights = normalize(weights, mass=1.0) #FIXME: below is atol, should be rtol?
  if not sum(abs(w) > tol for w in weights):
      yw = ((0.0,0.0),)
  else: #XXX: parallel map?
      yw = [(f(x),w) for (x,w) in zip(samples, weights) if abs(w) > tol]
  return mean(*zip(*yw))
  ##XXX: at around len(samples) == 150, the following is faster
  #aw = asarray(weights)
  #ax = asarray(samples)
  #w = aw > tol
  #return mean(ax[w], aw[w])

def _expected_moment(f, samples, weights=None, order=1, tol=0.0):
  """calculate the (weighted) nth-order expected moment of a function

Args:
    f (func): a function that takes a list and returns a number
    samples (list): a list of sample points
    weights (list, default=None): a list of sample weights
    order (int, default=1): the degree, a positive integer
    tol (float, default=0.0): a tolerance, where any ``weight <= tol`` is zero

Returns:
    the weighted nth-order expected moment of f on a list of sample points
"""
  if order < 0:
    raise NotImplementedError
  if weights is None:
    y = [f(x) for x in samples] #XXX: parallel map?
    return moment(y, weights, order) #XXX: tol?
  # skip evaluation of f(x) if the corresponding weight <= tol
  if not sum(abs(w) > tol for w in weights):
      yw = ((0.0,0.0),)
  else: #XXX: parallel map?
      yw = [(f(x),w) for (x,w) in zip(samples, weights) if abs(w) > tol]
  return moment(*zip(*yw), order=order)


def expected_variance(f, samples, weights=None, tol=0.0):
  """calculate the (weighted) expected variance of a function

Args:
    f (func): a function that takes a list and returns a number
    samples (list): a list of sample points
    weights (list, default=None): a list of sample weights
    tol (float, default=0.0): a tolerance, where any ``weight <= tol`` is zero

Returns:
    the weighted expected variance of f on a list of sample points
"""
  return _expected_moment(f, samples, weights, order=2, tol=tol)


def expected_std(f, samples, weights=None, tol=0.0):
  """calculate the (weighted) expected standard deviation of a function

Args:
    f (func): a function that takes a list and returns a number
    samples (list): a list of sample points
    weights (list, default=None): a list of sample weights
    tol (float, default=0.0): a tolerance, where any ``weight <= tol`` is zero

Returns:
    the weighted expected standard deviation of f on a list of sample points
"""
  from numpy import sqrt
  return sqrt(expected_variance(f, samples, weights, tol=tol))


def mean(samples, weights=None, tol=0):
  """calculate the (weighted) mean for a list of points

Args:
    samples (list): a list of sample points
    weights (list, default=None): a list of sample weights
    tol (float, default=0.0): a tolerance, where any ``mean <= tol`` is zero

Returns:
    the weighted mean for a list of sample points
"""
  if weights is None:
    weights = [1.0/float(len(samples))] * len(samples)
  # get weighted sum
  ssum = sum(i*j for i,j in zip(samples, weights))
  # normalize by sum of the weights
  wts = float(sum(weights))
  if wts:
    ssum = ssum / wts
    return 0.0 if abs(ssum) <= tol else ssum
  from numpy import inf
  return ssum * inf  # protect against ZeroDivision

def support_index(weights, tol=0): #XXX: no relative tolerance near zero
  """get the indices of the positions which have non-zero weight

Args:
    weights (list): a list of sample weights
    tol (float, default=0.0): a tolerance, where any ``weight <= tol`` is zero

Returns:
    a list of indices of positions with non-zero weight
"""
  return [i for (i,w) in enumerate(weights) if w > tol]

def support(samples, weights, tol=0): #XXX: no relative tolerance near zero
  """get the positions which have non-zero weight

Args:
    samples (list): a list of sample points
    weights (list): a list of sample weights
    tol (float, default=0.0): a tolerance, where any ``weight <= tol`` is zero

Returns:
    a list of positions with non-zero weight
"""
  return [samples[i] for (i,w) in enumerate(weights) if w > tol]

def moment(samples, weights=None, order=1, tol=0): #, _mean=None):
  """calculate the (weighted) nth-order moment for a list of points

Args:
    samples (list): a list of sample points
    weights (list, default=None): a list of sample weights
    order (int, default=1): the degree, a positive integer
    tol (float, default=0.0): a tolerance, where any ``mean <= tol`` is zero

Returns:
    the weighted nth-order moment for a list of sample points
"""
  if order == 0: return 1.0 #XXX: error if order < 0
  if order == 1: return 0.0
  if weights is None:
    weights = [1.0/float(len(samples))] * len(samples)
 #if _mean is None:
  _mean = mean(samples, weights)
  mom = [(s - _mean)**order for s in samples] #XXX: abs(s - _mean) ???
  return mean(mom,weights,tol) #XXX: sample_moment (Bessel correction) *N/(N-1)?

def standard_moment(samples, weights=None, order=1, tol=0):
  """calculate the (weighted) nth-order standard moment for a list of points

``standard_moment(x,w,order) = moment(x,w,order)/std(x,w)^order``

Args:
    samples (list): a list of sample points
    weights (list, default=None): a list of sample weights
    order (int, default=1): the degree, a positive integer
    tol (float, default=0.0): a tolerance, where any ``mean <= tol`` is zero

Returns:
    the weighted nth-order standard moment for a list of sample points
"""
  if order == 2: return 1.0 #XXX: error if order < 0
  return moment(samples, weights, order, tol)/std(samples, weights)**order

def variance(samples, weights=None): #,tol=0, _mean=None):
  """calculate the (weighted) variance for a list of points

Args:
    samples (list): a list of sample points
    weights (list, default=None): a list of sample weights

Returns:
    the weighted variance for a list of sample points
"""
  return moment(samples, weights, order=2)

def std(samples, weights=None): #,tol=0, _mean=None):
  """calculate the (weighted) standard deviation for a list of points

Args:
    samples (list): a list of sample points
    weights (list, default=None): a list of sample weights

Returns:
    the weighted standard deviation for a list of sample points
"""
  from numpy import sqrt
  return sqrt(variance(samples, weights)) # _mean)

def skewness(samples, weights=None): #,tol=0, _mean=None):
  """calculate the (weighted) skewness for a list of points

Args:
    samples (list): a list of sample points
    weights (list, default=None): a list of sample weights

Returns:
    the weighted skewness for a list of sample points
"""
  return standard_moment(samples, weights, order=3) # _mean)

def kurtosis(samples, weights=None): #,tol=0, _mean=None):
  """calculate the (weighted) kurtosis for a list of points

Args:
    samples (list): a list of sample points
    weights (list, default=None): a list of sample weights

Returns:
    the weighted kurtosis for a list of sample points
"""
  return standard_moment(samples, weights, order=4) # _mean)


##### coordinate shift methods #####
from numpy import asarray
def impose_mean(m, samples, weights=None): #,tol=0):
  """impose a mean on a list of (weighted) points

Args:
    m (float): the target mean
    samples (list): a list of sample points
    weights (list, default=None): a list of sample weights

Returns:
    a list of sample points with the desired weighted mean

Notes:
    this function does not alter the weighted range or the weighted variance
"""
 #XXX: this is as expected... mean(impose_mean(2.0, samples, weights), weights)
 #XXX: this is unexpected?... mean(impose_mean(2.0, samples, weights))
  samples = asarray(list(samples)) #XXX: faster to use x = array(x, copy=True) ?
  shift = m - mean(samples, weights)
  samples = samples + shift  #NOTE: is "range-preserving"
  return list(samples)


def impose_variance(v, samples, weights=None): #,tol=0):
  """impose a variance on a list of (weighted) points

Args:
    v (float): the target variance
    samples (list): a list of sample points
    weights (list, default=None): a list of sample weights

Returns:
    a list of sample points with the desired weighted variance

Notes:
    this function does not alter the weighted mean
"""
  m = mean(samples, weights)
  samples = asarray(list(samples)) #XXX: faster to use x = array(x, copy=True) ?
  sv = variance(samples,weights) #,m)
  if not sv:  # protect against ZeroDivision when variance = 0
    if not v: # variance is to be 0
      return [float(i) for i in samples] #samples.tolist()
    from numpy import nan
    return [nan]*len(samples) #XXX: better to space pts evenly across range?
  from numpy import sqrt, seterr
  err = seterr(invalid='ignore')
  scale = sqrt(float(v) / sv)
  seterr(**err)
  samples = samples * scale  #NOTE: not "mean-preserving", until the next line
  return impose_mean(m, samples, weights) #NOTE: not range preserving

#FIXME: for range and variance to be 'mutually preserving'...
#       must reconcile scaling by sqrt(v2/v1) & (r2/r1)
#       ...so likely, must scale the weights... or scale each point differently

def impose_std(s, samples, weights=None):
  """impose a standard deviation on a list of (weighted) points

Args:
    s (float): the target standard deviation
    samples (list): a list of sample points
    weights (list, default=None): a list of sample weights

Returns:
    a list of sample points with the desired weighted standard deviation

Notes:
    this function does not alter the weighted mean
"""
  return impose_variance(s**2, samples, weights)


def impose_moment(m, samples, weights=None, order=1, tol=0, skew=None):
  """impose the selected moment on a list of (weighted) points

Args:
    m (float): the target moment
    samples (list): a list of sample points
    weights (list, default=None): a list of sample weights
    order (int, default=1): the degree, a positive integer
    tol (float, default=0.0): a tolerance, where any ``mean <= tol`` is zero
    skew (bool, default=None): if True, allow skew in the samples

Returns:
    a list of sample points with the desired weighted moment

Notes:
    this function does not alter the weighted mean

    if *skew* is None, then allow *skew* when *order* is odd
"""
  v = m #NOTE: change of variables, so code is consistent with impose_variance
  if order == 0:
    if v == 1: return [float(i) for i in samples]
    else:
      from numpy import nan
      return [nan]*len(samples)
  if order == 1:
    if not v: return [float(i) for i in samples]
    else:
      from numpy import nan
      return [nan]*len(samples)
  if not order%2 and v < 0.0:
    from numpy import nan
    return [nan]*len(samples)
  m = mean(samples, weights)
  if skew is None: skew = order%2 # if odd
  if skew: samples = [i**2 for i in samples]
  sv = moment(samples,weights,order,tol)
  if not sv: #moment == 0     #NOTE: caution if ~0.0 or -v
    if not order%2: # not odd
      return [m]*len(samples)
    if not v: # moment is to be 0
      return [float(i) for i in samples] #samples.tolist()
   #if skew: then unskewed should have worked. XXX: handle this?
    from numpy import nan
    return [nan]*len(samples)
  samples = asarray(list(samples))
  from numpy import power, flipud
  fact = float(v)/sv
  # temporarily account for negative fact
  if order%2 and fact < 0: flip = True
  else: flip = False
 #try:
  scale = power(abs(fact), 1./order)
 #except ZeroDivisionError:
 #  from numpy import nan
 #  return [nan]*len(samples)
  if flip: samples = max(samples) + min(samples) - samples
 #if flip: samples = flipud(max(samples) + min(samples) - samples)
  samples = samples * scale
  return impose_mean(m, samples, weights)


def impose_spread(r, samples, weights=None): #FIXME: fails if len(samples) = 1
  """impose a range on a list of (weighted) points

Args:
    r (float): the target range
    samples (list): a list of sample points
    weights (list, default=None): a list of sample weights

Returns:
    a list of sample points with the desired weighted range

Notes:
    this function does not alter the weighted mean
"""
  m = mean(samples, weights)
  samples = asarray(list(samples)) #XXX: faster to use x = array(x, copy=True) ?
  sr = spread(samples)
  if not sr:  # protect against ZeroDivision when range = 0
    from numpy import nan
    return [nan]*len(samples) #XXX: better to space pts evenly across range?
  scale = float(r) / sr
  samples = samples * scale  #NOTE: not "mean-preserving", until the next line
  return impose_mean(m, samples, weights) #NOTE: not variance preserving


#NOTE: backward incompatible 08/26/18: (param=(m,D),...) --> (m,...,tol=D) 
def impose_expectation(m, f, npts, bounds=None, weights=None, **kwds):
  """impose a given expectation value ``E`` on a given function *f*,
where ``E = m +/- tol`` and ``E = mean(f(x))`` for ``x`` in *bounds*

Args:
    m (float): target expected mean
    f (func): a function that takes a list and returns a number
    npts (tuple(int)): a tuple of dimensions of the target product measure
    bounds (tuple, default=None): tuple is ``(lower_bounds, upper_bounds)``
    weights (list, default=None): a list of sample weights
    tol (float, default=None): maximum allowable deviation from ``m``
    constraints (func, default=None): a function that takes a nested list of
        ``N x 1D`` discrete measure positions and weights, with the intended
        purpose of kernel-transforming ``x,w`` as ``x' = constraints(x, w)``
    npop (int, default=200): size of the trial solution population
    maxiter (int, default=1000): the maximum number of iterations to perform
    maxfun (int, default=1e+6): the maximum number of function evaluations

Returns:
    a list of sample positions, with expectation ``E``

Notes:
    Expectation value ``E`` is calculated by minimizing ``mean(f(x)) - m``,
    over the given *bounds*, and will terminate when ``E`` is found within
    deviation ``tol`` of the target mean ``m``.  If ``tol`` is not provided,
    then a relative deviation of 1% of ``m`` will be used.

    This function does not preserve the mean, variance, or range, as there
    is no initial list of samples to draw the mean, variance, and etc from

    *bounds* is tuple with ``length(bounds) == 2``, composed of all the lower
    bounds, then all the upper bounds, for each parameter

Examples:
    >>> # provide the dimensions and bounds
    >>> nx = 3;  ny = 2;  nz = 1
    >>> x_lb = [10.0];  y_lb = [0.0];  z_lb = [10.0]
    >>> x_ub = [50.0];  y_ub = [9.0];  z_ub = [90.0]
    >>> 
    >>> # prepare the bounds
    >>> lb = (nx * x_lb) + (ny * y_lb) + (nz * z_lb)
    >>> ub = (nx * x_ub) + (ny * y_ub) + (nz * z_ub)
    >>>
    >>> # generate a list of samples with mean +/- dev imposed
    >>> mean = 2.0;  dev = 0.01
    >>> samples = impose_expectation(mean, f, (nx,ny,nz), (lb,ub), tol=dev)
    >>>
    >>> # test the results by calculating the expectation value for the samples
    >>> expectation(f, samples)
    >>> 2.000010010122465
"""
  # param[0] is the target mean
  # param[1] is the acceptable deviation from the target mean
  tol = kwds['tol'] if 'tol' in kwds else None
  param = (m,tol or 0.01*m)

  # FIXME: the following is a HACK to recover from lost 'weights' information
  #        we 'mimic' discrete measures using the product measure weights
  # plug in the 'constraints' function:  samples' = constrain(samples, weights)
  constrain = None   # default is no constraints
  if 'constraints' in kwds: constrain = kwds['constraints']
  if not constrain:  # if None (default), there are no constraints
    constraints = lambda x: x
  else: #XXX: better to use a standard "xk' = constrain(xk)" interface ?
    def constraints(rv):
      coords = _pack( _nested(rv,npts) )
      coords = list(zip(*coords))              # 'mimic' a nested list
      coords = constrain(coords, [weights for i in range(len(coords))])
      coords = list(zip(*coords))              # revert back to a packed list
      return _flat( _unpack(coords,npts) )

  # construct cost function to reduce deviation from expectation value
  def cost(rv):
    """compute cost from a 1-d array of model parameters,
    where:  cost = | E[model] - m |**2 """
    # from mystic.math.measures import _pack, _nested, expectation
    samples = _pack( _nested(rv,npts) )
    Ex = expectation(f, samples, weights)
    return (Ex - param[0])**2

  # if bounds are not set, use the default optimizer bounds
  if not bounds:
    lower_bounds = []; upper_bounds = []
    for n in npts:
      lower_bounds += [None]*n
      upper_bounds += [None]*n
  else: 
    lower_bounds, upper_bounds = bounds

  # construct and configure optimizer
  debug = kwds['debug'] if 'debug' in kwds else False
  npop = kwds.pop('npop', 200)
  maxiter = kwds.pop('maxiter', 1000)
  maxfun = kwds.pop('maxfun', 1e+6)
  crossover = 0.9; percent_change = 0.9

  def optimize(cost, bounds, tolerance, _constraints):
    (lb,ub) = bounds
    from mystic.solvers import DifferentialEvolutionSolver2
    from mystic.termination import VTR
    from mystic.strategy import Best1Exp
    from mystic.monitors import VerboseMonitor, Monitor
    from mystic.tools import random_seed
    if debug: random_seed(123)
    evalmon = Monitor();  stepmon = Monitor()
    if debug: stepmon = VerboseMonitor(10)

    ndim = len(lb)
    solver = DifferentialEvolutionSolver2(ndim,npop)
    solver.SetRandomInitialPoints(min=lb,max=ub)
    solver.SetStrictRanges(min=lb,max=ub)
    solver.SetEvaluationLimits(maxiter,maxfun)
    solver.SetEvaluationMonitor(evalmon)
    solver.SetGenerationMonitor(stepmon)
    solver.Solve(cost,termination=VTR(tolerance),strategy=Best1Exp, \
                 CrossProbability=crossover,ScalingFactor=percent_change, \
                 constraints = _constraints) #XXX: parallel map?

    solved = solver.Solution()
    diameter_squared = solver.bestEnergy
    func_evals = len(evalmon)
    return solved, diameter_squared, func_evals

  # use optimization to get expectation value
  tolerance = (param[1])**2
  results = optimize(cost, (lower_bounds, upper_bounds), tolerance, constraints)

  # repack the results
  samples = _pack( _nested(results[0],npts) )
  return samples
 

def _impose_expected_moment(m, f, npts, bounds=None, weights=None, **kwds):
  """impose a given expected moment ``E`` on a given function *f*,
where ``E = m +/- tol`` and ``E = moment(f(x))`` for ``x`` in *bounds*

Args:
    m (float): target expected moment
    f (func): a function that takes a list and returns a number
    npts (tuple(int)): a tuple of dimensions of the target product measure
    bounds (tuple, default=None): tuple is ``(lower_bounds, upper_bounds)``
    weights (list, default=None):  a list of sample weights
    order (int, default=1): the degree, a positive integer
    tol (float, default=None): maximum allowable deviation from ``m``
    constraints (func, default=None): a function that takes a nested list of
        ``N x 1D`` discrete measure positions and weights, with the intended
        purpose of kernel-transforming ``x,w`` as ``x' = constraints(x, w)``
    npop (int, default=200): size of the trial solution population
    maxiter (int, default=1000): the maximum number of iterations to perform
    maxfun (int, default=1e+6): the maximum number of function evaluations

Returns:
    a list of sample positions, with expected moment ``E``

Notes:
    Expected moment ``E`` is calculated by minimizing ``moment(f(x)) - m``,
    over the given *bounds*, and will terminate when ``E`` is found within
    deviation ``tol`` of the target moment ``m``. If ``tol`` is not provided,
    then a relative deviation of 1% of ``m`` will be used.

    This function does not preserve the mean, variance, or range, as there
    is no initial list of samples to draw the mean, variance, and etc from

    *bounds* is tuple with ``length(bounds) == 2``, composed of all the lower
    bounds, then all the upper bounds, for each parameter

Examples:
    >>> # provide the dimensions and bounds
    >>> nx = 3;  ny = 2;  nz = 1
    >>> x_lb = [10.0];  y_lb = [0.0];  z_lb = [10.0]
    >>> x_ub = [50.0];  y_ub = [9.0];  z_ub = [90.0]
    >>> 
    >>> # prepare the bounds
    >>> lb = (nx * x_lb) + (ny * y_lb) + (nz * z_lb)
    >>> ub = (nx * x_ub) + (ny * y_ub) + (nz * z_ub)
    >>>
    >>> # generate a list of samples with moment +/- dev imposed
    >>> mom = 2.0;  dev = 0.01
    >>> samples = _impose_expected_moment(mom, f, (nx,ny,nz), (lb,ub), \
    ...                                            order=2, tol=dev)
    >>>
    >>> # test the results by calculating the expected moment for the samples
    >>> _expected_moment(f, samples, order=2)
    >>> 2.000010010122465
"""
  # param[0] is the target moment (i.e. mean, variance, ...)
  # param[1] is the acceptable deviation from the target moment
  # param[2] is the order of the moment (i.e. 2=variance, ...)
  tol = kwds['tol'] if 'tol' in kwds else None
  order = kwds['order'] if 'order' in kwds else 1
  param = (m, tol or 0.01*m, order)

  if param[-1] < 0:
     msg = 'order must be greater than zero (order = %s)' % param[-1]
     raise ValueError(msg)
  if param[-1] == 1 and not (param[1] >= -param[0] >= -param[1]):
     msg = 'if order == 1, then moment == 0 (moment = %s +/- %s)' % param[:-1]
     raise ValueError(msg)
  if param[-1] == 0 and not (param[1] >= 1-param[0] >= -param[1]):
     msg = 'if order == 0, then moment == 1 (moment %s +/- %s)' % param[:-1]
     raise ValueError(msg)

  # FIXME: the following is a HACK to recover from lost 'weights' information
  #        we 'mimic' discrete measures using the product measure weights
  # plug in the 'constraints' function:  samples' = constrain(samples, weights)
  constrain = None   # default is no constraints
  if 'constraints' in kwds: constrain = kwds['constraints']
  if not constrain:  # if None (default), there are no constraints
    constraints = lambda x: x
  else: #XXX: better to use a standard "xk' = constrain(xk)" interface ?
    def constraints(rv):
      coords = _pack( _nested(rv,npts) )
      coords = list(zip(*coords))              # 'mimic' a nested list
      coords = constrain(coords, [weights for i in range(len(coords))])
      coords = list(zip(*coords))              # revert back to a packed list
      return _flat( _unpack(coords,npts) )

  # construct cost function to reduce deviation from expected moment
  def cost(rv):
    """compute cost from a 1-d array of model parameters,
    where:  cost = | E_moment[model] - m |**2 """
    # from mystic.math.measures import _pack, _nested, _expected_moment
    samples = _pack( _nested(rv,npts) )
    Ex = _expected_moment(f, samples, weights, param[-1])
    return (Ex - param[0])**2

  # if bounds are not set, use the default optimizer bounds
  if not bounds:
    lower_bounds = []; upper_bounds = []
    for n in npts:
      lower_bounds += [None]*n
      upper_bounds += [None]*n
  else:
    lower_bounds, upper_bounds = bounds

  if len(lower_bounds) == 1 and param[-1] > 1 \
     and not (param[1] >= -param[0] >= -param[1]):
     msg = 'if size == 1, then moment == 0 (moment %s +/- %s)' % param[:-1]
     raise ValueError(msg)

  # construct and configure optimizer
  debug = kwds['debug'] if 'debug' in kwds else False
  npop = kwds.pop('npop', 200)
  maxiter = 1 if param[-1] <= 1 else 1000
  maxiter = kwds.pop('maxiter', maxiter)
  maxfun = kwds.pop('maxfun', 1e+6)
  crossover = 0.9; percent_change = 0.9

  def optimize(cost, bounds, tolerance, _constraints):
    (lb,ub) = bounds
    from mystic.solvers import DifferentialEvolutionSolver2
    from mystic.termination import VTR
    from mystic.strategy import Best1Exp
    from mystic.monitors import VerboseMonitor, Monitor
    from mystic.tools import random_seed
    if debug: random_seed(123)
    evalmon = Monitor();  stepmon = Monitor()
    if debug: stepmon = VerboseMonitor(10)

    ndim = len(lb)
    solver = DifferentialEvolutionSolver2(ndim,npop)
    solver.SetRandomInitialPoints(min=lb,max=ub)
    solver.SetStrictRanges(min=lb,max=ub)
    solver.SetEvaluationLimits(maxiter,maxfun)
    solver.SetEvaluationMonitor(evalmon)
    solver.SetGenerationMonitor(stepmon)
    solver.Solve(cost,termination=VTR(tolerance),strategy=Best1Exp, \
                 CrossProbability=crossover,ScalingFactor=percent_change, \
                 constraints = _constraints) #XXX: parallel map?

    solved = solver.Solution()
    diameter_squared = solver.bestEnergy
    func_evals = len(evalmon)
    return solved, diameter_squared, func_evals

  # use optimization to get expected moment
  tolerance = (param[1])**2
  results = optimize(cost, (lower_bounds, upper_bounds), tolerance, constraints)

  # repack the results
  samples = _pack( _nested(results[0],npts) )
  return samples


def impose_expected_variance(v, f, npts, bounds=None, weights=None, **kwds):
  """impose a given expected variance ``E`` on a given function *f*,
where ``E = v +/- tol`` and ``E = variance(f(x))`` for ``x`` in *bounds*

Args:
    v (float): target expected variance
    f (func): a function that takes a list and returns a number
    npts (tuple(int)): a tuple of dimensions of the target product measure
    bounds (tuple, default=None): tuple is ``(lower_bounds, upper_bounds)``
    weights (list, default=None):  a list of sample weights
    tol (float, default=None): maximum allowable deviation from ``v``
    constraints (func, default=None): a function that takes a nested list of
        ``N x 1D`` discrete measure positions and weights, with the intended
        purpose of kernel-transforming ``x,w`` as ``x' = constraints(x, w)``
    npop (int, default=200): size of the trial solution population
    maxiter (int, default=1000): the maximum number of iterations to perform
    maxfun (int, default=1e+6): the maximum number of function evaluations

Returns:
    a list of sample positions, with expected variance ``E``

Notes:
    Expected variance ``E`` is calculated by minimizing ``variance(f(x)) - v``,
    over the given *bounds*, and will terminate when ``E`` is found within
    deviation ``tol`` of the target variance ``v``. If ``tol`` is not provided,
    then a relative deviation of 1% of ``v`` will be used.

    This function does not preserve the mean, variance, or range, as there
    is no initial list of samples to draw the mean, variance, and etc from

    *bounds* is tuple with ``length(bounds) == 2``, composed of all the lower
    bounds, then all the upper bounds, for each parameter

Examples:
    >>> # provide the dimensions and bounds
    >>> nx = 3;  ny = 2;  nz = 1
    >>> x_lb = [10.0];  y_lb = [0.0];  z_lb = [10.0]
    >>> x_ub = [50.0];  y_ub = [9.0];  z_ub = [90.0]
    >>> 
    >>> # prepare the bounds
    >>> lb = (nx * x_lb) + (ny * y_lb) + (nz * z_lb)
    >>> ub = (nx * x_ub) + (ny * y_ub) + (nz * z_ub)
    >>>
    >>> # generate a list of samples with variance +/- dev imposed
    >>> var = 2.0;  dev = 0.01
    >>> samples = impose_expected_variance(var, f, (nx,ny,nz), (lb,ub), tol=dev)
    >>>
    >>> # test the results by calculating the expected variance for the samples
    >>> expected_variance(f, samples)
    >>> 2.000010010122465
"""
  # param[0] is the target variance
  # param[1] is the acceptable deviation from the target variance
  tol = kwds['tol'] if 'tol' in kwds else None
  param = (v, tol or 0.01*v)

  # FIXME: the following is a HACK to recover from lost 'weights' information
  #        we 'mimic' discrete measures using the product measure weights
  # plug in the 'constraints' function:  samples' = constrain(samples, weights)
  constrain = None   # default is no constraints
  if 'constraints' in kwds: constrain = kwds['constraints']
  if not constrain:  # if None (default), there are no constraints
    constraints = lambda x: x
  else: #XXX: better to use a standard "xk' = constrain(xk)" interface ?
    def constraints(rv):
      coords = _pack( _nested(rv,npts) )
      coords = list(zip(*coords))              # 'mimic' a nested list
      coords = constrain(coords, [weights for i in range(len(coords))])
      coords = list(zip(*coords))              # revert back to a packed list
      return _flat( _unpack(coords,npts) )

  # construct cost function to reduce deviation from expected variance
  def cost(rv):
    """compute cost from a 1-d array of model parameters,
    where:  cost = | E_var[model] - m |**2 """
    # from mystic.math.measures import _pack, _nested, expected_variance
    samples = _pack( _nested(rv,npts) )
    Ex = expected_variance(f, samples, weights)
    return (Ex - param[0])**2

  # if bounds are not set, use the default optimizer bounds
  if not bounds:
    lower_bounds = []; upper_bounds = []
    for n in npts:
      lower_bounds += [None]*n
      upper_bounds += [None]*n
  else:
    lower_bounds, upper_bounds = bounds

  if len(lower_bounds) == 1 and not (param[1] >= -param[0] >= -param[1]):
     msg = 'if size == 1, then variance == 0 (variance %s +/- %s)' % param
     raise ValueError(msg)

  # construct and configure optimizer
  debug = kwds['debug'] if 'debug' in kwds else False
  npop = kwds.pop('npop', 200)
  maxiter = kwds.pop('maxiter', 1000)
  maxfun = kwds.pop('maxfun', 1e+6)
  crossover = 0.9; percent_change = 0.9

  def optimize(cost, bounds, tolerance, _constraints):
    (lb,ub) = bounds
    from mystic.solvers import DifferentialEvolutionSolver2
    from mystic.termination import VTR
    from mystic.strategy import Best1Exp
    from mystic.monitors import VerboseMonitor, Monitor
    from mystic.tools import random_seed
    if debug: random_seed(123)
    evalmon = Monitor();  stepmon = Monitor()
    if debug: stepmon = VerboseMonitor(10)

    ndim = len(lb)
    solver = DifferentialEvolutionSolver2(ndim,npop)
    solver.SetRandomInitialPoints(min=lb,max=ub)
    solver.SetStrictRanges(min=lb,max=ub)
    solver.SetEvaluationLimits(maxiter,maxfun)
    solver.SetEvaluationMonitor(evalmon)
    solver.SetGenerationMonitor(stepmon)
    solver.Solve(cost,termination=VTR(tolerance),strategy=Best1Exp, \
                 CrossProbability=crossover,ScalingFactor=percent_change, \
                 constraints = _constraints) #XXX: parallel map?

    solved = solver.Solution()
    diameter_squared = solver.bestEnergy
    func_evals = len(evalmon)
    return solved, diameter_squared, func_evals

  # use optimization to get expected_variance
  tolerance = (param[1])**2
  results = optimize(cost, (lower_bounds, upper_bounds), tolerance, constraints)

  # repack the results
  samples = _pack( _nested(results[0],npts) )
  return samples


def impose_expected_std(s, f, npts, bounds=None, weights=None, **kwds):
  """impose a given expected std ``E`` on a given function *f*,
where ``E = s +/- tol`` and ``E = std(f(x))`` for ``x`` in *bounds*

Args:
    s (float): target expected standard deviation
    f (func): a function that takes a list and returns a number
    npts (tuple(int)): a tuple of dimensions of the target product measure
    bounds (tuple, default=None): tuple is ``(lower_bounds, upper_bounds)``
    weights (list, default=None):  a list of sample weights
    tol (float, default=None): maximum allowable deviation from ``s``
    constraints (func, default=None): a function that takes a nested list of
        ``N x 1D`` discrete measure positions and weights, with the intended
        purpose of kernel-transforming ``x,w`` as ``x' = constraints(x, w)``
    npop (int, default=200): size of the trial solution population
    maxiter (int, default=1000): the maximum number of iterations to perform
    maxfun (int, default=1e+6): the maximum number of function evaluations

Returns:
    a list of sample positions, with expected standard deviation ``E``

Notes:
    Expected std ``E`` is calculated by minimizing ``std(f(x)) - s``,
    over the given *bounds*, and will terminate when ``E`` is found within
    deviation ``tol`` of the target std ``s``. If ``tol`` is not provided,
    then a relative deviation of 1% of ``s`` will be used.

    This function does not preserve the mean, variance, or range, as there
    is no initial list of samples to draw the mean, variance, and etc from

    *bounds* is tuple with ``length(bounds) == 2``, composed of all the lower
    bounds, then all the upper bounds, for each parameter

Examples:
    >>> # provide the dimensions and bounds
    >>> nx = 3;  ny = 2;  nz = 1
    >>> x_lb = [10.0];  y_lb = [0.0];  z_lb = [10.0]
    >>> x_ub = [50.0];  y_ub = [9.0];  z_ub = [90.0]
    >>> 
    >>> # prepare the bounds
    >>> lb = (nx * x_lb) + (ny * y_lb) + (nz * z_lb)
    >>> ub = (nx * x_ub) + (ny * y_ub) + (nz * z_ub)
    >>>
    >>> # generate a list of samples with std +/- dev imposed
    >>> std = 2.0;  dev = 0.01
    >>> samples = impose_expected_std(std, f, (nx,ny,nz), (lb,ub), tol=dev)
    >>>
    >>> # test the results by calculating the expected std for the samples
    >>> expected_std(f, samples)
    >>> 2.000010010122465
"""
  # param[0] is the target standard deviation
  # param[1] is the acceptable deviation from the target standard deviation
  tol = kwds['tol'] if 'tol' in kwds else None
  param = (s, tol or 0.01*s)

  # FIXME: the following is a HACK to recover from lost 'weights' information
  #        we 'mimic' discrete measures using the product measure weights
  # plug in the 'constraints' function:  samples' = constrain(samples, weights)
  constrain = None   # default is no constraints
  if 'constraints' in kwds: constrain = kwds['constraints']
  if not constrain:  # if None (default), there are no constraints
    constraints = lambda x: x
  else: #XXX: better to use a standard "xk' = constrain(xk)" interface ?
    def constraints(rv):
      coords = _pack( _nested(rv,npts) )
      coords = list(zip(*coords))              # 'mimic' a nested list
      coords = constrain(coords, [weights for i in range(len(coords))])
      coords = list(zip(*coords))              # revert back to a packed list
      return _flat( _unpack(coords,npts) )

  # construct cost function to reduce deviation from expected std
  def cost(rv):
    """compute cost from a 1-d array of model parameters,
    where:  cost = | E_std[model] - m |**2 """
    # from mystic.math.measures import _pack, _nested, expected_std
    samples = _pack( _nested(rv,npts) )
    Ex = expected_std(f, samples, weights)
    return (Ex - param[0])**2

  # if bounds are not set, use the default optimizer bounds
  if not bounds:
    lower_bounds = []; upper_bounds = []
    for n in npts:
      lower_bounds += [None]*n
      upper_bounds += [None]*n
  else:
    lower_bounds, upper_bounds = bounds

  if len(lower_bounds) == 1 and not (param[1] >= -param[0] >= -param[1]):
     msg = 'if size == 1, then std == 0 (std %s +/- %s)' % param
     raise ValueError(msg)

  # construct and configure optimizer
  debug = kwds['debug'] if 'debug' in kwds else False
  npop = kwds.pop('npop', 200)
  maxiter = kwds.pop('maxiter', 1000)
  maxfun = kwds.pop('maxfun', 1e+6)
  crossover = 0.9; percent_change = 0.9

  def optimize(cost, bounds, tolerance, _constraints):
    (lb,ub) = bounds
    from mystic.solvers import DifferentialEvolutionSolver2
    from mystic.termination import VTR
    from mystic.strategy import Best1Exp
    from mystic.monitors import VerboseMonitor, Monitor
    from mystic.tools import random_seed
    if debug: random_seed(123)
    evalmon = Monitor();  stepmon = Monitor()
    if debug: stepmon = VerboseMonitor(10)

    ndim = len(lb)
    solver = DifferentialEvolutionSolver2(ndim,npop)
    from mystic.termination import VTR
    from mystic.strategy import Best1Exp
    from mystic.monitors import VerboseMonitor, Monitor
    from mystic.tools import random_seed
    if debug: random_seed(123)
    evalmon = Monitor();  stepmon = Monitor()
    if debug: stepmon = VerboseMonitor(10)

    ndim = len(lb)
    solver = DifferentialEvolutionSolver2(ndim,npop)
    solver.SetRandomInitialPoints(min=lb,max=ub)
    solver.SetStrictRanges(min=lb,max=ub)
    solver.SetEvaluationLimits(maxiter,maxfun)
    solver.SetEvaluationMonitor(evalmon)
    solver.SetGenerationMonitor(stepmon)
    solver.Solve(cost,termination=VTR(tolerance),strategy=Best1Exp, \
                 CrossProbability=crossover,ScalingFactor=percent_change, \
                 constraints = _constraints) #XXX: parallel map?

    solved = solver.Solution()
    diameter_squared = solver.bestEnergy
    func_evals = len(evalmon)
    return solved, diameter_squared, func_evals

  # use optimization to get expected std
  tolerance = (param[1])**2
  results = optimize(cost, (lower_bounds, upper_bounds), tolerance, constraints)

  # repack the results
  samples = _pack( _nested(results[0],npts) )
  return samples


#XXX: better ...variance(param, f, ...) or ...variance(m,v, f, ...) ?
def impose_expected_mean_and_variance(param, f, npts, bounds=None, weights=None, **kwds):
  """impose a given expected mean ``E`` on a given function *f*,
where ``E = m +/- tol`` and ``E = mean(f(x))`` for ``x`` in *bounds*.
Additionally, impose a given expected variance ``R`` on *f*,
where ``R = v +/- tol`` and ``R = variance(f(x))`` for ``x`` in *bounds*.

Args:
    param (tuple(float)): target parameters, ``(mean, variance)``
    f (func): a function that takes a list and returns a number
    npts (tuple(int)): a tuple of dimensions of the target product measure
    bounds (tuple, default=None): tuple is ``(lower_bounds, upper_bounds)``
    weights (list, default=None):  a list of sample weights
    tol (float, default=None): maximum allowable deviation from ``m`` and ``v``
    constraints (func, default=None): a function that takes a nested list of
        ``N x 1D`` discrete measure positions and weights, with the intended
        purpose of kernel-transforming ``x,w`` as ``x' = constraints(x, w)``
    npop (int, default=200): size of the trial solution population
    maxiter (int, default=1000): the maximum number of iterations to perform
    maxfun (int, default=1e+6): the maximum number of function evaluations

Returns:
    a list of sample positions, with expected mean ``E`` and variance ``R``

Notes:
    Expected mean ``E`` and expected variance ``R`` are calculated by
    minimizing the sum of the absolute values of ``mean(f(x)) - m`` and
    ``variance(f(x)) - v`` over the given *bounds*, and will terminate when
    ``E`` and ``R`` are found within tolerance ``tol`` of the target mean ``m``
    and variance ``v``, respectively. If ``tol`` is not provided, then a
    relative deviation of 1% of ``max(m,v)`` will be used.

    This function does not preserve the mean, variance, or range, as there
    is no initial list of samples to draw the mean, variance, and etc from

    *bounds* is tuple with ``length(bounds) == 2``, composed of all the lower
    bounds, then all the upper bounds, for each parameter

Examples:
    >>> # provide the dimensions and bounds
    >>> nx = 3;  ny = 2;  nz = 1
    >>> x_lb = [10.0];  y_lb = [0.0];  z_lb = [10.0]
    >>> x_ub = [50.0];  y_ub = [9.0];  z_ub = [90.0]
    >>> 
    >>> # prepare the bounds
    >>> lb = (nx * x_lb) + (ny * y_lb) + (nz * z_lb)
    >>> ub = (nx * x_ub) + (ny * y_ub) + (nz * z_ub)
    >>>
    >>> # generate a list of samples with mean and variance imposed
    >>> mean = 5.0;  var = 2.0;  tol = 0.01
    >>> samples = impose_expected_mean_and_variance((mean,var), f, (nx,ny,nz), \
    ...                                             (lb,ub), tol=tol)
    >>>
    >>> # test the results by calculating the expected mean for the samples
    >>> expected_mean(f, samples)
    >>> 
    >>> # test the results by calculating the expected variance for the samples
    >>> expected_variance(f, samples)
    >>> 2.000010010122465
"""
  # param[0] is the target mean
  # param[1] is the target variance
  # param[2] is the acceptable deviation from the target mean and variance
  tol = kwds['tol'] if 'tol' in kwds else None
  param = tuple(param) + (tol or 0.01*max(param),)

  # FIXME: the following is a HACK to recover from lost 'weights' information
  #        we 'mimic' discrete measures using the product measure weights
  # plug in the 'constraints' function:  samples' = constrain(samples, weights)
  constrain = None   # default is no constraints
  if 'constraints' in kwds: constrain = kwds['constraints']
  if not constrain:  # if None (default), there are no constraints
    constraints = lambda x: x
  else: #XXX: better to use a standard "xk' = constrain(xk)" interface ?
    def constraints(rv):
      coords = _pack( _nested(rv,npts) )
      coords = list(zip(*coords))              # 'mimic' a nested list
      coords = constrain(coords, [weights for i in range(len(coords))])
      coords = list(zip(*coords))              # revert back to a packed list
      return _flat( _unpack(coords,npts) )

  # construct cost function to reduce deviation from expected mean and variance
  def cost(rv):
    """compute cost from a 1-d array of model parameters,
    where:  cost = | E[model] - m |**2 + | E_var[model] - n |**2 """
    # from mystic.math.measures import _pack, _nested
    # from mystic.math.measures import expectation, expected_variance
    samples = _pack( _nested(rv,npts) )
    Em = expectation(f, samples, weights)
    Ev = expected_variance(f, samples, weights)
    return (Em - param[0])**2 + (Ev - param[1])**2

  # if bounds are not set, use the default optimizer bounds
  if not bounds:
    lower_bounds = []; upper_bounds = []
    for n in npts:
      lower_bounds += [None]*n
      upper_bounds += [None]*n
  else:
    lower_bounds, upper_bounds = bounds

  if len(lower_bounds) == 1 and not (param[2] >= -param[1] >= -param[2]):
     msg = 'if size == 1, then variance == 0 (variance %s +/- %s)' % param[1:]
     raise ValueError(msg)

  # construct and configure optimizer
  debug = kwds['debug'] if 'debug' in kwds else False
  npop = kwds.pop('npop', 200)
  maxiter = kwds.pop('maxiter', 1000)
  maxfun = kwds.pop('maxfun', 1e+6)
  crossover = 0.9; percent_change = 0.9

  def optimize(cost, bounds, tolerance, _constraints):
    (lb,ub) = bounds
    from mystic.solvers import DifferentialEvolutionSolver2
    from mystic.termination import VTR
    from mystic.strategy import Best1Exp
    from mystic.monitors import VerboseMonitor, Monitor
    from mystic.tools import random_seed
    if debug: random_seed(123)
    evalmon = Monitor();  stepmon = Monitor()
    if debug: stepmon = VerboseMonitor(10)

    ndim = len(lb)
    solver = DifferentialEvolutionSolver2(ndim,npop)
    solver.SetRandomInitialPoints(min=lb,max=ub)
    solver.SetStrictRanges(min=lb,max=ub)
    solver.SetEvaluationLimits(maxiter,maxfun)
    solver.SetEvaluationMonitor(evalmon)
    solver.SetGenerationMonitor(stepmon)
    solver.Solve(cost,termination=VTR(tolerance),strategy=Best1Exp, \
                 CrossProbability=crossover,ScalingFactor=percent_change, \
                 constraints = _constraints) #XXX: parallel map?

    solved = solver.Solution()
    diameter_squared = solver.bestEnergy
    func_evals = len(evalmon)
    return solved, diameter_squared, func_evals

  # use optimization to get expected mean and variance
  tolerance = (param[-1])**2 #XXX: correct? or (?*D)**2
  results = optimize(cost, (lower_bounds, upper_bounds), tolerance, constraints)

  # repack the results
  samples = _pack( _nested(results[0],npts) )
  return samples


##### weight shift methods #####
def impose_weight_norm(samples, weights, mass=1.0):
  """normalize the weights for a list of (weighted) points
  (this function is 'mean-preserving')

Inputs:
    samples -- a list of sample points
    weights -- a list of sample weights
    mass -- float target of normalized weights
"""
  m = mean(samples, weights)
  wts = normalize(weights,mass) #NOTE: not mean-preserving, until next line
  return impose_mean(m, samples, wts), wts


def normalize(weights, mass='l2', zsum=False, zmass=1.0):
  """normalize a list of points (e.g. normalize to 1.0)

Inputs:
    weights -- a list of sample weights
    mass -- float target of normalized weights (or string for Ln norm)
    zsum -- use counterbalance when mass = 0.0
    zmass -- member scaling when mass = 0.0

Notes: if mass='l1', will use L1-norm; if mass='l2' will use L2-norm; etc.
"""
  try:
    mass = int(mass.lstrip('l'))
    fixed = False
  except AttributeError:
    fixed = True
  weights = asarray(list(weights)) #XXX: faster to use x = array(x, copy=True) ?

  if fixed:
    w = sum(abs(weights))
  else:
    mass = int(min(200, mass)) # x**200 is ~ x**inf
    w = Lnorm(weights,mass)
    mass = 1.0

  if not w:
    if not zsum: return list(weights * 0.0)
    from numpy import inf, nan
    weights[weights == 0.0] = nan
    return list(weights * inf)  # protect against ZeroDivision

  if float(mass) or not zsum:
    w = weights / w #FIXME: not "mean-preserving"
    if not fixed: return list(w) # <- scaled so sum(abs(x)) = 1
    #REMAINING ARE fixed mean
    m = sum(w)
    w = mass * w
    if not m:  #XXX: do similar to zsum (i.e. shift) when sum(weights)==0 ?
      if not zsum: return list(weights * 0.0)
      from numpy import inf, nan
      weights[weights == 0.0] = nan
      return list(weights * inf)  # protect against ZeroDivision
    return list(w/m) # <- scaled so sum(x) = 1

  # force selected member to satisfy sum = 0.0
  zsum = -1
  weights[zsum] = -(sum(weights) - weights[zsum])
  mass = zmass
  return list(mass * weights / w)  #FIXME: not "mean-preserving"


def impose_reweighted_mean(m, samples, weights=None, solver=None):
    """impose a mean on a list of points by reweighting weights"""
    ndim = len(samples)
    if weights is None:
        weights = [1.0/ndim] * ndim
    if solver is None or solver == 'fmin':
        from mystic.solvers import fmin as solver
    elif solver == 'fmin_powell':
        from mystic.solvers import fmin_powell as solver
    elif solver == 'diffev':
        from mystic.solvers import diffev as solver
    elif solver == 'diffev2':
        from mystic.solvers import diffev2 as solver
    norm = sum(weights)

    inequality = ""; equality = ""; equality2 = ""
    for i in range(ndim):
        inequality += "x%s >= 0.0\n" % (i) # positive
        equality += "x%s + " % (i)         # normalized
        equality2 += "%s * x%s + " % (float(samples[i]),(i)) # mean

    equality += "0.0 = %s\n" % float(norm)
    equality += equality2 + "0.0 = %s*%s\n" % (float(norm),m)

    penalties = generate_penalty(generate_conditions(inequality))
    constrain = generate_constraint(generate_solvers(solve(equality)))

    def cost(x): return sum(x)

    results = solver(cost, weights, constraints=constrain, \
                     penalty=penalties, disp=False, full_output=True)
    wts = list(results[0])
    _norm = results[1] # should have _norm == norm
    warn = results[4]  # nonzero if didn't converge

    #XXX: better to fail immediately if xlo < m < xhi... or the below?
    if warn or not almostEqual(_norm, norm):
        print("Warning: could not impose mean through reweighting")
        return None #impose_mean(m, samples, weights), weights

    return wts #samples, wts


def impose_reweighted_variance(v, samples, weights=None, solver=None):
    """impose a variance on a list of points by reweighting weights"""
    ndim = len(samples)
    if weights is None:
        weights = [1.0/ndim] * ndim
    if solver is None or solver == 'fmin':
        from mystic.solvers import fmin as solver
    elif solver == 'fmin_powell':
        from mystic.solvers import fmin_powell as solver
    elif solver == 'diffev':
        from mystic.solvers import diffev as solver
    elif solver == 'diffev2':
        from mystic.solvers import diffev2 as solver
    norm = sum(weights)
    m = mean(samples, weights)

    inequality = ""
    equality = ""; equality2 = ""; equality3 = ""
    for i in range(ndim):
        inequality += "x%s >= 0.0\n" % (i) # positive
        equality += "x%s + " % (i)         # normalized
        equality2 += "%s * x%s + " % (float(samples[i]),(i)) # mean
        equality3 += "x%s*(%s-%s)**2 + " % ((i),float(samples[i]),m) # var

    equality += "0.0 = %s\n" % float(norm)
    equality += equality2 + "0.0 = %s*%s\n" % (float(norm),m)
    equality += equality3 + "0.0 = %s*%s\n" % (float(norm),v)

    penalties = generate_penalty(generate_conditions(inequality))
    constrain = generate_constraint(generate_solvers(solve(equality)))

    def cost(x): return sum(x)

    results = solver(cost, weights, constraints=constrain, \
                     penalty=penalties, disp=False, full_output=True)
    wts = list(results[0])
    _norm = results[1] # should have _norm == norm
    warn = results[4]  # nonzero if didn't converge

    #XXX: better to fail immediately if xlo < m < xhi... or the below?
    if warn or not almostEqual(_norm, norm):
        print("Warning: could not impose mean through reweighting")
        return None #impose_variance(v, samples, weights), weights

    return wts #samples, wts  # "mean-preserving"

def impose_reweighted_std(s, samples, weights=None, solver=None):
    """impose a standard deviation on a list of points by reweighting weights"""
    return impose_reweighted_variance(s**2, samples, weights, solver)

##### sampling statistics methods #####
def _sort(samples, weights=None):
    "sort (weighted) samples; returns 2-D array of samples and weights"
    import numpy as np
    if weights is None:
        x = np.ones((2,len(samples))) #XXX: casts to a float
        x[0] = np.sort(samples)
        return x
    x = np.vstack([samples,weights]).T
    return x[x[:,0].argsort()].T


def median(samples, weights=None):
    """calculate the (weighted) median for a list of points

Inputs:
    samples -- a list of sample points
    weights -- a list of sample weights
"""
    import numpy as np
    x,w = _sort(samples,weights)
    s = sum(w)
    return np.mean(x[s/2. - np.cumsum(w) <= 0][0:2-x.size%2])


def mad(samples, weights=None): #, scale=1.4826):
    """calculate the (weighted) median absolute deviation for a list of points

Inputs:
    samples -- a list of sample points
    weights -- a list of sample weights
"""
    s = asarray(samples)
    return median(abs(s - median(samples,weights)),weights) # * scale


#XXX: impose median by ensuring equal # samples < & >... (and not shifting) ?
def impose_median(m, samples, weights=None):
    """impose a median on a list of (weighted) points
    (this function is 'range-preserving' and 'mad-preserving')

Inputs:
    m -- the target median
    samples -- a list of sample points
    weights -- a list of sample weights
"""
    s = asarray(samples)
    return (s + (m - median(samples, weights))).tolist()


def impose_mad(s, samples, weights=None):
    """impose a median absolute deviation on a list of (weighted) points
    (this function is 'median-preserving')

Inputs:
    s -- the target median absolute deviation
    samples -- a list of sample points
    weights -- a list of sample weights
"""
    import numpy as np
    m = median(samples, weights)
    samples = np.asarray(list(samples))
    _mad = mad(samples,weights)
    if not _mad: # protect against ZeroDivision when mad = 0
        return [np.nan]*len(samples)
    scale = float(s) / _mad
    samples = samples * scale #NOTE: not "median-preserving" until next line
    return impose_median(m, samples, weights) #NOTE: not "range-preserving"


def _k(weights, k=0, clip=False, norm=False, eps=15): #XXX: better 9 ?
    "trim weights at k%; if clip is True, winsorize instead of trim"
    #NOTE: eps is tolerance for cancellation of similar values
    import numpy as np
    try:
        klo,khi = k
    except TypeError:
        klo = khi = k
    if klo + khi > 100:
        msg = "cannot crop '%s + %s > 100' percent" % (klo,khi)
        raise ValueError(msg)
    elif klo < 0 or khi < 0:
        msg = "cannot crop negative percent '%s + %s'" % (klo,khi)
        raise ValueError(msg)
    else:
        klo,khi = .01*klo,.01*khi
    w = np.array(weights, dtype=float)/sum(weights)  #XXX: no dtype?
    w_lo, w_hi = np.cumsum(w), np.cumsum(w[::-1])
    # calculate the cropped indices
    lo = len(w) - sum((w_lo - klo).round(eps) > 0)
    hi = sum((w_hi - khi).round(eps) > 0) - 1
    # flip indices if flipped
    if lo > hi:
        lo,hi = hi,lo
    if not clip:
        # find the values at k%
        w_lo = w_lo[lo]
        w_hi = w_hi[len(w)-1-hi]
        # reset the values at k%
        if klo + khi == 1:
            w[lo] = w[hi] = 0
        elif lo == hi:
            w[lo] = max(1 - khi - klo,0)
        else: 
            w[lo] = max(w_lo - klo,0)
            w[hi] = max(w_hi - khi,0)
    else:
        # reset the values at k%
        if lo == hi:
            w[lo] = sum(w)
        else: 
            w[lo] += sum(w[:lo])
            w[hi] += sum(w[hi+1:])
    # trim the remaining weights
    w[:lo] = 0
    w[hi+1:] = 0
    if not norm: w *= sum(weights)
    return w.tolist()


def tmean(samples, weights=None, k=0, clip=False):
    """calculate the (weighted) trimmed mean for a list of points

Inputs:
    samples -- a list of sample points
    weights -- a list of sample weights
    k -- percent samples to trim (k%) [tuple (lo,hi) or float if lo=hi]
    clip -- if True, winsorize instead of trimming k% of samples

NOTE: if all samples are excluded, will return nan
"""
    samples,weights = _sort(samples,weights)
    weights = _k(weights,k,clip)
    return sum(samples * weights)/sum(weights)


def tvariance(samples, weights=None, k=0, clip=False):
    """calculate the (weighted) trimmed variance for a list of points

Inputs:
    samples -- a list of sample points
    weights -- a list of sample weights
    k -- percent samples to trim (k%) [tuple (lo,hi) or float if lo=hi]
    clip -- if True, winsorize instead of trimming k% of samples

NOTE: if all samples are excluded, will return nan
"""
    samples,weights = _sort(samples,weights)
    weights = _k(weights,k,clip)
    trim_mean = sum(samples * weights)/sum(weights)
    return mean(abs(samples - trim_mean)**2, weights) #XXX: correct ?


def tstd(samples, weights=None, k=0, clip=False):
    """calculate the (weighted) trimmed standard deviation for a list of points

Inputs:
    samples -- a list of sample points
    weights -- a list of sample weights
    k -- percent samples to trim (k%) [tuple (lo,hi) or float if lo=hi]
    clip -- if True, winsorize instead of trimming k% of samples

NOTE: if all samples are excluded, will return nan
"""
    import numpy as np
    return np.sqrt(tvariance(samples, weights, k, clip))


#XXX: use reweighting to impose tmean, tvariance, tstd, median, & mad ?
def impose_tmean(m, samples, weights=None, k=0, clip=False):
    """impose a trimmed mean (at k%) on a list of (weighted) points
    (this function is 'range-preserving' and 'tvariance-preserving')

Inputs:
    m -- the target trimmed mean
    samples -- a list of sample points
    weights -- a list of sample weights
    k -- percent samples to be trimmed (k%) [tuple (lo,hi) or float if lo=hi]
    clip -- if True, winsorize instead of trimming k% of samples
"""
    s = asarray(samples)
    return (s + (m - tmean(samples, weights, k=k, clip=clip))).tolist()


def impose_tvariance(v, samples, weights=None, k=0, clip=False):
    """impose a trimmed variance (at k%) on a list of (weighted) points
    (this function is 'tmean-preserving')

Inputs:
    v -- the target trimmed variance
    samples -- a list of sample points
    weights -- a list of sample weights
    k -- percent samples to be trimmed (k%) [tuple (lo,hi) or float if lo=hi]
    clip -- if True, winsorize instead of trimming k% of samples
"""
    import numpy as np
    m = tmean(samples, weights, k=k, clip=clip)
    samples = np.asarray(list(samples))

    tvar = tvariance(samples,weights,k=k,clip=clip)
    if not tvar: # protect against ZeroDivision when tvar = 0
        return [np.nan]*len(samples) #XXX: k?
    scale = np.sqrt(float(v) / tvar)
    samples = samples * scale #NOTE: not "tmean-preserving" until next line
    return impose_tmean(m, samples, weights, k=k, clip=clip) #NOTE: not "range-preserving"


def impose_tstd(s, samples, weights=None, k=0, clip=False):
    """impose a trimmed std (at k%) on a list of (weighted) points
    (this function is 'tmean-preserving')

Inputs:
    s -- the target trimmed standard deviation
    samples -- a list of sample points
    weights -- a list of sample weights
    k -- percent samples to be trimmed (k%) [tuple (lo,hi) or float if lo=hi]
    clip -- if True, winsorize instead of trimming k% of samples
"""
    return impose_tvariance(s**2, samples, weights, k=k, clip=clip)


##### collapse methods #####
#FIXME: add collapse weight/position methods to math.discrete.measure?
def impose_support(index, samples, weights): #XXX: toggle norm_preserving?
    """set all weights not appearing in 'index' to zero

Inputs:
    samples -- a list of sample points
    weights -- a list of sample weights
    index -- a list of desired support indices (weights will be non-zero)

For example:
    >>> impose_support([0,1],[1,2,3,4,5],[.2,.2,.2,.2,.2])
    ([2.5, 3.5, 4.5, 5.5, 6.5], [0.5, 0.5, 0.0, 0.0, 0.0])
    >>> impose_support([0,1,2,3],[1,2,3,4,5],[.2,.2,.2,.2,.2])
    ([1.5, 2.5, 3.5, 4.5, 5.5], [0.25, 0.25, 0.25, 0.25, 0.0])
    >>> impose_support([4],[1,2,3,4,5],[.2,.2,.2,.2,.2])
    ([-1.0, 0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0, 1.0])

Notes: is 'mean-preserving' for samples and 'norm-preserving' for weights
"""
    if index is None: index = range(len(weights))
    # allow negative indexing
    index = set(len(weights)+i if i<0 else i for i in index)
    m = mean(samples, weights)
    n = sum(weights)
    weights = [w if i in index else 0. for (i,w) in enumerate(weights)]
    weights = normalize(weights, n)
    return impose_mean(m, samples, weights), weights


#XXX: alternate to the above
def impose_unweighted(index, samples, weights, nullable=True):
    """set all weights appearing in 'index' to zero

Inputs:
    samples -- a list of sample points
    weights -- a list of sample weights
    index -- a list of indices where weight is to be zero
    nullable -- if False, avoid null weights by reweighting non-index weights

For example:
    >>> impose_unweighted([0,1,2],[1,2,3,4,5],[.2,.2,.2,.2,.2])
    ([-0.5, 0.5, 1.5, 2.5, 3.5], [0.0, 0.0, 0.0, 0.5, 0.5])
    >>> impose_unweighted([3,4],[1,2,3,4,5],[.2,.2,.2,.2,.2])
    ([2.0, 3.0, 4.0, 5.0, 6.0], [0.33333333333333331, 0.33333333333333331, 0.33333333333333331, 0.0, 0.0])

Notes: is 'mean-preserving' for samples and 'norm-preserving' for weights
"""
    if index is None: index = ()
    # allow negative indexing
    index = set(len(weights)+i if i<0 else i for i in index)
    m = mean(samples, weights)
    n = sum(weights)
    _weights = [0. if i in index else w for (i,w) in enumerate(weights)]
    if not nullable and not sum(_weights):
        _weights = [0. if i in index else 1. for (i,w) in enumerate(weights)]
    weights = normalize(_weights, n)
    return impose_mean(m, samples, weights), weights


def impose_collapse(pairs, samples, weights):
    """collapse the weight and position of each pair (i,j) in pairs

Collapse is defined as weight[j] += weight[i] and weights[i] = 0,
with samples[j] = samples[i].

Inputs:
    samples -- a list of sample points
    weights -- a list of sample weights
    pairs -- set of tuples of indices (i,j) where collapse occurs

For example:
    >>> impose_collapse({(0,1),(0,2)},[1,2,3,4,5],[.2,.2,.2,.2,.2])
    ([1.5999999999999996, 1.5999999999999996, 1.5999999999999996, 4.5999999999999996, 5.5999999999999996], [0.6000000000000001, 0.0, 0.0, 0.2, 0.2])
    >>> impose_collapse({(0,1),(3,4)},[1,2,3,4,5],[.2,.2,.2,.2,.2])
    ([1.3999999999999999, 1.3999999999999999, 3.3999999999999999, 4.4000000000000004, 4.4000000000000004], [0.4, 0.0, 0.2, 0.4, 0.0])

Notes: is 'mean-preserving' for samples and 'norm-preserving' for weights
"""
    samples, weights = list(samples), list(weights) # don't edit inputs
    m = mean(samples, weights)
    # allow negative indexing
    pairs = zip(*tuple(tuple(len(weights)+i if i<0 else i for i in j) for j in zip(*pairs)))
    #XXX: any vectorized way to do this?
    from mystic.tools import connected
    pairs = connected(pairs)
    for i,j in getattr(pairs,'iteritems',pairs.items)():
        v = weights[i]
        for k in j:
            v += weights[k]
            weights[k] = type(v)(0.0)
            samples[k] = samples[i]
        weights[i] = v
    return impose_mean(m, samples, weights), weights
    #XXX: any vectorized way to do this?
#   for i,j in sorted(sorted(pair) for pair in pairs): 
#       weight = weights[i]+weights[j]
#       weights[i],weights[j] = type(weight)(0.0),weight #FIXME: also wrong
#       samples[j] = samples[i] #FIXME: all paired collapses should be equal
#   return impose_mean(m, samples, weights), weights


##### misc methods #####
def impose_sum(mass, weights, zsum=False, zmass=1.0):
  """impose a sum on a list of points

Inputs:
    mass -- target sum of weights
    weights -- a list of sample weights
    zsum -- use counterbalance when mass = 0.0
    zmass -- member scaling when mass = 0.0
"""
  return normalize(weights, mass, zsum, zmass)

def impose_product(mass, weights, zsum=False, zmass=1.0):
  """impose a product on a list of points

Inputs:
    mass -- target product of weights
    weights -- a list of sample weights
    zsum -- use counterbalance when mass = 0.0
    zmass -- member scaling when mass = 0.0
"""
  from numpy import product
  weights = asarray(list(weights)) #XXX: faster to use x = array(x, copy=True) ?
  w = float(product(weights))
  n = len(weights)
  if not w:  #XXX: is this the best behavior?
    from numpy import inf
    return list(weights * inf)  # protect against ZeroDivision
  if float(mass):
    if w/mass < 0.0:
      return list(-weights / (-w/mass)**(1./n))  #FIXME: not "mean-preserving"
    return list(weights / (w/mass)**(1./n))      #FIXME: not "mean-preserving"
  # force selected member to satisfy product = 0.0
  if not zsum:
    return list(weights * 0.0)  #FIXME: not "mean-preserving"
  zsum = -1
  p, weights[zsum] = weights[zsum], 0.0
  w = (w/p)
  n = n-1
  mass = zmass
  if w/mass >= 0.0:
    return list(weights[:-1]/(w/mass)**(1./n))+[0.]#FIXME: not "mean-preserving"
  return list(-weights[:-1]/(-w/mass)**(1./n))+[0.]#FIXME: not "mean-preserving"


#--------------------------------------------------------------------
# <helper methods>

##### packing / unpacking
# >>> a = [1,2,3]; b = [4,5]
# >>> zip(a[0:]+a[:0],b) + zip(a[1:]+a[:1],b) + zip(a[2:]+a[:2],b)
# [(1, 4), (2, 5), (3, 6), (2, 4), (3, 5), (1, 6), (3, 4), (1, 5), (2, 6)]
def _pack(samples):
  """'pack' a list of discrete measure sample points 
into a list of product measure sample points

Inputs:
    samples -- a list of sample points for N discrete measures

For example:
  >>> _pack([[1,2,3], [4,5], [6,7]])
  [(1,4,6), (2,4,6), (3,4,6), (1,5,6), (2,5,6), (3,5,6), \
   (1,4,7), (2,4,7), (3,4,7), (1,5,7), (2,5,7), (3,5,7)]
"""
 #from numpy import product, array, ones
 #ndim = len(samples)
 #npts = [len(s) for s in samples]
 #z = []
 #for i in range(ndim):
 #  tmp = list(array([n*ones(product(npts[:i])) for n in samples[i]]).flatten())
 #  z.append( product(npts[i+1:])*tmp )
 #del tmp
 #zT = []
 #for i in range(len(z[0])):
 #  zT.append( tuple([y.pop(0) for y in z]) )
 #return zT
# from numpy import product, array, ones
# ndim = len(samples)
# npts = [len(s) for s in samples]
# z = ones((ndim, product(npts)))  # z.T of what's needed
# for i in range(ndim):
#   tmp = list(array([n*ones(product(npts[:i])) for n in samples[i]]).flatten())
#   z[i] = product(npts[i+1:])*tmp
# return [tuple(i) for i in z.T]
  ndim = len(samples)
  currentx=[0.]*ndim
  _samples = []
  def recurse(next):
    if next == -1:
      _samples.append(tuple(currentx))
      return
    else:
      for xpt in samples[next]:
        currentx[next]=xpt
        recurse(next - 1)
  recurse(ndim-1)
  return _samples

def _unpack(samples, npts):
  """'unpack' a list of product measure sample points 
into a list of discrete measure sample points

Inputs:
    samples -- a list of sample points for a product measure
    npts -- a tuple of dimensions of the target discrete measure

For example:
  >>> _unpack( [(1,4,6), (2,4,6), (3,4,6), (1,5,6), (2,5,6), (3,5,6), \
  ...           (1,4,7), (2,4,7), (3,4,7), (1,5,7), (2,5,7), (3,5,7)], (3,2,2) \
  ...        )
  [[1,2,3], [4,5], [6,7]]
"""
# from numpy import product, array
# ndim = len(npts)
# z = []
# for i in range(ndim):
#   tmp = array(samples[:int(len(samples)/product(npts[i+1:]))]).T[i]
#   z.append( list(tmp[::int(product(npts[:i]))]) )
# return z
  _samples = []
  ndim = len(npts)
  temp = [npts[0]]*ndim
  _samples.append([j[0] for j in samples][:npts[0]])
  def recurse(next):
    if next == ndim:
      return
    else:
      temp[next] = temp[next-1]*npts[next]
      currentindex = temp[next]
      lastindex = temp[next-1]
      _samples.append([j[next] for j in samples][:currentindex:lastindex])
      recurse(next + 1)
  recurse(1)
  return _samples


def _flat(params):
  """
converts a nested parameter list into flat parameter list

Inputs:
    params -- a nested list of weights or positions

For example:
    >>> par = [['x','x','x'], ['y','y'], ['z']]
    >>> _flat(par)
    ['x','x','x','y','y','z']
"""
  from mystic.tools import flatten
  return list(flatten(params))


def _nested(params, npts):
  """
converts a flat parameter list into nested parameter list

Inputs:
    params -- a flat list of weights or positions
    npts -- a tuple describing the shape of the target list

For example:
    >>> nx = 3;  ny = 2;  nz = 1
    >>> par = ['x']*nx + ['y']*ny + ['z']*nz
    >>> _nested(par, (nx,ny,nz))
    [['x','x','x'], ['y','y'], ['z']]
"""
  coords = []
  ind = 0
  for i in range(len(npts)):
    coords.append(params[ind:ind+npts[i]])
    ind += npts[i]
  return coords


def _nested_split(params, npts):
  """
splits a flat parameter list into a list of weights and a list of positions;
weights and positions are expected to have the same dimensions (given by npts)

Inputs:
    params -- a flat list of weights and positions (formatted as noted below)
    npts -- a tuple describing the shape of the target lists

For example:
    >>> nx = 3;  ny = 2;  nz = 1
    >>> par = ['wx']*nx + ['x']*nx + ['wy']*ny + ['y']*ny + ['wz']*nz + ['z']*nz
    >>> weights, positions = _nested_split(par, (nx,ny,nz))
    >>> weights
    [['wx','wx','wx'], ['wy','wy'], ['wz']]
    >>> positions
    [['x','x','x'], ['y','y'], ['z']]
"""
  weights = []
  coords = []
  ind = 0
  for i in range(len(npts)):
    weights.append(params[ind:ind+npts[i]])
    ind += npts[i]
    coords.append(params[ind:ind+npts[i]])
    ind += npts[i]
  return weights, coords


def split_param(params, npts):
  """
splits a flat parameter list into a flat list of weights
and a flat list of positions;  weights and positions are expected
to have the same dimensions (given by npts)

Inputs:
    params -- a flat list of weights and positions (formatted as noted below)
    npts -- a tuple describing the shape of the target lists

For example:
    >>> nx = 3;  ny = 2;  nz = 1
    >>> par = ['wx']*nx + ['x']*nx + ['wy']*ny + ['y']*ny + ['wz']*nz + ['z']*nz
    >>> weights, positions = split_param(par, (nx,ny,nz))
    >>> weights
    ['wx','wx','wx','wy','wy','wz']
    >>> positions
    ['x','x','x','y','y','z']
"""
  w,x = _nested_split(params, npts)
  return _flat(w), _flat(x)


# backward compatibility
_flat_split = split_param

# EOF

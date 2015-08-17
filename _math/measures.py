#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
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

def weighted_select(samples, weights, mass=1.0):
  """randomly select a sample from weighted set of samples

Inputs:
    samples -- a list of sample points
    weights -- a list of sample weights
    mass -- sum of normalized weights
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
  """calculate the range of a list of points   [range(x) = min(x) - min(x)]

Inputs:
    samples -- a list of sample points
"""
  return max(samples) - min(samples)

def norm(weights):
  """calculate the norm of a list of points   [norm(x) = mean(x)]

Inputs:
    weights -- a list of sample weights
"""
  return mean(weights)

def maximum(f, samples):
  """calculate the max of function for the given list of points

Inputs: 
    f -- a function that takes a list and returns a number
    samples -- a list of sample points
"""
  y = [f(x) for x in samples]
  return max(y)

def ess_maximum(f, samples, weights=None, tol=0.):
  """calculate the max of function for support on the given list of points

Inputs: 
    f -- a function that takes a list and returns a number
    samples -- a list of sample points
    weights -- a list of sample weights
    tol -- weight tolerance, where any weight <= tol is considered zero
"""
  if weights is None:
    return maximum(f, samples)
  return maximum(f, support(samples, weights, tol))

def minimum(f, samples):
  """calculate the min of function for the given list of points

Inputs: 
    f -- a function that takes a list and returns a number
    samples -- a list of sample points
"""
  y = [f(x) for x in samples]
  return min(y)

def ess_minimum(f, samples, weights=None, tol=0.):
  """calculate the min of function for support on the given list of points

Inputs: 
    f -- a function that takes a list and returns a number
    samples -- a list of sample points
    weights -- a list of sample weights
    tol -- weight tolerance, where any weight <= tol is considered zero
"""
  if weights is None:
    return minimum(f, samples)
  return minimum(f, support(samples, weights, tol))

def expectation(f, samples, weights=None, tol=0.0):
  """calculate the (weighted) expectation of a function for a list of points

Inputs:
    f -- a function that takes a list and returns a number
    samples -- a list of sample points
    weights -- a list of sample weights
    tol -- weight tolerance, where any weight <= tol is ignored
"""
  if weights is None:
    y = [f(x) for x in samples]
    return mean(y, weights)
  # contributed by TJS #
  # to prevent function evaluation if weight is "too small":
  # skip evaluation of f(x) if the corresponding weight <= tol
  yw = [(f(x),w) for (x,w) in zip(samples, weights) if abs(w) > tol]
  return mean(*zip(*yw))
  ##XXX: at around len(samples) == 150, the following is faster
  #aw = asarray(weights)
  #ax = asarray(samples)
  #w = aw > tol
  #return mean(ax[w], aw[w])

def mean(samples, weights=None):
  """calculate the (weighted) mean for a list of points

Inputs:
    samples -- a list of sample points
    weights -- a list of sample weights
"""
  if weights is None:
    weights = [1.0/float(len(samples))] * len(samples)
  # get weighted sum
  ssum = sum(i*j for i,j in zip(samples, weights))
  # normalize by sum of the weights
  wts = float(sum(weights))
  if wts: return ssum / wts
  from numpy import inf
  return ssum * inf  # protect against ZeroDivision

def support_index(weights, tol=0):
  """get the indicies of the positions which have non-zero weight

Inputs:
    weights -- a list of sample weights
    tol -- weight tolerance, where any weight <= tol is considered zero
"""
  return [i for (i,w) in enumerate(weights) if w > tol]

def support(samples, weights, tol=0):
  """get the positions which have non-zero weight

Inputs:
    samples -- a list of sample points
    weights -- a list of sample weights
    tol -- weight tolerance, where any weight <= tol is considered zero
"""
  return [samples[i] for (i,w) in enumerate(weights) if w > tol]

def variance(samples, weights=None): #, _mean=None):
  """calculate the (weighted) variance for a list of points

Inputs:
    samples -- a list of sample points
    weights -- a list of sample weights
"""
  if weights is None:
    weights = [1.0/float(len(samples))] * len(samples)
 #if _mean is None:
  _mean = mean(samples, weights)
  svar = [abs(s - _mean)**2 for s in samples]
  return mean(svar, weights)

def std(samples, weights=None): #, _mean=None):
  """calculate the (weighted) standard deviation for a list of points

Inputs:
    samples -- a list of sample points
    weights -- a list of sample weights
"""
  from numpy import sqrt
  return sqrt(variance(samples, weights)) # _mean)


##### coordinate shift methods #####
from numpy import asarray
def impose_mean(m, samples, weights=None):
  """impose a mean on a list of (weighted) points
  (this function is 'range-preserving' and 'variance-preserving')

Inputs:
    m -- the target mean
    samples -- a list of sample points
    weights -- a list of sample weights
"""
 #XXX: this is as expected... mean(impose_mean(2.0, samples, weights), weights)
 #XXX: this is unexpected?... mean(impose_mean(2.0, samples, weights))
  samples = asarray(list(samples)) #XXX: faster to use x = array(x, copy=True) ?
  shift = m - mean(samples, weights)
  samples = samples + shift  #NOTE: is "range-preserving"
  return list(samples)


def impose_variance(v, samples, weights=None):
  """impose a variance on a list of (weighted) points
  (this function is 'mean-preserving')

Inputs:
    v -- the target variance
    samples -- a list of sample points
    weights -- a list of sample weights
"""
  m = mean(samples, weights)
  samples = asarray(list(samples)) #XXX: faster to use x = array(x, copy=True) ?
  sv = variance(samples,weights) #,m)
  if not sv:  # protect against ZeroDivision when variance = 0
    from numpy import nan
    return [nan]*len(samples) #XXX: better to space pts evenly across range?
  from numpy import sqrt
  scale = sqrt(float(v) / sv)
  samples = samples * scale  #NOTE: not "mean-preserving", until the next line
  return impose_mean(m, samples, weights) #NOTE: not range preserving

#FIXME: for range and variance to be 'mutually preserving'...
#       must reconcile scaling by sqrt(v2/v1) & (r2/r1)
#       ...so likely, must scale the weights... or scale each point differently

def impose_std(s, samples, weights=None):
  """impose a standard deviation on a list of (weighted) points
  (this function is 'mean-preserving')

Inputs:
    s -- the target standard deviation
    samples -- a list of sample points
    weights -- a list of sample weights
"""
  return impose_variance(s**2, samples, weights)

def impose_spread(r, samples, weights=None): #FIXME: fails if len(samples) = 1
  """impose a range on a list of (weighted) points
  (this function is 'mean-preserving')

Inputs:
    r -- the target range
    samples -- a list of sample points
    weights -- a list of sample weights
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


def impose_expectation(param, f, npts, bounds=None, weights=None, **kwds):
  """impose a given expextation value (m +/- D) on a given function f.
Optimiziation on f over the given bounds seeks a mean 'm' with deviation 'D'.
  (this function is not 'mean-, range-, or variance-preserving')

Inputs:
    param -- a tuple of target parameters: param = (mean, deviation)
    f -- a function that takes a list and returns a number
    npts -- a tuple of dimensions of the target product measure
    bounds -- a tuple of sample bounds:   bounds = (lower_bounds, upper_bounds)
    weights -- a list of sample weights

Additional Inputs:
    constraints -- a function that takes a nested list of N x 1D discrete
        measure positions and weights   x' = constraints(x, w)

Outputs:
    samples -- a list of sample positions

For example:
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
    >>> samples = impose_expectation((mean,dev), f, (nx,ny,nz), (lb,ub))
    >>>
    >>> # test the results by calculating the expectation value for the samples
    >>> expectation(f, samples)
    >>> 2.00001001012246015
"""
  # param[0] is the target mean
  # param[1] is the acceptable deviation from the target mean

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
      coords = zip(*coords)              # 'mimic' a nested list
      coords = constrain(coords, [weights for i in range(len(coords))])
      coords = zip(*coords)              # revert back to a packed list
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
  npop = 200
  maxiter = 1000;  maxfun = 1e+6
  crossover = 0.9; percent_change = 0.9

  def optimize(cost,(lb,ub),tolerance,_constraints):
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
                 constraints = _constraints)

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

Note: if mass='l1', will use L1-norm; if mass='l2' will use L2-norm; etc.
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
        print "Warning: could not impose mean through reweighting"
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
        print "Warning: could not impose mean through reweighting"
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
    # calculate the cropped indicies
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

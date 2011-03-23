#! /usr/bin/env python
"""
Methods to support discrete measures
"""
# what about sample_unsafe ?
from mystic.math.stats import *
from mystic.math.samples import *
from mystic.math.integrate import *


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

def expectation(f, samples, weights=None):
  """calculate the (weighted) expectation of a function for a list of points

Inputs:
    f -- a function that takes a list and returns a number
    samples -- a list of sample points
    weights -- a list of sample weights
"""
  y = [f(x) for x in samples]
  return mean(y, weights)

def mean(samples, weights=None):
  """calculate the (weighted) mean for a list of points

Inputs:
    samples -- a list of sample points
    weights -- a list of sample weights
"""
  if weights == None:
    weights = [1.0/float(len(samples))] * len(samples)
  # get weighted sum
  ssum = 0.0
  for i in range(len(samples)):
    ssum += samples[i]*weights[i]
  # normalize by sum of the weights
  wts = float(sum(weights))
  if wts: return ssum / wts
  from numpy import inf
  return ssum * inf  # protect against ZeroDivision

def variance(samples, weights=None): #, _mean=None):
  """calculate the (weighted) variance for a list of points

Inputs:
    samples -- a list of sample points
    weights -- a list of sample weights
"""
  if weights == None:
    weights = [1.0/float(len(samples))] * len(samples)
 #if _mean == None:
  _mean = mean(samples, weights)
  svar = [abs(s - _mean)**2 for s in samples]
  return mean(svar, weights)


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
  samples = asarray(samples)
  shift = m - mean(samples, weights)
  samples += shift  #NOTE: is "range-preserving"
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
  samples = asarray(samples)
  sv = variance(samples,weights) #,m)
  if not sv:  # protect against ZeroDivision when variance = 0
    from numpy import nan
    return [nan]*len(samples) #XXX: better to space pts evenly across range?
  from numpy import sqrt
  scale = sqrt(float(v) / sv)
  samples *= scale  #NOTE: not "mean-preserving", until the next line
  return impose_mean(m, samples, weights) #NOTE: not range preserving

#FIXME: for range and variance to be 'mutually preserving'...
#       must reconcile scaling by sqrt(v2/v1) & (r2/r1)
#       ...so likely, must scale the weights... or scale each point differently


def impose_spread(r, samples, weights=None): #FIXME: fails if len(samples) = 1
  """impose a range on a list of (weighted) points
  (this function is 'mean-preserving')

Inputs:
    r -- the target range
    samples -- a list of sample points
    weights -- a list of sample weights
"""
  m = mean(samples, weights)
  samples = asarray(samples)
  sr = spread(samples)
  if not sr:  # protect against ZeroDivision when range = 0
    from numpy import nan
    return [nan]*len(samples) #XXX: better to space pts evenly across range?
  scale = float(r) / sr
  samples *= scale  #NOTE: not "mean-preserving", until the next line
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
    constrain -- an optional user-supplied constraints function,
        where (x', w') = constrain(x, w); and (x, w) are (samples, weights)
        in product_measure space

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

  # plug in 'constraints' function provided by user
  constrain = lambda x,w: (x,w)
  if kwds.has_key('constraints'): constrain = kwds['constraints']
  if not constrain:
    constraints = lambda x,w: (x,w)
  else: #XXX: better to use a standard "xk' = constrain(xk)" interface ?
    def constraints(rv):
      # assumes: samples', weights' = constrain(samples, weights)
      samples = _nested(rv,npts)
      smp, wts = constrain(samples, weights)
      return _flat(smp)

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
  debug = False
  npop = 200
  maxiter = 1000;  maxfun = 1e+6
  crossover = 0.9; percent_change = 0.9

  def optimize(cost,(lb,ub),tolerance,_constraints):
    from mystic.differential_evolution import DifferentialEvolutionSolver2
    from mystic.termination import VTR
    from mystic.strategy import Best1Exp
    from mystic import random_seed, VerboseSow, Sow
    if debug: random_seed(123)
    evalmon = Sow();  stepmon = Sow()
    if debug: stepmon = VerboseSow(10)

    ndim = len(lb)
    solver = DifferentialEvolutionSolver2(ndim,npop)
    solver.SetRandomInitialPoints(min=lb,max=ub)
    solver.SetStrictRanges(min=lb,max=ub)
    solver.SetEvaluationLimits(maxiter,maxfun)
    solver.Solve(cost,termination=VTR(tolerance),strategy=Best1Exp, \
                 CrossProbability=crossover,ScalingFactor=percent_change, \
                 StepMonitor=stepmon, EvaluationMonitor=evalmon, \
                 constraints = _constraints)

    solved = solver.Solution()
    diameter_squared = solver.bestEnergy
    func_evals = len(evalmon.y)
    return solved, diameter_squared, func_evals

  # use optimization to get expectation value
  tolerance = (param[1])**2
  results = optimize(cost, (lower_bounds, upper_bounds), tolerance, constraints)

  # repack the results
  samples = _pack( _nested(results[0],npts) )
  return samples


##### weight shift methods #####
def impose_weight_norm(samples, weights): #XXX: should allow setting norm != 1
  """normalize the weights for a list of (weighted) points
  (this function is 'mean-preserving')

Inputs:
    samples -- a list of sample points
    weights -- a list of sample weights
"""
  m = mean(samples, weights)
  wts = normalize(weights)  #NOTE: not "mean-preserving", until the next line
  return impose_mean(m, samples, wts), wts

def normalize(weights):
  """normalize a list of points to unity (i.e. normalize to 1.0)

Inputs:
    weights -- a list of sample weights
"""
  weights = asarray(weights)
  w = float(sum(weights))
  if w: return list(weights / w) #FIXME: not "mean-preserving"
  from numpy import inf
  return list(weights * inf)  # protect against ZeroDivision


#--------------------------------------------------------------------
# <helper methods>

##### packing / unpacking
# >>> a = [1,2,3]; b = [4,5]
# >>> zip(a[:],b) + zip(a[1:]+a[-1:],b) + zip(a[2:]+a[-2:],b)
# [(1, 4), (2, 5), (2, 4), (3, 5), (3, 4), (2, 5)]
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
  _samples = []
  ndim = len(npts)
  temp = [npts[0]]*ndim
  _samples.append([j[0] for j in samples][:npts[0]])
 #ndim = len(samples[0])
 #npts = int(len(samples)**(1.0/ndim))
 #temp = [npts]*ndim
 #_samples.append([j[0] for j in samples][:npts])

  def recurse(next):
    if next == ndim:
      return
    else:
     #temp[next] = temp[next-1]*npts
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


def _flat_split(params, npts):
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
    >>> weights, positions = _flat_split(par, (nx,ny,nz))
    >>> weights
    ['wx','wx','wx','wy','wy','wz']
    >>> positions
    ['x','x','x','y','y','z']
"""
  w,x = _nested_split(params, npts)
  return _flat(w), _flat(x)


# EOF

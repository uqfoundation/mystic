#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2025 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
tools for generating points on a grid
"""

def binnedpts(lb, ub, nbins, dist=None, clip=False): # bounded gridpts
    """
takes lower and upper bounds (e.g. lb = [1,3], ub = [2,4])
and a list of number of bins on each axis (e.g. nbins = [2,2])
produces a list of points on a grid g = [[1,3],[1,4],[2,3],[2,4]]

Inputs:
    lb  --  a list of the lower bounds
    ub  --  a list of the upper bounds
    nbins  --  a list of number of grid points per axis
    dist  --  a mystic.math.Distribution instance (or list of Distributions)
    clip  --  if True, clip at bounds, else resample [default = False]

Notes: if clip is None, do not clip or resample (may exceed bounds)
    """
    from numbers import Integral
    if isinstance(nbins, Integral):
        nbins = randomly_bin(nbins, len(lb), ones=True, exact=True)
    # generate arrays of points defining a grid in parameter space
    bins = []
    for i,nbin in enumerate(nbins):
        step = 1. * abs(ub[i] - lb[i])/nbin
        bins.append( [lb[i] + (j+0.5)*step for j in range(nbin)] )
    bins = gridpts(bins)
    from mystic.math.samples import _bounded_samples
    bins = _bounded_samples(lb, ub, bins, dist, clip)
    return bins.T.tolist()


def gridpts(q, dist=None):
    """
takes a list of lists of arbitrary length q = [[1,2],[3,4]]
produces a list of points on a grid g = [[1,3],[1,4],[2,3],[2,4]]

Inputs:
    q  --  a list of lists of integers denoting grid points per axis
    dist  --  a mystic.math.Distribution instance (or list of Distributions)

Notes: if a mystic.math.Distribution is provided, use it to inject randomness
    """
    w = [[] for i in range(len(q[-1]))]
    for j in range(len(q)-1,-1,-1):
      for k in range(len(q[j])):
        for l in range(k*len(w)//len(q[j]), (k+1)*len(w)//len(q[j])):
          w[l].append(q[j][k])
      if j: w += [i[:] for i in w[:]*(len(q[j-1])-1)]
    pts = [list(reversed(w[i])) for i in range(len(w))]
    # inject some randomness
    if dist is None: return pts
    if not len(pts): return pts
    if hasattr(dist, '__len__'): #FIXME: isiterable
      import numpy as np
      pts += np.array(tuple(di(len(pts)) for di in dist)).T
    else:
      pts += dist((len(pts),len(pts[0])))
    return pts.tolist()


def samplepts(lb, ub, npts, dist=None, clip=False):
    """
takes lower and upper bounds (e.g. lb = [0,3], ub = [2,4])
produces a list of sample points s = [[1,3],[1,4],[2,3],[2,4]]

Inputs:
    lb  --  a list of the lower bounds
    ub  --  a list of the upper bounds
    npts  --  number of sample points
    dist  --  a mystic.math.Distribution instance (or list of Distributions)
    clip  --  if True, clip at bounds, else resample [default = False]

Notes: if clip is None, do not clip or resample (may exceed bounds)
    """
    from mystic.math.samples import random_samples
    q = random_samples(lb, ub, npts, dist, clip)
    return q.T.tolist()
   #q = [list(i) for i in q]
   #q = zip(*q)
   #return [list(i) for i in q]


def fillpts(lb, ub, npts, data=None, rtol=None, dist=None, clip=False):
    """
takes lower and upper bounds (e.g. lb = [0,3], ub = [2,4])
finds npts that are at least rtol away from legacy data
produces a list of sample points s = [[1,3],[1,4],[2,3],[2,4]]

Inputs:
    lb  --  a list of the lower bounds
    ub  --  a list of the upper bounds
    npts  --  number of sample points
    data  --  a list of legacy sample points
    rtol  --  target radial distance from each point
    dist  --  a mystic.math.Distribution instance (or list of Distributions)
    clip  --  if True, clip at bounds, else resample [default = False]

Notes: if rtol is None, use max rtol; if rtol < 0, use quick-n-dirty method
Notes: if clip is None, do not clip or resample (may exceed bounds)
    """
    bounds = list(zip(lb,ub))
    from mystic.math.distance import euclidean as metric
    #XXX: expose solver settings to user? #XXX: better npop,ftol? faster?
    ###if rtol is None or type(rtol) in (int, float):
    from mystic.solvers import diffev as solver
    kwds = dict(npop=20, ftol=1e-4, gtol=None, disp=0, full_output=0)
    ###initial = lambda : bounds
    ###else: # assume it's a string
    #   from mystic.solvers import fmin_powell as solver
    #   kwds = dict(xtol=1e-8, ftol=1e-8, gtol=2, disp=0, full_output=0)
    #   if rtol == 'None': rtol = None
    #   else: rtol = float(rtol)
    #   import random as rd
    #   initial = lambda : [rd.randrange(l,u)+rd.random() for (l,u) in bounds]
    #from numpy import round
    #kwds['constraints'] = lambda x: round(x, 3)
    # copy the legacy data points (e.g. monitor.x) #XXX: more efficient method?
    pts = [] if data is None else list(data)
    # 'min', 'np.sum()', 'np.min()' are also a choice of distance metric
    if rtol and rtol < 0: # neg radius uses quick-n-dirty method
        def holes(x):
            return (metric(pts,x,axis=0).min(axis=0) < -rtol).sum()
    elif rtol is None: # no radius finds max distance away
        def holes(x):
            res =  metric(pts,x,axis=0).min()
            return -res
    else: # all points should be at least rtol away
        def holes(x):
            res =  metric(pts,x,axis=0).min()
            return -res if res < rtol else 0.0
    # iteratively find a point away from all other points
    for pt in range(npts):
        res = solver(holes, x0=bounds, bounds=bounds, **kwds)
        #res,cost = res[0],res[1]
        pts.append(res.ravel().tolist())
    pts = pts[-npts:] if npts else pts[:0]
    # inject some randomness #XXX: what are alternatives? some sampling?
    from mystic.math.samples import _bounded_samples
    pts = _bounded_samples(lb, ub, pts, dist, clip)
    #print(pts.T)
    return pts.T.tolist()


def errorpts(lb, ub, npts, data=None, error=None, mtol=None, dist=None, clip=False):
    """
takes lower and upper bounds (e.g. lb = [0,3], ub = [2,4])
sample npts near maxima of surface interpolated from {data:error}
produces a list of sample points s = [[1,3],[1,4],[2,3],[2,4]]

Inputs:
    lb  --  a list of the lower bounds
    ub  --  a list of the upper bounds
    npts  --  number of sample points
    data  --  a list of legacy sample points
    error  --  a list of legacy sample error values (must be non-negative)
    mtol  --  iteration tolerance solving for maximum error
    dist  --  a mystic.math.Distribution instance (or list of Distributions)
    clip  --  if True, clip at bounds, else resample [default = False]

Notes: if mtol is None, defaults to mtol=0.1 (must be non-negative)
Notes: if clip is None, do not clip or resample (may exceed bounds)
    """ #NOTE: error cannot be multi-valued
    #XXX: should draw be exposed or fixed? should default to str?
    #XXX: should dist not default to None? (or have a rtol?)
    smooth = 0 #XXX: expose this to the user?
    npop = 8 #FIXME: change default? expose this to the user?
    draw = 'diffev' if npop else 'fmin'
    # draw  --  if False, use npts largest error values, else use weighted draw
    # if draw is a mystic.solver, interpolate error and solve for npts maxima
    #
    import numpy as np
    if data is None: data = []
    if error is None: error = np.ones(len(data)) #XXX: does this make sense?
    if len(data) != len(error):
        msg = "'data' and 'error' must have the same length"
        raise ValueError(msg)
    data = np.asarray(data)
    error = np.asarray(error)
    #print(error)
    if mtol is None: mtol = 10 #NOTE: default is 0.1
    else: mtol = int(100 * mtol)
    if not len(error): # randomly select points (all error is the same)
        return fillpts(lb,ub,npts,dist=dist)
    if error.min() < 0:
        msg = "all 'error' must be non-negative"
        raise ValueError(msg)
    elif not error.sum(): error[:] = 1
    else: #NOTE: if contains 'nan', 'p' in random choice will fail
        eix = np.isinf(error)
        if eix.sum(): # only retain where inf
            error[eix] = 1; error[~eix] = 0
    if len(error) == 1: # select points away from data (error is the same)
        return fillpts(lb,ub,npts,data=data,dist=dist)
    # argsort npts based on maximum values
    idx = np.argsort(error)[::-1][:npts]
    pts = max(npts - len(idx), 0)
    if draw is True:
        # replace with a weighted draw expanded over maximum values
        idx = np.array([np.random.choice(idx[:i+1], size=1, p=error[idx[:i+1]]/error[idx[:i+1]].sum()) for i in range(len(idx))]).flatten()
    # sample remaining npts using simple weighted draw
    idx = np.concatenate((idx, np.random.choice(len(error), size=pts, p=error/error.sum())))
    result = data[idx].tolist()
    if draw:
        # generate error model from sample points and values #XXX: len(error)>1
        from mystic.math.interpolate import interpf, _to_objective
        error = interpf(data, error, 'thin_plate', smooth=smooth)
        error = _to_objective(error)
        bounds = list(zip(lb,ub))
        kwds = dict(bounds=bounds,disp=0,xtol=1e-8,ftol=1e-8,gtol=None)
        kwds.update(dict(npop=npop,maxiter=mtol)) #NOTE: expose to user
        # get solver
        from mystic import solvers, ensemble
        swap = dict(BuckshotSolver='buckshot', LatticeSolver='lattice',
                    SparsitySolver='sparsity', #MixedSolver='mixed',
                    DifferentialEvolutionSolver='diffev',
                    DifferentialEvolutionSolver2='diffev2',
                    NelderMeadSimplexSolver='fmin',
                    PowellDirectionalSolver='fmin_powell')
        draw = getattr(draw, '__name__', draw) # object to name
        draw = swap.get(draw, draw) # one-liner or True
        solver = getattr(solvers, draw, None) if type(draw) is type('') else None
        # if solver then optimize
        if solver is not None:
            if draw in dir(ensemble):
                result = [len(bounds)]*len(result)
            result = np.array([solver(lambda x: -error(x),x0,**kwds) for x0 in result]).tolist()
    # inject some randomness #XXX: what are alternatives? some sampling?
    from mystic.math.samples import _bounded_samples
    result = _bounded_samples(lb, ub, result, dist, clip)
    #print(result.T)
    return result.T.tolist()


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
        for i in chain([2],range(3,n+1,2)):
            s = 0
            while n%i == 0:
                n //= i
                s += 1
            result.extend([i]*s)
            if n == 1:
                return result
    result = factors(N)
    dim = nfact = len(result)
    prime = nfact == 1
    if ndim: result += [1] * (ndim - (nfact // ndim));  dim = ndim
    elif ones: result += [1] # add some 'randomness' by adding a "1"
    # if ones, mix in the 1s; otherwise, only use 1s when ndim < len(result)
    if ones: result = sorted(result, key=lambda v: random())
    else: result[:nfact] = sorted(result[:nfact], key=lambda v: random())
    from numpy import prod as product
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

  # build a grid of starting points
  initial_values = binnedpts(lower, upper, nbins)
  print("grid: %s" % initial_values)

  npts = 10
  lower = [0.0, 3.0, 6.0]
  upper = [2.0, 5.0, 8.0]

  # generate a set of random starting points
  initial_values = samplepts(lower, upper, npts)
  print("scatter: %s" % initial_values)

  npts = 8
  lower = [0.0, 0.0, 0.0]
  upper = [6.0, 6.0, 6.0]
  data = [[3,3,3]]

  # generate a set of space-filling points
  initial_values = fillpts(lower, upper, npts, data)
  print("filled: %s" % initial_values)

  npts = 6
  lower = [0.0, 0.0, 0.0] 
  upper = [4.0, 4.0, 4.0]
  data = [[0,0,0],[2,2,2],[4,4,4]]
  error = [0, 1, 2]

  # generate a set of space-filling points
  initial_values = errorpts(lower, upper, npts, data, error)
  print("errors: %s" % initial_values)


# EOF

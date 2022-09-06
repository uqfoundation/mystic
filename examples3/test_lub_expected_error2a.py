#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''
hyperparameter tuning for least upper bound of Error on |model - surrogate|

Test function is y = F(x), where:
  y0 = x0 + x1 * | x2 * x3**2 - (x4 / x1)**2 |**.5
  y1 = x0 - x1 * | x2 * x3**2 + (x4 / x1)**2 |**.5
  y2 = x0 - | x1 * x2 * x3 - x4 |

toy = lambda x: F(x)[0]
truth = lambda x: toy(x + .01) - .01
model = lambda x: toy(x - .05) + .05
surrogate is interpolated from data sampled from truth,
with hyperparameters z = [smooth]
error = lambda x: (model(x) - surrogate(x))**2

Calculate least upper bound on E|error(x)|, where:
  x in [(0,1), (1,10), (0,10), (0,10), (0,10)]
  wx in [(0,1), (1,1), (1,1), (1,1), (1,1)]
  npts = [2, 1, 1, 1, 1] (i.e. two Dirac masses on x[0], one elsewhere)
  sum(wx[i]_j) for j in [0,npts], for each i
  mean(x[0]) = 5e-1 +/- 1e-3
  var(x[0]) = 5e-3 +/- 1e-4
  z in [(0,10)]

Solves for z producing least upper bound on E|(model(x) - surrogate(x|z))**2|,
given the bounds, normalization, and moment constraints.

Creates 'log' of inner optimizations, 'result' for outer optimization,
and 'truth' database of stored evaluations.

Check results while running (using log reader):
  $ mystic_log_reader log.txt -n 0 -g -p '0:2,2:4,5,7,9,11'
  $ mystic_log_reader result.txt -g -p '0'
'''
from ouq_models import *

# build truth model, F'(x|a'), with selected a'
# generate truth data

# build "expensive" model of truth, F(x|a), with F ?= F' and a ?= a'

# pick hyperparm A, inerpolate G'(x|A) from G(d), trained on "truth data"
# (workflow: iterate interpolate to minimize graphical distance, for given A)
# expected upper bound on f(x) = F(x|a) - G'(x|A)

# 1) F(x|a) = F'(x|a'). Tune A for optimal G.
# 2) F(x|a) != F'(x|a'). Tune A for optimal G, or "a" for optimal F.
# 3) |F(x|a) - F'(x|a')|, no G. Tune "a" for optimal F.
# *) F(x|a) "is callable". Update "d" in G(d), then actively update G(x|A).

# CASE 2a: F(x|a) != F'(x|a'). Tune A for optimal G.


if __name__ == '__main__':

    #from toys import cost5x3 as toy; nx = 5; ny = 3
    #from toys import function5x3 as toy; nx = 5; ny = 3
    #from toys import cost5x1 as toy; nx = 5; ny = 1
    #from toys import function5x1 as toy; nx = 5; ny = 1
    #from toys import cost5 as toy; nx = 5; ny = None
    from toys import function5 as toy; nx = 5; ny = None

    # update 'inner-loop' optimization parameters
    from misc import param, npts, wlb, wub, is_cons, scons
    from ouq import ExpectedValue
    from mystic.bounds import MeasureBounds
    from mystic.monitors import VerboseLoggingMonitor, Monitor, VerboseMonitor
    from mystic.termination import VTRChangeOverGeneration as VTRCOG
    from mystic.termination import Or, VTR, ChangeOverGeneration as COG
    param['opts']['termination'] = COG(1e-10, 100) #NOTE: short stop?
    param['npop'] = 160 #NOTE: increase if results.txt is not monotonic
    param['stepmon'] = VerboseLoggingMonitor(1, 20, filename='log.txt', label='output')

    # build inner-loop and outer-loop bounds
    bnd = MeasureBounds((0,1,0,0,0)[:nx],(1,10,10,10,10)[:nx], n=npts[:nx], wlb=wlb[:nx], wub=wub[:nx])
    bounds = [(0,10)] #NOTE: smooth

    # build a model representing 'truth', and generate some data
    #print("building truth F'(x|a')...")
    true = dict(mu=.01, sigma=0., zmu=-.01, zsigma=0.)
    truth = NoisyModel('truth', model=toy, nx=nx, ny=ny, **true)
    #print('sampling truth...')
    data = truth.sample([(0,1),(1,10)]+[(0,10)]*(nx-2), pts=-16)

    # build a model that approximates 'truth'
    #print('building model F(x|a) of truth...')
    approx = dict(mu=-.05, sigma=0., zmu=.05, zsigma=0.)
    model = NoisyModel('model', model=toy, nx=nx, ny=ny, **approx)

    # get initial guess, a monitor, and a counter
    import mystic._counter as it
    counter = it.Counter()
    import numpy as np
    in_bounds = lambda a,b: (b-a) * np.random.rand() + a
    from pathos.maps import Map
    from pathos.pools import ProcessPool as Pool
    from mystic.solvers import fmin_powell, lattice, PowellDirectionalSolver
    axis = 0 #FIXME: calculation axis (for upper_bound, and thus cost)
    Ns = 25 #XXX: number of samples, when model has randomness
    #_solver, x0 = diffev2, bounds
    #_solver, x0 = fmin_powell, [.5 * sum(i) for i in bounds]
    _solver, x0 = fmin_powell, [in_bounds(*i) for i in bounds]
    #_solver, x0 = lattice, len(bounds)
    stepmon = VerboseLoggingMonitor(1, 1, 1, filename='result.txt', label='solved')

    def cost(x, axis=None, samples=Ns):
        """upper bound on expected model error, for approx and Interp(data)

    Inputs:
        x: list of InterpModel hyperparameters
        axis: int, the index of y on which to find bound (all, by default)
        samples: int, number of samples, for a non-deterministic OUQ model

    Returns:
        upper bound on expected value of model error
        """
        # CASE 2a: F(x|a) != F'(x|a'). Tune A for optimal G.
        kwds = dict(smooth=x[0], noise=0.0, method='thin_plate', extrap=False)

        #print('building estimator G(x) from truth data...')
        surrogate = InterpModel('surrogate', nx=nx, ny=ny, data=truth, **kwds)
        #print('building UQ model of model error...')
        error = ErrorModel('error', model=model, surrogate=surrogate)

        rnd = samples if error.rnd else None
        #print('building UQ objective of expected model error...')
        b = ExpectedValue(error, bnd, constraint=scons, cvalid=is_cons, samples=rnd)
        i = counter.count()
        #print('solving for upper bound on expected model error...')
        solved = b.upper_bound(axis=axis, id=i, **param)
        if type(solved) is not tuple:
            solved = (solved,)
        if axis is None:
            results = tuple(-s for s in solved) #NOTE: -1 for LUB
        else:
            results = -solved[axis] #NOTE: -1 for LUB
        return results

    # outer-loop solver configuration and execution
    settings = dict(args=(axis,Ns), bounds=bounds, maxiter=1000, maxfun=100000,
                    disp=1, full_output=1, itermon=stepmon, ftol=1e-6,
                    npop=4, gtol=4, solver=PowellDirectionalSolver)# xtol=1e-6)
    _map = Map(Pool, settings['npop'])
    result = _solver(cost, x0, map=_map, **settings)
    _map.close(); _map.join(); _map.clear()

    # get the best result (generally the same as returned by solver)
    m = stepmon.min()
    print("%s @ %s" % (m.y, m.x)) #NOTE: -1 for max, 1 for min
    #print("%s @ %s" % (result[1], result[0])) #NOTE: -1 for max, 1 for min


#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''
hyperparameter tuning for least upper bound of ExpectedValue on model

Test function is y = F(x), where:
  y0 = x0 + x1 * | x2 * x3**2 - (x4 / x1)**2 |**.5
  y1 = x0 - x1 * | x2 * x3**2 + (x4 / x1)**2 |**.5
  y2 = x0 - | x1 * x2 * x3 - x4 |

toy = lambda x: F(x)[0]
model = lambda x: toy(x), with x[-2:] = (d,e)
for hyperparameters z = [d, e]

Calculate least upper bound on E|model(x)|, where:
  x in [(0,1), (1,10), (0,10)]
  wx in [(0,1), (1,1), (1,1)]
  npts = [2, 1, 1] (i.e. two Dirac masses on x[0], one elsewhere)
  sum(wx[i]_j) for j in [0,npts], for each i
  mean(x[0]) = 5e-1 +/- 1e-3
  var(x[0]) = 5e-3 +/- 1e-4
  z in [(0,10), (0,10)]

Solves for z that produces least upper bound on E|model(x|z)|,
given the bounds, normalization, and moment constraints.

Creates 'log' of inner optimizations, and 'result' for outer optimization.

Check results while running (using log reader):
  $ mystic_log_reader log.txt -n 0 -g -p '0:2,2:4,5,7'
  $ mystic_log_reader result.txt -g -p '0,1'
'''
from ouq_models import *


if __name__ == '__main__':

    #from toys import cost5x3 as toy; nx = 5; ny = 3
    #from toys import function5x3 as toy; nx = 5; ny = 3
    #from toys import cost5x1 as toy; nx = 5; ny = 1
    #from toys import function5x1 as toy; nx = 5; ny = 1
    #from toys import cost5 as toy; nx = 5; ny = None
    from toys import function5 as toy; nx = 5; ny = None
    from toys import wrap; nx = nx-2 #NOTE: will reduce nx by 2

    # update 'inner-loop' optimization parameters
    from misc import param, npts, wlb, wub, is_cons, scons
    from ouq import ExpectedValue
    from mystic.bounds import MeasureBounds
    from mystic.monitors import VerboseLoggingMonitor, Monitor, VerboseMonitor
    from mystic.termination import VTRChangeOverGeneration as VTRCOG
    from mystic.termination import Or, VTR, ChangeOverGeneration as COG
    param['opts']['termination'] = COG(1e-10, 100) #NOTE: short stop?
    param['npop'] = 40 #NOTE: increase if results.txt is not monotonic
    param['stepmon'] = VerboseLoggingMonitor(1, 20, filename='log.txt', label='output')

    # build inner-loop and outer-loop bounds
    bnd = MeasureBounds((0,1,0,0,0)[:nx],(1,10,10,10,10)[:nx], n=npts[:nx], wlb=wlb[:nx], wub=wub[:nx])
    bounds = [(0,10),(0,10)] #NOTE: x[-2],x[-1]

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
        """upper bound on expected model output

    Inputs:
        x: list of model hyperparameters
        axis: int, the index of y on which to find bound (all, by default)
        samples: int, number of samples, for a non-deterministic OUQ model

    Returns:
        upper bound on expected value of model output
        """
        # build a model, F(x|a), and tune "a" for optimal F.
        toy_ = wrap(d=x[0], e=x[1])(toy) #NOTE: reduces nx by 2
        #print('building model F(x|a)...')
        model = WrapModel('model', model=toy_, nx=nx, ny=ny, rnd=False)

        rnd = samples if model.rnd else None
        #print('building UQ objective of expected model output...')
        b = ExpectedValue(model, bnd, constraint=scons, cvalid=is_cons, samples=rnd)
        i = counter.count()
        #print('solving for upper bound on expected model output...')
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


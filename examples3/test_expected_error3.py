#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''
calculate lower and upper bound on expected Error on |truth - model|

Test function is y = F(x), where:
  y0 = x0 + x1 * | x2 * x3**2 - (x4 / x1)**2 |**.5
  y1 = x0 - x1 * | x2 * x3**2 + (x4 / x1)**2 |**.5
  y2 = x0 - | x1 * x2 * x3 - x4 |

toy = lambda x: F(x)[0]
truth = lambda x: toy(x + .01) - .01
model = lambda x: toy(x - .05) + .05
error = lambda x: (truth(x) - model(x))**2

Calculate lower and upper bound on E|error(x)|, where:
  x in [(0,1), (1,10), (0,10), (0,10), (0,10)]
  wx in [(0,1), (1,1), (1,1), (1,1), (1,1)]
  npts = [2, 1, 1, 1, 1] (i.e. two Dirac masses on x[0], one elsewhere)
  sum(wx[i]_j) for j in [0,npts], for each i
  mean(x[0]) = 5e-1 +/- 1e-3
  var(x[0]) = 5e-3 +/- 1e-4

Solves for two scenarios of x that produce lower bound on E|error(x)|,
given the bounds, normalization, and moment constraints. Repeats
calculation for upper bound on E|error(x)|.

Creates 'log' of optimizations.
'''
from ouq_models import *


if __name__ == '__main__':

    #from toys import cost5x3 as toy; nx = 5; ny = 3
    #from toys import function5x3 as toy; nx = 5; ny = 3
    #from toys import cost5x1 as toy; nx = 5; ny = 1
    #from toys import function5x1 as toy; nx = 5; ny = 1
    #from toys import cost5 as toy; nx = 5; ny = None
    from toys import function5 as toy; nx = 5; ny = None

    # update optimization parameters
    from misc import param, npts, wlb, wub, is_cons, scons
    from ouq import ExpectedValue
    from mystic.bounds import MeasureBounds
    from mystic.monitors import VerboseLoggingMonitor, Monitor, VerboseMonitor
    from mystic.termination import VTRChangeOverGeneration as VTRCOG
    from mystic.termination import Or, VTR, ChangeOverGeneration as COG
    param['opts']['termination'] = COG(1e-10, 100) #NOTE: short stop?
    param['npop'] = 80 #NOTE: increase if poor convergence
    param['stepmon'] = VerboseLoggingMonitor(1, 20, filename='log.txt', label='lower')

    # build bounds
    bnd = MeasureBounds((0,1,0,0,0)[:nx],(1,10,10,10,10)[:nx], n=npts[:nx], wlb=wlb[:nx], wub=wub[:nx])

    # build a model representing 'truth'
    #print("building truth F'(x|a')...")
    true = dict(mu=.01, sigma=0., zmu=-.01, zsigma=0.)
    truth = NoisyModel('truth', model=toy, nx=nx, ny=ny, **true)
    Ns = 25 #XXX: number of samples, when model has randomness

    # build a model that approximates 'truth'
    #print('building model F(x|a) of truth...')
    approx = dict(mu=-.05, sigma=0., zmu=.05, zsigma=0.)
    model = NoisyModel('model', model=toy, nx=nx, ny=ny, **approx)
    #print('building UQ model of model error...')
    error = ErrorModel('error', model=truth, surrogate=model)

    rnd = Ns if error.rnd else None
    #print('building UQ objective of expected model error...')
    b = ExpectedValue(error, bnd, constraint=scons, cvalid=is_cons, samples=rnd)
    #print('solving for lower bound on expected model error...')
    b.lower_bound(axis=None, id=0, **param)
    print("lower bound per axis:")
    for axis,solver in b._lower.items():
        print("%s: %s @ %s" % (axis, solver.bestEnergy, solver.bestSolution))

    #print('solving for upper bound on expected model error...')
    param['opts']['termination'] = COG(1e-10, 200) #NOTE: short stop?
    param['npop'] = 160 #NOTE: increase if poor convergence
    param['stepmon'] = VerboseLoggingMonitor(1, 20, filename='log.txt', label='upper')
    b.upper_bound(axis=None, id=1, **param)
    print("upper bound per axis:")
    for axis,solver in b._upper.items():
        print("%s: %s @ %s" % (axis, -solver.bestEnergy, solver.bestSolution))

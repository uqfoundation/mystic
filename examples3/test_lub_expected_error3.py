#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

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

# CASE 3: |F(x|a) - F'(x|a')|, no G. Tune "a" for optimal F.


if __name__ == '__main__':

    #from toys import cost5x3 as toy; nx = 5; ny = 3
    #from toys import function5x3 as toy; nx = 5; ny = 3
    #from toys import cost5x1 as toy; nx = 5; ny = 1
    #from toys import function5x1 as toy; nx = 5; ny = 1
    #from toys import cost5 as toy; nx = 5; ny = None
    from toys import function5 as toy; nx = 5; ny = None

    from misc import param, npts, wlb, wub, is_cons, scons
    from ouq import ExpectedValue
    from mystic.monitors import VerboseLoggingMonitor, Monitor, VerboseMonitor
    from mystic.termination import VTRChangeOverGeneration as VTRCOG
    from mystic.termination import Or, VTR, ChangeOverGeneration as COG
    # update 'inner-loop' optimization parameters
    param['opts']['termination'] = COG(1e-10, 200)
    param['npop'] = 160
    param['stepmon'] = VerboseLoggingMonitor(1, 20, filename='log.txt', label='output')
    bnd = MeasureBounds((0,0,0,0,0)[:nx],(1,10,10,10,10)[:nx], n=npts[:nx], wlb=wlb[:nx], wub=wub[:nx])
    bounds = [(-10,10),(0,0),(-10,10),(0,0)] #NOTE: mu,sigma,zmu,zsigma
    import counter as it
    counter = it.Counter()

    #print("building truth F'(x|a')...")
    true = dict(mu=.01, sigma=0., zmu=-.01, zsigma=0.)
    truth = NoisyModel('truth', model=toy, nx=nx, ny=ny, **true)

    import numpy as np
    in_bounds = lambda a,b: (b-a) * np.random.rand() + a
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
        # CASE 3: |F(x|a) - F'(x|a')|, no G. Tune "a" for optimal F.
        approx = dict(mu=x[0], sigma=x[1], zmu=x[2], zsigma=x[3])
        #print('building model F(x|a) of truth...')
        model = NoisyModel('model', model=toy, nx=nx, ny=ny, **approx)

        #print('building UQ model of model error...')
        error = ErrorModel('error', model=truth, surrogate=model)

        rnd = Ns if error.rnd else None
        #print('building UQ objective of expected model error...')
        b = ExpectedValue(error, bnd, constraint=scons, cvalid=is_cons, samples=rnd)
        i = counter.count()
        #print('solving for upper bound on expected model error...')
        solver = b.upper_bound(axis=axis, id=i, **param)
        if type(solver) is not tuple:
            solver = (solver,) #FIXME: save solver to DB (or pkl)
        if axis is None:
            results = tuple(-s.bestEnergy for s in solver) #NOTE: -1 for GUB
            #print('[id: %s] %s' % (i, tuple(s.bestSolution for s in solver)))
        else:
            results = -solver[axis].bestEnergy #NOTE: -1 for GUB
            #print('[id: %s] %s' % (i, solver[axis].bestSolution))
        return results

    settings = dict(args=(axis,Ns), bounds=bounds, maxiter=1000, maxfun=100000,
                    disp=1, full_output=1, itermon=stepmon, ftol=1e-6,
                    npop=4, gtol=4, solver=PowellDirectionalSolver)# xtol=1e-6)
    pool = Pool(settings['npop'])
    _map = pool.map
    result = _solver(cost, x0, map=_map, **settings)
    pool.close(); pool.join(); pool.clear()
    print("%s @ %s" % (result[1], result[0])) #NOTE: -1 for max, 1 for min


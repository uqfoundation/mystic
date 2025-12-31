#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2022-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
We estimate the error on the lattice parameters as σ ~ 1e-4, as if N(μ,σ^2).

Given uncertainty in the selected lattice parameters, find the:
1) expected variance of "beta"
2) upper/lower bound on the expected variance of "beta"
3) expected VAR of "beta"
4) upper/lower bound on the expected VAR of "beta"
5) expected negativity of "beta"
6) upper/lower bound on the expected negativity of "beta"

lattice parameters x3 = (a_alpha, c_alpha, a_steel) +/- error3 in bounds3,
where: x3 = [2.9306538, 4.6817646, 3.6026807] (potentially μ)
       error3 = [1e-4, 1e-4, 1e-4] (potentially σ)
       bounds3 = [(0.95 * i, 1.05 * i) for i in x3]
looking for impact on a_beta, we could also impose a_beta +/- error,
where: a_beta = [3.233392] (potentially μ)
       error = [1e-4] (potentially σ)
"""

if __name__ == '__main__':

    from misc3 import *
    from ouq import Variance, ValueAtRisk
    from ouq_ import Negativity

    try: # parallel maps
        from pathos.maps import Map
        from pathos.pools import ProcessPool, ThreadPool, _ThreadPool
        pmap = Map(ProcessPool) if Ns else Map() # for sampling
        param['map'] = Map()#Map(ThreadPool) # for objective
        if ny: param['axmap'] = Map(_ThreadPool, join=True) # for multi-axis
        smap = Map(_ThreadPool, join=True) if ny else Map() # for expected value
    except ImportError:
        pmap = smap = None

    import datetime
    kwds.update(dict(smap=smap, verbose=False))
    axis = None

    # where x,F(x) has uncertainty, calculate:
    # expected variance
    print('\n%s'% datetime.datetime.now())
    param['id'] = 0 #NOTE: to distinguish solver in multi-run logging
    b = Variance(model, bnd, constraint=ccons, cvalid=is_cons, samples=Ns, map=pmap)
    b.expected(archive='ave.db', axis=axis, **kwds, **param)
    print("expected variance per axis:")
    for ax,solver in b._expect.items():
        print("%s: %s @ %s" % (ax, solver.bestEnergy, solver.bestSolution))

    # lower bound on variance
    if param['id'] is not None: param['id'] += 1
    print('\n%s'% datetime.datetime.now())
    b.lower_bound(axis=axis, **param)
    print("lower bound on expected variance per axis:")
    for ax,solver in b._lower.items():
        print("%s: %s @ %s" % (ax, solver.bestEnergy, solver.bestSolution))

    # calculate upper bound on variance
    if param['id'] is not None: param['id'] += 1
    print('\n%s'% datetime.datetime.now())
    b.upper_bound(axis=axis, **param)
    print("upper bound on expected variance per axis:")
    for ax,solver in b._upper.items():
        print("%s: %s @ %s" % (ax, -solver.bestEnergy, solver.bestSolution))

    # expected VAR of value
    if param['id'] is not None: param['id'] += 1
    print('\n%s'% datetime.datetime.now())
    b = ValueAtRisk(model, bnd, constraint=ccons, cvalid=is_cons, samples=Ns, map=pmap)
    b.expected(archive='min.db', axis=axis, **kwds, **param)
    print("expected VAR per axis:")
    for ax,solver in b._expect.items():
        print("%s: %s @ %s" % (ax, solver.bestEnergy, solver.bestSolution))

    # lower bound on VAR
    if param['id'] is not None: param['id'] += 1
    print('\n%s'% datetime.datetime.now())
    b.lower_bound(axis=axis, **param)
    print("lower bound on expected VAR per axis:")
    for ax,solver in b._lower.items():
        print("%s: %s @ %s" % (ax, solver.bestEnergy, solver.bestSolution))

    # upper bound on VAR
    if param['id'] is not None: param['id'] += 1
    print('\n%s'% datetime.datetime.now())
    b.upper_bound(axis=axis, **param)
    print("upper bound on expected VAR per axis:")
    for ax,solver in b._upper.items():
        print("%s: %s @ %s" % (ax, -solver.bestEnergy, solver.bestSolution))

    # expected negativity of value
    if param['id'] is not None: param['id'] += 1
    print('\n%s'% datetime.datetime.now())
    b = Negativity(model, bnd, constraint=ccons, cvalid=is_cons, samples=Ns, map=pmap)
    b.expected(archive='max.db', axis=axis, **kwds, **param)
    print("expected negativity per axis:")
    for ax,solver in b._expect.items():
        print("%s: %s @ %s" % (ax, solver.bestEnergy, solver.bestSolution))

    # lower bound on negativity
    if param['id'] is not None: param['id'] += 1
    print('\n%s'% datetime.datetime.now())
    b.lower_bound(axis=axis, **param)
    print("lower bound on expected negativity per axis:")
    for ax,solver in b._lower.items():
        print("%s: %s @ %s" % (ax, solver.bestEnergy, solver.bestSolution))

    # upper bound on negativity
    if param['id'] is not None: param['id'] += 1
    print('\n%s'% datetime.datetime.now())
    b.upper_bound(axis=axis, **param)
    print("upper bound on expected negativity per axis:")
    for ax,solver in b._upper.items():
        print("%s: %s @ %s" % (ax, -solver.bestEnergy, solver.bestSolution))

    # shutdown
    print('\n%s'% datetime.datetime.now())
    if pmap is not None:
        pmap.close(); pmap.join(); pmap.clear()
    pmap = param['map']
    if pmap is not None:
        pmap.close(); pmap.join(); pmap.clear()
    if ny:
        pmap = param['axmap']
        if pmap is not None:
            pmap.close(); pmap.join(); pmap.clear()
    if smap is not None:
        smap.close(); smap.join(); smap.clear()

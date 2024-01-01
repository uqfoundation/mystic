#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2022-2024 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
We estimate the error on the lattice parameters as σ ~ 1e-4, as if N(μ,σ^2).

Given uncertainty in the selected lattice parameters, find the:
1) expected value of "wR"
2) upper/lower bound on the expected value of "wR"
3) min/ave/max upper/lower bound on the value of "wR"

lattice params x4 = (a_alpha, c_alpha, a_steel, a_beta) +/- error4 in bounds4,
where: x4 = [2.9306538, 4.6817646, 3.6026807, 3.233392] (potentially μ)
       error4 = [1e-4, 1e-4, 1e-4, 1e-4] (potentially σ)
       bounds4 = [(0.95 * i, 1.05 * i) for i in x4]
looking for impact on wR
"""

if __name__ == '__main__':

    from misc4 import *
    from ouq import ExpectedValue, MinimumValue, MaximumValue

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
    # expected value
    print('\n%s'% datetime.datetime.now())
    param['id'] = 0 #NOTE: to distinguish solver in multi-run logging
    b = ExpectedValue(model, bnd, constraint=ccons, cvalid=is_cons, samples=Ns, map=pmap)
    b.expected(archive='ave.db', axis=axis, **kwds, **param)
    print("expected value per axis:")
    for ax,solver in b._expect.items():
        print("%s: %s @ %s" % (ax, solver.bestEnergy, solver.bestSolution))

    # lower bound on expected value
    if param['id'] is not None: param['id'] += 1
    print('\n%s'% datetime.datetime.now())
    b.lower_bound(axis=axis, **param)
    print("lower bound on expected value per axis:")
    for ax,solver in b._lower.items():
        print("%s: %s @ %s" % (ax, solver.bestEnergy, solver.bestSolution))

    # calculate upper bound on expected value
    if param['id'] is not None: param['id'] += 1
    print('\n%s'% datetime.datetime.now())
    b.upper_bound(axis=axis, **param)
    print("upper bound on expected value per axis:")
    for ax,solver in b._upper.items():
        print("%s: %s @ %s" % (ax, -solver.bestEnergy, solver.bestSolution))

    # expected lower bound on value
    if param['id'] is not None: param['id'] += 1
    print('\n%s'% datetime.datetime.now())
    b = MinimumValue(model, bnd, constraint=scons, cvalid=is_cons, samples=Ns, map=pmap)
    b.expected(archive='min.db', axis=axis, **kwds, **param)
    print("expected lower bound per axis:")
    for ax,solver in b._expect.items():
        print("%s: %s @ %s" % (ax, solver.bestEnergy, solver.bestSolution))

    # least lower bound on value
    if param['id'] is not None: param['id'] += 1
    print('\n%s'% datetime.datetime.now())
    b.lower_bound(axis=axis, **param)
    print("least lower bound per axis:")
    for ax,solver in b._lower.items():
        print("%s: %s @ %s" % (ax, solver.bestEnergy, solver.bestSolution))

    # greatest lower bound on value
    if param['id'] is not None: param['id'] += 1
    print('\n%s'% datetime.datetime.now())
    b.upper_bound(axis=axis, **param)
    print("greatest lower bound per axis:")
    for ax,solver in b._upper.items():
        print("%s: %s @ %s" % (ax, -solver.bestEnergy, solver.bestSolution))

    # expected upper bound on value
    if param['id'] is not None: param['id'] += 1
    print('\n%s'% datetime.datetime.now())
    b = MaximumValue(model, bnd, constraint=scons, cvalid=is_cons, samples=Ns, map=pmap)
    b.expected(archive='max.db', axis=axis, **kwds, **param)
    print("expected upper bound per axis:")
    for ax,solver in b._expect.items():
        print("%s: %s @ %s" % (ax, solver.bestEnergy, solver.bestSolution))

    # least upper bound on value
    if param['id'] is not None: param['id'] += 1
    print('\n%s'% datetime.datetime.now())
    b.lower_bound(axis=axis, **param)
    print("least upper bound per axis:")
    for ax,solver in b._lower.items():
        print("%s: %s @ %s" % (ax, solver.bestEnergy, solver.bestSolution))

    # greatest upper bound on value
    if param['id'] is not None: param['id'] += 1
    print('\n%s'% datetime.datetime.now())
    b.upper_bound(axis=axis, **param)
    print("greatest upper bound per axis:")
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

#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2022-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
We estimate the error on the lattice parameters as σ ~ 1e-4, as if N(μ,σ^2).

Given uncertainty in the selected lattice parameters, find the:
1) expected value of "beta"
2) upper/lower bound on the expected value of "beta"
3) min/ave/max upper/lower bound on the value of "beta"

lattice parameters x3 = (a_alpha, c_alpha, a_steel) +/- error3 in bounds3,
where: x3 = [2.9306538, 4.6817646, 3.6026807] (potentially μ)
       error3 = [1e-4, 1e-4, 1e-4] (potentially σ)
       bounds3 = [(0.95 * i, 1.05 * i) for i in x3]
looking for impact on a_beta, we could also impose a_beta +/- error,
where: a_beta = [*unknown*] (potentially μ)
       error = [1e-4] (potentially σ)
"""

if __name__ == '__main__':

    from misc3b import *
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
    b = MinimumValue(model, bnd, constraint=ccons, cvalid=is_cons, samples=Ns, map=pmap)
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
    b = MaximumValue(model, bnd, constraint=ccons, cvalid=is_cons, samples=Ns, map=pmap)
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

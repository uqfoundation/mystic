#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2022-2025 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
We estimate the error on the detector parameters as σ, as if N(μ,σ^2).

Given uncertainty in the selected detector parameters, find the:
1) expected value of "wR"
2) upper/lower bound on the expected value of "wR"
3) min/ave/max upper/lower bound on the value of "wR"

detector params x6 = (dist_spec, center_x, center_y, detc_2theta, detc_phiDA, detc_omegaDN) +/- error6 in bounds6,
where: x6 = [65.0458, 2.540605, -0.28430015, -27.21818, -0.44651195, 1.1400392] (potentially μ)
       error6 = [0.02124893, 0.028312204, 0.10003452, 0.011049773, 1.0831603, 2.0910518] (potentially σ)
       bounds6 = [(0.95 * i, 1.05 * i) for i in x6]
looking for impact on wR
"""

if __name__ == '__main__':

    from misc6 import *
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

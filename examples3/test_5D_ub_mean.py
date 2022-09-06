#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
calculate upper bound on expected value

Test function is y = F(x), where:
  y0 = x0 + x1 * | x2 * x3**2 - (x4 / x1)**2 |**.5
  y1 = x0 - x1 * | x2 * x3**2 + (x4 / x1)**2 |**.5
  y2 = x0 - | x1 * x2 * x3 - x4 |

model = lambda x: F(x)[0]

Calculate upper bound on E|model(x)|, where:
  x in [(0,1), (1,10), (0,10), (0,10), (0,10)]
  wx in [(0,1), (1,1), (1,1), (1,1), (1,1)]
  npts = [2, 1, 1, 1, 1] (i.e. two Dirac masses on x[0], one elsewhere)
  sum(wx[i]_j) for j in [0,npts], for each i
  E|model(x)| = 11.0 +/- 1.0

Solves for two scenarios of x that produce upper bound on E|model(x)|,
given the bounds, normalization, and moment constraints.
"""


if __name__ == '__main__':

    from spec5D import *
    from ouq import ExpectedValue
    from mystic.bounds import MeasureBounds
    from ouq_models import WrapModel
    #from toys import cost5x3 as toy; nx = 5; ny = 3
    #from toys import function5x3 as toy; nx = 5; ny = 3
    #from toys import cost5x1 as toy; nx = 5; ny = 1
    #from toys import function5x1 as toy; nx = 5; ny = 1
    #from toys import cost5 as toy; nx = 5; ny = None
    from toys import function5 as toy; nx = 5; ny = None
    Ns = 25

    try: # parallel maps
        from pathos.maps import Map
        from pathos.pools import ThreadPool, _ThreadPool
        pmap = Map(ThreadPool) if Ns else Map() # for sampling
        if ny: param['axmap'] = Map(_ThreadPool, join=True) # for multi-axis
    except ImportError:
        pmap = None

    # build a model representing 'truth'
    nargs = dict(nx=nx, ny=ny, rnd=False)
    model = WrapModel('model', toy, **nargs)

    # calculate upper bound on expected value, where x[0] has uncertainty
    bnd = MeasureBounds(xlb, xub, n=npts, wlb=wlb, wub=wub)
    b = ExpectedValue(model, bnd, constraint=scons, cvalid=is_cons, samples=Ns, map=pmap)
    b.upper_bound(axis=None, **param)
    print("upper bound per axis:")
    for axis,solver in b._upper.items():
        print("%s: %s @ %s" % (axis, -solver.bestEnergy, solver.bestSolution))

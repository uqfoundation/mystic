#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
calculate upper bound on probability of failure

Test function is y = F(x), where:
  y0 = x0 + x1 * | x2 * x3**2 - (x4 / x1)**2 |**.5
  y1 = x0 - x1 * | x2 * x3**2 + (x4 / x1)**2 |**.5
  y2 = x0 - | x1 * x2 * x3 - x4 |

model = lambda x: F(x)
success = all( model(x) >= cutoff )

Calculate upper bound on E|success(x)|, where:
  x in [(0,1), (0,10), (0,10), (0,0), (0,10)]
  wx in [(0,1), (1,1), (1,1), (1,1), (1,1)]
  npts = [2, 1, 1, 1, 1] (i.e. two Dirac masses on x[0], one elsewhere)
  sum(wx[i]_j) for j in [0,npts], for each i
  mean(x[0]) = 5e-1 +/- 1e-3
  var(x[0]) = 5e-3 +/- 1e-4

Solves for two scenarios of x that produce upper bound on E|success(x)|,
for y0, given the bounds, normalization, and moment constraints.
"""


if __name__ == '__main__':

    from misc import *
    from mystic.math.discrete import product_measure
    from ouq import ProbOfFailure
    from mystic.bounds import MeasureBounds
    from ouq_models import WrapModel, SuccessModel
    #from toys import cost5x3 as toy; nx = 5; ny = 3
    from toys import function5x3 as toy; nx = 5; ny = 3
    #from toys import cost5x1 as toy; nx = 5; ny = 1
    #from toys import function5x1 as toy; nx = 5; ny = 1
    #from toys import cost5 as toy; nx = 5; ny = None
    #from toys import function5 as toy; nx = 5; ny = None
    Ns = 25

    # build a model representing 'truth'
    nargs = dict(nx=nx, ny=ny, rnd=False)
    model = WrapModel('model', toy, **nargs)

    # build a model of success, relative to the cutoff
    sargs = dict(cutoff=(.5, .5, .5), nx=nx, ny=ny)
    success = SuccessModel('success', model, **sargs)

    # calculate upper bound on expected success, where x[0] has uncertainty
    bnd = MeasureBounds((0,0,0,0,0),(1,10,10,0,10), n=npts, wlb=wlb, wub=wub)
    rnd = Ns if success.rnd else None
    d = ProbOfFailure(success, bnd, constraint=scons, cvalid=is_cons, samples=rnd)
    d.upper_bound(axis=0, **param)
    print("upper bound per axis:")
    for axis,solver in d._upper.items():
        print("%s: %s @ %s" % (axis, -solver.bestEnergy, solver.bestSolution))

#!/usr/bin/env python
# 
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
calculate upper bound on expected value

Test function is y = F(x), where:
  F is defined in surrogate.py

model = F(x)

Calculate upper bound on E|model(x)|, where:
  x in [(20.0,150.0), (0.0,30.0), (2.1,2.8)]
  wx in [(0,1), (1,1), (1,1)]
  npts = [2, 1, 1] (i.e. two Dirac masses on x[0], one elsewhere)
  sum(wx[i]_j) for j in [0,npts], for each i
  E|model(x)| = 6.5 +/- 1.0

Solves for two scenarios of x that produce upper bound on E|model(x)|,
given the bounds, normalization, and moment constraints.
"""


if __name__ == '__main__':

    from spec3D import *
    from ouq import ExpectedValue
    from mystic.bounds import MeasureBounds
    from ouq_models import WrapModel
    from surrogate import marc_surr as toy; nx = 3; ny = None
    Ns = 25

    # build a model representing 'truth'
    nargs = dict(nx=nx, ny=ny, rnd=False)
    model = WrapModel('model', toy, **nargs) 

    # calculate upper bound on expected value, where F(x) has uncertainty
    bnd = MeasureBounds(xlb, xub, n=npts, wlb=wlb, wub=wub)
    b = ExpectedValue(model, bnd, constraint=scons, cvalid=is_cons, samples=Ns)
    b.upper_bound(axis=None, **param)
    print("upper bound per axis:")
    for axis,solver in b._upper.items():
        print("%s: %s @ %s" % (axis, -solver.bestEnergy, solver.bestSolution))

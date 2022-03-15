#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
calculate upper bound on mean value of 0th input

Test function is y = F(x), where:
  F is defined in surrogate.py

model = F(x)

Calculate upper bound on mean(x[0]), where:
  x in [(20.0,150.0), (0.0,30.0), (2.1,2.8)]
  wx in [(0,1), (1,1), (1,1)]
  npts = [2, 1, 1] (i.e. two Dirac masses on x[0], one elsewhere)
  sum(wx[i]_j) for j in [0,npts], for each i
  E|model(x)| = 6.5 +/- 1.0

Solves for two scenarios of x that produce upper bound on mean(x[0]),
given the bounds, normalization, and moment constraints.
"""


if __name__ == '__main__':

    # get model, bounds, constraints
    from spec3D import *
    from ouq_ import MeanValue
    Ns = None #FIXME: non-deterministic model is NotImplemented

    # calculate upper bound on mean value, where x[0] has uncertainty
    b = MeanValue(model, bnd, constraint=scons, cvalid=is_cons, samples=Ns, idx=0) #FIXME: idx is ignored; fixed in objective
    b.upper_bound(axis=None, **param)
    print("upper bound per axis:")
    for axis,solver in b._upper.items():
        print("%s: %s @ %s" % (axis, -solver.bestEnergy, solver.bestSolution))

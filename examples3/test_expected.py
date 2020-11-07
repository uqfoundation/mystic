#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
calculate upper bound on expected value
"""


if __name__ == '__main__':

    from misc import *
    from ouq import ExpectedValue
    from bounds import MeasureBounds
    from ouq_models import NoisyModel
    #from toys import cost5x3 as toy; nx = 5; ny = 3
    #from toys import function5x3 as toy; nx = 5; ny = 3
    #from toys import cost5x1 as toy; nx = 5; ny = 1
    #from toys import function5x1 as toy; nx = 5; ny = 1
    #from toys import cost5 as toy; nx = 5; ny = None
    from toys import function5 as toy; nx = 5; ny = None
    Ns = 5 # 25

    # build a model representing 'truth'
    nargs = dict(sigma=5.0, nx=nx, ny=ny)
    model = NoisyModel('model', toy, **nargs)

    # calculate upper bound on expected value, where x[0] has uncertainty
    bnd = MeasureBounds((0,0,0,0,0),(1,10,10,10,10), n=npts, wlb=wlb, wub=wub)
    b = ExpectedValue(model, bnd, constraint=scons, cvalid=is_cons, samples=Ns)
    solver = b.upper_bound(axis=None, **param)
    if type(solver) is not tuple:
        solver = (solver,)
    for s in solver:
        print("%s @ %s" % (-s.bestEnergy, s.bestSolution))

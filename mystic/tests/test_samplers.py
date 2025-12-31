#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2025-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from mystic.solvers import PowellDirectionalSolver
from mystic.termination import NormalizedChangeOverGeneration as NCOG
from mystic.samplers import (SparsitySampler, BuckshotSampler, LatticeSampler,
                             ResidualSampler, MixedSampler)
from mystic.models import sphere as model
from mystic.math import almostEqual
from mystic.bounds import Bounds
bounds = Bounds(-1,1,n=4)
x0 = [0,0,0,0]
y0 = 0
N = 8


for sampler in (SparsitySampler, BuckshotSampler, LatticeSampler,
                ResidualSampler, MixedSampler):
    s = sampler(bounds, model, npts=N, id=0, maxiter=8000, maxfun=1e6,
                solver=PowellDirectionalSolver, termination=NCOG(1e-6, 10))
    s.sample_until(terminated=all)
    x = [list(i) for i in s._sampler._all_bestSolution]
    y = s._sampler._all_bestEnergy

    assert s._sampler._npts == N
    for xi,yi in zip(x,y):
        assert almostEqual(xi, x0, tol=1e-12)
        assert almostEqual(yi, y0)

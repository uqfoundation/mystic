#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
misc user-defined items (solver configuration, moment constraints)
"""
from mystic.solvers import DifferentialEvolutionSolver2
from mystic.monitors import VerboseMonitor, Monitor
from mystic.termination import ChangeOverGeneration as COG
from mystic.bounds import Bounds, MeasureBounds

# kwds for solver
opts = dict(termination=COG(1e-10, 100))
param = dict(solver=DifferentialEvolutionSolver2,
             npop=80,
             maxiter=1500,
             maxfun=1e+6,
             x0=None, # use RandomInitialPoints
             nested=None, # don't use SetNested
             map=None, # don't use SetMapper
             stepmon=VerboseMonitor(1, label='output'), # monitor config
             #evalmon=Monitor(), # monitor config (re-initialized in solve)
             # kwds to pass directly to Solve(objective, **opt)
             opts=opts,
            )

# kwds for sampling
kwds = dict(npts=500, ipts=None, itol=1e-8, iter=5)

from mystic.constraints import and_, integers, sorting
from mystic.coupler import outer, additive
from ouq_misc import (flatten, unflatten, normalize_moments, constrained, check,
                      constrain_moments, constrain_expected, constrained_out)

# lower and upper bound for parameters and weights
xlb = (0,1,0,0,0)
xub = (1,10,10,10,10)
wlb = (0,1,1,1,1)
wub = (1,1,1,1,1)
# number of Dirac masses to use for each parameter
npts = (2,1,1,1,1) #NOTE: rv = (w0,w0,x0,x0,w1,x1,w2,x2,w3,x3,w4,x4)
index = (5,)       #NOTE: rv[5] -> x1
ordered = lambda constraint: sorting(index=(0,1))(constraint)

# moments and uncertainty in first parameter
a_ave = 5e-1
a_var = 5e-3
a_ave_err = 1e-3
a_var_err = 1e-4
# moments and uncertainty in second parameter
b_ave = None
b_var = None
b_ave_err = None
b_var_err = None
# moments and uncertainty in output
o_ave = None
o_var = None
o_ave_err = None
o_var_err = None


# constrain parameters at given index(es) to be ints
integer_indices = integers(ints=float, index=index)(lambda rv: rv)

## moment-based constraints ##
normcon = normalize_moments()
momcons = constrain_moments(a_ave, a_var, a_ave_err, a_var_err)
is_cons = constrained(a_ave, a_var, a_ave_err, a_var_err)
#momcon0 = constrain_moments(a_ave, a_var, a_ave_err, a_var_err, idx=0)
#momcon1 = constrain_moments(b_ave, b_var, b_ave_err, b_var_err, idx=1)
#is_con0 = constrained(a_ave, a_var, a_ave_err, a_var_err, idx=0)
#is_con1 = constrained(b_ave, b_var, b_ave_err, b_var_err, idx=1)
#is_cons = lambda c: bool(additive(is_con0)(is_con1)(c))

## index-based constraints ##
# impose constraints sequentially (faster, but assumes are decoupled)
#scons = outer(integer_indices)(flatten(npts)(outer(momcons)(normcon)))
scons = flatten(npts)(outer(momcons)(unflatten(npts)(ordered(flatten(npts)(normcon)))))
#scons = flatten(npts)(outer(momcon1)(outer(momcon0)(normcon)))

# impose constraints concurrently (slower, but safer)
#ccons = and_(flatten(npts)(normcon), flatten(npts)(momcons), integer_indices)
ccons = and_(ordered(flatten(npts)(normcon)), flatten(npts)(momcons))
#ccons = and_(flatten(npts)(normcon), flatten(npts)(momcon0), flatten(npts)(momcon1))

# check parameters (instead of measures)
iscon = check(npts)(is_cons)

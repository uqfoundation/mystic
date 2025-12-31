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
opts = dict(termination=COG(1e-6, 40), CrossProbability=0.9, ScalingFactor=0.9)
param = dict(solver=DifferentialEvolutionSolver2,
             npop=10, #XXX:npop
             maxiter=1000,
             maxfun=1e+6,
             x0=None, # use RandomInitialPoints
             nested=None, # don't use SetNested
             map=None, # don't use SetMapper
             stepmon=VerboseMonitor(1, label='output'), # monitor config
             #evalmon=Monitor(), # monitor config (re-initialized in solve)
             # kwds to pass directly to Solve(objective, **opt)
             opts=opts,
            )


from mystic.constraints import and_, integers
from mystic.coupler import outer, additive
from ouq_misc import (flatten, unflatten, normalize_moments, constrained, check,
                      constrain_moments, constrain_expected, constrained_out)

# lower and upper bound for parameters and weights
xlb = (20.0,0.0,2.1)
xub = (150.0,30.0,2.8)
wlb = (0,1,1)
wub = (1,1,1)
# number of Dirac masses to use for each parameter
npts = (2,1,1) #NOTE: rv = (w0,w0,x0,x0,w1,x1,w2,x2)
index = (5,)       #NOTE: rv[5] -> x1
# moments and uncertainty in first parameter
a_ave = None
a_var = None
a_ave_err = None
a_var_err = None
# moments and uncertainty in second parameter
b_ave = None
b_var = None
b_ave_err = None
b_var_err = None
# moments and uncertainty in output
o_ave = 6.5
o_var = None
o_ave_err = 1.0
o_var_err = None


# build a model representing 'truth'
from ouq_models import WrapModel
from surrogate import marc_surr as toy; nx = 3; ny = None; Ns = None
nargs = dict(nx=nx, ny=ny, rnd=(True if Ns else False))
model = WrapModel('model', toy, **nargs)

# set the bounds
bnd = MeasureBounds(xlb, xub, n=npts, wlb=wlb, wub=wub)

# constrain parameters at given index(es) to be ints
integer_indices = integers(ints=float, index=index)(lambda rv: rv)

## moment-based constraints ##
normcon = normalize_moments()
###momcons = constrain_moments(a_ave, a_var, a_ave_err, a_var_err)
###is_cons = constrained(a_ave, a_var, a_ave_err, a_var_err)
#momcon0 = constrain_moments(a_ave, a_var, a_ave_err, a_var_err, idx=0)
#momcon1 = constrain_moments(b_ave, b_var, b_ave_err, b_var_err, idx=1)
#is_con0 = constrained(a_ave, a_var, a_ave_err, a_var_err, idx=0)
#is_con1 = constrained(b_ave, b_var, b_ave_err, b_var_err, idx=1)
#is_cons = lambda c: bool(additive(is_con0)(is_con1)(c))
#momcons = constrain_expected(model, o_ave, o_var, o_ave_err, o_var_err, bnd, constraints=normcon)
momcons = constrain_expected(model, o_ave, o_var, o_ave_err, o_var_err, bnd)
is_cons = constrained_out(model, o_ave, o_var, o_ave_err, o_var_err)

## index-based constraints ##
# impose constraints sequentially (faster, but assumes are decoupled)
#scons = outer(integer_indices)(flatten(npts)(outer(momcons)(normcon)))
#scons = flatten(npts)(outer(momcon1)(outer(momcon0)(normcon)))
scons = flatten(npts)(outer(momcons)(normcon))
#scons = flatten(npts)(momcons)

# impose constraints concurrently (slower, but safer)
#ccons = and_(flatten(npts)(normcon), flatten(npts)(momcons), integer_indices)
#ccons = and_(flatten(npts)(normcon), flatten(npts)(momcon0), flatten(npts)(momcon1))
ccons = and_(flatten(npts)(normcon), flatten(npts)(momcons))
#ccons = scons

# check parameters (instead of measures)
iscon = check(npts)(is_cons)

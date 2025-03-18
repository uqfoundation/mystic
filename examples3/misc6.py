#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2025 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
misc user-defined items (solver configuration, moment constraints)
"""
from mystic.solvers import DifferentialEvolutionSolver2, NelderMeadSimplexSolver
from mystic.termination import ChangeOverGeneration as COG
from mystic.monitors import LoggingMonitor, VerboseLoggingMonitor
from mystic.bounds import Bounds, MeasureBounds

# kwds for solver #TODO: tune
opts = dict(termination=COG(1e-12, 200))
param = dict(solver=DifferentialEvolutionSolver2,
             npop=40,
             maxiter=5000,
             maxfun=1e+6,
             x0=None, # use RandomInitialPoints
             nested=None, # use SetNested
             map=None, # use SetMapper
             stepmon=VerboseLoggingMonitor(1,50, filename='log.txt'), # monitor
             #evalmon=LoggingMonitor(1, 'eval.txt'),# monitor (re-init in solve)
             # kwds to pass directly to Solve(objective, **opt)
             opts=opts,
            )

# kwds for sampling
kwds = dict(npts=500, ipts=None, itol=1e-8, iter=5)

from mystic.constraints import and_, integers, sorting
from mystic.coupler import outer, additive
from emulators import cost6, x6, bounds6, error6, wR
from ouq_misc import (flatten, unflatten, normalize_moments, constrained, check,
                      constrain_moments, constrain_expected, constrained_out)

# lower and upper bound for parameters and weights
xlb, xub = zip(*bounds6)
wlb = (0,0,0,0,0,0)
wub = (1,1,1,1,1,1)
# number of Dirac masses to use for each parameter
npts = (2,2,2,2,2,2) #NOTE: rv = (w0,w0,x0,x0,w1,w1,x1,x1,w2,w2,x2,x2,w3,w3,x3,x3,w4,w4,x4,x4,w5,w5,x5,x5)
index = (2,3,6,7,10,11,14,15,18,19,22,23)  #NOTE: rv[index] -> x0,x0,x1,x1,x2,x2,x3,x3,x4,x4,x5,x5
ordered = lambda constraint: sorting(index=(0,1))(sorting(index=(4,5))(sorting(index=(8,9))(sorting(index=(12,13))(sorting(index=(16,17))(sorting(index=(20,21))(constraint))))))

# moments and uncertainty in first parameter
a_ave = x6[0]
a_var = .5 * error6[0]**2
a_ave_err = 2 * a_var
a_var_err = a_var
# moments and uncertainty in second parameter
b_ave = x6[1]
b_var = .5 * error6[1]**2
b_ave_err = 2 * b_var
b_var_err = b_var
# moments and uncertainty in third parameter
c_ave = x6[2]
c_var = .5 * error6[2]**2
c_ave_err = 2 * c_var
c_var_err = c_var
# moments and uncertainty in fourth parameter
d_ave = x6[3]
d_var = .5 * error6[3]**2
d_ave_err = 2 * d_var
d_var_err = d_var
# moments and uncertainty in fifth parameter
e_ave = x6[4]
e_var = .5 * error6[4]**2
e_ave_err = 2 * e_var
e_var_err = e_var
# moments and uncertainty in sixth parameter
f_ave = x6[5]
f_var = .5 * error6[5]**2
f_ave_err = 2 * f_var
f_var_err = f_var
# moments and uncertainty in output
o_ave = None
o_var = None
o_ave_err = None
o_var_err = None


# build a model representing 'truth' F(x)
from ouq_models import WrapModel
nx = 6; ny = None
Ns = None #500 # number of samples of F(x) in the objective
nargs = dict(nx=nx, ny=ny, rnd=(True if Ns else False))
model = WrapModel('model', cost6, **nargs)

# set the bounds
bnd = MeasureBounds(xlb, xub, n=npts, wlb=wlb, wub=wub)

# constrain parameters at given index(es) to be ints
integer_indices = integers(ints=float, index=index)(lambda rv: rv)

## moment-based constraints ##
normcon = normalize_moments()
momcon0 = constrain_moments(a_ave, a_var, a_ave_err, a_var_err, idx=0)
momcon1 = constrain_moments(b_ave, b_var, b_ave_err, b_var_err, idx=1)
momcon2 = constrain_moments(c_ave, c_var, c_ave_err, c_var_err, idx=2)
momcon3 = constrain_moments(d_ave, d_var, d_ave_err, d_var_err, idx=3)
momcon4 = constrain_moments(e_ave, e_var, e_ave_err, e_var_err, idx=4)
momcon5 = constrain_moments(f_ave, f_var, f_ave_err, f_var_err, idx=5)
is_con0 = constrained(a_ave, a_var, a_ave_err, a_var_err, idx=0)
is_con1 = constrained(b_ave, b_var, b_ave_err, b_var_err, idx=1)
is_con2 = constrained(c_ave, c_var, c_ave_err, c_var_err, idx=2)
is_con3 = constrained(d_ave, d_var, d_ave_err, d_var_err, idx=3)
is_con4 = constrained(e_ave, e_var, e_ave_err, e_var_err, idx=4)
is_con5 = constrained(f_ave, f_var, f_ave_err, f_var_err, idx=5)
is_cons = lambda c: bool(additive(is_con5)(additive(is_con4)(additive(is_con3)(additive(is_con2)(additive(is_con1)(is_con0)))))(c))

## position-based constraints ##
# impose constraints sequentially (faster, but assumes are decoupled)
scons = flatten(npts)(outer(momcon5)(outer(momcon4)(outer(momcon3)(outer(momcon2)(outer(momcon1)(outer(momcon0)(unflatten(npts)(ordered(flatten(npts)(normcon))))))))))

# impose constraints concurrently (slower, but safer)
ccons = and_(ordered(flatten(npts)(normcon)), flatten(npts)(momcon0), flatten(npts)(momcon1), flatten(npts)(momcon2), flatten(npts)(momcon3), flatten(npts)(momcon4), flatten(npts)(momcon5))

# check parameters (instead of measures)
iscon = check(npts)(is_cons)

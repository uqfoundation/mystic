#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2021 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
misc user-defined items (solver configuration, moment constraints)
"""
from mystic.solvers import DifferentialEvolutionSolver2
from mystic.monitors import VerboseMonitor, Monitor
from mystic.termination import ChangeOverGeneration as COG

# kwds for solver
opts = dict(termination=COG(1e-10, 100))
param = dict(solver=DifferentialEvolutionSolver2,
             npop=80,
             maxiter=1500,
             maxfun=1e+6,
             x0=None, # use RandomInitialPoints
             nested=None, # don't use SetNested
             pool=None, # don't use SetMapper
             stepmon=VerboseMonitor(1, label='output'), # monitor config
             evalmon=Monitor(), # monitor config (re-initialized in solve)
             # kwds to pass directly to Solve(objective, **opt)
             opts=opts,
            )


from mystic.math.discrete import product_measure
from mystic.math import almostEqual as almost
from mystic.constraints import and_, integers
from mystic.coupler import outer

# lower and upper bound for parameter weights
wlb = (0,1,1,1,1)
wub = (1,1,1,1,1)
# number of Dirac masses to use for each parameter
npts = (2,1,1,1,1) #NOTE: rv = (w0,w0,x0,x0,w1,x1,w2,x2,w3,x3,w4,x4)
index = (5,)       #NOTE: rv[5] -> x1
# moments and uncertainty in first parameter
a_ave = 5e-1
a_var = 5e-3
a_ave_err = 1e-3
a_var_err = 1e-4
b_ave = None
b_var = None
b_ave_err = None
b_var_err = None


def flatten(npts):
    'convert a moment constraint to a "flattened" constraint'
    def dec(f):
        def func(rv):
            c = product_measure().load(rv, npts)
            c = f(c)
            return c.flatten()
        return func
    return dec


def normalize_moments(mass=1.0, tol=1e-18, rel=1e-7):
    'normalize (using weights) on all measures'
    def func(c):
        for measure in c:
            if not almost(float(measure.mass), mass, tol=tol, rel=rel):
                measure.normalize()
        return c
    return func


def constrain_moments(ave=None, var=None, ave_err=None, var_err=None, idx=0):
    'impose mean and variance constraints on the selected measure'
    if ave is None: ave = float('nan')
    if var is None: var = float('nan')
    if ave_err is None: ave_err = 0
    if var_err is None: var_err = 0
    def func(c):
        E = float(c[idx].mean)
        if E > (ave + ave_err) or E < (ave - ave_err):
            c[idx].mean = ave
        E = float(c[idx].var)
        if E > (var + var_err) or E < (var - var_err):
            c[idx].var = var
        return c
    return func


@integers(ints=float, index=index)
def integer_indices(rv):
    'constrain parameters at given index(es) to be ints'
    return rv


def constrained_integers(index=()):
    'check integer constraint is properly applied'
    def func(rv):
        return all(int(j) == j for i,j in enumerate(rv) if i in index)
    return func


def constrained(ave=None, var=None, ave_err=None, var_err=None, idx=0, debug=False):
    'check mean and variance on the selected measure are properly constrained'
    if ave is None: ave = float('nan')
    if var is None: var = float('nan')
    if ave_err is None: ave_err = 0
    if var_err is None: var_err = 0
    def func(c):
        E = float(c[idx].mean)
        if E > (ave + ave_err) or E < (ave - ave_err):
            if debug: print("skipping mean: %s" % E)
            return False
        E = float(c[idx].var)
        if E > (var + var_err) or E < (var - var_err):
            if debug: print("skipping var: %s" % E)
            return False
        return True
    return func


def check(npts):
    'convert a moment check to a "flattened" check'
    def dec(f):
        def func(rv):
            c = product_measure().load(rv, npts)
            return f(c)
        return func
    return dec


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
scons = flatten(npts)(outer(momcons)(normcon))
#scons = flatten(npts)(outer(momcon1)(outer(momcon0)(normcon)))

# impose constraints concurrently (slower, but safer)
#ccons = and_(flatten(npts)(normcon), flatten(npts)(momcons), integer_indices)
ccons = and_(flatten(npts)(normcon), flatten(npts)(momcons))
#ccons = and_(flatten(npts)(normcon), flatten(npts)(momcon0), flatten(npts)(momcon1))

# check parameters (instead of measures)
iscon = check(npts)(is_cons)
#rvcon = constrained_integers(index)

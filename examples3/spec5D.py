#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
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
             npop=80, #XXX:npop
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


from mystic.math.discrete import product_measure
from mystic.math import almostEqual as almost
from mystic.constraints import and_, integers
from mystic.coupler import outer, additive

# lower and upper bound for parameters and weights
xlb = (0,1,0,0,0)
xub = (1,10,10,10,10)
wlb = (0,1,1,1,1)
wub = (1,1,1,1,1)
# number of Dirac masses to use for each parameter
npts = (2,1,1,1,1) #NOTE: rv = (w0,w0,x0,x0,w1,x1,w2,x2,w3,x3,w4,x4)
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
o_ave = 11.0
o_var = None
o_ave_err = 1.0
o_var_err = None


def flatten(npts):
    'convert a moment constraint to a "flattened" constraint'
    def dec(f):
        def func(rv):
            c = product_measure().load(rv, npts)
            c = f(c)
            return c.flatten()
        return func
    return dec


def unflatten(npts):
    'convert a "flattened" constraint to a moment constraint'
    def dec(f):
        def func(c):
            return product_measure().load(f(c.flatten()), npts)
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


#NOTE: model has single-value output
def constrain_expected(model, ave=None, var=None, ave_err=None, var_err=None, bounds=None, **kwds):
    'impose mean and variance constraints on the measure'
    if ave is None: ave = float('nan')
    if var is None: var = float('nan'); kwds['k'] = 0
    if ave_err is None: ave_err = 0
    if var_err is None: var_err = 0
    if 'npop' not in kwds: kwds['npop'] = 200 #XXX: better default?
    if isinstance(bounds, Bounds): bounds = (bounds.xlower,bounds.xupper)
    samples = None #kwds.pop('samples', None) #NOTE: int or None
    if samples is None:
        def func(c):
            E = float(c.expect(model))
            Ev = float(c.expect_var(model))
            if E > (ave + ave_err) or E < (ave - ave_err) or \
               Ev > (var + var_err) or Ev < (var - var_err):
                c.set_expect_mean_and_var((ave,var), model, bounds, tol=(ave_err,var_err), **kwds) #NOTE: debug, maxiter, k
            return c
    else:
        def func(c):
            E = float(c.sampled_expect(model, samples)) #TODO: map
            Ev = float(c.sampled_variance(model, samples)) #TODO: map
            if E > (ave + ave_err) or E < (ave - ave_err) or \
               Ev > (var + var_err) or Ev < (var - var_err):
                c.set_expect_mean_and_var((ave,var), model, bounds, tol=(ave_err,var_err), **kwds) #NOTE: debug, maxiter, k #FIXME: Ns=samples
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


#NOTE: model has single-value output
def constrained_out(model, ave=None, var=None, ave_err=None, var_err=None, debug=False, **kwds):
    'check the expected output is properly constrained'
    if ave is None: ave = float('nan')
    if var is None: var = float('nan')
    if ave_err is None: ave_err = 0
    if var_err is None: var_err = 0
    samples = None #kwds.pop('samples', None) #NOTE: int or None
    if samples is None:
        def func(c):
            E = float(c.expect(model))
            Ev = float(c.expect_var(model))
            if E > (ave + ave_err) or E < (ave - ave_err) or \
               Ev > (var + var_err) or Ev < (var - var_err):
                if debug: print("skipping expected value,var: %s, %s" % (E,Ev))
                return False
            return True
    else:
        def func(c):
            E = float(c.sampled_expect(model, samples)) #TODO: map
            Ev = float(c.sampled_variance(model, samples)) #TODO: map
            if E > (ave + ave_err) or E < (ave - ave_err) or \
               Ev > (var + var_err) or Ev < (var - var_err):
                if debug: print("skipping expected value,var: %s, %s" % (E,Ev))
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


# build a model representing 'truth'
from ouq_models import WrapModel
from toys import function5 as toy; nx = 5; ny = None; Ns = None
nargs = dict(nx=nx, ny=ny, rnd=(True if Ns else False))
model = WrapModel('model', toy, **nargs)


# set the bounds
bnd = MeasureBounds(xlb, xub, n=npts, wlb=wlb, wub=wub)


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
#rvcon = constrained_integers(index)

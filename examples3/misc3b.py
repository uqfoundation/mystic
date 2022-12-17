#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
misc user-defined items (solver configuration, moment constraints)
"""
from mystic.solvers import DifferentialEvolutionSolver2, LatticeSolver, NelderMeadSimplexSolver
from mystic.termination import ChangeOverGeneration as COG
from mystic.monitors import Monitor, VerboseMonitor, LoggingMonitor, VerboseLoggingMonitor
from mystic.bounds import Bounds, MeasureBounds

# kwds for solver #TODO: tune
opts = dict(termination=COG(1e-6, 100))
param = dict(solver=DifferentialEvolutionSolver2,
             npop=10,
             maxiter=2000,
             maxfun=1e+6,
             x0=None, # use RandomInitialPoints
             nested=None, # use SetNested
             map=None, # use SetMapper
             stepmon=VerboseLoggingMonitor(1,10, filename='log.txt'), # monitor
             #evalmon=LoggingMonitor(1, 'eval.txt'),# monitor (re-init in solve)
             # kwds to pass directly to Solve(objective, **opt)
             opts=opts,
            )

# kwds for sampling
kwds = dict(npts=500, ipts=4, itol=1e-8, iter=5)

from mystic.math.discrete import product_measure
from mystic.math import almostEqual as almost
from mystic.constraints import and_, integers
from mystic.coupler import outer, additive
from emulators import cost3, x3, bounds3, error3, a_beta, a_beta_error
from mystic import suppressed

# lower and upper bound for parameters and weights
xlb, xub = zip(*bounds3)
wlb = (0,0,0)
wub = (1,1,1)
# number of Dirac masses to use for each parameter
npts = (2,2,2) #NOTE: rv = (w0,w0,x0,x0,w1,w1,x1,x1,w2,w2,x2,x2)
index = (2,3,6,7,10,11)  #NOTE: rv[index] -> x0,x0,x1,x1,x2,x2
# moments and uncertainty in first parameter
a_ave = x3[0]
a_var = .5 * error3[0]**2
a_ave_err = 2 * a_var
a_var_err = a_var
# moments and uncertainty in second parameter
b_ave = x3[1]
b_var = .5 * error3[1]**2
b_ave_err = 2 * b_var
b_var_err = b_var
# moments and uncertainty in third parameter
c_ave = x3[2]
c_var = .5 * error3[2]**2
c_ave_err = 2 * c_var
c_var_err = c_var
# moments and uncertainty in output
o_ave = None
o_var = .5 * a_beta_error**2
o_ave_err = None
o_var_err = o_var


def flatten(npts):
    'convert a moment constraint to a "flattened" constraint'
    def dec(f):
        #@suppressed(1e-3)
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
    if 'npop' not in kwds: kwds['npop'] = 20 #XXX: better default?
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


# build a model representing 'truth' F(x)
from ouq_models import WrapModel
nx = 3; ny = None
Ns = None #500 # number of samples of F(x) in the objective
nargs = dict(nx=nx, ny=ny, rnd=(True if Ns else False))
model = WrapModel('model', cost3, **nargs)

# set the bounds
bnd = MeasureBounds(xlb, xub, n=npts, wlb=wlb, wub=wub)

## moment-based constraints ##
normcon = normalize_moments()
momcon0 = constrain_moments(a_ave, a_var, a_ave_err, a_var_err, idx=0)
momcon1 = constrain_moments(b_ave, b_var, b_ave_err, b_var_err, idx=1)
momcon2 = constrain_moments(c_ave, c_var, c_ave_err, c_var_err, idx=2)
is_con0 = constrained(a_ave, a_var, a_ave_err, a_var_err, idx=0)
is_con1 = constrained(b_ave, b_var, b_ave_err, b_var_err, idx=1)
is_con2 = constrained(c_ave, c_var, c_ave_err, c_var_err, idx=2)
is_ocon = constrained_out(model, ave=o_ave, var=o_var, ave_err=o_ave_err, var_err=o_var_err, debug=False)
is_cons = lambda c: bool(additive(is_ocon)(additive(is_con2)(additive(is_con1)(is_con0)))(c))

## position-based constraints ##
# impose constraints sequentially (faster, but assumes are decoupled)
_scons = outer(momcon2)(outer(momcon1)(outer(momcon0)(normcon)))
scons = flatten(npts)(constrain_expected(model, ave=o_ave, var=o_var, ave_err=o_ave_err, var_err=o_var_err, bounds=bnd, constraints=_scons, maxiter=50))
#scons = flatten(npts)(outer(constrain_expected(model, ave=o_ave, var=o_var, ave_err=o_ave_err, var_err=o_var_err, bounds=bnd))(_scons))

# impose constraints concurrently (slower, but safer)
#_ccons = unflatten(npts)(and_(flatten(npts)(normcon), flatten(npts)(momcon0), flatten(npts)(momcon1), flatten(npts)(momcon2))) #FIXME: broadcasting error
#ccons = flatten(npts)(constrain_expected(model, ave=o_ave, var=o_var, ave_err=o_ave_err, var_err=o_var_err, bounds=bnd, constraints=_ccons, maxiter=50))
_ccons = flatten(npts)(constrain_expected(model, ave=o_ave, var=o_var, ave_err=o_ave_err, var_err=o_var_err, bounds=bnd, maxiter=1000, debug=False))
ccons = and_(flatten(npts)(normcon), flatten(npts)(momcon0), flatten(npts)(momcon1), flatten(npts)(momcon2), _ccons)

# check parameters (instead of measures)
iscon = check(npts)(is_cons)

#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2024-2025 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
misc moment constraints
"""
from mystic.math.discrete import product_measure
from mystic.math import almostEqual as almost
from mystic.bounds import Bounds
#from mystic import suppressed

def flatten(npts):
    'convert a moment constraint to a "flattened" constraint'
    def dec(f):
        #@suppressed(1e-3)
        def func(rv):
            return f(product_measure().load(rv, npts)).flatten()
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
            return f(product_measure().load(rv, npts))
        return func
    return dec


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


def constrained_integers(index=()):
    'check integer constraint is properly applied'
    def func(rv):
        return all(int(j) == j for i,j in enumerate(rv) if i in index)
    return func

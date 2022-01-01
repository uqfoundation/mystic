#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Function Couplers

These methods can be used to couple two functions together,
and represent some common patterns found in applying constraints
and penalty methods.

For example, the "outer" method called on y = f(x), with outer=c(x),
will convert y = f(x) to y' = c(f(x)).  Similarly, the "inner" method
called on y = f(x), with inner=c(x), will convert y = f(x) to  y' = f(c(x)).
"""

def outer(outer=lambda x:x, args=None, kwds=None):
    """wrap a function around another function: convert y = f(x) to y' = c(f(x))

This is a useful function for nesting one constraint in another constraint.
A constraints function takes an iterable x as input, returning a modified x.

    For example:
    >>> def squared(x):
    ...   return x**2             
    ... 
    >>> # equivalent to: ((x+1)**2) 
    >>> @outer(squared)
    ... def constrain(x):
    ...   return x+1
    ... 
    >>> from numpy import array
    >>> x = array([1,2,3,4,5])
    >>> constrain(x)
    array([ 4,  9, 16, 25, 36])
    """
    if args is None: args=()
    if kwds is None: kwds={}
    def dec(f):
        def func(x, *argz, **kwdz):
            return outer(f(x, *argz, **kwdz), *args, **kwds)
        return func
    return dec


def inner(inner=lambda x:x, args=None, kwds=None):
    """nest a function within another function: convert y = f(x) to y' = f(c(x))

This is a useful function for nesting one constraint in another constraint.
A constraints function takes an iterable x as input, returning a modified x.
The "inner" coupler is utilized by mystic.solvers to bind constraints to a cost
function; thus the constraints are imposed every cost function evaluation.

    For example:
    >>> def squared(x):
    ...   return x**2             
    ... 
    >>> # equivalent to: ((x**2)+1) 
    >>> @inner(squared)
    ... def constrain(x):
    ...   return x+1
    ... 
    >>> from numpy import array
    >>> x = array([1,2,3,4,5])
    >>> constrain(x)
    array([ 2,  5, 10, 17, 26])
    """
    if args is None: args=()
    if kwds is None: kwds={}
    def dec(f):
        def func(x, *argz, **kwdz):
            return f(inner(x, *args, **kwds), *argz, **kwdz)
        return func
    return dec


def inner_proxy(inner=lambda x:x, args=None, kwds=None):
    """nest a function within another function: convert y = f(x) to y' = f(c(x))

This is a useful function for nesting one constraint in another constraint.
A constraints function takes an iterable x as input, returning a modified x.

This function applies the "inner" coupler pattern. However, it does not preserve
decorated function signature -- it passes args and kwds to the inner function
instead of the decorated function.
    """
    if args is None: args=()
    if kwds is None: kwds={}
    def dec(f):
        def func(*argz, **kwdz):
            return f(inner(*argz, **kwdz), *args, **kwds)
        return func
    return dec


def outer_proxy(outer=lambda x:x, args=None, kwds=None):
    """wrap a function around another function: convert y = f(x) to y' = c(f(x))

This is a useful function for nesting one constraint in another constraint.
A constraints function takes an iterable x as input, returning a modified x.

This function applies the "outer" coupler pattern. However, it does not preserve
decorated function signature -- it passes args and kwds to the outer function
instead of the decorated function.
    """
    if args is None: args=()
    if kwds is None: kwds={}
    def dec(f):
        def func(x, *argz, **kwdz):
            return outer(f(x, *args, **kwds), *argz, **kwdz)
        return func
    return dec


####################################################
def additive_proxy(penalty=lambda x:0.0, args=None, kwds=None):
    """penalize a function with another function: y = f(x) to y' = f(x) + p(x)
    
This is useful, for example, in penalizing a cost function where the constraints
are violated; thus, the satisfying the constraints will be preferred at every
cost function evaluation.

This function does not preserve decorated function signature, but passes args
and kwds to the penalty function.
    """
    if args is None: args=()
    if kwds is None: kwds={}
    def dec(f):
        def func(x, *argz, **kwdz):
            return f(x, *args, **kwds) + penalty(x, *argz, **kwdz)
        return func
    return dec

def additive(penalty=lambda x:0.0, args=None, kwds=None):
    """penalize a function with another function: y = f(x) to y' = f(x) + p(x)
    
This is useful, for example, in penalizing a cost function where the constraints
are violated; thus, the satisfying the constraints will be preferred at every
cost function evaluation.

    For example:
    >>> def squared(x):
    ...   return x**2             
    ... 
    >>> # equivalent to: (x+1) + (x**2) 
    >>> @additive(squared)
    ... def constrain(x):
    ...   return x+1
    ... 
    >>> from numpy import array
    >>> x = array([1,2,3,4,5])
    >>> constrain(x)
    array([ 3,  7, 13, 21, 31])
    """
    if args is None: args=()
    if kwds is None: kwds={}
    def dec(f):
        def func(x, *argz, **kwdz):
            return f(x, *argz, **kwdz) + penalty(x, *args, **kwds)
        return func
    return dec

#XXX: can do multiple @additive; but better is compound penalty with And,Or,..?
#XXX: create a counter for n += 1 ?

# penalty coupling "language"  #XXX: was initially 'constraints.combined'
#XXX: should have the same interface as 'inner' and 'outer'? ...or ???
def and_(*penalties, **settings): #XXX: not a decorator, should be?
    """combine several penalties into a single penalty function by summation

Inputs:
    penalties -- penalty functions

Additional Inputs:
    ptype -- penalty function type [default: linear_equality]
    args -- arguments for the penalty function [default: ()]
    kwds -- keyword arguments for the penalty function [default: {}]
    k -- penalty multiplier [default: 1]
    h -- iterative multiplier [default: 5]

NOTE: The defaults provide a linear combination of the individual penalties
    without any scaling. A different ptype (from 'mystic.penalty') will
    apply a nonlinear scaling to the combined penalty, while a different
    k will apply a linear scaling.

NOTE: This function is also useful for combining constraints solvers
    into a single constraints solver, however can not do so directly.  
    Constraints solvers must first be converted to penalty functions
    (i.e. with 'as_penalty'), then combined, then can be converted to
    a constraints solver (i.e. with 'as_constraint'). The resulting
    constraints will likely be more expensive to evaluate and less
    accurate than writing the constraints solver from scratch.
    """
    k = settings.setdefault('k', 1)
    if k is None: del settings['k']
    if 'ptype' in settings:
        ptype = settings['ptype']; del settings['ptype']
    else: ptype = None
    if ptype is None:
        from mystic.penalty import linear_equality as ptype
    penalty = lambda x: sum(p(x) for p in penalties) #XXX: ZeroDivisionError?
    pf = ptype(penalty, **settings)(lambda x:0.)
    pfdoc = "\n-- AND --\n".join(p.__doc__ for p in penalties if p.__doc__)
    pfdoc = '{ '+ pfdoc +' }' if pfdoc else pfdoc #XXX: can be {p} w/no AND
    pf.__doc__ = pfdoc.rstrip('\n')
    pf.__name__ = 'penalty'
    return pf


#XXX: should have the same interface as 'inner' and 'outer'? ...or ???
def or_(*penalties, **settings): #XXX: not a decorator, should be?
    """create a single penalty that selects the minimum of several penalties

Inputs:
    penalties -- penalty functions

Additional Inputs:
    ptype -- penalty function type [default: linear_equality]
    args -- arguments for the penalty function [default: ()]
    kwds -- keyword arguments for the penalty function [default: {}]
    k -- penalty multiplier [default: 1]
    h -- iterative multiplier [default: 5]

NOTE: The defaults provide a linear combination of the individual penalties
    without any scaling. A different ptype (from 'mystic.penalty') will
    apply a nonlinear scaling to the combined penalty, while a different
    k will apply a linear scaling.

NOTE: This function is also useful for combining constraints solvers
    into a single constraints solver, however can not do so directly.  
    Constraints solvers must first be converted to penalty functions
    (i.e. with 'as_penalty'), then combined, then can be converted to
    a constraints solver (i.e. with 'as_constraint'). The resulting
    constraints will likely be more expensive to evaluate and less
    accurate than writing the constraints solver from scratch.
    """
    k = settings.setdefault('k', 1)
    if k is None: del settings['k']
    if 'ptype' in settings:
        ptype = settings['ptype']; del settings['ptype']
    else: ptype = None
    if ptype is None:
        from mystic.penalty import linear_equality as ptype
    penalty = lambda x: min(p(x) for p in penalties) #XXX: ZeroDivisionError?
    pf = ptype(penalty, **settings)(lambda x:0.)
    pfdoc = "\n-- OR --\n".join(p.__doc__ for p in penalties if p.__doc__)
    pfdoc = '[ '+ pfdoc +' ]' if pfdoc else pfdoc #XXX: can be [p] w/no OR
    pf.__doc__ = pfdoc.rstrip('\n')
    pf.__name__ = 'penalty'
    return pf


#XXX: should have the same interface as 'inner' and 'outer'? ...or ???
def not_(penalty, **settings): #XXX: not a decorator, should be?
    """invert, so penalizes the region where the given penalty is valid

Inputs:
    penalty -- a penalty function

Additional Inputs:
    ptype -- penalty function type [default: linear_equality]
    args -- arguments for the penalty function [default: ()]
    kwds -- keyword arguments for the penalty function [default: {}]
    k -- penalty multiplier [default: 1]
    h -- iterative multiplier [default: 5]
    """
    k = settings.setdefault('k', 1)
    if k is None: del settings['k']
    if 'ptype' in settings:
        ptype = settings['ptype']; del settings['ptype']
    else: ptype = None
    if ptype is None:
        import mystic.penalty as mp
        try:
            ptype = getattr(mp, penalty.ptype)
        except AttributeError: 
            ptype = mp.linear_equality
    try:
        condition = penalty.func # is a penalty
    except AttributeError:
        condition = penalty # is a raw condition
    if ptype.__name__.endswith('_inequality'):
        _penalty = lambda x: 0 - condition(x) #XXX: ZeroDivisionError?
    else:
        _penalty = lambda x: not condition(x) #XXX: ZeroDivisionError?
    pf = ptype(_penalty, **settings)(lambda x:0.)
    pfdoc = 'NOT( '+ penalty.__doc__ +' )' if penalty.__doc__ else ""
    pf.__doc__ = pfdoc.rstrip('\n')
    pf.__name__ = 'penalty'
    return pf



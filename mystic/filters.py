#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2019 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Input/output 'filters'
"""

# 'filters'
def identity(x):
    """identity filter, F, where F(x) yields x"""
    return x


def component(n, multiplier = 1.):
    """component filter, F, where F(x) yields x[n]"""
    def _(x):
        try:
            from numpy import asarray
            xx = asarray(x)
            xx = multiplier * xx[n,:]
            if isinstance(x[n],list):
                xx = xx.tolist()
            elif isinstance(x[n],tuple):
                xx = tuple(xx.tolist())
            elif type(x[n]) != type(xx):
                xx = type(x[n])(xx)
            return xx
        except IndexError:
            return multiplier * x[n]
    return _


# 'checkers'
def null_check(params, evalpts, *args):
    """null validity check"""
    return None


# monitor masks and filters
def generate_mask(x=None, y=None):
    """generate a monitor mask, based on constraints for x and/or y

    x: mystic.constraint function where x' = c(x) and x is parameter array
    y: mystic.constraint function where [y'] = c([y]) and y is cost array

    returns a function that produces a masking function for monitors, where
    list[bool] = mask(monitor), with True where constraints are satisfied

    For example:
    >>> mon = Monitor()
    >>> mon([0.0,0.5,1.0],2)
    >>> mon([2.0,3.0,4.0],3)
    >>> mon([4.0,5.0,6.0],4)
    >>> mon([5.0,5.5,6.5],6)
    >>> 
    >>> @impose_bounds((0,5))
    ... def inputs(x):
    ...   return x
    ... 
    >>> generate_mask(inputs)(mon)
    [True, True, False, False]
    >>> generate_mask(y=inputs)(mon)
    [True, True, True, False]
    >>>
    >>> @integers()
    ... def identity(x):
    ...   return x
    ... 
    >>> generate_mask(identity)(mon)
    [False, True, True, False]
    >>> generate_mask(y=identity)(mon)
    [True, True, True, True]
    """
    def func(mon):
        from numpy import ones_like
        res = ones_like(mon._y, dtype=bool)
        if x is not None:
            res = res & [i == x(i) for i in mon._x]
        if y is not None:
            res = res & [(i,) == y((i,)) for i in mon._y]
        return res.tolist()
    return func


def generate_filter(mask):
    """generate a monitor filter from a monitor masking function

    mask: masking function (built with generate_mask) or a boolean mask

    returns a function that filters a monitor, based on the given mask
    monitor' = filter(monitor), where filter removes values where mask is False

    For example:
    >>> mon = Monitor()
    >>> mon([0.0,0.5,1.0],2)
    >>> mon([2.0,3.0,4.0],3)
    >>> mon([4.0,5.0,6.0],4)
    >>> mon([5.0,5.5,6.5],6)
    >>> 
    >>> @impose_bounds((0,5))
    ... def inputs(x):
    ...   return x
    ... 
    >>> m = generate_filter(generate_mask(inputs))(mon)
    >>> m._x
    [[0.0, 0.5, 1.0], [2.0, 3.0, 4.0]]
    >>> m._y
    [2, 3]
    >>>
    >>> @integers()
    ... def identity(x):
    ...   return x
    ... 
    >>> m = generate_filter(generate_mask(identity))(mon)
    >>> m._x
    [[2.0, 3.0, 4.0], [4.0, 5.0, 6.0]]
    >>> m._y
    [3, 4]
    """
    def func(mon):
        if not hasattr(mask, '__call__'):
            _mask = mask
        else:
            _mask = mask(mon)
        from copy import copy
        m = copy(mon)
        from numpy import array
        m._x = array(m._x)[_mask].tolist()
        m._y = array(m._y)[_mask].tolist()
        try:
            m._id = array(m._id)[_mask].tolist()
        except IndexError:
            pass #XXX: m._id = []
        return m
    return func


# EOF

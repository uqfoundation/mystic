#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
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
def generate_mask(cx=None, cy=None):
    """generate a data mask, based on constraints for x and/or y

    cx: mystic.constraint function where x' = cx(x) and x is parameter array
    cy: mystic.constraint function where [y'] = cy([y]) and y is cost array

    returns a masking function for (x,y) data or monitors, where
    list[bool] = mask(x,y), with True where constraints are satisfied

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
    def func(x, z=None):
        _x = getattr(x, '_x', x)
        _y = x._y if z is None else z
        _x = _x.tolist() if hasattr(_x, 'tolist') else _x
        _y = _y.tolist() if hasattr(_y, 'tolist') else _y
        from numpy import ones_like
        res = ones_like(_y, dtype=bool)
        if cx is not None:
            res = res & [i == cx(i) for i in _x]
        if cy is not None:
            res = res & [(i,) == cy((i,)) for i in _y]
        return res.tolist()
    return func


def generate_filter(mask):
    """generate a data filter from a data masking function

    mask: masking function (built with generate_mask) or a boolean mask

    returns a function that filters (x,y) or a monitor, based on the given mask
    x',y' = filter(x,y), where filter removes values where mask is False

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
    def func(x, z=None):
        if not hasattr(mask, '__call__'):
            _mask = mask
        else:
            _mask = mask(x, z)
        if z is None and hasattr(x, '_x') and hasattr(x, '_y'):
            from copy import copy
            m = copy(x)
            mon = True
        else:
            from mystic.monitors import Monitor
            m = Monitor()
            m._x,m._y = x,z
            mon = False
        ax = True if hasattr(m._x, 'tolist') else False
        ay = True if hasattr(m._y, 'tolist') else False
        from numpy import asarray
        m._x = asarray(m._x)[_mask]
        m._y = asarray(m._y)[_mask]
        if not ax: m._x = m._x.tolist()
        if not ay: m._y = m._y.tolist()
        if mon:
            try:
                ai = True if hasattr(m._id, 'tolist') else False
                m._id = array(m._id)[_mask]
                if not ai: m._id = m._id.tolist()
            except IndexError:
                pass #XXX: m._id = []
            return m
        return m._x, m._y
    return func


# EOF

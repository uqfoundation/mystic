#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
test functions, including those with multivalued returns and the axis keyword
"""

def function5x3(x):
    """a 5x3 test function

  For example:
    >>> function5x3([1,2,3,4,5])
    [13.922847983320086, -13.730919862656235, -18.0]
    """
    a,b,c,d,e = x
    y = [
        a + b*abs(c*d**2 - (e/b)**2)**.5, 
        a - b*abs(c*d**2 + (e/b)**2)**.5, 
        a - abs(b*(c*d - (e/b)))
    ]
    return y


def cost5x3(x, axis=None):
    """a 5x3 test function, using axis

  For example:
    >>> cost5x3([1,2,3,4,5], axis=None)
    [13.922847983320086, -13.730919862656235, -18.0]
    >>> cost5x3([1,2,3,4,5], axis=0)
    13.922847983320086
    """
    y = function5x3(x)
    return y if axis is None else y[axis]


def function5x1(x):
    """a 5x1 test function

  For example:
    >>> function5x1([1,2,3,4,5])
    [13.922847983320086]
    """
    return function5x3(x)[:1]


def cost5x1(x, axis=None):
    """a 5x1 test function, using axis

  For example:
    >>> cost5x1([1,2,3,4,5], axis=None)
    [13.922847983320086]
    >>> cost5x1([1,2,3,4,5], axis=0)
    13.922847983320086
    """
    y = function5x1(x)
    return y if axis is None else y[axis]


def function5(x):
    """a 5d test function

  For example:
    >>> function5([1,2,3,4,5])
    13.922847983320086
    """
    return function5x3(x)[0]


def cost5(x, axis=None): #XXX: axis ignored, or better raise if not None?
    """a 5d test function, with axis keyword (ignored)

  For example:
    >>> cost5([1,2,3,4,5], axis=None)
    13.922847983320086
    >>> cost5([1,2,3,4,5], axis=0)
    13.922847983320086
    """
    y = function5(x)
    return y


def wrap(**kwds):
    """reduce dimensionality of the input by providing fixed values

  For example:
    >>> wrap(a=0,c=2)(function5)([1,3,4])
    1.4142135623730951
    >>> function5([0,1,2,3,4])
    1.4142135623730951
    >>> 
    >>> wrap(d=0,e=2)(function5)([1,3,4])
    3.0
    >>> function5([1,3,4,0,2])
    3.0

  NOTE: wraps functions with a single argument (y = f(x), not y = f(x, axis))
  NOTE: assumes wrapped function has 5d input, with x = [a, b, c, d, e]
    """
    var = list('abcde')
    def func(f):
        def dec(x):
            i = iter(x)
            y = []
            for v in var:
                k = kwds.get(v, None)
                y.extend((next(i) if k is None else k,))
            return f(y)
        return dec
    return func


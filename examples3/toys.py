#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
test functions, including those with multivalued returns and the axis keyword
"""

def function5x3(x):
    "a 5x3 test function"
    a,b,c,d,e = x
    y = [
        a + b*abs(c*d**2 - (e/b)**2)**.5, 
        a - b*abs(c*d**2 + (e/b)**2)**.5, 
        a - abs(b*(c*d - (e/b)))
    ]
    return y


def cost5x3(x, axis=None):
    "a 5x3 test function, using axis"
    y = function5x3(x)
    return y if axis is None else y[axis]


def function5x1(x):
    "a 5x1 test function"
    return function5x3(x)[:1]


def cost5x1(x, axis=None):
    "a 5x1 test function, using axis"
    y = function5x1(x)
    return y if axis is None else y[axis]


def function5(x):
    "a 5d test function"
    return function5x3(x)[0]


def cost5(x, axis=None): #XXX: axis ignored
    "a 5d test function, using axis"
    y = function5(x)
    return y


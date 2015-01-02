#!/usr/bin/env python
#
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""
Input/output 'filters'
"""

# 'filters'
def Identity(x):
    """identity filter, F, where F(x) yields x"""
    return x

def PickComponent(n, multiplier = 1.):
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
def NullChecker(params, evalpts, *args):
    """null validity check"""
    return None

# End of file

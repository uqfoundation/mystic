#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                       Patrick Hung & Mike McKerns, Caltech
#                        (C) 1998-2008  All Rights Reserved
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

"""
Input/output 'filters'
"""

# 'filters'
def Identity(x): return x

def PickComponent(n, multiplier = 1.):
    def _(x):
        return multiplier * x[n,:]
    return _

# 'checkers'
def NullChecker(params, evalpts, *args):
    return None

# End of file

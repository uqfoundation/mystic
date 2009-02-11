#!/usr/bin/env python

"""
F from test_br8.py
"""

def F(alpha):
    a1,a2,a3,a4,a5 = alpha
    def _(t):
        return a1 + a2*exp(-t/a4) + a3*exp(-t/a5)
    return _


# End of file

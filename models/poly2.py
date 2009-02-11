#!/usr/bin/env python

"""
polynomial2 functions
"""

from numpy import array

def ForwardFactory(params):
    a,b,c = params
    def forward_poly(x):
        """ x should be a 1D (1 by N) numpy array """
        return forward((a,b,c),x)
    return forward_poly

def forward(params,x):
    """ x should be a 1D (1 by N) numpy array """
    a,b,c = params
    return array((a*x*x + b*x + c))


# End of file

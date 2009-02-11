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
Forward Mogi Model

This is a reference implementation of the ModelFactory interface
"""

from math import pi
from numpy import array

def ForwardMogiFactory(params):
    x0,y0,z0,dV = params
    def forward_mogi(evalpts):
        """ evalpts should be a 2D (2 by N) numpy array """
        dx = evalpts[0,:] - x0
        dy = evalpts[1,:] - y0
        dz = 0 - z0
        c = dV * 3. / 4. * pi
        # or equivalently c= (3/4) a^3 dP / rigidity
        # where a = sphere radius, dP = delta Pressure
        r2 = dx*dx + dy*dy + dz*dz
        C = c / pow(r2, 1.5)
        return array((C*dx,C*dy,C*dz))
    return forward_mogi



# End of file

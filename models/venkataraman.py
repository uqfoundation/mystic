#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
__doc__ = _doc = """
This is drawn from examples in Applied Optimization with MATLAB programming,
with the function definition found in [1].

References:
    1. Venkataraman, P. "Applied Optimization with MATLAB Programming",
       John Wiley and Sons, Hoboken NJ, 2nd Edition, 2009.
"""
from .abstract_model import AbstractFunction

from math import sin

class Sinc(AbstractFunction):
    __doc__ = \
    """a Venkataraman's sinc function generator

Venkataraman's sinc function [1] has the global minimum at the center
of concentric rings of local minima, with well depth decreasing with
distance from center.

The generated function f(x) is identical to equation (9.5) of example
9.1 of [1], and requires len(x) == 2.
    """ + _doc
    def __init__(self, ndim=2): # is 2-dimensional
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates Venkataraman's sinc function for a list of coeffs

f(x) = -20 * \sin(r(x))/r(x)

Where:
r(x) = \sqrt((x_0 - 4)^2 + (x_1 - 4)^2 + 0.1)

Inspect with mystic_model_plotter using::
    mystic.models.venkat91 -b "-10:10:.1, -10:10:.1" -d

The minimum is f(x)=-19.668329370585823 at x=(4.0, 4.0)"""
        x,y = coeffs
        R = ((x-4.)**2 + (y-4.)**2 + 0.1)**.5
        return -20. * sin(R)/R

    minimizers = None #XXX: there are many local minima
    pass

# cleanup
del _doc

# prepared instances
venkat91 = Sinc().function

# End of file

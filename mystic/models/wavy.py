#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
__doc__ = _doc = """
Multi-minima example functions with vector outputs, which require
a 'reducing' function to provide scalar return values.

References:
    None
"""
from .abstract_model import AbstractFunction

from numpy import absolute as abs
from numpy import asarray
from numpy import sin, pi

class Wavy1(AbstractFunction): #XXX: not a standard test function...?
    __doc__ = \
    r"""a wavy1 function generator

A wavy1 function has a vector return value, and oscillates
similarly to x+\sin(x) in each direction. When a reduction
function, like 'numpy.add' is applied, the surface can be
visualized. The global minimum is at the center of a
cross-hairs running along x_i = -pi, with periodic local
minima in each direction.

The generated function f(x) requires len(x) > 0, and a
reducing function for use in most optimizers.
    """ + _doc
    def __init__(self, ndim=2): # is n-dimensional
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates the wavy1 function for a list of coeffs

f(x) = \abs(x + 3*\sin(x + \pi) + \pi)

Inspect with mystic_model_plotter using::
    mystic.models.wavy1 -b "-20:20:.5, -20:20:.5" -d -r numpy.add

The minimum is f(x)=0.0 at x_i=-pi for all i"""
        x = asarray(coeffs) #XXX: must be numpy.array
        return abs(x+3.*sin(x+pi)+pi)

    minimizers = [-pi]
    pass


class Wavy2(AbstractFunction): #XXX: not a standard test function...?
    r"""a wavy2 function generator

A wavy2 function has a vector return value, and oscillates
similarly to \sin(x) in each direction. When a reduction
function, like 'numpy.add' is applied, the surface can be
visualized. There are degenerate global minima which are
periodic at 2*\pi, and similarly periodic local minima
at nearly the same location and magnitude.

The generated function f(x) requires len(x) > 0, and a
reducing function for use in most optimizers.
    """ + _doc
    def __init__(self, ndim=2): # is n-dimensional
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates the wavy2 function for a list of coeffs

f(x) = 4*\sin(x)+\sin(4*x)+\sin(8*x)+\sin(16*x)+\sin(32*x)+\sin(64*x)

Inspect with mystic_model_plotter using::
    mystic.models.wavy2 -b "-10:10:.2, -10:10:.2" -d -r numpy.add

The function has degenerate global minima of f(x)=-6.987594
at x_i = 4.489843526 + 2*k*pi for all i, and k is an integer"""
        x = asarray(coeffs) #XXX: must be a numpy.array
        return 4*sin(x)+sin(4*x)+sin(8*x)+sin(16*x)+sin(32*x)+sin(64*x)

    minimizers = None #FIXME: degenerate minimum...
                      # minimum is ???
    pass

# cleanup
del _doc

# prepared instances
wavy1 = Wavy1().function
wavy2 = Wavy2().function

# End of file

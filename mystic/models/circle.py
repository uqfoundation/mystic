#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
2d array representation of a circle

References:
    None
"""
__all__ = ['Circle', 'gencircle', 'gendata', 'circle', 'dense_circle',
           'sparse_circle', 'minimal_circle']

#FIXME: cost function seems to apply penalty when r=R0... it should not
from .abstract_model import AbstractModel

from numpy import array, pi, arange
from numpy import sin, cos
from math import floor, sqrt
import random

# random.seed(123)

class Circle(AbstractModel):
    """Computes 2D array representation of a circle
where the circle minimally bounds the 2D data points

data points with [minimal, sparse, or dense] packing=[~0.2, ~1.0, or ~5.0]
setting packing = None will constrain all points to the circle's radius"""
    def __init__(self,packing=None,name='circle',sigma=1.0):
        AbstractModel.__init__(self,name,sigma)
        if packing == None: packing = 0.0
        self.__packing__ = packing
        return

    def __call__(self,x,y,r,*args,**kwds):
        return self.forward((x,y,r),*args,**kwds)

    def forward(self,coeffs,npts=None):
        """generate a 2D array of points contained within a circle

    Args:
        coeffs (list[float]): (x, y, and radius) defining a circle
        npts (int, default=None): number of points to generate

    Returns:
        a 2D array of points contained within the defined circle

    Notes:
        default ``npts`` is ``packing * floor(pi * radius**2)``
        """
        if not npts:
            # generate # of points based on packing and given radius
            npts = self.__packing__ * floor(pi*(coeffs[-1])**2)
        return gendata(coeffs,npts)

    def ForwardFactory(self,coeffs):
        """generate a circle instance from a sequence of coefficients

    Args:
        coeffs (list[float]): (x, y, and radius) defining a circle

    Returns:
        a function returning a 2D array of points contained within the circle
        """
        x,y,r = coeffs
        def forward_circle(npts=None):
            """generate a 2D array of points within the defined circle

    Args:
        npts (int, default=None): number of points to generate

    Returns:
        a 2D array of points contained within the circle (x,y,r) = (%s,%s,%s)

    Notes:
        default ``npts`` is ``packing * floor(pi * radius**2)``
            """ % (x,y,r)
            return self.forward((x,y,r),npts)
        return forward_circle

    def CostFactory(self,target,npts=None):
        """generate a cost function from target coefficients

    Args:
        target (list[float]): (x, y, and radius) defining the target circle
        npts (int, default=None): number of points to generate

    Returns:
        a function returning cost of minimum enclosing circle for npts

    Notes:
        default ``npts`` is ``packing * floor(pi * radius**2)``
        """
        datapts = self.forward(target,npts)
        def cost(params):
            """cost of minimum enclosing circle for a 2D set of points

    Args:
        params (list[float]): (x, y, and radius) defining a circle

    Returns:
        a float representing radius and number of points outside the circle

    Notes:
        fit to points generated on the circle defined by (x,y,r) = (%s,%s,%s)
            """ % (target[0], target[1], target[2])
            x,y,r = params
            if r<0:
                return -999. * r
            penalty = 0
            for xx,yy in datapts:
               # compute distance to origin
               d = sqrt((xx-x)*(xx-x) + (yy-y)*(yy-y))
               if d > r:
                   # each violation adds 1 to the cost plus amount of violation
                   penalty += 1+d-r
            return self.__sigma__ * (r+penalty)
        self.__cost__ = cost
        return self.__cost__

    def CostFactory2(self,datapts):
        """generate a cost function from a 2D array of data points

    Args:
        datapts (array[float,float]): (x,y) location of points in target circle

    Returns:
        a function returning cost of minimum enclosing circle for datapts
        """
        def cost(params):
            """cost of minimum enclosing circle for a 2D set of points

    Args:
        params (list[float]): (x, y, and radius) defining a circle

    Returns:
        a float representing radius and number of points outside the circle
            """
            x,y,r = params
            if r<0:
                return -999. * r
            penalty = 0
            for xx,yy in datapts:
               # compute distance to origin
               d = sqrt((xx-x)*(xx-x) + (yy-y)*(yy-y))
               if d > r:
                   # each violation adds 1 to the cost plus amount of violation
                   penalty += 1+d-r
            return self.__sigma__ * (r+penalty)
        self.__cost__ = cost
        return self.__cost__

    pass

# prepared instances
circle = Circle()
dense_circle = Circle(packing=5.0)
sparse_circle = Circle(packing=1.0)
minimal_circle = Circle(packing=0.2)


# helper functions
def gencircle(coeffs, interval=0.02):
    """generate a 2D array representation of a circle of given coeffs
coeffs = (x,y,r)"""
    x,y,r = coeffs
    theta = arange(0, 2*pi, interval)
    xy = array(list(zip(r*cos(theta)+x, r*sin(theta)+y)))
    return xy

def gendata(coeffs, npts=20):
    """Generate a 2D dataset of npts enclosed in circle of given coeffs,
where coeffs = (x,y,r).

NOTE: if npts == None, constrain all points to circle of given radius"""
    if not npts:
        return gencircle(coeffs)
    def points_circle(N):
        # generate N random points in a unit circle
        n = 0
        while n < N:
            x = random.random()*2.-1.
            y = random.random()*2.-1.
            if x*x + y*y <= 1:
                n = n+1
                yield [x,y]
    x0,y0,R0 = coeffs
    xy = array(list(points_circle(npts)))*R0
    xy[:,0] += x0
    xy[:,1] += y0
    return xy


# End of file

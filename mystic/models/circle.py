#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
2d array representation of a circle

References:
    None
"""
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
built from a list of coefficients
(x,y,r) = coeffs

default npts = packing * floor( pi*radius^2 )"""
        if not npts:
            # generate # of points based on packing and given radius
            npts = self.__packing__ * floor(pi*(coeffs[-1])**2)
        return gendata(coeffs,npts)

    def ForwardFactory(self,coeffs):
        """generates a circle instance from a list of coefficients
(x,y,r) = coeffs"""
        x,y,r = coeffs
        def forward_circle(npts=None):
            """generate a 2D array representation of a circle
with (x,y,r) = (%s,%s,%s)""" % (x,y,r)
            return self.forward((x,y,r),npts)
        return forward_circle

    def CostFactory(self,target,npts=None):
        """generates a cost function instance from list of coefficients & number of evaluation points
(x,y,r) = target coeffs"""
        datapts = self.forward(target,npts)
        def cost(params):
            """cost function for minimum enclosing circle for a 2D set of points"""
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
        """generates a cost function instance from a 2D array of datapoints"""
        def cost(params):
            """cost function for minimum enclosing circle for a 2D set of points"""
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
def gencircle(coeffs,interval=0.02):
    """generate a 2D array representation of a circle of given coeffs
coeffs = (x,y,r)"""
    x,y,r = coeffs
    theta = arange(0, 2*pi, interval)
    xy = array(list(zip(r*cos(theta)+x, r*sin(theta)+y)))
    return xy

def gendata(coeffs,npts=20):
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

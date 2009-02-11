#!/usr/bin/env python

"""
from test_circle.py
"""

from numpy import random, sqrt

def generate(N):
    # generate N random points in a unit circle
    n = 0
    while n < N:
        x = random.random()*2.-1.
        y = random.random()*2.-1.
        if x*x + y*y <= 1:
            n = n+1
            yield [x,y]


def cost(params):
    x,y,r = params
    if r<0:
        return -999. * r
    penalty = 0
    for xx,yy in xy:
       # compute distance to origin
       d = sqrt((xx-x)*(xx-x) + (yy-y)*(yy-y))
       if d > r:
           # each violation adds 1 to the cost plus amount of violation
           penalty += 1+d-r
    return r+penalty


# End of file

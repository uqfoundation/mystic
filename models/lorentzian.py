#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Lorentzian peak model

References:
    None
"""
from .abstract_model import AbstractModel

from numpy import sum as numpysum
from numpy import array, pi, asarray, arange
import random

class Lorentzian(AbstractModel):
    """
Computes lorentzian
    """

    def __init__(self,name='lorentz',metric=lambda x: numpysum(x*x),sigma=1.0):
        AbstractModel.__init__(self,name,metric,sigma)
        return

    def evaluate(self,coeffs,evalpts):
        """evaluate lorentzian with given coeffs over given evalpts
coeffs = (a1,a2,a3,A0,E0,G0,n)"""
        a1,a2,a3,A0,E0,G0,n = coeffs
        x = asarray(evalpts) #XXX: requires a numpy.array
        return (a1 + a2*x + a3*x*x + A0 * ( G0/(2*pi) )/( (x-E0)*(x-E0)+(G0/2)*(G0/2) ))/n

    def ForwardFactory(self,coeffs):
        """generates a lorentzian model instance from a list of coefficients"""
        a1,a2,a3,A0,E0,G0,n = coeffs
        def forward_lorentzian(evalpts):
            """a lorentzian peak over a 1D numpy array
with (a1,a2,a3,A0,E0,G0,n) = (%s,%s,%s,%s,%s,%s,%s)""" % (a1,a2,a3,A0,E0,G0,n)
            return self.evaluate((a1,a2,a3,A0,E0,G0,n),evalpts)
        return forward_lorentzian

    pass
 

# prepared instances
lorentzian = Lorentzian()

def gendata(params,xmin,xmax,npts=4000):
    """Generate a lorentzian dataset of npts between [min,max] from given params"""
    F = lorentzian.ForwardFactory
    def gensample(F, xmin, xmax):
        from numpy import arange
        import random
        a = arange(xmin, xmax, (xmax-xmin)/200.)
        ymin = 0
        ymax = F(a).max()
        while 1:
            t1 = random.random() * (xmax-xmin) + xmin
            t2 = random.random() * (ymax-ymin) + ymin
            t3 = F(t1)
            if t2 < t3:
                return t1
    fwd = F(params)
    try:
        xrange
    except NameError:
        xrange = range
    return array([gensample(fwd, xmin,xmax) for i in xrange(npts)])

# probably shouldn't be in here...
from numpy import histogram as numpyhisto
def histogram(data,binwidth, xmin,xmax):
    """generate bin-centered histogram of provided data
return bins of given binwidth (and histogram) generated between [xmin,xmax]"""
    bins = arange(xmin,xmax, binwidth)
    binsc = bins + (0.5 * binwidth)
    try: #FIXME: use updated numpy.histogram
        histo = numpyhisto(data, bins, new=False)[0]
    except:
        histo = numpyhisto(data, bins)[0]
    return binsc[:len(histo)], histo


# End of file

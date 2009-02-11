#!/usr/bin/env python

"""
Tools to support the Nelder-Mead Simplex class
"""
import numpy
from numpy import absolute
abs = absolute

#################### #################### #################### ####################
#  These are factories that give termination conditions
#################### #################### #################### ####################

def IterationRelativeError(xtol=1e-4, ftol=1e-4):
    def _(inst):
         sim = inst.population
         fsim = inst.popEnergy
         #FIXME: abs(inf - inf) will raise a warning...
         errdict = numpy.seterr(invalid='ignore') #FIXME: turn off warning 
         answer = (max(numpy.ravel(abs(sim[1:]-sim[0]))) <= xtol \
                  and max(abs(fsim[0]-fsim[1:])) <= ftol)
         numpy.seterr(invalid=errdict['invalid']) #FIXME: turn on warnings
         return answer
    return _


# end of file

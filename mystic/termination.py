#!/usr/bin/env python

"""
Factories that provide termination conditions for a mystic.solver
"""

import numpy
from numpy import absolute
abs = absolute

# a module level singleton.
EARLYEXIT = 0

# Factories that give termination conditions
def VTR(tolerance = 0.005):
    """cost of last iteration is < tolerance:

cost[-1] < tolerance"""
    def _(inst):
         hist = inst.energy_history
         return hist[-1] < tolerance
    return _

def ChangeOverGeneration(tolerance = 1e-6, generations = 30):
    """change in cost is < tolerance over a number of generations:

cost[-g] - cost[-1] < tolerance, where g=generations"""
    def _(inst):
         hist = inst.energy_history
         lg = len(hist)
         return lg > generations and (hist[-generations]-hist[-1]) < tolerance
    return _

def NormalizedChangeOverGeneration(tolerance = 1e-4, generations = 2):
    """normalized change in cost is < tolerance over number of generations:

(cost[-g] - cost[-1]) /  0.5*(abs(cost[-g]) + abs(cost[-1])) <= tolerance"""
    eta = 1e-20
    def _(inst):
         hist = inst.energy_history
         lg = len(hist)
         diff = tolerance*(abs(hist[-generations])+abs(hist[-1])) + eta
         return lg > generations and 2.0*(hist[-generations]-hist[-1]) <= diff
    return _
              
def CandidateRelativeTolerance(xtol=1e-4, ftol=1e-4):
    """absolute difference in candidates is < tolerance:

abs(xi-x0) <= xtol & abs(fi-f0) <= ftol, where x=params & f=cost"""
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

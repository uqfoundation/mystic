#!/usr/bin/env python

"""
Factories that provide termination conditions for a mystic.solver
"""

import numpy
from numpy import absolute
abs = absolute
Inf = numpy.Inf

# a module level singleton.
EARLYEXIT = 0

# Factories that give termination conditions
def VTR(tolerance = 0.005):
    """cost of last iteration is < tolerance:

cost[-1] < tolerance"""
    def _VTR(inst):
         hist = inst.energy_history
         return hist[-1] < tolerance
    return _VTR

def ChangeOverGeneration(tolerance = 1e-6, generations = 30):
    """change in cost is < tolerance over a number of generations:

cost[-g] - cost[-1] < tolerance, where g=generations"""
    def _ChangeOverGeneration(inst):
         hist = inst.energy_history
         lg = len(hist)
         if lg <= generations: return False
         return (hist[-generations]-hist[-1]) < tolerance
    return _ChangeOverGeneration

def NormalizedChangeOverGeneration(tolerance = 1e-4, generations = 10):
    """normalized change in cost is < tolerance over number of generations:

(cost[-g] - cost[-1]) /  0.5*(abs(cost[-g]) + abs(cost[-1])) <= tolerance"""
    eta = 1e-20
    def _NormalizedChangeOverGeneration(inst):
         hist = inst.energy_history
         lg = len(hist)
         if lg <= generations: return False
         diff = tolerance*(abs(hist[-generations])+abs(hist[-1])) + eta
         return 2.0*(hist[-generations]-hist[-1]) <= diff
    return _NormalizedChangeOverGeneration
              
def CandidateRelativeTolerance(xtol=1e-4, ftol=1e-4):
    """absolute difference in candidates is < tolerance:

abs(xi-x0) <= xtol & abs(fi-f0) <= ftol, where x=params & f=cost"""
    #NOTE: this termination expects nPop > 1
    def _CandidateRelativeTolerance(inst):
         sim = numpy.array(inst.population)
         fsim = numpy.array(inst.popEnergy)
         if not len(fsim[1:]):
             print "Warning: Invalid termination condition (nPop < 2)"
             return True
         #   raise ValueError, "Invalid termination condition (nPop < 2)"
         #FIXME: abs(inf - inf) will raise a warning...
         errdict = numpy.seterr(invalid='ignore') #FIXME: turn off warning 
         answer = max(numpy.ravel(abs(sim[1:]-sim[0]))) <= xtol
         answer = answer and max(abs(fsim[0]-fsim[1:])) <= ftol
         numpy.seterr(invalid=errdict['invalid']) #FIXME: turn on warnings
         return answer
    return _CandidateRelativeTolerance

def SolutionImprovement(tolerance = 1e-5):  
    """sum of change in each parameter is < tolerance:

sum(abs(last_params - current_params)) <= tolerance"""
    def _SolutionImprovement(inst):
        update = inst.bestSolution - inst.trialSolution #XXX: if inf - inf ?
        answer = numpy.add.reduce(abs(update)) <= tolerance
        return answer
    return _SolutionImprovement

def NormalizedCostTarget(fval = None, tolerance = 1e-6, generations = 30):
    """normalized absolute difference from given cost value is < tolerance:
(if fval is not provided, then terminate when no improvement over g iterations)

abs(cost[-1] - fval)/fval <= tolerance *or* (cost[-1] - cost[-g]) = 0 """
    #NOTE: modified from original behavior
    #  original --> if generations: then return cost[-g] - cost[-1] < 0
    #           --> else: return fval != 0 and abs((best - fval)/fval) < tol
    def _NormalizedCostTarget(inst):
         if generations and fval == None:
             hist = inst.energy_history
             lg = len(hist)
             #XXX: throws error when hist is shorter than generations ?
             return lg > generations and (hist[-generations]-hist[-1]) < 0
         if not generations and fval == None: return True
         return abs(inst.bestEnergy-fval) <= abs(tolerance * fval)
    return _NormalizedCostTarget

def VTRChangeOverGenerations(ftol = 0.005, gtol = 1e-6, generations = 30):
    """change in cost is < gtol over a number of generations,
or cost of last iteration is < ftol:

cost[-g] - cost[-1] < gtol, where g=generations *or* cost[-1] < ftol."""
    def _VTRChangeOverGenerations(inst):
         hist = inst.energy_history
         lg = len(hist)
         #XXX: throws error when hist is shorter than generations ?
         return (lg > generations and (hist[-generations]-hist[-1]) < gtol)\
                or ( hist[-1] < ftol )
    return _VTRChangeOverGenerations

def PopulationSpread(tolerance=1e-6):
    """normalized absolute deviation from best candidate is < tolerance:

abs(params - params[0]) < tolerance"""
    def _PopulationSpread(inst):
         sim = numpy.array(inst.population)
         #if not len(sim[1:]):
         #    print "Warning: Invalid termination condition (nPop < 2)"
         #    return True
         return all(abs(sim - sim[0]) <= abs(tolerance * sim[0]))
    return _PopulationSpread

def GradientNormTolerance(tolerance=1e-5, norm=Inf): 
    """gradient norm is < tolerance, given user-supplied norm:

sum( abs(gradient)**norm )**(1.0/norm) < tolerance"""
    def _GradientNormTolerance(inst):
        try:
            gfk = inst.gfk #XXX: need to ensure that gfk is an array ?
        except:
            print "Warning: Invalid termination condition (no gradient)"
            return True
        if norm == Inf:
            gnorm = numpy.amax(abs(gfk))
        elif norm == -Inf:
            gnorm = numpy.amin(abs(gfk))
        else: #XXX: throws error when norm = 0.0
           #XXX: as norm > large, gnorm approaches amax(abs(gfk)) --> then inf
           #XXX: as norm < -large, gnorm approaches amin(abs(gfk)) --> then -inf
            gnorm = numpy.sum(abs(gfk)**norm,axis=0)**(1.0/norm)
        return gnorm <= tolerance
    return _GradientNormTolerance

# end of file

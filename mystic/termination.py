#!/usr/bin/env python

"""
Factories that provide termination conditions for a mystic.solver
"""

import numpy
from numpy import absolute
from mystic.tools import flatten
abs = absolute
Inf = numpy.Inf
null = ""

# a module level singleton.
EARLYEXIT = 0

# Factories that extend termintaion conditions
class And(tuple):
  """couple termination conditions with "and".

Terminates when all given conditions are satisfied."""
  def __new__(self, *args):
    """
Takes one or more termination conditions:
    args    -- tuple of termination conditions

Usage:
    >>> from mystic.termination import And, VTR, ChangeOverGeneration
    >>> term = And( VTR(), ChangeOverGeneration() )"""
    return tuple.__new__(self, args)

  def __call__(self, solver, info=False):
    """check if the termination conditions are satisfied.

Inputs:
    solver  -- the solver instance

Additional Inputs:
    info    -- if True, return information about the satisfied conditions"""
    stop = {}
    [stop.update({f : f(solver, info)}) for f in self]
    _all = all(stop.values())
    # return T/F if the conditions are met
    if not info: return _all
    # return the satisfied conditions themselves
    if info == 'self': return tuple(set(flatten(stop.keys()))) if _all else ()
    # return info about the satisfied conditions
    return "; ".join(set("; ".join(stop.values()).split("; "))) if _all else ""

  def __repr__(self):
    return "And%s" % str(tuple([i for i in self]))
  #XXX: need method to 'remove' satisfied conditions from all conditions


class Or(tuple):
  """couple termination conditions with "or".

Terminates when any of the given conditions are satisfied."""
  def __new__(self, *args):
    """
Takes one or more termination conditions:
    args    -- tuple of termination conditions

Usage:
    >>> from mystic.termination import Or, VTR, ChangeOverGeneration
    >>> term = Or( VTR(), ChangeOverGeneration() )"""
    return tuple.__new__(self, args)

  def __call__(self, solver, info=False):
    """check if the termination conditions are satisfied.

Inputs:
    solver  -- the solver instance

Additional Inputs:
    info    -- if True, return information about the satisfied conditions"""
    stop = {}
    [stop.update({f : f(solver, info)}) for f in self]
    _any = any(stop.values())
    # return T/F if the conditions are met
    if not info: return _any
    [stop.pop(cond) for (cond,met) in stop.items() if not met]
    # return the satisfied conditions themselves
    if info == 'self': return tuple(set(flatten(stop.keys())))
    # return info about the satisfied conditions
    return "; ".join(set("; ".join(stop.values()).split("; ")))

  def __repr__(self):
    return "Or%s" % str(tuple([i for i in self]))
  #XXX: need method to 'remove' satisfied conditions from all conditions


class When(And):
  """provide a termination condition with more reporting options.

Terminates when the given condition is satisfied."""
  def __init__(self, arg):
    """
Takes a termination condition:
    arg     -- termination condition

Usage:
    >>> from mystic.termination import When, VTR
    >>> term = When( VTR() )"""
    super(And, self).__init__(arg)
    return

  def __repr__(self):
    return "When(%s)" % str(self[0])
  #XXX: need method to 'remove' satisfied conditions from all conditions


# Factories that give termination conditions
def VTR(tolerance = 0.005, target = 0.0):
    """cost of last iteration is < tolerance:

cost[-1] <= tolerance"""
    doc = "VTR with %s" % {'tolerance':tolerance, 'target':target}
    def _VTR(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        hist = inst.energy_history
        if not len(hist): return info(null)
        if abs(hist[-1] - target) <= tolerance: return info(doc)
        return info(null)
   #_VTR.__doc__ = "%s(**%s)" % tuple(doc.split(" with "))
    _VTR.__doc__ = doc
    return _VTR

def ChangeOverGeneration(tolerance = 1e-6, generations = 30):
    """change in cost is < tolerance over a number of generations:

cost[-g] - cost[-1] <= tolerance, where g=generations"""
    doc = "ChangeOverGeneration with %s" % {'tolerance':tolerance,
                                            'generations':generations}
    def _ChangeOverGeneration(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        hist = inst.energy_history
        lg = len(hist)
        if lg <= generations: return info(null)
        if (hist[-generations]-hist[-1]) <= tolerance: return info(doc)
        return info(null)
    _ChangeOverGeneration.__doc__ = doc
    return _ChangeOverGeneration

def NormalizedChangeOverGeneration(tolerance = 1e-4, generations = 10):
    """normalized change in cost is < tolerance over number of generations:

(cost[-g] - cost[-1]) /  0.5*(abs(cost[-g]) + abs(cost[-1])) <= tolerance"""
    eta = 1e-20
    doc = "NormalizedChangeOverGeneration with %s" % {'tolerance':tolerance,
                                                      'generations':generations}
    def _NormalizedChangeOverGeneration(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        hist = inst.energy_history
        lg = len(hist)
        if lg <= generations: return info(null)
        diff = tolerance*(abs(hist[-generations])+abs(hist[-1])) + eta
        if 2.0*(hist[-generations]-hist[-1]) <= diff: return info(doc)
        return info(null)
    _NormalizedChangeOverGeneration.__doc__ = doc
    return _NormalizedChangeOverGeneration
              
def CandidateRelativeTolerance(xtol = 1e-4, ftol = 1e-4):
    """absolute difference in candidates is < tolerance:

abs(xi-x0) <= xtol & abs(fi-f0) <= ftol, where x=params & f=cost"""
    #NOTE: this termination expects nPop > 1
    doc = "CandidateRelativeTolerance with %s" % {'xtol':xtol, 'ftol':ftol}
    def _CandidateRelativeTolerance(inst, info=False):
        sim = numpy.array(inst.population)
        fsim = numpy.array(inst.popEnergy)
        if not len(fsim[1:]):
            warn = "Warning: Invalid termination condition (nPop < 2)"
            print warn
            return warn
        #   raise ValueError, "Invalid termination condition (nPop < 2)"
        if info: info = lambda x:x
        else: info = bool
        #FIXME: abs(inf - inf) will raise a warning...
        errdict = numpy.seterr(invalid='ignore') #FIXME: turn off warning 
        answer = max(numpy.ravel(abs(sim[1:]-sim[0]))) <= xtol
        answer = answer and max(abs(fsim[0]-fsim[1:])) <= ftol
        numpy.seterr(invalid=errdict['invalid']) #FIXME: turn on warnings
        if answer: return info(doc)
        return info(null)
    _CandidateRelativeTolerance.__doc__ = doc
    return _CandidateRelativeTolerance

def SolutionImprovement(tolerance = 1e-5):  
    """sum of change in each parameter is < tolerance:

sum(abs(last_params - current_params)) <= tolerance"""
    doc = "SolutionImprovement with %s" % {'tolerance':tolerance}
    def _SolutionImprovement(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        best = numpy.array(inst.bestSolution)
        trial = numpy.array(inst.trialSolution)
        update = best - trial #XXX: if inf - inf ?
        answer = numpy.add.reduce(abs(update)) <= tolerance
        if answer: return info(doc)
        return info(null)
    _SolutionImprovement.__doc__ = doc
    return _SolutionImprovement

def NormalizedCostTarget(fval = None, tolerance = 1e-6, generations = 30):
    """normalized absolute difference from given cost value is < tolerance:
(if fval is not provided, then terminate when no improvement over g iterations)

abs(cost[-1] - fval)/fval <= tolerance *or* (cost[-1] - cost[-g]) = 0 """
    #NOTE: modified from original behavior
    #  original --> if generations: then return cost[-g] - cost[-1] < 0
    #           --> else: return fval != 0 and abs((best - fval)/fval) < tol
    doc = "NormalizedCostTarget with %s" % {'fval':fval, 'tolerance':tolerance,
                                            'generations':generations}
    def _NormalizedCostTarget(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        if generations and fval == None:
            hist = inst.energy_history
            lg = len(hist)
            #XXX: throws error when hist is shorter than generations ?
            if lg > generations and (hist[-generations]-hist[-1]) <= 0:
                return info(doc)
            return info(null)
        if not generations and fval == None: return info(doc)
        if abs(inst.bestEnergy-fval) <= abs(tolerance * fval): return info(doc)
        return info(null)
    _NormalizedCostTarget.__doc__ = doc
    return _NormalizedCostTarget

def VTRChangeOverGeneration(ftol = 0.005, gtol = 1e-6, generations = 30,
                                                            target = 0.0):
    """change in cost is < gtol over a number of generations,
or cost of last iteration is < ftol:

cost[-g] - cost[-1] <= gtol, where g=generations *or* cost[-1] <= ftol."""
    doc = "VTRChangeOverGeneration with %s" % {'ftol':ftol, 'gtol':gtol,
                                               'generations':generations,
                                               'target':target}
    def _VTRChangeOverGeneration(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        hist = inst.energy_history
        lg = len(hist)
        #XXX: throws error when hist is shorter than generations ?
        if (lg > generations and (hist[-generations]-hist[-1]) <= gtol)\
               or ( abs(hist[-1] - target) <= ftol ): return info(doc)
        return info(null)
    _VTRChangeOverGeneration.__doc__ = doc
    return _VTRChangeOverGeneration

def PopulationSpread(tolerance = 1e-6):
    """normalized absolute deviation from best candidate is < tolerance:

abs(params - params[0]) <= tolerance"""
    doc = "PopulationSpread with %s" % {'tolerance':tolerance}
    def _PopulationSpread(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        sim = numpy.array(inst.population)
        #if not len(sim[1:]):
        #    print "Warning: Invalid termination condition (nPop < 2)"
        #    return True
        if numpy.all(abs(sim - sim[0]) <= abs(tolerance * sim[0])): return info(doc)
        return info(null)
    _PopulationSpread.__doc__ = doc
    return _PopulationSpread

def GradientNormTolerance(tolerance = 1e-5, norm = Inf): 
    """gradient norm is < tolerance, given user-supplied norm:

sum( abs(gradient)**norm )**(1.0/norm) <= tolerance"""
    doc = "GradientNormTolerance with %s" % {'tolerance':tolerance, 'norm':norm}
    def _GradientNormTolerance(inst, info=False):
        try:
            gfk = inst.gfk #XXX: need to ensure that gfk is an array ?
        except:
            warn = "Warning: Invalid termination condition (no gradient)"
            print warn
            return warn
        if info: info = lambda x:x
        else: info = bool
        if norm == Inf:
            gnorm = numpy.amax(abs(gfk))
        elif norm == -Inf:
            gnorm = numpy.amin(abs(gfk))
        else: #XXX: throws error when norm = 0.0
           #XXX: as norm > large, gnorm approaches amax(abs(gfk)) --> then inf
           #XXX: as norm < -large, gnorm approaches amin(abs(gfk)) --> then -inf
            gnorm = numpy.sum(abs(gfk)**norm,axis=0)**(1.0/norm)
        if gnorm <= tolerance: return info(doc)
        return info(null)
    _GradientNormTolerance.__doc__ = doc
    return _GradientNormTolerance

# end of file

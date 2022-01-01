#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Copyright (c) 2010-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Factories that provide termination conditions for a mystic.solver
"""

import numpy
abs = numpy.absolute
inf = Inf = numpy.Inf
null = ""
_type = type #NOTE: save builtin type
import mystic.collapse as ct #XXX: avoid if move Collapse* to collapse
from mystic.math.distance import Lnorm #XXX: avoid this and following?
from mystic._scipy060optimize import approx_fprime, _epsilon

# a module level singleton
EARLYEXIT = 0

# termination condition interrogator functions
#FIXME FIXME: assumes NO DUPLICATE TYPES of termination conditions
def state(condition):
    '''get state (dict of kwds) used to create termination condition'''
    #NOTE: keys are termination name; values are termination state
    _state = {}
    for term in iter(condition) if isinstance(condition, tuple) else iter((condition,)):
        termdoc = term.__doc__# or 'MISSING'
        if termdoc is None: #HACK
            import warnings
            warnings.warn('Collapse termination mishandled')
            pass #FIXME: HACK: shouldn't be missing (due to ensemble + Mapper)
        elif not termdoc.split(None,1)[-1].startswith('with '):# Or, And, ...
            _state.update(state(term))
        else:
            kind,kwds = termdoc.split(' with ', 1)
           #_state[kind] = eval(kwds)
            _state[termdoc] = eval(kwds)
    return _state

def type(condition):
    '''get object that generated the given termination instance'''
    if isinstance(condition, _type(lambda :None)): #XXX: type of term conditions
        try:
            import importlib
            module = importlib.import_module(condition.__module__)
        except ImportError:
            module = __import__(condition.__module__, globals(), locals(), ['object'], -1)
        return getattr(module, condition.__name__[1:]) #XXX: start w/ _
    # otherwise, just figure it's a class or standard object
    return _type(condition)


# Factories that extend termination conditions
class When(tuple):
  """provide a termination condition with more reporting options.

Terminates when the given condition is satisfied."""
  def __new__(self, arg):
    """
Takes a termination condition:
    arg    -- termination condition

Usage:
    >>> from mystic.termination import When, VTR
    >>> term = When( VTR() )
    >>> term(solver)  # where solver is a mystic.solver instance"""
    if isinstance(arg, tuple) and len(arg) == 1: arg = arg[0] # for pickling
    #XXX: need better filter on inputs
    if getattr(arg, '__module__', None) != self.__module__:
      raise TypeError("'%s' object is not a condition" % arg.__class__.__name__)
    if not getattr(arg, '__len__', None): arg = [arg]
    return tuple.__new__(self, arg)

  def __call__(self, solver, info=False):
    """check if the termination conditions are satisfied.

Inputs:
    solver -- the solver instance

Additional Inputs:
    info   -- if True, return information about the satisfied conditions"""
    # return the unsatisfied conditions
    if info == 'not':
      return tuple(set([f for f in self if f not in self(solver, 'self')]))
    # do some filtering...
    stop = {}
    [stop.update({f : f(solver, info)}) for f in self]
    _all = all(stop.values())
    # return T/F if the conditions are met
    if not info: return _all
    # return the satisfied conditions
    if info == 'self': return tuple(set(stop.keys())) if _all else ()
    # return info about the satisfied conditions
    return "; ".join(set("; ".join(getattr(stop, 'itervalues', stop.values)()).split("; "))) if _all else ""

  def __repr__(self):
    return "When(%s)" % str(self[0])

class And(When):
  """couple termination conditions with "and".

Terminates when all given conditions are satisfied."""
  def __new__(self, *args):
    """
Takes one or more termination conditions:
    args   -- tuple of termination conditions

Usage:
    >>> from mystic.termination import And, VTR, ChangeOverGeneration
    >>> term = And( VTR(), ChangeOverGeneration() )
    >>> term(solver)  # where solver is a mystic.solver instance"""
    if isinstance(args, tuple) and len(args) == 1: args = args[0] # for pickling
    #XXX: need better filter on inputs
    if not getattr(args, '__len__', None): args = [args]
    #XXX: check if every arg in args has __module__ == self.__module__ ?
    return tuple.__new__(self, args)

  def __repr__(self):
    return "And%s" % str(tuple([f for f in self]))


class Or(When):
  """couple termination conditions with "or".

Terminates when any of the given conditions are satisfied."""
  def __new__(self, *args):
    """
Takes one or more termination conditions:
    args   -- tuple of termination conditions

Usage:
    >>> from mystic.termination import Or, VTR, ChangeOverGeneration
    >>> term = Or( VTR(), ChangeOverGeneration() )
    >>> term(solver)  # where solver is a mystic.solver instance"""
    if isinstance(args, tuple) and len(args) == 1: args = args[0] # for pickling
    #XXX: need better filter on inputs
    if not getattr(args, '__len__', None): args = [args]
    #XXX: check if every arg in args has __module__ == self.__module__ ?
    return tuple.__new__(self, args)

  def __call__(self, solver, info=False):
    """check if the termination conditions are satisfied.

Inputs:
    solver -- the solver instance

Additional Inputs:
    info   -- if True, return information about the satisfied conditions"""
    # return the unsatisfied conditions
    if info == 'not':
      return tuple(set([f for f in self if f not in self(solver, 'self')]))
    # do some filtering...
    stop = {}
    [stop.update({f : f(solver, info)}) for f in self]
    _any = any(stop.values())
    # return T/F if the conditions are met
    if not info: return _any
    [stop.pop(cond) for (cond,met) in tuple(getattr(stop, 'iteritems', stop.items)()) if not met]
    # return the satisfied conditions
    if info == 'self': return tuple(set(stop.keys()))
    # return info about the satisfied conditions
    return "; ".join(set("; ".join(getattr(stop, 'itervalues', stop.values)()).split("; ")))

  def __repr__(self):
    return "Or%s" % str(tuple([f for f in self]))


# Factories that give termination conditions
#FIXME: the following should be refactored into classes
def VTR(tolerance=0.005, target=0.0):
    """cost of last iteration is < tolerance from target:

``abs(cost[-1] - target) <= tolerance``
"""
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

def ChangeOverGeneration(tolerance=1e-6, generations=30):
    """change in cost is < tolerance over a number of generations:

``cost[-g] - cost[-1] <= tolerance``, with ``g=generations``
"""
    doc = "ChangeOverGeneration with %s" % {'tolerance':tolerance,
                                            'generations':generations}
    def _ChangeOverGeneration(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        hist = inst.energy_history
        lg = len(hist)
        if not lg: return info(null)
        if lg <= generations: return info(null)
        if (hist[-generations]-hist[-1]) <= tolerance: return info(doc)
        if (hist[-generations] == hist[-1]): return info(doc)
        return info(null)
    _ChangeOverGeneration.__doc__ = doc
    return _ChangeOverGeneration

def NormalizedChangeOverGeneration(tolerance=1e-4, generations=10):
    """normalized change in cost is < tolerance over number of generations:

``(cost[-g] - cost[-1]) / 0.5*(abs(cost[-g]) + abs(cost[-1])) <= tolerance``
"""
    eta = 1e-20
    doc = "NormalizedChangeOverGeneration with %s" % {'tolerance':tolerance,
                                                      'generations':generations}
    def _NormalizedChangeOverGeneration(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        hist = inst.energy_history
        lg = len(hist)
        if not lg: return info(null)
        if lg <= generations: return info(null)
        if (hist[-generations] == hist[-1]): return info(doc)
        diff = tolerance*(abs(hist[-generations])+abs(hist[-1])) + eta
        if 2.0*(hist[-generations]-hist[-1]) <= diff: return info(doc)
        return info(null)
    _NormalizedChangeOverGeneration.__doc__ = doc
    return _NormalizedChangeOverGeneration
              
def CandidateRelativeTolerance(xtol=1e-4, ftol=1e-4):
    """absolute difference in candidates is < tolerance:

``abs(xi-x0) <= xtol`` & ``abs(fi-f0) <= ftol``, with ``x=params`` & ``f=cost``
"""
    #NOTE: this termination expects nPop > 1
    doc = "CandidateRelativeTolerance with %s" % {'xtol':xtol, 'ftol':ftol}
    def _CandidateRelativeTolerance(inst, info=False):
        sim = numpy.array(inst.population)
        fsim = numpy.array(inst.popEnergy)
        if not len(fsim[1:]):
            warn = "Warning: Invalid termination condition (nPop < 2)"
            print(warn)
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

def SolutionImprovement(tolerance=1e-5):  
    """sum of change in each parameter is < tolerance:

``sum(abs(last_params - current_params)) <= tolerance``
"""
    doc = "SolutionImprovement with %s" % {'tolerance':tolerance}
    def _SolutionImprovement(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        best = numpy.array(inst.bestSolution)
        trial = numpy.array(inst.trialSolution)
        update = abs(best - trial) #XXX: if inf - inf ?
        answer = numpy.add.reduce(update.T)
        if isinstance(answer, numpy.ndarray): # if trialPop, take 'best' answer
            answer = max(answer)              #XXX: is this 'best' or 'worst'?
        answer = answer <= tolerance
        if answer: return info(doc)
        return info(null)
    _SolutionImprovement.__doc__ = doc
    return _SolutionImprovement

def NormalizedCostTarget(fval=None, tolerance=1e-6, generations=30):
    """normalized absolute difference from given cost value is < tolerance:
(if fval is not provided, then terminate when no improvement over g iterations)

``abs(cost[-1] - fval)/fval <= tolerance`` or ``(cost[-1] - cost[-g]) = 0``
"""
    #NOTE: modified from original behavior
    #  original --> if generations: then return cost[-g] - cost[-1] < 0
    #           --> else: return fval != 0 and abs((best - fval)/fval) < tol
    doc = "NormalizedCostTarget with %s" % {'fval':fval, 'tolerance':tolerance,
                                            'generations':generations}
    def _NormalizedCostTarget(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        hist = inst.energy_history
        lg = len(hist)
        if not lg: return info(null)
        if generations and fval is None:
            #XXX: throws error when hist is shorter than generations ?
            if lg > generations and ((hist[-generations]-hist[-1]) <= 0 or \
                                     (hist[-generations] == hist[-1])):
                return info(doc)
            return info(null)
        if not generations and fval is None: return info(doc)
        if abs(hist[-1]-fval) <= abs(tolerance * fval): return info(doc)
        return info(null)
    _NormalizedCostTarget.__doc__ = doc
    return _NormalizedCostTarget

def VTRChangeOverGeneration(ftol=0.005, gtol=1e-6, generations=30, target=0.0):
    """change in cost is < gtol over a number of generations,
or cost of last iteration is < ftol from target:

``cost[-g] - cost[-1] <= gtol`` or ``abs(cost[-1] - target) <= ftol``
"""
    doc = "VTRChangeOverGeneration with %s" % {'ftol':ftol, 'gtol':gtol,
                                               'generations':generations,
                                               'target':target}
    def _VTRChangeOverGeneration(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        hist = inst.energy_history
        lg = len(hist)
        if not lg: return info(null)
        #XXX: throws error when hist is shorter than generations ?
        if (lg > generations and ((hist[-generations]-hist[-1]) <= gtol or \
                                  (hist[-generations] == hist[-1]))) or \
           ( abs(hist[-1] - target) <= ftol ): return info(doc)
        return info(null)
    _VTRChangeOverGeneration.__doc__ = doc
    return _VTRChangeOverGeneration

def PopulationSpread(tolerance=1e-6):
    """normalized absolute deviation from best candidate is < tolerance:

``abs(params - params[0]) <= tolerance``
"""
    doc = "PopulationSpread with %s" % {'tolerance':tolerance}
    def _PopulationSpread(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        sim = numpy.array(inst.population)
        #if not len(sim[1:]):
        #    warn = "Warning: Invalid termination condition (nPop < 2)"
        #    print(warn)
        #    return warn
        if numpy.all(abs(sim - sim[0]) <= abs(tolerance * sim[0])): return info(doc)
        return info(null)
    _PopulationSpread.__doc__ = doc
    return _PopulationSpread

def GradientNormTolerance(tolerance=1e-5, norm=Inf): 
    """gradient norm is < tolerance, given user-supplied norm:

``sum( abs(gradient)**norm )**(1.0/norm) <= tolerance``
"""
    doc = "GradientNormTolerance with %s" % {'tolerance':tolerance, 'norm':norm}
    def _GradientNormTolerance(inst, info=False):
        grad = getattr(inst, 'gradient', [None])[-1]
        if grad is None:
            soln = inst.bestSolution
            cost = inst._cost[1]
            grad = approx_fprime(soln, cost, _epsilon)
           #warn = "Warning: using approximate gradient"
           #print(warn)
           #return warn
        if info: info = lambda x:x
        else: info = bool
        gnorm = Lnorm(grad, p=norm, axis=0)
        if gnorm <= tolerance: return info(doc)
        return info(null)
    _GradientNormTolerance.__doc__ = doc
    return _GradientNormTolerance

def EvaluationLimits(generations=None, evaluations=None):
    """number of iterations is > generations,
or number of function calls is > evaluations:

``iterations >= generations`` or ``fcalls >= evaluations``
"""
    #NOTE: default settings use solver defaults (_maxfun and _maxiter)
    doc = "EvaluationLimits with %s" % {'generations':generations, \
                                        'evaluations':evaluations}
    maxfun = [evaluations]
    maxiter = [generations]
    if maxfun[0] is None: maxfun[0] = Inf 
    if maxiter[0] is None: maxiter[0] = Inf
    def _EvaluationLimits(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        gens = inst.generations
        eval = inst._fcalls[0]
        if (eval >= maxfun[0]) or (gens >= maxiter[0]): return info(doc)
        return info(null)
   #_EvaluationLimits.__doc__ = "%s(**%s)" % tuple(doc.split(" with "))
    _EvaluationLimits.__doc__ = doc
    return _EvaluationLimits

def SolverInterrupt(): #XXX: enable = True ?
    """handler is enabled and interrupt is given:

``_EARLYEXIT == True``
"""
    doc = "SolverInterrupt with %s" % {}
    def _SolverInterrupt(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        if inst._EARLYEXIT: return info(doc)
        return info(null)
    _SolverInterrupt.__doc__ = doc
    return _SolverInterrupt

##### parameter collapse conditions #####
def CollapseWeight(tolerance=0.005, generations=50, mask=None, **kwds):
    """value of weights are < tolerance over a number of generations,
where mask is (row,column) indices of the selected weights:

``bool(collapse_weight(monitor, **kwds))``
"""
    _kwds = {'tolerance':tolerance, 'generations':generations, 'mask':mask}
    kwds.update(_kwds)
    doc = "CollapseWeight with %s" % kwds #XXX: better kwds or _kwds?
    def _CollapseWeight(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        hist = inst.energy_history
        lg = len(hist)
        if not lg: return info(null)
        if lg <= generations: return info(null)
        #XXX: might want to log/utilize *where* collapse happens...
#       if collapse_weight(inst._stepmon, **kwds): return info(doc)
        collapsed = ct.collapse_weight(inst._stepmon, **kwds)
        if collapsed: return info(doc + ' at %s' % str(collapsed))
        # otherwise bail out
        return info(null) 
    _CollapseWeight.__doc__ = doc
    return _CollapseWeight

def CollapsePosition(tolerance=0.005, generations=50, mask=None, **kwds):
    """max(pairwise(positions)) < tolerance over a number of generations,
where (measures,indices) are (row,column) indices of selected positions:

``bool(collapse_position(monitor, **kwds))``
"""
    _kwds = {'tolerance':tolerance, 'generations':generations, 'mask':mask}
    kwds.update(_kwds)
    doc = "CollapsePosition with %s" % kwds #XXX: better kwds or _kwds
    def _CollapsePosition(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        hist = inst.energy_history
        lg = len(hist)
        if not lg: return info(null)
        if lg <= generations: return info(null)
        #XXX: might want to log/utilize *where* collapse happens...
#       if collapse_weight(inst._stepmon, **kwds): return info(doc)
        collapsed = ct.collapse_position(inst._stepmon, **kwds)
        if collapsed: return info(doc + ' at %s' % str(collapsed))
        # otherwise bail out
        return info(null) 
    _CollapsePosition.__doc__ = doc
    return _CollapsePosition

def CollapseAt(target=None, tolerance=1e-4, generations=50, mask=None):
    """change(x[i]) is < tolerance over a number of generations,
where target can be a single value or a list of values of x length,
change(x[i]) = max(x[i]) - min(x[i]) if target=None else abs(x[i] - target),
and mask is column indices of selected params:

``bool(collapse_at(monitor, **kwds))``
"""
    kwds = {'tolerance':tolerance, 'generations':generations,
            'target':target, 'mask':mask}
    doc = "CollapseAt with %s" % kwds
    def _CollapseAt(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        hist = inst.energy_history
        lg = len(hist)
        if not lg: return info(null)
        if lg <= generations: return info(null)
        #XXX: might want to log/utilize *where* collapse happens...
#       if ct.collapse_at(inst._stepmon, **kwds): return info(doc)
        collapsed = ct.collapse_at(inst._stepmon, **kwds)
        if collapsed: return info(doc + ' at %s' % str(collapsed))
        # otherwise bail out
        return info(null) 
    _CollapseAt.__doc__ = doc
    return _CollapseAt

def CollapseAs(offset=False, tolerance=1e-4, generations=50, mask=None):
    """max(pairwise(x)) is < tolerance over a number of generations,
and mask is column indices of selected params:

``bool(collapse_as(monitor, **kwds))``
"""
    kwds = {'tolerance':tolerance, 'generations':generations,
            'offset':offset, 'mask':mask}
    doc = "CollapseAs with %s" % kwds
    def _CollapseAs(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        hist = inst.energy_history
        lg = len(hist)
        if not lg: return info(null)
        if lg <= generations: return info(null)
        #XXX: might want to log/utilize *where* collapse happens...
#       if ct.collapse_as(inst._stepmon, **kwds): return info(doc)
        collapsed = ct.collapse_as(inst._stepmon, **kwds)
        if collapsed: return info(doc + ' at %s' % str(collapsed))
        # otherwise bail out
        return info(null) 
    _CollapseAs.__doc__ = doc
    return _CollapseAs

##### bounds collapse conditions #####
def CollapseCost(clip=False, limit=1.0, samples=50, mask=None):
    """cost(x) - min(cost) is >= limit for all samples within an interval,
where if clip is True, then clip beyond the space sampled the optimizer,
and mask is a dict of {index:bounds} where bounds are provided as an
interval (min,max), or a list of intervals:

``bool(collapse_cost(monitor, **kwds))``
"""
    kwds = {'limit':limit, 'samples':samples,
            'clip':clip, 'mask':mask}
    doc = "CollapseCost with %s" % kwds
    def _CollapseCost(inst, info=False):
        if info: info = lambda x:x
        else: info = bool
        hist = inst.energy_history
        lg = len(hist)
        if not lg: return info(null)
        if lg <= samples: return info(null)
        #XXX: mask = interval_overlap(mask, solver_bounds(inst))?
        #XXX: might want to log/utilize *where* collapse happens...
#       if collapse_cost(inst._stepmon, **kwds): return info(doc)
        collapsed = ct.collapse_cost(inst._stepmon, **kwds)
        if collapsed: return info(doc + ' at %s' % str(collapsed))
        # otherwise bail out
        return info(null)
    _CollapseCost.__doc__ = doc
    return _CollapseCost

# end of file

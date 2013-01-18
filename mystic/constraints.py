#!/usr/bin/env python

"""Tools for building and applying constraints and penalties.
"""

__all__ = ['with_penalty','with_constraint','as_penalty','as_constraint',
           'with_mean','with_variance','with_spread','normalized',
           'issolution','solve']

from mystic.math.measures import *
from mystic.math import almostEqual

def with_penalty(ptype, *args, **kwds):
    """convert a condition to a penalty function of the chosen type

condition f(x) is satisfied when f(x) == 0.0 for equality constraints
and f(x) <= 0.0 for inequality constraints. ptype is a mystic.penalty type.

    For example:
    >>> @with_penalty(quadratic_equality, kwds={'target':5.0})
    ... def penalty(x, target):
    ...   return mean(x) - target
    >>> 
    >>> penalty([1,2,3,4,5])
    400.0
    >>> penalty([3,4,5,6,7])
    7.8886090522101181e-29
    """
    def dec(condition):
      @ptype(condition, *args, **kwds)
      def penalty(x):
          return 0.0
      return penalty
    return dec


#XXX: better: wrap_constraint( constraint, func, ctype=inner) ?
def with_constraint(ctype, *args, **kwds): #XXX: is this *at all* useful?
    """convert a set transformation to a constraints solver of the chosen type

transformation f(x) is a mapping between x and x', where x' = f(x).
ctype is a mystic.coupler type [inner, outer, inner_proxy, outer_proxy].

    For example:
    >>> @with_constraint(inner, kwds={'target':5.0})
    ... def constraint(x, target):
    ...   return impose_mean(target, x)
    ... 
    >>> x = constraint([1,2,3,4,5])
    >>> print x
    [3.0, 4.0, 5.0, 6.0, 7.0]
    >>> mean(x)
    5.0
    """
    def dec(condition):
      @ctype(condition, *args, **kwds)
      def constraint(x):
          return x
      return constraint
    return dec


# XXX: provide the below from measures, which from bounded?, and ...generic?
'''
mean(samples, weights=None)
variance(samples, weights=None)
spread(samples)
expectation(f, samples, weights=None, tol=0.0)
# impose_mean(m, samples, weights=None)
# impose_variance(v, samples, weights=None)
# impose_spread(r, samples, weights=None)
impose_expectation(param, f, npts, bounds=None, weights=None, **kwds)
impose_weight_norm(samples, weights, mass=1.0)
# normalize(weights, mass=1.0, zsum=False, zmass=1.0)
'''

def with_mean(target):
    """bind a mean constraint to a given constraints function.

Inputs:
    target -- the target mean

A constraints function takes an iterable x as input, returning a modified x.
This function is an "outer" coupling of "impose_mean" onto another constraints
function c(x), such that:  x' = impose_mean(target, c(x)).

    For example:
    >>> @with_mean(5.0)
    ... def constraint(x):
    ...   x[-1] = x[0]
    ...   return x
    ... 
    >>> x = constraint([1,2,3,4])
    >>> print x
    [4.25, 5.25, 6.25, 4.25]
    >>> mean(x)
    5.0
    """
    def decorate(constraints):
        def factory(x, *args, **kwds):
            # apply decorated constraints function
            x = constraints(x, *args, **kwds)
            # constrain x such that mean(x) == target
            if not almostEqual(mean(x), target):
                x = impose_mean(target, x)#, weights=weights)
            return x
        return factory
    return decorate


def with_variance(target):
    """bind a variance constraint to a given constraints function.

Inputs:
    target -- the target variance

A constraints function takes an iterable x as input, returning a modified x.
This function is an "outer" coupling of "impose_variance" onto another
constraints function c(x), such that:  x' = impose_variance(target, c(x)).

    For example:
    >>> @with_variance(1.0)
    ... def constraint(x):
    ...   x[-1] = x[0]
    ...   return x
    ... 
    >>> x = constraint([1,2,3])
    >>> print x
    [0.6262265521467858, 2.747546895706428, 0.6262265521467858]
    >>> variance(x)
    0.99999999999999956
    """
    def decorate(constraints):
        def factory(x, *args, **kwds):
            # apply decorated constraints function
            x = constraints(x, *args, **kwds)
            # constrain x such that variance(x) == target
            if not almostEqual(variance(x), target):
                x = impose_variance(target, x)#, weights=weights)
            return x
        return factory
    return decorate


def with_spread(target):
    """bind a range constraint to a given constraints function.

Inputs:
    target -- the target range

A constraints function takes an iterable x as input, returning a modified x.
This function is an "outer" coupling of "impose_spread" onto another constraints
function c(x), such that:  x' = impose_spread(target, c(x)).

    For example:
    >>> @with_spread(10.0)
    ... def constraint(x):
    ...   return [i**2 for i in x]
    ... 
    >>> x = constraint([1,2,3,4])
    >>> print x
    [3.1666666666666665, 5.1666666666666661, 8.5, 13.166666666666666]
    >>> spread(x)
    10.0
    """
    def decorate(constraints):
        def factory(x, *args, **kwds):
            # apply decorated constraints function
            x = constraints(x, *args, **kwds)
            # constrain x such that spread(x) == target
            if not almostEqual(spread(x), target):
                x = impose_spread(target, x)#, weights=weights)
            return x
        return factory
    return decorate


from numpy import sum
def normalized(mass=1.0): #XXX: order matters when multiple decorators
    """bind a normalization constraint to a given constraints function.

Inputs:
    mass -- the target sum of normalized weights

A constraints function takes an iterable x as input, returning a modified x.
This function is an "outer" coupling of "normalize" onto another constraints
function c(x), such that:  x' = normalize(c(x), mass).

    For example:
    >>> @normalized()
    ... def constraint(x):
    ...   return x
    ... 
    >>> constraint([1,2,3])
    [0.16666666666666666, 0.33333333333333331, 0.5]
    """
    def decorate(constraints):
        def factory(x, *args, **kwds):
            # apply decorated constraints function
            x = constraints(x, *args, **kwds)
            # constrain x such that sum(x) == mass
            if not almostEqual(sum(x), mass):
                x = normalize(x, mass=mass)
            return x
        return factory
    return decorate


def issolution(constraints, guess, tol=1e-3):
    """Returns whether the guess is a solution to the constraints

Input:
    constraints -- a constraints solver function or a penalty function
    guess -- list of parameter values prposed to solve the constraints
    tol -- residual error magnitude for which constraints are considered solved
    """
    if hasattr(constraints, 'error'):
        error = constraints.error(guess)
    else: # is a constraints solver
        try:
            constrained = guess.copy()
        except:
            from copy import deepcopy
            constrained = deepcopy(guess)
        error = 0.0
        constrained = constraints(constrained)
        for i in range(len(guess)):
            error += (constrained[i] - guess[i])**2  #XXX: better rnorm ?
        error = error**0.5

    return error <= tol


#XXX: nice if penalty.error could give error for each condition... or total


def solve(constraints, guess=None, nvars=None, solver=None, \
          lower_bounds=None, upper_bounds=None, termination=None):
    """Use optimization to find a solution to a set of constraints.

Inputs:
    constraints -- a constraints solver function or a penalty function

Additional Inputs:
    guess -- list of parameter values proposed to solve the constraints.
    lower_bounds -- list of lower bounds on solution values.
    upper_bounds -- list of upper bounds on solution values.
    nvars -- number of parameter values.
    solver -- the mystic solver to use in the optimization
    termination -- the mystic termination to use in the optimization

NOTE: The resulting constraints will likely be more expensive to evaluate
    and less accurate than writing the constraints solver from scratch.
    """
    ndim = 1 #XXX: better, increase in while loop catching IndexError ?
    if nvars is not None: ndim = nvars
    elif guess is not None: ndim = len(guess)
    elif lower_bounds is not None: ndim = len(lower_bounds)
    elif upper_bounds is not None: ndim = len(upper_bounds)

    def cost(x): return 1.

    #XXX: don't allow solver string as a short-cut?
    if solver is None or solver == 'diffev':
        from mystic.solvers import DifferentialEvolutionSolver as TheSolver
        solver = TheSolver(ndim, min(40, ndim*5))
    elif solver == 'diffev2':
        from mystic.solvers import DifferentialEvolutionSolver2 as TheSolver
        solver = TheSolver(ndim, min(40, ndim*5))
    elif solver == 'fmin_powell': #XXX: better as the default? (it's not random)
        from mystic.solvers import PowellDirectionalSolver as TheSolver
        solver = TheSolver(ndim)
    elif solver == 'fmin':
        from mystic.solvers import NelderMeadSimplexSolver as TheSolver
        solver = TheSolver(ndim)
    
    if termination is None:
        from mystic.termination import ChangeOverGeneration as COG
        termination = COG()
    if guess != None:
        solver.SetInitialPoints(guess) #XXX: nice if 'diffev' also had methods
    else:
        solver.SetRandomInitialPoints(lower_bounds, upper_bounds)
    if lower_bounds or upper_bounds:
        solver.SetStrictRanges(lower_bounds, upper_bounds)
    if hasattr(constraints, 'iter') and hasattr(constraints, 'error'):
        solver.SetPenalty(constraints) #i.e. is a penalty function
    else: # is a constraints solver
        solver.SetConstraints(constraints)
    solver.Solve(cost, termination)
    soln = solver.bestSolution

    from numpy import ndarray, array
    if isinstance(soln, ndarray) and not isinstance(guess, ndarray):
        soln = soln.tolist()
    elif isinstance(guess, ndarray) and not isinstance(soln, ndarray):
        soln = array(soln)  #XXX: or always return a list ?

    return soln #XXX: check with 'issolution' ?


def as_constraint(penalty, *args, **kwds):
    """Convert a penalty function to a constraints solver.

Inputs:
    penalty -- a penalty function

Additional Inputs:
    lower_bounds -- list of lower bounds on solution values.
    upper_bounds -- list of upper bounds on solution values.
    nvars -- number of parameter values.
    solver -- the mystic solver to use in the optimization
    termination -- the mystic termination to use in the optimization
    """
    def constraint(x): #XXX: better to enable args kwds for penalty ?
        return solve(penalty, x, *args, **kwds)
    return constraint


def as_penalty(constraint, ptype=None, *args, **kwds):
    """Convert a constraints solver to a penalty function.

Inputs:
    constraint -- a constraints solver
    ptype -- penalty function type [default: quadratic_equality]

Additional Inputs:
    args -- arguments for the constraints solver [default: ()]
    kwds -- keyword arguments for the constraints solver [default: {}]
    k -- penalty multiplier
    h -- iterative multiplier
    """
    def rnorm(x, *argz, **kwdz):
        error = 0.0
        constrained = constraint(x, *argz, **kwdz)
        for i in range(len(x)):
            error += (constrained[i] - x[i])**2  #XXX: better rnorm ?
        error = error**0.5
        return error

    if ptype is None:
        from mystic.penalty import quadratic_equality
        ptype = quadratic_equality

    @ptype(rnorm, *args, **kwds) #XXX: yes to k,h... but otherwise?
    def penalty(x):
        return 0.0

    return penalty


# EOF

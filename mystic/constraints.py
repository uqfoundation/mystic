#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""Tools for building and applying constraints and penalties.
"""

__all__ = ['with_penalty','with_constraint','as_penalty','as_constraint',
           'with_mean','with_variance','with_std','with_spread','normalized',
           'issolution','solve','discrete','integers','near_integers',
           'unique','has_unique','impose_unique','bounded','impose_bounds',
           'impose_as','impose_at','impose_measure','impose_position',
           'impose_weight','and_','or_','not_','vectorize','boundsconstrain']

from mystic.math.measures import *
from mystic.math import almostEqual

def vectorize(constraint, axis=1):
    """vectorize a constraint for 2D input, `x' = k(x)` where `x` is 2D

    Input:
        constraint -- a mystic constraint, c, where `x' = c(x)`, `x` is a list
        axis -- axis to apply constraints to, must be 0 or 1 (default is 1)

    Output:
        transform -- a transform function, k, where `x' = k(x)`, `x` is 2D array

    Notes:
        Produces a constraints function that is of the form required by
        sklearn.preprocessing.FunctionTransformer(func=transform).
        Input to the tranform is a 2D numpy array of shape (samples, features).

    For example:
        >>> from mystic.constraints import (impose_bounds, integers,
        ...                                 with_mean, and_)
        >>> cons = and_(impose_bounds([(0,5),(7,10)])(lambda x:x),
        ...             integers()(lambda x:x), with_mean(6.0)(lambda x:x))
        >>> import numpy as np
        >>> data = np.random.randn(6,4)
        >>> c = vectorize(cons, axis=0)
        >>> c(data)
        array([[ 3,  9,  3, 10],
               [ 7,  8,  7,  4],
               [ 9,  3,  7,  7],
               [ 7,  3,  8,  7],
               [ 3,  5,  4,  4],
               [ 7,  8,  7,  4]])
        >>> _.mean(axis=0)
        array([6., 6., 6., 6.])
        >>> c = vectorize(cons, axis=1)
        >>> c(data)
        array([[ 3,  3,  9,  9],
               [ 8,  3,  4,  9],
               [ 5, 10,  4,  5],
               [ 7,  8,  7,  2],
               [ 2,  4,  8, 10],
               [ 7, 10,  5,  2]])
        >>> _.mean(axis=1)
        array([6., 6., 6., 6., 6., 6.])
        >>> k = FunctionTransformer(func=c)
        >>> k.fit(data).transform(data).mean(axis=1)
        array([6., 6., 6., 6., 6., 6.])
    """
    if not axis in (0,1):
        msg = "axis = %s, must be either 0 or 1" % axis
        raise ValueError(msg)
    _doc = """mystic kernel transform `x' = k(x)` with x a 2D numpy array.

        """
    if axis:
        def transform(x, *args, **kwds):
            import numpy as np
            c = lambda x,*args,**kwds: np.array([constraint(xi, *args, **kwds) for xi in x])
            return c(x, *args, **kwds)
    else:
        def transform(x, *args, **kwds):
            import numpy as np
            c = lambda x,*args,**kwds: np.array([constraint(xi, *args, **kwds) for xi in x.T]).T
            return c(x, *args, **kwds)
    doc = constraint.__doc__
    transform.__doc__ = _doc + doc if doc else _doc
    return transform


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
      penalty.func = condition
      penalty.ptype = ptype.__name__ 
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
    >>> print(x)
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
impose_expectation(m, f, npts, bounds=None, weights=None, tol=None, **kwds)
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
    >>> print(x)
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
    >>> print(x)
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


def with_std(target):
    """bind a standard deviation constraint to a given constraints function.

Inputs:
    target -- the target standard deviation

A constraints function takes an iterable x as input, returning a modified x.
This function is an "outer" coupling of "impose_std" onto another
constraints function c(x), such that:  x' = impose_std(target, c(x)).

    For example:
    >>> @with_std(1.0)
    ... def constraint(x):
    ...   x[-1] = x[0]
    ...   return x
    ... 
    >>> x = constraint([1,2,3])
    >>> print(x)
    [0.6262265521467858, 2.747546895706428, 0.6262265521467858]
    >>> std(x)
    0.99999999999999956
    """
    return with_variance(target**2)


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
    >>> print(x)
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
    guess -- list of parameter values proposed to solve the constraints
    tol -- residual error magnitude for which constraints are considered solved

    For example:
    >>> @normalized()
    ... def constraint(x):
    ...   return x
    ... 
    >>> constraint([.5,.5])
    [0.5, 0.5]
    >>> issolution(constraint, [.5,.5])
    True
    >>> 
    >>> from mystic.penalty import quadratic_inequality
    >>> @quadratic_inequality(lambda x: x[0] - x[1] + 10)
    ... def penalty(x):
    ...   return 0.0
    ... 
    >>> penalty([-10,.5])
    0.0
    >>> issolution(penalty, [-10,.5])
    True
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
          lower_bounds=None, upper_bounds=None, termination=None, \
          tightrange=None, cliprange=None):
    """Use optimization to find a solution to a set of constraints.

Inputs:
    constraints -- a constraints solver function or a penalty function

Additional Inputs:
    guess -- list of parameter values proposed to solve the constraints.
    lower_bounds -- list of lower bounds on solution values.
    upper_bounds -- list of upper bounds on solution values.
    nvars -- number of parameter values.
    solver -- the mystic solver to use in the optimization.
    termination -- the mystic termination to use in the optimization.
    tightrange -- if True, impose bounds and constraints concurrently.
    cliprange -- if True, bounding constraints clip exterior values.

NOTE: The resulting constraints will likely be more expensive to evaluate
    and less accurate than writing the constraints solver from scratch.

NOTE: The ensemble solvers are available, using the default NestedSolver,
    where the keyword 'guess' can be used to set the number of solvers.

NOTE: The default solver is 'diffev', with npop=min(40, ndim*5). The default
    termination is ChangeOverGeneration(), and the default guess is randomly
    selected points between the upper and lower bounds.
    """
    npts = 8
    if type(guess) is int: npts, guess = guess, None

    ndim = 1 #XXX: better, increase in while loop catching IndexError ?
    if nvars is not None: ndim = nvars
    elif guess is not None: ndim = len(guess)
    elif lower_bounds is not None: ndim = len(lower_bounds)
    elif upper_bounds is not None: ndim = len(upper_bounds)

    def cost(x): return 1.

    #XXX: don't allow solver string as a short-cut?
    ensemble = False
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
    elif solver == 'buckshot':
        from mystic.solvers import BuckshotSolver as TheSolver
        solver = TheSolver(ndim, max(8, npts)) #XXX: needs better default?
        ensemble = True
    elif solver == 'lattice':
        from mystic.solvers import LatticeSolver as TheSolver
        solver = TheSolver(ndim, max(8, npts)) #XXX: needs better default?
        ensemble = True
    elif solver == 'sparsity':
        from mystic.solvers import SparsitySolver as TheSolver
        solver = TheSolver(ndim, max(8, npts)) #XXX: needs better default?
        ensemble = True
    
    if termination is None:
        from mystic.termination import ChangeOverGeneration as COG
        termination = COG()
    if not ensemble:
        if guess is not None:
            solver.SetInitialPoints(guess) #XXX: nice if 'diffev' had methods
        else:
            solver.SetRandomInitialPoints(lower_bounds, upper_bounds)
    if lower_bounds or upper_bounds: #XXX: disable/hardwire tight, clip?
        kwds = dict(tight=tightrange, clip=cliprange)
        solver.SetStrictRanges(lower_bounds, upper_bounds, **kwds)
    if hasattr(constraints, 'iter') and hasattr(constraints, 'error'):
        solver.SetPenalty(constraints) #i.e. is a penalty function
    else: # is a constraints solver
        solver.SetConstraints(constraints)
    from numpy import seterr
    settings = seterr(all='ignore')
    solver.Solve(cost, termination)
    seterr(**settings)
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
    solver -- the mystic solver to use in the optimization.
    termination -- the mystic termination to use in the optimization.
    tightrange -- if True, impose bounds and constraints concurrently.
    cliprange -- if True, bounding constraints clip exterior values.

NOTE: The default solver is 'diffev', with npop=min(40, ndim*5). The default
    termination is ChangeOverGeneration(), and the default guess is randomly
    selected points between the upper and lower bounds.
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

    penalty.func = rnorm
    penalty.ptype = ptype.__name__ 
    return penalty


# constraints 'language'
#XXX: should have the same interface as 'inner' and 'outer'?
def and_(*constraints, **settings): #XXX: not a decorator, should be?
    """combine several constraints into a single constraint

Inputs:
    constraints -- constraint functions

Additional Inputs:
    maxiter -- maximum number of iterations to attempt to solve [default: 100]
    onexit -- function x' = f(x) to call on success [default: None]
    onfail -- function x' = f(x) to call on failure [default: None]

NOTE: 
    If a repeating cycle is detected, some of the inputs may be randomized.
    """
    import itertools as it
    import random as rnd
    n = len(constraints)
    if 'maxiter' in settings:
        maxiter = settings['maxiter'] * n; del settings['maxiter']
    else: maxiter = 100 * n
    if 'onexit' in settings:
        onexit = settings['onexit']; del settings['onexit']
    else: onexit = None
    if 'onfail' in settings:
        onfail = settings['onfail']; del settings['onfail']
    else: onfail = None
    def _constraint(x): #XXX: inefficient, rewrite without append
        x = [x.tolist() if hasattr(x, 'tolist') else x[:]]
        # apply all constraints once
        e = None
        for c in constraints:
            try:
                ci = c(x[-1][:])
            except ZeroDivisionError as e:
                ci = x[-1][:] #XXX: do something else?
            x.append(ci.tolist() if hasattr(ci, 'tolist') else ci)
        if all(xi == x[-1] for xi in x[1:]) and e is None:
            return x[-1] if onexit is None else onexit(x[-1][:])
        # cycle constraints until there's no change
        _constraints = it.cycle(constraints) 
        for j in range(n,maxiter):
            e = None
            try:
                ci = next(_constraints)(x[-1][:])
            except ZeroDivisionError as e:
                ci = x[-1][:] #XXX: do something else?
            x.append(ci.tolist() if hasattr(ci, 'tolist') else ci)
            if all(xi == x[-1] for xi in x[-n:]) and e is None:
                return x[-1] if onexit is None else onexit(x[-1][:])
            # may be trapped in a cycle... randomize
            if x[-1] == x[-(n+1)]:
                x[-1] = [(i+rnd.randint(-1,1))*rnd.random() for i in x[-1]]
            if not j%(2*n):
                del x[:n]
        # give up #XXX: or fail with Error?
        return x[-1] if onfail is None else onfail(x[-1][:])
    cf = lambda x: _constraint(x)
    cfdoc = "\n-- AND --\n".join(c.__doc__ for c in constraints if c.__doc__)
    cfdoc = '{ '+ cfdoc +' }' if cfdoc else cfdoc #XXX: can be {c} w/no AND
    cf.__doc__ = cfdoc.rstrip('\n')
    cf.__name__ = 'constraint'
    return cf


#XXX: should have the same interface as 'inner' and 'outer'?
def or_(*constraints, **settings): #XXX: not a decorator, should be?
    """create a constraint that is satisfied if any constraints are satisfied

Inputs:
    constraints -- constraint functions

Additional Inputs:
    maxiter -- maximum number of iterations to attempt to solve [default: 100]
    onexit -- function x' = f(x) to call on success [default: None]
    onfail -- function x' = f(x) to call on failure [default: None]

NOTE: 
    If a repeating cycle is detected, some of the inputs may be randomized.
    """
    import itertools as it
    import random as rnd
    n = len(constraints)
    if 'maxiter' in settings:
        maxiter = settings['maxiter'] * n; del settings['maxiter']
    else: maxiter = 100 * n
    if 'onexit' in settings:
        onexit = settings['onexit']; del settings['onexit']
    else: onexit = None
    if 'onfail' in settings:
        onfail = settings['onfail']; del settings['onfail']
    else: onfail = None
    def _constraint(x): #XXX: inefficient, rewrite without append
        x = [x.tolist() if hasattr(x, 'tolist') else x[:]]
        # check if initial input is valid
        e = None
        for c in constraints:
            try:
                ci = c(x[0][:])
            except ZeroDivisionError as e:
                ci = x[0][:] #XXX: do something else?
            x.append(ci.tolist() if hasattr(ci, 'tolist') else ci)
            if x[-1] == x[0] and e is None:
                return x[-1] if onexit is None else onexit(x[-1][:])
        # cycle constraints until there's no change
        _constraints = it.cycle(constraints) 
        for j in range(n,maxiter):
            e = None
            try:
                ci = next(_constraints)(x[-n][:])
            except ZeroDivisionError as e:
                ci = x[-n][:] #XXX: do something else?
            x.append(ci.tolist() if hasattr(ci, 'tolist') else ci)
            if x[-1] == x[-(n+1)] and e is None:
                return x[-1] if onexit is None else onexit(x[-1][:])
            else: # may be trapped in a rut... randomize
                x[-1] = x[-rnd.randint(1,n)]
            if not j%(2*n):
                del x[:n]
        # give up #XXX: or fail with Error?
        return x[-1] if onfail is None else onfail(x[-1][:])
    cf = lambda x: _constraint(x)
    cfdoc = "\n-- OR --\n".join(c.__doc__ for c in constraints if c.__doc__)
    cfdoc = '[ '+ cfdoc +' ]' if cfdoc else cfdoc #XXX: can be [c] w/no OR
    cf.__doc__ = cfdoc.rstrip('\n')
    cf.__name__ = 'constraint'
    return cf


def not_(constraint, **settings): #XXX: not a decorator, should be?
    """invert the region where the given constraints are valid, then solve

Inputs:
    constraint -- constraint function

Additional Inputs:
    maxiter -- maximum number of iterations to attempt to solve [default: 100]
    onexit -- function x' = f(x) to call on success [default: None]
    onfail -- function x' = f(x) to call on failure [default: None]

NOTE: 
    If a repeating cycle is detected, some of the inputs may be randomized.
    """
    import random as rnd
    if 'maxiter' in settings:
        maxiter = settings['maxiter']; del settings['maxiter']
    else: maxiter = 100
    if 'onexit' in settings:
        onexit = settings['onexit']; del settings['onexit']
    else: onexit = None
    if 'onfail' in settings:
        onfail = settings['onfail']; del settings['onfail']
    else: onfail = None
    def _constraint(x):
        # check if initial input is valid, else randomize and try again
        for j in range(0,maxiter):
            try:
                if constraint(x[:]) != x:
                    return x[:] if onexit is None else onexit(x[:])
            except ZeroDivisionError as e:
                pass #XXX: do something else?
            x = [(i+rnd.randint(-1,1))*rnd.random() for i in x]
        # give up #XXX: or fail with Error?
        return x[:] if onfail is None else onfail(x[:])
    cf = lambda x: _constraint(x)
    cfdoc = 'NOT( '+ constraint.__doc__ +' )' if constraint.__doc__ else ""
    cf.__doc__ = cfdoc.rstrip('\n')
    cf.__name__ = 'constraint'
    return cf


from numpy import asfarray, asarray, choose, zeros, ones, ndarray
from numpy import shape, broadcast, empty, atleast_1d
#from random import sample, choice
def discrete(samples, index=None):
    """impose a discrete set of input values for the selected function

The function's input will be mapped to the given discrete set

>>> @discrete([1.0, 2.0])
... def identity(x):
...     return x

>>> identity([0.123, 1.789, 4.000])
[1.0, 2.0, 2.0]

>>> @discrete([1,3,5,7], index=(0,3))
... def squared(x):
....    return [i**2 for i in x]

>>> squared([0,2,4,6,8,10])
[1, 4, 16, 25, 64, 100]"""
    samples = [asarray(samples)]
    samples[0].sort()
    if isinstance(index, int): index = (index,)
    index = [index]

    #XXX: refactor to use points_factory(samples)
    def _points(alist):
        alist = asarray(alist)
        alist.sort()
        samples[0] = alist

    def _index(alist=None):
        index[0] = alist

    #XXX: refactor to use argnear_factory(samples)
    def _argnear(xi):
        arghi = sum(xi > samples[0])
        arglo = max(0, arghi - 1) # minimum = x[0]
        if arghi == len(samples[0]): 
            arghi = arglo         # maximum = x[-1]
        return arglo, arghi

    def _near(xi, lo, hi):
        if hi - xi < xi - lo: 
            return hi
        return lo

#   argnear = vectorize(_argnear)  #XXX: doesn't pickle well in all cases
#   near = vectorize(_near)        #XXX: doesn't pickle well in all cases
    def argnear(x):
        #RESULT: (array([0,0,1]),array([0,1,2])) *or* (array(0),array(1))
        flatten = False
        if not len(shape(x)):
            flatten = True
            x = [x]
        result = tuple(i for i in asarray(list(map(_argnear, x))).T)
        if flatten:
            result = tuple(asarray(i[0]) for i in result)
        return result
    def near(x, l, h):
        #RESULT: array(3) & array([3, 4])
        a = broadcast(x,l,h)
        has_iterable = a.shape
        x,l,h = tuple(atleast_1d(i) for i in (x,l,h))
        b = broadcast(x,l,h)
        _x,_l,_h = (empty(b.shape), empty(b.shape), empty(b.shape))
        _x.flat = [i for i in x]
        _l.flat = [i for i in l]
        _h.flat = [i for i in h]
        result = asarray(list(map(_near, x, l, h)))
        if not has_iterable:
            result = asarray(result[0])
        return result

    def dec(f):
        def func(x, *args, **kwds):
            if isinstance(x, ndarray): xtype = asarray
            else: xtype = type(x)
            arglo, arghi = argnear(x)
            xp = near(x, samples[0][arglo], samples[0][arghi])
            # create a choice array from given indices
            #FIXME: better ways to do the following
            if index[0] is None: 
                mask = ones(xp.size, dtype=bool)
            else:
                mask = zeros(xp.size, dtype=bool)
                try: mask[sorted(index[0], key=abs)] = True
                except IndexError: pass
            xp = xtype(choose(mask, (x,xp)))
            return f(xp, *args, **kwds)
        func.samples = _points
        func.index = _index
        return func
    return dec


from numpy import round, abs
def integers(ints=True, index=None):
    """impose the set of integers (by rounding) for the given function

The function's input will be mapped to the ints, where:
  - if ints is True, return results as ints; otherwise, use floats
  - if index tuple provided, only round at the given indices

>>> @integers()
... def identity(x):
...     return x

>>> identity([0.123, 1.789, 4.000])
[0, 2, 4]

>>> @integers(ints=float, index=(0,3,4))
... def squared(x):
....    return [i**2 for i in x]

>>> squared([0.12, 0.12, 4.01, 4.01, 8, 8])
[0.0, 0.0144, 16.080099999999998, 16.0, 64.0, 64.0]"""
    #HACK: allow ints=False or ints=int
    _ints = [(int if ints else float) if isinstance(ints, bool) else ints]
    if isinstance(index, int): index = (index,)
    index = [index]

    def _index(alist=None):
        index[0] = alist

    def _type(ints=None):
        _ints[0] = (int if ints else float) if isinstance(ints, bool) else ints

    def dec(f):
        def func(x,*args,**kwds):
            if isinstance(x, ndarray): xtype = asarray
            else: xtype = type(x)
            ##### ALT #####
           #pos = range(len(x))
           #pos = [pos[i] for i in index[0]] if index[0] else pos
           #xp = [float(round(xi) if i in pos else xi) for i,xi in enumerate(x)]
            ###############
            xp = round(x)
            if index[0] is None:
                mask = ones(xp.size, dtype=bool)
            else:
                mask = zeros(xp.size, dtype=bool)
                try: mask[sorted(index[0], key=abs)] = True
                except IndexError: pass
            xp = choose(mask, (x,xp)).astype(_ints[0])
            ###############
            return f(xtype(xp), *args, **kwds)
        func.index = _index
        func.type = _type
        return func
    return dec


from random import randrange, shuffle
def unique(seq, full=None):
    """replace the duplicate values with unique values in 'full'

    If full is a type (int or float), then unique values of the given type
    are selected from range(min(seq),max(seq)). If full is a dict of
    {'min':min, 'max':max}, then unique floats are selected from
    range(min(seq),max(seq)). If full is a sequence (list or set), then
    unique values are selected from the given sequence. 

    For example:
    >>> unique([1,2,3,1,2,4], range(11))
    [1, 2, 3, 9, 8, 4]
    >>> unique([1,2,3,1,2,9], range(11))
    [1, 2, 3, 8, 5, 9]
    >>> try:
    ...     unique([1,2,3,1,2,13], range(11))
    ... except ValueError:
    ...     pass
    ...
    >>>
    >>> unique([1,2,3,1,2,4], {'min':0, 'max':11})
    [1, 2, 3, 4.175187820357143, 2.5407265707465716, 4]
    >>> unique([1,2,3,1,2,4], {'min':0, 'max':11, 'type':int})
    [1, 2, 3, 6, 8, 4]
    >>> unique([1,2,3,1,2,4], float)
    [1, 2, 3, 1.012375036824941, 3.9821250727509905, 4]
    >>> unique([1,2,3,1,2,10], int)
    [1, 2, 3, 9, 6, 10]
    >>> try:
    ...     unique([1,2,3,1,2,4], int)
    ... except ValueError:
    ...     pass
    ...
    """
    unique = set()
    # replace all duplicates with 'None'
    seq = [x if x not in unique and not unique.add(x) else None for x in seq]
    lseq = len(seq)
    # check type if full not specified
    if full is None:
        if all([isinstance(x, int) for x in unique]): full = int
        else: full = float
        ok = True
    else: ok = False
    # check all unique show up in 'full'
    if full in (int,float): # specified type, not range
        ok = ok or full==float or all([isinstance(x, int) for x in unique])
        msg = "not all items are of type='%s'" % full.__name__
        _min = min(unique)
        _max = max(unique)
    elif isinstance(full, dict): # specified min/max for floats
        if 'type' in full: #NOTE: undocumented keys: min,max,type
            _type = full['type']; del full['type']
        else: _type = float
        minu = min(unique)
        maxu = max(unique)
        _min = full['min'] if 'min' in full else minu
        _max = full['max'] if 'max' in full else maxu
        ok = minu >= _min and maxu < _max
        if not ok:
            oops = list(unique - set(range(_min,_max)))
            msg = "x=%s not in %s <= x < %s" % (oops[-1],_min,_max)
        full = _type
    else: # full is a list of all possible values
        ok = unique.issubset(full)
        if not ok:
            oops = list(unique - set(full))
            msg = "%s not in given set" % oops[-1]
        _min = _max = None
    if not ok: raise ValueError(msg)
    # check if a unique sequence is possible to build
    if full is float:
        if _min == _max and lseq > 1:
            msg = "no unique len=%s sequence with %s <= x <= %s" % (lseq,_min,_max)
            raise ValueError(msg)
        # replace the 'None' values in seq with 'new' values
        #XXX: HIGHLY UNLIKELY two numbers will be the same, but possible
        return [randrange(_min,_max,_int=float) if x is None else x for x in seq]
    # generate all possible values not found in 'unique'
    if full is int:
        if max(lseq - (_max+1 - _min), 0):
            msg = "no unique len=%s sequence with %s <= x <= %s" % (lseq,_min,_max)
            raise ValueError(msg)
        new = list(set(range(_min,_max+1)) - unique)
    else:
        if lseq > len(full):
            msg = "no unique len=%s sequence in given set" % lseq
            raise ValueError(msg)
        new = list(set(full) - unique)
    # ensure randomly ordered
    shuffle(new)
    # replace the 'None' values in seq with 'new' values
    return [new.pop() if x is None else x for x in seq]


#XXX: enable impose_unique on selected members of x? (see constraints.integers)

def impose_unique(seq=None):
    """ensure all values are unique and found in the given set

    For example:
    >>> @impose_unique(range(11))
    ... def doit(x):
    ...     return x
    ... 
    >>> doit([1,2,3,1,2,4])
    [1, 2, 3, 9, 8, 4]
    >>> doit([1,2,3,1,2,10])
    [1, 2, 3, 8, 5, 10]
    >>> try:
    ...     doit([1,2,3,1,2,13])
    ... except ValueError:
    ...     print("Bad Input")
    ...
    Bad Input
"""
    def dec(f):
        def func(x,*args,**kwds):
            return f(unique(x, seq),*args,**kwds)
        return func
    return dec

from numpy import array, intersect1d, inf, isnan, where, choose, clip as _clip
from numpy.random import uniform, choice
def bounded(seq, bounds, index=None, clip=True, nearest=True):
    """bound a sequence by bounds = [min,max]

    For example:
    >>> sequence = [0.123, 1.244, -4.755, 10.731, 6.207]
    >>> 
    >>> bounded(sequence, (0,5))
    array([0.123, 1.244, 0.   , 5.   , 5.   ])
    >>> 
    >>> bounded(sequence, (0,5), index=(0,2,4))
    array([ 0.123,  1.244,  0.   , 10.731,  5.   ])
    >>> 
    >>> bounded(sequence, (0,5), clip=False)
    array([0.123     , 1.244     , 3.46621839, 1.44469038, 4.88937466])
    >>> 
    >>> bounds = [(0,5),(7,10)]
    >>> my.constraints.bounded(sequence, bounds)
    array([ 0.123,  1.244,  0.   , 10.   ,  7.   ])
    >>> my.constraints.bounded(sequence, bounds, nearest=False)
    array([ 0.123,  1.244,  7.   , 10.   ,  5.   ])
    >>> my.constraints.bounded(sequence, bounds, nearest=False, clip=False) 
    array([0.123     , 1.244     , 0.37617154, 8.79013111, 7.40864242])
    >>> my.constraints.bounded(sequence, bounds, clip=False)
    array([0.123     , 1.244     , 2.38186577, 7.41374049, 9.14662911])
    >>> 
"""
    seq = array(seq) #XXX: asarray?
    if bounds is None or not bounds: return seq
    if isinstance(index, int): index = (index,)
    if not hasattr(bounds[0], '__len__'): bounds = (bounds,)
    bounds = asfarray(bounds).T  # is [(min,min,...),(max,max,...)]
    # convert None to -inf or inf
    bounds[0][isnan(bounds[0])] = -inf
    bounds[1][isnan(bounds[1])] = inf
    # find indicies of the elements that are out of bounds
    at = where(sum((lo <= seq)&(seq <= hi) for (lo,hi) in bounds.T).astype(bool) == False)[-1]
    # find the intersection of out-of-bound and selected indicies
    at = at if index is None else intersect1d(at, index)
    if not len(at): return seq
    if clip:
        if nearest: # clip at closest bounds
            seq_at = seq[at]
            seq[at] = _clip(seq_at, *(b[abs(seq_at.reshape(-1,1)-b).argmin(axis=1)] for b in bounds))
        else: # clip in randomly selected interval
            picks = choice(len(bounds.T), size=at.shape)
            seq[at] = _clip(seq[at], bounds[0][picks], bounds[1][picks])
        return seq
    # limit to +/- 1e300 #XXX: better defaults?
    bounds[0][bounds[0] < -1e300] = -1e300
    bounds[1][bounds[1] > 1e300] = 1e300
    if nearest:
        seq_at = seq[at]
        seq[at] = choose(array([abs(seq_at.reshape(-1,1) - b).min(axis=1) for b in bounds.T]).argmin(axis=0), [uniform(0,1, size=at.shape) * (hi - lo) + lo for (lo,hi) in bounds.T])
    else: # randomly choose a value in one of the intervals
        seq[at] = choose(choice(len(bounds.T), size=at.shape), [uniform(0,1, size=at.shape) * (hi - lo) + lo for (lo,hi) in bounds.T])
    return seq

def impose_bounds(bounds, index=None, clip=True, nearest=True):
    """generate a function where bounds=[min,max] on a sequence are imposed

    For example:
    >>> sequence = [0.123, 1.244, -4.755, 10.731, 6.207]
    >>> 
    >>> @impose_bounds((0,5))       
    ... def simple(x):
    ...   return x
    ... 
    >>> simple(sequence)
    [0.123, 1.244, 0.0, 5.0, 5.0]
    >>> 
    >>> @impose_bounds((0,5), index=(0,2,4))
    ... def double(x):
    ...   return [i*2 for i in x]
    ... 
    >>> double(sequence)
    [0.246, 2.488, 0.0, 21.462, 10.0]
    >>> 
    >>> @impose_bounds((0,5), index=(0,2,4), clip=False)
    ... def square(x):
    ...   return [i*i for i in x]
    ... 
    >>> square(sequence)
    [0.015129, 1.547536, 14.675791119810688, 115.154361, 1.399551896073788]
    >>> 
    >>> @impose_bounds([(0,5),(7,10)])
    ... def simple(x):
    ...   return x
    ... 
    >>> simple(sequence)
    [0.123, 1.244, 0.0, 10.0, 7.0]
    >>> 
    >>> @impose_bounds([(0,5),(7,10)], nearest=False)
    ... def simple(x):
    ...   return x
    ... 
    >>> simple(sequence)
    [0.123, 1.244, 0.0, 5.0, 5.0]
    >>> simple(sequence)
    [0.123, 1.244, 7.0, 10.0, 5.0]
    >>> 
    >>> @impose_bounds({0:(0,5), 2:(0,5), 4:[(0,5),(7,10)]})
    ... def simple(x):
    ...   return x
    ... 
    >>> simple(sequence)
    [0.123, 1.244, 0.0, 10.731, 7.0]
    >>>
    >>> @impose_bounds({0:(0,5), 2:(0,5), 4:[(0,5),(7,10)]}, index=(0,2)) 
    ... def simple(x):
    ...   return x
    ... 
    >>> simple(sequence)
    [0.123, 1.244, 0.0, 10.731, 6.207]
    """
    ### bounds is a dict, index filters the dict
    ### keys of bounds are the index, if index=None apply to all
    ### *but* if bounds is given as list (of tuples), apply to seleted index
    if isinstance(index, int): index = (index,)
    # bounds are list, index is tuple => {i:bounds} for each i in index
    if type(bounds) is not dict:
        if index is not None:
            bounds = dict((i,bounds) for i in index)
            index = None
        # bounds are list, index is None => {None:bounds} apply to all index
        else: bounds = {None:bounds}
    # bounds are dict, index is None => do nothing -- is the preferred input
    # bounds are dict, index is tuple => filter bounds, then as above
    else:
        if None in bounds and len(bounds) > 1: # has either entries or None
            msg = "bounds got multiple entries for '%d'" % set(bounds).difference((None,)).pop()
            raise TypeError(msg)
        if index is not None:
            if None in index and len(index) > 1: # has either entries or None
                msg = "index got multiple entries for '%d'" % set(index).difference((None,)).pop()
                raise TypeError(msg)
            if None in bounds:
                bounds = dict((i,bounds[None]) for i in index)
            else:
                index = set(index).intersection(bounds)
                bounds = dict((i,bounds[i]) for i in index)
            index = None

    clip = [clip]
    nearest = [nearest]

    def _clip(clipped=True):
        clip[0] = clipped

    def _near(nearer=True):
        nearest[0] = nearer

    def dec(f):
        def func(x,*args,**kwds):
            if isinstance(x, ndarray): xtype = asarray
            else: xtype = type(x)
            xp = x
            for i in bounds:
                xp = bounded(xp, bounds[i], i, clip[0], nearest[0])
            return f(xtype(xp), *args, **kwds)
        func.clip = _clip
        func.nearest = _near
        func.__bounds__ = bounds
        func.__wrapped__ = f
        func.__doc__ = f.__doc__
        return func
    return dec


def boundsconstrain(min, max, **kwds):
    """build a constraint from the bounds (min,max)

Input:
    min: list of floats specifying the lower bounds
    max: list of floats specifying the upper bounds

Additional Input:
    symbolic: bool, if True, use symbolic constraints [default: True]
    clip: bool, if True, clip exterior values to the bounds [default: True]

Notes:
    Prepares a constraint function that imposes the bounds. The intent is
    so that the bounds and other constraints can be imposed concurrently
    (e.g. using `mystic.constraints.and_`). This is slower but more robust
    than applying the bounds sequential with the other constraints (the
    default for a solver).

    For entries where there is no upper bound, use either `inf` or `None`.
    Similarly for entries where there is no lower bound. When no upper
    bound exists for any of the entries, ``max`` should be an iterable
    with all entries of `inf` or `None`. Again, similarly for lower bounds.

    If `symbolic=True`, use symbolic constraints to impose the bounds;
    otherwise use `mystic.constraints.impose_bounds`. Using `clip=False`
    requires `symbolic=False`.
    """
    #NOTE: symbolic_bounds doesn't handle non-iterables (min=None or min=0)
    symbolic = kwds['symbolic'] if 'symbolic' in kwds else True
    clip = kwds['clip'] if 'clip' in kwds else True
    if symbolic and not clip:
        raise NotImplementedError("symbolic must clip to the nearest bound")
    # build the constraint
    if not symbolic: #XXX: much slower than symbolic
        cons = dict((i,j) for (i,j) in enumerate(zip(min, max)))
        cons = impose_bounds(cons, clip=clip)(lambda x: x)
        return cons
    import mystic.symbolic as ms #XXX: randomness due to sympy?
    cons = ms.symbolic_bounds(min, max) #XXX: how clipping with symbolic?
    cons = ms.generate_constraint(ms.generate_solvers(ms.simplify(cons))) #join?
    return cons


#XXX: enable near_integers and has_unique on selected members of x?
#FIXME: the following don't seem to belong in 'mystic.constraints'

from numpy import round, abs
def near_integers(x): # use as a penalty for int programming
    """the sum of all deviations from int values"""
    return abs(x - round(x)).sum()

def has_unique(x): # use as a penalty for unique numbers
    """check for uniqueness of the members of x"""
    return sum(x.count(xi) for xi in x)
   #return len(x) - len(set(x))


##### collapse constraints #####
from mystic.tools import synchronized, connected

#XXX: this is just a less-flexible tools.synchronized... is it useful?
def _impose_as(mask):
    """generate a function, where some input tracks another input

mask should be a dictionary of positional index and tracked index (e.g. {0:1}),
where keys and values should be different integers. However, if a tuple is
provided instead of the tracked index (e.g. {0:(1,2)}), the second member of
the tuple will be used as an additive offset for the tracked index. The mask
will be applied to the input, before the decorated function is called.

functions are expected to take a single argument, a n-dimensional list or array,
where the mask will be applied to the input array.

operations within a single mask are unordered. If a specific ordering of
operations is required, apply multiple masks in the desired order.

For example,
    >>> @_impose_as({0:1, 2:(3,10)})
    ... def same(x):
    ...     return x
    ... 
    >>> same([0,1,2,3,4,5])
    [1, 1, 13, 3, 4, 5]
    >>> same([-1,-2,-3])
    [-2, -2, -3]
    >>> same([-1,-2,-3,-4])
    [-2, -2, 6, -4]
    >>> 
    >>> @_impose_as({1:2})
    ... @_impose_as({0:1})
    ... def doit(x):
    ...     return [i+1 for i in x]
    ... 
    >>> doit([0,1,2,3,4,5])
    [3, 3, 3, 4, 5, 6]
    >>> doit([-1,-2,-3])
    [-2, -2, -2]
    >>> doit([-1,-2,-3,-4])
    [-2, -2, -2, -3]
    """
    offset = lambda i: (i if isinstance(i, int) else (i[0], lambda x:(x+i[1])))
    return synchronized(dict((i,offset(j)) for (i,j) in getattr(mask, 'iteritems', mask.items)()))


def impose_as(mask, offset=None):
    """generate a function, where some input tracks another input

mask should be a set of tuples of positional index and tracked index,
where the tuple should contain two different integers. The mask will be
applied to the input, before the decorated function is called.

The offset is applied to the second member of the tuple, and can accumulate.

For example,
    >>> @impose_as([(0,1),(3,1),(4,5),(5,6),(5,7)])
    ... def same(x):
    ...   return x
    ... 
    >>> same([9,8,7,6,5,4,3,2,1])
    [9, 9, 7, 9, 5, 5, 5, 5, 1]
    >>> same([0,1,0,1])
    [0, 0, 0, 0]
    >>> same([-1,-2,-3,-4,-5,-6,-7])
    [-1, -1, -3, -1, -5, -5, -5]
    >>> 
    >>> @impose_as([(0,1),(3,1),(4,5),(5,6),(5,7)], 10)
    ... def doit(x):
    ...   return x
    ... 
    >>> doit([9,8,7,6,5,4,3,2,1])
    [9, 19, 7, 9, 5, 15, 25, 25, 1]
    >>> doit([0,1,0,1])
    [0, 10, 0, 0]
    >>> doit([-1,-2,-3,-4,-5,-6])
    [-1, 9, -3, -1, -5, 5]
    >>> doit([-1,-2,-3,-4,-5,-6,-7])
    [-1, 9, -3, -1, -5, 5, 15]
    """
    import copy
    if offset is None: offset = 0
    def dec(f):
        def func(x, *args, **kwds):
            x = copy.copy(x) #XXX: inefficient
            pairs = connected(mask)
            pairs = getattr(pairs, 'iteritems', pairs.items)()
            for i,j in pairs:
                for k in j:
                    try: x[k] = x[i]
                    except IndexError: pass
            pairs = list(mask) #XXX: inefficient
            while pairs: # deal with the offset
                indx,trac = zip(*pairs)
                trac = set(trac)
                for i in trac:
                    try: x[i] += offset
                    except IndexError: pass
                indx = trac.intersection(indx)
                pairs = [m for m in pairs if m[0] in indx]
            return f(x, *args, **kwds)
        func.__wrapped__ = f   #XXX: getattr(f, '__wrapped__', f) ?
        func.__doc__ = f.__doc__
        func.mask = mask
        return func
    return dec


def impose_at(index, target=0.0):
    """generate a function, where some input is set to the target

index should be a set of indices to be fixed at the target. The target
can either be a single value (e.g. float), or a list of values.

For example,
    >>> @impose_at([1,3,4,5,7], -99)
    ... def same(x):
    ...   return x
    ... 
    >>> same([1,1,1,1,1,1,1])
    [1, -99, 1, -99, -99, -99, 1]
    >>> same([1,1,1,1])
    [1, -99, 1, -99]
    >>> same([1,1])
    [1, -99]
    >>> 
    >>> @impose_at([1,3,4,5,7], [0,2,4,6])
    ... def doit(x):
    ...   return x
    ... 
    >>> doit([1,1,1,1,1,1,1])
    [1, 0, 1, 2, 4, 6, 1]
    >>> doit([1,1,1,1])
    [1, 0, 1, 2]
    >>> doit([1,1])
    [1, 0]
    """
    def dec(f):
        def func(x, *args, **kwds):
            xtype = type(x)
            x = asarray(list(x)) #XXX: faster to use array(x, copy=True) ?
            x[[i for i in index if i < len(x)]] = target
            if not type(x) is xtype: x = xtype(x) #XXX: xtype(x.tolist()) ?
            return f(x, *args, **kwds)
        func.__wrapped__ = f   #XXX: getattr(f, '__wrapped__', f) ?
        func.__doc__ = f.__doc__
        func.target = target
        return func
    return dec


#NOTE: the above is intended to be used thusly... (e.g. produce constraints)
"""
>>> term = Or(ChangeOverGeneration(),CollapseAt(0.0),CollapseAs())
>>> collapses = collapsed(term(solver, True))
>>> _fixed = collapses.get(term[1].__doc__)
>>> _pairs = collapses.get(term[2].__doc__)
>>> 
>>> @impose_as(_pairs)
... @impose_at(_fixed, 0.0)
... def constrain(x):
...   return x
... 
>>> 
"""
#NOTE: and for product_measures... (less decorators the better/faster)
"""
>>> stop = Or(ChangeOverGeneration(),CollapsePosition(),CollapseWeight())
>>> collapses = collapsed(stop(solver, True))
>>> _pos = collapses.get(stop[1].__doc__)
>>> _wts = collapses.get(stop[2].__doc__)
>>>
>>> @impose_measure(npts, _pos, _wts)
... def constrain(x):
...   return x
...
>>> 
"""


from mystic.math.discrete import product_measure
#XXX: split to two decorators?
def impose_measure(npts, tracking={}, noweight={}):
    """generate a function, that constrains measure positions and weights

npts is a tuple of the product_measure dimensionality

tracking is a dict of collapses, or a tuple of dicts of collapses.
a tracking collapse is a dict of {measure: {pairs of indices}}, where the
pairs of indices are where the positions will be constrained to have the
same value, and the weight from the second index in the pair will be removed
and added to the weight of the first index

noweight is a dict of collapses, or a tuple of dicts of collapses.
a noweight collapse is a dict of {measure: {indices}), where the
indices are where the measure will be constrained to have zero weight

For example,
    >>> pos = {0: {(0,1)}, 1:{(0,2)}}
    >>> wts = {0: {1}, 1: {1, 2}}
    >>> npts = (3,3)
    >>> 
    >>> @impose_measure(npts, pos)
    ... def same(x):
    ...   return x
    ... 
    >>> same([.5, 0., .5, 2., 4., 6., .25, .5, .25, 6., 4., 2.])
    [0.5, 0.0, 0.5, 2.0, 2.0, 6.0, 0.5, 0.5, 0.0, 5.0, 3.0, 5.0]
    >>> same([1./3, 1./3, 1./3, 1., 2., 3., 1./3, 1./3, 1./3, 1., 2., 3.])
    [0.6666666666666666, 0.0, 0.3333333333333333, 1.3333333333333335, 1.3333333333333335, 3.3333333333333335, 0.6666666666666666, 0.3333333333333333, 0.0, 1.6666666666666667, 2.666666666666667, 1.6666666666666667]
    >>> 
    >>> @impose_measure(npts, {}, wts)
    ... def doit(x):
    ...   return x
    ... 
    >>> doit([.5, 0., .5, 2., 4., 6., .25, .5, .25, 6., 4., 2.])
    [0.5, 0.0, 0.5, 2.0, 4.0, 6.0, 1.0, 0.0, 0.0, 4.0, 2.0, 0.0]
    >>> doit([1./3, 1./3, 1./3, 1., 2., 3., 1./3, 1./3, 1./3, 1., 2., 3.])
    [0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 2.0, 3.0, 4.0]
    >>> 
    >>> @impose_measure(npts, pos, wts)
    ... def both(x):
    ...   return x
    ... 
    >>> both([1./3, 1./3, 1./3, 1., 2., 3., 1./3, 1./3, 1./3, 1., 2., 3.])
    [0.66666666666666663, 0.0, 0.33333333333333331, 1.3333333333333335, 1.3333333333333335, 3.3333333333333335, 1.0, 0.0, 0.0, 2.0, 3.0, 2.0]
    >>> 
    """
    # takes a dict of collapses, or a tuple of dicts of collapses
    if type(tracking) is dict: tracking = (tracking,)
    if type(noweight) is dict: noweight = (noweight,)
    def dec(f):
        def func(x, *args, **kwds):
            # populate a product measure with params
            c = product_measure()
            c.load(x, npts)
            # apply all collapses
            for clps in tracking:
                for k,v in getattr(clps, 'iteritems', clps.items)():
                    c[k].positions, c[k].weights = \
                      impose_collapse(v, c[k].positions, c[k].weights)
            for clps in noweight:
                for k,v in getattr(clps, 'iteritems', clps.items)():
                    c[k].positions, c[k].weights = \
                      impose_unweighted(v, c[k].positions, c[k].weights, False)
            # convert to params and apply function
            return f(c.flatten(), *args, **kwds)
        func.__wrapped__ = f   #XXX: getattr(f, '__wrapped__', f) ?
        func.__doc__ = f.__doc__
        func.npts = npts
        return func
    return dec


def impose_position(npts, tracking):
    """generate a function, that constrains measure positions

npts is a tuple of the product_measure dimensionality

tracking is a dict of collapses, or a tuple of dicts of collapses.
a tracking collapse is a dict of {measure: {pairs of indices}}, where the
pairs of indices are where the positions will be constrained to have the
same value, and the weight from the second index in the pair will be removed
and added to the weight of the first index

For example,
    >>> pos = {0: {(0,1)}, 1:{(0,2)}}
    >>> npts = (3,3)
    >>> 
    >>> @impose_position(npts, pos)
    ... def same(x):
    ...   return x
    ... 
    >>> same([.5, 0., .5, 2., 4., 6., .25, .5, .25, 6., 4., 2.])
    [0.5, 0.0, 0.5, 2.0, 2.0, 6.0, 0.5, 0.5, 0.0, 5.0, 3.0, 5.0]
    >>> same([1./3, 1./3, 1./3, 1., 2., 3., 1./3, 1./3, 1./3, 1., 2., 3.])
    [0.6666666666666666, 0.0, 0.3333333333333333, 1.3333333333333335, 1.3333333333333335, 3.3333333333333335, 0.6666666666666666, 0.3333333333333333, 0.0, 1.6666666666666667, 2.666666666666667, 1.6666666666666667]
    >>> 
    """
    return impose_measure(npts, tracking, {})


def impose_weight(npts, noweight):
    """generate a function, that constrains measure weights

npts is a tuple of the product_measure dimensionality

noweight is a dict of collapses, or a tuple of dicts of collapses.
a noweight collapse is a dict of {measure: {indices}), where the
indices are where the measure will be constrained to have zero weight

For example,
    >>> wts = {0: {1}, 1: {1, 2}}
    >>> npts = (3,3)
    >>> 
    >>> @impose_weight(npts, wts)
    ... def doit(x):
    ...   return x
    ... 
    >>> doit([.5, 0., .5, 2., 4., 6., .25, .5, .25, 6., 4., 2.])
    [0.5, 0.0, 0.5, 2.0, 4.0, 6.0, 1.0, 0.0, 0.0, 4.0, 2.0, 0.0]
    >>> doit([1./3, 1./3, 1./3, 1., 2., 3., 1./3, 1./3, 1./3, 1., 2., 3.])
    [0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 2.0, 3.0, 4.0]
    >>> 
    """
    return impose_measure(npts, {}, noweight)


# EOF

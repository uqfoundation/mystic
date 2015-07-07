#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Copyright (c) 1997-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
"""Tools for building and applying constraints and penalties.
"""

__all__ = ['with_penalty','with_constraint','as_penalty','as_constraint',
           'with_mean','with_variance','with_std','with_spread','normalized',
           'issolution','solve','discrete','integers']

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
    >>> print x
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

NOTE: The default solver is 'diffev', with npop=min(40, ndim*5). The default
    termination is ChangeOverGeneration(), and the default guess is randomly
    selected points between the upper and lower bounds.
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
    if guess is not None:
        solver.SetInitialPoints(guess) #XXX: nice if 'diffev' also had methods
    else:
        solver.SetRandomInitialPoints(lower_bounds, upper_bounds)
    if lower_bounds or upper_bounds:
        solver.SetStrictRanges(lower_bounds, upper_bounds)
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
    solver -- the mystic solver to use in the optimization
    termination -- the mystic termination to use in the optimization

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

    return penalty


from numpy import asfarray, asarray, choose, zeros, ones, ndarray
from numpy import vectorize, shape, broadcast, empty, atleast_1d
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
        result = tuple(i for i in asarray(map(_argnear, x)).T)
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
        result = asarray(map(_near, x, l, h))
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
  - if index tuple provided, only round at the given indicies

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
        ok = min(unique) >= full['min'] and max(unique) < full['max']
        if not ok:
            oops = list(unique - set(range(full['min'],full['max'])))
            msg = "x=%s not in %s <= x < %s" % (oops[-1],full['min'],full['max'])
        _min = full['min']
        _max = full['max']
        full = float
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
    """ensure all values are unique and found in the given set"""
    def dec(f):
        def func(x,*args,**kwds):
            return f(unique(x, seq),*args,**kwds)
        return func
    return dec


#XXX: enable near_integers and has_unique on selected members of x?

from numpy import round, abs
def near_integers(x): # use as a penalty for int programming
    """the sum of all deviations from int values"""
    return abs(x - round(x)).sum()

def has_unique(x): # use as a penalty for unique numbers
    """check for uniqueness of the members of x"""
    return sum(x.count(xi) for xi in x)
   #return len(x) - len(set(x))



# EOF

# EOF

#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
#
# flatten was adapted from the python cookbook
# wrap_function was adapted from numpy
# wrap_bounds was adapted from park

"""
Various python tools

Main functions exported are:: 
    - isiterable: check if an object is iterable
    - itertype: get the 'underlying' type used to construct x
    - multiply: recursive elementwise casting multiply of x by n
    - divide: recursive elementwise casting divide of x by n
    - flatten: flatten a sequence
    - flatten_array: flatten an array 
    - getch: provides "press any key to quit"
    - random_seed: sets the seed for calls to 'random()'
    - random_state: build a localized random generator
    - wrap_nested: nest a function call within a function object
    - wrap_penalty: append a function call to a function object
    - wrap_function: bind an EvaluationMonitor and an evaluation counter
        to a function object
    - wrap_bounds: impose bounds on a function object
    - wrap_reducer: convert a reducer function to an arraylike interface
    - reduced: apply a reducer function to reduce output to a single value
    - masked: generate a masked function, given a function and mask provided
    - partial: generate a function where some input has fixed values
    - insert_missing: return a sequence with the 'missing' elements inserted
    - unpair: convert a 1D array of N pairs to two 1D arrays of N values
    - src: extract source code from a python code object

Other tools of interest are in::
    `mystic.mystic.filters` and `mystic.models.poly`
"""

def isiterable(x):
    """check if an object is iterable"""
   #try:
   #    from collections import Iterable
   #    return isinstance(x, Iterable)
   #except ImportError:
    try:
        iter(x)
        return True
    except TypeError: return False
   #return hasattr(x, '__len__') or hasattr(x, '__iter__')

def itertype(x, default=tuple):
    """get the 'underlying' type used to construct x"""
    _type = type(x)
    try:
        if _type((0,)): return _type # non-iterator iterables
    except TypeError: pass
    try:
        if _type(1): return _type # non-iterables
    except TypeError: pass
    # iterators, etc
    _type = type(x).__name__.split('iterator')[0]
    types = ('xrange','generator','') # and ...?
    if _type in types: return default
    try:
        return eval(_type)
    except NameError: return default

def _kdiv(num, denom, type=None):
    """'special' scalar division for 'k'"""
    if num is denom is None: return None
    if denom is None: denom = 1
    if num is None: num = 1
    if type is not None: num = type(num)
    return num/denom

def multiply(x, n, type=list, recurse=False): # list, iter, numpy.array, ...
    """multiply: recursive elementwise casting multiply of x by n"""
    # short-circuit cases for speed
    if n is None: return x
    try:   # for scalars and vectors (numpy.arrays)
        -x # x*n might not throw an error
        return x*n
    except TypeError: pass
    if n is 1: return type(x)
    if type.__name__ == 'array': return type(x)*n
    # multiply by n != 1 for iterables (iter and non-iter)
    if recurse:
        return type(multiply(i,n,type) for i in x)
    return type(i*n for i in x)

def divide(x, n, type=list, recurse=False): # list, iter, numpy.array, ...
    """elementwise division of x by n, returning the selected type"""
    # short-circuit cases for speed
    if n is None: return x
    try:   # for scalars and vectors (numpy.arrays)
        return x/n
    except TypeError: pass
    if n is 1: return type(x)
    if type.__name__ == 'array': return type(x)/n
    # divide by n != 1 for iterables (iter and non-iter)
    if recurse:
        return type(divide(i,n,type) for i in x)
    return type(i/n for i in x)

def _multiply(x, n):
    """elementwise multiplication of x by n, as if x were an array"""
    # short-circuit cases for speed
    if n is None: return x
    try:   # for scalars and vectors (numpy.arrays)
        -x # x*n might not throw an error
        return x*n
    except TypeError: pass
    if n is 1: return itertype(x)(x)
    # multiply by n != 1 for iterables (iter and non-iter)
    xn = (_multiply(i,n) for i in x)
    # return astype used to construct x, if possible
    return itertype(x)(xn)

def _divide(x, n):
    """elementwise division of x by n, as if x were an array"""
    # short-circuit cases for speed
    if n is None: return x
    try:   # for scalars and vectors (numpy.arrays)
        return x/n
    except TypeError: pass
    if n is 1: return itertype(x)(x)
    # divide by n != 1 for iterables (iter and non-iter)
    xn = (_divide(i,n) for i in x)
    # return astype used to construct x, if possible
    return itertype(x)(xn)

def _imultiply(x, n):
    """iterator for elementwise 'array-like' multiplication of x by n"""
    # short-circuit cases for speed
    if n is None: return x
    try:   # for scalars and vectors (numpy.arrays)
        -x # x*n might not throw an error
        return iter(x*n)
    except TypeError: pass
    if n is 1: return iter(x)
    # multiply by n != 1 for iterables (iter and non-iter)
    return (_multiply(i,n) for i in x)

def _idivide(x, n):
    """iterator for elementwise 'array-like' division of x by n"""
    # short-circuit cases for speed
    if n is None: return x
    try:   # for scalars and vectors (numpy.arrays)
        return iter(x/n)
    except TypeError: pass
    if n is 1: return iter(x)
    # divide by n != 1 for iterables (iter and non-iter)
    return (_divide(i,n) for i in x)

def _amultiply(x, n):
    """elementwise 'array-casting' multiplication of x by n"""
    # short-circuit cases for speed
    if n is None: return x
    # convert to numpy array
    import numpy
    x = numpy.asarray(x)
    if n is 1: return x
    return x*n

def _adivide(x, n):
    """elementwise 'array-casting' division of x by n"""
    # short-circuit cases for speed
    if n is None: return x
    # convert to numpy array
    import numpy
    x = numpy.asarray(x)
    if n is 1: return x
    return x/n

def list_or_tuple(x): # set, ...?
    "True if x is a list or a tuple"
    return isinstance(x, (list, tuple))

def list_or_tuple_or_ndarray(x): # set, ...?
    "True if x is a list, tuple, or a ndarray"
    import numpy
    return isinstance(x, (list, tuple, numpy.ndarray))

def listify(x):
    "recursivly convert all members of a sequence to a list"
    if not isiterable(x): return x
    if x is iter(x): return listify(list(x))
    try: # e.g. if array(1)
        if x.ndim == 0: return x.flatten()[0]
    except Exception: pass
    return [listify(i) for i in x]

def flatten_array(sequence, maxlev=999, lev=0):
    "flatten a sequence; returns a ndarray"
    import numpy
    return numpy.array(list(flatten(sequence, maxlev, list_or_tuple_or_ndarray, lev)))

def flatten(sequence, maxlev=999, to_expand=list_or_tuple, lev=0):
    """flatten a sequence; returns original sequence type

example usage...
    >>> A = [1,2,3,[4,5,6],7,[8,[9]]]
    >>> 
    >>> # Flatten.
    >>> flatten(A)
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> 
    >>> # Flatten only one level deep.
    >>> flatten(A,1)
    [1, 2, 3, 4, 5, 6, 7, 8, [9]]
    >>> 
    >>> # Flatten twice. 
    >>> flatten(A,2)
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> 
    >>> # Flatten zero levels deep (i.e. don't flatten).
    >>> flatten(A,0)
    [1, 2, 3, [4, 5, 6], 7, [8, [9]]]

    """
    for item in sequence:
        if lev < maxlev and to_expand(item):
            for subitem in flatten(item, maxlev, to_expand, lev+1):
                 yield subitem
        else:
            yield item

def getch(str="Press any key to continue"):
    "configurable pause of execution"
    import sys, subprocess
    if sys.stdin.isatty():
       if str is not None:
          print str
       subprocess.call('stty raw', shell=True)
       a = sys.stdin.read(1)
       subprocess.call('stty cooked', shell=True)
       return a
    else:
       if str is not None:
           print str + " and press enter"
       return raw_input()

def random_seed(s=None):
    "sets the seed for calls to 'random()'"
    import random
    random.seed(s)
    try: 
        from numpy import random
        random.seed(s)
    except:
        pass
    return

def random_state(module='random', new=False, seed='!'):
    """return a (optionally manually seeded) random generator

For a given module, return an object that has random number generation (RNG)
methods available.  If new=False, use the global copy of the RNG object.
If seed='!', do not reseed the RNG (using seed=None 'removes' any seeding).
If seed='*', use a seed that depends on the process id (PID); this is useful
for building RNGs that are different across multiple threads or processes.
    """
    import random
    if module == 'random':
        rng = random
    elif not isinstance(module, type(random)):
        # convienence for passing in 'numpy'
        if module == 'numpy': module = 'numpy.random'
        try:
            import importlib
            rng = importlib.import_module(module)
        except ImportError:
            rng = __import__(module, fromlist=module.split('.')[-1:])
    elif module.__name__ == 'numpy': # convienence for passing in numpy
        from numpy import random as rng
    else: rng = module

    _rng = getattr(rng, 'RandomState', None) or \
           getattr(rng, 'Random') # throw error if no rng found
    if new:
        rng = _rng()

    if seed == '!': # special case: don't reset the seed
        return rng
    if seed == '*': # special case: random seeding for multiprocessing
        try:
            try:
                import multiprocessing as mp
            except ImportError:
                import processing as mp
            try:
                seed = mp.current_process().pid
            except AttributeError:
                seed = mp.currentProcess().getPid()
        except:   
            seed = 0
        import time
        seed += int(time.time()*1e6)

    # set the random seed (or 'reset' with None)
    rng.seed(seed)
    return rng

def wrap_nested(outer_function, inner_function):
    """nest a function call within a function object

This is useful for nesting a constraints function in a cost function;
thus, the constraints will be enforced at every cost function evaluation.
    """
    def function_wrapper(x):
        _x = x[:] #XXX: trouble if x not a list or ndarray... maybe "deepcopy"?
        return outer_function(inner_function(_x))
    return function_wrapper

def wrap_penalty(cost_function, penalty_function):
    """append a function call to a function object

This is useful for binding a penalty function to a cost function;
thus, the penalty will be evaluated at every cost function evaluation.
    """
    def function_wrapper(x):
        _x = x[:] #XXX: trouble if x not a list or ndarray... maybe "deepcopy"?
        return cost_function(_x) + penalty_function(_x)
    return function_wrapper

# slight break to backward compatability: renamed 'args' to 'extra_args'
def wrap_function(the_function, extra_args, EvaluationMonitor, scale=1):
    """bind an EvaluationMonitor and evaluation counter to a function object"""
    # scale=-1 intended to seek min(-f) == -max(f)
    ncalls = [0]
    from numpy import array
    def function_wrapper(x):
        ncalls[0] += 1
        fval = the_function(x, *extra_args)
        EvaluationMonitor(x, fval)
        return scale*fval
    return ncalls, function_wrapper

def wrap_bounds(target_function, min=None, max=None):
    "impose bounds on a function object"
    from numpy import asarray, any, inf, seterr
    bounds = True
    if min is not None and max is not None: #has upper & lower bound
        min = asarray(min)
        max = asarray(max)
    elif min is not None: #has lower bound
        min = asarray(min)
        max = asarray([inf for i in min])
    elif max is not None: #has upper bound
        max = asarray(max)
        min = asarray([-inf for i in max])
    else: #not bounded
        bounds = False
    if bounds:
        def function_wrapper(x):
            settings = seterr(all='ignore') #XXX: slow to supress warnings?
            if any((x<min)|(x>max)): #if violate bounds, evaluate as inf
                seterr(**settings)
                return inf
            seterr(**settings)
            return target_function(x)
    else:
        def function_wrapper(x):
            return target_function(x)
    return function_wrapper

def wrap_reducer(reducer_function):
    """convert a reducer function to an arraylike interface

This is useful for converting a function that used python's 'y = reduce(f, x)'
interface to an arraylike interface 'y = f(x)'.  Example usage...
    >>> acum = wrap_reducer(numpy.add)
    >>> acum([1,2,3,4])
    10
    >>> prod = wrap_reducer(lambda x,y: x*y)
    >>> prod([1,2,3,4])
    24
    """
    def _reduce(x): # NOTE: not a decorator
        return reduce(reducer_function, x)
    return _reduce


def reduced(reducer=None, arraylike=False):
    """apply a reducer function to reduce output to a single value

example usage...
    >>> @reduced(lambda x,y: x)
    ... def first(x):
    ...   return x
    ... 
    >>> first([1,2,3])
    1
    >>> 
    >>> @reduced(min)
    ... def minimum(x):
    ...   return x
    ... 
    >>> minimum([3,2,1])
    1
    >>> @reduced(lambda x,y: x+y)
    ... def add(x):
    ...   return x
    ... 
    >>> add([1,2,3])
    6
    >>> @reduced(sum, arraylike=True)
    ... def added(x):
    ...   return x
    ... 
    >>> added([1,2,3])
    6

    """
    if reducer is None:
        reducer = lambda x: x
        arraylike = True
    def dec(f):
        if arraylike:
            def func(*args, **kwds):
                result = f(*args, **kwds)
                iterable = isiterable(result)
                return reducer(result) if iterable else result
        else:
            def func(*args, **kwds):
                result = f(*args, **kwds)
                iterable = isiterable(result)
                return reduce(reducer, result) if iterable else result
        func.__wrapped__ = f
        func.__doc__ = f.__doc__
        return func
    return dec 


def insert_missing(x, missing=None):
    """return a sequence with the 'missing' elements inserted

missing should be a dictionary of positional index and a value (e.g. {0:1.0}),
where keys must be integers, and values can be any object (typically a float).

For example,
    >>> insert_missing([1,2,4], missing={0:10, 3:-1})
    [10, 1, 2, -1, 4]
    """
    import dill
    if missing is None: _mask = {}
    elif isinstance(missing, str): _mask = eval('{%s}' % missing)
    else: _mask = missing
    # raise KeyError if key out of bounds #XXX: also has *any* non-int object
    first = min([0]+_mask.keys())
    if first < 0:
        raise KeyError('invalid argument index: %s' % first)
    last = max([-1]+_mask.keys())
    if last > len(x)+len(_mask)-1:
        raise KeyError('invalid argument index: %s' % last)

    # preserve type(x)
    _locals = {}
    _locals['xtype'] = type(x)
    code = "%s" % dill.source.getimport(x, alias='xtype')
    if "import" in code:
        code = compile(code, '<string>', 'exec')
        exec code in _locals
    xtype = _locals['xtype']

    # find the new indices due to the mask
    _x = list(x)
    for (k,v) in sorted(_mask.items()):
        _x.insert(k,v)
    # get the new sequence
    return xtype(_x)


def masked(mask=None):
    """generate a masked function, given a function and mask provided

mask should be a dictionary of the positional index and a value (e.g. {0:1.0}),
where keys must be integers, and values can be any object (typically a float).

functions are expected to take a single argument, a n-dimensional list or array,
where the mask will be applied to the input array.  Hence, instead of masking
the inputs, the function is "masked".  Conceptually, f(mask(x)) ==> f'(x),
instead of f(mask(x)) ==> f(x').

For example,
    >>> @masked({0:10,3:-1})
    ... def same(x):
    ...     return x
    ...
    >>> same([1,2,3])
    [10, 1, 2, -1, 3]
    >>> 
    >>> @masked({0:10,3:-1})
    ... def foo(x):
            w,x,y,z = x # requires a lenth-4 sequence
    ...     return w+x+y+z
    ...
    >>> foo([-5,2])     # produces [10,-5,2,-1]
    6
    """
    def dec(f):
        def func(x, *args, **kwds):
            return f(insert_missing(x, mask), *args, **kwds)
        func.__wrapped__ = f
        func.__doc__ = f.__doc__
        func.mask = mask
        return func
    return dec


def partial(mask):
    """generate a function, where some input has fixed values

mask should be a dictionary of the positional index and a value (e.g. {0:1.0}),
where keys must be integers, and values can be any object (typically a float).

functions are expected to take a single argument, a n-dimensional list or array,
where the mask will be applied to the input array.

For example,
    >>> @partial({0:10,3:-1})
    ... def same(x):
    ...     return x
    ...
    >>> same([-5,9])
    [10, 9]
    >>> same([0,1,2,3,4])
    [10, 1, 2, -1, 4]
    """
    def dec(f):
        def func(x, *args, **kwds):
            for i,j in mask.items():
                try: x[i] = j
                except IndexError: pass
            return f(x, *args, **kwds)
        func.__wrapped__ = f
        func.__doc__ = f.__doc__
        func.mask = mask
        return func
    return dec


def wrap_cf(CF, REG=None, cfmult=1.0, regmult=0.0):
    "wrap a cost function..."
    def _(*args, **kwargs):
         if REG is not None:
             return cfmult * CF(*args,**kwargs) + regmult * REG(*args,**kwargs)
         else:
             return cfmult * CF(*args,**kwargs)
    return _


def unpair(pairs):
    '''convert a 1D array of N pairs to two 1D arrays of N values

example usage...
    >>> unpair([(a0,b0),(a1,b1),(a2,b2)])
    [a0,a1,a2],[b0,b1,b2]
    '''
    from numpy import asarray
    pairsT = asarray(pairs).transpose()
    return [i.tolist() for i in pairsT]


try:
    from itertools import permutations
except ImportError:
    def permutations(iterable, r=None):
        """return successive r-length permutations of elements in the iterable.
Produces a generator object.

For example, 
    >>> print list( permutations(range(3),2) ) 
    [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
    >>> print list( permutations(range(3)) )
    [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
        """
        # code from http://docs.python.org/library/itertools.html
        pool = tuple(iterable)
        n = len(pool)
        r = n if r is None else r
        if r > n:
            return
        indices = range(n)
        cycles = range(n, n-r, -1)
        yield tuple(pool[i] for i in indices[:r])
        while n:
            for i in reversed(range(r)):
                cycles[i] -= 1
                if cycles[i] == 0:
                    indices[i:] = indices[i+1:] + indices[i:i+1]
                    cycles[i] = n - i
                else:
                    j = cycles[i]
                    indices[i], indices[-j] = indices[-j], indices[i]
                    yield tuple(pool[i] for i in indices[:r])
                    break
            else:
                return
        return


# backward compatibility
from dill.source import getblocks as parse_from_history
from dill.source import getsource as src
from monitors import Monitor as Sow
from monitors import VerboseMonitor as VerboseSow
from monitors import LoggingMonitor as LoggingSow
from monitors import CustomMonitor as CustomSow
from monitors import Null

def isNull(mon):
    if isinstance(mon, Null): # is Null()
        return True
    if mon == Null:  # is Null
        return True
    return False


if __name__=='__main__':
    import doctest
    doctest.testmod(verbose=True)

# End of file

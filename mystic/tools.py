#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
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
    - factor: generator for factors of a number
    - flatten: flatten a sequence
    - flatten_array: flatten an array 
    - getch: provides "press any key to quit"
    - random_seed: sets the seed for calls to 'random()'
    - random_state: build a localized random generator
    - wrap_nested: nest a function call within a function object
    - wrap_penalty: append a function call to a function object
    - wrap_function: bind an eval_monitor and an evaluation counter
        to a function object
    - wrap_bounds: impose bounds on a function object
    - wrap_reducer: convert a reducer function to an arraylike interface
    - reduced: apply a reducer function to reduce output to a single value
    - masked: generate a masked function, given a function and mask provided
    - partial: generate a function where some input has fixed values
    - synchronized: generate a function, where some input tracks another input
    - insert_missing: return a sequence with the 'missing' elements inserted
    - clipped: generate a function where values outside of bounds are clipped
    - suppressed: generate a function where values less than tol are suppressed
    - suppress: suppress small values less than tol
    - chain: chain together decorators into a single decorator
    - connected: generate dict of connected members of a set of tuples (pairs)
    - unpair: convert a 1D array of N pairs to two 1D arrays of N values
    - pairwise: convert an array of positions to an array of pairwise distances
    - measure_indices: get the indices corresponding to weights and to positions
    - select_params: get params for the given indices as a tuple of index,values
    - solver_bounds: return a dict of tightest bounds defined for the solver
    - interval_overlap: find the intersection of intervals in the given bounds
    - indicator_overlap: find the intersection of dicts of sets of indices
    - no_mask: build dict of termination info {with mask: without mask}
    - unmasked_collapse: apply the embedded mask to the given collapse
    - masked_collapse: extract the mask, and combine with the given collapse
    - src: extract source code from a python code object

Other tools of interest are in::
    `mystic.mystic.filters` and `mystic.models.poly`
"""
from functools import reduce
import collections
try:
    import collections.abc
except ImportError:
    pass
_Callable = getattr(collections, 'Callable', None) or getattr(collections.abc, 'Callable')

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
    if n == 1: return type(x)
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
    if n == 1: return type(x)
    if type.__name__ == 'array': return type(x)/n
    # divide by n != 1 for iterables (iter and non-iter)
    if recurse:
        return type(divide(i,n,type) for i in x)
    return type(i/n for i in x)

def _cmultiply(x, n, type=list):
    """elementwise casting multiplication of x by n, as if x were an array"""
    if type is list:
        return _multiply(x, n)
    if type is iter:
        return _imultiply(x, n)
    if type.__name__ == 'array': 
        return _amultiply(x, n)
    return type(_multiply(x, n))

def _cdivide(x, n, type=list):
    """elementwise casting division of x by n, as if x were an array"""
    if type is list:
        return _divide(x, n)
    if type is iter:
        return _idivide(x, n)
    if type.__name__ == 'array': 
        return _adivide(x, n)
    return type(_divide(x, n))

def _multiply(x, n):
    """elementwise multiplication of x by n, as if x were an array"""
    # short-circuit cases for speed
    if n is None: return x
    try:   # for scalars and vectors (numpy.arrays)
        -x # x*n might not throw an error
        return x*n
    except TypeError: pass
    if n == 1: return itertype(x)(x)
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
    if n == 1: return itertype(x)(x)
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
    if n == 1: return iter(x)
    # multiply by n != 1 for iterables (iter and non-iter)
    return (_multiply(i,n) for i in x)

def _idivide(x, n):
    """iterator for elementwise 'array-like' division of x by n"""
    # short-circuit cases for speed
    if n is None: return x
    try:   # for scalars and vectors (numpy.arrays)
        return iter(x/n)
    except TypeError: pass
    if n == 1: return iter(x)
    # divide by n != 1 for iterables (iter and non-iter)
    return (_divide(i,n) for i in x)

def _amultiply(x, n):
    """elementwise 'array-casting' multiplication of x by n"""
    # short-circuit cases for speed
    if n is None: return x
    # convert to numpy array
    import numpy
    x = numpy.asarray(x)
    if n == 1: return x
    return x*n

def _adivide(x, n):
    """elementwise 'array-casting' division of x by n"""
    # short-circuit cases for speed
    if n is None: return x
    # convert to numpy array
    import numpy
    x = numpy.asarray(x)
    if n == 1: return x
    return x/n

def factor(n):
    "generator for factors of a number"
    #yield 1
    i = 2
    limit = n**0.5
    while i <= limit:
        if n % i == 0:
            yield i
            n = n / i
            limit = n**0.5
        else:
            i += 1
    if n > 1:
        yield n

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

For example:
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
          print(str)
       if sys.platform[:3] != 'win':
          raw,cooked = 'stty raw','stty cooked'
       else:
          raw,cooked = '',''
       subprocess.call(raw, shell=True)
       a = sys.stdin.read(1)
       subprocess.call(cooked, shell=True)
       return a
    else:
       if str is not None:
           print(str + " and press enter")
       return input()

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
        seed += int(time.time()) #NOTE: don't *1e6, numpy max is 2**32-1

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

# slight break to backward compatibility: renamed 'args' to 'extra_args'
def wrap_function(the_function, extra_args, eval_monitor, scale=1, start=0):
    """bind an eval_monitor and evaluation counter to a function object"""
    # scale=-1 intended to seek min(-f) == -max(f) #XXX: default extra_args=()?
    ncalls = [start] # [0] or [len(eval_monitor)]
    from numpy import array
    def function_wrapper(x):
        ncalls[0] += 1
        fval = the_function(x, *extra_args)
        eval_monitor(x, fval)
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
            settings = seterr(all='ignore') #XXX: slow to suppress warnings?
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
    """convert a reducer function to have an arraylike interface

Args:
    reducer_function (func): a function ``f`` of the form ``y = reduce(f, x)``.

Returns:
    a function ``f`` of the form ``y = f(x)``, where ``x`` is array-like.

Examples:
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

For example:
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

For example:
    >>> insert_missing([1,2,4], missing={0:10, 3:-1})
    [10, 1, 2, -1, 4]
    """
    import dill
    if missing is None: _mask = {}
    elif isinstance(missing, str): _mask = eval('{%s}' % missing)
    else: _mask = missing
    # raise KeyError if key out of bounds #XXX: also has *any* non-int object
    first = min([0]+list(_mask.keys()))
    if first < 0:
        raise KeyError('invalid argument index: %s' % first)
    last = max([-1]+list(_mask.keys()))
    if last > len(x)+len(_mask)-1:
        raise KeyError('invalid argument index: %s' % last)

    # preserve type(x)
    _locals = {}
    _locals['xtype'] = type(x)
    code = "%s" % dill.source.getimport(x, alias='xtype')
    if "import" in code:
        code = compile(code, '<string>', 'exec')
        exec(code, _locals)
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

For example:
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

For example:
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
            for i,j in getattr(mask, 'iteritems', mask.items)():
                try: x[i] = j
                except IndexError: pass
            return f(x, *args, **kwds)
        func.__wrapped__ = f
        func.__doc__ = f.__doc__
        func.mask = mask
        return func
    return dec


def synchronized(mask):
    """generate a function, where some input tracks another input

mask should be a dictionary of positional index and tracked index (e.g. {0:1}),
where keys and values should be different integers. However, if a tuple is
provided instead of the tracked index (e.g. {0:(1,lambda x:2*x)} or {0:(1,2)}),
the second member of the tuple will be used to scale the tracked index.

functions are expected to take a single argument, a n-dimensional list or array,
where the mask will be applied to the input array.

operations within a single mask are unordered. If a specific ordering of
operations is required, apply multiple masks in the desired order.

For example:
    >>> @synchronized({0:1,3:-1})
    ... def same(x):
    ...     return x
    ...
    >>> same([-5,9])
    [9, 9]
    >>> same([0,1,2,3,4])
    [1, 1, 2, 4, 4]
    >>> same([0,9,2,3,6])
    [9, 9, 2, 6, 6]
    >>> 
    >>> @synchronized({0:(1,lambda x:1/x),3:(1,-1)})
    ... def doit(x):
    ...   return x
    ... 
    >>> doit([-5.,9.])
    [0.1111111111111111, 9.0]
    >>> doit([0.,1.,2.,3.,4.])
    [1.0, 1.0, 2.0, -1.0, 4.0]
    >>> doit([0.,9.,2.,3.,6.])
    [0.1111111111111111, 9.0, 2.0, -9.0, 6.0]
    >>>
    >>> @synchronized({1:2})
    ... @synchronized({0:1})
    ... def invert(x):
    ...   return [-i for i in x]
    ... 
    >>> invert([0,1,2,3,4])
    [-2, -2, -2, -3, -4]
    """
    def dec(f):
        def func(x, *args, **kwds):
            for i,j in getattr(mask, 'iteritems', mask.items)():
                try: x[i] = x[j]
                except TypeError: # value is tuple with f(x) or constant
                  j0,j1 = (j[:2] + (1,))[:2]
                  try: x[i] = j1(x[j0]) if isinstance(j1, _Callable) else j1*x[j0]
                  except IndexError: pass
                except IndexError: pass
            return f(x, *args, **kwds)
        func.__wrapped__ = f   #XXX: getattr(f, '__wrapped__', f) ?
        func.__doc__ = f.__doc__
        func.mask = mask
        return func
    return dec


def suppress(x, tol=1e-8, clip=True):
    """suppress small values less than tol"""
    from numpy import asarray, abs
    x = asarray(list(x))
    mask = abs(x) < tol
    if not clip:
        # preserve sum by spreading suppressed values to the non-zero elements
        x[mask==False] = (x + sum(x[mask])/(len(mask)-sum(mask)))[mask==False]
    x[mask] = 0.0
    return x.tolist()


def suppressed(tol=1e-8, exit=False, clip=True):
    """generate a function, where values less than tol are suppressed

For example:
    >>> @suppressed(1e-8)
    ... def square(x):
    ...     return [i**2 for i in x]
    ... 
    >>> square([1e-8, 2e-8, 1e-9])
    [1.00000000e-16, 4.00000000e-16, 0.00000000e+00]
    >>> 
    >>> from mystic.math.measures import normalize
    >>> @suppressed(1e-8, exit=True, clip=False)
    ... def norm(x):
    ...     return normalize(x, mass=1)
    ... 
    >>> norm([1e-8, 2e-8, 1e-16, 5e-9])
    [0.28571428585034014, 0.5714285707482993, 0.0, 0.14285714340136055]
    >>> sum(_)
    1.0
    """
    def dec(f):
        if exit:
            def func(x, *args, **kwds):
                return suppress(f(x, *args, **kwds), tol, clip)
        else:
            def func(x, *args, **kwds):
                return f(suppress(x, tol, clip), *args, **kwds)
        func.__wrapped__ = f
        func.__doc__ = f.__doc__
        return func
    return dec


def clipped(min=None, max=None, exit=False):
    """generate a function, where values outside of bounds are clipped
    """
    from numpy import clip
    def dec(f):
        if exit:
            def func(x, *args, **kwds):
                return clip(f(x, *args, **kwds), min, max).tolist()
        else:
            def func(x, *args, **kwds):
                return f(clip(x, min, max).tolist(), *args, **kwds)
        func.__wrapped__ = f
        func.__doc__ = f.__doc__
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


def chain(*decorators):
    """chain together decorators into a single decorator

For example:
    >>> wm = with_mean(5.0)
    >>> wv = with_variance(5.0)
    >>> 
    >>> @chain(wm, wv)  
    ... def doit(x):
    ...     return x
    ... 
    >>> res = doit([1,2,3,4,5])
    >>> mean(res), variance(res)
    (5.0, 5.0000000000000018)
"""
    def dec(f):
        for _dec in reversed(decorators):
            f = _dec(f)
        return f
    return dec


def connected(pairs):
    """generate dict of connected members of a set of tuples (pairs)

For example:
    >>> connected({(0,3),(4,2),(3,1),(4,5),(2,6)})
    {0: set([1, 3]), 4: set([2, 5, 6])}
    >>> connected({(0,3),(3,1),(4,5),(2,6)})
    {0: set([1, 3]), 2: set([6]), 4: set([5])}}
"""
    collapse = {}
    #XXX: any vectorized way to do this?
    for i,j in pairs: #XXX: sorted(sorted(pair) for pair in pairs): # ordering?
        found = False
        for k,v in getattr(collapse, 'iteritems', collapse.items)():
            if i in (k,) or i in v:
                v.add(j); found = True; break
            if j in (k,) or j in v:
                v.add(i); found = True; break
        if not found:
            collapse[i] = set((j,))
    return collapse


def unpair(pairs):
    '''convert a 1D array of N pairs to two 1D arrays of N values

For example:
    >>> unpair([(a0,b0),(a1,b1),(a2,b2)])
    [a0,a1,a2],[b0,b1,b2]
    '''
    from numpy import asarray
    pairsT = asarray(pairs).transpose()
    return [i.tolist() for i in pairsT]


def pairwise(x, indices=False):
    '''convert an array of positions to an array of pairwise distances

    if indices=True, also return indices to relate input and output arrays'''
    import numpy as np
    x = np.asarray(x)
    shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    idx = np.triu_indices(x.shape[-1],k=1)  # get upper triangle indices
    z = np.zeros(x.shape[:-1] + (idx[0].shape[0],))
    for i in range(z.shape[-2]):
        z[i] = np.subtract.outer(x[i],x[i])[idx]
    z.shape = shape[:-1]+(z.shape[-1],)
    return abs(z),list(zip(*idx)) if indices else abs(z)  #XXX: abs(z) or z?


def _inverted(pairs): # assumes pairs is a list of tuples
    '''return a list of tuples, where each tuple has been reversed'''
    # >>> _inverted([(1,2),(3,4),(5,6)])
    # [(2, 1), (4, 3), (6, 5)]
    return list(map(tuple, map(reversed, pairs)))


def _symmetric(pairs): # assumes pairs is a set of tuples
    '''returns a set of tuples, where each tuple includes it's inverse'''
    # >>> _symmetric([(1,2),(3,4),(5,6)])
    # set([(1, 2), (5, 6), (2, 1), (4, 3), (3, 4), (6, 5)])
    return set(list(pairs) + _inverted(pairs))


try:
    from itertools import permutations
except ImportError:
    def permutations(iterable, r=None):
        """return successive r-length permutations of elements in the iterable.
Produces a generator object.

For example: 
    >>> print(list( permutations(range(3),2) ))
    [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
    >>> print(list( permutations(range(3)) ))
    [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
        """
        # code from http://docs.python.org/library/itertools.html
        pool = tuple(iterable)
        n = len(pool)
        r = n if r is None else r
        if r > n:
            return
        indices = list(range(n))
        cycles = list(range(n, n-r, -1))
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


def measure_indices(npts):
    '''get the indices corresponding to weights and to positions'''
    wts, pos = [], []
    for (i,n) in enumerate(npts):
        indx = 2*reduce(lambda x,y:x+y, (0,)+npts[:i])
        wts.extend(range(indx,n+indx))
        pos.extend(range(indx+npts[0],n+indx+npts[0]))
    return wts, pos


def select_params(params, index):
    """get params for the given indices as a tuple of index,values"""
    if isinstance(index, int): index = (index,)
    try: # was passed a solver instance
        params = params.bestSolution
    except AttributeError:
        try: # was passed a monitor instance
            if type(params).__module__ == 'mystic.monitors':
                params = params._x[-1]
        except: pass
    import itertools
    # returns (tuple(index), tuple(params[index]))
    return tuple(getattr(itertools, 'izip', zip)(*((i,params[i]) for i in index)))


def solver_bounds(solver):
    """return a dict {index:bounds} of tightest bounds defined for the solver"""
    return dict(enumerate(zip((solver._strictMin or solver._defaultMin),(solver._strictMax or solver._defaultMax))))


def _interval_invert(bounds, lb=None, ub=None):
    """invert the given intervals

    bounds is a list of tuples [(lo,hi),...],
    where lb and ub, if not given, are the extrema of bounds
    """
    if not len(bounds): return bounds
    import numpy
    bounds = numpy.asarray(bounds)
    _a,a_ = bounds.min(),bounds.max()
    lb = _a if lb is None else lb
    ub = a_ if ub is None else ub
    return [tuple(i) for i in numpy.array([lb]+bounds.ravel().tolist()+[ub])[2*(lb == _a):-2*(ub == a_) or None].reshape(-1,2)]


def _interval_intersection(bounds1, bounds2):
    """find the intersection of the intervals in the given bounds

    bounds is a list of tuples [(lo,hi),...]
    """
    if not len(bounds2):
        return bounds1 if len(bounds1) else []
    if not len(bounds1): return bounds2
    results = []
    for lb,ub in bounds1: #XXX: is there a better way?
        for lo,hi in bounds2:
            r = l,h = max(lb,lo),min(ub,hi)
            if l < h: #XXX: what about when l == h?
                results.append(r)
    return results


def _interval_union(bounds1, bounds2):
    """find the union of the intervals in the given bounds

    bounds is a list of tuples [(lo,hi),...]
    """
    if not len(bounds2) or not len(bounds1): return []
    import numpy
    bounds1,bounds2 = numpy.asarray(bounds1),numpy.asarray(bounds2)
    _a,a_ = bounds1.min(),bounds1.max()
    _b,b_ = bounds2.min(),bounds2.max()
    lb,ub = min(_a,_b),max(a_,b_)
    return _interval_invert(_interval_intersection(_interval_invert(bounds1,lb,ub),_interval_invert(bounds2,lb,ub)),lb,ub)


#XXX: generalize to *bounds?
#FIXME: what about keys of None?
def interval_overlap(bounds1, bounds2, union=False):
    """find the intersection of intervals in the given bounds

    bounds1 and bounds2 are a dict of {index:bounds},
    where bounds is a list of tuples [(lo,hi),...]
    """
    # ensure we have a list of tuples
    for (k,v) in getattr(bounds1, 'iteritems', bounds1.items)():
        if not hasattr(v[0], '__len__'):
            bounds1[k] = [v]
    # ensure we have a list of tuples
    for (k,v) in getattr(bounds2, 'iteritems', bounds2.items)():
        if not hasattr(v[0], '__len__'):
            bounds2[k] = [v]
    results = {}
    # get all entries in bounds1
    for k,v in getattr(bounds1, 'iteritems', bounds1.items)():
        m = bounds2.get(k, None) 
        if m is None:
            if union is False:
                results[k] = v
            continue
        # find intersection of all tuples of bounds
        if union is True:
            results[k] = _interval_union(m,v)
        else:
            results[k] = _interval_intersection(m,v)
        if not results[k]:
            del results[k]
    if union is True: # get the union of the bounds
        return results
    # get all entries in bounds2 not in bounds1
    for k in set(bounds2).difference(bounds1):
        results[k] = bounds2[k]
    return results


def indicator_overlap(dict1, dict2, union=False):
    """find the intersection for dicts of sets of indices

    dict1 and dict2 are dicts of {index:set},
    where set is a set of indices.
    """
    if union:
        return dict((i,dict1.get(i,set()).union(dict2.get(i,set()))) for i in set(dict1.keys()).union(dict2.keys()))
    return dict((i,dict1.get(i,set()).intersection(dict2.get(i,set()))) for i in set(dict1.keys()).intersection(dict2.keys()))


def no_mask(termination):
    """build dict of termination info {with mask: without mask}

    termination is a termination condition (with collapse)
    """
    from mystic.termination import state
    return dict((k,''.join([k.split('{',1)[0],str(v)])) for (k,v) in state(termination).items() if v.pop('mask',None) or k.startswith('Collapse'))


def _no_mask(info):
    """return termination info without mask

    info is info from a termination condition (with collapse)
    """
    if not info: return info
    from numpy import inf
    x,r = info.split('{',1)
    r = eval('{'+r)
    r.pop('mask')
    return ''.join((x,str(r)))


def unmasked_collapse(collapse, union=False):
    """apply the embedded mask to the given collapse

    collapse is a dict returned from a solver's Collapsed method
    """ #XXX: is this useful?
    def get_mask(k):
        return (k, eval(k.split(' with ',1)[-1]).get('mask',None))
    i,j,k = [get_mask(k)+(v,) for k,v in collapse.items()][0]
    if i.startswith(('CollapseCost','CollapseGrad')):
        j = interval_overlap(j,k,union)
    elif i.startswith(('CollapsePosition','CollapseWeight')):
        j = indicator_overlap(j,k,union)
    elif i.startswith(('CollapseAt','CollapseAs')):
        j = j.union(k) if union else j.intersection(k)
    else:
        j = {}
    return {i:j} if j else {} #XXX: pop value if (-inf,inf)?


def _masked_collapse(termination):
    """reformat the termination mask as a collapse dict

    termination is a termination condition
    """
    from mystic.termination import state
    return dict((i,j['mask']) for (i,j) in state(termination).items() if i.startswith('Collapse') and j['mask']) #XXX: pop value if (-inf,inf)?


def masked_collapse(termination, collapse=None, union=False):
    """extract mask from termination, and combine it with the given collapse

    termination is a termination condition, and collapse is a collapse dict
    """
    if collapse is None: collapse = {}
    masked = _masked_collapse(termination)
    masked = [(j,interval_overlap(collapse.get(j,{}),masked.get(j,{}), union)) for j in set(collapse.keys() + masked.keys())]
    return dict((j,i) for (j,i) in masked if i) #XXX: pop value if (-inf,inf)?


# a multiprocessing-friendly counter
from mystic._counter import Counter

# backward compatibility
from dill.source import getblocks as parse_from_history
from dill.source import getsource as src
from mystic.monitors import Monitor as Sow
from mystic.monitors import VerboseMonitor as VerboseSow
from mystic.monitors import LoggingMonitor as LoggingSow
from mystic.monitors import CustomMonitor as CustomSow
from mystic.monitors import Null

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

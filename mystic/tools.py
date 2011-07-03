#!/usr/bin/env python
#
# Patrick Hung & Mike McKerns, Caltech
#
# flatten was adapted from the python cookbook
# Null was adapted (and bugfixed) from the python cookbook
# wrap_function was adapted from numpy
# wrap_bounds was adapted from park
# src & parse_from_history were copied from pathos.pyina.ez_map

"""
Various python tools

Main functions exported are:: 
    - flatten: flatten a sequence
    - flatten_array: flatten an array 
    - Null: a Null object pattern
    - getch: provides "press any key to quit"
    - random_seed: sets the seed for calls to 'random()'
    - wrap_nested: nest a function call within a function object
    - wrap_function: bind an EvaluationMonitor and an evaluation counter
        to a function object
    - wrap_bounds: impose bounds on a funciton object
    - unpair: convert a 1D array of N pairs to two 1D arrays of N values
    - src: extract source code from a python code object

Other tools of interest are in::
    `mystic.mystic.filters` and `mystic.models.poly`
"""

def list_or_tuple(x):
    "True if x is a list or a tuple"
    return isinstance(x, (list, tuple))

def list_or_tuple_or_ndarray(x):
    "True if x is a list, tuple, or a ndarray"
    import numpy
    return isinstance(x, (list, tuple, numpy.ndarray))

def listify(x):
    "recursivly convert all members of a sequence to a list"
    if not list_or_tuple_or_ndarray(x): return x
    #if isinstance(x, numpy.ndarray) and x.ndim == 0: return x # i.e. array(1)
    if not list_or_tuple(x) and x.ndim == 0: return x.flatten()[0]
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

class Null(object):
    """A Null object

Null objects always and reliably "do nothing." """
    # optional optimization: ensure only one instance per subclass
    # (essentially just to save memory, no functional difference)
    #
    # from the Python cookbook, but type.__new__ replaced by object.__new__
    #
    def __new__(cls, *args, **kwargs):
        if '_inst' not in vars(cls):
            cls._inst = object.__new__(cls) #, *args, **kwargs)
        return cls._inst
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return self
    def __repr__(self): return "Null( )"
    def __nonzero__(self): return False
    def __getattr__(self, name): return self
    def __setattr__(self, name, value): return self
    def __delattr__(self, name): return self

def getch(str="Press any key to continue"):
    "configurable pause of execution"
    import os, sys
    if sys.stdin.isatty():
       if str != None:
          print str
       os.system('stty raw')
       a = sys.stdin.read(1)
       os.system('stty cooked')
       return a
    else:
       if str != None:
           print str + " and press enter"
       return raw_input()

def random_seed(s):
    "sets the seed for calls to 'random()'"
    import random
    random.seed(s)
    try: 
        from numpy import random
        random.seed(s)
    except:
        pass
    return

def wrap_nested(function, inner_function):
    """nest a function call within a function object

This is useful for nesting a constraints function in a cost function;
thus, the constraints will be enforced at every cost function evaluation.
    """
    def function_wrapper(x):
        _x = x[:] #XXX: trouble if x not a list or ndarray... maybe "deepcopy"?
        return function(inner_function(_x))
    return function_wrapper

def wrap_function(function, args, EvaluationMonitor):
    """bind an EvaluationMonitor and an evaluation counter
to a function object"""
    ncalls = [0]
    from numpy import array
    def function_wrapper(x):
        ncalls[0] += 1
        fval =  function(x, *args)
        EvaluationMonitor(x, fval)
        return fval
    return ncalls, function_wrapper

def wrap_bounds(function, min=None, max=None):
    "impose bounds on a funciton object"
    from numpy import asarray, any, inf
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
            if any((x<min)|(x>max)): #if violate bounds, evaluate as inf
                return inf
            return function(x)
    else:
        def function_wrapper(x):
            return function(x)
    return function_wrapper

def wrap_cf(CF, REG=None, cfmult = 1.0, regmult = 0.0):
    "wrap a cost function..."
    def _(*args, **kwargs):
         if REG is not None:
             return cfmult * CF(*args, **kwargs) + regmult * REG(*args, **kwargs)
         else:
             return cfmult * CF(*args, **kwargs)
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

def parse_from_history(object):
    """extract code blocks from a code object using stored history"""
    import readline, inspect
    lbuf = readline.get_current_history_length()
    code = [readline.get_history_item(i)+'\n' for i in range(1,lbuf)]
    lnum = 0
    codeblocks = []
    while lnum < len(code)-1:
       if code[lnum].startswith('def'):    
           block = inspect.getblock(code[lnum:])
           lnum += len(block)
           if block[0].startswith('def %s' % object.func_name):
               codeblocks.append(block)
       else:
           lnum +=1
    return codeblocks

def src(object):
    """Extract source code from python code object.

This function is designed to work with simple functions, and will not
work on any general callable. However, this function can extract source
code from functions that are defined interactively.
    """
    import inspect
    # no try/except (like the normal src function)
    if hasattr(object,'func_code') and object.func_code.co_filename == '<stdin>':
        # function is typed in at the python shell
        lines = parse_from_history(object)[-1]
    else:
        lines, lnum = inspect.getsourcelines(object)
    return ''.join(lines)

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
from monitors import Monitor as Sow
from monitors import VerboseMonitor as VerboseSow
from monitors import LoggingMonitor as LoggingSow
from monitors import CustomMonitor as CustomSow


if __name__=='__main__':
    import doctest
    doctest.testmod(verbose=True)

# End of file

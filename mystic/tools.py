#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                       Patrick Hung & Mike McKerns, Caltech
#                        (C) 1998-2008  All Rights Reserved
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

"""
Various python tools

Main function exported are: 

 -- flatten (adapted from the python cookbook)

 -- flatten_array 

 -- Null object pattern (from the python cookbook, bugfixed)

 -- getch (for "press any key to quit")

 -- Sow : Class whose instances are callable (compatible with EvaluationMonitor).

 -- VerboseSow : A verbose version of the basic Sow

 -- CustomSow : A customizable 'n-variable' version of the basic Sow

 -- random_seed : sets the seeds for python's random and numpy's random to seed value

 -- wrap_function : modified numpy's wrap_funciton to accept EvaluationMonitor

 -- wrap_bounds : impose bounds on a funciton object

 -- unpair : convert a 1D array of N pairs to two 1D arrays of N values

"""

def list_or_tuple(x):
    return isinstance(x, (list, tuple))

def list_or_tuple_or_ndarray(x):
    import numpy
    return isinstance(x, (list, tuple, numpy.ndarray))

def flatten_array(sequence, maxlev=999, lev=0):
    import numpy
    return numpy.array(list(flatten(sequence, maxlev, list_or_tuple_or_ndarray, lev)))

def flatten(sequence, maxlev=999, to_expand=list_or_tuple, lev=0):
    """
Example: Flatten([1,2,3,[4,5,6],7,[8,[9]]]) -> [1,2,3,4,5,6,7,8,9]
Example: Flatten([1,2,3,[4,5,6],7,[8,[9]]], 1) -> [1,2,3,4,5,6,7,8,[9]]

>>> A = [1,2,3,[4,5,6],7,[8,[9]]]

Flattens.
>>> list(flatten(A))
[1, 2, 3, 4, 5, 6, 7, 8, 9]

Flattens only one level deep
>>> list(flatten(A,1))
[1, 2, 3, 4, 5, 6, 7, 8, [9]]

Flattens twice. 
>>> list(flatten(A,2))
[1, 2, 3, 4, 5, 6, 7, 8, 9]

The original
>>> list(flatten(A,0))
[1, 2, 3, [4, 5, 6], 7, [8, [9]]]
    """
    for item in sequence:
        if lev < maxlev and to_expand(item):
            for subitem in flatten(item, maxlev, to_expand, lev+1):
                 yield subitem
        else:
            yield item

class Null(object):
    """ Null objects always and reliably "do nothing." """
    # optional optimization: ensure only one instance per subclass
    # (essentially just to save memory, no functional difference)
    #
    # comes from the Python cookbook, but type.__new__ replaced by object.__new__
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

class Sow(object):
    """
# Instances of objects can be passed as parameters to solver.Solve():

sow = Sow()
solver.Solve(rosen, x0, EvaulationMonitor=sow)

# Then parameters used to call the CostFunction can be retrieved via
sow.x

# the corresponding costs via
sow.y

    """
    def __init__(self):
        self._x = []   
        self._y = []   

    def __call__(self, x, y):
        self._x.append(x)
        self._y.append(y)
     
    def get_x(self):   
        return self._x

    def get_y(self):   
        return self._y

    x = property(get_x, doc = "Params")
    y = property(get_y, doc = "Costs")
    pass

class VerboseSow(Sow):
    import numpy
    def __init__(self, interval = 10, xinterval = numpy.inf):
        Sow.__init__(self)
        self._step = 0
        self._yinterval = interval
        self._xinterval = xinterval
        return
    def __call__(self, x, y):
        from numpy import ndarray
        Sow.__call__(self, x, y)
        self._step += 1
        if isinstance(y,(list,ndarray)):
            y = y[0] #XXX: get the "best" fit... which should be in y[0]
        if isinstance(x[0],(list,ndarray)): #XXX: x should always be iterable
            x = x[0] #XXX: get the "best" fit... which should be in x[0]
        if int(self._step % self._yinterval) == 0:
           #print "Generation %d has best Chi-Squared: %s" % (self._step, y)
            print "Generation %d has best Chi-Squared: %f" % (self._step, y)
        if int(self._step % self._xinterval) == 0:
            print "Generation %d has bet fit parameters: %s" % (self._step, x)
        return
    pass

def CustomSow(*args,**kwds):
    """
    generate a custom Sow

    takes *args & **kwds, where args will be required inputs for the Sow
      - args: property name strings (i.e. 'x')
      - kwds: must be in the form: property="doc" (i.e. x='Params')

    example usage...
      >>> sow = CustomSow('x','y',x="Params",y="Costs",e="Error",d="Deriv")
      >>> sow(1,1)
      >>> sow(2,4,e=0)
      >>> sow.x
      [1,2]
      >>> sow.y
      [1,4]
      >>> sow.e
      [0]
      >>> sow.d
      []
    """
    from _genSow import genSow
    return genSow(**kwds)(*args)

def random_seed(s):
    import random
    random.seed(s)
    try: 
        from numpy import random
        random.seed(s)
    except:
        pass
    return

def wrap_function(function, args, EvaluationMonitor):
    ncalls = [0]
    from numpy import array
    def function_wrapper(x):
        ncalls[0] += 1
        fval =  function(x, *args)
        EvaluationMonitor(x, fval)
        return fval
    return ncalls, function_wrapper

def wrap_bounds(function, min=None, max=None):
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
    def _(*args, **kwargs):
         if REG is not None:
             return cfmult * CF(*args, **kwargs) + regmult * REG(*args, **kwargs)
         else:
             return cfmult * CF(*args, **kwargs)
    return _


def unpair(pairs):
    '''convert a 1D array of N pairs to two 1D arrays of N values
For example:
    [a0,a1,a2],[b0,b1,b2] = unpair([(a0,b0),(a1,b1),(a2,b2)])'''
    from numpy import asarray
    pairsT = asarray(pairs).transpose()
    return [i.tolist() for i in pairsT]


if __name__=='__main__':
    import doctest
    doctest.testmod(verbose=True)

# End of file

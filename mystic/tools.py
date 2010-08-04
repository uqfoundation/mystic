#!/usr/bin/env python
#
# Patrick Hung & Mike McKerns, Caltech
#
# flatten was adapted from the python cookbook
# Null was adapted (and bugfixed) from the python cookbook
# wrap_function was adapted from numpy
# wrap_bounds was adapted from park

"""
Various python tools

Main functions exported are:: 
    - flatten: flatten a sequence
    - flatten_array: flatten an array 
    - Null: a Null object pattern
    - getch: provides "press any key to quit"
    - Sow: A class whose instances are callable (to be used as monitors)
    - VerboseSow: A verbose version of the basic Sow
    - LoggingSow: A version of the basic Sow that logs to a file
    - CustomSow: A customizable 'n-variable' version of the basic Sow
    - random_seed: sets the seed for calls to 'random()'
    - wrap_function: bind an EvaluationMonitor and an evaluation counter
        to a function object
    - wrap_bounds: impose bounds on a funciton object
    - unpair: convert a 1D array of N pairs to two 1D arrays of N values

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

class Sow(object):
    """
Instances of objects that can be passed as monitors.
The Sow logs the parameters and corresponding costs,
retrievable by accessing the Sow's member variables.

example usage...
    >>> sow = Sow()
    >>> solver.Solve(rosen, x0, EvaulationMonitor=sow)
    >>> sow.x   # get log of parameters
    >>> sow.y   # get log of costs

    """
    def __init__(self, **kwds):#, all=True):
        self._x = []
        self._y = []
        self._id = []
       #self._all = all

    def __call__(self, x, y, id=None, **kwds):#, best=0):
        self._x.append(listify(x)) #XXX: better to save as-is?
        self._y.append(listify(y)) #XXX: better to save as-is?
        self._id.append(id)
       #if not self._all and list_or_tuple_or_ndarray(x):
       #    self._x[-1] = self._x[-1][best]
       #if not self._all and list_or_tuple_or_ndarray(y):
       #    self._y[-1] = self._y[-1][best]

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_id(self):
        return self._id

    x = property(get_x, doc = "Params")
    y = property(get_y, doc = "Costs")
    id = property(get_id, doc = "Id")
    pass

class VerboseSow(Sow):
    """A verbose version of the basic Sow.

Prints ChiSq every 'interval', and optionally prints
current parameters every 'xinterval'.
    """
    import numpy
    def __init__(self, interval = 10, xinterval = numpy.inf, all=True):
        super(VerboseSow,self).__init__()
        self._step = 0
        self._yinterval = interval
        self._xinterval = xinterval
        self._all = all
        return
    def __call__(self, x, y, id=None, best=0):
        super(VerboseSow,self).__call__(x, y, id)
        if int(self._step % self._yinterval) == 0:
            if not list_or_tuple_or_ndarray(y):
                who = ''
                y = " %f" % self._y[-1]
            elif self._all:
                who = ''
                y = " %s" % self._y[-1]
            else:
                who = ' best'
                y = " %f" % self._y[-1][best]
            msg = "Generation %d has%s Chi-Squared:%s" % (self._step, who, y)
            if id != None: msg = "[id: %d] " % (id) + msg
            print msg
        if int(self._step % self._xinterval) == 0:
            if not list_or_tuple_or_ndarray(x):
                who = ''
                x = " %f" % self._x[-1]
            elif self._all:
                who = ''
                x = "\n %s" % self._x[-1]
            else:
                who = ' best'
                x = "\n %s" % self._x[-1][best]
            msg = "Generation %d has%s fit parameters:%s" % (self._step, who, x)
            if id != None: msg = "[id: %d] " % (id) + msg
            print msg
        self._step += 1
        return
    pass

class LoggingSow(Sow):
    """A version of the basic Sow that writes to a file at specified intervals.

Logs ChiSq and parameters to a file every 'interval'
    """
    import numpy
    def __init__(self, interval=1, filename='log.txt', new=False, all=True):
        import datetime
        super(LoggingSow,self).__init__()
        self._filename = filename
        self._step = 0
        self._yinterval = interval
        self._xinterval = interval
        if new: ind = 'w'
        else: ind = 'a'
        self._file = open(self._filename,ind)
        self._file.write("# %s\n" % datetime.datetime.now().ctime() )
        self._file.write("# ___#___  __ChiSq__  __params__\n")
        self._file.close()
        self._all = all
        return
    def __call__(self, x, y, id=None, best=0):
        self._file = open(self._filename,'a')
        super(LoggingSow,self).__call__(x, y, id)
        if int(self._step % self._yinterval) == 0:
            if not list_or_tuple_or_ndarray(y):
                y = "%f" % self._y[-1]
            elif self._all:
                y = "%s" % self._y[-1]
            else:
                y = "%f" % self._y[-1][best]
            if not list_or_tuple_or_ndarray(x):
                x = "[%f]" % self._x[-1]
            elif self._all:
                xa = self._x[-1]
                if not list_or_tuple_or_ndarray(xa):
                  x = "[%f]" % xa
                else:
                  x = "%s" % xa
            else:
                xb = self._x[-1][best]
                if not list_or_tuple_or_ndarray(xb):
                  x = "[%f]" % xb
                else:
                  x = "%s" % xb
            step = [self._step]
            if id != None: step.append(id)
            self._file.write("  %s     %s   %s\n" % (tuple(step), y, x))
        self._step += 1
        self._file.close()
        return
    pass

def CustomSow(*args,**kwds):
    """
generate a custom Sow

takes *args & **kwds, where args will be required inputs for the Sow::
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
    "sets the seed for calls to 'random()'"
    import random
    random.seed(s)
    try: 
        from numpy import random
        random.seed(s)
    except:
        pass
    return

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


if __name__=='__main__':
    import doctest
    doctest.testmod(verbose=True)

# End of file

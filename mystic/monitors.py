#!/usr/bin/env python
#
# Null was adapted (and bugfixed) from the python cookbook
"""
monitors: callable class instances that record data


Monitors
========

Montors provide the ability to monitor progress as the optimization
is underway. Monitors also can be used to extract and prepare information
for mystic's analysis viewers. Each of mystic's monitors are customizable,
and provide the user with a different type of output. The following
monitors are available::
    - Monitor        -- the basic monitor; only writes to internal state
    - VerboseMonitor -- a verbose monitor; also writes to stdout/stderr
    - LoggingMonitor -- a logging monitor; also writes to a logfile
    - CustomMonitor  -- a customizable 'n-variable' version of Monitor
    - Null           -- a null object, which reliably does nothing


Usage
=====

Typically monitors are either bound to a model function by a modelFactory,
or bound to a cost function by a Solver.  The typical usage pattern is::

    >>> # get and configure monitors
    >>> from mystic.monitors import Monitor, VerboseMonitor
    >>> evalmon = Monitor()
    >>> stepmon = VerboseMonitor(5)
    >>>
    >>> # instantiate and configure the solver
    >>> from mystic.solvers import NelderMeadSimplexSolver
    >>> from mystic.termination import CandidateRelativeTolerance as CRT
    >>> solver = NelderMeadSimplexSolver(len(x0))
    >>> solver.SetInitialPoints(x0)
    >>>
    >>> # associate the monitor with a solver, then solve
    >>> solver.SetEvaluationMonitor(evalmon)
    >>> solver.SetGenerationMonitor(stepmon)
    >>> solver.Solve(rosen, CRT())
    >>>
    >>> # access the 'iteration' history
    >>> stepmon.x     # parameters after each iteration
    >>> stepmon.y     # cost after each iteration
    >>>
    >>> # access the 'evaluation' history
    >>> evalmon.x     # parameters after each evaluation
    >>> evalmon.y     # cost after each evaluation


"""

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

class Monitor(object):
    """
Instances of objects that can be passed as monitors.
Typically, a Monitor logs a list of parameters and the
corresponding costs, retrievable by accessing the Monitor's
member variables.

example usage...
    >>> sow = Monitor()
    >>> sow([1,2],3)
    >>> sow([4,5],6)
    >>> sow.x
    [[1, 2], [4, 5]]
    >>> sow.y
    [3, 6]

    """
    def __init__(self, **kwds):#, all=True):
        self._x = []
        self._y = []
        self._id = []
       #self._all = all

    def __call__(self, x, y, id=None, **kwds):#, best=0):
        from mystic.tools import listify
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

class VerboseMonitor(Monitor):
    """A verbose version of the basic Monitor.

Prints ChiSq every 'interval', and optionally prints
current parameters every 'xinterval'.
    """
    import numpy
    def __init__(self, interval = 10, xinterval = numpy.inf, all=True):
        super(VerboseMonitor,self).__init__()
        self._step = 0
        self._yinterval = interval
        self._xinterval = xinterval
        self._all = all
        return
    def __call__(self, x, y, id=None, best=0):
        from mystic.tools import list_or_tuple_or_ndarray
        super(VerboseMonitor,self).__call__(x, y, id)
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

class LoggingMonitor(Monitor):
    """A basic Monitor that writes to a file at specified intervals.

Logs ChiSq and parameters to a file every 'interval'
    """
    import numpy
    def __init__(self, interval=1, filename='log.txt', new=False, all=True):
        import datetime
        super(LoggingMonitor,self).__init__()
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
        from mystic.tools import list_or_tuple_or_ndarray
        self._file = open(self._filename,'a')
        super(LoggingMonitor,self).__call__(x, y, id)
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

def CustomMonitor(*args,**kwds):
    """
generate a custom Monitor

takes *args & **kwds, where args will be required inputs for the Monitor::
    - args: property name strings (i.e. 'x')
    - kwds: must be in the form: property="doc" (i.e. x='Params')

example usage...
    >>> sow = CustomMonitor('x','y',x="Params",y="Costs",e="Error",d="Deriv")
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


# end of file

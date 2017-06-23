#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2017 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
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
    - LoggingMonitor -- a logging monitor; also writes to a logfile
    - VerboseMonitor -- a verbose monitor; also writes to stdout/stderr
    - VerboseLoggingMonitor -- a verbose logging monitor; best of both worlds
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
__all__ = ['Null', 'Monitor', 'VerboseMonitor', 'LoggingMonitor',
           'VerboseLoggingMonitor', 'CustomMonitor', 
           '_solutions', '_measures', '_positions', '_weights', '_load']

import os
import sys
import numpy
from mystic.tools import list_or_tuple_or_ndarray
from mystic.tools import listify, multiply, divide, _kdiv
from functools import reduce

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
    def __repr__(self): return "Null()"
    def __bool__(self): return False
    def __nonzero__(self): return False
    def __getattr__(self, name): return self
    def __setattr__(self, name, value): return self
    def __delattr__(self, name): return self
    def __len__(self): return
    def __getnewargs__(self): return ()
# comply with monitor interface
Null.info = Null()
Null.k = None
#XXX: should also have Null.x, Null.y ?


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
        self._info = []
       #self._all = all
        self.k = kwds.pop('k', None)
        self._npts = kwds.pop('npts', None)

    def __len__(self):
        return len(self.x)

    def info(self, message):
        self._info.append("%s" % "".join(["",str(message)]))
        return

    def __call__(self, x, y, id=None, **kwds):#, best=0):
        self._x.append(listify(x)) #XXX: listify?
        self._y.append(listify(self._k(y, iter))) #XXX: listify?
        self._id.append(id)
       #if not self._all and list_or_tuple_or_ndarray(x):
       #    self._x[-1] = self._x[-1][best]
       #if not self._all and list_or_tuple_or_ndarray(y):
       #    self._y[-1] = self._y[-1][best]

    def extend(self, monitor):
        """append the contents of the given monitor"""
        if isinstance(monitor, Monitor): # is Monitor()
            pass
        elif (monitor == Null) or isinstance(monitor, Null): # Null or Null()
            monitor = Monitor()
        elif hasattr(monitor, '__module__') and \
            monitor.__module__ in ['mystic._genSow']: # CustomMonitor()
                pass #XXX: CustomMonitor may fail...
        else:
            raise TypeError("'%s' is not a monitor instance" % monitor)
        self._x.extend(monitor._x)
        self._y.extend(self._get_y(monitor))      # scalar, up to 2x faster
       #self._y.extend(self._k(monitor.iy, iter)) # vector, results like numpy
        self._id.extend(monitor._id)
        self._info.extend(monitor._info)

    def prepend(self, monitor):
        """prepend the contents of the given monitor"""
        if isinstance(monitor, Monitor): # is Monitor()
            pass
        elif (monitor == Null) or isinstance(monitor, Null): # Null or Null()
            monitor = Monitor()
        elif hasattr(monitor, '__module__') and \
            monitor.__module__ in ['mystic._genSow']: # CustomMonitor()
                pass #XXX: CustomMonitor may fail...
        else:
            raise TypeError("'%s' is not a monitor instance" % monitor)
        [self._x.insert(*i) for i in enumerate(monitor._x)]
        [self._y.insert(*i) for i in enumerate(self._get_y(monitor))]
       #[self._y.insert(*i) for i in enumerate(self._k(monitor.iy, iter))]
        [self._id.insert(*i) for i in enumerate(monitor._id)]
        [self._info.insert(*i) for i in enumerate(monitor._info)]
        #XXX: may be faster ways of doing the above...
        #     (e.g. deepcopy(monitor) allows enumerate w/o list())

    def get_x(self):
        return self._x

    def get_id(self):
        return self._id

    def get_info(self):
        return self._info

    def __step(self):
        return len(self.x)

    ##### measures #####
    def get_iwts(self):
        wts = []
        if self._npts is None: return self._npts
        for (i,n) in enumerate(self._npts):
            indx = 2*reduce(lambda x,y:x+y, (0,)+self._npts[:i])
            wts.extend(range(indx,n+indx))
        return wts

    def get_ipos(self):
        pos = []
        if self._npts is None: return self._npts
        for (i,n) in enumerate(self._npts):
            indx = 2*reduce(lambda x,y:x+y, (0,)+self._npts[:i])
            pos.extend(range(indx+self._npts[0],n+indx+self._npts[0])) 
        return pos

    def get_wts(self):
        wts = self._wts
        if wts is None: return wts
        wts = numpy.array(self.x)[:, wts]
        wts.shape = (wts.shape[0], len(self._npts), -1)
        return wts.tolist()  #XXX: as list or array?

    def get_pos(self):
        pos = self._pos
        if pos is None: return pos
        pos = numpy.array(self.x)[:, pos]
        pos.shape = (pos.shape[0], len(self._npts), -1)
        return pos.tolist()  #XXX: as list or array?
    ####################

    #BELOW: madness due to monitor k-conversion

    def get_y(self): # can be slow if k not in (1, None)
        return divide(self._y, self.k, list)
        #XXX: better if everywhere y = _y, as opposed to y = _ik(_y) ?
        #     better if k only applied to 'output' of __call__ ?
        #     better if k ionly applied on 'exit' from solver ?

    def _get_y(self, monitor):
        "avoid double-conversion by combining k's"
        _ik = _kdiv(monitor.k, self.k, float) #XXX: always a float?
        return divide(monitor._y, _ik, iter)

    def get_ix(self):
        return divide(self._y, 1, iter) #XXX: _y ?

    def get_ax(self):
        return divide(self._y, 1, numpy.array) #XXX: _y ?

    def get_iy(self):
        return divide(self._y, self.k, iter)

    def get_ay(self):
        return divide(self._y, self.k, numpy.array)

    def _k(self, y, type=list):
        return multiply(y, self.k, type)

    def _ik(self, y, k=False, type=list):
        if k: return y # k's already applied, so don't un-apply it
        return divide(y, self.k, type)

    _step = property(__step)
    x = property(get_x, doc = "Params")
    ix = property(get_ix, doc = "Params")
    ax = property(get_ax, doc = "Params")
    y = property(get_y, doc = "Costs")
    iy = property(get_iy, doc = "Costs")
    ay = property(get_ay, doc = "Costs")
    id = property(get_id, doc = "Id")
    wts = property(get_wts, doc = "Weights")
    pos = property(get_pos, doc = "Positions")
    _wts = property(get_iwts, doc = "Weights")
    _pos = property(get_ipos, doc = "Positions")
    pass

class VerboseMonitor(Monitor):
    """A verbose version of the basic Monitor.

Prints ChiSq every 'interval', and optionally prints
current parameters every 'xinterval'.
    """
    def __init__(self, interval=10, xinterval=numpy.inf, all=True, **kwds):
        super(VerboseMonitor,self).__init__(**kwds)
        if not interval or interval is numpy.nan: interval = numpy.inf
        if not xinterval or xinterval is numpy.nan: xinterval = numpy.inf
        self._yinterval = interval
        self._xinterval = xinterval
        self._all = all
        return
    def info(self, message):
        super(VerboseMonitor,self).info(message)
        print("%s" % "".join(["",str(message)]))
        return
    def __call__(self, x, y, id=None, best=0, k=False):
        super(VerboseMonitor,self).__call__(x, y, id, k=k)
        if self._yinterval is not numpy.inf and \
           int((self._step-1) % self._yinterval) == 0:
            if not list_or_tuple_or_ndarray(y):
                who = ''
                y = " %f" % self._ik(self._y[-1], k)
            elif self._all:
                who = ''
                y = " %s" % self._ik(self._y[-1], k)
            else:
                who = ' best'
                y = " %f" % self._ik(self._y[-1][best], k)
            msg = "Generation %d has%s Chi-Squared:%s" % (self._step-1,who,y)
            if id is not None: msg = "[id: %d] " % (id) + msg
            print(msg)
        if self._xinterval is not numpy.inf and \
           int((self._step-1) % self._xinterval) == 0:
            if not list_or_tuple_or_ndarray(x):
                who = ''
                x = " %f" % self._x[-1]
            elif self._all:
                who = ''
                x = "\n %s" % self._x[-1]
            else:
                who = ' best'
                x = "\n %s" % self._x[-1][best]
            msg = "Generation %d has%s fit parameters:%s" % (self._step-1,who,x)
            if id is not None: msg = "[id: %d] " % (id) + msg
            print(msg)
        return
    pass

class LoggingMonitor(Monitor):
    """A basic Monitor that writes to a file at specified intervals.

Logs ChiSq and parameters to a file every 'interval'
    """
    def __init__(self, interval=1, filename='log.txt', new=False, all=True, info=None, **kwds):
        import datetime
        super(LoggingMonitor,self).__init__(**kwds)
        self._filename = filename
        if not interval or interval is numpy.nan: interval = numpy.inf
        self._yinterval = interval
        self._xinterval = interval
        if new: ind = 'w'
        else: ind = 'a'
        self._file = open(self._filename,ind)
        self._file.write("# %s\n" % datetime.datetime.now().ctime() )
        if info: self._file.write("# %s\n" % str(info))
        self._file.write("# ___#___  __ChiSq__  __params__\n")
        self._file.close()
        self._all = all
        return
    def info(self, message):
        super(LoggingMonitor,self).info(message)
        self._file = open(self._filename,'a')
        self._file.write("# %s\n" % str(message))
        self._file.close()
        return
    def __call__(self, x, y, id=None, best=0, k=False):
        self._file = open(self._filename,'a')
        super(LoggingMonitor,self).__call__(x, y, id, k=k)
        if self._yinterval is not numpy.inf and \
           int((self._step-1) % self._yinterval) == 0:
            if not list_or_tuple_or_ndarray(y):
                y = "%f" % self._ik(self._y[-1], k)
            elif self._all:
                y = "%s" % self._ik(self._y[-1], k)
            else:
                y = "%f" % self._ik(self._y[-1][best], k)
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
            step = [self._step-1]
            if id is not None: step.append(id)
            self._file.write("  %s     %s   %s\n" % (tuple(step), y, x))
        self._file.close()
        return
    def __reduce__(self):
        interval = self._yinterval        
        filename = self._filename
        new = False
        all = self._all
        info = None
        args = (interval, filename, new, all, info)
        k = self.k
        state = dict(_x=self._x,_y=self._y,_id=self._id,_info=self._info,k=k)
        return (self.__class__, args, state)
    def __setstate__(self, state):
        self.__dict__.update(state)
        return
    pass

class VerboseLoggingMonitor(LoggingMonitor):
    """A Monitor that writes to a file and the screen at specified intervals.

Logs ChiSq and parameters to a file every 'interval', print every 'yinterval'
    """
    def __init__(self, interval=1, yinterval=10, xinterval=numpy.inf, filename='log.txt', new=False, all=True, info=None, **kwds):
        super(VerboseLoggingMonitor,self).__init__(interval,filename,new,all,info,**kwds)
        if not yinterval or yinterval is numpy.nan: yinterval = numpy.inf
        if not xinterval or xinterval is numpy.nan: xinterval = numpy.inf
        self._vyinterval = yinterval
        self._vxinterval = xinterval
        return
    def info(self, message):
        super(VerboseLoggingMonitor,self).info(message)
        print("%s" % "".join(["",str(message)]))
        return
    def __call__(self, x, y, id=None, best=0, k=False):
        super(VerboseLoggingMonitor,self).__call__(x, y, id, best, k=k)
        if self._vyinterval is not numpy.inf and \
           int((self._step-1) % self._vyinterval) == 0:
            if not list_or_tuple_or_ndarray(y):
                who = ''
                y = " %f" % self._ik(self._y[-1], k)
            elif self._all:
                who = ''
                y = " %s" % self._ik(self._y[-1], k)
            else:
                who = ' best'
                y = " %f" % self._ik(self._y[-1][best], k)
            msg = "Generation %d has%s Chi-Squared:%s" % (self._step-1,who,y)
            if id is not None: msg = "[id: %d] " % (id) + msg
            print(msg)
        if self._vxinterval is not numpy.inf and \
           int((self._step-1) % self._vxinterval) == 0:
            if not list_or_tuple_or_ndarray(x):
                who = ''
                x = " %f" % self._x[-1]
            elif self._all:
                who = ''
                x = "\n %s" % self._x[-1]
            else:
                who = ' best'
                x = "\n %s" % self._x[-1][best]
            msg = "Generation %d has%s fit parameters:%s" % (self._step-1,who,x)
            if id is not None: msg = "[id: %d] " % (id) + msg
            print(msg)
        return
    def __reduce__(self):
        interval = self._yinterval
        yint = self._vyinterval
        xint = self._vxinterval
        filename = self._filename
        new = False
        all = self._all
        info = None
        args = (interval, yint, xint, filename, new, all, info)
        k = self.k
        state = dict(_x=self._x,_y=self._y,_id=self._id,_info=self._info,k=k)
        return (self.__class__, args, state)
    def __setstate__(self, state):
        self.__dict__.update(state)
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
    from mystic._genSow import genSow
    return genSow(**kwds)(*args)


##### loaders ##### #XXX: should be class method?
if (sys.hexversion >= 0x30000f0):
    exec_string = 'exec(code, _globals)'
else:
    exec_string = 'exec code in _globals'
#FIXME: remove this head-standing to workaround python2.6 exec bug
def_load = """
def _load(path, monitor=None, verbose=False): #XXX: duplicate in mystic.munge?
    '''load npts, params, and cost into monitor from file at given path'''

    base = os.path.splitext(os.path.basename(path))[0]
    root = os.path.realpath(os.path.dirname(path))

    string = '''
from {base} import params as ___params, cost as ___cost;
try: from {base} import npts as ___npts;
except ImportError: ___npts = None;
import sys;
sys.modules.pop('{base}', None);
'''.format(base=base)

    try:
        sys.path.insert(0, root)
        _globals = {}
        _globals.update(globals())
        code = compile(string, '<string>', 'exec')
        %s
        npts = _globals['___npts'] if '___npts' in _globals else None
        params = _globals['___params'] if '___params' in _globals else None
        cost = _globals['___cost'] if '___cost' in _globals else None
        del _globals
    except: #XXX: should only catch the appropriate exceptions
        raise OSError("error reading '{path}'".format(path=path))

    finally:
        sys.path.remove(root)

    _new = monitor is None or not verbose
    m = Monitor() if _new else monitor

    #for p,c in zip(zip(*params), cost):
    for p,c in zip((zip(*i)[0] for i in zip(*params)),cost):
        m(p,c)

    if _new:
        monitor = m
    else:
        monitor.extend(m)

    monitor._npts = npts

    return monitor
""" % exec_string
exec(def_load)
del def_load, exec_string

##### readers ##### #XXX: should be class methods?
def _solutions(monitor, last=None):
    '''return the params from the last N entries in a monitor'''
    indx = last if last is None else -last
    return numpy.array(monitor.x[indx:])


def _measures(monitor, last=None, weights=False):
    '''return positions or weights from the last N entries in a monitor

    this function requires a montor that is monitoring a product_measure'''
    indx = last if last is None else -last
    return numpy.array(monitor.wts[indx:] if weights else monitor.pos[indx:])


def _positions(monitor, last=None):
    '''return positions from the last N entries in a monitor

    this function requires a montor that is monitoring a product_measure'''
    return _measures(monitor, last, weights=False)


def _weights(monitor, last=None):
    '''return weights from the last N entries in a monitor
    
    this function requires a montor that is monitoring a product_measure'''
    return _measures(monitor, last, weights=True)


"""
def __measures(monitor, last=None, weights=False):
    '''return positions or weights from the last N entries in a monitor

    this function requires a montor that is monitoring a product_measure'''
    #XXX: alternate, using product_measure
    from mystic.math.discrete import product_measure
    # get npts and solution_vector
    npts, history = monitor._npts, _solutions(monitor, last)
    # get product_measure.wts for all generations
    xxx = 'wts' if weights else 'pos'
    for (i,step) in enumerate(history):
        c = product_measure()
        c.load(step, npts)
        history[i] = getattr(c,xxx)
    return history
"""


# end of file

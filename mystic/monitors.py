#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Patrick Hung (patrickh @caltech)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
#
# Null was adapted (and bugfixed) from the python cookbook
"""
monitors: callable class instances that record data


Monitors
========

Monitors provide the ability to monitor progress as the optimization
is underway. Monitors also can be used to extract and prepare information
for mystic's analysis viewers. Each of mystic's monitors are customizable,
and provide the user with a different type of output. The following
monitors are available::

    Monitor        -- the basic monitor; only writes to internal state
    LoggingMonitor -- a logging monitor; also writes to a logfile
    VerboseMonitor -- a verbose monitor; also writes to stdout/stderr
    VerboseLoggingMonitor -- a verbose logging monitor; best of both worlds
    CustomMonitor  -- a customizable 'n-variable' version of Monitor
    Null           -- a null object, which reliably does nothing


Usage
=====

Typically monitors are either bound to a model function by a ``modelFactory``,
or bound to a cost function by a ``Solver``.

Examples:
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
from functools import reduce
from mystic.tools import list_or_tuple_or_ndarray
from mystic.tools import listify, _kdiv, _divide, _idivide, \
                         _adivide, _cdivide, _cmultiply

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
    def __len__(self): return 0
    def __getnewargs__(self): return ()
    def __getitem__(self, y): return self
    def __setitem__(self, i, y): return
    def min(self): return self
# comply with monitor interface (are these the best responses?)
Null.info = Null()
Null.k = None
Null.x = Null._x = ()
Null.y = Null._y = ()
Null._id = ()
Null._npts = None
Null.label = None


#XXX: use mon._g (for gradient) if provided, else None/np.nan?
#XXX: enable pointer linking mon2._x to mon._x? or index mon._x in mon2._x?
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
        if 'k' in kwds:
            self.k = kwds['k']; del kwds['k']
        else: self.k = None
        if 'npts' in kwds:
            self._npts = kwds['npts']; del kwds['npts']
        else: self._npts = None
        if 'label' in kwds:
            self.label = kwds['label']; del kwds['label']
        else: self.label = 'ChiSquare'

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

    def __add__(self, monitor):
        """add the contents of self and the given monitor"""
        #m = self.__class__()
        #m.extend(self)
        #XXX: alternately (below: preserve name and other properties)
        import copy
        m = copy.deepcopy(self)
        m.extend(monitor)
        return m

    def __getitem__(self, y):
        """x.__getitem__(y) <==> x[y]"""
        if type(y) is int:
            return self.x[y],self.y[y]
        import copy
        m = copy.deepcopy(self)
        m._info = [] #XXX: best to lose all or retain all?
        #m = self.__class__() #XXX: workaround (duplicates entries by copy)
        if type(y) in (list,numpy.ndarray):
            m._x = numpy.array(self._x)[y].tolist()
            m._y = numpy.array(self._y)[y].tolist()
            m._id = numpy.array(self._id)[y].tolist()
        elif type(y) is tuple: #XXX: good idea? Needs more testing...
            nn,nx,ny = len(y),numpy.ndim(self._x),numpy.ndim(self._y)
            ni = numpy.ndim(self._id)
            m._x = numpy.array(self._x)[y if nn == nx else y[0]].tolist()
            m._y = numpy.array(self._y)[y if nn == ny else y[0]].tolist()
            m._id = numpy.array(self._id)[y if nn == ni else y[0]].tolist()
        else:
            m._x = self._x[y]
            m._y = self._y[y]
            m._id = self._id[y]
        return m

    def __setitem__(self, i, y):
        """x.__setitem__(i, y) <==> x[i]=y"""
        if isinstance(y, Monitor): # is Monitor()
            pass
        elif (y == Null) or isinstance(y, Null): # Null or Null()
            y = Monitor()
        elif hasattr(y, '__module__') and \
            y.__module__ in ['mystic._genSow']: # CustomMonitor()
                pass #XXX: CustomMonitor may fail...
        else:
            raise TypeError("'%s' is not a monitor instance" % y)
        if type(i) is int:
            self._x[i:i+1] = y._x
            self._y[i:i+1] = y._y
            self._id[i:i+1] = y._id
            return
        if type(i) in (list,numpy.ndarray):
            x = numpy.array(self._x)
            x[i] = y._x
            self._x[:] = x.tolist()
            x = numpy.array(self._y)
            x[i] = y._y
            self._y[:] = x.tolist()
            x = numpy.array(self._id)
            x[i] = y._id
            self._id[:] = x.tolist()
       #elif type(i) is tuple: #XXX: good idea? Needs more testing...
       #    nn,nx,ny = len(i),numpy.ndim(self._x),numpy.ndim(self._y)
       #    x = numpy.array(self._x)
       #    x[i if nn == nx else i[0]] = y._x
       #    self._x[:] = x.tolist()
       #    x = numpy.array(self._y)
       #    x[i if nn == ny else i[0]] = y._y
       #    self._y[:] = x.tolist()
        else:
            self._x[i] = y._x
            self._y[i] = y._y
            self._id[i] = y._id
        return

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

    def min(self): # should be the last entry, generally
        """get the minimum monitor entry"""
        return self[self.ay.argmin()]

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
        return _divide(self._y, 1 if self.k is None else self.k)
        #XXX: better if everywhere y = _y, as opposed to y = _ik(_y) ?
        #     better if k only applied to 'output' of __call__ ?
        #     better if k ionly applied on 'exit' from solver ?

    def _get_y(self, monitor):
        "avoid double-conversion by combining k's"
        _ik = _kdiv(monitor.k, self.k, float) #XXX: always a float?
        return _idivide(monitor._y, _ik)

    def get_ix(self):
        return _idivide(self._x, 1) #XXX: _y ?

    def get_ax(self):
        return _adivide(self._x, 1) #XXX: _y ?

    def get_iy(self):
        return _idivide(self._y, 1 if self.k is None else self.k)

    def get_ay(self):
        return _adivide(self._y, 1 if self.k is None else self.k)

    def _k(self, y, type=list):
        return _cmultiply(y, self.k, type)

    def _ik(self, y, k=False, type=list):
        if k: return y # k's already applied, so don't un-apply it
        return _cdivide(y, self.k, type)

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

Prints output 'y' every 'interval', and optionally prints
input parameters 'x' every 'xinterval'.
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
            msg = "Generation %d has%s %s:%s" % (self._step-1,who,self.label,y)
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

Logs output 'y' and input parameters 'x' to a file every 'interval'.
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
        self._file.write("# ___#___  __%s__  __params__\n" % self.label)
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
        state = dict(_x=self._x,_y=self._y,_id=self._id,_info=self._info,k=k,label=self.label)
        return (self.__class__, args, state)
    def __setstate__(self, state):
        self.__dict__.update(state)
        return
    pass

class VerboseLoggingMonitor(LoggingMonitor):
    """A Monitor that writes to a file and the screen at specified intervals.

Logs output 'y' and input parameters 'x' to a file every 'interval', also
print every 'yinterval'.
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
            msg = "Generation %d has%s %s:%s" % (self._step-1,who,self.label,y)
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
        state = dict(_x=self._x,_y=self._y,_id=self._id,_info=self._info,k=k,label=self.label)
        return (self.__class__, args, state)
    def __setstate__(self, state):
        self.__dict__.update(state)
        return
    pass

def CustomMonitor(*args,**kwds):
    """generate a custom Monitor

Args:
    args (tuple(str)): tuple of the required Monitor inputs (e.g. ``x``).
    kwds (dict(str)): dict of ``{"input":"doc"}`` (e.g. ``x='Params'``).

Returns:
    a customized monitor instance

Examples:
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
    for p,c in zip((list(zip(*i))[0] for i in zip(*params)),cost):
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

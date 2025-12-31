#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from mystic.tools import list_or_tuple_or_ndarray as sequence
from mystic.tools import isNull

import sys

# generalized history reader

def read_history(source, iter=False):
    """read parameter history and cost history from the given source

source is a monitor, logfile, support file, solver restart file, dataset, etc
iter is a bool, where if True, `iter` is also returned

Returns (iter, params, cost) if iter is True, else returns (params, cost), where
    - iter is a list of tuples of ints (iteration, id), where id may be None
    - params is input to cost; a list of lists of floats
    - cost is output of cost; a list of floats, or a list of tuples of floats
    """
    monitor = solver = data = arxiv = False
    from mystic.monitors import Monitor, Null
    from mystic.abstract_solver import AbstractSolver
    from mystic.math.legacydata import dataset
    from klepto.archives import archive, cache
    import io
    file = io.IOBase
    if isinstance(source, file):
        return read_history(source.name, iter)
    if isinstance(source, str):
        import re
        source = re.sub(r'\.py*.$', '', source)  # strip off .py* extension
    elif isinstance(source, Monitor):
        monitor = True
    elif isinstance(source, dataset):
        data = True
    elif isinstance(source, cache):
        source = source.__archive__
        arxiv = True
    elif isinstance(source, archive):
        arxiv = True
    elif isinstance(source, AbstractSolver):
        solver = True
    elif isinstance(source, Null): #XXX: or source.x, source.y or Error?
        return ([],[],[]) if iter else ([],[])
    else:
        raise IOError("a history filename or instance is required")
    ids = None
    try:  # read standard logfile (or monitor)
        if monitor:
            if iter:
                params, cost, ids = read_monitor(source, id=True)
                if not len(ids): ids = None
                ids = _process_ids(ids, len(cost))
            else:
                params, cost = read_monitor(source)
        elif data:
            params, cost = source.coords, source.values
            if iter: ids = _process_ids(source.ids, len(cost))
        elif arxiv: #NOTE: does not store ids
            params = list(list(k) for k in source.keys())
            cost = source.values()
            cost = cost if type(cost) is list else list(cost)
            if iter: ids = _process_ids(None, len(cost))
        elif solver: #NOTE: ids are single-valued, unless ensemble
            params, cost = source.solution_history, source.energy_history
            if iter: ids = _process_ids(source.id, len(cost))
        else:
            try:
                from mystic.solvers import LoadSolver
                return read_history(LoadSolver(source), iter)
            except: #KeyError
                ids, params, cost = logfile_reader(source, iter=True)
                if iter: ids = _process_ids(ids, len(cost))
                #FIXME: doesn't work for multi-id logfile; select id?
        params, cost = raw_to_support(params, cost)
    except:
        return read_raw_file(source+'.py', iter) #NOTE: assumes 'support' format
    return (ids, params, cost) if iter else (params, cost)


# logfile reader
#XXX: if have grad ([i,j,k]) not cost, 'cost' is list of lists [[i,j,k],...]
#XXX: if have grad and cost, 'cost' is mixed list [n,[i,j,k],m,[a,b,c],...]
#TODO: (mixed) create a function that splits/filters list/float in 'cost'
#TODO: (mixed) split/filter values in 'step','param' using indices from 'cost'
#TODO: behavior is identical for 'raw_to_support'... apply changes there also

def logfile_reader(filename, iter=False):
  """read a log file (e.g. written by a LoggingMonitor) in three-column format

  iter is a bool, where if True, `iter` is also returned
  filename is a log file path, with format:

  `__iter__  __energy__  __params__`

  `iter` is a tuple of (iteration, id), with `id` an int or None
  `energy` is a float or tuple[float], and is the output of the cost function
  `params` is a list[float], and is the input to the cost function

  If iter=True, returns tuple of (iter, params, energy), as defined above.
  If iter=False, returns tuple of (params, energy).
  """
  import numpy as np
  inf, nan = np.inf, np.nan
  f = open(filename,"r")
  file = f.read()
  f.close()
  contents = file.split("\n")
  # parse file contents to get (i,id), cost, and parameters
  step = []; cost = []; param = [];
  locals_ = locals().copy()
  for line in contents[:-1]:
    if line.startswith(("#","inf =","nan =")): pass
    else:
      values = line.split("   ")
      step.append(eval(values[0], {}, locals_))  #XXX: yields (i,id)
      cost.append(eval(values[1], {}, locals_))
      param.append(eval(values[2], {}, locals_))
  return (step, param, cost) if iter else (param, cost)

def read_trajectories(source, iter=False):
  """read trajectories from a monitor instance or three-column format log file

  iter is a bool, where if True, `iter` is also returned
  source is a monitor instance or a log file path, with format:

  `__iter__  __energy__  __params__`

  `iter` is a tuple of (iteration, id), with `id` an int or None
  `energy` is a float or tuple[float], and is the output of the cost function
  `params` is a list[float], and is the input to the cost function

  If iter=True, returns tuple of (iter, params, energy), as defined above.
  If iter=False, returns tuple of (params, energy).
  """
  if isinstance(source, str):
    return logfile_reader(source, iter)
  param, cost = source.x, source.y
  if not iter:
    return param, cost
  # otherwise, id may need some processing...
  return _process_ids(source.id, len(cost)), param, cost


def _reduce_ids(ids):
    """convert ids from list of tuples of (iterations, ids) to a list of ids
    """
    if not hasattr(ids, '__len__'):
        return ids
    if len(ids) and len(ids[0]) == 1:
        ids = [None]*len(ids)
    else:
        ids = [i[-1] for i in ids]
    return ids


def _process_ids(ids, n=None): #NOTE: n only needed when ids is single value
  """convert ids to list of tuples of (iterations, ids)

  ids is a list of ints, an int, or None
  n is target length (generally used when ids is a single-value)
  """
  if ids is None:
    return [(i,) for i in range(n)] if n else None
  from numbers import Integral
  if isinstance(ids, Integral):
    return [(i,ids) for i in range(n)] if n else ids
  if not len(ids):
    return [(i,) for i in range(n)] if n else []
  # ids is [(iter,id)] where id may be None
  if len(ids) and isinstance(ids[0], tuple):
    step = ids
  else:
    step = ids[:]
    for i,j in enumerate(ids):
      step[i] = (list(zip(*step[:i]))[1].count(j), j) if i else (i,j)
  if len(ids) == ids.count(None):
    step = [(i,) for (i,j) in step]
  else:
    step = list(step)
  return step[:n]


# read and write monitor (to and from raw data)

def read_monitor(mon, id=False):
  """read trajectories from a monitor instance

  mon is a monitor instance
  id is a bool, where if True, `id` is also returned

  Returns tuple of (mon.x, mon.y) or (mon.x, mon.y, mon.id)
  """
  steps = mon.x[:]
  energy = mon.y[:]
  if not id:
    return steps, energy
  id = mon.id[:]
  return steps, energy, id 

def write_monitor(steps, energy, id=None, k=None):
  """write trajectories to a monitor instance

  `steps` is a list[float], and is the input to the cost function
  `energy` is a float or tuple[float], and is the output of the cost function
  `id` is a list[int], and is the `id` of the monitored object
  `k` is float multiplier on `energy` (e.g. k=-1 inverts the cost function)

  Returns a mystic.monitors.Monitor instance
  """
  from mystic.monitors import Monitor
  mon = Monitor()
  mon.k = k
  mon._x.extend(steps)
  mon._y.extend(mon._k(energy, iter))
  if id is not None: mon._id.extend(id)
  return mon

# converters 

def converge_to_support(steps, energy):
  steps = zip(*steps)
  steps = [list(i) for i in steps]
  return steps, energy

def raw_to_converge(steps, energy):
  if len(steps) > 0:
    if not sequence(steps[0][0]):
      steps = [[step] for step in steps]  # needed when steps = [1,2,3,...]
    steps = [list(zip(*step)) for step in steps] # also can be used to revert 'steps'
  if len(energy) > 0:
    if hasattr(energy[0], 'tolist'):
      energy = [e.tolist() for e in energy]
  return steps, energy

def raw_to_support(steps, energy):
  return converge_to_support( *raw_to_converge(steps, energy) )

# monitor to file (support file, converge file, raw file)
## FIXME: 'converge' and 'raw' files are virtually unused and unsupported

def write_raw_file(mon,log_file='paramlog.py',**kwds):
  """write parameter and solution trajectory to a log file in 'raw' format

  mon is a monitor; log_file is a str log file path, with format:

  # header
  id = ...
  params = ...
  cost = ...

  `id` is a tuple of (iteration, id), with `id` an int or None
  `params` is a list[float], and is the input to the cost function
  `cost` is a float or tuple[float], and is the output of the cost function

  if header is provided, then the given string is written as the file header
  all other kwds are written as file entries
  """
  if isNull(mon): return  #XXX: throw error? warning? ???
  steps, energy, ids = read_monitor(mon, id=True) 
  if not len(ids): #XXX: is manipulating ids a good idea?
    ids = None
  elif ids.count(ids[0]) == len(ids): #XXX: generally, all None or all ints
    ids = ids[0]
  f = open(log_file,'w')
  if 'header' in kwds:
    f.write('# %s\n' % kwds['header'])
    del kwds['header']
  f.write("inf = float('inf')\n") # define special values
  f.write("nan = float('nan')\n") # define special values
  for variable,value in kwds.items():
    f.write('%s = %s\n' % (variable,value))# write remaining kwds as variables
  if ids is not None:
    f.write('id = %s\n' % ids)
 #f.write('# %s\n' % energy[-1])
  f.write('params = %s\n' % steps)
  f.write('cost = %s\n' % energy)
  f.close()
  return

def write_support_file(mon,log_file='paramlog.py',**kwds):
  """write parameter and solution trajectory to a log file in 'support' format

  mon is a monitor; log_file is a str log file path, with format:

  # header
  id = ...
  params = ...
  cost = ...

  `id` is a tuple of (iteration, id), with `id` an int or None
  `params` is a list[float], and is the input to the cost function
  `cost` is a float or tuple[float], and is the output of the cost function

  if header is provided, then the given string is written as the file header
  all other kwds are written as file entries

  NOTE: params are the transpose of how they are stored in monitor.x
  """
  if isNull(mon): return  #XXX: throw error? warning? ???
  monitor = write_monitor( *raw_to_support( *read_monitor(mon) ) )
  monitor._id = mon._id[:] #HACK: workaround loss of id above
  monitor.k = mon.k #HACK: workaround loss of k above (ensure is copy?)
  header = "written in 'support' format"
  if 'header' in kwds:
    header += "\n# " + str(kwds['header'])
    del kwds['header']
  write_raw_file(monitor,log_file,header=header,**kwds)
  return

def write_converge_file(mon,log_file='paramlog.py',**kwds):
  if isNull(mon): return  #XXX: throw error? warning? ???
  monitor = write_monitor( *raw_to_converge( *read_monitor(mon) ) )
  monitor._id = mon._id[:] #HACK: workaround loss of id above
  monitor.k = mon.k #HACK: workaround loss of k above (ensure is copy?)
  header = "written in 'converge' format"
  if 'header' in kwds:
    header += "\n# " + str(kwds['header'])
    del kwds['header']
  write_raw_file(monitor,log_file,header=header,**kwds)
  return

# file to data (support file, converge file, raw file)

def read_raw_file(file_in, iter=False):
    """read parameter and solution trajectory log file in 'raw' format

  file_in is a str log file path, with format:

  # header
  id = ...
  params = ...
  cost = ...

  `id` is a tuple of (iteration, id), with `id` an int or None
  `params` is a list[float], and is the input to the cost function
  `cost` is a float or tuple[float], and is the output of the cost function

  if iter is True, return (id,params,cost) otherwise return (params,cost)

  NOTE: params are stored how they are stored in monitor.x
    """
    if iter:
        ids, params, cost = read_import(file_in,'id','params','cost')
        return _process_ids(ids, len(cost)), params, cost
    else:
        return read_import(file_in,'params','cost')

#TODO: check impact of having gradient ([i,j,k]) and/not cost
def read_import(file, *targets):
  "import the targets; targets are name strings"
  import re, os, sys
  _dir, file = os.path.split(file)
  file = re.sub(r'\.py*.$', '', file) #XXX: strip .py* extension
  curdir = os.path.abspath(os.curdir)
  sys.path.append('.')
  results = []
  globals = {}
  try:
    if _dir: os.chdir(_dir)
    if len(targets):
      for target in targets:
        code = "from {0} import {1} as result".format(file, target)
        code = compile(code, '<string>', 'exec')
        try:
            exec(code, globals)
        except ModuleNotFoundError:
            raise RuntimeError('Module: {0} not found'.format(file))
        except ImportError:
            globals['result'] = None #XXX: or throw error?
        results.append(globals['result'])
    else:
        code = "import {0} as result".format(file)
        code = compile(code, '<string>', 'exec')
        try:
            exec(code, globals)
        except ModuleNotFoundError:
            raise RuntimeError('Module: {0} not found'.format(file))
        except ImportError:
            globals['result'] = None #XXX: or throw error?
        results.append(globals['result'])
  except FileNotFoundError:
    raise RuntimeError('File: {0} not found'.format(file))
  finally:
    if _dir: os.chdir(curdir)
    sys.path.pop()
  if not len(results): return None
  return results[-1] if (len(results) == 1) else results

def read_converge_file(file_in, iter=False):
  data = read_raw_file(file_in, iter)
  return (data[0], raw_to_converge(*data[1:])) if iter else raw_to_converge(*data)
 #return ids, raw_to_converge(params,cost)
 #params = [zip(*param) for param in params] # alternate to revert 'params'

def read_support_file(file_in, iter=False):
  """read parameter and solution trajectory log file in 'support' format

  file_in is a str log file path, with format:

  # header
  id = ...
  params = ...
  cost = ...

  `id` is a tuple of (iteration, id), with `id` an int or None
  `params` is a list[float], and is the input to the cost function
  `cost` is a float or tuple[float], and is the output of the cost function

  if iter is True, return (id,params,cost) otherwise return (params,cost)

  NOTE: params are the transpose of how they are stored in monitor.x
  """
  data = read_raw_file(file_in, iter)
  return (data[0], raw_to_support(*data[1:])) if iter else raw_to_support(*data)
 #return ids, raw_to_support(params,cost)

# file converters

def raw_to_converge_converter(file_in,file_out):
  steps, energy = read_raw_file(file_in)
  write_raw_file(write_monitor( *raw_to_converge(steps,energy) ),
                 file_out,header="written in 'converge' format")
  return

def raw_to_support_converter(file_in,file_out):
  steps, energy = read_raw_file(file_in)
  write_raw_file(write_monitor( *raw_to_support(steps,energy) ),
                 file_out,header="written in 'support' format")
  return

def converge_to_support_converter(file_in,file_out):
  steps, energy = read_converge_file(file_in)
  write_raw_file(write_monitor( *converge_to_support(steps,energy) ),
                 file_out,header="written in 'support' format")
  return

# old #

def read_old_support_file(file_in):
  steps, energy = read_raw_file(file_in)
  steps = [[(i,) for i in steps[j]] for j in range(len(steps))]
  energy = [(i,) for i in energy]
  return write_monitor(steps, energy)

def old_to_new_support_converter(file_in,file_out):
  mon = read_old_support_file(file_in)
  write_raw_file(mon,file_out)
  return

def __orig_write_support_file(mon,log_file='paramlog.py'):
  if isNull(mon): return  #XXX: throw error? warning? ???
  steps, energy = read_monitor(mon)
  log = []
  if len(steps) > 0:
    for p in range(len(steps[0])):
      q = []
      for s in range(len(steps)):
        q.append(steps[s][p])
      log.append(q)  
  monitor = write_monitor(log, energy)
  write_raw_file(monitor,log_file)
  return

def __orig_write_converge_file(mon,log_file='paramlog.py'):
  write_raw_file(mon,log_file)
  return


if __name__ == '__main__':
  pass

# EOF

#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from mystic.tools import list_or_tuple_or_ndarray as sequence
from mystic.tools import isNull

import sys
if (sys.hexversion >= 0x30000f0):
    exec_string = 'exec(code, globals)'
else:
    exec_string = 'exec code in globals'

# generalized history reader

def read_history(source):
    """read parameter history and cost history from the given source

source is a monitor, logfile, support file, solver restart file, dataset, etc
    """
    monitor = solver = data = arxiv = False
    from mystic.monitors import Monitor, Null
    from mystic.abstract_solver import AbstractSolver
    from mystic.math.legacydata import dataset
    from klepto.archives import archive, cache
    try:
        basestring
    except NameError:
        basestring = str
        import io
        file = io.IOBase
    if isinstance(source, file):
        return read_history(source.name)
    if isinstance(source, basestring):
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
    elif isinstance(source, Null):
        return [],[] #XXX: or source.x, source.y (e.g. Null(),Null())? or Error?
    else:
        raise IOError("a history filename or instance is required")
    try:  # read standard logfile (or monitor)
        if monitor:
            params, cost = read_monitor(source)
        elif data:
            params, cost = source.coords, source.values
        elif arxiv:
            params = list(list(k) for k in source.keys())
            cost = source.values()
            cost = cost if type(cost) is list else list(cost)
        elif solver:
            params, cost = source.solution_history, source.energy_history
        else:
            try:
                from mystic.solvers import LoadSolver
                return read_history(LoadSolver(source))
            except: #KeyError
                _step, params, cost = logfile_reader(source)
                #FIXME: doesn't work for multi-id logfile; select id?
        params, cost = raw_to_support(params, cost)
    except:
        params, cost = read_import(source+'.py', 'params', 'cost')
    return params, cost


# logfile reader
#XXX: if have grad ([i,j,k]) not cost, 'cost' is list of lists [[i,j,k],...]
#XXX: if have grad and cost, 'cost' is mixed list [n,[i,j,k],m,[a,b,c],...]
#TODO: (mixed) create a function that splits/filters list/float in 'cost'
#TODO: (mixed) split/filter values in 'step','param' using indices from 'cost'
#TODO: behavior is identical for 'raw_to_support'... apply changes there also

def logfile_reader(filename):
  from numpy import inf, nan
  f = open(filename,"r")
  file = f.read()
  f.close()
  contents = file.split("\n")
  # parse file contents to get (i,id), cost, and parameters
  step = []; cost = []; param = [];
  for line in contents[:-1]:
    if line.startswith("#"): pass
    else:
      values = line.split("   ")
      step.append(eval(values[0]))  #XXX: yields (i,id)
      cost.append(eval(values[1]))
      param.append(eval(values[2]))
  return step, param, cost

def read_trajectories(source):
  """read trajectories from a convergence logfile or a monitor

source can either be a monitor instance or a logfile path
  """
  if isinstance(source, str):
    step, param, cost = logfile_reader(source)
  else:
    if source.id and isinstance(source.id[0], tuple):
      step = source.id
    else:
      step = enumerate(source.id)
    if len(source) == source.id.count(None):
      step = [(i,) for (i,j) in step]
    else:
      step = list(step)
    param, cost = source.x, source.y
  return step, param, cost


# read and write monitor (to and from raw data)

def read_monitor(mon, id=False):
  steps = mon.x[:]
  energy = mon.y[:]
  if not id:
    return steps, energy
  id = mon.id[:]
  return steps, energy, id 

def write_monitor(steps, energy, id=[], k=None):
  from mystic.monitors import Monitor
  mon = Monitor()
  mon.k = k
  mon._x.extend(steps)
  mon._y.extend(mon._k(energy, iter))
  mon._id.extend(id)
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
  return steps, energy

def raw_to_support(steps, energy):
  return converge_to_support( *raw_to_converge(steps, energy) )

# monitor to file (support file, converge file, raw file)
## FIXME: 'converge' and 'raw' files are virtually unused and unsupported

def write_raw_file(mon,log_file='paramlog.py',**kwds):
  if isNull(mon): return  #XXX: throw error? warning? ???
  steps, energy = read_monitor(mon)
  f = open(log_file,'w')
  if 'header' in kwds:
    f.write('# %s\n' % kwds['header'])
    del kwds['header']
  f.write("inf = float('inf')\n") # define special values
  f.write("nan = float('nan')\n") # define special values
  for variable,value in getattr(kwds, 'iteritems', kwds.items)():
    f.write('%s = %s\n' % (variable,value))# write remaining kwds as variables
 #f.write('# %s\n' % energy[-1])
  f.write('params = %s\n' % steps)
  f.write('cost = %s\n' % energy)
  f.close()
  return

def write_support_file(mon,log_file='paramlog.py',**kwds):
  if isNull(mon): return  #XXX: throw error? warning? ???
  monitor = write_monitor( *raw_to_support( *read_monitor(mon) ) )
  header = "written in 'support' format"
  if 'header' in kwds:
    header += "\n# " + str(kwds['header'])
    del kwds['header']
  write_raw_file(monitor,log_file,header=header,**kwds)
  return

def write_converge_file(mon,log_file='paramlog.py',**kwds):
  if isNull(mon): return  #XXX: throw error? warning? ???
  monitor = write_monitor( *raw_to_converge( *read_monitor(mon) ) )
  header = "written in 'converge' format"
  if 'header' in kwds:
    header += "\n# " + str(kwds['header'])
    del kwds['header']
  write_raw_file(monitor,log_file,header=header,**kwds)
  return

# file to data (support file, converge file, raw file)

def read_raw_file(file_in):
  steps, energy = read_import(file_in, "params", "cost")
  return steps, energy  # was 'from file_in import params as steps', etc

#TODO: check impact of having gradient ([i,j,k]) and/not cost
def_read_import = r'''
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
        %(exec_string)s
        results.append(globals['result'])
    else:
        code = "import {0} as result".format(file)
        code = compile(code, '<string>', 'exec')
        %(exec_string)s
        results.append(globals['result'])
  except ImportError:
    raise RuntimeError('File: {0} not found'.format(file))
  finally:
    if _dir: os.chdir(curdir)
    sys.path.pop()
  if not len(results): return None
  return results[-1] if (len(results) == 1) else results
''' % dict(exec_string=exec_string)

exec(def_read_import)
del def_read_import

def read_converge_file(file_in):
  steps, energy = read_raw_file(file_in)
 #steps = [zip(*step) for step in steps] # also can be used to revert 'steps'
  return raw_to_converge(steps,energy)

def read_support_file(file_in):
  steps, energy = read_raw_file(file_in)
  return raw_to_support(steps,energy)
 #return converge_to_support(steps,energy)

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


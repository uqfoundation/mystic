from mystic.tools import list_or_tuple_or_ndarray as sequence

# logfile reader

def logfile_reader(filename):
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


# read and write monitor (to and from raw data)

def read_monitor(mon, id=False):
  steps = mon.x[:]
  energy = mon.y[:]
  if not id:
    return steps, energy
  id = mon.id[:]
  return steps, energy, id 

def write_monitor(steps, energy, id=[]):
  from mystic.monitors import Monitor
  mon = Monitor()
  mon._x = steps[:]
  mon._y = energy[:]
  mon._id = id[:]
  return mon

# converters 

def converge_to_support(steps, energy):
  steps = zip(*steps)
  steps = [list(i) for i in steps]
  return steps, energy

def raw_to_converge(steps, energy):
  if not sequence(steps[0][0]):
    steps = [[step] for step in steps]  # needed when steps = [1,2,3,...]
  steps = [zip(*step) for step in steps] # also can be used to revert 'steps'
  return steps, energy

def raw_to_support(steps, energy):
  return converge_to_support( *raw_to_converge(steps, energy) )

# monitor to file (support file, converge file, raw file)
## FIXME: 'converge' and 'raw' files are virtually unused and unsupported

def write_raw_file(mon,log_file='paramlog.py',**kwds):
  steps, energy = read_monitor(mon)
  f = open(log_file,'w')
  if kwds.has_key('header'):
    f.write('# %s\n' % kwds['header'])
 #f.write('# %s\n' % energy[-1])
  f.write('params = %s' % steps)
  f.write('\ncost = %s\n' % energy)
  f.close()
  return

def write_support_file(mon,log_file='paramlog.py'):
  monitor = write_monitor( *raw_to_support( *read_monitor(mon) ) )
  write_raw_file(monitor,log_file,header="written in 'support' format")
  return

def write_converge_file(mon,log_file='paramlog.py'):
  monitor = write_monitor( *raw_to_converge( *read_monitor(mon) ) )
  write_raw_file(monitor,log_file,header="written in 'converge' format")
  return

# file to data (support file, converge file, raw file)

def read_raw_file(file_in):
  import re
  file_in = re.sub('\.py*.$', '', file_in)  #XXX: strip off .py* extension
  exec "from %s import params as steps" % file_in
  exec "from %s import cost as energy" % file_in
  return steps, energy

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
  steps, energy = read_monitor(mon)
  log = []
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


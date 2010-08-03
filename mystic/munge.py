from mystic import list_or_tuple_or_ndarray as sequence

def write_support_file(mon,log_file='paramlog.py'):
  steps = mon.x[:]
  energy = mon.y[:]
  if not sequence(steps[0][0]):
    steps = [[step] for step in steps]  # needed when steps = [1,2,3,...]
  steps = [zip(*step) for step in steps]
  steps = zip(*steps)
  steps = [list(i) for i in steps]
  f = open(log_file,'w')
 #f.write('# %s\n' % energy[-1])
  f.write('params = %s' % steps)
  f.write('\ncost = %s\n' % energy)
  f.close()
  return

def write_converge_file(mon,log_file='paramlog.py'):
  steps = mon.x[:]
  energy = mon.y[:]
  if not sequence(steps[0][0]):
    steps = [[step] for step in steps]  # needed when steps = [1,2,3,...]
  steps = [zip(*step) for step in steps] # also can be used to revert 'steps'
  f = open(log_file,'w')
 #f.write('# %s\n' % energy[-1])
  f.write('params = %s' % steps)
  f.write('\ncost = %s\n' % energy)
  f.close()
  return

def write_raw_file(mon,log_file='paramlog.py'):
  steps = mon.x[:]
  energy = mon.y[:]
  f = open(log_file,'w')
 #f.write('# %s\n' % energy[-1])
  f.write('params = %s' % steps)
  f.write('\ncost = %s\n' % energy)
  f.close()
  return

def old_to_new_support_converter(file_in,file_out):
  import re
  file_in = re.sub('\.py*.$', '', file_in)  #XXX: strip off .py* extension
  exec "from %s import params as steps" % file_in
  exec "from %s import cost as energy" % file_in
  steps = [[(i,) for i in steps[j]] for j in range(len(steps))]
  energy = [(i,) for i in energy]
  f = open(file_out,'w')
 #f.write('# %s\n' % energy[-1])
  f.write('params = %s' % steps)
  f.write('\ncost = %s\n' % energy)
  f.close()
  return

def raw_to_converge_converter(file_in,file_out):
  import re
  file_in = re.sub('\.py*.$', '', file_in)  #XXX: strip off .py* extension
  exec "from %s import params as steps" % file_in
  exec "from %s import cost as energy" % file_in
  steps = [zip(*step) for step in steps] # also can be used to revert 'steps'
  f = open(file_out,'w')
 #f.write('# %s\n' % energy[-1])
  f.write('params = %s' % steps)
  f.write('\ncost = %s\n' % energy)
  f.close()
  return

def converge_to_support_converter(file_in,file_out):
  import re
  file_in = re.sub('\.py*.$', '', file_in)  #XXX: strip off .py* extension
  exec "from %s import params as steps" % file_in
  exec "from %s import cost as energy" % file_in
  steps = zip(*steps)
  steps = [list(i) for i in steps]
  f = open(file_out,'w')
 #f.write('# %s\n' % energy[-1])
  f.write('params = %s' % steps)
  f.write('\ncost = %s\n' % energy)
  f.close()
  return

def __orig_write_converge_file(mon,log_file='paramlog.py'):
  steps = mon.x[:]
  energy = mon.y[:]
  f = open(log_file,'w')
 #f.write('# %s\n' % energy[-1])
  f.write('params = %s' % steps)
  f.write('\ncost = %s\n' % energy)
  f.close()
  return

def __orig_write_support_file(mon,log_file='paramlog.py'):
  log = []
  steps = mon.x[:]
  energy = mon.y[:]
  for p in range(len(steps[0])):
    q = []
    for s in range(len(steps)):
      q.append(steps[s][p])
    log.append(q)  
  f = open(log_file,'w')
 #f.write('# %s\n' % energy[-1])
  f.write('params = %s' % log)
  f.write('\ncost = %s\n' % energy)
  f.close()
  return


if __name__ == '__main__':
  pass

# EOF


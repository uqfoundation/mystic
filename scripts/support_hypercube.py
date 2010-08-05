#!/usr/bin/env python
__doc__ = """
generate parameter support plots from file written with 'write_support_file'

Usage: python support_hypercube.py [filename] [bounds] [xyz] [iters] [id]
    [filename] - name of the python convergence logfile (e.g paramlog.py)
    [bounds] - string indicator to set hypercube bounds (see note below)
    [xyz] - string indicator to set which params belong to which axis
    [iters] - string indicator to select iterations to plot [optional]
    [id] - select the id'th of simultaneous points to plot [optional]

The bounds should be given as a quoted list of tuples.  For example, using
bounds = "[(60,105),(0,30),(2.1,2.8)]" will set the lower and upper bounds for
x to be (60,105), y to be (0,30), and z to be (2.1,2.8).  Similarly, xyz accepts
a quoted list of tuples. However, for xyz, the first tuple indicates which
parameters are along the x direction, the second tuple for the y direction, and
the third tuple for the z direction. Hence, xyz = "[(2,3),(6,7),(10,11)]"
would set the 2nd and 3rd parameters along x. Iters is also similar, however
accepts a list of strings instead of a list of tuples. For example,
iters = "[':']" will plot all iters in a single plot. Alternatively,
iters = "[':2','2:']" will split the iters into two plots, while
iters = "['0']" will only plot the first iteration.

CAN PLOT MULTIPLE ITERS IN ONE PLOT. IGNORES WEIGHTS. 
"""

from support_convergence import best_dimensions


if __name__ == '__main__':

  import sys
  if '--help' in sys.argv:
    print __doc__
    sys.exit(0)

  try:  # get the name of the parameter log file
    file = sys.argv[1]
    import re
    file = re.sub('\.py*.$', '', file)  #XXX: strip off .py* extension
  except:
    file = 'paramlog'
  exec "from %s import params" % file
  #exec "from %s import meta" % file
  # would be nice to use meta = ['wx','wx2','x','x2','wy',...]

  try: # select the bounds
    bounds = eval(sys.argv[2])  # format is "[(60,105),(0,30),(2.1,2.8)]"
  except:
    bounds = [(None,None),(None,None),(None,None)]
    
  try: # select which params are along which axes
    xyz = eval(sys.argv[3])  # format is "[(0,1),(4,5),(8,9)]"
  except:
    xyz = [(0,),(1,),(2,)]
    
  x = params[max(xyz[0])]
  try: # select which iterations to plot
    select = eval(sys.argv[4])  # format is "[':2','2:4','5','6:']"
  except:
    select = ['-1']
   #select = [':']
   #select = [':1']
   #select = [':2','2:']
   #select = [':1','1:2','2:3','3:']
   #select = ['0','1','2','3']

  # would like to collapse non-consecutive iterations into a single plot...
  #   "['1','100','300','700','*']", with '*' indicating 'flatten'
  flatten = False
  while select.count("*"):
    flatten = True
    select.remove("*")

  try: # select which 'id' to plot results for
    id = int(sys.argv[5])
  except:
    id = None # i.e. 'all' **or** use id=0, which should be 'best' energy ?

  # ensure all terms of bounds and xyz are tuples
  for bound in bounds:
    if not isinstance(bound, tuple):
      raise TypeError, "bounds should be tuples of (lower_bound,upper_bound)"
  for i in range(len(xyz)):
    if isinstance(xyz[i], int):
      xyz[i] = (xyz[i],)
    elif not isinstance(xyz[i], tuple):
      raise TypeError, "xyz should be tuples of (param1,param2,param3,...)"

  # ensure all terms of select are strings that have a ":"
  for i in range(len(select)):
    if isinstance(select[i], int): select[i] = str(select[i])
    if select[i] == '-1': select[i] = 'len(x)-1:len(x)'
    elif not select[i].count(':'):
      select[i] += ':' + str(int(select[i])+1)

  # take only the selected 'id'
  if id != None:
    param = []
    for j in range(len(params)):
      param.append([p[id] for p in params[j]])
    params = param[:]

  # at this point, we should have:
  #bounds = [(60,105),(0,30),(2.1,2.8)] or [(None,None),(None,None),(None,None)]
  #xyz = [(0,1),(4,5),(8,9)] for any length tuple
  #select = ['-1:'] or [':'] or [':1','1:2','2:3','3:'] or similar
  #id = 0 or None

  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.pyplot as plt
  from matplotlib.axes import subplot_class_factory
  Subplot3D = subplot_class_factory(Axes3D)

  plots = len(select)
  if not flatten:
    dim1,dim2 = best_dimensions(plots)
  else: dim1,dim2 = 1,1

  # declare bounds 'invalid' (i.e. ignore them) if any member is "None"
  invalid = [None in [i[j] for i in bounds] for j in range(len(bounds[0]))]

  # correctly bound the first plot.  there must be at least one plot
  fig = plt.figure()
  ax1 = Subplot3D(fig, dim1,dim2,1)
  if not invalid[0]: ax1.plot([bounds[0][0]],[bounds[1][0]],[bounds[2][0]])
  if not invalid[1]: ax1.plot([bounds[0][1]],[bounds[1][1]],[bounds[2][1]])
  if not flatten:
    exec "plt.title('iterations[%s]')" % select[0]
  else: 
    exec "plt.title('iterations[*]')"
  ax1.set_xlabel('x')
  ax1.set_ylabel('y')
  ax1.set_zlabel('z')
  a = [ax1]

  # set up additional plots
  if not flatten:
    for i in range(2, plots + 1):
      exec "ax%d = Subplot3D(fig, dim1,dim2,%d)" % (i,i)
      if not invalid[0]:
        exec "ax%d.plot([bounds[0][0]],[bounds[1][0]],[bounds[2][0]])" % i
      if not invalid[1]:
        exec "ax%d.plot([bounds[0][1]],[bounds[1][1]],[bounds[2][1]])" % i
      exec "plt.title('iterations[%s]')" % select[i - 1]
      exec "ax%d.set_xlabel('x')" % i
      exec "ax%d.set_ylabel('y')" % i
      exec "ax%d.set_zlabel('z')" % i
      exec "a.append(ax%d)" % i

  # turn each "n:m" in select to a list
  _select = []
  for sel in select:
    if sel[0] == ':': _select.append("0"+sel)
    else: _select.append(sel)
  for i in range(len(_select)):
    if _select[i][-1] == ':': select[i] = _select[i]+str(len(x))
    else: select[i] = _select[i]
  for i in range(len(select)):
    p = select[i].split(":")
    if p[0][0] == '-': p[0] = "len(x)"+p[0]
    if p[1][0] == '-': p[1] = "len(x)"+p[1]
    select[i] = p[0]+":"+p[1]
  steps = [eval("range(%s)" % sel.replace(":",",")) for sel in select]

  # at this point, we should have:
  #xyz = [(0,1),(4,5),(8,9)] for any length tuple
  #steps = [[0,1],[1,2],[2,3],[3,4,5,6,7,8]] or similar
  if flatten:
    from mystic import flatten
    steps = [list(flatten(steps))]

  # build all the plots
  from numpy import inf
  for v in range(len(steps)):
    if len(steps[v]) > 1: qp = float(max(steps[v]))
    else: qp = inf 
    for s in steps[v]:
      t = str(s/qp) # dot color determined by number of simultaneous iterations
      for i in eval("[params[q][%s] for q in xyz[0]]" % s):
        for j in eval("[params[q][%s] for q in xyz[1]]" % s):
          for k in eval("[params[q][%s] for q in xyz[2]]" % s):
            a[v].plot(i,j,k,marker='o',color=t,ms=10)

  plt.show()

# EOF

#!/usr/bin/env python
__doc__ = """
generate parameter support plots from file written with 'write_support_file'

Usage: python support_hypercube2.py [filename] [bounds] [xyz] [wxyz] [iters] [id]
    [filename] - name of the python convergence logfile (e.g paramlog.py)
    [bounds] - string indicator to set hypercube bounds (see note below)
    [xyz] - string indicator to set which params belong to which axis
    [wxyz] - string indicator to set which weights belong to which params
    [iters] - string indicator to select iterations to plot [optional]
    [id] - select the id'th of simultaneous points to plot [optional]

The bounds should be given as a quoted list of tuples.  For example, using
bounds = "[(60,105),(0,30),(2.1,2.8)]" will set the lower and upper bounds for
x to be (60,105), y to be (0,30), and z to be (2.1,2.8).  Similarly, xyz accepts
a quoted list of tuples. However, for xyz, the first tuple indicates which
parameters are along the x direction, the second tuple for the y direction,
and the third tuple for the z direction. Hence, xyz = "[(2,3),(6,7),(10,11)]"
would set the 2nd and 3rd parameters along x. The corresponding weights are
given with wxyz, where wxyz = "[(0,1),(4,5),(8,9)] would set the 0th and 1st
parameters along x.  Iters is also similar, however only accept a list of ints. 
Hence, iters = "[-1]" will plot the last iteration, while iters = "[0,300,700]" 
will plot the 0th, 300th, and 700th in three plots.

CAN PLOT ONE ITER PER PLOT. UTILIZES WEIGHTS.
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
    xyz = [(1,),(3,),(5,)]
    
  try: # select which params are along which axes
    wxyz = eval(sys.argv[4])  # format is "[(0,1),(4,5),(8,9)]"
  except:
    wxyz = [(0,),(2,),(4,)]
    
  x = params[max(xyz[0])]
  try: # select which iterations to plot
    select = eval(sys.argv[5])  # format is "[2,4,5,6]"
  except:
    select = ['-1']
   #select = ['0','1','2','3']

  # would like to collapse non-consecutive iterations into a single plot...
  #   "['1','100','300','700','*']", with '*' indicating 'flatten'
  flatten = False
  while select.count("*"):
    flatten = True
    select.remove("*")

  try: # select which 'id' to plot results for
    id = int(sys.argv[6])
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
  for i in range(len(wxyz)):
    if isinstance(wxyz[i], int):
      wxyz[i] = (wxyz[i],)
    elif not isinstance(wxyz[i], tuple):
      raise TypeError, "wxyz should be tuples of (param1,param2,param3,...)"

  # ensure all terms of select are strings of ints
  for i in range(len(select)):
    if isinstance(select[i], int): select[i] = str(select[i])
    if select[i].count(':'):
      raise ValueError, "iters should be ints"
   #if select[i] == '-1': select[i] = 'len(x)-1:len(x)'
   #elif not select[i].count(':'):
   #  select[i] += ':' + str(int(select[i])+1)

  # take only the selected 'id'
  if id != None:
    param = []
    for j in range(len(params)):
      param.append([p[id] for p in params[j]])
    params = param[:]

  # at this point, we should have:
  #bounds = [(60,105),(0,30),(2.1,2.8)] or [(None,None),(None,None),(None,None)]
  #xyz = [(2,3),(6,7),(10,11)] for any length tuple
  #wxyz = [(0,1),(4,5),(8,9)] for any length tuple (should match up with xyz)
  #select = ['-1'] or ['1','2','3','-1'] or similar
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
 #_select = []
 #for sel in select:
 #  if sel[0] == ':': _select.append("0"+sel)
 #  else: _select.append(sel)
 #for i in range(len(_select)):
 #  if _select[i][-1] == ':': select[i] = _select[i]+str(len(x))
 #  else: select[i] = _select[i]
 #for i in range(len(select)):
 #  p = select[i].split(":")
 #  if p[0][0] == '-': p[0] = "len(x)"+p[0]
 #  if p[1][0] == '-': p[1] = "len(x)"+p[1]
 #  select[i] = p[0]+":"+p[1]
 #steps = [eval("range(%s)" % sel.replace(":",",")) for sel in select]
  steps = [eval("[int(%s)]" % sel) for sel in select]

  # at this point, we should have:
  #xyz = [(2,3),(6,7),(10,11)] for any length tuple
  #wxyz = [(0,1),(4,5),(8,9)] for any length tuple (should match up with xyz)
  #steps = [[0],[1],[3],[8]] or similar
  if flatten:
    from mystic.tools import flatten
    steps = [list(flatten(steps))]

  # dot color is based on a product of weights
  t = []
  for v in range(len(steps)):
    t.append([])
    for s in steps[v]:
      for i in eval("[params[q][%s] for q in wxyz[0]]" % s):
        for j in eval("[params[q][%s] for q in wxyz[1]]" % s):
          for k in eval("[params[q][%s] for q in wxyz[2]]" % s):
            t[v].append([str(i[q]*j[q]*k[q]) for q in range(len(i))])

  # build all the plots
  for v in range(len(steps)):
    for s in steps[v]:
      u = 0
      for i in eval("[params[q][%s] for q in xyz[0]]" % s):
        for j in eval("[params[q][%s] for q in xyz[1]]" % s):
          for k in eval("[params[q][%s] for q in xyz[2]]" % s):
            for q in range(len(t[v][u])):
              a[v].plot([i[q]],[j[q]],[k[q]],marker='o',color=t[v][u][q],ms=10)
            u += 1

  plt.show()

# EOF

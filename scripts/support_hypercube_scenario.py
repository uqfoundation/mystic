#!/usr/bin/env python
__doc__ = """
support_hypercube_scenario.py [options] filename (datafile)

generate scenario support plots from file written with 'write_support_file';
generate legacy data and cones from a dataset file, if provided

The options "bounds", "dim", and "iters" all take indicator strings.
The bounds should be given as a quoted list of tuples.  For example, using
bounds = "[(.062,.125),(0,30),(2300,3200)]" will set the lower and upper bounds
for x to be (.062,.125), y to be (0,30), and z to be (2300,3200). I bounds
are to not be strictly enforced, append an asterisk '*' to the bounds. 
The dim (dimensions of the scenario) should be a quoted tuple.  For example,
dim = "(1,1,2)" will convert the params to a two-member 3-D dataset. Iters
takes a list of strings instead of a tuple or a list of tuples. For example,
iters = "[':']" will plot all iters in a single plot. Alternatively,
iters = "[':2','2:']" will split the iters into two plots, while
iters = "['0']" will only plot the first iteration.

The option "label" takes a list of strings. For example, label = "['x','y','']"
will place 'x' on the x-axis, 'y' on the y-axis, and nothing on the z-axis.
LaTeX is also accepted. For example, label = "[r'$ h $',r'$ {\alpha}$',r'$ v$']"
will label the axes with standard LaTeX math formatting. Note that the leading
space is required, while the trailing space aligns the text with the axis
instead of the plot frame.  The option "filter" is used to select datapoints
from a given dataset, and takes a quoted list.  A "mask" can be given as an
integer or a tuple of integers; when the mask is a tuple, the plot will be 2D.  
The option "vertical" will plot the dataset values on the vertical axis; for
2D plots, cones are always plotted on the vertical axis.

Required Inputs:
  filename            name of the python convergence logfile (e.g. paramlog.py)

Additional Inputs:
  datafile            name of the dataset textfile (e.g. StAlDataset.txt)
"""
ZERO = 1.0e-6  # zero-ish

#### building the cone primitive ####
def cone_builder(slope, bounds, strict=True):
  """ factory to create a cone primitive

  slope -- slope multiplier for cone on the X,Y,Z axes (for mesh construction)
  bounds -- list of tuples of bounds for the plot; (lower,upper) for each axis
"""
  from mystic.math import almostEqual
  import numpy as np
  def cone_mesh(length):
    """ construct a conical mesh for a given length of cone """
    L1,L2,L3 = slope
    radius = length / L3 #XXX: * 0.5
    r0 = ZERO

    if almostEqual(radius, r0, tol=r0): radius = r0
    r = np.linspace(radius,radius,6) 
    r[0]= np.zeros(r[0].shape) 
    r[1] *= r0/radius
    r[5] *= r0/radius
    r[3]= np.zeros(r[3].shape) 

    p = np.linspace(0,2*np.pi,50) 
    R,P = np.meshgrid(r,p) 
    X,Y = L1 * R*np.cos(P), L2 * R*np.sin(P) 

    tmp=list() 
    for i in range(np.size(p)): 
      tmp.append([0,0,length,length,length,0]) # len = size(r)
    Z = np.array(tmp) 
    return X,Z,Y

  lb,ub = zip(*bounds)
  # if False, the cone surface may violate bounds
 #strict = True # always respect the bounds

  def cone(position, top=True):
    """ construct a cone primitive, with vertex at given position

    position -- tuple of vertex position
    top -- boolean for if 'top' or 'bottom' cone
"""
    from numpy import asarray
    position = asarray(position)
    d_hi = ub - position
    d_lo = lb - position
    if strict and (any(d_hi < 0) or any(d_lo > 0)):
      return None  # don't plot cone if vertex is out of bounds

    if top: # distance of vertex from upper edge
      l = d_hi[2]
      if not l: l = ZERO
    else:   # distance of vertex from lower edge
      l = d_lo[2]
      if not l: l = -ZERO
    X,Z,Y = cone_mesh(l)
    X = X + position[0]
    Y = Y + position[1]
    Z = Z + position[2]
    return X,Z,Y

  return cone    


#### plotting the cones ####
def plot_bowtie(ax, data, slope, bounds, color='0.75', axis=None):
  if axis not in range(len(bounds)-1): return ax
  from numpy import asarray, inf
  data = asarray(data)
  sl = slope[axis]
  ylo = sl * (bounds[0][0] - data.T[0]) + data.T[1]
  yhi = sl * (bounds[0][1] - data.T[0]) + data.T[1]
  zhi = -sl * (bounds[0][0] - data.T[0]) + data.T[1]
  zlo = -sl * (bounds[0][1] - data.T[0]) + data.T[1]
  xlb = bounds[0][0]
  xub = bounds[0][1]
  ylb = bounds[1][0]
  yub = bounds[1][1]

  for pt,yl,yu,zl,zu in zip(data,ylo,yhi,zlo,zhi):
   #ax.plot([xlb,pt[0],xub], [yl,pt[1],yu], color='black')
   #ax.plot([xlb,pt[0],xub], [zu,pt[1],zl], color='black')
    ax.fill_between([xlb,pt[0],xub], [ylb]*3, [yl,pt[1],zl], \
                                     facecolor=color, alpha=0.2)
    ax.fill_between([xlb,pt[0],xub], [yub]*3, [zu,pt[1],yu], \
                                     facecolor=color, alpha=0.2)
  return ax

def plot_cones(ax, data, slope, bounds, color='0.75', axis=None, strict=True):
  """plot double cones for a given dataset

  ax -- matplotlib 'Axes3D' plot object
  data -- list of datapoints, where datapoints are 3-tuples (i.e. x,y,z)
  slope -- slope multiplier for cone on the X,Y,Z axes (for mesh construction)
  bounds -- list of tuples of bounds for the plot; (lower,upper) for each axis
  color -- string name (or rbg value) of color to use for datapoints
  axis -- the axis of the cone
"""
  # adjust slope, bounds, and data so cone axis is last 
  slope = swap(slope, axis) 
  bounds = swap(bounds, axis) 
  data = [swap(pt,axis) for pt in data]
  cone = cone_builder(slope, bounds, strict=strict)
  # plot the upper and lower cone
  for datapt in data:
    _cone = cone(datapt, top=True)
    if _cone:
      X,Z,Y = swap(_cone, axis) # 'unswap' the axes
      ax.plot_surface(X, Y,Z, rstride=1, cstride=1, color=color, \
                                         linewidths=0, alpha=.1)
    _cone = cone(datapt, top=False)
    if _cone:
      X,Z,Y = swap(_cone, axis) # 'unswap' the axes
      ax.plot_surface(X, Y,Z, rstride=1, cstride=1, color=color, \
                                         linewidths=0, alpha=.1)
  return ax

def plot_data(ax, data, bounds, color='red', strict=True):
  """plot datapoints for a given dataset

  ax -- matplotlib 'Axes3D' plot object
  data -- list of datapoints, where datapoints are 3-tuples (i.e. x,y,z)
  bounds -- list of tuples of bounds for the plot; (lower,upper) for each axis
  color -- string name (or rbg value) of color to use for datapoints
"""
# strict = True # always respect the bounds
  lb,ub = zip(*bounds)
  # plot the datapoints themselves
  from numpy import asarray
  for datapt in data:
    position = asarray(datapt)
    d_hi = ub - position
    d_lo = lb - position
    if strict and (any(d_hi < 0) or any(d_lo > 0)): #XXX: any or just in Y ?
      pass  # don't plot if datapt is out of bounds
    else:
      if _2D:
        ax.plot([datapt[0]], [datapt[1]], \
                color=color, marker='o', markersize=10)
      else:
        ax.plot([datapt[0]], [datapt[1]], [datapt[2]], \
                color=color, marker='o', markersize=10)
  return ax

def clip_axes(ax, bounds):
  """ clip plots to be within given lower and upper bounds

  ax -- matplotlib 'Axes3D' plot object
  bounds -- list of tuples of bounds for the plot; (lower,upper) for each axis
"""
  lb,ub = zip(*bounds)
  # plot only within [lb,ub]
  ax.set_xlim3d(lb[0], ub[0])
  ax.set_ylim3d(lb[1], ub[1])
  ax.set_zlim3d(lb[2], ub[2]) # cone "center" axis
  return ax

def label_axes(ax, labels):
  """ label plots with given string labels

  ax -- matplotlib 'Axes3D' plot object
  labels -- list of string labels for the plot
"""
  ax.set_xlabel(labels[0])
  ax.set_ylabel(labels[1])
  if not _2D:
    ax.set_zlabel(labels[2]) # cone "center" axis
  return ax

def get_slope(data, replace=None, mask=None):
  """ replace one slope in a list of slopes with '1.0'
  (i.e. replace a slope from the dataset with the unit slope)

  data -- dataset object, where coordinates are 3-tuples and values are floats
  replace -- selected axis (an int) to plot values NOT coords
  """
  slope = data.lipschitz
  if mask in range(len(slope)):
    slope = swap(slope, mask)
  if replace not in range(len(slope)):  # don't replace an axis
    return slope
  return slope[:replace] + [1.0] + slope[replace+1:]

def get_coords(data, replace=None, mask=None):
  """ replace one coordiate axis in a 3-D data set with 'data values'
  (i.e. replace an 'x' axis with the 'y' values of the data)

  data -- dataset object, where coordinates are 3-tuples and values are floats
  replace -- selected axis (an int) to plot values NOT coords
  """
  slope = data.lipschitz
  coords = data.coords
  values = data.values
  if mask in range(len(slope)):
    coords = [swap(pt,mask) for pt in coords]
  if replace not in range(len(slope)):  # don't replace an axis
    return coords
  return [list(coords[i][:replace]) + [values[i]] + \
          list(coords[i][replace+1:]) for i in range(len(coords))]

def swap(alist, index=None):
  """ swap the selected list element with the last element in alist

  alist -- a list of objects
  index -- the selected element
  """
  if index not in range(len(alist)):  # don't swap an element
    return alist 
  return alist[:index] + alist[index+1:] + alist[index:index+1]

from support_convergence import best_dimensions


if __name__ == '__main__':
  #XXX: note that 'argparse' is new as of python2.7
  from optparse import OptionParser
  parser = OptionParser(usage=__doc__)
  parser.add_option("-b","--bounds",action="store",dest="bounds",\
                    metavar="STR",default="[(0,1),(0,1),(0,1)]",
                    help="indicator string to set hypercube bounds")
  parser.add_option("-i","--iters",action="store",dest="iters",\
                    metavar="STR",default="['-1']",
                    help="indicator string to select iterations to plot")
  parser.add_option("-l","--label",action="store",dest="label",\
                    metavar="STR",default="['','','']",
                    help="string to assign label to axis")
  parser.add_option("-d","--dim",action="store",dest="dim",\
                    metavar="STR",default="None",
                    help="indicator string set to scenario dimensions")
  parser.add_option("-p","--filter",action="store",dest="filter",\
                    metavar="STR",default="None",
                    help="filter to select datapoints to plot")
  parser.add_option("-m","--mask",action="store",dest="replace",\
                    metavar="INT",default=None,
                    help="id # of the coordinate axis not to be plotted")
  parser.add_option("-n","--nid",action="store",dest="id",\
                    metavar="INT",default=None,
                    help="id # of the nth simultaneous points to plot")
  parser.add_option("-s","--scale",action="store",dest="scale",\
                    metavar="INT",default=1.0,
                    help="grayscale contrast multiplier for points in plot")
  parser.add_option("-o","--data",action="store_true",dest="legacy",\
                    default=False,help="plot legacy data, if provided")
  parser.add_option("-v","--cones",action="store_true",dest="cones",\
                    default=False,help="plot cones, if provided")
  parser.add_option("-z","--vertical",action="store_true",dest="vertical",\
                    default=False,help="always plot values on vertical axis")
  parser.add_option("-f","--flat",action="store_true",dest="flatten",\
                    default=False,help="show selected iterations in a single plot")
  parsed_opts, parsed_args = parser.parse_args()

  try:  # get the name of the parameter log file
    file = parsed_args[0]
    import re
    file = re.sub('\.py*.$', '', file)  #XXX: strip off .py* extension
  except:
    raise IOError, "please provide log file name"
  try:  # read standard logfile
    from mystic.munge import logfile_reader, raw_to_support
    _step, params, _cost = logfile_reader(file)
    params, _cost = raw_to_support(params, _cost)
  except:
    exec "from %s import params" % file
    #exec "from %s import meta" % file
    # would be nice to use meta = ['wx','wx2','x','x2','wy',...]

  from mystic.math.dirac_measure import scenario
  from mystic.math.legacydata import dataset
  try: # select whether to plot the cones
    cones = parsed_opts.cones
  except:
    cones = False

  try: # select whether to plot the legacy data
    legacy = parsed_opts.legacy
  except:
    legacy = False

  try: # get dataset filter
    filter = eval(parsed_opts.filter) # should be an iterable
  except:
    filter = None

  try: # select the scenario dimensions
    npts = eval(parsed_opts.dim)  # format is "(1,1,1)"
    if npts is None: # npts may have been logged
      exec "from %s import npts" % file
  except:
    npts = (1,1,1) #XXX: better in parsed_args ?

  try: # get the name of the dataset file
    file = parsed_args[1]
    from mystic.math.legacydata import load_dataset
    data = load_dataset(file, filter)
  except:
#   raise IOError, "please provide dataset file name"
    data = dataset()
    cones = False
    legacy = False

  try: # select the bounds
    _bounds = parsed_opts.bounds
    if _bounds[-1] == "*": #XXX: then bounds are NOT strictly enforced
      _bounds = _bounds[:-1]
      strict = False
    else:
      strict = True
    bounds = eval(_bounds)  # format is "[(60,105),(0,30),(2.1,2.8)]"
  except:
    strict = True
    bounds = [(0,1),(0,1),(0,1)]

  try: # select labels for the axes
    label = eval(parsed_opts.label)  # format is "['x','y','z']"
  except:
    label = ['','','']

  x = params[-1]
  try: # select which iterations to plot
    select = eval(parsed_opts.iters)  # format is "[':2','2:4','5','6:']"
  except:
    select = ['-1']
   #select = [':']
   #select = [':1']
   #select = [':2','2:']
   #select = [':1','1:2','2:3','3:']
   #select = ['0','1','2','3']

  try: # collapse non-consecutive iterations into a single plot...
    flatten = parsed_opts.flatten
  except:
    flatten = False

  try: # select which 'id' to plot results for
    id = int(parsed_opts.id)
  except:
    id = None # i.e. 'all' **or** use id=0, which should be 'best' energy ?

  try: # scale the color in plotting the weights
    scale = float(parsed_opts.scale)
  except:
    scale = 1.0 # color = color**scale

  _2D = False # if False, use 3D plots; if True, use 3D plots
  cs = None
  try: # select which axis to plot 'values' on  (3D plot)
    xs = int(parsed_opts.replace)
  except:
    try: # select which axes to mask (2D plot)
      xs = eval(parsed_opts.replace)  # format is "(1,2)"
      xs = list(reversed(sorted(set(xs))))
      cs = int(xs[-1]) if xs[-1] != xs[0] else None
      xs = int(xs[0])
      xs,cs = cs,xs # cs will swap coord axes; xs will swap with value axis
      _2D = True #NOTE: always apply cs swap before xs swap
    except:
      xs = None # don't plot values; plot values on 'x' axis with xs = 0

  try: # always plot cones on vertical axis
    vertical_cones = parsed_opts.vertical
  except:
    vertical_cones = False
  if _2D: # always plot cones on vertical axis
    vertical_cones = True

  # ensure all terms of bounds are tuples
  for bound in bounds:
    if not isinstance(bound, tuple):
      raise TypeError, "bounds should be tuples of (lower_bound,upper_bound)"

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
  #select = ['-1:'] or [':'] or [':1','1:2','2:3','3:'] or similar
  #id = 0 or None

  # get dataset coords (and values) for selected axes
  if data:
    slope = get_slope(data, xs, cs)
    coords = get_coords(data, xs, cs)
    #print "bounds: %s" % bounds
    #print "slope: %s" % slope
    #print "coords: %s" % coords
 #else:
 #  slope = []
 #  coords = []

  import matplotlib.pyplot as plt
  if not _2D:
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.axes import subplot_class_factory
    Subplot3D = subplot_class_factory(Axes3D)

  plots = len(select)
  if not flatten:
    dim1,dim2 = best_dimensions(plots)
  else: dim1,dim2 = 1,1

  # use the default bounds where not specified
  bounds = [list(i) for i in bounds]
  for i in range(len(bounds)):
    if bounds[i][0] is None: bounds[i][0] = 0
    if bounds[i][1] is None: bounds[i][1] = 1

  # swap the axes appropriately when plotting cones (when replacing an axis)
  if _2D and xs == 0:
    if data:
      slope[0],slope[1] = slope[1],slope[0]
      coords = [list(reversed(pt[:2]))+pt[2:] for pt in coords]
   #bounds[0],bounds[1] = bounds[1],bounds[0]
   #label[0],label[1] = label[1],label[0]
    axis = xs #None
  elif not _2D and vertical_cones and xs in range(len(bounds)):
    # adjust slope, bounds, and data so cone axis is last 
    if data:
      slope = swap(slope, xs) 
      coords = [swap(pt,xs) for pt in coords]
    bounds = swap(bounds, xs) 
    label = swap(label, xs)
    axis = None
  else:
    axis = xs

  # correctly bound the first plot.  there must be at least one plot
  fig = plt.figure() 
  if _2D:
    ax1 = fig.add_subplot(dim1,dim2,1)
    ax1.plot([bounds[0][0]],[bounds[1][0]])
    ax1.plot([bounds[0][1]],[bounds[1][1]])
  else:
    ax1 = Subplot3D(fig, dim1,dim2,1)
    ax1.plot([bounds[0][0]],[bounds[1][0]],[bounds[2][0]])
    ax1.plot([bounds[0][1]],[bounds[1][1]],[bounds[2][1]])
  if not flatten:
    exec "plt.title('iterations[%s]')" % select[0]
  else:
    exec "plt.title('iterations[*]')"
  if cones and data and xs in range(len(bounds)):
    if _2D:
      plot_bowtie(ax1,coords,slope,bounds,axis=axis)
    else:
      plot_cones(ax1,coords,slope,bounds,axis=axis,strict=strict)
  if legacy and data:
    plot_data(ax1,coords,bounds,strict=strict)
 #clip_axes(ax1,bounds)
  label_axes(ax1,label)
  a = [ax1]

  # set up additional plots
  if not flatten:
    for i in range(2, plots + 1):
      if _2D:
        exec "ax%d = fig.add_subplot(dim1,dim2,%d)" % (i,i)
        exec "ax%d.plot([bounds[0][0]],[bounds[1][0]])" % i
        exec "ax%d.plot([bounds[0][1]],[bounds[1][1]])" % i
      else:
        exec "ax%d = Subplot3D(fig, dim1,dim2,%d)" % (i,i)
        exec "ax%d.plot([bounds[0][0]],[bounds[1][0]],[bounds[2][0]])" % i
        exec "ax%d.plot([bounds[0][1]],[bounds[1][1]],[bounds[2][1]])" % i
      exec "plt.title('iterations[%s]')" % select[i - 1]
      if cones and data and xs in range(len(bounds)):
        if _2D:
          exec "plot_bowtie(ax%d,coords,slope,bounds,axis=axis)" % i
        else:
          exec "plot_cones(ax%d,coords,slope,bounds,axis=axis,strict=strict)" % i
      if legacy and data:
        exec "plot_data(ax%d,coords,bounds,strict=strict)" % i
     #exec "clip_axes(ax%d,bounds)" % i
      exec "label_axes(ax%d,label)" % i
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
  #steps = [[0,1],[1,2],[2,3],[3,4,5,6,7,8]] or similar
  if flatten:
    from mystic.tools import flatten
    steps = [list(flatten(steps))]

  # plot all the scenario "data"
  from numpy import inf, e
  scale = e**(scale - 1.0)
  for v in range(len(steps)):
    if len(steps[v]) > 1: qp = float(max(steps[v]))
    else: qp = inf
    for s in steps[v]:
      par = eval("[params[q][%s][0] for q in range(len(params))]" % s)
      pm = scenario()
      pm.load(par, npts)
      d = dataset()
      d.load(pm.coords, pm.values)
      # dot color determined by number of simultaneous iterations
      t = str((s/qp)**scale)
      # get and plot dataset coords for selected axes      
      _coords = get_coords(d, xs, cs)
      # check if we are replacing an axis
      if _2D and xs == 0:
        if data: # adjust data so cone axis is last 
          _coords = [list(reversed(pt[:2]))+pt[2:] for pt in _coords]
      elif not _2D and vertical_cones and xs in range(len(bounds)):
        if data: # adjust data so cone axis is last 
          _coords = [swap(pt,xs) for pt in _coords]
      plot_data(a[v], _coords, bounds, color=t, strict=strict)

  if _2D: # strict = True
    plt.xlim((bounds[0][0],bounds[0][1]))
    plt.ylim((bounds[1][0],bounds[1][1]))
  plt.show() 

# EOF

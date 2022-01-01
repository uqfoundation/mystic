#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
__doc__ = """
functional interfaces for mystic's visual diagnistics for support files
"""

__all__ = ['convergence', 'hypercube', 'hypercube_measures', \
           'hypercube_scenario', 'best_dimensions', 'swap']

# from mpl_toolkits.mplot3d import axes3d

# globals
__quit = False
ZERO = 1.0e-6  # zero-ish
import sys
if (sys.hexversion >= 0x30000f0):
    PY3 = True
    exec_string = 'exec(code, globals)'
else:
    PY3 = False
    exec_string = 'exec code in globals'
NL = '\n'
NL2 = '\n\n'
#FIXME: remove this head-standing to workaround python2.6 exec bug

def best_dimensions(n):
  """get the 'best' dimensions ``(i x j)`` for arranging plots

Args:
  n (int): number of plots

Returns:
  tuple ``(i,j)`` of ``i`` rows ``j`` columns, where ``i*j`` is roughly ``n``
  """
  from mystic.tools import factor
  allfactors = list(factor(n))
  from numpy import product
  cand = [1] + [product(allfactors[:i+1]) for i in range(len(allfactors))]
 #return cand[-1], n/cand[-1]
  best = [cand[len(cand)//2], n//cand[len(cand)//2]]
  best.sort(reverse=True)
  return tuple(best)
# if len(cand)%2:
#   return cand[len(cand)/2], cand[len(cand)/2]
# return cand[len(cand)/2], cand[len(cand)/2 - 1]


#### building the cone primitive ####
def _cone_builder(slope, bounds, strict=True):
  """factory to create a cone primitive

  slope -- slope multiplier for cone on the X,Y,Z axes (for mesh construction)
  bounds -- list of tuples of bounds for the plot; (lower,upper) for each axis
"""
  from mystic.math import almostEqual
  import numpy as np
  def cone_mesh(length):
    """construct a conical mesh for a given length of cone"""
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
def _plot_bowtie(ax, data, slope, bounds, color='0.75', axis=None, tol=0.0):
  """plot (2D) double cones for a given dataset

  ax -- matplotlib 'Axes3D' plot object
  data -- list of datapoints, where datapoints are 3-tuples (i.e. x,y,z)
  slope -- slope multiplier for cone on the X,Y,Z axes (for mesh construction)
  bounds -- list of tuples of bounds for the plot; (lower,upper) for each axis
  color -- string name (or rbg value) of color to use for datapoints
  axis -- the axis of the cone
  tol -- distance between center of mass of the double cones and a cone vertex
"""
  if axis not in range(len(bounds)-1): return ax
  from numpy import asarray, inf
  data = asarray(data)
  sl = slope[axis]
  gap = max(0., tol)
  # find the endpoints of the cone:  y = slope * (x0 - x) + y0
  ylo = sl * (bounds[0][0] - data.T[0]) + data.T[1]
  yhi = sl * (bounds[0][1] - data.T[0]) + data.T[1]
  zhi = -sl * (bounds[0][0] - data.T[0]) + data.T[1]
  zlo = -sl * (bounds[0][1] - data.T[0]) + data.T[1]
  xlb = bounds[0][0]
  xub = bounds[0][1]
  ylb = bounds[1][0]
  yub = bounds[1][1]

  for pt,yl,yu,zl,zu in zip(data,ylo,yhi,zlo,zhi):
   #ax.plot([xlb, pt[0], xub], [yl, pt[1], yu], color='black')
   #ax.plot([xlb, pt[0], xub], [zu, pt[1], zl], color='black')
    ax.fill_between([xlb, pt[0], xub], [ylb]*3, [yl-gap, pt[1]-gap, zl-gap], \
                                     facecolor=color, alpha=0.2)
    ax.fill_between([xlb, pt[0], xub], [yub]*3, [zu+gap, pt[1]+gap, yu+gap], \
                                     facecolor=color, alpha=0.2)
  return ax


def _plot_cones(ax, data, slope, bounds, color='0.75', axis=None, strict=True, tol=0.0):
  """plot (3D) double cones for a given dataset

  ax -- matplotlib 'Axes3D' plot object
  data -- list of datapoints, where datapoints are 3-tuples (i.e. x,y,z)
  slope -- slope multiplier for cone on the X,Y,Z axes (for mesh construction)
  bounds -- list of tuples of bounds for the plot; (lower,upper) for each axis
  color -- string name (or rbg value) of color to use for datapoints
  axis -- the axis of the cone
  tol -- distance between center of mass of the double cones and a cone vertex
"""
  from numpy import asarray, zeros
  # adjust slope, bounds, and data so cone axis is last 
  slope = swap(slope, axis) 
  bounds = swap(bounds, axis) 
  data = [swap(pt,axis) for pt in data]
  cone = _cone_builder(slope, bounds, strict=strict)
  # prepare to apply the 'gap' tolerance
  gap = zeros(len(slope))
  gap[-1:] = [max(0., tol)] #XXX: bad behavior if len(slope) is 0
  # plot the upper and lower cone
  for datapt in data:
    datapt = asarray(datapt)
    _cone = cone(datapt + gap, top=True)
    if _cone:
      X,Z,Y = swap(_cone, axis) # 'unswap' the axes
      ax.plot_surface(X, Y,Z, rstride=1, cstride=1, color=color, \
                                         linewidths=0, alpha=.1)
    _cone = cone(datapt - gap, top=False)
    if _cone:
      X,Z,Y = swap(_cone, axis) # 'unswap' the axes
      ax.plot_surface(X, Y,Z, rstride=1, cstride=1, color=color, \
                                         linewidths=0, alpha=.1)
  return ax


def _plot_data(ax, data, bounds, color='red', strict=True, **kwds):
  """plot datapoints for a given dataset

  ax -- matplotlib 'Axes3D' plot object
  data -- list of datapoints, where datapoints are 3-tuples (i.e. x,y,z)
  bounds -- list of tuples of bounds for the plot; (lower,upper) for each axis
  color -- string name (or rbg value) of color to use for datapoints
"""
  _2D = kwds.get('_2D', False)
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


def _clip_axes(ax, bounds):
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


def _label_axes(ax, labels, **kwds):
  """ label plots with given string labels

  ax -- matplotlib 'Axes3D' plot object
  labels -- list of string labels for the plot
"""
  _2D = kwds.get('_2D', False)
  ax.set_xlabel(labels[0])
  ax.set_ylabel(labels[1])
  if not _2D:
    ax.set_zlabel(labels[2]) # cone "center" axis
  return ax


def _get_slope(data, replace=None, mask=None):
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


def _get_coords(data, replace=None, mask=None):
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
  """swap the selected list element with the last element in alist

Args:
  alist (list): a list of objects
  index (int, default=None): the selected element

Returns:
  list with the elements swapped as indicated
  """
  if index not in range(len(alist)):  # don't swap an element
    return alist 
  return alist[:index] + alist[index+1:] + alist[index:index+1]


def_convergence = '''
def convergence(filename, **kwds):
    """
generate parameter convergence plots from file written with ``write_support_file``

Available from the command shell as::

    support_convergence filename [options]

or as a function call::

    mystic.support.convergence(filename, **options)

Args:
    filename (str): name of the convergence logfile (e.g ``paramlog.py``)

Returns:
    None

Notes:
    - The option *out* takes a string of the filepath for the generated plot.
      If ``out = True``, return the Figure object instead of generating a plot.
    - The option *iter* takes an integer of the largest iteration to plot.
    - The option *param* takes an indicator string. The indicator string is
      built from comma-separated array slices. For example, ``param = ":"``
      will plot all parameters in a single plot.  Alternatively,
      ``param = ":2, 2:"`` will split the parameters into two plots, and
      ``param = "0"`` will only plot the first parameter.
    - The option *label* takes comma-separated strings. For example,
      ``label = "x,y,"`` will label the y-axis of the first plot with 'x', a
      second plot with 'y', and not add a label to a third or subsequent plots.      If more labels are given than plots, then the last label will be used
      for the y-axis of the 'cost' plot. LaTeX is also accepted. For example,
      ``label = "$ h$, $ a$, $ v$"`` will label the axes with standard
      LaTeX math formatting. Note that the leading space is required, and the
      text is aligned along the axis.
    - The option *nid* takes an integer of the nth simultaneous points to plot.
    - The option *cost* takes a boolean, and will also plot the parameter cost.
    - The option *legend* takes a boolean, and will display the legend.
"""
    import shlex
    try:
        basestring
        from StringIO import StringIO
    except NameError:
        basestring = str
        from io import StringIO
    global __quit
    __quit = False
    _out = False

    instance = None
    # handle the special case where list is provided by sys.argv
    if isinstance(filename, (list,tuple)) and not kwds:
        cmdargs = filename # (above is used by script to parse command line)
    elif isinstance(filename, basestring) and not kwds:
        cmdargs = shlex.split(filename)
    # 'everything else' is essentially the functional interface
    else:
        cmdargs = kwds.get('kwds', '')
        if not cmdargs:
            out = kwds.get('out', None)
            iter = kwds.get('iter', None)
            param = kwds.get('param', None)
            label = kwds.get('label', None)
            nid = kwds.get('nid', None)
            cost = kwds.get('cost', False)
            legend = kwds.get('legend', False)

            if isinstance(out, bool): _out, out = out, None

            # process "commandline" arguments
            cmdargs = ''
            cmdargs += '' if out is None else '--out={} '.format(out)
            cmdargs += '' if iter is None else '--iter={} '.format(iter)
            cmdargs += '' if param is None else '--param="{}" '.format(param)
            cmdargs += '' if label is None else '--label="{}" '.format(label)
            cmdargs += '' if nid is None else '--nid={} '.format(nid)
            cmdargs += '' if cost == False else '--cost '
            cmdargs += '' if legend == False else '--legend '
        else:
            cmdargs = ' ' + cmdargs
        if isinstance(filename, basestring):
            cmdargs = filename.split() + shlex.split(cmdargs)
        else: # special case of passing in monitor instance
            instance = filename
            cmdargs = shlex.split(cmdargs)

    #XXX: note that 'argparse' is new as of python2.7
    from optparse import OptionParser
    def _exit(self, errno=None, msg=None):
        global __quit
        __quit = True
        if errno or msg:
            msg = msg.split(': error: ')[-1].strip()
            raise IOError(msg)
    OptionParser.exit = _exit

    parser = OptionParser(usage=convergence.__doc__.split(NL2+'Options:')[0])
    parser.add_option("-u","--out",action="store",dest="out",\
                      metavar="STR",default=None,
                      help="filepath to save generated plot")
    parser.add_option("-i","--iter",action="store",dest="step",metavar="INT",\
                      default=None,help="the largest iteration to plot")
    parser.add_option("-p","--param",action="store",dest="param",\
                      metavar="STR",default=":",
                      help="indicator string to select parameters")
    parser.add_option("-l","--label",action="store",dest="label",\
                      metavar="STR",default="",
                      help="string to assign label to y-axis")
    parser.add_option("-n","--nid",action="store",dest="id",\
                      metavar="INT",default=None,
                      help="id # of the nth simultaneous points to plot")
    parser.add_option("-c","--cost",action="store_true",dest="cost",\
                      default=False,help="also plot the parameter cost")
    parser.add_option("-g","--legend",action="store_true",dest="legend",\
                      default=False,help="show the legend")

    if PY3:
        f = StringIO()
        parser.print_help(file=f)
        f.seek(0)
        if 'Options:' not in convergence.__doc__:
            convergence.__doc__ += NL+'Options:{0}'.format(f.read().split('Options:')[-1])
        f.close()
    else:
        if 'Options:' not in convergence.__doc__:
            convergence.__doc__ += NL+'Options:{0}'.format(parser.format_help().split('Options:')[-1])

    try:
        parsed_opts, parsed_args = parser.parse_args(cmdargs)
    except UnboundLocalError:
        pass
    if __quit: return

    # get the name of the parameter log file
    if instance is None:
        instance = parsed_args[0]
    from mystic.munge import read_history
    params, cost = read_history(instance)

    if parsed_opts.cost: # also plot the cost
       #exec "from {file} import cost".format(file=file)
        pass
    else:
        cost = None

    try: # path of plot output file
      out = parsed_opts.out  # e.g. 'myplot.png'
      if "None" == out: out = None
    except:
      out = None

    if parsed_opts.legend: # show the legend
        legend = True
    else:
        legend = False

    try: # select which iteration to stop plotting at
        step = int(parsed_opts.step)
    except:
        step = None

    try: # select which parameters to plot
        select = parsed_opts.param.split(',')  # format is ":2, 2:4, 5, 6:"
    except:
        select = [':']
       #select = [':1']
       #select = [':2','2:']
       #select = [':1','1:2','2:3','3:']
       #select = ['0','1','2','3']
    plots = len(select)

    try: # select labels for the axes
        label = parsed_opts.label.split(',')  # format is "x, y, z"
        label += [''] * max(0, plots - len(label))
    except:
        label = [''] * plots

    try: # select which 'id' to plot results for
        id = int(parsed_opts.id)
    except:
        id = None # i.e. 'all' **or** use id=0, which should be 'best' energy ?

    # ensure all terms of select have a ":"
    for i in range(plots):
        if isinstance(select[i], int): select[i] = str(select[i])
        if select[i] == '-1': select[i] = 'len(params)-1:len(params)'
        elif not select[i].count(':'):
            select[i] += ':' + str(int(select[i])+1)

    # take only the first 'step' iterations
    params = [var[:step] for var in params]
    if cost:
        cost = cost[:step]

    # take only the selected 'id'
    if id != None:
        param = []
        for j in range(len(params)):
            param.append([p[id] for p in params[j]])
        params = param[:]

    # handle special case where only plot the cost
    if cost and len(select) == 1 and select[0].endswith(':0'):
        dim1,dim2 = best_dimensions(1)

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(dim1,dim2,1)
        ax1.set_ylabel(label[-1])
        ax1.plot(cost,label='cost')
        if legend: plt.legend()
        cost = None
    else:
        if cost: j = 1
        else: j = 0
        dim1,dim2 = best_dimensions(plots + j)

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(dim1,dim2,1)
        ax1.set_ylabel(label[0])
        locals = dict(params=params)
        data = eval("params[{0}]".format(select[0]), locals)
        try:
            n = int(select[0].split(":")[0])
        except ValueError:
            n = 0
        for line in data:
            ax1.plot(line,label=str(n))#, marker='o')
            n += 1
        if legend: plt.legend()

    globals = {'fig':fig, 'dim1':dim1, 'dim2':dim2, 'ax1':ax1, 'label':label}
    for i in range(2, plots + 1):
        code = "ax{0:d} = fig.add_subplot(dim1,dim2,{0:d}, sharex=ax1);".format(i)
        code += "ax{0:d}.set_ylabel(label[{1:d}]);".format(i,i-1)
        code = compile(code, '<string>', 'exec')
        %(exec_string)s
        data = eval("params[{0}]".format(select[i-1]), locals)
        try:
            n = int(select[i-1].split(":")[0])
        except ValueError:
            n = 0
        for line in data:
            globals['line'] = line
            code = "ax{0:d}.plot(line,label='{1}')#, marker='o')".format(i,n)
            code = compile(code, '<string>', 'exec')
            %(exec_string)s
            n += 1
        if legend: plt.legend()
    if cost:
        globals['cost'] = cost
        code = "cx1 = fig.add_subplot(dim1,dim2,{0:d}, sharex=ax1);".format(int(plots+1))
        code += "cx1.plot(cost,label='cost');"#, marker='o')"
        if max(0, len(label) - plots): code += "cx1.set_ylabel(label[-1]);"
        code = compile(code, '<string>', 'exec')
        %(exec_string)s
        if legend: plt.legend()

    # process inputs
    if _out: out = _out
    if out: out = _out or out

    if not out:
        plt.show()
    elif out is True:
        return fig #XXX: better?: fig.axes if len(fig.axes) > 1 else ax
    else:
        fig.savefig(out)
''' % dict(exec_string=exec_string)

    ### USUAL WAY OF CREATING PLOTS ###
    #fig = plt.figure()
    #ax1 = fig.add_subplot(3,2,1)
    ##ax1.ylim(60,105)
    #ax1.plot(x)
    #ax1.plot(x2)
    #plt.title('convergence for thickness support')
    ##plt.xlabel('iterations')
    #plt.ylabel('thickness')
    #
    #ax2 = fig.add_subplot(3,2,2, sharex=ax1)
    ##ax2.ylim(0,1)
    #ax2.plot(wx)
    #ax2.plot(wx2)
    #plt.title('convergence for weight(thickness)')
    ##plt.xlabel('iterations')
    #plt.ylabel('weight')
    #
    #plt.show()
    ###################################


def_hypercube = '''
def hypercube(filename, **kwds):
    """
generate parameter support plots from file written with ``write_support_file``

Available from the command shell as::

    support_hypercube filename [options]

or as a function call::

    mystic.support.hypercube(filename, **options)

Args:
    filename (str): name of the convergence logfile (e.g ``paramlog.py``)

Returns:
    None

Notes:
    - The option *out* takes a string of the filepath for the generated plot.
      If ``out = True``, return the Figure object instead of generating a plot.
    - The options *bounds*, *axes*, and *iters* all take indicator strings.
      The bounds should be given as comma-separated slices. For example, using
      ``bounds = "60:105, 0:30, 2.1:2.8"`` will set the lower and upper bounds
      for x to be (60,105), y to be (0,30), and z to be (2.1,2.8). Similarly,
      axes also accepts comma-separated groups of ints; however, for axes, each      entry indicates which parameters are to be plotted along each axis -- the
      first group for the x direction, the second for the y direction, and
      third for z. Thus, ``axes = "2 3, 6 7, 10 11"`` would set 2nd and 3rd
      parameters along x. Iters also accepts strings built from comma-separated      array slices. For example, ``iters = ":"`` will plot all iters in a
      single plot. Alternatively, ``iters = ":2, 2:"`` will split the iters
      into two plots, while ``iters = "0"`` will only plot the first iteration.
    - The option *label* takes comma-separated strings. Thus ``label = "x,y,"``      will place 'x' on the x-axis, 'y' on the y-axis, and nothing on the
      z-axis. LaTeX, such as ``label = "$ h $, $ a$, $ v$"`` will label the
      axes with standard LaTeX math formatting. Note that the leading space is
      required, while a trailing space aligns the text with the axis instead of      the plot frame.
    - The option *nid* takes an integer of the nth simultaneous points to plot.
    - The option *scale* takes an integer as a grayscale contrast multiplier.
    - The option *flat* takes a boolean, to plot results in a single plot.
"""
    import shlex
    try:
        basestring
        from StringIO import StringIO
    except NameError:
        basestring = str
        from io import StringIO
    global __quit
    __quit = False
    _out = False

    instance = None
    # handle the special case where list is provided by sys.argv
    if isinstance(filename, (list,tuple)) and not kwds:
        cmdargs = filename # (above is used by script to parse command line)
    elif isinstance(filename, basestring) and not kwds:
        cmdargs = shlex.split(filename)
    # 'everything else' is essentially the functional interface
    else:
        cmdargs = kwds.get('kwds', '')
        if not cmdargs:
            out = kwds.get('out', None)
            bounds = kwds.get('bounds', None)
            axes = kwds.get('axes', None)
            iters = kwds.get('iters', None)
            label = kwds.get('label', None)
            nid = kwds.get('nid', None)
            scale = kwds.get('scale', None)
            flat = kwds.get('flat', False)

            if isinstance(out, bool): _out, out = out, None

            # process "commandline" arguments
            cmdargs = ''
            cmdargs += '' if out is None else '--out={} '.format(out)
            cmdargs += '' if bounds is None else '--bounds="{}" '.format(bounds)
            cmdargs += '' if axes is None else '--axes="{}" '.format(axes)
            cmdargs += '' if iters is None else '--iters="{}" '.format(iters)
            cmdargs += '' if label is None else '--label="{}" '.format(label)
            cmdargs += '' if nid is None else '--nid={} '.format(nid)
            cmdargs += '' if scale is None else '--scale={} '.format(scale)
            cmdargs += '' if flat == False else '--flat '
        else:
            cmdargs = ' ' + cmdargs
        if isinstance(filename, basestring):
            cmdargs = filename.split() + shlex.split(cmdargs)
        else: # special case of passing in monitor instance
            instance = filename
            cmdargs = shlex.split(cmdargs)

    #XXX: note that 'argparse' is new as of python2.7
    from optparse import OptionParser
    def _exit(self, errno=None, msg=None):
        global __quit
        __quit = True
        if errno or msg:
            msg = msg.split(': error: ')[-1].strip()
            raise IOError(msg)
    OptionParser.exit = _exit

    parser = OptionParser(usage=hypercube.__doc__.split(NL2+'Options:')[0])
    parser.add_option("-u","--out",action="store",dest="out",\
                      metavar="STR",default=None,
                      help="filepath to save generated plot")
    parser.add_option("-b","--bounds",action="store",dest="bounds",\
                      metavar="STR",default="0:1, 0:1, 0:1",
                      help="indicator string to set hypercube bounds")
    parser.add_option("-x","--axes",action="store",dest="xyz",\
                      metavar="STR",default="0, 1, 2",
                      help="indicator string to assign parameter to axis")
    parser.add_option("-i","--iters",action="store",dest="iters",\
                      metavar="STR",default="-1",
                      help="indicator string to select iterations to plot")
    parser.add_option("-l","--label",action="store",dest="label",\
                      metavar="STR",default=",,",
                      help="string to assign label to axis")
    parser.add_option("-n","--nid",action="store",dest="id",\
                      metavar="INT",default=None,
                      help="id # of the nth simultaneous points to plot")
    parser.add_option("-s","--scale",action="store",dest="scale",\
                      metavar="INT",default=1.0,
                      help="grayscale contrast multiplier for points in plot")
    parser.add_option("-f","--flat",action="store_true",dest="flatten",\
                      default=False,help="show selected iterations in a single plot")

    if PY3:
        f = StringIO()
        parser.print_help(file=f)
        f.seek(0)
        if 'Options:' not in hypercube.__doc__:
            hypercube.__doc__ += NL+'Options:{0}'.format(f.read().split('Options:')[-1])
        f.close()
    else:
        if 'Options:' not in hypercube.__doc__:
            hypercube.__doc__ += NL+'Options:{0}'.format(parser.format_help().split('Options:')[-1])

    try:
        parsed_opts, parsed_args = parser.parse_args(cmdargs)
    except UnboundLocalError:
        pass
    if __quit: return

    # get the name of the parameter log file
    if instance is None:
        instance = parsed_args[0]
    from mystic.munge import read_history
    params, _cost = read_history(instance)
    # would be nice to use meta = ['wx','wx2','x','x2','wy',...]
    # exec "from {file} import meta".format(file=file)

    try: # select the bounds
        bounds = parsed_opts.bounds.split(",")  # format is "60:105, 0:30, 2.1:2.8"
        bounds = [tuple(float(j) for j in i.split(':')) for i in bounds]
    except:
        bounds = [(0,1),(0,1),(0,1)]

    try: # select which params are along which axes
        xyz = parsed_opts.xyz.split(",")  # format is "0 1, 4 5, 8 9"
        xyz = [tuple(int(j) for j in i.split()) for i in xyz]
    except:
        xyz = [(0,),(1,),(2,)]

    try: # select labels for the axes
        label = parsed_opts.label.split(',')  # format is "x, y, z"
    except:
        label = ['','','']

    x = params[max(xyz[0])]
    try: # select which iterations to plot
        select = parsed_opts.iters.split(',')  # format is ":2, 2:4, 5, 6:"
    except:
        select = ['-1']
       #select = [':']
       #select = [':1']
       #select = [':2','2:']
       #select = [':1','1:2','2:3','3:']
       #select = ['0','1','2','3']

    try: # path of plot output file
      out = parsed_opts.out  # e.g. 'myplot.png'
      if "None" == out: out = None
    except:
      out = None

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

    # ensure all terms of bounds and xyz are tuples
    for bound in bounds:
        if not isinstance(bound, tuple):
            raise TypeError("bounds should be tuples of (lower_bound,upper_bound)")
    for i in range(len(xyz)):
        if isinstance(xyz[i], int):
            xyz[i] = (xyz[i],)
        elif not isinstance(xyz[i], tuple):
            raise TypeError("xyz should be tuples of (param1,param2,param3,...)")

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
    #bounds = [(60,105),(0,30),(2.1,2.8)] or [(None,None)]*3
    #xyz = [(0,1),(4,5),(8,9)] for any length tuple
    #select = ['-1:'] or [':'] or [':1','1:2','2:3','3:'] or similar
    #id = 0 or None

    plots = len(select)
    if not flatten:
        dim1,dim2 = best_dimensions(plots)
    else: dim1,dim2 = 1,1

    # use the default bounds where not specified
    bounds = [list(i) for i in bounds]
    for i in range(len(bounds)):
        if bounds[i][0] is None: bounds[i][0] = 0
        if bounds[i][1] is None: bounds[i][1] = 1

    # correctly bound the first plot.  there must be at least one plot
    import matplotlib.pyplot as plt
    fig = plt.figure()
    try:
        Subplot3D = None
        ax1 = fig.add_subplot(dim1, dim2, 1, projection='3d')
        newsubplotcode = "ax = ax{0:d} = fig.add_subplot(dim1,dim2,{0:d}, projection='3d');"
    except:
        from mpl_toolkits.mplot3d import Axes3D as _Axes3D
        from matplotlib.axes import subplot_class_factory
        Subplot3D = subplot_class_factory(_Axes3D)
        ax1 = Subplot3D(fig, dim1,dim2,1)
        newsubplotcode = "ax = ax{0:d} = Subplot3D(fig, dim1,dim2,{0:d});"
    ax1.plot([bounds[0][0]],[bounds[1][0]],[bounds[2][0]])
    ax1.plot([bounds[0][1]],[bounds[1][1]],[bounds[2][1]])
    globals = {'plt':plt,'fig':fig,'dim1':dim1,'dim2':dim2,\
               'Subplot3D':Subplot3D,'bounds':bounds,'label':label}
    if not flatten:
        code = "plt.title('iterations[{0}]');".format(select[0])
    else: 
        code = "plt.title('iterations[*]');"
    code = compile(code, '<string>', 'exec')
    %(exec_string)s
    ax1.set_xlabel(label[0])
    ax1.set_ylabel(label[1])
    ax1.set_zlabel(label[2])
    a = [ax1]

    # set up additional plots
    if not flatten:
        for i in range(2, plots + 1):
            code = newsubplotcode.format(i)
            code += "ax.plot([bounds[0][0]],[bounds[1][0]],[bounds[2][0]]);"
            code += "ax.plot([bounds[0][1]],[bounds[1][1]],[bounds[2][1]]);"
            code += "plt.title('iterations[{0}]');".format(select[i - 1])
            code += "ax.set_xlabel(label[0]);"
            code += "ax.set_ylabel(label[1]);"
            code += "ax.set_zlabel(label[2]);"
            code = compile(code, '<string>', 'exec')
            %(exec_string)s
            a.append(globals['ax'])

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
    locals = dict(x=x, params=params, xyz=xyz)
    steps = [eval("list(range({0}))".format(sel.replace(":",",")), locals) for sel in select]

    # at this point, we should have:
    #xyz = [(0,1),(4,5),(8,9)] for any length tuple
    #steps = [[0,1],[1,2],[2,3],[3,4,5,6,7,8]] or similar
    if flatten:
        from mystic.tools import flatten
        steps = [list(flatten(steps))]

    # build all the plots
    from numpy import inf, e
    scale = e**(scale - 1.0)
    for v in range(len(steps)):
        if len(steps[v]) > 1: qp = float(max(steps[v]))
        else: qp = inf 
        for s in steps[v]:
            # dot color determined by number of simultaneous iterations
            t = str((s/qp)**scale)
            for i in eval("[params[q][{0}] for q in xyz[0]]".format(s), locals):
                for j in eval("[params[q][{0}] for q in xyz[1]]".format(s), locals):
                    for k in eval("[params[q][{0}] for q in xyz[2]]".format(s), locals):
                        a[v].plot(i,j,k,marker='o',color=t,ms=10)

    # process inputs
    if _out: out = _out
    if out: out = _out or out

    if not out:
        plt.show()
    elif out is True:
        return fig #XXX: better?: fig.axes if len(fig.axes) > 1 else ax
    else:
        fig.savefig(out)
''' % dict(exec_string=exec_string)


def_hypercube_measures = '''
def hypercube_measures(filename, **kwds):
    """
generate measure support plots from file written with ``write_support_file``

Available from the command shell as::

    support_hypercube_measures filename [options]

or as a function call::

    mystic.support.hypercube_measures(filename, **options)

Args:
    filename (str): name of the convergence logfile (e.g ``paramlog.py``)

Returns:
    None

Notes:
    - The option *out* takes a string of the filepath for the generated plot.
      If ``out = True``, return the Figure object instead of generating a plot.
    - The options *bounds*, *axes*, *weight*, and *iters* all take indicator
      strings. The bounds should be given as comma-separated slices. For
      example, using ``bounds = "60:105, 0:30, 2.1:2.8"`` will set lower and
      upper bounds for x to be (60,105), y to be (0,30), and z to be (2.1,2.8).      Similarly, axes also accepts comma-separated groups of ints; however, for      axes, each entry indicates which parameters are to be plotted along each
      axis -- the first group for the x direction, the second for the y
      direction, and third for z. Thus, ``axes = "2 3, 6 7, 10 11"`` would set
      2nd and 3rd parameters along x. The corresponding weights are used to
      color the measure points, where 1.0 is black and 0.0 is white. For
      example, using ``weight = "0 1, 4 5, 8 9"`` would use the 0th and 1st
      parameters to weight x. Iters is also similar, however only accepts
      comma-separated ints. Hence, ``iters = "-1"`` will plot the last
      iteration, while ``iters = "0, 300, 700"`` will plot the 0th, 300th, and
      700th in three plots.
    - The option *label* takes comma-separated strings. Thus ``label = "x,y,"``      will place 'x' on the x-axis, 'y' on the y-axis, and nothing on the
      z-axis. LaTeX, such as ``label = "$ h $, $ a$, $ v$"`` will label the
      axes with standard LaTeX math formatting. Note that the leading space is
      required, while a trailing space aligns the text with the axis instead of      the plot frame.
    - The option *nid* takes an integer of the nth simultaneous points to plot.
    - The option *scale* takes an integer as a grayscale contrast multiplier.
    - The option *flat* takes a boolean, to plot results in a single plot.

Warning:
    This function is intended to visualize weighted measures (i.e. weights
    and positions), where the weights must be normalized (to 1) or an error
    will be thrown.
"""
    import shlex
    try:
        basestring
        from StringIO import StringIO
    except NameError:
        basestring = str
        from io import StringIO
    global __quit
    __quit = False
    _out = False

    instance = None
    # handle the special case where list is provided by sys.argv
    if isinstance(filename, (list,tuple)) and not kwds:
        cmdargs = filename # (above is used by script to parse command line)
    elif isinstance(filename, basestring) and not kwds:
        cmdargs = shlex.split(filename)
    # 'everything else' is essentially the functional interface
    else:
        cmdargs = kwds.get('kwds', '')
        if not cmdargs:
            out = kwds.get('out', None)
            bounds = kwds.get('bounds', None)
            axes = kwds.get('axes', None)
            weight = kwds.get('weight', None)
            iters = kwds.get('iters', None)
            label = kwds.get('label', None)
            nid = kwds.get('nid', None)
            scale = kwds.get('scale', None)
            flat = kwds.get('flat', False)

            if isinstance(out, bool): _out, out = out, None

            # process "commandline" arguments
            cmdargs = ''
            cmdargs += '' if out is None else '--out={} '.format(out)
            cmdargs += '' if bounds is None else '--bounds="{}" '.format(bounds)
            cmdargs += '' if axes is None else '--axes="{}" '.format(axes)
            cmdargs += '' if weight is None else '--weight="{}" '.format(weight)
            cmdargs += '' if iters is None else '--iters="{}" '.format(iters)
            cmdargs += '' if label is None else '--label="{}" '.format(label)
            cmdargs += '' if nid is None else '--nid={} '.format(nid)
            cmdargs += '' if scale is None else '--scale={} '.format(scale)
            cmdargs += '' if flat == False else '--flat '
        else:
            cmdargs = ' ' + cmdargs
        if isinstance(filename, basestring):
            cmdargs = filename.split() + shlex.split(cmdargs)
        else: # special case of passing in monitor instance
            instance = filename
            cmdargs = shlex.split(cmdargs)

    #XXX: note that 'argparse' is new as of python2.7
    from optparse import OptionParser
    def _exit(self, errno=None, msg=None):
        global __quit
        __quit = True
        if errno or msg:
            msg = msg.split(': error: ')[-1].strip()
            raise IOError(msg)
    OptionParser.exit = _exit

    parser = OptionParser(usage=hypercube_measures.__doc__.split(NL2+'Options:')[0])
    parser.add_option("-u","--out",action="store",dest="out",\
                      metavar="STR",default=None,
                      help="filepath to save generated plot")
    parser.add_option("-b","--bounds",action="store",dest="bounds",\
                      metavar="STR",default="0:1, 0:1, 0:1",
                      help="indicator string to set hypercube bounds")
    parser.add_option("-x","--axes",action="store",dest="xyz",\
                      metavar="STR",default="1, 3, 5",
                      help="indicator string to assign spatial parameter to axis")
    parser.add_option("-w","--weight",action="store",dest="wxyz",\
                      metavar="STR",default="0, 2, 4",
                      help="indicator string to assign weight parameter to axis")
    parser.add_option("-i","--iters",action="store",dest="iters",\
                      metavar="STR",default="-1",
                      help="indicator string to select iterations to plot")
    parser.add_option("-l","--label",action="store",dest="label",\
                      metavar="STR",default=",,",
                      help="string to assign label to axis")
    parser.add_option("-n","--nid",action="store",dest="id",\
                      metavar="INT",default=None,
                      help="id # of the nth simultaneous points to plot")
    parser.add_option("-s","--scale",action="store",dest="scale",\
                      metavar="INT",default=1.0,
                      help="grayscale contrast multiplier for points in plot")
    parser.add_option("-f","--flat",action="store_true",dest="flatten",\
                      default=False,help="show selected iterations in a single plot")

    if PY3:
        f = StringIO()
        parser.print_help(file=f)
        f.seek(0)
        if 'Options:' not in hypercube_measures.__doc__:
            hypercube_measures.__doc__ += NL+'Options:{0}'.format(f.read().split('Options:')[-1])
        f.close()
    else:
        if 'Options:' not in hypercube_measures.__doc__:
            hypercube_measures.__doc__ += NL+'Options:{0}'.format(parser.format_help().split('Options:')[-1])

    try:
        parsed_opts, parsed_args = parser.parse_args(cmdargs)
    except UnboundLocalError:
        pass
    if __quit: return

    # get the name of the parameter log file
    if instance is None:
        instance = parsed_args[0]
    from mystic.munge import read_history
    params, _cost = read_history(instance)
    # would be nice to use meta = ['wx','wx2','x','x2','wy',...]
    # exec "from {file} import meta".format(file=file)

    try: # path of plot output file
      out = parsed_opts.out  # e.g. 'myplot.png'
      if "None" == out: out = None
    except:
      out = None

    try: # select the bounds
        bounds = parsed_opts.bounds.split(",")  # format is "60:105, 0:30, 2.1:2.8"
        bounds = [tuple(float(j) for j in i.split(':')) for i in bounds]
    except:
        bounds = [(0,1),(0,1),(0,1)]

    try: # select which params are along which axes
        xyz = parsed_opts.xyz.split(",")  # format is "0 1, 4 5, 8 9"
        xyz = [tuple(int(j) for j in i.split()) for i in xyz]
    except:
        xyz = [(1,),(3,),(5,)]

    try: # select which params are along which axes
        wxyz = parsed_opts.wxyz.split(",")  # format is "0 1, 4 5, 8 9"
        wxyz = [tuple(int(j) for j in i.split()) for i in wxyz]
    except:
        wxyz = [(0,),(2,),(4,)]

    try: # select labels for the axes
        label = parsed_opts.label.split(',')  # format is "x, y, z"
    except:
        label = ['','','']

    x = params[max(xyz[0])]
    try: # select which iterations to plot
        select = parsed_opts.iters.split(',')  # format is "2, 4, 5, 6"
    except:
        select = ['-1']
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

    # ensure all terms of bounds and xyz are tuples
    for bound in bounds:
        if not isinstance(bound, tuple):
            raise TypeError("bounds should be tuples of (lower_bound,upper_bound)")
    for i in range(len(xyz)):
        if isinstance(xyz[i], int):
            xyz[i] = (xyz[i],)
        elif not isinstance(xyz[i], tuple):
            raise TypeError("xyz should be tuples of (param1,param2,param3,...)")
    for i in range(len(wxyz)):
        if isinstance(wxyz[i], int):
            wxyz[i] = (wxyz[i],)
        elif not isinstance(wxyz[i], tuple):
            raise TypeError("wxyz should be tuples of (param1,param2,param3,...)")

    # ensure all terms of select are strings of ints
    for i in range(len(select)):
        if isinstance(select[i], int): select[i] = str(select[i])
        if select[i].count(':'):
            raise ValueError("iters should be ints")
       #if select[i] == '-1': select[i] = 'len(x)-1:len(x)'
       #elif not select[i].count(':'):
       #    select[i] += ':' + str(int(select[i])+1)

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

    plots = len(select)
    if not flatten:
        dim1,dim2 = best_dimensions(plots)
    else: dim1,dim2 = 1,1

    # use the default bounds where not specified
    bounds = [list(i) for i in bounds]
    for i in range(len(bounds)):
        if bounds[i][0] is None: bounds[i][0] = 0
        if bounds[i][1] is None: bounds[i][1] = 1

    # correctly bound the first plot.  there must be at least one plot
    import matplotlib.pyplot as plt
    fig = plt.figure()
    try:
        Subplot3D = None
        ax1 = fig.add_subplot(dim1, dim2, 1, projection='3d')
        newsubplotcode = "ax = ax{0:d} = fig.add_subplot(dim1,dim2,{0:d}, projection='3d');"
    except:
        from mpl_toolkits.mplot3d import Axes3D as _Axes3D
        from matplotlib.axes import subplot_class_factory
        Subplot3D = subplot_class_factory(_Axes3D)
        ax1 = Subplot3D(fig, dim1,dim2,1)
        newsubplotcode = "ax = ax{0:d} = Subplot3D(fig, dim1,dim2,{0:d});"
    ax1.plot([bounds[0][0]],[bounds[1][0]],[bounds[2][0]])
    ax1.plot([bounds[0][1]],[bounds[1][1]],[bounds[2][1]])
    globals = {'plt':plt,'fig':fig,'dim1':dim1,'dim2':dim2,\
               'Subplot3D':Subplot3D,'bounds':bounds,'label':label}
    if not flatten:
        code = "plt.title('iterations[{0}]');".format(select[0])
    else: 
        code = "plt.title('iterations[*]');"
    code = compile(code, '<string>', 'exec')
    %(exec_string)s
    ax1.set_xlabel(label[0])
    ax1.set_ylabel(label[1])
    ax1.set_zlabel(label[2])
    a = [ax1]

    # set up additional plots
    if not flatten:
        for i in range(2, plots + 1):
            code = newsubplotcode.format(i)
            code += "ax.plot([bounds[0][0]],[bounds[1][0]],[bounds[2][0]]);"
            code += "ax.plot([bounds[0][1]],[bounds[1][1]],[bounds[2][1]]);"
            code += "plt.title('iterations[{0}]');".format(select[i - 1])
            code += "ax.set_xlabel(label[0]);"
            code += "ax.set_ylabel(label[1]);"
            code += "ax.set_zlabel(label[2]);"
            code = compile(code, '<string>', 'exec')
            %(exec_string)s
            a.append(globals['ax'])

    # turn each "n:m" in select to a list
   #_select = []
   #for sel in select:
   #    if sel[0] == ':': _select.append("0"+sel)
   #    else: _select.append(sel)
   #for i in range(len(_select)):
   #    if _select[i][-1] == ':': select[i] = _select[i]+str(len(x))
   #    else: select[i] = _select[i]
   #for i in range(len(select)):
   #    p = select[i].split(":")
   #    if p[0][0] == '-': p[0] = "len(x)"+p[0]
   #    if p[1][0] == '-': p[1] = "len(x)"+p[1]
   #    select[i] = p[0]+":"+p[1]
   #steps = [eval("range({0})".format(sel.replace(":",","))) for sel in select]
    locals = dict(x=x, params=params, wxyz=wxyz, xyz=xyz)
    steps = [eval("[int({0})]".format(sel), locals) for sel in select]

    # at this point, we should have:
    #xyz = [(2,3),(6,7),(10,11)] for any length tuple
    #wxyz = [(0,1),(4,5),(8,9)] for any length tuple (should match up with xyz)
    #steps = [[0],[1],[3],[8]] or similar
    if flatten:
        from mystic.tools import flatten
        steps = [list(flatten(steps))]

    # adjust for logarithmic scaling of intensity
    from numpy import e
    scale = e**(scale - 1.0)

    # dot color is based on a product of weights
    t = []
    for v in range(len(steps)):
        t.append([])
        for s in steps[v]:
            for i in eval("[params[q][{0}] for q in wxyz[0]]".format(s), locals):
                for j in eval("[params[q][{0}] for q in wxyz[1]]".format(s), locals):
                    for k in eval("[params[q][{0}] for q in wxyz[2]]".format(s), locals):
                        t[v].append([str((1.0 - i[q]*j[q]*k[q])**scale) for q in range(len(i))])
                        if float(t[v][-1][-1]) > 1.0 or float(t[v][-1][-1]) < 0.0:
                            raise ValueError("Weights must be in range 0-1. Check normalization and/or assignment.")

    # build all the plots
    for v in range(len(steps)):
        for s in steps[v]:
            u = 0
            for i in eval("[params[q][{0}] for q in xyz[0]]".format(s), locals):
                for j in eval("[params[q][{0}] for q in xyz[1]]".format(s), locals):
                    for k in eval("[params[q][{0}] for q in xyz[2]]".format(s), locals):
                        for q in range(len(t[v][u])):
                            a[v].plot([i[q]],[j[q]],[k[q]],marker='o',color=t[v][u][q],ms=10)
                        u += 1

    # process inputs
    if _out: out = _out
    if out: out = _out or out

    if not out:
        plt.show()
    elif out is True:
        return fig #XXX: better?: fig.axes if len(fig.axes) > 1 else ax
    else:
        fig.savefig(out)
''' % dict(exec_string=exec_string)


def_hypercube_scenario = '''
def hypercube_scenario(filename, datafile=None, **kwds):
    """
generate scenario support plots from file written with ``write_support_file``;
and generate legacy data and cones from a dataset file, if provided

Available from the command shell as::

    support_hypercube_scenario filename (datafile) [options]

or as a function call::

    mystic.support.hypercube_scenario(filename, datafile=None, **options)

Args:
    filename (str): name of the convergence logfile (e.g. ``paramlog.py``)
    datafile (str, default=None): name of the dataset file (e.g. ``data.txt``)

Returns:
    None

Notes:
    - The option *out* takes a string of the filepath for the generated plot.
      If ``out = True``, return the Figure object instead of generating a plot.
    - The options *bounds*, *dim*, and *iters* all take indicator strings.
      The bounds should be given as comma-separated slices. For example, using
      ``bounds = ".062:.125, 0:30, 2300:3200"`` will set lower and upper bounds
      for x to be (.062,.125), y to be (0,30), and z to be (2300,3200). If all
      bounds are to not be strictly enforced, append an asterisk ``*`` to the
      string. The dim (dimensions of the scenario) should comma-separated ints.      For example, ``dim = "1, 1, 2"`` will convert the params to a two-member
      3-D dataset. Iters accepts a string built from comma-separated array
      slices. Thus, ``iters = ":"`` will plot all iters in a single plot.
      Alternatively, ``iters = ":2, 2:"`` will split the iters into two plots,
      while ``iters = "0"`` will only plot the first iteration.
    - The option *label* takes comma-separated strings. Thus ``label = "x,y,"``
      will place 'x' on the x-axis, 'y' on the y-axis, and nothing on the
      z-axis. LaTeX, such as ``label = "$ h $, $ a$, $ v$"`` will label the
      axes with standard LaTeX math formatting. Note that the leading space is
      required, while a trailing space aligns the text with the axis instead of      the plot frame.
    - The option "filter" is used to select datapoints from a given dataset,
      and takes comma-separated ints.
    - A "mask" is given as comma-separated ints. When the mask has more than
      one int, the plot will be 2D.
    - The option "vertical" will plot the dataset values on the vertical axis;
      for 2D plots, cones are always plotted on the vertical axis.
    - The option *nid* takes an integer of the nth simultaneous points to plot.
    - The option *scale* takes an integer as a grayscale contrast multiplier.
    - The option *gap* takes an integer distance from cone center to vertex.
    - The option *data* takes a boolean, to plot legacy data, if provided.
    - The option *cones* takes a boolean, to plot cones, if provided.
    - The option *flat* takes a boolean, to plot results in a single plot.
"""
    import shlex
    try:
        basestring
        from StringIO import StringIO
    except NameError:
        basestring = str
        from io import StringIO
    global __quit
    __quit = False
    _out = False

    instance = None
    # handle the special case where list is provided by sys.argv
    if isinstance(filename, (list,tuple)) and not kwds:
        cmdargs = filename # (above is used by script to parse command line)
    elif isinstance(filename, basestring) and not kwds:
        cmdargs = shlex.split(filename)
    # 'everything else' is essentially the functional interface
    else:
        cmdargs = kwds.get('kwds', '')
        if not cmdargs:
            out = kwds.get('out', None)
            bounds = kwds.get('bounds', None)
            iters = kwds.get('iters', None)
            label = kwds.get('label', None)
            dim = kwds.get('dim', None)
            filter = kwds.get('filter', None)
            mask = kwds.get('mask', None)
            nid = kwds.get('nid', None)
            scale = kwds.get('scale', None)
            gap = kwds.get('gap', None)
            data = kwds.get('data', False)
            cones = kwds.get('cones', False)
            vertical = kwds.get('vertical', False)
            flat = kwds.get('flat', False)

            if isinstance(out, bool): _out, out = out, None

            # process "commandline" arguments
            cmdargs = ''
            cmdargs += '' if out is None else '--out={} '.format(out)
            cmdargs += '' if bounds is None else '--bounds="{}" '.format(bounds)
            cmdargs += '' if iters is None else '--iters="{}" '.format(iters)
            cmdargs += '' if label is None else '--label="{}" '.format(label)
            cmdargs += '' if dim is None else '--dim="{}" '.format(dim)
            cmdargs += '' if filter is None else '--filter="{}" '.format(filter)
            cmdargs += '' if mask is None else '--mask={} '.format(mask)
            cmdargs += '' if nid is None else '--nid={} '.format(nid)
            cmdargs += '' if scale is None else '--scale={} '.format(scale)
            cmdargs += '' if gap is None else '--gap={} '.format(gap)
            cmdargs += '' if data == False else '--data '
            cmdargs += '' if cones == False else '--cones '
            cmdargs += '' if vertical == False else '--vertical '
            cmdargs += '' if flat == False else '--flat '
        else:
            cmdargs = ' ' + cmdargs
        if isinstance(filename, basestring):
            cmdargs = filename.split() + shlex.split(cmdargs)
        else: # special case of passing in monitor instance
            instance = filename
            cmdargs = shlex.split(cmdargs)

    #XXX: note that 'argparse' is new as of python2.7
    from optparse import OptionParser
    def _exit(self, errno=None, msg=None):
        global __quit
        __quit = True
        if errno or msg:
            msg = msg.split(': error: ')[-1].strip()
            raise IOError(msg)
    OptionParser.exit = _exit

    parser = OptionParser(usage=hypercube_scenario.__doc__.split(NL2+'Options:')[0])
    parser.add_option("-u","--out",action="store",dest="out",\
                      metavar="STR",default=None,
                      help="filepath to save generated plot")
    parser.add_option("-b","--bounds",action="store",dest="bounds",\
                      metavar="STR",default="0:1, 0:1, 0:1",
                      help="indicator string to set hypercube bounds")
    parser.add_option("-i","--iters",action="store",dest="iters",\
                      metavar="STR",default="-1",
                      help="indicator string to select iterations to plot")
    parser.add_option("-l","--label",action="store",dest="label",\
                      metavar="STR",default=",,",
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
    parser.add_option("-g","--gap",action="store",dest="gap",\
                      metavar="INT",default=0.0,
                      help="distance from double cone center to cone vertex")
    parser.add_option("-o","--data",action="store_true",dest="legacy",\
                      default=False,help="plot legacy data, if provided")
    parser.add_option("-v","--cones",action="store_true",dest="cones",\
                      default=False,help="plot cones, if provided")
    parser.add_option("-z","--vertical",action="store_true",dest="vertical",\
                      default=False,help="always plot values on vertical axis")
    parser.add_option("-f","--flat",action="store_true",dest="flatten",\
                      default=False,help="show selected iterations in a single plot")

    if PY3:
        f = StringIO()
        parser.print_help(file=f)
        f.seek(0)
        if 'Options:' not in hypercube_scenario.__doc__:
            hypercube_scenario.__doc__ += NL+'Options:{0}'.format(f.read().split('Options:')[-1])
        f.close()
    else:
        if 'Options:' not in hypercube_scenario.__doc__:
            hypercube_scenario.__doc__ += NL+'Options:{0}'.format(parser.format_help().split('Options:')[-1])

    try:
        parsed_opts, parsed_args = parser.parse_args(cmdargs)
    except UnboundLocalError:
        pass
    if __quit: return

    # get the name of the parameter log file
    if instance is None:
        instance = parsed_args[0]
    from mystic.munge import read_history
    params, _cost = read_history(instance)
    # would be nice to use meta = ['wx','wx2','x','x2','wy',...]
    # exec "from {file} import meta".format(file=file)

    from mystic.math.discrete import scenario
    from mystic.math.legacydata import dataset
    try: # select whether to plot the cones
        cones = parsed_opts.cones
    except:
        cones = False

    try: # select whether to plot the legacy data
        legacy = parsed_opts.legacy
    except:
        legacy = False

    try: # path of plot output file
      out = parsed_opts.out  # e.g. 'myplot.png'
      if "None" == out: out = None
    except:
      out = None

    try: # get dataset filter
        filter = parsed_opts.filter
        if "None" == filter: filter = None
        else: filter = [int(i) for i in filter.split(",")] # format is "1,5,9"
    except:
        filter = None

    try: # select the scenario dimensions
        npts = parsed_opts.dim
        if "None" == npts: # npts may have been logged
            from mystic.munge import read_import
            npts = read_import(parsed_args[0], 'npts')
        else: npts = tuple(int(i) for i in dim.split(",")) # format is "1,1,1"
    except:
        npts = (1,1,1) #XXX: better in parsed_args ?

    try: # get the name of the dataset file
        file = parsed_args[1]
        from mystic.math.legacydata import load_dataset
        data = load_dataset(file, filter)
    except:
#       raise IOError("please provide dataset file name")
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
        bounds = _bounds.split(",")  # format is "60:105, 0:30, 2.1:2.8"
        bounds = [tuple(float(j) for j in i.split(':')) for i in bounds]
    except:
        strict = True
        bounds = [(0,1),(0,1),(0,1)]

    try: # select labels for the axes
        label = parsed_opts.label.split(',')  # format is "x, y, z"
    except:
        label = ['','','']

    x = params[-1]
    try: # select which iterations to plot
        select = parsed_opts.iters.split(',')  # format is ":2, 2:4, 5, 6:"
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

    try: # gap between double cone center and cone vertex
        gap = float(parsed_opts.gap)
    except:
        gap = 0.0

    _2D = False # if False, use 3D plots; if True, use 3D plots
    cs = None
    try: # select which axis to plot 'values' on  (3D plot)
        xs = int(parsed_opts.replace)
    except:
        try: # select which axes to mask (2D plot)
            xs = (int(i) for i in parsed_opts.replace.split(",")) # format is "1,2"
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
            raise TypeError("bounds should be tuples of (lower_bound,upper_bound)")

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
    #bounds = [(60,105),(0,30),(2.1,2.8)] or [(None,None)]*3
    #select = ['-1:'] or [':'] or [':1','1:2','2:3','3:'] or similar
    #id = 0 or None

    # get dataset coords (and values) for selected axes
    if data:
        slope = _get_slope(data, xs, cs)
        coords = _get_coords(data, xs, cs)
        #print("bounds: {0}".format(bounds))
        #print("slope: {0}".format(slope))
        #print("coords: {0}".format(coords))
   #else:
   #    slope = []
   #    coords = []

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
    import matplotlib.pyplot as plt
    fig = plt.figure()
    try:
        Subplot3D = None
        if _2D:
            ax1 = fig.add_subplot(dim1, dim2, 1)
            newsubplotcode = "ax{0:d} = ax = fig.add_subplot(dim1,dim2,{0:d});"
        else:
            ax1 = fig.add_subplot(dim1, dim2, 1, projection='3d')
            newsubplotcode = "ax{0:d} = ax = fig.add_subplot(dim1,dim2,{0:d}, projection='3d');"
    except:
        from mpl_toolkits.mplot3d import Axes3D as _Axes3D
        from matplotlib.axes import subplot_class_factory
        Subplot3D = subplot_class_factory(_Axes3D)
        ax1 = Subplot3D(fig, dim1,dim2,1)
        newsubplotcode = "ax{0:d} = ax = Subplot3D(fig, dim1,dim2,{0:d});"
    if _2D:
        ax1.plot([bounds[0][0]],[bounds[1][0]])
        ax1.plot([bounds[0][1]],[bounds[1][1]])
    else:
        ax1.plot([bounds[0][0]],[bounds[1][0]],[bounds[2][0]])
        ax1.plot([bounds[0][1]],[bounds[1][1]],[bounds[2][1]])
    globals = {'plt':plt,'fig':fig,'dim1':dim1,'dim2':dim2,\
               'Subplot3D':Subplot3D,'bounds':bounds,'label':label}
    if not flatten:
        code = "plt.title('iterations[{0}]');".format(select[0])
    else: 
        code = "plt.title('iterations[*]');"
    code = compile(code, '<string>', 'exec')
    %(exec_string)s
    if cones and data and xs in range(len(bounds)):
        if _2D:
            _plot_bowtie(ax1,coords,slope,bounds,axis=axis,tol=gap)
        else:
            _plot_cones(ax1,coords,slope,bounds,axis=axis,strict=strict,tol=gap)
    if legacy and data:
        _plot_data(ax1,coords,bounds,strict=strict,_2D=_2D)
   #_clip_axes(ax1,bounds)
    _label_axes(ax1,label,_2D=_2D)
    a = [ax1]

    # set up additional plots
    if not flatten:
        for i in range(2, plots + 1):
            code = newsubplotcode.format(i)
            if _2D:
                code += "ax.plot([bounds[0][0]],[bounds[1][0]]);"
                code += "ax.plot([bounds[0][1]],[bounds[1][1]]);"
            else:
                code += "ax.plot([bounds[0][0]],[bounds[1][0]],[bounds[2][0]]);"
                code += "ax.plot([bounds[0][1]],[bounds[1][1]],[bounds[2][1]]);"
            code += "plt.title('iterations[{0}]');".format(select[i - 1])
            code = compile(code, '<string>', 'exec')
            %(exec_string)s
            ax = globals['ax']
            if cones and data and xs in range(len(bounds)):
                if _2D:
                    _plot_bowtie(ax,coords,slope,bounds,axis=axis,tol=gap)
                else:
                    _plot_cones(ax,coords,slope,bounds,axis=axis,strict=strict,tol=gap)
            if legacy and data:
                _plot_data(ax,coords,bounds,strict=strict,_2D=_2D)
           #_clip_axes(ax,bounds)
            _label_axes(ax,label,_2D=_2D)
            a.append(ax)

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
    locals = dict(x=x, params=params)
    steps = [eval("range({0})".format(sel.replace(":",",")), locals) for sel in select]

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
            par = eval("[params[q][{0}][0] for q in range(len(params))]".format(s), locals)
            pm = scenario()
            pm.load(par, npts)
            d = dataset()
            d.load(pm.coords, pm.values)
            # dot color determined by number of simultaneous iterations
            t = str((s/qp)**scale)
            # get and plot dataset coords for selected axes
            _coords = _get_coords(d, xs, cs)
            # check if we are replacing an axis
            if _2D and xs == 0:
                if data: # adjust data so cone axis is last 
                    _coords = [list(reversed(pt[:2]))+pt[2:] for pt in _coords]
            elif not _2D and vertical_cones and xs in range(len(bounds)):
                if data: # adjust data so cone axis is last 
                    _coords = [swap(pt,xs) for pt in _coords]
            _plot_data(a[v], _coords, bounds, color=t, strict=strict, _2D=_2D)

    if _2D: # strict = True
        plt.xlim((bounds[0][0],bounds[0][1]))
        plt.ylim((bounds[1][0],bounds[1][1]))

    # process inputs
    if _out: out = _out
    if out: out = _out or out

    if not out:
        plt.show()
    elif out is True:
        return fig #XXX: better?: fig.axes if len(fig.axes) > 1 else ax
    else:
        fig.savefig(out)
''' % dict(exec_string=exec_string)

exec(def_convergence)
exec(def_hypercube)
exec(def_hypercube_measures)
exec(def_hypercube_scenario)
del exec_string, def_convergence, def_hypercube
del def_hypercube_measures, def_hypercube_scenario


if __name__ == '__main__':
    pass


# EOF

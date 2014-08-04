#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

#FIXME: make a script with commandline options

def read_log(file): #FIXME: should also read "solver restart file"
  "read monitor or logfile (or support file) any return params and cost"
 #try:  # get the name of the parameter log file
 #  file = parsed_args[0]
  if isinstance(file, str):
    import re
    file = re.sub('\.py*.$', '', file)  #XXX: strip off .py* extension
    monitor = False
  else: monitor = True
 #except:
 #  raise IOError, "please provide log file name"
  try:  # read standard logfile (or monitor)
    from mystic.munge import logfile_reader, raw_to_support
    if monitor:
      params, cost = file.x, file.y
    else: #FIXME: 'logfile_reader' should work for both file and monitor
      _step, params, cost = logfile_reader(file)
    params, cost = raw_to_support(params, cost)
  except:
    exec "from %s import params" % file
    exec "from %s import cost" % file
  return params, cost


def parse_axes(option):
    """parse option string into grid axes; using modified numpy.ogrid notation

For example:
  option='-1:10' yields x=ogrid[-1:10] and y=0,
  option='-1:10, 2' yields x=ogrid[-1:10] and y=2,
  option='-1:10:.1, 0:10:.1' yields x,y=ogrid[-1:10:.1,0:10:.1],

Returns tuple (x,y,z) with 'x,y' defined above, and 'z' is a boolean
where if a third member is included return z=True, else return z=False.

For example:
  option='-1:10:.1, 0:10:.1, z' yields x,y=ogrid[-1:10:.1,0:10:.1] and z=True
    """
    import numpy
    option = option.split(',')
    opt = dict(zip(['x','y','z'],option))
    if len(option) > 3 or len(option) < 1:
        raise ValueError("invalid format string: '%s'" % ','.join(option))
    z = True if len(option) == 3 else False
    if len(option) == 1: opt['y'] = '0'
    xd = True if ':' in opt['x'] else False
    yd = True if ':' in opt['y'] else False
    #XXX: accepts option='3:1', '1:1', and '1:2:10' (try to catch?)
    if xd and yd:
        try: # x,y form a 2D grid
            exec('x,y = numpy.ogrid[%s,%s]' % (opt['x'],opt['y']))
        except: # AttributeError:
            raise ValueError("invalid format string: '%s'" % ','.join(option))
    elif xd and not z:
        try:
            exec('x = numpy.ogrid[%s]' % opt['x'])
            y = float(opt['y'])
        except: # (AttributeError, SyntaxError, ValueError):
            raise ValueError("invalid format string: '%s'" % ','.join(option))
    elif yd and not z:
        try:
            x = float(opt['x'])
            exec('y = numpy.ogrid[%s]' % opt['y'])
        except: # (AttributeError, SyntaxError, ValueError):
            raise ValueError("invalid format string: '%s'" % ','.join(option))
    else:
        raise ValueError("invalid format string: '%s'" % ','.join(option))
    return x,y,z


def nearest(target, vector): #XXX: useful, but not in this context...
    """find the value in vector that is nearest to target

    vector should be a 1D array of len=N or have shape=(N,1) or (1,N)
    """
    import numpy
    diff = numpy.min(numpy.abs(vector.reshape(-1) - target))
    if numpy.any(vector.reshape(-1) == diff + target):
        return diff + target
    return -diff + target


def draw_projection(file, plotx=True, scale=True, shift=False, color=None, figure=None):
    """draw a solution trajectory (for overlay on a 1D plot)

file is monitor or logfile of solution trajectories
if plotx is True, project along the x-axis
if scale is provided, scale the intensity as 'z = log(4*z*scale+1)+2'
if shift is provided, shift the intensity as 'z = z+shift' (useful for -z's)
if color is provided, set the line color (e.g. 'w', 'k')
if figure is provided, plot to an existing figure
    """
    # params are the parameter trajectories
    # cost is the solution trajectory
    params, cost = read_log(file)
    #XXX: take 'params,cost=None' instead of 'file,surface=False'?
    if len(params) < 2:
        x = y = params[0]
    else:
        x,y = params # requires two parameters

    if not figure: figure = plt.figure()
    ax = figure.gca()
    ax.autoscale(tight=True)

    if color in [None, False]:
        color = 'k'
    import numpy
    if shift: 
        if shift is True: #NOTE: MAY NOT be the exact minimum
            shift = max(-numpy.min(cost), 0.0) + 0.5 # a good guess
        cost = numpy.asarray(cost)+shift
    cost = numpy.asarray(cost)
    if scale:
        cost = numpy.log(4*cost*scale+1)+2

    if plotx:
        ax.plot(x,cost, color+'-o', linewidth=2, markersize=4)
    else:
        ax.plot(y,cost, color+'-o', linewidth=2, markersize=4)
    #XXX: need to 'correct' the z-axis (or provide easy conversion)
    return figure


def draw_trajectory(file, surface=False, scale=True, shift=False, color=None, figure=None):
    """draw a solution trajectory (for overlay on a contour plot)

file is monitor or logfile of solution trajectories
if surface is True, plot the trajectories as a 3D projection
if scale is provided, scale the intensity as 'z = log(4*z*scale+1)+2'
if shift is provided, shift the intensity as 'z = z+shift' (useful for -z's)
if color is provided, set the line color (e.g. 'w', 'k')
if figure is provided, plot to an existing figure
    """
    # params are the parameter trajectories
    # cost is the solution trajectory
    params, cost = read_log(file)
    #XXX: take 'params,cost=None' instead of 'file,surface=False'?
    x,y = params # requires two parameters

    if not figure: figure = plt.figure()
    
    if surface: kwds = {'projection':'3d'} # 3D
    elif surface is None: # 1D
        raise NotImplementedError('need to add an option string parser')
    else: kwds = {}                        # 2D
    ax = figure.gca(**kwds)

    if color in [None, False]:
        color = 'w'#if not scale else 'k'
    if surface: # is 3D, cost is needed
        import numpy
        if shift: 
            if shift is True: #NOTE: MAY NOT be the exact minimum
                shift = max(-numpy.min(cost), 0.0) + 0.5 # a good guess
            cost = numpy.asarray(cost)+shift
        if scale:
            cost = numpy.asarray(cost)
            cost = numpy.log(4*cost*scale+1)+2
        ax.plot(x,y,cost, color+'-o', linewidth=2, markersize=4)
        #XXX: need to 'correct' the z-axis (or provide easy conversion)
    else:    # is 2D, cost not needed
        ax.plot(x,y, color+'-o', linewidth=2, markersize=4)
    return figure


def draw_slice(f, x, y=None, scale=True, shift=False):
    """plot a slice of a 2D function 'f' in 1D

x is an array used to set up the axis
y is a fixed value for the 2nd axis
if scale is provided, scale the intensity as 'z = log(4*z*scale+1)+2'
if shift is provided, shift the intensity as 'z = z+shift' (useful for -z's)

NOTE: when plotting the 'y-axis' at fixed 'x',
pass the array to 'y' and the fixed value to 'x'
    """
    import numpy

    if y is None:
        y = 0.0
    x, y = numpy.meshgrid(x, y)
    plotx = True if numpy.all(y == y[0,0]) else False

    z = 0*x
    s,t = x.shape
    for i in range(s):
        for j in range(t):
            xx,yy = x[i,j], y[i,j]
            z[i,j] = f([xx,yy])
    if shift:
        if shift is True: shift = max(-numpy.min(z), 0.0) + 0.5 # exact minimum
        z = z+shift
    if scale: z = numpy.log(4*z*scale+1)+2
    #XXX: need to 'correct' the z-axis (or provide easy conversion)

    fig = plt.figure()
    ax = fig.gca()
    ax.autoscale(tight=True)
    if plotx:
        ax.plot(x.reshape(-1), z.reshape(-1))
    else:
        ax.plot(y.reshape(-1), z.reshape(-1))
    return fig


def draw_contour(f, x, y=None, surface=False, fill=True, scale=True, shift=False, density=5):
    """draw a contour plot for a given 2D function 'f'

x and y are arrays used to set up a 2D mesh grid
if fill is True, color fill the contours
if surface is True, plot the contours as a 3D projection
if scale is provided, scale the intensity as 'z = log(4*z*scale+1)+2'
if shift is provided, shift the intensity as 'z = z+shift' (useful for -z's)
use density to adjust the number of contour lines
    """
    import numpy

    if y is None:
        y = x
    x, y = numpy.meshgrid(x, y)

    z = 0*x
    s,t = x.shape
    for i in range(s):
        for j in range(t):
            xx,yy = x[i,j], y[i,j]
            z[i,j] = f([xx,yy])
    if shift:
        if shift is True: shift = max(-numpy.min(z), 0.0) + 0.5 # exact minimum
        z = z+shift
    if scale: z = numpy.log(4*z*scale+1)+2
    #XXX: need to 'correct' the z-axis (or provide easy conversion)

    fig = plt.figure()
    if surface and fill is None: # 'hidden' option; full 3D surface plot
        ax = fig.gca(projection='3d')
        d = max(11 - density, 1) # or 1/density ?
        kwds = {'rstride':d,'cstride':d,'cmap':cm.jet,'linewidth':0}
        ax.plot_surface(x, y, z, **kwds)
    else:
        if surface: kwds = {'projection':'3d'} # 3D
        elif surface is None: # 1D
            raise NotImplementedError('need to add an option string parser')
        else: kwds = {}                        # 2D
        ax = fig.gca(**kwds)
        density = 10*density
        if fill: plotf = ax.contourf  # filled contours
        else: plotf = ax.contour      # wire contours
        plotf(x, y, z, density, cmap=cm.jet)
    return fig


if __name__ == '__main__':
   #FIXME: for a script, need to: 
   # - build an appropriately-sized default grid, or parse "slices" info
   # - get a model ('package.file.model' or 'mystic.models.Foo(2)')
   # - do the other option parsing magic...
   # - enable 'skip' plotting points (points or line or both)?
   # - enable adding constraints 'mask' (set cost=nan where violate constraints)
   #FIXME: should be able to:
   # - apply a constraint as a region of NaN
   # - apply a penalty by shifting the surface (maybe plot using alpha?)
   # - read logfile with multiple trajectories (i.e. parallel batch)
   #FIXME: current issues:
   # - should be able to plot from solver.genealogy (multi-monitor?) [1D,2D,3D?]
   # - should be able to scale 'z-axis' instead of scaling 'z' itself
   #   (see https://github.com/matplotlib/matplotlib/issues/209)

    import numpy
    from mystic.models import rosen, step, quartic, shekel
    from mystic.models import zimmermann, griewangk, fosc3d
    from mystic.models.corana import corana2d
    from mystic.models import wavy1, wavy2
    from mystic.solvers import fmin, fmin_powell, diffev
    from mystic.tools import reduced

    ### INPUTS ###
   #spec = '-50:50:.5, -50:50:.5, z'
    spec = '-1:10:.1, -1:10:.1, z'
    scale = True
    fill = False
    source = 'log.txt'
    ##############

    # process inputs
    x,y,z = parse_axes(spec)
    color = 'w' if fill else 'k'
   #model = reduced(numpy.add, arraylike=False)(wavy1)
    model = zimmermann

    # for demo purposes, pick a solver (then solve)
    solver = fmin
    initial = (0,0)
   #solver = diffev
   #initial = ((-1,10),(-1,10))
    from mystic.monitors import VerboseLoggingMonitor
    itermon = VerboseLoggingMonitor(filename=source, new=True)
    sol = solver(model, x0=initial, itermon=itermon)

    ### INPUTS? ###
    # read trajectories from monitor (comment out to use logfile)
    source = itermon
    # if negative minimum, shift by the 'solved minimum' plus an epsilon
    shift = max(-numpy.min(itermon.y), 0.0) + 0.5 # a good guess
   #shift = False
    ###############

    # project trajectory on a 1D slice of the model surface #XXX: useful?
#   fig0 = draw_slice(model, x=x, y=sol[-1], scale=scale, shift=shift)
#   draw_projection(source, plotx=True, color=color, scale=scale, shift=shift, figure=fig0)

    # plot the trajectory on the model surface (2D and 3D)
    fig1 = draw_contour(model, x, y, fill=fill, scale=scale, shift=shift)
    draw_trajectory(source, color=color, scale=scale, shift=shift, figure=fig1)

    fig2 = draw_contour(model, x, y, surface=True, fill=fill, scale=scale, shift=shift)
    draw_trajectory(source, surface=True, color=color, scale=scale, shift=shift, figure=fig2)

#   draw_contour(model, x, y, fill=fill, scale=scale)
#   draw_contour(model, x, y, surface=True, fill=fill, scale=scale)
##  draw_contour(model, x, y, surface=True, fill=None, scale=scale)
    plt.show()


# EOF

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

def read_log(file):
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
    else: #FIXME: 'reader' should work for both file and monitor
      _step, params, cost = logfile_reader(file)
    params, cost = raw_to_support(params, cost)
  except:
    exec "from %s import params" % file
    exec "from %s import cost" % file
  return params, cost


def draw_trajectory(file, surface=False, scale=True, color=None, figure=None):
    """draw a solution trajectory (for overlay on a contour plot)

file is monitor or logfile of solution trajectories
if surface is True, plot the trajectories as a 3D projection
if scale is provided, scale the intensity as 'z = log(4*z*scale+1)+2'
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
    else: kwds = {}                        # 2D
    ax = figure.gca(**kwds)

    if color in [None, False]:
        color = 'w'#if not scale else 'k'
    if surface: # is 3D, cost is needed
        if scale:
            import numpy
            cost = numpy.asarray(cost)
            cost = numpy.log(4*cost*scale+1)+2
        ax.plot(x,y,cost, color+'-o', linewidth=2, markersize=4)
    else:    # is 2D, cost not needed
        ax.plot(x,y, color+'-o', linewidth=2, markersize=4)
    return figure


def draw_contour(f, x, y=None, surface=False, fill=True, scale=True, density=5):
    """draw a contour plot for a given 2D function 'f'

x and y are arrays used to set up a 2D mesh grid
if fill is True, color fill the contours
if surface is True, plot the contours as a 3D projection
if scale is provided, scale the intensity as 'z = log(4*z*scale+1)+2'
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
    if scale: z = numpy.log(4*z*scale+1)+2

    fig = plt.figure()
    if surface and fill is None: # 'hidden' option; full 3D surface plot
        ax = fig.gca(projection='3d')
        d = max(11 - density, 1) # or 1/density ?
        kwds = {'rstride':d,'cstride':d,'cmap':cm.jet,'linewidth':0}
        ax.plot_surface(x, y, z, **kwds)
    else:
        if surface: kwds = {'projection':'3d'} # 3D
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
   #FIXME: should also work for 1D function plot
   # - plot function (and trajectory) on selected axis
   # - should be able to plot from solver.genealogy (multi-monitor?) [1D,2D,3D?]

    import numpy
   #x, y = numpy.ogrid[-50:50:0.5,-50:50:0.5]
    x, y = numpy.ogrid[-1:10:0.1,-1:10:0.1]
    from mystic.models import rosen, step, quartic, shekel
    from mystic.models import corana, zimmermann, griewangk, fosc3d
    from mystic.solvers import fmin, fmin_powell#, diffev

    scale = True
    fill = False
    model = zimmermann
    solver = fmin
    source = 'log.txt'
    color = 'w' if fill else 'k'

    from mystic.monitors import VerboseLoggingMonitor
    itermon = VerboseLoggingMonitor(filename=source, new=True)
    sol = solver(model, x0=(0,0), itermon=itermon)
    # read trajectories from monitor (comment out to use logfile)
    source = itermon

    # build the plots
    fig1 = draw_contour(model, x, y, fill=fill, scale=scale)
    draw_trajectory(source, color=color, scale=scale, figure=fig1)

    fig2 = draw_contour(model, x, y, surface=True, fill=fill, scale=scale)
    draw_trajectory(source, surface=True, color=color, scale=scale, figure=fig2)

#   draw_contour(model, x, y, fill=fill, scale=scale)
#   draw_contour(model, x, y, surface=True, fill=fill, scale=scale)
##  draw_contour(model, x, y, surface=True, fill=None, scale=scale)
    plt.show()


# EOF

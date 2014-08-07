#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

from mystic.munge import read_history

def get_instance(location, *args, **kwds):
    """given the import location of a model or model class, return the model

args and kwds will be passed to the constructor of the model class
    """
    package, target = location.rsplit('.',1)
    exec "from %s import %s as model" % (package, target)
    import inspect
    if inspect.isclass(model):
        model = model(*args, **kwds)
    return model


def parse_input(option):
    """parse 'option' string into 'select', 'axes', and 'mask'

select contains the dimension specifications on which to plot
axes holds the indicies of the parameters selected to plot
mask is a dictionary of the parameter indicies and fixed values

For example,
    >>> select, axes, mask = parse_input("-1:10:.1, 0.0, 5.0, -50:50:.5")
    >>> select
    [0, 3]
    >>> axes
    "-1:10:.1, -50:50:.5"
    >>> mask
    {1: 0.0, 2: 5.0}
    """
    option = option.split(',')
    select = []
    axes = []
    mask = {}
    for index,value in enumerate(option):
        if ":" in value:
            select.append(index)
            axes.append(value)
        else:
            mask.update({index:float(value)})
    axes = ','.join(axes)
    return select, axes, mask


def parse_axes(option, grid=True):
    """parse option string into grid axes; using modified numpy.ogrid notation

For example:
  option='-1:10:.1, 0:10:.1' yields x,y=ogrid[-1:10:.1,0:10:.1],

If grid is False, accept options suitable for line plotting.
For example:
  option='-1:10' yields x=ogrid[-1:10] and y=0,
  option='-1:10, 2' yields x=ogrid[-1:10] and y=2,

Returns tuple (x,y) with 'x,y' defined above.
    """
    import numpy
    option = option.split(',')
    opt = dict(zip(['x','y','z'],option))
    if len(option) > 2 or len(option) < 1:
        raise ValueError("invalid format string: '%s'" % ','.join(option))
    z = bool(grid)
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
    return x,y


def draw_projection(file, select=0, scale=True, shift=False, color=None, figure=None):
    """draw a solution trajectory (for overlay on a 1D plot)

file is monitor or logfile of solution trajectories
select is the parameter index (e.g. 0 -> param[0]) selected for plotting
if scale is provided, scale the intensity as 'z = log(4*z*scale+1)+2'
if shift is provided, shift the intensity as 'z = z+shift' (useful for -z's)
if color is provided, set the line color (e.g. 'w', 'k')
if figure is provided, plot to an existing figure
    """
    # params are the parameter trajectories
    # cost is the solution trajectory
    params, cost = read_history(file)
    d = {'x':0, 'y':1, 'z':2} #XXX: remove the easter egg?
    if select in d: select = d[select]
    x = params[int(select)] # requires one parameter

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

    ax.plot(x,cost, color+'-o', linewidth=2, markersize=4)
    #XXX: need to 'correct' the z-axis (or provide easy conversion)
    return figure


def draw_trajectory(file, select=None, surface=False, scale=True, shift=False, color=None, figure=None):
    """draw a solution trajectory (for overlay on a contour plot)

file is monitor or logfile of solution trajectories
select is a len-2 list of parameter indicies (e.g. 0,1 -> param[0],param[1])
if surface is True, plot the trajectories as a 3D projection
if scale is provided, scale the intensity as 'z = log(4*z*scale+1)+2'
if shift is provided, shift the intensity as 'z = z+shift' (useful for -z's)
if color is provided, set the line color (e.g. 'w', 'k')
if figure is provided, plot to an existing figure
    """
    # params are the parameter trajectories
    # cost is the solution trajectory
    params, cost = read_history(file)
    if select is None: select = (0,1)
    d = {'x':0, 'y':1, 'z':2} #XXX: remove the easter egg?
    for ind,val in enumerate(select):
        if val in d: select[ind] = d[val]
    params = [params[int(i)] for i in select[:2]]
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
   # - do the other option parsing magic...
   # - enable 'skip' plotting points (points or line or both)?
   #FIXME: should be able to:
   # - apply a constraint as a region of NaN -- apply when 'xx,yy=x[ij],y[ij]'
   # - apply a penalty by shifting the surface (plot w/alpha?) -- as above
   # - read logfile with multiple trajectories (i.e. parallel batch)
   # - build an appropriately-sized default grid (from logfile info)
   #FIXME: current issues:
   # - 1D slice and projection work for 2D function, but aren't "pretty"
   # - 1D slice and projection for 1D function needs further testing...
   # - should be able to plot from solver.genealogy (multi-monitor?) [1D,2D,3D?]
   # - should be able to scale 'z-axis' instead of scaling 'z' itself
   #   (see https://github.com/matplotlib/matplotlib/issues/209)

    from mystic.tools import reduced, masked, partial

    ### INPUTS ###
    options = '-1:10:.1, -1:10:.1' #, 1.0'
   #options = '-50:50:.5, -50:50:.5'
    model = 'mystic.models.rosen'
    reducer = 'numpy.add'
    source = None #'log.txt'
    surface = True
    scale = True
    shift = False
    fill = False
    demo = False
    ##############
    #NOTE: The demo constrains params explicitly in the solver, then reduces
    #      dimensions appropriately so results can be 2D contour plotted.
    #      When working with legacy results that have more than 2 params,
    #      the trajectory WILL NOT follow the masked surface generated
    #      because the masked params were NOT fixed when the solver was run.

    # process inputs
    color = 'w' if fill else 'k'
    select, spec, mask = parse_input(options)
    x,y = parse_axes(spec, grid=True) # grid=False for 1D plots
    #FIXME: does grid=False still make sense here...?
    reducer = get_instance(reducer)
    if demo and (not source or not model):
        raise RuntimeError('a model and results filename are required')
    elif not source and not model:
        raise RuntimeError('a model or a results file is required')
    if model:
        model = get_instance(model)
        model = reduced(reducer, arraylike=False)(model) # need if returns array

    if demo:
        # for demo purposes... pick a solver
        solver = 'mystic.solvers.fmin'
        solver = get_instance(solver)
        xlen = len(select)+len(mask)
        if solver.__name__.startswith('diffev'):
            initial = [(-1,1)]*xlen
        else:
            initial = [0]*xlen
        from mystic.monitors import VerboseLoggingMonitor
        itermon = VerboseLoggingMonitor(filename=source, new=True)
        # explicitly constrain parameters
        model = partial(mask)(model)
        # solve
        sol = solver(model, x0=initial, itermon=itermon)

        #-OVERRIDE-INPUTS-# 
        import numpy
        # read trajectories from monitor (comment out to use logfile)
        source = itermon
        # if negative minimum, shift by the 'solved minimum' plus an epsilon
        shift = max(-numpy.min(itermon.y), 0.0) + 0.5 # a good guess
        #-----------------#

    if model: # for plotting, implicitly constrain by reduction
        model = masked(mask)(model)

    # project trajectory on a 1D slice of the model surface #XXX: useful?
#   fig0 = draw_slice(model, x=x, y=sol[-1], scale=scale, shift=shift)
#   draw_projection(source, select=0, color=color, scale=scale, shift=shift, figure=fig0)

    # plot the trajectory on the model surface (2D and 3D)
    if model: # plot the surface
        fig1 = draw_contour(model, x, y, fill=fill, scale=scale, shift=shift)
    else:
        fig1 = None
    if source: # plot the trajectory
        draw_trajectory(source, select=select, color=color, scale=scale, shift=shift, figure=fig1)

    if model: # plot the surface
        fig2 = draw_contour(model, x, y, surface=True, fill=fill, scale=scale, shift=shift)
    else:
        fig2 = None
    if source: # plot the trajectory
        draw_trajectory(source, select=select, surface=True, color=color, scale=scale, shift=shift, figure=fig2)

#   draw_contour(model, x, y, fill=fill, scale=scale)
#   draw_contour(model, x, y, surface=True, fill=fill, scale=scale)
##  draw_contour(model, x, y, surface=True, fill=None, scale=scale)
    plt.show()


# EOF

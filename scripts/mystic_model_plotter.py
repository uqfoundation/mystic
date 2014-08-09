#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE

__doc__ = """
mystic_model_plotter.py [options] model (filename)

generate surface contour plots for model, specified by full import path
generate model trajectory from logfile, if provided

The option "bounds" takes an indicator string, where the bounds should
be given as comma-separated slices. For example, using bounds = "-1:10, 0:20"
will set the lower and upper bounds for x to be (-1,10) and y to be (0,20).
The "step" can also be given, to control the number of lines plotted in the
grid. Thus "-1:10:.1, 0:20" would set the bounds as above, but use increments
of .1 along x and the default step along y.  For models with > 2D, the bounds
can be used to specify 2 dimensions plus fixed values for remaining dimensions.
Thus, "-1:10, 0:20, 1.0" would plot the 2D surface where the z-axis was fixed
at z=1.0.

The option "label" takes comma-separated strings. For example, label = "x,y,"
will place 'x' on the x-axis, 'y' on the y-axis, and nothing on the z-axis.
LaTeX is also accepted. For example, label = "$ h $, $ {\\alpha}$, $ v$" will
label the axes with standard LaTeX math formatting. Note that the leading
space is required, while a trailing space aligns the text with the axis
instead of the plot frame.

The option "reduce" can be given to reduce the output of a model to a scalar,
thus converting 'model(params)' to 'reduce(model(params))'. A reducer is given
by the import path (e.g. 'numpy.add'). The option "scale" will convert the plot
to log-scale, and scale the cost by 'z=log(4*z*scale+1)+2'. This is useful for
visualizing small contour changes around the minimium. If using log-scale
produces negative numbers, the option "shift" can be used to shift the cost
by 'z=z+shift'. Both shift and scale are intended to help visualize contours.

Required Inputs:
  model               full import path for the model (e.g. mystic.models.rosen)

Additional Inputs:
  filename            name of the convergence logfile (e.g. log.txt)
"""

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


def draw_projection(file, select=0, scale=True, shift=False, style=None, figure=None):
    """draw a solution trajectory (for overlay on a 1D plot)

file is monitor or logfile of solution trajectories
select is the parameter index (e.g. 0 -> param[0]) selected for plotting
if scale is provided, scale the intensity as 'z = log(4*z*scale+1)+2'
if shift is provided, shift the intensity as 'z = z+shift' (useful for -z's)
if style is provided, set the line style (e.g. 'w-o', 'k-', 'ro')
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

    if style in [None, False]:
        style = 'k-o'
    import numpy
    if shift: 
        if shift is True: #NOTE: MAY NOT be the exact minimum
            shift = max(-numpy.min(cost), 0.0) + 0.5 # a good guess
        cost = numpy.asarray(cost)+shift
    cost = numpy.asarray(cost)
    if scale:
        cost = numpy.log(4*cost*scale+1)+2

    ax.plot(x,cost, style, linewidth=2, markersize=4)
    #XXX: need to 'correct' the z-axis (or provide easy conversion)
    return figure


def draw_trajectory(file, select=None, surface=False, scale=True, shift=False, style=None, figure=None):
    """draw a solution trajectory (for overlay on a contour plot)

file is monitor or logfile of solution trajectories
select is a len-2 list of parameter indicies (e.g. 0,1 -> param[0],param[1])
if surface is True, plot the trajectories as a 3D projection
if scale is provided, scale the intensity as 'z = log(4*z*scale+1)+2'
if shift is provided, shift the intensity as 'z = z+shift' (useful for -z's)
if style is provided, set the line style (e.g. 'w-o', 'k-', 'ro')
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

    if style in [None, False]:
        style = 'w-o' #if not scale else 'k-o'
    if surface: # is 3D, cost is needed
        import numpy
        if shift: 
            if shift is True: #NOTE: MAY NOT be the exact minimum
                shift = max(-numpy.min(cost), 0.0) + 0.5 # a good guess
            cost = numpy.asarray(cost)+shift
        if scale:
            cost = numpy.asarray(cost)
            cost = numpy.log(4*cost*scale+1)+2
        ax.plot(x,y,cost, style, linewidth=2, markersize=4)
        #XXX: need to 'correct' the z-axis (or provide easy conversion)
    else:    # is 2D, cost not needed
        ax.plot(x,y, style, linewidth=2, markersize=4)
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

    #XXX: note that 'argparse' is new as of python2.7
    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option("-b","--bounds",action="store",dest="bounds",\
                      metavar="STR",default="0:1:.1, 0:1:.1",
                      help="indicator string to set plot bounds and density")
    parser.add_option("-l","--label",action="store",dest="label",\
                      metavar="STR",default=",,",
                      help="string to assign label to axis")
#   parser.add_option("-n","--nid",action="store",dest="id",\
#                     metavar="INT",default=None,
#                     help="id # of the nth simultaneous points to plot")
#   parser.add_option("-i","--iters",action="store",dest="iters",\
#                     metavar="STR",default=":",
#                     help="indicator string to select iterations to plot")
    parser.add_option("-r","--reduce",action="store",dest="reducer",\
                      metavar="STR",default="None",
                      help="import path of output reducer function")
    parser.add_option("-x","--scale",action="store",dest="zscale",\
                      metavar="INT",default=0.0,
                      help="scale plotted cost by z=log(4*z*scale+1)+2")
    parser.add_option("-z","--shift",action="store",dest="zshift",\
                      metavar="INT",default=0.0,
                      help="shift plotted cost by z=z+shift")
    parser.add_option("-f","--fill",action="store_true",dest="fill",\
                      default=False,help="plot using filled contours")
    parser.add_option("-d","--depth",action="store_true",dest="surface",\
                      default=False,help="plot contours showing depth in 3D")
    parser.add_option("-o","--dots",action="store_true",dest="dots",\
                      default=False,help="show trajectory points in plot")
    parser.add_option("-j","--join",action="store_true",dest="line",\
                      default=False,help="connect trajectory points in plot")
    parsed_opts, parsed_args = parser.parse_args()

    # get the import path for the model
    model = parsed_args[0]  # e.g. 'mystic.models.rosen'

    try: # get the name of the parameter log file
      source = parsed_args[1]  # e.g. 'log.txt'
    except:
      source = None

    try: # select the bounds
      options = parsed_opts.bounds  # format is "-1:10:.1, -1:10:.1, 1.0"
    except:
      options = "0:1:.1, 0:1:.1"

    try: # plot using filled contours
      fill = parsed_opts.fill
    except:
      fill = False

    try: # plot contours showing depth in 3D
      surface = parsed_opts.surface
    except:
      surface = False

    #XXX: can't do '-x' with no argument given  (use T/F instead?)
    try: # scale plotted cost by z=log(4*z*scale+1)+2
      scale = float(parsed_opts.zscale)
      if not scale: scale = False
    except:
      scale = False

    #XXX: can't do '-z' with no argument given
    try: # shift plotted cost by z=z+shift
      shift = float(parsed_opts.zshift)
      if not shift: shift = False
    except:
      shift = False

    try: # import path of output reducer function
      reducer = parsed_opts.reducer  # e.g. 'numpy.add'
      if "None" == reducer: reducer = None
    except:
      reducer = None

    style = '-' # default linestyle
    if parsed_opts.dots:
      mark = 'o' # marker=mark
      # when using 'dots', also can turn off 'line'
      if not parsed_opts.line:
        style = '' # linestyle='None'
    else:
      mark = ''
    color = 'w' if fill else 'k'
    style = color + style + mark

    try: # select labels for the axes
      label = parsed_opts.label.split(',')  # format is "x, y, z"
    except:
      label = ['','','']

#   try: # select which 'id' to plot results for
#     id = (int(parsed_opts.id),) #XXX: allow selecting more than one id ?
#   except:
#     id = None # i.e. 'all' **or** use id=0, which should be 'best' energy ?

#   try: # select which iterations to plot
#     iters = parsed_opts.iters.split(',')  # format is ":2, 2:4, 5, 6:"
#   except:
#     iters = [':']

    #################################################
    solver = None  # set to 'mystic.solvers.fmin' (or similar) for 'live' fits
    #NOTE: 'live' runs constrain params explicitly in the solver, then reduce
    #      dimensions appropriately so results can be 2D contour plotted.
    #      When working with legacy results that have more than 2 params,
    #      the trajectory WILL NOT follow the masked surface generated
    #      because the masked params were NOT fixed when the solver was run.
    #################################################

    from mystic.tools import reduced, masked, partial

    # process inputs
    select, spec, mask = parse_input(options)
    x,y = parse_axes(spec, grid=True) # grid=False for 1D plots
    #FIXME: does grid=False still make sense here...?
    if reducer: reducer = get_instance(reducer)
    if solver and (not source or not model):
        raise RuntimeError('a model and results filename are required')
    elif not source and not model:
        raise RuntimeError('a model or a results file is required')
    if model:
        model = get_instance(model)
        # need a reducer if model returns an array
        if reducer: model = reduced(reducer, arraylike=False)(model)

    if solver:
        # if 'live'... pick a solver
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
#   draw_projection(source, select=0, style=style, scale=scale, shift=shift, figure=fig0)

    # plot the trajectory on the model surface (2D or 3D)
    if model: # plot the surface
        fig = draw_contour(model, x, y, surface=surface, fill=fill, scale=scale, shift=shift)
    else:
        fig = None
    if source: # plot the trajectory
        fig = draw_trajectory(source, select=select, surface=surface, style=style, scale=scale, shift=shift, figure=fig)

    # add labels to the axes
    if surface: kwds = {'projection':'3d'} # 3D
    else: kwds = {}                        # 2D
    ax = fig.gca(**kwds)
    ax.set_xlabel(label[0])
    ax.set_ylabel(label[1])
    if surface: ax.set_zlabel(label[2])

    plt.show()


# EOF

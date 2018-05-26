import collections
#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Jean-Christophe Fillion-Robin (jchris.fillionr @kitware.com)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2018 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
__doc__ = """
functional interfaces for mystic's visual analytics scripts
"""

__all__ = ['model_plotter','log_reader','collapse_plotter']

# globals
__quit = False
import sys
if (sys.hexversion >= 0x30000f0):
    exec_string = 'exec(code, globals)'
    PY3 = True
else:
    exec_string = 'exec code in globals'
    PY3 = False 


#XXX: better if reads single id only? (e.g. same interface as read_history)
def _get_history(source, ids=None):
    """get params and cost from the given source

source is the name of the trajectory logfile (or solver instance)
if provided, ids are the list of 'run ids' to select
    """
    try: # if it's a logfile, it might be multi-id
        from mystic.munge import read_trajectories
        step, param, cost = read_trajectories(source)
    except: # it's not a logfile, so read and return
        from mystic.munge import read_history
        param, cost = read_history(source)
        return [param],[cost]

    # split (i,id) into iteration and id
    multinode = len(step[0]) - 1  #XXX: what if step = []?
    if multinode: id = [i[1] for i in step]
    else: id = [0 for i in step]

    params = [[] for i in range(max(id) + 1)]
    costs = [[] for i in range(len(params))]
    # populate params for each id with the corresponding (param,cost)
    for i in range(len(id)):
        if ids is None or id[i] in ids: # take only the selected 'id'
            params[id[i]].append(param[i])
            costs[id[i]].append(cost[i])
    params = [r for r in params if len(r)] # only keep selected 'ids'
    costs = [r for r in costs if len(r)] # only keep selected 'ids'

    # convert to support format
    from mystic.munge import raw_to_support
    for i in range(len(params)):
        params[i], costs[i] = raw_to_support(params[i], costs[i])
    return params, costs


def_get_instance = '''
def _get_instance(location, *args, **kwds):
    """given the import location of a model or model class, return the model

args and kwds will be passed to the constructor of the model class
    """
    globals = {}
    package, target = location.rsplit('.',1)
    code = "from {0} import {1} as model".format(package, target)
    code = compile(code, '<string>', 'exec')
    %(exec_string)s
    model = globals['model']
    import inspect
    if inspect.isclass(model):
        model = model(*args, **kwds)
    return model
''' % dict(exec_string=exec_string)

exec(def_get_instance)
del def_get_instance


def _parse_input(option):
    """parse 'option' string into 'select', 'axes', and 'mask'

select contains the dimension specifications on which to plot
axes holds the indices of the parameters selected to plot
mask is a dictionary of the parameter indices and fixed values

For example,
    >>> select, axes, mask = _parse_input("-1:10:.1, 0.0, 5.0, -50:50:.5")
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


def_parse_axes = '''
def _parse_axes(option, grid=True):
    """parse option string into grid axes; using modified numpy.ogrid notation

For example:
  option='-1:10:.1, 0:10:.1' yields x,y=ogrid[-1:10:.1,0:10:.1],

If grid is False, accept options suitable for line plotting.
For example:
  option='-1:10' yields x=ogrid[-1:10] and y=0,
  option='-1:10, 2' yields x=ogrid[-1:10] and y=2,

Returns tuple (x,y) with 'x,y' defined above.
    """
    option = option.split(',')
    msg = "invalid format string: '{0}'".format(','.join(option))
    opt = dict(zip(['x','y','z'],option))
    if len(option) > 2 or len(option) < 1:
        msg = "invalid format string: '{0}'".format(','.join(option))
        raise ValueError(msg)
    z = bool(grid)
    if len(option) == 1: opt['y'] = '0'
    xd = True if ':' in opt['x'] else False
    yd = True if ':' in opt['y'] else False
    #XXX: accepts option='3:1', '1:1', and '1:2:10' (try to catch?)
    globals = {}
    code = 'import numpy;'
    if xd and yd:
        try: # x,y form a 2D grid
            code += 'x,y = numpy.ogrid[{0},{1}]'.format(opt['x'],opt['y'])
            code = compile(code, '<string>', 'exec')
            %(exec_string)s
            x = globals['x']
            y = globals['y']
        except: # AttributeError:
            msg = "invalid format string: '{0}'".format(','.join(option))
            raise ValueError(msg)
    elif xd and not z:
        try:
            code += 'x = numpy.ogrid[{0}]'.format(opt['x'])
            code = compile(code, '<string>', 'exec')
            %(exec_string)s
            x = globals['x']
            y = float(opt['y'])
        except: # (AttributeError, SyntaxError, ValueError):
            msg = "invalid format string: '{0}'".format(','.join(option))
            raise ValueError(msg)
    elif yd and not z:
        try:
            x = float(opt['x'])
            code += 'y = numpy.ogrid[{0}]'.format(opt['y'])
            code = compile(code, '<string>', 'exec')
            %(exec_string)s
            y = globals['y']
        except: # (AttributeError, SyntaxError, ValueError):
            msg = "invalid format string: '{0}'".format(','.join(option))
            raise ValueError(msg)
    else:
        msg = "invalid format string: '{0}'".format(','.join(option))
        raise ValueError(msg)
    if not x.size or not y.size:
        msg = "invalid format string: '{0}'".format(','.join(option))
        raise ValueError(msg)
    return x,y
''' % dict(exec_string=exec_string)

exec(def_parse_axes)
del def_parse_axes


def _draw_projection(x, cost, scale=True, shift=False, style=None, figure=None):
    """draw a solution trajectory (for overlay on a 1D plot)

x is the sequence of values for one parameter (i.e. a parameter trajectory)
cost is the sequence of costs (i.e. the solution trajectory)
if scale is provided, scale the intensity as 'z = log(4*z*scale+1)+2'
if shift is provided, shift the intensity as 'z = z+shift' (useful for -z's)
if style is provided, set the line style (e.g. 'w-o', 'k-', 'ro')
if figure is provided, plot to an existing figure
    """
    import matplotlib.pyplot as plt
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


def _draw_trajectory(x, y, cost=None, scale=True, shift=False, style=None, figure=None):
    """draw a solution trajectory (for overlay on a contour plot)

x is a sequence of values for one parameter (i.e. a parameter trajectory)
y is a sequence of values for one parameter (i.e. a parameter trajectory)
cost is the solution trajectory (i.e. costs); if provided, plot a 3D contour
if scale is provided, scale the intensity as 'z = log(4*z*scale+1)+2'
if shift is provided, shift the intensity as 'z = z+shift' (useful for -z's)
if style is provided, set the line style (e.g. 'w-o', 'k-', 'ro')
if figure is provided, plot to an existing figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    if not figure: figure = plt.figure()

    if cost: kwds = {'projection':'3d'} # 3D
    else: kwds = {}                     # 2D
    ax = figure.gca(**kwds)

    if style in [None, False]:
        style = 'w-o' #if not scale else 'k-o'
    if cost: # is 3D, cost is needed
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


def _draw_slice(f, x, y=None, scale=True, shift=False):
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

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca()
    ax.autoscale(tight=True)
    if plotx:
        ax.plot(x.reshape(-1), z.reshape(-1))
    else:
        ax.plot(y.reshape(-1), z.reshape(-1))
    return fig


def _draw_contour(f, x, y=None, surface=False, fill=True, scale=True, shift=False, density=5):
    """draw a contour plot for a given 2D function 'f'

x and y are arrays used to set up a 2D mesh grid
if fill is True, color fill the contours
if surface is True, plot the contours as a 3D projection
if scale is provided, scale the intensity as 'z = log(4*z*scale+1)+2'
if shift is provided, shift the intensity as 'z = z+shift' (useful for -z's)
use density to adjust the number of contour lines
    """
    import numpy
    from matplotlib import cm

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

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
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


def model_plotter(model, logfile=None, **kwds):
    """
generate surface contour plots for model, specified by full import path; and
generate model trajectory from logfile (or solver restart file), if provided

Available from the command shell as::

    mystic_model_plotter.py model (logfile) [options]

or as a function call::

    mystic.model_plotter(model, logfile=None, **options)

Args:
    model (str): full import path for the model (e.g. ``mystic.models.rosen``)
    logfile (str, default=None): name of convergence logfile (e.g. ``log.txt``)

Returns:
    None

Notes:
    - The option *out* takes a string of the filepath for the generated plot.
    - The option *bounds* takes an indicator string, where bounds are given as
      comma-separated slices. For example, using ``bounds = "-1:10, ^0:20"``
      will set lower and upper bounds for x to be (-1,10) and y to be (0,20).
      The "step" can also be given, to control the number of lines plotted in
      the grid. Thus ``"-1:10:.1, 0:20"`` sets the bounds as above, but uses
      increments of .1 along x and the default step along y.  For models > 2D,
      the bounds can be used to specify 2 dimensions plus fixed values for
      remaining dimensions. Thus, ``"-1:10, 0:20, 1.0"`` plots the 2D surface
      where the z-axis is fixed at z=1.0. When called from a script, slice
      objects can be used instead of a string, thus ``"-1:10:.1, 0:20, 1.0"``
      becomes ``(slice(-1,10,.1), slice(20), 1.0)``.
    - The option *label* takes comma-separated strings. For example,
      ``label = "x,y,"`` will place 'x' on the x-axis, 'y' on the y-axis, and
      nothing on the z-axis.  LaTeX is also accepted. For example,
      ``label = "$ h $, $ {\\alpha}$, $ v$"`` will label the axes with standard
      LaTeX math formatting. Note that the leading space is required, while a
      trailing space aligns the text with the axis instead of the plot frame.
    - The option *nid* takes an integer of the nth simultaneous points to plot.
    - The option *iter* takes an integer of the largest iteration to plot.
    - The option *reduce* can be given to reduce the output of a model to a
      scalar, thus converting ``model(params)`` to ``reduce(model(params))``.
      A reducer is given by the import path (e.g. ``numpy.add``).
    - The option *scale* will convert the plot to log-scale, and scale the
      cost by ``z=log(4*z*scale+1)+2``. This is useful for visualizing small
      contour changes around the minimium.
    - If using log-scale produces negative numbers, the option *shift* can be
      used to shift the cost by ``z=z+shift``. Both *shift* and *scale* are
      intended to help visualize contours.
    - The option *fill* takes a boolean, to plot using filled contours.
    - The option *depth* takes a boolean, to plot contours in 3D.
    - The option *dots* takes a boolean, to show trajectory points in the plot.
    - The option *join* takes a boolean, to connect trajectory points.
    - The option *verb* takes a boolean, to print the model documentation.
"""
    #FIXME: should be able to:
    # - apply a constraint as a region of NaN -- apply when 'xx,yy=x[ij],y[ij]'
    # - apply a penalty by shifting the surface (plot w/alpha?) -- as above
    # - build an appropriately-sized default grid (from logfile info)
    # - move all mulit-id param/cost reading into read_history
    #FIXME: current issues:
    # - 1D slice and projection work for 2D function, but aren't "pretty"
    # - 1D slice and projection for 1D function, is it meaningful and correct?
    # - should be able to plot from solver.genealogy (multi-monitor?) [1D,2D,3D?]
    # - should be able to scale 'z-axis' instead of scaling 'z' itself
    #   (see https://github.com/matplotlib/matplotlib/issues/209)
    # - if trajectory outside contour grid, will increase bounds
    #   (see support_hypercube.py for how to fix bounds)
    import shlex
    try:
        basestring
        from StringIO import StringIO
    except NameError:
        basestring = str
        from io import StringIO
    global __quit
    __quit = False
    _model = None
    _reducer = None
    _solver = None

    instance = None
    # handle the special case where list is provided by sys.argv
    if isinstance(model, (list,tuple)) and not logfile and not kwds:
        cmdargs = model # (above is used by script to parse command line)
    elif isinstance(model, basestring) and not logfile and not kwds:
        cmdargs = shlex.split(model)
    # 'everything else' is essentially the functional interface
    else:
        cmdargs = kwds.get('kwds', '')
        if not cmdargs:
            out = kwds.get('out', None)
            bounds = kwds.get('bounds', None)
            label = kwds.get('label', None)
            nid = kwds.get('nid', None)
            iter = kwds.get('iter', None)
            reduce = kwds.get('reduce', None)
            scale = kwds.get('scale', None)
            shift = kwds.get('shift', None)
            fill = kwds.get('fill', False)
            depth = kwds.get('depth', False)
            dots = kwds.get('dots', False)
            join = kwds.get('join', False)
            verb = kwds.get('verb', False)

            # special case: bounds passed as list of slices
            if not isinstance(bounds, (basestring, type(None))):
                cmdargs = ''
                for b in bounds:
                    if isinstance(b, slice):
                        cmdargs += "{}:{}:{}, ".format(b.start, b.stop, b.step)
                    else:
                        cmdargs += "{}, ".format(b)
                bounds = cmdargs[:-2]
                cmdargs = ''

            if isinstance(reduce, collections.Callable): _reducer, reduce = reduce, None

        # special case: model passed as model instance
       #model.__doc__.split('using::')[1].split()[0].strip()
        if isinstance(model, collections.Callable): _model, model = model, "None"

        # handle logfile if given
        if logfile:
            if isinstance(logfile, basestring):
                model += ' ' + logfile
            else: # special case of passing in monitor instance
                instance = logfile

        # process "commandline" arguments
        if not cmdargs:
            cmdargs = ''
            cmdargs += '' if out is None else '--out={} '.format(out)
            cmdargs += '' if bounds is None else '--bounds="{}" '.format(bounds)
            cmdargs += '' if label is None else '--label={} '.format(label)
            cmdargs += '' if nid is None else '--nid={} '.format(nid)
            cmdargs += '' if iter is None else '--iter={} '.format(iter)
            cmdargs += '' if reduce is None else '--reduce={} '.format(reduce)
            cmdargs += '' if scale is None else '--scale={} '.format(scale)
            cmdargs += '' if shift is None else '--shift={} '.format(shift)
            cmdargs += '' if fill == False else '--fill '
            cmdargs += '' if depth == False else '--depth '
            cmdargs += '' if dots == False else '--dots '
            cmdargs += '' if join == False else '--join '
            cmdargs += '' if verb == False else '--verb '
        else:
            cmdargs = ' ' + cmdargs
        cmdargs = model.split() + shlex.split(cmdargs)

    #XXX: note that 'argparse' is new as of python2.7
    from optparse import OptionParser
    def _exit(self, errno=None, msg=None):
      global __quit
      __quit = True
      if errno or msg:
        msg = msg.split(': error: ')[-1].strip()
        raise IOError(msg)
    OptionParser.exit = _exit

    parser = OptionParser(usage=model_plotter.__doc__.split('\n\nOptions:')[0])
    parser.add_option("-u","--out",action="store",dest="out",\
                      metavar="STR",default=None,
                      help="filepath to save generated plot")
    parser.add_option("-b","--bounds",action="store",dest="bounds",\
                      metavar="STR",default="-5:5:.1, -5:5:.1",
                      help="indicator string to set plot bounds and density")
    parser.add_option("-l","--label",action="store",dest="label",\
                      metavar="STR",default=",,",
                      help="string to assign label to axis")
    parser.add_option("-n","--nid",action="store",dest="id",\
                      metavar="INT",default=None,
                      help="id # of the nth simultaneous points to plot")
    parser.add_option("-i","--iter",action="store",dest="stop",\
                      metavar="STR",default=":",
                      help="string for smallest:largest iterations to plot")
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
    parser.add_option("-v","--verb",action="store_true",dest="verbose",\
                      default=False,help="print model documentation string")

#   import sys
#   if 'mystic_model_plotter.py' not in sys.argv:
    if PY3:
      f = StringIO()
      parser.print_help(file=f)
      f.seek(0)
      if 'Options:' not in model_plotter.__doc__:
        model_plotter.__doc__ += '\nOptions:%s' % f.read().split('Options:')[-1]
      f.close()
    else:
      if 'Options:' not in model_plotter.__doc__:
        model_plotter.__doc__ += '\nOptions:%s' % parser.format_help().split('Options:')[-1]

    try:
      parsed_opts, parsed_args = parser.parse_args(cmdargs)
    except UnboundLocalError:
      pass
    if __quit: return

    # get the import path for the model
    model = parsed_args[0]  # e.g. 'mystic.models.rosen'
    if "None" == model: model = None

    try: # get the name of the parameter log file
      source = parsed_args[1]  # e.g. 'log.txt'
    except:
      source = None

    try: # select the bounds
      options = parsed_opts.bounds  # format is "-1:10:.1, -1:10:.1, 1.0"
    except:
      options = "-5:5:.1, -5:5:.1"

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

    try: # select which 'id' to plot results for
      ids = (int(parsed_opts.id),) #XXX: allow selecting more than one id ?
    except:
      ids = None # i.e. 'all'

    try: # select which iteration to stop plotting at
      stop = parsed_opts.stop  # format is "1:10:1"
      stop = stop if ":" in stop else ":"+stop
    except:
      stop = ":"

    try: # select whether to be verbose about model documentation
      verbose = bool(parsed_opts.verbose)
    except:
      verbose = False

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
    if _model: model = _model
    if _reducer: reducer = _reducer
    if _solver: solver = _solver
    select, spec, mask = _parse_input(options)
    x,y = _parse_axes(spec, grid=True) # grid=False for 1D plots
    #FIXME: does grid=False still make sense here...?
    if reducer: reducer = _reducer or _get_instance(reducer)
    if solver and (not source or not model): #XXX: not instance?
        raise RuntimeError('a model and results filename are required')
    elif not source and not model and not instance:
        raise RuntimeError('a model or a results file is required')
    if model:
        model = _model or _get_instance(model)
        if verbose: print(model.__doc__)
        # need a reducer if model returns an array
        if reducer: model = reduced(reducer, arraylike=False)(model)

    if solver:
        # if 'live'... pick a solver
        solver = 'mystic.solvers.fmin'
        solver = _solver or _get_instance(solver)
        xlen = len(select)+len(mask)
        if solver.__name__.startswith('diffev'):
            initial = [(-1,1)]*xlen
        else:
            initial = [0]*xlen
        from mystic.monitors import VerboseLoggingMonitor
        if instance:
            itermon = VerboseLoggingMonitor(new=True)
            itermon.prepend(instance)
        else:
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

       ## plot the surface in 1D
       #if solver: v=sol[-1]
       #elif source: v=cost[-1]
       #else: v=None
       #fig0 = _draw_slice(model, x=x, y=v, scale=scale, shift=shift)
        # plot the surface in 2D or 3D
        fig = _draw_contour(model, x, y, surface=surface, fill=fill, scale=scale, shift=shift)
    else:
       #fig0 = None
        fig = None

    if instance: source = instance
    if source:
        # params are the parameter trajectories
        # cost is the solution trajectory
        params, cost = _get_history(source, ids)
        if len(cost) > 1: style = style[1:] # 'auto-color' #XXX: or grayscale?

        for p,c in zip(params, cost):
           ## project trajectory on a 1D slice of model surface #XXX: useful?
           #s = select[0] if len(select) else 0
           #px = p[int(s)] # _draw_projection requires one parameter
           ## ignore everything after 'stop'
           #_c = eval('c[%s]' % stop)
           #_x = eval('px[%s]' % stop)
           #fig0 = _draw_projection(_x,_c, style=style, scale=scale, shift=shift, figure=fig0)

            # plot the trajectory on the model surface (2D or 3D)
            # get two selected params #XXX: what if len(select)<2? or len(p)<2?
            p = [p[int(i)] for i in select[:2]]
            px,py = p # _draw_trajectory requires two parameters
            # ignore everything after 'stop'
            locals = dict(px=px, py=py, c=c)
            _x = eval('px[%s]' % stop, locals)
            _y = eval('py[%s]' % stop, locals)
            _c = eval('c[%s]' % stop, locals) if surface else None
            fig = _draw_trajectory(_x,_y,_c, style=style, scale=scale, shift=shift, figure=fig)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    # add labels to the axes
    if surface: kwds = {'projection':'3d'} # 3D
    else: kwds = {}                        # 2D
    ax = fig.gca(**kwds)
    ax.set_xlabel(label[0])
    ax.set_ylabel(label[1])
    if surface: ax.set_zlabel(label[2])

    if not parsed_opts.out:
        plt.show()
    else:
        fig.savefig(parsed_opts.out)


def collapse_plotter(filename, **kwds):
    """
generate cost convergence rate plots from file written with ``write_support_file``

Available from the command shell as::

    mystic_collapse_plotter.py filename [options]

or as a function call::

    mystic.collapse_plotter(filename, **options)

Args:
    filename (str): name of the convergence logfile (e.g ``paramlog.py``).

Returns:
    None

Notes:
    - The option *dots* takes a boolean, and will show data points in the plot.
    - The option *linear* takes a boolean, and will plot in a linear scale.
    - The option *out* takes a string of the filepath for the generated plot.
    - The option *iter* takes an integer of the largest iteration to plot.
    - The option *label* takes a label string. For example, ``label = "y"``
      will label the plot with a 'y', while ``label = " log-cost,
      $ log_{10}(\hat{P} - \hat{P}_{max})$"`` will label the y-axis with
      standard LaTeX math formatting. Note that the leading space is required,
      and that the text is aligned along the axis.
    - The option *col* takes a string of comma-separated integers indicating
      iteration numbers where parameter collapse has occurred. If a second set
      of integers is provided (delineated by a semicolon), the additional set
      of integers will be plotted with a different linestyle (to indicate a
      different type of collapse).
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
            dots = kwds.get('dots', False)
           #line = kwds.get('line', False)
            linear = kwds.get('linear', False)
            iter = kwds.get('iter', None)
            label = kwds.get('label', None)
            col = kwds.get('col', None)

            # process "commandline" arguments
            cmdargs = ''
            cmdargs += '' if out is None else '--out={} '.format(out)
            cmdargs += '' if dots == False else '--dots '
            cmdargs += '' if linear == False else '--linear '
           #cmdargs += '' if line == False else '--line '
            cmdargs += '' if iter is None else '--iter={} '.format(iter)
            cmdargs += '' if label == None else '--label={} '.format(label)
            cmdargs += '' if col is None else '--col="{}" '.format(col)
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

    parser = OptionParser(usage=collapse_plotter.__doc__.split('\n\nOptions:')[0])
    parser.add_option("-d","--dots",action="store_true",dest="dots",\
                      default=False,help="show data points in plot")
    #parser.add_option("-l","--line",action="store_true",dest="line",\
    #                  default=False,help="connect data points with a line")
    parser.add_option("-y","--linear",action="store_true",dest="linear",\
                      default=False,help="plot y-axis in linear scale")
    parser.add_option("-u","--out",action="store",dest="out",\
                      metavar="STR",default=None,
                      help="filepath to save generated plot")
    parser.add_option("-i","--iter",action="store",dest="stop",metavar="INT",\
                      default=None,help="the largest iteration to plot")
    parser.add_option("-l","--label",action="store",dest="label",\
                      metavar="STR",default="",\
                      help="string to assign label to y-axis")
    parser.add_option("-c","--col",action="store",dest="collapse",\
                      metavar="STR",default="",
                      help="string to indicate collapse indices")
#   import sys
#   if 'mystic_collapse_plotter.py' not in sys.argv:
    if PY3:
      f = StringIO()
      parser.print_help(file=f)
      f.seek(0)
      if 'Options:' not in collapse_plotter.__doc__:
        collapse_plotter.__doc__ += '\nOptions:%s' % f.read().split('Options:')[-1]
      f.close()
    else:
      if 'Options:' not in collapse_plotter.__doc__:
        collapse_plotter.__doc__ += '\nOptions:%s' % parser.format_help().split('Options:')[-1]

    try:
      parsed_opts, parsed_args = parser.parse_args(cmdargs)
    except UnboundLocalError:
      pass
    if __quit: return

    style = '-' # default linestyle
    if parsed_opts.dots:
      mark = 'o'
      # when using 'dots', also turn off 'line'
      #if not parsed_opts.line:
      #  style = 'None'
    else:
      mark = ''

    try: # select labels for the axes
      label = parsed_opts.label  # format is "x" or " $x$"
    except:
      label = 'log-cost, $log_{10}(y - y_{min})$'

    try: # get logfile name
      filename = parsed_args[0]
    except:
      raise IOError("please provide log file name")

    try: # select which iteration to stop plotting at
      stop = int(parsed_opts.stop)
    except:
      stop = None

    try: # select collapse boundaries to plot
      collapse = parsed_opts.collapse.split(';')  # format is "2, 3; 4, 5, 6; 7"
      collapse = [eval("(%s,)" % i) if i.strip() else () for i in collapse]
    except:
      collapse = []

    # read file
    from mystic.munge import read_history
    params, cost = read_history(filename)

    # ignore everything after 'stop'
    cost = cost[:stop]
    params = params[:stop]

    # get the minimum cost
    import numpy as np
    cost_min = min(cost)

    # convert to log scale
    x = np.arange(len(cost))
    settings = np.seterr(all='ignore')
    if parsed_opts.linear:
      y = np.array(cost)
     #y = np.abs(cost_min - np.array(cost))
    else:
      y = np.log10(np.abs(cost_min - np.array(cost)))
    np.seterr(**settings)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(x, y, linestyle=style, marker=mark, markersize=1)

    colors = ['orange','red','brown','pink']
    linestyles = ['--','-.',':','-']

    for param,color,style in zip(collapse,colors,linestyles):
      for clps in set(param):
        plt.axvline(x=clps, ymin=-10, ymax=1, hold=None, linestyle=style, linewidth=param.count(clps), color=color)

    if label:
        #plt.title('convergence rate')
        plt.xlabel('iteration number, $n$')
        plt.ylabel(label)
        #plt.ylabel('$log-error,\; log_{10}(\hat{P} - \hat{P}_{max})$')

    if not parsed_opts.out:
        plt.show()
    else:
        fig.savefig(parsed_opts.out)


def log_reader(filename, **kwds):
    """
plot parameter convergence from file written with ``LoggingMonitor``

Available from the command shell as::

    mystic_log_reader.py filename [options]

or as a function call::

    mystic.log_reader(filename, **options)

Args:
    filename (str): name of the convergence logfile (e.g ``log.txt``).

Returns:
    None

Notes:
    - The option *out* takes a string of the filepath for the generated plot.
    - The option *dots* takes a boolean, and will show data points in the plot.
    - The option *line* takes a boolean, and will connect the data with a line.
    - The option *iter* takes an integer of the largest iteration to plot.
    - The option *legend* takes a boolean, and will display the legend.
    - The option *nid* takes an integer of the nth simultaneous points to plot.
    - The option *param* takes an indicator string. The indicator string is
      built from comma-separated array slices. For example, ``params = ":"``
      will plot all parameters.  Alternatively, ``params = ":2, 3:"`` will plot      all parameters except for the third parameter, while ``params = "0"``
      will only plot the first parameter.
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
            dots = kwds.get('dots', False)
            line = kwds.get('line', False)
            iter = kwds.get('iter', None)
            legend = kwds.get('legend', False)
            nid = kwds.get('nid', None)
            param = kwds.get('param', None)

            # process "commandline" arguments
            cmdargs = ''
            cmdargs += '' if out is None else '--out={} '.format(out)
            cmdargs += '' if dots == False else '--dots '
            cmdargs += '' if line == False else '--line '
            cmdargs += '' if iter is None else '--iter={} '.format(iter)
            cmdargs += '' if legend == False else '--legend '
            cmdargs += '' if nid is None else '--nid={} '.format(nid)
            cmdargs += '' if param is None else '--param="{}" '.format(param)
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

    parser = OptionParser(usage=log_reader.__doc__.split('\n\nOptions:')[0])
    parser.add_option("-u","--out",action="store",dest="out",\
                      metavar="STR",default=None,
                      help="filepath to save generated plot")
    parser.add_option("-d","--dots",action="store_true",dest="dots",\
                      default=False,help="show data points in plot")
    parser.add_option("-l","--line",action="store_true",dest="line",\
                      default=False,help="connect data points in plot with a line")
    parser.add_option("-i","--iter",action="store",dest="stop",metavar="INT",\
                      default=None,help="the largest iteration to plot")
    parser.add_option("-g","--legend",action="store_true",dest="legend",\
                      default=False,help="show the legend")
    parser.add_option("-n","--nid",action="store",dest="id",\
                      metavar="INT",default=None,
                      help="id # of the nth simultaneous points to plot")
    parser.add_option("-p","--param",action="store",dest="param",\
                      metavar="STR",default=":",
                      help="indicator string to select parameters")
    #parser.add_option("-f","--file",action="store",dest="filename",metavar="FILE",\
    #                  default='log.txt',help="log file name")

#   import sys
#   if 'mystic_log_reader.py' not in sys.argv:
    if PY3:
      f = StringIO()
      parser.print_help(file=f)
      f.seek(0)
      if 'Options:' not in log_reader.__doc__:
        log_reader.__doc__ += '\nOptions:%s' % f.read().split('Options:')[-1]
      f.close()
    else:
      if 'Options:' not in log_reader.__doc__:
        log_reader.__doc__ += '\nOptions:%s' % parser.format_help().split('Options:')[-1]

    try:
      parsed_opts, parsed_args = parser.parse_args(cmdargs)
    except UnboundLocalError:
      pass
    if __quit: return

    style = '-' # default linestyle
    if parsed_opts.dots:
      mark = 'o'
      # when using 'dots', also can turn off 'line'
      if not parsed_opts.line:
        style = 'None'
    else:
      mark = ''

    try: # get logfile name
      if instance:
        filename = instance
      else:
        filename = parsed_args[0]
    except:
      raise IOError("please provide log file name")

    try: # select which iteration to stop plotting at
      stop = int(parsed_opts.stop)
    except:
      stop = None

    try: # select which 'id' to plot results for
      runs = (int(parsed_opts.id),) #XXX: allow selecting more than one id ?
    except:
      runs = None # i.e. 'all' **or** use id=0, which should be 'best' energy ?

    try: # select which parameters to plot
      select = parsed_opts.param.split(',')  # format is ":2, 2:4, 5, 6:"
    except:
      select = [':']

    # ensure all terms of select have a ":"
    for i in range(len(select)):
      if isinstance(select[i], int): select[i] = str(select[i])
      if select[i] == '-1': select[i] = 'len(params)-1:len(params)'
      elif not select[i].count(':'):
        select[i] += ':' + str(int(select[i])+1)


    # == Possible results ==
    # iter = (i,id) or (i,) 
    # split => { (i,) then (i+1,) } or { (i,) then (0,) }
    # y x = { float list } or { list [list1, ...] }

    # == Use Cases ==
    # (i,id) + { (i,) then (i+1,) } + { float list }
    # (i,) + { (i,) then (i+1,) } + { float list }
    # (i,id) + { (i,) then (i+1,) } + { list [list1, ...] }
    # (i,) + { (i,) then (i+1,) } + { list [list1, ...] }
    # (i,id) + { (i,) then (0,) } + { float list }
    # (i,) + { (i,) then (0,) } + { float list }
    # (i,id) + { (i,) then (0,) } + { list [list1, ...] }
    # (i,) + { (i,) then (0,) } + { list [list1, ...] }
    # NOTES:
    #   Legend is different for list versus [list1,...]
    #   Plot should be discontinuous for (i,) then (0,)

    # parse file contents to get (i,id), cost, and parameters
    try:
        instance = instance if instance else filename
        from mystic.munge import read_trajectories
        step, param, cost = read_trajectories(instance)
    except SyntaxError:
        from mystic.munge import read_raw_file
        read_raw_file(filename)
        msg = "incompatible file format, try 'support_convergence %s'" % filename
        raise SyntaxError(msg)

    # ignore everything after 'stop'
    step = step[:stop]
    cost = cost[:stop]
    param = param[:stop]

    # split (i,id) into iteration and id
    multinode = len(step[0]) - 1  #XXX: what if step = []?
    iter = [i[0] for i in step]
    if multinode:
      id = [i[1] for i in step]
    else:
      id = [0 for i in step]

    # build the list of selected parameters
    params = list(range(len(param[0])))
    selected = []
    locals = dict(params=params)
    for i in select:
      selected.extend(eval("params[%s]" % i, locals))
    selected = list(set(selected))

    results = [[] for i in range(max(id) + 1)]

    # populate results for each id with the corresponding (iter,cost,param)
    for i in range(len(id)):
      if runs is None or id[i] in runs: # take only the selected 'id'
        results[id[i]].append((iter[i],cost[i],param[i]))
    # NOTE: for example...  results = [[(0,...)],[(0,...),(1,...)],[],[(0,...)]]

    # build list of parameter (and cost) convergences for each id
    conv = []; cost_conv = []; iter_conv = []
    for i in range(len(results)):
      conv.append([])#; cost_conv.append([]); iter_conv.append([])
      if len(results[i]):
        for k in range(len(results[i][0][2])):
          conv[i].append([results[i][j][2][k] for j in range(len(results[i]))])
        cost_conv.append([results[i][j][1] for j in range(len(results[i]))])
        iter_conv.append([results[i][j][0] for j in range(len(results[i]))])
      else:
        conv[i] = [[] for k in range(len(param[0]))]
        cost_conv.append([])
        iter_conv.append([])

    #print("iter_conv = %s" % iter_conv)
    #print("cost_conv = %s" % cost_conv)
    #print("conv = %s" % conv)

    import matplotlib.pyplot as plt
    fig = plt.figure()

    #FIXME: These may fail when conv[i][j] = [[],[],[]] and cost = []. Verify this.
    ax1 = fig.add_subplot(2,1,1)
    for i in range(len(conv)):
      if runs is None or i in runs: # take only the selected 'id'
        for j in range(len(param[0])):
          if j in selected: # take only the selected 'params'
            tag = "%d,%d" % (j,i) # label is 'parameter,id'
            ax1.plot(iter_conv[i],conv[i][j],label="%s" % tag,marker=mark,linestyle=style)
    if parsed_opts.legend: plt.legend()

    ax2 = fig.add_subplot(2,1,2)
    for i in range(len(conv)):
      if runs is None or i in runs: # take only the selected 'id'
        tag = "%d" % i # label is 'cost id'
        ax2.plot(iter_conv[i],cost_conv[i],label='cost %s' % tag,marker=mark,linestyle=style)
    if parsed_opts.legend: plt.legend()

    if not parsed_opts.out:
        plt.show()
    else:
        fig.savefig(parsed_opts.out)


# initialize doc
try: log_reader()
except TypeError:
    pass
try: model_plotter()
except TypeError:
    pass
try: collapse_plotter()
except TypeError:
    pass



if __name__ == '__main__':
    pass


# EOF

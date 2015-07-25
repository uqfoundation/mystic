#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
__doc__ = """
functional interfaces for mystic's visual diagnistics for support files
"""

__all__ = ['convergence', 'hypercube', 'hypercube_measures']

from mpl_toolkits.mplot3d import Axes3D as _Axes3D
from matplotlib.axes import subplot_class_factory
Subplot3D = subplot_class_factory(_Axes3D)

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

from mystic.munge import read_history
from mystic.munge import logfile_reader, raw_to_support
from mystic.tools import factor, flatten

# globals
__quit = False


def best_dimensions(n):
  "get the 'best' dimensions (n x m) for arranging plots"
  allfactors = list(factor(n))
  from numpy import product
  cand = [1] + [product(allfactors[:i+1]) for i in range(len(allfactors))]
 #return cand[-1], n/cand[-1]
  best = [cand[len(cand)/2], n/cand[len(cand)/2]]
  best.sort(reverse=True)
  return tuple(best)
# if len(cand)%2:
#   return cand[len(cand)/2], cand[len(cand)/2]
# return cand[len(cand)/2], cand[len(cand)/2 - 1]


def convergence(filename, **kwds):
    """
generate parameter convergence plots from file written with 'write_support_file'

Available from the command shell as:
  support_convergence.py filename [options]

or as a function call as:
  mystic.support.convergence(filename, **options)

The option "param" takes an indicator string. The indicator string is built
from comma-separated array slices. For example, params = ":" will plot all
parameters in a single plot.  Alternatively, params = ":2, 2:" will split the
parameters into two plots, and params = "0" will only plot the first parameter.

The option "label" takes comma-separated strings. For example, label = "x,y,"
will label the y-axis of the first plot with 'x', a second plot with 'y', and
not add a label to a third or subsequent plots. If more labels are given than
plots, then the last label will be used for the y-axis of the 'cost' plot.
LaTeX is also accepted. For example, label = "$ h$, $ {\\alpha}$, $ v$" will
label the axes with standard LaTeX math formatting. Note that the leading
space is required, and the text is aligned along the axis.

Required Inputs:
  filename            name of the python convergence logfile (e.g paramlog.py)
"""
    import shlex
    global __quit
    __quit = False

    # handle the special case where list is provided by sys.argv
    if isinstance(filename, (list,tuple)) and not kwds:
        cmdargs = filename # (above is used by script to parse command line)
    elif isinstance(filename, basestring) and not kwds:
        cmdargs = shlex.split(filename)
    # 'everything else' is essentially the functional interface
    else:
        out = kwds.get('out', None)
        iter = kwds.get('iter', None)
        param = kwds.get('param', None)
        label = kwds.get('label', None)
        nid = kwds.get('nid', None)
        cost = kwds.get('cost', False)
        legend = kwds.get('legend', False)

        # process "commandline" arguments
        cmdargs = ''
        cmdargs += '' if out is None else '--out={} '.format(out)
        cmdargs += '' if iter is None else '--iter={} '.format(iter)
        cmdargs += '' if param is None else '--param="{}" '.format(param)
        cmdargs += '' if label is None else '--label="{}" '.format(label)
        cmdargs += '' if nid is None else '--nid={} '.format(nid)
        cmdargs += '' if cost == False else '--cost '
        cmdargs += '' if legend == False else '--legend '
        cmdargs = filename.split() + shlex.split(cmdargs)

    #XXX: note that 'argparse' is new as of python2.7
    from optparse import OptionParser
    def _exit(self, **kwds):
        global __quit
        __quit = True
    OptionParser.exit = _exit

    parser = OptionParser(usage=convergence.__doc__.split('\n\nOptions:')[0])
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
    parsed_opts, parsed_args = parser.parse_args(cmdargs)

    from StringIO import StringIO
    f = StringIO()
    parser.print_help(file=f)
    f.seek(0)
    if 'Options:' not in convergence.__doc__:
        convergence.__doc__ += '\nOptions:%s' % f.read().split('Options:')[-1]
    f.close()

    if __quit: return

    # get the name of the parameter log file
    params, cost = read_history(parsed_args[0])

    if parsed_opts.cost: # also plot the cost
       #exec "from %s import cost" % file
        pass
    else:
        cost = None

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

    if cost: j = 1
    else: j = 0
    dim1,dim2 = best_dimensions(plots + j)

    fig = plt.figure()
    ax1 = fig.add_subplot(dim1,dim2,1)
    ax1.set_ylabel(label[0])
    data = eval("params[%s]" % select[0])
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
        code = "ax%d = fig.add_subplot(dim1,dim2,%d, sharex=ax1);" % (i,i)
        code += "ax%d.set_ylabel(label[%d]);" % (i,i-1)
        code = compile(code, '<string>', 'exec')
        exec code in globals
        data = eval("params[%s]" % select[i-1])
        try:
            n = int(select[i-1].split(":")[0])
        except ValueError:
            n = 0
        for line in data:
            globals['line'] = line
            code = "ax%d.plot(line,label='%s')#, marker='o')" % (i,n)
            code = compile(code, '<string>', 'exec')
            exec code in globals
            n += 1
        if legend: plt.legend()
    if cost:
        globals['cost'] = cost
        code = "cx1 = fig.add_subplot(dim1,dim2,%d, sharex=ax1);" % int(plots+1)
        code += "cx1.plot(cost,label='cost');"#, marker='o')"
        if max(0, len(label) - plots): code += "cx1.set_ylabel(label[-1]);"
        code = compile(code, '<string>', 'exec')
        exec code in globals
        if legend: plt.legend()

    if not parsed_opts.out:
        plt.show()
    else:
        fig.savefig(parsed_opts.out)

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


def hypercube(filename, **kwds):
    """
generate parameter support plots from file written with 'write_support_file'

Available from the command shell as:
  support_hypercube.py filename [options]

or as a function call as:
  mystic.support.hypercube(filename, **options)

The options "bounds", "axes", and "iters" all take indicator strings.
The bounds should be given as comma-separated slices. For example, using
bounds = "60:105, 0:30, 2.1:2.8" will set the lower and upper bounds for
x to be (60,105), y to be (0,30), and z to be (2.1,2.8).  Similarly, axes
also accepts comma-separated groups of ints; however, for axes, each entry
indicates which parameters are to be plotted along each axis -- the first
group for the x direction, the second for the y direction, and third for z. 
Thus, axes = "2 3, 6 7, 10 11" would set 2nd and 3rd parameters along x.
Iters also accepts a string built from comma-separated array slices. For
example, iters = ":" will plot all iters in a single plot. Alternatively,
iters = ":2, 2:" will split the iters into two plots, while iters = "0" will
only plot the first iteration.

The option "label" takes comma-separated strings. For example, label = "x,y,"
will place 'x' on the x-axis, 'y' on the y-axis, and nothing on the z-axis.
LaTeX is also accepted. For example, label = "$ h $, $ {\\alpha}$, $ v$" will
label the axes with standard LaTeX math formatting. Note that the leading
space is required, while a trailing space aligns the text with the axis
instead of the plot frame.

Required Inputs:
  filename            name of the python convergence logfile (e.g paramlog.py)
"""
    import shlex
    global __quit
    __quit = False

    # handle the special case where list is provided by sys.argv
    if isinstance(filename, (list,tuple)) and not kwds:
        cmdargs = filename # (above is used by script to parse command line)
    elif isinstance(filename, basestring) and not kwds:
        cmdargs = shlex.split(filename)
    # 'everything else' is essentially the functional interface
    else:
        out = kwds.get('out', None)
        bounds = kwds.get('bounds', None)
        axes = kwds.get('axes', None)
        iters = kwds.get('iters', None)
        label = kwds.get('label', None)
        nid = kwds.get('nid', None)
        scale = kwds.get('scale', None)
        flat = kwds.get('flat', False)

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
        cmdargs = filename.split() + shlex.split(cmdargs)

    #XXX: note that 'argparse' is new as of python2.7
    from optparse import OptionParser
    def _exit(self, **kwds):
        global __quit
        __quit = True
    OptionParser.exit = _exit

    parser = OptionParser(usage=hypercube.__doc__.split('\n\nOptions:')[0])
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
    parsed_opts, parsed_args = parser.parse_args(cmdargs)

    from StringIO import StringIO
    f = StringIO()
    parser.print_help(file=f)
    f.seek(0)
    if 'Options:' not in hypercube.__doc__:
        hypercube.__doc__ += '\nOptions:%s' % f.read().split('Options:')[-1]
    f.close()

    if __quit: return

    # get the name of the parameter log file
    params, _cost = read_history(parsed_args[0])
    # would be nice to use meta = ['wx','wx2','x','x2','wy',...]
    # exec "from %s import meta" % file

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
    fig = plt.figure()
    ax1 = Subplot3D(fig, dim1,dim2,1)
    ax1.plot([bounds[0][0]],[bounds[1][0]],[bounds[2][0]])
    ax1.plot([bounds[0][1]],[bounds[1][1]],[bounds[2][1]])
    globals = {'plt':plt,'fig':fig,'dim1':dim1,'dim2':dim2,\
               'Subplot3D':Subplot3D,'bounds':bounds,'label':label}
    if not flatten:
        code = "plt.title('iterations[%s]');" % select[0]
    else: 
        code = "plt.title('iterations[*]');"
    code = compile(code, '<string>', 'exec')
    exec code in globals
    ax1.set_xlabel(label[0])
    ax1.set_ylabel(label[1])
    ax1.set_zlabel(label[2])
    a = [ax1]

    # set up additional plots
    if not flatten:
        for i in range(2, plots + 1):
            code = "ax = ax%d = Subplot3D(fig, dim1,dim2,%d);" % (i,i)
            code += "ax.plot([bounds[0][0]],[bounds[1][0]],[bounds[2][0]]);"
            code += "ax.plot([bounds[0][1]],[bounds[1][1]],[bounds[2][1]]);"
            code += "plt.title('iterations[%s]');" % select[i - 1]
            code += "ax.set_xlabel(label[0]);"
            code += "ax.set_ylabel(label[1]);"
            code += "ax.set_zlabel(label[2]);"
            code = compile(code, '<string>', 'exec')
            exec code in globals
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
    steps = [eval("range(%s)" % sel.replace(":",",")) for sel in select]

    # at this point, we should have:
    #xyz = [(0,1),(4,5),(8,9)] for any length tuple
    #steps = [[0,1],[1,2],[2,3],[3,4,5,6,7,8]] or similar
    if flatten:
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
            for i in eval("[params[q][%s] for q in xyz[0]]" % s):
                for j in eval("[params[q][%s] for q in xyz[1]]" % s):
                    for k in eval("[params[q][%s] for q in xyz[2]]" % s):
                        a[v].plot(i,j,k,marker='o',color=t,ms=10)

    if not parsed_opts.out:
        plt.show()
    else:
        fig.savefig(parsed_opts.out)


def hypercube_measures(filename, **kwds):
    """
generate measure support plots from file written with 'write_support_file'

Available from the command shell as:
  support_hypercube_measures.py filename [options]

or as a function call as:
  mystic.support.hypercube_measures(filename, **options)

The options "bounds", "axes", 'weight", and "iters" all take indicator strings.
The bounds should be given as comma-separated slices. For example, using
bounds = "60:105, 0:30, 2.1:2.8" will set the lower and upper bounds for
x to be (60,105), y to be (0,30), and z to be (2.1,2.8).  Similarly, axes
also accepts comma-separated groups of ints; however, for axes, each entry
indicates which parameters are to be plotted along each axis -- the first
group for the x direction, the second for the y direction, and third for z. 
Thus, axes = "2 3, 6 7, 10 11" would set 2nd and 3rd parameters along x. The
corresponding weights are used to color the measure points, where 1.0 is black
and 0.0 is white. For example, using weight = "0 1, 4 5, 8 9" would use
the 0th and 1st parameters to weight x. Iters is also similar, however only
accepts comma-separated ints. Hence, iters = "-1" will plot the last iteration,
while iters = "0, 300, 700" will plot the 0th, 300th, and 700th in three plots.
***Note that if weights are not normalized (to 1), an error will be thrown.***

The option "label" takes comma-separated strings. For example, label = "x,y,"
will place 'x' on the x-axis, 'y' on the y-axis, and nothing on the z-axis.
LaTeX is also accepted. For example, label = "$ h $, $ {\\alpha}$, $ v$" will
label the axes with standard LaTeX math formatting. Note that the leading
space is required, while a trailing space aligns the text with the axis
instead of the plot frame.

INTENDED FOR VISUALIZING WEIGHTED MEASURES (i.e. weights and positions)

Required Inputs:
  filename            name of the python convergence logfile (e.g paramlog.py)
"""
    import shlex
    global __quit
    __quit = False

    # handle the special case where list is provided by sys.argv
    if isinstance(filename, (list,tuple)) and not kwds:
        cmdargs = filename # (above is used by script to parse command line)
    elif isinstance(filename, basestring) and not kwds:
        cmdargs = shlex.split(filename)
    # 'everything else' is essentially the functional interface
    else:
        out = kwds.get('out', None)
        bounds = kwds.get('bounds', None)
        axes = kwds.get('axes', None)
        weight = kwds.get('weight', None)
        iters = kwds.get('iters', None)
        label = kwds.get('label', None)
        nid = kwds.get('nid', None)
        scale = kwds.get('scale', None)
        flat = kwds.get('flat', False)

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
        cmdargs = filename.split() + shlex.split(cmdargs)

    #XXX: note that 'argparse' is new as of python2.7
    from optparse import OptionParser
    def _exit(self, **kwds):
        global __quit
        __quit = True
    OptionParser.exit = _exit

    parser = OptionParser(usage=hypercube_measures.__doc__.split('\n\nOptions:')[0])
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
    parsed_opts, parsed_args = parser.parse_args(cmdargs)

    from StringIO import StringIO
    f = StringIO()
    parser.print_help(file=f)
    f.seek(0)
    if 'Options:' not in hypercube_measures.__doc__:
        hypercube_measures.__doc__ += '\nOptions:%s' % f.read().split('Options:')[-1]
    f.close()

    if __quit: return

    # get the name of the parameter log file
    params, _cost = read_history(parsed_args[0])
    # would be nice to use meta = ['wx','wx2','x','x2','wy',...]
    # exec "from %s import meta" % file

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
    fig = plt.figure()
    ax1 = Subplot3D(fig, dim1,dim2,1)
    ax1.plot([bounds[0][0]],[bounds[1][0]],[bounds[2][0]])
    ax1.plot([bounds[0][1]],[bounds[1][1]],[bounds[2][1]])
    globals = {'plt':plt,'fig':fig,'dim1':dim1,'dim2':dim2,\
               'Subplot3D':Subplot3D,'bounds':bounds,'label':label}
    if not flatten:
        code = "plt.title('iterations[%s]');" % select[0]
    else: 
        code = "plt.title('iterations[*]');"
    code = compile(code, '<string>', 'exec')
    exec code in globals
    ax1.set_xlabel(label[0])
    ax1.set_ylabel(label[1])
    ax1.set_zlabel(label[2])
    a = [ax1]

    # set up additional plots
    if not flatten:
        for i in range(2, plots + 1):
            code = "ax = ax%d = Subplot3D(fig, dim1,dim2,%d);" % (i,i)
            code += "ax.plot([bounds[0][0]],[bounds[1][0]],[bounds[2][0]]);"
            code += "ax.plot([bounds[0][1]],[bounds[1][1]],[bounds[2][1]]);"
            code += "plt.title('iterations[%s]');" % select[i - 1]
            code += "ax.set_xlabel(label[0]);"
            code += "ax.set_ylabel(label[1]);"
            code += "ax.set_zlabel(label[2]);"
            code = compile(code, '<string>', 'exec')
            exec code in globals
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
   #steps = [eval("range(%s)" % sel.replace(":",",")) for sel in select]
    steps = [eval("[int(%s)]" % sel) for sel in select]

    # at this point, we should have:
    #xyz = [(2,3),(6,7),(10,11)] for any length tuple
    #wxyz = [(0,1),(4,5),(8,9)] for any length tuple (should match up with xyz)
    #steps = [[0],[1],[3],[8]] or similar
    if flatten:
        steps = [list(flatten(steps))]

    # adjust for logarithmic scaling of intensity
    from numpy import e
    scale = e**(scale - 1.0)

    # dot color is based on a product of weights
    t = []
    for v in range(len(steps)):
        t.append([])
        for s in steps[v]:
            for i in eval("[params[q][%s] for q in wxyz[0]]" % s):
                for j in eval("[params[q][%s] for q in wxyz[1]]" % s):
                    for k in eval("[params[q][%s] for q in wxyz[2]]" % s):
                        t[v].append([str((1.0 - i[q]*j[q]*k[q])**scale) for q in range(len(i))])
                        if float(t[v][-1][-1]) > 1.0 or float(t[v][-1][-1]) < 0.0:
                            raise ValueError, "Weights must be in range 0-1. Check normalization and/or assignment."

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

    if not parsed_opts.out:
        plt.show()
    else:
        fig.savefig(parsed_opts.out)


if __name__ == '__main__':
    pass


# EOF

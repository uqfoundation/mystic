#!/usr/bin/env python
#
# Author: Lan Huong Nguyen (lanhuong @stanford)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2012-2015 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE

__doc__ = """
generate cost convergence rate plots from file written with 'write_support_file'

Available from the command shell as:
  mystic_collapse_plotter.py filename [options]

The option "collapse" takes a string of comma-separated integers indicating
iteration numbers where parameter collapse has occurred.  If a second set of
integers is provided (delineated by a semicolon), the additional set of integers
will be plotted with a different linestyle (to indicate a different type of
collapse).

The option "label" takes a label string. For example, label = "y"
will label the y-axis plot with 'y'. LaTeX is also accepted. For example,
label = " log-cost, $ log_{10}(\hat{P} - \hat{P}_{max})$" will label the
y-axis with standard LaTeX math formatting. Note that the leading space is
required, and the text is aligned along the axis.

Required Inputs:
  filename            name of the python convergence logfile (e.g paramlog.py)
"""

#XXX: note that 'argparse' is new as of python2.7
from optparse import OptionParser
parser = OptionParser(usage=__doc__)
parser.add_option("-d","--dots",action="store_true",dest="dots",\
                  default=False,help="show data points in plot")
#parser.add_option("-l","--line",action="store_true",dest="line",\
#                  default=False,help="connect data points in plot with a line")
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
parsed_opts, parsed_args = parser.parse_args()


style = '-' # default linestyle
if parsed_opts.dots:
  mark = 'o'
  # when using 'dots', also turn off 'line'
  #if not parsed_opts.line:
  style = 'None'
else:
  mark = ''

try: # select labels for the axes
    label = parsed_opts.label  # format is "x" or " $x$"
except:
    label = 'log-cost, $log_{10}(y - y_{min})$'

try: # get logfile name
  filename = parsed_args[0]
except:
  raise IOError, "please provide log file name"

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


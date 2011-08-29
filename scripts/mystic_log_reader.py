#!/usr/bin/env python
__doc__ = """
mystic_log_reader.py [options] filename

plot parameter convergence from file written with 'LoggingMonitor'


Required Inputs:
  filename            name of the convergence logfile (e.g log.txt)
"""

#XXX: note that 'argparse' is new as of python2.7
from optparse import OptionParser
parser = OptionParser(usage=__doc__)
parser.add_option("-d","--dots",action="store_true",dest="dots",\
                  default=False,help="show data points in plot")
parser.add_option("-l","--line",action="store_true",dest="line",\
                  default=False,help="connect data points in plot with a line")
parser.add_option("-i","--iter",action="store",dest="stop",metavar="INT",\
                  default=None,help="the largest iteration to plot")
parser.add_option("-g","--legend",action="store_true",dest="legend",\
                  default=False,help="show the legend")
#parser.add_option("-f","--file",action="store",dest="filename",metavar="FILE",\
#                  default='log.txt',help="log file name")
parsed_opts, parsed_args = parser.parse_args()


style = '-' # default linestyle
if parsed_opts.dots:
  mark = 'o'
  # when using 'dots', also can turn off 'line'
  if not parsed_opts.line:
    style = 'None'
else:
  mark = ''

try: # get logfile name
  filename = parsed_args[0]
except:
  raise IOError, "please provide log file name"

try: # select which iteration to stop plotting at
  stop = int(parsed_opts.stop)
except:
  stop = None

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
from mystic.munge import logfile_reader
step, param, cost = logfile_reader(filename)

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

results = [[] for i in range(max(id) + 1)]

# populate results for each id with the corresponding (iter,cost,param)
for i in range(len(id)):
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

#print "iter_conv = %s" % iter_conv
#print "cost_conv = %s" % cost_conv
#print "conv = %s" % conv

import matplotlib.pyplot as plt

fig = plt.figure()

#FIXME: These may fail when conv[i][j] = [[],[],[]] and cost = []. Verify this.
ax1 = fig.add_subplot(2,1,1)
for j in range(len(param[0])):
  for i in range(len(conv)):
    tag = "%d,%d" % (i,j)
    ax1.plot(iter_conv[i],conv[i][j],label="%s" % tag,marker=mark,linestyle=style)
if parsed_opts.legend: plt.legend()

ax2 = fig.add_subplot(2,1,2)
for i in range(len(conv)):
  tag = "%d" % i
  ax2.plot(iter_conv[i],cost_conv[i],label='cost %s' % tag,marker=mark,linestyle=style)
if parsed_opts.legend: plt.legend()

plt.show()

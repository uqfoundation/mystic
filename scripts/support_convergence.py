#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE

__doc__ = """
support_convergence.py [options] filename

generate parameter convergence plots from file written with 'write_support_file'

The option "param" takes an indicator string. This indicator string is a list
of strings, with each entry in the list corresponding to an array slice.
For example, params = "[':']" will plot all parameters in a single plot.
Alternatively, params = "[':2','2:']" will split the parameters into two plots,
while params = "['0']" will only plot the first parameter.

The option "label" takes a list of strings. For example, label = "['x','y','']"
will label the y-axis of the first plot with 'x', a second plot with 'y', and
not add a label to a third or subsequent plots. If more labels are given than
plots, then the last label will be used for the y-axis of the 'cost' plot.
LaTeX is also accepted. For example, label = "[r'$ h$',r'$ {\alpha}$',r'$ v$']"
will label the axes with standard LaTeX math formatting. Note that the leading
space is required, and the text is aligned along the axis.

Required Inputs:
  filename            name of the python convergence logfile (e.g paramlog.py)
"""

def factor(n):
  "generator for factors of a number"
  #yield 1
  i = 2
  limit = n**0.5
  while i <= limit:
    if n % i == 0:
      yield i
      n = n / i
      limit = n**0.5
    else:
      i += 1
  if n > 1:
    yield n

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


if __name__ == '__main__':
  #print __doc__

  #XXX: note that 'argparse' is new as of python2.7
  from optparse import OptionParser
  parser = OptionParser(usage=__doc__)
  parser.add_option("-i","--iter",action="store",dest="step",metavar="INT",\
                    default=None,help="the largest iteration to plot")
  parser.add_option("-p","--param",action="store",dest="param",\
                    metavar="STR",default="[':']",
                    help="indicator string to select parameters")
  parser.add_option("-l","--label",action="store",dest="label",\
                    metavar="STR",default="['']",
                    help="string to assign label to y-axis")
  parser.add_option("-n","--nid",action="store",dest="id",\
                    metavar="INT",default=None,
                    help="id # of the nth simultaneous points to plot")
  parser.add_option("-c","--cost",action="store_true",dest="cost",\
                    default=False,help="also plot the parameter cost")
  parser.add_option("-g","--legend",action="store_true",dest="legend",\
                    default=False,help="show the legend")
  parsed_opts, parsed_args = parser.parse_args()

  try:  # get the name of the parameter log file
    file = parsed_args[0]
    import re
    file = re.sub('\.py*.$', '', file)  #XXX: strip off .py* extension
  except:
    raise IOError, "please provide log file name"
  try:  # read standard logfile
    from mystic.munge import logfile_reader, raw_to_support
    _step, params, cost = logfile_reader(file)
    params, cost = raw_to_support(params, cost)
  except: 
    exec "from %s import params" % file
    exec "from %s import cost" % file

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
    select = eval(parsed_opts.param)  # format is "[':2','2:4','5','6:']"
  except:
    select = [':']
   #select = [':1']
   #select = [':2','2:']
   #select = [':1','1:2','2:3','3:']
   #select = ['0','1','2','3']
  plots = len(select)

  try: # select labels for the axes
    label = eval(parsed_opts.label)  # format is "['x','y','z']"
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

  import matplotlib.pyplot as plt

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

  for i in range(2, plots + 1):
    exec "ax%d = fig.add_subplot(dim1,dim2,%d, sharex=ax1)" % (i,i)
    exec "ax%d.set_ylabel(label[%d])" % (i,i-1)
    data = eval("params[%s]" % select[i-1])
    try:
      n = int(select[i-1].split(":")[0])
    except ValueError:
      n = 0
    for line in data:
      exec "ax%d.plot(line,label='%s')#, marker='o')" % (i,n)
      n += 1
    if legend: plt.legend()
  if cost:
    exec "cx1 = fig.add_subplot(dim1,dim2,%d, sharex=ax1)" % int(plots+1)
    exec "cx1.plot(cost,label='cost')#, marker='o')"
    if max(0, len(label) - plots): exec "cx1.set_ylabel(label[-1])"
    if legend: plt.legend()

  plt.show()

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

# EOF

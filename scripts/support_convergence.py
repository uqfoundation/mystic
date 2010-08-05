#!/usr/bin/env python
__doc__ = """
generate parameter convergence plots from file written with 'write_support_file'

Usage: python support_convergence.py [filename] [maxiter] [params] [id]
    [filename] - name of the python convergence logfile (e.g paramlog.py)
    [maxiter] - the largest iteration to plot (from 1 to maxiter) [optional]
    [params] - string indicator to select params (see following note) [optional]
    [id] - select the id'th of simultaneous points to plot [optional]

For example, params = "[':']" will plot all params in a single plot.
Alternatively, params = "[':2','2:']" will split the params into two plots,
while params = "['0']" will only plot the first param.
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
  if len(cand)%2:
    return cand[len(cand)/2], cand[len(cand)/2]
  return cand[len(cand)/2], cand[len(cand)/2 - 1]


if __name__ == '__main__':
  #print __doc__

  import sys
  if '--help' in sys.argv:
    print __doc__
    sys.exit(0)

  try:  # get the name of the parameter log file
    file = sys.argv[1]
    import re
    file = re.sub('\.py*.$', '', file)  #XXX: strip off .py* extension
  except:
    file = 'paramlog'
  exec "from %s import params" % file
  #exec "from %s import meta" % file
  # no need to edit meta  ==>   meta = ['wx','wx2','x','x2','wy',...]

  try: # select which iteration to stop plotting at
    step = int(sys.argv[2])
  except:
    step = None

  try: # select which parameters to plot
    select = eval(sys.argv[3])  # format is "[':2','2:4','5','6:']"
  except:
    select = [':']
   #select = [':1']
   #select = [':2','2:']
   #select = [':1','1:2','2:3','3:']
   #select = ['0','1','2','3']

  try: # select which 'id' to plot results for
    id = int(sys.argv[4])
  except:
    id = None # i.e. 'all' **or** use id=0, which should be 'best' energy ?

  # ensure all terms of select have a ":"
  for i in range(len(select)):
    if isinstance(select[i], int): select[i] = str(select[i])
    if select[i] == '-1': select[i] = 'len(params)-1:len(params)'
    elif not select[i].count(':'):
      select[i] += ':' + str(int(select[i])+1)

  # take only the first 'step' iterations
  params = [var[:step] for var in params]

  # take only the selected 'id'
  if id != None:
    param = []
    for j in range(len(params)):
      param.append([p[id] for p in params[j]])
    params = param[:]

  import matplotlib.pyplot as plt

  plots = len(select)
  dim1,dim2 = best_dimensions(plots)

  fig = plt.figure()
  ax1 = fig.add_subplot(dim1,dim2,1)
  data = eval("params[%s]" % select[0])
  for line in data:
    ax1.plot(line)#, marker='o')

  for i in range(2, plots + 1):
    exec "ax%d = fig.add_subplot(dim1,dim2,%d, sharex=ax1)" % (i,i)
    data = eval("params[%s]" % select[i-1])
    for line in data:
      exec  "ax%d.plot(line)#, marker='o')" % i

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

#!/usr/bin/env python

def tmp_pickle(func, suffix='.pik'):
    """ standard pickle.dump of function to a NamedTemporaryFile """
    import dill as pickle
    import tempfile
    file = tempfile.NamedTemporaryFile(suffix=suffix, dir='.')
    pickle.dump(func, file)
    file.flush()
    return file


def unpickle(filename):
  """ standard pickle.load of function from a File """
  import dill as pickle
  return pickle.load(open(filename,'r'))


from numpy import asarray

def ndim_meshgrid(*arrs):
    """n-dimensional analogue to numpy.meshgrid"""
    arrs = tuple(reversed(arrs))  #edit
    lens = map(len, arrs)
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz*=s

    ans = []    
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j!=i:
                arr2 = arr2.repeat(sz, axis=j) 
        ans.append(arr2)

    return tuple(ans)


def gridpts(q):
    """
takes a list of lists of equal length q = [[1,2],[3,4]]
and produces a list of gridpoints g = [[1,3],[1,4],[2,3],[2,4]]
    """
    q = list(reversed(q))
    w = ndim_meshgrid(*q)
    for i in range(len(q)):
        q[i] = list( w[i].reshape(w[i].size) )
    q = zip(*q)
    return [list(i) for i in q]


def random_samples(lb,ub,npts=10000):
  "generate npts random samples between given lb & ub"
  from numpy.random import random
  dim = len(lb)
  pts = random((dim,npts))
  for i in range(dim):
    pts[i] = (pts[i] * abs(ub[i] - lb[i])) + lb[i]
  return pts


def samplepts(lb,ub,npts):
    """
takes upper and lower bounds (e.g. ub = [2,4], lb = [0,3])
produces a list of sample points s = [[1,3],[1,4],[2,3],[2,4]]
    """
    q = random_samples(lb,ub,npts)
    q = [list(i) for i in q]
    q = zip(*q)
    return [list(i) for i in q]



#######################################################################
if __name__ == '__main__':

  nbins = 2
  lower = [0, 3, 6]
  upper = [2, 5, 8]

  # generate arrays of points defining a grid in parameter space
  grid_dimensions = len(lower)
  bins = []
  for i in range(grid_dimensions):  #XXX: different nbins for each direction?
    step = abs(upper[i] - lower[i])/nbins
    bins.append( [lower[i] + (j+0.5)*step for j in range(nbins)] )

  # build a grid of starting points
  initial_values = gridpts(bins)
  print "grid: %s" % initial_values

  npts = 10
  lower = [0.0, 3.0, 6.0]
  upper = [2.0, 5.0, 8.0]

  # generate a set of random starting points
  initial_values = samplepts(lower,upper,npts)
  print "scatter: %s" % initial_values


# EOF

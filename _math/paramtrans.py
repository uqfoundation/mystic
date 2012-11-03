#! /usr/bin/env python
"""
...associated transformations for legacy data module

lipschitz_distance: ...

graphical_distance: for a given dataset (x,y) and a given model (F),
  find the radius(x') that minimizes the graph between reality, y = G(x),
  and an approximating function, y' = F(x')
"""
debug = False 

def absolute_distance(x, xp, up=False, dmin=0):
  """distance = |x - x'|;  (see euclidean_distance for notes)"""
  from numpy import abs, asarray, newaxis as nwxs
  from __builtin__ import max
  # cast as arrays of the same dimension
  x = asarray(x); xp = asarray(xp)
  xsize = max(len(x.shape), len(xp.shape), dmin)
  while len(x.shape) < xsize: x = x[nwxs]
  while len(xp.shape) < xsize: xp = xp[nwxs]
  # 'upconvert' with a newaxis  #XXX: only works correctly for dim=0,1,2 ?
  if up: x = x[:,nwxs] if xsize > 0 else x[nwxs]
  return abs(x - xp)

def euclidean_distance(x, xp, up=True, dmin=0):
  """1-D euclidean distance between points

  d[i] = | x[i] - x'[i] |

Input:
  up   = True if upconvert x with x[:,newaxis]
  dmin = upconvert to x,x' to dimension >= dmin

Notes:
  for standard array behavior, use up=False
  for element-wise across all elements, use up=True
"""
  return absolute_distance(x,xp,up=up,dmin=dmin).swapaxes(0,1).T

def manhattan_distance(x, xp, **kwds):
  """1-D manhattan distance between points

  d[ij] = | x[i] - x'[j] |

Notes:
  manhattan matrix where x'==x will be symmetric with zeros on the diagonal 
  use dmin=2 for forced upconversion of 1-D arrays
"""
  # dmin: force upconvert to x,x' to dimension >= dmin
  dmin = 0 # default
  if kwds.has_key('dmin'):
    dmin = kwds['dmin']
  from numpy import abs, asarray, newaxis as nwxs
  from __builtin__ import max
  # cast as arrays of the same dimension
  x = asarray(x); xp = asarray(xp)
  xsize = max(len(x.shape), len(xp.shape), dmin)
  while len(x.shape) < xsize: x = x[nwxs]
  while len(xp.shape) < xsize: xp = xp[nwxs]
  # prep for build manhattan matrix in single operation
  if xsize >= 2:
    x = x.T[:,:,nwxs]; xp = xp.T[:,nwxs]
  elif xsize == 1:
    x = x[:,nwxs]#     x.T[:,nwxs];  xp = xp.T
  else:#xsize == 0:
    x = x[nwxs]
  return abs(x - xp)#.T

###############################################################################
# Note: manhattan and euclidean are now essentially 'the same'...
#       (is sum done correctly? is orientation correct? ...)
###############################################################################

def lipschitz_metric(L, x, xp):
  """sum of lipschitz-weighted distance between points

  d = sum( L[i] * |x[i] - x'[i]| )
"""
  #FIXME: merge with lipschitz cone (distance/contains)
  from numpy import sum, asarray
  d = manhattan_distance(x,xp) #XXX: ,dmin=2)
  return sum(L * d.T, axis=-1).T

def _npts(*x):
  """get len(product measure), given lengths of each underlying measure"""
  from numpy import product
  return product(x)

def _get_xy(points):
  """extract the list of positions and the list of values for given points"""
  from numpy import asarray
  if hasattr(points, 'coords'):
    x  = points.coords
    y  = points.values
  elif len(asarray(points).shape) >= 1:
    x = [p.position for p in points] 
    y = [p.value for p in points] 
  else:
    x = points.position
    y = points.value
  return x,y

###########################################################################
# distance metrics  #NOTE: euclidean_distance and the like need renaming!!!
###########################################################################

def chebyshev(x,xp, up=True, dmin=0, axis=None):
  """infinity norm distance between points in euclidean space

  d(inf) =  max( |x[0] - x[0]'|, |x[1] - x[1]'|, ..., |x[n] - x[n]'| ) 

Input:
  x    = array of points, x
  xp   = array pf points, x'

Additional Input:
  up   = True if upconvert x with x[:,newaxis]
  dmin = upconvert to x,x' to dimension >= dmin
  axis = if not None, reduce across the selected axis

Notes:
  for standard array behavior, use up=False
  for element-wise across all elements, use up=True
"""
  d = euclidean_distance(x,xp,up=up,dmin=dmin)
  from numpy import max
  return max(d, axis=axis)


def minkowski(x,xp, up=True, dmin=0, p=3, axis=None):
  """p-norm distance between points in euclidean space

  d(p) = sum( |x[0] - x[0]'|^p, |x[1] - x[1]'|^p, ..., |x[n] - x[n]'|^p )^(1/p)

Input:
  x    = array of points, x
  xp   = array pf points, x'

Additional Input:
  up   = True if upconvert x with x[:,newaxis]
  dmin = upconvert to x,x' to dimension >= dmin
  axis = if not None, reduce across the selected axis
  p    = value of p for the p-norm (default = 3)

Notes:
  for standard array behavior, use up=False
  for element-wise across all elements, use up=True
"""
  d = euclidean_distance(x,xp,up=up,dmin=dmin)
  from numpy import sum
  return sum(d**p, axis=axis)**(1./p)


def euclidean(x,xp, up=True, dmin=0, axis=None):
  """L-2 norm distance between points in euclidean space

  d(2) = sqrt(sum( |x[0] - x[0]'|^2, |x[1] - x[1]'|^2, ..., |x[n] - x[n]'|^2 ))

Input:
  x    = array of points, x
  xp   = array pf points, x'

Additional Input:
  up   = True if upconvert x with x[:,newaxis]
  dmin = upconvert to x,x' to dimension >= dmin
  axis = if not None, reduce across the selected axis

Notes:
  for standard array behavior, use up=False
  for element-wise across all elements, use up=True
"""
  return minkowski(x,xp,up=up,dmin=dmin,p=2,axis=axis)


def manhattan(x,xp, up=True, dmin=0, axis=None):
  """L-1 norm distance between points in euclidean space

  d(1) = sum( |x[0] - x[0]'|, |x[1] - x[1]'|, ..., |x[n] - x[n]'| )

Input:
  x    = array of points, x
  xp   = array pf points, x'

Additional Input:
  up   = True if upconvert x with x[:,newaxis]
  dmin = upconvert to x,x' to dimension >= dmin
  axis = if not None, reduce across the selected axis

Notes:
  for standard array behavior, use up=False
  for element-wise across all elements, use up=True
"""
  return minkowski(x,xp,up=up,dmin=dmin,p=1,axis=axis)

###########################################################################

def is_feasible(distance, cutoff=0.0):
  """determine if the distance exceeds the given cutoff distance

Input:
  distance = measure of feasibility for each point
  cutoff = maximum acceptable distance

Returns:
  True where the distance is less than cutoff
"""
  from numpy import asarray
  d = infeasibility(distance, cutoff) > 0.0
  if len(asarray(d).shape) is 0:
    return not d
  return -d


def infeasibility(distance, cutoff=0.0): 
  """amount by which the distance exceeds the given cutoff distance

Input:
  distance = measure of feasibility for each point
  cutoff = maximum acceptable distance

Returns:
  array of distances by which each point is infeasbile
"""
  from numpy import array
  distance = array(distance)
  # zero-out all distances less than tolerated
  if cutoff is not None:
    if len(distance.shape) is 0:
      return 0.0 if distance <= cutoff else distance
    distance[distance <= cutoff] = 0.0
  return distance


def lipschitz_distance(L, points1, points2, **kwds):
  """calculate the lipschitz distance between two sets of datapoints

Inputs:
 L = list of lipschitz constants
 points1 = dataset or list of 'datapoint' or 'cone.vertex' objects
 points2 = dataset or list of 'datapoint' or 'cone.vertex' objects

Additional Inputs:
 tol = maximum acceptable deviation from shortness
 cutoff = zero out distances less than cutoff; typically: tol, 0.0, or None

Returns:
 list of lipschitz distances

Notes:
 Each point x,y can be thought to have an associated double-cone with slope
 equal to the lipschitz constant. Shortness with respect to another point is
 defined by the first point not being inside the cone of the second. We can
 allow for some error in shortness, a short tolerance 'tol', for which the
 point x,y is some acceptable y-distance inside the cone. While very tightly
 related, cutoff and tol play distinct roles; tol is subtracted from
 calculation of the lipschitz_distance, while cutoff zeros out the value
 of any element less than the cutoff.
"""
  #FIXME: merge with lipschitz cone (distance/contains)
  x,y   = _get_xy(points1)
  xp,yp = _get_xy(points2)

  # get tolerance in y
  tol = kwds.pop('tol', 0.0)
  cutoff = tol  # default is to zero out distances less than tolerance
  if kwds.has_key('cutoff'): cutoff = kwds.pop('cutoff')
  if cutoff is True: cutoff = tol
  elif cutoff is False: cutoff = None

  # calculate the distance matrix
  md = manhattan_distance(y,yp) - max(0.0, tol)
  lm = lipschitz_metric(L,x,xp)
  d = md - lm
  # zero-out all distances less than tolerated 
  return infeasibility(d, cutoff)


def graphical_distance(model, points, **kwds):
  """find the radius(x') that minimizes the graph between reality, y = G(x),
and an approximating function, y' = F(x')

Inputs:
  model = the model function, y' = F(x'), that approximates reality, y = G(x)
  points = object of type 'datapoint' to validate against; defines y = G(x)

Additional Inputs:
  ytol = maximum acceptable difference |y - F(x')|; a single value
  xtol = maximum acceptable difference |x - x'|; an iterable or single value
  cutoff = zero out distances less than cutoff; typically: ytol, 0.0, or None
  hausdorff = norm; where if given, ytol = |y - F(x')| + |x - x'|/norm

Returns:
  radius = minimum distance from x,G(x) to x',F(x') for each x

Notes:
  xtol defines the n-dimensional base of a pilar of height ytol, centered at
  each point. The region inside the pilar defines the space where a "valid"
  model must intersect. If xtol is not specified, then the base of the pilar
  will be a dirac at x' = x. This function performs an optimization for each
  x to find an appropriate x'. While cutoff and ytol are very tightly related,
  they play a distinct role; ytol is used to set the optimization termination
  for an acceptable |y - F(x')|, while cutoff is applied post-optimization.
  If we are using the hausdorff norm, then ytol will set the optimization
  termination for an acceptable |y - F(x')| + |x - x'|/norm, where the x
  values are normalized by norm = hausdorff.
"""
 #NotImplemented:
 #L = list of lipschitz constants, for use when lipschitz metric is desired
 #constraints = constraints function for finding minimum distance
  from mystic.math.legacydata import dataset
  from numpy import asarray, sum, isfinite, zeros, seterr
  from mystic.solvers import diffev2, fmin_powell
  from mystic.monitors import Monitor, VerboseMonitor

  # ensure target xe and ye is a dataset
  target = dataset()
  target.load(*_get_xy(points))
  nyi = target.npts             # y's are target.values
  nxi = len(target.coords[-1])  # nxi = len(x) / len(y)
  
  # NOTE: the constraints function is a function over a single xe,ye
  #       because each underlying optimization is over a single xe,ye.
  #       thus, we 'pass' on using constraints at this time...
  constraints = None   # default is no constraints
  if kwds.has_key('constraints'): constraints = kwds.pop('constraints')
  if not constraints:  # if None (default), there are no constraints
    constraints = lambda x: x

  # get tolerance in y and wiggle room in x
  ytol = kwds.pop('ytol', 0.0)
  xtol = kwds.pop('xtol', 0.0) # default is to not allow 'wiggle room' in x 

  cutoff = ytol  # default is to zero out distances less than tolerance
  if kwds.has_key('cutoff'): cutoff = kwds.pop('cutoff')
  if cutoff is True: cutoff = ytol
  elif cutoff is False: cutoff = None
  ipop = kwds.pop('ipop', min(20, 3*nxi)) #XXX: tune ipop?
  imax = kwds.pop('imax', 1000) #XXX: tune imax?

  # get range for the dataset (normalization for hausdorff distance)
  hausdorff = kwds.pop('hausdorff', False)
  if not hausdorff:  # False, (), None, ...
    ptp = [0.0]*nxi
  elif hausdorff is True:
    from mystic.math.measures import spread
    ptp = [spread(xi) for xi in zip(*target.coords)]
  else:
    try: #iterables
      ptp = len(hausdorff) == nxi #XXX: should be the same length
      ptp = hausdorff
    except TypeError: #non-iterables
      ptp = [hausdorff]*nxi

  #########################################################################
  def radius(model, point, ytol=0.0, xtol=0.0, ipop=None, imax=None):
    """graphical distance between a single point x,y and a model F(x')"""
    # given a single point x,y: find the radius = |y - F(x')| + delta
    # radius is just a minimization over x' of |y - F(x')| + delta
    # where we apply a constraints function (of box constraints) of
    # |x - x'| <= xtol  (for each i in x)
    #
    # if hausdorff = some iterable, delta = |x - x'|/hausdorff
    # if hausdorff = True, delta = |x - x'|/spread(x); using the dataset range
    # if hausdorff = False, delta = 0.0
    #
    # if ipop, then DE else Powell; ytol is used in VTR(ytol)
    # and will terminate when cost <= ytol
    x,y = _get_xy(point)
    y = asarray(y)

    # build the cost function
    if hausdorff: # distance in all directions
      def cost(rv):
        '''cost = |y - F(x')| + |x - x'| for each x,y (point in dataset)'''
        errs = seterr(invalid='ignore', divide='ignore') # turn off warning 
        z = abs((asarray(x) - rv)/ptp)  # normalize by range
        seterr(invalid=errs['invalid'], divide=errs['divide']) # turn on warning
        return abs(y - model(rv)) + sum(z[isfinite(z)])
    else:  # vertical distance only
      def cost(rv):
        '''cost = |y - F(x')| for each x,y (point in dataset)'''
        return abs(y - model(rv))

    if debug:
      print "rv: %s" % str(x)
      print "cost: %s" % cost(x)

    # if xtol=0, radius is difference in x,y and x,F(x); skip the optimization
    try:
      if not imax or not max(xtol): #iterables
        return cost(x)
    except TypeError:
      if not xtol: #non-iterables
        return cost(x)

    # set the range constraints
    xtol = asarray(xtol)
    bounds = zip( x - xtol, x + xtol )

    # optimize where initially x' = x
    stepmon = Monitor()
    if debug: stepmon = VerboseMonitor(1)
    #XXX: edit settings?
    MINMAX = 1 #XXX: confirm MINMAX=1 is minimization
    if ipop: # use VTR
      results = diffev2(cost, bounds, ipop, ftol=ytol, gtol=None, \
                        itermon = stepmon, maxiter=imax, bounds=bounds, \
                        full_output=1, disp=0, handler=False)
    else: # use VTR
      results = fmin_powell(cost, x, ftol=ytol, gtol=None, \
                            itermon = stepmon, maxiter=imax, bounds=bounds, \
                            full_output=1, disp=0, handler=False)
   #solved = results[0]            # x'
    func_opt = MINMAX * results[1] # cost(x')
    if debug:
      print "solved: %s" % results[0]
      print "cost: %s" % func_opt

    # get the minimum distance |y - F(x')|
    return func_opt
  #########################################################################

  #XXX: better to do a single optimization rather than for each point ???
  d = [radius(model, point, ytol, xtol, ipop, imax) for point in target]
  return infeasibility(d, cutoff)


#def split_xy(params, npts):
#  """split params_{w,x,y} to params_{wx}, params_{y}
#  npts is [len(measure1),...,len(measureN)]  i.e. pm.pts"""
# #if not isinstance(npts,int):
#  _len = 2 * sum(npts)
#  return params[:_len], params[_len:]  
# #return params[:-npts], params[-npts:]

#def mend_xy(wx_param, y_param):
#  """append params_{y} to params_{wx} yielding params_{w,x,y}"""
#  param = wx_param[:]
#  param.extend(y_param)
#  return param


if __name__ == '__main__':
  ### conversions ###
  print "building a list of params(w,x,Y)..."
  pts = (2,2,2)
  param1 = [.5,.5,1,2, .25,.75,3,4, .125,.875,5,6, -1,-2,-3,-4,-5,-6,-7,-8]
  print "pts: %s" % str(pts)
  print "params: %s" % param1
 
  print "\nbuilding a scenario from the params..."
  # [store Y as 'values' OR register(F) for Y=F(X) OR points store y as 'val' ?]
  from mystic.math.dirac_measure import scenario
  pm = scenario()
  pm.load(param1, pts)
  print "pm.wts: %s" % str(pm.wts)
  print "pm.pos: %s" % str(pm.pos)
  W = pm.weights
  X = pm.coords
  Y = pm.values
  print "pm.weights: %s" % str(W)
  print "pm.coords: %s" % str(X)
  print "pm.values: %s" % str(Y)

  print "\nbuilding a dataset from the scenario..."
  # build a dataset (using X,Y)
  # [store W as 'weights' ?]
  from mystic.math.legacydata import dataset
  d = dataset()
  d.load(X, Y) 
  print "d.coords: %s" % str(d.coords) 
  print "d.values: %s" % str(d.values) 

  print "\nedit the dataset..."
  d[0].value = 0
  print "d.values: %s" % str(d.values) 
  # DON'T EDIT d[0].position... IT BREAKS PRODUCT MEASURE!!!

  print "\nupdate the scenario from the dataset..."
  # update pm(w,x,Y | W,X) from dataset(X,Y)
  pm.coords, pm.values = d.fetch()
  print "then, build a new list of params from the scenario..."
  # convert pm(w,x,Y | W,X) to params(w,x,Y)
  param1 = pm.flatten(all=True)
  print "params: %s" % str(param1)

  ### lipschitz ###
  param2 = [1.,0.,1,1, .75,.25,3,4, .5,.5,1,2, -8,-2,4,4,-3,-8,-8,0]
  qm = scenario()
  qm.load(param2, pts)
  L = [.25,.5,1.]

  print ""
 #print "param1: %s" % param1
 #print "param2: %s" % param2
  print "pm1.coords: %s" % pm.coords
  print "pm2.coords: %s" % qm.coords
  print "L = %s" % L

  print "manhattan distance..."
  print "pm1[0:3],pm2[0:3] =>\n", manhattan_distance(pm.coords[0:3],qm.coords[0:3]), "\n"
  print "pm1[0:1],pm2[0:3] =>\n", manhattan_distance(pm.coords[0:1],qm.coords[0:3]), "\n"
  print "pm1[0:1],pm2[0:1] =>\n", manhattan_distance(pm.coords[0:1],qm.coords[0:1]), "\n"

  print "pm1[0:3],pm2[0:3] values =>\n", manhattan_distance(pm.values[0:3],qm.values[0:3]), "\n"
  print "pm1[0:1],pm2[0:3] values =>\n", manhattan_distance(pm.values[0:1],qm.values[0:3]), "\n"
  print "pm1[0:1],pm2[0:1] values =>\n", manhattan_distance(pm.values[0:1],qm.values[0:1]), "\n"
   
  print "lipschitz metric..."
  print "pm1[0:3],pm2[0:3] =>\n", lipschitz_metric(L, pm.coords[0:3], qm.coords[0:3]), "\n"
  print "pm1[0:1],pm2[0:3] =>\n", lipschitz_metric(L, pm.coords[0:1], qm.coords[0:3]), "\n"
  print "pm1[0;1],pm2[0:1] =>\n", lipschitz_metric(L, pm.coords[0:1], qm.coords[0:1]), "\n"
  
  print "lipschitz distance..."
  print "(don't cutoff):\n%s" % lipschitz_distance(L, pm, qm, cutoff=None)
  print "from measures:\n%s" % lipschitz_distance(L, pm, qm)
  b = dataset()
  id = ['A','B','C','D','E','F','G','H']
  b.load(qm.coords, qm.values, ids=id) 
  print "from datasets:\n%s" % lipschitz_distance(L, d, b)
  print "from list of points:\n%s" % lipschitz_distance(L, d.raw, b.raw)
  print "individual points:\n%s" % lipschitz_distance(L, d.raw[0:1], b.raw[0:1])

  print ""
  print "is short:\n%s" % is_feasible( lipschitz_distance(L, pm, qm) )

  ### updates ###
  print "\nupdates to dataset..."
  print "original:\n%s" % b
  b.update(qm.coords, qm.values) 
  print "points from pm2:\n%s" % b
  b.update(pm.coords, pm.values) 
  print "points from pm1:\n%s" % b

  print "\nupdates to product measure..."
  print "orig: %s\n %s\n from %s" % (pm, pm.values, pm.flatten(all=True))
  par = [.25,.75,3,4, .5,.5,1,2, .125,.875,5,6, 0,-1,-2,-3,-4,-5,-6,-7]
  pm.update(par)
  print "alt: %s\n %s\n from %s" % (pm, pm.values, par)
  par = [.25,.75,3,4, .5,.5,1,2, .125,.875,5,6, -1,-2,-3,-4,-5,-6,-7,-8,-9]
  pm.update(par)
  print "alt2: %s\n %s\n from %s" % (pm, pm.values, par)
  par = [.25,.75,3,4, .5,.5,1,2, .125,.875,5,6, -2,-4,-6,-8]
  pm.update(par)
  print "alt3: %s\n %s\n from %s" % (pm, pm.values, par)
  par = [.5,.5,1,2, .25,.75,3,4]
  pm.update(par)
  print "alt4: %s\n %s\n from %s" % (pm, pm.values, par)

  ##### indexing #####
  b.update(pm.coords, pm.values)
  assert b[0].value == pm.values[0]
  assert b[1].value == pm.values[1]
  assert b[0].position == pm.coords[0] # slow
  assert b[0].position == pm.select(0, reduce=False)[0] # fast
  assert b[1].position == pm.coords[1] # slow
  assert b[1].position == pm.select(1) # fast
  assert pm.select(0,2) == [pm.select(0),pm.select(2)]

  # member calls #
  print "\ntesting mean_value..."
  print "mean_value: %s" % pm.get_mean_value()
  pm.set_mean_value(5.0)
  print "mean_value: %s" % pm.get_mean_value()

  print "\ntesting shortness, feasibility, validity..."
  assert pm.short_wrt_data(b) == True
  b.lipschitz = L
  assert pm.short_wrt_self(L) == False
  assert pm.short_wrt_data(b) == False
  pm.set_feasible(b)
  assert pm.short_wrt_self(L) == True
  assert pm.short_wrt_data(b) == True

  Cy = 0.1; Cx = 0.0
  model = lambda x:x[0]
  assert b.valid(model, ytol=Cy, xtol=Cx) == False
  assert pm.valid_wrt_model(model, ytol=Cy, xtol=Cx) == False
  pm.set_valid(model, cutoff=Cy, xtol=Cx)
  assert pm.valid_wrt_model(model, ytol=Cy, xtol=Cx) == True
  print "...done\n"


# EOF

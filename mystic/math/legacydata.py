#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
data structures for legacy data observations of lipschitz functions
"""
from numpy import inf, asarray
NULLSLOPE = inf

def _fails(filter, data=None):
  """filter a data array for failures using a boolean filter

  Inputs:
    filter -- a list of booleans (same size as data)
    data -- a list of data points

  Notes:
    if data=None, return the pairs of indices where data fails
  """
  from numpy import where, asarray
  failures = asarray(where(filter == False)).T
  # if data not given, then return indices for failure
  if not data: return failures
  # otherwise, return the list of 'bad' points
  failures = set( failures.flat )
  return asarray([data[i] for i in failures])

class lipschitzcone(list):
  """ Lipschitz double cone around a data point, with vertex and slope

 queries:
  vertex -- coordinates of lipschitz cone vertex
  slopes -- lipschitz slopes for the cone (should be same dimension as 'vertex')

 methods:
  contains -- return True if a given point is within the cone
  distance -- sum of lipschitz-weighted distance between a point and the vertex
"""

  def __init__(self, datapoint, slopes=None):
    self.vertex = datapoint
    if not slopes: slopes = [NULLSLOPE]*len(self.vertex.position)
    self.slopes = slopes
    return

  def __repr__(self):
    return "cone(%s @%s)" % (self.slopes, self.vertex.position)

  def distance(self, point):
    """ sum of lipschitz-weighted distance between a point and the vertex """
    x = point.position
    Z = self.vertex.position
    L = self.slopes
    return sum([L[i]*abs(x[i] - Z[i]) for i in range(len(x))])

  def contains(self, point):
    """ return True if a given point is within the cone """
    L = self.slopes
    if any([i == NULLSLOPE for i in L]): return True
    x = point.position
    Z = self.vertex.position
    if all([x[i] == Z[i] for i in range(len(x))]): return True
    y = point.value
    G = self.vertex.value
    return abs(y - G) <= self.distance(point)

  pass

class point(object):
  """ n-d data point with position and value but no id (i.e. 'raw')

 queries:
  p.value   --  returns value
  p.position  --  returns position
  p.rms  --  returns the square root of sum of squared position

 settings:
  p.value = v1  --  set the value
  p.position = (x1, x2, ..., xn)  --  set the position
"""

  def __init__(self, position, value):
    self.value = value
    self.position = position
    return
  
  def __repr__(self):
    return "(%s @%s)" % (self.value, self.position)

  def __rms(self): # square root of sum of squared positions
    from math import sqrt
    return sqrt(sum([i**2 for i in self.position]))

  ## compare
  def __eq__(self, other): # equal if value and position are equal
    if type(other) == type(self):
      return other.value == self.value and other.position == self.position
    return False

  def __ne__(self, other):
    return not other == self

  # sorts by rms(position) then by value... better by value first ???
  def __lt__(self, other):
    if type(other) == type(self):
      if self.position != other.position:
        return other.rms - self.rms > 0.0
      return other.value - self.value > 0.0
    return NotImplemented

  def __gt__(self, other):
    if type(other) == type(self):
      if self.position != other.position:
        return other.rms - self.rms < 0.0
      return other.value - self.value < 0.0
    return NotImplemented

  def __le__(self, other):
    if type(other) == type(self):
      if self.position != other.position:
        return other.rms - self.rms >= 0.0
      return other.value - self.value >= 0.0
    return NotImplemented

  def __ge__(self, other):
    if type(other) == type(self):
      if self.position != other.position:
        return other.rms - self.rms <= 0.0
      return other.value - self.value <= 0.0
    return NotImplemented
  ## end compare

  # interface
  rms = property(__rms)

  pass


class datapoint(object):
  """ n-d data point with position and value 

 queries:
  p.value   --  returns value
  p.position  --  returns position

 settings:
  p.value = v1  --  set the value
  p.position = (x1, x2, ..., xn)  --  set the position

 notes:
  a datapoint can have an assigned id and cone;
  also has utilities for comparison against other
  datapoints (intended for use in a dataset)
"""

  __hash_id = id  # save a copy of the builtin id function

  def __init__(self, position, value=None, id=None, lipschitz=None):
    self.raw = point([],None)
    if type(position) == type(self.raw):
      self.raw = position
    else:
      self.raw.position = position 
      if value is None: raise ValueError("value not set for the datapoint")
      self.raw.value = value
    if id is None: id = self.__hash_id(self)  #XXX: default id is hash id
    self.id = id
    self.cone = lipschitzcone(self,lipschitz)
    return

  # compare (defer to 'raw')
  def __eq__(self, other):
    if type(other) == type(self):
      return other.raw == self.raw and other.id == self.id  #XXX: consider id?
    return False

  def __ne__(self, other):
    return not other == self

  def __lt__(self, other):
    if type(other) == type(self):
      return self.raw < other.raw
    return NotImplemented

  def __gt__(self, other):
    if type(other) == type(self):
      return self.raw > other.raw
    return NotImplemented

  def __le__(self, other):
    if type(other) == type(self):
      if self.raw == other.raw:     #XXX: consider id?
        return self.id <= other.id  #XXX: consider id?
      return self.raw <= other.raw
    return NotImplemented

  def __ge__(self, other):
    if type(other) == type(self):
      if self.raw == other.raw:     #XXX: consider id?
        return self.id >= other.id  #XXX: consider id?
      return self.raw >= other.raw
    return NotImplemented
  # end compare

  # integrity
  def __is_duplicate(self, pt): # same raw & id
    return False if self is pt else self == pt

  def __is_repeat(self, pt): # same raw, different id
    return False if self == pt else self.raw == pt.raw
   #return self.id != pt.id and self == pt

  def __is_conflict(self, pt): # same id, different raw
    return False if self == pt else self.id == pt.id
   #return self != pt and self.id == pt.id

  def __is_collision(self, pt): # same position, different value
    return False if self.raw == pt.raw else self.position == pt.position

  def duplicates(self, pts):
    """ return True where a point exists with same 'raw' and 'id' """
    return [self.__is_duplicate(i) for i in pts]

  def repeats(self, pts):
    """ return True where a point exists with same 'raw' but different 'id' """
    return [self.__is_repeat(i) for i in pts]

  def conflicts(self, pts):
    """ return True where a point exists with same 'id' but different 'raw' """
    return [self.__is_conflict(i) for i in pts]

  def collisions(self, pts):
    """ return True where a point exists with same 'position' and different 'value' """
    return [self.__is_collision(i) for i in pts]
  # end integrity

  def __repr__(self):
    return "(%r: %r)" % (self.id, self.raw)

  def __position(self):
    return self.raw.position

  def __value(self):
    return self.raw.value

  def __set_position(self, position):
    self.raw.position = position
    return

  def __set_value(self, value):
    self.raw.value = value
    return

  # interface
  position = property(__position, __set_position)
  value = property(__value, __set_value)

  pass

class dataset(list):  #FIXME: should just accept datapoints
  """ a collection of data points
  s = dataset([point1, point2, ..., pointN])

 queries:
  s.values  --  returns list of values
  s.coords  --  returns list of positions
  s.ids  --  returns list of ids
  s.raw  --  returns list of points
  s.npts  --  returns the number of points
  s.lipschitz  --  returns list of lipschitz constants

 settings:
  s.lipschitz = [s1, s2, ..., sn]  --  sets lipschitz constants

 methods:
  short  -- check for shortness with respect to given data (or self)
  valid  -- check for validity with respect to given model
  update  -- update the positions and values in the dataset 
  load  -- load a list of positions and a list of values to the dataset 
  fetch  -- fetch the list of positions and the list of values in the dataset
  intersection  -- return the set intersection between self and query
  filter  -- return dataset entries where mask array is True 
  has_id  -- return True where dataset ids are in query
  has_position  -- return True where dataset coords are in query
  has_point  -- return True where dataset points are in query
  has_datapoint  -- return True where dataset entries are in query

 notes:
  - datapoints should not be edited; except possibly for id
  - assumes that s.n = len(s.coords) == len(s.values)
  - all datapoints in a dataset should have the same cone.slopes
"""
  # - when datapoint added to dataset, SHOULD build cone from dataset.lipschitz
  #   dataset.__set_id should check 'dataset.contains(id)', so no duplicate ids

  __name__ = None

  def short(self, data=None, L=None, blamelist=False, pairs=True, \
                                     all=False, raw=False, **kwds):
    """check for shortness with respect to given data (or self)

Args:
    data (list, default=None): a list of data points, or the dataset itself.
    L (float, default=None): the lipschitz constant, or the dataset's constant.
    blamelist (bool, default=False): if True, indicate the infeasible points.
    pairs (bool, default=True): if True, indicate indices of infeasible points.
    all (bool, default=False): if True, get results for each individual point.
    raw (bool, default=False): if False, get boolean results (i.e. non-float).
    tol (float, default=0.0): maximum acceptable deviation from shortness.
    cutoff (float, default=tol): zero out distances less than cutoff.

Notes:
    Each point x,y can be thought to have an associated double-cone with slope
    equal to the lipschitz constant. Shortness with respect to another point is
    defined by the first point not being inside the cone of the second. We can
    allow for some error in shortness, a short tolerance *tol*, for which the
    point x,y is some acceptable y-distance inside the cone. While very tightly
    related, *cutoff* and *tol* play distinct roles; *tol* is subtracted from
    calculation of the lipschitz_distance, while *cutoff* zeros out the value
    of any element less than the *cutoff*.
"""
    tol = kwds['tol'] if 'tol' in kwds else 0.0 # get tolerance in y
    # default is to zero out distances less than tolerance
    cutoff = kwds['cutoff'] if 'cutoff' in kwds else tol
    if cutoff is True: cutoff = tol
    elif cutoff is False: cutoff = None

    if L is None: L = self.lipschitz
    if data is None: data = self
    from mystic.math.distance import lipschitz_distance, is_feasible
    # calculate the shortness
    Rv = lipschitz_distance(L, self, data, **kwds)
    ld = is_feasible(Rv, cutoff)
    if raw:
      x = Rv
    else:
      x = ld
    if not blamelist: return ld.all() if not all else x
    if pairs: return _fails(ld)
    # else lookup failures
    return _fails(ld, data)

  def valid(self, model, blamelist=False, pairs=True, \
                         all=False, raw=False, **kwds):
    """check for validity with respect to given model

Args:
    model (func): the model function, ``y' = F(x')``.
    blamelist (bool, default=False): if True, indicate the infeasible points.
    pairs (bool, default=True): if True, indicate indices of infeasible points.
    all (bool, default=False): if True, get results for each individual point.
    raw (bool, default=False): if False, get boolean results (i.e. non-float).
    ytol (float, default=0.0): maximum acceptable difference ``|y - F(x')|``.
    xtol (float, default=0.0): maximum acceptable difference ``|x - x'|``.
    cutoff (float, default=ytol): zero out distances less than cutoff.
    hausdorff (bool, default=False): hausdorff ``norm``, where if given,
        then ``ytol = |y - F(x')| + |x - x'|/norm``.

Notes:
    *xtol* defines the n-dimensional base of a pilar of height *ytol*,
    centered at each point. The region inside the pilar defines the space
    where a "valid" model must intersect. If *xtol* is not specified, then
    the base of the pilar will be a dirac at ``x' = x``. This function
    performs an optimization for each ``x`` to find an appropriate ``x'``.

    *ytol* is a single value, while *xtol* is a single value or an iterable.
    *cutoff* takes a float or a boolean, where ``cutoff=True`` will set the
    value of *cutoff* to the default. Typically, the value of *cutoff* is
    *ytol*, 0.0, or None. *hausdorff* can be False (e.g. ``norm = 1.0``),
    True (e.g. ``norm = spread(x)``), or a list of points of ``len(x)``.

    While *cutoff* and *ytol* are very tightly related, they play a distinct
    role; *ytol* is used to set the optimization termination for an acceptable
    ``|y - F(x')|``, while *cutoff* is applied post-optimization.

    If we are using the *hausdorff* norm, then *ytol* will set the optimization
    termination for an acceptable ``|y - F(x')| + |x - x'|/norm``, where the
    ``x`` values are normalized by ``norm = hausdorff``.
"""
    ytol = kwds['ytol'] if 'ytol' in kwds else 0.0 # get tolerance in y
    # default is to zero out distances less than tolerance
    cutoff = kwds['cutoff'] if 'cutoff' in kwds else ytol
    if cutoff is True: cutoff = ytol
    elif cutoff is False: cutoff = None

    from mystic.math.distance import graphical_distance, is_feasible
    # calculate the model validity
    Rv = graphical_distance(model, self, **kwds)
    ld = is_feasible(Rv, cutoff)
    if raw:
      x = Rv
    else:
      x = ld
    if not blamelist: return ld.all() if not all else x
    if pairs: return _fails(ld)
    # else lookup failures
    return _fails(ld, self)

  def update(self, positions, values):#ids=None):# positions,values are iterable
    """ update the positions and values in the dataset 

Returns:
    self (dataset): the dataset itself

Notes:
    positions and values provided must be iterable
"""
    lip = self.lipschitz
    ids = self.ids +\
          [None] * max(0, min(len(positions), len(values)) - len(self.ids))
    z = list(zip(positions, values, ids))
    self[:len(z)] = [datapoint(i,j,id=k) for (i,j,k) in z]
    self.lipschitz = lip
    return self

  def load(self, positions, values, ids=[]): # positions,values are iterable
    """load a list of positions and a list of values to the dataset 

Returns:
    self (dataset): the dataset itself

Notes:
    positions and values provided must be iterable
"""
    z = list(zip(positions, values))
    self.extend([datapoint(i,j) for (i,j) in z])
    if ids: #XXX: must be at least as long as 'z'
      for i in range(len(z)):
        self[-len(z)+i].id = ids[i]
    return self

# def generate(self, model, positions, ids=[]):
#   """use a function to generate a dataset from the given list of positions"""
#   values = map(model, positions)    #XXX: readily extended to parallel
#   self.load(positions, values, ids)
#   return

# def evaluate(self, model):
#   """evaluate a function at dataset positions, overwriting existing values"""
#   values = map(model, self.coords) #XXX: readily extended to parallel
#   self.update(self.coords, values)
#   return

  def fetch(self):
    """fetch the list of positions and the list of values in the dataset"""
    return self.coords, self.values

  # integrity
  def __has_duplicate(self):
    return [any(i.duplicates(self)) for i in self]
   #return [i.duplicates(self) for i in self]

  def __has_repeat(self):
    return [any(i.repeats(self)) for i in self]

  def __has_conflict(self):
    return [any(i.conflicts(self)) for i in self]

  def __has_collision(self):
    return [any(i.collisions(self)) for i in self]

# def __has_duplicate(self, pt): #XXX XXX: checks if pt is duplicated in self
#   return True if self.count(pt) > 1 else False
#   # could return w/o duplicated by: dataset(set(self))

# def __has_conflict(self, pt): #XXX XXX: assumes repeated id not duplicate
#   return True if self.ids.count(pt.id) > 1 else False

# def __has_collision(self, pt): #XXX XXX: assumes repeated coord not duplicate
#   return True if self.coords.count(pt.position) > 1 else False

##def __collision(self, pt):
##  return not pt in self and pt.position in self.coords
  # end integrity

  def has_id(self, query): #FIXME: assume is iterable & appropriate
    """return True where dataset ids are in query

Notes:
    query must be iterable
"""
    return [i in query for i in self.ids]

  def has_position(self, query): #FIXME: assume is iterable & appropriate
    """return True where dataset coords are in query

Notes:
    query must be iterable
"""
    #XXX: allow query to contain lists, and not only tuples?
    return [i in query for i in self.coords]

  def has_point(self, query): #FIXME: assume is iterable & appropriate
    """return True where dataset points are in query

Notes:
    query must be iterable
"""
    return [i in query for i in self.raw]

  def has_datapoint(self, query): #FIXME: assume is iterable & appropriate
    """return True where dataset entries are in query

Notes:
    query must be iterable
"""
    return [i in query for i in self]

  def intersection(self, query):
    "return the set intersection between self and query"
    return dataset(set(self).intersection(query))

  def filter(self, mask): #XXX: assumes len(mask) = len(self); mask[i] is bool
    """return dataset entries where mask array is True 

Inputs:
    mask -- a boolean array of the same length as dataset
"""
    from numpy import array, where
    _self = array(self)
    return dataset(_self[where(mask)])

  def __data(self):
    return list(self)

  def __values(self):
    return [i.value for i in self]

  def __coords(self):
    return [i.position for i in self]

  def __ids(self):
    return [i.id for i in self]

  def __raw(self):
    return [i.raw for i in self]

  def __lipschitz(self):
    if not len(self): return []
#   # if not all datapoints have cones, throw error
#   lip = [self[i].cone == NULLCONE for i in range(len(self))]
#   if any(lip):
#     raise ValueError. "Lipschitz constants not set for the dataset."
    # if all cones don't have the same slopes, throw error
    lip = [self[i].cone.slopes != self[0].cone.slopes for i in range(len(self))]
    if any(lip):
      raise ValueError("Lipschitz constants not set for the dataset.")
    # return the lipschitz constants
    return self[0].cone.slopes

  def __npts(self):
    return len(self)

  def __repr__(self):
    return "dset(%r)" % ([pt for pt in self])

  def __set_lipschitz(self, slopes): #XXX: need len(slopes) == len(coords)
    for i in range(len(self)):
      self[i].cone = lipschitzcone(self[i], slopes)
    return

  def __set_ids(self, ids):
    for i in range(len(self)):
      self[i].id = ids[i]
    return

  def __set_values(self, values):
    for i in range(len(self)):
      self[i].value = values[i]
    return

  def __set_coords(self, coords):
    for i in range(len(self)):
      self[i].position = coords[i]
    return

  # interface
  values = property(__values)#, __set_values)
  coords = property(__coords)#, __set_coords)
  ids = property(__ids)#, __set_ids)
  raw = property(__raw)
  npts = property(__npts)
  lipschitz = property(__lipschitz, __set_lipschitz)
  duplicates = property(__has_duplicate)
  repeats = property(__has_repeat)
  conflicts = property(__has_conflict)
  collisions = property(__has_collision)
  pass


#######################################################
# legacy data file IO
#######################################################

def load_dataset(filename, filter=None):
  """ read dataset from selected file

  filename -- string name of dataset file
  filter -- tuple of points to select ('False' to ignore filter stored in file)
"""
  from os.path import split, splitext
  name = splitext(split(filename)[-1])[0]  # default name is filename
  lipschitz = None
  f = open(filename,"r")
  file = f.read()
  f.close()
  contents = file.split("\n")
  # parse file contents to get (i,id), cost, and parameters
  pt = []; _filter = None
  for line in contents[:-1]:
    if line.startswith("# data name"):
      name = eval(line.split(":")[-1])
    elif line.startswith("# lipschitz"):
      lipschitz = list( eval(line.split(":")[-1]) )
    elif line.startswith("# filter") and filter != False:
      _filter = eval(line.split(":")[-1])
    elif line.startswith("#"):
      pass
    else:
      data = line.split("   ")
      sid = eval(data[0])
      value = eval(data[1])
      coords = list(eval(data[-1]))
      pt.append(datapoint(list(coords), value, id=sid))

  # apply filter(s)
  from numpy import asarray
  if _filter is not None and filter != False:
    _filter = asarray(_filter)
    _filter = _filter[_filter < len(pt)]
    pt = [pt[i] for i in _filter]
  if filter is not None and filter != False:
    filter = asarray(filter)
    filter = filter[filter < len(pt)]
    pt = [pt[i] for i in filter]

  # build dataset
  mydataset = dataset(pt)
  mydataset.__name__ = name
  mydataset.lipschitz = lipschitz
  return mydataset

def save_dataset(data, filename='dataset.txt', filter=None, new=True):
  """ save dataset to selected file

  data -- data set
  filename -- string name of dataset file
  filter -- tuple, filter to apply to dataset upon reading
  new -- boolean, False if appending to existing file
"""
  import datetime
  if new: ind = 'w'
  else: ind = 'a'
  file = open(filename, ind)
  file.write("# %s\n" % datetime.datetime.now().ctime() )
  if data.__name__: file.write('# data name: "%s"\n' % data.__name__ )
  if data.lipschitz:
    file.write("# lipschitz: %s\n" % str(tuple(data.lipschitz)) )
  if filter is not None:
    file.write("# filter: %s\n" % str(filter))
  file.write("# ___id___  __value__  __coords__\n")
  for pt in data:
    x = "%s" % str(tuple(pt.position))
    y = "%s" % pt.value
    file.write("  '%s'     %s     %s\n" % (pt.id, y, x))
  file.close()
  return


if __name__ == '__main__':
  x = [1,1,0]; x3 = [1,1,1]; x4 = [0,0,0]
  y = 1; y2 = 2; y3 = 0; y4 = 2; y5 = 3
  print("creating a datapoint...")
  pt1 = datapoint(x,y,id='1')
  print(pt1)
  print("with: %s\n" % pt1.cone)

  print("creating a second datapoint...")
  pt2 = datapoint(x,y2,lipschitz=[1,1,1])
  print(pt2)
  print("with: %s\n" % pt2.cone)

  print("creating a third datapoint...")
  pt3 = datapoint(x3,y3,lipschitz=[1,1,1])
  print(pt3)
  print("with: %s\n" % pt3.cone)

  print("creating a fourth datapoint...")
  pt4 = datapoint(x4,y4,lipschitz=[0.25,0.25,0.25])
  print(pt4)
  print("with: %s\n" % pt4.cone)

  print("creating a fifth datapoint...")
  pt5 = datapoint(x4,y5,lipschitz=[0.25,0.25,0.25])
  print(pt5)
  print("with: %s\n" % pt5.cone)

  print("testing 'cone.contains'...")
  print("1st cone contains 1st, 2nd, 3rd, 4th, 5th point? %s, %s, %s, %s, %s" % (pt1.cone.contains(pt1), pt1.cone.contains(pt2), pt1.cone.contains(pt3), pt1.cone.contains(pt4), pt1.cone.contains(pt5)))
  print("2nd cone contains 1st, 2nd, 3rd, 4th, 5th point? %s, %s, %s, %s, %s" % (pt2.cone.contains(pt1), pt2.cone.contains(pt2), pt2.cone.contains(pt3), pt2.cone.contains(pt4), pt2.cone.contains(pt5)))
  print("3rd cone contains 1st, 2nd, 3rd, 4th, 5th point? %s, %s, %s, %s, %s" % (pt3.cone.contains(pt1), pt3.cone.contains(pt2), pt3.cone.contains(pt3), pt3.cone.contains(pt4), pt3.cone.contains(pt5)))
  print("4th cone contains 1st, 2nd, 3rd, 4th, 5th point? %s, %s, %s, %s, %s" % (pt4.cone.contains(pt1), pt4.cone.contains(pt2), pt4.cone.contains(pt3), pt4.cone.contains(pt4), pt4.cone.contains(pt5)))
  print("5th cone contains 1st, 2nd, 3rd, 4th, 5th point? %s, %s, %s, %s, %s\n" % (pt5.cone.contains(pt1), pt5.cone.contains(pt2), pt5.cone.contains(pt3), pt5.cone.contains(pt4), pt5.cone.contains(pt5)))

  print( "creating a dataset...")
  dset = dataset([pt1,pt2,pt3])
  print(dset)
  print("values: %s:" % dset.values)
  print("coords: %s:" % dset.coords)
  print("ids: %s" % dset.ids)
  print("raw: %s\n" % dset.raw)

  print("setting lipschitz constants for dataset...")
 #print(dset.lipschitz)
  dset.lipschitz = pt2.cone.slopes
  print("all lipschitz: %s" % dset.lipschitz)
  print("resulting cones: %s\n" % [p.cone for p in dset])


  ##### comparisons #####
  p0 = point([1,1,1],1)
  p1 = point([1,1,1],1)
  p2 = point([1,1,1],2)
  p3 = point([1,2,1],1)
  p4 = point([1,1,1],1)
  p5 = point([1,1,1],1)
  assert p0 == p1 and p0 != p2 and p0 != p3 and p0 != ([1,1,1],1)
  assert p2 > p1 and p3 > p1 and p2 < p3  #XXX: p3 > p2 ?

  d0 = datapoint(p0, id='0') # (original)
  d1 = datapoint(p1, id='1') #  conflict (?)
  d2 = datapoint(p2, id='2') #  collision
  d3 = datapoint(p3, id='3') #    ---
  d4 = datapoint(p4, id='0') #  duplicate
  d5 = datapoint(p5, id='3') #  conflict (?)
  assert d0 != d1 and d0 != d2 and d0 != d3 and d0 != p0
  assert d2 > d1 and d3 > d1 and d2 < d3  #XXX: d3 > d2 ?

  ##### intersections #####
  print("testing membership...")
  d = dataset([d0,d1,d2])
  print("point:")
  query = [p1,p2]
  print("%r in %r is %r" % (query, d.raw, d.filter(d.has_point(query)).raw))
  query = [p0,p3]
  print("%r in %r is %r" % (query, d.raw, d.filter(d.has_point(query)).raw))
  print("coords:")
  query = [p0.position, p2.position]
  print("%r in %r is %r" % (query, d.coords, d.filter(d.has_position(query)).coords))
  query = [p1.position, p3.position]
  print("%r in %r is %r" % (query, d.coords, d.filter(d.has_position(query)).coords))
  query = [p3.position]
  print("%r in %r is %r" % (query, d.coords, d.filter(d.has_position(query)).coords))
  print("id:")
  query = ['0','2']
  print("%r in %r is %r" % (query, d.ids, d.filter(d.has_id(query)).ids))
  query = ['1','3']
  print("%r in %r is %r" % (query, d.ids, d.filter(d.has_id(query)).ids))
  print("datapoint:")
  query = [d1,d2]
  print("%r in %r is %r" % (dataset(query).ids, d.ids, d.filter(d.has_datapoint(query)).ids))
  print("datapoint by intersection:")
  query = [d0,d2]
  print("%r in %r is %r" % (dataset(query).ids, d.ids, d.intersection(query).ids))

  ##### integrity #####
  print("testing integrity...")
  d = dataset([d0,d1,d2,d3,d4,d5])
  print("duplicates: %r" % d.duplicates)
  print("repeats: %r" % d.repeats)
  print("conflicts: %r" % d.conflicts)
  print("collisions: %r" % d.collisions)

  print("%r collides: %r" % (d[0], d[0].collisions(d)))
  print("%r collides: %r" % (d[1], d[1].collisions(d)))
  print("%r collides: %r" % (d[2], d[2].collisions(d)))
  print("%r collides: %r" % (d[3], d[3].collisions(d)))
  print("%r collides: %r" % (d[4], d[4].collisions(d)))
  print("%r collides: %r" % (d[5], d[5].collisions(d)))


# EOF

#!/usr/bin/env python
"""
decorators that provide rounding
"""

__all__ = ['deep_round', 'shallow_round', 'simple_round']

from mystic.tools import isiterable
#FIXME: these seem *slow*... and a bit convoluted.  Maybe rewrite as classes?

def deep_round_factory(tol):
  """helper function for deep_round (a factory for deep_round functions)"""
  def deep_round(*args, **kwds):
    argstype = type(args) 
    _args = list(args)
    _kwds = kwds.copy()
    for i,j in enumerate(args):
      if isinstance(j, float): _args[i] = round(j, tol) # don't round int
      elif isinstance(j, (str, unicode, type(BaseException()))): continue
      elif isinstance(j, dict): _args[i] = deep_round(**j)[1]
      elif isiterable(j): #XXX: fails on the above, so don't iterate them
        jtype = type(j)
        _args[i] = jtype(deep_round(*j)[0])
    for i,j in kwds.items():
      if isinstance(j, float): _kwds[i] = round(j, tol)
      elif isinstance(j, (str, unicode, type(BaseException()))): continue
      elif isinstance(j, dict): _kwds[i] = deep_round(**j)[1]
      elif isiterable(j): #XXX: fails on the above, so don't iterate them
        jtype = type(j)
        _kwds[i] = jtype(deep_round(*j)[0])
    return argstype(_args), _kwds
  return deep_round

"""
>>> deep_round = deep_round_factory(tol=0)  #FIXME: convert to decorator !!!
>>> deep_round([1.12,2,{'x':1.23, 'y':[4.56,5.67]}], x=set([11.22,44,'hi']))
(([1.0, 2, {'y': [5.0, 6.0], 'x': 1.0}],), {'x': set([11.0, 'hi', 44])})
"""

def deep_round(tol=0):
  """decorator for rounding a function's input argument and keywords to the
  given precision *tol*.  This decorator always rounds to a floating point
  number.

  Rounding is done recursively for each element of all arguments and keywords.

  For example:
  >>> @deep_round(tol=1)
  ... def add(x,y):
  ...   return x+y
  ...
  >>> add(2.54, 5.47)
  8.0
  >>>
  >>> # rounds each float, regardless of depth in an object
  >>> add([2.54, 5.47],['x','y'])
  [2.5, 5.5, 'x', 'y']
  >>>
  >>> # rounds each float, regardless of depth in an object
  >>> add([2.54, 5.47],['x',[8.99, 'y']])
  [2.5, 5.5, 'x', [9.0, 'y']]
  """
  def dec(f):
    def func(*args, **kwds):
      if tol is None:
        _args,_kwds = args,kwds
      else:
        _deep_round = deep_round_factory(tol)
        _args,_kwds = _deep_round(*args, **kwds)
      return f(*_args, **_kwds)
    return func
  return dec


def simple_round_factory(tol):
  """helper function for simple_round (a factory for simple_round functions)"""
  def simple_round(*args, **kwds):
    argstype = type(args) 
    _args = list(args)
    _kwds = kwds.copy()
    for i,j in enumerate(args):
      if isinstance(j, float): _args[i] = round(j, tol) # don't round int
    for i,j in kwds.items():
      if isinstance(j, float): _kwds[i] = round(j, tol)
    return argstype(_args), _kwds
  return simple_round

def simple_round(tol=0): #NOTE: only rounds floats, nothing else
  """decorator for rounding a function's input argument and keywords to the
  given precision *tol*.  This decorator always rounds to a floating point
  number.

  Rounding is only done for arguments or keywords that are floats.

  For example:
  >>> @simple_round(tol=1)
  ... def add(x,y):
  ...   return x+y
  ... 
  >>> add(2.54, 5.47)
  8.0
  >>>
  >>> # does not round elements of iterables, only rounds at the top-level
  >>> add([2.54, 5.47],['x','y'])
  [2.54, 5.4699999999999998, 'x', 'y']
  >>>
  >>> # does not round elements of iterables, only rounds at the top-level
  >>> add([2.54, 5.47],['x',[8.99, 'y']])
  [2.54, 5.4699999999999998, 'x', [8.9900000000000002, 'y']]
  """
  def dec(f):
    def func(*args, **kwds):
      if tol is None:
        _args,_kwds = args,kwds
      else:
        _simple_round = simple_round_factory(tol)
        _args,_kwds = _simple_round(*args, **kwds)
      return f(*_args, **_kwds)
    return func
  return dec


def shallow_round_factory(tol):
  """helper function for shallow_round (a factory for shallow_round functions)"""
  from numpy import round as around #XXX: try or numpy makes this slow
  def shallow_round(*args, **kwds):
    argstype = type(args) 
    _args = list(args)
    _kwds = kwds.copy()
    for i,j in enumerate(args):
      try:
        jtype = type(j)
        _args[i] = jtype(around(j, tol))
      except: pass
    for i,j in kwds.items():
      try:
        jtype = type(j)
        _kwds[i] = jtype(around(j, tol))
      except: pass
    return argstype(_args), _kwds
  return shallow_round

def shallow_round(tol=0): #NOTE: rounds floats, lists, arrays one level deep
  """decorator for rounding a function's input argument and keywords to the
  given precision *tol*.  This decorator always rounds to a floating point
  number.

  Rounding is done recursively for each element of all arguments and keywords,
  however the rounding is shallow (a max of one level deep into each object).

  For example:
  >>> @shallow_round(tol=1)
  ... def add(x,y):
  ...   return x+y
  ... 
  >>> add(2.54, 5.47)
  8.0
  >>>
  >>> # rounds each float, at the top-level or first-level of each object.
  >>> add([2.54, 5.47],['x','y'])
  [2.5, 5.5, 'x', 'y']
  >>>
  >>> # rounds each float, at the top-level or first-level of each object.
  >>> add([2.54, 5.47],['x',[8.99, 'y']])
  [2.5, 5.5, 'x', [8.9900000000000002, 'y']]
  """
  def dec(f):
    def func(*args, **kwds):
      if tol is None:
        _args,_kwds = args,kwds
      else:
        _shallow_round = shallow_round_factory(tol)
        _args,_kwds = _shallow_round(*args, **kwds)
      return f(*_args, **_kwds)
    return func
  return dec


# EOF

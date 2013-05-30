from mystic.math.rounding import *

@deep_round(tol=1)
def add(x,y):
  return x+y

result = add(2.54, 5.47)
assert result == 8.0

# rounds each float, regardless of depth in an object
result = add([2.54, 5.47],['x','y'])
assert result == [2.5, 5.5, 'x', 'y']

# rounds each float, regardless of depth in an object
result = add([2.54, 5.47],['x',[8.99, 'y']])
assert result == [2.5, 5.5, 'x', [9.0, 'y']]

@simple_round(tol=1)
def add(x,y):
  return x+y

result = add(2.54, 5.47)
assert result == 8.0

# does not round elements of iterables, only rounds at the top-level
result = add([2.54, 5.47],['x','y'])
assert result == [2.54, 5.4699999999999998, 'x', 'y']

# does not round elements of iterables, only rounds at the top-level
result = add([2.54, 5.47],['x',[8.99, 'y']])
assert result == [2.54, 5.4699999999999998, 'x', [8.9900000000000002, 'y']]

@shallow_round(tol=1)
def add(x,y):
  return x+y

result = add(2.54, 5.47)
assert result == 8.0

# rounds each float, at the top-level or first-level of each object.
result = add([2.54, 5.47],['x','y'])
assert result == [2.5, 5.5, 'x', 'y']

# rounds each float, at the top-level or first-level of each object.
result = add([2.54, 5.47],['x',[8.99, 'y']])
assert result == [2.5, 5.5, 'x', [8.9900000000000002, 'y']]


# EOF

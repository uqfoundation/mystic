#!/usr/bin/env python

"""Functions consolidated from samples.py and cut.py in branches/UQ/math/ for general use.
sample() is from cut.py, and everything else is from samples.py"""

def random_samples(lb,ub,npts=10000):
  "generate npts random samples between given lb & ub"
  from numpy.random import random
  dim = len(lb)
  pts = random((dim,npts))
  for i in range(dim):
    pts[i] = (pts[i] * abs(ub[i] - lb[i])) + lb[i]
  return pts

# sample() looks like it only works for n=3
def sample(f,lb,ub,npts=10000):
  from numpy.random import random
  pts = random((3,npts))
  for i in range(3):
    pts[i] = (pts[i] * abs(ub[i] - lb[i])) + lb[i]

  failure = 0; success = 0
  for i in range(npts):
    if f([pts[0][i],pts[1][i],pts[2][i]]):
      success += 1
    else:
      failure += 1
  return failure,success


def sampled_pof(f,pts):
  "use sampling to calculate 'exact' PoF"
  failure = 0
  for i in range(len(pts[0])):
    Fx = f([pts[0][i],pts[1][i],pts[2][i]])
    if not Fx:
      failure += 1
  pof = float(failure) / float(len(pts[0]))
  return pof


def sampled_pts(pts,lb,ub):
  from numpy import inf
  def identity(x):
    return x
  f = wrap_bounds(identity,lb,ub)
  npts = 0
  for i in range(len(pts[0])):
    Fx = f([pts[0][i],pts[1][i],pts[2][i]])
    if Fx != -inf: # outside of bounds evaluates to -inf
      npts += 1
  return npts


def sampled_prob(pts,lb,ub):
  prob = float(sampled_pts(pts,lb,ub)) / float(len(pts[0]))
  return prob


def sampled_mean(model, pts,lb,ub):
  from numpy import inf
  f = wrap_bounds(model,lb,ub)
  ave = 0; count = 0
  for i in range(len(pts[0])):
    Fx = f([pts[0][i],pts[1][i],pts[2][i]])
    if Fx != -inf: # outside of bounds evaluates to -inf
      ave += Fx
      count += 1
  if not count: return None  #XXX: define 0/0 = None
  ave = float(ave) / float(count)
  return ave


def wrap_bounds(f,lb,ub):
  from numpy import asarray, any, inf
  lb = asarray(lb) 
  ub = asarray(ub) 
  def function_wrapper(x): #x bounded on [lb,ub)
    if any((x < lb) | (x >= ub)): #if violates bounds, evaluate as -inf
      return -inf
    return f(x)
  return function_wrapper


# These two functions require scale and model....
#def minF(x):
#  return scale * model(x)


#def maxF(x):
#  return -scale * model(x)


from math import log
def alpha(n,diameter,epsilon=0.01):
 #return diameter * n**(-0.5) * (-log(epsilon))**(0.5)
  return diameter * (-log(epsilon) / (2.0 * n))**(0.5)


#######################################################################

def test1():
  # From branches/UQ/math/samples.py
  num_sample_points = 5
  lower = [10.0, 0.0, 2.1]
  upper = [100.0, 30.0, 2.8]

  pts = random_samples(lower,upper,num_sample_points)
  print pts

def test2():
  # From branches/UQ/math/cut.py
  lower = [-60.0, -10.0, -50.0]
  upper = [105.0, 30.0, 75.0]

  def model(x):
    x1,x2,x3 = x
    if x1 > (x2 + x3): return x1*x2 - x3
    return 0.0
  
  failure,success = sample(model,lower,upper)
  pof = float(failure) / float(failure + success)
  print "Exact PoF: %s" % pof

  print "Exact PoF: %s" % sampled_pof(model,lower,upper)


if __name__ == '__main__':
    test1()
    test2()


# EOF

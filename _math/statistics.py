#!/usr/bin/env python

#######################################################################
# statistics
# probability mass, expectation, mean, diameter, McDiarmid
#######################################################################
def volume(lb,ub):
  """volume for a uniform distribution in n-dimensions"""
  vol = 1
  for i in range(len(ub)):
    vol *= abs(ub[i] - lb[i])
  return vol


def prob_mass(volume,norm):
  """probability mass ..."""
  return volume / norm  #XXX: norm != 0


from scipy.integrate import quad, dblquad, tplquad #FIXME: remove scipy depend!
def expectation_value(f,lb,ub):  #XXX: should be generalized to n-dimensions
  """expectation value for an n-dimensional function; n in [1,2,3]"""
  if len(lb) == 3:
    def func(z,y,x): return f([x,y,z])
    def qf(x,y): return lb[2]
    def rf(x,y): return ub[2]
    def gf(x): return lb[1]
    def hf(x): return ub[1]
    expectation,confidence = tplquad(func,lb[0],ub[0],gf,hf,qf,rf)
    return expectation
  if len(lb) == 2:
    def func(y,x): return f([x,y])
    def gf(x): return lb[1]
    def hf(x): return ub[1]
    expectation,confidence = dblquad(func,lb[0],ub[0],gf,hf)
    return expectation 
  if len(lb) == 1:
    expectation,confidence = quad(f,lb[0],ub[0])
    return expectation 
 #raise Exception, "Function must be either 1-D, 2-D, or 3-D"
 #### FIXME: instead of exception above, use hack for > 3D
  print "WARNING: Dimensions > 3-D are assumed as constant at lower bound"
  def func(z,y,x): return f([x,y,z]+lb[3:])
  def qf(x,y): return lb[2]
  def rf(x,y): return ub[2]
  def gf(x): return lb[1]
  def hf(x): return ub[1]
  expectation,confidence = tplquad(func,lb[0],ub[0],gf,hf,qf,rf)
  return expectation
 #### END HACK


def mean(expectation,volume):
  """mean ..."""
  return expectation / volume  #XXX: volume != 0

def integrated_mean(f, lb, ub):
  """Puts together the functions mean, volume, and expectation_value."""
  expectation = expectation_value(f, lb, ub)
  vol = volume(lb, ub)
  return mean(expectation, vol)


def mcdiarmid_bound(mean,diameter):
  """McDiarmid ..."""
  from math import exp
  if not diameter: return 1.0  #XXX: define e^(0/0) = 1
  return exp(-2.0 * (max(0,mean))**2 / diameter**2) #XXX: (or diameter**1 ?)


#######################################################################
if __name__ == '__main__':

  bounds = [(0.1,1.0),(2.0,4.0),(4.0,9.0)]
  lower = [i[0] for i in bounds]
  upper = [i[1] for i in bounds]

  cuboid_volume = volume(lower,upper)
  probability_mass = prob_mass(cuboid_volume,cuboid_volume)
  print " probability mass: %s" % probability_mass

# EOF

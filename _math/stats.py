#!/usr/bin/python
"""
shortcut (?) math tools related to statistics;
also, math tools related to gaussian distributions
"""
import math

#----------------------------------------------------------------
# Gaussian helper functions

def erf(x):
  """Error function approximation. 
Source: http://www.johndcook.com/python_erf.html
"""
  # constants
  a1 =  0.254829592
  a2 = -0.284496736
  a3 =  1.421413741
  a4 = -1.453152027
  a5 =  1.061405429
  p  =  0.3275911

  # Save the sign of x
  sign = 1
  if x < 0:
    sign = -1
  x = abs(x)

  # A&S formula 7.1.26
  t = 1.0/(1.0 + p*x)
  y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)
  return sign*y


def cdf_factory(mean, variance):
  """
Returns cumulative distribution function (as a Python function)
for a Gaussian, given the mean and variance
"""
  def cdf(x):
    """Cumulative distribution function."""
    try:
      from scipy.special import erf
    except ImportError:
      pass
    return 0.5*(1. + erf((x-mean)/math.sqrt(2.*variance)))
  return cdf


def pdf_factory(mean, variance):
  """
Returns a probability density function (as a Python function)
for a Gaussian, given the mean and variance
"""
  def pdf(x):
    """Probability density function."""
    return 1./math.sqrt(2.*math.pi*variance)* \
           math.exp(-(x - mean)**2/(2.*variance))
  return pdf


#----------------------------------------------------------------
# shortcut (?) statistics functions

def volume(lb,ub):
  """calculates volume for a uniform distribution in n-dimensions"""
  vol = 1
  for i in range(len(ub)):
    vol *= abs(ub[i] - lb[i])
  return vol


def prob_mass(volume,norm):
  """calculates probability mass given volume and norm"""
  return volume / norm  #XXX: norm != 0


def mean(expectation,volume):
  """calculates mean given expectation and volume"""
  return expectation / volume  #XXX: volume != 0


def mcdiarmid_bound(mean,diameter):
  """calculates McDiarmid bound given mean and McDiarmid diameter"""
  if not diameter: return 1.0  #XXX: define e^(0/0) = 1
  return math.exp(-2.0 * (max(0,mean))**2 / diameter**2) #XXX: or diameter**1 ?


def __mean(xarr):
  """mean = x.sum() / len(x)""" #interface is [lb,ub]; not lb,ub
  from numpy import mean
  return mean(xarr)


def __variance(xarr):
  """var = mean(abs(x - x.mean())**2) / 3""" #interface is [lb,ub]; not lb,ub
  from numpy import var
  return var(xarr) / 3.0


#----------------------------------------------------------------------------
# Tests

def __test_probability_mass():
  bounds = [(0.1,1.0),(2.0,4.0),(4.0,9.0)]
  lower = [i[0] for i in bounds]
  upper = [i[1] for i in bounds]

  cuboid_volume = volume(lower,upper)
  probability_mass = prob_mass(cuboid_volume,cuboid_volume)
  print "probability mass: %s" % probability_mass


if __name__ == '__main__':
  __test_probability_mass()

# EOF

#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
shortcut (?) math tools related to statistics;
also, math tools related to gaussian distributions
"""
import math

#----------------------------------------------------------------
# Gaussian helper functions

def erf(x):
    "evaluate the error function at x" 
    try:                                               
        from math import erf
    except ImportError:
        try:
            from ctypes import util, CDLL, c_double
            erf = CDLL(util.find_library('m')).erf
            erf.argtypes = [c_double]
            erf.restype = c_double
        except ImportError:
            erf = _erf
    if hasattr(x, '__len__'):
        from numpy import vectorize
        erf = vectorize(erf)
    return erf(x)

def gamma(x):
    "evaluate the gamma function at x"
    try:
        from math import gamma
    except ImportError:
        try:
            from ctypes import util, CDLL, c_double
            gamma = CDLL(util.find_library('m')).gamma
            gamma.argtypes = [c_double]
            gamma.restype = c_double
        except ImportError:
            gamma = _gamma
    if hasattr(x, '__len__'):
        from numpy import vectorize
        gamma = vectorize(gamma)
    return gamma(x) 

def lgamma(x):
    "evaluate the natual log of the abs value of the gamma function at x"
    try:
        from math import lgamma
    except ImportError:
        try:
            from ctypes import util, CDLL, c_double
            lgamma = CDLL(util.find_library('m')).lgamma
            lgamma.argtypes = [c_double]
            lgamma.restype = c_double
        except ImportError:
            lgamma = _lgamma
    if hasattr(x, '__len__'):
        from numpy import vectorize
        lgamma = vectorize(lgamma)
    return lgamma(x) 


def _erf(x):
    "approximate the error function at x"
    # Source: http://picomath.org/python/erf.py.html
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


def _gamma(x):
    "approximate the gamma function at x"
    # Source: http://picomath.org/python/gamma.py.html
    if x <= 0:
        raise ValueError("Invalid input")

    # Split the function domain into three intervals:
    # (0, 0.001), [0.001, 12), and (12, infinity)

    ###########################################################################
    # First interval: (0, 0.001)
    #
    # For small x, 1/Gamma(x) has power series x + gamma x^2  - ...
    # So in this range, 1/Gamma(x) = x + gamma x^2 with error order of x^3.
    # The relative error over this interval is less than 6e-7.

    gamma = 0.577215664901532860606512090 # Euler's gamma constant

    if x < 0.001:
        return 1.0/(x*(1.0 + gamma*x))

    ###########################################################################
    # Second interval: [0.001, 12)

    if x < 12.0:
        # The algorithm directly approximates gamma over (1,2) and uses
        # reduction identities to reduce other arguments to this interval.
        
        y = x
        n = 0
        arg_was_less_than_one = (y < 1.0)

        # Add or subtract integers as necessary to bring y into (1,2)
        # Will correct for this below
        if arg_was_less_than_one:
            y += 1.0
        else:
            n = int(math.floor(y)) - 1  # will use n later
            y -= n

        # numerator coefficients for approximation over the interval (1,2)
        p = [
            -1.71618513886549492533811E+0,
             2.47656508055759199108314E+1,
            -3.79804256470945635097577E+2,
             6.29331155312818442661052E+2,
             8.66966202790413211295064E+2,
            -3.14512729688483675254357E+4,
            -3.61444134186911729807069E+4,
             6.64561438202405440627855E+4
        ]

        # denominator coefficients for approximation over the interval (1,2)
        q = [
            -3.08402300119738975254353E+1,
             3.15350626979604161529144E+2,
            -1.01515636749021914166146E+3,
            -3.10777167157231109440444E+3,
             2.25381184209801510330112E+4,
             4.75584627752788110767815E+3,
            -1.34659959864969306392456E+5,
            -1.15132259675553483497211E+5
        ]

        num = 0.0
        den = 1.0

        z = y - 1
        for i in range(8):
            num = (num + p[i])*z
            den = den*z + q[i]
        result = num/den + 1.0

        # Apply correction if argument was not initially in (1,2)
        if arg_was_less_than_one:
            # Use identity gamma(z) = gamma(z+1)/z
            # The variable "result" now holds gamma of the original y + 1
            # Thus we use y-1 to get back the orginal y.
            result /= (y-1.0)
        else:
            # Use the identity gamma(z+n) = z*(z+1)* ... *(z+n-1)*gamma(z)
            for _ in range(n):
                result *= y
                y += 1

        return result

    ###########################################################################
    # Third interval: [12, infinity)

    if x > 171.624:
        # Correct answer too large to display. 
        return 1.0/0 # float infinity

    return math.exp(log_gamma(x))


def _lgamma(x):
    "approximate the natual log of the abs value of the gamma function at x"
    # Source: http://picomath.org/python/gamma.py.html
    if x <= 0:
        raise ValueError("Invalid input")

    if x < 12.0:
        return math.log(abs(gamma(x)))

    # Abramowitz and Stegun 6.1.41
    # Asymptotic series should be good to at least 11 or 12 figures
    # For error analysis, see Whittiker and Watson
    # A Course in Modern Analysis (1927), page 252

    c = [
         1.0/12.0,
        -1.0/360.0,
         1.0/1260.0,
        -1.0/1680.0,
         1.0/1188.0,
        -691.0/360360.0,
         1.0/156.0,
        -3617.0/122400.0
    ]
    z = 1.0/(x*x)
    sum = c[7]
    for i in range(6, -1, -1):
        sum *= z
        sum += c[i]
    series = sum/x

    halfLogTwoPi = 0.91893853320467274178032973640562
    logGamma = (x - 0.5)*math.log(x) - x + halfLogTwoPi + series
    return logGamma

#----------------------------------------------------------------

def _lefttail(percent):
    "calculate left-area percent from center-area percent"
    return 100 - .5*(100 - percent)

def stderr(std, npts):
    "standard error"
    return std/npts**.5

def meanconf(std, npts, percent=95):
    "mean confidence interval: returns conf, where interval = mean +/- conf"
    import scipy.stats as ss
    scale = ss.norm.ppf(.01*_lefttail(percent))
    return scale * stderr(std, npts)

def sampvar(var, npts):
    "sample variance from variance"
    return var * npts/(npts-1)

def _varconf(var, npts, percent=95):
    "var confidence interval: returns interval, where var in interval"
    s = var * npts
    import scipy.stats as ss
    chihi = ss.chi2.isf(.01*_lefttail(percent), npts-1)
    chilo = ss.chi2.isf(1 - .01*_lefttail(percent), npts-1)
    return s/chihi, s/chilo

def varconf(var, npts, percent=95, tight=False):
    "var confidence interval: returns max interval distance from var"
    hi,lo = _varconf(var, npts, percent)
    select = min if tight else max
    return select(abs(hi-var),abs(var-lo))

#----------------------------------------------------------------

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
  print("probability mass: %s" % probability_mass)


if __name__ == '__main__':
  __test_probability_mass()

# EOF

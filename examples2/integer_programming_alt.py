#!/usr/bin/env python
#
# Problem definition:
# Example in google/or-tools
# https://github.com/google/or-tools/blob/master/examples/python/integer_programming.py
# with Copyright 2010-2013 Google
# and disclamer as stated at the above reference link.
# 
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from integer_programming import objective, bounds, xs, ys
# bounds = [(0,11)]*2  #XXX: MOD = range(11) instead of LARGE

from mystic.penalty import quadratic_inequality, quadratic_equality
from mystic.constraints import as_constraint, discrete

def penalty1(x): # <= 0.0
    return -3*x[0] - 2*x[1] + 17.0

def penalty2(x): # == 0.0
    from numpy import abs, round
    return abs(x - round(x)).sum() # penalize when not an 'int'

#@quadratic_equality(penalty2)
@quadratic_inequality(penalty1)
def penalty(x):
    return 0.0

solver = as_constraint(penalty)
#solver = discrete(range(11))(solver)  #XXX: MOD = range(11) instead of LARGE
#FIXME: constrain to 'int' with discrete is very fragile!  required #MODs

def constraint(x):
    from numpy import round
    return round(solver(x))

# better is to constrain to integers, penalize otherwise
from mystic.constraints import integers

@integers()
def round(x):
  return x


if __name__ == '__main__':

    from mystic.solvers import diffev2
    from mystic.math import almostEqual

    result = diffev2(objective, x0=bounds, bounds=bounds, penalty=penalty, constraints=round, npop=30, gtol=50, disp=True, full_output=True)

    print(result[0])
    assert almostEqual(result[0], xs, tol=1e-8) #XXX: fails b/c rel & zero?
    assert almostEqual(result[1], ys, tol=1e-4)


# EOF

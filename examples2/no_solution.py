#!/usr/bin/env python
#
# Problem definition and original response:
# https://stackoverflow.com/q/12942153/2379433
# https://stackoverflow.com/a/43173143/2379433
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2018 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''
Attempt to solve equations with no solution, only using bounds constraints.
'''
from mystic import reduced

@reduced(lambda x,y: abs(x)+abs(y)) #choice changes answer
def objective(x, a, b, c):
  x,y = x
  eqns = (\
    (x - a[0])**2 + (y - b[0])**2 - c[0]**2,
    (x - a[1])**2 + (y - b[1])**2 - c[1]**2,
    (x - a[2])**2 + (y - b[2])**2 - c[2]**2)
  return eqns

bounds = [(None,None),(None,None)] #unnecessary

a = (0,2,0)
b = (0,0,2)
c = (.88,1,.75)
args = a,b,c

from mystic.solvers import diffev2
from mystic.monitors import VerboseMonitor
mon = VerboseMonitor(10)

result = diffev2(objective, args=args, x0=bounds, bounds=bounds, npop=40, \
                 ftol=1e-8, disp=False, full_output=True, itermon=mon)

print(result[0])
print(result[1])

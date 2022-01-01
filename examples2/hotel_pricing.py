#!/usr/bin/env python
#
# Problem definition and original response:
# https://stackoverflow.com/q/48088516/2379433
# https://stackoverflow.com/a/48494431/2379433
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2018-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
Maximize:

  sum_{i=1 to max i} P_i O_i


Subject to:

  O_i <= C_i
  P_i <= 0


where the decision variables are:

  P_i: price allocated for night i


and these are the computed auxiliary variables:

  X_a,L: number of rooms allocated for stay of type (a,L)


defined as:

  X_a,L = d_a,L * ([sum_{i=a to a+L-i} P_i]/[L * P_nominal])^e


and:

  O_i: number of rooms reserved in a given night


defined as:

  O_i = sum_{a,L in N_i} X_a,L


The input parameters are:

  P_nominal: nominal price of the hotel (average historical price)

  e: elasticity between price and demand

  d_a,L and N_i: as defined in the classical model

  C_i: total number of rooms available on the hotel

"""
import math
n_days = 7
n_rooms = 50
P_nom = 85
P_bounds = 0,None
P_elastic = 2

class hotel(object):
    def __init__(self, rooms, price_ave, price_elastic):
        self.rooms = rooms
        self.price_ave = price_ave
        self.price_elastic = price_elastic

    def profit(self, P):
        # assert len(P) == len(self.rooms)
        return sum(j * self._reserved(P, i) for i,j in enumerate(P))

    def remaining(self, P): # >= 0
        C = self.rooms
        # assert len(P) == C
        return [C[i] - self._reserved(P, i) for i,j in enumerate(P)]

    def _reserved(self, P, day):
        max_days = len(self.rooms)
        As = range(0, day)
        return sum(self._allocated(P, a, L) for a in As
                   for L in range(day-a+1, max_days+1))
        
    def _allocated(self, P, a, L):
        P_nom = self.price_ave
        e = self.price_elastic
        return math.ceil(self._demand(a, L)*(sum(P[a:a+L])/(P_nom*L))**e)

    def _demand(self, a,L):
        return abs(1-a)/L + 2*(a**2)/L**2


h = hotel([n_rooms]*n_days, P_nom, P_elastic)

def objective(price, hotel):
    return -hotel.profit(price) 

def constraint(price, hotel): # <= 0
    return -min(hotel.remaining(price))

bounds = [P_bounds]*n_days
guess = [P_nom]*n_days

import mystic as my

@my.penalty.quadratic_inequality(constraint, kwds=dict(hotel=h))
def penalty(x):
    return 0.0

# using a local optimizer, starting from the nominal price
solver = my.solvers.fmin
mon = my.monitors.VerboseMonitor(50)

kwds = dict(disp=True, full_output=True, itermon=mon,
            args=(h,),  xtol=1e-8, ftol=1e-8, maxfun=10000, maxiter=2000)
result = solver(objective, guess, bounds=bounds, penalty=penalty, **kwds)

print([round(i,2) for i in result[0]])


# however, we can do better using a global optimizer
solver = my.solvers.diffev
mon = my.monitors.VerboseMonitor(50)

kwds = dict(disp=True, full_output=True, itermon=mon, npop=40,
            args=(h,),  gtol=250, ftol=1e-8, maxfun=30000, maxiter=2000)
result = solver(objective, bounds, bounds=bounds, penalty=penalty, **kwds)

print([round(i,2) for i in result[0]])

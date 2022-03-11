#!/usr/bin/env python
# 
# Problem definition and original response:
# https://stackoverflow.com/q/69655167
# https://stackoverflow.com/a/69885831
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2021-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''
Solve a bounded knapsack problem.

Maximize:
  profit = SUM_i (quantity_i * profit_i)

where:
  profit_i = sell_i - buy_i

We have a list of items that we can ship in a truck. Each item has:
  - a buy price (at the source)
  - a sell price (at the destination)
  - a per-unit mass
  - an upper limit on how many can be purchased

Let's say we have 10 items, with:
  buy_price: [123, 104, 149, 175, 199, 120, 164, 136, 194, 111]
  profit: [13, 24, 10, 29, 29, 39, 28, 35, 33, 39] 
  unit_mass: [10, 15, 20, 18, 34, 75, 11, 49, 68, 55]
  item_limit: [300, 500, 200, 300, 200, 350, 100, 600, 1000, 50]

Also, we have constraints:
  - our truck is limited in the amount of mass it can carry
  - we have an upper limit on how much we can "invest" (spend at the source)

Let's say we have:
max_load = 75000  # max limit on mass can carry
max_spend = 350000  # max limit to spend at source


Which items, and what quantity of each, should be purchased to maximize profit?
'''
import mystic as my
import mystic.symbolic as ms
import mystic.constraints as mc

class item(object):
    def __init__(self, id, mass, buy, net, limit):
        self.id = id
        self.mass = mass
        self.buy = buy
        self.net = net
        self.limit = limit
    def __repr__(self):
        return 'item(%s, mass=%s, buy=%s, net=%s, limit=%s)' % (self.id, self.mass, self.buy, self.net, self.limit)

# data
buy_price = [123, 104, 149, 175, 199, 120, 164, 136, 194, 111]
profit = [13, 24, 10, 29, 29, 39, 28, 35, 33, 39] 
unit_mass = [10, 15, 20, 18, 34, 75, 11, 49, 68, 55]
item_limit = [300, 500, 200, 300, 200, 350, 100, 600, 1000, 50]
ids = range(len(item_limit))

# maxima
max_load = 75000  # max limit on mass can carry
max_spend = 350000  # max limit to spend at source

# items
items = [item(*i) for i in zip(ids, unit_mass, buy_price, profit, item_limit)]

# profit
def fixnet(net):
    def profit(x):
        return sum(xi*pi for xi,pi in zip(x,net))
    return profit

profit = fixnet([i.net for i in items])

# item constraints
load = [i.mass for i in items]
invest = [i.buy for i in items]
constraints = ms.linear_symbolic(G=[load, invest], h=[max_load, max_spend])

# bounds (on x)
bounds = [(0, i.limit) for i in items]

# bounds constraints
lo = 'x%s >= %s'
lo = '\n'.join(lo % (i,str(float(j[0])).lstrip('0')) for (i,j) in enumerate(bounds))
hi = 'x%s <= %s'
hi = '\n'.join(hi % (i,str(float(j[1])).lstrip('0')) for (i,j) in enumerate(bounds))
constraints = '\n'.join([lo, hi]).strip() + '\n' + constraints
cf = ms.generate_constraint(ms.generate_solvers(ms.simplify(constraints)), join=mc.and_)
pf = ms.generate_penalty(ms.generate_conditions(ms.simplify(constraints)))


# integer constraints
#constrain = mc.and_(mc.integers(float)(lambda x:x), cf)
constrain = mc.integers(float)(lambda x:x)

# solve
mon = my.monitors.VerboseMonitor(10)
result = my.solvers.diffev2(lambda x: -profit(x), bounds, npop=400, bounds=bounds, ftol=1e-6, gtol=100, itermon=mon, disp=True, full_output=True, constraints=constrain, penalty=pf)

result, cost = result[:2]
print('\nmax profit: %s' % -cost)
print("load: %s <= %s" % (sum(i*j for i,j in zip(result, load)), max_load))
print("spend: %s <= %s" % (sum(i*j for i,j in zip(result, invest)), max_spend))
print('')

for item,quantity in enumerate(result):
    print("item %d: %s" % (item, quantity))


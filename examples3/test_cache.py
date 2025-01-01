#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2022-2025 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''
create a standard file archive that captures function input and output, and retains calling order
'''
# generate a function that writes to the archive, where it:
#  - uses function input as the key, and function output as the value
#  - is ordered by function call
import mystic.cache as mc
a = mc.archive.read('cache.db', type=mc.archive.file_archive)
from toys import function5
objective = mc.cached(archive=a)(function5)

# set some bounds
xlb = (0,1,0,0,0)
xub = (1,10,10,10,10)
bounds = list(zip(xlb, xub))

# optimize, using a evaluation monitor
import mystic as my
mon = my.monitors.Monitor()
my.solvers.fmin(objective, x0=xlb, bounds=bounds, evalmon=mon, full_output=True, disp=False)

# archived rv and cost should match evaluation monitor, regardless of size
# (to look up the last entry, we need to use keys(), so we just check all keys)
# file archives preserve order, but we convert to nested lists to compare
xs,ys = list(list(k) for k in a.keys()), list(a.values())
assert xs == mon.x
assert ys == mon.y

# remove the archive
import os
os.remove('cache.db')


'''
** alternate **
create a limited-size customized directory archive that captures function input and output, and retains calling order
'''
# generate a global counter
from mystic._counter import Counter
counter = Counter(0)
run = lambda : counter.count()

# generate a function that writes to the archive, where it:
#  - uses run number as the key
#  - uses the given kwds as the values
#  - any kwds input to the cached function is ignored in the keys
#  - has a maximum archive size of 500
import mystic.cache as mc
@mc.cached(type=mc.lru_cache, archive='cache', maxsize=500, ignore='**')
def archive(run, **kwds): # key: run[int]
    return kwds # value: kwds[dict]

# embed writing to the archive within the objective
def objective(rv):
    from toys import function5
    cost = function5(rv)
    archive([run()], rv=rv, cost=cost)
    return cost

# set some bounds
xlb = (0,1,0,0,0)
xub = (1,10,10,10,10)
bounds = list(zip(xlb, xub))

# optimize, using a evaluation monitor
import mystic as my
mon = my.monitors.Monitor()
my.solvers.fmin(objective, x0=xlb, bounds=bounds, evalmon=mon, full_output=True, disp=False)

# read the archive
a = mc.archive.read('cache')

# archived rv and cost should match evaluation monitor
# (for up to 500 function evaluations -- the max size of the archive)
# hence, we will just check the last entry 
last = a[len(a)-1]
assert last['rv'] == mon.x[-1]
assert last['cost'] == mon.y[-1]

# directory archives do not preserve order, so we need to sort by key
xs,ys = zip(*[(a[i]['rv'],a[i]['cost']) for i in sorted(a.keys())])
assert list(xs) == mon.x
assert list(ys) == mon.y

# remove the archive
import shutil
shutil.rmtree('cache')

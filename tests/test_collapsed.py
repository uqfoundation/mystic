#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

import mystic.collapse as ct
import mystic.mask as ma
import mystic as my
m = my.monitors._load('_log.py')
# cleanup *pyc
import os
try: os.remove('_log.pyc')
except OSError: pass

import mystic.termination as mt
from mystic.solvers import DifferentialEvolutionSolver
solver = DifferentialEvolutionSolver(2*sum(m._npts))
solver.SetRandomInitialPoints()
solver.SetGenerationMonitor(m)
##############################################

# print('collapse_at')
ix = ct.collapse_at(m)
# (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17)
assert not ct.collapse_at(m, mask=ix)
ix = ct.collapse_at(m, target=0.0)
# (1, 7, 8, 9, 10, 11, 12, 14)
assert not ct.collapse_at(m, target=0.0, mask=ix)
ix = ct.collapse_at(m, target=m.x[-1]) 
assert not ct.collapse_at(m, target=m.x[-1], mask=ix)

# print('collapse_as')
ix = ct.collapse_as(m)
# set([(10, 11), (7, 12), (10, 12), (8, 9), (11, 14), (7, 11), (1, 11), (16, 17), (8, 14), (1, 14), (8, 10), (9, 11), (7, 10), (1, 10), (7, 14), (9, 14), (12, 14), (8, 11), (9, 10), (1, 9), (11, 12), (7, 9), (1, 12), (8, 12), (3, 4), (1, 8), (10, 14), (6, 13), (1, 7), (7, 8), (9, 12)])
assert not ct.collapse_as(m, mask=ix)

# print('collapse_weight')
ix = ct.collapse_weight(m)
# {0: {1}, 1: {1, 2}, 2: {0, 2}}
assert not ct.collapse_weight(m, mask=ix)

# print('collapse_position')
ix = ct.collapse_position(m)
# {0: {(0,1)}, 1:{(0,1),(0,2),(1,2)}, 2:{(1,2)}}
assert not ct.collapse_position(m, mask=ix)

##############################################
# print(mt.CollapseAt()(solver, True))
# print(mt.CollapseAt(target=0.0)(solver, True))
# print(mt.CollapseAs()(solver, True))
# print(mt.CollapseWeight()(solver, True))
# print(mt.CollapseWeight(mask=set())(solver, True))
# print(mt.CollapseWeight(mask=())(solver, True))
# print(mt.CollapsePosition()(solver, True))
# print(mt.CollapsePosition(mask=set())(solver, True))
# print(mt.CollapsePosition(mask=())(solver, True))

def test_cc(termination, seed=None):
    term = termination(mask=seed)
    collapse = ct.collapsed(term(solver, True))
    mask = list(collapse.values())[-1]
    # full results as mask
    _term = termination(mask=mask)
    assert not ct.collapsed(_term(solver, True))
    # update mask with full results
    _term = ma.update_mask(term, collapse)
    assert mask == ma.get_mask(_term)
    assert not ct.collapsed(_term(solver, True))

for term in (mt.CollapseAt, mt.CollapseAs):
    test_cc(term)

for term in (mt.CollapseWeight, mt.CollapsePosition):
    for seed in (None, {}, set(), ()):
        test_cc(term, seed)

##############################################
stop = mt.Or((mt.ChangeOverGeneration(), \
              mt.CollapseWeight(), \
              mt.CollapsePosition()))

message = stop(solver, True)

_stop = ma.update_mask(stop, ct.collapsed(message))

message = _stop(solver, True)
assert message
collapse = ct.collapsed(message)
assert not collapse
stop_ = ma.update_mask(_stop, collapse)
assert mt.state(stop_) == mt.state(_stop)



# EOF

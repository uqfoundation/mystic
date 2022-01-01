#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

import numpy as np
import mystic.math.measures as mm
x = np.arange(1., 5)
y = np.array([-1., -2, 0, 3])
z = np.array([-1., -2, 0, 7])

# L1 and L2 norms

q = mm.normalize(x, mass='l1', zsum=True)
p = [0.10000000000000001, 0.20000000000000001, 0.29999999999999999, 0.40000000000000002]
assert np.sum(np.array(q) - p) <= 1e-15

q = mm.normalize(x, mass='l2', zsum=True)
p = [0.18257418583505536, 0.36514837167011072, 0.54772255750516607, 0.73029674334022143]
assert np.sum(np.array(q) - p) <= 1e-15

q = mm.normalize(y, mass='l2', zsum=True)
p = [-0.2672612419124244, -0.53452248382484879, 0.0, 0.80178372573727319]
assert np.sum(np.array(q) - p) <= 1e-15

q = mm.normalize(y, mass='l1', zsum=True)
p = [-0.16666666666666666, -0.33333333333333331, 0.0, 0.5]
assert np.sum(np.array(q) - p) <= 1e-15

q = mm.normalize(z, mass='l1', zsum=True)
p = [-0.10000000000000001, -0.20000000000000001, 0.0, 0.69999999999999996]
assert np.sum(np.array(q) - p) <= 1e-15

q = mm.normalize(z, mass='l2', zsum=True)
p = [-0.13608276348795434, -0.27216552697590868, 0.0, 0.95257934441568037]
assert np.sum(np.array(q) - p) <= 1e-15

# fixed mass

q = mm.normalize(x, mass=1, zsum=True)
p = [0.10000000000000001, 0.20000000000000001, 0.29999999999999999, 0.40000000000000002]
assert np.sum(np.array(q) - p) <= 1e-15
assert np.sum(q) - 1 <= 1e-15

q = mm.normalize(x, mass=0, zsum=True)
p = [0.10000000000000001, 0.20000000000000001, 0.29999999999999999, -0.59999999999999998]
assert np.sum(np.array(q) - p) <= 1e-15
assert np.sum(q) - 0 <= 1e-15

q = mm.normalize(x, mass=0, zsum=False)
p = [0.0, 0.0, 0.0, 0.0]
assert np.sum(np.array(q) - p) <= 1e-15
assert np.sum(q) - 0 <= 1e-15

q = mm.normalize(x, mass=1, zsum=False)
p = [0.10000000000000001, 0.20000000000000001, 0.29999999999999999, 0.40000000000000002]
assert np.sum(np.array(q) - p) <= 1e-15
assert np.sum(q) - 1 <= 1e-15

q = mm.normalize(y, mass=1, zsum=False)
p = [-np.inf, -np.inf, np.nan, np.inf] #XXX: is this the desired answer?
# assert np.sum(np.array(q) - p) <= 1e-15

q = mm.normalize(y, mass=0, zsum=False)
p = [-0.0, -0.0, 0.0, 0.0]
assert np.sum(np.array(q) - p) <= 1e-15
assert np.sum(q) - 0 <= 1e-15

q = mm.normalize(y, mass=0, zsum=True)
p = [-0.16666666666666666, -0.33333333333333331, 0.0, 0.5]
assert np.sum(np.array(q) - p) <= 1e-15
assert np.sum(q) - 0 <= 1e-15

q = mm.normalize(y, mass=1, zsum=True)
p = [-np.inf, -np.inf, np.nan, np.inf] #XXX: is this the desired answer?
# assert np.sum(np.array(q) - p) <= 1e-15

q = mm.normalize(z, mass=0, zsum=False)
p = [-0.0, -0.0, 0.0, 0.0]
assert np.sum(np.array(q) - p) <= 1e-15
assert np.sum(q) - 0 <= 1e-15

q = mm.normalize(z, mass=1, zsum=False)
p = [-0.25000000000000006, -0.50000000000000011, 0.0, 1.7500000000000002]
assert np.sum(np.array(q) - p) <= 1e-15
assert np.sum(q) - 1 <= 1e-15

q = mm.normalize(z, mass=1, zsum=True)
p = [-0.25000000000000006, -0.50000000000000011, 0.0, 1.7500000000000002]
assert np.sum(np.array(q) - p) <= 1e-15
assert np.sum(q) - 1 <= 1e-15

q = mm.normalize(z, mass=0, zsum=True)
p = [-0.10000000000000001, -0.20000000000000001, 0.0, 0.29999999999999999]
assert np.sum(np.array(q) - p) <= 1e-15
assert np.sum(q) - 0 <= 1e-15

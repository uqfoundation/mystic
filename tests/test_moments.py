#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

import mystic.math.measures as mo
from numpy import nan, isnan
x = [1,2,3,4,5]
y = [1,2,3,4,5,6]
z = [1,5,5,5,5,5]
w = [1,2,3,4,10]

for i in (x,y,z,w):
    assert mo.moment(i, order=1) == 0.0
    assert mo.impose_moment(0.0, i, order=1) == i
    assert sum(isnan(mo.impose_moment(5.0, i, order=1))) == len(i)
    assert mo.impose_moment(0.0, i, order=2) == [mo.mean(i)]*len(i)
    assert sum(isnan(mo.impose_moment(5.0, i, order=2))) == 0
    assert mo.impose_moment(0.0, i, order=3) == [mo.mean(i)]*len(i)
    assert sum(isnan(mo.impose_moment(5.0, i, order=3))) == 0

assert sum(isnan(mo.impose_moment(5.0, x, order=3, skew=False))) == len(x)
assert sum(isnan(mo.impose_moment(5.0, y, order=3, skew=False))) == 0 #XXX?
assert sum(isnan(mo.impose_moment(5.0, z, order=3, skew=False))) == 0
assert sum(isnan(mo.impose_moment(5.0, w, order=3, skew=False))) == 0

for i in (x,y,z,w):
    tol = 1e-12
    _i = [max(i)+min(i)-j for j in i]
    assert mo.moment(i, order=3) + mo.moment(_i, order=3) <= tol
    assert mo.moment(i, order=5) + mo.moment(_i, order=5) <= tol

for i in (x,y,z,w):
    t = 5.0
    q = mo.impose_moment(t, i, order=3)
    s = mo.moment(q, order=3)
    assert round(t, 12) == round(s, 12)
    t = -t
    q = mo.impose_moment(t, i, order=3)
    s = mo.moment(q, order=3)
    assert round(t, 12) == round(s, 12)

for i in (x,y,z,w):
    t = 5.0
    q = mo.impose_moment(t, i, order=2)
    s = mo.moment(q, order=2)
    assert round(t, 12) == round(s, 12)
    t = -t
    q = mo.impose_moment(t, i, order=2)
    assert sum(isnan(q)) == len(i)


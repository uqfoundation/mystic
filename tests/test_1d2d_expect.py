#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE

from __future__ import division
from past.utils import old_div
from mystic.math.measures import *
from mystic.math.discrete import *
from mystic.math import almostEqual
def f(x): return old_div(sum(x),len(x))

def test_setexpect(m, expect=5.0, tol=0.001):
    e = (expect, tol)
    #e_lo = expect - expect
    #e_hi = expect + expect
    m.set_expect(e, f)#, bounds=([e_lo]*m.npts, [e_hi]*m.npts)) 
    assert almostEqual(m.expect(f), e[0], tol=e[1])

def test_expect(m, f):
    pos = m.positions
    if not (len(pos) and isinstance(pos[0], tuple)): # then m is a measure
        pos = [[i] for i in pos]
    expect = expectation(f, pos, m.weights)
    assert almostEqual(m.expect(f), expect)


if __name__ == '__main__':
    d2 = measure([point_mass(1.0, 1.0), point_mass(3.0, 2.0)])
    d1 = measure([point_mass(3.0, 2.0)])
    d2b = measure([point_mass(2.0, 4.0), point_mass(4.0, 2.0)])
    p1 = product_measure([d1])
    p1b = product_measure([d2])
    p2 = product_measure([d2,d2b])
    p21 = product_measure([d2,d1])

    # expect for product_measure and measure
    test_expect(d2, f)
    test_expect(d2b, f)
    test_expect(p2, f)

    # set_expect for measure
    test_setexpect(d2)
    test_setexpect(d2b)

    # set_expect for product_measure
    test_setexpect(p2)

    # again, but for single-point measures
    # expect for product_measure and measure
    test_expect(d1, f)
    test_expect(p1, f)
    test_expect(p1b, f)

    # set_expect for measure
    test_setexpect(d1)

    # set_expect for product_measure
    test_setexpect(p1)
    test_setexpect(p1b)

    # again, but for mixed measures
    test_setexpect(p21)


# EOF

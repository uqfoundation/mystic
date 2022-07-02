#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from mystic.math.measures import *
from mystic.math.discrete import *
from mystic.math import almostEqual
def f(x): return sum(x)/len(x)

def test_setexpect(m, expect=5.0, tol=0.001):
    #e_lo = expect - expect
    #e_hi = expect + expect
    m.set_expect(expect, f, tol=tol)#, bounds=([e_lo]*m.npts, [e_hi]*m.npts)) 
    assert almostEqual(m.expect(f), expect, tol=tol)

def test_setexpectvar(m, expect=5.0, tol=0.001):
    m.set_expect_var(expect, f, tol=tol)
    assert almostEqual(m.expect_var(f), expect, tol=tol)

def test_setexpectmeanvar(m, expect=5.0, tol=0.001):
    m.set_expect_mean_and_var((expect,expect), f, tol=tol)
    assert almostEqual(m.expect(f), expect, tol=tol)
    assert almostEqual(m.expect_var(f), expect, tol=tol)

def test_expect(m, f):
    pos = m.positions
    if not (len(pos) and isinstance(pos[0], tuple)): # then m is a measure
        pos = [[i] for i in pos]
    expect = expectation(f, pos, m.weights)
    assert almostEqual(m.expect(f), expect)

def test_expectvar(m, f):
    pos = m.positions
    if not (len(pos) and isinstance(pos[0], tuple)): # then m is a measure
        pos = [[i] for i in pos]
    expect = expected_variance(f, pos, m.weights)
    assert almostEqual(m.expect_var(f), expect)

def fail(f, val):
    try:
        f(val)
        raise AssertionError('should throw ValueError')
    except ValueError:
        pass
    except:
        raise AssertionError('should throw ValueError')


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
    test_expectvar(d2, f)
    test_expectvar(d2b, f)
    test_expectvar(p2, f)

    # set_expect for measure
    test_setexpect(d2)
    test_setexpect(d2b)
    test_setexpectvar(d2)
    test_setexpectvar(d2b)
    test_setexpectmeanvar(d2)
    test_setexpectmeanvar(d2b)

    # set_expect for product_measure
    test_setexpect(p2)
    test_setexpectvar(p2)
    test_setexpectmeanvar(p2)

    # again, but for single-point measures
    # expect for product_measure and measure
    test_expect(d1, f)
    test_expect(p1, f)
    test_expect(p1b, f)
    test_expectvar(d1, f)
    test_expectvar(p1, f)
    test_expectvar(p1b, f)

    # set_expect for measure
    test_setexpect(d1)
    test_setexpectvar(d1, expect=0.0) 
    test_setexpectmeanvar(d1, expect=0.0) 

    # set_expect for product_measure
    test_setexpect(p1)
    test_setexpect(p1b)
    test_setexpectvar(p1, expect=0.0)
    test_setexpectvar(p1b)
    test_setexpectmeanvar(p1, expect=0.0)
    test_setexpectmeanvar(p1b)

    # again, but for mixed measures
    test_setexpect(p21)
    test_setexpectvar(p21)
    test_setexpectmeanvar(p21)

    # failure cases (variance w/ one point_mass == 0.0)
    fail(test_setexpectvar, d1) 
    fail(test_setexpectmeanvar, d1)
    fail(test_setexpectvar, p1)
    fail(test_setexpectmeanvar, p1)


# EOF

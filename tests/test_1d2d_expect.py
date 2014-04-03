#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2014 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE

from mystic.math.measures import *
from mystic.math.discrete import *
from mystic.math import almostEqual
def f(x): return sum(x)/len(x)

d2 = measure([point_mass(1.0, 1.0), point_mass(3.0, 2.0)])
d1 = measure([point_mass(3.0, 2.0)])
d2b = measure([point_mass(2.0, 4.0), point_mass(4.0, 2.0)])
p1 = product_measure([d1])
p1b = product_measure([d2])
p2 = product_measure([d2,d2b])
p21 = product_measure([d2,d1])


# expect for product_measure and measure
assert almostEqual(d2.expect(f), 2.3333333333333335)
assert almostEqual(p2.expect(f), 2.5)

# set_expect for measure
d2.set_expect((2.0,0.001), f, bounds=([0.,0.], [10.,10.])) 
assert almostEqual(d2.expect(f), 2.0, tol=0.001)
#print p2.expect(f)

# set_expect for product_measure
p2.set_expect((5.0,0.001), f, bounds=([0.,0.,0.,0.],[10.,10.,10.,10.]))
assert almostEqual(p2.expect(f), 5.0, tol=0.001)


# again, but for single-point measures
# expect for product_measure and measure
assert almostEqual(d1.expect(f), 3.0)
assert almostEqual(p1.expect(f), 3.0)

# set_expect for measure
d1.set_expect((2.0,0.001), f, bounds=([0.], [10.])) 
assert almostEqual(d1.expect(f), 2.0, tol=0.001)
#print p1.expect(f)

# set_expect for product_measure
p1.set_expect((5.0,0.001), f, bounds=([0.],[10.]))
assert almostEqual(p1.expect(f), 5.0, tol=0.001)


# again, but for mixed measures
p21.set_expect((6.0,0.001), f)
assert almostEqual(p21.expect(f), 6.0, tol=0.001)


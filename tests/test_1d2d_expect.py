from mystic.math.measures import *
from mystic.math.dirac_measure import *
from mystic.math import almostEqual
def f(x): return sum(x)/len(x)

d2 = dirac_measure([point(1.0, 1.0), point(3.0, 2.0)])
d1 = dirac_measure([point(3.0, 2.0)])
d2b = dirac_measure([point(2.0, 4.0), point(4.0, 2.0)])
p1 = product_measure([d1])
p1b = product_measure([d2])
p2 = product_measure([d2,d2b])
p21 = product_measure([d2,d1])


# get_expect for product_measure and dirac_measure
assert almostEqual(d2.get_expect(f), 2.3333333333333335)
assert almostEqual(p2.get_expect(f), 2.5)

# set_expect for dirac_measure
d2.set_expect((2.0,0.001), f, bounds=([0.,0.], [10.,10.])) 
assert almostEqual(d2.get_expect(f), 2.0, tol=0.001)
#print p2.get_expect(f)

# set_expect for product_measure
p2.set_expect((5.0,0.001), f, bounds=([0.,0.,0.,0.],[10.,10.,10.,10.]))
assert almostEqual(p2.get_expect(f), 5.0, tol=0.001)


# again, but for single-point measures
# get_expect for product_measure and dirac_measure
assert almostEqual(d1.get_expect(f), 3.0)
assert almostEqual(p1.get_expect(f), 3.0)

# set_expect for dirac_measure
d1.set_expect((2.0,0.001), f, bounds=([0.], [10.])) 
assert almostEqual(d1.get_expect(f), 2.0, tol=0.001)
#print p1.get_expect(f)

# set_expect for product_measure
p1.set_expect((5.0,0.001), f, bounds=([0.],[10.]))
assert almostEqual(p1.get_expect(f), 5.0, tol=0.001)


# again, but for mixed measures
p21.set_expect((6.0,0.001), f)
assert almostEqual(p21.get_expect(f), 6.0, tol=0.001)


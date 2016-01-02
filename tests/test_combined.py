#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# License: 3-clause BSD.  The full license text is available at:
#  - http://trac.mystic.cacr.caltech.edu/project/mystic/browser/mystic/LICENSE
'''
combine several penalty conditions to build a single constraint solver
'''

from mystic.constraints import combined
 

if __name__ == '__main__':
    import numpy as np
    from mystic.penalty import linear_equality, quadratic_equality
    from mystic.constraints import as_constraint

    x = x1,x2,x3 = (5., 5., 1.)
    f = f1,f2,f3 = (np.sum, np.prod, np.average)

    k = 100
    solver = 'fmin_powell' #'diffev'
    ptype = quadratic_equality



    # case #1: couple penalties into a single constraint

#   p = [lambda x: abs(xi - fi(x)) for (xi,fi) in zip(x,f)] #XXX
    p1 = lambda x: abs(x1 - f1(x))
    p2 = lambda x: abs(x2 - f2(x))
    p3 = lambda x: abs(x3 - f3(x))
    p = (p1,p2,p3)
    p = [ptype(pi)(lambda x:0.) for pi in p]
    penalty = combined(*p, k=k)
    constraint = as_constraint(penalty, solver=solver)

    x = [1,2,3,4,5]
    x_ = constraint(x)

#   print "target: %s, %s, %s" % (x1, x2, x3)
#   print "solved: %s, %s, %s" % (f1(x_), f2(x_), f3(x_))
    assert round(f1(x_)) == round(x1)
    assert round(f2(x_)) == round(x2)
    assert round(f3(x_)) == round(x3)



   # case #2: couple constraints into a single constraint

    from mystic.math.measures import impose_product, impose_sum, impose_mean
    from mystic.constraints import as_penalty

    t = t1,t2,t3 = (impose_sum, impose_product, impose_mean)
   #c = [lambda x: ti(xi, x) for (xi,ti) in zip(x,t)] #XXX
    c1 = lambda x: t1(x1, x)
    c2 = lambda x: t2(x2, x)
    c3 = lambda x: t3(x3, x)
    c = (c1,c2,c3)
    
    k=1
    solver = 'buckshot' #'diffev'
    ptype = linear_equality #quadratic_equality

    p = [as_penalty(ci, ptype) for ci in c]
    penalty = combined(*p, k=k)
    constraint = as_constraint(penalty, solver=solver)

    x = [1,2,3,4,5]
    x_ = constraint(x)

#   print "target: %s, %s, %s" % (x1, x2, x3)
#   print "solved: %s, %s, %s" % (f1(x_), f2(x_), f3(x_))
    assert round(f1(x_)) == round(x1)
    assert round(f2(x_)) == round(x2)
    assert round(f3(x_)) == round(x3)



# EOF

#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''
combine several penalty conditions to build a single constraint solver,
then show examples of using (and_, or_, not_) for penalties and constraints
'''

from mystic.coupler import and_, or_, not_
from mystic.constraints import and_ as _and, or_ as _or, not_ as _not
 

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
    penalty = and_(*p, k=k)
    constraint = as_constraint(penalty, solver=solver)

    x = [1,2,3,4,5]
    x_ = constraint(x)

#   print("target: %s, %s, %s" % (x1, x2, x3))
#   print("solved: %s, %s, %s" % (f1(x_), f2(x_), f3(x_)))
    assert round(f1(x_)) == round(x1)
    assert round(f2(x_)) == round(x2)
    assert round(f3(x_)) == round(x3)



   # case #2: couple constraints into a single constraint

    from mystic.math.measures import impose_product, impose_sum, impose_mean
    from mystic.constraints import as_penalty
    from mystic import random_seed
    random_seed(123)

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
    penalty = and_(*p, k=k)
    constraint = as_constraint(penalty, solver=solver)

    x = [1,2,3,4,5]
    x_ = constraint(x)

#   print("target: %s, %s, %s" % (x1, x2, x3))
#   print("solved: %s, %s, %s" % (f1(x_), f2(x_), f3(x_)))
    assert round(f1(x_)) == round(x1)
    assert round(f2(x_)) == round(x2)
    assert round(f3(x_)) == round(x3)



    # etc: more coupling of constraints
    from mystic.constraints import with_mean, discrete

    @with_mean(5.0)
    def meanie(x):
      return x
    
    @discrete(list(range(11)))
    def integers(x):
      return x
    
    c = _and(integers, meanie)
    x = c([1,2,3])
    assert x == integers(x) == meanie(x)
    x = c([9,2,3])
    assert x == integers(x) == meanie(x)
    x = c([0,-2,3])
    assert x == integers(x) == meanie(x)
    x = c([9,-200,344])
    assert x == integers(x) == meanie(x)

    c = _or(meanie, integers)
    x = c([1.1234, 4.23412, -9])
    assert x == meanie(x) and x != integers(x)
    x = c([7.0, 10.0, 0.0])
    assert x == integers(x) and x != meanie(x)
    x = c([6.0, 9.0, 0.0])
    assert x == integers(x) == meanie(x)
    x = c([3,4,5])
    assert x == integers(x) and x != meanie(x)
    x = c([3,4,5.5])
    assert x == meanie(x) and x != integers(x)

    c = _not(integers)
    x = c([1,2,3])
    assert x != integers(x) and x != [1,2,3] and x == c(x)
    x = c([1.1,2,3])
    assert x != integers(x) and x == [1.1,2,3] and x == c(x)
    c = _not(meanie)
    x = c([1,2,3])
    assert x != meanie(x) and x == [1,2,3] and x == c(x)
    x = c([4,5,6])
    assert x != meanie(x) and x != [4,5,6] and x == c(x)
    c = _not(_and(meanie, integers))
    x = c([4,5,6])
    assert x != meanie(x) and x != integers(x) and x != [4,5,6] and x == c(x)


    # etc: more coupling of penalties
    from mystic.penalty import quadratic_inequality

    p1 = lambda x: sum(x) - 5
    p2 = lambda x: min(i**2 for i in x)
    p = p1,p2

    p = [quadratic_inequality(pi)(lambda x:0.) for pi in p]
    p1,p2 = p
    penalty = and_(*p)

    x = [[1,2],[-2,-1],[5,-5]]
    for xi in x:
        assert p1(xi) + p2(xi) == penalty(xi)

    penalty = or_(*p)
    for xi in x:
        assert min(p1(xi),p2(xi)) == penalty(xi)

    penalty = not_(p1)
    for xi in x:
        assert bool(p1(xi)) != bool(penalty(xi))
    penalty = not_(p2)
    for xi in x:
        assert bool(p2(xi)) != bool(penalty(xi))


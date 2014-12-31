# A-R Hedar and M Fukushima, "Derivative-Free Filter Simulated Annealing
# Method for Constrained Continuous Global Optimization", Journal of
# Global Optimization, 35(4), 521-549 (2006).
# 
# code for function PrG2c and PrG2f
# translated from Matlab Code written by A. Hedar (Nov. 23, 2005).
# http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/go.htm

from g02 import objective, bounds, xs, ys
from g02_alt import penalty1, penalty2, quadratic_inequality

from mystic.math.measures import impose_product, impose_sum

def constraint1(x): # impose exactly
    return impose_product(0.75, x)

def constraint2(x): # impose exactly
    return impose_sum(7.5*len(x), x)

def penalty(x): return 0.0
penalty2 = quadratic_inequality(penalty2)(penalty)
penalty1 = quadratic_inequality(penalty1)(penalty)



if __name__ == '__main__':
    bounds = bounds(len(xs))

    from mystic.solvers import diffev2
    from mystic.math import almostEqual

    result = diffev2(objective, x0=bounds, bounds=bounds, constraint=constraint2, penalty=penalty1, npop=40, disp=False, full_output=True)

    assert almostEqual(result[0], xs, rel=1e-2)
    assert almostEqual(result[1], ys, rel=1e-2)



# EOF

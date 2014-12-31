# A-R Hedar and M Fukushima, "Derivative-Free Filter Simulated Annealing
# Method for Constrained Continuous Global Optimization", Journal of
# Global Optimization, 35(4), 521-549 (2006).
# 
# code for function PrG7c and PrG7f
# translated from Matlab Code written by A. Hedar (Nov. 23, 2005).
# http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/go.htm

def objective(x):
    x0,x1,x2,x3,x4,x5,x6,x7,x8,x9 = x
    return x0**2 + x1**2 + x0*x1 - 14*x0 - 16*x1 + (x2-10)**2 + \
           4*(x3-5)**2 + (x4-3)**2 + 2*(x5-1)**2 + 5*x6**2 + \
           7*(x7-11)**2 + 2*(x8-10)**2 + (x9-7)**2 + 45.0

bounds = [(-10,10)]*10
# with penalty='penalty' applied, solution is:
xs = [2.171996, 2.363683, 8.773926, 5.095984, 0.9906548,
      1.430574, 1.321644, 9.828726, 8.280092, 8.375927]
ys = 24.3062091

from mystic.symbolic import generate_constraint, generate_solvers, solve
from mystic.symbolic import generate_penalty, generate_conditions

equations = """
4*x0 + 5*x1 - 3*x6 + 9*x7 - 105.0 <= 0.0
10*x0 - 8*x1 - 17*x6 + 2*x7 <= 0.0
-8*x0 + 2*x1 + 5*x8 - 2*x9 - 12.0 <= 0.0
3*(x0-2)**2 + 4*(x1-3)**2 + 2*x2**2 - 7*x3 - 120.0 <= 0.0
5*x0**2 + 8*x1 + (x2-6)**2 - 2*x3 - 40.0 <= 0.0
0.5*(x0-8)**2 + 2*(x1-4)**2 + 3*x4**2 - x5 - 30.0 <= 0.0
x0**2 + 2*(x1-2)**2 - 2*x0*x1 + 14*x4 - 6*x5 <= 0.0
-3*x0 + 6*x1 + 12*(x8-8)**2 - 7*x9 <= 0.0
"""
#cf = generate_constraint(generate_solvers(solve(equations))) #XXX: inequalities
pf = generate_penalty(generate_conditions(equations))

from mystic.constraints import as_constraint

cf = as_constraint(pf)



if __name__ == '__main__':
    x = [0]*len(xs)

    from mystic.solvers import fmin_powell
    from mystic.math import almostEqual

    result = fmin_powell(objective, x0=x, bounds=bounds, penalty=pf, maxiter=1000, maxfun=100000, ftol=1e-12, xtol=1e-12, gtol=10, disp=False, full_output=True)

    assert almostEqual(result[0], xs, tol=1e-2)
    assert almostEqual(result[1], ys, rel=1e-2)



# EOF

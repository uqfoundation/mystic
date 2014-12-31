# A-R Hedar and M Fukushima, "Derivative-Free Filter Simulated Annealing
# Method for Constrained Continuous Global Optimization", Journal of
# Global Optimization, 35(4), 521-549 (2006).
# 
# code for function PrG11c and PrG11f
# translated from Matlab Code written by A. Hedar (Nov. 23, 2005).
# http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/go.htm

def objective(x):
    x0,x1 = x
    return x0**2 + (x1 - 1)**2

bounds = [(-1,1)]*2
# with penalty='penalty' applied, solution is:
xs, xs_ = [(0.5)**.5, 0.5], [-(0.5)**.5, 0.5]
ys = 0.75

from mystic.symbolic import generate_constraint, generate_solvers, solve
from mystic.symbolic import generate_penalty, generate_conditions

equations = """
x1 - x0**2 = 0.0
"""
cf = generate_constraint(generate_solvers(solve(equations, target='x1')))
pf = generate_penalty(generate_conditions(equations), k=1e12)



if __name__ == '__main__':

    from mystic.solvers import diffev2
    from mystic.math import almostEqual

    result = diffev2(objective, x0=bounds, bounds=bounds, constraints=cf, npop=40, disp=False, full_output=True)

    assert almostEqual(result[0], xs, tol=1e-2) \
        or almostEqual(result[0], xs_, tol=1e-2)
    assert almostEqual(result[1], ys, rel=1e-2)



# EOF

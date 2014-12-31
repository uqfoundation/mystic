# A-R Hedar and M Fukushima, "Derivative-Free Filter Simulated Annealing
# Method for Constrained Continuous Global Optimization", Journal of
# Global Optimization, 35(4), 521-549 (2006).
# 
# code for function PrG2c and PrG2f
# translated from Matlab Code written by A. Hedar (Nov. 23, 2005).
# http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/go.htm

def objective(x):
    from numpy import abs, sum, cos, product, sqrt
    sum_jx = 0.0
    for j in range(len(x)): sum_jx = sum_jx + (j+1) * x[j]**2
    return -abs((sum(cos(x)**4) - 2*product(cos(x)**2))/sqrt(sum_jx))

def bounds(len=3):
    return [(0.0,10.0)]*len

# with penalty='penalty' applied, solution is:
xs = [ 3.04295933, 1.48286963, 0.16618875]
ys = -0.51578987

"""
for len(x) = 10,
xs ~ [3.12388714, 3.06913834, 3.01426760, 2.95755412, 1.46603517,
      0.36802963, 0.36346912, 0.35912472, 0.35493945, 0.35095372]
ys ~ -0.74732020
"""

from mystic.symbolic import generate_constraint, generate_solvers, solve
from mystic.symbolic import generate_penalty, generate_conditions

equations = """
-prod([x0, x1, x2]) + 0.75 <= 0.0
sum([x0, x1, x2]) - 7.5*3 <= 0.0
"""
#cf = generate_constraint(generate_solvers(solve(equations))) #XXX: inequalities
pf = generate_penalty(generate_conditions(equations))

from mystic.constraints import as_constraint

cf = as_constraint(pf)



if __name__ == '__main__':
    bounds = bounds(len(xs))

    from mystic.solvers import diffev2
    from mystic.math import almostEqual

    result = diffev2(objective, x0=bounds, bounds=bounds, penalty=pf, npop=40, disp=False, full_output=True)

    assert almostEqual(result[0], xs, rel=1e-2)
    assert almostEqual(result[1], ys, rel=1e-2)



# EOF

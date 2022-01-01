#!/usr/bin/env python
#
# Problem definition:
# A-R Hedar and M Fukushima, "Derivative-Free Filter Simulated Annealing
# Method for Constrained Continuous Global Optimization", Journal of
# Global Optimization, 35(4), 521-549 (2006).
# 
# Original Matlab code written by A. Hedar (Nov. 23, 2005)
# http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/go.htm
# and ported to Python by Mike McKerns (December 2014)
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"Welded Beam Design"

def objective(x):
    x0,x1,x2,x3 = x
    return 1.10471*x0**2*x1 + 0.04811*x2*x3*(14.0 + x1)

bounds = [(0.125,10)] + [(0.1,10)]*3
# with penalty='penalty' applied, solution is:
xs = [ 0.20572964, 7.09241428, 9.03662391, 0.20572964]
ys = 2.21815086

# default parameters
P = 6000.
L = 14.
E = 30.e+6
G = 12.e+6
t_max = 13600.
s_max = 30000.
d_max = 0.25

# parameter equations
def M(x, P=P, L=L): return P*(L + x[1]/2.)
def R(x): return (0.25*(x[1]**2 + (x[0] + x[2])**2))**.5
def J(x): return 2./2**.5 * x[0] * x[1] * (x[1]**2/12. + 0.25*(x[0] + x[2])**2)
def P_c(x, L=L, E=E, G=G): return (4.013*E/(6*L**2)) * x[2]*x[3]**3 * (1-0.25*x[2]*(E/G)**.5/L)
def t1(x, P=P): return P/(2**.5 * x[0] * x[1])
def t2(x, P=P, L=L): return M(x,P,L)*R(x)/J(x)
def t(x, P=P, L=L): return (t1(x,P)**2 + t1(x,P)*t2(x,P,L)*x[1]/R(x) + t2(x,P,L)**2)**.5
def s(x, P=P, L=L): return 6*P*L/(x[3] * x[2]**2)
def d(x, P=P, L=L, E=E): return 4*P*L**3/(E * x[3] * x[2]**3)

from mystic.penalty import quadratic_inequality

def generate_penalty(**kwds):
    "enable override of P,L,E,G,t_max,s_max,d_max in penalties"

    def penalty1(x, P=P, L=L, t_max=t_max, **kwd): # <= 0.0
        return t(x,P,L) - t_max

    def penalty2(x, P=P, L=L, s_max=s_max, **kwd): # <= 0.0
        return s(x,P,L) - s_max

    def penalty3(x, **kwd): # <= 0.0
        return x[0] - x[3]

    def penalty4(x, **kwd): # <= 0.0
        return 0.10471*x[0]**2 + 0.04811*x[2]*x[3]*(14.0 + x[1]) - 5.0

    def penalty5(x, P=P, L=L, E=E, d_max=d_max, **kwd): # <= 0.0
        return d(x,P,L,E) - d_max

    def penalty6(x, P=P, L=L, E=E, G=G, **kwd): # <= 0.0
        return P - P_c(x,L,E,G)

    @quadratic_inequality(penalty1, k=1e12, kwds=kwds)
    @quadratic_inequality(penalty2, k=1e12, kwds=kwds)
    @quadratic_inequality(penalty3, k=1e12, kwds=kwds)
    @quadratic_inequality(penalty4, k=1e12, kwds=kwds)
    @quadratic_inequality(penalty5, k=1e12, kwds=kwds)
    @quadratic_inequality(penalty6, k=1e12, kwds=kwds)
    def penalty(x):
        return 0.0

    return penalty



if __name__ == '__main__':

    parameters = {
      'P': 6000.,
      'L': 14.,
      'E': 30.e+6,
      'G': 12.e+6,
      't_max': 13600.,
      's_max': 30000.,
      'd_max': 0.25,
    }

    from mystic.solvers import diffev2
    from mystic.math import almostEqual

    penalty = generate_penalty(**parameters)

    result = diffev2(objective, x0=bounds, bounds=bounds, penalty=penalty, npop=40, gtol=500, disp=False, full_output=True)

    assert almostEqual(result[0], xs, rel=1e-2)
    assert almostEqual(result[1], ys, rel=1e-2)



# EOF

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

from g01 import objective, bounds, xs, ys
from g01_alt import penalty1, penalty2, penalty3, penalty4, penalty5, \
                    penalty6, penalty7, penalty8, penalty9

from mystic.constraints import as_constraint
from mystic.penalty import linear_inequality
from mystic.coupler import and_ as combined

penalties = (penalty1,penalty2,penalty3,penalty4,penalty5,\
             penalty6,penalty7,penalty8,penalty9)

penalty = combined(*[linear_inequality(pi)(lambda x:0.) for pi in penalties])
solver = as_constraint(penalty)



if __name__ == '__main__':
    x = [0]*len(xs)

    from mystic.solvers import fmin_powell, diffev
    from mystic.math import almostEqual

    result = fmin_powell(objective, x0=x, bounds=bounds, penalty=penalty, disp=False, full_output=True)

    assert almostEqual(result[0], xs, tol=1e-2)
    assert almostEqual(result[1], ys, tol=1e-2)



# EOF

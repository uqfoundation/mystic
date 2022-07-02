#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2021-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

import numpy as np
import mystic as my
import mystic.symbolic as ms
import mystic.constraints as mc
import mystic.models as mm
from mystic.solvers import PowellDirectionalSolver, NelderMeadSimplexSolver, \
                           DifferentialEvolutionSolver, \
                           DifferentialEvolutionSolver2, \
                           LatticeSolver, BuckshotSolver, SparsitySolver

almostEqual = my.math.almostEqual
rosen = mm.rosen
bounds = [(0.0001, 0.0002)]*3
guess = [0.001,0.001,0.001]

# generate the constraints
_eqn = 'x0 + x1 + x2 <= .0005'
_cons = ms.generate_constraint(ms.generate_solvers(ms.simplify(_eqn)), join=mc.and_)

# explicitly include bounds in constraints
eqn_ = '\n'.join([_eqn, ms.symbolic_bounds(*zip(*bounds))])
cons_ = ms.generate_constraint(ms.generate_solvers(ms.simplify(eqn_)), join=mc.and_)


def test_constrained(TheSolver, tight, clip):
    ndim = len(bounds)
    solver = TheSolver(ndim)
    if TheSolver in [PowellDirectionalSolver, NelderMeadSimplexSolver]:
        solver.SetInitialPoints(guess)
    elif TheSolver in [DifferentialEvolutionSolver, DifferentialEvolutionSolver2]:
        solver.SetRandomInitialPoints(*zip(*bounds))
    else: pass
    solver.SetStrictRanges(*zip(*bounds), tight=tight, clip=clip)
    solver.SetConstraints(_cons if tight else cons_)
    solver.Solve(rosen)
    fx = solver.bestEnergy
    xk = solver.Solution()
    x0,x1,x2 = xk
    #print('{T:%s, C:%s} %s @ %s' % (tight, clip, fx, xk))
    assert almostEqual(fx, 1.999205, tol=5e-4) # minimum energy is found
    assert eval(_eqn) # constraint is satisfied
    assert (not np.any(xk - np.clip(xk, *zip(*bounds)))) # bounds are satisfied


test_constrained(NelderMeadSimplexSolver, tight=True, clip=None) #ftol=1e-8?
test_constrained(NelderMeadSimplexSolver, tight=True, clip=True)
test_constrained(NelderMeadSimplexSolver, tight=None, clip=None)
test_constrained(PowellDirectionalSolver, tight=True, clip=None) #gtol=?
test_constrained(PowellDirectionalSolver, tight=True, clip=True)
test_constrained(PowellDirectionalSolver, tight=None, clip=None)
test_constrained(DifferentialEvolutionSolver, tight=True, clip=None) #npop=10?
test_constrained(DifferentialEvolutionSolver, tight=True, clip=True)
test_constrained(DifferentialEvolutionSolver, tight=None, clip=None)
test_constrained(DifferentialEvolutionSolver2, tight=True, clip=None) #npop=10?
test_constrained(DifferentialEvolutionSolver2, tight=True, clip=True)
test_constrained(DifferentialEvolutionSolver2, tight=None, clip=None)
#test_constrained(LatticeSolver, tight=True, clip=None) #nbin=8?
#test_constrained(LatticeSolver, tight=True, clip=True)
#test_constrained(LatticeSolver, tight=None, clip=None)
#test_constrained(BuckshotSolver, tight=True, clip=None) #npts=8?
#test_constrained(BuckshotSolver, tight=True, clip=True)
#test_constrained(BuckshotSolver, tight=None, clip=None)
#test_constrained(SparsitySolver, tight=True, clip=None) #npts=8?
#test_constrained(SparsitySolver, tight=True, clip=True)
#test_constrained(SparsitySolver, tight=None, clip=None)

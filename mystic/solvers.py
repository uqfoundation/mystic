#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
solvers: minimal and expanded interfaces for optimization algorithms


Standard Interface
==================

All of mystic's optimizers derive from the solver API, which provides
each optimizer with a standard, but highly-customizable interface.
A description of the solver API is found in ``mystic.models.abstract_model``,
and in each derived optimizer.  Mystic's optimizers are::

    ** Global Optimizers **
    DifferentialEvolutionSolver  -- Differential Evolution algorithm
    DifferentialEvolutionSolver2 -- Price & Storn's Differential Evolution
    ** Pseudo-Global Optimizers **
    SparsitySolver               -- N Solvers sampled where point desity is low
    BuckshotSolver               -- Uniform Random Distribution of N Solvers
    LatticeSolver                -- Distribution of N Solvers on a Regular Grid
    ** Local-Search Optimizers **
    NelderMeadSimplexSolver      -- Nelder-Mead Simplex algorithm
    PowellDirectionalSolver      -- Powell's (modified) Level Set algorithm


Minimal Interface
=================

Most of mystic's optimizers can be called from a minimal (i.e. one-line)
interface. The collection of arguments is often unique to the optimizer,
and if the underlying solver derives from a third-party package, the
original interface is reproduced. Minimal interfaces to these optimizers
are provided::

    ** Global Optimizers **
    diffev      -- DifferentialEvolutionSolver
    diffev2     -- DifferentialEvolutionSolver2
    ** Pseudo-Global Optimizers **
    sparsity    -- SparsitySolver
    buckshot    -- BuckshotSolver
    lattice     -- LatticeSolver
    ** Local-Search Optimizers **
    fmin        -- NelderMeadSimplexSolver
    fmin_powell -- PowellDirectionalSolver


More Information
================

For more information, please see the solver documentation found here:
    - ``mystic.differential_evolution``   [differential evolution solvers]
    - ``mystic.scipy_optimize``           [scipy local-search solvers]
    - ``mystic.ensemble``                 [pseudo-global solvers]

or the API documentation found here:
    - ``mystic.abstract_solver``          [the solver API definition]
    - ``mystic.abstract_map_solver``      [the parallel solver API]
    - ``mystic.abstract_ensemble_solver`` [the ensemble solver API]
"""
# global optimizers
from mystic.differential_evolution import DifferentialEvolutionSolver
from mystic.differential_evolution import DifferentialEvolutionSolver2
from mystic.differential_evolution import diffev, diffev2

# pseudo-global optimizers
from mystic.ensemble import SparsitySolver
from mystic.ensemble import BuckshotSolver
from mystic.ensemble import LatticeSolver
from mystic.ensemble import sparsity, buckshot, lattice

# local-search optimizers
from mystic.scipy_optimize import NelderMeadSimplexSolver
from mystic.scipy_optimize import PowellDirectionalSolver
from mystic.scipy_optimize import fmin, fmin_powell


# load a solver from a restart file
def LoadSolver(filename=None, **kwds):
    """load solver state from a restart file"""
    if filename is None:
        filename = kwds['_state'] if '_state' in kwds else None
    #XXX: only allow a list override keys (lookup values from self)
#   if filename is None: filename = self._state
#   if filename is None:
#       solver = self
#   else:
    import dill
    if filename: f = open(filename, 'rb')
    else: return
    try:
        solver = dill.load(f)
        _locals = {}
        _locals['solver'] = solver
        code = "from mystic.solvers import %s;" % solver._type
        code += "self = %s(solver.nDim);" % solver._type
        code = compile(code, '<string>', 'exec')
        exec(code, _locals)
        self = _locals['self']
    finally:
        f.close()
    # transfer state from solver to self, allowing overrides
    self._AbstractSolver__load_state(solver, **kwds)
    self._state = filename
    self._stepmon.info('LOADED("%s")' % filename)
    return self


# end of file

#!/usr/bin/env python
"""
solvers: minimal and expanded interfaces for optimization algorithms


Standard Interface
==================

All of mystic's optimizers derive from the solver API, which provides
each optimizer with a standard, but highly-customizable interface.
A description of the solver API is found in `mystic.models.abstract_model`,
and in each derived optimizer.  Mystic's optimizers are::
    == Global Optimizers ==
    DifferentialEvolutionSolver  -- Differential Evolution algorithm
    DifferentialEvolutionSolver2 -- Price & Storn's Differential Evolution
    == Pseudo-Global Optimizers ==
    BuckshotSolver               -- Uniform Random Distribution of N Solvers
    LatticeSolver                -- Distribution of N Solvers on a Regular Grid
    == Local-Search Optimizers ==
    NelderMeadSimplexSolver      -- Nelder-Mead Simplex algorithm
    PowellDirectionalSolver      -- Powell's (modified) Level Set algorithm


Minimal Interface
=================

Most of mystic's optimizers can be called from a minimal (i.e. one-line)
interface. The collection of arguments is often unique to the optimizer,
and if the underlying solver derives from a third-party package, the
original interface is reproduced. Minimal interfaces to these optimizers
are provided::
    == Global Optimizers ==
    diffev      -- DifferentialEvolutionSolver
    diffev2     -- DifferentialEvolutionSolver2
    == Local-Search Optimizers ==
    fmin        -- NelderMeadSimplexSolver
    fmin_powell -- PowellDirectionalSolver


More Information
================

For more information, please see the solver documentation found here::
    - mystic.mystic.differential_evolution [differential evolution solvers]
    - mystic.mystic.scipy_optimize         [scipy local-search solvers]
    - mystic.mystic.nested                 [pseudo-global solvers]

or the API documentation found here::
    - mystic.mystic.abstract_solver        [the solver API definition]
    - mystic.mystic.abstract_map_solver    [the parallel solver API]
    - mystic.mystic.abstract_nested_solver [the nested solver API]


"""
# global optimizers
from differential_evolution import DifferentialEvolutionSolver
from differential_evolution import DifferentialEvolutionSolver2
from differential_evolution import diffev, diffev2

# pseudo-global optimizers
from nested import BuckshotSolver
from nested import LatticeSolver

# local-search optimizers
from scipy_optimize import NelderMeadSimplexSolver
from scipy_optimize import PowellDirectionalSolver
from scipy_optimize import fmin, fmin_powell


# end of file

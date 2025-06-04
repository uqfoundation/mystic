#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2019-2025 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
samplers: optimizer-guided directed sampling
"""
__all__  = ['LatticeSampler','BuckshotSampler','SparsitySampler',\
            'MisfitSampler','MixedSampler']

from mystic.abstract_sampler import AbstractSampler


class LatticeSampler(AbstractSampler):
    """
optimizer-directed sampling starting at the centers of N grid points
    """
    def _init_solver(self, **kwd):
        """initialize the ensemble solver"""
        from mystic.ensemble import LatticeSolver
        s = LatticeSolver(len(self._bounds), nbins=self._npts)
        s.SetStrictRanges(*zip(*self._bounds), **kwd)
        s.SetObjective(self._model)
        return s


class BuckshotSampler(AbstractSampler):
    """
optimizer-directed sampling starting at N randomly sampled points
    """
    def _init_solver(self, **kwd):
        """initialize the ensemble solver"""
        from mystic.ensemble import BuckshotSolver
        s = BuckshotSolver(len(self._bounds), npts=self._npts)
        s.SetStrictRanges(*zip(*self._bounds), **kwd)
        s.SetObjective(self._model)
        return s


class SparsitySampler(AbstractSampler):
    """
optimizer-directed sampling starting at N points sampled in sparse reigons
    """
    def __init__(self, bounds, model, npts=None, **kwds):
        self._rtol = kwds.pop('rtol', None)
        super(SparsitySampler, self).__init__(bounds, model, npts, **kwds)
        return
    __init__.__doc__ = AbstractSampler.__init__.__doc__
    def _init_solver(self, **kwd):
        """initialize the ensemble solver"""
        from mystic.ensemble import SparsitySolver
        s = SparsitySolver(len(self._bounds), npts=self._npts, rtol=self._rtol)
        s.SetStrictRanges(*zip(*self._bounds), **kwd)
        s.SetObjective(self._model)
        return s


class MisfitSampler(AbstractSampler):
    """
optimizer-directed sampling starting at N points sampled near largest misfit
    """
    def __init__(self, bounds, model, npts=None, **kwds):
        self._mtol = kwds.pop('mtol', None)
        super(MisfitSampler, self).__init__(bounds, model, npts, **kwds)
        return
    __init__.__doc__ = AbstractSampler.__init__.__doc__
    def _init_solver(self, **kwd):
        """initialize the ensemble solver"""
        from mystic.ensemble import MisfitSolver
        s = MisfitSolver(len(self._bounds), npts=self._npts, mtol=self._mtol, func=self._model) #FIXME: TEST (mon?)
        s.SetStrictRanges(*zip(*self._bounds), **kwd)
        s.SetObjective(self._model)
        return s


class MixedSampler(AbstractSampler):
    """
optimizer-directed sampling using N points from a mixture of ensemble solvers
    """
    def _init_solver(self, **kwd):
        """initialize the ensemble solver"""
        from mystic.ensemble import MixedSolver
        s = MixedSolver(len(self._bounds), samp=self._npts)
        s.SetStrictRanges(*zip(*self._bounds), **kwd)
        s.SetObjective(self._model)
        return s


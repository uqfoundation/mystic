#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2019-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
samplers: optimizer-guided directed sampling
"""
__all__  = ['LatticeSampler','BuckshotSampler','SparsitySampler']

from mystic.abstract_sampler import AbstractSampler


class LatticeSampler(AbstractSampler):
    """
optimizer-directed sampling starting at the centers of N grid points
    """
    def _init_solver(self):
        """initialize the ensemble solver"""
        from mystic.ensemble import LatticeSolver
        return LatticeSolver(len(self._bounds), nbins=self._npts)


class BuckshotSampler(AbstractSampler):
    """
optimizer-directed sampling starting at N randomly sampled points
    """
    def _init_solver(self):
        """initialize the ensemble solver"""
        from mystic.ensemble import BuckshotSolver
        return BuckshotSolver(len(self._bounds), npts=self._npts)


class SparsitySampler(AbstractSampler):
    """
optimizer-directed sampling starting at N points sampled in sparse reigons
    """
    def __init__(self, bounds, model, npts=None, **kwds):
        self._rtol = kwds.pop('rtol', None)
        super(SparsitySampler, self).__init__(bounds, model, npts, **kwds)
        return
    __init__.__doc__ = AbstractSampler.__init__.__doc__
    def _init_solver(self):
        """initialize the ensemble solver"""
        from mystic.ensemble import SparsitySolver
        return SparsitySolver(len(self._bounds), npts=self._npts, rtol=self._rtol)


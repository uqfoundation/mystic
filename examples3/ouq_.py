#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2024 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
OUQ classes for calculating bounds on statistical quantities
"""
from mystic.math.discrete import product_measure
from ouq import BaseOUQ


class MeanValue(BaseOUQ):
    def objective(self, rv, axis=None, idx=0): #FIXME: idx != None, is fixed
        """calculate mean value of input, under uncertainty

    Input:
        rv: list of input parameters
        axis: int, the index of output to calculate (all, by default)
        idx: int, the index of input to calculate (all, by default) #FIXME

    Returns:
        the mean value for the specified index

    NOTE:
        respects constraints on input parameters and product measure

    NOTE:
        for product_measure, use sampled_expect if samples, else expect
        """
        # check constraints
        c = product_measure().load(rv, self.npts)
        if not self.cvalid(c) or not self.xvalid(rv): #FIXME: set model,samples in constraints
            return self._invalid
        # get mean value
        if axis is None and self.axes is not None:
            model = (lambda x: self.model(x, axis=i) for i in range(self.axes))
            if self.samples is None:
                return NotImplemented #FIXME: set model,samples in constraints
            # else use sampled support
            return NotImplemented #FIXME: set model,samples in constraints
        # else, get mean value for the given axis
        if axis is None:
            model = lambda x: self.model(x)
        else:
            model = lambda x: self.model(x, axis=axis)
        if self.samples is None:
            if idx is None:
                res = (c[i].mean for i in self.nx)
                return tuple(res)
            return c[idx].mean #FIXME: set model,samples in constraints
        return NotImplemented #FIXME: set model,samples in constraints


class Negativity(BaseOUQ):

    def objective(self, rv, axis=None, iter=True):
        """calculate expected negativity for model, under uncertainty

    Input:
        rv: list of input parameters
        axis: int, the index of output to calculate (all, by default)
        iter: bool, if True, calculate per axis, else calculate for all axes

    Returns:
        the expected negativity for the specified axis (or all axes)

    NOTE:
        respects constraints on input parameters and product measure

    NOTE:
        for product_measure, use sampled_pof(func) if samples, else pof(func)
        where func = lambda x: model(x) >= 0
        """
        # check constraints
        c = product_measure().load(rv, self.npts)
        if not self.cvalid(c) or not self.xvalid(rv):
            if axis is None and self.axes is not None:
                return (self._invalid,) * (self.axes or 1) #XXX:?
            return self._invalid
        # get expected negativity
        if iter and axis is None and self.axes is not None:
            model = (lambda x: (self.model(x,axis=i) >= 0) for i in range(self.axes))
            if self.samples is None:
                return tuple(c.pof(m) for m in model)
            # else use sampled support
            return tuple(c.sampled_pof(m, self.samples, map=self.map) for m in model)
        # else, get expected negativity for the given axis
        if axis is None:
            model = lambda x: (self.model(x) >= 0)
        else:
            model = lambda x: (self.model(x, axis=axis) >= 0)
        if self.samples is None:
            return c.pof(model)
        return c.sampled_pof(model, self.samples, map=self.map)


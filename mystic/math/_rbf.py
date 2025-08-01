#!/usr/bin/env python
# rbf.py module by John Travers, Robert Hetland, and Travis Oliphant
#
# Forked by: Mike McKerns (October 2018)
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2018-2025 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""rbf - Radial basis functions for interpolation/smoothing scattered Nd data.
"""
#
# original documentation/license (below)
"""
Written by John Travers <jtravs@gmail.com>, February 2007
Based closely on Matlab code by Alex Chirokov
Additional, large, improvements by Robert Hetland
Some additional alterations by Travis Oliphant
Further additions and adjustments by Mike McKerns @uqfoundation.org

Permission to use, modify, and distribute this software is given under the
terms of the SciPy (BSD style) license.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.

Copyright (c) 2006-2007, Robert Hetland <hetland@tamu.edu>
Copyright (c) 2007, John Travers <jtravs@gmail.com>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of Robert Hetland nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np

get_function_code = lambda x: x.__code__
get_method_function = lambda x: x.__func__


__all__ = ['Rbf']

#@np.vectorize
def xlogy(x, y):
    settings = np.seterr(all='ignore')
    y = x*np.log(y) #XXX: throws RuntimeWarning
    np.seterr(**settings)
    y[x == 0] = 0
    return y
    #return 0 if (x == 0 and not np.isnan(y)) else x * np.log(y)


class Rbf(object):
    """
    Rbf(*args)

    A class for radial basis function approximation/interpolation of
    n-dimensional scattered data.

    Parameters
    ----------
    *args : arrays
        x, y, z, ..., d, where x, y, z, ... are the coordinates of the nodes
        and d is the array of values at the nodes
    function : str or callable, optional
        The radial basis function, based on the radius, r, given by the norm
        (default is Euclidean distance); the default is 'multiquadric'::

            'multiquadric': sqrt((r/epsilon)**2 + 1)
            'inverse_multiquadric': 1.0/sqrt((r/epsilon)**2 + 1)
            'inverse_quadratic': 1.0/((r/epsilon)**2 + 1)
            'gaussian': exp(-(r/epsilon)**2)
            'hyperbolic_tangent': r * tanh(r/epsilon)
            'bump': exp(-1.0/(1 - (r/epsilon)**2)) if r < epsilon else 0.0
            'linear': r
            'cubic': r**3
            'quintic': r**5
            'thin_plate': r**2 * log(r)
            'quartic': r**4 * log(r)

        If callable, then it must take 2 arguments (self, r).  The epsilon
        parameter will be available as self.epsilon.  Other keyword
        arguments passed in will be available as well.

    epsilon : float, optional
        Adjustable shape parameter for infinitely smooth functions
        - defaults to approximate average distance between nodes (which is
        a good start).
    smooth : float, optional
        Values greater than zero increase the smoothness of the
        approximation.  0 is for interpolation (default), the function will
        always go through the nodal points in this case.
    norm : callable, optional
        A function that returns the 'distance' between two points, with
        inputs as arrays of positions (x, y, z, ...), and an output as an
        array of distance.  E.g, the default::

            def euclidean_norm(x1, x2):
                return sqrt( ((x1 - x2)**2).sum(axis=0) )

        which is called with ``x1 = x1[ndims, newaxis, :]`` and
        ``x2 = x2[ndims, : ,newaxis]`` such that the result is a matrix of the
        distances from each point in ``x1`` to each point in ``x2``.

    Examples
    --------
    >>> from rbf import Rbf
    >>> x, y, z, d = np.random.rand(4, 50)
    >>> rbfi = Rbf(x, y, z, d)  # radial basis function interpolator instance
    >>> xi = yi = zi = np.linspace(0, 1, 20)
    >>> di = rbfi(xi, yi, zi)   # interpolated values
    >>> di.shape
    (20,)

    """

    def _euclidean_norm(self, x1, x2):
        return np.sqrt(np.square(x1 - x2).sum(axis=0))

    def _h_multiquadric(self, r):
        return np.sqrt(np.square(1.0/self.epsilon*r) + 1)

    def _h_inverse_multiquadric(self, r):
        return 1.0/np.sqrt(np.square(1.0/self.epsilon*r) + 1)

    def _h_inverse_quadratic(self, r):
        return 1.0/(np.square(1.0/self.epsilon*r) + 1)

    def _h_hyperbolic_tangent(self, r):
        return r*np.tanh(1.0/self.epsilon*r)

    def _h_gaussian(self, r):
        return np.exp(-np.square(1.0/self.epsilon*r))

    def _h_linear(self, r):
        return r

    def _h_cubic(self, r):
        return np.power(r, 3)

    def _h_quintic(self, r):
        return np.power(r, 5)

    def _h_thin_plate(self, r):
        return xlogy(np.square(r), r)

    def _h_quartic(self, r):
        return xlogy(np.power(r, 4), r)

    def _h_bump(self, r):
        if r < self.epsilon:
            return np.exp(-1.0/(1 - np.square(1.0/self.epsilon*r)))
        return 0.0

    # Setup self._function and do smoke test on initial r
    def _init_function(self, r):
        if isinstance(self.function, str):
            self.function = self.function.lower()
            _mapped = {'inverse': 'inverse_multiquadric',
                       'inverse multiquadric': 'inverse_multiquadric',
                       'inverse quadratic': 'inverse_quadratic',
                       'hyperbolic tangent': 'hyperbolic_tangent',
                       'hyperbolic': 'hyperbolic_tangent',
                       'tanh': 'hyperbolic_tangent',
                       'thin plate': 'thin_plate',
                       'thin-plate': 'thin_plate'}
            if self.function in _mapped:
                self.function = _mapped[self.function]

            func_name = "_h_" + self.function
            if hasattr(self, func_name):
                self._function = getattr(self, func_name)
            else:
                functionlist = [x[3:] for x in dir(self) if x.startswith('_h_')]
                raise ValueError("function must be a callable or one of " +
                                     ", ".join(functionlist))
            self._function = getattr(self, "_h_"+self.function)
        elif callable(self.function):
            allow_one = False
            if hasattr(self.function, '__code__'):
                val = self.function
                allow_one = True
            elif hasattr(self.function, "__func__"):
                val = get_method_function(self.function)
            elif hasattr(self.function, "__call__"):
                val = get_method_function(self.function.__call__)
            else:
                raise ValueError("Cannot determine number of arguments to function")

            argcount = get_function_code(val).co_argcount
            if allow_one and argcount == 1:
                self._function = self.function
            elif argcount == 2:
                self._function = self.function.__get__(self, Rbf)
            else:
                raise ValueError("Function argument must take 1 or 2 arguments.")

        a0 = self._function(r)
        if a0.shape != r.shape:
            raise ValueError("Callable must take array and return array of the same shape")
        return a0

    def __init__(self, *args, **kwargs):
        self.xi = np.asarray([np.asarray(a, dtype=np.float64).flatten()
                           for a in args[:-1]])
        self.N = self.xi.shape[-1]
        self.di = np.asarray(args[-1]).flatten()

        if not all([x.size == self.di.size for x in self.xi]):
            raise ValueError("All arrays must be equal length.")

        self.norm = kwargs.pop('norm', self._euclidean_norm)
        self.epsilon = kwargs.pop('epsilon', None)
        if self.epsilon is None:
            # default epsilon is the "the average distance between nodes" based
            # on a bounding hypercube
            dim = self.xi.shape[0]
            ximax = np.amax(self.xi, axis=1)
            ximin = np.amin(self.xi, axis=1)
            edges = ximax-ximin
            edges = edges[np.nonzero(edges)]
            power = 1.0/edges.size if edges.size else np.inf
            self.epsilon = np.power(np.prod(edges)/self.N, power)
        self.smooth = kwargs.pop('smooth', 0.0)

        self.function = kwargs.pop('function', 'multiquadric')

        # attach anything left in kwargs to self
        #  for use by any user-callable function or
        #  to save on the object returned.
        for item, value in kwargs.items():
            setattr(self, item, value)

        # if self.A.size == 1 and self.A.flat[0] == 0. and smooth == 0.
        #  produces singular matrix if function returns 0 at r == 0
        self.nodes = np.linalg.solve(self.A, self.di)

    @property
    def A(self):
        # this only exists for backwards compatibility: self.A was available
        # and, at least technically, public.
        r = self._call_norm(self.xi, self.xi)
        return self._init_function(r) - np.eye(self.N)*self.smooth

    def _call_norm(self, x1, x2):
        if len(x1.shape) == 1:
            x1 = x1[np.newaxis, :]
        if len(x2.shape) == 1:
            x2 = x2[np.newaxis, :]
        x1 = x1[..., :, np.newaxis]
        x2 = x2[..., np.newaxis, :]
        return self.norm(x1, x2)

    def __call__(self, *args):
        args = [np.asarray(x) for x in args]
        if not all([x.shape == y.shape for x in args for y in args]):
            raise ValueError("Array lengths must be equal")
        shp = args[0].shape
        xa = np.asarray([a.flatten() for a in args], dtype=np.float64)
        r = self._call_norm(xa, self.xi)
        return np.dot(self._function(r), self.nodes).reshape(shp)

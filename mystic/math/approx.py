#!/usr/bin/env python
#
# almostEqual is a repackaging of numpy.allclose, but at different 'tol'
# approx_equal is similar to almostEqual, and can be treated as deprecated
#
# Forked by: Mike McKerns (May 2010)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2010-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
tools for measuring equality
"""

def _float_approx_equal(x, y, tol=1e-18, rel=1e-7):
    if tol is rel is None:
        raise TypeError('cannot specify both absolute and relative errors are None')
    tests = []
    if tol is not None: tests.append(tol)
    if rel is not None: tests.append(rel*abs(x))
    assert tests
    return abs(x - y) <= max(tests)


def approx_equal(x, y, *args, **kwargs):
    """Return True if x and y are approximately equal, otherwise False.

Args:
    x (object): first object to compare
    y (object): second object to compare
    tol (float, default=1e-18): absolute error
    rel (float, default=1e-7): relative error

Returns:
    True if x and y are equal within tolerance, otherwise returns False.

Notes:
    If x and y are floats, return True if y is within either absolute error
    tol or relative error rel of x. You can disable either the absolute or
    relative check by passing ``None`` as *tol* or *rel* (but not both).

    For any other objects, x and y are checked in that order for a method
    ``__approx_equal__``, and the result of that is returned as a bool. Any
    optional arguments are passed to the ``__approx_equal__`` method.

    ``__approx_equal__`` can return ``NotImplemented`` to signal it doesn't
    know how to perform the specific comparison, in which case the other
    object is checked instead. If neither object has the method, or both defer
    by returning ``NotImplemented``, then fall back on the same numeric
    comparison that is used for floats.

Examples:
    >>> approx_equal(1.2345678, 1.2345677)
    True
    >>> approx_equal(1.234, 1.235)
    False
    """
    if not (type(x) is type(y) is float):
        # Skip checking for __approx_equal__ in the common case of two floats.
        methodname = '__approx_equal__'
        # Allow the objects to specify what they consider "approximately equal",
        # giving precedence to x. If either object has the appropriate method, we
        # pass on any optional arguments untouched.
        for a,b in ((x, y), (y, x)):
            try:
                method = getattr(a, methodname)
            except AttributeError:
                continue
            else:
                result = method(b, *args, **kwargs)
                if result is NotImplemented:
                    continue
                return bool(result)
    # If we get here without returning, then neither x nor y knows how to do an
    # approximate equal comparison (or are both floats). Fall back to a numeric
    # comparison.
    return _float_approx_equal(x, y, *args, **kwargs)


def almostEqual(x, y, tol=1e-18, rel=1e-7):
    """
    Returns True if two arrays are element-wise equal within a tolerance.
    
    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.
    
    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    
    Returns
    -------
    y : bool
        Returns True if the two arrays are equal within the given
        tolerance; False otherwise. If either array contains NaN, then
        False is returned.

    Notes
    -----
    If the following equation is element-wise True, then almostEqual returns
    True.
    
     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))
    
    Examples
    --------
    >>> almostEqual([1e10,1.2345678], [1e10,1.2345677])
    True
    >>> almostEqual([1e10,1.234], [1e10,1.235])
    False
"""
    from numpy import allclose
    return allclose(x, y, rtol=rel, atol=tol)


def tolerance(x, tol=1e-15, rel=1e-15):
    """relative plus absolute difference"""
    return tol + abs(x)*rel


# end of file

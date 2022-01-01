#!/usr/bin/env python
# ******NOTICE***************
# optimize.py module by Travis E. Oliphant
#
# You may copy and use this module as you see fit with no
# guarantee implied provided you keep this notice in all copies.
# *****END NOTICE************
#
# Forked by: Mike McKerns (February 2009)
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2009-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""modified algorithms from local copy of scipy.optimize"""

#__all__ = ['fmin', 'fmin_powell', 'fmin_ncg', 'fmin_cg', 'fmin_bfgs',
#           'fminbound','brent', 'golden','bracket','rosen','rosen_der',
#           'rosen_hess', 'rosen_hess_prod', 'brute', 'approx_fprime',
#           'line_search', 'check_grad']

from mystic._scipy060optimize import *
from mystic._scipy060optimize import _cubicmin, _quadmin, _epsilon
from mystic._scipy060optimize import _linesearch_powell, _endprint

def line_search(f, myfprime, xk, pk, gfk, old_fval, old_old_fval,
                args=(), c1=1e-4, c2=0.9, amax=50, self=None):
    """Find alpha that satisfies strong Wolfe conditions.

    :Parameters:

        f : callable f(x,*args)
            Objective function.
        myfprime : callable f'(x,*args)
            Objective function gradient (can be None).
        xk : ndarray
            Starting point.
        pk : ndarray
            Search direction.
        gfk : ndarray
            Gradient value for x=xk (xk being the current parameter
            estimate).
        args : tuple
            Additional arguments passed to objective function.
        c1 : float
            Parameter for Armijo condition rule.
        c2 : float
            Parameter for curvature condition rule.

    :Returns:

        alpha0 : float
            Alpha for which ``x_new = x0 + alpha * pk``.
        fc : int
            Number of function evaluations made.
        gc : int
            Number of gradient evaluations made.

    :Notes:

        Uses the line search algorithm to enforce strong Wolfe
        conditions.  See Wright and Nocedal, 'Numerical Optimization',
        1999, pg. 59-60.

        For the zoom phase it uses an algorithm by [...].

    """

    global _ls_fc, _ls_gc, _ls_ingfk
    _ls_fc = 0
    _ls_gc = 0
    _ls_ingfk = None
    def phi(alpha):
        global _ls_fc
        _ls_fc += 1
        return f(xk+alpha*pk,*args)

    if isinstance(myfprime,type(())):
        def phiprime(alpha):
            global _ls_fc, _ls_ingfk
            _ls_fc += len(xk)+1
            eps = myfprime[1]
            fprime = myfprime[0]
            newargs = (f,eps) + args
            _ls_ingfk = fprime(xk+alpha*pk,*newargs)  # store for later use
            return numpy.dot(_ls_ingfk,pk)
    else:
        fprime = myfprime
        def phiprime(alpha):
            global _ls_gc, _ls_ingfk
            _ls_gc += 1
            _ls_ingfk = fprime(xk+alpha*pk,*args)  # store for later use
            return numpy.dot(_ls_ingfk,pk)

    alpha0 = 0
    phi0 = old_fval
    derphi0 = numpy.dot(gfk,pk)

    alpha1 = min(1.0,1.01*2*(phi0-old_old_fval)/derphi0)

    if alpha1 == 0:
        # This shouldn't happen. Perhaps the increment has slipped below
        # machine precision?  For now, set the return variables skip the
        # useless while loop, and raise warnflag=2 due to possible imprecision.
        alpha_star = None
        fval_star = old_fval
        old_fval = old_old_fval
        fprime_star = None

    phi_a1 = phi(alpha1)
    #derphi_a1 = phiprime(alpha1)  evaluated below

    phi_a0 = phi0
    derphi_a0 = derphi0

    i = 1
    maxiter = 10
    while 1:         # bracketing phase
        if alpha1 == 0:
            break
        if (phi_a1 > phi0 + c1*alpha1*derphi0) or \
           ((phi_a1 >= phi_a0) and (i > 1)):
            alpha_star, fval_star, fprime_star = \
                        zoom(alpha0, alpha1, phi_a0,
                             phi_a1, derphi_a0, phi, phiprime,
                             phi0, derphi0, c1, c2)
            break

        derphi_a1 = phiprime(alpha1)
        if (abs(derphi_a1) <= -c2*derphi0):
            alpha_star = alpha1
            fval_star = phi_a1
            fprime_star = derphi_a1
            break

        if (derphi_a1 >= 0):
            alpha_star, fval_star, fprime_star = \
                        zoom(alpha1, alpha0, phi_a1,
                             phi_a0, derphi_a1, phi, phiprime,
                             phi0, derphi0, c1, c2)
            break

        alpha2 = 2 * alpha1   # increase by factor of two on each iteration
        i = i + 1
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1)
        derphi_a0 = derphi_a1

        # stopping test if lower function not found
        if (i > maxiter):
            alpha_star = alpha1
            fval_star = phi_a1
            fprime_star = None
            break

        if self._EARLYEXIT: # In case it gets stuck in this loop, 'exit' will work
            alpha_star = alpha1
            fval_star = phi_a1
            fprime_star = None
            break

    if fprime_star is not None:
        # fprime_star is a number (derphi) -- so use the most recently
        # calculated gradient used in computing it derphi = gfk*pk
        # this is the gradient at the next step no need to compute it
        # again in the outer loop.
        fprime_star = _ls_ingfk

    return alpha_star, _ls_fc, _ls_gc, fval_star, old_fval, fprime_star

def line_search_BFGS(f, xk, pk, gfk, old_fval, args=(), c1=1e-4, alpha0=1,\
                     self=None):
    """Minimize over alpha, the function ``f(xk+alpha pk)``.

    Uses the interpolation algorithm (Armiijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pg. 56-57

    :Returns: (alpha, fc, gc)

    """

    xk = atleast_1d(xk)
    fc = 0
    phi0 = old_fval # compute f(xk) -- done in past loop
    phi_a0 = f(*((xk+alpha0*pk,)+args))
    fc = fc + 1
    derphi0 = numpy.dot(gfk,pk)

    if (phi_a0 <= phi0 + c1*alpha0*derphi0):
        return alpha0, fc, 0, phi_a0

    # Otherwise compute the minimizer of a quadratic interpolant:

    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = f(*((xk+alpha1*pk,)+args))
    fc = fc + 1

    if (phi_a1 <= phi0 + c1*alpha1*derphi0): 
        return alpha1, fc, 0, phi_a1

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satifies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.

    while 1:       # we are assuming pk is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + numpy.sqrt(abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = f(*((xk+alpha2*pk,)+args))
        fc = fc + 1

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, fc, 0, phi_a2

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

        # In case it gets stuck in this loop, to make 'exit' work.
        if self._EARLYEXIT:
            return alpha2, fc, 0, phi_a2

def approx_fprime(xk,f,epsilon,*args,**kwds):
    """Finite-difference approximation of the gradient of a scalar function.

    Parameters
    ----------
    xk : array_like
        The coordinate vector at which to determine the gradient of `f`.
    f : callable
        The function of which to determine the gradient (partial derivatives).
        Should take `xk` as first argument, other arguments to `f` can be
        supplied in ``*args``.  Should return a scalar, the value of the
        function at `xk`.
    epsilon : array_like
        Increment to `xk` to use for determining the function gradient.
        If a scalar, uses the same finite difference delta for all partial
        derivatives.  If an array, should contain one value per element of
        `xk`.
    f0 : float, optional
        Initial function value.
    \\*args : args, optional
        Any other arguments that are to be passed to `f`.

    Returns
    -------
    grad : ndarray
        The partial derivatives of `f` to `xk`.

    Notes
    -----
    The function gradient is determined by the forward finite difference
    formula::

                 f(xk[i] + epsilon[i]) - f(xk[i])
        f'[i] = ---------------------------------
                            epsilon[i]

    The main use of `approx_fprime` is in scalar function optimizers like
    `fmin_bfgs`, to determine numerically the Jacobian of a function.

    Examples
    --------
    >>> def func(x, c0, c1):
    ...     "Coordinate vector `x` should be an array of size two."
    ...     return c0 * x[0]**2 + c1*x[1]**2

    >>> x = np.ones(2)
    >>> c0, c1 = (1, 200)
    >>> eps = np.sqrt(np.finfo(float).eps)
    >>> approx_fprime(x, func, [eps, np.sqrt(200) * eps], c0, c1)
    array([   2.        ,  400.00004198])
    """
    xk = xk.copy()  #XXX: this line has been added for scipy_ncg
    f0 = kwds['f0'] if 'f0' in kwds else f(*((xk,)+args))
    grad = numpy.zeros((len(xk),), float)
    ei = numpy.zeros((len(xk),), float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        grad[k] = (f(*((xk+d,)+args)) - f0) / d[k]
        ei[k] = 0.0
    return grad


if __name__ == "__main__":
    pass

# End of file

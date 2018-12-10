#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2018 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""functions related to interpolation
1-D and n-D interpolation, building a grid of points, calculating gradients
"""
#XXX: interpf produces f(*x) and gradient takes f(*x), use f(x) instead?
#XXX: make interpolation more accurate, at least on the given data points?
#XXX: utilize numdifftools/theano? scipy.misc.derivative?

from mystic.math import _rbf
Rbf = _rbf.Rbf


def _sort(x, y=None, param=0):
    '''sort x (and y, if provided) by the given parameter

    For example:
    >>> _sort([[1,5,9],[7,8,3],[4,2,6]], param=0)
    array([[1, 5, 9],
           [4, 2, 6],
           [7, 8, 3]])
    >>> _sort([[1,5,9],[7,8,3],[4,2,6]], param=1)
    array([[4, 2, 6],
           [1, 5, 9],
           [7, 8, 3]])
    >>> _sort([[1,5,9],[7,8,3],[4,2,6]], [4,3,2])
    (array([[1, 5, 9],
           [4, 2, 6],
           [7, 8, 3]]), array([4, 2, 3]))
    >>>
    '''
    import numpy as np
    x = np.asarray(x)
    i = np.argsort(x, axis=0)[:, param]
    if y is None:
        return x[i]
    y = np.asarray(y)
    return x[i],y[i]


def _isin(i, x):
    '''check if i is in x, where i is an iterable'''
    import numpy as np
    x = np.asarray(x) #XXX: doesn't verify that i is iterable
    if x.ndim is 1: #FIXME: expects i not nessarily an iterable
        return np.equal(i,x).any()
    return np.equal(i,x).all(axis=-1).any()


def _to_objective(function):
    '''convert f(*xargs, **kwds) to f(x, *args, **kwds) where xargs=x+args

    Input:
      objective: a function of the form f(*xargs, **kwds)

    Output:
      a function of the form f(x, *args, **kwds)

    For example:
      >>> @_to_objective
      ... def cost(x,y,z):
      ...   return x+y+z
      ... 
      >>> x = [1,2,3]
      >>> cost(x)
      6
    '''
    def objective(x, *args, **kwds):
        return function(*(tuple(x)+args), **kwds)
    objective.__doc__ = function.__doc__
    return objective


def _to_function(objective, ndim=None):
    '''convert f(x, *args, **kwds) to f(*xargs, **kwds) where xargs=x+args

    Input:
      objective: a function of the form f(x, *args, **kwds)
      ndim: an int, if provided, is the length of x in f(x, *args, **kwds)

    Output:
      a function of the form f(*xargs, **kwds)

    For example:
      >>> @_to_function
      ... def model(x):
      ...   return sum(x)
      ... 
      >>> model(1,2,3)
      6
    '''
    obj = objective
    if ndim is None:
        def function(*args, **kwds):
            return obj(args, **kwds)
    else:
        def function(*args, **kwds):
            return obj(args[:ndim], *args[ndim:], **kwds)
    function.__doc__ = objective.__doc__
    return function


def _unique(x, z=None, sort=False, index=False): #XXX: move to tools?
    '''return the unique values of x, and corresponding z (if provided)

    Input:
      x: an array of shape (npts, dim) or (npts,)
      z: an array of shape (npts,)
      sort: boolean, if True, return arrays sorted on x [default=True]
      index: boolean, if True, also return an index array that recovers x

    Output:
      unique (potentially sorted) x, and z (if provided) and/or index
    ''' # avoid LinAlgError when interpolating
    import numpy as np
    x,i,v = np.unique(x, return_index=True, return_inverse=True, axis=0)
    if not z is None: z = np.asarray(z)[i]
    if not sort:
        i = i.argsort()
        x = x[i]
        if not z is None: z = z[i]
        # reorder v, so that it recovers the original x
        k = np.arange(0,len(v))
        k[i] = np.arange(0,len(i))
        v = k[v]; del k
    if index is True:
        return (x, z, v) if (not z is None) else (x,v)
    return (x, z) if (not z is None) else x


def sort_axes(*axes, **kwds):
    '''sort axes along the selected primary axis

    Input:
      axes: a tuple of arrays of points along each axis
      axis: an integer corresponding to the axis upon which to sort

    Output:
      a tuple of arrays sorted with regard to the selected axis
    ''' #NOTE: last entry might be 'values' (i.e. f(x))
    import numpy as np
    #XXX: add an option (or new function) for monotonic increasing?
    # instead of string matching, use dict lookup
    axes = np.asarray(axes)
    axis = kwds['axis'] if 'axis' in kwds else 0
    i = axes[axis].argsort()
    return tuple(ax[i] for ax in axes)


def axes(mins, maxs, npts=None):
    '''generate a tuple of arrays defining axes on a grid, given bounds

    Input:
      mins: a tuple of lower bounds for coordinates on the grid
      maxs: a tuple of upper bounds for coordinates on the grid
      npts: a tuple of grid shape, or integer number or points on grid

    Output:
      a tuple of arrays defining the axes of the coordinate grid

    NOTE:
      If npts is not provided, a default of 50 points in each direction is used.
    ''' #NOTE: ensures all axes will be ordered (monotonically increasing)
    import numpy as np
    if not hasattr(mins, '__len__'):
        mins = (mins,)
    if not hasattr(maxs, '__len__'):
        maxs = (maxs,)
    if npts is None: npts = 50**len(mins) #XXX: better default?
    if not hasattr(npts, '__len__'):
        npts = (max(int(npts**(1./len(mins))),1),)*len(mins)
    return tuple(np.linspace(*r) for r in zip(mins,maxs,npts))


def _swapvals(x, i):
    '''swap values from column 0 with column i'''
    x = np.asarray(x).T
    x[[0,i]] = x[[i,0]]
    return x.T


def _axes(x):
    '''convert measured data 'x' to a tuple of 'axes'

    Input:
      x: an array of shape (npts, dim) or (npts,)

    Output:
      a tuple of arrays (x0, ..., xm), where m = dim-1 and len(xi) = npts
    '''
    import numpy as np
    x = np.asarray(x)
    if x.ndim is 1:
        return (x,)
    return tuple(x.T)


def grid(*axes):
    '''generate tuple of (irregular) coordinate grids, given coordinate axes

    Input:
      axes: a tuple of arrays defining the axes of the coordinate grid

    Output:
      the resulting coordinate grid
    '''
    import numpy as np #FIXME: fails large len(axes)
    return tuple(np.meshgrid(*axes, indexing='ij'))


def _noisy(x, scale=1e-8): #XXX: move to tools?
    '''add random gaussian noise of the given scale, or None if scale=None

    Input:
      x: an array of shape (npts, dim) or (npts,)

    Output:
      an array of shape (npts, dim) or (npts,), where noise has been added
    '''
    import numpy as np
    return x if not scale else x + np.random.normal(scale=scale, size=x.shape)


def interpolate(x, z, xgrid, method=None):
    '''interpolate to find z = f(x) sampled at points defined by xgrid

    Input:
      x: an array of shape (npts, dim) or (npts,)
      z: an array of shape (npts,)
      xgrid: (irregular) coordinate grid on which to sample z = f(x)
      method: string for kind of interpolator

    Output:
      interpolated points on a grid, where z = f(x) has been sampled on xgrid

    NOTE:
      if scipy is not installed, will use np.interp for 1D (non-rbf),
      or mystic's rbf otherwise. default method is 'nearest' for
      1D and 'linear' otherwise. method can be one of ('rbf','linear',
      'nearest','cubic','inverse','gaussian','quintic','thin_plate').
    '''
    x,z = _unique(x,z,sort=True)
    # avoid nan as first value #XXX: better choice than 'nearest'?
    if method is None: method = 'nearest' if x.ndim is 1 else 'linear'
    methods = dict(rbf=0, linear=1, nearest=2, cubic=3)
    functions = {0:'multiquadric', 1:'linear', 2:'nearest', 3:'cubic'}
    # also: ['thin_plate','inverse','gaussian','quintic']
    kind = methods[method] if method in methods else None
    function = functions[kind] if (not kind is None) else method
    if kind is None: kind = 0 #XXX: or raise KeyError ?
    try:
        import scipy.interpolate as si
    except ImportError:
        if not kind is 0: # non-rbf
            if x.ndim is 1: # is 1D, so use np.interp
                import numpy as np
                return np.interp(*xgrid, xp=x, fp=z)
            kind = 0 # otherwise, utilize mystic's rbf
        si = _rbf
    if kind is 0: # 'rbf' -> Rbf
        import numpy as np
        rbf = si.Rbf(*np.vstack((x.T, z)), function=function, smooth=0)
        return rbf(*xgrid)
    # method = 'linear' -> LinearNDInterpolator
    # method = 'nearest' -> NearestNDInterpolator
    # method = 'cubic' -> (1D: spline; 2D: CloughTocher2DInterpolator)
    return si.griddata(x, z, xgrid, method=method)#, rescale=False)


def interpf(x, z, method=None): #XXX: return f(*x) or f(x)?
    '''interpolate to find f, where z = f(*x)

    Input:
      x: an array of shape (npts, dim) or (npts,)
      z: an array of shape (npts,)
      method: string for kind of interpolator

    Output:
      interpolated function f, where z = f(*x)

    NOTE:
      if scipy is not installed, will use np.interp for 1D (non-rbf),
      or mystic's rbf otherwise. default method is 'nearest' for
      1D and 'linear' otherwise. method can be one of ('rbf','linear',
      'nearest','cubic','inverse','gaussian','quintic','thin_plate').
    '''
    x,z = _unique(x,z,sort=True)
    # avoid nan as first value #XXX: better choice than 'nearest'?
    if method is None: method = 'nearest' if x.ndim is 1 else 'linear'
    methods = dict(rbf=0, linear=1, nearest=2, cubic=3)
    functions = {0:'multiquadric', 1:'linear', 2:'nearest', 3:'cubic'}
    # also: ['thin_plate','inverse','gaussian','quintic']
    kind = methods[method] if method in methods else None
    function = functions[kind] if (not kind is None) else method
    if kind is None: kind = 0 #XXX: or raise KeyError ?
    try:
        import scipy.interpolate as si
    except ImportError:
        if not kind is 0: # non-rbf
            if x.ndim is 1: # is 1D, so use np.interp
                import numpy as np
                return lambda xn: np.interp(xn, xp=x, fp=z)
            kind = 0 # otherwise, utilize mystic's rbf
        si = _rbf
    if kind is 0: # 'rbf'
        import numpy as np
        return si.Rbf(*np.vstack((x.T, z)), function=function, smooth=0)
    elif x.ndim is 1: 
        return si.interp1d(x, z, fill_value='extrapolate', bounds_error=False, kind=method)
    elif kind is 1: # 'linear'
        return si.LinearNDInterpolator(x, z, rescale=False)
    elif kind is 2: # 'nearest'
        return si.NearestNDInterpolator(x, z, rescale=False)
    #elif x.ndim is 1: # 'cubic'
    #    return lambda xn: si.spline(x, z, xn)
    return si.CloughTocher2DInterpolator(x, z, rescale=False)


def _gradient(x, grid):
    '''find gradient of f(x), sampled on the coordinate grid defined by x

    Input:
      x: an array of shape (npts, dim) or (npts,)
      grid: (irregular) coordinate grid of z = f(x), with axes = _axes(x)

    Output:
      list of length dim (or array in 1D), gradient of the points on grid

    NOTE:
      output will be of the form (dim,)+grid.shape
    ''' #XXX: can unique be used in this function?
    import numpy as np
    err = np.seterr(all='ignore') # silence warnings (division by nan)
    z = np.gradient(grid, *_axes(x))
    np.seterr(**err)
    return z #XXX: for (N,1) & (N,), should return a tuple?


def _fprime(x, fx, method=None):
    '''find gradient of fx at x, where fx is a function z=fx(x)

    Input:
      x: an array of shape (npts, dim) or (npts,)
      fx: a function, z = fx(x)
      method: string for kind of gradient method

    Output:
      array of dimensions x.shape, gradient of the points at (x,fx)

    NOTE:
      if method is 'approx' (the default) use mystic's approx_fprime,
      which uses a local gradient approximation; other choices are
      'symbolic', which uses mpmath.diff if installed.
    '''
    x,i = _unique(x, index=True)
    if method is None or method == 'approx':
        import numpy as np
        from mystic._scipyoptimize import approx_fprime, _epsilon
        err = np.seterr(all='ignore') # silence warnings (division by nan)
        #fx = _to_objective(fx) # conform to gradient interface
        x,s = np.atleast_2d(x),x.shape
        z = np.array([approx_fprime(xi, fx, _epsilon) for xi in x]).reshape(*s)
        np.seterr(**err)
        return z[i]
    try: #XXX: mpmath.diff is more error prone -- don't use it?
        from mpmath import diff
    except ImportError:
        return _fprime(x, fx, method=None)[i]
    import numpy as np
    err = np.seterr(all='ignore') # silence warnings (division by nan)
    #fx = _to_objective(fx) # conform to gradient interface
    k = range(s[-1])
    z = np.array([[diff(lambda *x: fx(_swapvals(x,j)), xk[_swapvals(k,j)], (1,)) for j in k] for xk in x], dtype=x.dtype).reshape(*s)
    np.seterr(**err)
    return z[i]


def gradient(x, fx, method=None, approx=True): #XXX: take f(*x) or f(x)?
    '''find gradient of fx at x, where fx is a function z=fx(*x) or an array z

    Input:
      x: an array of shape (npts, dim) or (npts,)
      fx: an array of shape (npts,) **or** a function, z = fx(*x)
      method: string for kind of interpolator
      approx: if True, use local approximation method

    Output:
      array of dimensions x.shape, gradient of the points at (x,fx)

    NOTE:
      if approx is True, use mystic's approx_fprime, which uses a local
      gradient approximation; otherwise use numpy's gradient method which
      performs a more memory-intensive calcuation on a grid.

    NOTE:
      if scipy is not installed, will use np.interp for 1D (non-rbf),
      or mystic's rbf otherwise. default method is 'nearest' for
      1D and 'linear' otherwise. method can be one of ('rbf','linear',
      'nearest','cubic','inverse','gaussian','quintic','thin_plate').
    ''' #NOTE: uses 'unique' in all cases
    #XXX: nice to have a test for smoothness, worth exploring?
    import numpy as np
    x = np.asarray(x)
    if not hasattr(fx, '__call__'):
        fx = interpf(x, fx, method=method)
    if approx is True:
        fx = _to_objective(fx) # conform to gradient interface
        if x.ndim is 1:
            return _fprime(x.reshape(x.shape+(1,)), fx).reshape(x.shape)
        return _fprime(x, fx)
    q = True #XXX: is q=True better for memory, worse for accuracy?
    if q is True: x,i = _unique(x, index=True)
    else: i = slice(None)
    gfx = _gradient(x, fx(*grid(*_axes(x)))) #XXX: diagonal w/o full grid?
    if type(gfx) is not list:
        gfx.shape = x.shape
        return gfx[i]
    idx = np.diag_indices(*x.shape)
    return np.array([j[idx] for j in gfx]).T[i]


# SEE: https://stackoverflow.com/questions/31206443
def _hessian(x, grid):
    '''find hessian of f(x), sampled on the coordinate grid defined by x

    Input:
      x: an array of shape (npts, dim) or (npts,)
      grid: (irregular) coordinate grid of z = f(x), with axes = _axes(x)

    Output:
      array of shape indicated in NOTE, hessian of the points on grid

    NOTE:
      output will be of the form (dim,dim)+grid.shape, where output hess[i,j]
      corresponds to the second derivative z_{i,j} with i,j in range(dim).
      For a 1D array x, output will be a 1D array of the same length.
      The hessian is calculated using finite differences.
    ''' #XXX: can unique be used in this function?
    import numpy as np
    x =  np.asarray(x)
    hess = np.empty((grid.ndim, grid.ndim) + grid.shape, dtype=grid.dtype)
    if grid.ndim is 1: #XXX: is (1,1,N) really desirable when x is (N,1)?
        hess[0,0] = _gradient(x, _gradient(x, grid))
        return hess.ravel() if x.ndim is 1 else hess
    for k, grad_k in enumerate(_gradient(x, grid)):
        # apply gradient to every component of the first derivative
        for l, grad_kl in enumerate(_gradient(x, grad_k)):
            hess[k,l] = grad_kl
    return hess


def hessian(x, fx, method=None, approx=True): #XXX: take f(*x) or f(x)?
    '''find hessian of fx at x, where fx is a function z=fx(*x) or an array z

    Input:
      x: an array of shape (npts, dim) or (npts,)
      fx: an array of shape (npts,) **or** a function, z = fx(*x)
      method: string for kind of interpolator
      approx: if True, use local approximation method

    Output:
      array of shape indicated in NOTE, hessian of the points at (x,fx)

    NOTE:
      output will be of the form x.shape+(dim,), where hess[:,i,j]
      corresponds to the second derivative z_{i,j} with i,j in range(dim).
      For a 1D array x, output will be a 1D array of the same length.
      
    NOTE:
      if approx is True, first use interpolation to build gradient functions
      in each direction, then use mystic's approx_fprime, which uses a local
      gradient approximation; otherwise use numpy's gradient method which
      performs a more memory-intensive calcuation on a grid.

    NOTE:
      if scipy is not installed, will use np.interp for 1D (non-rbf),
      or mystic's rbf otherwise. default method is 'nearest' for
      1D and 'linear' otherwise. method can be one of ('rbf','linear',
      'nearest','cubic','inverse','gaussian','quintic','thin_plate').

    NOTE:
      method string can provide either one or two methods (i.e. 'rbf
      or 'rbf, cubic'), where if two methods are provided, the first
      will be used to interpolate f(x) and the second will be used to
      interpolate the gradient of f(x).
    ''' #NOTE: uses 'unique' in all cases
    import numpy as np
    x = np.asarray(x)
    if method is None:
        method = (None,)
    else: #NOTE: accepts either 'one', 'one, two' or 'one; two'
        method = [s.strip() for s in method.replace(';',',').split(',')]
    if not hasattr(fx, '__call__'):
        fx = interpf(x, fx, method=method[0])
    if approx is True:
        fx = _to_objective(fx) # conform to gradient interface
        #XXX: alternate: find grid w/_gradient, use _fprime to find hessian
        if x.ndim is 1:
            gfx = _fprime(x.reshape(x.shape+(1,)), fx).reshape(x.shape)
            gfx = interpf(x, gfx, method=method[-1])
            return _fprime(x.reshape(x.shape+(1,)), gfx).reshape(x.shape)
        gfx = _fprime(x, fx).T
        gfx = [_to_objective(interpf(x, i, method=method[-1])) for i in gfx]
        hess = np.empty(x.shape + x.shape[-1:], dtype=x.dtype)
        for i, gf in enumerate(gfx):
            hess[:,:,i] = _fprime(x, gf)
        return hess
    q = True #XXX: is q=True better for memory, worse for accuracy?
    if q is True: x,i = _unique(x, index=True)
    else: i = slice(None)
    hess = _hessian(x, fx(*grid(*_axes(x))))
    if hess.size is hess.shape[-1]:
        hess.shape = x.shape #XXX: if (N,1), is (N,1,1) or (N,1) right shape?
        return hess[i]
    idx = np.diag_indices(*x.shape)
    return np.array([[k[idx] for k in j] for j in hess]).T[i]#XXX: right shape?


def hessian_diagonal(x, fx, method=None, approx=True):
    '''find hessian diagonal of fx at x, with fx a function z=fx(*x) or array z

    Input:
      x: an array of shape (npts, dim) or (npts,)
      fx: an array of shape (npts,) **or** a function, z = fx(*x)
      method: string for kind of interpolator
      approx: if True, use local approximation method

    Output:
      array of dimensions x.shape, hessian diagonal of the points at (x,fx)

    NOTE:
      if approx is True, first use interpolation to build gradient functions
      in each direction, then use mystic's approx_fprime, which uses a local
      gradient approximation; otherwise use numpy's gradient method which
      performs a more memory-intensive calcuation on a grid.

    NOTE:
      if scipy is not installed, will use np.interp for 1D (non-rbf),
      or mystic's rbf otherwise. default method is 'nearest' for
      1D and 'linear' otherwise. method can be one of ('rbf','linear',
      'nearest','cubic','inverse','gaussian','quintic','thin_plate').
    '''
    hess = hessian(x, fx, method)
    if hess.ndim is not 3: # (i.e. is 1 or 2)
        return hess
    import numpy as np
    x = np.asarray(x)
    return np.array([hess[:,i,i] for i in range(x.ndim+1)]).T
    #TODO: calculate/approximate without calculating the full hessian


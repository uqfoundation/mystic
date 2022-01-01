#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2018-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
calculate graphical distance of (interpolated) function from a dataset 
converters for klepto.archives.dir_archive and mystic.math.legacydata.dataset
read (params,cost) from LoggingMonitor archive or klepto.archive.dir_archive
"""

def _getitem(data, axis):
    """get the selected axis of the tuple-valued dataset

    Inputs:
      data: a mystic.math.legacydata.dataset of i points, M inputs, N outputs
      axis: int, the desired index the tuple-valued dataset [0,N]
    """
    if len(data.values) and type(data.values[0]) not in (tuple,list):
        msg = "cannot get axis %s for single-valued dataset.values" % axis
        raise ValueError(msg)
    # assumes axis is an int; select values corresponding to axis
    from mystic.math.legacydata import dataset, datapoint as datapt
    ds = dataset()
    for pt in data:
        ds.append(datapt(pt.position, pt.value[axis], pt.id, pt.cone.slopes))
    return ds


# somewhat hacky include of tuple-valued dataset
def interpolate(data, step=None, **kwds):
    """generate interpolated function y=f(x) from data (x,y)

    Inputs:
      data: mystic.math.legacydata.dataset of i points, M inputs, N outputs
      step: int, stepsize for interpolation (default: do not skip any points)

    Additional Inputs:
      method: string for kind of interpolator
      maxpts: int, maximum number of points (x,z) to use from the monitor
      noise: float, amplitude of gaussian noise to remove duplicate x
      extrap: if True, extrapolate a bounding box (can reduce # of nans)
      arrays: if True, return a numpy array; otherwise don't return arrays
      axis: int in [0,N], index of z on which to interpolate (all, by default)

    NOTE:
      if scipy is not installed, will use np.interp for 1D (non-rbf),
      or mystic's rbf otherwise. default method is 'nearest' for
      1D and 'linear' otherwise. method can be one of ('rbf','linear',
      'nearest','cubic','inverse','gaussian','quintic','thin_plate').

    NOTE:
      additional keyword arguments (epsilon, smooth, norm) are avaiable
      for use with a Rbf interpolator. See mystic.math.interpolate.Rbf
      for more details.
    """
    # interpolate for each member of tuple-valued data, unless axis provided
    axis = kwds.pop('axis', None) # axis for tuple-valued data
    if axis is None:
        if len(data.values) and type(data.values[0]) in (tuple,list):
            # iterate over each axis, build a 'combined' interpf
            def objective(x, axis=None):
                fs = objective.__axis__
                if axis is None:
                    return tuple(fi(x) for fi in fs)
                return fs[axis](x)
            objective.__axis__ = [interpolate(data, axis=ax, **kwds) for ax,val in enumerate(data.values[0])]
            return objective
        # else: data is single-valued
    else: # axis is not None
        data = _getitem(data, axis)
    #XXX: what if dataset is empty? (i.e. len(data.values) == 0)
    from interpolator import interpolate as interp
    ii = interp(data.coords[::step], z=data.values[::step], **kwds)
    from mystic.math.interpolate import _to_objective
    function = _to_objective(ii.function)   
    function.__axis__ = axis #XXX: bad idea: list of funcs, or int/None ?
    return function


def distance(data, function=None, hausdorff=True, **kwds):
    """get graphical distance between function y=f(x) and a dataset

    Inputs:
      data: a mystic.math.legacydata.dataset of i points, M inputs, N outputs
      function: a function y=f(*x) of data (x,y)
      hausdorff: if True, use Hausdorff norm

    Additional Inputs:
      method: string for kind of interpolator
      maxpts: int, maximum number of points (x,z) to use from the monitor
      noise: float, amplitude of gaussian noise to remove duplicate x
      extrap: if True, extrapolate a bounding box (can reduce # of nans)
      arrays: if True, return a numpy array; otherwise don't return arrays
      axis: int in [0,N], index of z on which to interpolate (all, by default)

    NOTE:
      if scipy is not installed, will use np.interp for 1D (non-rbf),
      or mystic's rbf otherwise. default method is 'nearest' for
      1D and 'linear' otherwise. method can be one of ('rbf','linear',
      'nearest','cubic','inverse','gaussian','quintic','thin_plate').

    NOTE:
      data and function may provide tuple-valued or single-valued output.
      Distance will be measured component-wise, resulting in a tuple of
      distances, unless an 'axis' is selected. If an axis is selected,
      then return distance for the selected component (i.e. axis) only.
    """
    """ #FIXME: the following is generally true (doesn't account for 'fax')
    # function is multi-value & has its components
      # data is multi-value
        # axis is None => apply function component-wise to data
        # axis is int => apply function component-wise to single axis
      # data is single-value
        # axis is None => ERROR: unknown which function component to apply
        # axis is int => apply selected function component to data
    # function is single-value (int or None)
      # data is multi-value
        # axis is None => ERROR: unknown which axis to apply function
        # axis is int => apply function to selected axis of data
      # data is single-value
        # axis is None => apply function to data
        # axis is int => ERROR: can't take axis of single-valued data
    # function is None
      # data is multi-value
        # axis is None => [fmv] => apply function component-wise to data
        # axis is int => [fsv] => apply function to selected axis of data
      # data is single-value
        # axis is None => [fsv] => apply function to data
        # axis is int => ERROR: can't take axis of single-valued data
    """
    axis = kwds.get('axis', None) # axis for tuple-valued data and/or function
    if function is None:
        function = interpolate(data, **kwds)
    import mystic.math.distance as md #FIXME: simplify axis vs fax
    from mystic.math.interpolate import _to_objective
    fax = getattr(function, '__axis__', None) # axis for tuple-valued function
    if axis is None:
        if len(data.values) and type(data.values[0]) in (tuple,list):
            if type(fax) is list: # multi-value func, multi-value data
                import numpy as np
                return np.array([md.graphical_distance(_to_objective(j), _getitem(data, i), hausdorff=hausdorff) for i,j in enumerate(fax)])
            elif type(fax) is int: # single-value func, multi-value data
                return md.graphical_distance(_to_objective(function), _getitem(data, fax), hausdorff=hausdorff)
            else: # single-value func, multi-value data
                msg = "axis required for multi-valued dataset.values"
                raise ValueError(msg)
        else:
            if type(fax) is list: # multi-value func, singe-value data
                msg = "axis required for multi-valued function"
                raise ValueError(msg)
            else: # single-value func, singe-value data
                return md.graphical_distance(_to_objective(function), data, hausdorff=hausdorff)
    else:
        if len(data.values) and type(data.values[0]) in (tuple,list):
            if type(fax) is list: # multi-value func, multi-value data
                return md.graphical_distance(_to_objective(fax[axis]), _getitem(data, axis), hausdorff=hausdorff)
            elif type(fax) is int and fax != axis: # single-value func, multi-value data
                msg = "inconsistent axis for multi-valued dataset.values"
                raise ValueError(msg)
            else: # single-value func, multi-value data
                return md.graphical_distance(_to_objective(function), _getitem(data, axis), hausdorff=hausdorff)
        else:
            if type(fax) is list: # multi-value func, singe-value data
                return md.graphical_distance(_to_objective(fax[axis]), data, hausdorff=hausdorff)
            elif type(fax) is int:
                if fax == axis: # single-value func, singe-value data
                    return md.graphical_distance(_to_objective(function), data, hausdorff=hausdorff)
                msg = "inconsistent axis for multi-valued function"
                raise ValueError(msg)
            else: # single-value func, singe-value data
                _getitem(data, axis) # raise ValueError
    return NotImplemented # should never get here


# reader for logfile(s)
def read_logfile(filename):
    """read 'parameters' and 'cost' from LoggingMonitor archive

    Inputs:
      filename: str path to location of klepto.archives.dir_archive
    """
    from mystic.munge import read_trajectories
    param, param, cost = read_trajectories(filename)
    return param, cost


# reader for archive(s)
def read_archive(filename, axis=None): #NOTE: could return iterators
    """read 'parameters' and 'cost' from klepto.dir_archive

    Inputs:
      filename: str path to location of klepto.archives.dir_archive
      axis: int, the desired index the tuple-valued dataset [0,N]
    """
    from klepto.archives import dir_archive
    arch = dir_archive(filename, cached=True)
    return for_monitor(arch, axis=axis)


def for_monitor(archive, inverted=False, axis=None):
    """convert klepto.dir_archive to param,cost for a monitor

    Inputs:
      archive: a klepto.archive instance
      inverted: if True, invert the z-axis of the Monitor (i.e. z => -z) 
      axis: int, the desired index the tuple-valued dataset [0,N]
    """
    arxiv = getattr(archive, '__archive__', archive)
    # param = list(arxiv.keys())
    # param = list(list(k[-1]) for k in arxiv.keys())
    param = list(list(k) for k in arxiv.keys())
    if inverted:
        if axis is None:
            cost = list(tuple(-i for i in k) for k in arxiv.values())
        else:
            cost = list(-k[axis] for k in arxiv.values())
    else:
        cost = arxiv.values()
        if axis is None:
            cost = cost if type(cost) is list else list(cost)
        else:
            cost = list(k[axis] for k in arxiv.values())
    return (param, cost)


def from_archive(cache, data=None, **kwds):
    '''convert a klepto.archive to a mystic.math.legacydata.dataset

    Inputs:
      cache: a klepto.archive instance
      data: a mystic.math.legacydata.dataset of i points, M inputs, N outputs
      axis: int, the desired index the tuple-valued dataset [0,N]
      ids: a list of ids for the data, or a function ids(x,y) to generate ids
    '''
    axis = kwds.get('axis', None)
    ids = kwds.get('ids', _iargsort) #XXX: better None?
    import numpy as np #FIXME: accept lipshitz coeffs as input
    if data is None:
        from mystic.math.legacydata import dataset
        data = dataset()
    # import klepto as kl
    # ca = kl.archives.dir_archive('__objective_5D_cache__', cached=False)
    # k = keymap=kl.keymaps.keymap()
    # c = kl.inf_cache(keymap=k, tol=1, ignore=('**','out'), cache=ca)
    # memo = c(lambda *args, **kwds: kwds['out'])
    # cache = memo.__cache__()
    y = np.array(list(cache.items()), dtype=object).T #NOTE: cache.items() slow
    if not len(y): return data
    if callable(ids): #XXX: don't repeat tolist
        ids = ids(y[0].tolist(), y[1].tolist(), axis=axis)
    if axis is None: #XXX: dataset should be single valued
        return data.load(y[0].tolist(), y[1].tolist(), ids=ids)
    return data.load(y[0].tolist(), [i[axis] for i in y[1]], ids=ids)


as_dataset = from_archive #NOTE: just an alias


def argsort(x, axis=None):
    """generate a list of indices i that sort x, so x[i] is in increasing order

    Inputs:
      x: an array of shape (npts, dim) or (npts,)
      axis: int in [0,dim], the primary index of x to sort

    NOTE:
      if axis=None and dim > 1, the 0th index will be used as the primary
    """
    if not len(x):
        return []
    import numpy as np
    from mystic import isiterable
    if axis is None:
        if isiterable(x[0]): # multi-value data, defaulting to axis=0
            return np.argsort(x, axis=0).T[0].tolist() #XXX: better to error?
        else: # single-value data
            return np.argsort(x).tolist()
    if isiterable(x[0]): # multi-value data
        return np.argsort(np.array(x)[:,axis]).tolist()
    msg = "cannot get axis %s for single-valued data" % axis
    raise ValueError(msg)

def iargsort(x, axis=None):
    """generate a list of indices that, when sorted, sort x in increasing order

    Inputs:
      x: an array of shape (npts, dim) or (npts,)
      axis: int in [0,dim], the primary index of x to sort

    NOTE:
      if axis=None and dim > 1, the 0th index will be used as the primary
    """
    import numpy as np
    i = argsort(x, axis=axis)
    ii = np.empty_like(i)
    ii[i] = np.arange(len(i))
    return ii.tolist()

def counting(x):
    """generate a list of counting indices of len(x)'

    Inputs:
      x: an array of shape (npts, dim) or (npts,)
    """
    return list(range(len(x)))

# interface f(x,y,axis) for use with ids (in from_archive)
_argsort = lambda x,y,axis=None: argsort(x,axis=axis)
_argsort.__doc__ = argsort.__doc__
_iargsort = lambda x,y,axis=None: iargsort(x,axis=axis)
_iargsort.__doc__ = iargsort.__doc__
_counting = lambda x,y,axis=None: counting(x)
_counting.__doc__ = counting.__doc__

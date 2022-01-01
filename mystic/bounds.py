#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
cartesian bounds and measure bounds instances
"""
__all__ = ['Bounds','MeasureBounds']

from mystic.tools import flatten


class Bounds(object):

    def __init__(self, *bounds, **kwds):
        """create a bounds instance

    bounds is a tuple of (lower, upper) bound

    additionally, one can specify:
        - xlb: lower bound
        - xub: upper bound
        - n: repeat

    For example:
        >>> b = Bounds(7, 8, n=2)
        >>> b.xlb, b.xub
        (7, 8)
        >>> b()
        [(7, 8), (7, 8)]
        >>> b.lower
        [7, 7]
        >>> b.upper
        [8, 8]
        >>> 
        >>> c = Bounds((0,1),(3,4), n=2)
        >>> c()
        [(0, 3), (0, 3), (1, 4), (1, 4)]
        """
        self.n = kwds.pop('n', 1)
        self.wub = None # 1.0
        self.wlb = None # 1.0

        xlb = -float('inf')
        xub = float('inf')

        blen = len(bounds)
        if blen == 0:
            self.xub = kwds.pop('xub', xub)
            self.xlb = kwds.pop('xlb', xlb)
        elif blen == 1:
            self.xlb = bounds[0]
            self.xub = kwds.pop('xub', xub)
            if 'xlb' in kwds:
                msg = "__init__() got multiple values for argument 'xlb'"
                raise TypeError(msg)
        else:
            self.xlb,self.xub = bounds
            if 'xlb' in kwds:
                msg = "__init__() got multiple values for argument 'xlb'"
                raise TypeError(msg)
            if 'xub' in kwds:
                msg = "__init__() got multiple values for argument 'xub'"
                raise TypeError(msg)
        # single xub stretched to xlb
        if hasattr(self.xlb, '__len__') and not hasattr(self.xub, '__len__'):
            self.xub = len(self.xlb) * (self.xub,)
        # single xlb stretched to xub
        elif hasattr(self.xub, '__len__') and not hasattr(self.xlb, '__len__'):
            self.xlb = len(self.xub) * (self.xlb,)
        # single n stretched to xub
        if hasattr(self.xub, '__len__') and not hasattr(self.n, '__len__'):
            self.n = len(self.xub) * (self.n,)
        # single xlb,xub stretched to n
        elif hasattr(self.n, '__len__') and not hasattr(self.xub, '__len__'):
            self.xub = len(self.n) * (self.xub,)
            self.xlb = len(self.n) * (self.xlb,)

    def __lower(self):
        "get list of lower bounds"
        n = (self.n,) if not hasattr(self.n, '__len__') else self.n
        xlb = (self.xlb,) if not hasattr(self.xlb, '__len__') else self.xlb
        return list(flatten(i*[j] for i,j in zip(n,xlb)))

    def __upper(self):
        "get list of upper bounds"
        n = (self.n,) if not hasattr(self.n, '__len__') else self.n
        xub = (self.xub,) if not hasattr(self.xub, '__len__') else self.xub
        return list(flatten(i*[j] for i,j in zip(n,xub)))

    def __call__(self):
        "get list of tuples of (lower, upper) bounds"
        return list(zip(self.lower, self.upper))

    def __add__(self, other): #FIXME: create new Bounds instance
        if not isinstance(other, Bounds):
            return NotImplemented
        return self() + other()

    def __set_lower(self, lb):
        return NotImplemented

    def __set_upper(self, ub):
        return NotImplemented

    def __repr__(self):
        return "%s(%s, %s, n=%s)" % (self.__class__.__name__, self.xlb, self.xub, self.n)

    lower = property(__lower, __set_lower)
    upper = property(__upper, __set_upper)


class MeasureBounds(Bounds):

    def __init__(self, *bounds, **kwds):
        """create a measure bounds instance

    bounds is a tuple of (lower, upper) bound

    additionally, one can specify:
        - wlb: weight lower bound
        - wub: weight upper bound
        - xlb: lower bound
        - xub: upper bound
        - n: repeat

    For example:
        >>> b = MeasureBounds(7, 8, n=2)
        >>> b.wlb, b.wub
        (0, 1)
        >>> b.xlb, b.xub
        (7, 8)
        >>> b()
        [(0, 1), (0, 1), (7, 8), (7, 8)]
        >>> b.lower
        [0, 0, 7, 7]
        >>> b.upper
        [1, 1, 8, 8]
        >>> 
        >>> c = MeasureBounds((0,1),(4,5), n=1, wlb=(0,1), wub=(2,3))
        >>> c.lower
        [0, 0, 1, 1]
        >>> c.upper
        [2, 4, 3, 5]
        >>> c()
        [(0, 2), (0, 4), (1, 3), (1, 5)]
        >>> 
        >>> c = MeasureBounds((0,1),(4,5), n=2)
        >>> c()
        [(0, 1), (0, 1), (0, 4), (0, 4), (0, 1), (0, 1), (1, 5), (1, 5)]
        """
        super(MeasureBounds, self).__init__(*bounds, **kwds)
        self.wub = kwds.pop('wub', 1)
        self.wlb = kwds.pop('wlb', 0)
        # single wlb stretched to wub
        if hasattr(self.wub, '__len__') and not hasattr(self.wlb, '__len__'):
            self.wlb = len(self.wub) * (self.wlb,)
        elif hasattr(self.wlb, '__len__') and not hasattr(self.wub, '__len__'):
            self.wub = len(self.wlb) * (self.wub,)
        # single wlb stretched to xlb
        if hasattr(self.xlb, '__len__') and not hasattr(self.wlb, '__len__'):
            self.wlb = len(self.xlb) * (self.wlb,)
            self.wub = len(self.xub) * (self.wub,)
        elif hasattr(self.wlb, '__len__') and not hasattr(self.xlb, '__len__'):
            self.n = len(self.wlb) * (self.n,)
            self.xlb = len(self.wlb) * (self.xlb,)
            self.xub = len(self.wub) * (self.xub,)

    def __lower(self):
        n = (self.n,) if not hasattr(self.n, '__len__') else self.n
        wlb = (self.wlb,) if not hasattr(self.wlb, '__len__') else self.wlb
        xlb = (self.xlb,) if not hasattr(self.xlb, '__len__') else self.xlb
        return list(flatten(i*[j] + i*[k] for i,j,k in zip(n,wlb,xlb)))

    def __upper(self):
        n = (self.n,) if not hasattr(self.n, '__len__') else self.n
        wub = (self.wub,) if not hasattr(self.wub, '__len__') else self.wub
        xub = (self.xub,) if not hasattr(self.xub, '__len__') else self.xub
        return list(flatten(i*[j] + i*[k] for i,j,k in zip(n,wub,xub)))

    def __add__(self, other): #FIXME: create new MeasureBounds instance
        if not isinstance(other, Bounds):
            return NotImplemented
        return self() + other()

    def __set_lower(self, lb):
        return NotImplemented

    def __set_upper(self, ub):
        return NotImplemented

    def __repr__(self):
        if self.wlb == 0 and self.wub == 1:
            return super(MeasureBounds, self).__repr__()
        return "%s(%s, %s, n=%s, wlb=%s, wub=%s)" % (self.__class__.__name__, self.xlb, self.xub, self.n, self.wlb, self.wub)

    lower = property(__lower, __set_lower)
    upper = property(__upper, __set_upper)


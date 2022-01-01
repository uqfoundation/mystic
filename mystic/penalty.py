#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
penalty methods: methods used to convert a function into a penalty function

Suppose a given condition ``f(x)`` is satisfied when ``f(x) == 0.0``
for equality constraints, and ``f(x) <= 0.0`` for inequality constraints.
This condition ``f(x)`` can be used as the basis for a ``mystic.penalty``
function.

Examples:
    >>> def penalty_mean(x, target):
    ...   return mean(x) - target
    ... 
    >>> @quadratic_equality(condition=penalty_mean, kwds={'target':5.0})
    ... def penalty(x):
    ...   return 0.0
    ... 
    >>> penalty([1,2,3,4,5])
    400.0
    >>> penalty([3,4,5,6,7])
    7.8886090522101181e-29

References:
    1. http://en.wikipedia.org/wiki/Penalty_method
    2. "Applied Optimization with MATLAB Programming", by Venkataraman,
       Wiley, 2nd edition, 2009.
    3. http://www.srl.gatech.edu/education/ME6103/Penalty-Barrier.ppt
    4. "An Augmented Lagrange Multiplier Based Method for Mixed Integer
       Discrete Continuous Optimization and Its Applications to
       Mechanical Design", by Kannan and Kramer, 1994.
"""

from numpy import inf, log
def quadratic_equality(condition=lambda x:0., args=None, kwds=None, k=100, h=5):
    """apply a quadratic penalty if the given equality constraint is violated

penalty is p(x) = pk*f(x)**2, with pk = k*pow(h,n) and n=0
where f.iter() can be used to increment n = n+1

the condition f(x) is satisfied when f(x) == 0.0
    """
    if args is None: args=()
    if kwds is None: kwds={}
    _n = [0] # current penalty iteration
    _f = [lambda x:0.] # decorated function
    _y = [] # stored results
    def error(x):
        try:
            rms = condition(x, *args, **kwds)**2
        except ZeroDivisionError:
            return inf
        if hasattr(_f[0], 'error'): rms += _f[0].error(x)**2
        return rms**0.5
    def iter(i=None):
        if i is None: _n[0] += 1
        else: _n[0] = i
        if hasattr(_f[0], 'iter'): _f[0].iter(i)
        return
    def iteration():
        return _n[0]
    def store(x,i=None): #XXX: couple to 'iter' as {n:y} ?
        if hasattr(_f[0], 'store'): _f[0].store(x,i)
        return
    def stored(i=None): # can take a slice
        if i is None: return _y[:]
        try: return _y[i]
        except IndexError: return 0.0
    def clear():
        _n[0] = 0
        [_y.pop() for i in range(len(_y))]
        if hasattr(_f[0], 'clear'): _f[0].clear()
        return
    def dec(f):
        _f[0] = f
        def func(x, *argz, **kwdz):
            try:
                pf = condition(x, *args, **kwds)
            except ZeroDivisionError:
                return inf
            _k = k * pow(h,_n[0])
            return float(_k)*pf**2 + f(x, *argz, **kwdz)
        func.func = condition
        func.ptype = 'quadratic_equality'  
        func.iter = iter
        func.iteration = iteration
        func.store = store
        func.clear = clear
        func.stored = stored
        func.error = error
        return func
    return dec

def linear_equality(condition=lambda x:0., args=None, kwds=None, k=100, h=5):
    """apply a linear penalty if the given equality constraint is violated

penalty is p(x) = pk*abs(f(x)), with pk = k*pow(h,n) and n=0
where f.iter() can be used to increment n = n+1

the condition f(x) is satisfied when f(x) == 0.0
    """
    if args is None: args=()
    if kwds is None: kwds={}
    _n = [0] # current penalty iteration
    _f = [lambda x:0.] # decorated function
    _y = [] # stored results
    def error(x):
        try:
            rms = condition(x, *args, **kwds)**2
        except ZeroDivisionError:
            return inf
        if hasattr(_f[0], 'error'): rms += _f[0].error(x)**2
        return rms**0.5
    def iter(i=None):
        if i is None: _n[0] += 1
        else: _n[0] = i
        if hasattr(_f[0], 'iter'): _f[0].iter(i)
        return
    def iteration():
        return _n[0]
    def store(x,i=None): #XXX: couple to 'iter' as {n:y} ?
        if hasattr(_f[0], 'store'): _f[0].store(x,i)
        return
    def stored(i=None): # can take a slice
        if i is None: return _y[:]
        try: return _y[i]
        except IndexError: return 0.0
    def clear():
        _n[0] = 0
        [_y.pop() for i in range(len(_y))]
        if hasattr(_f[0], 'clear'): _f[0].clear()
        return
    def dec(f):
        _f[0] = f
        def func(x, *argz, **kwdz):
            try:
                pf = condition(x, *args, **kwds)
            except ZeroDivisionError:
                return inf
            _k = k * pow(h,_n[0])
            return float(_k)*abs(pf) + f(x, *argz, **kwdz)
        func.func = condition
        func.ptype = 'linear_equality'  
        func.iter = iter
        func.iteration = iteration
        func.store = store
        func.clear = clear
        func.stored = stored
        func.error = error
        return func
    return dec

def uniform_equality(condition=lambda x:0., args=None, kwds=None, k=inf, h=5):
    """apply a uniform penalty if the given equality constraint is violated

penalty is p(x) = pk, with pk = k*pow(h,n) and n=0
where f.iter() can be used to increment n = n+1

the condition f(x) is satisfied when f(x) == 0.0
    """ #XXX: this is a rather special case penalty
    if args is None: args=()
    if kwds is None: kwds={}
    _n = [0] # current penalty iteration
    _f = [lambda x:0.] # decorated function
    _y = [] # stored results
    def error(x):
        try:
            rms = condition(x, *args, **kwds)**2
        except ZeroDivisionError:
            return inf
        if hasattr(_f[0], 'error'): rms += _f[0].error(x)**2
        return rms**0.5
    def iter(i=None):
        if i is None: _n[0] += 1
        else: _n[0] = i
        if hasattr(_f[0], 'iter'): _f[0].iter(i)
        return
    def iteration():
        return _n[0]
    def store(x,i=None): #XXX: couple to 'iter' as {n:y} ?
        if hasattr(_f[0], 'store'): _f[0].store(x,i)
        return
    def stored(i=None): # can take a slice
        if i is None: return _y[:]
        try: return _y[i]
        except IndexError: return 0.0
    def clear():
        _n[0] = 0
        [_y.pop() for i in range(len(_y))]
        if hasattr(_f[0], 'clear'): _f[0].clear()
        return
    def dec(f):
        _f[0] = f
        def func(x, *argz, **kwdz):
            try:
                pf = condition(x, *args, **kwds)
            except ZeroDivisionError:
                return inf
            _k = float(k) * pow(h,_n[0]) if pf else 0.0
            return _k + f(x, *argz, **kwdz)
        func.func = condition
        func.ptype = 'uniform_equality'  
        func.iter = iter
        func.iteration = iteration
        func.store = store
        func.clear = clear
        func.stored = stored
        func.error = error
        return func
    return dec

def uniform_inequality(condition=lambda x:0., args=None, kwds=None, k=inf, h=5):
    """apply a uniform penalty if the given inequality constraint is violated

penalty is p(x) = pk, with pk = k*pow(h,n) and n=0
where f.iter() can be used to increment n = n+1

the condition f(x) is satisfied when f(x) <= 0.0
    """
    if args is None: args=()
    if kwds is None: kwds={}
    _n = [0] # current penalty iteration
    _f = [lambda x:0.] # decorated function
    _y = [] # stored results
    def error(x):
        try:
            rms = max(0., condition(x, *args, **kwds))**2
        except ZeroDivisionError:
            return inf
        if hasattr(_f[0], 'error'): rms += _f[0].error(x)**2
        return rms**0.5
    def iter(i=None):
        if i is None: _n[0] += 1
        else: _n[0] = i
        if hasattr(_f[0], 'iter'): _f[0].iter(i)
        return
    def iteration():
        return _n[0]
    def store(x,i=None): #XXX: couple to 'iter' as {n:y} ?
        if hasattr(_f[0], 'store'): _f[0].store(x,i)
        return
    def stored(i=None): # can take a slice
        if i is None: return _y[:]
        try: return _y[i]
        except IndexError: return 0.0
    def clear():
        _n[0] = 0
        [_y.pop() for i in range(len(_y))]
        if hasattr(_f[0], 'clear'): _f[0].clear()
        return
    def dec(f):
        _f[0] = f
        def func(x, *argz, **kwdz):
            try:
                pf = condition(x, *args, **kwds)
            except ZeroDivisionError:
                return inf
            _k = float(k) * pow(h,_n[0]) if pf > 0 else 0.0
            return _k + f(x, *argz, **kwdz)
        func.func = condition
        func.ptype = 'uniform_inequality'  
        func.iter = iter
        func.iteration = iteration
        func.store = store
        func.clear = clear
        func.stored = stored
        func.error = error
        return func
    return dec

def barrier_inequality(condition=lambda x:0., args=None, kwds=None, k=100, h=5):
    """apply a infinite barrier if the given inequality constraint is violated,
and a logarithmic penalty if the inequality constraint is satisfied

penalty is p(x) = inf if constraint is violated, otherwise
penalty is p(x) = -1/pk*log(-f(x)), with pk = 2k*pow(h,n) and n=0
where f.iter() can be used to increment n = n+1

the condition f(x) is satisfied when f(x) <= 0.0
    """
    if args is None: args=()
    if kwds is None: kwds={}
    _n = [0] # current penalty iteration
    _f = [lambda x:0.] # decorated function
    _y = [] # stored results
    def error(x):
        try:
            rms = max(0., condition(x, *args, **kwds))**2
        except ZeroDivisionError:
            return inf
        if hasattr(_f[0], 'error'): rms += _f[0].error(x)**2
        return rms**0.5
    def iter(i=None):
        if i is None: _n[0] += 1
        else: _n[0] = i
        if hasattr(_f[0], 'iter'): _f[0].iter(i)
        return
    def iteration():
        return _n[0]
    def store(x,i=None): #XXX: couple to 'iter' as {n:y} ?
        if hasattr(_f[0], 'store'): _f[0].store(x,i)
        return
    def stored(i=None): # can take a slice
        if i is None: return _y[:]
        try: return _y[i]
        except IndexError: return 0.0
    def clear():
        _n[0] = 0
        [_y.pop() for i in range(len(_y))]
        if hasattr(_f[0], 'clear'): _f[0].clear()
        return
    def dec(f):
        _f[0] = f
        def func(x, *argz, **kwdz):
            try:
                pf = condition(x, *args, **kwds)
            except ZeroDivisionError:
                return inf
            if pf > 0:  # inequality constraint is violated
                return inf
            # inequality constraint is satisfied
            _k = k * pow(h,_n[0])
            return -.5/_k*log(-pf) + f(x, *argz, **kwdz) #XXX: use 2*k or k=200?
        func.func = condition
        func.ptype = 'barrier_inequality'  
        func.iter = iter
        func.iteration = iteration
        func.store = store
        func.clear = clear
        func.stored = stored
        func.error = error
        return func
    return dec

def quadratic_inequality(condition=lambda x:0., args=None, kwds=None, k=100, h=5):
    """apply a quadratic penalty if the given inequality constraint is violated

penalty is p(x) = pk*f(x)**2, with pk = 2k*pow(h,n) and n=0
where f.iter() can be used to increment n = n+1

the condition f(x) is satisfied when f(x) <= 0.0
    """
    if args is None: args=()
    if kwds is None: kwds={}
    _n = [0] # current penalty iteration
    _f = [lambda x:0.] # decorated function
    _y = [] # stored results
    def error(x):
        try:
            rms = max(0., condition(x, *args, **kwds))**2
        except ZeroDivisionError:
            return inf
        if hasattr(_f[0], 'error'): rms += _f[0].error(x)**2
        return rms**0.5
    def iter(i=None):
        if i is None: _n[0] += 1
        else: _n[0] = i
        if hasattr(_f[0], 'iter'): _f[0].iter(i)
        return
    def iteration():
        return _n[0]
    def store(x,i=None): #XXX: couple to 'iter' as {n:y} ?
        if hasattr(_f[0], 'store'): _f[0].store(x,i)
        return
    def stored(i=None): # can take a slice
        if i is None: return _y[:]
        try: return _y[i]
        except IndexError: return 0.0
    def clear():
        _n[0] = 0
        [_y.pop() for i in range(len(_y))]
        if hasattr(_f[0], 'clear'): _f[0].clear()
        return
    def dec(f):
        _f[0] = f
        def func(x, *argz, **kwdz):
            try:
                pf = condition(x, *args, **kwds)
            except ZeroDivisionError:
                return inf
            _k = k * pow(h,_n[0])
            return float(2*_k)*max(0., pf)**2 + f(x, *argz, **kwdz) #XXX: use 2*k or k=200?
        func.func = condition
        func.ptype = 'quadratic_inequality'  
        func.iter = iter
        func.iteration = iteration
        func.store = store
        func.clear = clear
        func.stored = stored
        func.error = error
        return func
    return dec

def linear_inequality(condition=lambda x:0., args=None, kwds=None, k=100, h=5):
    """apply a linear penalty if the given inequality constraint is violated

penalty is p(x) = pk*abs(f(x)), with pk = 2k*pow(h,n) and n=0
where f.iter() can be used to increment n = n+1

the condition f(x) is satisfied when f(x) <= 0.0
    """
    if args is None: args=()
    if kwds is None: kwds={}
    _n = [0] # current penalty iteration
    _f = [lambda x:0.] # decorated function
    _y = [] # stored results
    def error(x):
        try:
            rms = max(0., condition(x, *args, **kwds))**2
        except ZeroDivisionError:
            return inf
        if hasattr(_f[0], 'error'): rms += _f[0].error(x)**2
        return rms**0.5
    def iter(i=None):
        if i is None: _n[0] += 1
        else: _n[0] = i
        if hasattr(_f[0], 'iter'): _f[0].iter(i)
        return
    def iteration():
        return _n[0]
    def store(x,i=None): #XXX: couple to 'iter' as {n:y} ?
        if hasattr(_f[0], 'store'): _f[0].store(x,i)
        return
    def stored(i=None): # can take a slice
        if i is None: return _y[:]
        try: return _y[i]
        except IndexError: return 0.0
    def clear():
        _n[0] = 0
        [_y.pop() for i in range(len(_y))]
        if hasattr(_f[0], 'clear'): _f[0].clear()
        return
    def dec(f):
        _f[0] = f
        def func(x, *argz, **kwdz):
            try:
                pf = condition(x, *args, **kwds)
            except ZeroDivisionError:
                return inf
            _k = k * pow(h,_n[0])
            return float(2*_k)*abs(max(0., pf)) + f(x, *argz, **kwdz) #XXX: use 2*k or k=200?
        func.func = condition
        func.ptype = 'linear_inequality'  
        func.iter = iter
        func.iteration = iteration
        func.store = store
        func.clear = clear
        func.stored = stored
        func.error = error
        return func
    return dec

def lagrange_inequality(condition=lambda x:0., args=None, kwds=None, k=20, h=5):
    """apply a quadratic penalty if the given inequality constraint is violated

penalty is p(x) = pk*mpf**2 + beta*mpf, with pk = k*pow(h,n) and n=0
also mpf = max(-beta/2k, f(x)) and lagrange multiplier beta = 2k*mpf
where f.iter() can be used to increment n = n+1

the condition f(x) is satisfied when f(x) <= 0.0
    """
    if args is None: args=()
    if kwds is None: kwds={}
    _n = [0] # current penalty iteration
    _f = [lambda x:0.] # decorated function
    _y = [] # stored results
    def error(x):
        try:
            rms = max(0., condition(x, *args, **kwds))**2
        except ZeroDivisionError:
            return inf
        if hasattr(_f[0], 'error'): rms += _f[0].error(x)**2
        return rms**0.5
    def iter(i=None):
        if i is None: _n[0] += 1
        else: _n[0] = i
        if hasattr(_f[0], 'iter'): _f[0].iter(i)
        return
    def iteration():
        return _n[0]
    def store(x,i=None): #XXX: couple to 'iter' as {n:y} ?
        try:
            y = condition(x, *args, **kwds) #XXX: don't re-evaluate
        except ZeroDivisionError:
            y = inf
        l = len(_y)
        if i is None: i = iteration()
        if i >= l: _y.extend([0.]*(i-l) + [y])
        else: _y[i] = y
        if hasattr(_f[0], 'store'): _f[0].store(x,i)
        return
    def stored(i=None): # can take a slice
        if i is None: return _y[:]
        try: return _y[i]
        except IndexError: return 0.0
    def clear():
        _n[0] = 0
        [_y.pop() for i in range(len(_y))]
        if hasattr(_f[0], 'clear'): _f[0].clear()
        return
    def dec(f):
        _f[0] = f
        def func(x, *argz, **kwdz):
            try:
                pf = condition(x, *args, **kwds)
            except ZeroDivisionError:
                return inf
            beta = 0.; _k = k
            for i in range(_n[0]):
                beta += 2.*_k*max(-beta/(2.*_k), stored(i))
                _k *= h
            mpf = max(-beta/(2.*_k), pf)
            return float(_k)*mpf**2 + beta*mpf + f(x, *argz, **kwdz)
        func.func = condition
        func.ptype = 'lagrange_inequality'  
        func.iter = iter
        func.iteration = iteration
        func.store = store
        func.clear = clear
        func.stored = stored
        func.error = error
        return func
    return dec

def lagrange_equality(condition=lambda x:0., args=None, kwds=None, k=20, h=5):
    """apply a quadratic penalty if the given equality constraint is violated

penalty is p(x) = pk*f(x)**2 + lam*f(x), with pk = k*pow(h,n) and n=0
also lagrange multiplier lam = 2k*f(x)
where f.iter() can be used to increment n = n+1

the condition f(x) is satisfied when f(x) = 0.0
    """
    if args is None: args=()
    if kwds is None: kwds={}
    _n = [0] # current penalty iteration
    _f = [lambda x:0.] # decorated function
    _y = [] # stored results
    def error(x):
        try:
            rms = condition(x, *args, **kwds)**2
        except ZeroDivisionError:
            return inf
        if hasattr(_f[0], 'error'): rms += _f[0].error(x)**2
        return rms**0.5
    def iter(i=None):
        if i is None: _n[0] += 1
        else: _n[0] = i
        if hasattr(_f[0], 'iter'): _f[0].iter(i)
        return
    def iteration():
        return _n[0]
    def store(x,i=None): #XXX: couple to 'iter' as {n:y} ?
        try:
            y = condition(x, *args, **kwds)
        except ZeroDivisionError:
            y = inf
        l = len(_y)
        if i is None: i = iteration()
        if i >= l: _y.extend([0.]*(i-l) + [y])
        else: _y[i] = y
        if hasattr(_f[0], 'store'): _f[0].store(x,i)
        return
    def stored(i=None): # can take a slice
        if i is None: return _y[:]
        try: return _y[i]
        except IndexError: return 0.0
    def clear():
        _n[0] = 0
        [_y.pop() for i in range(len(_y))]
        if hasattr(_f[0], 'clear'): _f[0].clear()
        return
    def dec(f):
        _f[0] = f
        def func(x, *argz, **kwdz):
            try:
                pf = condition(x, *args, **kwds)
            except ZeroDivisionError:
                return inf
            lam = 0.; _k = k
            for i in range(_n[0]):
                lam += 2.*_k*stored(i)
                _k *= h
            return float(_k)*pf**2 + lam*pf + f(x, *argz, **kwdz)
        func.func = condition
        func.ptype = 'lagrange_equality'  
        func.iter = iter
        func.iteration = iteration
        func.store = store
        func.clear = clear
        func.stored = stored
        func.error = error
        return func
    return dec


# EOF 

#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from mystic.constraints import *

def _symbolic():
    import mystic.symbolic as ms
    eqn = """x0 - 2*x1 > 0
             x3 - 3*x2 < 0"""
    eqn = ms.simplify(eqn, target=['x1','x3'])
    cons = ms.generate_constraint(ms.generate_solvers(eqn))
    return cons

def _mean():
    import mystic.constraints as mc
    return mc.with_mean(5.0)(lambda x: x)

def _parameter():
    def constraint(x):
        x[-1] = x[0]
        return x
    return constraint

def _compound():
    import mystic.constraints as mc
    return mc.or_(_parameter(), _mean(), _symbolic())

def test_vectorize(constraint, axis):
    # generate data
    import numpy as np
    data = np.random.randn(10,4)

    # build transform
    ineq = vectorize(constraint, axis)

    # test transform
    res = ineq(data)
    assert np.all(ineq(res) == res)

def test_vectorize_sklearn(constraint, axis):
    # get dataset
    from sklearn.datasets import load_iris
    iris = load_iris()

    # build transform
    ineq = vectorize(constraint, axis)
    from sklearn.preprocessing import FunctionTransformer
    t = FunctionTransformer(func=ineq, validate=False) #XXX: inverse?

    # test transform
    import numpy as np
    iris_ = t.fit(iris.data).transform(iris.data)
    assert np.all(t._transform(iris_) == iris_)


if __name__ == '__main__':
    try:
        import sklearn
        HAS_SKLEARN = True
    except ImportError:
        HAS_SKLEARN = False

    for cons in (_symbolic, _mean, _parameter, _compound):
        for axis in (0,1):
            test_vectorize(cons(), axis)
            if HAS_SKLEARN: test_vectorize_sklearn(cons(), axis)

# EOF

#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2020-2025 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
  Example applying mystic to sklearn

  Use a linear regression to fit sparse data generated from:
            f(x) = a*x3**3 + b*x2**2 + c*x1 + d*x0
            a,b,c,d = 0.661, -1.234, 2.983, -16.5571

  Where the following information is utilized: 
            f(x) is a polynomial of order=3
            3*b + c > -0.75
            4.5*b - d > 11.0
"""
import numpy as np
from sklearn import preprocessing as pre
from sklearn import linear_model as lin
from mystic.symbolic import generate_constraint, generate_solvers, simplify
from mystic.constraints import vectorize
from mystic import random_seed
random_seed(123)

# define a model
a,b,c,d = 0.661, -1.234, 2.983, -16.5571
def model(x):
  x0,x1,x2,x3 = x
  return a*x3**3 + b*x2**2 + c*x1 + d*x0

# generate some sparse training data
xtrain = np.random.uniform(0,100, size=(10,4))
target = model(xtrain.T).T
# generate sparse test data outside the training region
xtest = np.random.uniform(100,200, size=(10,4))
test = model(xtest.T).T

# define some model constraints
equations = """
3*b + c > -0.75
4.5*b - d > 11.0
"""
var = list('abcd')
equations = simplify(equations, variables=var)
cf = generate_constraint(generate_solvers(equations, variables=var))


if __name__ == '__main__':
    # build a kernel-transformed regressor
    ta = pre.FunctionTransformer(func=vectorize(cf, axis=1))
    tp = pre.PolynomialFeatures(degree=3)
    e = lin.LinearRegression()

    # build a pipeline, then train
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([('ta', ta), ('tp', tp), ('e', e)])
    pipe = pipe.fit(xtrain, target)

    # get training score and test score
    assert 1.0 == pipe.score(xtrain, target)
    assert 1 - pipe.score(xtest, test) <= 1e-2

# EOF

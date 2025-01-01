#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2024-2025 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
  Example applying mystic to keras with scikeras

  Use a linear regression to fit sparse data generated from:
            f(x) = a*x3**3 + b*x2**2 + c*x1 + d*x0
            a,b,c,d = 0.661, -1.234, 2.983, -16.5571

  Where the following information is utilized: 
            f(x) is a polynomial of order=3
            3*b + c > -0.75
            4.5*b - d > 11.0
"""
import warnings
from tensorflow import get_logger
get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", message="setting tensorflow random state")
import numpy as np
import scikeras
from scikeras.wrappers import KerasRegressor


class LinearRegression(KerasRegressor):
    def __init__(self, hidden_layer_sizes=(100,),
                 optimizer="adam", optimizer__learning_rate=0.001,
                 epochs=200, verbose=0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose

    def _keras_build_fn(self, compile_kwargs):
        import keras
        model = keras.Sequential()
        inp = keras.layers.Input(shape=(self.n_features_in_,))
        model.add(inp)
        for hidden_layer_size in self.hidden_layer_sizes:
            layer = keras.layers.Dense(hidden_layer_size, activation="relu")
            model.add(layer)
        out = keras.layers.Dense(1)
        model.add(out)
        model.compile(loss="mse", optimizer=compile_kwargs["optimizer"])
        return model


if __name__=='__main__':
    # build a kernel-transformed regressor
    from constrained_sklearn import *
    ta = pre.FunctionTransformer(func=vectorize(cf, axis=1))
    tp = pre.PolynomialFeatures(degree=3)
    e = LinearRegression()

    # build a pipeline, then train
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([('ta', ta), ('tp', tp), ('e', e)])
    pipe = pipe.fit(xtrain, target)

    # get training score and test score
    assert 1 - pipe.score(xtrain, target) <= 1e-4
    assert 1 - pipe.score(xtest, test) <= 1e-1

#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2024-2025 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
  Example applying mystic to torch with skorch

  Use a linear regression to fit sparse data generated from:
            f(x) = a*x3**3 + b*x2**2 + c*x1 + d*x0
            a,b,c,d = 0.661, -1.234, 2.983, -16.5571

  Where the following information is utilized: 
            f(x) is a polynomial of order=3
            3*b + c > -0.75
            4.5*b - d > 11.0
"""
import torch
import numpy as np
import torch.nn as nn
import skorch
from skorch import NeuralNetRegressor
from torch.optim.lr_scheduler import StepLR
from skorch.callbacks.lr_scheduler import LRScheduler
import torch.optim as optim


class LinearRegression(nn.Module):
    def __init__(self, n_in, n_h=100, n_layers=1, dropout=0.0):
        super().__init__()
        self.n_in = n_in
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_in, n_h))
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.ReLU())
        for _ in range(n_layers):
            self.layers.append(nn.Linear(n_h, n_h))
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(n_h, 1))

    def forward(self, x):
        y = x.view(-1, self.n_in)
        for _, layer in enumerate(self.layers):
            y = layer(y)
        return y


if __name__=='__main__':
    # convert training and test data to the format used by torch
    from constrained_sklearn import *
    target = target.reshape(target.shape[0], -1).astype(np.float32)
    xtrain = xtrain.astype(np.float32)
    test = test.reshape(test.shape[0], -1).astype(np.float32)
    xtest = xtest.astype(np.float32)

    # build a kernel-transformed regressor
    ta = pre.FunctionTransformer(func=vectorize(cf, axis=1))
    tp = pre.PolynomialFeatures(degree=3)
    n_in = tp.fit_transform(ta.fit_transform(xtrain)).shape[1]
    lr_policy = LRScheduler(StepLR, step_size=15, gamma=0.5)
    e = NeuralNetRegressor(LinearRegression, criterion=torch.nn.MSELoss,
                           module__n_in=n_in, module__n_h=300,
                           module__n_layers=3, module__dropout=0.2,
                           optimizer=torch.optim.Adam, optimizer__lr=.005,
                           max_epochs=200, callbacks=[lr_policy],
                           device='cpu', batch_size=64, verbose=0)

    # build a pipeline, then train
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([('ta', ta), ('tp', tp), ('e', e)])
    pipe = pipe.fit(xtrain, target)

    # get training score and test score
    assert 1 - pipe.score(xtrain, target) <= 1e-1
    assert 1 - pipe.score(xtest, test) <= 1e-1

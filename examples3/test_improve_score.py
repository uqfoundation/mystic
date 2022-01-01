#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
'''
demonstrate iterative improvement of R^2 score for an ANN model
'''
from ml import *


if __name__ == '__main__':

    # get model data
    from sklearn.datasets import load_iris
    data = load_iris()
    d = MLData(*traintest(data.data[:,:3], data.data[:,3], .2))

    # build estimator and perform initial fit
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    xfm = StandardScaler().fit(d.xtrain)
    mlp = MLPRegressor().fit(xfm.transform(d.xtrain), d.ytrain)
    e = Estimator(mlp, xfm)

    # check the initial score
    print('initial score')
    print(e.score(d.xtest, d.ytest))

    # iteratively improve the score
    print('improving...')
    ee = improve_score(e, d, verbose=True)
    print('checking improved score')
    print(ee.score(d.xtest, d.ytest))
    print('improving...')
    ee = improve_score(ee, d, verbose=True)
    print('checking improved score')
    print(ee.score(d.xtest, d.ytest))


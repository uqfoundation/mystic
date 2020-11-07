#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2019-2020 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
mahine learning containers and assorted tools
"""

import dataset as ds
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

class Estimator(object):
    "a container for a trained estimator and transform (not a pipeline)"
    def __init__(self, estimator, transform):
        """a container for a trained estimator and transform

    Input:
        estimator: a fitted sklearn estimator
        transform: a fitted sklearn transform
        """
        self.estimator = estimator
        self.transform = transform
        self.function = lambda *x: float(self.estimator.predict(self.transform.transform(np.array(x).reshape(1,-1))).reshape(-1))
    def __call__(self, *x):
        "f(*x) for x of xtest and predict on fitted estimator(transform(xtest))"
        import numpy as np
        return self.function(*x)
    def map(self, xtrain): # doesn't have n_samples_seen_
        "fit the transform on the training data"
        self.transform.fit(xtrain)
        return self
    def apply(self, xtrain): # has n_samples_seen_
        "apply the transform to the training data"
        return self.transform.transform(xtrain)
    def fit(self, xtrain, ytrain): # doesn't have n_iter_
        "fit the estimator on training data"
        self.estimator.fit(xtrain, ytrain)
        return self
    def predict(self, xtest): # has n_iter_
        "predict ytest using fitted estimator"
        return self.estimator.predict(xtest)
    def train(self, xtrain, ytrain):
        "fit the estimator and tranform on training data"
        xscale = self.map(xtrain).apply(xtrain)
        self.fit(xscale, ytrain)
        return self
    def test(self, xtest):
        "predict ytest using fitted estimator and transform"
        return self.predict(self.apply(xtest))


class MLData(object):
    "a container for training and test data"
    def __init__(self, xtrain, xtest, ytrain, ytest):
        """a container for training and test data

    Input:
        xtrain: x training data
        xtest: x test data
        ytrain: y training data
        ytest: y test data

    For example:
        >>> (x,xt,y,yt) = ([1,2], [3,4], [5,6], [7,8])
        >>> data = MLData(xtrain=x, xtest=xt, ytrain=y, ytest=yt)
        >>> data.xtrain is x
        True
        >>> (x,xt,y,yt) == data()
        True
        """
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest
    def __call__(self):
        "get (xtrain, xtest, ytrain, ytest)"
        return self.xtrain, self.xtest, self.ytrain, self.ytest


def plot_train_pred(x, t, y, xaxis=None, yaxis=None, mark=None, ax=None):
    """generate a plot of training and testing data, and model predictions

    Input:
        x: array[float] of x data
        t: array[float] of 'true' y data
        y: array[float] of 'predicted' y data
        xaxis: tuple[int] of x axes to plot [default: (0,1)]
        yaxis: int of y axis to plot [default: 0 (or None for 1D)]
        mark: string (or tuple) of symbol markers (or color and marker)
        ax: matplotlib subplot axis object

    NOTE:
        the default mark is 'ox', which is equvalent to ('ko','rx'),
        drawing a black filled circle for 't' and a red cross for 'y'.
    """
    import matplotlib.pyplot as plt
    if xaxis is None: xaxis = (0,1)
    if yaxis is None and t.ndim > 1: yaxis = 0
    x0,x1 = xaxis
    y0 = yaxis
    # draw plot of truth vs predicted
    if ax is None:
        figure = plt.figure()
        kwds = {'projection':'3d'}
        ax = figure.gca(**kwds)
    ax.autoscale(tight=True)
    if mark is None:
        s0,s1 = 'ko','rx'
    elif type(mark) in (tuple,list):
        s0,s1 = mark
    else:
        s0,s1 = 'k'+mark[0],'r'+mark[1]
    if t.ndim > 1:
        ax.plot(x[:,x0], x[:,x1], t[:,y0], s0, linewidth=2, markersize=4)
        ax.plot(x[:,x0], x[:,x1], y[:,y0], s1, linewidth=2, markersize=4)
    else:
        ax.plot(x[:,x0], x[:,x1], t, s0, linewidth=2, markersize=4)
        ax.plot(x[:,x0], x[:,x1], y, s1, linewidth=2, markersize=4)
    return ax


def score(i, estimator, xyt, delta=.0001, tries=10, verbose=False):
    """return (mlp instance, y predicted, score) for the given axis, i

    Inputs:
        i: int, the axis
        estimator: an already-fit estimator instance (mlp+ss)
        xyt: MLData instance (xtrain, xtest, ytrain, ytest)
        delta: float, the change in target, given target is satisfied
        tries: int, number of tries to exceed target before giving up

    Returns:
        tuple of (estimator, array of predicted y, R^2 score) 
    """
    est = estimator
    scor = _scor = -float('inf')
    target = 0 #XXX: expose target?
    while True:
        est,ypred,scor = _score(i, est, xyt, target, tries, verbose)
        if scor >= target:
            target = scor + delta
        if scor > _scor:
            _scor = scor
        else:
            if verbose: print('{0}: no improvement'.format(i))
            break
    return est, ypred, scor


def _score(i, estimator, xyt, target=0, tries=10, verbose=False):
    """return (mlp instance, y predicted, score) for the given axis, i

    Inputs:
        i: int, the axis
        estimator: an already-fit estimator instance (mlp+ss)
        xyt: MLData instance (xtrain, xtest, ytrain, ytest)
        target: float, the target score
        tries: int, number of tries to exceed target before giving up

    Returns:
        tuple of (estimator, array of predicted y, R^2 score) 
    """
    import warnings
    import sklearn.metrics as sm
    from sklearn.base import clone #NOTE: use to copy an estimator instance
    from sklearn.exceptions import ConvergenceWarning
    warnings.simplefilter('ignore', ConvergenceWarning)
    warnings.simplefilter('ignore', RuntimeWarning)
    est = estimator
    xscale,xstest,ytrain,ytest = xyt()
    if target is None: target = -float('inf')
    if not hasattr(est.estimator, 'n_iter_'): # is already fit
        with warnings.catch_warnings() as w:
            ypred = est.fit(xscale, ytrain if i is None else ytrain[:,i]).predict(xstest)
    else:
        ypred = est.predict(xstest)
    score = sm.r2_score(ytest if i is None else ytest[:,i], ypred)
    _ypred = _score = -float('inf')
    if verbose: print('{0}: score = {1}'.format(i,max(score,_score)))
    ntry = 1
    while score < target and ntry < tries:
        if score <= _score:
            ntry += 1
        else: # there is improvement
            _ypred, _score = ypred, score
            best_mlp = est.estimator # fitted
            ntry = 0
        est.estimator = clone(best_mlp) # unfitted
        with warnings.catch_warnings() as w:
            ypred = est.fit(xscale, ytrain if i is None else ytrain[:,i]).predict(xstest)
        score = sm.r2_score(ytest if i is None else ytest[:,i], ypred)
        if verbose: print('{0}: score = {1}'.format(i,max(score,_score)))
    if score >= target: return est,ypred,score
    # didn't come to a happy end, so return the best we have thus far
    if verbose: print('{0}: reached max tries'.format(i))
    if score <= _score:
        est.estimator = best_mlp
        return est,_ypred,_score
    return est,ypred,score


#XXX: may want to iteratively adjust to test data, instead of all at once
def traintest(x, y, test_size=None, random_state=None):
    """get train-test split of data from archive

    Inputs:
        x: x input data
        y: y input data
        test_size: float, % data to use for test [0,1] (default: duplicate)
        random_state: int, seed for splitting data

    Returns:
        xtrain,xtest,ytrain,ytest: arrays[float] of training and test data
    """
    # build train/test data
    xx = np.array(x)
    yy = np.array(y)
    if test_size is None:
        return xx,xx,yy,yy
    from sklearn.model_selection import train_test_split as split
    return split(xx, yy, test_size=test_size, random_state=random_state)



if __name__ == '__main__':

    # get access to data in archive
    from mystic.monitors import Monitor
    m = Monitor()
    m._x,m._y = ds.read_archive('demo')
    if not len(m._x):
        msg = "the 'demo' archive is empty."
        raise ValueError(msg)
    xtrain,xtest,ytrain,ytest = traintest(m._x, m._y, test_size=.2, random_state=42)

    import sklearn.preprocessing as pre
    import sklearn.neural_network as nn

    # build dicts of hyperparameters for ANN instance
    args,barg,carg = dict(alpha=0.0001, batch_size='auto', beta_1=0.9, beta_2=0.999, epsilon=1e-08, hidden_layer_sizes=(100,75,50,25), learning_rate_init=0.001, max_fun=15000, max_iter=1000, momentum=0.9, n_iter_no_change=5, power_t=0.5, tol=0.0001, validation_fraction=0.1), dict(early_stopping=False, nesterovs_momentum=True, shuffle=True), {} #dict(activation='relu', learning_rate='constant', solver='lbfgs')

    # modify MLP hyperparameters
    param = dict(alpha=0.0001, batch_size='auto', beta_1=0.9, beta_2=0.999, epsilon=1e-08, hidden_layer_sizes=(100,75,50,25), learning_rate_init=0.001, max_fun=15000, max_iter=1000, momentum=0.9, n_iter_no_change=5, power_t=0.5, tol=0.0001, validation_fraction=0.1)

    # get tuples of estimator functions, distances, and scores
    extra = dict(
        #verbose = True, # if True, print intermediate scores
        #pure = True, # if True, don't use test data until final scoring
        #delta = .01, # step size
        #tries = 5, # attempts to increase score with no improvement
        solver = 'lbfgs',
        learning_rate = 'constant',
        activation = 'relu'
    #NOTE: activation: 'identity','logistic','tanh','relu'
    #NOTE: learning_rate: 'constant','invscaling','adaptive'
    #NOTE: solver: 'lbfgs','sgd','adam'
    )

    args.update(param)
    carg.update(extra)
    mlp = nn.MLPRegressor(**args, **barg, **carg)

    ss = pre.StandardScaler()
    fpred = Estimator(mlp, ss)

    i = 0 #0,1,2,None
    fpred.train(xtrain, ytrain[:,i])

    print('as estimator...')
    ypred = fpred.test(xtest)

    import sklearn.metrics as sm
    score = sm.r2_score(ytest[:,i], ypred)
    print(score)

    print('as function...')
    score_ = sm.r2_score(ytest[:,i], [fpred(*x) for x in xtest])
    print(score_)

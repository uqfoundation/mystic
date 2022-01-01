#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @uqfoundation)
# Copyright (c) 2019-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
mahine learning containers and assorted tools
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

class Estimator(object): #XXX: use pipeline?
    "a container for a trained estimator and transform (not a pipeline)"
    def __init__(self, estimator, transform=None):
        """a container for a trained estimator and transform

    Input:
        estimator: a fitted sklearn estimator
        transform: a fitted sklearn transform

    For example:
        >>> from sklearn.datasets import load_iris
        >>> data = load_iris()
        >>> d = MLData(*traintest(data.data[:,:3], data.data[:,3], .2))
        >>> from sklearn.linear_model import LinearRegression
        >>> from sklearn.preprocessing import StandardScaler
        >>> xfm = StandardScaler().fit(d.xtrain)
        >>> lnr = LinearRegression().fit(xfm.transform(d.xtrain), d.ytrain)
        >>> e = Estimator(lnr, xfm)
        >>> [e(*i) for i in d.xtest[:2]]
        [1.7802194778123053, 1.3775908988859642]
        >>> e.test(d.xtest)[:2].tolist()
        [1.7802194778123053, 1.3775908988859642]
        >>> d.ytest[:2].tolist()
        [1.8, 1.3]
        >>> e.score(d.xtest, d.ytest)
        0.9440222526291645
        """
        self.estimator = estimator
        if transform is None:
            import sklearn.preprocessing as pre
            transform = pre.FunctionTransformer() #XXX: or StandardScaler ?
        self.transform = transform
        self.function = lambda *x: float(self.test(np.array(x).reshape(1,-1)).reshape(-1))
    def __call__(self, *x):
        "f(*x) for x of xtest and predict on fitted estimator(transform(xtest))"
        import numpy as np
        return self.function(*x)
    def map(self, xtrain): # doesn't have n_samples_seen_
        """fit the transform on the training data

    Inputs:
        xtrain: 2D array of shape (pts, nx) of raw input data

    Returns:
        instance where transform.fit(xtrain) has been called
        """
        self.transform.fit(xtrain)
        return self
    def apply(self, xtrain): # has n_samples_seen_
        """apply the transform to the training data

    Inputs:
        xtrain: 2D array of shape (pts, nx) of raw input data

    Returns:
        instance where transform.transform(xtrain) has been called
        """
        return self.transform.transform(xtrain)
    def fit(self, xtrain, ytrain): # doesn't have n_iter_
        """fit the estimator on training data

    Inputs:
        xtrain: 2D array of shape (pts, nx) of transformed input data
        ytrain: 1D array of shape (pts,) of raw output data

    Returns:
        instance where estimator.fit(xtrain, ytrain) has been called
        """
        self.estimator.fit(xtrain, ytrain)
        return self
    def predict(self, xtest): # has n_iter_
        """predict ytest using fitted estimator

    Inputs:
        xtest: 2D array of shape (pts, nx) of transformed input data

    Returns:
        predicted ytest from estimator.predict(xtest)
        """
        return self.estimator.predict(xtest)
    def train(self, xtrain, ytrain):
        """fit the estimator and tranform on training data

    Inputs:
        xtrain: 2D array of shape (pts, nx) of raw input data
        ytrain: 1D array of shape (pts,) of raw output data

    Returns:
        instance where fit(map(xtrain).apply(xtrain), ytrain) has been called
        """
        xscale = self.map(xtrain).apply(xtrain)
        self.fit(xscale, ytrain)
        return self
    def test(self, xtest):
        """predict ytest using fitted estimator and transform

    Inputs:
        xtest: 2D array of shape (pts, nx) of raw input data

    Returns:
        predicted ytest from predict(apply(xtest))
        """
        return self.predict(self.apply(xtest))
    def score(self, xtest, ytest):
        """score predicted versus ytest, using r2_score

    Inputs:
        xtest: 2D array of shape (pts, nx) of raw input data
        ytest: 1D array of shape (pts,) of raw output data

    Returns:
        score from r2_score(ytest, test(xtest))
        """
        import sklearn.metrics as sm
        return sm.r2_score(ytest, self.test(xtest))


class MLData(object):
    "a container for training and test data"
    def __init__(self, xtrain, xtest, ytrain, ytest=None):
        """a container for training and test data

    Input:
        xtrain: x training data
        xtest: x test data
        ytrain: y training data
        ytest: y test data

    For example:
        >>> x = [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2]]
        >>> y = [0, 1, 0, 0, 2]
        >>> d = MLData(*traintest(x, y, test_size=.4))
        >>> len(d.xtest)/len(x) == len(d.ytest)/len(y) == .4
        True
        >>> len(d.xtrain)/len(x) == len(d.ytrain)/len(y) == 1-.4
        True
        >>> a,b,c,z = d()
        >>> (a == d.xtrain).all() and (b == d.xtest).all() and (c == d.ytrain).all() and (z == d.ytest).all()
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
    """generate a 3D plot of input vs true, overlayed with input vs predicted

    Input:
        x: array[[float]] of 2D input data
        t: array[float] of 'true' 1D output data
        y: array[float] of 'predicted' 1D output data
        xaxis: tuple[int] of x axes to plot [default: (0,1)]
        yaxis: int of y axis to plot [default: 0 (or None for 1D)]
        mark: string (or tuple) of symbol markers (or color and marker)
        ax: matplotlib subplot axis object

    Returns:
        ax: matplotlib subplot axis object (with plotted data)

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
        ax = figure.axes[0] if figure.axes else plt.axes(**kwds)
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


def improve_score(estimator, xyt, delta=.0001, tries=10, **kwds):
    """iteratively improve R^2 score for an estimator with randomness in fit

    Inputs:
        estimator: a fitted Estimator instance
        xyt: a MLData instance (xtrain, xtest, ytrain, ytest)
        delta: float, the change in target, given target is satisfied
        tries: int, number of tries to exceed target before giving up

    Additional Inputs:
        verbose: bool, if True, be verbose with intermediate results
        scaled: bool, if True, don't call estimator.apply on xtrain,xtest
        axis: int, the index of y to estimate (all, by default)

    Returns:
        estimator with best score

    NOTE:
        can take xyt with either raw xtrain,xtest (will call estimator.map)
        or can take "transformed" xtrain,xtest (estimator.map has been called)
    """
    ax = kwds.pop('axis', None)
    return _improve_score(ax, estimator, xyt, delta=delta, tries=tries, **kwds)[0]


#XXX: use Pipeline or ouq.LearnedModel?
#XXX: refactor to use estimator.score? allow user-supplied scoring metric?
#XXX: add as an Estimator method?
def _improve_score(axis, estimator, xyt, delta=.0001, tries=10, **kwds):
    """iteratively improve R^2 score for an estimator with randomness in fit

    Inputs:
        axis: int, the index of y to estimate (all, by default)
        estimator: a fitted Estimator instance
        xyt: a MLData instance (xtrain, xtest, ytrain, ytest)
        delta: float, the change in target, given target is satisfied
        tries: int, number of tries to exceed target before giving up

    Additional Inputs:
        verbose: bool, if True, be verbose with intermediate results
        scaled: bool, if True, don't call estimator.apply on xtrain,xtest

    Returns:
        tuple of (best estimator, array of y predicted, best score) for axis

    NOTE:
        can take xyt with either raw xtrain,xtest (will call estimator.map)
        or can take "transformed" xtrain,xtest (estimator.map has been called)
    """
    verbose = kwds.get('verbose', False)
    scaled = kwds.get('scaled', False)
    est = estimator
    scor = _scor = -float('inf')
    target = 0 #XXX: expose target?
    if not scaled:
        if not hasattr(est.transform, 'n_samples_seen'): # is already fit
            import warnings
            with warnings.catch_warnings() as w:
                est.map(xyt.xtrain)
        xyt = MLData(est.apply(xyt.xtrain), est.apply(xyt.xtest), xyt.ytrain, xyt.ytest) #NOTE: thus 'while' fits transformed x
    while True:
        est,ypred,scor = _rescore(axis, est, xyt, target, tries, verbose)
        if scor >= target:
            target = scor + delta
        if scor > _scor:
            _scor = scor
        else:
            if verbose:
                if axis is None:
                    print('no improvement'.format(axis))
                else:
                    print('{0}: no improvement'.format(axis))
            break
    return est, ypred, scor


def _rescore(axis, estimator, xyt, target=0, tries=10, verbose=False):
    """iteratively improve R^2 score for an estimator with randomness in fit

    Inputs:
        axis: int, the index of y to estimate (all, by default)
        estimator: a fitted Estimator instance
        xyt: a MLData instance (xtrain, xtest, ytrain, ytest)
        target: float, the target score
        tries: int, number of tries to exceed target before giving up
        verbose: bool, if True, be verbose with intermediate results

    Returns:
        tuple of (best estimator, array of y predicted, best score) for axis

    NOTE:
        assumes xtrain,xtest in xyt have been "transformed" by estimator.apply
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
            ypred = est.fit(xscale, ytrain if axis is None else ytrain[:,axis]).predict(xstest)
    else:
        ypred = est.predict(xstest)
    # check if there's something to test against
    if ytest is None:
        if verbose:
            print('no truth velues to compare')
        return est,ypred,-float('inf')
    # we have ytest, let's try to make improvements...
    score = sm.r2_score(ytest if axis is None else ytest[:,axis], ypred)
    _ypred = _score = -float('inf')
    if verbose:
        if axis is None:
            print('score = {1}'.format(axis,max(score,_score)))
        else:
            print('{0}: score = {1}'.format(axis,max(score,_score)))
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
            ypred = est.fit(xscale, ytrain if axis is None else ytrain[:,axis]).predict(xstest)
        score = sm.r2_score(ytest if axis is None else ytest[:,axis], ypred)
        if verbose:
            if axis is None:
                print('score = {1}'.format(axis,max(score,_score)))
            else:
                print('{0}: score = {1}'.format(axis,max(score,_score)))
    if score >= target: return est,ypred,score
    # didn't come to a happy end, so return the best we have thus far
    if verbose:
        if axis is None:
            print('reached max tries'.format(axis))
        else:
            print('{0}: reached max tries'.format(axis))
    if score <= _score:
        est.estimator = best_mlp
        return est,_ypred,_score
    return est,ypred,score


#XXX: may want to iteratively adjust to test data, instead of all at once
def traintest(x, y, test_size=None, random_state=None):
    """get train-test split of data from archive

    Inputs:
        x: 2D input data array of shape (pts, nx)
        y: 1D output data array of shape (pts,)
        test_size: float, % data to use for test [0,1] (default: copy [1,1])
        random_state: int, seed for splitting data

    Returns:
        xtrain,xtest,ytrain,ytest: arrays of training and test data

    For example:
      >>> x = [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2]]
      >>> y = [0, 1, 0, 0, 2]
      >>> 
      >>> xx,xt,yy,yt = traintest(x, y, test_size=.4)
      >>> len(xt)/len(x) == len(yt)/len(y) == .4
      True
      >>> len(xx)/len(x) == len(yy)/len(y) == 1-.4
      True
      >>> 
      >>> xx,xt,yy,yt = traintest(x, y)
      >>> len(yy) == len(yt) == len(y)
      True
      >>> len(xx) == len(xt) == len(x)
      True
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
    import dataset as ds
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
    args = dict(hidden_layer_sizes=(100,75,50,25), max_iter=1000, n_iter_no_change=5)
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

    args.update(extra)
    mlp = nn.MLPRegressor(**args)

    ss = pre.StandardScaler()
    fpred = Estimator(mlp, ss)

    i = 0 #0,1,2,None
    fpred.train(xtrain, ytrain[:,i])

    print('as estimator...')
    score = fpred.score(xtest, ytest[:,i])
    print(score)

    print('as function...')
    import sklearn.metrics as sm
    score_ = sm.r2_score(ytest[:,i], [fpred(*x) for x in xtest])
    print(score_)

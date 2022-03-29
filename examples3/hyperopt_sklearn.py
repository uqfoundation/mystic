#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2020-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"Hyperparameter optimization"

from sklearn.svm import SVR
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as tts
iris = load_iris()
X_train, X_test, y_train, y_test = tts(iris.data, iris.target, random_state=1)


def objective(x):
    estimator = SVR(kernel='linear', C=x[0])
    estimator.fit(X_train, y_train)
    return 1-estimator.score(X_test, y_test)


bounds = [(0,10)]
# for the given split, the solution is, roughly:
xs = [0.01213213]
ys =  0.08483955


if __name__ == '__main__':

    from mystic.solvers import diffev2
    from mystic.math import almostEqual
    from mystic.monitors import VerboseMonitor
    mon = VerboseMonitor(10)

    result = diffev2(objective, x0=bounds, bounds=bounds, npop=40, ftol=1e-8, gtol=75, disp=False, full_output=True, itermon=mon)

    assert almostEqual(result[0], xs, rel=1e-2)
    assert almostEqual(result[1], ys, rel=1e-2)


# EOF

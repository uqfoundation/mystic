#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2022-2024 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
"""
plot the sampled data (and cumulative average) from xrd_design*.py
"""

def plot_db(archive):
    'plot cumulative average for sampled data in archive'
    import mystic.cache as mc
    db = mc.archive.read(archive, type=mc.archive.file_archive)
    xx = list(db.values())
    xtype = type(xx[-1])
    multi = hasattr(xtype, '__len__')
    import numpy as np
    xs = np.array(xx)
    var = xtype(np.nanvar(xs, axis=0)) if multi else np.nanvar(xs)
    if xs.ndim == 1: xs = np.atleast_2d(xs).T
    ys = np.arange(1, xs.shape[0]+1)
    y_ = ys - np.cumsum(np.isnan(xs), axis=0).T # nan-adjusted
    xs = (np.nancumsum(xs, axis=0).T/y_).T.tolist()
    del y_; ave = xtype(xs[-1]) if multi else xs[-1][0]
    print("%s: len = %s, ave = %s, var = %s" % (archive, len(db), ave, var))
    import matplotlib.pyplot as plt
    if type(ave) is tuple:
        fig, axs = plt.subplots(len(ave))
        for i,ax in enumerate(axs):
            ax.plot(np.array(xs).T[i])
            ax.plot(np.array(xx).T[i], 'x')
    else:
        fig, ax = plt.subplots(1)
        ax.plot(np.array(xs).reshape(-1))
        ax.plot(np.array(xx).reshape(-1), 'x')
    plt.title(archive)
    plt.show()


if __name__ == '__main__':
    import os
    archives = ('ave.db', 'min.db', 'max.db')

    for archive in archives:
        a = None
        if os.path.exists(archive):
            try:
                plot_db(archive)
            except Exception:
                pass


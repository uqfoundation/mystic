#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2021-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

from ouq_models import WrapModel
from mystic.models import rosen

# generate a sampled dataset for the model
model = WrapModel('rosen', rosen, cached=True)
bounds = [(0,10),(0,10)]
data = model.sample(bounds, pts=-8)

# plot the model and sampled data
import mystic as my
m = my.monitors.Monitor()
m._x, m._y = data.coords, data.values
b = '0:10, 0:10'
my.model_plotter(rosen, m, depth=True, bounds=b, dots=True, join=False)

# read the archive of sampled data
import mystic.cache as mc
a = mc.archive.read('rosen')
print(a.archive.state)

# convert archive to a dataframe
import klepto as kl
p = kl.archives._to_frame(a)
print(p.head())

# plot the data
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.axes[0] if fig.axes else plt.axes(projection='3d')
p.reset_index(inplace=True)
p.columns = 'x','y','rosen'
ax.scatter(p['x'], p['y'], p['rosen'])
plt.show()

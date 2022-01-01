#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Alta Fang (altafang @caltech and alta @princeton)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE
__doc__ = _doc = """
This is part of Pohlheim's "GEATbx" test suite in [1], with function
definitions drawn from [1], [2], [3], [4], [5], [6], and [7].

References:
    1. Pohlheim, H. "GEATbx: Genetic and Evolutionary Algorithm Toolbox
       for use with MATLAB", Version 3.80, 2006. http://www.geatbx.com/docu
    2. Schwefel, H.-P. "Numerical Optimization of Computer Models",
       John Wiley and Sons, Chichester UK, 1981.
    3. Ackley, D.H. "A Connectionist Machine for Genetic Hillclimbing",
       Kluwer Academic Publishers, Boston MA, 1987.
    4. Michalewicz, Z. "Genetic Algorithms + Data Structures = Evolution
       Programs", Springer-Verlag, Berlin, Heidelberg, New York, 1992.
    5. Branin, F.K. "A Widely Convergent Method for Finding Multiple
       Solutions of Simultaneous Nonlinear Equations", IBM J. Res. Develop.,
       504-522, Sept 1972.
    6. Easom, E.E. "A Survey of Global Optimization Techniques", M. Eng.
       Thesis, U. Louisville, Louisville KY, 1990.
    7. Goldstein, A.A. and Price, I.F. "On Descent from Local Minima",
       Math. Comput., (25) 115, 1971.
"""
#   8. Dixon, L.C.W. and Szego, G.P. "The Optimization Problem: An
#      Introduction", in "Toward Global Optimization II", North Holland,
#      New York, 1978.
from .abstract_model import AbstractFunction

from math import sin, cos, sqrt, pi, exp

class Schwefel(AbstractFunction):
    __doc__ = \
    """a Schwefel's function generator

Schwefel's function [1,2] has alternating rows of peaks and valleys,
with the global minimum near the edge of the bounded parameter
space.  This funciton can be misleading for optimizers as the next
best local minima are near the other corners of the bounded
parameter space.  The intensity of the peaks and valleys increases
as one moves away from the origin.

The generated function f(x) is identical to function (7)
of [1], where len(x) >= 0.
    """ + _doc
    def __init__(self, ndim=2): # is n-dimensional
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates Schwefel's function for a list of coeffs

f(x) = \sum_(i=0)^(N-1) -x_i * \sin(\sqrt(\abs(x_i)))

Where abs(x_i) <= 500.

Inspect with mystic_model_plotter using::
    mystic.models.schwefel -b "-500:500:10, -500:500:10" -d

The minimum is f(x)=-(N+1)*418.98288727243374 at x_i=420.9687465 for all i"""
        return sum(-c * sin(sqrt(abs(c))) for c in coeffs)

    minimizers = None #XXX: there are many local minima
    pass


class HyperEllipsoid(AbstractFunction):
    __doc__ = \
    """a Pohlheim's rotated hyper-ellipsoid function generator

Pohlheim's rotated hyper-ellipsoid function [1] is continuous,
convex, and unimodal. The global minimum is located at the
center of the N-dimensional axis parallel hyper-ellipsoid.

The generated function f(x) is identical to function (1b)
of [1], where len(x) >= 0.
    """ + _doc
    def __init__(self, ndim=2): # is n-dimensional
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates the rotated hyper-ellipsoid function for a list of coeffs

f(x) = \sum_(i=0)^(N-1) (\sum_(j=0)^(i) x_j)^2

Inspect with mystic_model_plotter using::
    mystic.models.ellipsoid -b "-5:5:.1, -5:5:.1" -d

The minimum is f(x)=0.0 at x_i=0.0 for all i"""
        ncoeffs = range(len(coeffs))
        return sum(sum(coeffs[0:i+1])**2 for i in ncoeffs)

    minimizers = [0.]
    pass


class Rastrigin(AbstractFunction):
    __doc__ = \
    """a Rastrigin's function generator

Rastrigin's function [1] is essentially De Jong's sphere with the
addition of cosine modulation to produce several regularly distributed
local minima. The global minimum is at the origin.

The generated function f(x) is identical to function (6)
of [1], where len(x) >= 0.
    """ + _doc
    def __init__(self, ndim=2): # is n-dimensional
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates Rastrigin's function for a list of coeffs

f(x) = 10 * N + \sum_(i=0)^(N-1) (x_(i)^2 - 10 * \cos(2 * \pi * x_(i)))

Inspect with mystic_model_plotter using::
    mystic.models.rastrigin -b "-5:5:.1, -5:5:.1" -d

The minimum is f(x)=0.0 at x_i=0.0 for all i"""
        return 10.*len(coeffs) + sum(c*c - 10.*cos(2*pi*c) for c in coeffs)

    minimizers = None #XXX: there are many local minima
    pass


class DifferentPowers(AbstractFunction):
    __doc__ = \
    """a Pohlheim's sum of different powers function generator

Pohlheim's sum of different powers function [1] is unimodal, and
similar to the hyper-ellipsoid and De Jong's sphere. The global
minimum is at the origin, at the center of a broad basin.

The generated function f(x) is identical to function (9)
of [1], where len(x) >= 0.
    """ + _doc
    def __init__(self, ndim=2): # is n-dimensional
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates the sum of different powers function for a list of coeffs

f(x) = \sum_(i=0)^(N-1) \abs(x_(i))^(i+2)

Inspect with mystic_model_plotter using::
    mystic.models.powers -b "-5:5:.1, -5:5:.1" -d

The minimum is f(x)=0.0 at x_i=0.0 for all i"""
        ncoeffs = range(len(coeffs))
        return sum(abs(coeffs[i])**(i+2) for i in ncoeffs)

    minimizers = [0.]
    pass


class Ackley(AbstractFunction):
    __doc__ = \
    """an Ackley's path function generator

At a very coarse level, Ackley's path function [1,3] is a slightly
parabolic plane, with a sharp cone-shaped depression at the origin.
The global minimum is found at the origin. There are several local
minima evenly distributed across the function surface, where the
surface modulates similarly to cosine.

The generated function f(x) is identical to function (10)
of [1], where len(x) >= 0.
    """ + _doc
    def __init__(self, ndim=2): # is n-dimensional
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates Ackley's function for a list of coeffs

f(x) = f_0(x) + f_1(x)

Where:
f_0(x) = -20 * \exp(-0.2 * \sqrt(1/N * \sum_(i=0)^(N-1) x_(i)^(2)))
and:
f_1(x) = -\exp(1/N * \sum_(i=0)^(N-1) cos(2 * \pi * x_(i))) + 20 + \exp(1)

Inspect with mystic_model_plotter using::
    mystic.models.ackley -b "-10:10:.1, -10:10:.1" -d

The minimum is f(x)=0.0 at x_i=0.0 for all i"""
        a=20.; b=0.2; d=2*pi
        n = len(coeffs)
        f0 = -a * exp(-b * sqrt(sum(c*c for c in coeffs)/n))
        f1 = -exp(sum(cos(d*c) for c in coeffs)/n) + a + exp(1)
        return f0 + f1

    minimizers = None #XXX: there are many local minima
    pass


class Michalewicz(AbstractFunction):
    __doc__ = \
    """a Michalewicz's function generator

Michalewicz's function [1,4] in general evaluates to zero. However,
there are long narrow channels that create local minima. At the
intersection of the channels, the function additionally has sharp
dips -- one of which is the global minimum.

The generated function f(x) is identical to function (12)
of [1], where len(x) >= 0.
    """ + _doc
    def __init__(self, ndim=5): # is n-dimensional
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates Michalewicz's function for a list of coeffs

f(x) = -\sum_(i=0)^(N-1) \sin(x_i) * (\sin((i+1) * (x_i)^(2) / \pi))^(20)

Inspect with mystic_model_plotter using::
    mystic.models.michal -b "0:3.14:.1, 0:3.14:.1, 1.28500168, 1.92305311, 1.72047194" -d

For x=(2.20289811, 1.57078059, 1.28500168, 1.92305311, 1.72047194, ...)[:N]
and c=(-0.801303, -1.0, -0.959092, -0.896699, -1.030564, ...)[:N],
the minimum is f(x)=sum(c) for all x_i=(0,pi)"""
        m = 20 # the larger the number the narrower the channel
        n = range(len(coeffs))
        r = -sum(sin(coeffs[i])*(sin((i+1)*coeffs[i]**2/pi))**m for i in n)
        return r #XXX: constrain x_i=(0,pi) ?

    minimizers = None #XXX: there are many local minima
    pass


class Branins(AbstractFunction):
    __doc__ = \
    """a Branins's rcos function generator

Branins's function [1,5] is very similar to Rosenbrock's saddle function.
However unlike Rosenbrock's saddle, Branins's function has a degenerate
global minimum.

The generated function f(x) is identical to function (13)
of [1], where len(x) >= 0.
    """ + _doc
    def __init__(self, ndim=2): # is 2-dimensional
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates Branins's function for a list of coeffs

f(x) = f_0(x) + f_1(x)

Where: 
f_0(x) = a * (x_1 - b * x_(0)^(2) + c * x_0 - d)^2
and
f_1(x) = e * (1 - f) * \cos(x_0) + e
and
a=1, b=5.1/(4*pi^2), c=5/pi, d=6, e=10, f=1/(8*pi)

Inspect with mystic_model_plotter using::
    mystic.models.branins -b "-10:20:.1, -5:25:.1" -d -x 1

The minimum is f(x)=0.397887 at x=((2 +/- (2*i)+1)*pi, 2.275 + 10*i*(i+1)/2)
for all i"""
        x1,x2 = coeffs
        a=1.; b=5.1/(4*pi**2); c=5./pi; d=6.; e=10.; f=1./(8*pi)
        f0 = a * (x2 - b * x1**2 + c * x1 - d)**2
        f1 = e * (1 - f) * cos(x1) + e
        return f0 + f1

    minimizers = None #XXX: there are degenerate global minima
    pass


class Easom(AbstractFunction):
    __doc__ = \
    """a Easom's function generator

Easom's function [1,6] is a unimodal function that exaluates
to zero everywhere except in the region around the global
minimum. The global minimum is at the bottom of a sharp well.

The generated function f(x) is identical to function (14)
of [1], where len(x) >= 0.
    """ + _doc
    def __init__(self, ndim=2): # is 2-dimensional
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates Easom's function for a list of coeffs

f(x) = -\cos(x_0) * \cos(x_1) *  \exp(-((x_0-\pi)^2+(x_1-\pi)^2))

Inspect with mystic_model_plotter using::
    mystic.models.easom -b "-5:10:.1, -5:10:.1" -d

The minimum is f(x)=-1.0 at x=(pi,pi)"""
        x0,x1 = coeffs
        return -cos(x0) * cos(x1) *  exp(-((x0-pi)**2+(x1-pi)**2))

    minimizers = [(pi,pi)]
    pass


class GoldsteinPrice(AbstractFunction):
    __doc__ = \
    """a Goldstein-Price's function generator

Goldstein-Price's function [1,7] provides a function with several
peaks surrounding a roughly flat valley. There are a few shallow
scorings across the valley, where the global minimum is found at
the intersection of the deepest of the two scorings. Local minima
occur at other intersections of scorings.

The generated function f(x) is identical to function (15)
of [1], where len(x) >= 0.
    """ + _doc
    def __init__(self, ndim=2): # is 2-dimensional
        AbstractFunction.__init__(self, ndim=ndim)
        return

    def function(self,coeffs):
        r"""evaluates Goldstein-Price's function for a list of coeffs

f(x) = (1 + (x_0 + x_1 + 1)^2 * f_0(x)) * (30 + (2*x_0 - 3*x_1)^2 * f_1(x))

Where:
f_0(x) = 19 - 14*x_0 + 3*x_(0)^2 - 14*x_1 + 6*x_(0)*x_(1) + 3*x_(1)^2
and
f_1(x) = 18 - 32*x_0 + 12*x_(0)^2 + 48*x_1 - 36*x_(0)*x_(1) + 27*x_(1)^2

Inspect with mystic_model_plotter using::
    mystic.models.goldstein -b "-5:5:.1, -5:5:.1" -d -x 1

The minimum is f(x)=3.0 at x=(0,-1)"""
        x0,x1 = coeffs
        f0 = 19 - 14*x0 + 3*x0**2 - 14*x1 + 6*x0*x1 + 3*x1**2
        f1 = 18 - 32*x0 + 12*x0**2 + 48*x1 - 36*x0*x1 + 27*x1**2
        return (1 + (x0 + x1 + 1)**2 * f0) * (30 + (2*x0 - 3*x1)**2 * f1)

    minimizers = [(0,-1),(-.6,-.4),(1.8,.2)]
    pass


#class SixHumpCamel(AbstractFunction):
#    __doc__ = \
#    """a Dixon's six-hump camelback function generator
#
#Dixon's six-hump camelback function [1,8] is a saddle
#with two small local minima on the back of the saddle.
#
#The generated function f(x) is identical to function (16)
#of [1], where len(x) >= 0.
#    """ + _doc
#    def __init__(self, ndim=2): # is 2-dimensional
#        AbstractFunction.__init__(self, ndim=ndim)
#        return
#
#    def function(self,coeffs):
#        r"""evaluates Dixon's six-hump camelback function for a list of coeffs
#
#f(x) = f_0(x) + f_1(x)
#
#Where:
#f_0(x) = (4 - 2.1*x_(0)^2 + x_(0)^(4/3)) * x_(0)^2
#and
#f_1(x) = x_(0)*x_(1) + (4*x_(1)^2 - 4) * x_(1)^2
#
#Inspect with mystic_model_plotter using::
#    mystic.models.camelback6 -b "-3:3:.1, -3:3:.1" -d -x 1
#
#The minimum is f(x)=-1.031313 at x=(-0.08834623*j,0.71256506*j) for j = -1,1"""
#        x0,x1 = coeffs
#        f0 = (4 - 2.1*x0**2 + (x0**4)**(1/3.)) * x0**2
#        f1 = x0*x1 + (-4 + 4*x1**2) * x1**2
#        return f0 + f1
#
#    minimizers = None #FIXME: global minimum = -inf
#    pass


# cleanup
del _doc

# prepared instances
schwefel = Schwefel().function
ellipsoid = HyperEllipsoid().function
rastrigin = Rastrigin().function
powers = DifferentPowers().function
ackley = Ackley().function
michal = Michalewicz().function
branins = Branins().function
easom = Easom().function
goldstein = GoldsteinPrice().function
#camelback6 = SixHumpCamel().function


# End of file

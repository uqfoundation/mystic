#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

import os
import sys
# drop support for older python
if sys.version_info < (3, 9):
    unsupported = 'Versions of Python before 3.9 are not supported'
    raise ValueError(unsupported)

# get distribution meta info
here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(here)
from version import (__version__, __author__, __contact__ as AUTHOR_EMAIL,
                     get_license_text, get_readme_as_rst, write_info_file)
LICENSE = get_license_text(os.path.join(here, 'LICENSE'))
README = get_readme_as_rst(os.path.join(here, 'README.md'))

# write meta info file
write_info_file(here, 'mystic', doc=README, license=LICENSE,
                version=__version__, author=__author__)
del here, get_license_text, get_readme_as_rst, write_info_file

# check if setuptools is available
try:
    from setuptools import setup
    from setuptools.dist import Distribution
    has_setuptools = True
except ImportError:
    from distutils.core import setup
    Distribution = object
    has_setuptools = False

# build the 'setup' call
setup_kwds = dict(
    name='mystic',
    version=__version__,
    description='constrained nonlinear optimization for scientific machine learning, UQ, and AI',
    long_description = README.strip(),
    author = __author__,
    author_email = AUTHOR_EMAIL,
    maintainer = __author__,
    maintainer_email = AUTHOR_EMAIL,
    license = 'BSD-3-Clause',
    platforms = ['Linux', 'Windows', 'Mac'],
    url = 'https://github.com/uqfoundation/mystic',
    download_url = 'https://pypi.org/project/mystic/#files',
    project_urls = {
        'Documentation':'http://mystic.rtfd.io',
        'Source Code':'https://github.com/uqfoundation/mystic',
        'Bug Tracker':'https://github.com/uqfoundation/mystic/issues',
    },
    python_requires = '>=3.9',
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ],
    packages = ['mystic','mystic.models','mystic.math','mystic.cache',
                'mystic.tests'],
    package_dir = {'mystic':'mystic','mystic.models':'mystic/models',
                   'mystic.math':'mystic/math','mystic.cache':'mystic/cache',
                   'mystic.tests':'mystic/tests'},
    scripts=['scripts/mystic_log_reader',
             'scripts/mystic_log_converter',
             'scripts/mystic_model_plotter',
             'scripts/mystic_collapse_plotter',
             'scripts/support_convergence',
             'scripts/support_hypercube',
             'scripts/support_hypercube_measures',
             'scripts/support_hypercube_scenario'],
)

# force python-, abi-, and platform-specific naming of bdist_wheel
class BinaryDistribution(Distribution):
    """Distribution which forces a binary package with platform name"""
    def has_ext_modules(foo):
        return True

# define dependencies
dill_version = 'dill>=0.4.0'
klepto_version = 'klepto>=0.2.7'
pathos_version = 'pathos>=0.3.4'
pyina_version = 'pyina>=0.3.1'
cython_version = 'cython>=0.29.30' #XXX: required to build numpy from source
numpy_version = 'numpy>=1.0'
sympy_version = 'sympy>=0.6.7'#, <0.7.4'
scipy_version = 'scipy>=0.6.0'
mpmath_version = 'mpmath>=0.19'
matplotlib_version = 'matplotlib>=0.91' #XXX: kiwisolver-1.3.0
# add dependencies
depend = [dill_version, klepto_version, numpy_version, sympy_version, mpmath_version]
extras = {'math': [scipy_version], 'parallel': [pathos_version, pyina_version], 'plotting': [matplotlib_version]}
# update setup kwds
if has_setuptools:
    setup_kwds.update(
        zip_safe=False,
        # distclass=BinaryDistribution,
        install_requires=depend,
        extras_require=extras,
    )

# call setup
setup(**setup_kwds)

# if dependencies are missing, print a warning
try:
    import numpy
    import sympy
    import mpmath
    import dill
    import klepto
    #import cython
    #import scipy
    #import matplotlib #XXX: has issues being zip_safe
    #import pathos
    #import pyina
except ImportError:
    print("\n***********************************************************")
    print("WARNING: One of the following dependencies is unresolved:")
    print("    %s" % numpy_version)
    print("    %s" % sympy_version)
    print("    %s" % mpmath_version)
    print("    %s" % dill_version)
    print("    %s" % klepto_version)
    #print("    %s" % cython_version)
    print("    %s (optional)" % scipy_version)
    print("    %s (optional)" % matplotlib_version)
    print("    %s (optional)" % pathos_version)
    print("    %s (optional)" % pyina_version)
    print("***********************************************************\n")

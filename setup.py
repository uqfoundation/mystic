#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 1997-2016 California Institute of Technology.
# Copyright (c) 2016-2022 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/mystic/blob/master/LICENSE

import os
import sys
# drop support for older python
if sys.version_info < (3, 7):
    unsupported = 'Versions of Python before 3.7 are not supported'
    raise ValueError(unsupported)

# get distribution meta info
here = os.path.abspath(os.path.dirname(__file__))
meta_fh = open(os.path.join(here, 'mystic/__init__.py'))
try:
    meta = {}
    for line in meta_fh:
        if line.startswith('__version__'):
            VERSION = line.split()[-1].strip("'").strip('"')
            break
    meta['VERSION'] = VERSION
    for line in meta_fh:
        if line.startswith('__author__'):
            AUTHOR = line.split(' = ')[-1].strip().strip("'").strip('"')
            break
    meta['AUTHOR'] = AUTHOR
    LONG_DOC = ""
    DOC_STOP = "FAKE_STOP_12345"
    for line in meta_fh:
        if LONG_DOC:
            if line.startswith(DOC_STOP):
                LONG_DOC = LONG_DOC.strip().strip("'").strip('"').lstrip()
                break
            else:
                LONG_DOC += line
        elif line.startswith('__doc__'):
            DOC_STOP = line.split(' = ')[-1]
            LONG_DOC = "\n"
    meta['LONG_DOC'] = LONG_DOC
finally:
    meta_fh.close()

# get version numbers, long_description, etc
AUTHOR = meta['AUTHOR']
VERSION = meta['VERSION']
LONG_DOC = meta['LONG_DOC'] #FIXME: near-duplicate of README.md
#LICENSE = meta['LICENSE'] #FIXME: duplicate of LICENSE
AUTHOR_EMAIL = 'mmckerns@uqfoundation.org'

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
    version=VERSION,
    description='highly-constrained non-convex optimization and uncertainty quantification',
    long_description = LONG_DOC,
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    maintainer = AUTHOR,
    maintainer_email = AUTHOR_EMAIL,
    license = '3-clause BSD',
    platforms = ['Linux', 'Windows', 'Mac'],
    url = 'https://github.com/uqfoundation/mystic',
    download_url = 'https://pypi.org/project/mystic/#files',
    project_urls = {
        'Documentation':'http://mystic.rtfd.io',
        'Source Code':'https://github.com/uqfoundation/mystic',
        'Bug Tracker':'https://github.com/uqfoundation/mystic/issues',
    },
    python_requires = '>=3.7',
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
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
dill_version = 'dill>=0.3.5.1'
klepto_version = 'klepto>=0.2.2'
pathos_version = 'pathos>=0.2.9'
pyina_version = 'pyina>=0.2.6'
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


if __name__=='__main__':
    pass

# end of file

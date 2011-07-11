#!/usr/bin/env python
#
# Michael McKerns
# mmckerns@caltech.edu

# check if easy_install is available
try:
#   import __force_distutils__ #XXX: uncomment to force use of distutills
    from setuptools import setup
    has_setuptools = True
except ImportError:
    from distutils.core import setup
    has_setuptools = False

# build the 'setup' call
setup_code = """
setup(name='mystic',
      version='0.2a2.dev',
      description='a simple interactive inversion analysis framework',
      author = 'Mike McKerns',
      maintainer = 'Mike McKerns',
      maintainer_email = 'mmckerns@caltech.edu',
      license = 'BSD',
      platforms = ['any'],
      url = 'http://www.cacr.caltech.edu/~mmckerns',
      classifiers = ('Intended Audience :: Developers',
                     'Programming Language :: Python',
                     'Topic :: Physics Programming'),

      packages = ['mystic','mystic.models','mystic.math'],
      package_dir = {'mystic':'mystic','mystic.models':'models',
                     'mystic.math':'_math'},
"""

# add dependencies
numpy_version = '>=1.0'
sympy_version = '>=0.6.7'
scipy_version = '>=0.6.0'
matplotlib_version = '>=0.91'
if has_setuptools:
    setup_code += """
      install_requires = ('numpy%s', 'sympy%s'),
""" % (numpy_version, sympy_version)

# close 'setup' call
setup_code += """    
      zip_safe=True,
"""

# add the scripts, and close 'setup' call
setup_code += """
    scripts=['scripts/mystic_log_reader.py',
             'scripts/support_convergence.py',
             'scripts/support_hypercube.py',
             'scripts/support_hypercube_measures.py'])
"""

# exec the 'setup' code
exec setup_code

# if dependencies are missing, print a warning
try:
    import numpy
    import sympy
    #import scipy
    #import matplotlib #XXX: has issues being zip_safe
except ImportError:
    print "\n***********************************************************"
    print "WARNING: One of the following dependencies is unresolved:"
    print "    numpy %s" % numpy_version
    print "    sympy %s" % sympy_version
    print "    scipy %s (optional)" % scipy_version
    print "    matplotlib %s (optional)" % matplotlib_version
    print "***********************************************************\n"


if __name__=='__main__':
    pass

# end of file

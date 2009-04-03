# -*- Makefile -*-


PROJECT = mystic
PACKAGE = examples_other

PROJ_TIDY += 
PROJ_CLEAN =

#--------------------------------------------------------------------------

# all: export
all: clean

release: tidy
	cvs release .

update: clean
	cvs update .

#--------------------------------------------------------------------------
#
# export

#EXPORT_PYTHON_MODULES = \
EXPORT_BINS = \
    CubeSection.py \
    test_argv.py \
#   test_gplot.py \
#   test_scem.py \
#   test_smo1.py \
#   test_svc1.py \
#   test_svc2.py \
#   test_svr1.py \
#   test_svr2.py \
#   sam_corana.py \
#   sam_corana2.py \
#   sam_rosenbrock.py \
#   sam_cg_rosenbrock.py \
#   sam_zimmermann.py \
#   sam_cg_zimmermann.py \
#   sam_mogi.py \
#   sam_circle_matlab.py \
#   qld_circle_dual.py \


# export:: export-package-python-modules
export:: export-binaries release-binaries


# End of file

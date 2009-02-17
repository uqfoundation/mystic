# -*- Makefile -*-
#

PROJECT = mystic
PACKAGE = examples

PROJ_TIDY += *.log *.png *.dat
PROJ_CLEAN =

#--------------------------------------------------------------------------

#all: export
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
    test_br8.py \
#   gplot_test_ffit.py \
    test_ffit.py \
    test_ffit2.py \
    test_ffitB.py \
    scipy_ffit.py \
    test_fosc3d.py \
    test_griewangk.py \
    test_dejong3.py \
    test_dejong4.py \
    test_dejong5.py \
    test_corana.py \
#   sam_corana.py \
#   sam_corana2.py \
    mpl_corana.py \
    test_rosenbrock.py \
    test_rosenbrock2.py \
#   sam_rosenbrock.py \
    cg_rosenbrock.py \
#   sam_cg_rosenbrock.py \
    test_zimmermann.py \
#   sam_zimmermann.py \
#   sam_cg_zimmermann.py \
    test_lorentzian.py \
    test_lorentzian2.py \
#   sam_mogi.py \
    test_mogi.py \
    test_mogi2.py \
#   test_mogi3.py \
#   test_mogi4.py \
    test_mogi_anneal.py \
#   test_mogi_leastsq.py \
#   sam_circle_matlab.py \
    test_circle.py \
#   qld_circle_dual.py \
#   metropolis.py \
#   test_twistedgaussian.py \
#   test_twistedgaussian2.py \
#   test_twistedgaussian3.py \
    test_wavy.py \
#   derun.py \
#   example.py \
    example_getCost.py \
#   forward_model.py \
    forward_mogi.py \


# export:: export-package-python-modules
export:: export-binaries release-binaries


# End of file

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
    TEST_ffitPP_b.py \
    batchgrid_example06.py \
    cg_rosenbrock.py \
##  derun.py \
##  dummy.py \
    example01.py \
    example02.py \
    example03.py \
    example04.py \
    example05.py \
    example06.py \
    example07.py \
    example08.py \
    example09.py \
    example10.py \
    example11.py \
    example12.py \
    ezmap_desolve.py \
    ezmap_desolve_rosen.py \
    forward_model.py \
#   gplot_test_ffit.py \
##  metropolis.py \
    mpl_corana.py \
    raw_chebyshev8.py \
    raw_chebyshev8b.py \
    raw_rosen.py \
    rosetta_parabola.py \
    rosetta_mogi.py \
    scattershot_example06.py \
    test_br8.py \
    test_circle.py \
    test_corana.py \
    test_dejong3.py \
    test_dejong4.py \
    test_dejong5.py \
    test_ffit.py \
    test_ffit2.py \
    test_ffitB.py \
    test_ffitC.py \
    test_ffitD.py \
    test_fosc3d.py \
    test_getCost.py \
    test_griewangk.py \
    test_lorentzian.py \
    test_lorentzian2.py \
    test_mogi.py \
    test_mogi2.py \
##  test_mogi3.py \
##  test_mogi4.py \
    test_mogi_anneal.py \
    test_mogi_leastsq.py \
    test_rosenbrock.py \
    test_rosenbrock2.py \
    test_rosenbrock3.py \
    test_rosenbrock3b.py \
##  test_twistedgaussian.py \
##  test_twistedgaussian2.py \
##  test_twistedgaussian3.py \
    test_wavy.py \
    test_zimmermann.py \


# export:: export-package-python-modules
export:: export-binaries release-binaries


# End of file

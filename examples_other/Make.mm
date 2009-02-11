# -*- Makefile -*-


PROJECT = mystic
PACKAGE = examples_other

PROJ_TIDY += 
PROJ_CLEAN =

#--------------------------------------------------------------------------

# all: export
all: clean

update: $(BUILD_DIRS)

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


# export:: export-package-python-modules
export:: export-binaries release-binaries


# End of file

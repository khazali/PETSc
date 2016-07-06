CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/impls/picell/examples/tutorials/
EXAMPLESC       = ex1.c
EXAMPLESF       =
MANSEC          = DM

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

NP=1

UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
# do something Linux-y
CPP_FLAGS = -I${ADVISOR_XE_2016_DIR}/include -DX2_HAVE_ADVISOR -DH5PART -DPARALLEL_IO -DMPICH_IGNORE_CXX_SEEK -I/usr/common/software/h5part/1.6.6/include/
H5PART_LIB = -Wl,-u__errno_location -L/usr/common/software/h5part/1.6.6/lib -lH5Part
endif
ifeq ($(UNAME), Darwin)
CPP_FLAGS = -I/Users/markadams/Codes/H5Part-1.6.6/src -I. -DH5PART -DPARALLEL_IO -DMPICH_IGNORE_CXX_SEEK
H5PART_LIB = /Users/markadams/Codes/H5Part-1.6.6/src/.libs/libH5Part.a
endif

ex1: ex1.o chkopts
	-${CLINKER} -o ex1.${PETSC_ARCH} ex1.o ${PETSC_DM_LIB} ${H5PART_LIB}
	${RM} -f ex1.o

runex1:
	-${MPIEXEC} -n ${NP} ./ex1.${PETSC_ARCH} -x2_dm_type p8est -petscspace_poly_tensor -petscspace_order 1 -x2_dm_forest_initial_refinement 1 -x2_dm_refine 1 -numMajor 8 -npart_flux_tube 1000 -mstep 20 -debug 2 -snes_monitor -snes_converged_reason -collisionPeriod 1 -run_type boxtorus -petscpartitioner_type parmetis -dm_view hdf5:sol.h5 -vec_view hdf5:sol.h5::append
	-${PETSC_DIR}/bin/petsc_gen_xdmf.py sol.h5
# -iter_vertex_file ITER-51vertex-quad-tri.txt -x2_dm_view -x2_dm_type p8est
# -run_type boxtorus  -bsp_chunksize 3000 -gtool "advixe-cl -collect survey:2,3"
# -nphi_particles 2 -x2_dm_view vtk:tokamac.vtu

val:
	-${MPIEXEC} -n ${NP} valgrind --dsymutil=yes --leak-check=yes --gen-suppressions=no --num-callers=20 --error-limit=no ./ex1.${PETSC_ARCH} -debug 2 -petscspace_poly_tensor -petscspace_order 1 -x2_dm_forest_initial_refinement 1 -x2_dm_refine 1 -numMajor 8 -npart_flux_tube 100 -mstep 3 -debug 2 -snes_monitor -snes_converged_reason -collisionPeriod 1 -petscpartitioner_type parmetis -run_type boxtorus
# --vgdb-error=0 -x2_dm_type p8est

debug:
	-@gdb --args ./ex1.${PETSC_ARCH} -petscspace_poly_tensor -petscspace_order 1 -x2_dm_forest_initial_refinement 1 -x2_dm_refine 1 -numMajor 8 -npart_flux_tube 10 -mstep 2 -debug 2 -snes_monitor -snes_converged_reason -collisionPeriod 1 -start_in_debugger -run_type boxtorus -petscpartitioner_type parmetis



include ${PETSC_DIR}/lib/petsc/conf/test

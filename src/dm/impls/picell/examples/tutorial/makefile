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

CPP_FLAGS = -I/Users/markadams/Codes/H5Part-1.6.6/src -I. -DH5PART -DPARALLEL_IO -DMPICH_IGNORE_CXX_SEEK

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_DM_LIB} /Users/markadams/Codes/H5Part-1.6.6/src/.libs/libH5Part.a
#	${RM} -f ex1.o

runex1:
	-echo ${MPIEXEC}
	-${MPIEXEC} -n 2 ./ex1 -x2_dm_type p8est -x2_dm_view vtk:tokamac.vtu -x2_dm_forest_initial_refinement 3 -numMajor 16 -npart_cell 10 -mstep 10 -debug 1 -snes_monitor -snes_converged_reason -collisionPeriod 5
# -iter_vertex_file ITER-51vertex-quad-tri.txt
# -run_type boxtorus  -bsp_chuncksize 3000
# -nphi_particles 2

val: ex1
	-${MPIEXEC} -n 1 valgrind --dsymutil=yes --leak-check=yes --gen-suppressions=no --num-callers=20 --error-limit=no ./ex1 -x2_dm_type p8est -x2_dm_view vtk:tokamac.vtu -x2_dm_forest_initial_refinement 5 -numMajor 4 -npart_cell 10 -debug 1 -mstep 2 -collisionPeriod 1

debug:
	-@gdb --args ./ex1 -x2_dm_type p8est -x2_dm_view vtk:tokamac.vtu -x2_dm_forest_initial_refinement 1 -numMajor 4 -use_electrons true -npart_cell 10 -mstep 2 -debug 1 -snes_monitor -snes_converged_reason -collisionPeriod 1



include ${PETSC_DIR}/lib/petsc/conf/test

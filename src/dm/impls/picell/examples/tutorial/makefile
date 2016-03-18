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
	-${MPIEXEC} -n 1 ./ex1 -torus_dm_type p8est -torus_dm_view vtk:tokamac.vtu -torus_dm_forest_initial_refinement 1 -numMajor 12 -use_electrons true -npart_cell 100 -mstep 20 -debug 1 -rMinor 200 -rMinorParticles 180 -snes_monitor -snes_converged_reason -bsp_chuncksize 3000 -collisionPeriod 1
# -nphi_particles 2

val: ex1
	-${MPIEXEC} -n 1 valgrind --dsymutil=yes --leak-check=yes --gen-suppressions=no --num-callers=20 --error-limit=no ./ex1 -torus_dm_type p8est -torus_dm_view vtk:tokamac.vtu -torus_dm_forest_initial_refinement 1 -numMajor 12 -use_electrons true -npart_cell 10 -mstep 1 -snes_converged_reason

debug:
	-@gdb --args ./ex1 -torus_dm_type p8est -torus_dm_view vtk:tokamac.vtu -torus_dm_forest_initial_refinement 1 -numMajor 12 -use_electrons true -npart_cell 1000 -mstep 20 -debug 2 -rMinor 200 -rMinorParticles 180 -snes_monitor -snes_converged_reason -bsp_chuncksize 3000



include ${PETSC_DIR}/lib/petsc/conf/test

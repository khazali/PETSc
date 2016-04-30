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

#CPP_FLAGS = -I/Users/markadams/Codes/H5Part-1.6.6/src -I. -DH5PART -DPARALLEL_IO -DMPICH_IGNORE_CXX_SEEK
#H5PART_LIB = /Users/markadams/Codes/H5Part-1.6.6/src/.libs/libH5Part.a

ex1: ex1.o chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_DM_LIB} ${H5PART_LIB}
#	${RM} -f ex1.o

runex1:
	-echo ${MPIEXEC}
	-${MPIEXEC} -n 1 ./ex1 -x2_dm_type p8est -x2_dm_view vtk:tokamac.vtu -petscspace_poly_tensor -petscspace_order 1 -x2_dm_forest_initial_refinement 1 -numMajor 4 -npart_flux_tube 10 -mstep 2 -debug 1 -snes_monitor -snes_converged_reason -collisionPeriod 1
# -iter_vertex_file ITER-51vertex-quad-tri.txt
# -run_type boxtorus  -bsp_chuncksize 3000
# -nphi_particles 2

val: ex1
	-${MPIEXEC} -n 1 valgrind --dsymutil=yes --leak-check=yes --gen-suppressions=no --num-callers=20 --error-limit=no ./ex1 -x2_dm_type p8est -x2_dm_view vtk:tokamac.vtu -petscspace_poly_tensor -petscspace_order 1 -x2_dm_forest_initial_refinement 1 -numMajor 4 -npart_flux_tube 10 -mstep 2 -debug 1 -snes_monitor -snes_converged_reason -collisionPeriod 1

debug:
#	-${MPIEXEC} -n 2 ./ex1  -x2_dm_type p8est -x2_dm_view vtk:tokamac.vtu -petscspace_poly_tensor -petscspace_order 1 -x2_dm_forest_initial_refinement 2 -numMajor 4 -npart_flux_tube 10 -mstep 2 -debug 1 -snes_monitor -snes_converged_reason -collisionPeriod 1 -start_in_debugger
	-@gdb --args ./ex1 -x2_dm_type p8est -x2_dm_view vtk:tokamac.vtu -petscspace_poly_tensor -petscspace_order 1 -x2_dm_forest_initial_refinement 2 -numMajor 4 -npart_flux_tube 10 -mstep 2 -debug 1 -snes_monitor -snes_converged_reason -collisionPeriod 1 -start_in_debugger



include ${PETSC_DIR}/lib/petsc/conf/test

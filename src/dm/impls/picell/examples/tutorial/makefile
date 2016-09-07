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
endif
ifeq ($(UNAME), Darwin)
EX1_LIB = /Users/markadams/Codes/H5Part-1.6.6/src/.libs/libH5Part.a
endif
CPP_FLAGS = ${EX1_INC}

ex1: ex1.o chkopts
	-${CLINKER} -o ex1.${PETSC_ARCH} ex1.o ${PETSC_DM_LIB} ${EX1_LIB}
	${RM} -f ex1.o

ex2: ex2.o chkopts
	-${CLINKER} -o ex2.${PETSC_ARCH} ex2.o ${PETSC_DM_LIB} ${EX1_LIB}
	${RM} -f ex2.o

runex1:
	-${MPIEXEC} -n ${NP} ./ex1.${PETSC_ARCH} -petscspace_poly_tensor -petscspace_order 1 -x2_dm_refine 2 -numMajor 4 -num_particles_proc 1000 -mstep 2 -debug 2 -snes_monitor -ksp_monitor -snes_converged_reason -run_type boxtorus -x2_dm_view hdf5:ex1.h5 -x2_vec_view hdf5:ex1.h5::append -collision_period 1 -max_vpar 30. -use_bsp 1  -petscpartitioner_type simple
	-${PETSC_DIR}/bin/petsc_gen_xdmf.py ex1.h5

runex2:
	-${MPIEXEC} -n ${NP}  ./ex2.${PETSC_ARCH} -petscspace_poly_tensor -petscspace_order 1 -x2_dm_refine 3 -num_particles_proc 10 -mstep 2 -debug 3 -snes_monitor -ksp_monitor -snes_converged_reason -x2_dm_view hdf5:ex2.h5 -x2_vec_view hdf5:ex2.h5::append -collision_period 1 -max_vpar 1.1 -snes_type ksponly -use_bsp 1 -petscpartitioner_type simple -solver_np 2,2,1 -ft_np 2,2,1 -dom_lo -1,-1,-2 -dom_hi 1,1,2 -b0 -.1,.1,1 -use_electrons false
	-${PETSC_DIR}/bin/petsc_gen_xdmf.py ex2.h5
# -solver_np 2,1,1 -ft_np 1,2,1 -dom_lo -2,-2,-2 -dom_hi 2,2,2 -b0 .2,.2,1.5 
runex1_p4:
	-${MPIEXEC} -n ${NP} ./ex1.${PETSC_ARCH} -x2_dm_type p8est -petscspace_poly_tensor -petscspace_order 1 -x2_dm_forest_initial_refinement 2 -numMajor 3 -x2_dm_refine 0 -num_particles_proc 1000 -mstep 2 -debug 3 -snes_monitor -ksp_monitor -snes_converged_reason -run_type iter -x2_dm_view hdf5:ex1.h5 -x2_vec_view hdf5:ex1.h5::append -use_bsp 0
	-${PETSC_DIR}/bin/petsc_gen_xdmf.py ex1.h5  -petscpartitioner_type simple

# -iter_vertex_file ITER-51vertex-quad-tri.txt -x2_dm_view -x2_dm_type p8est
# -run_type boxtorus -gtool "advixe-cl -collect survey:2,3"
# -nphi_particles 2 -x2_dm_view vtk:tokamac.vtu -ft_np 1,1,1 -chunksize 165536

val:
	-${MPIEXEC} -n ${NP} valgrind --dsymutil=yes --leak-check=yes --gen-suppressions=no --num-callers=20 --error-limit=no ./ex1.${PETSC_ARCH} -petscspace_poly_tensor -petscspace_order 1 -x2_dm_refine 2 -numMajor 4 -num_particles_proc 100 -mstep 2 -debug 2 -snes_monitor -ksp_monitor -snes_converged_reason -run_type boxtorus -x2_dm_view hdf5:ex1.h5 -x2_vec_view hdf5:ex1.h5::append -collision_period 1 -max_vpar 30.  -petscpartitioner_type simple
# --vgdb-error=0 -x2_dm_type p8est

debug:
	-@gdb --args ./ex1.${PETSC_ARCH} -petscspace_poly_tensor -petscspace_order 1 -x2_dm_forest_initial_refinement 0 -x2_dm_refine 2 -numMajor 8 -num_particles_proc 20 -mstep 2 -debug 2 -snes_monitor -snes_converged_reason -start_in_debugger -run_type boxtorus -petscpartitioner_type simple

include ${PETSC_DIR}/lib/petsc/conf/test

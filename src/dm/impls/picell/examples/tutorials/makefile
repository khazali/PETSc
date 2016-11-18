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

ex3: ex3.o chkopts
	-${CLINKER} -o ex3.${PETSC_ARCH} ex3.o ${PETSC_DM_LIB} ${EX1_LIB}
	${RM} -f ex3.o

runex1_p4est:
	-${MPIEXEC} -n ${NP} ./ex1.${PETSC_ARCH} -x2_dm_type p8est -petscspace_poly_tensor -petscspace_order 1 -mstep 2 -debug 2 -snes_monitor -ksp_monitor -snes_converged_reason -domain_type torus -max_vpar 30. -use_v_update true -use_bsp 1  -petscpartitioner_type simple -collision_period 10 -x2_dm_view hdf5:ex1.h5 -x2_vec_view hdf5:ex1.h5::append -x2_dm_forest_initial_refinement 2 -num_phi_cells 3 -num_particles_proc 10
	-${PETSC_DIR}/bin/petsc_gen_xdmf.py ex1.h5

runex2:
	-${MPIEXEC} -n ${NP} ./ex2.${PETSC_ARCH} -petscspace_poly_tensor -petscspace_order 1 -mstep 1 -debug 3 -snes_monitor -ksp_monitor -snes_converged_reason -max_vpar .05 -snes_type ksponly -ksp_max_it 5 -use_bsp 1 -petscpartitioner_type simple -use_electrons false -use_v_update true -x2_dm_view hdf5:ex2.h5 -x2_vec_view hdf5:ex2.h5::append -x2_dm_refine 3 -num_particles_proc 1000 -periodic_domain false -use_mms true -plot false -domain_lo 0,0,0
	-${PETSC_DIR}/bin/petsc_gen_xdmf.py ex2.h5
# -solver_np 2,2,1 -ft_np 2,2,1 -domain_lo -2,-2,-2 -domain_hi 2,2,2 -b0 .2,.2,1.5

runex1:
	-${MPIEXEC} -n ${NP} ./ex1.${PETSC_ARCH} -petscspace_poly_tensor -petscspace_order 2 -mstep 2 -debug 3 -snes_monitor -ksp_monitor -snes_converged_reason -domain_type torus -x2_dm_view hdf5:ex1.h5 -x2_vec_view hdf5:ex1.h5::append -x2_dm_refine 1 -use_bsp 1 -num_particles_proc 100 -num_phi_cells 8
	-${PETSC_DIR}/bin/petsc_gen_xdmf.py ex1.h5

# -iter_vertex_file ITER-51vertex-quad-tri.txt -x2_dm_view -x2_dm_type p8est
# -domain_type boxtorus -gtool "advixe-cl -collect survey:2,3"
# -nphi_particles 2 -x2_dm_view vtk:tokamac.vtu -ft_np 1,1,1 -chunksize 165536 -petscspace_poly_tensor -petscspace_order 1

runex3:
	-${MPIEXEC} -n ${NP} ./ex3.${PETSC_ARCH}                                                                                                  -dm_type p8est -dm_forest_maximum_refinement 6 -dm_forest_minimum_refinement 2 -dm_forest_initial_refinement 1 -petscfv_type leastsquares -petscfv_compute_gradients 0 -petsclimiter_type minmod -ts_view -ts_ssp_type rks2 -ts_ssp_nstages 10 -mstep 8 -debug 3 -ts_monitor -domain_type torus -petscpartitioner_type simple -use_amr false -view_initial_sol true -view_sol true -refine_tol 0.5 -coarsen_tol 1.e-2 -num_phi_cells 3 -num_particles_proc 0 -petscfv_view -dm_view hdf5:ex3.h5 -vec_view hdf5:ex3.h5::append -initial_dm_view hdf5:u0.h5 -initial_vec_view hdf5:u0.h5::append -domain_type torus -section_length 5 -radius_major 6e2 -cfl .25
	-@${MV} u0.h5 ex3.0.h5
	-@for f in ex3.*.h5; do      \
	  echo $$f;  \
	  ${PETSC_DIR}/bin/petsc_gen_xdmf.py  $$f;  \
	done

runex3_plex:
	-${MPIEXEC} -n ${NP} ./ex3.${PETSC_ARCH}  -dm_refine 2 -petscfv_type leastsquares -petscfv_compute_gradients 0 -petsclimiter_type minmod -ts_view -ts_ssp_type rks2 -ts_ssp_nstages 10 -mstep 2 -debug 2 -ts_monitor -domain_type torus -petscpartitioner_type simple -use_amr false -view_initial true -num_phi_cells 8 -num_particles_proc 0 -petscfv_view -dm_view hdf5:ex3.h5 -vec_view hdf5:ex3.h5::append -dm_initial_view hdf5:init.h5 -vec_initial_view hdf5:init.h5::append -use_line_section false -section_length 10 -radius_major 6e2
	-${MC} ex3.h5 ex3.0.h5

val:
	-${MPIEXEC} -n ${NP} valgrind --dsymutil=yes --leak-check=yes --gen-suppressions=no --num-callers=20 --error-limit=no ./ex3.${PETSC_ARCH} -dm_type p8est -dm_forest_maximum_refinement 6 -dm_forest_minimum_refinement 1 -dm_forest_initial_refinement 1 -petscfv_type leastsquares -petscfv_compute_gradients 0 -petsclimiter_type minmod -ts_view -ts_ssp_type rks2 -ts_ssp_nstages 10 -mstep 2 -debug 3 -ts_monitor -domain_type torus -petscpartitioner_type simple -use_amr false -view_initial true -view_sol true -refine_tol 0.5 -coarsen_tol 1.e-2 -num_phi_cells 3 -num_particles_proc 0 -petscfv_view -dm_view hdf5:ex3.h5 -vec_view hdf5:ex3.h5::append -initial_dm_view hdf5:u0.h5 -initial_vec_view hdf5:u0.h5::append -use_line_section false -section_length 10 -radius_major 6e2

debug:
	-@gdb --args ./ex3.${PETSC_ARCH} -dm_type p8est -dm_forest_maximum_refinement 6 -dm_forest_minimum_refinement 1 -dm_forest_initial_refinement 1 -petscfv_type leastsquares -petscfv_compute_gradients 0 -petsclimiter_type minmod -ts_view -ts_ssp_type rks2 -ts_ssp_nstages 10 -mstep 2 -debug 2 -ts_monitor -domain_type torus -petscpartitioner_type simple -use_amr false -view_initial_sol true -view_sol false -refine_tol 0.5 -coarsen_tol 1.e-2 -num_phi_cells 3 -num_particles_proc 0 -petscfv_view -dm_view hdf5:ex3.h5 -vec_view hdf5:ex3.h5::append -use_line_section false -section_length 10 -radius_major 6e2

include ${PETSC_DIR}/lib/petsc/conf/test

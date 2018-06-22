
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/dm/impls/swarm/examples/tests/
EXAMPLESC       =
EXAMPLESF       =
MANSEC          = DM

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o ${PETSC_DM_LIB}
#	${RM} -f ex1.o

NP=1
N=9
P=16
VALGRIND_PROG=valgrind
VALGRIND_ARGS=--dsymutil=yes --leak-check=full --gen-suppressions=no --num-callers=20 --error-limit=no
VAL=$(VALGRIND_PROG) $(VALGRIND_ARGS)

run-bc-periodic:
	-${MPIEXEC} -n ${NP} ./ex1 -dim 2 -nbrVerEdge ${N} -simplex 0 -dm_plex_separate_marker 0 -pc_type gamg -ksp_type cg -ksp_rtol 1.e-14 -petscspace_order 2 -petscfe_default_quadrature_order 4 -particles_cell ${P} -particle_perturbation 1.e-2 -mesh_perturbation 1.e-20 -x_boundary periodic

run:
	-${MPIEXEC} -n ${NP} ./ex1 -dim 2 -nbrVerEdge ${N} -simplex 0 -dm_plex_separate_marker 0 -pc_type gamg -ksp_type cg -ksp_rtol 1.e-14 -petscspace_order 2 -petscfe_default_quadrature_order 4 -particles_cell ${P} -particle_perturbation 1.e-2 -mesh_perturbation 1.e-20
#-dm_view hdf5:sol.h5 -vec_view hdf5:sol.h5::append -particle_interp_mat_view ascii:f.m:ascii_matlab
#	-@${PETSC_DIR}/lib/petsc/bin/petsc_gen_xdmf.py sol.h5
#-petscfe_default_quadrature_order 4

val:
	-${MPIEXEC} -n ${NP} $(VAL) ./ex1 -dim 2 -nbrVerEdge ${N} -simplex 0 -dm_plex_separate_marker 0 -pc_type gamg -ksp_type cg -ksp_rtol 1.e-14 -petscspace_order 2 -particles_cell ${P} -x_boundary none

# -dm_plex_periodic_cut -dm_view hdf5:sol.h5 -vec_view hdf5:sol.h5::append -particle_interp_mat_view ascii:f.m:ascii_matlab 

lldb:
	-@lldb ./ex1 -- -dim 2 -nbrVerEdge 4 -simplex 0 -dm_plex_separate_marker 0 -dm_view -sw_view -pc_type lu -petscspace_order 1

include ${PETSC_DIR}/lib/petsc/conf/test

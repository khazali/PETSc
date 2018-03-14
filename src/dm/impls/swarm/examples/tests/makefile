
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
	${RM} -f ex1.o

NP=1
VALGRIND_PROG=valgrind
VALGRIND_ARGS=--dsymutil=yes --leak-check=full --gen-suppressions=no --num-callers=20 --error-limit=no
VAL=$(VALGRIND_PROG) $(VALGRIND_ARGS)

run:
	-${MPIEXEC} -n ${NP} ./ex1 -dim 2 -nbrVerEdge 4 -simplex 0 -dm_plex_separate_marker 0 -pc_type lu -petscspace_order 1 -particles_cell 4 # -particle_mass_mat_view ascii:f.m:ascii_matlab -dm_plex_periodic_cut -dm_view hdf5:sol.h5 -vec_view hdf5:sol.h5::append
	-@${PETSC_DIR}/bin/petsc_gen_xdmf.py sol.h5

val:
	-${MPIEXEC} -n ${NP} $(VAL) ./ex1 -dim 2 -nbrVerEdge 4 -simplex 0 -dm_plex_separate_marker 0 -pc_type lu -petscspace_order 1 -particles_cell 4 -dm_view
# -dm_plex_periodic_cut -dm_view hdf5:sol.h5 -vec_view hdf5:sol.h5::append

lldb:
	-@lldb ./ex1 -- -dim 2 -nbrVerEdge 4 -simplex 0 -dm_plex_separate_marker 0 -dm_view -sw_view -pc_type lu -petscspace_order 1
include ${PETSC_DIR}/lib/petsc/conf/test

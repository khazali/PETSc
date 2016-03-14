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
	${RM} -f ex1.o
runex1:
	-@${MPIEXEC} -n 1 ./ex1 -dm_type forest -dm_forest_topology tokamac -dm_view vtk:tokamac.vtu -npart_cell 10000 -mstep 10

#--------------------------------------------------------------------------

include ${PETSC_DIR}/lib/petsc/conf/test

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
	-@${MPIEXEC} -n 3 ./ex1 -dm_type p8est -dm_forest_topology shell -dm_view vtk:tokamac.vtu:VTK_VTU -use_electrons true -npart_cell 10 -mstep 10 -nphi_particles 3

#-bsp_chuncksize 32000

val: ex1
	-${MPIEXEC} -n 3 valgrind --dsymutil=yes --leak-check=yes --gen-suppressions=no --num-callers=20 --error-limit=no ./ex1 -dm_type p8est -dm_forest_topology shell -dm_view vtk:tokamac.vtu -use_electrons true -npart_cell 10 -mstep 10000 -nphi_particles 3


include ${PETSC_DIR}/lib/petsc/conf/test


LOCDIR		 = src/ksp/ksp/examples/tutorials/network
MANSEC           = KSP

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

ex1_nest: ex1_nest.o chkopts
	-${CLINKER} -o ex1_nest ex1_nest.o ${PETSC_KSP_LIB}
	${RM} ex1_nest.o


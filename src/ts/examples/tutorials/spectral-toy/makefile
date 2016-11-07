
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/spectral-toy
EXAMPLESC       = ex1.c ex2.c
EXAMPLESF       =
MANSEC          = TS
DIRS            =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o petscgll.o  chkopts
	-${CLINKER} -o ex1 ex1.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o ex1.o

ex2: ex2.o petscgll.o  chkopts
	-${CLINKER} -o ex2 ex2.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o ex2.o

TESTEXAMPLES_C =

include ${PETSC_DIR}/lib/petsc/conf/test

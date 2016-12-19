
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/spectral-toy
EXAMPLESC       = ex1.c ex2.c ex3.c heat.c heatgll.c
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

ex3: ex3.o petscgll.o  chkopts
	-${CLINKER} -o ex3 ex3.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o ex3.o

heat: heat.o petscgll.o  chkopts
	-${CLINKER} -o heat heat.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o heat.o

heatgll: heatgll.o petscgll.o  chkopts
	-${CLINKER} -o heatgll heatgll.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o heatgll.o


TESTEXAMPLES_C =

include ${PETSC_DIR}/lib/petsc/conf/test

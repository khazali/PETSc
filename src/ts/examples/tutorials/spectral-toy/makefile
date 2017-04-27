
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/spectral-toy
EXAMPLESC       = ex1.c ex2.c heat.c heatgll.c ex3.c
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

heatEDMlinearF: heatEDMlinearF.o petscgll.o  chkopts
	-${CLINKER} -o heatEDMlinearF heatEDMlinearF.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o heatEDMlinearf.o

heatEDMlinear: heatEDMlinear.o petscgll.o  chkopts
	-${CLINKER} -o heatEDMlinear heatEDMlinear.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o heatEDMlinear.o

heatEDMlinear2d: heatEDMlinear2d.o petscgll.o  chkopts
	-${CLINKER} -o heatEDMlinear2d heatEDMlinear2d.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o heatEDMlinear2d.o

heatEDMlinear_adj: heatEDMlinear_adj.o petscgll.o  chkopts
	-${CLINKER} -o heatEDMlinear_adj heatEDMlinear_adj.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o heatEDMlinear_adj.o

heatNlinear: heatNlinear.o petscgll.o  chkopts
	-${CLINKER} -o heatNlinear heatNlinear.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o heatNlinear.o

heatNlinear2d: heatNlinear2d.o petscgll.o specoplib.o chkopts
	-${CLINKER} -o heatNlinear2d heatNlinear2d.o specoplib.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o specoplib.oheatNlinear2d.o

heatNlinearIJ: heatNlinearIJ.o petscgll.o  chkopts
	-${CLINKER} -o heatNlinearIJ heatNlinearIJ.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o heatNlinearIJ.o

ex3: ex3.o petscgll.o  chkopts
	-${CLINKER} -o ex3 ex3.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o ex3.o

TESTEXAMPLES_C =

include ${PETSC_DIR}/lib/petsc/conf/test

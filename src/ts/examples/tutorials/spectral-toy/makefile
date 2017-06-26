CFLAGS          = -g
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

all: remove burgersEDMnonlinear_adj 

remove:
	rm -f PDEadjoint/*

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

convNlinear: convNlinear.o petscgll.o  chkopts
	-${CLINKER} -o convNlinear convNlinear.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o convNlinear.o

bkcpheat: bkcpheat.o petscgll.o  chkopts
	-${CLINKER} -o bkcpheat bkcpheat.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o bkcpheat.o

heatEDMlinear: heatEDMlinear.o petscgll.o  chkopts
	-${CLINKER} -o heatEDMlinear heatEDMlinear.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o heatEDMlinear.o

heatEDMlinear2d: heatEDMlinear2d.o petscgll.o  chkopts
	-${CLINKER} -o heatEDMlinear2d heatEDMlinear2d.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o heatEDMlinear2d.o

heatEDMlinear_adj: heatEDMlinear_adj.o petscgll.o  chkopts
	-${CLINKER} -o heatEDMlinear_adj heatEDMlinear_adj.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o heatEDMlinear_adj.o

heatEDMlinear_adj_B: heatEDMlinear_adj_B.o petscgll.o  chkopts
	-${CLINKER} -o heatEDMlinear_adj_B heatEDMlinear_adj_B.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o heatEDMlinear_adj_B.o

heatwithfcall: heatwithfcall.o petscgll.o  chkopts
	-${CLINKER} -o heatwithfcall heatwithfcall.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o heatwithfcall.o

theburg: theburg.o petscgll.o  chkopts
	-${CLINKER} -o theburg theburg.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o theburg.o

convEDMlinear_loc_adj: convEDMlinear_loc_adj.o petscgll.o  chkopts
	-${CLINKER} -o convEDMlinear_loc_adj convEDMlinear_loc_adj.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o convEDMlinear_loc_adj.o

convEDMlinear_adj: convEDMlinear_adj.o petscgll.o  chkopts
	-${CLINKER} -o convEDMlinear_adj convEDMlinear_adj.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o convEDMlinear_adj.o

convEDMnonlinear_adj: convEDMnonlinear_adj.o petscgll.o  chkopts
	-${CLINKER} -o convEDMnonlinear_adj convEDMnonlinear_adj.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o convEDMnonlinear_adj.o

burgersEDMnonlinear_adj: burgersEDMnonlinear_adj.o petscgll.o  chkopts
	-${CLINKER} -o burgersEDMnonlinear_adj burgersEDMnonlinear_adj.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o burgersEDMnonlinear_adj.o

heatNlinear: heatNlinear.o petscgll.o  chkopts
	-${CLINKER} -o heatNlinear heatNlinear.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o heatNlinear.o

heatNlinear2d: heatNlinear2d.o petscgll.o specoplib.o chkopts
	-${CLINKER} -o heatNlinear2d heatNlinear2d.o specoplib.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o specoplib.oheatNlinear2d.o

heatNlinearIJ: heatNlinearIJ.o petscgll.o  chkopts
	-${CLINKER} -o heatNlinearIJ heatNlinearIJ.o petscgll.o  ${PETSC_TS_LIB}
	${RM} petscgll.o heatNlinearIJ.o

TESTEXAMPLES_C =

include ${PETSC_DIR}/lib/petsc/conf/test


CLFAGS          =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/advection-diffusion-reaction/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c
EXAMPLESF       = shashi.F90
EXAMPLESFH      =
MANSEC          = TS
DIRS            =
CLEANFILES      =  SA-data/*

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

shashi: shashi.o  chkopts
	-${FLINKER} -o shashi shashi.o  ${PETSC_TS_LIB}
	${RM} shashi.o

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_TS_LIB}
	${RM} ex1.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o  ${PETSC_TS_LIB}
	${RM} ex2.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o  ${PETSC_TS_LIB}
	${RM} ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o  ${PETSC_TS_LIB}
	${RM} ex4.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o  ${PETSC_TS_LIB}
	${RM} ex5.o

ex5adj: ex5adj.o  chkopts
	-${CLINKER} -o ex5adj ex5adj.o  ${PETSC_TS_LIB}
	${RM} ex5adj.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_TS_LIB}
	${RM} ex6.o

runex5adj_ao:
	-@${MPIEXEC} -n 1 ./ex5adj -ts_max_steps 10 -ts_monitor -tsgradient_adjoint_ts_monitor -da_grid_x 16 -da_grid_y 16 -adjointode > ex5adj_ao.tmp 2>&1; \
	   if (${DIFF} output/ex5adj_ao.out ex5adj_ao.tmp) then true; \
	   else printf "${PWD}\n Possible problem with ex5adj_ao, diffs above\n=======================================\n"; fi; \
	   ${RM} -f ex5adj_ao.tmp SA-data/*


include ${PETSC_DIR}/lib/petsc/conf/test

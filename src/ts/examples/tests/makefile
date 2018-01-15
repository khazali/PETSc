
CFLAGS	        =
FFLAGS	        =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tests/
EXAMPLESC       = ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c ex9.c ex10.c ex11.c ex14.c ex25.c
EXAMPLESF       =
EXAMPLESFH      =
MANSEC          = TS

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_TS_LIB}
	${RM} ex1.o

ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o  ${PETSC_TS_LIB}
	${RM} ex1f.o

ex2: ex2.o  chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_TS_LIB}
	${RM} ex2.o

ex3: ex3.o  chkopts
	-${CLINKER} -o ex3 ex3.o ${PETSC_TS_LIB}
	${RM} ex3.o

ex4: ex4.o  chkopts
	-${CLINKER} -o ex4 ex4.o ${PETSC_TS_LIB}
	${RM} ex4.o

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o ${PETSC_TS_LIB}
	${RM} ex5.o

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o ${PETSC_TS_LIB}
	${RM} ex6.o

ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o ${PETSC_TS_LIB}
	${RM} ex7.o

ex8: ex8.o  chkopts
	-${CLINKER} -o ex8 ex8.o ${PETSC_TS_LIB}
	${RM} ex8.o

ex9: ex9.o  chkopts
	-${CLINKER} -o ex9 ex9.o ${PETSC_TS_LIB}
	${RM} ex9.o

ex10: ex10.o  chkopts
	-${CLINKER} -o ex10 ex10.o ${PETSC_TS_LIB}
	${RM} ex10.o

ex11: ex11.o  chkopts
	-${CLINKER} -o ex11 ex11.o ${PETSC_TS_LIB}
	${RM} ex11.o

ex14: ex14.o  chkopts
	-${CLINKER} -o ex14 ex14.o ${PETSC_TS_LIB}
	${RM} ex14.o

ex22: ex22.o  chkopts
	-${CLINKER} -o ex22 ex22.o  ${PETSC_TS_LIB}
	${RM} ex22.o

ex23: ex23.o  chkopts
	-${CLINKER} -o ex23 ex23.o  ${PETSC_TS_LIB}
	${RM} ex23.o

ex25: ex25.o  chkopts
	-${CLINKER} -o ex25 ex25.o ${PETSC_TS_LIB}
	${RM} ex25.o

runex14_lorentz:
	-@${MPIEXEC} -n 1 ./ex14 -ts_type rk -tf 3  -lorentz -ts_rtol 1.e-6 -ts_atol 1.e-6 -ts_trajectory_type memory > ex14.tmp 2>&1;  \
           ${DIFF} output/ex14_lorentz.out ex14.tmp || printf "${PWD}\nPossible problem with ex14_lorentz, diffs above\n=========================================\n"; \
           ${RM} -rf ex14.tmp

runex14_lorentz_i_noifunc:
	-@${MPIEXEC} -n 1 ./ex14 -ts_type bdf -ts_bdf_order 3 -tlm_ts_bdf_order 3 -adjoint_tlm_ts_bdf_order 3 -tf 0.5  -lorentz -ts_rtol 1.e-6 -ts_atol 1.e-6 -ts_trajectory_type memory > ex14.tmp 2>&1;  \
           ${DIFF} output/ex14_lorentz.out ex14.tmp || printf "${PWD}\nPossible problem with ex14_lorentz_i_noifunc, diffs above\n=========================================\n"; \
           ${RM} -rf ex14.tmp

runex14_lorentz_i:
	-@${MPIEXEC} -n 1 ./ex14 -ts_type bdf -ts_bdf_order 3 -tlm_ts_bdf_order 3 -adjoint_tlm_ts_bdf_order 3 -tf 0.5  -lorentz -ts_rtol 1.e-6 -ts_atol 1.e-6 -ts_trajectory_type memory -ifunc > ex14.tmp 2>&1;  \
           ${DIFF} output/ex14_lorentz.out ex14.tmp || printf "${PWD}\nPossible problem with ex14_lorentz_i, diffs above\n=========================================\n"; \
           ${RM} -rf ex14.tmp

runex14_oscillator:
	-@${MPIEXEC} -n 1 ./ex14 -ts_type rk -ts_rk_type 5dp -tlm_ts_rk_type 5dp -adjoint_tlm_ts_rk_type 5dp -tf 10  -omega 3 -gamma 2 -ts_trajectory_type memory > ex14.tmp 2>&1;  \
           ${DIFF} output/ex14_oscillator.out ex14.tmp || printf "${PWD}\nPossible problem with ex14_oscillator, diffs above\n=========================================\n"; \
           ${RM} -rf ex14.tmp

runex14_oscillator_i_noifunc:
	-@${MPIEXEC} -n 1 ./ex14 -ts_type cn -tf 10 -dt 0.1 -omega 3 -gamma 2 -ts_trajectory_type memory > ex14.tmp 2>&1;  \
           ${DIFF} output/ex14_oscillator.out ex14.tmp || printf "${PWD}\nPossible problem with ex14_oscillator_i_noifunc, diffs above\n=========================================\n"; \
           ${RM} -rf ex14.tmp

runex14_oscillator_i:
	-@${MPIEXEC} -n 1 ./ex14 -ts_type cn -tf 10 -dt 0.1 -omega 3 -gamma 2 -ts_trajectory_type memory -ifunc > ex14.tmp 2>&1;  \
           ${DIFF} output/ex14_oscillator.out ex14.tmp || printf "${PWD}\nPossible problem with ex14_oscillator_i, diffs above\n=========================================\n"; \
           ${RM} -rf ex14.tmp

include ${PETSC_DIR}/lib/petsc/conf/test

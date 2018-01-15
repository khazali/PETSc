
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c \
                ex9.c ex10.c ex11.c ex12.c ex13.c ex14.c ex15.c ex16.c ex17.c \
                ex19.c ex20.c ex21.c ex22.c ex23.c ex24.c ex25.c ex26.c \
                ex28.c ex31.c ex34.c ex35.cxx extchem.c\
                ex16adj.c ex16opt_p.c ex16opt_ic.c  \
                ex20adj.c ex20opt_p.c ex20opt_ic.c  \
                ex40.c ex41.c ex42.c ex48.c
EXAMPLESF       = ex1f.F ex22f.F ex22f_mf.F90
MANSEC          = TS
DIRS            = phasefield advection-diffusion-reaction eimex power_grid network
CLEANFILES      = SA-data/*
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o  chkopts
	-${CLINKER} -o ex1 ex1.o  ${PETSC_TS_LIB}
	${RM} ex1.o

ex1f: ex1f.o  chkopts
	-${FLINKER} -o ex1f ex1f.o  ${PETSC_TS_LIB}
	${RM} ex1f.o

ex2: ex2.o chkopts
	-${CLINKER} -o ex2 ex2.o ${PETSC_TS_LIB}
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

ex6: ex6.o  chkopts
	-${CLINKER} -o ex6 ex6.o  ${PETSC_TS_LIB}
	${RM} ex6.o

ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_TS_LIB}
	${RM} ex7.o

ex8: ex8.o  chkopts
	-${CLINKER} -o ex8 ex8.o  ${PETSC_TS_LIB}
	${RM} ex8.o

ex9: ex9.o  chkopts
	-${CLINKER} -o ex9 ex9.o  ${PETSC_TS_LIB}
	${RM} ex9.o

ex10: ex10.o  chkopts
	-${CLINKER} -o ex10 ex10.o  ${PETSC_TS_LIB}
	${RM} ex10.o

ex11: ex11.o  chkopts
	-${CLINKER} -o ex11 ex11.o  ${PETSC_TS_LIB}
	${RM} ex11.o

ex11_sa: ex11_sa.o chkopts
	-${CLINKER} -o ex11_sa ex11_sa.o ${PETSC_TS_LIB}
	${RM} ex11_sa.o

ex12: ex12.o  chkopts
	-${CLINKER} -o ex12 ex12.o  ${PETSC_TS_LIB}
	${RM} ex12.o

ex13: ex13.o  chkopts
	-${CLINKER} -o ex13 ex13.o  ${PETSC_TS_LIB}
	${RM} ex13.o

ex14: ex14.o  chkopts
	-${CLINKER} -o ex14 ex14.o  ${PETSC_TS_LIB}
	${RM} ex14.o

ex15: ex15.o  chkopts
	-${CLINKER} -o ex15 ex15.o  ${PETSC_TS_LIB}
	${RM} ex15.o

ex16: ex16.o  chkopts
	-${CLINKER} -o ex16 ex16.o  ${PETSC_TS_LIB}
	${RM} ex16.o

ex16adj: ex16adj.o chkopts
	-${CLINKER} -o ex16adj ex16adj.o ${PETSC_TS_LIB}
	${RM} ex16adj.o

ex16opt_p: ex16opt_p.o chkopts
	-${CLINKER} -o ex16opt_p ex16opt_p.o ${PETSC_LIB}
	${RM} ex16opt_p.o

ex16opt_ic: ex16opt_ic.o chkopts
	-${CLINKER} -o ex16opt_ic ex16opt_ic.o ${PETSC_LIB}
	${RM} ex16opt_ic.o

ex17: ex17.o  chkopts
	-${CLINKER} -o ex17 ex17.o  ${PETSC_TS_LIB}
	${RM} ex17.o

ex18: ex18.o  chkopts
	-${CLINKER} -o ex18 ex18.o  ${PETSC_TS_LIB}
	${RM} ex18.o

ex19: ex19.o  chkopts
	-${CLINKER} -o ex19 ex19.o  ${PETSC_TS_LIB}
	${RM} ex19.o

ex20: ex20.o  chkopts
	-${CLINKER} -o ex20 ex20.o  ${PETSC_TS_LIB}
	${RM} ex20.o

ex20adj: ex20adj.o  chkopts
	-${CLINKER} -o ex20adj ex20adj.o  ${PETSC_TS_LIB}
	${RM} ex20adj.o

ex20fwd: ex20fwd.o  chkopts
	-${CLINKER} -o ex20fwd ex20fwd.o  ${PETSC_TS_LIB}
	${RM} ex20fwd.o

ex20opt_ic: ex20opt_ic.o chkopts
	-${CLINKER} -o ex20opt_ic ex20opt_ic.o ${PETSC_LIB}
	${RM} ex20opt_ic.o

ex20opt_p: ex20opt_p.o chkopts
	-${CLINKER} -o ex20opt_p ex20opt_p.o ${PETSC_LIB}
	${RM} ex20opt_p.o

ex21: ex21.o  chkopts
	-${CLINKER} -o ex21 ex21.o  ${PETSC_TS_LIB}
	${RM} ex21.o

ex22: ex22.o chkopts
	-${CLINKER} -o ex22 ex22.o ${PETSC_TS_LIB}
	${RM} ex22.o

ex22f: ex22f.o  chkopts
	-${FLINKER} -o ex22f ex22f.o ${PETSC_TS_LIB}
	${RM} ex22f.o

ex22f_mf: ex22f_mf.o  chkopts
	-${FLINKER} -o ex22f_mf ex22f_mf.o ${PETSC_TS_LIB}
	${RM} ex22f_mf.o

ex23: ex23.o chkopts
	-${CLINKER} -o ex23 ex23.o ${PETSC_TS_LIB}
	${RM} ex23.o

ex24: ex24.o chkopts
	-${CLINKER} -o ex24 ex24.o ${PETSC_TS_LIB}
	${RM} ex24.o

ex25: ex25.o chkopts
	-${CLINKER} -o ex25 ex25.o ${PETSC_TS_LIB}
	${RM} ex25.o

ex26: ex26.o chkopts
	-${CLINKER} -o ex26 ex26.o ${PETSC_TS_LIB}
	${RM} ex26.o

ex27: ex27.o chkopts
	-${CLINKER} -o ex27 ex27.o ${PETSC_TS_LIB}
	${RM} ex27.o

ex28: ex28.o chkopts
	-${CLINKER} -o ex28 ex28.o ${PETSC_TS_LIB}
	${RM} ex28.o

ex29: ex29.o chkopts
	-${CLINKER} -o ex29 ex29.o ${PETSC_TS_LIB}
	${RM} ex29.o

ex31: ex31.o chkopts
	-${CLINKER} -o ex31 ex31.o ${PETSC_TS_LIB}
	${RM} ex31.o

ex34: ex34.o chkopts
	-${CLINKER} -o ex34 ex34.o ${PETSC_TS_LIB}
	${RM} ex34.o

ex35: ex35.o chkopts
	-${CLINKER} -o ex35 ex35.o ${PETSC_TS_LIB}
	${RM} ex35.o

ex36: ex36.o chkopts
	-${CLINKER} -o ex36 ex36.o ${PETSC_TS_LIB}
	${RM} ex36.o

ex36SE: ex36SE.o chkopts
	-${CLINKER} -o ex36SE ex36SE.o ${PETSC_TS_LIB}
	${RM} ex36SE.o

ex36A: ex36A.o chkopts
	-${CLINKER} -o ex36A ex36A.o ${PETSC_TS_LIB}
	${RM} ex36A.o

extchem: extchem.o chkopts
	-${CLINKER} -o extchem extchem.o ${PETSC_TS_LIB}
	${RM} extchem.o

extchemfield: extchemfield.o chkopts
	-${CLINKER} -o extchemfield extchemfield.o ${PETSC_TS_LIB}
	${RM} extchemfield.o

ex40: ex40.o chkopts
	-${CLINKER} -o ex40 ex40.o ${PETSC_TS_LIB}
	${RM} ex40.o

ex41: ex41.o chkopts
	-${CLINKER} -o ex41 ex41.o ${PETSC_TS_LIB}
	${RM} ex41.o

ex42: ex42.o chkopts
	-${CLINKER} -o ex42 ex42.o ${PETSC_TS_LIB}
	${RM} ex42.o

ex43: ex43.o chkopts
	-${CLINKER} -o ex43 ex43.o ${PETSC_TS_LIB}
	${RM} ex43.o

ex44: ex44.o chkopts
	-${CLINKER} -o ex44 ex44.o ${PETSC_TS_LIB}
	${RM} ex44.o

ex48: ex48.o chkopts
	-${CLINKER} -o ex48 ex48.o ${PETSC_TS_LIB}
	${RM} ex48.o

runex16opt_p_ao:
	-@${MPIEXEC} -n 1 ./ex16opt_p -ts_trajectory_type memory -adjointode -tao_view -tao_monitor -tao_gttol 1.e-5 -tsgradient_adjoint_ts_atol 1.e-9 > ex16opt_p_ao.tmp 2>&1; \
	   ${DIFF} output/ex16opt_p_ao.out ex16opt_p_ao.tmp || printf "${PWD}\nPossible problem with ex16opt_p_ao, diffs above\n=========================================\n"; \
	   ${RM} -f ex16opt_p_ao.tmp

runex16opt_p_ao_hessian_mf:
	-@${MPIEXEC} -n 1 ./ex16opt_p -ts_trajectory_type memory -adjointode -tao_view -tao_monitor -tao_gttol 1.e-5 -tsgradient_adjoint_ts_atol 1.e-10 -tao_type tron -tao_mf_hessian > ex16opt_p_ao_hessian_mf.tmp 2>&1; \
	   ${DIFF} output/ex16opt_p_ao_hessian_mf.out ex16opt_p_ao_hessian_mf.tmp || printf "${PWD}\nPossible problem with ex16opt_p_ao_hessian_mf, diffs above\n=========================================\n"; \
	   ${RM} -f ex16opt_p_ao_hessian_mf.tmp

runex16opt_p_ao_hessian:
	-@${MPIEXEC} -n 1 ./ex16opt_p -ts_trajectory_type memory -adjointode -tao_view -tao_monitor -tao_gttol 1.e-5 -tsgradient_adjoint_ts_atol 1.e-10 -tshessian_soadjoint_ts_atol 1.e-10 -tshessian_foadjoint_ts_atol 1.e-10 -tshessian_tlm_ts_atol 1.e-10 -tao_type tron > ex16opt_p_ao_hessian.tmp 2>&1; \
	   ${DIFF} output/ex16opt_p_ao_hessian.out ex16opt_p_ao_hessian.tmp || printf "${PWD}\nPossible problem with ex16opt_p_ao_hessian, diffs above\n=========================================\n"; \
	   ${RM} -f ex16opt_p_ao_hessian.tmp

runex16opt_ic_ao:
	-@${MPIEXEC} -n 1 ./ex16opt_ic -ts_trajectory_type memory -adjointode -monitor 0 -tao_view -tao_monitor -tao_gttol 1.e-5 > ex16opt_ic_ao.tmp 2>&1; \
	   ${DIFF} output/ex16opt_ic_ao.out ex16opt_ic_ao.tmp || printf "${PWD}\nPossible problem with ex16opt_ic_ao, diffs above\n=========================================\n"; \
	   ${RM} -f ex16opt_ic_ao.tmp

runex16opt_ic_ao_hessian:
	-@${MPIEXEC} -n 1 ./ex16opt_ic -ts_trajectory_type memory -adjointode -monitor 0 -tao_view -tao_monitor -tao_gttol 1.e-5 -tao_type tron > ex16opt_ic_ao_hessian.tmp 2>&1; \
	   ${DIFF} output/ex16opt_ic_ao_hessian.out ex16opt_ic_ao_hessian.tmp || printf "${PWD}\nPossible problem with ex16opt_ic_ao_hessian, diffs above\n=========================================\n"; \
	   ${RM} -f ex16opt_ic_ao_hessian.tmp

runex16opt_ic_ao_hessian_mf:
	-@${MPIEXEC} -n 1 ./ex16opt_ic -ts_trajectory_type memory -adjointode -monitor 0 -tao_view -tao_monitor -tao_gttol 1.e-5 -tao_type tron -tao_mf_hessian > ex16opt_ic_ao_hessian_mf.tmp 2>&1; \
	   ${DIFF} output/ex16opt_ic_ao_hessian_mf.out ex16opt_ic_ao_hessian_mf.tmp || printf "${PWD}\nPossible problem with ex16opt_ic_ao_hessian_mf, diffs above\n=========================================\n"; \
	   ${RM} -f ex16opt_ic_ao_hessian_mf.tmp

runex20opt_p_ao:
	-@${MPIEXEC} -n 1 ./ex20opt_p -adjointode -tao_monitor -monitor 0 -ts_trajectory_type memory -tao_view > ex20opt_p_ao.tmp 2>&1; \
	  ${DIFF} output/ex20opt_p_ao.out ex20opt_p_ao.tmp || printf "${PWD}\nPossible problem with ex20opt_p_ao, diffs above\n=========================================\n"; \
	  ${RM} -f ex20opt_p_ao.tmp

runex20opt_p_ao_hessian:
	-@${MPIEXEC} -n 1 ./ex20opt_p -adjointode -tao_monitor -monitor 0 -tao_view -tao_type tron -ts_trajectory_type memory > ex20opt_p_ao_hessian.tmp 2>&1; \
	  ${DIFF} output/ex20opt_p_ao_hessian.out ex20opt_p_ao_hessian.tmp || printf "${PWD}\nPossible problem with ex20opt_p_ao_hessian, diffs above\n=========================================\n"; \
	  ${RM} -f ex20opt_p_ao_hessian.tmp

runex20opt_p_ao_hessian_mf:
	-@${MPIEXEC} -n 1 ./ex20opt_p -adjointode -tao_monitor -monitor 0 -tao_view -tao_type tron -tao_mf_hessian -ts_trajectory_type memory > ex20opt_p_ao_hessian_mf.tmp 2>&1; \
	  ${DIFF} output/ex20opt_p_ao_hessian_mf.out ex20opt_p_ao_hessian_mf.tmp || printf "${PWD}\nPossible problem with ex20opt_p_ao_hessian_mf, diffs above\n=========================================\n"; \
	  ${RM} -f ex20opt_p_ao_hessian_mf.tmp

runex23_1:
	-@${MPIEXEC} -n 1 ./ex23 -t0 1.1 -tf 2.3 -ts_type rk -ts_adapt_type basic -ts_atol 1.e-9 -ts_rtol 1.e-9 -test_event_final -p 1.3 -ts_trajectory_type memory > ex23_1.tmp 2>&1; \
	   ${DIFF} output/ex23_1.out ex23_1.tmp || printf "${PWD}\nPossible problem with ex23_1, diffs above\n=========================================\n";  \
	   ${RM} -rf ex23_1.tmp SA-data

runex23_2:
	-@${MPIEXEC} -n 1 ./ex23 -t0 1.6 -tf 2.12 -ts_type bdf -ts_adapt_type basic -ts_atol 1.e-9 -ts_rtol 1.e-9 -test_event_final -p 1.3 > ex23_2.tmp 2>&1; \
	   ${DIFF} output/ex23_2.out ex23_2.tmp || printf "${PWD}\nPossible problem with ex23_2, diffs above\n=========================================\n";  \
	   ${RM} -rf ex23_2.tmp SA-data

runex23_3:
	-@${MPIEXEC} -n 1 ./ex23 -t0 1.6 -tf 2.12 -ts_type bdf -ts_adapt_type basic -ts_atol 1.e-9 -ts_rtol 1.e-9 -test_event_final -p 1.3 -test_ifunc -test_nulljac_IC -test_nullgrad_M > ex23_3.tmp 2>&1; \
	   ${DIFF} output/ex23_3.out ex23_3.tmp || printf "${PWD}\nPossible problem with ex23_3, diffs above\n=========================================\n";  \
	   ${RM} -rf ex23_3.tmp SA-data

runex23_4:
	-@${MPIEXEC} -n 1 ./ex23 -t0 1.1 -tf 2.3 -ts_type rk -ts_adapt_type basic -ts_atol 1.e-9 -ts_rtol 1.e-9 -test_event_constant -test_rhsjacconst -ts_trajectory_reconstruction_order 3 > ex23_4.tmp 2>&1; \
	   ${DIFF} output/ex23_4.out ex23_4.tmp || printf "${PWD}\nPossible problem with ex23_4, diffs above\n=========================================\n";  \
	   ${RM} -rf ex23_4.tmp SA-data

runex23_5:
	-@${MPIEXEC} -n 1 ./ex23 -t0 0.7 -tf 1.2 -dt 0.01 -ts_type cn -test_event_constant -p 0.8 -test_ifunc -ts_trajectory_reconstruction_order 2 -test_pjac 0 > ex23_5.tmp 2>&1; \
	   ${DIFF} output/ex23_5.out ex23_5.tmp || printf "${PWD}\nPossible problem with ex23_5, diffs above\n=========================================\n";  \
	   ${RM} -rf ex23_5.tmp SA-data

runex23_6:
	-@${MPIEXEC} -n 1 ./ex23 -t0 0.01 -tf 0.8 -b 0.3 -a 1.7 -p 1 -ts_type rk -ts_adapt_type basic -ts_atol 1.e-8 -ts_rtol 1.e-8 -test_event_func -ts_trajectory_reconstruction_order 4 -tshessian_mffd > ex23_6.tmp 2>&1; \
	   ${DIFF} output/ex23_6.out ex23_6.tmp || printf "${PWD}\nPossible problem with ex23_6, diffs above\n=========================================\n";  \
	   ${RM} -rf ex23_6.tmp SA-data

runex23_7:
	-@${MPIEXEC} -n 1 ./ex23 -t0 0 -tf 0.125 -dt 0.001 -b 0.3 -a 1.7 -p 1 -ts_type rosw -test_ifunc -test_event_func -ts_adapt_type none -tshessian_mffd > ex23_7.tmp 2>&1; \
	   ${DIFF} output/ex23_7.out ex23_7.tmp || printf "${PWD}\nPossible problem with ex23_7, diffs above\n=========================================\n";  \
	   ${RM} -rf ex23_7.tmp SA-data

runex23_8:
	-@${MPIEXEC} -n 1 ./ex23 -t0 0 -tf 0.125 -b 0.5 -a 1.1 -p 0.4 -ts_type bdf -test_mix -test_pjac 0 -test_event_constant -ts_rtol 1.e-8 -ts_atol 1.e-8 > ex23_8.tmp 2>&1; \
	   ${DIFF} output/ex23_8.out ex23_8.tmp || printf "${PWD}\nPossible problem with ex23_8, diffs above\n=========================================\n";  \
	   ${RM} -rf ex23_8.tmp SA-data

runex23_9:
	-@${MPIEXEC} -n 5 ./ex23 -t0 -0.3 -tf 0.5 -b 1.2 -a 2.1 -p 0.3 -ts_type rk -test_general_fixed -test_general_final -test_general -test_event_func -test_event_constant -test_event_final -ts_rtol 1.e-6 -ts_atol 1.e-6 -tshessian_mffd > ex23_9.tmp 2>&1; \
	   ${DIFF} output/ex23_9.out ex23_9.tmp || printf "${PWD}\nPossible problem with ex23_9, diffs above\n=========================================\n";  \
	   ${RM} -rf ex23_9.tmp SA-data

runex23_10:
	-@${MPIEXEC} -n 5 ./ex23 -t0 -0.3 -tf 0.5 -b 1.2 -a 2.1 -p 0.3 -ts_type bdf -test_general_fixed -test_general_final -test_general -test_event_func -test_event_constant -test_event_final -ts_rtol 1.e-6 -ts_atol 1.e-6 -tshessian_mffd > ex23_10.tmp 2>&1; \
	   ${DIFF} output/ex23_10.out ex23_10.tmp || printf "${PWD}\nPossible problem with ex23_10, diffs above\n=========================================\n";  \
	   ${RM} -rf ex23_10.tmp SA-data

runex23_11:
	-@${MPIEXEC} ./ex23 -t0 0.41 -tf 0.47 -b 0.3 -a 1.25 -p 2.3 -ts_type rk -test_general_final -test_general -test_event_constant -test_event_final -ts_rtol 1.e-10 -ts_atol 1.e-10 -ts_trajectory_type memory -test_forward_event > ex23_11.tmp 2>&1; \
	   ${DIFF} output/ex23_11.out ex23_11.tmp || printf "${PWD}\nPossible problem with ex23_11, diffs above\n=========================================\n";  \
	   ${RM} -rf ex23_11.tmp

include ${PETSC_DIR}/lib/petsc/conf/test

ADIC_RUNTIME    = -L/Users/hongzhang/Projects/adic_runtime/lib -lruntimedense 
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/ts/examples/tutorials/
EXAMPLESC       = ex1.c ex2.c ex3.c ex4.c ex5.c ex6.c ex7.c ex8.c \
                ex9.c ex10.c ex11.c ex12.c ex13.c ex14.c ex15.c ex16.c ex17.c \
                ex19.c ex20.c ex21.c ex22.c ex24.c ex25.c ex26.c \
                ex28.c ex31.c ex34.c ex35.cxx extchem.c\
                ex16adj.c ex16opt_p.c ex16opt_ic.c  \
                ex20adj.c ex20opt_p.c ex20opt_ic.c  \
                ex40.c ex41.c ex42.c ex48.c ex49.c ex50.c
EXAMPLESF       = ex1f.F ex22f.F ex22f_mf.F90
MANSEC          = TS
DIRS            = phasefield advection-diffusion-reaction eimex power_grid network
CLEANFILES      = SA-data/*
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex20_mf: ex20_mf.o  chkopts
	-${CLINKER} -o ex20_mf ex20_mf.o vanderpol1/build/driver.o vanderpol1/build/libhead_r_la-head_r.cn.xb.pp.o vanderpol1/build/libhead_f_la-head_f.cn.xb.pp.o ${PETSC_TS_LIB} ${ADIC_RUNTIME}
	${RM} ex20_mf.o

include ${PETSC_DIR}/lib/petsc/conf/test

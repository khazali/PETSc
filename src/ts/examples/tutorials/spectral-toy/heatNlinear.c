
static char help[] ="Solves a simple time-dependent linear PDE (the heat equation).\n\
Input parameters include:\n\
  -m <points>, where <points> = number of grid points\n\
  -time_dependent_rhs : Treat the problem as having a time-dependent right-hand side\n\
  -debug              : Activate debugging printouts\n\
  -nox                : Deactivate x-window graphics\n\n";

/*
   Concepts: TS^time-dependent linear problems
   Concepts: TS^heat equation
   Concepts: TS^diffusion equation
   Processors: 1
*/

/* ------------------------------------------------------------------------

   This program solves the one-dimensional heat equation (also called the
   diffusion equation),
       u_t = u_xx,
   on the domain 0 <= x <= 1, with the boundary conditions
       u(t,0) = 0, u(t,1) = 0,
   and the initial condition
       u(0,x) = sin(6*pi*x) + 3*sin(2*pi*x).
   This is a linear, second-order, parabolic equation.

   We discretize the right-hand side using finite differences with
   uniform grid spacing h:
       u_xx = (u_{i+1} - 2u_{i} + u_{i-1})/(h^2)
   We then demonstrate time evolution using the various TS methods by
   running the program via
       ex3 -ts_type <timestepping solver>

   We compare the approximate solution with the exact solution, given by
       u_exact(x,t) = exp(-36*pi*pi*t) * sin(6*pi*x) +
                      3*exp(-4*pi*pi*t) * sin(2*pi*x)

   Notes:
   This code demonstrates the TS solver interface to two variants of
   linear problems, u_t = f(u,t), namely
     - time-dependent f:   f(u,t) is a function of t
     - time-independent f: f(u,t) is simply f(u)

    The parallel version of this code is ts/examples/tutorials/ex4.c

  ------------------------------------------------------------------------- */

/*
   Include "petscts.h" so that we can use TS solvers.  Note that this file
   automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h  - vectors
     petscmat.h  - matrices
     petscis.h     - index sets            petscksp.h  - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h   - preconditioners
     petscksp.h   - linear solvers        petscsnes.h - nonlinear solvers
*/

#include <petscts.h>
#include "petscgll.h"
#include <petscdraw.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/
typedef struct {
  Vec         solution;          /* global exact solution vector */
  PetscInt    Nl;                 /* total number of grid points */
  PetscReal   *Z;                 /* mesh grid */
  PetscScalar *W;                 /* weights */
  PetscBool   debug;             /* flag (1 indicates activation of debugging printouts) */
  PetscViewer viewer1,viewer2;  /* viewers for the solution and error */
  PetscReal   norm_2,norm_max;  /* error norms */
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode RHSMatrixHeatgll(TS,PetscReal,Vec,Mat*,Mat,void*);
extern PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void*);
extern PetscErrorCode ExactSolution(PetscReal,Vec,AppCtx*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  TS             ts;                     /* timestepping context */
  Mat            A,B,K;                      /* matrix data structure */
  Vec            u;                      /* approximate solution vector */
  PetscReal      time_total_max = 4e-5; /* default max total time */
  PetscInt       time_steps_max = 20;   /* default max timesteps */
  PetscErrorCode ierr;
  PetscInt       steps,Nl=40,i,rows[2];
  PetscMPIInt    size;
  PetscReal      dt,**temp,v;
  PetscBool      flg;
  PetscGLLIP     gll;
  PetscViewer    viewfile;
   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  //ierr = PetscOptionsGetInt(NULL,NULL,"-m",&Nl,NULL);CHKERRQ(ierr);
  //ierr = PetscOptionsHasName(NULL,NULL,"-debug",&appctx.debug);CHKERRQ(ierr);

  appctx.Nl        = Nl;
  appctx.norm_2   = 0.0;
  appctx.norm_max = 0.0;

  ierr = PetscPrintf(PETSC_COMM_SELF,"Solving a linear TS problem on 1 processor\n");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscGLLIPCreate(Nl,PETSCGLLIP_VIA_LINEARALGEBRA,&gll);CHKERRQ(ierr);
  
  ierr= PetscMalloc1(Nl, &appctx.Z);
  ierr= PetscMalloc1(Nl, &appctx.W);

  for(i=0; i<Nl; i++)
     { 
     appctx.Z[i]=(gll.nodes[i]+1.0);
      }

  for(i=0; i<Nl; i++)
     { 
     appctx.W[i]=gll.weights[i];
      }
   
  /*
     Create vector data structures for approximate and exact solutions
  */
  ierr = VecCreateSeq(PETSC_COMM_SELF,Nl,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.solution);CHKERRQ(ierr);


  /* Open viewer for binary output */
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);

 /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set optional user-defined monitoring routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSMonitorSet(ts,Monitor,&appctx,NULL);CHKERRQ(ierr);

 /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

     Create matrix data structure; set matrix evaluation routine.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

 
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,Nl,Nl,NULL,&A);CHKERRQ(ierr);
  //ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,Nl,Nl);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

 
      /*
       For linear problems with a time-dependent f(u,t) in the equation
       u_t = f(u,t), the user provides the discretized right-hand-side
       as a time-dependent matrix.
    */
  ierr = RHSMatrixHeatgll(ts,0.0,u,&A,A,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,A,A,TSComputeRHSJacobianConstant,&appctx);CHKERRQ(ierr);
  

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solution vector and initial timestep
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  dt   = 0.2e-5;
  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize timestepping solver:
       - Set the solution method to be the Backward Euler method.
       - Set timestepping duration info
     Then set runtime options, which can override these defaults.
     For example,
          -ts_max_steps <maxsteps> -ts_final_time <maxtime>
     to override the defaults set by TSSetDuration().
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSSetDuration(ts,time_steps_max,time_total_max);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the problem
     - - - - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - */
 
  /*
     Evaluate initial conditions
  */
  ierr = InitialConditions(u,&appctx);CHKERRQ(ierr);
  //ierr = VecView(u,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
 
 /*
     Run the timestepping solver
  */
  ierr = TSSolve(ts,u);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     View timestepping solver info
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscPrintf(PETSC_COMM_SELF,"avg. error (2 norm) = %g, avg. error (max norm) = %g\n",(double)(appctx.norm_2/steps),(double)(appctx.norm_max/steps));CHKERRQ(ierr);
  ierr = TSView(ts,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     For matlab output
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"mat.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = VecView(u,viewfile);CHKERRQ(ierr);
    //ierr = MatView(K,viewfile);CHKERRQ(ierr);
    ierr = VecView(appctx.solution,viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewfile);CHKERRQ(ierr);
 
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.solution);CHKERRQ(ierr);
  ierr = PetscGLLIPDestroy(&gll);CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).
  */
  ierr = PetscFinalize();
  return ierr;
}
/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "InitialConditions"
/*
   InitialConditions - Computes the solution at the initial time.

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode InitialConditions(Vec u,AppCtx *appctx)
{
  PetscScalar    *u_localptr;
  PetscErrorCode ierr;
  PetscInt       i;

  /*
    Get a pointer to vector data.
    - For default PETSc vectors, VecGetArray() returns a pointer to
      the data array.  Otherwise, the routine is implementation dependent.
    - You MUST call VecRestoreArray() when you no longer need access to
      the array.
    - Note that the Fortran interface to VecGetArray() differs from the
      C version.  See the users manual for details.
  */
  ierr = VecGetArray(u,&u_localptr);CHKERRQ(ierr);

  /*
     We initialize the solution array by simply writing the solution
     directly into the array locations.  Alternatively, we could use
     VecSetValues() or VecSetValuesLocal().
  */
  for (i=0; i<appctx->Nl; i++) 
       u_localptr[i] = PetscSinScalar(PETSC_PI*6.*appctx->Z[i]) + 3.*PetscSinScalar(PETSC_PI*2.*appctx->Z[i]);

  /*
     Restore vector
  */
  ierr = VecRestoreArray(u,&u_localptr);CHKERRQ(ierr);

  /*
     Print debugging information if desired
  */
  if (appctx->debug) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Initial guess vector\n");CHKERRQ(ierr);
    //ierr = VecView(u,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  return 0;
}
/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "ExactSolution"
/*
   ExactSolution - Computes the exact solution at a given time.

   Input Parameters:
   t - current time
   solution - vector in which exact solution will be computed
   appctx - user-defined application context

   Output Parameter:
   solution - vector with the newly computed exact solution
*/
PetscErrorCode ExactSolution(PetscReal t,Vec solution,AppCtx *appctx)
{
  PetscScalar    *s_localptr,ex1,ex2,sc1,sc2,tc = t;
  PetscErrorCode ierr;
  PetscReal      xx;
  PetscInt       i;

  /*
     Get a pointer to vector data.
  */
  ierr = VecGetArray(solution,&s_localptr);CHKERRQ(ierr);

  /*
     Simply write the solution directly into the array locations.
     Alternatively, we culd use VecSetValues() or VecSetValuesLocal().
  */
  ex1 = PetscExpScalar(-36.*PETSC_PI*PETSC_PI*tc);
  ex2 = PetscExpScalar(-4.*PETSC_PI*PETSC_PI*tc);
  sc1 = PETSC_PI*6.0;                 sc2 = PETSC_PI*2.0;
  printf("solution \n");  
  for (i=0; i<appctx->Nl; i++) 
         {xx=appctx->Z[i]; 
          s_localptr[i] = PetscSinScalar(sc1*xx)*ex1 + 3.*PetscSinScalar(sc2*xx)*ex2;
          printf("%f\n",s_localptr[i]);  
         } 
  /*
     Restore vector
  */
  ierr = VecRestoreArray(solution,&s_localptr);CHKERRQ(ierr);
  return 0;
}
/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "Monitor"
/*
   Monitor - User-provided routine to monitor the solution computed at
   each timestep.  This example plots the solution and computes the
   error in two different norms.

   This example also demonstrates changing the timestep via TSSetTimeStep().

   Input Parameters:
   ts     - the timestep context
   step   - the count of the current step (with 0 meaning the
             initial condition)
   time   - the current time
   u      - the solution at this timestep
   ctx    - the user-provided context for this monitoring routine.
            In this case we use the application context which contains
            information about the problem size, workspace and the exact
            solution.
*/
PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal time,Vec u,void *ctx)
{
  AppCtx         *appctx = (AppCtx*) ctx;   /* user-defined application context */
  PetscErrorCode ierr;
  PetscReal      norm_2,norm_max,dt,dttol,norm;
  Vec f;

  /*
     View a graph of the current iterate
  */
  //ierr = VecView(u,appctx->viewer2);CHKERRQ(ierr);

  /*
     Compute the exact solution
  */
  ierr = ExactSolution(time,appctx->solution,appctx);CHKERRQ(ierr);

  /*
     Print debugging information if desired
  */
  if (appctx->debug) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Computed solution vector\n");CHKERRQ(ierr);
    ierr = VecView(u,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Exact solution vector\n");CHKERRQ(ierr);
    ierr = VecView(appctx->solution,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }


   /*
     Compute the 2-norm and max-norm of the error
  */
  
/* 
    //all this now has to change to have a proper integration with all weights
    // compute the L^2 norm of the error 
    ierr = VecGetArray(appctx->solution,&f);CHKERRQ(ierr);
    ierr = VecAXPY(f,-1.0,u);CHKERRQ(ierr);
    ierr = PetscGLLIPIntegrate(appctx->W,f,&norm);CHKERRQ(ierr);
    norm = PetscSqrtReal(norm);
    //ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF,"L^2 norm of the error %D %g\n",n,(double)norm);CHKERRQ(ierr);
    ierr = VecDestroy(&f);CHKERRQ(ierr);
    //xc   = (PetscReal)n;
    //yc   = PetscLogReal(norm);
    //ierr = PetscDrawLGAddPoint(lg,&xc,&yc);CHKERRQ(ierr);
    //ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);

*/

  /*
     Compute the 2-norm and max-norm of the error
  */
  ierr   = VecAXPY(appctx->solution,-1.0,u);CHKERRQ(ierr);
  ierr   = VecNorm(appctx->solution,NORM_2,&norm_2);CHKERRQ(ierr);
  ierr   = VecNorm(appctx->solution,NORM_MAX,&norm_max);CHKERRQ(ierr);
  if (norm_2   < 1e-14) norm_2   = 0;
  if (norm_max < 1e-14) norm_max = 0;

  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  //ierr = PetscPrintf(PETSC_COMM_WORLD,"Timestep %3D: step size = %g, time = %g, 2-norm error = %g, max norm error = %g\n",step,(double)dt,(double)time,(double)norm_2,(double)norm_max);CHKERRQ(ierr);

  appctx->norm_2   += norm_2;
  appctx->norm_max += norm_max;

  dttol = .0001;
  ierr  = PetscOptionsGetReal(NULL,NULL,"-dttol",&dttol,NULL);CHKERRQ(ierr);
  if (dt < dttol) {
    dt  *= .999;
    //ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  }

  /*
     View a graph of the error
  */
  //ierr = VecView(appctx->solution,appctx->viewer1);CHKERRQ(ierr);

  /*
     Print debugging information if desired
  */
  if (appctx->debug) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error vector\n");CHKERRQ(ierr);
    ierr = VecView(appctx->solution,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  return 0;
}

/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "RHSMatrixHeatgll"

/*
   RHSMatrixHeat - User-provided routine to compute the right-hand-side
   matrix for the heat equation.

   Input Parameters:
   ts - the TS context
   t - current time
   global_in - global input vector
   dummy - optional user-defined context, as set by TSetRHSJacobian()

   Output Parameters:
   AA - Jacobian matrix
   BB - optionally different preconditioning matrix
   str - flag indicating matrix structure

   Notes:
   Recall that MatSetValues() uses 0-based row and column numbers
   in Fortran as well as in C.
*/
PetscErrorCode RHSMatrixHeatgll(TS ts,PetscReal t,Vec X,Mat *AA,Mat BB,void *ctx)
{
  Mat            A;                /* Jacobian matrix */
  PetscReal      **temp, p;
  PetscGLLIP     gll;
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  Mat            K, B;
  PetscInt       N=appctx->Nl;
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    v; 
  PetscInt       rows[2];
  PetscViewer    viewfile;
  
       /*
       Creates the element stiffness matrix for the given gll
    */
   ierr = PetscGLLIPCreate(N,PETSCGLLIP_VIA_LINEARALGEBRA,&gll);CHKERRQ(ierr);
   ierr = PetscGLLIPElementStiffnessCreate(&gll,&temp);CHKERRQ(ierr);
   ierr = MatCreateSeqDense(PETSC_COMM_SELF,N,N,&temp[0][0],&K);CHKERRQ(ierr);

    
   MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);

     /*
       Creates the element mass matrix for the given gll
    */
   ierr = MatCreateAIJ(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,N,N,1,NULL,0,NULL,&B);CHKERRQ(ierr);
   for (i=0; i<N; i++) {
       v=-1.0/appctx->W[i]; //appctx->W[i];
       MatSetValues(B,1,&i,1,&i,&v,INSERT_VALUES);
       }
   
   MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);
  
    
   // one needs mappings to ref element to proceed further 
   
   MatMatMult(B,K,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A);


   /* 
       Set BCs 
    */
   rows[0] = 0;
   rows[1] = N-1;
   ierr = MatZeroRowsColumns(A,2,rows,0.0,appctx->solution,appctx->solution);CHKERRQ(ierr);


   MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
   //MatView(A,PETSC_VIEWER_STDOUT_WORLD);

  
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"inside.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    //ierr = MatView(K,viewfile);CHKERRQ(ierr);
    //err = MatView(B,viewfile);CHKERRQ(ierr);
    ierr = MatView(A,viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewfile);CHKERRQ(ierr);

   ierr = PetscGLLIPElementStiffnessDestroy(&gll,&temp);CHKERRQ(ierr);
   //ierr=MatCopy(B,A,SAME_NONZERO_PATTERN);
   ierr = MatDestroy(&K);CHKERRQ(ierr);
   ierr = MatDestroy(&B);CHKERRQ(ierr);
   *AA=A;
     /*
     Set and option to indicate that we will never add a new nonzero location
     to the matrix. If we do, it will generate an error.
  */
  //ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  return 0;
}


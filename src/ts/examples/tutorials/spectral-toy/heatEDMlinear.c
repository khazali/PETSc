
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
//./heatNlinear -ts_monitor -ts_view -log_summary -ts_exact_final_time matchstep -ts_type beuler -ts_exact_final_time matchstep -ts_dt 1e-2 -Nl 40
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
#include "petscvec.h" 
#include "petscgll.h"
#include <petscdraw.h>
#include <petscdmda.h>
/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/
typedef struct {
  Vec         solution;          /* global exact solution vector */
  DM          da;                /* distributed array data structure */
  Vec         localwork;         /* local ghosted work vector */
  Vec         u_local;           /* local ghosted approximate solution vector */
  Vec         grid;              /* total grid */   
  Vec         mass;              /* mass matrix for total integration */
  PetscInt    Nl;                 /* total number of grid points */
  PetscInt    E;                 /* number of elements */
  PetscReal   *Z;                 /* mesh grid */
  PetscReal   *mult;                 /* multiplicity*/
  PetscScalar *W;                 /* weights */
  PetscBool   debug;             /* flag (1 indicates activation of debugging printouts) */
  PetscViewer viewer1,viewer2;  /* viewers for the solution and error */
  PetscReal   norm_2,norm_max,norm_L2;  /* error norms */
} AppCtx;
/*
   User-defined routines
*/
extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode RHSMatrixHeatgllDM(TS,PetscReal,Vec,Mat*,Mat,void*);
extern PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void*);
extern PetscErrorCode ExactSolution(PetscReal,Vec,AppCtx*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  TS             ts;                     /* timestepping context */
  Mat            A;                      /* matrix data structure */
  Vec            u;                      /* approximate solution vector */
  PetscReal      time_total_max = 0.0001; /* default max total time */
  PetscInt       time_steps_max = 100;   /* default max timesteps */
  PetscErrorCode ierr;
  PetscInt       steps,Nl=15,i, E=5, xs, xm, ind, j, lenglob;
  PetscMPIInt    size;
  PetscReal      dt=1e-3, x, *wrk_ptr1, *wrk_ptr2, L=2.0, Le;
  //PetscBool      flg;
  PetscGLLIP     gll;
  PetscViewer    viewfile;
   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBegin;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  ierr = PetscOptionsGetInt(NULL,NULL,"-Nl",&Nl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-E",&E,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-debug",&appctx.debug);CHKERRQ(ierr);

  appctx.Nl       = Nl;
  appctx.E        = E;
  Le=L/appctx.E;
  appctx.norm_2   = 0.0;
  appctx.norm_max = 0.0;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Solving a linear TS problem on 1 processor\n");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscGLLIPCreate(Nl,PETSCGLLIP_VIA_LINEARALGEBRA,&gll);CHKERRQ(ierr);
  
  ierr= PetscMalloc1(Nl, &appctx.Z);
  ierr= PetscMalloc1(Nl, &appctx.W);
  ierr= PetscMalloc1(Nl, &appctx.mult);

  
  for(i=0; i<Nl; i++)
     { 
     appctx.Z[i]=(gll.nodes[i]+1.0);
     appctx.W[i]=gll.weights[i];
     appctx.mult[i]=1.0;
      }

  appctx.mult[0]=0.5;
  appctx.mult[Nl-1]=0.5; 

  //lenloc   = E*Nl;
  lenglob  = E*(Nl-1)+1;

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
     and to set up the ghost point communication pattern.  There are E*(Nl-1)+1
     total grid values spread equally among all the processors, except first and last
  */

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,lenglob,1,1,NULL,&appctx.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(appctx.da);CHKERRQ(ierr);
  ierr = DMSetUp(appctx.da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(appctx.da,NULL,&E,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
 
  /*
     Extract global and local vectors from DMDA; we use these to store the
     approximate solution.  Then duplicate these for remaining vectors that
     have the same types.
  */
  ierr = DMCreateGlobalVector(appctx.da,&u);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(appctx.da,&appctx.u_local);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.solution);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.grid);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.mass);CHKERRQ(ierr);
 
  ierr = DMDAGetCorners(appctx.da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(appctx.da,appctx.grid,&wrk_ptr1);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx.da,appctx.mass,&wrk_ptr2);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
    xs=xs/(appctx.Nl-1);
    xm=xm/(appctx.Nl-1);
  
  /* 
     Build total grid and mass over entire mesh (multi-elemental) 
  */ 

   for (i=xs; i<xs+xm; i++) {
      for (j=0; j<appctx.Nl; j++)
      {
      x = (Le/2.0)*(appctx.Z[j])+Le*i; 
      ind=i*(appctx.Nl-1)+j;
      wrk_ptr1[ind]=x;
      wrk_ptr2[ind]=appctx.W[j];
      } 
  }

  ierr = DMDAVecRestoreArray(appctx.da,appctx.grid,&wrk_ptr1);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(appctx.da,appctx.mass,&wrk_ptr2);CHKERRQ(ierr);
 
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);
  ierr = TSSetDM(ts,appctx.da);CHKERRQ(ierr);
 /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set optional user-defined monitoring routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSMonitorSet(ts,Monitor,&appctx,NULL);CHKERRQ(ierr);

 /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

     Create matrix data structure; set matrix evaluation routine.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 
      /*
       For linear problems with a time-dependent f(u,t) in the equation
       u_t = f(u,t), the user provides the discretized right-hand-side
       as a time-dependent matrix.
    */

  ierr = RHSMatrixHeatgllDM(ts,0.0,u,&A,A,&appctx);CHKERRQ(ierr);

  ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,A,A,TSComputeRHSJacobianConstant,&appctx);CHKERRQ(ierr);
  
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
  ierr = TSSetDM(ts,appctx.da);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solution vector and initial timestep
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  ierr = InitialConditions(u,&appctx);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Initial cond");CHKERRQ(ierr);
  ierr = VecView(u,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

 /*
     Run the timestepping solver
  */
  ierr = TSSolve(ts,u);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     View timestepping solver info
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscPrintf(PETSC_COMM_SELF,"avg. error (2 norm) = %g, avg. error (max norm) = %g\n",(double)(appctx.norm_2/steps),(double)(appctx.norm_L2/steps));CHKERRQ(ierr);
  ierr = TSView(ts,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     For matlab output
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"solution.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = VecView(u,viewfile);CHKERRQ(ierr);
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
    ierr = DMDestroy(&appctx.da);CHKERRQ(ierr);

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
  PetscInt       i,j,ind,xs,xm;
  PetscReal      x,xx,Le;

  /*
    Get a pointer to vector data.
    - For default PETSc vectors, VecGetArray() returns a pointer to
      the data array.  Otherwise, the routine is implementation dependent.
    - You MUST call VecRestoreArray() when you no longer need access to
      the array.
    - Note that the Fortran interface to VecGetArray() differs from the
      C version.  See the users manual for details.
  */
  
  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArray(appctx->da,u,&u_localptr);CHKERRQ(ierr);
  
  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */

    Le=2.0/appctx->E;
    xs=xs/(appctx->Nl-1);
    xm=xm/(appctx->Nl-1);
    //I could also apply this to the entire grid as a function using PF

   for (i=xs; i<xs+xm; i++) {
      for (j=0; j<appctx->Nl; j++)
      {
      x = (Le/2.0)*(appctx->Z[j])+Le*i; 
      xx= PetscSinScalar(PETSC_PI*6.*x) + 3.*PetscSinScalar(PETSC_PI*2.*x);
      ind=i*(appctx->Nl-1)+j;
      u_localptr[ind]=xx;
      } 
  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(appctx->da,u,&u_localptr);CHKERRQ(ierr);

  //minor test.. to be removed
  ierr = PetscPrintf(PETSC_COMM_SELF,"Initial cond inside routine \n");CHKERRQ(ierr);
  ierr = VecView(u,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  /*
     Print debugging information if desired
  */
  if (appctx->debug) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Initial guess vector\n");CHKERRQ(ierr);
    ierr = VecView(u,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
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
  PetscReal      xx, x, Le;
  PetscInt       i, xs, xm, j, ind;

  
  /*
     Simply write the solution directly into the array locations.
     Alternatively, we culd use VecSetValues() or VecSetValuesLocal().
  */
  ex1 = PetscExpScalar(-36.*PETSC_PI*PETSC_PI*tc);
  ex2 = PetscExpScalar(-4.*PETSC_PI*PETSC_PI*tc);
  sc1 = PETSC_PI*6.0;                 
  sc2 = PETSC_PI*2.0;
 

  ierr = DMDAVecGetArray(appctx->da,solution,&s_localptr);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
    xs=xs/(appctx->Nl-1);
    xm=xm/(appctx->Nl-1);

    Le=2.0/appctx->E;
    //xg = []; for e=1:E xg=[xg; x(1:end-1)+Le*(e-1)]; end;

   for (i=xs; i<xs+xm; i++) {
      for (j=0; j<appctx->Nl; j++)
      {
      x = (Le/2.0)*(appctx->Z[j])+Le*i; 
      xx= PetscSinScalar(sc1*x)*ex1 + 3.*PetscSinScalar(sc2*x)*ex2;
      ind=i*(appctx->Nl-1)+j;
      s_localptr[ind]=xx;
      } 
  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(appctx->da,solution,&s_localptr);CHKERRQ(ierr);
  
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
  PetscReal      norm_2,norm_max,dt,dttol,norm_L2;
  Vec            f;
  
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
     Compute the L2-norm and max-norm of the error
  */
  ierr   = VecAXPY(appctx->solution,-1.0,u);CHKERRQ(ierr);
  ierr   = VecDuplicate(appctx->solution,&f);CHKERRQ(ierr);
 
  ierr   = VecPointwiseMult(f,appctx->solution,appctx->solution);CHKERRQ(ierr);
  ierr   = VecDot(f,appctx->mass,&norm_L2);CHKERRQ(ierr);


  ierr   = VecNorm(appctx->solution,NORM_2,&norm_2);CHKERRQ(ierr);
  ierr   = VecNorm(appctx->solution,NORM_MAX,&norm_max);CHKERRQ(ierr);
  if (norm_L2   < 1e-14) norm_L2   = 0;
  if (norm_2   < 1e-14) norm_2   = 0;
  if (norm_max < 1e-14) norm_max = 0;

  printf(" norm L2 %f \n", norm_L2);
  norm_L2=sqrt(norm_L2);
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Timestep %3D: step size = %g, time = %g, L2-norm error = %g, max norm error = %g\n",step,(double)dt,(double)time,(double)norm_L2,(double)norm_max);CHKERRQ(ierr);

  appctx->norm_2   += norm_2;
  appctx->norm_max += norm_max;
  appctx->norm_L2  += norm_L2;

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
#define __FUNCT__ "RHSMatrixHeatgllDM"

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
PetscErrorCode RHSMatrixHeatgllDM(TS ts,PetscReal t,Vec X,Mat *AA,Mat BB,void *ctx)
{
  Mat            A;                /* Jacobian matrix */
  PetscReal      **temp, Le;
  PetscGLLIP     gll;
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  Mat            K, B;
  PetscInt       N=appctx->Nl;
  PetscInt       E=appctx->E;
  PetscErrorCode ierr;
  PetscInt       i,xs,xn,l,id,j;
  PetscScalar    v; 
  PetscInt       rows[2], *rowsDM;
  PetscViewer    viewfile;
  
  Le=2.0/E; // this should be in the appctx, but I think I need a new struct only for grid info

    /*
       Creates the element stiffness matrix for the given gll
    */
   ierr = PetscGLLIPCreate(N,PETSCGLLIP_VIA_LINEARALGEBRA,&gll);CHKERRQ(ierr);
   ierr = PetscGLLIPElementStiffnessCreate(&gll,&temp);CHKERRQ(ierr);

    /*
        Create the global stiffness matrix and add the element stiffness for each local element
    */
    ierr = DMCreateMatrix(appctx->da,&K);CHKERRQ(ierr);
    ierr = MatSetOption(K,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL);CHKERRQ(ierr);

    xs   = xs/(N-1);
    xn   = xn/(N-1);

    ierr = PetscMalloc1(N,&rowsDM);CHKERRQ(ierr);

    /*
        loop over local elements
    */
    for (j=xs; j<xs+xn; j++) {
      for (l=0; l<N; l++) 
          {rowsDM[l] = j*(N-1)+l;
           }
      ierr = MatSetValues(K,N,rowsDM,N,rowsDM,&temp[0][0],ADD_VALUES);CHKERRQ(ierr);
    }

   MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY); 

   /*
       Creates the element mass matrix for the given gll
   */
    ierr = DMCreateMatrix(appctx->da,&B);CHKERRQ(ierr);
    ierr = MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL);CHKERRQ(ierr);
    xs   = xs/(N-1);
    xn   = xn/(N-1);
  
    for (j=xs; j<xs+xn; j++) {
     for (i=0; i<N; i++) {
       v=-4.0/(Le*Le)*(appctx->mult[i]/appctx->W[i]); //note here I took the multiplicities in
       id=j*(N-1)+i;
       MatSetValues(B,1,&id,1,&id,&v,INSERT_VALUES);
      
      } 
     }

   MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); 

   //matdiagonal scale would be more suitable
   ierr = MatDuplicate(K,MAT_DO_NOT_COPY_VALUES,&A);CHKERRQ(ierr); 
   MatMatMult(B,K,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A);
   /* 
       Set BCs 
    */
   rows[0] = 0;
   rows[1] = E*(N-1);
   ierr = MatZeroRowsColumns(A,2,rows,0.0,appctx->solution,appctx->solution);CHKERRQ(ierr);

   MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
  
   // Output only for testing
   ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"inside.m",&viewfile);CHKERRQ(ierr);
   ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
   ierr = MatView(K,viewfile);CHKERRQ(ierr);
   ierr = MatView(B,viewfile);CHKERRQ(ierr);
   ierr = MatView(A,viewfile);CHKERRQ(ierr);
   ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
   ierr = PetscViewerDestroy(&viewfile);CHKERRQ(ierr);

   ierr = PetscGLLIPElementStiffnessDestroy(&gll,&temp);CHKERRQ(ierr);
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


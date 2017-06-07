
static char help[] ="Solves a simple time-dependent linear PDE (the Advection equation).\n\
Input parameters include:\n\
  -m <points>, where <points> = number of grid points\n\
  -time_dependent_rhs : Treat the problem as having a time-dependent right-hand side\n\
  -debug              : Activate debugging printouts\n\
  -nox                : Deactivate x-window graphics\n\n";

/*
   Concepts: TS^time-dependent linear problems
   Concepts: TS^Advection equation
   Concepts: TS^diffusion equation
   Processors: 1
*/
// Run with
//./AdvectionEDMlinear -ts_monitor -ts_view -log_summary -ts_exact_final_time matchstep -ts_type beuler -ts_exact_final_time matchstep -ts_dt 1e-2 -Nl 20
/* ------------------------------------------------------------------------

   This program solves the one-dimensional Advection equation (also called the
   diffusion equation),
       u_t = u_xx,
   on the domain 0 <= x <= 1, with the boundary conditions
       u(t,0) = 0, u(t,1) = 0,
   This is a linear, second-order, parabolic equation.

   We discretize the right-hand side using the spectral element method
   We then demonstrate time evolution using the various TS methods by
   running the program via AdvectionEDMlinear -ts_type <timestepping solver>

   Adjoints version 
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
#include <petscmath.h>
#include <petscdmda.h>
#include <petscmat.h>  
#include <petsctao.h>   
#include "petscdmlabel.h" 
#include "stdio.h"
/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/

typedef struct {
  PetscInt    N;             /* grid points per elements*/
  PetscInt    E;              /* number of elements */
  PetscReal   tol_L2,tol_max; /* error norms */
  PetscInt    steps;          /* number of timesteps */
  PetscReal   Tend;           /* endtime */
  PetscReal   mu;             /* viscosity */
  PetscReal   dt;             /* timestep*/
  PetscReal   L;              /* total length of domain */   
  PetscReal   Le; 
  PetscReal   Tadj;
} PetscParam;

typedef struct {
  Vec         obj;               /* desired end state */
  Vec         u_local;           /* local ghosted approximate solution vector */
  Vec         grad;
  Vec         ic;
  Vec         curr_sol;
  PetscReal   *Z;                 /* mesh grid */
  PetscReal   *mult;              /* multiplicity*/
  PetscScalar *W;                 /* weights */
} PetscData;

typedef struct {
  Vec         u_local;           /* local ghosted approximate solution vector */
  Vec         grid;              /* total grid */   
  Vec         mass;              /* mass matrix for total integration */
  Vec         massinv;              /* mass matrix for total integration */
  Mat         stiff;             // stifness matrix
  Mat         grad;             // stifness matrix
  Mat         adj;           //adjoint jacobian    
  PetscGLLIP  gll;
} PetscSEMOperators;

typedef struct {
  DM                da;                /* distributed array data structure */
  PetscSEMOperators SEMop;
  PetscParam        param;
  PetscData         dat;
  PetscBool         debug;
  TS                ts;
  PetscReal         initial_dt;
  PetscBool         AdjTestFD;
  PetscBool         AdjTestLinear;  /* just test if the adjoint is consistent (works only for leanr problems becasue we don't have the damn TLM in PETSc*/
  PetscReal         AdjTestLinearForward;
  PetscReal         AdjTestLinearBackward;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);
extern PetscErrorCode RHSMatrixAdvectiongllDM(TS,PetscReal,Vec,Mat*,Mat,void*);
extern PetscErrorCode RHSAdjointgllDM(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSFunctionAdvection(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode Objective(PetscReal,Vec,AppCtx*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  Tao            tao;
  KSP            ksp;
  PC             pc;
  Mat            A;                      /* matrix data structure */
  Vec            u;                      /* approximate solution vector */
  PetscErrorCode ierr;
  PetscInt       i, xs, xm, ind, j, lenglob;
  PetscInt       VecLength;
  PetscMPIInt    size;
  PetscReal      x, *wrk_ptr1, *wrk_ptr2, *wrk_ptr3, wrk1, wrk2,varepsilon;
  Vec            wrk_vec,wrk1_vec,wrk2_vec,wrk3_vec,wrk4_vec;
  PetscViewer    viewfile;
     /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBegin;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");
 

  appctx.AdjTestLinear=PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_adjoint_linear_problem",&appctx.AdjTestLinear,NULL);CHKERRQ(ierr);
  appctx.AdjTestFD=PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_adjoint_finite_difference",&appctx.AdjTestFD,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&appctx.param.N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-E",&appctx.param.E,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-debug",&appctx.debug);CHKERRQ(ierr);

  /*initialize parameters */ 
  appctx.param.N  = 10;
  appctx.param.E  = 4;
  appctx.param.L  = 1.0;
  appctx.param.Le = appctx.param.L/appctx.param.E;

  appctx.param.mu    = 0.001; 

  appctx.param.steps =20000000;
  appctx.initial_dt    = 1e-3;

  appctx.param.Tend = 0.001; //appctx.param.steps*appctx.param.dt;  
  appctx.param.Tadj =appctx.param.Tend+0.7;

  //ierr = PetscPrintf(PETSC_COMM_WORLD,"Solving a linear TS problem on 1 processor\n");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscGLLIPCreate(appctx.param.N,PETSCGLLIP_VIA_LINEARALGEBRA,&appctx.SEMop.gll);CHKERRQ(ierr);
  //ierr = PetscGLLIPCreate(appctx.param.N,PETSCGLLIP_VIA_NEWTON,&appctx.SEMop.gll);CHKERRQ(ierr);
  
  ierr = PetscMalloc1(appctx.param.N, &appctx.dat.Z);
  ierr = PetscMalloc1(appctx.param.N, &appctx.dat.W);
  ierr = PetscMalloc1(appctx.param.N, &appctx.dat.mult);

  for(i=0; i<appctx.param.N; i++)
     { 
     appctx.dat.Z[i]=(appctx.SEMop.gll.nodes[i]+1.0);
     appctx.dat.W[i]=appctx.SEMop.gll.weights[i]; 
     appctx.dat.mult[i]=1.0;
      }

  appctx.dat.mult[0]=0.5;
  appctx.dat.mult[appctx.param.N-1]=0.5; 

  //lenloc   = appctx.param.E*appctx.param.N; //only if I want to do it totally local for explicit
  lenglob  = appctx.param.E*(appctx.param.N-1)+1;

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
     and to set up the ghost point communication pattern.  There are E*(Nl-1)+1
     total grid values spread equally among all the processors, except first and last
  */

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,lenglob,1,1,NULL,&appctx.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(appctx.da);CHKERRQ(ierr);
  ierr = DMSetUp(appctx.da);CHKERRQ(ierr);
  //ierr = DMDAGetInfo(appctx.da,NULL,&E,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
 
  /*
     Extract global and local vectors from DMDA; we use these to store the
     approximate solution.  Then duplicate these for remaining vectors that
     have the same types.
  */

  ierr = DMCreateGlobalVector(appctx.da,&u);CHKERRQ(ierr);
  //ierr = DMCreateLocalVector(appctx.da,&appctx.u_local);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.dat.ic);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.dat.obj);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.dat.grad);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.SEMop.grid);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.SEMop.mass);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.SEMop.massinv);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.dat.curr_sol);CHKERRQ(ierr);
 
  ierr = DMDAGetCorners(appctx.da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(appctx.da,appctx.SEMop.grid,&wrk_ptr1);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx.da,appctx.SEMop.mass,&wrk_ptr2);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx.da,appctx.SEMop.massinv,&wrk_ptr3);CHKERRQ(ierr);
  //Compute function over the locally owned part of the grid
  
    xs=xs/(appctx.param.N-1);
    xm=xm/(appctx.param.N-1);
  
  /* 
     Build total grid and mass over entire mesh (multi-elemental) 
  */ 

  for (i=xs; i<xs+xm; i++) {
      for (j=0; j<appctx.param.N; j++)
      {
      x = (appctx.param.Le/2.0)*(appctx.dat.Z[j])+appctx.param.Le*i; 
      ind=i*(appctx.param.N-1)+j;
      wrk_ptr1[ind]=x;
      wrk_ptr2[ind]=appctx.param.Le/2.0*appctx.dat.W[j]*appctx.dat.mult[j];
      if (j==0 || j==appctx.param.N)
             {wrk_ptr2[ind]=appctx.param.Le/2.0*appctx.dat.W[j];}
      wrk_ptr3[ind]=1./wrk_ptr2[ind];
      } 
   }

   ierr = DMDAVecRestoreArray(appctx.da,appctx.SEMop.grid,&wrk_ptr1);CHKERRQ(ierr);
   ierr = DMDAVecRestoreArray(appctx.da,appctx.SEMop.mass,&wrk_ptr2);CHKERRQ(ierr);
   ierr = DMDAVecRestoreArray(appctx.da,appctx.SEMop.massinv,&wrk_ptr3);CHKERRQ(ierr);

   //Set Objective and Initial conditions for the problem 
   ierr = Objective(appctx.param.Tadj,appctx.dat.obj,&appctx);CHKERRQ(ierr);
   ierr = InitialConditions(appctx.dat.ic,&appctx);CHKERRQ(ierr);

  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"IC_OBJ.m",&viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx.dat.ic,"ic");
  ierr = VecView(appctx.dat.ic,viewfile);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx.SEMop.grid,"xg");
  ierr = VecView(appctx.SEMop.grid,viewfile);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx.dat.obj,"obj");
  ierr = VecView(appctx.dat.obj,viewfile);CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

   ierr = TSCreate(PETSC_COMM_WORLD,&appctx.ts);CHKERRQ(ierr);
   ierr = TSSetProblemType(appctx.ts,TS_LINEAR);CHKERRQ(ierr);
   ierr = TSSetType(appctx.ts,TSRK);CHKERRQ(ierr);
   ierr = TSSetDM(appctx.ts,appctx.da);CHKERRQ(ierr);
   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set time
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = TSSetTime(appctx.ts,0.0);CHKERRQ(ierr);
    ierr = TSSetInitialTimeStep(appctx.ts,0.0,appctx.initial_dt);CHKERRQ(ierr);
    ierr = TSSetDuration(appctx.ts,appctx.param.steps,appctx.param.Tend);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(appctx.ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

    ierr = TSSetTolerances(appctx.ts,1e-7,NULL,1e-7,NULL);CHKERRQ(ierr);
    ierr = TSSetFromOptions(appctx.ts);
    /* Need to save initial timestep user may have set with -ts_dt so it can be reset for each new TSSolve() */
    ierr = TSGetTimeStep(appctx.ts,&appctx.initial_dt);CHKERRQ(ierr);
   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set matrix evaluation routine.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

   ierr = DMSetMatrixPreallocateOnly(appctx.da, PETSC_TRUE);CHKERRQ(ierr);
   ierr = DMCreateMatrix(appctx.da,&A);CHKERRQ(ierr);
   ierr = DMCreateMatrix(appctx.da,&appctx.SEMop.grad);CHKERRQ(ierr);
    
    /*
       For linear problems with a time-dependent f(u,t) in the equation
       u_t = f(u,t), the user provides the discretized right-hand-side
       as a time-dependent matrix.
    */
   ierr = RHSMatrixAdvectiongllDM(appctx.ts,0.0,u,&A,A,&appctx);CHKERRQ(ierr);
   ierr = MatDuplicate(A,MAT_COPY_VALUES,&appctx.SEMop.grad);CHKERRQ(ierr);
   ierr = MatScale(A, 1.0);
   ierr = MatDuplicate(A,MAT_COPY_VALUES,&appctx.SEMop.adj);CHKERRQ(ierr);

   //exit(1);

  ierr = TSSetRHSFunction(appctx.ts,NULL,TSComputeRHSFunctionLinear,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(appctx.ts,appctx.SEMop.grad,appctx.SEMop.grad,TSComputeRHSJacobianConstant,&appctx);CHKERRQ(ierr);
  //ierr = PetscPrintf(PETSC_COMM_SELF,"avg. error (2 norm) = %g, avg. error (max norm) = %g\n",(double)(appctx.norm_2/steps),(double)(appctx.norm_L2/steps));CHKERRQ(ierr);
  //ierr = TSView(ts,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
 
  // Create TAO solver and set desired solution method 
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOCG);CHKERRQ(ierr);

  ierr = TaoSetInitialVector(tao,appctx.dat.ic);CHKERRQ(ierr);

  // Set routine for function and gradient evaluation 
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void *)&appctx);CHKERRQ(ierr);

  // Check for any TAO command line options 
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  ierr = TaoGetKSP(tao,&ksp);CHKERRQ(ierr);
  if (ksp) {
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  }

  ierr = TaoSetTolerances(tao,1e-8,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

  if(!appctx.AdjTestLinear && !appctx.AdjTestFD) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Solving the optimization problem:\n");CHKERRQ(ierr);
    ierr = TaoSolve(tao); CHKERRQ(ierr);
  }
  if (appctx.AdjTestLinear){
    ierr = PetscPrintf(PETSC_COMM_SELF,"Testing the adjoint of the linear problem:\n");CHKERRQ(ierr);
    ierr = VecDuplicate(appctx.dat.ic,&wrk_vec); CHKERRQ(ierr);
    ierr = VecCopy(appctx.dat.ic,wrk_vec); CHKERRQ(ierr);
    ierr = VecDuplicate(appctx.dat.ic,&wrk2_vec); CHKERRQ(ierr);
    ierr = FormFunctionGradient(tao,wrk_vec,&wrk1,wrk2_vec,&appctx);CHKERRQ(ierr);
    ierr = VecDot(appctx.dat.ic,wrk2_vec,&(appctx.AdjTestLinearBackward));CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"Cost=%g, forward test = %g = %g = backward test \n",wrk1,appctx.AdjTestLinearForward,appctx.AdjTestLinearBackward);
    ierr = VecDestroy(&wrk_vec); CHKERRQ(ierr);
    ierr = VecDestroy(&wrk2_vec); CHKERRQ(ierr);
  }
  if (appctx.AdjTestFD){
    appctx.AdjTestLinear = PETSC_FALSE;
    ierr = PetscPrintf(PETSC_COMM_SELF,"Testing the adjoint by using finite differences:\n");CHKERRQ(ierr);
    ierr = VecDuplicate(appctx.dat.ic,&wrk_vec); CHKERRQ(ierr);
    ierr = VecCopy(appctx.dat.ic,wrk_vec); CHKERRQ(ierr);
    ierr = VecDuplicate(appctx.dat.ic,&wrk1_vec); CHKERRQ(ierr);
    ierr = VecDuplicate(appctx.dat.ic,&wrk2_vec); CHKERRQ(ierr);
    ierr = VecDuplicate(appctx.dat.ic,&wrk3_vec); CHKERRQ(ierr);
    ierr = VecDuplicate(appctx.dat.ic,&wrk4_vec); CHKERRQ(ierr);

    ierr = FormFunctionGradient(tao,wrk_vec,&wrk1,wrk2_vec,&appctx);CHKERRQ(ierr);
    /* Note computed gradient is in wrk2_vec, original cost is in wrk1 */
    ierr = VecZeroEntries(wrk3_vec);
    ierr = VecGetSize(wrk_vec,&VecLength); CHKERRQ(ierr);
    varepsilon = 1e-05;
    for (i=0; i<VecLength; i++) {
      ierr = VecCopy(appctx.dat.ic,wrk_vec); CHKERRQ(ierr); //reset J(eps) for each point
      VecSetValue(wrk_vec,i, varepsilon,INSERT_VALUES);
      ierr = FormFunctionGradient(tao,wrk_vec,&wrk2,wrk4_vec,&appctx);CHKERRQ(ierr);
      /* Note original cost is in wrk1, perturbed cost in wrk2 */
      VecSetValue(wrk3_vec,i,(PetscScalar)((wrk2-wrk1)/varepsilon),INSERT_VALUES);
      VecSetValue(wrk1_vec,i,(PetscScalar)(wrk2),INSERT_VALUES);
    }
    PetscPrintf(PETSC_COMM_WORLD,"Cost original J=%g \n",wrk1);
    /* Note finite difference gradient is in wrk3_vec */
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"fd.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    //ierr = PetscObjectSetName((PetscObject)wrk2_vec,"gradj");
    //ierr = VecView(wrk2_vec,viewfile);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)wrk3_vec,"gradj");
    ierr = VecView(wrk3_vec,viewfile);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)wrk1_vec,"Jeps");
    ierr = VecView(wrk1_vec,viewfile);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)wrk2_vec,"J");
    ierr = VecView(wrk2_vec,viewfile);CHKERRQ(ierr);
    ierr=PetscViewerPopFormat(viewfile);

    ierr = VecDestroy(&wrk_vec); CHKERRQ(ierr);
    ierr = VecDestroy(&wrk2_vec); CHKERRQ(ierr);
    ierr = VecDestroy(&wrk3_vec); CHKERRQ(ierr);
    ierr = VecDestroy(&wrk4_vec); CHKERRQ(ierr);
    exit(1); 
  }

  // Free TAO data structures 
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);

  ierr = TSDestroy(&appctx.ts);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.SEMop.grad);CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.SEMop.adj);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.obj);CHKERRQ(ierr);
  ierr = PetscGLLIPDestroy(&appctx.SEMop.gll);CHKERRQ(ierr);
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
  PetscScalar    *s_localptr, *xg_localptr;
  PetscErrorCode ierr;
  PetscInt       i,lenglob;

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
  ierr = DMDAVecGetArray(appctx->da,u,&s_localptr);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx->da,appctx->SEMop.grid,&xg_localptr);CHKERRQ(ierr);

  lenglob  = appctx->param.E*(appctx->param.N-1)+1;
  /*
  for (i=0; i<lenglob; i++) {
      s_localptr[i]=PetscSinScalar(2.0*PETSC_PI*xg_localptr[i]);
      } 
   */
  for (i=0; i<lenglob; i++) {
      s_localptr[i]=PetscSinScalar(2.0*PETSC_PI*xg_localptr[i]);
      }

//printf("sinfunc %2.20f \n",PetscSinScalar(2.0*PETSC_PI*xg_localptr[lenglob])*PetscExpScalar(-0.4*tc));

  //Restore vectors

  ierr = DMDAVecRestoreArray(appctx->da,appctx->SEMop.grid,&xg_localptr);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(appctx->da,appctx->dat.ic,&s_localptr);CHKERRQ(ierr);
  
  //  Print debugging information if desired
  
  if (appctx->debug) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Initial guess vector\n");CHKERRQ(ierr);
    ierr = VecView(u,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  return 0;
}
/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "Objective"
/*
   Sets the profile at end time

   Input Parameters:
   t - current time
   obj - vector storing the end function
   appctx - user-defined application context

   Output Parameter:
   solution - vector with the newly computed exact solution
*/
PetscErrorCode Objective(PetscReal t,Vec obj,AppCtx *appctx)
{
  PetscScalar    *s_localptr,*xg_localptr;
  PetscErrorCode ierr;
  PetscInt       i, lenglob;
  
  /*
     Simply write the solution directly into the array locations.
     Alternatively, we culd use VecSetValues() or VecSetValuesLocal().
  */
  
  ierr = DMDAVecGetArray(appctx->da,obj,&s_localptr);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx->da,appctx->SEMop.grid,&xg_localptr);CHKERRQ(ierr);

  lenglob  = appctx->param.E*(appctx->param.N-1)+1;
  /*
  for (i=0; i<lenglob; i++) {
      s_localptr[i]=*PetscExpScalar(-0.4*tc);
      } 
  */
  for (i=0; i<lenglob; i++) {
      s_localptr[i]=PetscSinScalar(2.0*PETSC_PI*(xg_localptr[i]-0.2));
      } 


/*
     Restore vectors
*/
  ierr = DMDAVecRestoreArray(appctx->da,appctx->SEMop.grid,&xg_localptr);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(appctx->da,appctx->dat.obj,&s_localptr);CHKERRQ(ierr);

  return 0;
}


/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "RHSMatrixAdvectiongllDM"

/*
   RHSMatrixAdvection - User-provided routine to compute the right-hand-side
   matrix for the Advection equation.

   Input Parameters:
   ts - the TS context
   t - current time
   global_in - global input vector
   dummy - optional user-defined context, as set by TSetRHSJacobian()

   Output Parameters:
   AA - Jacobian matrix
   BB - optionally different preconditioning matrix
   str - flag indicating matrix structure

*/
PetscErrorCode RHSMatrixAdvectiongllDM(TS ts,PetscReal t,Vec X,Mat *AA,Mat BB,void *ctx)
{
  PetscReal      **temp, init;
  PetscReal      vv;
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  PetscErrorCode ierr;
  PetscInt       i,xs,xn,l,j,id1,id2, N=appctx->param.N;
  PetscInt       *rowsDM;
   PetscViewer    viewfile;
   Mat            K,Q,A;
     /*
       Creates the advection matrix for the given gll
    */
    ierr = PetscGLLIPElementAdvectionCreate(&appctx->SEMop.gll,&temp);CHKERRQ(ierr);

/*
    // scale by the mass matrix
    for (i=0; i<appctx->param.N; i++) 
    {
       vv=2.0/appctx->param.Le*appctx->dat.mult[i]*1.0/appctx->dat.W[i]; //note here I took the multiplicities in
       for (j=0; j<appctx->param.N; j++)
               {
                temp[i][j]=temp[i][j]*vv; 
               }
      } 
*/
    ierr = DMCreateMatrix(appctx->da,&K);CHKERRQ(ierr);
    ierr = MatSetOption(K,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL);CHKERRQ(ierr);

    xs   = xs/(appctx->param.N-1);
    xn   = xn/(appctx->param.N-1);

    ierr = PetscMalloc1(appctx->param.N,&rowsDM);CHKERRQ(ierr);
    /*
        loop over local elements
    */
    for (j=xs; j<xs+xn; j++) {
      for (l=0; l<appctx->param.N; l++) 
          {rowsDM[l] = j*(appctx->param.N-1)+l;
           }

      ierr = MatSetValues(K,appctx->param.N,rowsDM,appctx->param.N,rowsDM,&temp[0][0],ADD_VALUES);CHKERRQ(ierr);
    }

   MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);

   MatDiagonalScale(K,appctx->SEMop.massinv,NULL);

  /*
       Creates the element BC matrix for the given gll
    */
  
    ierr = DMCreateMatrix(appctx->da,&Q);CHKERRQ(ierr);
    ierr = MatSetOption(Q,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL);CHKERRQ(ierr);
    xs   = xs/(N-1);
    xn   = xn/(N-1);
  
    for (j=xs; j<xs+xn; j++) {
     for (i=0; i<N; i++) {
       vv=appctx->dat.mult[i]; //note here I took the multiplicities in
       id1=j*(N-1)+i;
       MatSetValues(Q,1,&id1,1,&id1,&vv,ADD_VALUES);
      } 
     }
   vv=0.25; 
   id1=0; id2=appctx->param.E*(appctx->param.N-1);
   ierr = MatSetValues(Q,1,&id1,1,&id2,&vv,ADD_VALUES);CHKERRQ(ierr);
   ierr = MatSetValues(Q,1,&id2,1,&id1,&vv,ADD_VALUES);CHKERRQ(ierr);
   //ierr = MatSetValues(Q,1,&id1,1,&id1,&vv,ADD_VALUES);CHKERRQ(ierr);
   //ierr = MatSetValues(Q,1,&id2,1,&id2,&vv,ADD_VALUES);CHKERRQ(ierr);


   MatAssemblyBegin(Q,MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd(Q,MAT_FINAL_ASSEMBLY);
    
   // one needs mappings to ref element to proceed further 
   ierr = MatDuplicate(K,MAT_DO_NOT_COPY_VALUES,&A);CHKERRQ(ierr); 
   MatMatMult(Q,K,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A);

   MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"convection.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)A,"grad");
    ierr = MatView(A,viewfile);CHKERRQ(ierr);
    ierr=PetscViewerPopFormat(viewfile);
   
   ierr = PetscGLLIPElementAdvectionDestroy(&appctx->SEMop.gll,&temp);CHKERRQ(ierr);
   
     /*
     Set and option to indicate that we will never add a new nonzero location
     to the matrix. If we do, it will generate an error.
  */
  //ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  *AA=A;
  return 0;
}

/* --------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "RHSAdjointgllDM"

PetscErrorCode RHSAdjointgllDM(TS ts,PetscReal t,Vec X,Mat A,Mat BB,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
   
  //ierr=MatCopy(A,appctx->stiff,SAME_NONZERO_PATTERN);
  //ierr=MatScale(A,-1.0);
  A=appctx->SEMop.adj;
  
  return 0;
}
/* ------------------------------------------------------------------ */
#undef __FUNCT__
#define __FUNCT__ "FormFunctionGradient"
/*
   FormFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetObjectiveAndGradientRoutine()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormFunctionGradient(Tao tao,Vec IC,PetscReal *f,Vec G,void *ctx)
{
  AppCtx           *appctx = (AppCtx*)ctx;     /* user-defined application context */
  PetscErrorCode    ierr;
  Vec               temp, temp2;
  PetscInt          its;
  PetscReal         ff, gnorm, cnorm, xdiff; 
  TaoConvergedReason reason;      
  PetscViewer        viewfile;
  char filename[24] ;
  char data[60] ;


  ierr = TSSetInitialTimeStep(appctx->ts,0.0,appctx->initial_dt);CHKERRQ(ierr);
  ierr = VecCopy(IC,appctx->dat.curr_sol);CHKERRQ(ierr);
   
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Save trajectory of solution so that TSAdjointSolve() may be used
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   ierr = TSSetSaveTrajectory(appctx->ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(appctx->ts,appctx->dat.curr_sol);CHKERRQ(ierr);
  /*VecView(appctx->dat.curr_sol,PETSC_VIEWER_STDOUT_WORLD);*/
  if(appctx->AdjTestLinear){
    ierr = VecDot(appctx->dat.curr_sol,appctx->dat.curr_sol,&ff);CHKERRQ(ierr);
    appctx->AdjTestLinearForward=ff;
  }

  /*
     Compute the L2-norm of the objective function, cost function is f
  */

  ierr = VecDuplicate(appctx->dat.obj,&temp);CHKERRQ(ierr);
  ierr = VecCopy(appctx->dat.obj,temp);CHKERRQ(ierr);
  ierr = VecAXPY(temp,-1.0,appctx->dat.curr_sol);CHKERRQ(ierr);

  //ierr = VecPointwiseMult(appctx->dat.grad,temp,appctx->SEMop.mass);CHKERRQ(ierr);
  
  ierr   = VecDuplicate(temp,&temp2);CHKERRQ(ierr);
  ierr   = VecPointwiseMult(temp2,temp,temp);CHKERRQ(ierr);
  //ierr   = VecDot(temp,temp,f);CHKERRQ(ierr);CHKERRQ(ierr);
  //ierr   = VecPointwiseDivide(temp2,temp2,appctx->SEMop.mass);CHKERRQ(ierr);
  ierr = VecDot(temp2,appctx->SEMop.mass,f);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*  
   Initial conditions for the adjoint integration, given by 2*obj'=temp (rewrite)
   */
  
  ierr = VecScale(temp, -2.0);
  ierr = VecCopy(temp,appctx->dat.grad);CHKERRQ(ierr);
  
 // if(appctx->AdjTestFD){
    /*VecView(appctx->dat.curr_sol,PETSC_VIEWER_STDOUT_WORLD);*/
 //   ierr = TSSetCostGradients(appctx->ts,1,&appctx->dat.curr_sol,NULL);CHKERRQ(ierr);
    
  //} else {
    ierr = TSSetCostGradients(appctx->ts,1,&appctx->dat.grad,NULL);CHKERRQ(ierr);
  //}

  ierr = TSAdjointSetUp(appctx->ts);CHKERRQ(ierr);
  ierr = TSSetDM(appctx->ts,appctx->da);CHKERRQ(ierr);
    
  /* Set RHS Jacobian  for the adjoint integration */
  /*ierr = TSSetRHSJacobian(appctx->ts,appctx->SEMop.adj,appctx->SEMop.adj,TSComputeRHSJacobianConstant,appctx);CHKERRQ(ierr);*/

  ierr = TSAdjointSolve(appctx->ts);CHKERRQ(ierr);
   
  ierr = VecCopy(appctx->dat.grad,G);CHKERRQ(ierr);
  ierr=  TaoGetSolutionStatus(tao, &its, &ff, &gnorm, &cnorm, &xdiff, &reason);
  PetscPrintf(PETSC_COMM_WORLD,"iteration=%D\t cost function (TAO)=%g, cost function (L2 %g)\n",its,(double)ff,*f);

  PetscSNPrintf(filename,sizeof(filename),"PDEadjoint/optimize%02d.m",its);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename,&viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  PetscSNPrintf(data,sizeof(data),"TAO(%D)=%g; L2(%D)= %g ;\n",its+1,(double)ff,its+1,*f);
  PetscViewerASCIIPrintf(viewfile,data);
  //ierr = MatView(appctx.grad,viewfile);CHKERRQ(ierr);
  //ierr = MatView(appctx.adj,viewfile);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx->dat.grad,"Grad");
  ierr = VecView(appctx->dat.grad,viewfile);CHKERRQ(ierr);
  //ierr = PetscObjectSetName((PetscObject)appctx->SEMop.grid,"Grid");
  //ierr = VecView(appctx->SEMop.grid,viewfile);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)temp,        "Init_adj");
  ierr = VecView(temp,viewfile);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx->dat.ic,  "Init_ts");
  ierr = VecView(appctx->dat.ic,viewfile);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx->dat.curr_sol,"Curr_sol");
  ierr = VecView(appctx->dat.curr_sol,viewfile);CHKERRQ(ierr);
  //ierr = PetscObjectSetName((PetscObject)appctx->dat.obj,  "Obj");
  //ierr = VecView(appctx->dat.obj,viewfile);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx->SEMop.mass, "Mass");
  ierr = VecView(appctx->SEMop.mass,viewfile);CHKERRQ(ierr);
  //ierr = PetscObjectSetName((PetscObject)appctx->SEMop.adj, "A_adj");
  //ierr = MatView(appctx->SEMop.adj,viewfile);CHKERRQ(ierr);
  //ierr = PetscObjectSetName((PetscObject)appctx->SEMop.grad, "A");
  //ierr = MatView(appctx->SEMop.grad,viewfile);CHKERRQ(ierr);
  //ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
  //ierr = PetscViewerDestroy(&viewfile);CHKERRQ(ierr);
  ierr=PetscViewerPopFormat(viewfile);

  PetscFunctionReturn(0);
}
/*

#undef __FUNCT__
#define __FUNCT__ "RHSFunctionAdvectiongllDM"
PetscErrorCode RHSFunctionAdvectiongllDM(TS ts,PetscReal t,Vec globalin,Vec globalout,void *ctx)
{
  PetscErrorCode ierr;
  
  AppCtx        *appctx = (AppCtx*)ctx;   
  PetscFunctionBeginUser;
  ierr = TSGetRHSJacobian(ts,&A,NULL,NULL,&ctx);CHKERRQ(ierr);
  ierr = RHSMatrixAdvectiongllDM(ts,t,globalin,A,NULL,ctx);CHKERRQ(ierr);
  // ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
  ierr = MatMult(A,globalin,globalout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
*/

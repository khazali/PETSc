
static char help[] ="Solves a simple data assimilation problem with one dimensional advection diffusion equation using TSAdjoint\n\n";

/*
-tao_type test -tao_test_gradient
    Not yet tested in parallel

*/
/*
   Concepts: TS^time-dependent linear problems
   Concepts: TS^heat equation
   Concepts: TS^diffusion equation
   Concepts: adjoints
   Processors: n
*/

/* ------------------------------------------------------------------------

   This program uses the one-dimensional advection-diffusion equation),
       u_t = mu*u_xx - a u_x,
   on the domain 0 <= x <= 1, with periodic boundary conditions

   to demonstrate solving a data assimilation problem of finding the initial conditions
   to produce a given solution at a fixed time.

   The operators are discretized with the spectral element method

  ------------------------------------------------------------------------- */

#include <petsctao.h>
#include <petscts.h>
#include <petscgll.h>
#include <petscdraw.h>
#include <petscdmda.h>
#include <petscblaslapack.h>
#include <petsc/private/petscimpl.h>
#include <codi.hpp>
#include "f2c.h"

typedef codi::RealForwardGen<double> t1s;
typedef codi::RealForwardGen<t1s> t2s;
typedef codi::RealReverseGen<double> a1s;

/* Subroutine */ int dgemm(char *transa, char *transb, integer *m, integer *
          n, integer *k, doublereal *alpha, doublereal *a, integer *lda, 
          doublereal *b, integer *ldb, doublereal *beta, doublereal *c__, 
          integer *ldc);
/* Subroutine */ int daxpy(integer *n, doublereal *da, doublereal *dx,
          integer *incx, doublereal *dy, integer *incy);

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/

typedef struct {
  PetscInt    N;             /* grid points per elements*/
  PetscInt    Ex;              /* number of elements */
  PetscInt    Ey;              /* number of elements */
  PetscReal   tol_L2,tol_max; /* error norms */
  PetscInt    steps;          /* number of timesteps */
  PetscReal   Tend;           /* endtime */
  PetscReal   mu;             /* viscosity */
  PetscReal   Lx;              /* total length of domain */ 
  PetscReal   Ly;              /* total length of domain */     
  PetscReal   Lex; 
  PetscReal   Ley; 
  PetscInt    lenx;
  PetscInt    leny;
  PetscReal   Tadj;
} PetscParam;

template <class T> class Field {
  public: 
    T u,v;   /* wind speed */
};


typedef struct {
  Vec         obj;               /* desired end state */
  Vec         grid;              /* total grid */   
  Vec         grad;
  Vec         ic;
  Vec         curr_sol;
  Vec         pass_sol;
  Vec         true_solution;     /* actual initial conditions for the final solution */
} PetscData;

typedef struct {
  Vec         grid;              /* total grid */   
  Vec         mass;              /* mass matrix for total integration */
  Mat         stiff;             /* stifness matrix */
  Mat         keptstiff;
  Mat         grad;
  Mat         opadd;
  PetscGLL    gll;
} PetscSEMOperators;

typedef struct {
  DM                da;                /* distributed array data structure */
  PetscSEMOperators SEMop;
  PetscParam        param;
  PetscData         dat;
  TS                ts;
  PetscReal         initial_dt;
  PetscReal         *solutioncoefficients;
  PetscInt          ncoeff;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);
extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode TrueSolution(Vec,AppCtx*);
extern PetscErrorCode ComputeObjective(PetscReal,Vec,AppCtx*);
extern PetscErrorCode MonitorError(Tao,void*);
extern PetscErrorCode ComputeSolutionCoefficients(AppCtx*);
extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode MyMatMult(Mat,Vec,Vec);
extern PetscErrorCode MyMatMultTransp(Mat,Vec,Vec);

int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  Tao            tao;
  Vec            u;                      /* approximate solution vector */
  PetscErrorCode ierr;
  PetscInt       xs, xm, ys,ym, ix,iy;
  PetscInt       indx,indy,m, nn;
  PetscReal      x,y;
  Field<double>  **bmass;
  DMDACoor2d     **coors;
  Vec            global,loc;
  DM             cda;
  PetscInt       jx,jy;
  PetscViewer    viewfile;
  Mat            H_shell;
  MatNullSpace   nsp;

 
   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBegin;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /*initialize parameters */
  appctx.param.N    = 4;  /* order of the spectral element */
  appctx.param.Ex    = 2;  /* number of elements */
  appctx.param.Ey    = 2;  /* number of elements */
  appctx.param.Lx    = 4.0;  /* length of the domain */
  appctx.param.Ly    = 4.0;  /* length of the domain */
  appctx.param.mu   = 0.005; /* diffusion coefficient */
  appctx.initial_dt = 5e-3;
  appctx.param.steps = PETSC_MAX_INT;
  appctx.param.Tend  = 4.0;
  appctx.ncoeff      = 2;

  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&appctx.param.N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Ex",&appctx.param.Ex,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Ey",&appctx.param.Ey,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ncoeff",&appctx.ncoeff,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-Tend",&appctx.param.Tend,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-mu",&appctx.param.mu,NULL);CHKERRQ(ierr);
  appctx.param.Lex = appctx.param.Lx/appctx.param.Ex;
  appctx.param.Ley = appctx.param.Ly/appctx.param.Ey;


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create GLL data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscGLLCreate(appctx.param.N,PETSCGLL_VIA_LINEARALGEBRA,&appctx.SEMop.gll);CHKERRQ(ierr);
  
  appctx.param.lenx = appctx.param.Ex*(appctx.param.N-1);
  appctx.param.leny = appctx.param.Ey*(appctx.param.N-1);

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
     and to set up the ghost point communication pattern.  There are E*(Nl-1)+1
     total grid values spread equally among all the processors, except first and last
  */

  
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_BOX,appctx.param.lenx,appctx.param.leny,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&appctx.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(appctx.da);CHKERRQ(ierr);
  ierr = DMSetUp(appctx.da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(appctx.da,0,"u");CHKERRQ(ierr);
  ierr = DMDASetFieldName(appctx.da,1,"v");CHKERRQ(ierr);
  
  /*
     Extract global and local vectors from DMDA; we use these to store the
     approximate solution.  Then duplicate these for remaining vectors that
     have the same types.
  */

  ierr = DMCreateGlobalVector(appctx.da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.dat.ic);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.dat.true_solution);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.dat.obj);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.SEMop.mass);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.dat.curr_sol);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.dat.pass_sol);CHKERRQ(ierr);
 
  ierr = DMDAGetCorners(appctx.da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
 /* Compute function over the locally owned part of the grid */
    xs=xs/(appctx.param.N-1);
    xm=xm/(appctx.param.N-1);
    ys=ys/(appctx.param.N-1);
    ym=ym/(appctx.param.N-1);
    
  VecSet(appctx.SEMop.mass,0.0);
  
  DMCreateLocalVector(appctx.da,&loc);
  ierr = DMDAVecGetArray(appctx.da,loc,&bmass);CHKERRQ(ierr);
  
  /*
     Build mass over entire mesh (multi-elemental) 

  */ 

   for (ix=xs; ix<xs+xm; ix++) 
     {for (jx=0; jx<appctx.param.N; jx++) 
      {for (iy=ys; iy<ys+ym; iy++) 
        {for (jy=0; jy<appctx.param.N; jy++)   
        {
        x = (appctx.param.Lex/2.0)*(appctx.SEMop.gll.nodes[jx]+1.0)+appctx.param.Lex*ix; 
        y = (appctx.param.Ley/2.0)*(appctx.SEMop.gll.nodes[jy]+1.0)+appctx.param.Ley*iy; 
        indx=ix*(appctx.param.N-1)+jx;
        indy=iy*(appctx.param.N-1)+jy;
        bmass[indy][indx].u +=appctx.SEMop.gll.weights[jx]*appctx.SEMop.gll.weights[jy]*.25*appctx.param.Ley*appctx.param.Lex;
        bmass[indy][indx].v +=appctx.SEMop.gll.weights[jx]*appctx.SEMop.gll.weights[jy]*.25*appctx.param.Ley*appctx.param.Lex;
     
          } 
         }
       }
     }
    
    DMDAVecRestoreArray(appctx.da,loc,&bmass);CHKERRQ(ierr);
    DMLocalToGlobalBegin(appctx.da,loc,ADD_VALUES,appctx.SEMop.mass);
    DMLocalToGlobalEnd(appctx.da,loc,ADD_VALUES,appctx.SEMop.mass); 
  
   
  DMDASetUniformCoordinates(appctx.da,0.0,appctx.param.Lx,0.0,appctx.param.Ly,0.0,0.0);
  DMGetCoordinateDM(appctx.da,&cda);
  
  DMGetCoordinates(appctx.da,&global);
  VecSet(global,0.0);
  DMDAVecGetArray(cda,global,&coors);
 
   for (ix=xs; ix<xs+xm; ix++) 
     {for (jx=0; jx<appctx.param.N-1; jx++) 
      {for (iy=ys; iy<ys+ym; iy++) 
        {for (jy=0; jy<appctx.param.N-1; jy++)   
        {
        //x = (appctx.param.Lex/2.0)*(appctx.SEMop.gll.nodes[jx]+1.0)+appctx.param.Lex*ix-0.5*PETSC_PI;
        //y = (appctx.param.Ley/2.0)*(appctx.SEMop.gll.nodes[jy]+1.0)+appctx.param.Ley*iy-0.5*PETSC_PI;
        x = (appctx.param.Lex/2.0)*(appctx.SEMop.gll.nodes[jx]+1.0)+appctx.param.Lex*ix-2.0; 
        y = (appctx.param.Ley/2.0)*(appctx.SEMop.gll.nodes[jy]+1.0)+appctx.param.Ley*iy-2.0;
        indx=ix*(appctx.param.N-1)+jx;
        indy=iy*(appctx.param.N-1)+jy;
        coors[indy][indx].x=x;
        coors[indy][indx].y=y;
       } 
       }
     }
     }
    DMDAVecRestoreArray(cda,global,&coors);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"tomesh.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)global,"grid");
    ierr = VecView(global,viewfile);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)appctx.SEMop.mass,"mass");
    ierr = VecView(appctx.SEMop.mass,viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewfile);
    ierr = PetscViewerDestroy(&viewfile);CHKERRQ(ierr);

 
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create matrix data structure; set matrix evaluation routine.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMSetMatrixPreallocateOnly(appctx.da, PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMCreateMatrix(appctx.da,&appctx.SEMop.stiff);CHKERRQ(ierr);
  ierr = DMCreateMatrix(appctx.da,&appctx.SEMop.grad);CHKERRQ(ierr);
  ierr = DMCreateMatrix(appctx.da,&appctx.SEMop.opadd);CHKERRQ(ierr);
    
   /*
       For linear problems with a time-dependent f(u,t) in the equation
       u_t = f(u,t), the user provides the discretized right-hand-side
       as a time-dependent matrix.
    */
  
  /* Create the TS solver that solves the ODE and its adjoint; set its options */
  ierr = TSCreate(PETSC_COMM_WORLD,&appctx.ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(appctx.ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(appctx.ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetDM(appctx.ts,appctx.da);CHKERRQ(ierr);
  ierr = TSSetTime(appctx.ts,0.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(appctx.ts,appctx.initial_dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(appctx.ts,appctx.param.steps);CHKERRQ(ierr);
  ierr = TSSetMaxTime(appctx.ts,appctx.param.Tend);CHKERRQ(ierr);
  ierr = TSSetDuration(appctx.ts,appctx.param.steps,appctx.param.Tend);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(appctx.ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  VecGetLocalSize(u,&m);
  VecGetSize(u,&nn);
  
  MatCreateShell(PETSC_COMM_WORLD,m,m,nn,nn,&appctx,&H_shell);
  MatShellSetOperation(H_shell,MATOP_MULT,(void(*)(void))MyMatMult);
  MatShellSetOperation(H_shell,MATOP_MULT_TRANSPOSE,(void(*)(void))MyMatMultTransp);
  
 /* attach the null space to the matrix, this probably is not needed but does no harm */
  
 /*
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&nsp);CHKERRQ(ierr);
  ierr = MatSetNullSpace(H_shell,nsp);CHKERRQ(ierr);
  ierr = MatNullSpaceTest(nsp,H_shell,NULL);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nsp);CHKERRQ(ierr);
  */

  ierr = TSSetTolerances(appctx.ts,1e-7,NULL,1e-7,NULL);CHKERRQ(ierr);
  ierr = TSSetFromOptions(appctx.ts);CHKERRQ(ierr);
  /* Need to save initial timestep user may have set with -ts_dt so it can be reset for each new TSSolve() */
  ierr = TSGetTimeStep(appctx.ts,&appctx.initial_dt);CHKERRQ(ierr);
  //ierr = TSSetRHSFunction(appctx.ts,NULL,TSComputeRHSFunctionLinear,&appctx);CHKERRQ(ierr);
  //ierr = TSSetRHSJacobian(appctx.ts,H_shell,H_shell,TSComputeRHSJacobianConstant,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(appctx.ts,H_shell,H_shell,RHSJacobian,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(appctx.ts,NULL,RHSFunction,&appctx);CHKERRQ(ierr);
  

  ierr = ComputeObjective(2.0,appctx.dat.obj,&appctx);CHKERRQ(ierr);
  ierr = TSSolve(appctx.ts,appctx.dat.obj);CHKERRQ(ierr);
  


  Vec   ref, wrk_vec, jac, vec_jac, vec_rhs, temp, vec_trans;
  Field<double> **s;
  PetscScalar vareps;
  PetscInt i;   
  PetscInt its=0;
  char var[15] ;
  
  ierr = VecDuplicate(appctx.dat.ic,&wrk_vec);CHKERRQ(ierr);
  //ierr = VecDuplicate(appctx.dat.ic,&temp);CHKERRQ(ierr);
  ierr = VecDuplicate(appctx.dat.ic,&vec_jac);CHKERRQ(ierr);
  ierr = VecDuplicate(appctx.dat.ic,&vec_rhs);CHKERRQ(ierr);
  ierr = VecDuplicate(appctx.dat.ic,&vec_trans);CHKERRQ(ierr);
  ierr = VecDuplicate(appctx.dat.ic,&ref);CHKERRQ(ierr);
  ierr = VecDuplicate(appctx.dat.ic,&jac);CHKERRQ(ierr);

  ierr = VecCopy(appctx.dat.ic,appctx.dat.pass_sol);CHKERRQ(ierr);
  //VecSet(appctx.dat.pass_sol,1.0);

  RHSFunction(appctx.ts,0.0,appctx.dat.ic,ref,&appctx);
  //ierr = FormFunctionGradient(tao,wrk_vec,&wrk1,wrk2_vec,&appctx);CHKERRQ(ierr);
  // Note computed gradient is in wrk2_vec, original cost is in wrk1 
  ierr = VecZeroEntries(vec_jac);
  ierr = VecZeroEntries(vec_rhs); 
  vareps = 1e-05;
    for (i=0; i<2*(appctx.param.lenx*appctx.param.leny); i++) 
    //for (i=0; i<6; i++) 
     {
      its=its+1;
      ierr = VecCopy(appctx.dat.ic,wrk_vec); CHKERRQ(ierr); //reset J(eps) for each point
      ierr = VecZeroEntries(jac);
      VecSetValue(wrk_vec,i, vareps,ADD_VALUES);
      VecSetValue(jac,i,1.0,ADD_VALUES);
      RHSFunction(appctx.ts,0.0,wrk_vec,vec_rhs,&appctx);
      VecAXPY(vec_rhs,-1.0,ref);
      VecScale(vec_rhs, 1.0/vareps);
      MyMatMult(H_shell,jac,vec_jac);
      //VecView(jac,0);
      MyMatMultTransp(H_shell,jac,vec_trans);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"testjac.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    PetscSNPrintf(var,sizeof(var),"jac(:,%d)",its);
    ierr = PetscObjectSetName((PetscObject)vec_jac,var);
    ierr = VecView(vec_jac,viewfile);CHKERRQ(ierr);
    PetscSNPrintf(var,sizeof(var),"rhs(:,%d)",its);
    ierr = PetscObjectSetName((PetscObject)vec_rhs,var);
    ierr = VecView(vec_rhs,viewfile);CHKERRQ(ierr);
    PetscSNPrintf(var,sizeof(var),"trans(:,%d)",its);
    ierr = PetscObjectSetName((PetscObject)vec_trans,var);
    ierr = VecView(vec_trans,viewfile);CHKERRQ(ierr);
    //ierr = PetscObjectSetName((PetscObject)ref,"ref");
    //ierr = VecView(ref,viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewfile);
    //printf("test i %d length %d\n",its, appctx.param.lenx*appctx.param.leny);
    } 
exit(1);

  //ierr = VecDuplicate(appctx.dat.ic,&uu);CHKERRQ(ierr);
  //ierr = VecCopy(appctx.dat.ic,uu);CHKERRQ(ierr);
  //MatView(H_shell,0);
  
   //ierr = VecDuplicate(appctx.dat.ic,&appctx.dat.curr_sol);CHKERRQ(ierr);
   //ierr = VecCopy(appctx.dat.ic,appctx.dat.curr_sol);CHKERRQ(ierr);
   //ierr = TSSolve(appctx.ts,appctx.dat.curr_sol);CHKERRQ(ierr);

/*
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"sol2d.m",&viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)appctx.dat.obj,"sol");
    ierr = VecView(appctx.dat.obj,viewfile);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)appctx.dat.ic,"ic");
    ierr = VecView(appctx.dat.ic,viewfile);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewfile);

 exit(1);

 */
  ierr = TSSetSaveTrajectory(appctx.ts);CHKERRQ(ierr);

  /* Set Objective and Initial conditions for the problem and compute Objective function (evolution of true_solution to final time */

  ierr = ComputeSolutionCoefficients(&appctx);CHKERRQ(ierr);
  ierr = InitialConditions(appctx.dat.ic,&appctx);CHKERRQ(ierr);
  ierr = ComputeObjective(4.0,appctx.dat.true_solution,&appctx);CHKERRQ(ierr);
  //ierr = TrueSolution(appctx.dat.true_solution,&appctx);CHKERRQ(ierr);
  //ierr = ComputeObjective(4.0,appctx.dat.obj,&appctx);CHKERRQ(ierr);

  /* Create TAO solver and set desired solution method  */
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetMonitor(tao,MonitorError,&appctx,NULL);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOBLMVM);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,appctx.dat.ic);CHKERRQ(ierr);
  /* Set routine for function and gradient evaluation  */
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void *)&appctx);CHKERRQ(ierr);
  /* Check for any TAO command line options  */
  ierr = TaoSetTolerances(tao,1e-8,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  ierr = TaoSolve(tao);CHKERRQ(ierr);

  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = PetscFree(appctx.solutioncoefficients);CHKERRQ(ierr);
  //ierr = MatDestroy(&appctx.SEMop.stiff);CHKERRQ(ierr);
  //ierr = MatDestroy(&appctx.SEMop.keptstiff);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.ic);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.true_solution);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.obj);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.SEMop.mass);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.curr_sol);CHKERRQ(ierr);
  ierr = PetscGLLDestroy(&appctx.SEMop.gll);CHKERRQ(ierr);
  ierr = DMDestroy(&appctx.da);CHKERRQ(ierr);
  ierr = TSDestroy(&appctx.ts);CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).
  */
    ierr = PetscFinalize();
    return ierr;
}

/*
    Computes the coefficients for the analytic solution to the PDE
*/
PetscErrorCode ComputeSolutionCoefficients(AppCtx *appctx)
{
  PetscErrorCode    ierr;
  PetscRandom       rand;
  PetscInt          i;

  ierr = PetscMalloc1(appctx->ncoeff,&appctx->solutioncoefficients);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand,.9,1.0);CHKERRQ(ierr);
  for (i=0; i<appctx->ncoeff; i++) 
    {
    ierr = PetscRandomGetValue(rand,&appctx->solutioncoefficients[i]);CHKERRQ(ierr);
    }
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  return 0;
}

/* --------------------------------------------------------------------- */
/*
   InitialConditions - Computes the initial conditions for the Tao optimization solve (these are also initial conditions for the first TSSolve()

                       The routine TrueSolution() computes the true solution for the Tao optimization solve which means they are the initial conditions for the objective function

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode InitialConditions(Vec u,AppCtx *appctx)
{
  PetscScalar       tt,pp;
  Field<double>     **s;
  PetscErrorCode    ierr;
  PetscInt          i,j;
  DM                cda;
  Vec          global;
  DMDACoor2d        **coors;

  ierr = DMDAVecGetArray(appctx->da,u,&s);CHKERRQ(ierr);
      
  DMGetCoordinateDM(appctx->da,&cda);
  DMGetCoordinates(appctx->da,&global);
  DMDAVecGetArray(cda,global,&coors);

  tt=0.0;
  for (i=0; i<appctx->param.lenx; i++) 
    {for (j=0; j<appctx->param.leny; j++) 
      {
      //s[j][i].u=PetscExpScalar(-appctx->param.mu*tt)*(PetscCosScalar(2.*PETSC_PI*coors[j][i].x)+PetscSinScalar(2.*PETSC_PI*coors[j][i].y))*2.0;
      //s[j][i].v=PetscExpScalar(-appctx->param.mu*tt)*(PetscSinScalar(2.*PETSC_PI*coors[j][i].x)+PetscCosScalar(2.*PETSC_PI*coors[j][i].y))*2.0;
      
      s[j][i].u=PetscExpScalar(-appctx->param.mu*tt)*(PetscCosScalar(0.5*PETSC_PI*coors[j][i].x)+PetscSinScalar(0.5*PETSC_PI*coors[j][i].y))/10.0;
      s[j][i].v=PetscExpScalar(-appctx->param.mu*tt)*(PetscSinScalar(0.5*PETSC_PI*coors[j][i].x)+PetscCosScalar(0.5*PETSC_PI*coors[j][i].y))/10.0;
      
      //pp=(coors[j][i].x*coors[j][i].x+coors[j][i].y*coors[j][i].y);
      //s[j][i].u=PetscExpScalar(- 7.0*pp)/5.0;
      //s[j][i].v=0.0;//PetscExpScalar(-appctx->param.mu*tt - 12.0*(coors[j][i].x*coors[j][i].x+coors[j][i].y*coors[j][i].y));


      //s[j][i].u=PetscExpScalar(-appctx->param.mu*tt)*(PetscSinScalar(2*PETSC_PI*));
      //s[j][i].v=PetscExpScalar(-appctx->param.mu*tt)*(PetscCosScalar(2*PETSC_PI*(coors[j][i].x-0.5*PETSC_PI)));
      //s[j][i].u=PetscMax(0.0,PetscSinReal(PetscSqrtReal(PETSC_PI*coors[j][i].x*coors[j][i].x+PETSC_PI*coors[j][i].y*coors[j][i].y)))+1.0;
      //s[j][i].v=0.0;
      } 
     }
  
  ierr = DMDAVecRestoreArray(appctx->da,u,&s);CHKERRQ(ierr);

  return 0;
}


/*
   TrueSolution() computes the true solution for the Tao optimization solve which means they are the initial conditions for the objective function. 

             InitialConditions() computes the initial conditions for the begining of the Tao iterations

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode TrueSolution(Vec u,AppCtx *appctx)
{
  PetscScalar       tt;
  Field<double>     **s;  
  PetscErrorCode    ierr;
  PetscInt          i,j;
  DM                cda;
  Vec               global;
  DMDACoor2d        **coors;

  ierr = DMDAVecGetArray(appctx->da,u,&s);CHKERRQ(ierr);
      
  DMGetCoordinateDM(appctx->da,&cda);
  DMGetCoordinates(appctx->da,&global);
  DMDAVecGetArray(cda,global,&coors);

  tt=4.0-appctx->param.Tend;
  for (i=0; i<appctx->param.lenx; i++) 
    {for (j=0; j<appctx->param.leny; j++) 
      {
      s[j][i].u=PetscExpScalar(-appctx->param.mu*tt)*(PetscCosScalar(0.5*PETSC_PI*coors[j][i].x)+PetscSinScalar(0.5*PETSC_PI*coors[j][i].y))/10.0;
      s[j][i].v=PetscExpScalar(-appctx->param.mu*tt)*(PetscSinScalar(0.5*PETSC_PI*coors[j][i].x)+PetscCosScalar(0.5*PETSC_PI*coors[j][i].y))/10.0;
      
      } 
     }
  
  ierr = DMDAVecRestoreArray(appctx->da,u,&s);CHKERRQ(ierr);
  /* make sure initial conditions do not contain the constant functions, since with periodic boundary conditions the constant functions introduce a null space */
   return 0;
}
/* --------------------------------------------------------------------- */
/*
   Sets the desired profile for the final end time

   Input Parameters:
   t - final time
   obj - vector storing the desired profile
   appctx - user-defined application context

*/
PetscErrorCode ComputeObjective(PetscReal t,Vec obj,AppCtx *appctx)
{
  Field<double>     **s; 
  PetscErrorCode    ierr;
  PetscInt          i,j;
  DM                cda;
  Vec               global;
  DMDACoor2d        **coors;
  PetscScalar       pp;


  ierr = DMDAVecGetArray(appctx->da,obj,&s);CHKERRQ(ierr);
      
  DMGetCoordinateDM(appctx->da,&cda);
  DMGetCoordinates(appctx->da,&global);
  DMDAVecGetArray(cda,global,&coors);

  for (i=0; i<appctx->param.lenx; i++) 
    {for (j=0; j<appctx->param.leny; j++) 
      {
      //s[j][i].u=PetscExpScalar(-appctx->param.mu*t)*(PetscCosScalar(2.*PETSC_PI*coors[j][i].x)+PetscSinScalar(2.*PETSC_PI*coors[j][i].y));
      //s[j][i].v=PetscExpScalar(-appctx->param.mu*t)*(PetscSinScalar(2.*PETSC_PI*coors[j][i].x)+PetscCosScalar(2.*PETSC_PI*coors[j][i].y));
      //s[j][i].u=PetscExpScalar(-appctx->param.mu*t - 10.0*(coors[j][i].x*coors[j][i].x+coors[j][i].y*coors[j][i].y));
      //s[j][i].v=PetscExpScalar(-appctx->param.mu*t - 12.0*(coors[j][i].x*coors[j][i].x+coors[j][i].y*coors[j][i].y));
      //pp=(coors[j][i].x*coors[j][i].x+coors[j][i].y*coors[j][i].y);
      //s[j][i].u=PetscExpScalar(- 2.0*pp*pp)/5.0;
      //s[j][i].v=0.0;

      s[j][i].u=PetscExpScalar(-appctx->param.mu*t)*(PetscCosScalar(0.5*PETSC_PI*coors[j][i].x)+PetscSinScalar(0.5*PETSC_PI*coors[j][i].y))/10.0;
      s[j][i].v=PetscExpScalar(-appctx->param.mu*t)*(PetscSinScalar(0.5*PETSC_PI*coors[j][i].x)+PetscCosScalar(0.5*PETSC_PI*coors[j][i].y))/10.0;
      //s[j][i].u=PetscExpScalar(-appctx->param.mu*t)*(PetscSinScalar(2*PETSC_PI*coors[j][i].x));
      //s[j][i].v=PetscExpScalar(-appctx->param.mu*t)*(PetscCosScalar(2*PETSC_PI*(coors[j][i].x-0.5*PETSC_PI)));
      } 
     }
  
  ierr = DMDAVecRestoreArray(appctx->da,obj,&s);CHKERRQ(ierr);


  return 0;
}

template <class T> PetscErrorCode ADRHSFunction (Field<T> **outl, Field<T> **ul, void *ctx) 
{
  PetscErrorCode  ierr;
  AppCtx          *appctx = (AppCtx*)ctx;  
  T               **wrk3, **wrk1, **wrk2, **wrk4, **wrk5, **wrk6, **wrk7;
  PetscScalar     **stiff, **mass, **grad;
  T               **astiff, **amass, **agrad;
  T               **ulb, **vlb;
  PetscInt        i,ix,iy,jx,jy, indx, indy;
  PetscInt        xs,xm,ys,ym, Nl, Nl2; 
  PetscViewer     viewfile;
  DM              cda;
  Vec             uloc, outloc, global, forcing;
  DMDACoor2d      **coors;
  PetscScalar     tt, alpha, beta, tempu, tempv,xpy;
  PetscScalar     aalpha, abeta; 
  PetscInt        inc;  
  static int its=0;
  char var[12] ;

  ierr = PetscGLLElementLaplacianCreate(&appctx->SEMop.gll,&stiff);CHKERRQ(ierr);
  ierr = PetscGLLElementMassCreate(&appctx->SEMop.gll,&mass);CHKERRQ(ierr);
  ierr = PetscGLLElementAdvectionCreate(&appctx->SEMop.gll,&grad);CHKERRQ(ierr);

  ierr = DMDAGetCorners(appctx->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  Nl    = appctx->param.N; 

  xs=xs/(Nl-1);
  xm=xm/(Nl-1);
  ys=ys/(Nl-1);
  ym=ym/(Nl-1); 

  inc=1;
  /*
     Initialize work arrays
  */ 
  astiff = new T*[appctx->param.N];
  astiff[0] = new T[appctx->param.N * appctx->param.N];
  for (i=1; i<Nl; i++) {
    astiff[i] = astiff[i-1]+Nl;
    for(int j=1; j<Nl; j++) {
      astiff[i][j] = stiff[i][j];
    }
  }

  amass = new T*[appctx->param.N];
  amass[0] = new T[appctx->param.N * appctx->param.N];
  for(int j=0; j<Nl; j++) {
      amass[0][j] = mass[0][j];
  }
  for (i=1; i<Nl; i++) {
    amass[i] = amass[i-1]+Nl;
    for(int j=0; j<Nl; j++) {
      amass[i][j] = mass[i][j];
    }
  }

  astiff = new T*[appctx->param.N];
  astiff[0] = new T[appctx->param.N * appctx->param.N];
  for(int j=0; j<Nl; j++) {
      astiff[0][j] = stiff[0][j];
  }
  for (i=1; i<Nl; i++) {
    astiff[i] = astiff[i-1]+Nl;
    for(int j=0; j<Nl; j++) {
      astiff[i][j] = stiff[i][j];
    }
  }

  agrad = new T*[appctx->param.N];
  agrad[0] = new T[appctx->param.N * appctx->param.N];
  for(int j=0; j<Nl; j++) {
      agrad[0][j] = grad[0][j];
  }
  for (i=1; i<Nl; i++) {
    agrad[i] = agrad[i-1]+Nl;
    for(int j=0; j<Nl; j++) {
      agrad[i][j] = grad[i][j];
    }
  }


  ulb = new T*[appctx->param.N];
  ulb[0] = new T[appctx->param.N * appctx->param.N];
  for (i=1; i<Nl; i++) ulb[i] = ulb[i-1]+Nl;

  vlb = new T*[appctx->param.N];
  vlb[0] = new T[appctx->param.N * appctx->param.N];
  for (i=1; i<Nl; i++) vlb[i] = vlb[i-1]+Nl;

  wrk1 = new T*[appctx->param.N];
  wrk1[0] = new T[appctx->param.N * appctx->param.N];
  for (i=1; i<Nl; i++) wrk1[i] = wrk1[i-1]+Nl;

  wrk2 = new T*[appctx->param.N];
  wrk2[0] = new T[appctx->param.N * appctx->param.N];
  for (i=1; i<Nl; i++) wrk2[i] = wrk2[i-1]+Nl;

  wrk3 = new T*[appctx->param.N];
  wrk3[0] = new T[appctx->param.N * appctx->param.N];
  for (i=1; i<Nl; i++) wrk3[i] = wrk3[i-1]+Nl;\

  wrk4 = new T*[appctx->param.N];
  wrk4[0] = new T[appctx->param.N * appctx->param.N];
  for (i=1; i<Nl; i++) wrk4[i] = wrk4[i-1]+Nl;

  wrk5 = new T*[appctx->param.N];
  wrk5[0] = new T[appctx->param.N * appctx->param.N];
  for (i=1; i<Nl; i++) wrk5[i] = wrk5[i-1]+Nl;

  wrk6 = new T*[appctx->param.N];
  wrk6[0] = new T[appctx->param.N * appctx->param.N];
  for (i=1; i<Nl; i++) wrk6[i] = wrk6[i-1]+Nl;

  wrk7 = new T*[appctx->param.N];
  wrk7[0] = new T[appctx->param.N * appctx->param.N];
  for (i=1; i<Nl; i++) wrk7[i] = wrk7[i-1]+Nl;

  alpha = 1.0;
  aalpha = alpha;
  beta  = 0.0;
  abeta = beta;
  Nl2=Nl*Nl;

   for (ix=xs; ix<xs+xm; ix++) 
      {for (iy=ys; iy<ys+ym; iy++) 
         { 
       for (jx=0; jx<appctx->param.N; jx++) 
        {for (jy=0; jy<appctx->param.N; jy++) 
               
           {ulb[jy][jx]=0.0;
            vlb[jy][jx]=0.0;
            indx=ix*(appctx->param.N-1)+jx;
            indy=iy*(appctx->param.N-1)+jy;
            ulb[jy][jx]=ul[indy][indx].u; 
            vlb[jy][jx]=ul[indy][indx].v; 
          }}

        //here the stifness matrix in 2d
        //first product (B x K_yy)u=W2 (u_yy)
        aalpha=appctx->param.Lex/2.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&aalpha,&amass[0][0],&Nl,&ulb[0][0],&Nl,&abeta,&wrk1[0][0],&Nl);
        aalpha=2./appctx->param.Ley;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&aalpha,&wrk1[0][0],&Nl,&astiff[0][0],&Nl,&abeta,&wrk2[0][0],&Nl);

        //second product (K_xx x B) u=W3 (u_xx)
        aalpha=2.0/appctx->param.Lex;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&aalpha,&astiff[0][0],&Nl,&ulb[0][0],&Nl,&abeta,&wrk1[0][0],&Nl);
        aalpha=appctx->param.Ley/2.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&aalpha,&wrk1[0][0],&Nl,&amass[0][0],&Nl,&abeta,&wrk3[0][0],&Nl);

        aalpha=1.0;
        BLASaxpy_(&Nl2,&aalpha, &wrk3[0][0],&inc,&wrk2[0][0],&inc); //I freed wrk3 and saved the laplacian in wrk2
       
        // for the v component now 
        //first product (B x K_yy)v=W3
        aalpha=appctx->param.Lex/2.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&aalpha,&amass[0][0],&Nl,&vlb[0][0],&Nl,&abeta,&wrk1[0][0],&Nl);
        aalpha=2.0/appctx->param.Ley;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&aalpha,&wrk1[0][0],&Nl,&astiff[0][0],&Nl,&abeta,&wrk3[0][0],&Nl);

        //second product (K_xx x B)v=W4
        aalpha=2.0/appctx->param.Lex;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&aalpha,&astiff[0][0],&Nl,&vlb[0][0],&Nl,&abeta,&wrk1[0][0],&Nl);
        aalpha=appctx->param.Ley/2.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&aalpha,&wrk1[0][0],&Nl,&amass[0][0],&Nl,&abeta,&wrk4[0][0],&Nl);

        aalpha=1.0;
        BLASaxpy_(&Nl2,&aalpha, &wrk4[0][0],&inc,&wrk3[0][0],&inc); //I freed wrk4 and saved the laplacian in wrk3


        //now the gradient operator for u
        // first (D_x x B) u =W4 this multiples u
        aalpha=appctx->param.Lex/2.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&aalpha,&amass[0][0],&Nl,&ulb[0][0],&Nl,&abeta,&wrk1[0][0],&Nl);
        aalpha=1.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&aalpha,&wrk1[0][0],&Nl,&agrad[0][0],&Nl,&abeta,&wrk4[0][0],&Nl);
        

        // first (B x D_y) u =W5 this mutiplies v
        aalpha=1.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&aalpha,&agrad[0][0],&Nl,&ulb[0][0],&Nl,&abeta,&wrk1[0][0],&Nl);
        aalpha=appctx->param.Ley/2.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&aalpha,&wrk1[0][0],&Nl,&amass[0][0],&Nl,&abeta,&wrk5[0][0],&Nl);


        //now the agradient operator for v
        // first (D_x x B) v =W6 this multiples u
        aalpha=appctx->param.Lex/2.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&aalpha,&amass[0][0],&Nl,&vlb[0][0],&Nl,&abeta,&wrk1[0][0],&Nl);
        aalpha=1.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&aalpha,&wrk1[0][0],&Nl,&agrad[0][0],&Nl,&abeta,&wrk6[0][0],&Nl);
        

        // first (B x D_y) v =W7 this mutiplies v
        aalpha=1.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&aalpha,&agrad[0][0],&Nl,&vlb[0][0],&Nl,&abeta,&wrk1[0][0],&Nl);
        aalpha=appctx->param.Ley/2.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&aalpha,&wrk1[0][0],&Nl,&amass[0][0],&Nl,&abeta,&wrk7[0][0],&Nl);


        for (jx=0; jx<appctx->param.N; jx++) 
        {for (jy=0; jy<appctx->param.N; jy++)   
           {indx=ix*(appctx->param.N-1)+jx;
            indy=iy*(appctx->param.N-1)+jy;
            
            outl[indy][indx].u +=appctx->param.mu*(wrk2[jy][jx])+vlb[jy][jx]*wrk5[jy][jx]+ulb[jy][jx]*wrk4[jy][jx];//+rr.u*mass[jy][jx];  
            outl[indy][indx].v +=appctx->param.mu*(wrk3[jy][jx])+ulb[jy][jx]*wrk6[jy][jx]+vlb[jy][jx]*wrk7[jy][jx];//+rr.v*mass[jy][jx];    
           }}
        }
     }
  ierr = PetscGLLElementLaplacianDestroy(&appctx->SEMop.gll,&stiff);CHKERRQ(ierr);
  ierr = PetscGLLElementAdvectionDestroy(&appctx->SEMop.gll,&grad);CHKERRQ(ierr);
  ierr = PetscGLLElementMassDestroy(&appctx->SEMop.gll,&mass);CHKERRQ(ierr);
  
  delete [] wrk1[0]; delete [] wrk1;
  delete [] wrk2[0]; delete [] wrk2;
  delete [] wrk3[0]; delete [] wrk3;
  delete [] wrk4[0]; delete [] wrk4;
  delete [] wrk5[0]; delete [] wrk5;
  delete [] wrk6[0]; delete [] wrk6;
  delete [] wrk7[0]; delete [] wrk7;
}

#undef __FUNCT__
#define __FUNCT__ "RHSFunction"
PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec globalin,Vec globalout,void *ctx)
{
  PetscErrorCode  ierr;
  AppCtx          *appctx = (AppCtx*)ctx;  
  PetscScalar     **wrk3, **wrk1, **wrk2, **wrk4, **wrk5, **wrk6, **wrk7;
  PetscScalar     **stiff, **mass, **grad;
  PetscScalar     **ulb, **vlb;
  Field<double>  **ul;
  Field<double>   **ff;
  Field<double>   **outl; 
  PetscInt        i,ix,iy,jx,jy, indx, indy;
  PetscInt        xs,xm,ys,ym, Nl, Nl2; 
  PetscViewer     viewfile;
  DM              cda;
  Vec             uloc, outloc, global, forcing;
  DMDACoor2d      **coors;
  PetscScalar     tt, alpha, beta, tempu, tempv,xpy; 
  PetscInt        inc;  
  static int its=0;
  char var[12] ;

  PetscFunctionBegin;

  ierr = PetscGLLElementLaplacianCreate(&appctx->SEMop.gll,&stiff);CHKERRQ(ierr);
  ierr = PetscGLLElementMassCreate(&appctx->SEMop.gll,&mass);CHKERRQ(ierr);
  ierr = PetscGLLElementAdvectionCreate(&appctx->SEMop.gll,&grad);CHKERRQ(ierr);
  
  /* unwrap local vector for the input solution */
  DMCreateLocalVector(appctx->da,&uloc);

  DMGlobalToLocalBegin(appctx->da,globalin,INSERT_VALUES,uloc);
  DMGlobalToLocalEnd(appctx->da,globalin,INSERT_VALUES,uloc);

  ierr = DMDAVecGetArrayRead(appctx->da,uloc,&ul);CHKERRQ(ierr);

  /* unwrap local vector for the output solution */
  DMCreateLocalVector(appctx->da,&outloc);
  
  ierr = DMDAVecGetArray(appctx->da,outloc,&outl);CHKERRQ(ierr);

  
  //ierr = DMDAVecGetArray(appctx->da,gradloc,&outgrad);CHKERRQ(ierr);
  ierr = ADRHSFunction<double> (outl, ul, ctx); 

  ierr = DMDAVecRestoreArrayRead(appctx->da,globalin,&uloc);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(appctx->da,outloc,&outl);CHKERRQ(ierr);
  VecSet(globalout,0.0);
  DMLocalToGlobalBegin(appctx->da,outloc,ADD_VALUES,globalout);
  DMLocalToGlobalEnd(appctx->da,outloc,ADD_VALUES,globalout);

  VecScale(globalout, -1.0);
 
  ierr = VecPointwiseDivide(globalout,globalout,appctx->SEMop.mass);CHKERRQ(ierr);

  
  DMGetCoordinateDM(appctx->da,&cda);
  DMGetCoordinates(appctx->da,&global);
  DMDAVecGetArray(cda,global,&coors);
  VecDuplicate(globalout,&forcing); 

  ierr = DMDAVecGetArray(appctx->da,forcing,&ff);CHKERRQ(ierr);
     
  /* 
  tt=t;
  for (ix=0; ix<appctx->param.lenx; ix++) 
    {for (jx=0; jx<appctx->param.leny; jx++) 
      {
      //ff[jx][ix].u=PetscExpScalar(-appctx->param.mu*tt)*(appctx->param.mu*(-1.0 + 4*PETSC_PI*PETSC_PI)*PetscCosScalar(2.*PETSC_PI*coors[jx][ix].x)
      //             +2.*PETSC_PI*ul[jx][ix].v*PetscCosScalar(2.*PETSC_PI*coors[jx][ix].y)-2.*PETSC_PI*ul[jx][ix].u*PetscSinScalar(2.*PETSC_PI*coors[jx][ix].x)-
      //               appctx->param.mu*PetscSinScalar(2.*PETSC_PI*coors[jx][ix].y));
      //ff[jx][ix].v=PetscExpScalar(-appctx->param.mu*tt)*((appctx->param.mu*(-1.0 + 4*PETSC_PI*PETSC_PI)+2.*PETSC_PI*ul[jx][ix].v)*
      //              PetscCosScalar(2.*PETSC_PI*coors[jx][ix].y)-(appctx->param.mu +2.*PETSC_PI*ul[jx][ix].u)*PetscSinScalar(2.*PETSC_PI*coors[jx][ix].x));
      //       xpy=(coors[jx][ix].x*coors[jx][ix].x+coors[jx][ix].y*coors[jx][ix].y);
      //       tempu=PetscExpScalar(-appctx->param.mu*tt - 10.0*xpy);
      //       tempv=PetscExpScalar(-appctx->param.mu*tt - 12.0*xpy);

      //ff[jx][ix].u=PetscExpScalar(-appctx->param.mu*tt - 10.0*xpy)*(appctx->param.mu*(19.0 - 400.0*coors[jx][ix].x*coors[jx][ix].x) - 20.0*(tempu*coors[jx][ix].x + tempv*coors[jx][ix].y));
      //ff[jx][ix].v=PetscExpScalar(-appctx->param.mu*tt - 12.0*xpy)*(-20.0*PetscExpScalar(2.0*xpy)*(tempu*coors[jx][ix].x + tempv*coors[jx][ix].y) + appctx->param.mu* (23.0 - 576.0*coors[jx][ix].y*coors[jx][ix].y));

        xpy=0.25*PETSC_PI*PETSC_PI;
        tempu=PetscExpScalar(-appctx->param.mu*tt)*(PetscCosScalar(0.5*PETSC_PI*coors[jx][ix].x)+PetscSinScalar(0.5*PETSC_PI*coors[jx][ix].y))/10.0;
        tempv=PetscExpScalar(-appctx->param.mu*tt)*(PetscSinScalar(0.5*PETSC_PI*coors[jx][ix].x)+PetscCosScalar(0.5*PETSC_PI*coors[jx][ix].y))/10.0;
ff[jx][ix].u=PetscExpScalar(-appctx->param.mu*tt) *((-0.1 + 0.1*xpy)*appctx->param.mu*PetscCosScalar(0.5*PETSC_PI*coors[jx][ix].x) + 0.1* 0.5*PETSC_PI*tempv* PetscCosScalar(0.5*PETSC_PI*coors[jx][ix].y) * 0.1*0.5*PETSC_PI*tempu*PetscSinScalar(0.5*PETSC_PI*coors[jx][ix].x) - 0.1*appctx->param.mu*PetscSinScalar(0.5*PETSC_PI*coors[jx][ix].y));
ff[jx][ix].v=PetscExpScalar(-appctx->param.mu*tt)* (((-0.1 + 0.1*xpy)*appctx->param.mu + 0.1*0.5*PETSC_PI*tempv)*PetscCosScalar(0.5*PETSC_PI*coors[jx][ix].y) + (-0.1* appctx->param.mu- 0.1*0.5*PETSC_PI*tempu)*PetscSinScalar(0.5*PETSC_PI*coors[jx][ix].x));
      } 
     }
  ierr = DMDAVecRestoreArray(appctx->da,forcing,&ff);CHKERRQ(ierr);
  VecAXPY(globalout,1.0,forcing);
  */


  ierr = PetscGLLElementLaplacianDestroy(&appctx->SEMop.gll,&stiff);CHKERRQ(ierr);
  ierr = PetscGLLElementAdvectionDestroy(&appctx->SEMop.gll,&grad);CHKERRQ(ierr);
  ierr = PetscGLLElementMassDestroy(&appctx->SEMop.gll,&mass);CHKERRQ(ierr);

  //ierr = VecDestroy(&outloc);CHKERRQ(ierr);
  //ierr = VecDestroy(&uloc);CHKERRQ(ierr);
/*
  its=its+1;
  //printf("time to write %f ",&t); 
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"rhsB.m",&viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  PetscSNPrintf(var,sizeof(var),"inr(:,%d)",its);
  ierr = PetscObjectSetName((PetscObject)globalin,var);
  ierr = VecView(globalin,viewfile);CHKERRQ(ierr);
  PetscSNPrintf(var,sizeof(var),"outr(:,%d)",its);
  ierr = PetscObjectSetName((PetscObject)globalout,var);
 ierr = VecView(globalout,viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewfile);
  
  exit(1);
  */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MyMatMult"

PetscErrorCode MyMatMult(Mat H, Vec in, Vec out)
 {
   AppCtx         *appctx;
   
   const Field<double> **ul, **uj;
   Field<double>   **outl;
   PetscScalar     **stiff, **mass, **grad;
   PetscScalar     **wrk1, **wrk2, **wrk3, **wrk4, **wrk5, **wrk6, **wrk7; 
   PetscScalar     **ulb, **vlb, **ujb, **vjb;
   PetscInt        Nl, Nl2, inc;
   PetscInt        xs,ys,xm,ym,ix,iy,jx,jy, indx,indy, i;
   PetscErrorCode  ierr;
   Vec             uloc, outloc, ujloc;
   PetscViewer     viewfile;
   PetscScalar     alpha, beta;
   static int its=0;
   char var[12] ;

 
  MatShellGetContext(H,&appctx);

  ierr = PetscGLLElementLaplacianCreate(&appctx->SEMop.gll,&stiff);CHKERRQ(ierr);
  ierr = PetscGLLElementMassCreate(&appctx->SEMop.gll,&mass);CHKERRQ(ierr); 
  ierr = PetscGLLElementAdvectionCreate(&appctx->SEMop.gll,&grad);CHKERRQ(ierr);

  /* unwrap local vector for the input solution */
  DMCreateLocalVector(appctx->da,&uloc);

  DMGlobalToLocalBegin(appctx->da,in,INSERT_VALUES,uloc);
  DMGlobalToLocalEnd(appctx->da,in,INSERT_VALUES,uloc);

  DMDAVecGetArrayRead(appctx->da,uloc,&ul);CHKERRQ(ierr);

  // vector form jacobian
  DMCreateLocalVector(appctx->da,&ujloc);

  DMGlobalToLocalBegin(appctx->da,appctx->dat.pass_sol,INSERT_VALUES,ujloc);
  DMGlobalToLocalEnd(appctx->da,appctx->dat.pass_sol,INSERT_VALUES,ujloc);

  DMDAVecGetArrayRead(appctx->da,ujloc,&uj);CHKERRQ(ierr);
 
  /* unwrap local vector for the output solution */
  DMCreateLocalVector(appctx->da,&outloc);
  VecSet(outloc,0.0);

  ierr = DMDAVecGetArray(appctx->da,outloc,&outl);CHKERRQ(ierr);
 
  ierr = DMDAGetCorners(appctx->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  Nl    = appctx->param.N; 
    
  xs=xs/(Nl-1);
  xm=xm/(Nl-1);
  ys=ys/(Nl-1);
  ym=ym/(Nl-1);
/*
  Field<t1s> **t1s_ul = new Field<t1s>*[Nl*xm];
  for(int i = 0; i < Nl*xm; i++) t1s_ul[i] = new Field<t1s>[Nl*ym];
  Field<t1s> **t1s_outl = new Field<t1s>*[Nl*xm];
  for(int i = 0; i < Nl*xm; i++) t1s_outl[i] = new Field<t1s>[Nl*ym];
  for(ix = 0; ix < (Nl-1)*xm + 1; ix++) 
  {
    for(iy = 0; iy < (Nl-1)*ym + 1; iy++) 
    {
    t1s_ul[iy][ix].v = uj[iy][ix].v;
    t1s_ul[iy][ix].u = uj[iy][ix].u;
    t1s_ul[iy][ix].v.setGradient(ul[iy][ix].v);
    t1s_ul[iy][ix].u.setGradient(ul[iy][ix].u);
    t1s_outl[iy][ix].v = 0.0;
    t1s_outl[iy][ix].u = 0.0;
    }
  }
  ierr = ADRHSFunction<t1s>(t1s_outl, t1s_ul, (void*) appctx);
  for(ix = 0; ix < (Nl-1)*xm + 1; ix++){ 
    for(iy = 0; iy < (Nl-1)*ym + 1; iy++)
    {
      outl[iy][ix].v = t1s_outl[iy][ix].v.gradient();
      outl[iy][ix].u = t1s_outl[iy][ix].u.gradient();
    }
  }
  for(int i = 0; i < Nl*xm; i++) delete [] t1s_ul[i];
  for(int i = 0; i < Nl*xm; i++) delete [] t1s_outl[i];
  delete [] t1s_ul;
  delete [] t1s_outl;  
  /*
     Initialize work arrays
  */ 

  ierr = PetscMalloc1(appctx->param.N,&ulb);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&ulb[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) ulb[i] = ulb[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&vlb);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&vlb[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) vlb[i] = vlb[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&ujb);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&ujb[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) ujb[i] = ujb[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&vjb);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&vjb[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) vjb[i] = vjb[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&wrk1);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&wrk1[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) wrk1[i] = wrk1[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&wrk2);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&wrk2[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) wrk2[i] = wrk2[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&wrk3);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&wrk3[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) wrk3[i] = wrk3[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&wrk4);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&wrk4[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) wrk4[i] = wrk4[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&wrk5);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&wrk5[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) wrk5[i] = wrk5[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&wrk6);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&wrk6[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) wrk6[i] = wrk6[i-1]+Nl;


  alpha = 1.0;
  beta  = 0.0;
  Nl2= Nl*Nl;
  inc=1;
   for (ix=xs; ix<xs+xm; ix++) 
      {for (iy=ys; iy<ys+ym; iy++) 
         { 
       for (jx=0; jx<appctx->param.N; jx++) 
        {for (jy=0; jy<appctx->param.N; jy++) 
               
           {ulb[jy][jx]=0.0;
            ujb[jy][jx]=0.0;
            vlb[jy][jx]=0.0;
            vjb[jy][jx]=0.0;
            indx=ix*(appctx->param.N-1)+jx;
            indy=iy*(appctx->param.N-1)+jy;
            ujb[jy][jx]=uj[indy][indx].u; 
            vjb[jy][jx]=uj[indy][indx].v; 
            ulb[jy][jx]=ul[indy][indx].u; 
            vlb[jy][jx]=ul[indy][indx].v; 
            wrk4[jy][jx]=0.0;
          }}
       //here the stifness matrix in 2d
        //first product (B x K_yy) u=W2 (u_yy)
        alpha=appctx->param.Lex/2.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&mass[0][0],&Nl,&ulb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=2./appctx->param.Ley;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&stiff[0][0],&Nl,&beta,&wrk2[0][0],&Nl);

        //second product (K_xx x B) u=W3 (u_xx)
        alpha=2.0/appctx->param.Lex;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&stiff[0][0],&Nl,&ulb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=appctx->param.Ley/2.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&mass[0][0],&Nl,&beta,&wrk3[0][0],&Nl);

        alpha=1.0;
        BLASaxpy_(&Nl2,&alpha, &wrk3[0][0],&inc,&wrk2[0][0],&inc); //I freed wrk3 and saved the lalplacian in wrk2
       
        // for the v component now 
        //first product (B x K_yy) v=W3
        alpha=appctx->param.Lex/2.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&mass[0][0],&Nl,&vlb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=2.0/appctx->param.Ley;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&stiff[0][0],&Nl,&beta,&wrk3[0][0],&Nl);

        //second product (K_xx x B) v=W4
        alpha=2.0/appctx->param.Lex;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&stiff[0][0],&Nl,&vlb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=appctx->param.Ley/2.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&mass[0][0],&Nl,&beta,&wrk4[0][0],&Nl);

        alpha=1.0;
        BLASaxpy_(&Nl2,&alpha, &wrk4[0][0],&inc,&wrk3[0][0],&inc); //I freed wrk4 and saved the lalplacian in wrk3

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    

       //now the gradient operator for u
        // first (D_x x B) wu the term ujb.(D_x x B) wu
        alpha=appctx->param.Lex/2.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&mass[0][0],&Nl,&ulb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=1.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&grad[0][0],&Nl,&beta,&wrk4[0][0],&Nl);

        PetscPointWiseMult(Nl2, &wrk4[0][0], &ujb[0][0], &wrk4[0][0]); 
        
       // (D_x x B) u the term ulb.(D_x x B) u
        alpha=appctx->param.Lex/2.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&mass[0][0],&Nl,&ujb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=1.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&grad[0][0],&Nl,&beta,&wrk5[0][0],&Nl);

        PetscPointWiseMult(Nl2, &wrk5[0][0], &ulb[0][0], &wrk5[0][0]); 

        alpha=1.0;
        BLASaxpy_(&Nl2,&alpha, &wrk5[0][0],&inc,&wrk4[0][0],&inc); // saving in wrk4



        // first (B x D_y) wu the term vjb.(B x D_x) wu 
        alpha=1.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&grad[0][0],&Nl,&ulb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=appctx->param.Ley/2.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&mass[0][0],&Nl,&beta,&wrk5[0][0],&Nl);

        PetscPointWiseMult(Nl2, &wrk5[0][0], &vjb[0][0], &wrk5[0][0]); 

        alpha=1.0;
        BLASaxpy_(&Nl2,&alpha, &wrk5[0][0],&inc,&wrk4[0][0],&inc); // saving in wrk4

        // first (B x D_y) u the term vlb.(B x D_x) u !!!
        alpha=1.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&grad[0][0],&Nl,&ujb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=appctx->param.Ley/2.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&mass[0][0],&Nl,&beta,&wrk5[0][0],&Nl);

        PetscPointWiseMult(Nl2, &wrk5[0][0], &vlb[0][0], &wrk5[0][0]); 

        alpha=1.0;
        BLASaxpy_(&Nl2,&alpha, &wrk5[0][0],&inc,&wrk4[0][0],&inc); // saving in wrk4


//////////////////////////////////// the second equation
        

       // (D_x x B) wv the term ujb.(D_x x B) wv
        alpha=appctx->param.Lex/2.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&mass[0][0],&Nl,&vlb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=1.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&grad[0][0],&Nl,&beta,&wrk5[0][0],&Nl);

        PetscPointWiseMult(Nl2, &wrk5[0][0], &ujb[0][0], &wrk5[0][0]); 

       // (D_x x B) v the term ulb.(D_x x B) v !!!
         alpha=appctx->param.Lex/2.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&mass[0][0],&Nl,&vjb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=1.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&grad[0][0],&Nl,&beta,&wrk6[0][0],&Nl);

        PetscPointWiseMult(Nl2, &wrk6[0][0], &ulb[0][0], &wrk6[0][0]); 

        alpha=1.0;
        BLASaxpy_(&Nl2,&alpha, &wrk6[0][0],&inc,&wrk5[0][0],&inc); // saving in wrk5

        // first (B x D_y) v the term vlb.(B x D_x) v
        alpha=1.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&grad[0][0],&Nl,&vjb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=appctx->param.Ley/2.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&mass[0][0],&Nl,&beta,&wrk6[0][0],&Nl);         

        PetscPointWiseMult(Nl2, &wrk6[0][0], &vlb[0][0], &wrk6[0][0]); 

        alpha=1.0;
        BLASaxpy_(&Nl2,&alpha, &wrk6[0][0],&inc,&wrk5[0][0],&inc); // saving in wrk5

      
        // first (B x D_y) wv the term vjb.(B x D_x) wv
        alpha=1.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&grad[0][0],&Nl,&vlb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=appctx->param.Ley/2.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&mass[0][0],&Nl,&beta,&wrk6[0][0],&Nl);

      
        PetscPointWiseMult(Nl2, &wrk6[0][0], &vjb[0][0], &wrk6[0][0]); 

        alpha=1.0;
        BLASaxpy_(&Nl2,&alpha, &wrk6[0][0],&inc,&wrk5[0][0],&inc); // saving in wrk5


        for (jx=0; jx<appctx->param.N; jx++) 
        {for (jy=0; jy<appctx->param.N; jy++)   
           {indx=ix*(appctx->param.N-1)+jx;
            indy=iy*(appctx->param.N-1)+jy;
            
            outl[indy][indx].u += appctx->param.mu*(wrk2[jy][jx])+wrk4[jy][jx];
            outl[indy][indx].v += appctx->param.mu*(wrk3[jy][jx])+wrk5[jy][jx];

            //printf("outl[%d][%d]=%0.15f\n", indx,indy, outl[indy][indx]);
           }}
       }
     }
  
  ierr = DMDAVecRestoreArray(appctx->da,outloc,&outl);CHKERRQ(ierr);
  DMDAVecRestoreArrayRead(appctx->da,in,&uloc);CHKERRQ(ierr);
  DMDAVecRestoreArrayRead(appctx->da,appctx->dat.pass_sol,&ujloc);CHKERRQ(ierr);

  VecSet(out,0.0);

  DMLocalToGlobalBegin(appctx->da,outloc,ADD_VALUES,out);
  DMLocalToGlobalEnd(appctx->da,outloc,ADD_VALUES,out);

  VecScale(out, -1.0);
  ierr = VecPointwiseDivide(out,out,appctx->SEMop.mass);CHKERRQ(ierr);

  //VecView(out,0);

  ierr = PetscGLLElementLaplacianDestroy(&appctx->SEMop.gll,&stiff);CHKERRQ(ierr);
  ierr = PetscGLLElementAdvectionDestroy(&appctx->SEMop.gll,&grad);CHKERRQ(ierr);
  ierr = PetscGLLElementMassDestroy(&appctx->SEMop.gll,&mass);CHKERRQ(ierr);
 
  ierr = PetscFree((wrk1)[0]);CHKERRQ(ierr);
  ierr = PetscFree(wrk1);CHKERRQ(ierr);
  ierr = PetscFree((wrk2)[0]);CHKERRQ(ierr);
  ierr = PetscFree(wrk2);CHKERRQ(ierr);
  ierr = PetscFree((wrk3)[0]);CHKERRQ(ierr);
  ierr = PetscFree(wrk3);CHKERRQ(ierr);
  ierr = PetscFree((wrk4)[0]);CHKERRQ(ierr);
  ierr = PetscFree(wrk4);CHKERRQ(ierr);
  ierr = PetscFree((wrk5)[0]);CHKERRQ(ierr);
  ierr = PetscFree(wrk5);CHKERRQ(ierr);
  ierr = PetscFree((wrk6)[0]);CHKERRQ(ierr);
  ierr = PetscFree(wrk6);CHKERRQ(ierr);
  

/*
  its=its+1;
  //printf("time to write %f ",&t); 
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"jacin.m",&viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  PetscSNPrintf(var,sizeof(var),"in(:,%d)",its);
  ierr = PetscObjectSetName((PetscObject)in,var);
  ierr = VecView(in,viewfile);CHKERRQ(ierr);
  PetscSNPrintf(var,sizeof(var),"out(:,%d)",its);
  ierr = PetscObjectSetName((PetscObject)out,var);
  ierr = VecView(out,viewfile);CHKERRQ(ierr);
  //PetscSNPrintf(var,sizeof(var),"mass",its);
  //ierr = PetscObjectSetName((PetscObject)appctx->SEMop.mass,var);
  //ierr = VecView(appctx->SEMop.mass,viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewfile);

 */
   return(0);
 }


#undef __FUNCT__
#define __FUNCT__ "MyMatMultTransp"

PetscErrorCode MyMatMultTransp(Mat H, Vec in, Vec out)
 {
   AppCtx         *appctx;
   
   const Field<double> **ul, **uj;
   Field<double>   **outl;
   PetscScalar     **stiff, **mass, **grad;
   PetscScalar     **wrk1, **wrk2, **wrk3, **wrk4, **wrk5, **wrk6, **wrk7; 
   PetscScalar     **ulb, **vlb, **ujb, **vjb;
   PetscInt        Nl, Nl2, inc;
   PetscInt        xs,ys,xm,ym,ix,iy,jx,jy, indx,indy, i;
   PetscErrorCode  ierr;
   Vec             uloc, outloc, ujloc, incopy;
   PetscViewer     viewfile;
   PetscScalar     alpha, beta;
   static int its=0;
   char var[12] ;

 
  MatShellGetContext(H,&appctx);

  ierr = PetscGLLElementLaplacianCreate(&appctx->SEMop.gll,&stiff);CHKERRQ(ierr);
  ierr = PetscGLLElementMassCreate(&appctx->SEMop.gll,&mass);CHKERRQ(ierr); 
  ierr = PetscGLLElementAdvectionCreate(&appctx->SEMop.gll,&grad);CHKERRQ(ierr);

  VecDuplicate(in,&incopy);
  VecCopy(in,incopy);
  ierr = VecPointwiseDivide(incopy,in,appctx->SEMop.mass);CHKERRQ(ierr);
  
  /* unwrap local vector for the input solution */
  DMCreateLocalVector(appctx->da,&uloc);

  DMGlobalToLocalBegin(appctx->da,incopy,INSERT_VALUES,uloc);
  DMGlobalToLocalEnd(appctx->da,incopy,INSERT_VALUES,uloc);

  DMDAVecGetArrayRead(appctx->da,uloc,&ul);CHKERRQ(ierr);

  // vector form jacobian
  DMCreateLocalVector(appctx->da,&ujloc);

  DMGlobalToLocalBegin(appctx->da,appctx->dat.pass_sol,INSERT_VALUES,ujloc);
  DMGlobalToLocalEnd(appctx->da,appctx->dat.pass_sol,INSERT_VALUES,ujloc);

  DMDAVecGetArrayRead(appctx->da,ujloc,&uj);CHKERRQ(ierr);
 
  /* unwrap local vector for the output solution */
  DMCreateLocalVector(appctx->da,&outloc);
  VecSet(outloc,0.0);

  ierr = DMDAVecGetArray(appctx->da,outloc,&outl);CHKERRQ(ierr);
 
  ierr = DMDAGetCorners(appctx->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  Nl    = appctx->param.N; 
    
  xs=xs/(Nl-1);
  xm=xm/(Nl-1);
  ys=ys/(Nl-1);
  ym=ym/(Nl-1);
  
  /*
     Initialize work arrays
  */ 

  ierr = PetscMalloc1(appctx->param.N,&ulb);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&ulb[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) ulb[i] = ulb[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&vlb);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&vlb[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) vlb[i] = vlb[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&ujb);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&ujb[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) ujb[i] = ujb[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&vjb);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&vjb[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) vjb[i] = vjb[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&wrk1);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&wrk1[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) wrk1[i] = wrk1[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&wrk2);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&wrk2[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) wrk2[i] = wrk2[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&wrk3);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&wrk3[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) wrk3[i] = wrk3[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&wrk4);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&wrk4[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) wrk4[i] = wrk4[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&wrk5);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&wrk5[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) wrk5[i] = wrk5[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&wrk6);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&wrk6[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) wrk6[i] = wrk6[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&wrk7);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&wrk7[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) wrk7[i] = wrk7[i-1]+Nl;


  alpha = 1.0;
  beta  = 0.0;
  Nl2= Nl*Nl;
  inc=1;
  for (ix=xs; ix<xs+xm; ix++) 
      {for (iy=ys; iy<ys+ym; iy++) 
         { 
       for (jx=0; jx<appctx->param.N; jx++) 
        {for (jy=0; jy<appctx->param.N; jy++) 
               
           {ulb[jy][jx]=0.0;
            ujb[jy][jx]=0.0;
            vlb[jy][jx]=0.0;
            vjb[jy][jx]=0.0;
            indx=ix*(appctx->param.N-1)+jx;
            indy=iy*(appctx->param.N-1)+jy;
            ujb[jy][jx]=uj[indy][indx].u; 
            vjb[jy][jx]=uj[indy][indx].v; 
            ulb[jy][jx]=ul[indy][indx].u; 
            vlb[jy][jx]=ul[indy][indx].v; 
            
          }}

       //here the stifness matrix in 2d
        //first product (B x K_yy)u=W2 (u_yy)
        alpha=appctx->param.Lex/2.0;
        BLASgemm_("T","N",&Nl,&Nl,&Nl,&alpha,&mass[0][0],&Nl,&ulb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=2./appctx->param.Ley;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&stiff[0][0],&Nl,&beta,&wrk2[0][0],&Nl);

        //second product (K_xx x B) u=W3 (u_xx)
        alpha=2.0/appctx->param.Lex;
        BLASgemm_("T","N",&Nl,&Nl,&Nl,&alpha,&stiff[0][0],&Nl,&ulb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=appctx->param.Ley/2.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&mass[0][0],&Nl,&beta,&wrk3[0][0],&Nl);

        alpha=1.0;
        BLASaxpy_(&Nl2,&alpha, &wrk3[0][0],&inc,&wrk2[0][0],&inc); //I freed wrk3 and saved the lalplacian in wrk2
       
        // for the v component now 
        //first product (B x K_yy)v=W3
        alpha=appctx->param.Lex/2.0;
        BLASgemm_("T","N",&Nl,&Nl,&Nl,&alpha,&mass[0][0],&Nl,&vlb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=2.0/appctx->param.Ley;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&stiff[0][0],&Nl,&beta,&wrk3[0][0],&Nl);

        //second product (K_xx x B)v=W4
        alpha=2.0/appctx->param.Lex;
        BLASgemm_("T","N",&Nl,&Nl,&Nl,&alpha,&stiff[0][0],&Nl,&vlb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=appctx->param.Ley/2.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&mass[0][0],&Nl,&beta,&wrk4[0][0],&Nl);

        alpha=1.0;
        BLASaxpy_(&Nl2,&alpha, &wrk4[0][0],&inc,&wrk3[0][0],&inc); //I freed wrk4 and saved the lalplacian in wrk3

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    

       //now the gradient operator for u
        // first (D_x x B) wu the term (D_x x B) wu.ujb

        PetscPointWiseMult(Nl2, &ulb[0][0], &ujb[0][0], &wrk6[0][0]); 

        alpha=appctx->param.Lex/2.0;
        BLASgemm_("T","N",&Nl,&Nl,&Nl,&alpha,&mass[0][0],&Nl,&wrk6[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=1.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&grad[0][0],&Nl,&beta,&wrk4[0][0],&Nl);

        
        // (D_x x B) u the term ulb.(D_x x B) u
        alpha=appctx->param.Lex/2.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&mass[0][0],&Nl,&ujb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=1.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&grad[0][0],&Nl,&beta,&wrk5[0][0],&Nl);

        PetscPointWiseMult(Nl2, &wrk5[0][0], &ulb[0][0], &wrk5[0][0]); //same term

        alpha=1.0;
        BLASaxpy_(&Nl2,&alpha, &wrk5[0][0],&inc,&wrk4[0][0],&inc); // saving in wrk4


        // first (B x D_y) wu the term vjb.(B x D_x) wu 

        PetscPointWiseMult(Nl2, &ulb[0][0], &vjb[0][0], &wrk6[0][0]); 

        alpha=1.0;
        BLASgemm_("T","N",&Nl,&Nl,&Nl,&alpha,&grad[0][0],&Nl,&wrk6[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
         alpha=appctx->param.Ley/2.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&mass[0][0],&Nl,&beta,&wrk5[0][0],&Nl);

        alpha=1.0;
        BLASaxpy_(&Nl2,&alpha, &wrk5[0][0],&inc,&wrk4[0][0],&inc); // saving in wrk4

         // (D_x x B) v the term vlb.(D_x x B) v
        alpha=appctx->param.Lex/2.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&mass[0][0],&Nl,&vjb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=1.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&grad[0][0],&Nl,&beta,&wrk5[0][0],&Nl);

        PetscPointWiseMult(Nl2, &wrk5[0][0], &vlb[0][0], &wrk5[0][0]); 

        alpha=1.0;
        BLASaxpy_(&Nl2,&alpha, &wrk5[0][0],&inc,&wrk4[0][0],&inc); // saving in wrk5


//////////////////////////////////// the second equation
        

       // (D_x x B) wv the term ujb.(D_x x B) wv

        PetscPointWiseMult(Nl2, &vlb[0][0], &ujb[0][0], &wrk7[0][0]); 
        alpha=appctx->param.Lex/2.0;
        BLASgemm_("T","N",&Nl,&Nl,&Nl,&alpha,&mass[0][0],&Nl,&wrk7[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=1.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&grad[0][0],&Nl,&beta,&wrk5[0][0],&Nl);

     

        // first (B x D_y) u the term ulb.(B x D_x) u       /////////same term B
        alpha=1.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&grad[0][0],&Nl,&ujb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=appctx->param.Ley/2.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&mass[0][0],&Nl,&beta,&wrk6[0][0],&Nl);

        PetscPointWiseMult(Nl2, &wrk6[0][0], &ulb[0][0], &wrk6[0][0]); 

        alpha=1.0;
        BLASaxpy_(&Nl2,&alpha, &wrk6[0][0],&inc,&wrk5[0][0],&inc); // saving in wrk5


        
        // first (B x D_y) v the term vjb.(B x D_x) wv
        PetscPointWiseMult(Nl2, &vlb[0][0], &vjb[0][0], &wrk7[0][0]); 
        alpha=1.0;
        BLASgemm_("T","N",&Nl,&Nl,&Nl,&alpha,&grad[0][0],&Nl,&wrk7[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=appctx->param.Ley/2.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&mass[0][0],&Nl,&beta,&wrk6[0][0],&Nl);         

        alpha=1.0;
        BLASaxpy_(&Nl2,&alpha, &wrk6[0][0],&inc,&wrk5[0][0],&inc); // saving in wrk5
        
        // first (B x D_y) wv the term vlb.(B x D_x) v
        alpha=1.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&grad[0][0],&Nl,&vjb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=appctx->param.Ley/2.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&mass[0][0],&Nl,&beta,&wrk6[0][0],&Nl);

      
        PetscPointWiseMult(Nl2, &wrk6[0][0], &vlb[0][0], &wrk6[0][0]); 

        alpha=1.0;
        BLASaxpy_(&Nl2,&alpha, &wrk6[0][0],&inc,&wrk5[0][0],&inc); // saving in wrk5


        for (jx=0; jx<appctx->param.N; jx++) 
        {for (jy=0; jy<appctx->param.N; jy++)   
           {indx=ix*(appctx->param.N-1)+jx;
            indy=iy*(appctx->param.N-1)+jy;
            
           outl[indy][indx].u += appctx->param.mu*(wrk2[jy][jx])+wrk4[jy][jx];
           outl[indy][indx].v += appctx->param.mu*(wrk3[jy][jx])+wrk5[jy][jx];
            //printf("outl[%d][%d]=%0.15f\n", indx,indy, outl[indy][indx]);
           }}
       }
     }
  
  ierr = DMDAVecRestoreArray(appctx->da,outloc,&outl);CHKERRQ(ierr);
  DMDAVecRestoreArrayRead(appctx->da,in,&uloc);CHKERRQ(ierr);
  DMDAVecRestoreArrayRead(appctx->da,appctx->dat.pass_sol,&ujloc);CHKERRQ(ierr);

  VecSet(out,0.0);

  DMLocalToGlobalBegin(appctx->da,outloc,ADD_VALUES,out);
  DMLocalToGlobalEnd(appctx->da,outloc,ADD_VALUES,out);

  VecScale(out, -1.0);
  //ierr = VecPointwiseDivide(out,out,appctx->SEMop.mass);CHKERRQ(ierr);


  ierr = PetscGLLElementLaplacianDestroy(&appctx->SEMop.gll,&stiff);CHKERRQ(ierr);
  ierr = PetscGLLElementAdvectionDestroy(&appctx->SEMop.gll,&grad);CHKERRQ(ierr);
  ierr = PetscGLLElementMassDestroy(&appctx->SEMop.gll,&mass);CHKERRQ(ierr);
 
  ierr = PetscFree((wrk1)[0]);CHKERRQ(ierr);
  ierr = PetscFree(wrk1);CHKERRQ(ierr);
  ierr = PetscFree((wrk2)[0]);CHKERRQ(ierr);
  ierr = PetscFree(wrk2);CHKERRQ(ierr);
  ierr = PetscFree((wrk3)[0]);CHKERRQ(ierr);
  ierr = PetscFree(wrk3);CHKERRQ(ierr);
  ierr = PetscFree((wrk4)[0]);CHKERRQ(ierr);
  ierr = PetscFree(wrk4);CHKERRQ(ierr);
  ierr = PetscFree((wrk5)[0]);CHKERRQ(ierr);
  ierr = PetscFree(wrk5);CHKERRQ(ierr);
  ierr = PetscFree((wrk6)[0]);CHKERRQ(ierr);
  ierr = PetscFree(wrk6);CHKERRQ(ierr);
  ierr = PetscFree((wrk7)[0]);CHKERRQ(ierr);
  ierr = PetscFree(wrk7);CHKERRQ(ierr);
/*
  its=its+1;
  //printf("time to write %f ",&t); 
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"jacin.m",&viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  PetscSNPrintf(var,sizeof(var),"in(:,%d)",its);
  ierr = PetscObjectSetName((PetscObject)in,var);
  ierr = VecView(in,viewfile);CHKERRQ(ierr);
  PetscSNPrintf(var,sizeof(var),"out(:,%d)",its);
  ierr = PetscObjectSetName((PetscObject)out,var);
  ierr = VecView(out,viewfile);CHKERRQ(ierr);
  //PetscSNPrintf(var,sizeof(var),"mass",its);
  //ierr = PetscObjectSetName((PetscObject)appctx->SEMop.mass,var);
  //ierr = VecView(appctx->SEMop.mass,viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewfile);

 */
   return(0);
 }




#undef __FUNCT__
#define __FUNCT__ "RHSJacobian"
PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec globalin, Mat A, Mat B,void *ctx)
{
  PetscErrorCode ierr;
  AppCtx         *appctx = (AppCtx*)ctx;  
  PetscFunctionBegin;

  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

  VecCopy(globalin, appctx->dat.pass_sol);
 
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------ */
/*
   FormFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   IC   - the input vector
   ctx - optional user-defined context, as set when calling TaoSetObjectiveAndGradientRoutine()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient

   Notes:

          The forward equation is
              M u_t = F(U)
          which is converted to
                u_t = M^{-1} F(u)
          in the user code since TS has no direct way of providing a mass matrix. The Jacobian of this is
                 M^{-1} J
          where J is the Jacobian of F. Now the adjoint equation is
                M v_t = J^T v
          but TSAdjoint does not solve this since it can only solve the transposed system for the 
          Jacobian the user provided. Hence TSAdjoint solves
                 w_t = J^T M^{-1} w  (where w = M v)
          since there is no way to indicate the mass matrix as a seperate entitity to TS. Thus one
          must be careful in initializing the "adjoint equation" and using the result. This is
          why
              G = -2 M(u(T) - u_d)
          below (instead of -2(u(T) - u_d) and why the result is
              G = G/appctx->SEMop.mass (that is G = M^{-1}w)
          below (instead of just the result of the "adjoint solve").


*/
PetscErrorCode FormFunctionGradient(Tao tao,Vec IC,PetscReal *f,Vec G,void *ctx)
{
  AppCtx           *appctx = (AppCtx*)ctx;     /* user-defined application context */
  PetscErrorCode    ierr;
  Vec               temp, bsol, adj;
  PetscInt          its;
  PetscReal         ff, gnorm, cnorm, xdiff,errex; 
  TaoConvergedReason reason;    
  PetscViewer        viewfile;
  //static int counter=0; it was considered for storing line search error
  char filename[24] ;
  char data[80] ;
  
  ierr = TSSetTime(appctx->ts,0.0);CHKERRQ(ierr);
  ierr = TSSetStepNumber(appctx->ts,0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(appctx->ts,appctx->initial_dt);CHKERRQ(ierr);
  ierr = VecCopy(IC,appctx->dat.curr_sol);CHKERRQ(ierr);

  ierr = TSSolve(appctx->ts,appctx->dat.curr_sol);CHKERRQ(ierr);
 //counter++; // this was for storing the error accross line searches
  /*
  PetscSNPrintf(filename,sizeof(filename),"inside.m",its);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename,&viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx->dat.curr_sol,"fwd");
  ierr = VecView(appctx->dat.curr_sol,viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);

*/
  /*
  Store current solution for comparison
  */
  ierr = VecDuplicate(appctx->dat.curr_sol,&bsol);CHKERRQ(ierr);
  ierr = VecCopy(appctx->dat.curr_sol,bsol);CHKERRQ(ierr);
  
  ierr = VecWAXPY(G,-1.0,appctx->dat.curr_sol,appctx->dat.obj);CHKERRQ(ierr);

  /*
     Compute the L2-norm of the objective function, cost function is f
  */
  ierr = VecDuplicate(G,&temp);CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp,G,G);CHKERRQ(ierr);
  ierr = VecDot(temp,appctx->SEMop.mass,f);CHKERRQ(ierr);
  ierr = VecDestroy(&temp);CHKERRQ(ierr);

  //local error evaluation   
  ierr = VecDuplicate(G,&temp);CHKERRQ(ierr);
  ierr = VecDuplicate(appctx->dat.ic,&temp);CHKERRQ(ierr);
  ierr = VecWAXPY(temp,-1.0,appctx->dat.ic,appctx->dat.true_solution);CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp,temp,temp);CHKERRQ(ierr);
  //for error evaluation
  ierr = VecDot(temp,appctx->SEMop.mass,&errex);CHKERRQ(ierr);
  ierr = VecDestroy(&temp);CHKERRQ(ierr);
  errex  = PetscSqrtReal(errex); 

/*
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename,&viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)G,"inb");
  ierr = VecView(G,viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
*/

/*
     Compute initial conditions for the adjoint integration. See Notes above
  */

  ierr = VecScale(G, -2.0);CHKERRQ(ierr);
  //VecView(G,0);
  ierr = VecPointwiseMult(G,G,appctx->SEMop.mass);CHKERRQ(ierr);
  ierr = TSSetCostGradients(appctx->ts,1,&G,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(G,&adj);CHKERRQ(ierr);
  ierr = VecCopy(G,adj);CHKERRQ(ierr);

  ierr = TSAdjointSolve(appctx->ts);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(G,G,appctx->SEMop.mass);CHKERRQ(ierr);

  ierr=  TaoGetSolutionStatus(tao, &its, &ff, &gnorm, &cnorm, &xdiff, &reason);

  //counter++; // this was for storing the error accross line searches
  PetscPrintf(PETSC_COMM_WORLD,"iteration=%D\t cost function (TAO)=%g, cost function (L2 %g), ic error %g\n",its,(double)ff,*f,errex);
  PetscSNPrintf(filename,sizeof(filename),"PDEadjoint/optimize%02d.m",its);
  ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename,&viewfile);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewfile,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  PetscSNPrintf(data,sizeof(data),"TAO(%D)=%g; L2(%D)= %g ; Err(%D)=%g\n",its+1,(double)ff,its+1,*f,its+1,errex);
  PetscViewerASCIIPrintf(viewfile,data);
  ierr = PetscObjectSetName((PetscObject)appctx->dat.obj,"obj");
  ierr = VecView(appctx->dat.obj,viewfile);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)G,"Init_adj");
  ierr = VecView(G,viewfile);CHKERRQ(ierr);
ierr = PetscObjectSetName((PetscObject)adj,"adj");
  ierr = VecView(adj,viewfile);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)IC,  "Init_ts");
  ierr = VecView(IC,viewfile);CHKERRQ(ierr);
  //ierr = PetscObjectSetName((PetscObject)appctx->dat.senmask,  "senmask");
  //ierr = VecView(appctx->dat.senmask,viewfile);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)bsol,"fwd");
  ierr = VecView(bsol,viewfile);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx->dat.curr_sol,"Curr_sol");
  ierr = VecView(appctx->dat.curr_sol,viewfile);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)appctx->dat.true_solution, "exact");
  ierr = VecView(appctx->dat.true_solution,viewfile);CHKERRQ(ierr);
   ierr = PetscViewerPopFormat(viewfile);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewfile);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode MonitorError(Tao tao,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;
  Vec            temp;
  PetscReal      nrm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(appctx->dat.ic,&temp);CHKERRQ(ierr);
  ierr = VecWAXPY(temp,-1.0,appctx->dat.ic,appctx->dat.true_solution);CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp,temp,temp);CHKERRQ(ierr);
  ierr = VecDot(temp,appctx->SEMop.mass,&nrm);CHKERRQ(ierr);
  ierr = VecDestroy(&temp);CHKERRQ(ierr);
  nrm  = PetscSqrtReal(nrm);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Error for initial conditions %g\n",(double)nrm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*TEST

   build:
     requires: !complex

   test:
     requires: !single
     args: -tao_monitor  -ts_adapt_dt_max 3.e-3 -E 10 -N 8 -ncoeff 5 

   test:
     suffix: cn
     requires: !single
     args: -tao_monitor -ts_type cn -ts_dt .003 -pc_type lu -E 10 -N 8 -ncoeff 5 

TEST*/

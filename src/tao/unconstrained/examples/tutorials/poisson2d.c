
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

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/

typedef struct {
  PetscInt    N;             /* grid points per elements*/
  PetscInt    Ex;              /* number of elements */
  PetscInt    Ey;              /* number of elements */
  PetscReal   tol_L2,tol_max; /* error norms */
  PetscReal   mu;             /* viscosity */
  PetscReal   Lx;              /* total length of domain */ 
  PetscReal   Ly;              /* total length of domain */     
  PetscReal   Lex; 
  PetscReal   Ley; 
  PetscInt    lenx;
  PetscInt    leny;
  PetscReal   Tadj;
} PetscParam;

typedef struct {
  PetscScalar u,v;   /* wind speed */
} Field;


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
extern PetscErrorCode PDEFunction(Vec,Vec,void*);
extern PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode MyMatMult(Mat,Vec,Vec);

int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  Tao            tao;
  Vec            u;                      /* approximate solution vector */
  PetscErrorCode ierr;
  PetscInt       xs, xm, ys,ym, ix,iy;
  PetscInt       indx,indy,m, nn;
  PetscReal      x,y;
  Field          **bmass;
  DMDACoor2d     **coors;
  Vec            global,loc;
  DM             cda;
  PetscInt       jx,jy;
  PetscViewer    viewfile;
  Mat            H_shell;
  
 
   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBegin;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /*initialize parameters */
  appctx.param.N    = 8;  /* order of the spectral element */
  appctx.param.Ex    = 6;  /* number of elements */
  appctx.param.Ey    = 6;  /* number of elements */
  appctx.param.Lx    = 4.0;  /* length of the domain */
  appctx.param.Ly    = 4.0;  /* length of the domain */
  appctx.param.mu   = 0.005; /* diffusion coefficient */
  appctx.ncoeff      = 2;

  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&appctx.param.N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Ex",&appctx.param.Ex,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Ey",&appctx.param.Ey,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ncoeff",&appctx.ncoeff,NULL);CHKERRQ(ierr);
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

  VecGetLocalSize(u,&m);
  VecGetSize(u,&nn);
  
  MatCreateShell(PETSC_COMM_WORLD,m,m,nn,nn,&appctx,&H_shell);
  MatShellSetOperation(H_shell,MATOP_MULT,(void(*)(void))MyMatMult);
  //MatShellSetOperation(H_shell,MATOP_MULT_TRANSPOSE,(void(*)(void))MyMatMultTransp);
  
  //ierr = TSSetRHSJacobian(appctx.ts,H_shell,H_shell,RHSJacobian,&appctx);CHKERRQ(ierr);
  //ierr = TSSetRHSFunction(appctx.ts,NULL,RHSFunction,&appctx);CHKERRQ(ierr);

  ierr = InitialConditions(appctx.dat.ic,&appctx);CHKERRQ(ierr);
  ierr = ComputeObjective(2.0,appctx.dat.obj,&appctx);CHKERRQ(ierr);

  ierr = ComputeObjective(4.0,appctx.dat.true_solution,&appctx);CHKERRQ(ierr);
  ierr = TrueSolution(appctx.dat.true_solution,&appctx);CHKERRQ(ierr);
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
 
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.ic);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.true_solution);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.obj);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.SEMop.mass);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.curr_sol);CHKERRQ(ierr);
  ierr = PetscGLLDestroy(&appctx.SEMop.gll);CHKERRQ(ierr);
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
  Field             **s;
  PetscErrorCode    ierr;
  PetscInt          i,j;
  DM                cda;
  Vec               global;
  DMDACoor2d        **coors;

  ierr = DMDAVecGetArray(appctx->da,u,&s);CHKERRQ(ierr);
      
  DMGetCoordinateDM(appctx->da,&cda);
  DMGetCoordinates(appctx->da,&global);
  DMDAVecGetArray(cda,global,&coors);

  for (i=0; i<appctx->param.lenx; i++) 
    {for (j=0; j<appctx->param.leny; j++) 
      {
         
      s[j][i].u=appctx->param.mu*(PetscCosScalar(0.5*PETSC_PI*coors[j][i].x)+PetscSinScalar(0.5*PETSC_PI*coors[j][i].y))/10.0;
      s[j][i].v=appctx->param.mu*(PetscSinScalar(0.5*PETSC_PI*coors[j][i].x)+PetscCosScalar(0.5*PETSC_PI*coors[j][i].y))/10.0;
      
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
   Field             **s;  
  PetscErrorCode    ierr;
  PetscInt          i,j;
  DM                cda;
  Vec               global;
  DMDACoor2d        **coors;

  ierr = DMDAVecGetArray(appctx->da,u,&s);CHKERRQ(ierr);
      
  DMGetCoordinateDM(appctx->da,&cda);
  DMGetCoordinates(appctx->da,&global);
  DMDAVecGetArray(cda,global,&coors);

  for (i=0; i<appctx->param.lenx; i++) 
    {for (j=0; j<appctx->param.leny; j++) 
      {
      s[j][i].u=(PetscCosScalar(0.5*PETSC_PI*coors[j][i].x)+PetscSinScalar(0.5*PETSC_PI*coors[j][i].y))/10.0;
      s[j][i].v=(PetscSinScalar(0.5*PETSC_PI*coors[j][i].x)+PetscCosScalar(0.5*PETSC_PI*coors[j][i].y))/10.0;
      
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
  Field             **s; 
  PetscErrorCode    ierr;
  PetscInt          i,j;
  DM                cda;
  Vec               global;
  DMDACoor2d        **coors;


  ierr = DMDAVecGetArray(appctx->da,obj,&s);CHKERRQ(ierr);
      
  DMGetCoordinateDM(appctx->da,&cda);
  DMGetCoordinates(appctx->da,&global);
  DMDAVecGetArray(cda,global,&coors);

  for (i=0; i<appctx->param.lenx; i++) 
    {for (j=0; j<appctx->param.leny; j++) 
      {
     
      s[j][i].u=-appctx->param.mu*(PetscCosScalar(0.5*PETSC_PI*coors[j][i].x)+PetscSinScalar(0.5*PETSC_PI*coors[j][i].y))/10.0;
      s[j][i].v=-appctx->param.mu*(PetscSinScalar(0.5*PETSC_PI*coors[j][i].x)+PetscCosScalar(0.5*PETSC_PI*coors[j][i].y))/10.0;
     
      } 
     }
  
  ierr = DMDAVecRestoreArray(appctx->da,obj,&s);CHKERRQ(ierr);


  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PDEFunction"
PetscErrorCode PDEFunction(Vec globalin,Vec globalout,void *ctx)
{
  PetscErrorCode  ierr;
  AppCtx          *appctx = (AppCtx*)ctx;  
  PetscScalar     **wrk3, **wrk1, **wrk2, **wrk4;
  PetscScalar     **stiff, **mass, **grad;
  PetscScalar     **ulb, **vlb;
  const Field     **ul;
  Field           **ff;
  Field           **outl; 
  PetscInt        i,ix,iy,jx,jy, indx, indy;
  PetscInt        xs,xm,ys,ym, Nl, Nl2; 
  DM              cda;
  Vec             uloc, outloc, global, forcing;
  DMDACoor2d      **coors;
  PetscScalar     alpha, beta; 
  PetscInt        inc;  
   
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
 
  ierr = DMDAGetCorners(appctx->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  Nl    = appctx->param.N; 

  //DMCreateGlobalVector(appctx->da,&gradgl);
    
  xs=xs/(Nl-1);
  xm=xm/(Nl-1);
  ys=ys/(Nl-1);
  ym=ym/(Nl-1); 

  inc=1;
  /*
     Initialize work arrays
  */ 

  ierr = PetscMalloc1(appctx->param.N,&ulb);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&ulb[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) ulb[i] = ulb[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&vlb);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&vlb[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) vlb[i] = vlb[i-1]+Nl;

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

  alpha = 1.0;
  beta  = 0.0;
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
        BLASaxpy_(&Nl2,&alpha, &wrk3[0][0],&inc,&wrk2[0][0],&inc); //I freed wrk3 and saved the laplacian in wrk2
       
        // for the v component now 
        //first product (B x K_yy)v=W3
        alpha=appctx->param.Lex/2.0;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&mass[0][0],&Nl,&vlb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=2.0/appctx->param.Ley;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&stiff[0][0],&Nl,&beta,&wrk3[0][0],&Nl);

        //second product (K_xx x B)v=W4
        alpha=2.0/appctx->param.Lex;
        BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&stiff[0][0],&Nl,&vlb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
        alpha=appctx->param.Ley/2.0;
        BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&mass[0][0],&Nl,&beta,&wrk4[0][0],&Nl);

        alpha=1.0;
        BLASaxpy_(&Nl2,&alpha, &wrk4[0][0],&inc,&wrk3[0][0],&inc); //I freed wrk4 and saved the laplacian in wrk3


        for (jx=0; jx<appctx->param.N; jx++) 
        {for (jy=0; jy<appctx->param.N; jy++)   
           {indx=ix*(appctx->param.N-1)+jx;
            indy=iy*(appctx->param.N-1)+jy;
            
            outl[indy][indx].u +=appctx->param.mu*(wrk2[jy][jx]);
            outl[indy][indx].v +=appctx->param.mu*(wrk3[jy][jx]);    
           }}
        }
     }

    for (ix=0; ix<appctx->param.lenx; ix++) 
    {for (jx=0; jx<appctx->param.leny; jx++) 
      {
      ff[jx][ix].u=(appctx->param.mu*(-1.0 + 4*PETSC_PI*PETSC_PI)*PetscCosScalar(2.*PETSC_PI*coors[jx][ix].x)
                   +2.*PETSC_PI*ul[jx][ix].v*PetscCosScalar(2.*PETSC_PI*coors[jx][ix].y)-2.*PETSC_PI*ul[jx][ix].u*PetscSinScalar(2.*PETSC_PI*coors[jx][ix].x)-
                     appctx->param.mu*PetscSinScalar(2.*PETSC_PI*coors[jx][ix].y));
      ff[jx][ix].v=((appctx->param.mu*(-1.0 + 4*PETSC_PI*PETSC_PI)+2.*PETSC_PI*ul[jx][ix].v)*
                    PetscCosScalar(2.*PETSC_PI*coors[jx][ix].y)-(appctx->param.mu +2.*PETSC_PI*ul[jx][ix].u)*PetscSinScalar(2.*PETSC_PI*coors[jx][ix].x));
     
      } 
     }
  ierr = DMDAVecRestoreArray(appctx->da,forcing,&ff);CHKERRQ(ierr);
 
  
  ierr = DMDAVecRestoreArrayRead(appctx->da,globalin,&uloc);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(appctx->da,outloc,&outl);CHKERRQ(ierr);
  VecSet(globalout,0.0);
  DMLocalToGlobalBegin(appctx->da,outloc,ADD_VALUES,globalout);
  DMLocalToGlobalEnd(appctx->da,outloc,ADD_VALUES,globalout);

  VecScale(globalout, -1.0);
  VecAXPY(globalout,1.0,forcing);

  ierr = VecPointwiseDivide(globalout,globalout,appctx->SEMop.mass);CHKERRQ(ierr);

  

  DMGetCoordinateDM(appctx->da,&cda);
  DMGetCoordinates(appctx->da,&global);
  DMDAVecGetArray(cda,global,&coors);
  VecDuplicate(globalout,&forcing); 

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

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MyMatMult"

PetscErrorCode MyMatMult(Mat H, Vec in, Vec out)
 {
   AppCtx         *appctx;
   
   const Field     **ul;
   Field           **outl;
   PetscScalar     **stiff, **mass, **grad;
   PetscScalar     **wrk1, **wrk2, **wrk3, **wrk4; 
   PetscScalar     **ulb, **vlb;
   PetscInt        Nl, Nl2, inc;
   PetscInt        xs,ys,xm,ym,ix,iy,jx,jy, indx,indy, i;
   PetscErrorCode  ierr;
   Vec             uloc, outloc;
   PetscScalar     alpha, beta;
   
 
  MatShellGetContext(H,&appctx);

  ierr = PetscGLLElementLaplacianCreate(&appctx->SEMop.gll,&stiff);CHKERRQ(ierr);
  ierr = PetscGLLElementMassCreate(&appctx->SEMop.gll,&mass);CHKERRQ(ierr); 
  ierr = PetscGLLElementAdvectionCreate(&appctx->SEMop.gll,&grad);CHKERRQ(ierr);

  /* unwrap local vector for the input solution */
  DMCreateLocalVector(appctx->da,&uloc);

  DMGlobalToLocalBegin(appctx->da,in,INSERT_VALUES,uloc);
  DMGlobalToLocalEnd(appctx->da,in,INSERT_VALUES,uloc);

  DMDAVecGetArrayRead(appctx->da,uloc,&ul);CHKERRQ(ierr);

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
            vlb[jy][jx]=0.0;
           
            indx=ix*(appctx->param.N-1)+jx;
            indy=iy*(appctx->param.N-1)+jy;
            ulb[jy][jx]=ul[indy][indx].u; 
            vlb[jy][jx]=ul[indy][indx].v; 
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


        for (jx=0; jx<appctx->param.N; jx++) 
        {for (jy=0; jy<appctx->param.N; jy++)   
           {indx=ix*(appctx->param.N-1)+jx;
            indy=iy*(appctx->param.N-1)+jy;
            
            outl[indy][indx].u += appctx->param.mu*(wrk2[jy][jx]);
            outl[indy][indx].v += appctx->param.mu*(wrk3[jy][jx]);

            //printf("outl[%d][%d]=%0.15f\n", indx,indy, outl[indy][indx]);
           }}
       }
     }
  
  ierr = DMDAVecRestoreArray(appctx->da,outloc,&outl);CHKERRQ(ierr);
  DMDAVecRestoreArrayRead(appctx->da,in,&uloc);CHKERRQ(ierr);
 
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
*/
PetscErrorCode FormFunctionGradient(Tao tao,Vec X,PetscReal *f,Vec G,void *ctx)
{
  AppCtx           *appctx = (AppCtx*)ctx;     /* user-defined application context */
  PetscErrorCode    ierr;
  Vec               temp, djdu, rhs_adj, grad, psi;
  PetscInt          its;
  PetscReal         ff, gnorm, cnorm, xdiff,errex; 
  TaoConvergedReason reason;    
  PetscViewer        viewfile;
  KSP               ksp; 
  //static int counter=0; it was considered for storing line search error
  char filename[24] ;
  char data[80] ;
  
  
  ierr = VecCopy(X,appctx->dat.curr_sol);CHKERRQ(ierr);

  /*
  Store current solution for comparison
  */
  ierr = VecDuplicate(appctx->dat.curr_sol,&temp);CHKERRQ(ierr);
  ierr = VecDuplicate(appctx->dat.curr_sol,&djdu);CHKERRQ(ierr);

  ierr = VecWAXPY(djdu,-1.0,X,appctx->dat.obj);CHKERRQ(ierr);

  /*
     Compute the L2-norm of the objective function, cost function is f
  */
  ierr = VecPointwiseMult(temp,djdu,djdu);CHKERRQ(ierr);
  ierr = VecDot(temp,appctx->SEMop.mass,f);CHKERRQ(ierr);
  ierr = VecDestroy(&temp);CHKERRQ(ierr);
/*
  //local error evaluation   
  ierr = VecDuplicate(G,&temp);CHKERRQ(ierr);
  ierr = VecDuplicate(appctx->dat.ic,&temp);CHKERRQ(ierr);
  ierr = VecWAXPY(temp,-1.0,appctx->dat.ic,appctx->dat.true_solution);CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp,temp,temp);CHKERRQ(ierr);
  //for error evaluation
  ierr = VecDot(temp,appctx->SEMop.mass,&errex);CHKERRQ(ierr);
  ierr = VecDestroy(&temp);CHKERRQ(ierr);
  errex  = PetscSqrtReal(errex); 
*/

/*
     Compute dJ/du which gives the right hand side and store it in rhs_adj
  */

  ierr = VecScale(djdu, -2.0);CHKERRQ(ierr);
  ierr = VecPointwiseMult(djdu,djdu,appctx->SEMop.mass);CHKERRQ(ierr);
  ierr = VecDuplicate(djdu,&rhs_adj);CHKERRQ(ierr);
  ierr = VecCopy(djdu,rhs_adj);CHKERRQ(ierr);

  ierr = VecDuplicate(djdu,&grad);CHKERRQ(ierr);
  ierr = VecSet(grad,1.0);CHKERRQ(ierr);
  KSPCreate(PETSC_COMM_WORLD,&ksp);
  KSPSetOperators(ksp,H,H);

  ierr= PetscErrorCode KSPSolve(ksp,rhs_adj,grad);
  
  PDEFunction(H,
  
  VecCopy(grad,G);


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

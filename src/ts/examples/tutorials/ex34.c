static const char help[]="Solve the Van Der Pol equation using a FLAVOR approach";
/*
As a test of a multiscale integration technique, consider the 'hidden' Van der Pol oscillator described in

 Tao, Owhadi, and Marsden, "Nonintrusive and Structure Preserving Multiscale Integration of Stiff ODEs, SDEs, and Hamiltonian Systems with Hidden Slow Dynamics via Flow Averaging", 2010

as example 6.1. (Note that the change of variables reverses the usual x/y convention, and that the conversion to a first order system differs from that in ex16)

Here we prototype a FLAVOR integrator to reproduce this example. We are able to use very naive integrators and still
get the same results of a more involved one 

The example in the paper uses forward Euler with h = 0.05\epsilon as a benchmark integrator, while here we use TSARKIMEX

Todo is to use simple adaptive methods inside a TSMULTI to try and beat TSARKIMEX

Accepts special arguments
  -epsilon
  -monitor

*/
#include <petscts.h>

/* User-defined Routines */
static PetscErrorCode FormRHSFunctionXYImex(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormIFunctionXYImex(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode FormIJacobianXYImex(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);
/* static PetscErrorCode FormRHSFunctionRTHFast(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormRHSFunctionRTHSlow(TS,PetscReal,Vec,Vec,void*); 
static PetscErrorCode FormRHSFunctionXYFast(TS,PetscReal,Vec,Vec,void*); */
static PetscErrorCode FormRHSFunctionXYSlow(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormRHSFunctionXYTotal(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FLAVOR_FE_Step(PetscReal,PetscReal,Vec,PetscBool,TSRHSFunction,TSRHSFunction,void*);
static PetscErrorCode FLAVOR_FE_loop(Vec,PetscReal,TSRHSFunction,TSRHSFunction,PetscInt*,PetscReal*,PetscBool useMonitor,PetscErrorCode(*)(TS,PetscInt,PetscReal,Vec,void*),void*);
static PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void*);
static PetscErrorCode FormInitialSolutionRTH(TS,Vec,void*);
static PetscErrorCode FormInitialSolutionXY(TS,Vec,void*);


/* User context */
typedef struct _User *User;
struct _User
{
  PetscReal epsilon;
  PetscViewer viewer;
};

/* 
   Main function 
*/
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscErrorCode  ierr;
  Vec             X;
  Mat             A;
  PetscReal       ftime, epsilon = 0.001, T = 5000;
  PetscBool       useMonitor = PETSC_FALSE;
  PetscInt        steps,  maxSteps = 10000000;
  PetscMPIInt     size;
  struct _User    user;
  TS              ts, ts_dummy;
  DM              dm;

  ierr = PetscInitialize(&argc, &argv, (char*) 0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only");

  ierr = PetscOptionsGetReal(NULL,"-epsilon",&epsilon,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-ts_final_time",&T,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,"-monitor",&useMonitor,NULL);CHKERRQ(ierr);
  /* Other options only affect the true TS, not the hand-coded loops */

  user.epsilon = epsilon;
  /* ==== TSMULTI === */
  if(useMonitor){
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"ex34_output_MULTI.txt",&user.viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(user.viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr); 
  }
  ierr = VecCreateSeq(PETSC_COMM_WORLD,2,&X);CHKERRQ(ierr); 
  ierr = FormInitialSolutionXY(ts,X,&user);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);

  ierr = TSSetType(ts,TSMULTI);CHKERRQ(ierr); 
  ierr = TSMultiSetType(ts,TSMULTIFLAVORFE);CHKERRQ(ierr); 
  ierr = TSMultiSetEpsilon(ts,user.epsilon);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSSetRHSPartitionFunction(dm,TS_MULTI_PARTITION,TS_MULTI_SLOW_SLOT,FormRHSFunctionXYSlow,&user);CHKERRQ(ierr);
  ierr = DMTSSetRHSPartitionFunction(dm,TS_MULTI_PARTITION,TS_MULTI_FULL_SLOT,FormRHSFunctionXYTotal,&user);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,maxSteps,T);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.2); /* this is the 'macro' step, currently 20 'meso' steps */
  if(useMonitor){
    ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);
  }
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"epsilon %G, steps %D, ftime %G\n",user.epsilon,steps,ftime);CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  if(useMonitor){
    ierr = PetscViewerDestroy(&user.viewer);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  /*  ===== An IMEX solver to compute a 'reference' solution ==== */
  if(useMonitor){
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"ex34_output_imex.txt",&user.viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(user.viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr); 
  }
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_WORLD,2,&X);CHKERRQ(ierr); 
  ierr = FormInitialSolutionXY(ts,X,&user);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,FormIFunctionXYImex,&user);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,A,A,FormIJacobianXYImex,&user);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunctionXYImex,&user);CHKERRQ(ierr);  
  ierr = TSSetDuration(ts,maxSteps,T);CHKERRQ(ierr);

  ierr = TSSetMaxSNESFailures(ts,-1); /* unlimited failures */
  if(useMonitor){
    ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);
  }
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"epsilon %G, steps %D, ftime %G\n",user.epsilon,steps,ftime);CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  if(useMonitor){
    ierr = PetscViewerDestroy(&user.viewer);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  /* ===== A hand-coded simple FLAVOR loop in the 'hidden' r,\theta formulation =====  */
  /*
  if(useMonitor){
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"ex34_output_flavor_rth.txt",&user.viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(user.viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr); 
  }
  ierr = VecCreateSeq(PETSC_COMM_WORLD,2,&X);CHKERRQ(ierr);
  ierr = FormInitialSolutionRTH(ts_dummy,X,&user);CHKERRQ(ierr);
  ierr = FLAVOR_FE_loop(X,T,FormRHSFunctionRTHFast,FormRHSFunctionRTHSlow,&steps,&ftime,useMonitor,Monitor,&user);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"epsilon %G, steps %D, ftime %G\n",user.epsilon,steps,ftime);CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  if(useMonitor){
    ierr = PetscViewerDestroy(&user.viewer);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&X);CHKERRQ(ierr);
*/

  /* ===== A hand-coded simple FLAVOR loop with the XY system to test =====  */
  if(useMonitor){
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"ex34_output_flavor_xy.txt",&user.viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(user.viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr); 
  }
  ierr = VecCreateSeq(PETSC_COMM_WORLD,2,&X);CHKERRQ(ierr);
  ierr = FormInitialSolutionXY(ts_dummy,X,&user);CHKERRQ(ierr);
  ierr = FLAVOR_FE_loop(X,T,FormRHSFunctionXYTotal,FormRHSFunctionXYSlow,&steps,&ftime,useMonitor,Monitor,&user);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"epsilon %G, steps %D, ftime %G\n",user.epsilon,steps,ftime);CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  if(useMonitor){
    ierr = PetscViewerDestroy(&user.viewer);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&X);CHKERRQ(ierr);

  PetscFinalize();
  return EXIT_SUCCESS;
}

/* 
RHS function for use with an imex integrator (in terms of xy)
*/
#undef __FUNCT__
#define __FUNCT__ "FormRHSFunctionXYImex"
static PetscErrorCode FormRHSFunctionXYImex(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  User           user = (User)ctx;
  PetscScalar    *x,*f;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = -user->epsilon * x[1];
  f[1] = 0;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
#undef __FUNCT__
#define __FUNCT__ "FormRHSFunctionXYFast"
static PetscErrorCode FormRHSFunctionXYFast(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  User           user = (User)ctx;
  PetscScalar    *x,*f;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = 0;
  f[1] =  (1./user->epsilon)*(x[0]+x[1]-(x[1]*x[1]*x[1]/3.));
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
*/

#undef __FUNCT__
#define __FUNCT__ "FormRHSFunctionXYSlow"
static PetscErrorCode FormRHSFunctionXYSlow(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  User           user = (User)ctx;
  PetscScalar    *x,*f;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = -user->epsilon * x[1];
  f[1] = 0;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "FormRHSFunctionXYTotal"
static PetscErrorCode FormRHSFunctionXYTotal(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  User           user = (User)ctx;
  PetscScalar    *x,*f;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = -user->epsilon * x[1];
  f[1] = (1./user->epsilon)*(x[0]+x[1]-(x[1]*x[1]*x[1]/3.));
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
IFunction for use with an imex integrator (in terms of xy)
*/
#undef __FUNCT__
#define __FUNCT__ "FormIFunctionXYImex"
static PetscErrorCode FormIFunctionXYImex(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  User           user = (User)ctx;
  PetscScalar    *x,*xdot,*f;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = xdot[0];
  f[1] = xdot[1] - (1./user->epsilon)*(x[0]+x[1]-(x[1]*x[1]*x[1]/3.));
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormIJacobianXYImex"
static PetscErrorCode FormIJacobianXYImex(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat *A,Mat *B,MatStructure *flag,void *ctx)
{
  PetscErrorCode ierr;
  User           user     = (User)ctx;
  PetscReal      mu       = 1./user->epsilon;
  PetscInt       rowcol[] = {0,1};
  PetscScalar    *x,J[2][2];

  PetscFunctionBeginUser;
  ierr    = VecGetArray(X,&x);CHKERRQ(ierr);
  J[0][0] = a;                      J[0][1] = 0;
  J[1][0] = -mu;                    J[1][1] = a - mu*(1 - x[1]*x[1]);

  ierr    = MatSetValues(*B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr    = VecRestoreArray(X,&x);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*A != *B) {
    ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}
/* 
Fast RHS Function (in terms of r \theta)
*/
#undef __FUNCT__
#define __FUNCT__ "FormRHSFunctionRTHFast"
/*
static PetscErrorCode FormRHSFunctionRTHFast(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{
  User             user = (User) ctx; 
  PetscErrorCode   ierr;
  PetscScalar      *x,*f, c, s, r, th;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  r = x[0]; th = x[1]; c = PetscCosScalar(th); s = PetscSinScalar(th);
  f[0] =   (1./user->epsilon) * r * c * (c + s - (r*r*c*c*c/3.)) ; 
  f[1] = - (1./user->epsilon) *     s * (c + s - (r*r*c*c*c/3.)); 
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
*/

/* 
Slow RHS Function (in terms of r \theta)
*/
/*
#undef __FUNCT__
#define __FUNCT__ "FormRHSFunctionRTHSlow"
static PetscErrorCode FormRHSFunctionRTHSlow(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{
  User             user = (User) ctx; 
  PetscErrorCode   ierr;
  PetscScalar      *x,*f, c, s, r, th;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  r = x[0]; th = x[1]; c = PetscCosScalar(th); s = PetscSinScalar(th);
  f[0] = - user->epsilon * r * c * s; 
  f[1] = - user->epsilon * c * c; 
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
*/

#undef __FUNCT__
#define __FUNCT__ "FLAVOR_FE_loop"
static PetscErrorCode FLAVOR_FE_loop(Vec X, PetscReal T, TSRHSFunction fast, TSRHSFunction slow, PetscInt *steps, PetscReal *ftime, PetscBool useMonitor, PetscErrorCode (*monitor)(TS,PetscInt,PetscReal,Vec,void*), void *ctx)
{
  PetscErrorCode ierr;
  User user = (User) ctx;
  PetscInt i;
  PetscReal t,tau,deltaMinusTau;
  TS ts_dummy;

  PetscFunctionBeginUser;
  tau = 0.05*user->epsilon;
  deltaMinusTau = 0.01-tau;

  i = 0; t = 0;
  while(t<T){
      if(useMonitor){
        ierr = monitor(ts_dummy,i,t,X,user);CHKERRQ(ierr);
      }
      
      /* Stiff step */
      ierr = FLAVOR_FE_Step(t,tau,X,PETSC_TRUE,fast,slow,user);CHKERRQ(ierr);
      t += tau;

      /* Coarse Step */
      ierr = FLAVOR_FE_Step(t,deltaMinusTau,X,PETSC_FALSE,NULL,slow,user);CHKERRQ(ierr);
      t += deltaMinusTau;
      ++i;
  }
  *ftime = t;
  *steps = i;
  PetscFunctionReturn(0);
}

/* 
Explicit Euler step with stiff terms on or off
*/
#undef __FUNCT__
#define __FUNCT__ "FLAVOR_FE_Step"
static PetscErrorCode FLAVOR_FE_Step(PetscReal t, PetscReal h, Vec X, PetscBool stiffOn, TSRHSFunction fastFunc, TSRHSFunction slowFunc, void* ctx)
{
  PetscErrorCode   ierr;
  Vec              F,F2;
  TS               ts_dummy;

  PetscFunctionBeginUser;
  ierr = VecCreateSeq(PETSC_COMM_WORLD,2,&F);CHKERRQ(ierr);
  ierr = slowFunc(ts_dummy,t,X,F,ctx);CHKERRQ(ierr);
  if(stiffOn){
    ierr = VecCreateSeq(PETSC_COMM_WORLD,2,&F2);CHKERRQ(ierr);
    ierr = fastFunc(ts_dummy,t,X,F2,ctx);CHKERRQ(ierr);
    ierr = VecAXPY(F,1,F2);CHKERRQ(ierr);
    ierr = VecDestroy(&F2);CHKERRQ(ierr);
  }
  ierr = VecAXPY(X,h,F);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

/*
Initial Conditions (xy)
*/
#undef __FUNCT__
#define __FUNCT__ "FormInitialSolutionXY"
static PetscErrorCode FormInitialSolutionXY(TS ts,Vec X,void *ctx)
{ 
  PetscErrorCode ierr;
  PetscScalar *x,r,th;

  PetscFunctionBeginUser;
  ierr = FormInitialSolutionRTH(ts,X,ctx);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  r = x[0]; th = x[1];
  x[0] = r * PetscSinScalar(th);
  x[1] = r * PetscCosScalar(th);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
Initial Conditions (r \theta)
*/
#undef __FUNCT__
#define __FUNCT__ "FormInitialSolutionRTH"
static PetscErrorCode FormInitialSolutionRTH(TS ts,Vec X,void *ctx)
{ 
  PetscErrorCode  ierr;
  PetscScalar     *x ;
  const PetscReal PI = 3.14159265358979323846;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  x[0] = PetscSqrtScalar(2);
  x[1] = PI/4;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
Monitor
*/
#undef __FUNCT__
#define __FUNCT__ "Monitor"
static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal time,Vec u,void *ctx)
{
  User            user = (User) ctx;
  PetscErrorCode  ierr;
  PetscScalar     *uarr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(u,&uarr);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(user->viewer,"%15.15g %15.15g %15.15g\n",
                                time, uarr[0],uarr[1]);CHKERRQ(ierr);
  ierr = VecRestoreArray(u,&uarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

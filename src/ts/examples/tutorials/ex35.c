static const char help[]="Solve a toy problem with an HMM";
/*

Solve a toy problem using TSMULTI and the FHMMFE algorithm from 
Ariel, Engquist, Kim, Lee, and Tsai. "A Multiscale Method for Highly Oscillatory Dynamical Systems
Using a Poincar\'e Map Type Technique," 2013


Additional RHSFunctions and IFunctions are provided to allow experimentation with other solvers

Example 3.2 in AEKLT2013

dx/dt = \epsilon^{-1} A x + f(x) ,   x_0 = [1 0 1 0]'

      0  2  0  0
A =  -2  0  0  0
      0  0  0  1
      0  0 -1  0

 f(x) = [0 0.5*x[2]^2 0 2*x[0]*x[2] ]', where x = [ x[0] x[1] x[2] x[3] ]'       

 A set of slow variables, which are not explicitly used here but which can be used for analysis:
 x[0]^2 + x[1]^2
 x[2]^2 + x[3]^2
 x[0] x[2]^2 + 2 x[1]x[2]x[3] - x[0]x[3]^2

Accepts extra options
  -epsilon
  -monitor
*/

#include <petscts.h>

/* User-defined Routines */
static PetscErrorCode FormRHSFunctionSlow(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormRHSFunctionFull(TS,PetscReal,Vec,Vec,void*); 
static PetscErrorCode FormRHSFunctionFast(TS,PetscReal,Vec,Vec,void*); 
static PetscErrorCode FormInitialSolution(TS,Vec,void*);
static PetscErrorCode FormIFunction(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode FormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);
static PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void*);

/* User context */
typedef struct _User *User;
struct _User
{
  PetscReal    epsilon;
  PetscViewer  viewer;
};

/* 
   Main function 
*/
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscErrorCode  ierr;
  struct _User    user;
  PetscReal       epsilon = 0.0001, ftime, T = 7, dt = 0.1;

  PetscInt        steps;
  Vec             X;
  PetscMPIInt     size;
  TS              ts;
  DM              dm; 
  PetscBool       useMonitor = PETSC_FALSE;

  ierr = PetscInitialize(&argc, &argv, (char*) 0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only");
  
  ierr = PetscOptionsGetReal(NULL,"-epsilon",&epsilon,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,"-monitor",&useMonitor,NULL);CHKERRQ(ierr);
  user.epsilon = epsilon;

  if(useMonitor){
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"ex35_output.txt",&user.viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(user.viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr); 
  }

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSMULTI);CHKERRQ(ierr);
  ierr = TSMultiSetType(ts,TSMULTIFHMMFE);CHKERRQ(ierr);
  ierr = TSMultiSetWindow(ts,15.0*epsilon);CHKERRQ(ierr);

  /* Set a slow/fast/full partition to use with TSMULTI */
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSSetRHSPartitionFunction(dm,TS_MULTI_PARTITION,TS_MULTI_SLOW_SLOT,FormRHSFunctionSlow,&user);CHKERRQ(ierr);
  ierr = DMTSSetRHSPartitionFunction(dm,TS_MULTI_PARTITION,TS_MULTI_FULL_SLOT,FormRHSFunctionFull,&user);CHKERRQ(ierr);
  ierr = DMTSSetRHSPartitionFunction(dm,TS_MULTI_PARTITION,TS_MULTI_FAST_SLOT,FormRHSFunctionFast,&user);CHKERRQ(ierr);

  /* Set some extra RHS and I Functions to allow testing with other integrators */
  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunctionFull,&user);CHKERRQ(ierr); 
  ierr = TSSetIFunction(ts,NULL,FormIFunction,&user);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,NULL,NULL,FormIJacobian,&user);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_WORLD,4,&X);CHKERRQ(ierr);
  ierr = FormInitialSolution(ts,X,&user);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,PETSC_MAX_INT,T);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
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
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  if(useMonitor){
    ierr = PetscViewerDestroy(&user.viewer);CHKERRQ(ierr);
  }
  PetscFinalize();
  return EXIT_SUCCESS;
}

/*
RHS Function
*/
#undef __FUNCT__
#define __FUNCT__ "FormRHSFunctionFull"
static PetscErrorCode FormRHSFunctionFull(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{
  User             user = (User) ctx; 
  PetscErrorCode   ierr;
  PetscScalar      *x,*f,mu;

  PetscFunctionBeginUser;
  mu = 1./user->epsilon;
  ierr = VecGetArray(F,&f);CHKERRQ(ierr); 
  ierr = VecGetArray(X,&x);CHKERRQ(ierr); 
  f[0] =  2*mu*x[1];
  f[1] = -2*mu*x[0] + x[2]*x[2]/2;
  f[2] =    mu*x[3];
  f[3]=    -mu*x[2] + 2*x[0]*x[2];
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr); 
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "FormRHSFunctionFast"
static PetscErrorCode FormRHSFunctionFast(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{
  User             user = (User) ctx; 
  PetscErrorCode   ierr;
  PetscScalar      *x,*f, mu;

  PetscFunctionBeginUser;
  mu = 1./user->epsilon;
  ierr = VecGetArray(F,&f);CHKERRQ(ierr); 
  ierr = VecGetArray(X,&x);CHKERRQ(ierr); 
  f[0] =  2*mu*x[1];
  f[1] = -2*mu*x[0];
  f[2] =    mu*x[3];
  f[3]=    -mu*x[2];
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr); 
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "FormRHSFunctionSlow"
static PetscErrorCode FormRHSFunctionSlow(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{
  PetscErrorCode   ierr;
  PetscScalar      *x,*f;

  PetscFunctionBeginUser;
  ierr = VecGetArray(F,&f);CHKERRQ(ierr); 
  ierr = VecGetArray(X,&x);CHKERRQ(ierr); 
  f[0] = 0;
  f[1] = x[2]*x[2]/2;
  f[2] = 0;
  f[3]=  2*x[0]*x[2];
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr); 
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormIJacobian"
static PetscErrorCode FormIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot, PetscReal a,Mat *A,Mat *B,MatStructure *str,void *ctx)
{
    User              user = (User)ctx;
    PetscErrorCode    ierr;
    PetscScalar       v[16], *x, mu=1./user->epsilon;
    PetscInt          idxm[4] = {0,1,2,3};
    
    PetscFunctionBeginUser;
    ierr = VecGetArray(X,&x);CHKERRQ(ierr);
   
    v[0]  = a;     v[1]  = -2*mu; v[2]  = 0;   v[3]  = 0;
    v[4]  = 2*mu;  v[5]  = a;     v[6]  = 0;   v[7]  = 0;
    v[8]  = 0;     v[9]  = 0;     v[10] = a;   v[11] = -mu;
    v[12] = 0;     v[13] = 0;     v[14] = mu;  v[15] = a;   
    
    ierr = MatSetValues(*B,4,idxm,4,idxm,v,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (*A != *B) {
        ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    *str = SAME_NONZERO_PATTERN;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormIFunction"
static PetscErrorCode FormIFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
    User             user = (User) ctx;
    PetscErrorCode   ierr;
    PetscScalar      *x,*xdot,*f, mu=1./user->epsilon;
    
    PetscFunctionBeginUser;
    ierr = VecGetArray(F,&f);CHKERRQ(ierr);
    ierr = VecGetArray(X,&x);CHKERRQ(ierr);
    ierr = VecGetArray(Xdot,&xdot);CHKERRQ(ierr);
    f[0] = xdot[0] - 2*mu*x[1];
    f[1] = xdot[1] + 2*mu*x[0];
    f[2] = xdot[2] -   mu*x[3];
    f[3]=  xdot[3] +   mu*x[2];
    ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
    ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
    ierr = VecRestoreArray(Xdot,&xdot);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

/*
Initial Conditions
*/
#undef __FUNCT__
#define __FUNCT__ "FormInitialSolution"
static PetscErrorCode FormInitialSolution(TS ts,Vec X,void *ctx)
{ 
  PetscErrorCode ierr;
  PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  x[0] = 1; x[1] = 0; x[2] = 1; x[3] = 0; 
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
  ierr = PetscViewerASCIIPrintf(user->viewer,"%15.15g %15.15g %15.15g %15.15g %15.15g\n",
                                time, uarr[0],uarr[1],uarr[2],uarr[3]);CHKERRQ(ierr);
  ierr = VecRestoreArray(u,&uarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

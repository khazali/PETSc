static const char help[] = "Demonstrate the interface to solve DAE systems with partitioned RHS functions\n";
/*
   Consider the following second order ODE:

   q''(t) + q(t) = \epsilon f(q) 
  
   where \epsilon may or may not be small

   As a first order system, we  have

   q'(t) = p(t)
   p'(t) = -q(t) + \epsilon f(q) 

   (We choose f(q) = -q^3)

   Write X = [p; q] and partition the righthand side

   X' = [q; p]' =  [p(t); 0] + [0; -q(t)] + [0 ; \epsilon f(q)]

                   -- P ---   ----------- Q -------------------
                   ------- Fast --------   ------ Slow --------  

  Different potential integration schemes would split the same RHS differently,
  perhaps based on the value of \epsilon

  Accepts an extra option -epsilon
*/

#include <petscts.h>

/* User-defined Functions */
static PetscErrorCode FormRHSFunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormRHSFunctionP(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormRHSFunctionQ(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormRHSFunctionSlow(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormRHSFunctionFast(TS,PetscReal,Vec,Vec,void*);

static PetscErrorCode  FormJacobian(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
static PetscErrorCode  FormJacobianP(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
static PetscErrorCode  FormJacobianQ(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
static PetscErrorCode  FormJacobianSlow(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
static PetscErrorCode  FormJacobianFast(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);

static PetscErrorCode FormInitialSolution(TS,Vec,void*);

/* User context */
typedef struct
{
  PetscReal epsilon;
} User;

/* 
   Main function 
*/
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char* argv[])
{
  PetscErrorCode ierr;
  PetscMPIInt size;
  Vec X;
  Mat J;
  TS ts;
  DM dm;
  User user;
  PetscScalar epsilon = 1;
  PetscReal dt = 0.1, maxtime = 50, ftime;
  PetscInt maxsteps = 1000, steps;
  TSConvergedReason reason;

  ierr = PetscInitialize(&argc, &argv, (char*) 0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only");
 
  ierr = PetscOptionsGetScalar(NULL,"-epsilon",&epsilon,NULL);CHKERRQ(ierr);

  /* User Parameters */
  user.epsilon = epsilon;

  /* Create TS */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
    
  /* Create Vector and Matrix for Solution and Jacobian */
  ierr = VecCreateSeq(PETSC_COMM_WORLD,2,&X);CHKERRQ(ierr);  
  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,2,2,2,NULL,&J);CHKERRQ(ierr);

  /* Set Initial Conditions and time Parameters*/
  ierr = FormInitialSolution(ts,X,&user);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,maxsteps,maxtime);

  /* Set RHS Functions amd Jacobians*/
  ierr = DMTSSetRHSFunction(dm,FormRHSFunction,&user);CHKERRQ(ierr); 

  /* Register Partitioned RHS Functions and Jacobians */
  ierr = DMTSSetRHSPartitionFunction(dm,SYMPLECTIC,SYMPLECTIC_P,FormRHSFunctionP,&user);CHKERRQ(ierr);
  ierr = DMTSSetRHSPartitionFunction(dm,SYMPLECTIC,SYMPLECTIC_Q,FormRHSFunctionQ,&user);CHKERRQ(ierr);
    
  ierr = DMTSSetRHSPartitionFunction(dm,EXPONENTIAL,EXPONENTIAL_FAST,FormRHSFunctionFast,&user);CHKERRQ(ierr);
  ierr = DMTSSetRHSPartitionFunction(dm,EXPONENTIAL,EXPONENTIAL_SLOW,FormRHSFunctionSlow,&user);CHKERRQ(ierr);
  
  ierr = TSSetRHSJacobian(ts, J, J, FormJacobian, &user);CHKERRQ(ierr); 
  ierr = DMTSSetRHSPartitionJacobian(dm, SYMPLECTIC,SYMPLECTIC_Q, FormJacobianQ, &user);CHKERRQ(ierr);
  ierr = DMTSSetRHSPartitionJacobian(dm, SYMPLECTIC,SYMPLECTIC_P, FormJacobianP, &user);CHKERRQ(ierr);
  ierr = DMTSSetRHSPartitionJacobian(dm, EXPONENTIAL,EXPONENTIAL, FormJacobianFast, &user);CHKERRQ(ierr);
  ierr = DMTSSetRHSPartitionJacobian(dm, EXPONENTIAL,EXPONENTIAL, FormJacobianSlow, &user);CHKERRQ(ierr);
    
  /* Set TS Type  */
  ierr = TSSetType(ts,TSSYMPEULER);CHKERRQ(ierr);

  /* Set from options */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* Solve */
  ierr = TSSolve(ts,X); CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %G after %D steps\n",TSConvergedReasons[reason],ftime,steps);CHKERRQ(ierr);

  /* Clean Up */
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
    
  return EXIT_SUCCESS;
}
 
/* 
Initial Conditions 
*/
#undef __FUNCT__
#define __FUNCT__ "FormInitialSolution"
static PetscErrorCode FormInitialSolution(TS ts,Vec X,void *ctx)
{ 
  PetscScalar    *x;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  x[0] = 1; 
  x[1] = 1;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
RHS function (complete) 
*/
#undef __FUNCT__
#define __FUNCT__ "FormRHSFunction"
static PetscErrorCode FormRHSFunction(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{
  User             *user = (User*) ctx;
  PetscScalar      *x, *f;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = x[1];
  f[1] = -x[0] -(user->epsilon * x[0] * x[0] * x[0]);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
RHS function for 'P' terms
*/
#undef __FUNCT__
#define __FUNCT__ "FormRHSFunctionP"
static PetscErrorCode FormRHSFunctionP(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{
  User            *user = (User*) ctx;
  PetscScalar     *x, *f;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = 0;
  f[1] = -x[0] -(user->epsilon * x[0] * x[0] *x[0]);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
RHS function for 'Q' terms
*/
#undef __FUNCT__
#define __FUNCT__ "FormRHSFunctionQ"
static PetscErrorCode FormRHSFunctionQ(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{
  PetscScalar     *x, *f;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = x[1];
  f[1] = 0;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
RHS function for 'fast' terms
*/
#undef __FUNCT__
#define __FUNCT__ "FormRHSFunctionFast"
static PetscErrorCode FormRHSFunctionFast(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{
  PetscScalar     *x, *f;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] =  x[1];
  f[1] = -x[0];
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
RHS function for 'slow' terms
*/
#undef __FUNCT__
#define __FUNCT__ "FormRHSFunctionSlow"
static PetscErrorCode FormRHSFunctionSlow(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{
  User           *user = (User*) ctx;
  PetscScalar    *x, *f;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = 0;
  f[1] = -user->epsilon*x[0]*x[0]*x[0];
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
RHS Jacobian (full)
*/
#undef __FUNCT__
#define __FUNCT__ "FormJacobian"
PetscErrorCode FormJacobian(TS ts,PetscReal t,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  User              *user = (User*)ctx;
  PetscErrorCode    ierr;
  PetscScalar       v[4], *x;
  PetscInt          idxm[2] = {0,1};

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  v[0] = 0;                                   v[1] = 1;
  v[2] = -1 -(user->epsilon * x[0] * x[0]);   v[3] = 0;
  ierr = MatSetValues(*B,2,idxm,2,idxm,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*J != *B) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

/*
RHS Jacobian (fast)
*/
#undef __FUNCT__
#define __FUNCT__ "FormJacobianFast"
PetscErrorCode FormJacobianFast(TS ts,PetscReal t,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       v[4];
  PetscInt          idxm[2] = {0,1};

  PetscFunctionBeginUser;
  v[0] = 0;    v[1] = 1;
  v[2] = -1;   v[3] = 0;
  ierr = MatSetValues(*B,2,idxm,2,idxm,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*J != *B) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

/*
RHS Jacobian (slow)
*/
#undef __FUNCT__
#define __FUNCT__ "FormJacobianSlow"
PetscErrorCode FormJacobianSlow(TS ts,PetscReal t,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  User              *user = (User*)ctx;
  PetscErrorCode    ierr;
  PetscScalar       v[4], *x;
  PetscInt          idxm[2] = {0,1};

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  v[0] = 0;                                   v[1] = 0;
  v[2] =  -(user->epsilon * x[0] * x[0]);     v[3] = 0;
  ierr = MatSetValues(*B,2,idxm,2,idxm,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*J != *B) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

/*
RHS Jacobian (P)
*/
#undef __FUNCT__
#define __FUNCT__ "FormJacobianP"
PetscErrorCode FormJacobianP(TS ts,PetscReal t,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  User              *user = (User*)ctx;
  PetscErrorCode    ierr;
  PetscScalar       v[4], *x;
  PetscInt          idxm[2] = {0,1};

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  v[0] = 0;                                   v[1] = 0;
  v[2] = -1 -(user->epsilon * x[0] * x[0]);   v[3] = 0;
  ierr = MatSetValues(*B,2,idxm,2,idxm,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*J != *B) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

/*
RHS Jacobian (Q)
*/
#undef __FUNCT__
#define __FUNCT__ "FormJacobianQ"
PetscErrorCode FormJacobianQ(TS ts,PetscReal t,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       v[4], *x;
  PetscInt          idxm[2] = {0,1};

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  v[0] = 0;   v[1] = 1;
  v[2] = 0;   v[3] = 0;
  ierr = MatSetValues(*B,2,idxm,2,idxm,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*J != *B) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

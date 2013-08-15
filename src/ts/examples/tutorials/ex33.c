static const char help[] = "Demonstrate the use of an integrator requiring a partitioned RHS\n";
/*

  Test for a Hamiltonian system with a variable mass matrix

  This follows Hairer/Lubich/Wanner, "Geometric Numerical Integration", 2nd ed.,  ch XIII example 10.1

  Coordinates are 
  q0 [angle]
  q1 [r-1]
  p0 = (1+q_1)^2 \dot q_0
  p1 = \dot q-1

  Note that this problem actually admits an explicit algorithm with an even finer splitting of the RHS (detailed in the book).
  Here we use it as a test for the symplectic Euler method with a nontrivial explicit step. 

  Note that in the current implementation, the 'P' terms are treated implicitly and the 'Q' terms explicitly. Thus, 
  a Jacobian is only provided for one of the partitioned RHS functions.

  Accepts an extra option -epsilon
*/

#include <petscts.h>

/* User-defined Functions */
static PetscErrorCode FormRHSFunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormRHSFunctionP(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormRHSFunctionQ(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode  FormJacobian(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
static PetscErrorCode  FormJacobianP(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
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
  PetscErrorCode      ierr;
  PetscMPIInt         size;
  Vec                 X;
  Mat                 J;
  TS                  ts;
  DM                  dm;
  User                user;
  PetscScalar         epsilon = 0.1;
  PetscReal           dt = 0.15,maxtime = 10,ftime;
  PetscInt            steps;
  TSConvergedReason   reason;

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
  ierr = VecCreateSeq(PETSC_COMM_WORLD,4,&X);CHKERRQ(ierr);  
  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,4,4,4,NULL,&J);CHKERRQ(ierr);

  /* Set Initial Conditions and time Parameters*/
  ierr = FormInitialSolution(ts,X,&user);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,PETSC_MAX_INT,maxtime);

  /* Set RHS Functions amd Jacobians*/
  ierr = DMTSSetRHSFunction(dm,FormRHSFunction,&user);CHKERRQ(ierr); 

  /* Register Partitioned RHS Functions and Jacobians */
  ierr = DMTSSetRHSPartitionFunction(dm,TS_SYMP_PARTITION,TS_SYMP_P_SLOT,FormRHSFunctionP,&user);CHKERRQ(ierr);
  ierr = DMTSSetRHSPartitionFunction(dm,TS_SYMP_PARTITION,TS_SYMP_Q_SLOT,FormRHSFunctionQ,&user);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts, J, J, FormJacobian, &user);CHKERRQ(ierr); 
  ierr = DMTSSetRHSPartitionJacobian(dm,TS_SYMP_PARTITION,TS_SYMP_P_SLOT, FormJacobianP, &user);CHKERRQ(ierr);
    
  /* Set TS Type  */
  ierr = TSSetType(ts,TSSYMPEULER);CHKERRQ(ierr);

  /* Set from Options */
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
  x[0] = 1.57079632679;
  x[1] = 0.2;
  x[2] = 0.3;
  x[3] = 0.1;
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
  PetscScalar      *x,*f,opq1;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  opq1 = 1 + x[1];
  f[0] = x[2]/(opq1*opq1);
  f[1] = x[3]; 
  f[2] = -(1 + x[1]) * PetscSinScalar(x[0]);
  f[3] = (x[2] * x[2])/(opq1 * opq1 * opq1) - x[1]/(user->epsilon * user->epsilon) + PetscCosScalar(x[0]);
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
  User             *user = (User*) ctx;
  PetscScalar      *x,*f,opq1;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  opq1 = 1 + x[1];
  f[0] = 0;
  f[1] = 0;
  f[2] = -(1 + x[1]) * PetscSinScalar(x[0]);
  f[3] = (x[2] * x[2])/(opq1 * opq1 * opq1) - x[1]/(user->epsilon * user->epsilon) + PetscCosScalar(x[0]);
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
  PetscScalar      *x,*f,opq1;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  opq1 = 1 + x[1];
  f[0] = x[2]/(opq1*opq1);
  f[1] = x[3]; 
  f[2] = 0;
  f[3] = 0;
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
  PetscScalar       v[16],*x,opq1;
  PetscInt          idxm[4] = {0,1,2,3};

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  opq1 = 1 + x[1];
 v[0]  = 1/(opq1*opq1);           v[1]  = 0; v[2] = 0;                           v[3]  = -2/(opq1*opq1*opq1)*x[2];                                               
 v[4]  = 0;                       v[5]  = 1; v[6] = 0;                           v[5]  = 0;                                                                     
 v[8]  = 0;                       v[9]  = 0; v[10] = -opq1*PetscCosScalar(x[0]); v[9]  = -PetscSinScalar(x[0]);                                                 
 v[12] = 2/(opq1*opq1*opq1)*x[2]; v[13] = 0; v[14] = -PetscSinScalar(x[0]);      v[15] = -3/(opq1*opq1*opq1*opq1)*x[2]*x[2] - 1/(user->epsilon*user->epsilon); 

  ierr = MatSetValues(*B,4,idxm,4,idxm,v,INSERT_VALUES);CHKERRQ(ierr);
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
  PetscScalar       v[16],*x,opq1;
  PetscInt          idxm[4] = {0,1,2,3};

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  opq1 = 1 + x[1];
 v[0]  = 0;                       v[1]  = 0; v[2] = 0;                           v[3]  = 0;                                               
 v[4]  = 0;                       v[5]  = 1; v[6] = 0;                           v[5]  = 0;                                                                     
 v[8]  = 0;                       v[9]  = 0; v[10] = -opq1*PetscCosScalar(x[0]); v[9]  = -PetscSinScalar(x[0]);                                                 
 v[12] = 2/(opq1*opq1*opq1)*x[2]; v[13] = 0; v[14] = -PetscSinScalar(x[0]);      v[15] = -3/(opq1*opq1*opq1*opq1)*x[2]*x[2] - 1/(user->epsilon*user->epsilon); 
 
  ierr = MatSetValues(*B,4,idxm,4,idxm,v,INSERT_VALUES);CHKERRQ(ierr);
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

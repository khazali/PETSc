static const char help[] = "Demonstrate the interface to solve DAE systems with partitioned RHS functions\n";

/*

   Consider the following second order ODE:

   q''(t) + q(t) = \epsilon f(q) 
  
   As a first order system, we  have

   q'(t) = p(t)
   p'(t) = -q(t) + \epsilon f(q) 
   
   Write X = [p; q] and partition the righthand side

   X' = [q; p]' =  [p(t); 0] + [0; -q(t)] + [0 ; \epsilon f(q)]

*/

#include <petscts.h>

/* User-defined Functions */
static PetscErrorCode FormRHSFunctionPFast(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormRHSFunctionQFast(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormRHSFunctionPSlow(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormInitialSolution(TS,Vec,void*);

/* User context */
typedef struct
{
  PetscReal epsilon;
} User;

/* Main function */
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char* argv[])
{
  PetscErrorCode ierr;
  PetscMPIInt size;
  Vec X;
  //Mat J;
  TS ts;
  DM dm;
  User user;
  PetscReal dt = 0.1, maxtime = 1.0, ftime;
  PetscInt maxsteps = 1000, steps;
  TSConvergedReason reason;

  ierr = PetscInitialize(&argc, &argv, (char*) 0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only");
  
  /* User Parameters */
  user.epsilon = 0.01;

  /* Create TS */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr); // Should create a DM for us to use (or you might create one yourself, for example a DMDA)

  /* Set Initial Conditions */
  ierr = VecCreateSeq(PETSC_COMM_WORLD,2,&X);CHKERRQ(ierr);  
  ierr = FormInitialSolution(ts,X,&user);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);

  /* Set RHS Functions */

  // TEMP - change to TSDMXX to confirm that works..
  //ierr = TSSetRHSFunction(ts,X,FormRHSFunctionQFast,&user);CHKERRQ(ierr); 
  ierr = DMTSSetRHSFunction(dm,FormRHSFunctionQFast,&user);CHKERRQ(ierr); 
  // TSSetRHSFunction does error checking and creates a vector if X isn't provided, but also registers a function with the SNES object associated with the TS, which we may also have to do ourselves

  //// ................. 

  // Set RHSJacobians ? 

  /* Set other TS Parameters */
  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,maxsteps,maxtime);

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
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
    
  return EXIT_SUCCESS;
}
 
/* Initial Conditions */
#undef __FUNCT__
#define __FUNCT__ "FormInitialSolution"
static PetscErrorCode FormInitialSolution(TS ts,Vec X,void *ctx)
{ 
  User           *user = (User*)ctx;
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
RHS function for 'fast P' terms
*/
#undef __FUNCT__
#define __FUNCT__ "FormRHSFunctionPFast"
static PetscErrorCode FormRHSFunctionPFast(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{

  PetscScalar *x, *f;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = 0;
  f[1] = -x[0];
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
RHS function for 'fast Q' terms
*/
#undef __FUNCT__
#define __FUNCT__ "FormRHSFunctionQFast"
static PetscErrorCode FormRHSFunctionQFast(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{

  PetscScalar *x, *f;
  PetscErrorCode ierr;

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
RHS function for 'slow P' terms
*/
#undef __FUNCT__
#define __FUNCT__ "FormRHSFunctionPSlow"
static PetscErrorCode FormRHSFunctionPSlow(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{

  User           *user = (User*) ctx;
  PetscScalar    *x, *f;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = 0;
  f[1] = -user->epsilon*x[0]*x[0]*x[0]; //choose f(q) = -q^3 
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

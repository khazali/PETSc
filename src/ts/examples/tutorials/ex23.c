static const char help[] = "Demonstrates the use of TSEvaluteGradient";
/*
  Computes the gradient of

    Obj(x,m) = \int^{TF}_{T0} ||u||^2 dt

  where u obeys the ODE:

    udot = b*u
    u(0) = a

  The design variable is m = [a,b]

*/
#include <petscts.h>

#if 0
  PetscInt       rl,cl;
  ierr = VecGetLocalSize(U,&rl);CHKERRQ(ierr);
  ierr = VecGetLocalSize(M,&cl);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J,rl,cl,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
#endif  
/* returns ||u||^2 */
static PetscErrorCode EvalCostFunctional(TS ts, PetscReal time, Vec U, Vec M, PetscScalar *val, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecDot(U,U,val);CHKERRQ(ierr);
  //*val = PetscSqrtScalar(*val);
  PetscFunctionReturn(0);
}

/* returns 2*u */
static PetscErrorCode EvalCostGradient_U(TS ts, PetscReal time, Vec U, Vec M, Vec grad, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecCopy(U,grad);CHKERRQ(ierr);
  ierr = VecScale(grad,2.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* returns 0 */
static PetscErrorCode EvalCostGradient_M(TS ts, PetscReal time, Vec U, Vec M, Vec grad, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecSet(grad,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* returns \partial_m F(U,Udot,t;M) for a fixed design M. F(U,Udot,t) the ODE in implicit form */
static PetscErrorCode EvalGradient(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Mat J, void *ctx)
{
  PetscInt       rst,ren,r;
  PetscScalar    *arr;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,(const PetscScalar**)&arr);CHKERRQ(ierr);
  ierr = MatZeroEntries(J);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(J,&rst,&ren);CHKERRQ(ierr);
  for (r = rst; r < ren; r++) {
    PetscInt    c = 1;
    PetscScalar v = -arr[r-rst];
    ierr = MatSetValues(J,1,&r,1,&c,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,(const PetscScalar**)&arr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* returns \partial_u0 G and \partial_m G, with G the initial conditions in implicit form */
static PetscErrorCode EvalICGradient(TS ts, PetscReal t0, Vec u0, Vec M, Mat G_u0, Mat G_m, void *ctx)
{
  PetscInt       rst,ren,r;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatZeroEntries(G_u0);CHKERRQ(ierr);
  ierr = MatShift(G_u0,1.0);CHKERRQ(ierr);
  ierr = MatZeroEntries(G_m);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(G_m,&rst,&ren);CHKERRQ(ierr);
  for (r = rst; r < ren; r++) {
    PetscInt    c = 0;
    PetscScalar v = -1;
    ierr = MatSetValues(G_m,1,&r,1,&c,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(G_m,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(G_m,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscScalar a;
  PetscScalar b;
} User;

static PetscErrorCode FormRHSFunction(TS ts,PetscReal time, Vec U, Vec F,void* ctx)
{
  User           *user = (User*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecCopy(U,F);CHKERRQ(ierr);
  ierr = VecScale(F,user->b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestPostStep(TS ts)
{
  PetscReal      time;
  PetscInt       step;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetTimeStepNumber(ts,&step);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&time);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"  Inside %s at (step,time) %d,%g\n",__func__,step,time);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char* argv[])
{
  TS             ts;
  Vec            U,M,Mgrad;
  User           user;
  PetscReal      t0 = 0.0, tf = 2.0, dt = 0.1;
  PetscScalar    obj,objtest;
  PetscMPIInt    np;
  PetscBool      testpoststep = PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  t0 = 0.0;
  tf = 2.0;
  dt = 0.1;
  user.a = 2.0;
  user.b = 3.0;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"PDE-constrained options","");
  ierr = PetscOptionsScalar("-a","Initial condition","",user.a,&user.a,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-b","Grow rate","",user.b,&user.b,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-t0","Initial time","",t0,&t0,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-tf","Final time","",tf,&tf,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-dt","Initial time","",dt,&dt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_poststep","Test with PostStep method","",testpoststep,&testpoststep,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* state vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr);
  ierr = VecSetSizes(U,1,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(U,VECSTANDARD);CHKERRQ(ierr);

  /* design vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&M);CHKERRQ(ierr);
  ierr = VecSetSizes(M,PETSC_DECIDE,2);CHKERRQ(ierr);
  ierr = VecSetType(M,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecSetValue(M,0,user.a,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValue(M,1,user.b,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(M);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(M);CHKERRQ(ierr);
  ierr = VecDuplicate(M,&Mgrad);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,t0,dt);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,PETSC_MAX_INT,tf);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetCostFunctional(ts,PETSC_MIN_REAL,EvalCostFunctional,NULL,EvalCostGradient_U,NULL,EvalCostGradient_M,NULL);CHKERRQ(ierr);
  ierr = TSSetEvalGradient(ts,NULL,EvalGradient,NULL);CHKERRQ(ierr);
  ierr = TSSetEvalICGradient(ts,NULL,NULL,EvalICGradient,NULL);CHKERRQ(ierr);
  if (testpoststep) {
    ierr = TSSetPostStep(ts,TestPostStep);CHKERRQ(ierr);
  }
  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,&user);CHKERRQ(ierr);
  //ierr = TSSetIFunction(ts,NULL,FormIFunction,&user);CHKERRQ(ierr);
  //ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
  //ierr = TSSolve(ts,NULL);CHKERRQ(ierr);

  /* Test objective function evaluation */
  ierr = VecSet(U,user.a);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
  ierr = TSEvaluateCostFunctionals(ts,U,M,&obj);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&tf);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&np);CHKERRQ(ierr);
  objtest = np * (user.a * user.a) / (2.0*user.b) * (PetscExpScalar(2.0*user.b*(tf-t0)) - 1.0);
  //objtest = np * (user.a) / (user.b) * (PetscExpScalar(user.b*(tf-t0)) - 1.0);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Objective function: time [%g,%g], val %g (should be %g)\n",t0,tf,(double)PetscRealPart(obj),(double)objtest);CHKERRQ(ierr);

  //ierr = TSEvaluateGradient(ts,U,M,Mgrad);CHKERRQ(ierr);
  //ierr = VecSet(U,user.a);CHKERRQ(ierr);
  //ierr = TSEvaluateGradient(ts,U,M,Mgrad);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&M);CHKERRQ(ierr);
  ierr = VecDestroy(&Mgrad);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

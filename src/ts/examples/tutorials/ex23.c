static const char help[] = "Demonstrates the use of TSEvaluteGradient";
/*
  Computes the gradient of

    Obj(x,m) = \int^{TF}_{T0} f(u) dt

  where u obeys the ODE:

    udot = b*u^p
    u(0) = a

  The integrand of the objective function can be either f(u) = ||u||^2 or f(u) = Sum(u)
  The design variables are m = [a,b,p].
*/
#include <petscts.h>

typedef struct {
  PetscBool isnorm;
} UserObjective;

/* returns ||u||^2  or Sum(u), depending on the objective function selected */
static PetscErrorCode EvalCostFunctional(TS ts, PetscReal time, Vec U, Vec M, PetscReal *val, void *ctx)
{
  UserObjective  *user = (UserObjective*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (user->isnorm) {
    ierr  = VecNorm(U,NORM_2,val);CHKERRQ(ierr);
    *val *= *val;
  } else {
    PetscScalar sval;
    ierr = VecSum(U,&sval);CHKERRQ(ierr);
    *val = PetscRealPart(sval);
  }
  PetscFunctionReturn(0);
}

/* returns 2*u or 1, depending on the objective function selected */
static PetscErrorCode EvalCostGradient_U(TS ts, PetscReal time, Vec U, Vec M, Vec grad, void *ctx)
{
  UserObjective  *user = (UserObjective*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (user->isnorm) {
    ierr = VecCopy(U,grad);CHKERRQ(ierr);
    ierr = VecScale(grad,2.0);CHKERRQ(ierr);
  } else {
    ierr = VecSet(grad,1.0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* returns \partial_m f(u) */
static PetscErrorCode EvalCostGradient_M(TS ts, PetscReal time, Vec U, Vec M, Vec grad, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecSet(grad,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  PetscScalar a;
  PetscScalar b;
  PetscReal   p;
} User;

/* returns \partial_m F(U,Udot,t;M) for a fixed design M. F(U,Udot,t) the ODE in implicit form */
static PetscErrorCode EvalGradient(TS ts, PetscReal time, Vec U, Vec Udot, Vec M, Mat J, void *ctx)
{
  User           *user = (User*)ctx;
  PetscInt       rst,ren,r;
  PetscScalar    *arr;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,(const PetscScalar**)&arr);CHKERRQ(ierr);
  ierr = MatZeroEntries(J);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(J,&rst,&ren);CHKERRQ(ierr);
  for (r = rst; r < ren; r++) {
    PetscInt    c = 1;
    PetscScalar v = -PetscPowScalarReal(arr[r-rst],user->p);
    /* F_b : -x^p */
    ierr = MatSetValues(J,1,&r,1,&c,&v,INSERT_VALUES);CHKERRQ(ierr);
    /* F_p : -b*x^p*log(x) */
    c = 2;
    v *= user->b*PetscLogReal(arr[r-rst]);
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
  if (G_u0) {
    ierr = MatZeroEntries(G_u0);CHKERRQ(ierr);
    ierr = MatShift(G_u0,1.0);CHKERRQ(ierr);
  }
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

static PetscErrorCode FormIFunction(TS ts,PetscReal time, Vec U, Vec Udot, Vec F,void* ctx)
{
  User           *user = (User*)ctx;
  PetscScalar    *aU,*aUdot,*aF;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(F,&aF);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,(const PetscScalar**)&aU);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,(const PetscScalar**)&aUdot);CHKERRQ(ierr);
  aF[0] = aUdot[0] - user->b*PetscPowScalarReal(aU[0],user->p);
  ierr = VecRestoreArrayRead(U,(const PetscScalar**)&aU);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,(const PetscScalar**)&aUdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&aF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormIJacobian(TS ts,PetscReal time, Vec U, Vec Udot, PetscReal shift, Mat A, Mat P, void* ctx)
{
  User           *user = (User*)ctx;
  PetscInt       i;
  PetscScalar    *aU,v;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = MatShift(A,shift);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,(const PetscScalar**)&aU);CHKERRQ(ierr);
  v    = -user->b*user->p*PetscPowScalarReal(aU[0],user->p - 1.0);
  ierr = MatGetOwnershipRange(A,&i,NULL);CHKERRQ(ierr);
  ierr = MatSetValues(A,1,&i,1,&i,&v,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,(const PetscScalar**)&aU);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSFunction(TS ts,PetscReal time, Vec U, Vec F,void* ctx)
{
  User           *user = (User*)ctx;
  PetscScalar    *aU,*aF;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArray(F,&aF);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,(const PetscScalar**)&aU);CHKERRQ(ierr);
  aF[0] = user->b*PetscPowScalarReal(aU[0],user->p);
  ierr = VecRestoreArrayRead(U,(const PetscScalar**)&aU);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&aF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSJacobian(TS ts,PetscReal time, Vec U, Mat A, Mat P, void* ctx)
{
  User           *user = (User*)ctx;
  PetscInt       i;
  PetscScalar    *aU,v;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,(const PetscScalar**)&aU);CHKERRQ(ierr);
  v    = user->b*user->p*PetscPowScalarReal(aU[0],user->p - 1.0);
  ierr = MatGetOwnershipRange(A,&i,NULL);CHKERRQ(ierr);
  ierr = MatSetValues(A,1,&i,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,(const PetscScalar**)&aU);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestPostStep(TS ts)
{
  PetscReal      time;
  PetscInt       step;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetStepNumber(ts,&step);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&time);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"  Inside %s at (step,time) %d,%g\n",__func__,step,time);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char* argv[])
{
  TS             ts;
  Mat            J,G_M,F_M,G_X;
  Vec            U,M,Mgrad;
  UserObjective  userobj;
  User           user;
  PetscScalar    one = 1.0;
  PetscReal      t0 = 0.0, tf = 2.0, dt = 0.1;
  PetscReal      obj,objtest;
  PetscInt       maxsteps;
  PetscMPIInt    np;
  PetscBool      testpoststep = PETSC_FALSE;
  PetscBool      testifunc = PETSC_FALSE;
  PetscBool      testrhsjacconst = PETSC_FALSE;
  PetscBool      testnullgradM = PETSC_FALSE;
  PetscBool      testnulljacIC = PETSC_FALSE;
  PetscBool      usefd = PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* Command line options */
  t0             = 0.0;
  tf             = 2.0;
  dt             = 1.0/512.0;
  user.a         = 1.0;
  user.b         = 1.0;
  user.p         = 1.0;
  userobj.isnorm = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"PDE-constrained options","");
  ierr = PetscOptionsScalar("-a","Initial condition","",user.a,&user.a,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-b","Grow rate","",user.b,&user.b,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-p","Nonlinearity","",user.p,&user.p,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-t0","Initial time","",t0,&t0,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tf","Final time","",tf,&tf,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt","Initial time","",dt,&dt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_objective_norm","Test f(u) = ||u||^2","",userobj.isnorm,&userobj.isnorm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_poststep","Test with PostStep method","",testpoststep,&testpoststep,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_ifunc","Test with IFunction interface","",testifunc,&testifunc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_rhsjacconst","Test with TSComputeRHSJacobianConstant","",testrhsjacconst,&testrhsjacconst,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_nullgrad_M","Test with NULL M gradient","",testnullgradM,&testnullgradM,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_nulljac_IC","Test with NULL G_X jacobian","",testnulljacIC,&testnulljacIC,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_fd","Use finite differencing to test gradient evaluation","",usefd,&usefd,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* we test with finite differencing the nonlinear case */
  if (user.p != 1.0) usefd = PETSC_TRUE;

  /* state vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr);
  ierr = VecSetSizes(U,1,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(U,VECSTANDARD);CHKERRQ(ierr);

  /* design vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&M);CHKERRQ(ierr);
  ierr = VecSetSizes(M,PETSC_DECIDE,3);CHKERRQ(ierr);
  ierr = VecSetType(M,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecSetRandom(M,NULL);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(M);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(M);CHKERRQ(ierr);
  ierr = VecDuplicate(M,&Mgrad);CHKERRQ(ierr);

  /* rhs jacobian */
  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,1,1,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Jacobian for F_m */
  ierr = MatCreate(PETSC_COMM_WORLD,&F_M);CHKERRQ(ierr);
  ierr = MatSetSizes(F_M,1,PETSC_DECIDE,PETSC_DECIDE,3);CHKERRQ(ierr);
  ierr = MatSetUp(F_M);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(F_M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(F_M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Jacobians for initial conditions */
  ierr = MatCreate(PETSC_COMM_WORLD,&G_X);CHKERRQ(ierr);
  ierr = MatSetSizes(G_X,1,1,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetUp(G_X);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(G_X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(G_X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&G_M);CHKERRQ(ierr);
  ierr = MatSetSizes(G_M,1,PETSC_DECIDE,PETSC_DECIDE,3);CHKERRQ(ierr);
  ierr = MatSetUp(G_M);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(G_M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(G_M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* TS solver */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetTime(ts,t0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,tf);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,PETSC_MAX_INT);CHKERRQ(ierr);
  if (testpoststep) {
    ierr = TSSetPostStep(ts,TestPostStep);CHKERRQ(ierr);
  }
  if (!testifunc) {
    ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,&user);CHKERRQ(ierr);
    if (testrhsjacconst) {
      ierr = FormRHSJacobian(ts,0.0,NULL,J,J,&user);CHKERRQ(ierr);
      ierr = TSSetRHSJacobian(ts,J,J,TSComputeRHSJacobianConstant,NULL);CHKERRQ(ierr);
    } else {
      ierr = TSSetRHSJacobian(ts,J,J,FormRHSJacobian,&user);CHKERRQ(ierr);
    }
  } else {
    ierr = TSSetIFunction(ts,NULL,FormIFunction,&user);CHKERRQ(ierr);
    ierr = TSSetIJacobian(ts,J,J,FormIJacobian,&user);CHKERRQ(ierr);
  }
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* force matchstep and get initial time and final time requested */
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
  ierr = TSGetMaxTime(ts,&tf);CHKERRQ(ierr);
  ierr = TSGetMaxSteps(ts,&maxsteps);CHKERRQ(ierr);

  /* Set cost functionals */
  if (testnullgradM) {
    ierr = TSSetCostFunctional(ts,PETSC_MIN_REAL,EvalCostFunctional,&userobj,EvalCostGradient_U,&userobj,NULL,NULL);CHKERRQ(ierr);
  } else {
    ierr = TSSetCostFunctional(ts,PETSC_MIN_REAL,EvalCostFunctional,&userobj,EvalCostGradient_U,&userobj,EvalCostGradient_M,&userobj);CHKERRQ(ierr);
  }

  /* Set dependence of F(Udot,U,t;M) = 0 from the parameters */
  ierr = TSSetEvalGradient(ts,F_M,EvalGradient,&user);CHKERRQ(ierr);

  /* Set dependence of initial conditions (in implicit form G(U(0);M) = 0) from the parameters */
  if (testnulljacIC) {
    ierr = TSSetEvalICGradient(ts,NULL,G_M,EvalICGradient,NULL);CHKERRQ(ierr);
  } else {
    ierr = TSSetEvalICGradient(ts,G_X,G_M,EvalICGradient,NULL);CHKERRQ(ierr);
  }

  /* Test objective function evaluation */
  ierr = VecSet(U,user.a);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
  ierr = TSEvaluateCostFunctionals(ts,U,M,&obj);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&tf);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&np);CHKERRQ(ierr);
  if (user.p == 1.0) {
    if (userobj.isnorm) {
      objtest = np * PetscRealPart((user.a * user.a) / (2.0*user.b) * (PetscExpScalar(2.0*(tf-t0)*user.b) - one));
    } else {
      objtest = np * PetscRealPart((user.a / user.b) * (PetscExpScalar((tf-t0)*user.b) - one));
    }
  } else {
    PetscReal   scale = userobj.isnorm ? 2.0 : 1.0;
    PetscScalar alpha = PetscPowScalarReal(user.a,1.0-user.p);
    PetscScalar  beta = user.b*(1.0-user.p);
    PetscReal   gamma = scale/(1.0-user.p);
    objtest = np / ((gamma + 1.0) * beta )* ( PetscPowScalar(beta*(tf-t0)+alpha,gamma+1.0) - PetscPowScalar(alpha,gamma+1.0) );

  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Objective function: time [%g,%g], val %g (should be %g)\n",t0,tf,(double)obj,(double)objtest);CHKERRQ(ierr);

  /* Test gradient evaluation */
  ierr = TSSetTime(ts,t0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,tf);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,maxsteps);CHKERRQ(ierr);
  ierr = VecSet(U,user.a);CHKERRQ(ierr);
  ierr = TSEvaluateGradient(ts,U,M,Mgrad);CHKERRQ(ierr);
  if (usefd) { /* we test against finite differencing the function evaluation */
    PetscScalar oa = user.a, ob = user.b, op = user.p, dx = PETSC_SMALL;
    PetscReal   objadx1,objbdx1,objpdx1;
    PetscReal   objadx2,objbdx2,objpdx2;

    user.a = oa + dx;
    user.b = ob;
    user.p = op;
    ierr = VecSetValue(M,0,user.a,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(M,1,user.b,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(M,2,user.p,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(M);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(M);CHKERRQ(ierr);
    ierr = VecSet(U,user.a);CHKERRQ(ierr);
    ierr = TSSetTime(ts,t0);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,tf);CHKERRQ(ierr);
    ierr = TSSetMaxSteps(ts,maxsteps);CHKERRQ(ierr);
    ierr = TSEvaluateCostFunctionals(ts,U,M,&objadx1);CHKERRQ(ierr);
    user.a = oa - dx;
    user.b = ob;
    user.p = op;
    ierr = VecSetValue(M,0,user.a,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(M,1,user.b,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(M,2,user.p,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(M);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(M);CHKERRQ(ierr);
    ierr = VecSet(U,user.a);CHKERRQ(ierr);
    ierr = TSSetTime(ts,t0);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,tf);CHKERRQ(ierr);
    ierr = TSSetMaxSteps(ts,maxsteps);CHKERRQ(ierr);
    ierr = TSEvaluateCostFunctionals(ts,U,M,&objadx2);CHKERRQ(ierr);

    user.a = oa;
    user.b = ob + dx;
    user.p = op;
    ierr = VecSetValue(M,0,user.a,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(M,1,user.b,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(M,2,user.p,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(M);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(M);CHKERRQ(ierr);
    ierr = VecSet(U,user.a);CHKERRQ(ierr);
    ierr = TSSetTime(ts,t0);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,tf);CHKERRQ(ierr);
    ierr = TSSetMaxSteps(ts,maxsteps);CHKERRQ(ierr);
    ierr = TSEvaluateCostFunctionals(ts,U,M,&objbdx1);CHKERRQ(ierr);
    user.a = oa;
    user.b = ob - dx;
    user.p = op;
    ierr = VecSetValue(M,0,user.a,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(M,1,user.b,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(M,2,user.p,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(M);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(M);CHKERRQ(ierr);
    ierr = VecSet(U,user.a);CHKERRQ(ierr);
    ierr = TSSetTime(ts,t0);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,tf);CHKERRQ(ierr);
    ierr = TSSetMaxSteps(ts,maxsteps);CHKERRQ(ierr);
    ierr = TSEvaluateCostFunctionals(ts,U,M,&objbdx2);CHKERRQ(ierr);

    user.a = oa;
    user.b = ob;
    user.p = op + dx;
    ierr = VecSetValue(M,0,user.a,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(M,1,user.b,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(M,2,user.p,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(M);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(M);CHKERRQ(ierr);
    ierr = VecSet(U,user.a);CHKERRQ(ierr);
    ierr = TSSetTime(ts,t0);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,tf);CHKERRQ(ierr);
    ierr = TSSetMaxSteps(ts,maxsteps);CHKERRQ(ierr);
    ierr = TSEvaluateCostFunctionals(ts,U,M,&objpdx1);CHKERRQ(ierr);
    user.a = oa;
    user.b = ob;
    user.p = op - dx;
    ierr = VecSetValue(M,0,user.a,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(M,1,user.b,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(M,2,user.p,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(M);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(M);CHKERRQ(ierr);
    ierr = VecSet(U,user.a);CHKERRQ(ierr);
    ierr = TSSetTime(ts,t0);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,tf);CHKERRQ(ierr);
    ierr = TSSetMaxSteps(ts,maxsteps);CHKERRQ(ierr);
    ierr = TSEvaluateCostFunctionals(ts,U,M,&objpdx2);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"1st component of gradient should be (approximated) %g\n",(double)((objadx1-objadx2)/PetscRealPart(2*dx)));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"2nd component of gradient should be (approximated) %g\n",(double)((objbdx1-objbdx2)/PetscRealPart(2*dx)));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"3rd component of gradient should be (approximated) %g\n",(double)((objpdx1-objpdx2)/PetscRealPart(2*dx)));CHKERRQ(ierr);
  } else { /* analytic solution */
    if (userobj.isnorm) {
      objtest = np * PetscRealPart( user.a / user.b * (PetscExpScalar(2.0*(tf-t0)*user.b) - one));
    } else {
      objtest = np * PetscRealPart(-(one - PetscExpScalar((tf-t0)*user.b))/user.b);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"1st component of gradient should be (analytic) %g\n",(double)objtest);CHKERRQ(ierr);
    if (userobj.isnorm) {
      objtest = np * PetscRealPart(0.5*user.a*user.a/ (user.b*user.b) * ( (2.0*user.b*(tf-t0) - one)*PetscExpScalar(2.0*(tf-t0)*user.b) + one));
    } else {
      objtest = np * PetscRealPart((user.a/user.b)*( (tf-t0)*PetscExpScalar((tf-t0)*user.b) - (PetscExpScalar((tf-t0)*user.b) - one)/user.b));
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"2nd component of gradient should be (analytic) %g\n",(double)objtest);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"3rd component of gradient not yet coded (use -use_fd to test it)\n");CHKERRQ(ierr);
  }
  ierr = VecView(Mgrad,NULL);CHKERRQ(ierr);

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&M);CHKERRQ(ierr);
  ierr = VecDestroy(&Mgrad);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = MatDestroy(&G_M);CHKERRQ(ierr);
  ierr = MatDestroy(&G_X);CHKERRQ(ierr);
  ierr = MatDestroy(&F_M);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

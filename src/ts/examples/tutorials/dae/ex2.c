/**
   This program solves the index 1 semi explicit DAE
     y' = -y^2 + z
     cost(y)-sqrt(z) = 0
   on the domain 0.1 <= t <= 0.2.

   y0 = 0.25
   z0 = cos(y0)^2
**/

#include <petscts.h>

typedef struct _n_User *User;
struct _n_User {
  PetscInt  steps;
  PetscReal stime,ftime;
  Mat       Jac; /* Jacobian matrix */
  Vec       sol;
};

static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec U,Vec G,void *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *g;
  const PetscScalar *u;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArray(G,&g);CHKERRQ(ierr);
  g[0] = -u[0]*u[0]+u[1];
  g[1] = 0.0;
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArray(G,&g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *u,*udot;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = udot[0];
  f[1] = PetscCosReal(u[0])-PetscSqrtReal(u[1]);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  ierr    = VecGetArrayRead(U,&u);CHKERRQ(ierr);

  J[0][0] = a;     J[0][1] = 0;
  J[1][0] = -PetscSinReal(u[0]);   J[1][1] = -0.5/PetscSqrtReal(u[1]);

  ierr = MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;
  PetscScalar    *x_ptr;
  PetscMPIInt    size;
  struct _n_User user;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,NULL,NULL);

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");
  user.steps       = 0;
  user.stime       = -1.;
  user.ftime       = 2.;
  ierr = MatCreate(PETSC_COMM_WORLD,&user.Jac);CHKERRQ(ierr);
  ierr = MatSetSizes(user.Jac,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user.Jac);CHKERRQ(ierr);
  ierr = MatSetUp(user.Jac);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.Jac,&user.sol,NULL);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,(void*)&user);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,IFunction,(void*)&user);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,user.Jac,user.Jac,(TSIJacobian)IJacobian,(void*)&user);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.01);CHKERRQ(ierr);
  ierr = TSSetTime(ts,user.stime);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,user.ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  ierr = VecGetArray(user.sol,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.25;
  x_ptr[1] = (PetscCosReal(x_ptr[0]))*(PetscCosReal(x_ptr[0]));
  ierr = VecRestoreArray(user.sol,&x_ptr);CHKERRQ(ierr);

  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts,user.sol);CHKERRQ(ierr);

  ierr = MatDestroy(&user.Jac);CHKERRQ(ierr);
  ierr = VecDestroy(&user.sol);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}

static char help[] = "Test threadcomm with type=pthread, model=loop with PETSc vector routines.\n\n";

#include <petscvec.h>
#include <petscthreadcomm.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec x, y;
  PetscErrorCode  ierr;
  PetscInt n=20;
  PetscScalar vnorm,alpha=3.0;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  //ierr = PetscOptionsGetInt(NULL,"-nthreads",&nthreads,NULL);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD,"\n\nRunning test 1\n");
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);

  // Run PETSc code
  ierr = VecSet(x,2.0);CHKERRQ(ierr);
  ierr = VecSet(y,3.0);CHKERRQ(ierr);
  ierr = VecAXPY(y,alpha,x);CHKERRQ(ierr);
  //VecView(y,PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecNorm(y,NORM_2,&vnorm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm=%f\n",vnorm);CHKERRQ(ierr);

  ierr = VecPointwiseMult(y,y,y);CHKERRQ(ierr);
  ierr = VecScale(y,2.0);CHKERRQ(ierr);
  ierr = VecAXPY(y,alpha,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&vnorm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm=%f\n",vnorm);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD,"\n\nRunning test 2\n");
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSet(x,2.0);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);
  ierr = VecSet(y,3.0);CHKERRQ(ierr);

  ierr = VecAXPY(y,alpha,x);CHKERRQ(ierr);
  //VecView(y,PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecNorm(y,NORM_2,&vnorm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Different Comms Norm=%f\n",vnorm);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}

static char help[] = "Test threadcomm with loop model or nothread thread type with PETSc vector routines.\n\n";

#include <petscvec.h>
#include <petscthreadcomm.h>

/*
   Example run commands:
   PThreads: ./ex8 -n 1000 -threadcomm_type pthread -threadcomm_model loop -threadcomm_nthreads $nthreads
   OpenMP: ./ex8 -n 1000 -threadcomm_type openmp -threadcomm_model loop -threadcomm_nthreads $nthreads
   TBB: ./ex8 -n 1000 -threadcomm_type tbb -threadcomm_model loop -threadcomm_nthreads $nthreads
   NoThread: ./ex8 -n 1000
*/

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec            x,y;
  PetscErrorCode ierr;
  PetscInt       n=20,nthreads;
  PetscScalar    vnorm,alpha=3.0;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nRunning test 1\n");CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);

  ierr = PetscThreadCommGetNThreads(PETSC_COMM_WORLD,&nthreads);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"PETSC_COMM_WORLD has %d threads\n",nthreads);CHKERRQ(ierr);

  /*Run PETSc code */
  ierr = VecSet(x,2.0);CHKERRQ(ierr);
  ierr = VecSet(y,3.0);CHKERRQ(ierr);
  ierr = VecAXPY(y,alpha,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&vnorm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test1 Part1 Norm=%f\n",vnorm);CHKERRQ(ierr);

  ierr = VecPointwiseMult(y,y,y);CHKERRQ(ierr);
  ierr = VecScale(y,2.0);CHKERRQ(ierr);
  ierr = VecAXPY(y,alpha,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&vnorm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test1 Part2 Norm=%f\n",vnorm);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nRunning test 2\n");CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSet(x,2.0);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);
  ierr = VecSet(y,3.0);CHKERRQ(ierr);

  ierr = VecAXPY(y,alpha,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&vnorm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test2 Norm=%f\n",vnorm);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

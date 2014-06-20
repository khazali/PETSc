static char help[] = "Test PetscThreadPool with pthreads with PETSc vector routines.\n\n";

#include <petscvec.h>
#include <petscthreadcomm.h>

void func(void *arg);

Vec x, y;
PetscScalar *ay;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  PetscInt n=20, nthreads, tnum;
  MPI_Comm comm1;
  PetscThreadComm tcomm1;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRCONTINUE(ierr);
  ierr = PetscThreadCommGetNThreads(PETSC_COMM_WORLD,&nthreads);CHKERRQ(ierr);

  // Create MPI_Comm and ThreadComm from PETSC_COMM_WORLD
  // Create worker threads in PETSc, master thread returns
  ierr = PetscThreadCommCreate(PETSC_COMM_WORLD,nthreads,PETSC_TRUE,&comm1,&tcomm1);

  // Create two vectors on MPIComm/ThreadComm
  ierr = VecCreate(comm1,&x);CHKERRCONTINUE(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRCONTINUE(ierr);
  ierr = VecSetFromOptions(x);CHKERRCONTINUE(ierr);
  ierr = VecDuplicate(x,&y);CHKERRCONTINUE(ierr);

  // Run PETSc code
  ierr = VecSet(x,2.0);CHKERRCONTINUE(ierr);
  ierr = VecSet(y,3.0);CHKERRCONTINUE(ierr);
  ierr = VecAXPY(y,alpha,x);CHKERRCONTINUE(ierr);
  //VecView(y,PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecNorm(y,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
  ierr = PetscPrintf(comm1,"Norm=%f\n",vnorm);

  ierr = VecDestroy(&x);CHKERRCONTINUE(ierr);
  ierr = VecDestroy(&y);CHKERRCONTINUE(ierr);
  printf("Calling PetscFinalize\n");
  PetscFinalize();

  return 0;
}

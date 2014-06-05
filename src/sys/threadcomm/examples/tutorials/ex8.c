static char help[] = "Test Splitting PetscThreadPool.\n\n";

#include <petscvec.h>
#include <omp.h>
#include <petscthreadcomm.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec             x;
  PetscErrorCode  ierr;
  PetscInt        prank, nthreads, n=20;
  PetscThreadPool pool;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(PETSC_COMM_WORLD,&nthreads);CHKERRQ(ierr);
  printf("nthreads=%d\n",nthreads);
  PetscThreadPoolCreate(PETSC_COMM_WORLD, &pool);

  #pragma omp parallel num_threads(nthreads)
  {
    int trank = omp_get_thread_num();

    // Give pool to PETSc to do work
    ierr = PetscThreadPoolJoin(pool,&prank);CHKERRCONTINUE(ierr);
    if(prank) {
      ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRCONTINUE(ierr);
      ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRCONTINUE(ierr);
      ierr = VecSetFromOptions(x);CHKERRCONTINUE(ierr);

      ierr = VecSet(x,1.0);CHKERRCONTINUE(ierr);
    }
    // Get pool back
    ierr = PetscThreadPoolReturn(pool,&prank);CHKERRCONTINUE(ierr);

    // Create multiple pools
    PetscThreadPoolSplit(pool,2,&prank);

    // Give both pools to PETSc
    ierr = PetscThreadPoolJoin(pool,&prank);CHKERRCONTINUE(ierr);
    // Use first pool
    if(prank==1) {
      ierr = VecDuplicate(x,&y);CHKERRCONTINUE(ierr);
      ierr = VecSet(y,2.0);CHKERRCONTINUE(ierr);
    }
    // Use second pool
    if(prank==2) {
      ierr = VecDuplicate(x,&z);CHKERRCONTINUE(ierr);
      ierr = VecSet(z,3.0);CHKERRCONTINUE(ierr);
    }
    // Get pools back
    ierr = PetscThreadPoolReturn(pool,&prank);CHKERRCONTINUE(ierr);

    // Merge pools together
    PetscThreadPoolMerge(pool,2,&prank);

    ierr = PetscThreadPoolJoin(pool,&prank);CHKERRCONTINUE(ierr);
    if(prank) {
      ierr = VecDestroy(&x);CHKERRCONTINUE(ierr);
      ierr = VecDestroy(&y);CHKERRCONTINUE(ierr);
      ierr = VecDestroy(&z);CHKERRCONTINUE(ierr);
    }
    ierr = PetscThreadPoolReturn(pool,&prank);CHKERRCONTINUE(ierr);
  }

  PetscThreadPoolDestroy(pool);

  PetscFinalize();
  return 0;
}

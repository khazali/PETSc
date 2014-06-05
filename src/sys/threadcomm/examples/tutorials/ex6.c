static char help[] = "Test PetscThreadPool.\n\n";

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
  PetscScalar     one = 1.0;
  PetscThreadPool pool;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(PETSC_COMM_WORLD,&nthreads);CHKERRQ(ierr);
  printf("nthreads=%d\n",nthreads);
  PetscThreadPoolCreate(PETSC_COMM_WORLD, &pool);

  #pragma omp parallel num_threads(nthreads)
  {
    int trank = omp_get_thread_num();

    ierr = PetscThreadPoolJoin(pool,&prank);CHKERRCONTINUE(ierr);
    printf("trank=%d joined pool prank=%d\n",trank,prank);
    if(prank) {
      printf("trank=%d working on vec\n",trank);
      ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRCONTINUE(ierr);
      ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRCONTINUE(ierr);
      ierr = VecSetFromOptions(x);CHKERRCONTINUE(ierr);

      printf("vec set\n");
      ierr = VecSet(x,one);CHKERRCONTINUE(ierr);

      //PetscThreadPoolBarrier(pool);
      VecView(x,PETSC_VIEWER_STDOUT_WORLD);

      printf("vec destroy\n");
      ierr = VecDestroy(&x);CHKERRCONTINUE(ierr);
    }
    ierr = PetscThreadPoolReturn(pool,&prank);CHKERRCONTINUE(ierr);
  }

  PetscThreadPoolDestroy(pool);

  PetscFinalize();
  return 0;
}

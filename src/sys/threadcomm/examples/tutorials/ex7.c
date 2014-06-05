static char help[] = "Test PetscThreadPool with openmp.\n\n";

#include <petscvec.h>
#include <omp.h>
#include <petscthreadcomm.h>
#include <../src/sys/threadcomm/impls/openmp/tcopenmpimpl.h>

PetscErrorCode parmainfunc(PetscInt trank);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt       n;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  PetscThreadCommExecute_OpenMP(&parmainfunc,PETSC_COMM_WORLD);

  PetscFinalize();
  return 0;
}

// This would be hidden from user in PETSc
// Note that PETSc MPI_Comm contains threadcomm and threadcomm contains threadpools
/*PetscErrorCode PetscThreadPoolExecuteOpenMP(void *func, MPI_Comm comm) {

  PetscInt

  ierr = PetscThreadCommGetNThreads(PETSC_COMM_WORLD,&nthreads);CHKERRQ(ierr);
  printf("nthreads=%d\n",nthreads);
  PetscThreadPoolCreate(comm, &pool);

  #pragma omp parallel num_threads(nthreads)
  {
    int trank = omp_get_thread_num();
    mainfunc(trank);
  }

  PetscThreadPoolDestroy(pool);
}*/

PetscErrorCode parmainfunc(PetscInt trank) {

  Vec x;
  PetscInt prank,n=100;
  PetscErrorCode ierr;

  // Insert parallel threaded user code here

  // User gives threads to PETSc to use for PETSc functions
  ierr = PetscThreadPoolJoin(PETSC_COMM_WORLD,trank,&prank);CHKERRCONTINUE(ierr);
  printf("trank=%d joined pool prank=%d\n",trank,prank);
  if(prank>=0) {
    printf("rank=%d working on vec\n",trank);
    ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRCONTINUE(ierr);
    ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRCONTINUE(ierr);
    ierr = VecSetFromOptions(x);CHKERRCONTINUE(ierr);

    printf("vec set\n");
    ierr = VecSet(x,1.0);CHKERRCONTINUE(ierr);

    //PetscThreadPoolBarrier(pool);
    VecView(x,PETSC_VIEWER_STDOUT_WORLD);

    printf("vec destroy\n");
    ierr = VecDestroy(&x);CHKERRCONTINUE(ierr);
  }

  printf("done with parallel work\n");

  // User takes back threads from PETSc once done calling PETSc functions
  ierr = PetscThreadPoolReturn(PETSC_COMM_WORLD,&prank);CHKERRCONTINUE(ierr);

  printf("Returned from pool\n");

  // Insert parallel threaded user code here

  return ierr;
}

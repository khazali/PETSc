static char help[] = "Test PetscThreadPool.\n\n";

#include <petscvec.h>
#include <omp.h>
#include <petscthreadcomm.h>
#include <../src/sys/threadcomm/impls/pthread/tcpthreadimpl.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec             x, y;
  PetscErrorCode  ierr;
  PetscInt        nthreads, n=20, *indices;
  PetscScalar     alpha=3.0;
  PetscScalar     *ay;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(PETSC_COMM_WORLD,&nthreads);CHKERRQ(ierr);
  printf("nthreads=%d\n",nthreads);

#pragma omp parallel num_threads(nthreads) default(shared) private(ierr)
  {
    int i, prank, start, end;
    PetscScalar vnorm=0.0;
    int trank = omp_get_thread_num();

    // User gives threads to PETSc for threaded PETSc work
    ierr = PetscThreadPoolJoin(PETSC_COMM_WORLD,trank,&prank);CHKERRCONTINUE(ierr);
    printf("trank=%d joined pool prank=%d\n",trank,prank);
    if(prank>=0) {
      printf("Working on vec\n");
      ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRCONTINUE(ierr);
      ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRCONTINUE(ierr);
      ierr = VecSetFromOptions(x);CHKERRCONTINUE(ierr);
      ierr = VecDuplicate(x,&y);CHKERRCONTINUE(ierr);

      printf("Vec set\n");
      ierr = VecSet(x,2.0);CHKERRCONTINUE(ierr);
      ierr = VecSet(y,3.0);CHKERRCONTINUE(ierr);
      ierr = VecAXPY(y,alpha,x);CHKERRCONTINUE(ierr);

      //VecView(y,PETSC_VIEWER_STDOUT_WORLD);
      ierr = VecNorm(y,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      printf("Norm=%f\n",vnorm);
      ierr = VecGetArray(y,&ay);CHKERRCONTINUE(ierr);
      ierr = PetscThreadCommGetOwnershipRanges(PETSC_COMM_WORLD,n,&indices);CHKERRCONTINUE(ierr);
    }
    ierr = PetscThreadPoolReturn(PETSC_COMM_WORLD,&prank);CHKERRCONTINUE(ierr);

    // Parallel threaded user code
    start = indices[trank];
    end = indices[trank+1];
    for(i=start; i<end; i++) {
      ay[i] = ay[i]*ay[i];
    }

    // User gives threads to PETSc for threaded PETSc work
    ierr = PetscThreadPoolJoin(PETSC_COMM_WORLD,trank,&prank);CHKERRCONTINUE(ierr);
    if(prank>=0) {
      ierr = VecRestoreArray(y,&ay);CHKERRCONTINUE(ierr);
      ierr = VecScale(y,2.0);CHKERRCONTINUE(ierr);
      ierr = VecAXPY(y,alpha,x);CHKERRCONTINUE(ierr);
      //VecView(y,PETSC_VIEWER_STDOUT_WORLD);
      ierr = VecNorm(y,NORM_2,&vnorm);
      printf("Norm=%f\n",vnorm);

      printf("Vec destroy\n");
      ierr = VecDestroy(&x);CHKERRCONTINUE(ierr);
      ierr = VecDestroy(&y);CHKERRCONTINUE(ierr);
    }
     ierr = PetscThreadPoolReturn(PETSC_COMM_WORLD,&prank);CHKERRCONTINUE(ierr);
  }

  PetscFinalize();
  return 0;
}

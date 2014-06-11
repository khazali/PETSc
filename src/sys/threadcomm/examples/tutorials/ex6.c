static char help[] = "Test PetscThreadPool with OpenMP with PETSc vector routines.\n\n";

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
  PetscInt        nthreads, n=20, *indices, pstart, pend, lsize, gsize;
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
      ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,n,&x);CHKERRCONTINUE(ierr);
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
      ierr = VecGetOwnershipRange(x,&pstart,&pend);
      ierr = VecGetLocalSize(x,&lsize);
      ierr = VecGetSize(x,&gsize);
      printf("localsize=%d globalsize=%d pstart=%d pend=%d\n",lsize,gsize,pstart,pend);
      ierr = PetscThreadCommGetOwnershipRanges(PETSC_COMM_WORLD,pend-pstart,&indices);CHKERRCONTINUE(ierr);
    }
    ierr = PetscThreadPoolReturn(PETSC_COMM_WORLD,&prank);CHKERRCONTINUE(ierr);

    // Parallel threaded user code
    start = indices[trank];
    end = indices[trank+1];
    printf("trank=%d start=%d end=%d\n",trank,start,end);
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

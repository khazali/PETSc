static char help[] = "Test Splitting PetscThreadPool.\n\n";

#include <petscvec.h>
#include <omp.h>
#include <petscthreadcomm.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec             x1, y1, x2, y2;
  PetscErrorCode  ierr;
  PetscInt        prank, nthreads, n=20;
  PetscScalar     alpha, vnorm;
  PetscThreadComm tcomm;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(PETSC_COMM_WORLD,&nthreads);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"nthreads=%d\n",nthreads);

  //Split ThreadComm into two comms of half size
  ierr = PetscCommGetThreadComm(PETSC_COMM_WORLD,&tcomm);CHKERRCONTINUE(ierr);
  PetscInt cthreads = nthreads/2;
  PetscInt commsize[2];
  commsize[0] = cthreads;
  commsize[1] = cthreads;
  PetscThreadComm splitcomms[2];
  ierr = PetscThreadCommSplit(tcomm,2,commsize,splitcomms);

  #pragma omp parallel num_threads(nthreads) default(shared) private(ierr)
  {
    int trank = omp_get_thread_num();

    // Threads in first comm do this work
    if(trank < cthreads) {
      ierr = PetscThreadPoolJoin(PETSC_COMM_WORLD,trank,&prank,splitcomms[0]);CHKERRCONTINUE(ierr);
      if(prank>=0) {
        ierr = VecCreate(PETSC_COMM_WORLD,&x1);CHKERRCONTINUE(ierr);
        ierr = VecSetSizes(x1,PETSC_DECIDE,n);CHKERRCONTINUE(ierr);
        ierr = VecSetFromOptions(x1);CHKERRCONTINUE(ierr);
        ierr = VecDuplicate(x1,&y1);CHKERRCONTINUE(ierr);

        alpha = 4.0;
        ierr = VecSet(x1,2.0);CHKERRCONTINUE(ierr);
        ierr = VecSet(y1,3.0);CHKERRCONTINUE(ierr);
        ierr = VecAXPY(y1,alpha,x1);CHKERRCONTINUE(ierr);
        ierr = VecNorm(y1,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"ThreadComm1 Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

        ierr = VecDestroy(&x1);CHKERRCONTINUE(ierr);
        ierr = VecDestroy(&y1);CHKERRCONTINUE(ierr);
      }
      ierr = PetscThreadPoolReturn(PETSC_COMM_WORLD,&prank);CHKERRCONTINUE(ierr);
    }
    // Threads in second comm do this work
    else
    {
      ierr = PetscThreadPoolJoin(PETSC_COMM_WORLD,trank,&prank,splitcomms[1]);CHKERRCONTINUE(ierr);
      if(prank>=0) {
        ierr = VecCreate(PETSC_COMM_WORLD,&x2);CHKERRCONTINUE(ierr);
        ierr = VecSetSizes(x2,PETSC_DECIDE,n);CHKERRCONTINUE(ierr);
        ierr = VecSetFromOptions(x2);CHKERRCONTINUE(ierr);
        ierr = VecDuplicate(x2,&y2);CHKERRCONTINUE(ierr);

        alpha = 8.0;
        ierr = VecSet(x2,3.0);CHKERRCONTINUE(ierr);
        ierr = VecSet(y2,6.0);CHKERRCONTINUE(ierr);
        ierr = VecAXPY(y2,alpha,x2);CHKERRCONTINUE(ierr);
        ierr = VecNorm(y2,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"ThreadComm1 Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

        ierr = VecDestroy(&x2);CHKERRCONTINUE(ierr);
        ierr = VecDestroy(&y2);CHKERRCONTINUE(ierr);
      }
      ierr = PetscThreadPoolReturn(PETSC_COMM_WORLD,&prank);CHKERRCONTINUE(ierr);
    }
  }

  // Destroy comms
  PetscThreadCommDestroy(&splitcomms[0]);
  PetscThreadCommDestroy(&splitcomms[1]);

  PetscFinalize();
  return 0;
}

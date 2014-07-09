static char help[] = "Test threadcomm with type=OpenMP,model=user with multiple threadcomms with PETSc vector routines.\n\n";

#include <petscvec.h>
#include <omp.h>
#include <petscthreadcomm.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  PetscInt        i, nthreads, n=20, ncomms=1;
  PetscScalar     alpha=3.0;
  MPI_Comm        comm, *splitcomms;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-ncomms",&ncomms,NULL);CHKERRQ(ierr);

  ierr = PetscThreadCommCreate(PETSC_COMM_WORLD,PETSC_DECIDE,&comm);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(comm,&nthreads);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"nthreads=%d\n",nthreads);CHKERRQ(ierr);

  printf("Creating splitcomm with %d comms\n",ncomms);
  ierr = PetscThreadCommSplitEvenly(comm,ncomms,&splitcomms);CHKERRQ(ierr);

  #pragma omp parallel num_threads(nthreads) default(shared) private(ierr)
  {
    Vec x, y, a, b;
    PetscInt prank;
    PetscScalar vnorm=0.0, xval, yval;
    int trank = omp_get_thread_num();

    // User gives threads to PETSc for threaded PETSc work
    printf("trank=%d joining pool\n",trank);
    ierr = PetscThreadPoolJoin(splitcomms,ncomms,trank,&prank);CHKERRCONTINUE(ierr);
    ierr = PetscPrintf(comm,"trank=%d joined pool prank=%d\n",trank,prank);CHKERRCONTINUE(ierr);
    if(prank>=0) {
      ierr = PetscPrintf(splitcomms[prank],"Creating vecs\n");CHKERRCONTINUE(ierr);
      ierr = VecCreateMPI(splitcomms[prank],PETSC_DECIDE,n,&x);CHKERRCONTINUE(ierr);
      ierr = VecSetFromOptions(x);CHKERRCONTINUE(ierr);
      ierr = VecDuplicate(x,&y);CHKERRCONTINUE(ierr);

      xval = 1.0*(prank+1.0);
      yval = 2.0*(prank+1.0);
      ierr = VecSet(x,xval);CHKERRCONTINUE(ierr);
      ierr = VecSet(y,yval);CHKERRCONTINUE(ierr);
      ierr = VecAXPY(y,alpha,x);CHKERRCONTINUE(ierr);
      ierr = VecNorm(y,NORM_1,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(splitcomms[prank],"Comm=%d Norm=%f\n",prank,vnorm);CHKERRCONTINUE(ierr);

      ierr = VecDestroy(&x);CHKERRCONTINUE(ierr);
      ierr = VecDestroy(&y);CHKERRCONTINUE(ierr);
    }
    ierr = PetscThreadPoolReturn(splitcomms,ncomms,trank,&prank);CHKERRCONTINUE(ierr);

    // User gives threads to PETSc for threaded PETSc work
    printf("Running single comm test trank=%d\n",trank);
    ierr = PetscThreadPoolJoin(&comm,1,trank,&prank);CHKERRCONTINUE(ierr);
    ierr = PetscPrintf(comm,"trank=%d joined pool prank=%d\n",trank,prank);CHKERRCONTINUE(ierr);
    if(prank>=0) {
      ierr = VecCreateMPI(comm,PETSC_DECIDE,n,&a);CHKERRCONTINUE(ierr);
      ierr = VecSetFromOptions(a);CHKERRCONTINUE(ierr);
      ierr = VecDuplicate(a,&b);CHKERRCONTINUE(ierr);

      ierr = VecSet(a,2.0);CHKERRCONTINUE(ierr);
      ierr = VecSet(b,3.0);CHKERRCONTINUE(ierr);
      ierr = VecAXPY(b,alpha,a);CHKERRCONTINUE(ierr);
      ierr = VecNorm(b,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(comm,"Single comm test Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

      ierr = VecDestroy(&a);CHKERRCONTINUE(ierr);
      ierr = VecDestroy(&b);CHKERRCONTINUE(ierr);
    }
    ierr = PetscThreadPoolReturn(&comm,1,trank,&prank);CHKERRCONTINUE(ierr);
  }

  // Destroy Threadcomms
  ierr = PetscPrintf(comm,"Destory and Finalize\n");CHKERRQ(ierr);
  for(i=0; i<ncomms; i++) {
    ierr = PetscCommDestroy(&splitcomms[i]);CHKERRQ(ierr);
  }
  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

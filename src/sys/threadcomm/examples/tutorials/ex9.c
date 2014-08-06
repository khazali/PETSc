static char help[] = "Test threadcomm with OpenMP thread type and user threading model with multiple threadcomms with PETSc vector routines.\n\n";

#include <petscvec.h>
#include <omp.h>
#include <petscthreadcomm.h>

/*
   Example run command: ./ex9 -n 100000 -threadcomm_type openmp -threadcomm_model user -threadcomm_nthreads $nthreads -ncomms $ncomms
*/

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,nthreads,nthreads2,ntcthreads,n=20,ncomms=1,*granks;
  PetscScalar    alpha=3.0;
  MPI_Comm       comm,shcomm,*splitcomms,*multcomms;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ncomms",&ncomms,PETSC_NULL);CHKERRQ(ierr);

  /* Create threadcomm using all threads */
  ierr = PetscThreadCommCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_NULL,&comm);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(comm,&nthreads);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Running tests with %d threads\n",nthreads);CHKERRQ(ierr);

  /* Create second threadcomm using every other thread */
  nthreads2 = ceil((PetscScalar)nthreads/2.0);
  ierr = PetscMalloc1(nthreads2,&granks);CHKERRQ(ierr);
  for(i=0; i<nthreads2; i++) granks[i] = i*2;
  ierr = PetscThreadCommCreateShare(comm,nthreads2,granks,&shcomm);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(shcomm,&ntcthreads);CHKERRQ(ierr);
  ierr = PetscPrintf(shcomm,"Creating shared comm with %d threads\n",ntcthreads);CHKERRQ(ierr);

  /* Create split threadcomm from comm */
  ierr = PetscPrintf(comm,"Creating splitcomm with %d comms\n",ncomms);CHKERRQ(ierr);
  ierr = PetscThreadCommSplit(comm,ncomms,PETSC_NULL,PETSC_NULL,&splitcomms);CHKERRQ(ierr);

  /* Create multiple threadcomms */
  ierr = PetscPrintf(comm,"Creating multcomm with %d comms\n",ncomms);CHKERRQ(ierr);
  ierr = PetscThreadCommCreateMultiple(PETSC_COMM_WORLD,ncomms,nthreads,PETSC_NULL,PETSC_NULL,&multcomms);

  #pragma omp parallel num_threads(nthreads) default(shared) private(ierr)
  {
    Vec         x,y,a,b;
    PetscInt    commrank;
    PetscScalar vnorm = 0.0, xval, yval;
    PetscInt    trank = omp_get_thread_num();

    ierr = PetscThreadInitialize();CHKERRCONTINUE(ierr);

    /* User gives threads to PETSc for threaded multiple comm PETSc work */
    ierr = PetscPrintf(comm,"\nRunning Multcomm test\n");CHKERRCONTINUE(ierr);
    ierr = PetscThreadCommJoinMultComms(multcomms,ncomms,trank,&commrank);CHKERRCONTINUE(ierr);
    if(commrank>=0) {
      ierr = VecCreateMPI(multcomms[commrank],PETSC_DECIDE,n,&x);CHKERRCONTINUE(ierr);
      ierr = VecSetFromOptions(x);CHKERRCONTINUE(ierr);
      ierr = VecDuplicate(x,&y);CHKERRCONTINUE(ierr);

      xval = 1.0*(commrank+1.0);
      yval = 2.0*(commrank+1.0);
      ierr = VecSet(x,xval);CHKERRCONTINUE(ierr);
      ierr = VecSet(y,yval);CHKERRCONTINUE(ierr);
      ierr = VecAXPY(y,alpha,x);CHKERRCONTINUE(ierr);
      ierr = VecNorm(y,NORM_1,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(multcomms[commrank],"Multcomm test commrank=%d norm=%f\n",commrank,vnorm);CHKERRCONTINUE(ierr);

      ierr = VecDestroy(&x);CHKERRCONTINUE(ierr);
      ierr = VecDestroy(&y);CHKERRCONTINUE(ierr);
    }
    ierr = PetscThreadCommReturnMultComms(multcomms,ncomms,trank,&commrank);CHKERRCONTINUE(ierr);

    #pragma omp barrier

    /* User gives threads to PETSc for threaded split comm PETSc work */
    ierr = PetscPrintf(comm,"\nRunning Splitcomm test\n",trank,commrank);CHKERRCONTINUE(ierr);
    ierr = PetscThreadCommJoinMultComms(splitcomms,ncomms,trank,&commrank);CHKERRCONTINUE(ierr);
    if(commrank>=0) {
      ierr = VecCreateMPI(splitcomms[commrank],PETSC_DECIDE,n,&x);CHKERRCONTINUE(ierr);
      ierr = VecSetFromOptions(x);CHKERRCONTINUE(ierr);
      ierr = VecDuplicate(x,&y);CHKERRCONTINUE(ierr);

      xval = 1.0*(commrank+1.0);
      yval = 2.0*(commrank+1.0);
      ierr = VecSet(x,xval);CHKERRCONTINUE(ierr);
      ierr = VecSet(y,yval);CHKERRCONTINUE(ierr);
      ierr = VecAXPY(y,alpha,x);CHKERRCONTINUE(ierr);
      ierr = VecNorm(y,NORM_1,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(splitcomms[commrank],"Splitcomm test commrank=%d norm=%f\n",commrank,vnorm);CHKERRCONTINUE(ierr);

      ierr = VecDestroy(&x);CHKERRCONTINUE(ierr);
      ierr = VecDestroy(&y);CHKERRCONTINUE(ierr);
    }
    ierr = PetscThreadCommReturnMultComms(splitcomms,ncomms,trank,&commrank);CHKERRCONTINUE(ierr);

    #pragma omp barrier

    /* User gives threads to PETSc for threaded shared comm PETSc work */
    ierr = PetscPrintf(comm,"\nRunning shared comm test\n");
    ierr = PetscThreadCommJoinMultComms(&shcomm,1,trank,&commrank);CHKERRCONTINUE(ierr);
    if(commrank>=0) {
      ierr = VecCreateMPI(shcomm,PETSC_DECIDE,n,&a);CHKERRCONTINUE(ierr);
      ierr = VecSetFromOptions(a);CHKERRCONTINUE(ierr);
      ierr = VecDuplicate(a,&b);CHKERRCONTINUE(ierr);

      ierr = VecSet(a,2.0);CHKERRCONTINUE(ierr);
      ierr = VecSet(b,3.0);CHKERRCONTINUE(ierr);
      ierr = VecAXPY(b,alpha,a);CHKERRCONTINUE(ierr);
      ierr = VecNorm(b,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(shcomm,"Shared comm test norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

      ierr = VecDestroy(&a);CHKERRCONTINUE(ierr);
      ierr = VecDestroy(&b);CHKERRCONTINUE(ierr);
    }
    ierr = PetscThreadCommReturnMultComms(&shcomm,1,trank,&commrank);CHKERRCONTINUE(ierr);

     #pragma omp barrier

    /* User gives threads to PETSc for threaded single comm PETSc work */
    ierr = PetscPrintf(comm,"\nRunning single comm test\n");
    ierr = PetscThreadCommJoinMultComms(&comm,1,trank,&commrank);CHKERRCONTINUE(ierr);
    if(commrank>=0) {
      ierr = VecCreateMPI(comm,PETSC_DECIDE,n,&a);CHKERRCONTINUE(ierr);
      ierr = VecSetFromOptions(a);CHKERRCONTINUE(ierr);
      ierr = VecDuplicate(a,&b);CHKERRCONTINUE(ierr);

      ierr = VecSet(a,2.0);CHKERRCONTINUE(ierr);
      ierr = VecSet(b,3.0);CHKERRCONTINUE(ierr);
      ierr = VecAXPY(b,alpha,a);CHKERRCONTINUE(ierr);
      ierr = VecNorm(b,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(comm,"Single comm test norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

      ierr = VecDestroy(&a);CHKERRCONTINUE(ierr);
      ierr = VecDestroy(&b);CHKERRCONTINUE(ierr);
    }
     ierr = PetscThreadCommReturnMultComms(&comm,1,trank,&commrank);CHKERRCONTINUE(ierr);

     ierr = PetscThreadFinalize();CHKERRCONTINUE(ierr);
  }

  /* Destroy Threadcomms */
  ierr = PetscPrintf(comm,"Destory and Finalize\n");CHKERRQ(ierr);
  for(i=0; i<ncomms; i++) {
    ierr = PetscCommDestroy(&splitcomms[i]);CHKERRQ(ierr);
    ierr = PetscCommDestroy(&multcomms[i]);CHKERRQ(ierr);
  }
  ierr = PetscCommDestroy(&shcomm);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);
  ierr = PetscFree(splitcomms);CHKERRQ(ierr);
  ierr = PetscFree(multcomms);CHKERRQ(ierr);
  ierr = PetscFree(granks);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

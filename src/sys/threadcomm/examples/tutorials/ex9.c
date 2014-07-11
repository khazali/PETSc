static char help[] = "Test threadcomm with type=OpenMP,model=user with multiple threadcomms with PETSc vector routines.\n\n";

#include <petscvec.h>
#include <omp.h>
#include <petscthreadcomm.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  PetscInt        i, nthreads, nthreads2, ntcthreads, n=20, ncomms=1, *granks;
  PetscScalar     alpha=3.0;
  MPI_Comm        comm, shcomm, *splitcomms, *multcomms, *mcomms;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ncomms",&ncomms,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscThreadCommCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_NULL,&comm);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(comm,&nthreads);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"nthreads=%d\n",nthreads);CHKERRQ(ierr);

  // Create second threadcomm using every other thread
  nthreads2 = ceil((PetscScalar)nthreads/2.0);
  ierr = PetscMalloc1(nthreads2,&granks);CHKERRQ(ierr);
  for(i=0; i<nthreads2; i++) {
    granks[i] = i*2;
  }
  ierr = PetscThreadCommCreateShare(comm,nthreads2,granks,&shcomm);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(shcomm,&ntcthreads);CHKERRQ(ierr);
  ierr = PetscPrintf(shcomm,"Created shared comm with %d threads\n",ntcthreads);CHKERRQ(ierr);

  printf("Creating splitcomm with %d comms\n",ncomms);
  ierr = PetscThreadCommSplitEvenly(comm,ncomms,&splitcomms);CHKERRQ(ierr);

  ierr = PetscThreadCommCreateMultiple(PETSC_COMM_WORLD,ncomms,nthreads,PETSC_NULL,PETSC_NULL,&multcomms);
  ierr = PetscThreadCommGetNThreads(multcomms[0],&ntcthreads);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"multcomm1 ntcthreads=%d\n",ntcthreads);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(multcomms[1],&ntcthreads);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"multcomm2 ntcthreads=%d\n",ntcthreads);CHKERRQ(ierr);

  mcomms = multcomms;

  #pragma omp parallel num_threads(nthreads) default(shared) private(ierr)
  {
    Vec x, y, a, b;
    Vec *xvecs, *yvecs;
    PetscInt commrank;
    PetscScalar vnorm=0.0, xval, yval;
    int trank = omp_get_thread_num();

    ierr = PetscMalloc1(ncomms,&xvecs);CHKERRCONTINUE(ierr);
    ierr = PetscMalloc1(ncomms,&yvecs);CHKERRCONTINUE(ierr);

    // User gives threads to PETSc for threaded multiple comm PETSc work
    printf("trank=%d joining comm\n",trank);
    mcomms = multcomms;
    ierr = PetscThreadCommJoin(mcomms,ncomms,trank,&commrank);CHKERRCONTINUE(ierr);
    if(commrank>=0) {
      ierr = PetscPrintf(comm,"Computing with trank=%d commrank=%d\n",trank,commrank);CHKERRCONTINUE(ierr);
      x = xvecs[commrank];
      y = yvecs[commrank];
      ierr = PetscPrintf(mcomms[commrank],"Creating vecs\n");CHKERRCONTINUE(ierr);
      ierr = VecCreateMPI(mcomms[commrank],PETSC_DECIDE,n,&x);CHKERRCONTINUE(ierr);
      ierr = VecSetFromOptions(x);CHKERRCONTINUE(ierr);
      ierr = VecDuplicate(x,&y);CHKERRCONTINUE(ierr);

      xval = 1.0*(commrank+1.0);
      yval = 2.0*(commrank+1.0);
      ierr = VecSet(x,xval);CHKERRCONTINUE(ierr);
      ierr = VecSet(y,yval);CHKERRCONTINUE(ierr);
      ierr = VecAXPY(y,alpha,x);CHKERRCONTINUE(ierr);
      ierr = VecNorm(y,NORM_1,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(mcomms[commrank],"Multcomm=%d Norm=%f\n",commrank,vnorm);CHKERRCONTINUE(ierr);

      ierr = VecDestroy(&x);CHKERRCONTINUE(ierr);
      ierr = VecDestroy(&y);CHKERRCONTINUE(ierr);
    }
    ierr = PetscThreadCommReturn(mcomms,ncomms,trank,&commrank);CHKERRCONTINUE(ierr);

    // User gives threads to PETSc for threaded split comm PETSc work
    printf("trank=%d joining comm\n",trank);
    mcomms = splitcomms;
    ierr = PetscThreadCommJoin(mcomms,ncomms,trank,&commrank);CHKERRCONTINUE(ierr);
    if(commrank>=0) {
      ierr = PetscPrintf(comm,"Computing with trank=%d commrank=%d\n",trank,commrank);CHKERRCONTINUE(ierr);
      x = xvecs[commrank];
      y = yvecs[commrank];
      ierr = PetscPrintf(mcomms[commrank],"Creating vecs\n");CHKERRCONTINUE(ierr);
      ierr = VecCreateMPI(mcomms[commrank],PETSC_DECIDE,n,&x);CHKERRCONTINUE(ierr);
      ierr = VecSetFromOptions(x);CHKERRCONTINUE(ierr);
      ierr = VecDuplicate(x,&y);CHKERRCONTINUE(ierr);

      xval = 1.0*(commrank+1.0);
      yval = 2.0*(commrank+1.0);
      ierr = VecSet(x,xval);CHKERRCONTINUE(ierr);
      ierr = VecSet(y,yval);CHKERRCONTINUE(ierr);
      ierr = VecAXPY(y,alpha,x);CHKERRCONTINUE(ierr);
      ierr = VecNorm(y,NORM_1,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(mcomms[commrank],"Splitcomm=%d Norm=%f\n",commrank,vnorm);CHKERRCONTINUE(ierr);

      ierr = VecDestroy(&x);CHKERRCONTINUE(ierr);
      ierr = VecDestroy(&y);CHKERRCONTINUE(ierr);
    }
    ierr = PetscThreadCommReturn(mcomms,ncomms,trank,&commrank);CHKERRCONTINUE(ierr);

    ierr = PetscFree(xvecs);CHKERRCONTINUE(ierr);
    ierr = PetscFree(yvecs);CHKERRCONTINUE(ierr);

    // User gives threads to PETSc for threaded shared comm PETSc work
    /*ierr = PetscPrintf(shcomm,"\n\n\nRunning shared comm test trank=%d\n",trank);
    ierr = PetscThreadCommJoin(&shcomm,1,trank,&commrank);CHKERRCONTINUE(ierr);
    ierr = PetscPrintf(shcomm,"trank=%d joined comm commrank=%d\n",trank,commrank);CHKERRCONTINUE(ierr);
    if(commrank>=0) {
      ierr = VecCreateMPI(shcomm,PETSC_DECIDE,n,&a);CHKERRCONTINUE(ierr);
      ierr = VecSetFromOptions(a);CHKERRCONTINUE(ierr);
      ierr = VecDuplicate(a,&b);CHKERRCONTINUE(ierr);

      ierr = VecSet(a,2.0);CHKERRCONTINUE(ierr);
      ierr = VecSet(b,3.0);CHKERRCONTINUE(ierr);
      ierr = VecAXPY(b,alpha,a);CHKERRCONTINUE(ierr);
      ierr = VecNorm(b,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(shcomm,"Single comm test Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);

      ierr = VecDestroy(&a);CHKERRCONTINUE(ierr);
      ierr = VecDestroy(&b);CHKERRCONTINUE(ierr);
    }
    ierr = PetscThreadCommReturn(&shcomm,1,trank,&commrank);CHKERRCONTINUE(ierr);

    // User gives threads to PETSc for threaded single comm PETSc work
    ierr = PetscPrintf(comm,"\n\n\nRunning single comm test trank=%d\n",trank);
    ierr = PetscThreadCommJoin(&comm,1,trank,&commrank);CHKERRCONTINUE(ierr);
    ierr = PetscPrintf(comm,"trank=%d joined comm commrank=%d\n",trank,commrank);CHKERRCONTINUE(ierr);
    if(commrank>=0) {
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
     ierr = PetscThreadCommReturn(&comm,1,trank,&commrank);CHKERRCONTINUE(ierr);*/
  }

  // Destroy Threadcomms
  ierr = PetscPrintf(comm,"Destory and Finalize\n");CHKERRQ(ierr);
  for(i=0; i<ncomms; i++) {
    ierr = PetscCommDestroy(&splitcomms[i]);CHKERRQ(ierr);
    ierr = PetscCommDestroy(&multcomms[i]);CHKERRQ(ierr);
  }
  ierr = PetscCommDestroy(&shcomm);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

static char help[] = "Test threadcomm splitting with type=pthread, model=auto with PETSc vector routines.\n\n";

#include <petscvec.h>
#include <petscthreadcomm.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  MPI_Comm comm, *splitcomms;
  Vec *xvec, *yvec;
  PetscScalar *vnorm, alpha=3.0, xval, yval;
  PetscInt i, n=20, nthreads, ncomms=1;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-ncomms",&ncomms,NULL);CHKERRQ(ierr);

  // Create MPI_Comm and ThreadComm from PETSC_COMM_WORLD
  // Create worker threads in PETSc, master thread returns
  printf("Creating threadcomm\n");
  ierr = PetscThreadCommCreate(PETSC_COMM_WORLD,PETSC_DECIDE,&comm);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(comm,&nthreads);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Created comm with %d threads\n",nthreads);CHKERRQ(ierr);

  // Split threadcomm into multiple smaller threadcomms
  /*PetscInt splitsize, *commsizes
  ierr = PetscMalloc1(ncomms,&commsizes);CHKERRQ(ierr);
  splitsize = nthreads/ncomms;
  for(i=0; i<ncomms; i++) {
    commsizes[i] = splitsize;
  }
  PetscThreadCommSplit(comm,ncomms,commsizes,&splitcomms1);*/

  // Split threads evenly among comms
  printf("Creating splitcomm with %d comms\n",ncomms);
  ierr = PetscThreadCommSplitEvenly(comm,ncomms,&splitcomms);CHKERRQ(ierr);

  ierr = PetscPrintf(comm,"Creating split comms with %d comms\n",ncomms);

  for(i=0; i<ncomms; i++) {
    ierr = PetscThreadCommGetNThreads(splitcomms[i],&nthreads);CHKERRQ(ierr);
    ierr = PetscPrintf(splitcomms[i],"Created splitcomm %d with %d threads\n",i,nthreads);CHKERRQ(ierr);
  }

  // Allocate arrays
  ierr = PetscMalloc1(ncomms,&xvec);CHKERRQ(ierr);
  ierr = PetscMalloc1(ncomms,&yvec);CHKERRQ(ierr);
  ierr = PetscMalloc1(ncomms,&vnorm);CHKERRQ(ierr);

  // Work on multiple vector at once.
  for(i=0; i<ncomms; i++) {
    ierr = VecCreate(splitcomms[i],&xvec[i]);CHKERRQ(ierr);
    ierr = VecSetSizes(xvec[i],PETSC_DECIDE,n);CHKERRQ(ierr);
    ierr = VecSetFromOptions(xvec[i]);CHKERRQ(ierr);
    ierr = VecDuplicate(xvec[i],&yvec[i]);CHKERRQ(ierr);

    xval = 1.0*(i+1.0);
    yval = 2.0*(i+1.0);
    printf("Comm=%d xval=%f yval=%f axpy=%f norm=%f\n",i,xval,yval,alpha*xval+yval,(alpha*xval+yval)*n);
    ierr = VecSet(xvec[i],xval);CHKERRQ(ierr);
    ierr = VecSet(yvec[i],yval);CHKERRQ(ierr);
    ierr = VecAXPY(yvec[i],alpha,xvec[i]);CHKERRQ(ierr);
  }

  printf("Main thread at barrier after distributing work\n");
  PetscThreadCommBarrier(comm);

  // Output final results
  for(i=0; i<ncomms; i++) {
    ierr = VecNorm(yvec[i],NORM_1,&vnorm[i]);CHKERRQ(ierr);
    ierr = PetscPrintf(splitcomms[i],"Splitcomm %d computed vnorm=%lf\n",i,vnorm[i]);
  }

  PetscThreadCommBarrier(comm);

  // Destroy MPI_Comms/threadcomms
  for(i=0; i<ncomms; i++) {
    ierr = VecDestroy(&xvec[i]);CHKERRQ(ierr);
    ierr = VecDestroy(&yvec[i]);CHKERRQ(ierr);
    ierr = PetscCommDestroy(&splitcomms[i]);CHKERRQ(ierr);
  }
  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

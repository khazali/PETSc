static char help[] = "Test PetscThreadPool with pthreads with PETSc vector routines.\n\n";

#include <petscvec.h>
#include <petscthreadcomm.h>

// Threaded user function
PetscErrorCode user_func(PetscInt trank,Vec y, MPI_Comm *comm) {

  PetscInt i, start, end, *indices, lsize;
  PetscScalar *ay;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("In user func trank=%d\n",trank);

  // Get data for local work
  ierr = VecGetArray(y,&ay);CHKERRCONTINUE(ierr);
  ierr = VecGetLocalSize(y,&lsize);CHKERRCONTINUE(ierr);
  ierr = PetscThreadCommGetOwnershipRanges(*comm,lsize,&indices);CHKERRCONTINUE(ierr);

  // Parallel threaded user code
  start = indices[trank];
  end = indices[trank+1];
  ierr = PetscPrintf(*comm,"trank=%d start=%d end=%d\n",trank,start,end);CHKERRCONTINUE(ierr);
  for(i=start; i<end; i++) {
    ay[i] = ay[i]*ay[i];
  }

  // Restore vector
  ierr = VecRestoreArray(y,&ay);CHKERRCONTINUE(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec x, y;
  PetscErrorCode  ierr;
  PetscInt i, n=20, nthreads, nthreads2, *granks;
  PetscScalar vnorm,alpha=3.0;
  MPI_Comm comm,comm1,comm2;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-nthreads",&nthreads,NULL);CHKERRQ(ierr);

  // Create MPI_Comm and ThreadComm from PETSC_COMM_WORLD
  // Create worker threads in PETSc, master thread returns
  ierr = PetscThreadCommCreate(PETSC_COMM_WORLD,nthreads,PETSC_TRUE,&comm1);

  // Create second ThreadComm using every other thread
  nthreads2 = ceil((PetscScalar)nthreads/2.0);
  ierr = PetscMalloc1(nthreads2,&granks);CHKERRQ(ierr);
  for(i=0; i<nthreads2; i++) {
    granks[i] = i*2;
  }
  ierr = PetscThreadCommCreateShare(comm1,nthreads2,granks,&comm2);
  comm = comm2;

  // Create two vectors on MPIComm/ThreadComm
  ierr = VecCreate(comm,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);

  // Run PETSc code
  ierr = VecSet(x,2.0);CHKERRQ(ierr);
  ierr = VecSet(y,3.0);CHKERRQ(ierr);
  ierr = VecAXPY(y,alpha,x);CHKERRQ(ierr);
  //VecView(y,PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecNorm(y,NORM_2,&vnorm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Norm=%f\n",vnorm);CHKERRQ(ierr);

  // Run User code
  ierr = PetscThreadCommRunKernel2(comm,(PetscThreadKernel)user_func,y,&comm);CHKERRQ(ierr);

  ierr = VecScale(y,2.0);CHKERRQ(ierr);
  ierr = VecAXPY(y,alpha,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&vnorm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Norm=%f\n",vnorm);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  printf("Calling PetscFinalize\n");
  PetscFinalize();

  return 0;
}

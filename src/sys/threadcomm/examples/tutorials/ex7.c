static char help[] = "Test threadcomm with PThreads and auto threading model with PETSc vector routines.\n\n";

#include <petscvec.h>
#include <petscthreadcomm.h>

/*
   Example run command: ./ex7 -n 1000 -threadcomm_type pthread -threadcomm_model auto -threadcomm_nthreads $nthreads
*/

/* Threaded user function */
PetscErrorCode user_func(PetscInt trank,Vec y, MPI_Comm *comm) {

  PetscInt i, start, end, *indices, lsize;
  PetscScalar *ay;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Get data for local work */
  ierr = VecGetArray(y,&ay);CHKERRCONTINUE(ierr);
  ierr = VecGetLocalSize(y,&lsize);CHKERRCONTINUE(ierr);
  ierr = PetscThreadCommGetOwnershipRanges(*comm,lsize,&indices);CHKERRCONTINUE(ierr);

  /* Parallel threaded user code */
  start = indices[trank];
  end = indices[trank+1];
  for(i=start; i<end; i++) ay[i] = ay[i]*ay[i];

  /* Restore vector */
  ierr = VecRestoreArray(y,&ay);CHKERRCONTINUE(ierr);
  ierr = PetscFree(indices);CHKERRCONTINUE(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec            x,y;
  PetscErrorCode ierr;
  PetscInt       i,n=20,nthreads,nthreads2,*granks,*affinities;
  PetscScalar    vnorm,alpha=3.0;
  MPI_Comm       comm,comm1,comm2,comm_a,comm_b;
  PetscInt       ntcthreads1,ntcthreads2;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

  /* Create MPI_Comm and ThreadComm from PETSC_COMM_WORLD
     Create worker threads in PETSc, master thread returns */
  ierr = PetscThreadCommCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_NULL,&comm1);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(comm1,&nthreads);CHKERRQ(ierr);
  ierr = PetscPrintf(comm1,"Created comm1 with %d threads\n",nthreads);CHKERRQ(ierr);

  /* Create second threadcomm using every other thread */
  nthreads2 = ceil((PetscScalar)nthreads/2.0);
  ierr = PetscMalloc1(nthreads2,&granks);CHKERRQ(ierr);
  for(i=0; i<nthreads2; i++) {
    granks[i] = i*2;
  }
  ierr = PetscThreadCommCreateShare(comm1,nthreads2,granks,&comm2);CHKERRQ(ierr);

  ierr = PetscThreadCommGetNThreads(comm1,&ntcthreads1);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(comm2,&ntcthreads2);CHKERRQ(ierr);
  ierr = PetscPrintf(comm2,"Comm1 has %d threads, created comm2 with %d threads\n",ntcthreads1,ntcthreads2);CHKERRQ(ierr);

  /* Set thread affinities for this threadcomm specifically */
  PetscMalloc1(nthreads,&affinities);
  for(i=0; i<nthreads; i++) affinities[i] = nthreads + i;
  /* Attach threadcomm to PETSC_COMM_WORLD */
  ierr = PetscThreadCommCreateAttach(PETSC_COMM_WORLD,nthreads,affinities);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(PETSC_COMM_WORLD,&ntcthreads1);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"PETSC_COMM_WORLD has %d threads\n",ntcthreads1);CHKERRQ(ierr);

  /* Run single comm test 1 */
  comm = comm1;
  ierr = PetscPrintf(comm,"\nRunning test with first single comm\n");
  /* Create two vectors on MPIComm/ThreadComm */
  ierr = VecCreate(comm,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);

  /* Run PETSc code */
  for(i=0; i<n; i++) {
    VecSetValue(x,i,i*1.0,INSERT_VALUES);
    VecSetValue(y,i,i*2.0,INSERT_VALUES);
  }
  ierr = VecAXPY(y,alpha,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&vnorm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Test1 Part1 Norm=%f\n",vnorm);CHKERRQ(ierr);

  /* Run User code */
  ierr = PetscThreadCommRunKernel2(comm,(PetscThreadKernel)user_func,y,&comm);CHKERRQ(ierr);

  ierr = VecScale(y,2.0);CHKERRQ(ierr);
  ierr = VecAXPY(y,alpha,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&vnorm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Test1 Part2 Norm=%f\n",vnorm);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);

  /* Run single comm test 2 */
  comm = comm2;
  ierr = PetscPrintf(comm,"\nRunning test with second single comm\n");
  /* Create two vectors on MPIComm/ThreadComm */
  ierr = VecCreate(comm,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);

  /* Run PETSc code */
  ierr = VecSet(x,2.0);CHKERRQ(ierr);
  ierr = VecSet(y,3.0);CHKERRQ(ierr);
  ierr = VecAXPY(y,alpha,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&vnorm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Test2 Part1 Norm=%f\n",vnorm);CHKERRQ(ierr);

  /* Run User code */
  ierr = PetscThreadCommRunKernel2(comm,(PetscThreadKernel)user_func,y,&comm);CHKERRQ(ierr);

  ierr = VecScale(y,2.0);CHKERRQ(ierr);
  ierr = VecAXPY(y,alpha,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&vnorm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Test2 Part2 Norm=%f\n",vnorm);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);

  /* Run test using vecs created on different comms in same petsc operation */
  comm_a = comm2;
  comm_b = comm1;

  ierr = PetscPrintf(comm_a,"\nTesting computations with vectors on different comms\n");
  ierr = VecCreate(comm_a,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSet(x,2.0);CHKERRQ(ierr);

  ierr = VecCreate(comm_b,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);
  ierr = VecSet(y,3.0);CHKERRQ(ierr);

  ierr = VecAXPY(y,alpha,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&vnorm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm_a,"Different Comms Norm=%f\n",vnorm);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);

  ierr = PetscFree(affinities);CHKERRQ(ierr);
  ierr = PetscFree(granks);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&comm1);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&comm2);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

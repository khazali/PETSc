static char help[] = "Test Splitting PetscThreadComm.\n\n";

#include <petscvec.h>
#include <petscthreadcomm.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  PetscInt i, n=20, nthreads;
  MPI_Comm comm1;
  PetscInt ntcthreads;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-nthreads",&nthreads,NULL);CHKERRQ(ierr);

  // Create MPI_Comm and ThreadComm from PETSC_COMM_WORLD
  // Create worker threads in PETSc, master thread returns
  ierr = PetscThreadCommCreate(PETSC_COMM_WORLD,nthreads,PETSC_TRUE,&comm1);
  PetscThreadCommGetNThreads(comm1,&ntcthreads);
  printf("After creating comm1, comm1 has %d threads\n\n\n",ntcthreads);

  printf("\n\nCreating split comms\n");
  // Create a set of split threadcomms that each use some threads
  MPI_Comm *splitcomms;
  int ncomms = 4;
  int *commsizes;
  PetscMalloc1(ncomms,&commsizes);
  int splitsize = floor((PetscScalar)nthreads/(PetscScalar)ncomms);
  for(i=0; i<ncomms; i++) {
    commsizes[i] = splitsize;
  }
  PetscThreadCommSplit(comm1,ncomms,commsizes,&splitcomms);

  for(i=0; i<ncomms; i++) {
    PetscThreadCommGetNThreads(splitcomms[i],&ntcthreads);
    printf("After creating splitcomms, comm[%d] has %d threads\n",i,ntcthreads);
  }

  for(i=0; i<ncomms; i++) {
    PetscCommDestroy(&splitcomms[i]);
  }
  PetscFree(splitcomms);

  PetscFinalize();
  return 0;
}

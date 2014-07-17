static char help[] = "PThreads code to test and debug threadcomm code.\n\n";

#include <petscvec.h>
#include <petscthreadcomm.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec x, y;
  PetscErrorCode  ierr;
  PetscInt i, n=20, nthreads, *granks, *affinities;
  PetscScalar vnorm,alpha=3.0;
  MPI_Comm comm;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-threadcomm_nthreads",&nthreads,PETSC_NULL);CHKERRQ(ierr);

  // Create MPI_Comm and ThreadComm from PETSC_COMM_WORLD
  // Create worker threads in PETSc, master thread returns
  PetscMalloc1(nthreads,&granks);
  PetscMalloc1(nthreads,&affinities);
  int j=0;
  for(i=0; i<nthreads; i+=2) {
    granks[i] = j;
    granks[i+1] = j;
    affinities[i] = j;
    affinities[i+1] = j;
    j++;
  }

  ierr = PetscThreadCommCreate(PETSC_COMM_WORLD,nthreads,granks,affinities,&comm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Created comm1 with %d threads\n",nthreads);CHKERRQ(ierr);

  // Create two vectors on MPIComm/ThreadComm
  ierr = VecCreate(comm,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);

  // Run PETSc code
  for(i=0; i<n; i++) {
    VecSetValue(x,i,i*1.0,INSERT_VALUES);
    VecSetValue(y,i,i*2.0,INSERT_VALUES);
  }
  //ierr = VecSet(x,2.0);CHKERRQ(ierr);
  //ierr = VecSet(y,3.0);CHKERRQ(ierr);
  ierr = VecAXPY(y,alpha,x);CHKERRQ(ierr);
  //VecView(y,PETSC_VIEWER_STDOUT_WORLD);
  ierr = VecNorm(y,NORM_2,&vnorm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Norm=%f\n",vnorm);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);

  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

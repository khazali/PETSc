static char help[] = "OpenMP code to test and debug threadcomm code.\n\n";

#include <petscvec.h>
#include <omp.h>
#include <petscthreadcomm.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec            x,y;
  PetscInt       nthreads=1, n=20;
  PetscScalar    alpha=3.0;
  PetscErrorCode ierr;
  MPI_Comm       comm;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscThreadCommCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_NULL,PETSC_NULL,&comm);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(comm,&nthreads);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"nthreads=%d\n",nthreads);CHKERRQ(ierr);

  ierr = PetscPrintf(comm,"Creating vecs\n");CHKERRQ(ierr);
  ierr = VecCreateMPI(comm,PETSC_DECIDE,n,&x);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);

  int mnthreads = nthreads*2;
#pragma omp parallel num_threads(mnthreads) default(shared) private(ierr)
  {
    PetscInt commrank;
    PetscScalar vnorm=0.0;
    int trank = omp_get_thread_num();

    // User gives threads to PETSc for threaded PETSc work
    if(trank<nthreads) {
      ierr = PetscThreadCommJoin(&comm,1,trank,&commrank);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(comm,"trank=%d joined comm commrank=%d\n",trank,commrank);CHKERRCONTINUE(ierr);
      if(commrank>=0) {
        ierr = VecSet(x,2.0);CHKERRCONTINUE(ierr);
        ierr = VecSet(y,3.0);CHKERRCONTINUE(ierr);
        ierr = VecAXPY(y,alpha,x);CHKERRCONTINUE(ierr);
        ierr = VecNorm(y,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
        ierr = PetscPrintf(comm,"Norm=%f\n",vnorm);CHKERRCONTINUE(ierr);
      }
      ierr = PetscThreadCommReturn(&comm,1,trank,&commrank);CHKERRCONTINUE(ierr);
    } else {
      printf("trank=%d thread not in petsc\n",trank);
      PetscInt sn = n, i;
      PetscScalar xvals[sn],yvals[sn],sum=0.0;
      printf("trank=%d sum should be %f\n",trank,1.0*trank + alpha*2.0*trank);
      for(i=0; i<sn; i++) {
        xvals[sn] = 1.0*trank;
        yvals[sn] = 2.0*trank;
        yvals[sn] = xvals[sn] + alpha*yvals[sn];
        sum += yvals[sn];
      }
      printf("trank=%d sum=%f\n",trank,sum);
    }
  }

  // Destroy Vecs
  ierr = PetscPrintf(comm,"Destory and Finalize\n");CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);

  // Destroy Threadcomms
  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

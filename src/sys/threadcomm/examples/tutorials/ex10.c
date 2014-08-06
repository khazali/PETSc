static char help[] = "Test threadcomm with OpenMP thread type and user thread model with PETSc matrix routines.\n\n";

#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>
#include <omp.h>
#include <petscthreadcomm.h>
#include <mpi.h>

/*
   Example run command: ./ex10 -n 100 -threadcomm_type openmp -threadcomm_model user -threadcomm_nthreads $nthreads
*/

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat             A;
  Vec             x, b;
  PetscErrorCode  ierr;
  PetscInt        nthreads,n=20,i,j,Ii,J;
  PetscScalar     v, vnorm;
  KSP             ksp;
  PC              pc;
  MPI_Comm        comm;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscThreadCommCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_NULL,&comm);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(comm,&nthreads);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Running test with %d threads\n",nthreads);CHKERRQ(ierr);

  /* Create vectors */
  ierr = VecCreateMPI(comm,PETSC_DECIDE,n*n,&x);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);

  #pragma omp parallel num_threads(nthreads) default(shared) private(ierr)
  {
    PetscInt           pstart,pend,lsize,gsize;
    PetscInt           commrank=0,trank = omp_get_thread_num();
    KSPConvergedReason reason;
    PetscScalar        rnorm;

    ierr = PetscThreadInitialize();CHKERRCONTINUE(ierr);

    ierr = PetscThreadCommJoinComm(comm,trank,&commrank);CHKERRCONTINUE(ierr);
    if(commrank>=0) {
      /* Set rhs */
      ierr = VecSet(b,2.0);CHKERRCONTINUE(ierr);

      /* Get ownership information */
      ierr = VecGetOwnershipRange(x,&pstart,&pend);CHKERRCONTINUE(ierr);
      ierr = VecGetLocalSize(x,&lsize);CHKERRCONTINUE(ierr);
      ierr = VecGetSize(x,&gsize);CHKERRCONTINUE(ierr);

      /* Create Matrix */
      ierr = PetscPrintf(comm,"Creating matrix\n");CHKERRCONTINUE(ierr);
      ierr = MatCreate(comm,&A);CHKERRCONTINUE(ierr);
      ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n*n,n*n);CHKERRCONTINUE(ierr);
      ierr = MatSetType(A,MATMPIAIJ);CHKERRCONTINUE(ierr);
      ierr = MatSetUp(A);CHKERRCONTINUE(ierr);

      /* Assemble matrix */
      for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
          v = -1.0;  Ii = j + n*i;
          if (i>0)   {J = Ii - n; MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);}
          if (i<n-1) {J = Ii + n; MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);}
          if (j>0)   {J = Ii - 1; MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);}
          if (j<n-1) {J = Ii + 1; MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);}
          v = 4.0; MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);
        }
      }
      ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRCONTINUE(ierr);
      ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRCONTINUE(ierr);

      /* Create linear solver and preconditioner */
      ierr = KSPCreate(comm,&ksp);CHKERRCONTINUE(ierr);
      ierr = KSPSetOperators(ksp,A,A);CHKERRCONTINUE(ierr);
      ierr = KSPSetType(ksp,KSPCG);CHKERRCONTINUE(ierr);
      ierr = KSPGetPC(ksp,&pc);CHKERRCONTINUE(ierr);
      ierr = PCSetType(pc,PCJACOBI);CHKERRCONTINUE(ierr);

      /* Solve linear system */
      ierr = KSPSetUp(ksp);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(comm,"Solving linear system\n");CHKERRCONTINUE(ierr);
      ierr = KSPSolve(ksp,b,x);CHKERRCONTINUE(ierr);

      ierr = VecNorm(x,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = KSPGetConvergedReason(ksp,&reason);CHKERRCONTINUE(ierr);
      ierr = KSPGetResidualNorm(ksp,&rnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(comm,"Residual=%f Converged=%d Soln norm=%f\n",rnorm,reason,vnorm);CHKERRCONTINUE(ierr);
    }
    ierr = PetscThreadCommReturnComm(comm,trank,&commrank);CHKERRCONTINUE(ierr);
    ierr = PetscThreadFinalize();CHKERRCONTINUE(ierr);
  }

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

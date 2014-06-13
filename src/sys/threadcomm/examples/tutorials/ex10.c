static char help[] = "Test PetscThreadPool with OpenMP with PETSc matrix routines.\n\n";

#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>
#include <omp.h>
#include <petscthreadcomm.h>
#include <../src/sys/threadcomm/impls/pthread/tcpthreadimpl.h>
#include <mpi.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat             A;
  Vec             x, b;
  PetscErrorCode  ierr;
  PetscInt        nthreads, n=20;
  PetscInt        i,j,Ii,J;
  PetscScalar     v, vnorm;
  KSP             ksp;
  PC              pc;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(PETSC_COMM_WORLD,&nthreads);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"nthreads=%d\n",nthreads);CHKERRQ(ierr);

  // Create vectors
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,n*n,&x);CHKERRCONTINUE(ierr);
  ierr = VecSetFromOptions(x);CHKERRCONTINUE(ierr);
  ierr = VecDuplicate(x,&b);CHKERRCONTINUE(ierr);

  #pragma omp parallel num_threads(nthreads) default(shared) private(ierr)
  {
    PetscInt pstart,pend,lsize,gsize;
    PetscInt prank,trank = omp_get_thread_num();

    ierr = PetscThreadPoolJoin(PETSC_COMM_WORLD,trank,&prank);CHKERRCONTINUE(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"trank=%d joined pool prank=%d\n",trank,prank);CHKERRCONTINUE(ierr);
    if(prank>=0) {
      // Set rhs
      ierr = VecSet(b,2.0);CHKERRCONTINUE(ierr);

      // Get ownership information
      ierr = VecGetOwnershipRange(x,&pstart,&pend);CHKERRCONTINUE(ierr);
      ierr = VecGetLocalSize(x,&lsize);CHKERRCONTINUE(ierr);
      ierr = VecGetSize(x,&gsize);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"localsize=%d globalsize=%d pstart=%d pend=%d\n",lsize,gsize,pstart,pend);CHKERRCONTINUE(ierr);

      // Create Matrix
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Creating matrix\n");CHKERRCONTINUE(ierr);
      ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRCONTINUE(ierr);
      ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n*n,n*n);CHKERRCONTINUE(ierr);
      ierr = MatSetType(A,MATMPIAIJ);CHKERRCONTINUE(ierr);
      ierr = MatSetUp(A);CHKERRCONTINUE(ierr);

      // Assemble matrix
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

      ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRCONTINUE(ierr);
      ierr = KSPSetOperators(ksp,A,A);CHKERRCONTINUE(ierr);
      ierr = KSPSetType(ksp,KSPCG);CHKERRCONTINUE(ierr);
      ierr = KSPGetPC(ksp,&pc);CHKERRCONTINUE(ierr);
      ierr = PCSetType(pc,PCJACOBI);CHKERRCONTINUE(ierr);

      ierr = KSPSetUp(ksp);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Solving linear system\n");CHKERRCONTINUE(ierr);
      ierr = KSPSolve(ksp,b,x);CHKERRCONTINUE(ierr);

      KSPConvergedReason reason;
      PetscScalar rnorm;
      ierr = VecNorm(x,NORM_2,&vnorm);CHKERRCONTINUE(ierr);
      ierr = KSPGetConvergedReason(ksp,&reason);CHKERRCONTINUE(ierr);
      ierr = KSPGetResidualNorm(ksp,&rnorm);CHKERRCONTINUE(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual=%f Converged=%d Soln norm=%f\n",rnorm,reason,vnorm);CHKERRCONTINUE(ierr);
    }
    ierr = PetscThreadPoolReturn(PETSC_COMM_WORLD,&prank);CHKERRCONTINUE(ierr);
  }

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}

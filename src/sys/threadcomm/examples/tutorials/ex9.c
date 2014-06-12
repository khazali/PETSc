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
  Vec             x, y;
  PetscErrorCode  ierr;
  PetscInt        nthreads, n=20;
  PetscInt        i,j;
  KSP             ksp;
  PC              pc;
  PetscScalar     t1, t2;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(PETSC_COMM_WORLD,&nthreads);CHKERRQ(ierr);
  printf("nthreads=%d\n",nthreads);

  #pragma omp parallel num_threads(nthreads) default(shared) private(ierr)
  {
    int prank;
    int trank = omp_get_thread_num();

    ierr = PetscThreadPoolJoin(PETSC_COMM_WORLD,trank,&prank);CHKERRCONTINUE(ierr);
    printf("trank=%d joined pool prank=%d\n",trank,prank);
    if(prank>=0) {
      // Create vectors
      ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,n,&x);CHKERRCONTINUE(ierr);
      ierr = VecSetFromOptions(x);CHKERRCONTINUE(ierr);
      ierr = VecDuplicate(x,&y);CHKERRCONTINUE(ierr);
      ierr = VecSet(x,2.0);CHKERRCONTINUE(ierr);

      // Create Matrix
      printf("Creating matrix\n");
      MatCreate(PETSC_COMM_WORLD,&A);
      MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);
      MatSetType(A,MATSEQAIJ);
      MatSetUp(A);

      // Assemble matrix
      printf("Assembling matrix\n");
      srand(0);
      double *vals = (double*)malloc(n*n*sizeof(double));
      int *ii = (int*)malloc(n*sizeof(int));
      int *jj = (int*)malloc(n*sizeof(int));
      for (i=0; i<n; i++) {
        ii[i] = i;
        jj[i] = i;
        for (j=0; j<n; j++) {
          vals[i+j*n] = rand() % 100;
        }
      }
      MatSetValues(A,n,ii,n,jj,vals,INSERT_VALUES);
      MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
      //MatView(A,PETSC_VIEWER_STDOUT_WORLD);

      printf("MatMultTranspose\n");
      t1 = MPI_Wtime();
      MatMultTranspose(A,x,y);
      t2 = MPI_Wtime();
      printf("Solve time=%f\n",t2-t1);

      printf("VecView\n");
      //VecView(y,PETSC_VIEWER_STDOUT_WORLD);

      KSPCreate(PETSC_COMM_WORLD,&ksp);
      KSPSetOperators(ksp,A,A);
      KSPSetType(ksp,KSPPREONLY);
      KSPGetPC(ksp,&pc);
      PCSetType(pc,PCLU);

      KSPSetUp(ksp);
      printf("Solving linear system\n");
      t1 = MPI_Wtime();
      KSPSolve(ksp,y,x);
      t2 = MPI_Wtime();
      printf("Solve time=%f\n",t2-t1);

      printf("VecView\n");
      //VecView(x,PETSC_VIEWER_STDOUT_WORLD);

      KSPDestroy(&ksp);
      MatDestroy(&A);
      VecDestroy(&x);
      VecDestroy(&y);
    }
    ierr = PetscThreadPoolReturn(PETSC_COMM_WORLD,&prank);CHKERRCONTINUE(ierr);
  }

  PetscFinalize();
  return 0;
}

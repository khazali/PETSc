
static char help[] = "Tests external Elemental direct solvers. Simplified from ex130.c\n\
Example: mpiexec -n <np> ./ex168 -f <matrix binary file> \n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A,A2,A3,A_elem,F;
  Vec            u,x,b,b_elem;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       m,n,nfact;
  PetscReal      norm,tol=1.e-12,Anorm;
  IS             perm,iperm;
  MatFactorInfo  info;
  PetscBool      flg,testMatSolve=PETSC_TRUE;
  PetscViewer    fd;              /* viewer */
  char           file[PETSC_MAX_PATH_LEN]; /* input file name */

  PetscInitialize(&argc,&args,(char*)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);

  /* Determine file from which we read the matrix A */
  ierr = PetscOptionsGetString(NULL,"-f",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f option");

  /* Load matrix A */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  if (m != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%d, %d)", m, n);
  ierr = MatCreateVecs(A,&b,&x);CHKERRQ(ierr);

  /* Test conversion routines */
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&A2);CHKERRQ(ierr);
  ierr = MatConvert(A2,MATELEMSPARSE,MAT_INITIAL_MATRIX,&A_elem);CHKERRQ(ierr);
  ierr = MatConvert(A_elem,MATAIJ,MAT_INITIAL_MATRIX,&A3);CHKERRQ(ierr);
  ierr = MatAXPY(A3,-1.0,A2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(A3,NORM_INFINITY,&Anorm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"AIJ-ELEMSPARSE-AIJ conversion error: %g\n",Anorm);CHKERRQ(ierr);
  ierr = MatDestroy(&A3);CHKERRQ(ierr);
  ierr = MatConvert(A_elem,MATAIJ,MAT_REUSE_MATRIX,&A_elem);CHKERRQ(ierr);
  ierr = MatAXPY(A2,-1.0,A_elem,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(A2,NORM_INFINITY,&Anorm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"AIJ-ELEMSPARSE-AIJ in place conversion error: %g\n",Anorm);CHKERRQ(ierr);
  ierr = MatDestroy(&A_elem);CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);

  /* test MatMult */
  ierr = MatConvert(A,MATELEMSPARSE,MAT_INITIAL_MATRIX,&A_elem);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&b_elem);CHKERRQ(ierr);
  ierr = VecSetRandom(x,NULL);CHKERRQ(ierr);
  ierr = MatMult(A,x,b);CHKERRQ(ierr);
  ierr = MatMult(A_elem,x,b_elem);CHKERRQ(ierr);
  ierr = VecAXPY(b_elem,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(b_elem,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"MatMult error %g\n",norm);CHKERRQ(ierr);
  ierr = MatDestroy(&A_elem);CHKERRQ(ierr);
  ierr = VecDestroy(&b_elem);CHKERRQ(ierr);

  /* Create random rhs */
  ierr = VecSetRandom(b,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr); /* save the true solution */

  /* Test Cholesky Factorization */
  ierr = MatNorm(A,NORM_INFINITY,&Anorm);CHKERRQ(ierr);
  ierr = MatGetOrdering(A,MATORDERINGNATURAL,&perm,&iperm);CHKERRQ(ierr);
  ierr = MatGetFactor(A,MATSOLVERELEMENTAL,MAT_FACTOR_CHOLESKY,&F);CHKERRQ(ierr);

  info.fill = 5.0;
  ierr = MatCholeskyFactorSymbolic(F,A,perm,&info);CHKERRQ(ierr);
  ierr = MatCholeskyFactorNumeric(F,A,&info);CHKERRQ(ierr);
  ierr = MatSolve(F,b,x);CHKERRQ(ierr);
  /* Check the residual */
  ierr = MatMult(A,x,u);CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"MatSolve: rel residual %g/%g = %g\n",norm,Anorm,norm/Anorm);CHKERRQ(ierr);

  /* Free data structures */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = ISDestroy(&iperm);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

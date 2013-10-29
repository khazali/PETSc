
static char help[] = "Solves a linear system in parallel. Contributed by S.C. Choi for testing MINRES and MINRESQLP. \n\n";
/* 
 Example: 
   mpiexec -n <np> ./ex42 -ksp_type minres -pc_type none -ksp_monitor_minres
   mpiexec -n <np> ./ex42 -ksp_type minresqlp -pc_type none -ksp_monitor_minresqlp
 */

#include <petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            x,b;      /* approx solution, RHS */
  Mat            A;        /* linear system matrix */
  KSP            ksp;      /* linear solver context */
  PetscInt       Ii,Istart,Iend,m = 11;
  PetscErrorCode ierr;
  PetscScalar    v,xnorm; 

  PetscInitialize(&argc,&args,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,"-m",&m,NULL);CHKERRQ(ierr);

  /* Create parallel diagonal matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,m);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,1,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

  for (Ii=Istart; Ii<Iend; Ii++) {
    v = (PetscScalar)Ii+1;
    ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  /* Make A singular */
  Ii = m - 1; /* last diagonal entry */
  v  = 0.0;
  ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* A is symmetric. Set symmetric flag to enable KSP_type = minres or minresqlp */
  ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  ierr = VecSet(b,1.0);CHKERRQ(ierr);
  ierr = VecSet(x,0.0);CHKERRQ(ierr);

  /* Create linear solver context */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  //ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&xnorm);CHKERRQ(ierr);   
  if (xnorm > 3.1826) {
     printf("\nERROR: TWO NORM OF X = %g IS TOO LARGE.\n", xnorm);
  }  else if (xnorm < 1.2448) {
     printf("\nERROR: TWO NORM OF X = %g IS TOO SMALL.\n", xnorm);
  } else {
     printf("\nxnorm = %g. x is a least-squares solution if ~ 3.1825. It is a pseudoinverse solution if ~ 1.2449.\n", xnorm);
  }

  /* Free work space. */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);  
  ierr = MatDestroy(&A);CHKERRQ(ierr);


  ierr = PetscFinalize();
  return 0;
}

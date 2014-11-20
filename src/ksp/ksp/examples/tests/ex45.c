
static char help[] = "Solves a linear system in parallel with KSP. \n\
Contributed by Stefano Zampini for testing repeated call of KSPSetOperators() with Elemental sparse solvers, partially copied by ex8.c\n\n";

#include <petscksp.h>
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            x,b,u;    /* approx solution, RHS, exact solution */
  Mat            A;        /* linear system matrix */
  KSP            ksp;      /* linear solver context */
  PC             pc;
  PetscRandom    rctx;     /* random number generator context */
  PetscInt       i,j,Ii,J,Istart,Iend,m = 8,n = 7;
  PetscScalar    v;
  PetscReal      error;
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;
  PetscBool      sameA=PETSC_FALSE;

  PetscInitialize(&argc,&args,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,"-nx",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-ny",&m,NULL);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ELEMENTAL)
  ierr = MatSetType(A,MATELEMSPARSE);CHKERRQ(ierr);
#endif
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  /* TODO ADD preallocation routine */
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = Ii + n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = Ii - 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = Ii + 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Create parallel vectors. */
  ierr = MatCreateVecs(A,&x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);

  /*
     Set exact solution; then compute right-hand-side vector.
     By default we use an exact solution of a vector with all
     elements of 1.0;  Alternatively, using the runtime option
     -random_sol forms a solution vector with random components.
  */
  ierr = PetscOptionsGetBool(NULL,"-random_exact_sol",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(u,rctx);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  } else {
    ierr = VecSet(u,1.0);CHKERRQ(ierr);
  }
  ierr = MatMult(A,u,b);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* Create linear solver context */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

  /* Set operators. */
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);

  ierr = KSPSetTolerances(ksp,1.e-2/((m+1)*(n+1)),1.e-50,PETSC_DEFAULT,
                          PETSC_DEFAULT);CHKERRQ(ierr);

  ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
  ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ELEMENTAL)
  ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERELEMENTAL);CHKERRQ(ierr);
#endif

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_INFINITY,&error);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Error first sol %g\n",error);
/*
  if (sameA) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Seting A\n");CHKERRQ(ierr);
    ierr = MatAXPY(A,1.1,B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  } else {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Seting B\n");CHKERRQ(ierr);
    ierr = MatAXPY(B,1.1,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,B,B);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }
*/
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = 4.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatMult(A,u,b);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_INFINITY,&error);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Error second sol %g\n",error);

  /* Free work space.*/
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

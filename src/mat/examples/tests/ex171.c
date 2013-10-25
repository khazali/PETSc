
static char help[] = "Tests vatious routines in MatTAIJ format.\n";

#include <petscmat.h>
#define IMAX 15
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A,B,TA;
  PetscScalar    *S,*T;
  PetscViewer    fd;
  char           file[PETSC_MAX_PATH_LEN];
  PetscInt       m,n,M,N,p=1,q=1,i,j;
  PetscMPIInt    rank,size;
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscInitialize(&argc,&args,(char*)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,1,"This example does not work with complex numbers");
#else

  /* Load aij matrix A */
  ierr = PetscOptionsGetString(NULL,"-f",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* Get dof, then create S and T */
  ierr = PetscOptionsGetInt(NULL,"-p",&p,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-q",&q,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc2(p*q,PetscScalar,&S,p*q,PetscScalar,&T);CHKERRQ(ierr);
  for (i=0; i<p*q; i++) S[i] = 0;

  for (i=0; i<p; i++) {
    for (j = 0; j<q; j++) {
      /* set some random non-zero values */
      S[i+p*j] = ((PetscReal) (i*j)) / ((PetscReal) (p+q));
      T[i+p*j] = ((PetscReal) ((p-i)+j)) / ((PetscReal) (p*q));
    }
  }

  /* Test TAIJ when both S & T are not NULL */

  /* create taij matrix TA */
  ierr = MatCreateTAIJ(A,p,q,S,T,&TA);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);

  if (size == 1) {
    ierr = MatConvert(TA,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  } else {
    ierr = MatConvert(TA,MATMPIAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  }

  /* Test MatMult() */
  ierr = MatMultEqual(TA,B,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 1: MatMult() for TAIJ matrix");
  /* Test MatMultAdd() */
  ierr = MatMultAddEqual(TA,B,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 1: MatMultAdd() for TAIJ matrix");

  ierr = MatDestroy(&TA);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  /* Test TAIJ when S is NULL */

  /* create taij matrix TA */
  ierr = MatCreateTAIJ(A,p,q,NULL,T,&TA);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);

  if (size == 1) {
    ierr = MatConvert(TA,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  } else {
    ierr = MatConvert(TA,MATMPIAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  }

  /* Test MatMult() */
  ierr = MatMultEqual(TA,B,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 2: MatMult() for TAIJ matrix");
  /* Test MatMultAdd() */
  ierr = MatMultAddEqual(TA,B,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 2: MatMultAdd() for TAIJ matrix");

  ierr = MatDestroy(&TA);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  
  /* Test TAIJ when T is NULL */

  /* create taij matrix TA */
  ierr = MatCreateTAIJ(A,p,q,S,NULL,&TA);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);

  if (size == 1) {
    ierr = MatConvert(TA,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  } else {
    ierr = MatConvert(TA,MATMPIAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  }

  /* Test MatMult() */
  ierr = MatMultEqual(TA,B,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 3: MatMult() for TAIJ matrix");
  /* Test MatMultAdd() */
  ierr = MatMultAddEqual(TA,B,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 3: MatMultAdd() for TAIJ matrix");

  ierr = MatDestroy(&TA);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  /* Done with all tests */

  ierr = PetscFree2(S,T);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
#endif
  return 0;
}

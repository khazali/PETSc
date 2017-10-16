
static char help[] = "Tests vatious routines in MatKAIJ format.\n";

#include <petscmat.h>
#define IMAX 15

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
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* Get dof, then create S and T */
  ierr = PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-q",&q,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc2(p*q,&S,p*q,&T);CHKERRQ(ierr);
  for (i=0; i<p*q; i++) S[i] = 0;

  for (i=0; i<p; i++) {
    for (j = 0; j<q; j++) {
      /* set some random non-zero values */
      S[i+p*j] = ((PetscReal) ((i+1)*(j+1))) / ((PetscReal) (p+q));
      T[i+p*j] = ((PetscReal) ((p-i)+j)) / ((PetscReal) (p*q));
    }
  }

  /* Test KAIJ when both S & T are not NULL */

  /* create kaij matrix TA */
  ierr = MatCreateKAIJ(A,p,q,S,T,&TA);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);

  if (size == 1) {
    ierr = MatConvert(TA,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  } else {
    ierr = MatConvert(TA,MATMPIAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  }

  /* Test MatMult() */
  ierr = MatMultEqual(TA,B,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 1: MatMult() for KAIJ matrix");
  /* Test MatMultAdd() */
  ierr = MatMultAddEqual(TA,B,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 1: MatMultAdd() for KAIJ matrix");

  ierr = MatDestroy(&TA);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  /* Test KAIJ when S is NULL */

  /* create kaij matrix TA */
  ierr = MatCreateKAIJ(A,p,q,NULL,T,&TA);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);

  if (size == 1) {
    ierr = MatConvert(TA,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  } else {
    ierr = MatConvert(TA,MATMPIAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  }

  /* Test MatMult() */
  ierr = MatMultEqual(TA,B,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 2: MatMult() for KAIJ matrix");
  /* Test MatMultAdd() */
  ierr = MatMultAddEqual(TA,B,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 2: MatMultAdd() for KAIJ matrix");

  ierr = MatDestroy(&TA);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  
  /* Test KAIJ when T is NULL */

  /* create kaij matrix TA */
  ierr = MatCreateKAIJ(A,p,q,S,NULL,&TA);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);

  if (size == 1) {
    ierr = MatConvert(TA,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  } else {
    ierr = MatConvert(TA,MATMPIAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  }

  /* Test MatMult() */
  ierr = MatMultEqual(TA,B,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 3: MatMult() for KAIJ matrix");
  /* Test MatMultAdd() */
  ierr = MatMultAddEqual(TA,B,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 3: MatMultAdd() for KAIJ matrix");

  ierr = MatDestroy(&TA);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  /* Test KAIJ when T is is an identity matrix */

  if (p == q) {

    for (i=0; i<p; i++) {
      for (j=0; j<q; j++) {
        if (i==j) T[i+j*p] = 1.0;
        else      T[i+j*p] = 0.0;
      }
    }

    /* create kaij matrix TA */
    ierr = MatCreateKAIJ(A,p,q,S,T,&TA);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
    ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);

    if (size == 1) {
      ierr = MatConvert(TA,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
    } else {
      ierr = MatConvert(TA,MATMPIAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
    }

    /* Test MatMult() */
    ierr = MatMultEqual(TA,B,10,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 4: MatMult() for KAIJ matrix");
    /* Test MatMultAdd() */
    ierr = MatMultAddEqual(TA,B,10,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error in Test 4: MatMultAdd() for KAIJ matrix");

    ierr = MatDestroy(&TA);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }

  /* Done with all tests */

  ierr = PetscFree2(S,T);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
#endif
  return 0;
}

/*TEST
  test:
    requires: datafilespath
    output_file: output/ex176.out
    nsize: {{1 2 3 4}}
    args: -f ${DATAFILESPATH}/matrices/small -p {{2 3 7}} -q {{3 7}} -viewer_binary_skip_info
TEST*/


static char help[] = "Tests vatious routines in MatTAIJ format.\n";

#include <petscmat.h>
#define IMAX 15
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A,B,TA;
  PetscScalar    *S;
  PetscViewer    fd;
  char           file[PETSC_MAX_PATH_LEN];
  PetscInt       m,n,M,N,dof=1,i;
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

  /* Get dof, then create the shift S */
  ierr = PetscOptionsGetInt(NULL,"-dof",&dof,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc(dof*dof*sizeof(PetscScalar),&S);CHKERRQ(ierr);
  for (i=0; i<dof*dof; i++) S[i] = 0;
  if (dof == 1) {
    S[0] = 1.0;
  } else if (dof == 2) {
    S[0] = 5.0/12.0;
    S[1] = 3.0/4.0;
    S[2] = -1.0/12.0;
    S[3] = 1.0/4.0;
  } else if (dof == 3) {
    S[0] = (88.0-7.0*PetscSqrtScalar(6.0))/360.0;
    S[1] = (296.0+169.0*PetscSqrtScalar(6.0))/1800.0;
    S[2] = (16.0-PetscSqrtScalar(6.0))/36.0;
    S[3] = (296.0+169.0*PetscSqrtScalar(6.0))/1800.0;
    S[4] = (88.0+7.0*PetscSqrtScalar(6.0))/360;
    S[5] = (16.0+PetscSqrtScalar(6.0))/36.0;
    S[6] = (-2.0+3.0*PetscSqrtScalar(6.0))/225.0;
    S[7] = (-2.0-3.0*PetscSqrtScalar(6.0))/225.0;
    S[8] = 1.0/9.0;
  } else {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error: This example is not set up for dof > 3");
  }

  /* create taij matrix TA */
  ierr = MatCreateTAIJ(A,dof,S,&TA);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);

  if (size == 1) {
    ierr = MatConvert(TA,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  } else {
    ierr = MatConvert(TA,MATMPIAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  }

  /* Test MatMult() */
  ierr = MatMultEqual(TA,B,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error: MatMult() for TAIJ matrix");
  /* Test MatMultAdd() */
  ierr = MatMultAddEqual(TA,B,10,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Error: MatMultAdd() for TAIJ matrix");

  ierr = PetscFree(S);CHKERRQ(ierr);
  ierr = MatDestroy(&TA);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscFinalize();
#endif
  return 0;
}

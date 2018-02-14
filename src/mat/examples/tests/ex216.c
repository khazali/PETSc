
static char help[] = "Tests MatIncreaseOverlap(), MatCreateSubMatrices() for sequential MatSBAIJ format. Derived from ex51.c\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat               A;
  PetscInt          bs=3,m=3,i,j,*rows,*cols,M,*idx;
  PetscErrorCode    ierr;
  PetscScalar       *vals,rval;
  PetscRandom       rand;
  const PetscScalar *bdiag;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mat_block_size",&bs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mat_size",&m,NULL);CHKERRQ(ierr);

  /* create a SeqBAIJ matrix A */
  M    = m*bs;
  ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,bs,M,M,1,NULL,&A);CHKERRQ(ierr);
  ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);

  ierr = PetscMalloc1(bs,&rows);CHKERRQ(ierr);
  ierr = PetscMalloc1(bs,&cols);CHKERRQ(ierr);
  ierr = PetscMalloc1(bs*bs,&vals);CHKERRQ(ierr);
  ierr = PetscMalloc1(M,&idx);CHKERRQ(ierr);

  /* Now set blocks of random values */
  /* first, set diagonal blocks as zero */
  for (j=0; j<bs*bs; j++) vals[j] = j+1;
  for (j=0; j<bs; j++) vals[j*bs + j] += 30;
  for (i=0; i<m; i++) {
    cols[0] = i*bs; rows[0] = i*bs;
    for (j=1; j<bs; j++) {
      rows[j] = rows[j-1]+1;
      cols[j] = cols[j-1]+1;
    }
    ierr = MatSetValues(A,bs,rows,bs,cols,vals,ADD_VALUES);CHKERRQ(ierr);
  }
  /* second, add random blocks */
  for (i=0; i<3*m; i++) {
    ierr    = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
    cols[0] = bs*(int)(PetscRealPart(rval)*m);
    ierr    = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
    rows[0] = bs*(int)(PetscRealPart(rval)*m);
    if (rows[0] == cols[0]) continue;
    for (j=1; j<bs; j++) {
      rows[j] = rows[j-1]+1;
      cols[j] = cols[j-1]+1;
    }

    for (j=0; j<bs*bs; j++) {
      ierr    = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
      vals[j] = rval;
    }
    ierr = MatSetValues(A,bs,rows,bs,cols,vals,ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);


  ierr = MatInvertBlockDiagonal(A,&bdiag);CHKERRQ(ierr);
  ierr = MatBAIJBlockDiagonalScale(A,bdiag);CHKERRQ(ierr);

  ierr = PetscFree(idx);CHKERRQ(ierr);
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = PetscFree(cols);CHKERRQ(ierr);
  ierr = PetscFree(vals);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      args: 

TEST*/

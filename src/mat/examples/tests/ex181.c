
static char help[] = "Tests MatReset(): creates a matrix, preallocates, inserts some values, views the matrix, resets, preallocates for more values, inserts more values (than the first time) and views again.\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            mat;
  PetscInt       M = 10,N = 10,bs=2,rstart,rend,m,i,*dnnz,*onnz,cols[3];
  PetscErrorCode ierr;
  PetscScalar    values[4] = {1.0,2.0,3.0,4.0};

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(PETSC_VIEWER_STDOUT_SELF,PETSC_VIEWER_ASCII_COMMON);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetBlockSize(mat,bs);CHKERRQ(ierr);
  ierr = MatSetFromOptions(mat);CHKERRQ(ierr);
  ierr = MatSetUp(mat);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(mat,&rstart,&rend);CHKERRQ(ierr);
  m = rend-rstart;
  ierr = PetscMalloc2(m,&dnnz,m,&onnz);CHKERRQ(ierr);
  for (i=0; i<m/bs; i++) {
    dnnz[i] = 1;
  }
  ierr = MatXAIJSetPreallocation(mat,bs,dnnz,NULL,dnnz,NULL);CHKERRQ(ierr);
  for (i=rstart/2; i<rend/2; i++) {
    ierr  = MatSetValuesBlocked(mat,1,&i,1,&i,values,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Original matrix\n");CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatReset(mat);CHKERRQ(ierr);
  for (i=0; i<m/bs; i++) {
    /* Inserting two values per row: one on the diagonal and the other elsewhere.
       In the parallel setting the second value may end up in the diagonal or off-diagonal
       matrix -- depending on the parallel decomposition -- and it may end up in different
       2x2 blocks that we are preallocating. So we allocate an extra block entry both in
       the diag and off-diag parts.
    */
    dnnz[i] = 2;
    onnz[i] = 1;
  }
  ierr = MatXAIJSetPreallocation(mat,bs,dnnz,onnz,dnnz,onnz);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    cols[0] = i;
    cols[1] = (i + 4) % 5;
    ierr  = MatSetValues(mat,1,&i,2,cols,values,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Matrix after the first rebuilding\n");CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatReset(mat);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    /* Inserting three values per row: one on the diagonal and the other two elsewhere.
       In the parallel setting the second and third value may end up in the diagonal or
       off-diagonal matrix -- depending on the parallel decomposition -- and in different
       2x2 blocks at that.  So we allocate two extra entries 2x2 blocks both in the diag
       and off-diag parts.  We have to be careful, however, not to request more preallocation
       than there are 2x2 blocks. Note also that doing preallocation with block size 1 is not
       an option, since that would amount to resetting the block size to 1 -- resetting
       of block sizes is not allowed.
    */
    dnnz[i] = PetscMin(3,m/bs);
    onnz[i] = PetscMin(2,(M-m)/bs);
  }
  ierr = MatXAIJSetPreallocation(mat,bs,dnnz,onnz,dnnz,onnz);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    cols[0] = i;
    cols[1] = (i + 4) % 5;
    cols[1] = (i + 7) % 5;
    ierr  = MatSetValues(mat,1,&i,3,cols,values,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Matrix after second rebuilding\n");CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscFree2(dnnz,onnz);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}


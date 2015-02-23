
static char help[] = "Tests MatReset(): creates a matrix, preallocates, inserts some values, views the matrix, resets, preallocates for more values, inserts more values (than the first time) and views again.\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            mat;
  PetscInt       M = 10,N = 10,bs=2,rstart,rend,m,i,*dnnz,*onnz,cols[2];
  PetscErrorCode ierr;
  PetscScalar    values[2] = {1.0,2.0};

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
  for (i=0; i<m; i++) {
    dnnz[i] = 1;
  }
  ierr = MatXAIJSetPreallocation(mat,bs,dnnz,NULL,dnnz,NULL);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr  = MatSetValues(mat,1,&i,1,&i,values,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Original matrix\n");CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatReset(mat);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    /* Inserting two values per row: one on the diagonal and the other elsewhere.
       In the parallel setting the second value may end up in the diagonal or off-diagonal
       matrix -- depending on the parallel decomposition.  So we allocate an extra entry
       both in the diag and off-diag parts.
    */
    dnnz[i] = 2;
    onnz[i] = 1;
  }
  ierr = MatXAIJSetPreallocation(mat,bs,dnnz,onnz,dnnz,onnz);CHKERRQ(ierr);
  ierr = PetscFree2(dnnz,onnz);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    cols[0] = i;
    cols[1] = (i + 4) % 5;
    ierr  = MatSetValues(mat,1,&i,2,cols,values,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"New matrix\n");CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}


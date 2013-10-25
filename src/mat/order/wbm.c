#include <petscmat.h>
#include <petsc-private/matorderimpl.h>

/*
  MatGetOrdering_WBM - Find the nonsymmetric reordering of the graph which maximizes the product of diagonal entries,
    using weighted bipartite graph matching. This is MC64 in the Harwell-Boeing library.
*/
#undef __FUNCT__
#define __FUNCT__ "MatGetOrdering_WBM"
PETSC_EXTERN PetscErrorCode MatGetOrdering_WBM(Mat mat, MatOrderingType type, IS *row, IS *col)
{
  PetscScalar    *a, *dw, cntl[1];
  const PetscInt *ia, *ja;
  const PetscInt  job = 5;
  PetscInt       *perm, nrow, ncol, nnz, num, liw, *iw, ldw, icntl[5], info[10], i;
  PetscBool       done;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  ncol = nrow;
  nnz  = ia[nrow];
  if (!done) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot get rows for matrix");
  ierr = MatSeqAIJGetArray(mat, &a);CHKERRQ(ierr);
  switch (job) {
  case 1: liw = 4*nrow +   ncol; ldw = 0;break;
  case 2: liw = 2*nrow + 2*ncol; ldw = ncol;break;
  case 3: liw = 8*nrow + 2*ncol + nnz; ldw = nnz;break;
  case 4: liw = 3*nrow + 2*ncol; ldw = 2*ncol + nnz;break;
  case 5: liw = 3*nrow + 2*ncol; ldw = nrow + 2*ncol + nnz;break;
  }
  icntl[0] = 0;/*-1*/
  icntl[1] = 0;/*-1*/
  icntl[2] = 0;/*-1*/
  icntl[3] = 0;
  icntl[4] = 4;/*-1*/
  cntl[0]  = 0.0;/*1e-8*/

  ierr = PetscMalloc3(liw,PetscInt,&iw,ldw,PetscScalar,&dw,nrow,PetscInt,&perm);CHKERRQ(ierr);
  ierr = HSLmc64AD(&job, &ncol, &nrow, &nnz, ia, ja, a, &num, perm, &liw, iw, &ldw, dw, icntl, cntl, info);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(mat, 1, PETSC_TRUE, PETSC_TRUE, NULL, &ia, &ja, &done);CHKERRQ(ierr);
  for (i = 0; i < nrow; ++i) perm[i]--;
  /* If job == 5, dw[0..ncols] contains the column scaling and dw[ncols..ncols+nrows] contains the row scaling */
  ierr = ISCreateStride(PETSC_COMM_SELF, nrow, 0, 1, row);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,col);CHKERRQ(ierr);
  ierr = PetscFree3(iw,dw,perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

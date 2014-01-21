#include <petsc-private/matimpl.h>      /*I "petscmat.h"  I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

#undef __FUNCT__
#define __FUNCT__ "MatColoringGetLocalGraph_Private"
PetscErrorCode MatColoringGetLocalGraph_Private(Mat G,PetscInt **ia,PetscInt **ja,PetscInt **iao,PetscInt **jao)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscInt       s,e,n;
  PetscBool      ismpiaij,isseqaij;
  Mat            dG,oG;
  PetscInt       *dia,*dja,*oia,*oja,ncols,dcols,ocols;
  const PetscInt *cols;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)G,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)G,MATSEQAIJ,&isseqaij);CHKERRQ(ierr);
  if (ismpiaij) {
    Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)G->data;
    dG = aij->A;
    oG = aij->B;
  } else if (isseqaij) {
    dG = G;
    oG = NULL;
  } else {
    SETERRQ(PetscObjectComm((PetscObject)G),PETSC_ERR_ARG_WRONG,"Only Seq/MPIAIJ matrices supported for coloring");
  }
  ierr = MatGetOwnershipRange(G,&s,&e);CHKERRQ(ierr);
  n = e-s;
  ierr = PetscMalloc1(n+1,&dia);CHKERRQ(ierr);
  if (oG) {
    ierr = PetscMalloc(n+1,&oia);CHKERRQ(ierr);
  }
  /* count the total entries and get the offsets */
  dcols=0;
  ocols=0;
  for (i=0;i<n;i++) {
    ierr = MatGetRow(dG,i,&ncols,NULL,NULL);CHKERRQ(ierr);
    dia[i]=dcols;
    dcols += ncols;
    ierr = MatRestoreRow(dG,i,&ncols,NULL,NULL);CHKERRQ(ierr);
    if (oG) {
      ierr = MatGetRow(oG,i,&ncols,NULL,NULL);CHKERRQ(ierr);
      oia[i]=ocols;
      ocols += ncols;
      ierr = MatRestoreRow(oG,i,&ncols,NULL,NULL);CHKERRQ(ierr);
    }
  }
  dia[n]=dcols;
  if (oG) {
    oia[n]=ocols;
  }
  ierr = PetscMalloc1(dcols,&dja);CHKERRQ(ierr);
  if (oG) {
    ierr = PetscMalloc1(ocols,&oja);CHKERRQ(ierr);
  }
  dcols=0;
  ocols=0;
  for (i=0;i<n;i++) {
    ierr = MatGetRow(dG,i,&ncols,&cols,NULL);CHKERRQ(ierr);
    for (j=0;j<ncols;j++) {
      dja[dcols+j] = cols[j];
    }
    dia[i]=dcols;
    dcols += ncols;
    ierr = MatRestoreRow(dG,i,&ncols,&cols,NULL);CHKERRQ(ierr);
    if (oG) {
      ierr = MatGetRow(oG,i,&ncols,&cols,NULL);CHKERRQ(ierr);
      for (j=0;j<ncols;j++) {
        oja[dcols+j] = cols[j];
      }
      oia[i]=ocols;
      ocols += ncols;
      ierr = MatRestoreRow(oG,i,&ncols,&cols,NULL);CHKERRQ(ierr);
    }
  }
  *ia = dia;
  *ja = dja;
  *iao = oia;
  *jao = oja;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringGetGraph_Private"
PetscErrorCode MatColoringGetGraph_Private(MatColoring mc,Mat *G)
{
  PetscErrorCode ierr;
  Mat            M=mc->mat;
  Mat            Gm=M,Gmn;
  PetscInt       i,k=mc->dist;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)Gm);CHKERRQ(ierr);
  for (i=0;i<k-1;i++) {
    ierr = MatMatMult(M,Gm,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Gmn);CHKERRQ(ierr);
    ierr = MatDestroy(&Gm);CHKERRQ(ierr);
    Gm = Gmn;
  }
  *G = Gm;
  PetscFunctionReturn(0);
  }

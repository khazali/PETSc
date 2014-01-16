#include <petsc-private/matimpl.h>      /*I "petscmat.h"  I*/

#undef __FUNCT__
#define __FUNCT__ "MatColoringGetDistanceKGraph"
PetscErrorCode MatColoringGetDistanceKGraph(Mat M,PetscInt k,PetscInt **dia,Mat *Mk)
{
  PetscErrorCode ierr;
  Mat            Mm=M,Mmn;
  PetscInt       i;

  PetscFunctionBegin;

  for (i=0;i<k-1;i++) {
    if (i%2) {
      ierr = MatMatMult(M,Mm,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Mmn);CHKERRQ(ierr);
      ierr = MatDestroy(&Mm);CHKERRQ(ierr);
      Mm = Mmn;
    } else {
      ierr = MatTransposeMatMult(M,Mm,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Mmn);CHKERRQ(ierr);
      ierr = MatDestroy(&Mm);CHKERRQ(ierr);
      Mm = Mmn;
    }
  }
  *Mk = Mm;

  PetscFunctionReturn(0);
}

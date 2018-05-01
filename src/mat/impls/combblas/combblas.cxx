
#include <../src/mat/impls/combblas/combblas.h>

/*MC
   MATCOMBBLAS = "combblas" - A matrix type for sparse matrices using the CombBLAS package

  Use ./configure --download-combblas to install PETSc to use CombBLAS

  Level: beginner

.seealso: MATType
M*/

PetscErrorCode MatCreate_CombBLAS(Mat A)
{
  Mat_CombBLAS   *a;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr       = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr       = PetscNewLog(A,&a);CHKERRQ(ierr);
  A->data    = (void*)a;
  a->combmat = new combblas::SpParMat<PetscInt,PetscScalar,combblas::SpDCCols<PetscInt, PetscScalar>>(comm);if (!a->combmat) SETERRQ(comm,PETSC_ERR_LIB,"Cannot construct SpParMat");
  PetscFunctionReturn(0);
}

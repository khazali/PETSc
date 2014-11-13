#if !defined(_matelemsparseimpl_h)
#define _matelemsparseimpl_h

#include <../src/mat/impls/elemental/matelemimpl.h> /*I "petscmat.h" I*/

typedef struct {
  MatStructure matstruc;
  PetscBool    CleanUpClique;    /* Boolean indicating if we call Clique clean step */
  MPI_Comm     cliq_comm;        /* Elemental MPI communicator                         */
  PetscInt     cutoff;           /* maximum size of leaf node */
  PetscInt     numDistSeps;      /* number of distributed separators to try */
  PetscInt     numSeqSeps;       /* number of sequential separators to try */

  El::DistSparseMatrix<PetscElemScalar>  *cmat;  /* Elemental sparse matrix */
  El::DistMap                            *inverseMap;
  El::DistSymmInfo                       *info;
  El::DistSymmFrontTree<PetscElemScalar> *frontTree;
  El::DistMultiVec<PetscElemScalar>        *rhs;
  El::DistNodalMultiVec<PetscElemScalar>   *xNodal;

  PetscErrorCode (*Destroy)(Mat);
} Mat_ElemSparse;

#endif

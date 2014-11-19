#if !defined(_matelemsparseimpl_h)
#define _matelemsparseimpl_h

#include <../src/mat/impls/elemental/matelemimpl.h> /*I "petscmat.h" I*/

typedef struct {
  MatStructure matstruc;
  PetscBool    freemv;
  PetscInt     cutoff;           /* maximum size of leaf node */
  PetscInt     numDistSeps;      /* number of distributed separators to try */
  PetscInt     numSeqSeps;       /* number of sequential separators to try */

  El::DistSparseMatrix<PetscElemScalar>  *cmat;  /* Elemental sparse matrix */
  El::DistMultiVec<PetscElemScalar>      *cvecr;  /* Elemental right vector for MatMults */
  El::DistMultiVec<PetscElemScalar>      *cvecl;  /* Elemental left vector for MatMults */
  El::DistSeparatorTree                  *sepTree;
  El::DistMap                            *map;
  El::DistMap                            *inverseMap;
  El::DistSymmInfo                       *info;
  El::DistSymmFrontTree<PetscElemScalar> *frontTree;
  El::DistNodalMultiVec<PetscElemScalar> *xNodal;

  PetscErrorCode (*Destroy)(Mat);
} Mat_ElemSparse;

#endif

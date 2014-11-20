#if !defined(_matelemsparseimpl_h)
#define _matelemsparseimpl_h

#include <../src/mat/impls/elemental/matelemimpl.h> /*I "petscmat.h" I*/

typedef struct {

  /* Mat */
  El::DistSparseMatrix<PetscElemScalar> *cmat;  /* Elemental sparse matrix */
  El::DistMultiVec<PetscElemScalar>     *cvecr;  /* Elemental right vector for MatMults */
  El::DistMultiVec<PetscElemScalar>     *cvecl;  /* Elemental left vector for MatMults */
  PetscBool    freemv;

  /* MatFactor */
  MatStructure                           matstruc;
  El::DistSeparatorTree                  *sepTree;
  El::DistMap                            *map;
  El::DistMap                            *inverseMap;
  El::DistSymmInfo                       *info;
  El::DistSymmFrontTree<PetscElemScalar> *frontTree;
  El::DistNodalMultiVec<PetscElemScalar> *xNodal;
  PetscErrorCode (*Destroy)(Mat);

  /* customization for sparse direct solver */
  PetscBool use_metis; /* use or not METIS/ParMETIS nested dissection */
  PetscBool seq_nd;    /* use sequential nested dissection from METIS instead of using ParMETIS */
  PetscInt  cutoff;    /* maximum size of leaf node */
  PetscInt  numSeps;   /* number of separators to try */
  PetscBool selInv;    /* use selective inversion */
  PetscBool intraPiv;  /* use frontal pivoting */
} Mat_ElemSparse;

#endif

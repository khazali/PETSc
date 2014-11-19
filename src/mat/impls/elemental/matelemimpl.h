#if !defined(_matelemimpl_h)
#define _matelemimpl_h

#include <El.hpp>
#include <petsc-private/matimpl.h>

PETSC_EXTERN PetscErrorCode MatFactorGetSolverPackage_elemental(Mat,const MatSolverPackage*);
PETSC_EXTERN PetscErrorCode MatGetFactor_elemdense_elemental(Mat,MatFactorType,Mat*);
PETSC_EXTERN PetscErrorCode MatGetFactor_aij_elemental(Mat,MatFactorType,Mat*);

#if defined(PETSC_USE_COMPLEX)
typedef El::Complex<PetscReal> PetscElemScalar;
#else
typedef PetscScalar PetscElemScalar;
#endif

#endif

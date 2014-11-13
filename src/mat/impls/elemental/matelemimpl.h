#if !defined(_matelemimpl_h)
#define _matelemimpl_h

#include <El.hpp>
#include <petsc-private/matimpl.h>

PetscErrorCode MatFactorGetSolverPackage_elemental_elemental(Mat,const MatSolverPackage*);
PetscErrorCode MatGetFactor_elemdense_elemdense(Mat,MatFactorType,Mat*);

#if defined(PETSC_USE_COMPLEX)
typedef El::Complex<PetscReal> PetscElemScalar;
#else
typedef PetscScalar PetscElemScalar;
#endif

#endif

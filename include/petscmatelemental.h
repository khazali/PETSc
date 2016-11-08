#if !defined(__PETSCMATELEMENTAL_H)
#define __PETSCMATELEMENTAL_H

#include <petscmat.h>

#if defined(PETSC_HAVE_ELEMENTAL) && defined(__cplusplus)
#include <El.hpp>
#if defined(PETSC_USE_COMPLEX)
typedef El::Complex<PetscReal> PetscElemScalar;
#else
typedef PetscScalar PetscElemScalar;
#endif
/* c++ prototypes requiring elemental datatypes. */
PETSC_EXTERN PetscErrorCode MatElementalHermitianGenDefEig(El::Pencil,El::UpperOrLower,Mat,Mat,Mat*,Mat*,const El::HermitianEigCtrl<PetscElemScalar>);
PETSC_EXTERN PetscErrorCode MatElementalSyrk(El::UpperOrLower,El::Orientation,PetscScalar,Mat,PetscScalar,Mat,PetscBool);
PETSC_EXTERN PetscErrorCode MatElementalHerk(El::UpperOrLower,El::Orientation,PetscScalar,Mat,PetscScalar,Mat);
PETSC_EXTERN PetscErrorCode MatElementalSyr2k(El::UpperOrLower,El::Orientation,PetscScalar,Mat,Mat,PetscScalar,Mat,PetscBool);
PETSC_EXTERN PetscErrorCode MatElementalHer2k(El::UpperOrLower,El::Orientation,PetscScalar,Mat,Mat,PetscScalar,Mat);

#endif

#endif /* __PETSCMATELEMENTAL_H */

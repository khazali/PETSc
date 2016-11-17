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
#if defined(PETSC_HAVE_COMPLEX)
PETSC_EXTERN PetscErrorCode MatElementalHermitianGenDefEig(El::Pencil,El::UpperOrLower,Mat,Mat,Mat*,Mat*,const El::HermitianEigCtrl<PetscElemScalar>);
#endif
PETSC_EXTERN PetscErrorCode MatElementalSyrk(El::UpperOrLower,El::Orientation,PetscScalar,Mat,PetscScalar,Mat,PetscBool);
PETSC_EXTERN PetscErrorCode MatElementalHerk(El::UpperOrLower,El::Orientation,PetscReal,Mat,PetscReal,Mat);
PETSC_EXTERN PetscErrorCode MatElementalSyr2k(El::UpperOrLower,El::Orientation,PetscScalar,Mat,Mat,PetscScalar,Mat,PetscBool);
PETSC_EXTERN PetscErrorCode MatElementalHer2k(El::UpperOrLower,El::Orientation,PetscScalar,Mat,Mat,PetscReal,Mat);
PETSC_EXTERN PetscErrorCode MatGetElementalMat(Mat,void**);
PETSC_EXTERN PetscErrorCode MatCreateElementalWithEl(const El::DistMatrix<PetscElemScalar>&,Mat*);

#endif

#endif /* __PETSCMATELEMENTAL_H */

#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

/*MC
  PETSCSPACETENSOR = "tensor" - A PetscSpace object that encapsulates a tensor product space.

  Level: intermediate

.seealso: PetscSpaceType, PetscSpaceCreate(), PetscSpaceSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Polynomial(PetscSpace sp)
{
  PetscSpace_Tensor *tens;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp,&tens);CHKERRQ(ierr);
  sp->data = poly;

  tens->numSubspaces = PETSC_DEFAULT;

  PetscFunctionReturn(0);
}

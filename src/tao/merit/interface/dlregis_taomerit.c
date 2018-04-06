#define TAOMERIT_DLL
#include <petsc/private/taomeritimpl.h>

PETSC_EXTERN PetscErrorCode TaoMeritCreate_OBJ(TaoMerit);
static PetscBool TaoMeritPackageInitialized = PETSC_FALSE;

/*@C
  TaoMeritFinalizePackage - This function destroys everything in the PETSc/TAO
  interface to the TaoMerit package. It is called from PetscFinalize().

  Level: developer
@*/
PetscErrorCode TaoMeritFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&TaoMeritList);CHKERRQ(ierr);
  TaoMeritInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  TaoMeritInitializePackage - This function registers the merit functions in TAO.
  When using static libraries, this function is called from the
  first entry to TaoCreate(); when using shared libraries, it is called
  from PetscDLLibraryRegister()

  Level: developer

.seealso: TaoMeritCreate()
@*/
PetscErrorCode TaoMeritInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TaoMeritPackageInitialized) PetscFunctionReturn(0);
  TaoMeritPackageInitialized=PETSC_TRUE;
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscClassIdRegister("TaoMerit",&TAOMERIT_CLASSID);CHKERRQ(ierr);
  ierr = TaoLineSearchRegister("obj",TaoMeritCreate_Obj);CHKERRQ(ierr);
#endif
  ierr = PetscRegisterFinalize(TaoMeritFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
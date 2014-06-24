#include <petsc-private/threadcommimpl.h>

static PetscBool PetscThreadPoolPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolFinalizePackage"
/*@C
   PetscThreadPoolFinalizePackage - Finalize PetscThreadPool package, called from PetscFinalize()

   Logically collective

   Level: developer

.seealso: PetscThreadPoolInitializePackage()
@*/
PetscErrorCode PetscThreadPoolFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscThreadPoolPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolInitializePackage"
/*@C
   PetscThreadPoolInitializePackage - Initializes threadpool package

   Logically collective

   Level: developer

.seealso: PetscThreadPoolFinalizePackage()
@*/
PetscErrorCode PetscThreadPoolInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscThreadPoolPackageInitialized) PetscFunctionReturn(0);

  ierr = PetscGetNCores(NULL);CHKERRQ(ierr);

  PetscThreadPoolPackageInitialized = PETSC_TRUE;

  ierr = PetscRegisterFinalize(PetscThreadPoolFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

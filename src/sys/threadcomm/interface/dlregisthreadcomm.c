#include <petsc-private/threadcommimpl.h>

static PetscBool PetscThreadCommPackageInitialized = PETSC_FALSE;

extern PetscBool PetscThreadPoolRegisterAllModelsCalled;
extern PetscBool PetscThreadPoolRegisterAllTypesCalled;

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommFinalizePackage"
/*@C
   PetscThreadCommFinalizePackage - Finalize PetscThreadComm package, called from PetscFinalize()

   Logically collective

   Level: developer

.seealso: PetscThreadCommInitializePackage()
@*/
PetscErrorCode PetscThreadCommFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&PetscThreadPoolTypeList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&PetscThreadPoolModelList);CHKERRQ(ierr);
  PetscThreadCommPackageInitialized      = PETSC_FALSE;
  PetscThreadPoolRegisterAllModelsCalled = PETSC_FALSE;
  PetscThreadPoolRegisterAllTypesCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInitializePackage"
/*@C
   PetscThreadCommInitializePackage - Initializes ThreadComm package

   Logically collective

   Level: developer

.seealso: PetscThreadCommFinalizePackage()
@*/
PetscErrorCode PetscThreadCommInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscThreadCommPackageInitialized) PetscFunctionReturn(0);

  ierr = PetscGetNCores(NULL);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("ThreadCommRunKer",  0, &ThreadComm_RunKernel);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("ThreadCommBarrie",  0, &ThreadComm_Barrier);CHKERRQ(ierr);

  PetscThreadCommPackageInitialized = PETSC_TRUE;

  ierr = PetscRegisterFinalize(PetscThreadCommFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

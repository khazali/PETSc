
#include <petsc-private/threadcommimpl.h>     /*I    "petscthreadcomm.h"  I*/

PETSC_EXTERN PetscErrorCode PetscThreadCommInit_NoThread(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_NoThread(PetscThreadComm);
#if defined(PETSC_HAVE_PTHREADCLASSES)
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_PThread(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_PThread(PetscThreadComm);
#endif
#if defined(PETSC_HAVE_OPENMP)
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_OpenMP(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_OpenMP(PetscThreadComm);
#endif
#if defined(PETSC_HAVE_TBB)
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_TBB(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_TBB(PetscThreadComm);
#endif

PETSC_EXTERN PetscErrorCode PetscThreadCommCreateModel_Loop(PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreateModel_Auto(PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreateModel_User(PetscThreadComm);

extern PetscBool PetscThreadCommRegisterAllModelsCalled;
extern PetscBool PetscThreadCommRegisterAllTypesCalled;

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRegisterAllModels"
/*@C
   PetscThreadCommRegisterAllModels - Registers of all the threadpool models

   Not Collective

   Level: advanced

.keywords: PetscThreadComm, register, all

.seealso: PetscThreadCommRegisterDestroy()
@*/
PetscErrorCode PetscThreadCommRegisterAllModels(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscThreadCommRegisterAllModelsCalled = PETSC_TRUE;

  ierr = PetscThreadCommModelRegister(LOOP,PetscThreadCommCreateModel_Loop);CHKERRQ(ierr);
  ierr = PetscThreadCommModelRegister(AUTO,PetscThreadCommCreateModel_Auto);CHKERRQ(ierr);
  ierr = PetscThreadCommModelRegister(USER,PetscThreadCommCreateModel_User);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRegisterAllTypes"
/*@C
   PetscThreadCommRegisterAllTypes - Registers of all the threadpool models

   Not Collective

   Level: advanced

.keywords: PetscThreadComm, register, all

.seealso: PetscThreadCommRegisterDestroy()
@*/
PetscErrorCode PetscThreadCommRegisterAllTypes(PetscThreadPool pool)
{
  PetscInt model;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscThreadCommRegisterAllTypesCalled = PETSC_TRUE;

  model = pool->model;
  printf("Registering Types\n");

  ierr = PetscThreadCommInitTypeRegister(NOTHREAD,PetscThreadCommInit_NoThread);CHKERRQ(ierr);
  ierr = PetscThreadCommTypeRegister(NOTHREAD,PetscThreadCommCreate_NoThread);CHKERRQ(ierr);

#if defined(PETSC_HAVE_PTHREADCLASSES)
  ierr = PetscThreadCommInitTypeRegister(PTHREAD,PetscThreadCommInit_PThread);CHKERRQ(ierr);
  ierr = PetscThreadCommTypeRegister(PTHREAD, PetscThreadCommCreate_PThread);CHKERRQ(ierr);
#endif

#if defined(PETSC_HAVE_OPENMP)
  ierr = PetscThreadCommInitTypeRegister(OPENMP,PetscThreadCommInit_OpenMP);CHKERRQ(ierr);
  ierr = PetscThreadCommTypeRegister(OPENMP,  PetscThreadCommCreate_OpenMP);CHKERRQ(ierr);
#endif

#if defined(PETSC_HAVE_TBB)
  ierr = PetscThreadCommInitTypeRegister(TBB,PetscThreadCommInit_TBB);CHKERRQ(ierr);
  ierr = PetscThreadCommTypeRegister(TBB,     PetscThreadCommCreate_TBB);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

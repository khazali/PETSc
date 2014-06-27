
#include <petsc-private/threadcommimpl.h>     /*I    "petscthreadcomm.h"  I*/

PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_NoThread(PetscThreadComm);
#if defined(PETSC_HAVE_PTHREADCLASSES)
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_PThreadLoop(PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_PThreadAuto(PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_PThreadUser(PetscThreadComm);
#endif
#if defined(PETSC_HAVE_OPENMP)
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_OpenMPLoop(PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_OpenMPUser(PetscThreadComm);
#endif
#if defined(PETSC_HAVE_TBB)
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
PetscErrorCode PetscThreadCommRegisterAllTypes(PetscThreadComm tcomm)
{
  PetscInt model;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscThreadCommRegisterAllTypesCalled = PETSC_TRUE;

  model = tcomm->model;
  printf("Registering Types = %d\n",model);

  ierr = PetscThreadCommTypeRegister(NOTHREAD,PetscThreadCommCreate_NoThread);CHKERRQ(ierr);

#if defined(PETSC_HAVE_PTHREADCLASSES)
  if (model==THREAD_MODEL_LOOP) {
    ierr = PetscThreadCommTypeRegister(PTHREAD, PetscThreadCommCreate_PThreadLoop);CHKERRQ(ierr);
  } else if (model==THREAD_MODEL_AUTO) {
    ierr = PetscThreadCommTypeRegister(PTHREAD, PetscThreadCommCreate_PThreadAuto);CHKERRQ(ierr);
  } else if (model==THREAD_MODEL_USER) {
    ierr = PetscThreadCommTypeRegister(PTHREAD, PetscThreadCommCreate_PThreadUser);CHKERRQ(ierr);
  }
#endif

#if defined(PETSC_HAVE_OPENMP)
  if (model==THREAD_MODEL_LOOP) {
    ierr = PetscThreadCommTypeRegister(OPENMP,  PetscThreadCommCreate_OpenMPLoop);CHKERRQ(ierr);
  } else if (model==THREAD_MODEL_USER) {
    ierr = PetscThreadCommTypeRegister(OPENMP,  PetscThreadCommCreate_OpenMPUser);CHKERRQ(ierr);
  }
#endif

#if defined(PETSC_HAVE_TBB)
  ierr = PetscThreadCommTypeRegister(TBB,     PetscThreadCommCreate_TBB);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

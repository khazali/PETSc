
#include <petsc-private/threadcommimpl.h>     /*I    "petscthreadcomm.h"  I*/

PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_NoThread(PetscThreadComm);
#if defined(PETSC_HAVE_PTHREADCLASSES)
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_PThreadLoop(PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_PThreadUser(PetscThreadComm);
#endif
#if defined(PETSC_HAVE_OPENMP)
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_OpenMP(PetscThreadComm);
#endif
#if defined(PETSC_HAVE_TBB)
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_TBB(PetscThreadComm);
#endif

PETSC_EXTERN PetscErrorCode PetscThreadModelCreate_Loop(PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadModelCreate_User(PetscThreadComm);

extern PetscBool PetscThreadCommRegisterAllModelsCalled;
extern PetscBool PetscThreadCommRegisterAllTypesCalled;

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRegisterAllModels"
/*@C
   PetscThreadCommRegisterAllModels - Registers of all the thread communicator models

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

  ierr = PetscThreadModelRegister(LOOP,PetscThreadModelCreate_Loop);CHKERRQ(ierr);
  ierr = PetscThreadModelRegister(USER,PetscThreadModelCreate_User);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRegisterAllTypes"
/*@C
   PetscThreadCommRegisterAllTypes - Registers of all the thread communicator models

   Not Collective

   Level: advanced

.keywords: PetscThreadComm, register, all

.seealso: PetscThreadCommRegisterDestroy()
@*/
PetscErrorCode PetscThreadCommRegisterAllTypes(PetscThreadComm tcomm)
{
  PetscInt type;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscThreadCommRegisterAllTypesCalled = PETSC_TRUE;

  type = tcomm->pool->model;
  printf("Registering Types = %d\n",type);

  ierr = PetscThreadCommRegister(NOTHREAD,PetscThreadCommCreate_NoThread);CHKERRQ(ierr);

#if defined(PETSC_HAVE_PTHREADCLASSES)
  if (type==THREAD_MODEL_LOOP) {
    ierr = PetscThreadCommRegister(PTHREAD, PetscThreadCommCreate_PThreadLoop);CHKERRQ(ierr);
  } else if (type==THREAD_MODEL_USER) {
    ierr = PetscThreadCommRegister(PTHREAD, PetscThreadCommCreate_PThreadUser);CHKERRQ(ierr);
  }
#endif

#if defined(PETSC_HAVE_OPENMP)
  ierr = PetscThreadCommRegister(OPENMP,  PetscThreadCommCreate_OpenMP);CHKERRQ(ierr);
#endif

#if defined(PETSC_HAVE_TBB)
  ierr = PetscThreadCommRegister(TBB,     PetscThreadCommCreate_TBB);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

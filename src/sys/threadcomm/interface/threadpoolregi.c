
#include <petsc-private/threadcommimpl.h>     /*I    "petscthreadcomm.h"  I*/

PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_NoThread(PetscThreadPool);
#if defined(PETSC_HAVE_PTHREADCLASSES)
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_PThreadLoop(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_PThreadUser(PetscThreadPool);
#endif
#if defined(PETSC_HAVE_OPENMP)
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_OpenMPLoop(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_OpenMPUser(PetscThreadPool);
#endif
#if defined(PETSC_HAVE_TBB)
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_TBB(PetscThreadPool);
#endif

PETSC_EXTERN PetscErrorCode PetscThreadPoolCreateModel_Loop(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadPoolCreateModel_User(PetscThreadPool);

extern PetscBool PetscThreadPoolRegisterAllModelsCalled;
extern PetscBool PetscThreadPoolRegisterAllTypesCalled;

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolRegisterAllModels"
/*@C
   PetscThreadPoolRegisterAllModels - Registers of all the threadpool models

   Not Collective

   Level: advanced

.keywords: PetscThreadPool, register, all

.seealso: PetscThreadPoolRegisterDestroy()
@*/
PetscErrorCode PetscThreadPoolRegisterAllModels(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscThreadPoolRegisterAllModelsCalled = PETSC_TRUE;

  ierr = PetscThreadPoolModelRegister(LOOP,PetscThreadPoolCreateModel_Loop);CHKERRQ(ierr);
  ierr = PetscThreadPoolModelRegister(USER,PetscThreadPoolCreateModel_User);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolRegisterAllTypes"
/*@C
   PetscThreadPoolRegisterAllTypes - Registers of all the threadpool models

   Not Collective

   Level: advanced

.keywords: PetscThreadPool, register, all

.seealso: PetscThreadPoolRegisterDestroy()
@*/
PetscErrorCode PetscThreadPoolRegisterAllTypes(PetscThreadPool pool)
{
  PetscInt model;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscThreadPoolRegisterAllTypesCalled = PETSC_TRUE;

  model = pool->model;
  printf("Registering Types = %d\n",model);

  ierr = PetscThreadPoolTypeRegister(NOTHREAD,PetscThreadCommCreate_NoThread);CHKERRQ(ierr);

#if defined(PETSC_HAVE_PTHREADCLASSES)
  if (model==THREAD_MODEL_LOOP) {
    ierr = PetscThreadPoolTypeRegister(PTHREAD, PetscThreadCommCreate_PThreadLoop);CHKERRQ(ierr);
  } else if (model==THREAD_MODEL_USER) {
    ierr = PetscThreadPoolTypeRegister(PTHREAD, PetscThreadCommCreate_PThreadUser);CHKERRQ(ierr);
  }
#endif

#if defined(PETSC_HAVE_OPENMP)
  if (model==THREAD_MODEL_LOOP) {
    ierr = PetscThreadPoolTypeRegister(OPENMP,  PetscThreadCommCreate_OpenMPLoop);CHKERRQ(ierr);
  } else if (model==THREAD_MODEL_USER) {
    ierr = PetscThreadPoolTypeRegister(OPENMP,  PetscThreadCommCreate_OpenMPUser);CHKERRQ(ierr);
  }
#endif

#if defined(PETSC_HAVE_TBB)
  ierr = PetscThreadPoolTypeRegister(TBB,     PetscThreadCommCreate_TBB);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

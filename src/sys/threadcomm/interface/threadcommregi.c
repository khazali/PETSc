
#include <petsc-private/threadcommimpl.h>     /*I    "petscthreadcomm.h"  I*/

/* Threadcomm initialization and creation routines */
PETSC_EXTERN PetscErrorCode PetscThreadInit_NoThread();
PETSC_EXTERN PetscErrorCode PetscThreadPoolInit_NoThread(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_NoThread(PetscThreadComm);
#if defined(PETSC_HAVE_PTHREADCLASSES)
PETSC_EXTERN PetscErrorCode PetscThreadInit_PThread();
PETSC_EXTERN PetscErrorCode PetscThreadPoolInit_PThread(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_PThread(PetscThreadComm);
#endif
#if defined(PETSC_HAVE_OPENMP)
PETSC_EXTERN PetscErrorCode PetscThreadInit_OpenMP();
PETSC_EXTERN PetscErrorCode PetscThreadPoolInit_OpenMP(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_OpenMP(PetscThreadComm);
#endif
#if defined(PETSC_HAVE_TBB)
PETSC_EXTERN PetscErrorCode PetscThreadInit_TBB();
PETSC_EXTERN PetscErrorCode PetscThreadPoolInit_TBB(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_TBB(PetscThreadComm);
#endif

/* Threadcomm model initialization routines */
PETSC_EXTERN PetscErrorCode PetscThreadCommInitModel_Loop(PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadCommInitModel_Auto(PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadCommInitModel_User(PetscThreadComm);

/* Variables to track model/type registration */
PETSC_EXTERN PetscBool PetscThreadCommRegisterAllModelsCalled;
PETSC_EXTERN PetscBool PetscThreadCommRegisterAllTypesCalled;

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRegisterAllModels"
/*@C
   PetscThreadCommRegisterAllModels - Registers all of the threadpool models

   Not Collective

   Level: advanced

   Notes:
   Function list is destroyed in PetscThreadCommFinalizePackage.

.keywords: PetscThreadComm, register, all
@*/
PetscErrorCode PetscThreadCommRegisterAllModels(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscThreadCommRegisterAllModelsCalled = PETSC_TRUE;

  ierr = PetscFunctionListAdd(&PetscThreadCommModelList,LOOP,PetscThreadCommInitModel_Loop);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscThreadCommModelList,AUTO,PetscThreadCommInitModel_Auto);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscThreadCommModelList,USER,PetscThreadCommInitModel_User);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRegisterAllTypes"
/*@C
   PetscThreadCommRegisterAllTypes - Registers all of the threadpool models

   Not Collective

   Level: advanced

   Notes:
   Function list is destroyed in PetscThreadCommFinalizePackage.

.keywords: PetscThreadComm, register, all
@*/
PetscErrorCode PetscThreadCommRegisterAllTypes()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscThreadCommRegisterAllTypesCalled = PETSC_TRUE;

  ierr = PetscFunctionListAdd(&PetscThreadTypeList,NOTHREAD,PetscThreadInit_NoThread);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscThreadPoolTypeList,NOTHREAD,PetscThreadPoolInit_NoThread);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscThreadCommTypeList,NOTHREAD,PetscThreadCommInit_NoThread);CHKERRQ(ierr);

#if defined(PETSC_HAVE_PTHREADCLASSES)
  ierr = PetscFunctionListAdd(&PetscThreadTypeList,PTHREAD,PetscThreadInit_PThread);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscThreadPoolTypeList,PTHREAD,PetscThreadPoolInit_PThread);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscThreadCommTypeList,PTHREAD,PetscThreadCommInit_PThread);CHKERRQ(ierr);
#endif

#if defined(PETSC_HAVE_OPENMP)
  ierr = PetscFunctionListAdd(&PetscThreadTypeList,OPENMP,PetscThreadInit_OpenMP);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscThreadPoolTypeList,OPENMP,PetscThreadPoolInit_OpenMP);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscThreadCommTypeList,OPENMP,PetscThreadCommInit_OpenMP);CHKERRQ(ierr);
#endif

#if defined(PETSC_HAVE_TBB)
  ierr = PetscFunctionListAdd(&PetscThreadTypeList,TBB,PetscThreadInit_TBB);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscThreadPoolTypeList,TBB,PetscThreadPoolInit_TBB);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscThreadCommTypeList,TBB,PetscThreadCommInit_OpenMP);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

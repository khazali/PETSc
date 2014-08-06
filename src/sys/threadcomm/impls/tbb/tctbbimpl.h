#if !defined(__TCTBBIMPLH)
#define __TCTBBIMPLH

#include <petsc-private/threadcommimpl.h>
#include "tbb/tbb.h"

PETSC_EXTERN PetscErrorCode PetscThreadInit_TBB();
PETSC_EXTERN PetscErrorCode PetscThreadPoolInit_TBB(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_TBB(PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel_TBB(PetscThreadComm,PetscThreadCommJobCtx);
PETSC_EXTERN PetscErrorCode PetscThreadCreate_TBB(PetscThread);
PETSC_EXTERN PetscErrorCode PetscThreadCommInitialize_TBB(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommSetAffinity_TBB(PetscThreadPool,PetscThread);
PETSC_EXTERN PetscErrorCode PetscThreadPoolDestroy_TBB(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadLockCreate_TBB(void**);
PETSC_EXTERN PetscErrorCode PetscThreadLockDestroy_TBB(void**);
PETSC_EXTERN PetscErrorCode PetscThreadLockAcquire_TBB(void*);
PETSC_EXTERN PetscErrorCode PetscThreadLockRelease_TBB(void*);

#endif


#if !defined(__NOTHREADIMPLH)
#define __NOTHREADIMPLH

#include <petsc-private/threadcommimpl.h>

PETSC_EXTERN PetscErrorCode PetscThreadInit_NoThread();
PETSC_EXTERN PetscErrorCode PetscThreadPoolInit_NoThread(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_NoThread(PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadLockAcquire_NoThread(void *lock);
PETSC_EXTERN PetscErrorCode PetscThreadLockRelease_NoThread(void *lock);

#endif

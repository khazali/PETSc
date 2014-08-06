
#if !defined(__NOTHREADIMPLH)
#define __NOTHREADIMPLH

#include <petsc-private/threadcommimpl.h>

PETSC_EXTERN PetscErrorCode PetscThreadInit_NoThread();
PETSC_EXTERN PetscErrorCode PetscThreadPoolInit_NoThread(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_NoThread(PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadLockCreate_NoThread(void**);
PETSC_EXTERN PetscErrorCode PetscThreadLockDestroy_NoThread(void**);
PETSC_EXTERN PetscErrorCode PetscThreadLockAcquire_NoThread(void*);
PETSC_EXTERN PetscErrorCode PetscThreadLockRelease_NoThread(void*);

#endif

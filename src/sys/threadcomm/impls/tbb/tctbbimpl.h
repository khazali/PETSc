#if !defined(__TCTBBIMPLH)
#define __TCTBBIMPLH

#include <petsc-private/threadcommimpl.h>
#include "tbb/tbb.h"

//typedef spin_mutex LockType;

typedef struct _p_PetscThreadLock_TBB {
  //LockType lock; /* lock for tbb routines */
};
typedef struct _p_PetscThreadLock_TBB *PetscThreadLock_TBB;

PETSC_EXTERN PetscErrorCode PetscThreadInit_TBB();
PETSC_EXTERN PetscErrorCode PetscThreadPoolInit_TBB(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_TBB(PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel_TBB(PetscThreadComm,PetscThreadCommJobCtx);
PETSC_EXTERN PetscErrorCode PetscThreadLockInitialize_TBB(void);
PETSC_EXTERN PetscErrorCode PetscThreadLockAcquire_TBB(void *lock);
PETSC_EXTERN PetscErrorCode PetscThreadLockRelease_TBB(void *lock);

#endif

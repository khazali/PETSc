
#if !defined(__TCOPENMPIMPLH)
#define __TCOPENMPIMPLH

#include <petsc-private/threadcommimpl.h>
#include <omp.h>

struct _p_PetscThreadComm_OpenMP {
  PetscInt  barrier_threads;
  PetscBool wait_inc, wait_dec;
};
typedef struct _p_PetscThreadComm_OpenMP *PetscThreadComm_OpenMP;

typedef omp_lock_t *PetscThreadLock_OpenMP;

PETSC_EXTERN PetscErrorCode PetscThreadPoolInit_OpenMP(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_OpenMP(PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel_OpenMPLoop(PetscThreadComm,PetscThreadCommJobCtx);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel_OpenMPUser(PetscThreadComm,PetscThreadCommJobCtx);
PETSC_EXTERN PetscErrorCode PetscThreadCommBarrier_OpenMP(PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadLockInitialize_OpenMP(void);
PETSC_EXTERN PetscErrorCode PetscThreadLockAcquire_OpenMP(void*);
PETSC_EXTERN PetscErrorCode PetscThreadLockRelease_OpenMP(void*);

#endif


#if !defined(__TCOPENMPIMPLH)
#define __TCOPENMPIMPLH

#include <petsc-private/threadcommimpl.h>

struct _p_PetscThreadComm_OpenMP {
  PetscInt  barrier_threads;
  PetscBool wait_inc, wait_dec;
};
typedef struct _p_PetscThreadComm_OpenMP *PetscThreadComm_OpenMP;

PETSC_EXTERN PetscErrorCode PetscThreadCommInit_OpenMP(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_OpenMP(PetscThreadComm);
extern PetscErrorCode PetscThreadCommRunKernel_OpenMPLoop(PetscThreadComm,PetscThreadCommJobCtx);
extern PetscErrorCode PetscThreadCommRunKernel_OpenMPUser(PetscThreadComm,PetscThreadCommJobCtx);
extern PetscErrorCode PetscThreadCommBarrier_OpenMP(PetscThreadComm);
extern PetscErrorCode PetscThreadCommGetCores_OpenMP(PetscThreadComm,PetscInt,PetscInt*);

#endif

#if !defined(__TCTBBIMPLH)
#define __TCTBBIMPLH

#include <petsc-private/threadcommimpl.h>

PETSC_EXTERN PetscErrorCode PetscThreadPoolInit_TBB(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_TBB(PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel_TBB(PetscThreadComm,PetscThreadCommJobCtx);

#endif

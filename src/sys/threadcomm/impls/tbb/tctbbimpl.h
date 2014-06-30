#if !defined(__TCTBBIMPLH)
#define __TCTBBIMPLH

#include <petsc-private/threadcommimpl.h>

PETSC_EXTERN PetscErrorCode PetscThreadCommInit_TBB(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_TBB(PetscThreadComm);
extern PetscErrorCode PetscThreadCommRunKernel_TBB(PetscThreadComm,PetscThreadCommJobCtx);

#endif

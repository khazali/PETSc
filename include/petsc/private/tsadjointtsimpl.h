#ifndef __TSADJOINTTSIMPL_H
#define __TSADJOINTTSIMPL_H

#include <petscts.h>

PETSC_EXTERN PetscErrorCode TSCreateAdjointTS(TS,TS*);
PETSC_EXTERN PetscErrorCode AdjointTSComputeInitialConditions(TS,Vec,PetscBool,PetscBool);
PETSC_EXTERN PetscErrorCode AdjointTSSetQuadratureVec(TS,Vec);
PETSC_EXTERN PetscErrorCode AdjointTSSetDesignVec(TS,Vec);
PETSC_EXTERN PetscErrorCode AdjointTSSetDirectionVec(TS,Vec);
PETSC_EXTERN PetscErrorCode AdjointTSSetTLMTSAndFOATS(TS,TS,TS);
PETSC_EXTERN PetscErrorCode AdjointTSSetTimeLimits(TS,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode AdjointTSEventHandler(TS);
PETSC_EXTERN PetscErrorCode AdjointTSFinalizeQuadrature(TS);

/* Check sanity of the AdjointTS */
#if !defined(PETSC_USE_DEBUG)
#define PetscCheckAdjointTS(a) do {} while (0)
#else
#define PetscCheckAdjointTS(a)                                                                                                                 \
  do {                                                                                                                                         \
    PetscErrorCode __ierr;                                                                                                                     \
    PetscContainer __c;                                                                                                                        \
    void *__ac,*__cc;                                                                                                                          \
    __ierr = TSGetApplicationContext((a),(void*)&__ac);CHKERRQ(__ierr);                                                                        \
    __ierr = PetscObjectQuery((PetscObject)(a),"_ts_adjctx",(PetscObject*)&__c);CHKERRQ(__ierr);                                               \
    if (!__c) SETERRQ(PetscObjectComm((PetscObject)(a)),PETSC_ERR_USER,"The TS was not obtained from calling TSCreateAdjointTS()");            \
    __ierr = PetscContainerGetPointer(__c,(void**)&__cc);CHKERRQ(__ierr);                                                                      \
    if (__cc != __ac) SETERRQ(PetscObjectComm((PetscObject)(a)),PETSC_ERR_USER,"You cannot change the application context for the AdjointTS"); \
  } while (0)
#endif

#endif

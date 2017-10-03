#ifndef __TSTLMTSIMPL_H
#define __TSTLMTSIMPL_H

#include <petscts.h>
PETSC_INTERN PetscErrorCode TSCreateTLMTS(TS,TS*);
PETSC_INTERN PetscErrorCode TLMTSGetRHSVec(TS,Vec*);
PETSC_INTERN PetscErrorCode TLMTSSetPerturbationVec(TS,Vec);
PETSC_INTERN PetscErrorCode TLMTSSetDesignVec(TS,Vec);
PETSC_INTERN PetscErrorCode TLMTSGetDesignVec(TS,Vec*);
PETSC_INTERN PetscErrorCode TLMTSGetModelTS(TS,TS*);
#endif

#ifndef __TSADJOINTTSIMPL_H
#define __TSADJOINTTSIMPL_H

#include <petscts.h>
PETSC_INTERN PetscErrorCode TSCreateAdjointTS(TS,TS*);
PETSC_INTERN PetscErrorCode AdjointTSComputeInitialConditions(TS,PetscReal,Vec,PetscBool,PetscBool);
PETSC_INTERN PetscErrorCode AdjointTSSetGradientVec(TS,Vec);
PETSC_INTERN PetscErrorCode AdjointTSSetDesignVec(TS,Vec);
PETSC_INTERN PetscErrorCode AdjointTSSetDirectionVec(TS,Vec);
PETSC_INTERN PetscErrorCode AdjointTSSetTLMTSAndFOATS(TS,TS,TS);
PETSC_INTERN PetscErrorCode AdjointTSSetTimeLimits(TS,PetscReal,PetscReal);
PETSC_INTERN PetscErrorCode AdjointTSEventHandler(TS);
PETSC_INTERN PetscErrorCode AdjointTSComputeFinalGradient(TS);
#endif

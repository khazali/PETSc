#ifndef __TSHISTORYIMPL_H
#define __TSHISTORYIMPL_H

#include <petsc/private/tsimpl.h>

struct _p_TSHistory {
  PETSCHEADER(PetscOps);

  PetscReal *hist;    /* time history */
  PetscInt  *hist_id; /* stores the stepid in time history */
  PetscInt  n;        /* current number of steps registered */
  PetscBool sorted;   /* if the history is sorted in ascending order */
  PetscInt  c;        /* current capacity of hist */
  PetscInt  s;        /* reallocation size */
};

PETSC_INTERN PetscErrorCode TSHistoryCreate(MPI_Comm,TSHistory*);
PETSC_INTERN PetscErrorCode TSHistoryDestroy(TSHistory*);
PETSC_INTERN PetscErrorCode TSHistorySetHistory(TSHistory,PetscInt,PetscReal[]);
PETSC_INTERN PetscErrorCode TSHistoryGetLocFromTime(TSHistory,PetscReal,PetscInt*);
PETSC_INTERN PetscErrorCode TSHistoryUpdate(TSHistory,PetscInt,PetscReal);
PETSC_INTERN PetscErrorCode TSHistoryGetTimeStep(TSHistory,PetscBool,PetscInt,PetscReal*);
PETSC_INTERN PetscErrorCode TSHistoryGetNumSteps(TSHistory,PetscInt*);
#endif

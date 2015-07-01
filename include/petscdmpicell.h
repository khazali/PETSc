/*
  DMPlex, for parallel unstructured distributed mesh problems.
*/
#if !defined(__PETSCDMPICELL_H)
#define __PETSCDMPICELL_H

#include <petscdm.h>
#include <petscdt.h>
#include <petscfe.h>
#include <petscfv.h>
#include <petscsftypes.h>

PETSC_EXTERN PetscErrorCode DMPICellAddDensity(DM, double, double, double, PetscReal);
PETSC_EXTERN PetscErrorCode DMPICellGetGrad(DM, double, double, double, double[]);

#endif

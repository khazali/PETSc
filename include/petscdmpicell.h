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

PETSC_EXTERN PetscErrorCode DMPICellAddSource(DM, Vec, Vec, PetscInt);
PETSC_EXTERN PetscErrorCode DMPICellGetJet(DM, Vec, PetscInt, Vec);

#endif

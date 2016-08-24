#if !defined(_PICELLIMPL_H)
#define _PICELLIMPL_H

#include <petscmat.h>       /*I      "petscmat.h"          I*/
#include <petscsnes.h>       /*I      "petscmat.h"          I*/
#include <petscdmplex.h> /*I      "petscdmplex.h"    I*/
#include <petscdmpicell.h> /*I      "petscdmpicell.h"    I*/
#include <petscbt.h>
#include <petscsf.h>
#include <petsc/private/dmimpl.h>
#include <petsc/private/isimpl.h>     /* for inline access to atlasOff */
#include <../src/sys/utils/hash.h>

PETSC_EXTERN PetscLogEvent DMPICell_Solve, DMPICell_SetUp, DMPICell_AddSource, DMPICell_LocateProcess, DMPICell_GetJet, DMPICell_Add1, DMPICell_Add2, DMPICell_Add3;

typedef struct {
  DM   dmgrid;
  DM   dmplex;
  Vec  phi;
  Vec  rho;
  SNES snes;
  PetscFE fem;
  PetscInt debug;   /* The debugging level */
} DM_PICell;

#endif /* _PICELLIMPL_H */

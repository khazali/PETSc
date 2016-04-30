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

PETSC_EXTERN PetscLogEvent DMPICELL_Add, DMPICELL_GetGrad, DMPICELL_Create;

typedef struct {
  DM   dmgrid;
  DM   dmplex;
  Vec  phi;
  Vec  rho;
  /* Vec grad; */
  SNES snes;
  PetscFE fem;
} DM_PICell;

#endif /* _PICELLIMPL_H */

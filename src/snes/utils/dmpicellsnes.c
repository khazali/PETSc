#include <petsc/private/dmpicellimpl.h>    /*I   "petscdmpicell.h"   I*/

/************************** DMPICellSolve *******************************/

#undef __FUNCT__
#define __FUNCT__ "DMPICellSolve"
PetscErrorCode DMPICellSolve(DM dm)
{
  DM_PICell      *dmpi = (DM_PICell *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscLogEventBegin(DMPICell_Solve,dm,0,0,0);CHKERRQ(ierr);

  /* solve for potential and zero density for next solve */
  ierr = SNESSolve(dmpi->snes, dmpi->rho, dmpi->phi);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(DMPICell_Solve,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(DMPICell_Solve,dm,0,0,0);CHKERRQ(ierr);
#endif
  ierr = VecZeroEntries(dmpi->phi);CHKERRQ(ierr);
  /* solve for potential */
  ierr = SNESSolve(dmpi->snes, dmpi->rho, dmpi->phi);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(DMPICell_Solve,dm,0,0,0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

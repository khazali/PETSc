#include <petsc/private/snesimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MSNESComputeUpdate_NRichardson"
/*
  MSNESComputeUpdate_NRichardson - Searches along the direction of steepest descent.

  Input Parameters:
+ snes - the SNES context
. X    - the current solution
- F    - the current residual

  Output Parameters:
+ X - the new solution X' = X + dX
- Y - the update direction Y = dX

  TODO REF Richardson1911
*/
PetscErrorCode MSNESComputeUpdate_NRichardson(SNES snes, Vec X, Vec F, Vec Y)
{
  SNESLineSearchReason lsresult;
  PetscReal            fnorm;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  /* TODO Either remove residual calculation from line search, or signal to SNES that we have already calculated F/fnorm */
  ierr = VecCopy(F, Y);CHKERRQ(ierr);
  ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
  ierr = SNESLineSearchApply(snes->linesearch, X, F, &fnorm, F);CHKERRQ(ierr);
  ierr = SNESLineSearchGetReason(snes->linesearch, &lsresult);CHKERRQ(ierr);
  if (lsresult && (++snes->numFailures >= snes->maxFailures)) {
    snes->reason = SNES_DIVERGED_LINE_SEARCH;
    if (snes->pc && snes->pc->jacobian) {
      PetscBool ismin;

      ierr = MSNESCheckLocalMin_Internal(snes->pc, snes->pc->jacobian, F, &ismin);CHKERRQ(ierr);
      if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
    }
  }
  PetscFunctionReturn(0);
}

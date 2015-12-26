#include <petsc/private/snesimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MSNESComputeUpdate_Newton"
/*
  MSNESComputeUpdate_Newton - Solves the Newton linear system, perhaps inexactly.

  Input Parameters:
+ snes - the SNES context
. X    - the current solution
- F    - the current residual

  Output Parameters:
+ X - the new solution X' = X + dX
- Y - the update direction Y = dX

  TODO REF Dembo, Eisenstat, Steihaug
*/
PetscErrorCode MSNESComputeUpdate_Newton(SNES snes, Vec X, Vec F, Vec Y)
{
  PetscInt       lits;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Solve J Y = F, where J is Jacobian matrix */
  ierr = SNESComputeJacobian(snes, X, snes->jacobian, snes->jacobian_pre);CHKERRQ(ierr);
  ierr = KSPSetOperators(snes->ksp, snes->jacobian, snes->jacobian_pre);CHKERRQ(ierr);
  ierr = KSPSolve(snes->ksp, F, Y);CHKERRQ(ierr);
  SNESCheckKSPSolve(snes);
  /* Update generic SNES variables */
  ierr = PetscObjectSAWsTakeAccess((PetscObject) snes);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(snes->ksp, &lits);CHKERRQ(ierr);
  snes->linear_its += lits;
  ierr = PetscObjectSAWsGrantAccess((PetscObject) snes);CHKERRQ(ierr);
  ierr = PetscInfo2(snes, "iter=%D, linear solve iterations=%D\n", snes->iter, lits);CHKERRQ(ierr);
  if (PetscLogPrintInfo) {ierr = MSNESCheckConsistency_Internal(snes, snes->jacobian, F, Y);CHKERRQ(ierr);}
  /* Compute new solution */
  ierr = VecAXPY(X, -1.0, Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

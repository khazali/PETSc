#include <petsc/private/snesimpl.h>      /*I "petscsnes.h"  I*/
#include <petscblaslapack.h>

/* Logging support */
PetscLogEvent MSNES_SolUpdate, MSNES_Restart;

#undef __FUNCT__
#define __FUNCT__ "MSNESCheckLocalMin_Internal"
/*
  MSNESCheckLocalMin_Internal - Checks if J^T F = 0, which implies we've found a local minimum of the norm of the function || F(u) ||_2 but not a zero, F(u) = 0.

  Input Parameters:
+ snes - the SNES
. J    - the Jacobian
- F    - the residual

  Output Parameters:
. ismin - PETSC_TRUE if we are at a local minimum

  Note:
  In the case when one cannot compute J^T F we use the fact that 0 = (J^T F)^T W = F^T J W iff W not in the null space of J.
  Thanks for Jorge More for this trick. One assumes that the probability that a random W is in the null space of J is very small.

  Level: developer
*/
PetscErrorCode MSNESCheckLocalMin_Internal(SNES snes, Mat J, Vec F, PetscBool *ismin)
{
  Vec            W;
  PetscReal      fnorm, a1;
  PetscBool      hastranspose;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *ismin = PETSC_FALSE;
  ierr = MatHasOperation(J, MATOP_MULT_TRANSPOSE, &hastranspose);CHKERRQ(ierr);
  ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
  ierr = VecDuplicate(F, &W);CHKERRQ(ierr);
  if (hastranspose) {
    /* Compute || J^T F|| */
    ierr = MatMultTranspose(J, F, W);CHKERRQ(ierr);
    ierr = VecNorm(W, NORM_2, &a1);CHKERRQ(ierr);
    a1  /= fnorm;
    ierr = PetscInfo1(snes, "|| J^T F||/||F|| %14.12e near zero implies found a local minimum\n", (double) a1);CHKERRQ(ierr);
    if (a1 < 1.e-4) *ismin = PETSC_TRUE;
  } else {
    Vec         work;
    PetscScalar result;
    PetscReal   wnorm;

    ierr = VecSetRandom(W, NULL);CHKERRQ(ierr);
    ierr = VecNorm(W, NORM_2, &wnorm);CHKERRQ(ierr);
    ierr = VecDuplicate(W, &work);CHKERRQ(ierr);
    ierr = MatMult(J, W, work);CHKERRQ(ierr);
    ierr = VecDot(F, work, &result);CHKERRQ(ierr);
    ierr = VecDestroy(&work);CHKERRQ(ierr);
    a1   = PetscAbsScalar(result)/(fnorm*wnorm);
    ierr = PetscInfo1(snes, "(F^T J random)/(|| F ||*||random||) %14.12e near zero implies found a local minimum\n", (double) a1);CHKERRQ(ierr);
    if (a1 < 1.e-4) *ismin = PETSC_TRUE;
  }
  ierr = VecDestroy(&W);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MSNESCheckConsistency_Internal"
/*
  MSNESCheckConsistency_Internal = Checks for consistency of the rhs in the linearized equation

  Input Parameters:
+ snes - the SNES
. J    - the Jacobian
. F    - the current residual
- X    - the current residual

  Note: This checks if J^T (F - J*X) = 0, which would indicate an inconsistent system.

  Level: developer
*/
PetscErrorCode MSNESCheckConsistency_Internal(SNES snes, Mat J, Vec F, Vec X)
{
  PetscReal      a1, a2;
  PetscBool      hastranspose;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatHasOperation(J, MATOP_MULT_TRANSPOSE, &hastranspose);CHKERRQ(ierr);
  if (hastranspose) {
    Vec W1, W2;

    ierr = VecDuplicate(F, &W1);CHKERRQ(ierr);
    ierr = VecDuplicate(F, &W2);CHKERRQ(ierr);
    ierr = MatMult(J, X, W1);CHKERRQ(ierr);
    ierr = VecAXPY(W1, -1.0, F);CHKERRQ(ierr);

    /* Compute || J^T W|| */
    ierr = MatMultTranspose(J, W1, W2);CHKERRQ(ierr);
    ierr = VecNorm(W1, NORM_2, &a1);CHKERRQ(ierr);
    ierr = VecNorm(W2, NORM_2, &a2);CHKERRQ(ierr);
    if (a1 != 0.0) {ierr = PetscInfo1(snes, "||J^T(F-Ax)||/||F-AX|| %14.12e near zero implies inconsistent rhs\n", (double) (a2/a1));CHKERRQ(ierr);}
    ierr = VecDestroy(&W1);CHKERRQ(ierr);
    ierr = VecDestroy(&W2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MSNESComputeInitialResidual"
/*
  MSNESComputeInitialResidual - Compute the initial residual for a preconditioned nonlinear iteration

  Input Parameters:
+ snes - the SNES context
- X    - the current solution

  Output Parameters:
+ F     - the current residual
- fnorm - the norm of F

  Level: intermediate

.seealso: SNESSolve()
*/
PetscErrorCode MSNESComputeInitialResidual(SNES snes, Vec X, Vec F, PetscReal *fnorm)
{
  SNESConvergedReason reason;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* TODO Do I need (snes->pc && snes->pcside == PC_LEFT && snes->functype == SNES_FUNCTION_PRECONDITIONED)? */
  if (snes->functype == SNES_FUNCTION_PRECONDITIONED) {
    ierr = SNESApplyNPC(snes, X, NULL, F);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes->pc, &reason);CHKERRQ(ierr);
    if (reason < 0  && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
  } else {
    if (!snes->vec_func_init_set) {ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);}
    else snes->vec_func_init_set = PETSC_FALSE;
  }
  ierr = VecNorm(F, NORM_2, fnorm);CHKERRQ(ierr);
  SNESCheckFunctionNorm(snes, *fnorm);
  snes->minnorm = *fnorm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MSNESComputeNextResidual"
/*
  MSNESComputeNextResidual - Compute the next residual for a preconditioned nonlinear iteration

  Input Parameters:
+ snes - the SNES context
- X    - the current solution

  Output Parameters:
+ F     - the current residual
- fnorm - the norm of F

  Level: intermediate

.seealso: SNESSolve()
*/
PetscErrorCode MSNESComputeNextResidual(SNES snes, Vec X, Vec F, PetscReal *fnorm)
{
  SNESConvergedReason reason;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* TODO Should we allow approximate residuals here, such as
      ierr = VecCopy(FM, FA);CHKERRQ(ierr);
      ierr = VecScale(FA, 1.0 - alph_total);CHKERRQ(ierr);
      ierr = VecMAXPY(FA, l, beta, Fdot);CHKERRQ(ierr);
   */
  if (snes->functype == SNES_FUNCTION_PRECONDITIONED) {
    ierr = SNESApplyNPC(snes, X, NULL, F);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes->pc, &reason);CHKERRQ(ierr);
    if (reason < 0  && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
  } else {
    ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
  }
  ierr = VecNorm(F, NORM_2, fnorm);CHKERRQ(ierr);
  SNESCheckFunctionNorm(snes, *fnorm);
  snes->minnorm = PetscMin(snes->minnorm, *fnorm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MSNESApplyNPCRight"
/*
  MSNESApplyNPCRight - Computes the action of the NPC at each iteration

  Input Parameters:
+ snes - the SNES context
- X    - the current solution

  Output Parameters:
+ F     - the current residual
- fnorm - the norm of F

  Level: intermediate

.seealso: SNESSolve(), SNESComputeInitialResidual()
*/
PetscErrorCode MSNESApplyNPCRight(SNES snes, Vec X, Vec B, Vec F, PetscReal *fnorm)
{
  SNESConvergedReason reason;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (!snes->pc) PetscFunctionReturn(0);
  if (snes->pcside == PC_RIGHT) {
    ierr = SNESSetInitialFunction(snes->pc, F);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(SNES_NPCSolve, snes->pc, X, B, 0);CHKERRQ(ierr);
    ierr = SNESSolve(snes->pc, B, X);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(SNES_NPCSolve, snes->pc, X, B, 0);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes->pc, &reason);CHKERRQ(ierr);
    if (reason < 0  && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
    ierr = SNESGetNPCFunction(snes, F, fnorm);CHKERRQ(ierr);
  } else if (snes->pcside == PC_LEFT && snes->functype == SNES_FUNCTION_UNPRECONDITIONED) {
    /* If SNES_FUNCTION_PRECONDITIONED, residual will have been computed in the convergence check */
    ierr = SNESApplyNPC(snes, X, F, F);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes->pc, &reason);CHKERRQ(ierr);
    if (reason < 0  && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
    ierr = VecNorm(F, NORM_2, fnorm);CHKERRQ(ierr);
    SNESCheckFunctionNorm(snes, *fnorm);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MSNESComputeUpdate"
/*
  MSNESComputeUpdate - Compute the next residual for a preconditioned nonlinear iteration

  Input Parameters:
+ snes - the SNES context
. X    - the current solution
- F    - the current residual

  Output Parameters:
. Y     - the solution udpate

  Level: intermediate

.seealso: SNESSolve()
*/
PetscErrorCode MSNESComputeUpdate(SNES snes, Vec X, Vec F, Vec Y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2); PetscCheckSameComm(snes, 1, X, 2);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 3); PetscCheckSameComm(snes, 1, F, 3);
  PetscValidHeaderSpecific(Y, VEC_CLASSID, 4); PetscCheckSameComm(snes, 1, Y, 4);
  ierr = PetscLogEventBegin(MSNES_SolUpdate,snes,X,F,Y);CHKERRQ(ierr);
  ierr = (*snes->ops->solupdate)(snes, X, F, Y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MSNES_SolUpdate,snes,X,F,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MSNESRestart"
/*
  MSNESRestart - Restart the subspace if necessary

  Input Parameters:
+ snes - the SNES context
. restartCount - the previous number of times the restart conditions have been satisfied
. X    - the current solution
- F    - the current residual

  Output Parameters:
. restartCount - the number of times the restart conditions have been satisfied

  Note: This only applies to methods with an approximation subspace, such as Anderson Mixing or Nonlinear GMRES

  Level: intermediate

.seealso: SNESSolve()
*/
PetscErrorCode MSNESRestart(SNES snes, Vec X, Vec F, PetscInt *restartCount)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Will dispatch to impl types */
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2); PetscCheckSameComm(snes, 1, X, 2);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 3); PetscCheckSameComm(snes, 1, F, 3);
  PetscValidPointer(restartCount, 4);
  if (snes->ops->restart) {
    ierr = PetscLogEventBegin(MSNES_Restart,snes,X,F,0);CHKERRQ(ierr);
    ierr = (*snes->ops->restart)(snes, X, F, restartCount);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MSNES_Restart,snes,X,F,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MSNESSolve"
PetscErrorCode MSNESSolve(SNES snes)
{
  Vec            X            = snes->vec_sol;  /* solution vector X^n */
  Vec            Y            = snes->vec_sol_update;  /* solution update vector Y^n */
  Vec            F            = snes->vec_func; /* residual vector */
  Vec            B            = snes->vec_rhs;  /* rhs vector */
  const PetscInt maxits       = snes->max_its;  /* maximum number of Newton iterates */
  PetscInt       restartCount = 0;              /* The number of times the restart criteria have been satisfied */
  PetscReal      fnorm, xnorm, ynorm;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Update generic SNES variables */
  ierr = PetscObjectSAWsTakeAccess((PetscObject) snes);CHKERRQ(ierr);
  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;
  snes->iter                   = 0;
  snes->iterRestart            = 1;
  snes->norm                   = 0.0;
  ierr = PetscObjectSAWsGrantAccess((PetscObject) snes);CHKERRQ(ierr);
  /* Compute initial residual */
  ierr = MSNESComputeInitialResidual(snes, X, F, &fnorm);CHKERRQ(ierr);
  /* Update generic SNES variables */
  ierr = PetscObjectSAWsTakeAccess((PetscObject) snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr = PetscObjectSAWsGrantAccess((PetscObject) snes);CHKERRQ(ierr);
  ierr = SNESLogConvergenceHistory(snes, fnorm, 0);CHKERRQ(ierr);
  ierr = SNESMonitor(snes, snes->iter, fnorm);CHKERRQ(ierr);
  /* Test convergence */
  snes->ttol = fnorm*snes->rtol; /* TODO set parameter for default relative tolerance convergence test */
  ierr = (*snes->ops->converged)(snes, 0, 0.0, 0.0, fnorm, &snes->reason, snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);
  /* TODO fix minnorm */
  /* Outer Iteration */
  for (i = 1; i < maxits+1; i++) {
    /* Call general purpose update function */
    if (snes->ops->update) {ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);}
    /* Compute the NPC effect */
    ierr = MSNESApplyNPCRight(snes, X, B, F, &fnorm);CHKERRQ(ierr);
    /* Compute update and norms */
    ierr = MSNESComputeUpdate(snes, X, F, Y);CHKERRQ(ierr);
    if (snes->reason < 0) break;
    if (ynorm > 0.0 && snes->stol*xnorm > ynorm) {snes->reason = SNES_CONVERGED_SNORM_RELATIVE; break;}
    if (snes->nfuncs >= snes->max_funcs)         {snes->reason = SNES_DIVERGED_FUNCTION_COUNT;  break;}
    /* Compute new residual and norm */
    ierr = MSNESComputeNextResidual(snes, X, F, &fnorm);CHKERRQ(ierr);
    /* Restart or enlarge subspace */
    ierr = MSNESRestart(snes, X, F, &restartCount);CHKERRQ(ierr);
    /* Update generic SNES variables */
    ierr = PetscObjectSAWsTakeAccess((PetscObject) snes);CHKERRQ(ierr);
    snes->iter = i;
    snes->norm = fnorm;
    ierr = PetscObjectSAWsGrantAccess((PetscObject) snes);CHKERRQ(ierr);
    ierr = SNESLogConvergenceHistory(snes, snes->norm, snes->linear_its);CHKERRQ(ierr);
    ierr = SNESMonitor(snes, snes->iter, snes->norm);CHKERRQ(ierr);
    /* Test for convergence */
    ierr = VecNormBegin(Y, NORM_2, &ynorm);CHKERRQ(ierr);
    ierr = VecNormBegin(X, NORM_2, &xnorm);CHKERRQ(ierr);
    ierr = VecNormEnd(Y, NORM_2, &ynorm);CHKERRQ(ierr);
    ierr = VecNormEnd(X, NORM_2, &xnorm);CHKERRQ(ierr);
    ierr = (*snes->ops->converged)(snes, snes->iter, xnorm, ynorm, fnorm, &snes->reason, snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;
  }
  if (i == maxits+1) {
    ierr = PetscInfo1(snes, "Maximum number of iterations has been reached: %D\n", maxits);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}

#include <petsc-private/snesimpl.h>

/*
   Private context for an active set Newton line search method
   for solving system of mixed complementarity equations.
 */

#if !defined(__SNES_IMPL_NEWTONAS_H)
#define __SNES_IMPL_NEWTONAS_H

typedef struct {
  SNESNEWTONASType type;
  /* TODO: do we need vec_soln_lambda_update? */
  Vec              vec_sol_lambda,vec_sol_lambda_update; /* These vectors are part of the solver's state and are expected to reflect the current proposed solution (converged soln at the end of the solve). */
  /* KSP and Jacobians */
  Mat              Bb_pre;           /* preconditioning matrix for the basis of the active constraints (as columns) */
  Mat              Bbt_pre;          /* preconditioning matrix for the basis of the active constraints (as rows Bbt = Bb^T) */
  KSP              ksp_aug;          /* augmented system solver. */
} SNES_NEWTONAS;

#endif

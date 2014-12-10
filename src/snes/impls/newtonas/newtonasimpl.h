#include <petsc-private/snesimpl.h>

/*
   Private context for an active set Newton line search method
   for solving system of mixed complementarity equations.
 */

#if !defined(__SNES_IMPL_NEWTONAS_H)
#define __SNES_IMPL_NEWTONAS_H

typedef struct {
  SNESNEWTONASType type;
  Vec              vec_lambda, vec_lambda_update; /* These are part of the solver's state and are expected to reflect the current solution or the computed update, not some intermediate stage. */
  Vec              vec_soln_aug, vec_func_aug, vec_soln_update_aug;  /* Augmented vectors: solution, residual and solution update -- include both solution and the lagrange multipliers. */
  Mat              Bb_pre;       /* preconditioning matrix for the basis of the active constraints (as columns) */
  Mat              Bbt_pre;      /* preconditioning matrix for the basis of the active constraints (as rows Bbt = Bb^T) */
  VecScatter       aug_to_x;
  VecScatter       aug_to_lambda;
  PetscReal        merit;
  Mat              jacobian_aug; /* augmented Jacobian */
  /* Constraint-space reusable work vectors. */
  Vec              work_Bdx,work_lambda;
} SNES_NEWTONAS;

#endif

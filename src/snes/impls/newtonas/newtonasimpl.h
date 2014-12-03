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
  Vec              *workg;      /* constraint-related work vectors; same size as vec_constr or vec_lambda */
  Mat              Bb_pre;      /* preconditioning matrix for the basis of the active constraints (as columns) */
  Mat              Bbt_pre;     /* preconditioning matrix for the basis of the active constraints (as rows Bbt = Bb^T) */
  Vec              ls_x;
  Vec              ls_f;
  Vec              ls_step;
  VecScatter       scat_ls_to_x;
  VecScatter       scat_ls_to_lambda;

} SNES_NEWTONAS;

#endif

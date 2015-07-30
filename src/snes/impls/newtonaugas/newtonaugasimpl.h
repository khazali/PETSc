#include <petsc-private/snesimpl.h>

/*
   Private context for an active set Newton line search method
   for solving system of mixed complementarity equations.
 */

#if !defined(__SNES_IMPL_NEWTONAS_H)
#define __SNES_IMPL_NEWTONAUGAS_H

typedef struct {
  /* TODO: do we need vec_soln_lambda_update? */
  Vec              vec_sol_lambda,vec_sol_lambda_update; /* These vectors are part of the solver's state and are expected to reflect the current proposed solution (converged soln at the end of the solve). */
  /* SADDLE: KSP */
  Mat              Br,Brt;             /* reduced constraint matrix and its transpose */
  Mat              Bb_pre,Bbt_pre;     /* preconditioners for the basis columns of the constraint matrix and its transpose. */
  KSP              kspr_aug;           /* reduced augmented system solver. */
} SNES_NEWTONAUGAS;

#endif

#include <petsc-private/snesimpl.h>

/*
   Private context for an active set Newton line search method
   for solving system of mixed complementarity equations.
 */

#if !defined(__SNES_IMPL_NEWTONAS_H)
#define __SNES_IMPL_NEWTONAS_H

typedef struct {
  SNESNEWTONASType type;
  Vec              *lambda;    /* lambda and dlambda */
#if 0
  /* These _might_ be needed in SNESNEWTONASLinearUpdate_Private() */
  Vec              tdlambda;    /* \tilde \delta \lambda; this vec's size is changing depending on the active set */
  Vec              tdlambdarhs; /* rhs for the linearized constraint equation; is a zero vector of the same size as vec_tdlambda */
#endif
  Mat              Bb_pre;      /* preconditioning matrix for the basis of the active constraints (as columns) */
  Mat              Bbt_pre;     /* preconditioning matrix for the basis of the active constraints (as rows Bbt = Bb^T) */
} SNES_NEWTONAS;

#endif

#include <petsc-private/snesimpl.h>

/*
   Private context for an active set Newton line search method
   for solving system of mixed complementarity equations.
 */

#if !defined(__SNES_IMPL_NEWTONAS_H)
#define __SNES_IMPL_NEWTONAS_H

typedef struct {
  /* FIXME: do we need this stuff? */
  PetscErrorCode (*checkredundancy)(SNES,IS,IS*,void*);
  void             *ctxP;           /* user defined check redundancy context */
  IS               IS_inact_prev;

  SNESNEWTONASType type;

} SNES_NEWTONAS;

#endif

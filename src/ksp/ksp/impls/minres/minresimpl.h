/*
    Private Krylov Context Structure (KSP) for Minres and Minresqlp
*/

#if !defined(__MINRESIMPL_H)
#define __MINRESIMPL_H

/*
        Defines the basic KSP object
*/
#include <petsc-private/kspimpl.h>

typedef struct {
  PetscReal haptol;
  PetscBool monitor_arnorm;
  PetscReal Arnorm,relArnorm;
} KSP_MINRES;

PETSC_EXTERN PetscErrorCode KSPMINRESMonitor(KSP,PetscInt,PetscReal,void *);

#endif

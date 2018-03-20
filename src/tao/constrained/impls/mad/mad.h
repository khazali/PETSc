#ifndef __TAO_MAD_H
#define __TAO_MAD_H
#include <petsc/private/taoimpl.h>

/*
 Context for Multisecant Accelerated Descent
*/

typedef struct {
  Vec     *Y, *R, *dY, *dR;
  Vec     kkt, grad, h, g; /* kkt vector and its components */
  Vec     step, x, lambda, mu; /* step vector and its components */
  Vec     cin, cb, cl, cu;
  Vec     g_work, cin_work, cb_work, cl_work, cu_work;
  Vec     h_work;
  Vec     lb, ub;
  IS      lb_idx, ub_idx; /* index sets for bounded design variables */
  
  PetscReal   f;
  PetscReal   alpha, beta, gamma; /* parameters for step calculation */
  PetscInt    k, q, nd, nlb, nub, nb, nineq, nh, ng; /* counters for vector space sizes */
} TAO_MAD;

#endif /* ifndef __TAO_MAD_H */

PETSC_INTERN PetscErrorCode TaoMADInitCompositeVecs(Tao);
PETSC_INTERN PetscErrorCode TaoMADUpdateHistory(Tao);
PETSC_INTERN PetscErrorCode TaoMADComputeDiffMats(Tao);
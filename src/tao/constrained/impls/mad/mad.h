#ifndef __TAO_MAD_H
#define __TAO_MAD_H
#include <petsc/private/taoimpl.h>
#include <petsc.h>
#include <petscblaslapack.h>

#define LAPACKgesvd_   PETSCBLAS(gesvd,GESVD)

/*
 Context for Multisecant Accelerated Descent
*/

typedef struct {
  Vec     *Y, *R, *dY, *dR;
  Vec     kkt, grad, h, g, cin, cb, cl, cu; /* kkt vector and its components */
  Vec     kkt_work, x_work, h_work, g_work, cin_work, cb_work, cl_work, cu_work;
  Vec     sol, lambda, mu, mu_cin, mu_b, mu_lb, mu_ub; /* step vector and its components */
  Vec     step, dlambda, dmu, dmu_cin, dmu_b, dmu_lb, dmu_ub; /* step vector and its components */
  Vec     lb, ub;
  IS      lb_idx, ub_idx; /* index sets for bounded design variables */

  PetscReal   alpha, beta, max_step, svd_cutoff; /* parameters for step calculation */
  PetscInt    k, q, nkkt, nd, nlb, nub, nb, nineq, nh, ng; /* counters for vector space sizes */
} TAO_MAD;

#endif /* ifndef __TAO_MAD_H */

PETSC_EXTERN void LAPACKgesvd_(const char*, const char*, PetscBLASInt*, PetscBLASInt*, PetscScalar*, PetscBLASInt*, PetscScalar*, PetscScalar*, PetscBLASInt*, PetscScalar*, PetscBLASInt*, PetscScalar*, PetscBLASInt*, PetscBLASInt*);

PETSC_INTERN PetscErrorCode TaoMADInitVecs(Tao);
PETSC_INTERN PetscErrorCode TaoMADSetInitPoint(Tao);
PETSC_INTERN PetscErrorCode TaoMADUpdateHistory(Tao, Vec, Vec);
PETSC_INTERN PetscErrorCode TaoMADComputeDiffMats(Tao);
PETSC_INTERN PetscErrorCode TaoMADComputeKKT(Tao, PetscReal*, PetscReal*, PetscReal*);
PETSC_INTERN PetscErrorCode TaoMADDiffMatMult(Tao, Vec*, PetscScalar*, Vec);
PETSC_INTERN PetscErrorCode TaoMADDiffMatMultTrans(Tao, Vec*, Vec, PetscScalar*);
PETSC_INTERN PetscErrorCode TaoMADSolveSubproblem(Tao, PetscScalar*);
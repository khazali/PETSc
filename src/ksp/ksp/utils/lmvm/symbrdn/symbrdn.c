#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Limited-memory Symmetric Broyden method for approximating both 
  the forward product and inverse application of a Jacobian.
*/

typedef struct {
  Vec *P, *Q;
  Vec work;
  PetscBool allocated;
  PetscReal *stp, *ytq, *yts;
  PetscReal phi, *psi;
} Mat_SymBrdn;

/*------------------------------------------------------------*/

/*
  The solution method below is the matrix-free implementation of 
  Equation 18 in Dennis and Wolkowicz "Sizing and Least Change Secant 
  Methods" (http://www.caam.rice.edu/caam/trs/90/TR90-05.pdf).
  
  Q[i] = (B_i)^{-1}*S[i] terms are computed ahead of time whenever 
  the matrix is updated with a new (S[i], Y[i]) pair. This allows 
  repeated calls of MatSolve without incurring redundant computation.
  
  dX <- J0^{-1} * F
  
  for i=0,1,2,...,k
    # Q[i] = (B_i)^T{-1} Y[i]
    
    rho = 1.0 / (Y[i]^T S[i])
    alpha = rho * (S[i]^T F)
    zeta = 1.0 / (Y[i]^T Q[i])
    gamma = zeta * (Y[i]^T dX)
    
    dX <- dX - (gamma * Q[i]) + (alpha * Y[i])
    W <- (rho * S[i]) - (zeta * Q[i])
    dX <- dX + (psi[i] * (Y[i]^T Q[i]) * (W^T F) * W)
  end
*/
static PetscErrorCode MatSolve_LMVMSymBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i;
  PetscReal         ytx, stf, wtf; 
  
  PetscFunctionBegin;
  /* Efficient shortcuts for pure BFGS and pure DFP configurations */
  if (lsb->phi == 0.0) {
    ierr = MatSolve_LMVMBFGS(B, F, dX);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (lsb->phi == 1.0) {
    ierr = MatSolve_LMVMDFP(B, F, dX);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  
  PetscValidHeaderSpecific(F, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(dX, VEC_CLASSID, 3);
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);

  /* Start the outer iterations for ((B^{-1}) * dX) */
  ierr = MatLMVMApplyJ0Inv(B, F, dX);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k; ++i) {
    /* Compute the necessary dot products -- store yTs and yTp for inner iterations later */
    ierr = VecDotBegin(lmvm->Y[i], dX, &ytx);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->S[i], F, &stf);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], dX, &ytx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], F, &stf);CHKERRQ(ierr);
    /* Compute the pure DFP component */
    ierr = VecAXPBYPCZ(dX, -ytx/lsb->ytq[i], stf/lsb->yts[i], 1.0, lsb->Q[i], lmvm->S[i]);CHKERRQ(ierr);
    /* Tack on the convexly scaled extras */
    ierr = VecAXPBYPCZ(lsb->work, 1.0/lsb->yts[i], -1.0/lsb->ytq[i], 0.0, lmvm->S[i], lsb->Q[i]);CHKERRQ(ierr);
    ierr = VecDot(lsb->work, F, &wtf);CHKERRQ(ierr);
    ierr = VecAXPY(dX, lsb->psi[i]*lsb->ytq[i]*wtf, lsb->work);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*
  The forward-product below is the matrix-free implementation of 
  Equation 16 in Dennis and Wolkowicz "Sizing and Least Change Secant 
  Methods" (http://www.caam.rice.edu/caam/trs/90/TR90-05.pdf).
  
  P[i] = (B_i)*S[i] terms are computed ahead of time whenever 
  the matrix is updated with a new (S[i], Y[i]) pair. This allows 
  repeated calls of MatMult inside KSP solvers without unnecessarily 
  recomputing P[i] terms in expensive nested-loops.
  
  Z <- J0 * X
  
  for i=0,1,2,...,k
    # P[i] = (B_k) * S[i]
    
    rho = 1.0 / (Y[i]^T S[i])
    alpha = rho * (Y[i]^T F)
    zeta = 1.0 / (S[i]^T P[i])
    gamma = zeta * (S[i]^T dX)
    
    dX <- dX - (gamma * P[i]) + (alpha * S[i])
    W <- (rho * Y[i]) - (zeta * P[i])
    dX <- dX + (phi * (S[i]^T P[i]) * (W^T F) * W)
  end
*/
static PetscErrorCode MatMult_LMVMSymBrdn(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          i;
  PetscReal         stz, ytx, wtx;
  
  
  PetscFunctionBegin;
  /* Efficient shortcuts for pure BFGS and pure DFP configurations */
  if (lsb->phi == 0.0) {
    ierr = MatMult_LMVMBFGS(B, X, Z);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } 
  if (lsb->phi == 1.0) {
    ierr = MatMult_LMVMDFP(B, X, Z);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* Start the outer iterations for (B * X) */
  ierr = MatLMVMApplyJ0Fwd(B, X, Z);CHKERRQ(ierr);
  for (i = 0; i <= lmvm->k; ++i) {
    /* Compute the necessary dot products */
    ierr = VecDotBegin(lmvm->S[i], Z, &stz);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Y[i], X, &ytx);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->S[i], Z, &stz);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Y[i], X, &ytx);CHKERRQ(ierr);
    /* Compute the pure BFGS component */
    ierr = VecAXPBYPCZ(Z, -stz/lsb->stp[i], ytx/lsb->yts[i], 1.0, lsb->P[i], lmvm->Y[i]);CHKERRQ(ierr);
    /* Tack on the convexly scaled extras */
    ierr = VecAXPBYPCZ(lsb->work, 1.0/lsb->yts[i], -1.0/lsb->stp[i], 0.0, lmvm->Y[i], lsb->P[i]);CHKERRQ(ierr);
    ierr = VecDot(lsb->work, X, &wtx);CHKERRQ(ierr);
    ierr = VecAXPY(Z, (1.0 - lsb->phi)*lsb->stp[i]*wtx, lsb->work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatUpdate_LMVMSymBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscInt          old_k, i, j;
  PetscReal         curvature, yty, sjtpi, yjtsi, wtsi, yjtqi, sjtyi, wtyi, numer;
  Vec               Ptmp, Qtmp;

  PetscFunctionBegin;
  if (lmvm->m == 0) PetscFunctionReturn(0);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    ierr = VecAXPBY(lmvm->Xprev, 1.0, -1.0, X);CHKERRQ(ierr);
    ierr = VecAXPBY(lmvm->Fprev, 1.0, -1.0, F);CHKERRQ(ierr);
    /* Test if the updates can be accepted */
    ierr = VecDot(lmvm->Xprev, lmvm->Fprev, &curvature);CHKERRQ(ierr);
    if (curvature > -lmvm->eps) {
      /* Update is good, accept it */
      old_k = lmvm->k;
      ierr = MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev);CHKERRQ(ierr);
      if (old_k == lmvm->k) {
        for (i = 0; i <= lmvm->k-1; ++i) {
          lsb->yts[i] = lsb->yts[i+1];
        }
      }
      lsb->yts[lmvm->k] = curvature;
      if (lsb->phi != 1.0) {
        /* If we hit the memory limit, shift the P and Q vectors */
        if (old_k == lmvm->k) {
          Ptmp = lsb->P[0];
          for (i = 0; i <= lmvm->k-1; ++i) {
            lsb->P[i] = lsb->P[i+1];
            lsb->stp[i] = lsb->stp[i+1];
          }
          lsb->P[lmvm->k] = Ptmp;
        }
        /* Start the loops for (P[i] = (B_i) * S[i]) */
        for (i = 0; i <= lmvm->k; ++i) {
          ierr = MatLMVMApplyJ0Fwd(B, lmvm->S[i], lsb->P[i]);CHKERRQ(ierr);
          for (j = 0; j <= i-1; ++j) {
            /* Compute the necessary dot products */
            ierr = VecDotBegin(lmvm->S[j], lsb->P[i], &sjtpi);CHKERRQ(ierr);
            ierr = VecDotBegin(lmvm->Y[j], lmvm->S[i], &yjtsi);CHKERRQ(ierr);
            ierr = VecDotEnd(lmvm->S[j], lsb->P[i], &sjtpi);CHKERRQ(ierr);
            ierr = VecDotEnd(lmvm->Y[j], lmvm->S[i], &yjtsi);CHKERRQ(ierr);
            /* Compute the pure BFGS component of the forward product */
            ierr = VecAXPBYPCZ(lsb->P[i], -sjtpi/lsb->stp[j], yjtsi/lsb->yts[j], 1.0, lsb->P[j], lmvm->Y[j]);CHKERRQ(ierr);
            /* Tack on the convexly scaled extras to the forward product */
            if (lsb->phi > 0.0) {
              ierr = VecAXPBYPCZ(lsb->work, 1.0/lsb->yts[j], -1.0/lsb->stp[j], 0.0, lmvm->Y[j], lsb->P[j]);CHKERRQ(ierr);
              ierr = VecDot(lsb->work, lmvm->S[i], &wtsi);CHKERRQ(ierr);
              ierr = VecAXPY(lsb->P[i], lsb->phi*lsb->stp[i]*wtsi, lsb->work);CHKERRQ(ierr);
            }
          }
          ierr = VecDot(lmvm->S[i], lsb->P[i], &lsb->stp[i]);CHKERRQ(ierr);
        }
      }
      if (lsb->phi != 0.0) {
        /* If we hit the memory limit, shift the P and Q vectors */
        if (old_k == lmvm->k) {
          Qtmp = lsb->Q[0];
          for (i = 0; i <= lmvm->k-1; ++i) {
            lsb->Q[i] = lsb->Q[i+1];
            lsb->ytq[i] = lsb->ytq[i+1];
          }
          lsb->Q[lmvm->k] = Qtmp;
        }
        /* Start the loop for (Q[k] = (B_k)^{-1} * Y[k]) */
        for (i = 0; i <= lmvm->k; ++i) {
          ierr = MatLMVMApplyJ0Inv(B, lmvm->Y[i], lsb->Q[i]);CHKERRQ(ierr);
          for (j = 0; j <= i-1; ++j) {
            /* Compute the necessary dot products */
            ierr = VecDotBegin(lmvm->Y[j], lsb->Q[i], &yjtqi);CHKERRQ(ierr);
            ierr = VecDotBegin(lmvm->S[j], lmvm->Y[i], &sjtyi);CHKERRQ(ierr);
            ierr = VecDotEnd(lmvm->Y[j], lsb->Q[i], &yjtqi);CHKERRQ(ierr);
            ierr = VecDotEnd(lmvm->S[j], lmvm->Y[i], &sjtyi);CHKERRQ(ierr);
            /* Compute the pure DFP component of the inverse application*/
            ierr = VecAXPBYPCZ(lsb->Q[i], -yjtqi/lsb->ytq[j], sjtyi/lsb->yts[j], 1.0, lsb->Q[j], lmvm->S[j]);CHKERRQ(ierr);
            /* Tack on the convexly scaled extras to the inverse application*/
            if (lsb->psi[j] > 0.0) {
              ierr = VecAXPBYPCZ(lsb->work, 1.0/lsb->yts[j], -1.0/lsb->ytq[j], 0.0, lmvm->S[j], lsb->Q[j]);CHKERRQ(ierr);
              ierr = VecDot(lsb->work, lmvm->Y[i], &wtyi);CHKERRQ(ierr);
              ierr = VecAXPY(lsb->Q[i], lsb->psi[j]*lsb->ytq[j]*wtyi, lsb->work);CHKERRQ(ierr);
            }
          }
          ierr = VecDot(lmvm->Y[i], lsb->Q[i], &lsb->ytq[i]);CHKERRQ(ierr);
          if (lsb->phi == 0.0) {
            lsb->psi[i] = 0.0;
          } else {
            numer = (1.0 - lsb->phi)*lsb->yts[i]*lsb->yts[i];
            lsb->psi[i] = numer / (numer + (lsb->phi*lsb->ytq[i]*lsb->stp[i]));
          }
        }
      }
      /* Update default J0 scaling */
      ierr = VecDot(lmvm->Y[lmvm->k], lmvm->Y[lmvm->k], &yty);CHKERRQ(ierr);
      lmvm->J0default = yty/curvature;
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
    }
  }

  /* Save the solution and function to be used in the next update */
  ierr = VecCopy(X, lmvm->Xprev);CHKERRQ(ierr);
  ierr = VecCopy(F, lmvm->Fprev);CHKERRQ(ierr);
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatReset_LMVMSymBrdn(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = PetscMemzero(lsb->psi, lmvm->m);CHKERRQ(ierr);
  if (destructive && lsb->allocated) {
    ierr = VecDestroy(&lsb->work);CHKERRQ(ierr);
    ierr = PetscFree4(lsb->stp, lsb->ytq, lsb->yts, lsb->psi);CHKERRQ(ierr);
    if (lmvm->m > 0) {
      if (lsb->phi != 1.0) {
        ierr = VecDestroyVecs(lmvm->m, &lsb->P);CHKERRQ(ierr);
      }
      if (lsb->phi != 0.0) {
        ierr = VecDestroyVecs(lmvm->m, &lsb->Q);CHKERRQ(ierr);
      }
    }
    lsb->allocated = PETSC_FALSE;
  }
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMSymBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!lsb->allocated) {
    ierr = VecDuplicate(X, &lsb->work);CHKERRQ(ierr);
    ierr = PetscMalloc3(lmvm->m, &lsb->stp, lmvm->m, &lsb->ytq, lmvm->m, &lsb->yts);CHKERRQ(ierr);
    ierr = PetscCalloc1(lmvm->m, &lsb->psi);CHKERRQ(ierr);
    if (lmvm->m > 0) {
      if (lsb->phi != 1.0) {
        ierr = VecDuplicateVecs(X, lmvm->m, &lsb->P);CHKERRQ(ierr);
      }
      if (lsb->phi != 0.0) {
        ierr = VecDuplicateVecs(X, lmvm->m, &lsb->Q);CHKERRQ(ierr);
      }
    }
    lsb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMSymBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (lsb->allocated) {
    ierr = VecDestroy(&lsb->work);CHKERRQ(ierr);
    ierr = PetscFree4(lsb->stp, lsb->ytq, lsb->yts, lsb->psi);CHKERRQ(ierr);
    if (lmvm->m > 0) {
      if (lsb->phi != 1.0) {
        ierr = VecDestroyVecs(lmvm->m, &lsb->P);CHKERRQ(ierr);
      }
      if (lsb->phi != 0.0) {
        ierr = VecDestroyVecs(lmvm->m, &lsb->Q);CHKERRQ(ierr);
      }
    }
    lsb->allocated = PETSC_FALSE;
  }
  ierr = PetscFree(lmvm->ctx);CHKERRQ(ierr);
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMSymBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!lsb->allocated) {
    ierr = VecDuplicate(lmvm->Xprev, &lsb->work);CHKERRQ(ierr);
    ierr = PetscMalloc3(lmvm->m, &lsb->stp, lmvm->m, &lsb->ytq, lmvm->m, &lsb->yts);CHKERRQ(ierr);
    ierr = PetscCalloc1(lmvm->m, &lsb->psi);CHKERRQ(ierr);
    if (lmvm->m > 0) {
      if (lsb->phi != 1.0) {
        ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lsb->P);CHKERRQ(ierr);
      }
      if (lsb->phi != 0.0) {
        ierr = VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lsb->Q);CHKERRQ(ierr);
      }
    }
    lsb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_LMVMSymBrdn(PetscOptionItems *PetscOptionsObject, Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_SymBrdn       *lsb = (Mat_SymBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatSetFromOptions_LMVM(PetscOptionsObject, B);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"Limited-memory Variable Metric matrix for approximating Jacobians");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_phi","(developer) convex ratio between BFGS and DFP components in the Broyden update","",lsb->phi,&lsb->phi,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if ((lsb->phi < 0.0) || (lsb->phi > 1.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio cannot be outside the range of [0, 1]");
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMSymBrdn(Mat B)
{
  Mat_SymBrdn       *lsb;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMSYMBRDN);CHKERRQ(ierr);
  ierr = MatSetOption(B, MAT_SPD, PETSC_TRUE);CHKERRQ(ierr);
  
  B->ops->solve = MatSolve_LMVMSymBrdn;
  B->ops->setfromoptions = MatSetFromOptions_LMVMSymBrdn;
  B->ops->setup = MatSetUp_LMVMSymBrdn;
  B->ops->destroy = MatDestroy_LMVMSymBrdn;
  
  Mat_LMVM *lmvm = (Mat_LMVM*)B->data;
  lmvm->square = PETSC_TRUE;
  lmvm->ops->allocate = MatAllocate_LMVMSymBrdn;
  lmvm->ops->reset = MatReset_LMVMSymBrdn;
  lmvm->ops->update = MatUpdate_LMVMSymBrdn;
  lmvm->ops->mult = MatMult_LMVMSymBrdn;
  
  ierr = PetscNewLog(B, &lsb);CHKERRQ(ierr);
  lmvm->ctx = (void*)lsb;
  lsb->allocated = PETSC_FALSE;
  lsb->phi = 0.125;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMSymBrdn - Creates a limited-memory Symmetric Broyden-type matrix used 
   for approximating Jacobians. L-SymBrdn is a convex combination of L-DFP and 
   L-BFGS such that SymBrdn = (1 - phi)*BFGS + phi*DFP. The combination factor 
   phi is restricted to the range [0, 1], where the L-SymBrdn matrix is guaranteed 
   to be symmetric positive-definite.
   
   The provided local and global sizes must match the solution and function vectors 
   used with MatLMVMUpdate() and MatSolve(). The resulting L-SymBrdn matrix will have 
   storage vectors allocated with VecCreateSeq() in serial and VecCreateMPI() in 
   parallel. To use the L-SymBrdn matrix with other vector types, the matrix must be 
   created using MatCreate() and MatSetType(), followed by MatLMVMAllocate(). 
   This ensures that the internal storage and work vectors are duplicated from the 
   correct type of vector.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  n - number of local rows for storage vectors
-  N - global size of the storage vectors

   Output Parameter:
.  B - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions()
   paradigm instead of this routine directly.

   Options Database Keys:
.   -mat_lmvm_num_vecs - maximum number of correction vectors (i.e.: updates) stored
.   -mat_lmvm_phi - (developer) convex ratio between BFGS and DFP components of the inverse

   Level: intermediate

.seealso: MatCreate(), MATLMVM, MATLMVMSYMBRDN, MatCreateLMVMDFP(), MatCreateLMVMSR1(), 
          MatCreateLMVMBFGS(), MatCreateLMVMBrdn(), MatCreateLMVMBadBrdn()
@*/
PetscErrorCode MatCreateLMVMSymBrdn(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatCreate(comm, B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(*B, MATLMVMSYMBRDN);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Zero-memory Symmetric Broyden method for explicitly approximating 
  the diagonal of a Jacobian.
*/

typedef struct {
  Vec D, ssT, HyyTH, wwT, Xwork, Fwork;
  PetscBool allocated;
  PetscReal phi;
} Mat_DiagBrdn;

/*------------------------------------------------------------*/

PetscErrorCode MatSolve_LMVMDiagBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(dX, VEC_CLASSID, 3);
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  ierr = VecPointwiseDivide(dX, F, ldb->D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatMult_LMVMDiagBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(dX, VEC_CLASSID, 3);
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(B, dX, 3, F, 2);
  ierr = VecPointwiseMult(dX, ldb->D, F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*
  The diagonal Hessian update is derived from Equation 2 in 
  Erway and Marcia "On Solving Large-scale Limited-Memory 
  Quasi-Newton Equations" (https://arxiv.org/pdf/1510.06378.pdf).
  In this "zero"-memory implementation, the matrix-matrix products 
  are replaced by pointwise multiplications between their diagonal 
  vectors. Unlike limited-memory methods, the incoming updates 
  are directly applied to the diagonal instead of being stored 
  for later use.
*/
PetscErrorCode MatUpdate_LMVMDiagBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscReal         ynorm2, rho, rhotol;
  PetscReal         phi_k, yts, ythy, stbs;

  PetscFunctionBegin;
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    ierr = VecAXPBY(lmvm->Xprev, 1.0, -1.0, X);CHKERRQ(ierr);
    ierr = VecAXPBY(lmvm->Fprev, 1.0, -1.0, F);CHKERRQ(ierr);
    /* Test if the updates can be accepted */
    ierr = VecDotBegin(lmvm->Fprev, lmvm->Xprev, &rho);CHKERRQ(ierr);
    ierr = VecDotBegin(lmvm->Fprev, lmvm->Fprev, &ynorm2);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Fprev, lmvm->Xprev, &rho);CHKERRQ(ierr);
    ierr = VecDotEnd(lmvm->Fprev, lmvm->Fprev, &ynorm2);CHKERRQ(ierr);
    rhotol = lmvm->eps * ynorm2;
    if (rho > rhotol) {
      /* Update is good, accept it */
      ++lmvm->nupdates;
      /* Compute B_k * s_k once here because we need it for norms */
      ierr = VecPointwiseMult(ldb->Xwork, ldb->D, lmvm->Xprev);CHKERRQ(ierr);
      /* Now compute H_k * y_k and stash it permanently */
      ierr = VecPointwiseDivide(ldb->Fwork, lmvm->Fprev, ldb->D);CHKERRQ(ierr);
      /* Compute the necessary norms */
      ierr = VecDotBegin(lmvm->Fprev, lmvm->Xprev, &yts);CHKERRQ(ierr);
      ierr = VecDotBegin(lmvm->Fprev, ldb->Fwork, &ythy);CHKERRQ(ierr);
      ierr = VecDotBegin(lmvm->Xprev, ldb->Xwork, &stbs);CHKERRQ(ierr);
      ierr = VecDotEnd(lmvm->Fprev, lmvm->Xprev, &yts);CHKERRQ(ierr);
      ierr = VecDotEnd(lmvm->Fprev, ldb->Fwork, &ythy);CHKERRQ(ierr);
      ierr = VecDotEnd(lmvm->Xprev, ldb->Xwork, &stbs);CHKERRQ(ierr);
      /* Compute S*S^T diagonal */
      ierr = VecPointwiseMult(ldb->ssT, lmvm->Xprev, lmvm->Xprev);CHKERRQ(ierr);
      /* Compute H*y*y^T*H diagonal */
      ierr = VecPointwiseMult(ldb->Xwork, lmvm->Fprev, ldb->D);CHKERRQ(ierr);
      ierr = VecPointwiseMult(ldb->HyyTH, ldb->Fwork, ldb->Xwork);CHKERRQ(ierr);
      /* Compute V*V^T diagonal */
      ierr = VecAXPBYPCZ(ldb->Xwork, 1.0/yts, -1.0/ythy, 0.0, lmvm->Xprev, ldb->Fwork);CHKERRQ(ierr);
      ierr = VecPointwiseMult(ldb->wwT, ldb->Xwork, ldb->Xwork);CHKERRQ(ierr);
      /* Compute phi_k */
      phi_k = ((1.0 - ldb->phi) * (yts*yts)) / (((1.0 - ldb->phi) * (yts * yts)) + (ldb->phi * ythy * stbs));
      /* Update the diagonal */
      ierr = VecAXPBYPCZ(ldb->D, 1.0/yts, -1.0/ythy, 1.0, ldb->ssT, ldb->HyyTH);CHKERRQ(ierr);
      ierr = VecAXPY(ldb->D, phi_k*ythy, ldb->wwT);CHKERRQ(ierr);
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

PETSC_INTERN PetscErrorCode MatReset_LMVMDiagBrdn(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  if (destructive && ldb->allocated) {
    ierr = VecDestroy(&ldb->D);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->ssT);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->HyyTH);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->wwT);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->Xwork);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->Fwork);CHKERRQ(ierr);
    ldb->allocated = PETSC_FALSE;
  } else {
    ierr = VecSet(ldb->D, 1.0);CHKERRQ(ierr);
  }
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatAllocate_LMVMDiagBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  lmvm->m = 0;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!ldb->allocated) {
    ierr = VecDuplicate(X, &ldb->D);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->ssT);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->HyyTH);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->wwT);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->Xwork);CHKERRQ(ierr);
    ierr = VecDuplicate(F, &ldb->Fwork);CHKERRQ(ierr);
    ldb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatDestroy_LMVMDiagBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (ldb->allocated) {
    ierr = VecDestroy(&ldb->D);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->ssT);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->HyyTH);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->wwT);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->Xwork);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->Fwork);CHKERRQ(ierr);
    ldb->allocated = PETSC_FALSE;
  }
  ierr = PetscFree(lmvm->ctx);CHKERRQ(ierr);
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatSetUp_LMVMDiagBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  lmvm->m = 0;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!ldb->allocated) {
    ierr = VecDuplicate(lmvm->Xprev, &ldb->D);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->ssT);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->HyyTH);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->wwT);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->Xwork);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Fprev, &ldb->Fwork);CHKERRQ(ierr);
    ldb->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PETSC_INTERN PetscErrorCode MatSetFromOptions_LMVMDiagBrdn(PetscOptionItems *PetscOptionsObject, Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatSetFromOptions_LMVM(PetscOptionsObject, B);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"Limited-memory Variable Metric matrix for approximating Jacobians");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_phi","(developer) convex ratio between BFGS and DFP components in the Broyden update","",ldb->phi,&ldb->phi,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if ((ldb->phi < 0.0) || (ldb->phi > 1.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio cannot be outside the range of [0, 1]");
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMDiagBrdn(Mat B)
{
  Mat_DiagBrdn       *ldb;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMSYMBRDN);CHKERRQ(ierr);
  ierr = MatSetOption(B, MAT_SPD, PETSC_TRUE);CHKERRQ(ierr);
  
  B->ops->mult = MatMult_LMVMDiagBrdn;
  B->ops->solve = MatSolve_LMVMDiagBrdn;
  B->ops->setfromoptions = MatSetFromOptions_LMVMDiagBrdn;
  B->ops->setup = MatSetUp_LMVMDiagBrdn;
  B->ops->destroy = MatDestroy_LMVMDiagBrdn;
  
  Mat_LMVM *lmvm = (Mat_LMVM*)B->data;
  lmvm->m = 0;
  lmvm->square = PETSC_TRUE;
  lmvm->ops->update = MatUpdate_LMVMDiagBrdn;
  lmvm->ops->allocate = MatAllocate_LMVMDiagBrdn;
  lmvm->ops->reset = MatReset_LMVMDiagBrdn;
  
  ierr = PetscNewLog(B, &ldb);CHKERRQ(ierr);
  lmvm->ctx = (void*)ldb;
  ldb->allocated = PETSC_FALSE;
  ldb->phi = 0.125;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMDiagBrdn - Creates a zero-memory symmetric Broyden approximation 
   for the diagonal of a Jacobian. This matrix does not store any LMVM update vectors, 
   and instead uses the full-memory symmetric Broyden formula to update a vector that 
   respresents the diagonal of a Jacobian.

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
.   -mat_lmvm_phi - (developer) convex ratio between BFGS and DFP components of the inverse

   Level: intermediate

.seealso: MatCreate(), MATLMVM, MATLMVMSYMBRDN, MatCreateLMVMDFP(), MatCreateLMVMSR1(), 
          MatCreateLMVMBFGS(), MatCreateLMVMBrdn(), MatCreateLMVMBadBrdn()
@*/
PetscErrorCode MatCreateLMVMDiagBrdn(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatCreate(comm, B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(*B, MATLMVMDIAGBRDN);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
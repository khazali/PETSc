#include <petscksp.h>
/* multigrid eigensolver for bootstrapping */

PetscErrorCode PCGAMGGeneralizedRayleighQuotient(PetscInt l,Mat *A,Mat *P,Vec *v,Vec w,PetscInt level,PetscScalar *rq)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    vtv,vtAv;

  PetscFunctionBegin;
  ierr = MatMult(A[level],w,v[level]);CHKERRQ(ierr);
  ierr = VecDot(w,v[level],&vtAv);CHKERRQ(ierr);
  ierr = VecCopy(w,v[level]);CHKERRQ(ierr);
  for (i=level;i>0;i--) {
    ierr = MatMult(P[i],v[i],v[i-1]);CHKERRQ(ierr);
  }
  for (i=1;i<=level;i++) {
    ierr = MatMultTranspose(P[i],v[i-1],v[i]);CHKERRQ(ierr);
  }
  ierr = VecDot(v[level],w,&vtv);CHKERRQ(ierr);
  *rq = vtAv/PetscRealPart(vtv);
  PetscFunctionReturn(0);
}

PetscErrorCode PCGAMGGetNearNullspace(PC pc,PetscInt l,Mat *A,Mat *P,PetscInt n,PetscScalar ***lambdas,Vec ***ws)
{
  PetscInt       h,i,j,k;
  PetscErrorCode ierr;
  KSP            *ksp; /* smoothers */
  Vec            **w,*v;
  PetscScalar    **lambda,lambdaold,lambdaerr;
  PC             spc;
  const char     *prefix;
  PetscRandom    rand;

  PetscFunctionBegin;
  /* setup smoothers */
  ierr = PetscMalloc1(l,&ksp);CHKERRQ(ierr);
  ierr = PetscMalloc1(l,&w);CHKERRQ(ierr);
  ierr = PetscMalloc1(l,&v);CHKERRQ(ierr);
  ierr = PetscMalloc1(l,&lambda);CHKERRQ(ierr);
  for (i=0;i<l;i++) {
    ierr = PetscMalloc1(n,&lambda[i]);CHKERRQ(ierr);
    ierr = MatGetVecs(A[i],NULL,&v[i]);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(v[i],n,&w[i]);CHKERRQ(ierr);
    ierr = KSPCreate(PetscObjectComm((PetscObject)pc),&ksp[i]);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp[i],A[i],A[i],SAME_NONZERO_PATTERN);
    if (i==l-1) {
      ierr = KSPSetType(ksp[i],KSPGMRES);CHKERRQ(ierr);
      ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ksp[i],prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ksp[i],"mgeig_coarse_");CHKERRQ(ierr);
      ierr = KSPGetPC(ksp[i],&spc);CHKERRQ(ierr);
    } else {
      ierr = KSPSetType(ksp[i],KSPCHEBYSHEV);CHKERRQ(ierr);
      ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ksp[i],prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ksp[i],"mgeig_");CHKERRQ(ierr);
      ierr = KSPGetPC(ksp[i],&spc);CHKERRQ(ierr);
      ierr = KSPSetTolerances(ksp[i],PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1);CHKERRQ(ierr);
    }
    ierr = KSPSetFromOptions(ksp[i]);CHKERRQ(ierr);
  }
  /* start with random initial guesses on the coarsest level */

  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)pc),&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetType(rand,PETSCRAND);CHKERRQ(ierr);
  for (i=l-1;i>=0;i--) {
    for (j=0;j<n;j++) {
      if (i==l-1) {
        ierr = VecSetRandom(w[i][j],rand);CHKERRQ(ierr);
        ierr = VecNormalize(w[i][j],NULL);CHKERRQ(ierr);
      }
      ierr = PCGAMGGeneralizedRayleighQuotient(l,A,P,v,w[i][j],i,&lambda[i][j]);CHKERRQ(ierr);
      PetscInt    size;
      ierr = VecGetSize(w[i][j],&size);CHKERRQ(ierr);
      /* ierr = PetscPrintf(PETSC_COMM_WORLD,"[%d] eigenvalue %d: %12.14g\n",size,j,lambda[i][j]);CHKERRQ(ierr); */
      for (h=0;h<100;h++) {
        /* smooth */
        ierr = VecScale(v[i],lambda[i][j]);CHKERRQ(ierr);
        ierr = KSPSetInitialGuessNonzero(ksp[i],PETSC_TRUE);CHKERRQ(ierr);
        ierr = KSPSolve(ksp[i],v[i],w[i][j]);CHKERRQ(ierr);
        /* orthogonalize against the previously computed eigenvectors */
        for (k=0;k<j;k++) {
          PetscScalar wdw;
          ierr = VecDot(w[i][j],w[i][k],&wdw);CHKERRQ(ierr);
          ierr = VecAXPY(w[i][j],-wdw,w[i][k]);CHKERRQ(ierr);
        }
        ierr = VecNormalize(w[i][j],NULL);CHKERRQ(ierr);
        lambdaold=lambda[i][j];
        ierr = PCGAMGGeneralizedRayleighQuotient(l,A,P,v,w[i][j],i,&lambda[i][j]);CHKERRQ(ierr);
        lambdaerr = PetscAbsScalar(lambdaold-lambda[i][j]);
        if (PetscAbsScalar(lambda[i][j]) > 1e-12) lambdaerr /= PetscAbsScalar(lambda[i][j]);
        if (PetscAbsScalar(lambdaerr) < 1e-3) break;
        /* ierr = PetscPrintf(PETSC_COMM_WORLD,"[%d] eigenvalue %d: %12.14g\n",size,j,lambda[i][j]);CHKERRQ(ierr); */
      }
      if (i==l-1) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"[%d] eigenvalue %d: %12.14g\n",size,j,lambda[i][j]);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"[%d] eigenvalue %d: %12.14g (%12.14g)\n",size,j,lambda[i][j],lambda[i+1][j]);CHKERRQ(ierr);
      }
    }
    /* project */
    if (i!=0) {
      for (j=0;j<n;j++) {
        ierr = MatMult(P[i],w[i][j],w[i-1][j]);CHKERRQ(ierr);
        lambda[i-1][j] = lambda[i][j];
      }
    }
  }
  /* destroy */
  for (i=0;i<l;i++) {
    ierr = PetscFree(lambda[i]);CHKERRQ(ierr);
    /* ierr = VecDestroyVecs(n,&w[i]);CHKERRQ(ierr); */
    ierr = VecDestroy(&v[i]);CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp[i]);CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscFree(lambda);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  /* ierr = PetscFree(w);CHKERRQ(ierr); */
  ierr = PetscFree(ksp);CHKERRQ(ierr);

  *lambdas = lambda;
  *ws = w;

  PetscFunctionReturn(0);
}

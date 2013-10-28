
#include <petsc-private/kspimpl.h>

typedef struct {
  PetscReal haptol;
  PetscBool monitor_arnorm,monitor_relrnorm;
  PetscReal Arnorm;
} KSP_MINRESQLP;

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_MINRESQLP"
PetscErrorCode KSPSetUp_MINRESQLP(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ksp->pc_side == PC_RIGHT) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"No right preconditioning for KSPMINRESQLP");
  else if (ksp->pc_side == PC_SYMMETRIC) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"No symmetric preconditioning for KSPMINRESQLP");
  ierr = KSPSetWorkVecs(ksp,9);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* modified from KSPMonitorDefault() by adding Arnorm */
#undef __FUNCT__
#define __FUNCT__ "KSPMINTRESQLPMonitor"
PetscErrorCode KSPMINTRESQLPMonitor(KSP ksp,PetscInt n,PetscReal rnorm,void *dummy)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = (PetscViewer) dummy;
  KSP_MINRESQLP  *minresqlp = (KSP_MINRESQLP*)ksp->data;

  PetscFunctionBegin;
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ksp),&viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
  if (n == 0 && ((PetscObject)ksp)->prefix) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Residual norms for %s solve.\n",((PetscObject)ksp)->prefix);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"%3D KSP Residual norm %14.12e, Arnorm %14.12e\n",n,(double)rnorm,(double)minresqlp->Arnorm);CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)ksp)->tablevel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_MINRESQLP"
PetscErrorCode KSPSetFromOptions_MINRESQLP(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_MINRESQLP  *minresqlp = (KSP_MINRESQLP*)ksp->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP MINRESQLP options");CHKERRQ(ierr);

  ierr = PetscOptionsBool("-ksp_monitor_minresqlp","Monitor rnorm and Arnorm","KSPMINRESQLPMonitor",minresqlp->monitor_arnorm,&minresqlp->monitor_arnorm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ksp_monitor_minresqlp_relrnorm","Monitor relative rnorm and Arnorm","KSPMINRESQLPMonitor",minresqlp->monitor_relrnorm,&minresqlp->monitor_relrnorm,NULL);CHKERRQ(ierr);
  printf("monitor_arnorm %d\n",minresqlp->monitor_arnorm);
  if (minresqlp->monitor_arnorm) {
    ierr = KSPMonitorSet(ksp,KSPMINTRESQLPMonitor,NULL,NULL);CHKERRQ(ierr);
  }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSolve_MINRESQLP"
PetscErrorCode  KSPSolve_MINRESQLP(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    alpha,beta,ibeta,betaold,eta,c=1.0,ceta,cold=1.0,coold,s=0.0,sold=0.0,soold;
  KSP_MINRESQLP  *minresqlp = (KSP_MINRESQLP*)ksp->data;
  PetscReal      Arnorm=minresqlp->Arnorm,root=0.0;
  PetscScalar    rho0,rho1,irho1,rho2,mrho2,rho3,mrho3,dp = 0.0;
  PetscReal      np;
  Vec            X,B,R,Z,U,V,W,UOLD,VOLD,WOLD,WOOLD;
  Mat            Amat,Pmat;
  MatStructure   pflag;
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  ierr    = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  X       = ksp->vec_sol;
  B       = ksp->vec_rhs;
  R       = ksp->work[0];
  Z       = ksp->work[1];
  U       = ksp->work[2];
  V       = ksp->work[3];
  W       = ksp->work[4];
  UOLD    = ksp->work[5];
  VOLD    = ksp->work[6];
  WOLD    = ksp->work[7];
  WOOLD   = ksp->work[8];
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);

  ksp->its = 0;

  ierr = VecSet(UOLD,0.0);CHKERRQ(ierr);           /*     u_old  <-   0   */
  ierr = VecCopy(UOLD,VOLD);CHKERRQ(ierr);         /*     v_old  <-   0   */
  ierr = VecCopy(UOLD,W);CHKERRQ(ierr);            /*     w      <-   0   */
  ierr = VecCopy(UOLD,WOLD);CHKERRQ(ierr);         /*     w_old  <-   0   */

  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);/*     r <- b - A*x    */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
    ierr = MatMult(Amat,R,WOOLD);CHKERRQ(ierr);    /* WOOLD <- A*r        */
    ierr = VecNorm(WOOLD,NORM_2,&Arnorm);CHKERRQ(ierr); /* Arnorm = norm(A*r) */
  } else { 
    ierr = VecCopy(B,R);CHKERRQ(ierr);             /*     r <- b (x is 0) */
    ierr = MatMult(Amat,B,WOOLD);CHKERRQ(ierr);    /* WOOLD <- A*b        */
    ierr = VecNorm(WOOLD,NORM_2,&Arnorm);CHKERRQ(ierr); /* Arnorm = norm(A*b) */
  }

  ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);       /*     z  <- B*r       */

  ierr = VecDot(R,Z,&dp);CHKERRQ(ierr);
  if (PetscRealPart(dp) < minresqlp->haptol) {
    ierr = PetscInfo2(ksp,"Detected indefinite operator %G tolerance %G\n",PetscRealPart(dp),minresqlp->haptol);CHKERRQ(ierr);
    ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
    PetscFunctionReturn(0);
  }

  dp   = PetscAbsScalar(dp);
  dp   = PetscSqrtScalar(dp);
  beta = dp;                                      /*  beta <- sqrt(r'*z)  */
  eta  = beta;

  ierr = VecCopy(R,V);CHKERRQ(ierr);
  ierr = VecCopy(Z,U);CHKERRQ(ierr);
  ibeta = 1.0 / beta;
  ierr = VecScale(V,ibeta);CHKERRQ(ierr);         /*    v <- r / beta     */
  ierr = VecScale(U,ibeta);CHKERRQ(ierr);         /*    u <- z / beta     */

  ierr = VecNorm(Z,NORM_2,&np);CHKERRQ(ierr);     /*   np <- ||z||        */

  KSPLogResidualHistory(ksp,np);
  ierr = KSPMonitor(ksp,0,np);CHKERRQ(ierr);
  ksp->rnorm = np;
  ierr = (*ksp->converged)(ksp,0,np,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);  /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  i = 0;
  do {
    ksp->its = i+1;

    /* Lanczos  */
    ierr = KSP_MatMult(ksp,Amat,U,R);CHKERRQ(ierr);   /*      r <- A*u   */
    ierr = VecDot(U,R,&alpha);CHKERRQ(ierr);          /*  alpha <- r'*u  */
    ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);        /*      z <- B*r   */

    ierr = VecAXPY(R,-alpha,V);CHKERRQ(ierr);     /*  r <- r - alpha v     */
    ierr = VecAXPY(Z,-alpha,U);CHKERRQ(ierr);     /*  z <- z - alpha u     */
    ierr = VecAXPY(R,-beta,VOLD);CHKERRQ(ierr);   /*  r <- r - beta v_old  */
    ierr = VecAXPY(Z,-beta,UOLD);CHKERRQ(ierr);   /*  z <- z - beta u_old  */

    betaold = beta;

    ierr = VecDot(R,Z,&dp);CHKERRQ(ierr);
    if ( PetscRealPart(dp) < minresqlp->haptol) {
      ierr = PetscInfo2(ksp,"Detected indefinite operator %G tolerance %G\n",PetscRealPart(dp),minresqlp->haptol);CHKERRQ(ierr);
      ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
      break;
    }

    dp = PetscAbsScalar(dp);
    beta = PetscSqrtScalar(dp);                   /*  beta <- sqrt(r'*z)   */

    /*    QR factorisation    */
    coold = cold; cold = c; soold = sold; sold = s;

    rho0 = cold * alpha - coold * sold * betaold;
    rho1 = PetscSqrtScalar(rho0*rho0 + beta*beta);
    rho2 = sold * alpha + coold * cold * betaold;
    rho3 = soold * betaold;

    /*     Givens rotation    */
    c = rho0 / rho1;
    s = beta / rho1;

    /*    Update    */
    ierr = VecCopy(WOLD,WOOLD);CHKERRQ(ierr);     /*  w_oold <- w_old      */
    ierr = VecCopy(W,WOLD);CHKERRQ(ierr);         /*  w_old  <- w          */

    ierr = VecCopy(U,W);CHKERRQ(ierr);            /*  w      <- u          */
    mrho2 = - rho2;
    ierr = VecAXPY(W,mrho2,WOLD);CHKERRQ(ierr);   /*  w <- w - rho2 w_old  */
    mrho3 = - rho3;
    ierr = VecAXPY(W,mrho3,WOOLD);CHKERRQ(ierr);  /*  w <- w - rho3 w_oold */
    irho1 = 1.0 / rho1;
    ierr = VecScale(W,irho1);CHKERRQ(ierr);       /*  w <- w / rho1        */

    ceta = c * eta;
    ierr = VecAXPY(X,ceta,W);CHKERRQ(ierr);       /*  x <- x + ceta w     */ 
    eta = - s * eta;

    ierr = VecCopy(V,VOLD);CHKERRQ(ierr);
    ierr = VecCopy(U,UOLD);CHKERRQ(ierr);
    ierr = VecCopy(R,V);CHKERRQ(ierr);
    ierr = VecCopy(Z,U);CHKERRQ(ierr);
    ibeta = 1.0 / beta;
    ierr = VecScale(V,ibeta);CHKERRQ(ierr);       /*  v <- r / beta       */
    ierr = VecScale(U,ibeta);CHKERRQ(ierr);       /*  u <- z / beta       */

    np = ksp->rnorm * PetscAbsScalar(s);
    root = PetscSqrtReal(rho0*rho0 + (cold*beta)*(cold*beta));
    Arnorm = ksp->rnorm * root;  

    printf("\n*** %3d-th  Arnorm %8.3g",(ksp->its)-1,Arnorm); 

    minresqlp->Arnorm = Arnorm; 
    if (Arnorm < minresqlp->haptol) {
      ierr = PetscInfo2(ksp,"Detected happy breakdown %G tolerance %G. It is a least-squares solution.\n",Arnorm,minresqlp->haptol);CHKERRQ(ierr);
      /* printf("Arnorm %g < minresqlp->haptol = %g, exit \n",Arnorm, minresqlp->haptol); */
      ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ksp->reason = KSP_CONVERGED_ATOL_NORMAL;
      break;
    }

    ksp->rnorm = np;
    KSPLogResidualHistory(ksp,np);
    ierr = KSPMonitor(ksp,i+1,np);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,np,&ksp->reason,ksp->cnvP);CHKERRQ(ierr); /* test for convergence */
    if (ksp->reason) break;
    i++;
  } while (i<ksp->max_it);

    
  ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);      /*     r <- A*x        */
  ierr = VecAXPY(R,-1.0,B);CHKERRQ(ierr);              /*     r <- A*x - b    */
  ierr = MatMult(Amat,R,WOOLD);CHKERRQ(ierr);          /* WOOLD <- A*r        */
  ierr = VecNorm(WOOLD,NORM_2,&Arnorm);CHKERRQ(ierr);  /* Arnorm = norm2(A*r) */
  minresqlp->Arnorm = Arnorm; 
  printf("\n~~~~ Final Arnorm %8.3g\n", Arnorm); 

  if (i >= ksp->max_it) {
    ksp->reason = KSP_DIVERGED_ITS;
  }
  PetscFunctionReturn(0);
}

/*MC
     KSPMINRESQLP - This code implements the MINRESQLP (Minimum Residual) method.

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes: The operator and the preconditioner must be symmetric and the preconditioner must
          be positive definite for this method.
          Supports only left preconditioning. ???

   References: S.-C. T. Choi, C. C. Paige and M. A. Saunders (2011).  MINRES-QLP: A Krylov Subspace Method for Indefinite or Singular Symmetric Systems. SIAM J. Scientific Computing 33, Number 4, 1810-1836. 
               S.-C. T. Choi and M. A. Saunders (2013). ALGORITHM: MINRES-QLP for Symmetric and Hermitian Linear Equations and Least-Squares Problems. ACM Transactions on Mathematical Software.  To appear.

   Contributed by: Robert Scheichl: maprs@maths.bath.ac.uk
                   Sou-Cheng Choi : sctchoi@mcs.anl.gov, schoi32@iit.edu
                   Hong Zhang     : hzhang@mcs.anl.gov

.seealso: KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPCG, KSPCR
M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_MINRESQLP"
PetscErrorCode  KSPCreate_MINRESQLP(KSP ksp)
{
  KSP_MINRESQLP  *minresqlp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = PetscNewLog(ksp,KSP_MINRESQLP,&minresqlp);CHKERRQ(ierr);
  minresqlp->haptol           = 1.e-18;
  minresqlp->monitor_arnorm   = PETSC_FALSE;
  minresqlp->monitor_relrnorm = PETSC_FALSE;
  minresqlp->Arnorm           = 1.0;
  ksp->data                   = (void*)minresqlp;

  /*
       Sets the functions that are associated with this data structure
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup                = KSPSetUp_MINRESQLP;
  ksp->ops->solve                = KSPSolve_MINRESQLP;
  ksp->ops->destroy              = KSPDestroyDefault;
  ksp->ops->setfromoptions       = KSPSetFromOptions_MINRESQLP;
  ksp->ops->buildsolution        = KSPBuildSolutionDefault;
  ksp->ops->buildresidual        = KSPBuildResidualDefault;
  PetscFunctionReturn(0);
}
EXTERN_C_END

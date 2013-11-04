
#include <../src/ksp/ksp/impls/minres/minresimpl.h>       /*I "petscksp.h" I*/

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

#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_MINRESQLP"
PetscErrorCode KSPSetFromOptions_MINRESQLP(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_MINRES  *minresqlp = (KSP_MINRES*)ksp->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP MINRESQLP options");CHKERRQ(ierr);

  ierr = PetscOptionsBool("-ksp_monitor_minresqlp","Monitor rnorm and Arnorm","KSPMINREMonitor",minresqlp->monitor_arnorm,&minresqlp->monitor_arnorm,NULL);CHKERRQ(ierr);
  if (minresqlp->monitor_arnorm) {
    ierr = KSPMonitorSet(ksp,KSPMINRESMonitor,NULL,NULL);CHKERRQ(ierr);
  }
  
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSolve_MINRESQLP"
PetscErrorCode  KSPSolve_MINRESQLP(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       iter, QLPiter = 0, lines = 1, headlines = 20;
  PetscScalar    alfa,betal,betan,ibeta,beta=0,phi,cs=1.0,taul=0.0,tau=0.0,cold=1.0,coold,sn=0.0,sold=0.0,soold,cr1=-1.0,sr1=0.0,cr2=-1.0,sr2=0.0;
  KSP_MINRES  *minresqlp = (KSP_MINRES*)ksp->data;
  PetscReal      Arnorm=minresqlp->Arnorm, root = 0.0, relArnorm = 0.0, pnorm = 0.0, Anorm2 = 0.0;
  PetscScalar    rho0,gama,irho1,dlta,dlta_QLP,mrho2,epln,mrho3,dp = 0.0;
  PetscReal      np; //=residual???, used for convergence test!
  Vec            X,B,R3,Z,V,R2,W,UOLD,R1,WL,WL2;
  Mat            Amat,Pmat;
  MatStructure   pflag;
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  ierr    = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Krylov method %sn does not support diagonal scaling",((PetscObject)ksp)->type_name);

  X       = ksp->vec_sol;
  B       = ksp->vec_rhs;
  R3       = ksp->work[0];
  Z       = ksp->work[1];
  V       = ksp->work[2];
  R2       = ksp->work[3];
  W       = ksp->work[4];
  UOLD    = ksp->work[5];
  R1    = ksp->work[6];
  WL    = ksp->work[7];
  WL2   = ksp->work[8];

  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);

  ksp->its = 0;

  ierr = VecSet(UOLD,0.0);CHKERRQ(ierr);           /*     u_old  <-   0   */
  ierr = VecCopy(UOLD,R1);CHKERRQ(ierr);           /*     r1     <-   0   */
  ierr = VecCopy(UOLD,W);CHKERRQ(ierr);            /*     w      <-   0   */
  ierr = VecCopy(UOLD,WL);CHKERRQ(ierr);           /*     wl     <-   0   */

  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R3);CHKERRQ(ierr);   /* r3    <- b - A*x     */
    ierr = VecAYPX(R3,-1.0,B);CHKERRQ(ierr);
    ierr = MatMult(Amat,R3,WL2);CHKERRQ(ierr);         /* wl2 <- A*r3          */
    ierr = VecNorm(WL2,NORM_2,&Arnorm);CHKERRQ(ierr);  /* Arnorm = norm(A*r3)  */
  } else {
    ierr = VecCopy(B,R3);CHKERRQ(ierr);                  /*    r3 <- b (x is 0) */
    ierr = MatMult(Amat,B,WL2);CHKERRQ(ierr);            /* wl2 <- A*b          */
    ierr = VecNorm(WL2,NORM_2,&Arnorm);CHKERRQ(ierr);    /* Arnorm = norm(A*b)  */
  }

  ierr = KSP_PCApply(ksp,R3,Z);CHKERRQ(ierr);            /*     z  <- B*r3      */

  ierr = VecDot(R3,Z,&dp);CHKERRQ(ierr);
  if (PetscRealPart(dp) < minresqlp->haptol) {
    ierr = PetscInfo2(ksp,"Detected indefinite operator %G tolerance %G\n",PetscRealPart(dp),minresqlp->haptol);CHKERRQ(ierr);
    ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
    PetscFunctionReturn(0);
  }

  dp   = PetscAbsScalar(dp);
  dp   = PetscSqrtScalar(dp);
  betan = dp;                                      /*  betan <- sqrt(r3'*z)  */
  phi  = betan;

  ierr = VecCopy(R3,R2);CHKERRQ(ierr);
  ierr = VecCopy(Z,V);CHKERRQ(ierr);
  ibeta = 1.0 / betan;
  ierr = VecScale(R2,ibeta);CHKERRQ(ierr);         /*    r2 <- r3 / betan    */
  ierr = VecScale(V,ibeta);CHKERRQ(ierr);          /*     v <- z / betan     */

  ierr = VecNorm(Z,NORM_2,&np);CHKERRQ(ierr);      /*    np <- ||z||         */

  KSPLogResidualHistory(ksp,np);
  ksp->rnorm = np;
  ierr = (*ksp->converged)(ksp,0,np,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);  /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  iter = 0;
  do {
    ksp->its = iter+1;
  
    /* Lanczos  */
    betal = beta;        beta = betan;
    ierr = KSP_MatMult(ksp,Amat,V,R3);CHKERRQ(ierr);   /*   r3 <- A*v    */
    ierr = VecDot(V,R3,&alfa);CHKERRQ(ierr);           /* alfa <- r3'*v  */
    ierr = KSP_PCApply(ksp,R3,Z);CHKERRQ(ierr);        /*    z <- B*r3   */   //TODO What is B? Is this for preconditioning?

    ierr = VecAXPY(R3,-alfa,R2);CHKERRQ(ierr);         /*   r3 <- r3 - alfa r2    */
    ierr = VecAXPY(Z,-alfa,V);CHKERRQ(ierr);           /*    z <- z - alfa v      */
    ierr = VecAXPY(R3,-betan,R1);CHKERRQ(ierr);        /*   r3 <- r3 - betan r1   */
    ierr = VecAXPY(Z,-betan,UOLD);CHKERRQ(ierr);       /*    z <- z - betan u_old */   // What is UOLD?

    ierr = VecDot(R3,Z,&dp);CHKERRQ(ierr);
    if ( PetscRealPart(dp) < minresqlp->haptol) {
      ierr = PetscInfo2(ksp,"Detected indefinite operator %G tolerance %G\n",PetscRealPart(dp),minresqlp->haptol);CHKERRQ(ierr);
      ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
      break;
    }

    dp = PetscAbsScalar(dp);
    betan = PetscSqrtScalar(dp);                   /*  betan <- sqrt(r3'*z)   */

    pnorm  = ksp->its == 1? PetscSqrtReal(alfa*alfa + betan*betan) :  PetscSqrtReal(beta*beta + alfa*alfa + betan*betan);
          //TODO pnorm  = ksp->its == 1? VecNorm([alfa, betan],NORM_2,&pnorm): VecNorm([betan, alfa, betan],NORM_2,&pnorm);
    Anorm2 = Anorm2 > pnorm? Anorm2 : pnorm;
    /* QR factorisation    */
    coold = cold; cold = cs; soold = sold; sold = sn;

    rho0 = cold * alfa - coold * sold * beta;
    gama = PetscSqrtScalar(rho0*rho0 + betan*betan);
    dlta = sold * alfa + coold * cold * beta;
    epln = soold * beta;

    /* Givens rotation    */
    cs = rho0 / gama;
    sn = betan / gama;
    
    
    /* Update xnorm */
      
    /* Update w. Update x except if it will become too big */
    ierr = VecCopy(WL,WL2);CHKERRQ(ierr);       /*  wl2 <- wl         */
    ierr = VecCopy(W,WL);CHKERRQ(ierr);         /*  wl  <- w          */
    ierr = VecCopy(V,W);CHKERRQ(ierr);          /*  w   <- v          */
    mrho2 = - dlta;
    ierr = VecAXPY(W,mrho2,WL);CHKERRQ(ierr);   /*  w <- w - dlta wl  */
    mrho3 = - epln;
    ierr = VecAXPY(W,mrho3,WL2);CHKERRQ(ierr);  /*  w <- w - epln wl2 */
    irho1 = 1.0 / gama;
    ierr = VecScale(W,irho1);CHKERRQ(ierr);     /*  w <- w / gama     */

    tau = cs * phi;
    ierr = VecAXPY(X,tau,W);CHKERRQ(ierr);      /*  x <- x + tau w    */
    phi = - sn * phi;

    ierr = VecCopy(R2,R1);CHKERRQ(ierr);
    ierr = VecCopy(V,UOLD);CHKERRQ(ierr);
    ierr = VecCopy(R3,R2);CHKERRQ(ierr);
    ierr = VecCopy(Z,V);CHKERRQ(ierr);
    ibeta = 1.0 / betan;
    ierr = VecScale(R2,ibeta);CHKERRQ(ierr);       /*  r2 <- r3 / betan      */
    ierr = VecScale(V,ibeta);CHKERRQ(ierr);        /*   v <-  z / betan      */

    root = PetscSqrtReal(rho0*rho0 + (cold*betan)*(cold*betan));  /* ? form two vector and use VecNorm */
    Arnorm = ksp->rnorm * root;
    relArnorm = root / Anorm2;
    printf("\n*** %3d-th  Arnorm %8.3g, rnorm %8.3g, Anorm %8.3g, relArnorm %8.3g\n",(ksp->its)-1, Arnorm, np, Anorm2, relArnorm);
    minresqlp->Arnorm    = Arnorm;
    minresqlp->relArnorm = relArnorm;   
    ierr = KSPMonitor(ksp,iter,np);CHKERRQ(ierr);
    if (relArnorm < minresqlp->haptol) {
      ierr = PetscInfo2(ksp,"Detected happy breakdown %G tolerance %G. It is a least-squares solution.\n",Arnorm,minresqlp->haptol);CHKERRQ(ierr);
      printf("Arnorm %g < minresqlp->haptol = %g, exit \n",Arnorm, minresqlp->haptol);
      ksp->reason = KSP_CONVERGED_ATOL_NORMAL;
      break;
    }

    np = ksp->rnorm * PetscAbsScalar(sn); //res???
    ksp->rnorm = np;
    KSPLogResidualHistory(ksp,np);
    ierr = (*ksp->converged)(ksp,iter+1,np,&ksp->reason,ksp->cnvP);CHKERRQ(ierr); /* test for convergence */
    if (ksp->reason) break;
    iter++;
  } while (iter<ksp->max_it);

  /* do we need following other than montoring final Arnorm??? */
  ierr = KSP_MatMult(ksp,Amat,X,R3);CHKERRQ(ierr);    /*    r3 <- A*x         */
  ierr = VecAXPY(R3,-1.0,B);CHKERRQ(ierr);            /*    r3 <- A*x - b     */
  ierr = MatMult(Amat,R3,WL2);CHKERRQ(ierr);          /*   wl2 <- A*r3        */
  ierr = VecNorm(WL2,NORM_2,&Arnorm);CHKERRQ(ierr);   /* Arnorm = norm2(A*r3) */
  relArnorm = Arnorm / Anorm2;
  minresqlp->Arnorm    = Arnorm; 
  minresqlp->relArnorm = relArnorm;
  printf("\n~~~~ Final Arnorm %8.3g, relArnorm %8.3g\n", Arnorm, relArnorm);
  ierr = KSPMonitor(ksp,iter,np);CHKERRQ(ierr);

  if (iter >= ksp->max_it) {
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

   References: S.-C. T. Choi (2006). Iterative Methods for Singular Linear Equations and Least-Squares Problems, PhD thesis, ICME, Stanford University.
               S.-C. T. Choi, C. C. Paige and M. A. Saunders (2011).  MINRES-QLP: A Krylov Subspace Method for Indefinite or Singular Symmetric Systems. SIAM J. Scientific Computing 33, Number 4, 1810-1836.
               S.-C. T. Choi and M. A. Saunders (2013). ALGORITHM: MINRES-QLP for Symmetric and Hermitian Linear Equations and Least-Squares Problems. ACM Transactions on Mathematical Software.  To appear.

   Contributed by: Sou-Cheng Choi : sctchoi@mcs.anl.gov, schoi32@iit.edu
 
   See also: MINRES-QLP project website at http://code.google.com/p/minres-qlp/

.seealso: KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPCG, KSPCR
M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_MINRESQLP"
PetscErrorCode  KSPCreate_MINRESQLP(KSP ksp)
{
  KSP_MINRES  *minresqlp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = PetscNewLog(ksp,KSP_MINRES,&minresqlp);CHKERRQ(ierr);
  minresqlp->haptol           = 1.e-18;
  minresqlp->monitor_arnorm   = PETSC_FALSE;
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

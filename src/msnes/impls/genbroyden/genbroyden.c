#include <petsc/private/snesimpl.h>
#include <petscblaslapack.h>

typedef struct {
  PetscInt    msizeMax;       /* maximum size of krylov space */
  PetscInt    msize;          /* size of krylov space */
  PetscViewer monitor;        /* debugging output for NGMRES */
  /* History and subspace data */
  Vec       *Fdot;            /* [msize] residual history */
  Vec       *Xdot;            /* [msize] solution history */
  PetscReal *fnorms;          /* [msize] residual norm history  */
  PetscReal *xnorms;          /* [msize] solution norm history */
  /* Least squares BLAS arguments */
  PetscScalar *h;              /* the constraint matrix H */
  PetscScalar *q;              /* the matrix formed as Q_ij = (rdot_i, rdot_j) */
  PetscScalar *beta;           /* rhs for the minimization problem */
  PetscScalar *xi;             /* the dot-product of the current and previous residual */
  PetscReal   *s;              /* the singular values */
  PetscScalar *work;           /* the work vector */
  PetscReal   *rwork;          /* the real work vector used for complex */
  PetscBLASInt lwork;          /* the size of the work vector */
  /* Restart parameters */
  PetscInt    restartTries;   /* if the restart conditions persist for restartTries iterations, then restart WAS restart_it */
  PetscInt    restart_periodic; /* number of iterations to restart after */
  PetscReal epsilonB;         /* Criterion B difference tolerance */
  PetscReal deltaB;           /* Criterion B residual tolerance */
  PetscReal gammaC;           /* Restart residual tolerance */
  SNESNGMRESRestartType restart_type;
} MSNES_GenBroyden;

#define H(i,j)  broy->h[i*broy->msizeMax + j]
#define Q(i,j)  broy->q[i*broy->msizeMax + j]

#undef __FUNCT__
#define __FUNCT__ "MSNESGenBroydenUpdateSubspace_Private"
/*
  MSNESGenBroydenUpdateSubspace_Private - Update the ivecth residual and solution vectors in the subspace

  Input Parameters:
+ snes   - SNES object
. ivec   - subspace vector to update
. m      - subspace size
. F      - residual
. fnorm  - residual norm, ||F||
- X      - approximate solution

  Level: developer

.seealso: SNESGenBroyden
*/
static PetscErrorCode MSNESGenBroydenUpdateSubspace_Private(SNES snes, PetscInt ivec, PetscInt m, Vec F, Vec X)
{
  MSNES_GenBroyden *broy = (MSNES_GenBroyden *) snes->data;
  Vec              *Fdot = broy->Fdot;
  Vec              *Xdot = broy->Xdot;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (m > broy->msize) SETERRQ2(PetscObjectComm((PetscObject) snes), PETSC_ERR_ARG_WRONGSTATE, "Subspace size %D cannot be greater than maximum %D!", m, broy->msize);
  if (ivec > m)          SETERRQ2(PetscObjectComm((PetscObject) snes), PETSC_ERR_ARG_WRONGSTATE, "Cannot update vector %d with space size %d!", ivec, m);
  ierr = VecCopy(F, Fdot[ivec]);CHKERRQ(ierr);
  ierr = VecCopy(X, Xdot[ivec]);CHKERRQ(ierr);
  ierr = VecNorm(F, NORM_2, &broy->fnorms[ivec]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MSNESCheckRestart_GenBroyden(SNES snes, Vec X, Vec F, PetscBool *replace, PetscInt *restartCount)
{
  MSNES_GenBroyden     *broy      = (MSNES_GenBroyden *) snes->data;
  Vec                   XM        = snes->work[3]; /* Prior solution before LS solve */
  Vec                  *Xdot      = broy->Xdot;
  SNESNGMRESRestartType rtype     = broy->restart_type;
  PetscInt              itrestart = snes->iterRestart;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  /* TODO Should we allow a line search along X^A - X^M? */
  *replace = PETSC_TRUE;
  if (rtype == SNES_NGMRES_RESTART_PERIODIC) {
    if (itrestart > broy->restart_periodic) {
      if (broy->monitor) {ierr = PetscViewerASCIIPrintf(broy->monitor, "periodic restart after %D iterations\n", itrestart);CHKERRQ(ierr);}
      *restartCount = broy->restartTries; /* Force restart */
      *replace      = PETSC_FALSE;
    }
  } else if (broy->restart_type == SNES_NGMRES_RESTART_DIFFERENCE) {
    Vec       D    = snes->work[1];
    PetscReal dmin = PETSC_MAX_REAL, dnorm;
    PetscReal fmin = snes->minnorm, fnorm;
    PetscInt  i;

    ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
    /* Calculate D = X - X^M and X - X_j */
    ierr = VecCopy(X, D);CHKERRQ(ierr);
    ierr = VecAXPY(D, -1.0, XM);CHKERRQ(ierr);
    ierr = VecNormBegin(D, NORM_2, &dnorm);CHKERRQ(ierr);
    for (i = 0; i < broy->msize; ++i) {
      ierr = VecCopy(Xdot[i], D);CHKERRQ(ierr);
      ierr = VecAXPY(D, -1.0, X);CHKERRQ(ierr);
      ierr = VecNormBegin(D, NORM_2, &broy->xnorms[i]);CHKERRQ(ierr);
    }
    ierr = VecNormEnd(D, NORM_2, &dnorm);CHKERRQ(ierr);
    for (i = 0; i < broy->msize; ++i) {
      ierr = VecNormEnd(D, NORM_2, &broy->xnorms[i]);CHKERRQ(ierr);
      dmin = PetscMin(dmin, broy->xnorms[i]);
    }

    /* TODO Peter has sqrts on all the norms. Why? Is he just insane? */
    /* difference stagnation restart, the choice of x^A is too close to some other choice */
    if ((broy->msize > 1) && (broy->epsilonB * dnorm > dmin) && (fnorm > broy->deltaB * fmin)) {
      if (broy->monitor) {ierr = PetscViewerASCIIPrintf(broy->monitor, "difference restart: %e > %e\n", broy->epsilonB*dnorm, dmin);CHKERRQ(ierr);}
      *replace = PETSC_FALSE;
    }
    /* residual stagnation restart, the norm of the function is increased above the minimum by too much */
    if (fnorm > broy->gammaC * fmin) {
      if (broy->monitor) {ierr = PetscViewerASCIIPrintf(broy->monitor, "residual restart: %e > %e\n", fnorm, broy->gammaC * fmin);CHKERRQ(ierr);}
      *replace = PETSC_FALSE;
    }
    if (!*replace) ++(*restartCount);
    else           *restartCount = 0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MSNESRestart_GenBroyden(SNES snes, Vec X, Vec F, PetscInt *restartCount)
{
  MSNES_GenBroyden *broy = (MSNES_GenBroyden *) snes->data;
  Vec               XM   = snes->work[3]; /* Prior solution before LS solve */
  Vec               FM   = snes->work[4]; /* Prior residual before LS solve */
  PetscBool         replace;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MSNESCheckRestart_GenBroyden(snes, X, F, &replace, restartCount);CHKERRQ(ierr);
  /* If restart condition were satisfied, revert to prior guess */
  if (!replace) {
    ierr = VecCopy(XM, X);CHKERRQ(ierr);
    ierr = VecCopy(FM, F);CHKERRQ(ierr);
  }
  /* Restart after restart conditions have persisted for restartTries iterations */
  if (*restartCount >= broy->restartTries) {
    if (broy->monitor) {ierr = PetscViewerASCIIPrintf(broy->monitor, "Restarted at iteration %d\n", snes->iterRestart);CHKERRQ(ierr);}
    *restartCount     = 0;
    snes->iterRestart = 1; /* WAS k_restart */
    broy->msize       = 1; /* WAS l */
  } else {
    /* Set the current size of the subspace */
    if (broy->msize < broy->msizeMax) ++broy->msize;
    ++snes->iterRestart;
  }
  /* Place the current solution and residual in the subspace, could possibly use (FM, fMnorm, XM) instead */
  ierr = MSNESGenBroydenUpdateSubspace_Private(snes, snes->iterRestart % broy->msizeMax, broy->msize, F, X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MSNESComputeUpdate_GenBroyden"
/*
  MSNESComputeUpdate_GenBroyden - Solve a least-squares system to compute an update.

  Input Parameters:
+ snes - the SNES context
. X    - the current solution
- F    - the current residual

  Output Parameters:
+ X - the new solution X' = X + dX
- Y - the update direction Y = dX

  TODO REF Anderson, JACM, 1965
           Oosterlee and Washio, SISC, 2000
           Fang and Saad, UMSI, 2007
           Ni, WPI, 2009
*/
PetscErrorCode MSNESComputeUpdate_GenBroyden(SNES snes, Vec X, Vec F, Vec Y)
{
  MSNES_GenBroyden *broy = (MSNES_GenBroyden *) snes->data;
  Vec               XM   = snes->work[3];        /* This is the right preconditioned starting guess (FAS from the paper) TODO Is this needed, or can we just use X? */
  Vec               FM   = snes->work[4];        /* This is the right preconditioned residual (FAS from the paper) */
  const PetscInt    ivec = snes->iterRestart % broy->msizeMax; /* replace the last used part of the subspace */
  Vec              *Fdot       = broy->Fdot;
  Vec              *Xdot       = broy->Xdot;
  PetscScalar      *beta       = broy->beta;
  PetscScalar      *xi         = broy->xi;
  const PetscInt    l          = broy->msize;
  PetscScalar       alph_total = 0.0;
  PetscReal         nu;
  PetscInt          i, j;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecNorm(FM, NORM_2, &nu);CHKERRQ(ierr);
  nu   = PetscSqr(nu);
  /* Store initial vector in subspace */
  ierr = MSNESGenBroydenUpdateSubspace_Private(snes, 0, 0, F, X);CHKERRQ(ierr);

  /* TODO Update for Generalized Broyden/Anderson Mixing */
  /*   Solve min_alpha || F(alpha) ||, where we approximate this F(alpha) \approx F(x^M) + \sum^m_{j=1} alpha_i (F(x_{i-j}) - F(x^M)) */

  /* TODO Redo this to form the LS system directly instead of the LS system and check solution */

  /* construct the right hand side and xi factors */
  if (l > 0) {
    ierr = VecMDotBegin(F, l, Fdot, xi);CHKERRQ(ierr);
    ierr = VecMDotBegin(Fdot[ivec], l, Fdot, beta);CHKERRQ(ierr);
    ierr = VecMDotEnd(F, l, Fdot, xi);CHKERRQ(ierr);
    ierr = VecMDotEnd(Fdot[ivec], l, Fdot, beta);CHKERRQ(ierr);
    for (i = 0; i < l; i++) {
      Q(i, ivec) = beta[i];
      Q(ivec, i) = beta[i];
    }
  } else {
    Q(0,0) = broy->fnorms[ivec]*broy->fnorms[ivec];
  }

  for (i = 0; i < l; ++i) beta[i] = nu - xi[i];

  /* construct h */
  for (j = 0; j < l; ++j) {
    for (i = 0; i < l; ++i) {
      H(i,j) = Q(i,j) - xi[i] - xi[j] + nu;
    }
  }
  if (l == 1) {
    /* simply set alpha[0] = beta[0] / H[0, 0] */
    if (H(0,0) != 0.0) beta[0] = beta[0]/H(0,0);
    else               beta[0] = 0.0;
  } else {
    PetscReal    rcond = -1.0; /* the exit condition */
    PetscBLASInt info  = 0;    /* the output condition */
    PetscBLASInt m, n, lda, ldb, rank, nrhs = 1;

#if defined(PETSC_MISSING_LAPACK_GELSS)
    SETERRQ(PetscObjectComm((PetscObject) snes), PETSC_ERR_SUP, "Generalized Broyden with Least-Squares solve requires the LAPACK GELSS routine.");
#else
    ierr = PetscBLASIntCast(l, &m);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(l, &n);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(broy->msizeMax, &lda);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(broy->msizeMax, &ldb);CHKERRQ(ierr);
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    PetscStackCallBLAS("LAPACKgelss", LAPACKgelss_(&m,&n,&nrhs,broy->h,&lda,broy->beta,&ldb,broy->s,&rcond,&rank,broy->work,&broy->lwork,broy->rwork,&info));
#else
    PetscStackCallBLAS("LAPACKgelss", LAPACKgelss_(&m,&n,&nrhs,broy->h,&lda,broy->beta,&ldb,broy->s,&rcond,&rank,broy->work,&broy->lwork,&info));
#endif
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    if (info < 0) SETERRQ(PetscObjectComm((PetscObject) snes), PETSC_ERR_LIB, "Bad argument to GELSS");
    if (info > 0) SETERRQ(PetscObjectComm((PetscObject) snes), PETSC_ERR_LIB, "SVD failed to converge");
#endif
  }
  for (i = 0; i < l; ++i) {if (PetscIsInfOrNanScalar(beta[i])) SETERRQ(PetscObjectComm((PetscObject) snes), PETSC_ERR_LIB, "SVD generated inconsistent output");}

  for (i = 0; i < l; ++i) alph_total += beta[i];
  ierr = VecCopy(X, XM);CHKERRQ(ierr);
  ierr = VecCopy(F, FM);CHKERRQ(ierr);
  ierr = VecScale(X, 1.0 - alph_total);CHKERRQ(ierr);
  ierr = VecMAXPY(X, l, beta, Xdot);CHKERRQ(ierr);
  /* Calculate update */
  ierr = VecCopy(X, Y);CHKERRQ(ierr);
  ierr = VecAXPY(Y, -1.0, XM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

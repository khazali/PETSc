#include <../src/tao/constrained/impls/mad/mad.h>

PetscErrorCode TaoMADInitVecs(Tao tao)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  MPI_Comm           comm;
  PetscErrorCode     ierr;
  Vec                xtmp, btmp;
  
  PetscFunctionBegin;
  mad->nlb = mad->nub = mad->nb = mad->ng = mad->nkkt = 0;
  ierr = PetscObjectGetComm((PetscObject)tao, &comm);CHKERRQ(ierr);
  
  /* make sure Tao bound vectors are initialized */
  if (!tao->XL && !tao->XU && tao->ops->computebounds) {
    ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);
  }
  
  /* get index set of lower-bounded design vars */
  ierr = VecDuplicate(tao->solution, &xtmp);CHKERRQ(ierr);
  if (tao->XL) {
    ierr = VecSet(xtmp,PETSC_NINFINITY);CHKERRQ(ierr);
    ierr = VecWhichGreaterThan(tao->XL, xtmp, &mad->lb_idx);CHKERRQ(ierr);
    ierr = ISGetSize(mad->lb_idx, &mad->nlb);CHKERRQ(ierr);
    ierr = VecGetSubVector(tao->XL, mad->lb_idx, &btmp);CHKERRQ(ierr);
    ierr = VecDuplicate(btmp, &mad->cl);CHKERRQ(ierr);
    ierr = VecDuplicate(btmp, &mad->cl_work);CHKERRQ(ierr);
    ierr = VecDuplicate(btmp, &mad->mu_lb);CHKERRQ(ierr);
    ierr = VecDuplicate(btmp, &mad->dmu_lb);CHKERRQ(ierr);
    ierr = VecDuplicate(btmp, &mad->lb);CHKERRQ(ierr);
    ierr = VecCopy(btmp, mad->lb);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(tao->XL, mad->lb_idx, &btmp);CHKERRQ(ierr);
    ierr = VecDestroy(&btmp);CHKERRQ(ierr);
  } else {
    mad->nlb=0;
  }
  
  /* get index set of upper-bounded design vars */
  if (tao->XU) {
    ierr = VecSet(xtmp,PETSC_INFINITY);CHKERRQ(ierr);
    ierr = VecWhichLessThan(tao->XU, xtmp, &mad->ub_idx);CHKERRQ(ierr);
    ierr = ISGetSize(mad->ub_idx, &mad->nub);CHKERRQ(ierr);
    ierr = VecGetSubVector(tao->XU, mad->ub_idx, &btmp);CHKERRQ(ierr);
    ierr = VecDuplicate(btmp, &mad->cu);CHKERRQ(ierr);
    ierr = VecDuplicate(btmp, &mad->cu_work);CHKERRQ(ierr);
    ierr = VecDuplicate(btmp, &mad->mu_ub);CHKERRQ(ierr);
    ierr = VecDuplicate(btmp, &mad->dmu_ub);CHKERRQ(ierr);
    ierr = VecDuplicate(btmp, &mad->ub);CHKERRQ(ierr);
    ierr = VecCopy(btmp, mad->ub);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(tao->XU, mad->ub_idx, &btmp);CHKERRQ(ierr);
    ierr = VecDestroy(&btmp);CHKERRQ(ierr);
  } else {
    mad->nub=0;
  }
  ierr = VecDestroy(&xtmp);CHKERRQ(ierr);
  
  /* create the composite bound vector */
  mad->nb = mad->nlb + mad->nub;
  if (mad->nb > 0) {
    if (mad->cl && mad->cu) {
      Vec cb[2] = {mad->cl, mad->cu};
      ierr = VecCreateNest(comm, 2, NULL, cb, &mad->cb);CHKERRQ(ierr);
      Vec cb_work[2] = {mad->cl_work, mad->cu_work};
      ierr = VecCreateNest(comm, 2, NULL, cb_work, &mad->cb_work);CHKERRQ(ierr);
      Vec mu_b[2] = {mad->mu_lb, mad->mu_ub};
      ierr = VecCreateNest(comm, 2, NULL, mu_b, &mad->mu_b);CHKERRQ(ierr);
      Vec dmu_b[2] = {mad->dmu_lb, mad->dmu_ub};
      ierr = VecCreateNest(comm, 2, NULL, dmu_b, &mad->dmu_b);CHKERRQ(ierr);
    } else if (mad->cl) {
      mad->cb = mad->cl;
      mad->cb_work = mad->cl_work;
      mad->mu_b = mad->mu_lb;
      mad->dmu_b = mad->dmu_lb;
    } else if (mad->cu) {
      mad->cb = mad->cu;
      mad->cb_work = mad->cu_work;
      mad->mu_b = mad->mu_ub;
      mad->dmu_b = mad->dmu_ub;
    } else {
      SETERRQ(PETSC_COMM_SELF,1,"Error in counting bound constraint sizes!");
    }
  }
  
  /* create the composite inequality constraint vector */
  mad->ng = mad->nb + mad->nineq;
  if (mad->ng > 0) {
    if (mad->cin && mad->cb) {
      Vec g[2] = {mad->cin, mad->cb};
      ierr = VecCreateNest(comm, 2, NULL, g, &mad->g);CHKERRQ(ierr);
      Vec g_work[2] = {mad->cin, mad->cb};
      ierr = VecCreateNest(comm, 2, NULL, g_work, &mad->g_work);CHKERRQ(ierr);
      Vec mu[2] = {mad->mu_cin, mad->mu_b};
      ierr = VecCreateNest(comm, 2, NULL, mu, &mad->mu);CHKERRQ(ierr);
      Vec dmu[2] = {mad->dmu_cin, mad->dmu_b};
      ierr = VecCreateNest(comm, 2, NULL, dmu, &mad->dmu);CHKERRQ(ierr);
    } else if (mad->cb) {
      mad->g = mad->cb;
      mad->g_work = mad->cb_work;
      mad->mu = mad->mu_b;
      mad->dmu = mad->dmu_b;
    } else if (mad->cin) {
      mad->g = mad->cin;
      mad->g_work = mad->cin_work;
      mad->mu = mad->mu_cin;
      mad->dmu = mad->dmu_cin;
    } else {
      SETERRQ(PETSC_COMM_SELF,1,"Error in counting inequality constraint sizes!");
    }
    ierr = VecDuplicate(mad->g, &mad->mu);CHKERRQ(ierr);
  }
  
  mad->nkkt = mad->nd + mad->nh + mad->ng;
  if (mad->nkkt == mad->nd) {
    /* there are no constraints of any type */
    mad->sol = tao->solution;
    mad->step = tao->stepdirection;
    mad->kkt = mad->grad;
    mad->kkt_work = mad->x_work;
  } else if (mad->nkkt > mad->nd) {
    if (mad->nh > 0 && mad->ng > 0) {
      /* there are both equality and inequality constraints */
      Vec sol[3] = {tao->solution, mad->lambda, mad->mu};
      Vec step[3] = {tao->stepdirection, mad->dlambda, mad->dmu};
      Vec kkt[3] = {mad->grad, mad->h, mad->g};
      Vec kkt_work[3] = {mad->x_work, mad->h_work, mad->g_work};
      ierr = VecCreateNest(comm, 3, NULL, sol, &mad->sol);CHKERRQ(ierr);
      ierr = VecCreateNest(comm, 3, NULL, step, &mad->step);CHKERRQ(ierr);
      ierr = VecCreateNest(comm, 3, NULL, kkt, &mad->kkt);CHKERRQ(ierr);
      ierr = VecCreateNest(comm, 3, NULL, kkt_work, &mad->kkt_work);CHKERRQ(ierr);
    } else if (mad->nh > 0) {
      /* there are only equality constraints */
      Vec sol[2] = {tao->solution, mad->lambda};
      Vec step[2] = {tao->stepdirection, mad->dlambda};
      Vec kkt[2] = {mad->grad, mad->h};
      Vec kkt_work[2] = {mad->x_work, mad->h_work};
      ierr = VecCreateNest(comm, 2, NULL, sol, &mad->sol);CHKERRQ(ierr);
      ierr = VecCreateNest(comm, 2, NULL, step, &mad->step);CHKERRQ(ierr);
      ierr = VecCreateNest(comm, 2, NULL, kkt, &mad->kkt);CHKERRQ(ierr);
      ierr = VecCreateNest(comm, 2, NULL, kkt_work, &mad->kkt_work);CHKERRQ(ierr);
    } else if (mad->ng > 0) {
      /* there are only inequality constraints */
      Vec sol[2] = {tao->solution, mad->mu};
      Vec step[2] = {tao->stepdirection, mad->dmu};
      Vec kkt[2] = {mad->grad, mad->g};
      Vec kkt_work[2] = {mad->x_work, mad->g_work};
      ierr = VecCreateNest(comm, 2, NULL, sol, &mad->sol);CHKERRQ(ierr);
      ierr = VecCreateNest(comm, 2, NULL, step, &mad->step);CHKERRQ(ierr);
      ierr = VecCreateNest(comm, 2, NULL, kkt, &mad->kkt);CHKERRQ(ierr);
      ierr = VecCreateNest(comm, 2, NULL, kkt_work, &mad->kkt_work);CHKERRQ(ierr);
    } else {
      /* we should never get here; if we did, something broke */
      SETERRQ(PETSC_COMM_SELF,1,"Error in counting constraint sizes!");
    }
  } else {
    /* we should never get here; if we did, something broke */
    SETERRQ(PETSC_COMM_SELF,1,"Error in counting KKT vector size!");
  }

  /* allocate arrays of vectors for the multisecant approx */
  ierr = VecDuplicateVecs(mad->sol, mad->q, &mad->Y);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(mad->kkt, mad->q, &mad->R);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(mad->sol, mad->q-1, &mad->dY);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(mad->kkt, mad->q-1, &mad->dR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADSetInitPoint(Tao tao)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  PetscErrorCode     ierr;
  
  PetscFunctionBegin;
  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);
  if (tao->XL && tao->XU) {
    ierr = VecMedian(tao->XL, tao->solution, tao->XU, tao->solution);CHKERRQ(ierr);
  }
  if (mad->nh > 0) {
    ierr = VecSet(mad->lambda, 0.0);CHKERRQ(ierr);
  }
  if (mad->ng > 0) {
    ierr = VecSet(mad->mu, 0.0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0); 
}

PetscErrorCode TaoMADUpdateHistory(Tao tao, Vec y_new, Vec r_new) 
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  PetscErrorCode     ierr;
  PetscInt           i, last;
  Vec                tmpY, tmpR;

  PetscFunctionBegin;
  /* add to history */
  if (mad->k < mad->q) {
    last = mad->k;
  } else {
    /* we hit the memory limit so we have to discard the oldest vectors */
    tmpY = mad->Y[0];
    tmpR = mad->R[0];
    for (i=1; i<mad->q; i++) {
      mad->Y[i-1] = mad->Y[i];
      mad->R[i-1] = mad->R[i];
    }
    mad->Y[mad->q-1] = tmpY;
    mad->R[mad->q-1] = tmpR;
    last = mad->q-1;
  }
  ierr = VecCopy(y_new, mad->Y[last]);CHKERRQ(ierr);
  ierr = VecCopy(r_new, mad->R[last]);CHKERRQ(ierr);
  mad->k++;
  mad->k = PetscMin(mad->k, mad->q);
  PetscFunctionReturn(0);        
}

PetscErrorCode TaoMADResetHistory(Tao tao) 
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;

  PetscFunctionBegin;
  mad->k = 0;
  PetscFunctionReturn(0);        
}

PetscErrorCode TaoMADComputeDiffMats(Tao tao) 
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  PetscErrorCode     ierr;
  PetscInt           i;
  VecType            kkt_type;
  
  PetscFunctionBegin;
  for (i=1; i<PetscMin(mad->k, mad->q); i++) {
    /* dY[i-1] = Y[i] - Y[i-1] */
    ierr = VecCopy(mad->Y[i], mad->dY[i-1]);CHKERRQ(ierr);
    ierr = VecAXPBY(mad->dY[i-1], -1.0, 1.0, mad->Y[i-1]);CHKERRQ(ierr);
    /* dR[i-1] = R[i] - R[i-1] + beta*(X[i-1], 0, 0) */
    ierr = VecCopy(mad->R[i], mad->dR[i-1]);CHKERRQ(ierr);
    ierr = VecAXPBY(mad->dR[i-1], -1.0, 1.0, mad->R[i-1]);CHKERRQ(ierr);
    ierr = VecGetType(mad->dR[i-1], &kkt_type);CHKERRQ(ierr);
    /* apply the Hessian correction, if necessary */
    if (kkt_type == VECNEST) {
      Vec r_x, x_old;
      ierr = VecNestGetSubVec(mad->dR[i-1], 0, &r_x);CHKERRQ(ierr);
      ierr = VecNestGetSubVec(mad->Y[i-1], 0, &x_old);CHKERRQ(ierr);
      ierr = VecAXPBY(r_x, mad->beta, 1.0, x_old);CHKERRQ(ierr);
    } else {
      ierr = VecAXPBY(mad->dR[i-1], mad->beta, 1.0, mad->Y[i-1]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADDiffMatMult(Tao tao, Vec *dM, PetscScalar *in, Vec out)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  PetscErrorCode     ierr;
  PetscInt           i;

  PetscFunctionBegin;
  ierr = VecSet(out, 0.0);CHKERRQ(ierr);
  for (i=0; i<PetscMin(mad->k, mad->q)-1; i++) {
    ierr = VecAXPBY(out, in[i], 1.0, dM[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADDiffMatMultTrans(Tao tao, Vec *dM, Vec in, PetscScalar *out)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  PetscErrorCode     ierr;
  PetscInt           i;

  PetscFunctionBegin;
  for (i=0; i<PetscMin(mad->k, mad->q)-1; i++) {
    ierr = VecDot(dM[i], in, &out[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADSolveSubproblem(Tao tao, PetscScalar *gamma)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  PetscErrorCode     ierr;
  PetscInt           i, j;
  PetscBLASInt       M = mad->nkkt; /* number of rows in the R matrix */
  PetscBLASInt       N = PetscMin(mad->k, mad->q); /* number of columns in the R matrix */
  PetscBLASInt       LDA = M, LDU = M, LDVT = N;
  PetscBLASInt       info, lwork;
  PetscScalar        R[LDA*N], S[N], U[LDU*N], VT[LDVT*N];
  PetscScalar        wkopt;
  PetscScalar        *work;
  const PetscScalar  *rtmp;
  PetscScalar        UTr[N], SUTr[N];

  PetscFunctionBegin;
  /* first convert the array of vectors in R into an actual matrix */
  for (j=0; j<N; j++) {
    ierr = VecGetArrayRead(mad->dR[j], &rtmp);CHKERRQ(ierr);
    for (i=0; i<M; i++) {
      R[i*N + j] = rtmp[j];
    }
    ierr = VecRestoreArrayRead(mad->dR[j], &rtmp);CHKERRQ(ierr);
  }
  /* now query LAPACK to get size of workspace and allocate it */
  lwork = -1;
  LAPACKdgesvd_("A", "A", &M, &N, R, &LDA, S, U, &LDU, VT, &LDVT, &wkopt, &lwork, &info);
  lwork = (PetscBLASInt)wkopt;
  ierr = PetscMalloc1(lwork, &work);CHKERRQ(ierr);
  
  /* compute SVD */
  LAPACKdgesvd_("A", "A", &M, &N, R, &LDA, S, U, &LDU, VT, &LDVT, &wkopt, &lwork, &info);
  if (info > 0) {
    ierr = PetscInfo(mad, "SVD calculation failed to converge!\n");CHKERRQ(ierr);
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
  
  /* truncate small singular values */
  for (i=0; i<N; i++) {
    if (S[i] < mad->svd_cutoff*S[0]) S[i] = 0.0;
  }
  
  /* apply the pseudo-inverse to the KKT vector, which computes gamma = (dR)^{-1} * r_k */
  ierr = VecGetArrayRead(mad->kkt, &rtmp);CHKERRQ(ierr);
  /* first we do U^T * r */
  for (j=0; j<N; j++) {
    UTr[j] = 0.0;
    for (i=0; i<LDU; i++) UTr[j] += U[i*N + j] * rtmp[i];
  }
  /* now we do sigma * (U^T * r) */
  for (j=0; j<N; j++) SUTr[j] = S[j]*UTr[j];
  /* finally hit this with V to compute gamma */
  for (j=0; j<N; j++) {
    gamma[j] = 0.0;
    for (i=0; i<LDVT; i++) gamma[j] += VT[i*N + j] * SUTr[i];
  }
  ierr = VecRestoreArrayRead(mad->kkt, &rtmp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADComputeKKT(Tao tao, PetscReal *f, PetscReal *opt_norm, PetscReal *feas_norm)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  PetscErrorCode     ierr;
  Vec                x_lb, x_ub;
  PetscReal          feas2_h = 0.0, feas2_g = 0.0;
  
  PetscFunctionBegin;
  /* start with the objective function and its gradient */
  ierr = TaoComputeObjectiveAndGradient(tao, tao->solution, f, tao->gradient);CHKERRQ(ierr);
  ierr = VecCopy(tao->gradient, mad->grad);CHKERRQ(ierr);
  if (mad->h) {
    /* evaluate equality constraints and their jacobian matrix */
    ierr = TaoComputeEqualityConstraints(tao, tao->solution, tao->constraints_equality);CHKERRQ(ierr);
    ierr = TaoComputeJacobianEquality(tao, tao->solution, tao->jacobian_equality, tao->jacobian_equality_pre);CHKERRQ(ierr);
    ierr = VecCopy(tao->constraints_equality, mad->h);CHKERRQ(ierr);
    /* compute feasibility component for equality constraits */
    ierr = VecDot(mad->h, mad->h, &feas2_h);CHKERRQ(ierr);
    /* add the correction to the gradient of the Lagrangian grad = grad - Ae^T lambda */
    ierr = MatMultTranspose(tao->jacobian_equality, mad->lambda, mad->x_work);CHKERRQ(ierr);
    ierr = VecAXPBY(mad->grad, -1.0, 1.0, mad->x_work);CHKERRQ(ierr);
  }
  if (mad->g) {
    if (mad->cin) {
      /* evaluate inequality constraints and their jacobian matrix */
      ierr = TaoComputeInequalityConstraints(tao, tao->solution, tao->constraints_inequality);CHKERRQ(ierr);
      ierr = TaoComputeJacobianInequality(tao, tao->solution, tao->jacobian_inequality, tao->jacobian_inequality_pre);CHKERRQ(ierr);
      ierr = VecCopy(tao->constraints_inequality, mad->cin);CHKERRQ(ierr);
      /* add the correction to the gradient of the Lagrangian grad = grad - Ai^T mu_cin*/
      ierr = MatMultTranspose(tao->jacobian_inequality, mad->mu_cin, mad->x_work);CHKERRQ(ierr);
      ierr = VecAXPBY(mad->grad, -1.0, 1.0, mad->x_work);CHKERRQ(ierr);
    }
    if (mad->cb) {
      if (mad->cl) {
        /* evaluate lower bound constraints cl = x_lb - lb >= 0 */
        ierr = VecGetSubVector(tao->solution, mad->lb_idx, &x_lb);CHKERRQ(ierr);
        ierr = VecCopy(x_lb, mad->cl);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(tao->solution, mad->lb_idx, &x_lb);CHKERRQ(ierr);
        ierr = VecAXPBY(mad->cl, -1.0, 1.0, mad->lb);CHKERRQ(ierr);
        /* add correction to the gradient of the Lagrangian grad = grad - mu_lb */
        ierr = VecGetSubVector(mad->grad, mad->lb_idx, &x_lb);CHKERRQ(ierr);
        ierr = VecAXPBY(x_lb, -1.0, 1.0, mad->mu_lb);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(mad->grad, mad->lb_idx, &x_lb);CHKERRQ(ierr);
      }
      if (mad->cu) {
        /* evaluate lower bound constraints cu = ub - x_ub >= 0 */
        ierr = VecGetSubVector(tao->solution, mad->ub_idx, &x_ub);CHKERRQ(ierr);
        ierr = VecCopy(x_ub, mad->cu);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(tao->solution, mad->ub_idx, &x_ub);CHKERRQ(ierr);
        ierr = VecAXPBY(mad->cu, 1.0, -1.0, mad->ub);CHKERRQ(ierr);
        /* add correction to the gradient of the Lagrangian grad = grad + mu_ub */
        ierr = VecGetSubVector(mad->grad, mad->ub_idx, &x_ub);CHKERRQ(ierr);
        ierr = VecAXPBY(x_ub, 1.0, 1.0, mad->mu_ub);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(mad->grad, mad->ub_idx, &x_ub);CHKERRQ(ierr);
      }
    }
    /* now handle the homotopy for inequality constraints  g = 0.5*(abs(g - mu) - g - mu) */
    ierr = VecCopy(mad->g, mad->g_work);CHKERRQ(ierr);
    ierr = VecAXPBY(mad->g_work, -1.0, 1.0, mad->mu);CHKERRQ(ierr);
    ierr = VecAbs(mad->g_work);CHKERRQ(ierr);
    ierr = VecAXPBY(mad->g, 1.0, -1.0, mad->g_work);CHKERRQ(ierr);
    ierr = VecAXPBY(mad->g, -1.0, 1.0, mad->mu);CHKERRQ(ierr);
    ierr = VecScale(mad->g, 0.5);CHKERRQ(ierr);
    /* compute feasibility component for inequality constraits */
    ierr = VecDot(mad->g, mad->g, &feas2_g);CHKERRQ(ierr);
  }
  /* compute final convergence norms */
  ierr = VecNorm(mad->grad, NORM_2, opt_norm);CHKERRQ(ierr);
  *feas_norm = PetscSqrtReal(feas2_h + feas2_g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSolve_MAD(Tao tao)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  PetscErrorCode     ierr;
  PetscReal          f, steplen=1.0, step_norm, opt_norm, opt_norm0, feas_norm, feas_norm0;
  PetscScalar        *gamma;

  PetscFunctionBegin;
  /* initialize all the vectors we need -- this also computes and processes bounds */
  ierr = TaoMADInitVecs(tao);CHKERRQ(ierr);
  ierr = TaoMADSetInitPoint(tao);CHKERRQ(ierr);
  ierr = TaoMADComputeKKT(tao, &f, &opt_norm0, &feas_norm0);CHKERRQ(ierr);
  
  /* convergence check at the initial point */
  tao->reason = TAO_CONTINUE_ITERATING;
  ierr = TaoLogConvergenceHistory(tao, f, opt_norm0, feas_norm0, tao->ksp_its);CHKERRQ(ierr);
  ierr = TaoMonitor(tao, tao->niter, f, opt_norm0, feas_norm0, steplen);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  
  /* record the initial point in history */
  ierr = TaoMADResetHistory(tao);CHKERRQ(ierr);
  ierr = TaoMADUpdateHistory(tao, mad->sol, mad->kkt);CHKERRQ(ierr);
  
  /* allocate the gamma array for the first time */
  ierr = PetscMalloc(PetscMin(mad->k, mad->q), &gamma);CHKERRQ(ierr);
  
  /* take a safe-guarded steepest descent step */
  ierr = VecCopy(mad->kkt, mad->step);CHKERRQ(ierr);
  ierr = VecScale(mad->step, -1.0);CHKERRQ(ierr);
  ierr = VecNorm(mad->step, NORM_2, &step_norm);CHKERRQ(ierr);
  if (step_norm > mad->max_step) {
    steplen = mad->max_step/step_norm;
  } else {
    steplen = 1.0;
  }
  ierr = VecAXPBY(mad->sol, steplen, 1.0, mad->step);CHKERRQ(ierr);
  
  tao->niter = 1;
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* compute KKT conditions at new point and check convergence*/
    ierr = TaoMADComputeKKT(tao, &f, &opt_norm, &feas_norm);CHKERRQ(ierr);
    ierr = TaoLogConvergenceHistory(tao, f, opt_norm, feas_norm, tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao, tao->niter, f, opt_norm, feas_norm, steplen);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
    if (tao->reason != TAO_CONTINUE_ITERATING) break;
    
    /* record the new point and compute difference matrices */
    ierr = TaoMADUpdateHistory(tao, mad->sol, mad->kkt);CHKERRQ(ierr);
    ierr = TaoMADComputeDiffMats(tao);CHKERRQ(ierr);
    
    /* solve the least-squares subproblem argmin_gamma ||r_k - R_k * gamma || */
    ierr = PetscRealloc(PetscMin(mad->k, mad->q), &gamma);
    ierr = TaoMADSolveSubproblem(tao, gamma);
    
    /* compute the new step */
    ierr = VecAXPBY(mad->step, -mad->alpha, 0.0, mad->kkt);CHKERRQ(ierr);
    ierr = TaoMADDiffMatMult(tao, mad->dY, gamma, mad->kkt_work);CHKERRQ(ierr);
    ierr = VecAXPBY(mad->step, -1.0, 1.0, mad->kkt_work);CHKERRQ(ierr);
    ierr = TaoMADDiffMatMult(tao, mad->dR, gamma, mad->kkt_work);CHKERRQ(ierr);
    ierr = VecAXPBY(mad->step, mad->alpha, 1.0, mad->kkt_work);CHKERRQ(ierr);
    
    /* safe-guard the step size */
    ierr = VecNorm(mad->step, NORM_2, &step_norm);CHKERRQ(ierr);
    if (step_norm > mad->max_step) {
      steplen = mad->max_step/step_norm;
    } else {
      steplen = 1.0;
    }
    
    /* accept the step */
    ierr = VecAXPBY(mad->sol, steplen, 1.0, mad->step);CHKERRQ(ierr);
    tao->niter++;
  }
  ierr = PetscFree(gamma);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetup_MAD(Tao tao)
{
  TAO_MAD        *mad = (TAO_MAD*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mad->nd = mad->nh  = mad->nineq = 0;
  ierr = VecGetSize(tao->solution, &mad->nd);CHKERRQ(ierr);
  if (!tao->gradient) {
    ierr = VecDuplicate(tao->solution, &tao->gradient);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &tao->stepdirection);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &mad->grad);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &mad->x_work);CHKERRQ(ierr);
  }
  if (tao->constraints_equality) {
    ierr = VecGetSize(tao->constraints_equality, &mad->nh);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality, &mad->lambda);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality, &mad->dlambda);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality, &mad->h);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality, &mad->h_work);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality, &tao->DE);CHKERRQ(ierr);
  }
  if (tao->constraints_inequality) {
    ierr = VecGetSize(tao->constraints_inequality, &mad->nineq);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality, &mad->cin);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality, &mad->mu_cin);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality, &mad->dmu_cin);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality, &tao->DI);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_MAD(Tao tao)
{
  TAO_MAD        *mad = (TAO_MAD*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&mad->kkt);CHKERRQ(ierr);
  ierr = VecDestroy(&mad->kkt_work);CHKERRQ(ierr);
  ierr = VecDestroy(&mad->lambda);CHKERRQ(ierr);
  ierr = VecDestroy(&mad->dlambda);CHKERRQ(ierr);
  ierr = VecDestroy(&mad->mu);CHKERRQ(ierr);
  ierr = VecDestroy(&mad->dmu);CHKERRQ(ierr);
  ierr = VecDestroy(&mad->lb);CHKERRQ(ierr);
  ierr = VecDestroy(&mad->ub);CHKERRQ(ierr);
  ierr = ISDestroy(&mad->lb_idx);CHKERRQ(ierr);
  ierr = ISDestroy(&mad->ub_idx);CHKERRQ(ierr);
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_MAD(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_MAD        *mad = (TAO_MAD*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Multisecant Accelerated Descent (MAD) method for constrained optimization");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-tao_mad_vectors","number of vectors used in the approximation",NULL,mad->q,&mad->q,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_mad_alpha","step length parameter",NULL,mad->alpha,&mad->alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_mad_beta","Hessian regularization parameters",NULL,mad->beta,&mad->beta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_mad_max_step","maximum step size (L2 norm)",NULL,mad->max_step,&mad->max_step,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_mad_svd_cutoff","relative tolerance for truncating small singular values",NULL,mad->svd_cutoff,&mad->svd_cutoff,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = KSPSetFromOptions(tao->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_MAD(Tao tao, PetscViewer viewer)
{
  return 0;
}

/*MC
  TAOMAD - Multisecant Accelerated Descent algorithm for generally constrained optimization.

  Option Database Keys:

  Notes: This algorithm is more of a place-holder for future constrained optimization algorithms and should not yet be used for large problems or production code.
  Level: beginner

M*/

PETSC_EXTERN PetscErrorCode TaoCreate_MAD(Tao tao)
{
  TAO_MAD        *mad;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tao->ops->setup = TaoSetup_MAD;
  tao->ops->solve = TaoSolve_MAD;
  tao->ops->view = TaoView_MAD;
  tao->ops->setfromoptions = TaoSetFromOptions_MAD;
  tao->ops->destroy = TaoDestroy_MAD;

  ierr = PetscNewLog(tao,&mad);CHKERRQ(ierr);
  tao->data = (void*)mad;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 200;
  if (!tao->max_funcs_changed) tao->max_funcs = 500;
  
  /* Vector counters for the approximation */
  mad->k = 0;
  mad->q = 15;
  mad->alpha = 0.1;
  mad->beta = 0.5;
  mad->max_step = 1.0;
  mad->svd_cutoff = 1e-6;
  PetscFunctionReturn(0);
}

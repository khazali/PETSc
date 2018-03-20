#include <petsctaolinesearch.h>
#include <../src/tao/constrained/impls/mad/mad.h> /*I "ipm.h" I*/

PetscErrorCode TaoMADInitCompositeVecs(Tao tao)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  MPI_Comm           comm;
  PetscErrorCode     ierr;
  Vec                xtmp, btmp;
  
  PetscFunctionBegin;
  mad->nlb = mad->nub = mad->nb = 0;
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
      Vec cb[2];
      cb[0] = mad->cl;
      cb[1] = mad->cu;
      ierr = VecCreateNest(comm, 2, NULL, cb, &mad->cb);CHKERRQ(ierr);
      Vec cb_work[2];
      cb_work[0] = mad->cl_work;
      cb_work[1] = mad->cu_work;
      ierr = VecCreateNest(comm, 2, NULL, cb, &mad->cb_work);CHKERRQ(ierr);
    } else if (mad->cl) {
      mad->cb = mad->cl;
      mad->cb_work = mad->cl_work;
    } else if (mad->cu) {
      mad->cb = mad->cu;
      mad->cb_work = mad->cu_work;
    } else {
      SETERRQ(PETSC_COMM_SELF,1,"Error in counting bound constraint sizes!");
    }
  }
  
  /* create the composite inequality constraint vector */
  mad->ng = mad->nb + mad->nineq;
  if (mad->ng > 0) {
    if (mad->cin && mad->cb) {
      Vec g[2];
      g[0] = mad->cb;
      g[1] = mad->cin;
      ierr = VecCreateNest(comm, 2, NULL, g, &mad->g);CHKERRQ(ierr);
      Vec g_work[2];
      g_work[0] = mad->cb;
      g_work[1] = mad->cin;
      ierr = VecCreateNest(comm, 2, NULL, g_work, &mad->g_work);CHKERRQ(ierr);
    } else if (mad->cb) {
      mad->g = mad->cb;
      mad->g_work = mad->cb_work;
    } else if (mad->cin) {
      mad->g = mad->cin;
      mad->g_work = mad->cin_work;
    } else {
      SETERRQ(PETSC_COMM_SELF,1,"Error in counting inequality constraint sizes!");
    }
    ierr = VecDuplicate(mad->g, &mad->mu);CHKERRQ(ierr);
  }
  
  /* create the composite KKT vector */
  if (mad->h && mad->g) {
    Vec step[3];
    step[0] = mad->x;
    step[1] = mad->lambda;
    step[2] = mad->mu;
    ierr = VecCreateNest(comm, 3, NULL, step, &mad->step);CHKERRQ(ierr);
    Vec kkt[3];
    kkt[0] = mad->grad;
    kkt[1] = mad->h;
    kkt[2] = mad->g;
    ierr = VecCreateNest(comm, 3, NULL, kkt, &mad->kkt);CHKERRQ(ierr);
  } else if (mad->h) {
    Vec step[3];
    step[0] = mad->x;
    step[1] = mad->lambda;
    ierr = VecCreateNest(comm, 2, NULL, step, &mad->step);CHKERRQ(ierr);
    Vec kkt[3];
    kkt[0] = mad->grad;
    kkt[1] = mad->h;
    ierr = VecCreateNest(comm, 2, NULL, kkt, &mad->kkt);CHKERRQ(ierr);
  } else if (mad->g) {
    Vec step[3];
    step[0] = mad->x;
    step[1] = mad->mu;
    ierr = VecCreateNest(comm, 2, NULL, step, &mad->step);CHKERRQ(ierr);
    Vec kkt[3];
    kkt[0] = mad->grad;
    kkt[1] = mad->g;
    ierr = VecCreateNest(comm, 2, NULL, kkt, &mad->kkt);CHKERRQ(ierr);
  } else {
    mad->kkt = mad->grad;
    mad->step = mad->x;
  }
  
  /* allocate vectors for the multisecant approx */
  ierr = VecDuplicateVecs(mad->step, mad->q, &mad->Y);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(mad->kkt, mad->q, &mad->R);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(mad->step, mad->q-1, &mad->dY);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(mad->kkt, mad->q-1, &mad->dR);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADUpdateHistory(Tao tao, Vec y_new, Vec r_new) 
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  PetscErrorCode     ierr;
  PetscInt           i, last;
  VecType            vec_type;
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

PetscErrorCode TaoMADComputeBounds(Tao tao)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  PetscErrorCode     ierr;
  Vec                x_lb, x_ub;

  PetscFunctionBegin;
  if (mad->cl) {
    ierr = VecGetSubVector(tao->solution, mad->lb_idx, &x_lb);CHKERRQ(ierr);
    ierr = VecCopy(x_lb, mad->cl);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(tao->solution, mad->lb_idx, &x_lb);CHKERRQ(ierr);
    ierr = VecAXPBY(mad->cl, -1.0, 1.0, mad->lb);CHKERRQ(ierr);
  }
  if (mad->cu) {
    ierr = VecGetSubVector(tao->solution, mad->ub_idx, &x_ub);CHKERRQ(ierr);
    ierr = VecCopy(x_ub, mad->cu);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(tao->solution, mad->ub_idx, &x_ub);CHKERRQ(ierr);
    ierr = VecAXPBY(mad->cu, 1.0, -1.0, mad->ub);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADComputeInequality(Tao tao)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  PetscErrorCode     ierr;
  
  PetscFunctionBegin;
  if (mad->cin) {
    ierr = TaoComputeInequalityConstraints(tao, tao->solution, tao->constraints_inequality);CHKERRQ(ierr);
    ierr = TaoComputeJacobianInequality(tao, tao->solution, tao->jacobian_inequality, tao->jacobian_inequality_pre);CHKERRQ(ierr);
    ierr = VecCopy(tao->constraints_inequality, mad->cin);CHKERRQ(ierr);
  }
  if (mad->cb) {
    ierr = TaoMADComputeBounds(tao);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADComputeKKTCond(Tao tao)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  PetscErrorCode     ierr;
  
  PetscFunctionBegin;
  ierr = TaoComputeObjectiveAndGradient(tao, tao->solution, mad->f, tao->gradient);CHKERRQ(ierr);
  ierr = VecCopy(tao->gradient, mad->grad);CHKERRQ(ierr);
  if (mad->h) {
    ierr = TaoComputeEqualityConstraints(tao, tao->solution, tao->constraints_equality);CHKERRQ(ierr);
    ierr = TaoComputeJacobianEquality(tao, tao->solution, tao->jacobian_equality, tao->jacobian_equality_pre);CHKERRQ(ierr);
    ierr = VecCopy(tao->constraints_equality, mad->h);CHKERRQ(ierr);
  }
  if (mad->g) {
    ierr = TaoMADComputeInequality
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSolve_MAD(Tao tao)
{
  TAO_MAD            *mad = (TAO_MAD*)tao->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;

  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetup_MAD(Tao tao)
{
  TAO_MAD        *mad = (TAO_MAD*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mad->nlb = mad->nub = mad->nb = mad->nineq = mad->nh = mad->ng = 0;
  ierr = VecGetSize(tao->solution, &mad->nd);CHKERRQ(ierr);
  if (!tao->gradient) {
    ierr = VecDuplicate(tao->solution, &tao->gradient);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &tao->stepdirection);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &mad->x);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->solution, &mad->x_work);CHKERRQ(ierr);
  }
  if (tao->constraints_equality) {
    ierr = VecGetSize(tao->constraints_equality, &mad->nh);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality, &mad->lambda);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality, &mad->lambda_work);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality, &mad->h);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality, &mad->h_work);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_equality, &tao->DE);CHKERRQ(ierr);
  }
  if (tao->constraints_inequality) {
    ierr = VecGetSize(tao->constraints_inequality, &mad->nineq);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality, &mad->cin);CHKERRQ(ierr);
    ierr = VecDuplicate(tao->constraints_inequality, &tao->DI);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_MAD(Tao tao)
{
  TAO_MAD        *mad = (TAO_MAD*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;

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
  PetscFunctionReturn(0);
}

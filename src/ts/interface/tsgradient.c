#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/

static PetscErrorCode TSGradientEvalCostFunctionals(TS ts, PetscReal time, Vec state, Vec design, PetscReal *val)
{
  PetscErrorCode     ierr;
  CostFunctionalLink link = ts->funchead;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidPointer(val,4);
  ierr = VecLockPush(state);CHKERRQ(ierr);
  ierr = VecLockPush(design);CHKERRQ(ierr);
  *val = 0.0;
  while (link) {
    PetscReal v = 0.0;
    if (link->f && link->fixedtime <= PETSC_MIN_REAL) {
      ierr = (*link->f)(ts,time,state,design,&v,link->f_ctx);CHKERRQ(ierr);
    }
    *val += v;
    link = link->next;
  }
  ierr = VecLockPop(state);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGradientEvalCostFunctionalsFixed(TS ts, PetscReal time, Vec state, Vec design, PetscReal *val)
{
  PetscErrorCode     ierr;
  CostFunctionalLink link = ts->funchead;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidPointer(val,4);
  ierr = VecLockPush(state);CHKERRQ(ierr);
  ierr = VecLockPush(design);CHKERRQ(ierr);
  *val = 0.0;
  while (link) {
    PetscReal v = 0.0;
    if (link->f && PetscAbsReal(link->fixedtime-time) < PETSC_SMALL) {
      ierr = (*link->f)(ts,time,state,design,&v,link->f_ctx);CHKERRQ(ierr);
    }
    *val += v;
    link = link->next;
  }
  ierr = VecLockPop(state);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGradientEvalCostGradientU(TS ts, PetscReal time, Vec state, Vec design, Vec work, Vec out)
{
  PetscErrorCode     ierr;
  CostFunctionalLink link = ts->funchead;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidHeaderSpecific(work,VEC_CLASSID,4);
  PetscValidHeaderSpecific(out,VEC_CLASSID,4);
  ierr = VecLockPush(state);CHKERRQ(ierr);
  ierr = VecLockPush(design);CHKERRQ(ierr);
  ierr = VecSet(out,0.0);CHKERRQ(ierr);
  while (link) {
    if (link->f_x && link->fixedtime <= PETSC_MIN_REAL) {
      ierr = VecSet(work,0.0);CHKERRQ(ierr);
      ierr = (*link->f_x)(ts,time,state,design,work,link->f_x_ctx);CHKERRQ(ierr);
      ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
    }
    link = link->next;
  }
  ierr = VecLockPop(state);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGradientEvalCostGradientUFixed(TS ts, PetscReal time, Vec state, Vec design, Vec work, Vec out)
{
  PetscErrorCode     ierr;
  CostFunctionalLink link = ts->funchead;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidHeaderSpecific(work,VEC_CLASSID,4);
  PetscValidHeaderSpecific(out,VEC_CLASSID,4);
  ierr = VecLockPush(state);CHKERRQ(ierr);
  ierr = VecLockPush(design);CHKERRQ(ierr);
  ierr = VecSet(out,0.0);CHKERRQ(ierr);
  while (link) {
    if (link->f_x && link->fixedtime > PETSC_MIN_REAL && PetscAbsReal(link->fixedtime-time) < PETSC_SMALL) {
      ierr = VecSet(work,0.0);CHKERRQ(ierr);
      ierr = (*link->f_x)(ts,time,state,design,work,link->f_x_ctx);CHKERRQ(ierr);
      ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
    }
    link = link->next;
  }
  ierr = VecLockPop(state);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGradientEvalCostGradientM(TS ts, PetscReal time, Vec state, Vec design, Vec work, Vec out)
{
  PetscErrorCode     ierr;
  CostFunctionalLink link = ts->funchead;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidHeaderSpecific(work,VEC_CLASSID,4);
  PetscValidHeaderSpecific(out,VEC_CLASSID,4);
  ierr = VecLockPush(state);CHKERRQ(ierr);
  ierr = VecLockPush(design);CHKERRQ(ierr);
  ierr = VecSet(out,0.0);CHKERRQ(ierr);
  while (link) {
    if (link->f_m && link->fixedtime <= PETSC_MIN_REAL) {
      ierr = VecSet(work,0.0);CHKERRQ(ierr);
      ierr = (*link->f_m)(ts,time,state,design,work,link->f_m_ctx);CHKERRQ(ierr);
      ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
    }
    link = link->next;
  }
  ierr = VecLockPop(state);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  Vec       design;        /* design vector (fixed) */
  TS        fwdts;         /* forward solver */
  PetscReal t0,tf;         /* time limits, for forward time recovery */
  Vec       *W;            /* work vectors W[0] and W[1] always store U and Udot at a given time */
  Mat       splitJ_U;      /* Jacobian : F_U (U,Udot,fwdt) */
  Mat       splitJ_Udot;   /* Jacobian : F_Udot(U,Udot,fwdt) */
  Mat       splitJ_dtUdot; /* Jacobian : d/dt F_Udot(U,Udot,fwdt) */
  PetscBool firststep;     /* used for trapz rule */
  Vec       gradient;      /* gradient we are evaluating */
  Vec       *wgrad;        /* gradient work vectors (for trapz rule) */
} AdjointCtx;

static PetscErrorCode AdjointTSDestroy_Private(void *ptr)
{
  AdjointCtx*    adj = (AdjointCtx*)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&adj->design);CHKERRQ(ierr);
  ierr = VecDestroyVecs(3,&adj->W);CHKERRQ(ierr);
  ierr = MatDestroy(&adj->splitJ_U);CHKERRQ(ierr);
  ierr = MatDestroy(&adj->splitJ_Udot);CHKERRQ(ierr);
  ierr = MatDestroy(&adj->splitJ_dtUdot);CHKERRQ(ierr);
  ierr = VecDestroy(&adj->gradient);CHKERRQ(ierr);
  ierr = VecDestroyVecs(2,&adj->wgrad);CHKERRQ(ierr);
  ierr = PetscFree(adj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSUpdateHistory(TS adjts, PetscReal time, PetscBool U, PetscBool Udot)
{
  AdjointCtx     *adj_ctx;
  PetscReal      ft;
  PetscInt       step = PETSC_MIN_INT;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  if (!adj_ctx->fwdts->trajectory) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Missing TSTrajectory object");
  ft   = adj_ctx->tf - time + adj_ctx->t0;
  ierr = VecLockPop(adj_ctx->W[0]);CHKERRQ(ierr);
  ierr = VecLockPop(adj_ctx->W[1]);CHKERRQ(ierr);
  ierr = TSTrajectoryGetVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,step,&ft,U ? adj_ctx->W[0] : NULL, Udot ? adj_ctx->W[1] : NULL);CHKERRQ(ierr);
  ierr = VecLockPush(adj_ctx->W[0]);CHKERRQ(ierr);
  ierr = VecLockPush(adj_ctx->W[1]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* The assumption here is that the RHSJacobian routine is called after the RHSFunction has been called with same time and U
   The evaluation of the forward RHS jacobian is always called with an updated history vector adj_ctx->W[0]
   since F_U needs to be evaluated in the RHSFunction */
static PetscErrorCode AdjointTSRHSJacobian(TS adjts, PetscReal time, Vec U, Mat A, Mat P, void *ctx)
{
  AdjointCtx     *adj_ctx;
  PetscReal      ft;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  ft   = adj_ctx->tf - time + adj_ctx->t0;
  ierr = TSComputeRHSJacobian(adj_ctx->fwdts,ft,adj_ctx->W[0],A,P);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSRHSFuncLinear(TS adjts, PetscReal time, Vec U, Vec F, void *ctx)
{
  AdjointCtx     *adj_ctx;
  PetscReal      fwdt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = AdjointTSUpdateHistory(adjts,time,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - time + adj_ctx->t0;
  ierr = TSGradientEvalCostGradientU(adj_ctx->fwdts,fwdt,adj_ctx->W[0],adj_ctx->design,adj_ctx->W[2],F);CHKERRQ(ierr);
  /* the adjoint formulation I'm using assumes F(U,Udot,t) = 0
     -> the forward PDE is Udot - G(U) = 0
     -> the adjoint PDE is F - L^T * G_U - Ldot^T in backward time
     -> the adjoint PDE is Ldot^T = L^T * G_U - F in forward time */
  ierr = VecScale(F,-1.0);CHKERRQ(ierr);
  ierr = TSComputeRHSJacobian(adjts,time,U,adjts->Arhs,NULL);CHKERRQ(ierr);
  ierr = MatMultTransposeAdd(adjts->Arhs,U,F,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeSplitJacobians(TS ts, PetscReal time, Vec U, Vec Udot, Mat A, Mat pA, Mat B, Mat pB, Mat C, Mat pC)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(U,VEC_CLASSID,3);
  PetscValidHeaderSpecific(Udot,VEC_CLASSID,4);
  PetscValidHeaderSpecific(A,MAT_CLASSID,5);
  PetscValidHeaderSpecific(pA,MAT_CLASSID,6);
  PetscValidHeaderSpecific(B,MAT_CLASSID,7);
  PetscValidHeaderSpecific(pB,MAT_CLASSID,8);
  if (A == B) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"A and B must be different matrices");
  if (pA == pB) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"pA and pB must be different matrices");
  /* PetscValidHeaderSpecific(C,MAT_CLASSID,9); */
  /* PetscValidHeaderSpecific(pC,MAT_CLASSID,10); */
  /* this is a fallback if the user does not provide the Jacobians through TSSetSplitJacobians */
  ierr = TSComputeIJacobian(ts,time,U,Udot,0.0,A,pA,PETSC_FALSE);CHKERRQ(ierr);
  ierr = TSComputeIJacobian(ts,time,U,Udot,1.0,B,pB,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatAXPY(B,-1.0,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  if (pB != B) {
    ierr = MatAXPY(pB,-1.0,pA,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  if (C) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Time derivative of F_Udot not yet implemented");
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSUpdateSplitJacobians(TS adjts, PetscReal time)
{
  AdjointCtx     *adj_ctx;
  PetscReal      fwdt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = AdjointTSUpdateHistory(adjts,time,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - time + adj_ctx->t0;
  ierr = TSComputeSplitJacobians(adj_ctx->fwdts,fwdt,adj_ctx->W[0],adj_ctx->W[1],
                                 adj_ctx->splitJ_U,adj_ctx->splitJ_U,
                                 adj_ctx->splitJ_Udot,adj_ctx->splitJ_Udot,
                                 adj_ctx->splitJ_dtUdot,adj_ctx->splitJ_dtUdot);CHKERRQ(ierr);
  if (adj_ctx->splitJ_dtUdot) {
    ierr = MatAXPY(adj_ctx->splitJ_U,1.0,adj_ctx->splitJ_dtUdot,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatZeroEntries(adj_ctx->splitJ_dtUdot);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* the assumption here is that the IJacobian routine is called after the IFunction (called with same time, U and Udot) */
static PetscErrorCode AdjointTSIJacobian(TS adjts, PetscReal time, Vec U, Vec Udot, PetscReal shift, Mat A, Mat B, void *ctx)
{
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  ierr = MatCopy(adj_ctx->splitJ_Udot,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatScale(A,shift);CHKERRQ(ierr);
  ierr = MatAXPY(A,1.0,adj_ctx->splitJ_U,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  if (B && A != B) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_SUP,"B != A not yet implemented");
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSIFuncLinear(TS adjts, PetscReal time, Vec U, Vec Udot, Vec F, void *ctx)
{
  AdjointCtx     *adj_ctx;
  PetscReal      fwdt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = AdjointTSUpdateHistory(adjts,time,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - time + adj_ctx->t0;
  ierr = TSGradientEvalCostGradientU(adj_ctx->fwdts,fwdt,adj_ctx->W[0],adj_ctx->design,adj_ctx->W[2],F);CHKERRQ(ierr);
  ierr = AdjointTSUpdateSplitJacobians(adjts,time);CHKERRQ(ierr);
  ierr = MatMultTransposeAdd(adj_ctx->splitJ_U,U,F,F);CHKERRQ(ierr);
  ierr = MatMultTransposeAdd(adj_ctx->splitJ_Udot,Udot,F,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSPostStep(TS adjts)
{
  Vec            lambda;
  PetscReal      dt,time,ptime,fwdt;
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  if (adjts->reason < 0) PetscFunctionReturn(0);
  ierr = TSGetTime(adjts,&time);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - time + adj_ctx->t0;

  /* trapezoidal rule */
  ierr = TSGetPrevTime(adjts,&ptime);CHKERRQ(ierr);
  dt   = time - ptime;

  /* first step:
       wgrad[0] has been initialized with the first backward evaluation of \lambda^T F_m at t0
       gradient with the forward contribution to the gradient (i.e. \int f_m )
  */
  if (adj_ctx->firststep) {
    ierr = VecScale(adj_ctx->wgrad[0],dt/2.0);CHKERRQ(ierr);
    ierr = VecAXPY(adj_ctx->gradient,1.0,adj_ctx->wgrad[0]);CHKERRQ(ierr);
    ierr = VecSet(adj_ctx->wgrad[1],0.0);CHKERRQ(ierr);
  }
  if (adj_ctx->fwdts->F_m) {
    PetscScalar tt[2];

    TS ts = adj_ctx->fwdts;
    if (ts->F_m_f) { /* non constant dependence */
      ierr = AdjointTSUpdateHistory(adjts,time,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
      ierr = (*ts->F_m_f)(ts,fwdt,adj_ctx->W[0],adj_ctx->W[1],adj_ctx->design,ts->F_m,ts->F_m_ctx);CHKERRQ(ierr);
    }
    ierr = TSGetSolution(adjts,&lambda);CHKERRQ(ierr);
    ierr = MatMultTranspose(ts->F_m,lambda,adj_ctx->wgrad[0]);CHKERRQ(ierr);
    tt[0] = tt[1] = dt/2.0;
    ierr = VecMAXPY(adj_ctx->gradient,2,tt,adj_ctx->wgrad);CHKERRQ(ierr);
    /* XXX this could be done more efficiently */
    ierr = VecCopy(adj_ctx->wgrad[0],adj_ctx->wgrad[1]);CHKERRQ(ierr);
  }
  adj_ctx->firststep = PETSC_FALSE;
  if (adjts->reason) { adj_ctx->tf = time; } /* prevent from accumulation errors XXX */
  PetscFunctionReturn(0);
}

static PetscErrorCode TSCreateAdjointTS(TS ts, TS* adjts)
{
  Mat             A,B;
  Vec             U,vatol,vrtol;
  PetscContainer  container;
  AdjointCtx      *adj;
  TSIFunction     ifunc;
  TSRHSFunction   rhsfunc;
  TSRHSJacobian   rhsjacfunc;
  TSType          type;
  const char      *prefix;
  PetscReal       atol,rtol;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = TSCreate(PetscObjectComm((PetscObject)ts),adjts);CHKERRQ(ierr);
  ierr = TSGetType(ts,&type);CHKERRQ(ierr);
  ierr = TSSetType(*adjts,type);CHKERRQ(ierr);
  ierr = TSGetTolerances(ts,&atol,&vatol,&rtol,&vrtol);CHKERRQ(ierr);
  ierr = TSSetTolerances(*adjts,atol,vatol,rtol,vrtol);CHKERRQ(ierr);

  /* application context */
  ierr = PetscNew(&adj);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(*adjts,(void *)adj);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(U,3,&adj->W);CHKERRQ(ierr);
  adj->fwdts = ts; /* we don't take reference on the forward ts, as adjts in not public */
  adj->t0 = adj->tf = PETSC_MAX_REAL;

  /* wrap application context in a container, so that it will be destroyed when calling TSDestroy on adjts */
  ierr = PetscContainerCreate(PetscObjectComm((PetscObject)(*adjts)),&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,adj);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(container,AdjointTSDestroy_Private);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)(*adjts),"_ts_gradient_adjctx",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);

  /* setup callbacks for adjoint ode: we reuse the same jacobian matrices of the forward solve */
  ierr = TSGetIFunction(ts,NULL,&ifunc,NULL);CHKERRQ(ierr);
  ierr = TSGetRHSFunction(ts,NULL,&rhsfunc,NULL);CHKERRQ(ierr);
  if (ifunc) {
    ierr = TSGetIJacobian(ts,&A,&B,NULL,NULL);CHKERRQ(ierr);
    ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&adj->splitJ_U);CHKERRQ(ierr);
    ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&adj->splitJ_Udot);CHKERRQ(ierr);
    /* ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&adj->splitJ_dtUdot);CHKERRQ(ierr); */
    ierr = TSSetIFunction(*adjts,NULL,AdjointTSIFuncLinear,NULL);CHKERRQ(ierr);
    ierr = TSSetIJacobian(*adjts,A,B,AdjointTSIJacobian,NULL);CHKERRQ(ierr);
  } else {
    if (!rhsfunc) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"TSSetIFunction or TSSetRHSFunction not called");
    ierr = TSSetRHSFunction(*adjts,NULL,AdjointTSRHSFuncLinear,NULL);CHKERRQ(ierr);
    ierr = TSGetRHSJacobian(ts,&A,&B,&rhsjacfunc,NULL);CHKERRQ(ierr);
    if (rhsjacfunc == TSComputeRHSJacobianConstant) {
      ierr = TSSetRHSJacobian(*adjts,A,B,TSComputeRHSJacobianConstant,NULL);CHKERRQ(ierr);
    } else {
      ierr = TSSetRHSJacobian(*adjts,A,B,AdjointTSRHSJacobian,NULL);CHKERRQ(ierr);
    }
  }

  /* prefix */
  ierr = TSGetOptionsPrefix(ts,&prefix);CHKERRQ(ierr);
  ierr = TSSetOptionsPrefix(*adjts,prefix);CHKERRQ(ierr);
  ierr = TSAppendOptionsPrefix(*adjts,"adjoint_");CHKERRQ(ierr);
  ierr = TSSetFromOptions(*adjts);CHKERRQ(ierr);

  /* adjoint ODE is linear */
  ierr = TSSetProblemType(*adjts,TS_LINEAR);CHKERRQ(ierr);

  /* XXX use KSPSolveTranspose to solve the adjoint */

  /* set special purpose post step method for incremental gradient evaluation */
  ierr = TSSetPostStep(*adjts,AdjointTSPostStep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSSetInitialGradient(TS adjts, Vec gradient)
{
  Vec            lambda;
  PetscReal      norm;
  PetscContainer c;
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  PetscValidHeaderSpecific(gradient,VEC_CLASSID,2);
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_gradient_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Missing adjoint container");
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  if (adj_ctx->t0 >= PETSC_MAX_REAL || adj_ctx->tf >= PETSC_MAX_REAL) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_ORDER,"You should call AdjointTSSetTimeLimits first");
  if (!adj_ctx->design) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_ORDER,"You should call AdjointTSSetDesign first");

  adj_ctx->firststep = PETSC_TRUE;
  ierr = PetscObjectReference((PetscObject)gradient);CHKERRQ(ierr);
  ierr = VecDestroy(&adj_ctx->gradient);CHKERRQ(ierr);
  adj_ctx->gradient = gradient;
  if (!adj_ctx->wgrad) {
    ierr = VecDuplicateVecs(gradient,2,&adj_ctx->wgrad);CHKERRQ(ierr);
  }

  /* Set initial conditions for the adjoint ode */
  ierr = TSGetSolution(adjts,&lambda);CHKERRQ(ierr);
  ierr = TSGradientEvalCostGradientUFixed(adj_ctx->fwdts,adj_ctx->tf,lambda,adj_ctx->design,adj_ctx->W[0],adj_ctx->W[1]);CHKERRQ(ierr);
  ierr = VecNorm(adj_ctx->W[1],NORM_2,&norm);CHKERRQ(ierr);

  /* these two vectors are locked: only AdjointTSUpdateHistory can unlock them */
  ierr = VecLockPush(adj_ctx->W[0]);CHKERRQ(ierr);
  ierr = VecLockPush(adj_ctx->W[1]);CHKERRQ(ierr);

  if (norm > PETSC_SMALL) {
    TSIJacobian ijac;

    ierr = TSGetIJacobian(adjts,NULL,NULL,&ijac,NULL);CHKERRQ(ierr);
    if (ijac) { /* lambda(T) = - (F_Udot)^T D_x, D_x the gradients of the functionals that sample the solution at the final time */
      SNES snes;
      KSP  ksp;

      ierr = AdjointTSUpdateSplitJacobians(adjts,adj_ctx->t0);CHKERRQ(ierr); /* split uses backward fime */
      ierr = TSGetSNES(adjts,&snes);CHKERRQ(ierr);
      ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,adj_ctx->splitJ_Udot,adj_ctx->splitJ_Udot);CHKERRQ(ierr);
      ierr = KSPSolveTranspose(ksp,adj_ctx->W[1],lambda);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(adj_ctx->W[1],lambda);CHKERRQ(ierr);
    }
    ierr = VecScale(lambda,-1.0);CHKERRQ(ierr);
  } else {
    ierr = VecSet(lambda,0.0);CHKERRQ(ierr);
  }

  /* initialize wgrad[0] */
  if (adj_ctx->fwdts->F_m) {
    TS ts = adj_ctx->fwdts;
    if (ts->F_m_f) { /* non constant dependence */
      ierr = AdjointTSUpdateHistory(adjts,adj_ctx->t0,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
      ierr = (*ts->F_m_f)(ts,adj_ctx->tf,adj_ctx->W[0],adj_ctx->W[1],adj_ctx->design,ts->F_m,ts->F_m_ctx);CHKERRQ(ierr);
    }
    ierr = TSGetSolution(adjts,&lambda);CHKERRQ(ierr);
    ierr = MatMultTranspose(ts->F_m,lambda,adj_ctx->wgrad[0]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSSetDesign(TS adjts, Vec design)
{
  PetscContainer c;
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  PetscValidHeaderSpecific(design,VEC_CLASSID,2);
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_gradient_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Missing adjoint container");
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)design);CHKERRQ(ierr);
  ierr = VecDestroy(&adj_ctx->design);CHKERRQ(ierr);
  adj_ctx->design = design;
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSSetTimeLimits(TS adjts, PetscReal t0, PetscReal tf, PetscReal dt)
{
  PetscContainer c;
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_gradient_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Missing adjoint container");
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  ierr = TSSetTime(adjts,t0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(adjts,PetscMin(dt,tf-t0));CHKERRQ(ierr);
  ierr = TSSetMaxSteps(adjts,PETSC_MAX_INT);CHKERRQ(ierr);
  ierr = TSSetMaxTime(adjts,tf);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(adjts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  /* update time limits in the application context
     they are needed to recover the forward time from the backward */
  adj_ctx->tf = tf;
  adj_ctx->t0 = t0;
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSComputeFinalGradient(TS adjts)
{
  TS             fwdts;
  PetscReal      tf;
  PetscContainer c;
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_gradient_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Missing adjoint container");
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  if (!adj_ctx->gradient) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_ORDER,"Missing gradient vector");
  ierr = TSGetTime(adjts,&tf);CHKERRQ(ierr);
  if (tf < adj_ctx->tf) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_ORDER,"Backward solve did not complete");

  /* initial condition contribution to the gradient */
  fwdts = adj_ctx->fwdts;
  if (fwdts->G_m) {
    Vec         lambda;
    TSIJacobian ijacfunc;

    ierr = TSGetSolution(adjts,&lambda);CHKERRQ(ierr);
    ierr = TSGetIJacobian(adjts,NULL,NULL,&ijacfunc,NULL);CHKERRQ(ierr);
    if (!ijacfunc) {
      ierr = VecCopy(lambda,adj_ctx->W[2]);CHKERRQ(ierr);
    } else {
      ierr = MatMultTranspose(adj_ctx->splitJ_Udot,lambda,adj_ctx->W[2]);CHKERRQ(ierr);
    }

    if (fwdts->G_x) { /* this is optional. If not provided, identity is assumed */
      KSP ksp;

      ierr = KSPCreate(PetscObjectComm((PetscObject)adjts),&ksp);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,fwdts->G_x,fwdts->G_x);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(ksp,"adjoint_G_");CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
      ierr = KSPSolveTranspose(ksp,adj_ctx->W[2],adj_ctx->W[2]);CHKERRQ(ierr);
      ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    }
    ierr = MatMultTransposeAdd(fwdts->G_m,adj_ctx->W[2],adj_ctx->gradient,adj_ctx->gradient);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

typedef struct {
  PetscErrorCode (*user)(TS); /* user post step method */
  Vec            design;      /* the design vector we are evaluating against */
  PetscBool      objeval;     /* indicates we have to evalute the cost functionals */
  PetscReal      obj;         /* objective function value */
  Vec            gradient;    /* used when f_m is not zero, and it is evaluated during the forward run */
  Vec            *wgrad;      /* gradient work vectors */
  PetscBool      firststep;   /* used for trapz rule */
  PetscReal      pval;        /* previous value (for trapz rule) */
} TSGradientPostStepCtx;

static PetscErrorCode TSGradientPostStep(TS ts)
{
  PetscContainer        container;
  Vec                   solution;
  TSGradientPostStepCtx *poststep_ctx;
  PetscReal             val = 0.0;
  PetscReal             dt,time,ptime;
  PetscInt              step;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->reason < 0) PetscFunctionReturn(0);
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_gradient_poststep",(PetscObject*)&container);CHKERRQ(ierr);
  if (!container) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Missing poststep container");
  ierr = PetscContainerGetPointer(container,(void**)&poststep_ctx);CHKERRQ(ierr);
  if (poststep_ctx->user) {
    ierr = (*poststep_ctx->user)(ts);CHKERRQ(ierr);
  }

  ierr = TSGetSolution(ts,&solution);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&time);CHKERRQ(ierr);
  if (poststep_ctx->objeval) {
    ierr = TSGradientEvalCostFunctionals(ts,time,solution,poststep_ctx->design,&val);CHKERRQ(ierr);
  }

  /* time step used */
  ierr = TSGetPrevTime(ts,&ptime);
  dt   = time - ptime;

  /* first step: obj has been initialized with the first function evaluation at t0
     and gradient with the first gradient evaluation */
  ierr = TSGetStepNumber(ts,&step);CHKERRQ(ierr);
  if (poststep_ctx->firststep) {
    poststep_ctx->obj *= dt/2.0;
    poststep_ctx->pval = 0.0;
  }
  poststep_ctx->obj += dt*(val+poststep_ctx->pval)/2.0;
  poststep_ctx->pval = val;
  if (poststep_ctx->objeval) {
    ierr = TSGradientEvalCostFunctionalsFixed(ts,time,solution,poststep_ctx->design,&val);CHKERRQ(ierr);
    poststep_ctx->obj += val;
  }
  if (poststep_ctx->gradient) {
    PetscScalar tt[2];

    if (poststep_ctx->firststep) {
      ierr = VecSet(poststep_ctx->wgrad[2],0.0);CHKERRQ(ierr);
      ierr = VecScale(poststep_ctx->gradient,dt/2.0);CHKERRQ(ierr);
    }
    ierr = TSGradientEvalCostGradientM(ts,time,solution,poststep_ctx->design,poststep_ctx->wgrad[0],poststep_ctx->wgrad[1]);CHKERRQ(ierr);
    tt[0] = tt[1] = dt/2.0;
    ierr = VecMAXPY(poststep_ctx->gradient,2,tt,poststep_ctx->wgrad+1);CHKERRQ(ierr);
    /* XXX this could be done more efficiently */
    ierr = VecCopy(poststep_ctx->wgrad[1],poststep_ctx->wgrad[2]);CHKERRQ(ierr);
  }
  poststep_ctx->firststep = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEvaluateCostFunctionals_Private(TS ts, Vec X, Vec design, Vec gradient, PetscReal *val)
{
  Vec                   U;
  PetscContainer        container;
  TSGradientPostStepCtx poststep_ctx;
  PetscReal             t0;
  PetscInt              tst;
  PetscBool             destroyX;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (!gradient && !val) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Either val or gradient should be asked");
  if (gradient) PetscValidHeaderSpecific(gradient,VEC_CLASSID,4);
  if (!ts->funchead) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Missing cost functionals");
  /* solution vector */
  destroyX = PETSC_FALSE;
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  if (!X) {
    if (!U) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Missing solution vector");
    ierr = VecDuplicate(U,&X);CHKERRQ(ierr);
    ierr = VecSet(X,0.0);CHKERRQ(ierr);
    destroyX = PETSC_TRUE;
  }
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);

  /* set special purpose post step method */
  poststep_ctx.user      = ts->poststep;
  poststep_ctx.design    = design;
  poststep_ctx.firststep = PETSC_TRUE;
  poststep_ctx.objeval   = val ? PETSC_TRUE : PETSC_FALSE;
  poststep_ctx.obj       = 0.0;
  poststep_ctx.gradient  = gradient;
  poststep_ctx.wgrad     = NULL;
  ierr = PetscContainerCreate(PetscObjectComm((PetscObject)ts),&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,(void*)&poststep_ctx);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradient_poststep",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  ierr = TSSetPostStep(ts,TSGradientPostStep);CHKERRQ(ierr);

  /* evaluate at initial time */
  ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
  if (poststep_ctx.objeval) {
    ierr = TSGradientEvalCostFunctionals(ts,t0,X,design,&poststep_ctx.obj);CHKERRQ(ierr);
  }
  if (poststep_ctx.gradient) {
    ierr = VecDuplicateVecs(poststep_ctx.gradient,3,&poststep_ctx.wgrad);CHKERRQ(ierr);
    ierr = TSGradientEvalCostGradientM(ts,t0,X,design,poststep_ctx.wgrad[0],poststep_ctx.gradient);CHKERRQ(ierr);
  }

  /* forward solve */
  ierr = TSSetUp(ts);CHKERRQ(ierr);

  ierr = TSGetStepNumber(ts,&tst);CHKERRQ(ierr);
  ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
  ierr = TSSolve(ts,NULL);CHKERRQ(ierr);

  /* restore */
  ierr = TSSetStepNumber(ts,tst);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradient_poststep",NULL);CHKERRQ(ierr);
  ierr = TSSetPostStep(ts,poststep_ctx.user);CHKERRQ(ierr);
  if (U) {
    ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
  }
  if (destroyX) {
    ierr = VecDestroy(&X);CHKERRQ(ierr);
  }
  ierr = VecDestroyVecs(3,&poststep_ctx.wgrad);CHKERRQ(ierr);

  /* get back value */
  if (val) *val = poststep_ctx.obj;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEvaluateGradient_Private(TS ts, Vec X, Vec design, Vec gradient, PetscReal *val)
{
  TS             adjts;
  TSTrajectory   otrj;
  PetscReal      t0,tf,dt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  otrj = ts->trajectory;
  ierr = TSTrajectoryCreate(PetscObjectComm((PetscObject)ts),&ts->trajectory);CHKERRQ(ierr);
  ierr = TSTrajectorySetType(ts->trajectory,ts,TSTRAJECTORYBASIC);CHKERRQ(ierr);
  ierr = TSTrajectorySetFromOptions(ts->trajectory,ts);CHKERRQ(ierr);

  /* sample initial condition dependency */
  ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
  if (ts->Ggrad) {
    ierr = (*ts->Ggrad)(ts,t0,X,design,ts->G_x,ts->G_m,ts->Ggrad_ctx);CHKERRQ(ierr);
  }

  /* forward solve */
  ierr = TSEvaluateCostFunctionals_Private(ts,X,design,gradient,val);CHKERRQ(ierr);

  /* adjoint */
  ierr = TSCreateAdjointTS(ts,&adjts);CHKERRQ(ierr);
  ierr = TSSetSolution(adjts,X);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&tf);CHKERRQ(ierr);
  ierr = TSGetPrevTime(ts,&dt);CHKERRQ(ierr);
  dt   = tf - dt;
  ierr = AdjointTSSetTimeLimits(adjts,t0,tf,dt);CHKERRQ(ierr);
  ierr = AdjointTSSetDesign(adjts,design);CHKERRQ(ierr);
  ierr = AdjointTSSetInitialGradient(adjts,gradient);CHKERRQ(ierr); /* it also initializes the adjoint variable */
  ierr = TSSolve(adjts,NULL);CHKERRQ(ierr);
  ierr = AdjointTSComputeFinalGradient(adjts);CHKERRQ(ierr);
  ierr = TSDestroy(&adjts);CHKERRQ(ierr);

  /* restore trajectory */
  ierr = TSTrajectoryDestroy(&ts->trajectory);CHKERRQ(ierr);
  ts->trajectory  = otrj;
  PetscFunctionReturn(0);
}

/*
   TSResetCostFunctionals - Resets the list of cost functionals for gradient computation.

   Logically Collective on TS

   Input Parameters:
.  ts - the TS context

   Level: developer

.seealso:
*/
PetscErrorCode TSResetCostFunctionals(TS ts)
{
  CostFunctionalLink link;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  link = ts->funchead;
  while (link) {
    CostFunctionalLink olink = link;

    link = link->next;
    ierr = PetscFree(olink);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   TSSetCostFunctional - Sets a cost functional for gradient computation.

   Logically Collective on TS

   Input Parameters:
+  ts      - the TS context obtained from TSCreate()
.  fixtime - the time at which the functional has to be evaluated (use PETSC_MIN_REAL for integrands)
.  f       - the function evaluation routine
.  f_ctx   - user-defined context for private data for the function evaluation routine (may be NULL)
.  f_x     - the function evaluation routine for the derivative with respect to the state variables (may be NULL)
.  f_x_ctx - user-defined context for private data for the function evaluation routine (may be NULL)
.  f_m     - the function evaluation routine for the derivative with respect to the design variables (may be NULL)
-  f_m_ctx - user-defined context for private data for the function evaluation routine (may be NULL)

   Calling sequence of f:
$  f(TS ts,PetscReal t,Vec u,Vec m,PetscReal *out,void *ctx);

+  t   - time at step/stage being solved
.  u   - state vector
.  m   - design vector
.  out - output value
-  ctx - [optional] user-defined context

   Calling sequence of f_x and f_m:
$  f(TS ts,PetscReal t,Vec u,Vec m,Vec out,void ctx);

+  t   - time at step/stage being solved
.  u   - state vector
.  m   - design vector
.  out - output vector
-  ctx - [optional] user-defined context

   Notes: the functions passed in are appended to a list. More functions can be passed in
          by simply calling TSSetCostFunctional with different arguments.
          The functionals are intendended to be used as integrand terms of a time integration (if fixtime == PETSC_MIN_REAL) or as evaluation at a given specific time.
          The size of the output vectors equals the size of the state and design vectors for f_x and f_m, respectively.

   Level: developer

.seealso: TSSetEvalGradient(), TSEvaluateGradient(), TSSetEvalICGradient()
*/
PetscErrorCode TSSetCostFunctional(TS ts, PetscReal fixtime, TSEvalCostFunctional f, void* f_ctx, TSEvalCostGradient f_x, void* f_x_ctx, TSEvalCostGradient f_m, void* f_m_ctx)
{
  CostFunctionalLink link;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!ts->funchead) {
    ierr = PetscNew(&ts->funchead);CHKERRQ(ierr);
    link = ts->funchead;
  } else {
    link = ts->funchead;
    while (link->next) link = link->next;
    ierr = PetscNew(&link->next);CHKERRQ(ierr);
    link = link->next;
  }
  link->f         = f;
  link->f_ctx     = f_ctx;
  link->f_x       = f_x;
  link->f_x_ctx   = f_x_ctx;
  link->f_m       = f_m;
  link->f_m_ctx   = f_m_ctx;
  link->fixedtime = fixtime;
  PetscFunctionReturn(0);
}

/*
   TSSetEvalGradient - Sets the function for the evaluation of F_m(t,x(t),x_t(t);m) for gradient computation.

   Logically Collective on TS

   Input Parameters:
+  ts      - the TS context obtained from TSCreate()
.  J       - the Mat object to hold F_m(t,x(t),x_t(t);m)
.  f       - the function evaluation routine
-  f_ctx   - user-defined context for private data for the function evaluation routine (may be NULL)

   Calling sequence of f:
$  f(TS ts,PetscReal t,Vec u,Vec u_t,Vec m,Mat J,ctx);

+  t   - time at step/stage being solved
.  u   - state vector
.  u_t - time derivative of state vector
.  m   - design vector
.  J   - the jacobian
-  ctx - [optional] user-defined context

   Notes: The J matrix doesn't need to assembled. Just the transposed action on the adjoint state via MatMultTranspose() is needed.

   Level: developer

.seealso: TSSetCostFunctional(), TSEvaluateGradient(), TSSetEvalICGradient(), MATSHELL
*/
PetscErrorCode TSSetEvalGradient(TS ts, Mat J, TSEvalGradient f, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(J,MAT_CLASSID,2);
  ierr        = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
  ierr        = MatDestroy(&ts->F_m);CHKERRQ(ierr);
  ts->F_m     = J;
  ts->F_m_f   = f;
  ts->F_m_ctx = ctx;
  PetscFunctionReturn(0);
}

/*
   TSSetEvalICGradient - Sets the callback function to compute the matrices g_x(x0,m) and g_m(x0,m), if there is any dependence of the PDE initial conditions from the design parameters.

   Logically Collective on TS

   Input Parameters:
+  ts      - the TS context obtained from TSCreate()
.  J_x     - the Mat object to hold g_x(x0,m) (optional, if NULL identity is assumed)
.  J_m     - the Mat object to hold g_m(x0,m)
.  f       - the function evaluation routine
-  f_ctx   - user-defined context for private data for the function evaluation routine (may be NULL)

   Calling sequence of f:
$  f(TS ts,PetscReal t,Vec u,Vec m,Mat Gx,Mat Gm,ctx);

+  t   - initial time
.  u   - state vector (at initial time)
.  m   - design vector
.  Gx  - the Mat Object representing the operator g_x(u,m)
.  Gm  - the Mat object for g_m(u,m)
-  ctx - [optional] user-defined context

   Notes: J_x is a square matrix of the same size of the state vector. J_m is a rectangular matrix with "state size" rows and "design size" columns.
          If f is not provided, J_x is assumed constant. The J_m matrix doesn't need to assembled.
          Just the transposed action on the adjoint state via MatMultTranspose() is needed.

   Level: developer

.seealso: TSSetCostFunctional(), TSSetEvalGradient(), TSEvaluateGradient(), MATSHELL
*/
PetscErrorCode TSSetEvalICGradient(TS ts, Mat J_x, Mat J_m, TSEvalICGradient f, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (J_x) PetscValidHeaderSpecific(J_x,MAT_CLASSID,2);
  PetscValidHeaderSpecific(J_m,MAT_CLASSID,3);

  if (J_x) {
    ierr    = PetscObjectReference((PetscObject)J_x);CHKERRQ(ierr);
    ierr    = MatDestroy(&ts->G_x);CHKERRQ(ierr);
    ts->G_x = J_x;
  }
  ierr    = PetscObjectReference((PetscObject)J_m);CHKERRQ(ierr);
  ierr    = MatDestroy(&ts->G_m);CHKERRQ(ierr);
  ts->G_m = J_m;

  ts->Ggrad     = f;
  ts->Ggrad_ctx = ctx;
  PetscFunctionReturn(0);
}

/*
   TSEvaluateCostFunctionals - Evaluates the cost functionals.

   Logically Collective on TS

   Input Parameters:
+  ts     - the TS context
.  X      - the initial vector for the state (can be NULL)
-  design - current design state

   Output Parameters:
.  value - the value of the functional

   Notes:

   Level: developer

.seealso: TSSetCostFunctional(), TSSetEvalGradient(), TSSetEvalICGradient(), TSEvaluateGradient(), TSEvaluateCostAndGradient()
*/
PetscErrorCode TSEvaluateCostFunctionals(TS ts, Vec X, Vec design, PetscReal *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (X) PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidPointer(val,4);
  ierr = VecLockPush(design);CHKERRQ(ierr);
  ierr = TSEvaluateCostFunctionals_Private(ts,X,design,NULL,val);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   TSEvaluateGradient - Evaluates the gradient of the cost functionals.

   Logically Collective on TS

   Input Parameters:
+  ts       - the TS context
.  X        - the initial vector for the state (can be NULL)
-  design   - current design state

   Output Parameters:
.  gradient - the computed gradient

   Notes:

   Level: developer

.seealso: TSSetCostFunctional(), TSSetEvalGradient(), TSSetEvalICGradient(), TSEvaluateCostFunctionals(), TSEvaluateCostAndGradient()
*/
PetscErrorCode TSEvaluateGradient(TS ts, Vec X, Vec design, Vec gradient)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidHeaderSpecific(gradient,VEC_CLASSID,4);
  ierr = VecLockPush(design);CHKERRQ(ierr);
  ierr = TSEvaluateGradient_Private(ts,X,design,gradient,NULL);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   TSEvaluateCostAndGradient - Evaluates the cost functionals and their gradient

   Logically Collective on TS

   Input Parameters:
+  ts       - the TS context
.  X        - the initial vector for the state
-  design   - current design state

   Output Parameters:
+  obj      - the value of the objective function
-  gradient - the computed gradient

   Notes:

   Level: developer

.seealso: TSSetCostFunctional(), TSSetEvalGradient(), TSSetEvalICGradient(), TSEvaluateCostFunctionals(), TSEvaluateGradient()
*/
PetscErrorCode TSEvaluateCostAndGradient(TS ts, Vec X, Vec design, Vec gradient, PetscReal *obj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidHeaderSpecific(gradient,VEC_CLASSID,4);
  PetscValidPointer(obj,5);

  ierr = VecLockPush(design);CHKERRQ(ierr);
  ierr = TSEvaluateGradient_Private(ts,X,design,gradient,obj);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

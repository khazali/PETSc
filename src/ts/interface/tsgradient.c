#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petsc/private/snesimpl.h>

/* ------------------ Helper routines for PDE-constrained support, namespaced with TSGradient ----------------------- */

/* Evaluates objective functions of the type f(U,param,t) */
static PetscErrorCode TSGradientEvalObjective(TS ts, PetscReal time, Vec state, Vec design, PetscReal *val)
{
  PetscErrorCode ierr;
  ObjectiveLink  link = ts->funchead;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,time,2);
  PetscValidHeaderSpecific(state,VEC_CLASSID,3);
  PetscValidHeaderSpecific(design,VEC_CLASSID,4);
  PetscValidPointer(val,5);
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

/* Evaluates objective functions of the type f(U,param,t = fixed)
   Since we just accumulate values to the objective function, we don't need an Event to detect the exact time.
   We test in the interval (ptime,time] */
static PetscErrorCode TSGradientEvalObjectiveFixed(TS ts, PetscReal ptime, PetscReal time, Vec state, Vec design, PetscReal *val)
{
  PetscErrorCode ierr;
  ObjectiveLink  link = ts->funchead;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,ptime,2);
  PetscValidLogicalCollectiveReal(ts,time,3);
  PetscValidHeaderSpecific(state,VEC_CLASSID,4);
  PetscValidHeaderSpecific(design,VEC_CLASSID,5);
  PetscValidPointer(val,6);
  ierr = VecLockPush(state);CHKERRQ(ierr);
  ierr = VecLockPush(design);CHKERRQ(ierr);
  *val = 0.0;
  while (link) {
    PetscReal v = 0.0;
    if (link->f && ptime < link->fixedtime && link->fixedtime <= time) {
      ierr = (*link->f)(ts,link->fixedtime,state,design,&v,link->f_ctx);CHKERRQ(ierr);
    }
    *val += v;
    link = link->next;
  }
  ierr = VecLockPop(state);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Evaluates derivative (wrt the state) of objective functions of the type f(U,param,t) */
static PetscErrorCode TSGradientEvalObjectiveGradientU(TS ts, PetscReal time, Vec state, Vec design, Vec work, Vec out)
{
  PetscErrorCode ierr;
  ObjectiveLink  link = ts->funchead;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,time,2);
  PetscValidHeaderSpecific(state,VEC_CLASSID,3);
  PetscValidHeaderSpecific(design,VEC_CLASSID,4);
  PetscValidHeaderSpecific(work,VEC_CLASSID,5);
  PetscValidHeaderSpecific(out,VEC_CLASSID,6);
  ierr = VecLockPush(state);CHKERRQ(ierr);
  ierr = VecLockPush(design);CHKERRQ(ierr);
  ierr = VecSet(out,0.0);CHKERRQ(ierr);
  while (link) {
    if (link->f_x && link->fixedtime <= PETSC_MIN_REAL) {
      ierr = (*link->f_x)(ts,time,state,design,work,link->f_x_ctx);CHKERRQ(ierr);
      ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
    }
    link = link->next;
  }
  ierr = VecLockPop(state);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Evaluates derivative (wrt the state) of objective functions of the type f(U,param,t = fixed)
   These may lead to Dirac's delta terms in the adjoint ODE if the fixed time is in between (t0,tf) */
static PetscErrorCode TSGradientEvalObjectiveGradientUFixed(TS ts, PetscReal time, Vec state, Vec design, Vec work, Vec out)
{
  PetscErrorCode ierr;
  ObjectiveLink  link = ts->funchead;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,time,2);
  PetscValidHeaderSpecific(state,VEC_CLASSID,3);
  PetscValidHeaderSpecific(design,VEC_CLASSID,4);
  PetscValidHeaderSpecific(work,VEC_CLASSID,5);
  PetscValidHeaderSpecific(out,VEC_CLASSID,6);
  ierr = VecLockPush(state);CHKERRQ(ierr);
  ierr = VecLockPush(design);CHKERRQ(ierr);
  ierr = VecSet(out,0.0);CHKERRQ(ierr);
  while (link) {
    if (link->f_x && link->fixedtime > PETSC_MIN_REAL && PetscAbsReal(link->fixedtime-time) < PETSC_SMALL) {
      ierr = (*link->f_x)(ts,link->fixedtime,state,design,work,link->f_x_ctx);CHKERRQ(ierr);
      ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
    }
    link = link->next;
  }
  ierr = VecLockPop(state);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Evaluates derivative (wrt the parameters) of objective functions of the type f(U,param,t) */
static PetscErrorCode TSGradientEvalObjectiveGradientM(TS ts, PetscReal time, Vec state, Vec design, Vec work, Vec out)
{
  PetscErrorCode ierr;
  ObjectiveLink  link = ts->funchead;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,time,2);
  PetscValidHeaderSpecific(state,VEC_CLASSID,3);
  PetscValidHeaderSpecific(design,VEC_CLASSID,4);
  PetscValidHeaderSpecific(work,VEC_CLASSID,5);
  PetscValidHeaderSpecific(out,VEC_CLASSID,6);
  ierr = VecLockPush(state);CHKERRQ(ierr);
  ierr = VecLockPush(design);CHKERRQ(ierr);
  ierr = VecSet(out,0.0);CHKERRQ(ierr);
  while (link) {
    if (link->f_m && link->fixedtime <= PETSC_MIN_REAL) {
      ierr = (*link->f_m)(ts,link->fixedtime,state,design,work,link->f_m_ctx);CHKERRQ(ierr);
      ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
    }
    link = link->next;
  }
  ierr = VecLockPop(state);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Evaluates derivative (wrt the parameters) of objective functions of the type f(U,param,t = tfixed)
   Regularizers fall into this category. They don't contribute to the adjoint ODE, only to the gradient
   They are evaluated in TSGradientPostStep
*/
static PetscErrorCode TSGradientEvalObjectiveGradientMFixed(TS ts, PetscReal ptime, PetscReal time, Vec state, Vec design, Vec work, Vec out)
{
  PetscErrorCode ierr;
  ObjectiveLink  link = ts->funchead;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,time,2);
  PetscValidHeaderSpecific(state,VEC_CLASSID,3);
  PetscValidHeaderSpecific(design,VEC_CLASSID,4);
  PetscValidHeaderSpecific(work,VEC_CLASSID,5);
  PetscValidHeaderSpecific(out,VEC_CLASSID,6);
  ierr = VecLockPush(state);CHKERRQ(ierr);
  ierr = VecLockPush(design);CHKERRQ(ierr);
  ierr = VecSet(out,0.0);CHKERRQ(ierr);
  while (link) {
    if (link->f_m && ptime < link->fixedtime && link->fixedtime <= time) {
      ierr = (*link->f_m)(ts,link->fixedtime,state,design,work,link->f_m_ctx);CHKERRQ(ierr);
      ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
    }
    link = link->next;
  }
  ierr = VecLockPop(state);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Apply "gradients" of initial conditions
   if transpose is true : y = G_x^-1 G_m x
   if transpose is false: y = G_m^t G_x^-T x
*/
static PetscErrorCode TSGradientICApply(TS ts, Vec x, Vec y, PetscBool transpose)
{
  PetscErrorCode ierr;
  KSP            ksp = NULL;
  Vec            workvec = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscValidLogicalCollectiveBool(ts,transpose,4);
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  if (!ts->G_m) PetscFunctionReturn(0);
  if (ts->G_x) { /* this is optional. If not provided, identity is assumed */
    ierr = PetscObjectQuery((PetscObject)ts,"_ts_gradient_G",(PetscObject*)&ksp);
    ierr = PetscObjectQuery((PetscObject)ts,"_ts_gradient_GW",(PetscObject*)&workvec);
    if (!ksp) {
      const char *prefix;
      ierr = KSPCreate(PetscObjectComm((PetscObject)ts),&ksp);CHKERRQ(ierr);
      ierr = TSGetOptionsPrefix(ts,&prefix);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(ksp,"G_");CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ksp,prefix);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradient_G",(PetscObject)ksp);
    }
    if (!workvec) {
      ierr = MatCreateVecs(ts->G_m,NULL,&workvec);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradient_GW",(PetscObject)workvec);
    }
    ierr = KSPSetOperators(ksp,ts->G_x,ts->G_x);CHKERRQ(ierr);
  }
  if (transpose) {
    if (ksp) {
      ierr = KSPSolveTranspose(ksp,x,workvec);CHKERRQ(ierr);
      ierr = MatMultTranspose(ts->G_m,workvec,y);CHKERRQ(ierr);
    } else {
      ierr = MatMultTranspose(ts->G_m,x,y);CHKERRQ(ierr);
    }
  } else {
    ierr = MatMult(ts->G_m,x,y);CHKERRQ(ierr);
    if (ksp) {
      ierr = KSPSolve(ksp,y,y);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
/* ------------------ Helper routines to update history vectors and compute split Jacobians, namespaced with TS ----------------------- */

/* Updates history vectors U and Udot, if present */
static PetscErrorCode TSTrajectoryUpdateHistoryVecs(TSTrajectory tj, TS ts, PetscReal time, Vec U, Vec Udot)
{
  PetscInt       step = PETSC_MIN_INT; /* inquire TSTrajectoryGetVecs by the time argument */
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  if (ts) PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  PetscValidLogicalCollectiveReal(tj,time,3);
  if (U) PetscValidHeaderSpecific(U,VEC_CLASSID,4);
  if (Udot) PetscValidHeaderSpecific(Udot,VEC_CLASSID,5);
  if (U)    { ierr = VecLockPop(U);CHKERRQ(ierr); }
  if (Udot) { ierr = VecLockPop(Udot);CHKERRQ(ierr); }
  ierr = TSTrajectoryGetVecs(ts->trajectory,ts,step,&time,U,Udot);CHKERRQ(ierr);
  if (U)    { ierr = VecLockPush(U);CHKERRQ(ierr); }
  if (Udot) { ierr = VecLockPush(Udot);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/* struct to hold split jacobians, shifts and split flag */
typedef struct {
  PetscScalar      shift;
  PetscObjectState Astate;
  PetscObjectId    Aid;
  PetscBool        splitdone;
  PetscBool        timeindep; /* true if the ODE is A Udot + B U -f = 0 */
  Mat              J_U;       /* Jacobian : F_U (U,Udot,t) */
  Mat              J_Udot;    /* Jacobian : F_Udot(U,Udot,t) */
  Mat              J_dtUdot;  /* Jacobian : d/dt F_Udot(U,Udot,t) */
} SplitJac;

static PetscErrorCode SplitJacDestroy_Private(void *ptr)
{
  SplitJac*      s = (SplitJac*)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&s->J_U);CHKERRQ(ierr);
  ierr = MatDestroy(&s->J_Udot);CHKERRQ(ierr);
  ierr = MatDestroy(&s->J_dtUdot);CHKERRQ(ierr);
  ierr = PetscFree(s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* This is an helper routine to get F_U and F_Udot. Can be generalized with some public API with callbacks.
   TODO: possibly also d/dt F_Udot, not yet coded */
static PetscErrorCode TSComputeSplitJacobians(TS ts, PetscReal time, Vec U, Vec Udot, Mat A, Mat pA, Mat B, Mat pB, Mat C, Mat pC)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,time,2);
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

/* Just access the split matrices */
static PetscErrorCode TSGetSplitJacobians(TS ts, Mat* JU, Mat *JUdot)
{
  PetscErrorCode ierr;
  PetscContainer c;
  SplitJac       *splitJ;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (JU) PetscValidPointer(JU,2);
  if (JUdot) PetscValidPointer(JUdot,3);
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_splitJac",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Missing splitJac container");
  ierr = PetscContainerGetPointer(c,(void**)&splitJ);CHKERRQ(ierr);
  if (JU) *JU = splitJ->J_U;
  if (JUdot) *JUdot = splitJ->J_Udot;
  PetscFunctionReturn(0);
}

/* Updates F_Udot (splitJ->J_Udot) and F_U + d/dt F_Udot (splitJ->J_U) at a given time */
static PetscErrorCode TSUpdateSplitJacobiansFromHistory(TS ts, PetscReal time, Vec U, Vec Udot)
{
  PetscContainer c;
  SplitJac       *splitJ;
  TSProblemType  type;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_splitJac",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Missing splitJac container");
  ierr = PetscContainerGetPointer(c,(void**)&splitJ);CHKERRQ(ierr);
  if (splitJ->timeindep && splitJ->splitdone) PetscFunctionReturn(0);
  ierr = TSGetProblemType(ts,&type);CHKERRQ(ierr);
  if (type > TS_LINEAR) {
    ierr = TSTrajectoryUpdateHistoryVecs(ts->trajectory,ts,time,U,Udot);CHKERRQ(ierr);
  }
  if (!splitJ->J_U) {
    Mat A;

    ierr = TSGetIJacobian(ts,&A,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&splitJ->J_U);CHKERRQ(ierr);
  }
  if (!splitJ->J_Udot) {
    Mat A;

    ierr = TSGetIJacobian(ts,&A,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&splitJ->J_Udot);CHKERRQ(ierr);
  }
  /* ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&adj->splitJ->dtUdot);CHKERRQ(ierr); */
  ierr = TSComputeSplitJacobians(ts,time,U,Udot,
                                 splitJ->J_U,     splitJ->J_U,
                                 splitJ->J_Udot,  splitJ->J_Udot,
                                 splitJ->J_dtUdot,splitJ->J_dtUdot);CHKERRQ(ierr);
  if (splitJ->J_dtUdot) {
    ierr = MatAXPY(splitJ->J_U,1.0,splitJ->J_dtUdot,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatZeroEntries(splitJ->J_dtUdot);CHKERRQ(ierr);
  }
  splitJ->splitdone = splitJ->timeindep ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* The assumption here is that the IJacobian routine is called after the IFunction (called with same time, U and Udot)
   This function is used in AdjointTSIFunctionLinear and TLMTSIFunctionLinear, that update the splits via TSUpdateSplitJacobiansFromHistory.
   This is why the time, U and Udot arguments are ignored */
static PetscErrorCode TSComputeIJacobianWithSplits(TS ts, PetscReal time, Vec U, Vec Udot, PetscReal shift, Mat A, Mat B, void *ctx)
{
  PetscObjectState Astate;
  PetscObjectId    Aid;
  PetscContainer   c;
  SplitJac         *splitJ;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_splitJac",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Missing splitJac container");
  ierr = PetscContainerGetPointer(c,(void**)&splitJ);CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject)A,&Astate);CHKERRQ(ierr);
  ierr = PetscObjectGetId((PetscObject)A,&Aid);CHKERRQ(ierr);
  if (splitJ->timeindep && PetscAbsScalar(splitJ->shift - shift) < PETSC_SMALL &&
      splitJ->Astate == Astate && splitJ->Aid == Aid) {
    PetscFunctionReturn(0);
  }
  ierr = MatCopy(splitJ->J_Udot,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatScale(A,shift);CHKERRQ(ierr);
  ierr = MatAXPY(A,1.0,splitJ->J_U,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject)A,&splitJ->Astate);CHKERRQ(ierr);
  ierr = PetscObjectGetId((PetscObject)A,&splitJ->Aid);CHKERRQ(ierr);
  splitJ->shift = shift;
  if (B && A != B) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"B != A not yet implemented");
  PetscFunctionReturn(0);
}

/* ------------------ Routines for adjoints of ODE, namespaced with AdjointTS ----------------------- */

typedef struct {
  PetscBool optimization;  /* whether we need an adjoint for PDE constrained optimization or a standard ODE adjoint */
  Vec       design;        /* design vector (fixed) */
  TS        fwdts;         /* forward solver */
  PetscReal t0,tf;         /* time limits, for forward time recovery */
  Vec       *W;            /* work vectors W[0] and W[1] always store U and Udot at a given time */
  PetscBool firststep;     /* used for trapz rule in PDE-constrained support */
  Vec       gradient;      /* gradient we are evaluating (PDE-constrained case only) */
  Vec       *wgrad;        /* gradient work vectors (for trapz rule, PDE-constrained case only) */
  PetscBool dirac_delta;   /* If true, means that a delta contribution needs to be added to lambda during the post step method (PDE-constrained case only) */
} AdjointCtx;

static PetscErrorCode AdjointTSDestroy_Private(void *ptr)
{
  AdjointCtx*    adj = (AdjointCtx*)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&adj->design);CHKERRQ(ierr);
  ierr = VecDestroyVecs(4,&adj->W);CHKERRQ(ierr);
  ierr = VecDestroy(&adj->gradient);CHKERRQ(ierr);
  ierr = VecDestroyVecs(2,&adj->wgrad);CHKERRQ(ierr);
  ierr = TSDestroy(&adj->fwdts);CHKERRQ(ierr);
  ierr = PetscFree(adj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSRHSJacobian(TS adjts, PetscReal time, Vec U, Mat A, Mat P, void *ctx)
{
  AdjointCtx     *adj_ctx;
  PetscReal      ft;
  TSProblemType  type;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  ierr = TSGetProblemType(adj_ctx->fwdts,&type);CHKERRQ(ierr);
  ft   = adj_ctx->tf - time + adj_ctx->t0;
  if (type > TS_LINEAR) {
    ierr = TSTrajectoryUpdateHistoryVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,ft,adj_ctx->W[0],NULL);CHKERRQ(ierr);
  }
  ierr = TSComputeRHSJacobian(adj_ctx->fwdts,ft,adj_ctx->W[0],A,P);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* The adjoint formulation I'm using assumes H(U,Udot,t) = 0
   -> the forward ODE is Udot - G(U) = 0 ( -> H(U,Udot,t) := Udot - G(U) )
   -> the adjoint ODE is F - L^T * G_U - Ldot^T in backward time (F is present just in optimization)
   -> the adjoint ODE is Ldot^T = L^T * G_U - F in forward time */
static PetscErrorCode AdjointTSRHSFuncLinear(TS adjts, PetscReal time, Vec U, Vec F, void *ctx)
{
  AdjointCtx     *adj_ctx;
  PetscReal      fwdt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - time + adj_ctx->t0;
  if (adj_ctx->optimization) {
    ierr = TSTrajectoryUpdateHistoryVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,fwdt,adj_ctx->W[0],NULL);CHKERRQ(ierr);
    ierr = TSGradientEvalObjectiveGradientU(adj_ctx->fwdts,fwdt,adj_ctx->W[0],adj_ctx->design,adj_ctx->W[3],F);CHKERRQ(ierr);
    ierr = VecScale(F,-1.0);CHKERRQ(ierr);
    ierr = TSComputeRHSJacobian(adjts,time,U,adjts->Arhs,NULL);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(adjts->Arhs,U,F,F);CHKERRQ(ierr);
  } else {
    ierr = TSComputeRHSJacobian(adjts,time,U,adjts->Arhs,NULL);CHKERRQ(ierr);
    ierr = MatMultTranspose(adjts->Arhs,U,F);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Given the forward ODE : H(U,Udot,t) = 0
   -> the adjoint ODE is : F - L^T * (H_U - d/dt H_Udot) - Ldot^T H_Udot = 0 (in backward time) (again, F is null for standard ODE adjoints)
   -> the adjoint ODE is : Ldot^T H_Udot + L^T * (H_U + d/dt H_Udot) + F = 0 (in forward time) */
static PetscErrorCode AdjointTSIFunctionLinear(TS adjts, PetscReal time, Vec U, Vec Udot, Vec F, void *ctx)
{
  AdjointCtx     *adj_ctx;
  Mat            J_U, J_Udot;
  PetscReal      fwdt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - time + adj_ctx->t0;
  if (adj_ctx->optimization) {
    ierr = TSTrajectoryUpdateHistoryVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,fwdt,adj_ctx->W[0],NULL);CHKERRQ(ierr);
    ierr = TSGradientEvalObjectiveGradientU(adj_ctx->fwdts,fwdt,adj_ctx->W[0],adj_ctx->design,adj_ctx->W[3],F);CHKERRQ(ierr);
    ierr = TSUpdateSplitJacobiansFromHistory(adj_ctx->fwdts,fwdt,adj_ctx->W[0],adj_ctx->W[1]);CHKERRQ(ierr);
    ierr = TSGetSplitJacobians(adj_ctx->fwdts,&J_U,&J_Udot);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(J_U,U,F,F);CHKERRQ(ierr);
  } else {
    ierr = TSTrajectoryUpdateHistoryVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,fwdt,adj_ctx->W[0],adj_ctx->W[1]);CHKERRQ(ierr);
    ierr = TSGetSplitJacobians(adj_ctx->fwdts,&J_U,&J_Udot);CHKERRQ(ierr);
    ierr = MatMultTranspose(J_U,U,F);CHKERRQ(ierr);
  }
  ierr = MatMultTransposeAdd(J_Udot,Udot,F,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSIJacobian(TS adjts, PetscReal time, Vec U, Vec Udot, PetscReal shift, Mat A, Mat B, void *ctx)
{
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  ierr = TSComputeIJacobianWithSplits(adj_ctx->fwdts,time,U,Udot,shift,A,B,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Handles the detection of Dirac's delta forcing terms (i.e. f_U(U,param,t = fixed)) in the adjoint equations */
static PetscErrorCode AdjointTSEventFunction(TS adjts, PetscReal t, Vec U, PetscScalar fvalue[], void *ctx)
{
  AdjointCtx     *adj_ctx;
  ObjectiveLink  link;
  TS             ts;
  PetscInt       cnt = 0;
  PetscReal      fwdt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - t + adj_ctx->t0;
  ts   = adj_ctx->fwdts;
  link = ts->funchead;
  while (link) { fvalue[cnt++] = (link->f_x && link->fixedtime > PETSC_MIN_REAL) ?  link->fixedtime - fwdt : 1.0; link = link->next; }
  PetscFunctionReturn(0);
}

/* Dirac's delta integration H_Udot^T ( L(+) - L(-) )  = - f_U -> L(+) = - H_Udot^-T f_U + L(-)
   We store the increment - H_Udot^-T f_U in adj_ctx->W[2] and apply it during the AdjointTSPostStep */
static PetscErrorCode AdjointTSPostEvent(TS adjts, PetscInt nevents, PetscInt event_list[], PetscReal t, Vec U, PetscBool forwardsolve, void* ctx)
{
  AdjointCtx     *adj_ctx;
  PetscReal      fwdt;
  TSIJacobian    ijac;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - t + adj_ctx->t0;
  ierr = TSTrajectoryUpdateHistoryVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,fwdt,adj_ctx->W[0],NULL);CHKERRQ(ierr);
  ierr = VecLockPop(adj_ctx->W[2]);CHKERRQ(ierr);
  ierr = TSGradientEvalObjectiveGradientUFixed(adj_ctx->fwdts,fwdt,adj_ctx->W[0],adj_ctx->design,adj_ctx->W[3],adj_ctx->W[2]);CHKERRQ(ierr);
  ierr = VecScale(adj_ctx->W[2],-1.0);CHKERRQ(ierr);
  ierr = TSGetIJacobian(adjts,NULL,NULL,&ijac,NULL);CHKERRQ(ierr);
  if (ijac) {
    SNES snes;
    KSP  ksp;
    Mat  J_Udot;

    ierr = TSTrajectoryUpdateHistoryVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,fwdt,adj_ctx->W[0],adj_ctx->W[1]);CHKERRQ(ierr);
    ierr = TSGetSplitJacobians(adj_ctx->fwdts,NULL,&J_Udot);CHKERRQ(ierr);
    ierr = TSGetSNES(adjts,&snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,J_Udot,J_Udot);CHKERRQ(ierr);
    ierr = KSPSolveTranspose(ksp,adj_ctx->W[2],adj_ctx->W[2]);CHKERRQ(ierr);
  }
  ierr = VecLockPush(adj_ctx->W[2]);CHKERRQ(ierr);
  adj_ctx->dirac_delta = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* Partial integration (via the trapezoidal rule) of the gradient terms f_M + L^T H_M */
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
  ierr = TSGetPrevTime(adjts,&ptime);CHKERRQ(ierr);
  dt   = time - ptime; /* current time step used */

  /* first step:
       wgrad[0] has been initialized with the first backward evaluation of L^T H_M at t0
       gradient with the forward contribution to the gradient (i.e. \int f_m)
       Note that f_m is not a regularizer: these are added to the gradient during the forward step
  */
  if (adj_ctx->firststep) {
    ierr = VecAXPY(adj_ctx->gradient,dt/2.0,adj_ctx->wgrad[0]);CHKERRQ(ierr);
    ierr = VecSet(adj_ctx->wgrad[1],0.0);CHKERRQ(ierr);
  }
  if (adj_ctx->fwdts->F_m) {
    PetscScalar tt[2];

    TS ts = adj_ctx->fwdts;
    if (ts->F_m_f) { /* non constant dependence */
      ierr = TSTrajectoryUpdateHistoryVecs(ts->trajectory,ts,fwdt,adj_ctx->W[0],adj_ctx->W[1]);CHKERRQ(ierr);
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

  /* We detected Dirac's delta terms -> add the increment here
     Re-evaluate L^T H_M and restart trapezoidal rule if needed */
  if (adj_ctx->dirac_delta) {
    TS ts = adj_ctx->fwdts;

    ierr = VecAXPY(lambda,1.0,adj_ctx->W[2]);CHKERRQ(ierr);
    if (ts->F_m) {
      ierr = VecSet(adj_ctx->wgrad[0],0.0);CHKERRQ(ierr);
      ierr = MatMultTranspose(ts->F_m,lambda,adj_ctx->wgrad[0]);CHKERRQ(ierr);
      adj_ctx->firststep = PETSC_TRUE;
    }
  }
  adj_ctx->dirac_delta = PETSC_FALSE;

  if (adjts->reason) { adj_ctx->tf = time; } /* prevent from accumulation errors XXX */
  PetscFunctionReturn(0);
}

/* Creates the adjoint TS */
static PetscErrorCode TSCreateAdjointTS(TS ts, TS* adjts, PetscBool optimization)
{
  SNES            snes;
  Mat             A,B;
  Vec             lambda,vatol,vrtol;
  PetscContainer  container;
  AdjointCtx      *adj;
  TSIFunction     ifunc;
  TSRHSFunction   rhsfunc;
  TSRHSJacobian   rhsjacfunc;
  TSI2Function    i2func;
  TSType          type;
  TSEquationType  eqtype;
  const char      *prefix;
  PetscReal       atol,rtol;
  SplitJac        *splitJ;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = TSGetEquationType(ts,&eqtype);CHKERRQ(ierr);
  if (eqtype != TS_EQ_UNSPECIFIED && eqtype != TS_EQ_EXPLICIT && eqtype != TS_EQ_ODE_EXPLICIT &&
      eqtype != TS_EQ_IMPLICIT && eqtype != TS_EQ_ODE_IMPLICIT)
      SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSEquationType %D\n",eqtype);
  ierr = TSGetI2Function(ts,NULL,&i2func,NULL);CHKERRQ(ierr);
  if (i2func) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Second order ODEs are not supported");
  ierr = TSCreate(PetscObjectComm((PetscObject)ts),adjts);CHKERRQ(ierr);
  ierr = TSGetType(ts,&type);CHKERRQ(ierr);
  ierr = TSSetType(*adjts,type);CHKERRQ(ierr);
  ierr = TSGetTolerances(ts,&atol,&vatol,&rtol,&vrtol);CHKERRQ(ierr);
  ierr = TSSetTolerances(*adjts,atol,vatol,rtol,vrtol);CHKERRQ(ierr);

  /* application context */
  ierr = PetscNew(&adj);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(*adjts,(void *)adj);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&lambda);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(lambda,4,&adj->W);CHKERRQ(ierr);
  adj->optimization = optimization;

  /* these three vectors are locked:
     - only TSUpdateHistoryVecs can modify W[0] and W[1]
       W[0] stores the updated forward state,  W[1] stores the updated derivative of the state vector (interpolated)
       If you need to update them, call TSUpdateHistoryVecs, a caching mechanism in
       TSTrajectoryGetVecs will prevent to reload/reinterpolate
     - only AdjointTSPostEvent can modify W[2], used to store delta contributions
  */
  ierr = VecLockPush(adj->W[0]);CHKERRQ(ierr);
  ierr = VecLockPush(adj->W[1]);CHKERRQ(ierr);
  ierr = VecLockPush(adj->W[2]);CHKERRQ(ierr);

  ierr = PetscObjectReference((PetscObject)ts);CHKERRQ(ierr);
  adj->fwdts = ts;
  adj->t0 = adj->tf = PETSC_MAX_REAL;

  /* caching to prevent from recomputation of Jacobians */
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_splitJac",(PetscObject*)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerGetPointer(container,(void**)&splitJ);CHKERRQ(ierr);
  } else {
    ierr = PetscNew(&splitJ);CHKERRQ(ierr);
    splitJ->Astate    = -1;
    splitJ->Aid       = PETSC_MIN_INT;
    splitJ->shift     = PETSC_MIN_REAL;
    splitJ->splitdone = PETSC_FALSE;
    ierr = PetscContainerCreate(PetscObjectComm((PetscObject)ts),&container);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container,splitJ);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container,SplitJacDestroy_Private);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)ts,"_ts_splitJac",(PetscObject)container);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  }

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
    ierr = TSSetIFunction(*adjts,NULL,AdjointTSIFunctionLinear,NULL);CHKERRQ(ierr);
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
  ierr = TSSetOptionsPrefix(*adjts,"adjoint_");CHKERRQ(ierr);
  ierr = TSAppendOptionsPrefix(*adjts,prefix);CHKERRQ(ierr);

  /* preliminary support for time-independent adjoints */
  ierr = TSGetOptionsPrefix(*adjts,&prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,prefix,"-time_independent",&splitJ->timeindep,NULL);CHKERRQ(ierr);
  splitJ->splitdone = PETSC_FALSE;

  /* adjoint ODE is linear */
  ierr = TSSetProblemType(*adjts,TS_LINEAR);CHKERRQ(ierr);

  /* use KSPSolveTranspose to solve the adjoint */
  ierr = TSGetSNES(*adjts,&snes);CHKERRQ(ierr);
  ierr = SNESKSPONLYSetUseTransposeSolve(snes,PETSC_TRUE);CHKERRQ(ierr);

  /* set special purpose post step method for incremental gradient evaluation */
  if (adj->optimization) {
    ierr = TSSetPostStep(*adjts,AdjointTSPostStep);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSSetInitialGradient(TS adjts, Vec gradient)
{
  Vec            lambda,fwdsol;
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
  if (!adj_ctx->optimization) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_USER,"Cannot set initial gradient with a standard ODE adjoint. Use TSCreateAdjointTS(ts,&adj,PETSC_TRUE)");
  if (adj_ctx->t0 >= PETSC_MAX_REAL || adj_ctx->tf >= PETSC_MAX_REAL) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_ORDER,"You should call AdjointTSSetTimeLimits first");
  if (!adj_ctx->design) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_ORDER,"You should call AdjointTSSetDesign first");

  adj_ctx->firststep = PETSC_TRUE;
  ierr = PetscObjectReference((PetscObject)gradient);CHKERRQ(ierr);
  ierr = VecDestroy(&adj_ctx->gradient);CHKERRQ(ierr);
  adj_ctx->gradient = gradient;
  if (!adj_ctx->wgrad) {
    ierr = VecDuplicateVecs(gradient,2,&adj_ctx->wgrad);CHKERRQ(ierr);
  }
  ierr = VecSet(adj_ctx->wgrad[0],0.0);CHKERRQ(ierr);

  /* Set initial conditions for the adjoint ode */
  ierr = TSGetSolution(adj_ctx->fwdts,&fwdsol);CHKERRQ(ierr);
  ierr = TSGetSolution(adjts,&lambda);CHKERRQ(ierr);
  ierr = TSGradientEvalObjectiveGradientUFixed(adj_ctx->fwdts,adj_ctx->tf,fwdsol,adj_ctx->design,adj_ctx->W[3],lambda);CHKERRQ(ierr);
  ierr = VecNorm(lambda,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > PETSC_SMALL) {
    TSIJacobian ijac;

    /* Dirac's delta initial condition */
    ierr = TSGetIJacobian(adjts,NULL,NULL,&ijac,NULL);CHKERRQ(ierr);
    if (ijac) { /* lambda(T) = - (F_Udot)^T D_x, D_x the gradients of the functionals that sample the solution at the final time */
      SNES snes;
      KSP  ksp;
      Mat  J_Udot;

      ierr = TSUpdateSplitJacobiansFromHistory(adj_ctx->fwdts,adj_ctx->tf,adj_ctx->W[0],adj_ctx->W[1]);CHKERRQ(ierr);
      ierr = TSGetSNES(adjts,&snes);CHKERRQ(ierr);
      ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
      ierr = TSGetSplitJacobians(adj_ctx->fwdts,NULL,&J_Udot);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,J_Udot,J_Udot);CHKERRQ(ierr);
      ierr = KSPSolveTranspose(ksp,lambda,lambda);CHKERRQ(ierr);
    }
    ierr = VecScale(lambda,-1.0);CHKERRQ(ierr);
  }

  /* initialize wgrad[0] */
  if (adj_ctx->fwdts->F_m) {
    TS ts = adj_ctx->fwdts;
    if (ts->F_m_f) { /* non constant dependence */
      ierr = TSTrajectoryUpdateHistoryVecs(ts->trajectory,ts,adj_ctx->tf,adj_ctx->W[0],adj_ctx->W[1]);CHKERRQ(ierr);
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

static PetscErrorCode AdjointTSSetTimeLimits(TS adjts, PetscReal t0, PetscReal tf)
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
  ierr = TSSetMaxSteps(adjts,PETSC_MAX_INT);CHKERRQ(ierr);
  ierr = TSSetMaxTime(adjts,tf);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(adjts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  /* update time limits in the application context
     they are needed to recover the forward time from the backward */
  adj_ctx->tf = tf;
  adj_ctx->t0 = t0;
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSEventHandler(TS adjts)
{
  PetscContainer c;
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;
  ObjectiveLink  link;
  PetscInt       cnt = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_gradient_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Missing adjoint container");
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);

  /* set event handler for Dirac's delta terms */
  link = adj_ctx->fwdts->funchead;
  while (link) { cnt++; link = link->next; }
  if (cnt) {
    PetscInt  *dir;
    PetscBool *term;

    ierr = PetscCalloc2(cnt,&dir,cnt,&term);CHKERRQ(ierr);
    ierr = TSSetEventHandler(adjts,cnt,dir,term,AdjointTSEventFunction,AdjointTSPostEvent,NULL);CHKERRQ(ierr);
    ierr = PetscFree2(dir,term);CHKERRQ(ierr);
  }
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
  if (!adj_ctx->optimization) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_USER,"Cannot compute final gradient with a standard ODE adjoint. Use TSCreateAdjointTS(ts,&adj,PETSC_TRUE)");
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
      ierr = VecCopy(lambda,adj_ctx->W[3]);CHKERRQ(ierr);
    } else {
      Mat J_Udot;
      ierr = TSGetSplitJacobians(adj_ctx->fwdts,NULL,&J_Udot);CHKERRQ(ierr);
      ierr = MatMultTranspose(J_Udot,lambda,adj_ctx->W[3]);CHKERRQ(ierr);
    }
    ierr = TSGradientICApply(fwdts,adj_ctx->W[3],adj_ctx->wgrad[0],PETSC_TRUE);CHKERRQ(ierr);
    ierr = VecAXPY(adj_ctx->gradient,1.0,adj_ctx->wgrad[0]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------ Helper routines specific to PDE-constrained support, namespaced with TSGradient ----------------------- */

typedef struct {
  PetscErrorCode (*user)(TS); /* user post step method */
  Vec            design;      /* the design vector we are evaluating against */
  PetscBool      objeval;     /* indicates we have to evalute the objective functions */
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
  if (ts->reason == TS_CONVERGED_TIME) {
    ierr = TSGetMaxTime(ts,&time);CHKERRQ(ierr);
  }
  if (poststep_ctx->objeval) {
    ierr = TSGradientEvalObjective(ts,time,solution,poststep_ctx->design,&val);CHKERRQ(ierr);
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
    ierr = TSGradientEvalObjectiveFixed(ts,ptime,time,solution,poststep_ctx->design,&val);CHKERRQ(ierr);
    poststep_ctx->obj += val;
  }
  if (poststep_ctx->gradient) {
    PetscScalar tt[3];

    if (poststep_ctx->firststep) {
      ierr = VecSet(poststep_ctx->wgrad[2],0.0);CHKERRQ(ierr);
      ierr = VecScale(poststep_ctx->gradient,dt/2.0);CHKERRQ(ierr);
    }
    ierr = TSGradientEvalObjectiveGradientM(ts,time,solution,poststep_ctx->design,poststep_ctx->wgrad[0],poststep_ctx->wgrad[1]);CHKERRQ(ierr);

    /* Regularizers */
    ierr = TSGradientEvalObjectiveGradientMFixed(ts,ptime,time,solution,poststep_ctx->design,poststep_ctx->wgrad[3],poststep_ctx->wgrad[0]);CHKERRQ(ierr);

    tt[0] = 1.0;    /* regularizer */
    tt[1] = dt/2.0; /* trapz */
    tt[2] = dt/2.0; /* trapz */
    ierr = VecMAXPY(poststep_ctx->gradient,3,tt,poststep_ctx->wgrad);CHKERRQ(ierr);

    /* XXX this could be done more efficiently */
    ierr = VecCopy(poststep_ctx->wgrad[1],poststep_ctx->wgrad[2]);CHKERRQ(ierr);
  }
  poststep_ctx->firststep = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEvaluateObjective_Private(TS ts, Vec X, Vec design, Vec gradient, PetscReal *val)
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
  if (!ts->funchead) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Missing objective functions");
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
    ierr = TSGradientEvalObjective(ts,t0,X,design,&poststep_ctx.obj);CHKERRQ(ierr);
  }
  if (poststep_ctx.gradient) {
    ierr = VecDuplicateVecs(poststep_ctx.gradient,4,&poststep_ctx.wgrad);CHKERRQ(ierr);
    ierr = TSGradientEvalObjectiveGradientM(ts,t0,X,design,poststep_ctx.wgrad[0],poststep_ctx.gradient);CHKERRQ(ierr);
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
  ierr = VecDestroyVecs(4,&poststep_ctx.wgrad);CHKERRQ(ierr);

  /* get back value */
  if (val) *val = poststep_ctx.obj;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEvaluateObjectiveGradient_Private(TS ts, Vec X, Vec design, Vec gradient, PetscReal *val)
{
  TS             adjts;
  Vec            lambda;
  TSTrajectory   otrj;
  PetscBool      isbasic;
  PetscReal      t0,tf,dt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  otrj = ts->trajectory;
  ierr = TSTrajectoryCreate(PetscObjectComm((PetscObject)ts),&ts->trajectory);CHKERRQ(ierr);
  ierr = TSTrajectorySetType(ts->trajectory,ts,TSTRAJECTORYBASIC);CHKERRQ(ierr);
  ierr = TSTrajectorySetFromOptions(ts->trajectory,ts);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)ts->trajectory,TSTRAJECTORYBASIC,&isbasic);CHKERRQ(ierr);
  if (!isbasic) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSTrajectory type %s",((PetscObject)ts->trajectory)->type_name);

  /* sample initial condition dependency */
  ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
  if (ts->Ggrad) {
    ierr = (*ts->Ggrad)(ts,t0,X,design,ts->G_x,ts->G_m,ts->Ggrad_ctx);CHKERRQ(ierr);
  }

  /* forward solve */
  ierr = TSEvaluateObjective_Private(ts,X,design,gradient,val);CHKERRQ(ierr);

  /* adjoint */
  ierr = TSCreateAdjointTS(ts,&adjts,PETSC_TRUE);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&lambda);CHKERRQ(ierr);
  ierr = TSSetSolution(adjts,lambda);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&tf);CHKERRQ(ierr);
  ierr = TSGetPrevTime(ts,&dt);CHKERRQ(ierr);
  dt   = tf - dt;
  ierr = TSSetTimeStep(adjts,PetscMin(dt,tf-t0));CHKERRQ(ierr);
  ierr = AdjointTSSetTimeLimits(adjts,t0,tf);CHKERRQ(ierr);
  ierr = AdjointTSSetDesign(adjts,design);CHKERRQ(ierr);
  ierr = AdjointTSSetInitialGradient(adjts,gradient);CHKERRQ(ierr); /* it also initializes the adjoint variable */
  ierr = AdjointTSEventHandler(adjts);CHKERRQ(ierr);
  ierr = TSSetFromOptions(adjts);CHKERRQ(ierr);
  ierr = AdjointTSSetTimeLimits(adjts,t0,tf);CHKERRQ(ierr);
  ierr = TSSolve(adjts,NULL);CHKERRQ(ierr);
  ierr = AdjointTSComputeFinalGradient(adjts);CHKERRQ(ierr);
  ierr = TSDestroy(&adjts);CHKERRQ(ierr);
  ierr = VecDestroy(&lambda);CHKERRQ(ierr);

  /* restore trajectory */
  ierr = TSTrajectoryDestroy(&ts->trajectory);CHKERRQ(ierr);
  ts->trajectory  = otrj;
  PetscFunctionReturn(0);
}

/*
   TSResetObjective - Resets the list of objective functions for gradient computation.

   Logically Collective on TS

   Input Parameters:
.  ts - the TS context

   Level: developer

.seealso:
*/
PetscErrorCode TSResetObjective(TS ts)
{
  ObjectiveLink  link;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  link = ts->funchead;
  while (link) {
    ObjectiveLink olink = link;

    link = link->next;
    ierr = PetscFree(olink);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   TSSetObjective - Sets a cost functional for gradient computation.

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

   Notes: the functions passed in are appended to a list. More functions can be passed by simply calling TSSetObjective multiple times.
          The functionals are intendended to be used as integrand terms of a time integration (if fixtime == PETSC_MIN_REAL) or as evaluation at a given specific time.
          Regularizers fall into this category: use f_x = NULL, and pass f and f_m with any fixtime in between the half-open interval (t0, tf] (i.e. start and end of the forward solve).
          The size of the output vectors equals the size of the state and design vectors for f_x and f_m, respectively.

   Level: developer

.seealso: TSSetEvalGradient(), TSEvaluateObjectiveGradient(), TSSetEvalICGradient()
*/
PetscErrorCode TSSetObjective(TS ts, PetscReal fixtime, TSEvalObjective f, void* f_ctx, TSEvalObjectiveGradient f_x, void* f_x_ctx, TSEvalObjectiveGradient f_m, void* f_m_ctx)
{
  ObjectiveLink  link;
  PetscErrorCode ierr;

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
   TSSetEvalGradient - Sets the function for the evaluation of F_m(t,x(t),x_t(t);m).

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

.seealso: TSSetObjective(), TSEvaluateGradient(), TSSetEvalICGradient(), MATSHELL
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
   TSSetEvalICGradient - Sets the callback function to compute the matrices g_x(x0,m) and g_m(x0,m), if there is any dependence of the ODE initial conditions from the design parameters.

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
          Just the transposed action on the current adjoint state (via MatMultTranspose()) is needed.

   Level: developer

.seealso: TSSetObjective(), TSSetEvalGradient(), TSEvaluateObjectiveGradient(), MATSHELL, MatMultTranspose()
*/
PetscErrorCode TSSetEvalICGradient(TS ts, Mat J_x, Mat J_m, TSEvalICGradient f, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (J_x) PetscValidHeaderSpecific(J_x,MAT_CLASSID,2);
  PetscValidHeaderSpecific(J_m,MAT_CLASSID,3);

  ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradient_G",NULL);
  ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradient_GW",NULL);
  if (J_x) {
    ierr = PetscObjectReference((PetscObject)J_x);CHKERRQ(ierr);
    ierr = MatDestroy(&ts->G_x);CHKERRQ(ierr);
    ts->G_x = J_x;
  } else {
    ierr = MatDestroy(&ts->G_x);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject)J_m);CHKERRQ(ierr);
  ierr = MatDestroy(&ts->G_m);CHKERRQ(ierr);
  ts->G_m = J_m;

  ts->Ggrad     = f;
  ts->Ggrad_ctx = ctx;
  PetscFunctionReturn(0);
}

/*
   TSEvaluateObjective - Evaluates the objective functions set with TSSetObjective.

   Logically Collective on TS

   Input Parameters:
+  ts     - the TS context
.  X      - the initial vector for the state (can be NULL)
-  design - current design vector

   Output Parameters:
.  value - the value of the functional

   Notes: A forward solve is performed.

   Level: developer

.seealso: TSSetObjective(), TSSetEvalGradient(), TSSetEvalICGradient(), TSEvaluateObjectiveGradient(), TSEvaluateObjectiveAndGradient()
*/
PetscErrorCode TSEvaluateObjective(TS ts, Vec X, Vec design, PetscReal *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (X) PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidPointer(val,4);
  ierr = VecLockPush(design);CHKERRQ(ierr);
  ierr = TSEvaluateObjective_Private(ts,X,design,NULL,val);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   TSEvaluateObjectiveGradient - Evaluates the gradient of the objective functions.

   Logically Collective on TS

   Input Parameters:
+  ts       - the TS context
.  X        - the initial vector for the state (can be NULL)
-  design   - current design vector

   Output Parameters:
.  gradient - the computed gradient

   Notes: A forward and backward solve is performed.

   Level: developer

.seealso: TSSetObjective(), TSSetEvalGradient(), TSSetEvalICGradient(), TSEvaluateObjective(), TSEvaluateObjectiveAndGradient()
*/
PetscErrorCode TSEvaluateObjectiveGradient(TS ts, Vec X, Vec design, Vec gradient)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidHeaderSpecific(gradient,VEC_CLASSID,4);
  ierr = VecLockPush(design);CHKERRQ(ierr);
  ierr = TSEvaluateObjectiveGradient_Private(ts,X,design,gradient,NULL);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   TSEvaluateObjectiveAndGradient - Evaluates the objective functions and the gradient.

   Logically Collective on TS

   Input Parameters:
+  ts       - the TS context
.  X        - the initial vector for the state
-  design   - current design vector

   Output Parameters:
+  obj      - the value of the objective function
-  gradient - the computed gradient

   Notes:

   Level: developer

.seealso: TSSetObjective(), TSSetEvalGradient(), TSSetEvalICGradient(), TSEvaluateObjective(), TSEvaluateObjectiveGradient()
*/
PetscErrorCode TSEvaluateObjectiveAndGradient(TS ts, Vec X, Vec design, Vec gradient, PetscReal *obj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidHeaderSpecific(gradient,VEC_CLASSID,4);
  PetscValidPointer(obj,5);
  ierr = VecLockPush(design);CHKERRQ(ierr);
  ierr = TSEvaluateObjectiveGradient_Private(ts,X,design,gradient,obj);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   This code is very much inspired to the papers
   [1] Cao, Li, Petzold. Adjoint sensitivity analysis for differential-algebraic equations: algorithms and software, JCAM 149, 2002.
   [2] Cao, Li, Petzold. Adjoint sensitivity analysis for differential-algebraic equations: the adjoint DAE system and its numerical solution, SISC 24, 2003.
   TODO: register citations
*/
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
static PetscErrorCode TSGradientEvalObjectiveGradientU(TS ts, PetscReal time, Vec state, Vec design, Vec work, PetscBool *has, Vec out)
{
  PetscErrorCode ierr;
  ObjectiveLink  link = ts->funchead;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,time,2);
  PetscValidHeaderSpecific(state,VEC_CLASSID,3);
  PetscValidHeaderSpecific(design,VEC_CLASSID,4);
  PetscValidHeaderSpecific(work,VEC_CLASSID,5);
  PetscValidPointer(has,6);
  PetscValidHeaderSpecific(out,VEC_CLASSID,7);
  *has = PETSC_FALSE;
  while (link) {
    if (link->f_x && link->fixedtime <= PETSC_MIN_REAL) *has = PETSC_TRUE;
    link = link->next;
  }
  if (*has) {
    PetscBool firstdone = PETSC_FALSE;

    link = ts->funchead;
    ierr = VecLockPush(state);CHKERRQ(ierr);
    ierr = VecLockPush(design);CHKERRQ(ierr);
    while (link) {
      if (link->f_x && link->fixedtime <= PETSC_MIN_REAL) {
        if (!firstdone) {
          ierr = (*link->f_x)(ts,time,state,design,out,link->f_x_ctx);CHKERRQ(ierr);
        } else {
          ierr = (*link->f_x)(ts,time,state,design,work,link->f_x_ctx);CHKERRQ(ierr);
          ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
        }
        firstdone = PETSC_TRUE;
      }
      link = link->next;
    }
    ierr = VecLockPop(state);CHKERRQ(ierr);
    ierr = VecLockPop(design);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Evaluates derivative (wrt the state) of objective functions of the type f(U,param,t = fixed)
   These may lead to Dirac's delta terms in the adjoint DAE if the fixed time is in between (t0,tf) */
static PetscErrorCode TSGradientEvalObjectiveGradientUFixed(TS ts, PetscReal time, Vec state, Vec design, Vec work, PetscBool *has, Vec out)
{
  PetscErrorCode ierr;
  ObjectiveLink  link = ts->funchead;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,time,2);
  PetscValidHeaderSpecific(state,VEC_CLASSID,3);
  PetscValidHeaderSpecific(design,VEC_CLASSID,4);
  PetscValidHeaderSpecific(work,VEC_CLASSID,5);
  PetscValidPointer(has,6);
  PetscValidHeaderSpecific(out,VEC_CLASSID,7);
  *has = PETSC_FALSE;
  while (link) {
    if (link->f_x && link->fixedtime > PETSC_MIN_REAL && PetscAbsReal(link->fixedtime-time) < PETSC_SMALL) *has = PETSC_TRUE;
    link = link->next;
  }
  if (*has) {
    PetscBool firstdone = PETSC_FALSE;

    link = ts->funchead;
    ierr = VecLockPush(state);CHKERRQ(ierr);
    ierr = VecLockPush(design);CHKERRQ(ierr);
    ierr = VecSet(out,0.0);CHKERRQ(ierr);
    while (link) {
      if (link->f_x && link->fixedtime > PETSC_MIN_REAL && PetscAbsReal(link->fixedtime-time) < PETSC_SMALL) {
        if (!firstdone) {
          ierr = (*link->f_x)(ts,link->fixedtime,state,design,out,link->f_x_ctx);CHKERRQ(ierr);
        } else {
          ierr = (*link->f_x)(ts,link->fixedtime,state,design,work,link->f_x_ctx);CHKERRQ(ierr);
          ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
        }
        firstdone = PETSC_TRUE;
      }
      link = link->next;
    }
    ierr = VecLockPop(state);CHKERRQ(ierr);
    ierr = VecLockPop(design);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Evaluates derivative (wrt the parameters) of objective functions of the type f(U,param,t) */
static PetscErrorCode TSGradientEvalObjectiveGradientM(TS ts, PetscReal time, Vec state, Vec design, Vec work, PetscBool *has, Vec out)
{
  PetscErrorCode ierr;
  ObjectiveLink  link = ts->funchead;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,time,2);
  PetscValidHeaderSpecific(state,VEC_CLASSID,3);
  PetscValidHeaderSpecific(design,VEC_CLASSID,4);
  PetscValidHeaderSpecific(work,VEC_CLASSID,5);
  PetscValidPointer(has,6);
  PetscValidHeaderSpecific(out,VEC_CLASSID,7);
  *has = PETSC_FALSE;
  while (link) {
    if (link->f_m && link->fixedtime <= PETSC_MIN_REAL) *has = PETSC_TRUE;
    link = link->next;
  }
  if (*has) {
    PetscBool firstdone = PETSC_FALSE;

    link = ts->funchead;
    ierr = VecLockPush(state);CHKERRQ(ierr);
    ierr = VecLockPush(design);CHKERRQ(ierr);
    while (link) {
      if (link->f_m && link->fixedtime <= PETSC_MIN_REAL) {
        if (!firstdone) {
          ierr = (*link->f_m)(ts,time,state,design,out,link->f_m_ctx);CHKERRQ(ierr);
        } else {
          ierr = (*link->f_m)(ts,time,state,design,work,link->f_m_ctx);CHKERRQ(ierr);
          ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
        }
        firstdone = PETSC_TRUE;
      }
      link = link->next;
    }
    ierr = VecLockPop(state);CHKERRQ(ierr);
    ierr = VecLockPop(design);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   Evaluates derivative (wrt the parameters) of objective functions of the type f(U,param,t = tfixed)
   Regularizers fall into this category. They don't contribute to the adjoint DAE, only to the gradient
   They are evaluated in TSGradientPostStep
*/
static PetscErrorCode TSGradientEvalObjectiveGradientMFixed(TS ts, PetscReal ptime, PetscReal time, Vec state, Vec design, Vec work, PetscBool *has, Vec out)
{
  PetscErrorCode ierr;
  ObjectiveLink  link = ts->funchead;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,time,2);
  PetscValidHeaderSpecific(state,VEC_CLASSID,3);
  PetscValidHeaderSpecific(design,VEC_CLASSID,4);
  PetscValidHeaderSpecific(work,VEC_CLASSID,5);
  PetscValidPointer(has,6);
  PetscValidHeaderSpecific(out,VEC_CLASSID,7);
  *has = PETSC_FALSE;
  while (link) {
    if (link->f_m && ptime < link->fixedtime && link->fixedtime <= time) *has = PETSC_TRUE;
    link = link->next;
  }
  if (*has) {
    PetscBool firstdone = PETSC_FALSE;

    link = ts->funchead;
    ierr = VecLockPush(state);CHKERRQ(ierr);
    ierr = VecLockPush(design);CHKERRQ(ierr);
    ierr = VecSet(out,0.0);CHKERRQ(ierr);
    while (link) {
      if (link->f_m && ptime < link->fixedtime && link->fixedtime <= time) {
        if (!firstdone) {
          ierr = (*link->f_m)(ts,link->fixedtime,state,design,out,link->f_m_ctx);CHKERRQ(ierr);
        } else {
          ierr = (*link->f_m)(ts,link->fixedtime,state,design,work,link->f_m_ctx);CHKERRQ(ierr);
          ierr = VecAXPY(out,1.0,work);CHKERRQ(ierr);
        }
        firstdone = PETSC_TRUE;
      }
      link = link->next;
    }
    ierr = VecLockPop(state);CHKERRQ(ierr);
    ierr = VecLockPop(design);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   Apply "Jacobians" of initial conditions
   if transpose is true : y = G_x^-1 G_m x
   if transpose is false: y = G_m^t G_x^-T x
   (x0,design) are the variables one needs to linearize against to get G_x and G_m
*/
static PetscErrorCode TSGradientICApply(TS ts, PetscReal t0, Vec x0, Vec design, Vec x, Vec y, PetscBool transpose)
{
  PetscErrorCode ierr;
  KSP            ksp = NULL;
  Vec            workvec = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,t0,2);
  PetscValidHeaderSpecific(x0,VEC_CLASSID,3);
  PetscValidHeaderSpecific(design,VEC_CLASSID,4);
  PetscValidHeaderSpecific(x,VEC_CLASSID,5);
  PetscValidHeaderSpecific(y,VEC_CLASSID,6);
  PetscValidLogicalCollectiveBool(ts,transpose,7);
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  if (!ts->G_m) PetscFunctionReturn(0);
  if (ts->Ggrad) {
    ierr = (*ts->Ggrad)(ts,t0,x0,design,ts->G_x,ts->G_m,ts->Ggrad_ctx);CHKERRQ(ierr);
  }
  if (ts->G_x) { /* this is optional. If not provided, identity is assumed */
    ierr = PetscObjectQuery((PetscObject)ts,"_ts_gradient_G",(PetscObject*)&ksp);
    ierr = PetscObjectQuery((PetscObject)ts,"_ts_gradient_GW",(PetscObject*)&workvec);
    if (!ksp) {
      const char *prefix;
      ierr = KSPCreate(PetscObjectComm((PetscObject)ts),&ksp);CHKERRQ(ierr);
      ierr = KSPSetTolerances(ksp,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
      ierr = TSGetOptionsPrefix(ts,&prefix);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(ksp,prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ksp,"JacIC_");CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradient_G",(PetscObject)ksp);
      ierr = PetscObjectDereference((PetscObject)ksp);CHKERRQ(ierr);
    }
    if (!workvec) {
      ierr = MatCreateVecs(ts->G_m,NULL,&workvec);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradient_GW",(PetscObject)workvec);
      ierr = PetscObjectDereference((PetscObject)workvec);CHKERRQ(ierr);
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

/* Updates history vectors U and Udot for a given (forward) time, if they are present */
static PetscErrorCode TSTrajectoryUpdateHistoryVecs(TSTrajectory tj, TS ts, PetscReal time, Vec U, Vec Udot)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  if (ts) PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  PetscValidLogicalCollectiveReal(tj,time,3);
  if (U) PetscValidHeaderSpecific(U,VEC_CLASSID,4);
  if (Udot) PetscValidHeaderSpecific(Udot,VEC_CLASSID,5);
  if (U)    { ierr = VecLockPop(U);CHKERRQ(ierr); }
  if (Udot) { ierr = VecLockPop(Udot);CHKERRQ(ierr); }
  ierr = TSTrajectoryGetVecs(ts->trajectory,ts,PETSC_DECIDE,&time,U,Udot);CHKERRQ(ierr);
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
  PetscBool        timeindep; /* true if the DAE is A Udot + B U -f = 0 */
  Mat              J_U;       /* Jacobian : F_U (U,Udot,t) */
  Mat              J_Udot;    /* Jacobian : F_Udot(U,Udot,t) */
} SplitJac;

static PetscErrorCode SplitJacDestroy_Private(void *ptr)
{
  SplitJac*      s = (SplitJac*)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&s->J_U);CHKERRQ(ierr);
  ierr = MatDestroy(&s->J_Udot);CHKERRQ(ierr);
  ierr = PetscFree(s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* This is an helper routine to get F_U and F_Udot, which can be generalized in a public API with callbacks.
   Right now, the default implementation can be superseded by function composition */
static PetscErrorCode TSComputeSplitJacobians(TS ts, PetscReal time, Vec U, Vec Udot, Mat A, Mat pA, Mat B, Mat pB)
{
  PetscErrorCode (*f)(TS,PetscReal,Vec,Vec,Mat,Mat,Mat,Mat);
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
  ierr = PetscObjectQueryFunction((PetscObject)ts,"TSComputeSplitJacobians_C",&f);CHKERRQ(ierr);
  if (!f) {
    ierr = TSComputeIJacobian(ts,time,U,Udot,0.0,A,pA,PETSC_FALSE);CHKERRQ(ierr);
    ierr = TSComputeIJacobian(ts,time,U,Udot,1.0,B,pB,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatAXPY(B,-1.0,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    if (pB && pB != B) {
      ierr = MatAXPY(pB,-1.0,pA,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
  } else {
    ierr = (*f)(ts,time,U,Udot,A,pA,B,pB);CHKERRQ(ierr);
  }
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
  if (JU) {
    if (!splitJ->J_U) {
      Mat A;

      ierr = TSGetIJacobian(ts,&A,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&splitJ->J_U);CHKERRQ(ierr);
    }
    *JU = splitJ->J_U;
  }
  if (JUdot) {
    if (!splitJ->J_Udot) {
      Mat A;

      ierr = TSGetIJacobian(ts,&A,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&splitJ->J_Udot);CHKERRQ(ierr);
    }
    *JUdot = splitJ->J_Udot;
  }
  PetscFunctionReturn(0);
}

/* Updates F_Udot (splitJ->J_Udot) and F_U (splitJ->J_U) at a given (forward) time */
static PetscErrorCode TSUpdateSplitJacobiansFromHistory(TS ts, PetscReal time, Vec U, Vec Udot)
{
  PetscContainer c;
  SplitJac       *splitJ;
  Mat            J_U,J_Udot;
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
  ierr = TSGetSplitJacobians(ts,&J_U,&J_Udot);CHKERRQ(ierr);
  ierr = TSComputeSplitJacobians(ts,time,U,Udot,J_U,J_U,J_Udot,J_Udot);CHKERRQ(ierr);
  splitJ->splitdone = splitJ->timeindep ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* This function is used in AdjointTSIJacobian and (optionally) in TLMTSIJacobian.
   The assumption here is that the IJacobian routine is called after the IFunction (called with same time, U and Udot)
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

/* ------------------ Routines for adjoints of DAE, namespaced with AdjointTS ----------------------- */

typedef struct {
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
   -> the forward DAE is Udot - G(U) = 0 ( -> H(U,Udot,t) := Udot - G(U) )
   -> the adjoint DAE is F - L^T * G_U - Ldot^T in backward time (F the derivative of the objective wrt U)
   -> the adjoint DAE is Ldot^T = L^T * G_U - F in forward time */
static PetscErrorCode AdjointTSRHSFuncLinear(TS adjts, PetscReal time, Vec U, Vec F, void *ctx)
{
  AdjointCtx     *adj_ctx;
  PetscReal      fwdt;
  PetscBool      has;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - time + adj_ctx->t0;
  ierr = TSTrajectoryUpdateHistoryVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,fwdt,adj_ctx->W[0],NULL);CHKERRQ(ierr);
  ierr = TSGradientEvalObjectiveGradientU(adj_ctx->fwdts,fwdt,adj_ctx->W[0],adj_ctx->design,adj_ctx->W[3],&has,F);CHKERRQ(ierr);
  ierr = TSComputeRHSJacobian(adjts,time,U,adjts->Arhs,NULL);CHKERRQ(ierr);
  if (has) {
    ierr = VecScale(F,-1.0);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(adjts->Arhs,U,F,F);CHKERRQ(ierr);
  } else {
    ierr = MatMultTranspose(adjts->Arhs,U,F);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Given the forward DAE : H(U,Udot,t) = 0
   -> the adjoint DAE is : F - L^T * (H_U - d/dt H_Udot) - Ldot^T H_Udot = 0 (in backward time) (again, F is null for standard DAE adjoints)
   -> the adjoint DAE is : Ldot^T H_Udot + L^T * (H_U + d/dt H_Udot) + F = 0 (in forward time) */
static PetscErrorCode AdjointTSIFunctionLinear(TS adjts, PetscReal time, Vec U, Vec Udot, Vec F, void *ctx)
{
  AdjointCtx     *adj_ctx;
  Mat            J_U, J_Udot;
  PetscReal      fwdt;
  PetscBool      has;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - time + adj_ctx->t0;
  ierr = TSTrajectoryUpdateHistoryVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,fwdt,adj_ctx->W[0],NULL);CHKERRQ(ierr);
  ierr = TSGradientEvalObjectiveGradientU(adj_ctx->fwdts,fwdt,adj_ctx->W[0],adj_ctx->design,adj_ctx->W[3],&has,F);CHKERRQ(ierr);
  ierr = TSUpdateSplitJacobiansFromHistory(adj_ctx->fwdts,fwdt,adj_ctx->W[0],adj_ctx->W[1]);CHKERRQ(ierr);
  ierr = TSGetSplitJacobians(adj_ctx->fwdts,&J_U,&J_Udot);CHKERRQ(ierr);
  if (has) {
    ierr = MatMultTransposeAdd(J_U,U,F,F);CHKERRQ(ierr);
  } else {
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
   We store the increment - H_Udot^-T f_U in adj_ctx->W[2] and apply it during the AdjointTSPostStep
   It also works for index-1 DAEs.
*/
static PetscErrorCode AdjointTSComputeInitialConditions(TS,PetscReal,Vec,PetscBool);

static PetscErrorCode AdjointTSPostEvent(TS adjts, PetscInt nevents, PetscInt event_list[], PetscReal t, Vec U, PetscBool forwardsolve, void* ctx)
{
  AdjointCtx     *adj_ctx;
  PetscReal      fwdt;
  Vec            lambda;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - t + adj_ctx->t0;
  ierr = TSTrajectoryUpdateHistoryVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,fwdt,adj_ctx->W[0],NULL);CHKERRQ(ierr);
  ierr = TSGetSolution(adjts,&lambda);CHKERRQ(ierr);
  ierr = VecLockPush(lambda);CHKERRQ(ierr);
  ierr = AdjointTSComputeInitialConditions(adjts,t,adj_ctx->W[0],PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecLockPop(lambda);CHKERRQ(ierr);
  adj_ctx->dirac_delta = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* Partial integration (via the trapezoidal rule) of the gradient terms f_M + L^T H_M */
/* TODO: use a TS to do integration? */
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
static PetscErrorCode TSCreateAdjointTS(TS ts, TS* adjts)
{
  SNES            snes;
  KSP             ksp;
  KSPType         ksptype;
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
  PetscReal       atol,rtol,dtol;
  PetscInt        maxits;
  SplitJac        *splitJ;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* some comments by looking at [1] and [2]
     - Need to implement the augmented formulation (25) in [2] for implicit problems
     - Initial conditions for the adjoint variable are fine as they are now for the cases:
       - integrand terms : all but index-2 DAEs
       - g(x,T,p)        : all but index-2 DAEs
  */
  ierr = TSGetEquationType(ts,&eqtype);CHKERRQ(ierr);
  if (eqtype != TS_EQ_UNSPECIFIED && eqtype != TS_EQ_EXPLICIT && eqtype != TS_EQ_ODE_EXPLICIT &&
      eqtype != TS_EQ_IMPLICIT && eqtype != TS_EQ_ODE_IMPLICIT && eqtype != TS_EQ_DAE_SEMI_EXPLICIT_INDEX1)
      SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSEquationType %D\n",eqtype);
  ierr = TSGetI2Function(ts,NULL,&i2func,NULL);CHKERRQ(ierr);
  if (i2func) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Second order DAEs are not supported");
  ierr = TSCreate(PetscObjectComm((PetscObject)ts),adjts);CHKERRQ(ierr);
  ierr = TSGetType(ts,&type);CHKERRQ(ierr);
  ierr = TSSetType(*adjts,type);CHKERRQ(ierr);
  ierr = TSGetTolerances(ts,&atol,&vatol,&rtol,&vrtol);CHKERRQ(ierr);
  ierr = TSSetTolerances(*adjts,atol,vatol,rtol,vrtol);CHKERRQ(ierr);
  if (ts->adapt) {
    ierr = TSAdaptCreate(PetscObjectComm((PetscObject)*adjts),&(*adjts)->adapt);CHKERRQ(ierr);
    ierr = TSAdaptSetType((*adjts)->adapt,((PetscObject)ts->adapt)->type_name);CHKERRQ(ierr);
  }

  /* application context */
  ierr = PetscNew(&adj);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(*adjts,(void *)adj);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&lambda);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(lambda,4,&adj->W);CHKERRQ(ierr);

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

  /* Options specific to ADJTS */
  ierr = TSGetOptionsPrefix(*adjts,&prefix);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)*adjts),prefix,"Adjoint options","TS");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-timeindependent","Whether or not the DAE Jacobians are time-independent",NULL,splitJ->timeindep,&splitJ->timeindep,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  splitJ->splitdone = PETSC_FALSE;

  /* The equation type is the same */
  ierr = TSSetEquationType(*adjts,eqtype);CHKERRQ(ierr);

  /* adjoint DAE is linear */
  ierr = TSSetProblemType(*adjts,TS_LINEAR);CHKERRQ(ierr);

  /* get info on linear solver */
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetType(ksp,&ksptype);CHKERRQ(ierr);
  ierr = KSPGetTolerances(ksp,&rtol,&atol,&dtol,&maxits);CHKERRQ(ierr);

  /* use KSPSolveTranspose to solve the adjoint */
  ierr = TSGetSNES(*adjts,&snes);CHKERRQ(ierr);
  ierr = SNESKSPONLYSetUseTransposeSolve(snes,PETSC_TRUE);CHKERRQ(ierr);

  /* propagate KSP info of the forward model */
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,ksptype);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,rtol,atol,dtol,maxits);CHKERRQ(ierr);

  /* set special purpose post step method for incremental gradient evaluation */
  ierr = TSSetPostStep(*adjts,AdjointTSPostStep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSSetInitialGradient(TS adjts, Vec gradient)
{
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
  ierr = VecSet(adj_ctx->wgrad[0],0.0);CHKERRQ(ierr);
  ierr = VecSet(adj_ctx->wgrad[1],0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* compute initial conditions */
static PetscErrorCode AdjointTSComputeInitialConditions(TS adjts, PetscReal time, Vec svec, PetscBool apply)
{
  PetscReal      ftime;
  PetscContainer c;
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;
  TSIJacobian    ijac;
  PetscBool      has_g;
  TSEquationType eqtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_gradient_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Missing adjoint container");
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  ftime = adj_ctx->tf - time + adj_ctx->t0;
  ierr = VecLockPop(adj_ctx->W[2]);CHKERRQ(ierr);
  ierr = VecSet(adj_ctx->W[2],0.0);CHKERRQ(ierr);
  ierr = TSGradientEvalObjectiveGradientUFixed(adj_ctx->fwdts,ftime,svec,adj_ctx->design,adj_ctx->W[3],&has_g,adj_ctx->W[2]);CHKERRQ(ierr);
  ierr = TSGetEquationType(adj_ctx->fwdts,&eqtype);CHKERRQ(ierr);
  ierr = TSGetIJacobian(adjts,NULL,NULL,&ijac,NULL);CHKERRQ(ierr);
  if (eqtype == TS_EQ_DAE_SEMI_EXPLICIT_INDEX1) { /* details in [1,Section 4.2] */
    KSP       kspM,kspD;
    Mat       M,B,C,D;
    IS        diff = NULL,alg = NULL;
    Vec       f_x;
    PetscBool has_f;

    ierr = VecDuplicate(adj_ctx->W[2],&f_x);CHKERRQ(ierr);
    ierr = TSGradientEvalObjectiveGradientU(adj_ctx->fwdts,ftime,svec,adj_ctx->design,adj_ctx->W[3],&has_f,f_x);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)adj_ctx->fwdts,"_ts_algebraic_is",(PetscObject*)&alg);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)adj_ctx->fwdts,"_ts_differential_is",(PetscObject*)&diff);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjoint_index1_kspM",(PetscObject*)&kspM);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjoint_index1_kspD",(PetscObject*)&kspD);CHKERRQ(ierr);
    if (!kspD) {
      const char *prefix;
      ierr = TSGetOptionsPrefix(adjts,&prefix);CHKERRQ(ierr);
      ierr = KSPCreate(PetscObjectComm((PetscObject)adjts),&kspD);CHKERRQ(ierr);
      ierr = KSPSetTolerances(kspD,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(kspD,prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(kspD,"index1_D_");CHKERRQ(ierr);
      ierr = KSPSetFromOptions(kspD);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)adjts,"_ts_adjoint_index1_kspD",(PetscObject)kspD);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)kspD);CHKERRQ(ierr);
    }
    if (!kspM) {
      const char *prefix;
      ierr = TSGetOptionsPrefix(adjts,&prefix);CHKERRQ(ierr);
      ierr = KSPCreate(PetscObjectComm((PetscObject)adjts),&kspM);CHKERRQ(ierr);
      ierr = KSPSetTolerances(kspM,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(kspM,prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(kspM,"index1_M_");CHKERRQ(ierr);
      ierr = KSPSetFromOptions(kspM);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)adjts,"_ts_adjoint_index1_kspM",(PetscObject)kspM);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)kspM);CHKERRQ(ierr);
    }
    if (ijac) {
      Mat      J_U,J_Udot;
      PetscInt m,n,N;

      ierr = TSUpdateSplitJacobiansFromHistory(adj_ctx->fwdts,ftime,adj_ctx->W[0],adj_ctx->W[1]);CHKERRQ(ierr);
      ierr = TSGetSplitJacobians(adj_ctx->fwdts,&J_U,&J_Udot);CHKERRQ(ierr);
      ierr = MatGetOwnershipRange(J_Udot,&m,&n);CHKERRQ(ierr);
      if (!diff) {
        if (alg) {
          ierr = ISComplement(alg,m,n,&diff);CHKERRQ(ierr);
        } else {
          ierr = MatChop(J_Udot,PETSC_SMALL);CHKERRQ(ierr);
          ierr = MatFindNonzeroRows(J_Udot,&diff);CHKERRQ(ierr);
          if (!diff) SETERRQ(PetscObjectComm((PetscObject)adj_ctx->fwdts),PETSC_ERR_USER,"The DAE does not appear to have algebraic variables");
        }
        ierr = PetscObjectCompose((PetscObject)adj_ctx->fwdts,"_ts_differential_is",(PetscObject)diff);CHKERRQ(ierr);
        ierr = PetscObjectDereference((PetscObject)diff);CHKERRQ(ierr);
      }
      if (!alg) {
        ierr = ISComplement(diff,m,n,&alg);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)adj_ctx->fwdts,"_ts_algebraic_is",(PetscObject)alg);CHKERRQ(ierr);
        ierr = PetscObjectDereference((PetscObject)alg);CHKERRQ(ierr);
      }
      ierr = ISGetSize(alg,&N);CHKERRQ(ierr);
      if (!N) SETERRQ(PetscObjectComm((PetscObject)adj_ctx->fwdts),PETSC_ERR_USER,"The DAE does not have algebraic variables");
      ierr = MatCreateSubMatrix(J_Udot,diff,diff,MAT_INITIAL_MATRIX,&M);CHKERRQ(ierr);
      ierr = MatCreateSubMatrix(J_U,diff,alg,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
      ierr = MatCreateSubMatrix(J_U,alg,diff,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
      ierr = MatCreateSubMatrix(J_U,alg,alg,MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)adj_ctx->fwdts),PETSC_ERR_USER,"IJacobian routine is missing");

    /* we first compute the contribution of the g(x,T,p) terms,
       the initial conditions are consistent by construction with the adjointed algebraic constraints, i.e.
       B^T lambda_d + D^T lambda_a = 0 */
    if (has_g) {
      Vec       g_d,g_a;
      PetscReal norm;

      ierr = VecGetSubVector(adj_ctx->W[2],diff,&g_d);CHKERRQ(ierr);
      ierr = VecGetSubVector(adj_ctx->W[2],alg,&g_a);CHKERRQ(ierr);
      ierr = VecNorm(g_a,NORM_2,&norm);CHKERRQ(ierr);
      if (norm) {
        ierr = KSPSetOperators(kspD,D,D);CHKERRQ(ierr);
        ierr = KSPSolveTranspose(kspD,g_a,g_a);CHKERRQ(ierr);
        ierr = VecScale(g_a,-1.0);CHKERRQ(ierr);
        ierr = MatMultTransposeAdd(C,g_a,g_d,g_d);CHKERRQ(ierr);
        if (adj_ctx->fwdts->F_m) { /* add fixed term to the gradient */
          TS ts = adj_ctx->fwdts;
          if (ts->F_m_f) { /* non constant dependence */
            ierr = TSTrajectoryUpdateHistoryVecs(ts->trajectory,ts,ftime,adj_ctx->W[0],adj_ctx->W[1]);CHKERRQ(ierr);
            ierr = (*ts->F_m_f)(ts,ftime,adj_ctx->W[0],adj_ctx->W[1],adj_ctx->design,ts->F_m,ts->F_m_ctx);CHKERRQ(ierr);
          }
          ierr = MatMultTransposeAdd(ts->F_m,g_a,adj_ctx->gradient,adj_ctx->gradient);CHKERRQ(ierr);
        }
      }
      ierr = KSPSetOperators(kspM,M,M);CHKERRQ(ierr);
      ierr = KSPSolveTranspose(kspM,g_d,g_d);CHKERRQ(ierr);
      ierr = MatMultTranspose(B,g_d,g_a);CHKERRQ(ierr);
      ierr = KSPSetOperators(kspD,D,D);CHKERRQ(ierr);
      ierr = KSPSolveTranspose(kspD,g_a,g_a);CHKERRQ(ierr);
      ierr = VecScale(g_d,-1.0);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(adj_ctx->W[2],diff,&g_d);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(adj_ctx->W[2],alg,&g_a);CHKERRQ(ierr);
#if 0
      {
        Mat J_U;
        Vec test,test_a;
        PetscReal norm;

        ierr = VecDuplicate(adj_ctx->W[2],&test);CHKERRQ(ierr);
        ierr = TSGetSplitJacobians(adj_ctx->fwdts,&J_U,NULL);CHKERRQ(ierr);
        ierr = MatMultTranspose(J_U,adj_ctx->W[2],test);CHKERRQ(ierr);
        ierr = VecGetSubVector(test,alg,&test_a);CHKERRQ(ierr);
        ierr = VecNorm(test_a,NORM_2,&norm);CHKERRQ(ierr);
        ierr = PetscPrintf(PetscObjectComm((PetscObject)test),"This should be zero %1.16e\n",norm);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(test,alg,&test_a);CHKERRQ(ierr);
        ierr = VecDestroy(&test);CHKERRQ(ierr);
      }
#endif
    }
    /* we then compute, and add, admissible initial conditions for the algebraic variables, since the rhs of the adjoint system will depend
       on the derivative of the intergrand terms in the objective function w.r.t to the state */
    if (has_f) {
      Vec f_a,lambda_a;

      ierr = VecGetSubVector(f_x,alg,&f_a);CHKERRQ(ierr);
      ierr = VecGetSubVector(adj_ctx->W[2],alg,&lambda_a);CHKERRQ(ierr);
      ierr = KSPSetOperators(kspD,D,D);CHKERRQ(ierr);
      ierr = KSPSolveTranspose(kspD,f_a,f_a);CHKERRQ(ierr);
      ierr = VecAXPY(lambda_a,-1.0,f_a);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(adj_ctx->W[2],alg,&lambda_a);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(f_x,alg,&f_a);CHKERRQ(ierr);
    }
    if (0) {
      Mat J_U;
      Vec test,test_a;
      PetscReal norm;

      ierr = VecDuplicate(adj_ctx->W[2],&test);CHKERRQ(ierr);
      ierr = TSGetSplitJacobians(adj_ctx->fwdts,&J_U,NULL);CHKERRQ(ierr);
      ierr = MatMultTranspose(J_U,adj_ctx->W[2],test);CHKERRQ(ierr);
      ierr = VecGetSubVector(test,alg,&test_a);CHKERRQ(ierr);
      ierr = VecNorm(test_a,NORM_2,&norm);CHKERRQ(ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)test),"FINAL: This should be zero %1.16e\n",norm);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(test,alg,&test_a);CHKERRQ(ierr);
      ierr = VecDestroy(&test);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&f_x);CHKERRQ(ierr);
    ierr = MatDestroy(&M);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr);
    ierr = MatDestroy(&D);CHKERRQ(ierr);
  } else {
    if (has_g) {
      if (ijac) { /* lambda_T(T) = (J_Udot)^T D_x, D_x the gradients of the functionals that sample the solution at the final time */
        SNES      snes;
        KSP       ksp;
        Mat       J_Udot;
        PetscReal rtol,atol;
        PetscInt  maxits;

        ierr = TSUpdateSplitJacobiansFromHistory(adj_ctx->fwdts,ftime,adj_ctx->W[0],adj_ctx->W[1]);CHKERRQ(ierr);
        ierr = TSGetSNES(adjts,&snes);CHKERRQ(ierr);
        ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
        ierr = TSGetSplitJacobians(adj_ctx->fwdts,NULL,&J_Udot);CHKERRQ(ierr);
        ierr = KSPSetOperators(ksp,J_Udot,J_Udot);CHKERRQ(ierr);
        ierr = KSPGetTolerances(ksp,&rtol,&atol,NULL,&maxits);CHKERRQ(ierr);
        ierr = KSPSetTolerances(ksp,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,10000);CHKERRQ(ierr);
        ierr = KSPSolveTranspose(ksp,adj_ctx->W[2],adj_ctx->W[2]);CHKERRQ(ierr);
        ierr = KSPSetTolerances(ksp,rtol,atol,PETSC_DEFAULT,maxits);CHKERRQ(ierr);
      }
      /* the lambdas we use are equivalent to -lambda_T in [1] */
      ierr = VecScale(adj_ctx->W[2],-1.0);CHKERRQ(ierr);
    }
  }
  ierr = VecLockPush(adj_ctx->W[2]);CHKERRQ(ierr);
  if (apply) {
    Vec lambda;

    ierr = TSGetSolution(adjts,&lambda);CHKERRQ(ierr);
    ierr = VecCopy(adj_ctx->W[2],lambda);CHKERRQ(ierr);
    /* initialize wgrad[0] */
    ierr = VecSet(adj_ctx->wgrad[0],0.0);CHKERRQ(ierr);
    if (adj_ctx->fwdts->F_m) {
      Vec lambda;

      ierr = TSGetSolution(adjts,&lambda);CHKERRQ(ierr);
      TS ts = adj_ctx->fwdts;
      if (ts->F_m_f) { /* non constant dependence */
        ierr = TSTrajectoryUpdateHistoryVecs(ts->trajectory,ts,adj_ctx->tf,adj_ctx->W[0],adj_ctx->W[1]);CHKERRQ(ierr);
        ierr = (*ts->F_m_f)(ts,adj_ctx->tf,adj_ctx->W[0],adj_ctx->W[1],adj_ctx->design,ts->F_m,ts->F_m_ctx);CHKERRQ(ierr);
      }
      ierr = MatMultTranspose(ts->F_m,lambda,adj_ctx->wgrad[0]);CHKERRQ(ierr);
    }
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

      ierr = TSUpdateSplitJacobiansFromHistory(fwdts,adj_ctx->t0,adj_ctx->W[0],adj_ctx->W[1]);CHKERRQ(ierr);
      ierr = TSGetSplitJacobians(fwdts,NULL,&J_Udot);CHKERRQ(ierr);
      ierr = MatMultTranspose(J_Udot,lambda,adj_ctx->W[3]);CHKERRQ(ierr);
    }
    ierr = TSTrajectoryUpdateHistoryVecs(fwdts->trajectory,fwdts,adj_ctx->t0,adj_ctx->W[0],NULL);CHKERRQ(ierr);
    ierr = TSGradientICApply(fwdts,adj_ctx->t0,adj_ctx->W[0],adj_ctx->design,adj_ctx->W[3],adj_ctx->wgrad[0],PETSC_TRUE);CHKERRQ(ierr);
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

/* TODO: Some of the taylor remainder tests fail, and my best guess is because the gradient integration is not accurate enough
   Do we have to integrate the gradient terms with a fake TS that runs the quadrature? */
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
    PetscBool   has_m,has_m_fixed;

    if (poststep_ctx->firststep) {
      ierr = VecSet(poststep_ctx->wgrad[2],0.0);CHKERRQ(ierr);
      ierr = VecScale(poststep_ctx->gradient,dt/2.0);CHKERRQ(ierr);
    }
    ierr = TSGradientEvalObjectiveGradientM(ts,time,solution,poststep_ctx->design,poststep_ctx->wgrad[0],&has_m,poststep_ctx->wgrad[1]);CHKERRQ(ierr);

    /* Regularizers */
    ierr = TSGradientEvalObjectiveGradientMFixed(ts,ptime,time,solution,poststep_ctx->design,poststep_ctx->wgrad[3],&has_m_fixed,poststep_ctx->wgrad[0]);CHKERRQ(ierr);

    tt[0] = 1.0;    /* regularizer */
    tt[1] = dt/2.0; /* trapz */
    tt[2] = dt/2.0; /* trapz */
    if (has_m && has_m_fixed) {
      ierr = VecMAXPY(poststep_ctx->gradient,3,tt,poststep_ctx->wgrad);CHKERRQ(ierr);
    } else if (has_m && !has_m_fixed) {
      ierr = VecMAXPY(poststep_ctx->gradient,2,tt+1,poststep_ctx->wgrad);CHKERRQ(ierr);
    } else if (!has_m && has_m_fixed) {
      ierr = VecAXPY(poststep_ctx->gradient,1,poststep_ctx->wgrad[0]);CHKERRQ(ierr);
    }
    if (has_m) { /* stash current objective evaluation for next step of trapz rule */
      ierr = VecCopy(poststep_ctx->wgrad[1],poststep_ctx->wgrad[2]);CHKERRQ(ierr);
    }
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
  ierr = PetscObjectReference((PetscObject)U);CHKERRQ(ierr);
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
    PetscBool has;

    ierr = VecDuplicateVecs(poststep_ctx.gradient,4,&poststep_ctx.wgrad);CHKERRQ(ierr);
    ierr = TSGradientEvalObjectiveGradientM(ts,t0,X,design,poststep_ctx.wgrad[0],&has,poststep_ctx.gradient);CHKERRQ(ierr);
    if (!has) {
      ierr = VecSet(poststep_ctx.gradient,0.0);CHKERRQ(ierr);
    }
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
  ierr = PetscObjectDereference((PetscObject)U);CHKERRQ(ierr);
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
  PetscReal      t0,tf,dt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  otrj = ts->trajectory;
  ierr = TSTrajectoryCreate(PetscObjectComm((PetscObject)ts),&ts->trajectory);CHKERRQ(ierr);
  ierr = TSTrajectorySetType(ts->trajectory,ts,TSTRAJECTORYBASIC);CHKERRQ(ierr);
  ierr = TSTrajectorySetSolutionOnly(ts->trajectory,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSTrajectorySetFromOptions(ts->trajectory,ts);CHKERRQ(ierr);
  /* we don't have an API for this right now */
  ts->trajectory->adjoint_solve_mode = PETSC_FALSE;
  ierr = TSTrajectorySetUp(ts->trajectory,ts);CHKERRQ(ierr);

  /* sample initial condition dependency */
  ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
  if (ts->Ggrad) {
    ierr = (*ts->Ggrad)(ts,t0,X,design,ts->G_x,ts->G_m,ts->Ggrad_ctx);CHKERRQ(ierr);
  }

  /* forward solve */
  ierr = TSEvaluateObjective_Private(ts,X,design,gradient,val);CHKERRQ(ierr);

  /* adjoint */
  ierr = TSCreateAdjointTS(ts,&adjts);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&lambda);CHKERRQ(ierr);
  ierr = TSSetSolution(adjts,lambda);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&tf);CHKERRQ(ierr);
  ierr = TSGetPrevTime(ts,&dt);CHKERRQ(ierr);
  dt   = tf - dt;
  ierr = TSSetTimeStep(adjts,PetscMin(dt,tf-t0));CHKERRQ(ierr);
  ierr = AdjointTSSetTimeLimits(adjts,t0,tf);CHKERRQ(ierr);
  ierr = AdjointTSSetDesign(adjts,design);CHKERRQ(ierr);
  ierr = AdjointTSSetInitialGradient(adjts,gradient);CHKERRQ(ierr);
  ierr = AdjointTSComputeInitialConditions(adjts,t0,X,PETSC_TRUE);CHKERRQ(ierr);
  ierr = AdjointTSEventHandler(adjts);CHKERRQ(ierr);
  ierr = TSSetFromOptions(adjts);CHKERRQ(ierr);
  ierr = AdjointTSSetTimeLimits(adjts,t0,tf);CHKERRQ(ierr);
  if (adjts->adapt) {
    PetscBool istrj;

    ierr = PetscObjectTypeCompare((PetscObject)adjts->adapt,TSADAPTTRAJECTORY,&istrj);CHKERRQ(ierr);
    ierr = TSAdaptTrajectorySetTrajectory(adjts->adapt,ts->trajectory,PETSC_TRUE);CHKERRQ(ierr);
    if (!istrj) {
      ierr = TSSetMaxSteps(adjts,PETSC_MAX_INT);CHKERRQ(ierr);
      ierr = TSSetExactFinalTime(adjts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
    } else {
      PetscInt nsteps = ts->trajectory->tsh->n;

      ierr = TSSetMaxSteps(adjts,nsteps-1);CHKERRQ(ierr);
      ierr = TSSetExactFinalTime(adjts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
    }
  }
  ierr = TSSolve(adjts,NULL);CHKERRQ(ierr);
  ierr = AdjointTSComputeFinalGradient(adjts);CHKERRQ(ierr);
  ierr = TSDestroy(&adjts);CHKERRQ(ierr);
  ierr = VecDestroy(&lambda);CHKERRQ(ierr);

  /* restore trajectory */
  ierr = TSTrajectoryDestroy(&ts->trajectory);CHKERRQ(ierr);
  ts->trajectory  = otrj;
  PetscFunctionReturn(0);
}

/* ------------------ Routines for the TS representing the tangent linear model, namespaced by TLMTS ----------------------- */

typedef struct {
  TS        model;
  PetscBool userijac;
  Vec       *W;
  Vec       design;
  Vec       mdelta;
  Mat       P;
} TLMTS_Ctx;

static PetscErrorCode TLMTSDestroy_Private(void *ptr)
{
  TLMTS_Ctx*     tlm = (TLMTS_Ctx*)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&tlm->design);CHKERRQ(ierr);
  ierr = VecDestroy(&tlm->mdelta);CHKERRQ(ierr);
  ierr = VecDestroyVecs(3,&tlm->W);CHKERRQ(ierr);
  ierr = MatDestroy(&tlm->P);CHKERRQ(ierr);
  ierr = TSDestroy(&tlm->model);CHKERRQ(ierr);
  ierr = PetscFree(tlm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* TLMTS can be called by AdjointTS, this is a shortcut */
static PetscErrorCode TLMTSComputeSplitJacobians(TS ts, PetscReal time, Vec U, Vec Udot, Mat A, Mat pA, Mat B, Mat pB)
{
  TLMTS_Ctx      *tlm_ctx;
  Mat            J_U, J_Udot;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A == B) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"A and B must be different matrices");
  if (pA == pB) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"pA and pB must be different matrices");
  if (pA && pA != A) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Pmats not yet supported");
  if (pB && pB != B) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Pmats not yet supported");
  ierr = TSGetApplicationContext(ts,(void*)&tlm_ctx);CHKERRQ(ierr);
  ierr = TSUpdateSplitJacobiansFromHistory(tlm_ctx->model,time,tlm_ctx->W[0],tlm_ctx->W[1]);CHKERRQ(ierr);
  ierr = TSGetSplitJacobians(tlm_ctx->model,&J_U,&J_Udot);CHKERRQ(ierr);
  if (A) { ierr = MatCopy(J_U,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  if (B) { ierr = MatCopy(J_Udot,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/* The TLM DAE is J_Udot * U_dot + J_U * U + f = 0, with f = dH/dm * deltam */
static PetscErrorCode TLMTSIFunctionLinear(TS lts, PetscReal time, Vec U, Vec Udot, Vec F, void *ctx)
{
  TLMTS_Ctx      *tlm_ctx;
  Mat            J_U, J_Udot;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(lts,(void*)&tlm_ctx);CHKERRQ(ierr);
  ierr = TSUpdateSplitJacobiansFromHistory(tlm_ctx->model,time,tlm_ctx->W[0],tlm_ctx->W[1]);CHKERRQ(ierr);
  ierr = TSGetSplitJacobians(tlm_ctx->model,&J_U,&J_Udot);CHKERRQ(ierr);
  ierr = MatMult(J_U,U,F);CHKERRQ(ierr);
  ierr = MatMultAdd(J_Udot,Udot,F,F);CHKERRQ(ierr);
  if (tlm_ctx->model->F_m) {
    TS ts = tlm_ctx->model;
    if (ts->F_m_f) { /* non constant dependence */
      ierr = TSTrajectoryUpdateHistoryVecs(ts->trajectory,ts,time,tlm_ctx->W[0],tlm_ctx->W[1]);CHKERRQ(ierr);
      ierr = (*ts->F_m_f)(ts,time,tlm_ctx->W[0],tlm_ctx->W[1],tlm_ctx->design,ts->F_m,ts->F_m_ctx);CHKERRQ(ierr);
      ierr = MatMult(ts->F_m,tlm_ctx->mdelta,tlm_ctx->W[2]);CHKERRQ(ierr);
    }
    ierr = VecAXPY(F,1.0,tlm_ctx->W[2]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TLMTSIJacobian(TS lts, PetscReal time, Vec U, Vec Udot, PetscReal shift, Mat A, Mat B, void *ctx)
{
  TLMTS_Ctx      *tlm_ctx;
  TS             model;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(lts,(void*)&tlm_ctx);CHKERRQ(ierr);
  model = tlm_ctx->model;
  ierr = TSTrajectoryUpdateHistoryVecs(model->trajectory,model,time,tlm_ctx->W[0],tlm_ctx->W[1]);CHKERRQ(ierr);
  if (tlm_ctx->userijac) {
    ierr = TSComputeIJacobian(model,time,tlm_ctx->W[0],tlm_ctx->W[1],shift,A,B,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = TSComputeIJacobianWithSplits(model,time,tlm_ctx->W[0],tlm_ctx->W[1],shift,A,B,ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* The TLM DAE is U_dot = J_U * U - f, with f = dH/dm * deltam */
static PetscErrorCode TLMTSRHSFunctionLinear(TS lts, PetscReal time, Vec U, Vec F, void *ctx)
{
  TLMTS_Ctx      *tlm_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(lts,(void*)&tlm_ctx);CHKERRQ(ierr);
  ierr = TSComputeRHSJacobian(lts,time,U,lts->Arhs,NULL);CHKERRQ(ierr);
  ierr = MatMult(lts->Arhs,U,F);CHKERRQ(ierr);
  if (tlm_ctx->model->F_m) {
    TS ts = tlm_ctx->model;
    if (ts->F_m_f) { /* non constant dependence */
      ierr = TSTrajectoryUpdateHistoryVecs(ts->trajectory,ts,time,tlm_ctx->W[0],tlm_ctx->W[1]);CHKERRQ(ierr);
      ierr = (*ts->F_m_f)(ts,time,tlm_ctx->W[0],tlm_ctx->W[1],tlm_ctx->design,ts->F_m,ts->F_m_ctx);CHKERRQ(ierr);
      ierr = MatMult(ts->F_m,tlm_ctx->mdelta,tlm_ctx->W[2]);CHKERRQ(ierr);
    }
    ierr = VecAXPY(F,-1.0,tlm_ctx->W[2]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TLMTSRHSJacobian(TS lts, PetscReal time, Vec U, Mat A, Mat P, void *ctx)
{
  TLMTS_Ctx      *tlm_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(lts,(void*)&tlm_ctx);CHKERRQ(ierr);
  ierr = TSTrajectoryUpdateHistoryVecs(tlm_ctx->model->trajectory,tlm_ctx->model,time,tlm_ctx->W[0],NULL);CHKERRQ(ierr);
  ierr = TSComputeRHSJacobian(tlm_ctx->model,time,tlm_ctx->W[0],A,P);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* creates the TS for the tangent linear model */
static PetscErrorCode TSCreateTLMTS(TS ts, TS* lts)
{
  SNES           snes;
  KSP            ksp;
  KSPType        ksptype;
  Mat            A,B;
  Vec            vatol,vrtol;
  PetscContainer container;
  TLMTS_Ctx      *tlm_ctx;
  TSIFunction    ifunc;
  TSRHSFunction  rhsfunc;
  TSI2Function   i2func;
  TSType         type;
  TSEquationType eqtype;
  const char     *prefix;
  PetscReal      atol,rtol,dtol;
  PetscInt       maxits;
  SplitJac       *splitJ;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(lts,2);
  ierr = TSGetI2Function(ts,NULL,&i2func,NULL);CHKERRQ(ierr);
  if (i2func) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Second order DAEs are not supported");
  ierr = TSCreate(PetscObjectComm((PetscObject)ts),lts);CHKERRQ(ierr);
  ierr = TSGetType(ts,&type);CHKERRQ(ierr);
  ierr = TSSetType(*lts,type);CHKERRQ(ierr);
  ierr = TSGetTolerances(ts,&atol,&vatol,&rtol,&vrtol);CHKERRQ(ierr);
  ierr = TSSetTolerances(*lts,atol,vatol,rtol,vrtol);CHKERRQ(ierr);
  ierr = TSAdaptCreate(PetscObjectComm((PetscObject)*lts),&(*lts)->adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetType((*lts)->adapt,TSADAPTTRAJECTORY);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)(*lts),"TSComputeSplitJacobians_C",TLMTSComputeSplitJacobians);CHKERRQ(ierr);

  ierr = PetscNew(&tlm_ctx);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(*lts,(void *)tlm_ctx);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)ts);CHKERRQ(ierr);
  tlm_ctx->model = ts;

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
  splitJ->splitdone = PETSC_FALSE;

  /* wrap application context in a container, so that it will be destroyed when calling TSDestroy on lts */
  ierr = PetscContainerCreate(PetscObjectComm((PetscObject)(*lts)),&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,tlm_ctx);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(container,TLMTSDestroy_Private);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)(*lts),"_ts_tlm_ctx",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);

  /* setup callbacks for the tangent linear model DAE: we reuse the same jacobian matrices of the forward model */
  ierr = TSGetIFunction(ts,NULL,&ifunc,NULL);CHKERRQ(ierr);
  ierr = TSGetRHSFunction(ts,NULL,&rhsfunc,NULL);CHKERRQ(ierr);
  if (ifunc) {
    ierr = TSGetIJacobian(ts,&A,&B,NULL,NULL);CHKERRQ(ierr);
    ierr = TSSetIFunction(*lts,NULL,TLMTSIFunctionLinear,NULL);CHKERRQ(ierr);
    ierr = TSSetIJacobian(*lts,A,B,TLMTSIJacobian,NULL);CHKERRQ(ierr);
  } else {
    if (!rhsfunc) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"TSSetIFunction or TSSetRHSFunction not called");
    ierr = TSGetRHSJacobian(ts,&A,&B,NULL,NULL);CHKERRQ(ierr);
    ierr = TSSetRHSFunction(*lts,NULL,TLMTSRHSFunctionLinear,NULL);CHKERRQ(ierr);
    ierr = TSSetRHSJacobian(*lts,A,B,TLMTSRHSJacobian,NULL);CHKERRQ(ierr);
  }

  /* prefix */
  ierr = TSGetOptionsPrefix(ts,&prefix);CHKERRQ(ierr);
  ierr = TSSetOptionsPrefix(*lts,"tlm_");CHKERRQ(ierr);
  ierr = TSAppendOptionsPrefix(*lts,prefix);CHKERRQ(ierr);

  /* Options specific to TLMTS */
  ierr = TSGetOptionsPrefix(*lts,&prefix);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)*lts),prefix,"Tangent Linear Model options","TS");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-userijacobian","Use the user-provided IJacobian routine, instead of the splits, to compute the Jacobian",NULL,tlm_ctx->userijac,&tlm_ctx->userijac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* The equation type is the same */
  ierr = TSGetEquationType(ts,&eqtype);CHKERRQ(ierr);
  ierr = TSSetEquationType(*lts,eqtype);CHKERRQ(ierr);

  /* get info on linear solver */
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetType(ksp,&ksptype);CHKERRQ(ierr);
  ierr = KSPGetTolerances(ksp,&rtol,&atol,&dtol,&maxits);CHKERRQ(ierr);

  /* tangent linear model DAE is linear */
  ierr = TSSetProblemType(*lts,TS_LINEAR);CHKERRQ(ierr);

  /* propagate KSP info of the forward model */
  ierr = TSGetSNES(*lts,&snes);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,ksptype);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,rtol,atol,dtol,maxits);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------ Routines for the Mat that represents the linearized propagator ----------------------- */

typedef struct {
  TS           model;
  TS           lts;
  TS           adjlts;
  Vec          x0;
  TSTrajectory tj;
  PetscReal    t0;
  PetscReal    tf;
} MatPropagator_Ctx;

static PetscErrorCode MatDestroy_Propagator(Mat A)
{
  MatPropagator_Ctx *prop;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void **)&prop);CHKERRQ(ierr);
  ierr = TSTrajectoryDestroy(&prop->tj);CHKERRQ(ierr);
  ierr = VecDestroy(&prop->x0);CHKERRQ(ierr);
  ierr = TSDestroy(&prop->adjlts);CHKERRQ(ierr);
  ierr = TSDestroy(&prop->lts);CHKERRQ(ierr);
  ierr = TSDestroy(&prop->model);CHKERRQ(ierr);
  ierr = PetscFree(prop);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Just a silly function to pass information to initialize the adjoint variables */
static PetscErrorCode TLMTS_dummyRHS(TS ts, PetscReal time, Vec U, Vec M, Vec grad, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(U,grad);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_Propagator(Mat A, Vec x, Vec y)
{
  MatPropagator_Ctx *prop;
  TLMTS_Ctx         *tlm;
  PetscErrorCode    ierr;
  PetscBool         istrj;
  PetscReal         dt;
  TSTrajectory      otrj;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void **)&prop);CHKERRQ(ierr);
  otrj = prop->model->trajectory;
  prop->model->trajectory = prop->tj;
  prop->lts->trajectory = prop->tj;
  ierr = TSGetApplicationContext(prop->lts,(void *)&tlm);CHKERRQ(ierr);
  if (!tlm->mdelta) {
    ierr = VecDuplicate(y,&tlm->mdelta);CHKERRQ(ierr);
  }
  if (!tlm->W) {
    ierr = VecDuplicateVecs(prop->x0,3,&tlm->W);CHKERRQ(ierr);
    ierr = VecLockPush(tlm->W[0]);CHKERRQ(ierr);
    ierr = VecLockPush(tlm->W[1]);CHKERRQ(ierr);
  }
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = AdjointTSSetDesign(prop->adjlts,tlm->design);CHKERRQ(ierr);
  ierr = AdjointTSSetInitialGradient(prop->adjlts,y);CHKERRQ(ierr);
  /* Initialize adjoint variables using P^T x or x */
  ierr = VecSet(tlm->W[2],0.0);CHKERRQ(ierr);
  if (tlm->P) {
    ierr = MatMultTranspose(tlm->P,x,tlm->W[2]);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(x,tlm->W[2]);CHKERRQ(ierr);
  }
  ierr = AdjointTSComputeInitialConditions(prop->adjlts,prop->t0,tlm->W[2],PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSSetStepNumber(prop->adjlts,0);CHKERRQ(ierr);
  ierr = TSSetTime(prop->adjlts,prop->t0);CHKERRQ(ierr);
  ierr = TSHistoryGetTimeStep(prop->tj->tsh,PETSC_TRUE,0,&dt);CHKERRQ(ierr);
  ierr = TSSetTimeStep(prop->adjlts,dt);CHKERRQ(ierr);
  istrj = PETSC_FALSE;
  if (prop->adjlts->adapt) {
    ierr = PetscObjectTypeCompare((PetscObject)prop->adjlts->adapt,TSADAPTTRAJECTORY,&istrj);CHKERRQ(ierr);
    ierr = TSAdaptTrajectorySetTrajectory(prop->adjlts->adapt,prop->tj,PETSC_TRUE);CHKERRQ(ierr);
  }
  if (!istrj) {
    ierr = TSSetMaxSteps(prop->adjlts,PETSC_MAX_INT);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(prop->adjlts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  } else {
    PetscInt nsteps = prop->tj->tsh->n;

    ierr = TSSetMaxSteps(prop->adjlts,nsteps-1);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(prop->adjlts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  }
  ierr = TSSetMaxTime(prop->adjlts,prop->tf);CHKERRQ(ierr);
  ierr = TSSolve(prop->adjlts,NULL);CHKERRQ(ierr);
  ierr = AdjointTSComputeFinalGradient(prop->adjlts);CHKERRQ(ierr);
  prop->lts->trajectory = NULL;
  prop->tj = prop->model->trajectory;
  prop->model->trajectory = otrj;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_Propagator(Mat A, Vec x, Vec y)
{
  MatPropagator_Ctx *prop;
  TLMTS_Ctx         *tlm;
  PetscErrorCode    ierr;
  PetscReal         dt;
  PetscBool         istrj;
  Vec               sol;
  TSTrajectory      otrj;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void **)&prop);CHKERRQ(ierr);
  otrj = prop->model->trajectory;
  prop->model->trajectory = prop->tj;
  istrj = PETSC_FALSE;
  if (prop->lts->adapt) {
    ierr = PetscObjectTypeCompare((PetscObject)prop->lts->adapt,TSADAPTTRAJECTORY,&istrj);CHKERRQ(ierr);
    ierr = TSAdaptTrajectorySetTrajectory(prop->lts->adapt,prop->tj,PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = TSHistoryGetTimeStep(prop->tj->tsh,PETSC_FALSE,0,&dt);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(prop->lts,(void *)&tlm);CHKERRQ(ierr);
  if (!tlm->mdelta) {
    ierr = VecDuplicate(x,&tlm->mdelta);CHKERRQ(ierr);
  }
  ierr = VecCopy(x,tlm->mdelta);CHKERRQ(ierr);
  ierr = VecLockPush(tlm->mdelta);CHKERRQ(ierr);
  if (!tlm->W) {
    ierr = VecDuplicateVecs(prop->x0,3,&tlm->W);CHKERRQ(ierr);
    ierr = VecLockPush(tlm->W[0]);CHKERRQ(ierr);
    ierr = VecLockPush(tlm->W[1]);CHKERRQ(ierr);
  }

  /* initialize tlm->W[2] if needed */
  ierr = VecSet(tlm->W[2],0.0);CHKERRQ(ierr);
  if (prop->lts->F_m) {
    TS ts = prop->lts;
    if (!ts->F_m_f) { /* constant dependence */
      ierr = MatMult(ts->F_m,tlm->mdelta,tlm->W[2]);CHKERRQ(ierr);
    }
  }

  /* sample initial condition dependency */
  ierr = TSGetSolution(prop->lts,&sol);CHKERRQ(ierr);
  ierr = TSGradientICApply(prop->lts,prop->t0,prop->x0,tlm->design,x,sol,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecScale(sol,-1.0);CHKERRQ(ierr);

  ierr = TSSetStepNumber(prop->lts,0);CHKERRQ(ierr);
  ierr = TSSetTime(prop->lts,prop->t0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(prop->lts,dt);CHKERRQ(ierr);
  ierr = TSSetMaxTime(prop->lts,prop->tf);CHKERRQ(ierr);
  if (istrj) {
    ierr = TSSetMaxSteps(prop->lts,prop->tj->tsh->n-1);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(prop->lts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  } else {
    ierr = TSSetMaxSteps(prop->lts,PETSC_MAX_INT);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(prop->lts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  }
  ierr = TSSolve(prop->lts,NULL);CHKERRQ(ierr);
  if (tlm->P) {
    ierr = MatMult(tlm->P,sol,y);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(sol,y);CHKERRQ(ierr);
  }
  prop->tj = prop->model->trajectory;
  prop->model->trajectory = otrj;
  ierr = VecLockPop(tlm->mdelta);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* solves the forward model and stores its trajectory */
static PetscErrorCode MatPropagatorUpdate_Propagator(Mat A, PetscReal t0, PetscReal dt, PetscReal tf, Vec x0)
{
  Vec               osol;
  TSTrajectory      otrj;
  MatPropagator_Ctx *prop;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void **)&prop);CHKERRQ(ierr);
  ierr = VecLockPop(prop->x0);CHKERRQ(ierr);
  ierr = VecCopy(x0,prop->x0);CHKERRQ(ierr);
  ierr = VecLockPush(prop->x0);CHKERRQ(ierr);
  prop->t0 = t0;
  prop->tf = tf;
  ierr = TSTrajectoryDestroy(&prop->tj);CHKERRQ(ierr);

  /* Create trajectory object */
  otrj = prop->model->trajectory;
  ierr = TSTrajectoryCreate(PetscObjectComm((PetscObject)prop->model),&prop->model->trajectory);CHKERRQ(ierr);
  ierr = TSTrajectorySetType(prop->model->trajectory,prop->model,TSTRAJECTORYBASIC);CHKERRQ(ierr);
  ierr = TSTrajectorySetSolutionOnly(prop->model->trajectory,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSTrajectorySetFromOptions(prop->model->trajectory,prop->model);CHKERRQ(ierr);
  /* we don't have an API for this right now */
  prop->model->trajectory->adjoint_solve_mode = PETSC_FALSE;
  ierr = TSTrajectorySetUp(prop->model->trajectory,prop->model);CHKERRQ(ierr);

  /* Solve the forward nonlinear model in the given time window */
  ierr = TSGetSolution(prop->model,&osol);CHKERRQ(ierr);
  ierr = VecCopy(prop->x0,osol);CHKERRQ(ierr);
  ierr = TSSetStepNumber(prop->model,0);CHKERRQ(ierr);
  ierr = TSSetTime(prop->model,t0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(prop->model,dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(prop->model,PETSC_MAX_INT);CHKERRQ(ierr);
  ierr = TSSetMaxTime(prop->model,tf);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(prop->model,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSolve(prop->model,osol);CHKERRQ(ierr);
  prop->tj = prop->model->trajectory;
  prop->model->trajectory = otrj;
  PetscFunctionReturn(0);
}

static PetscErrorCode TLMTSSetProjection(TS lts, Mat P)
{
  PetscContainer c;
  TLMTS_Ctx      *tlm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lts,TS_CLASSID,1);
  PetscValidHeaderSpecific(P,MAT_CLASSID,2);
  ierr = PetscObjectQuery((PetscObject)lts,"_ts_tlm_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)lts),PETSC_ERR_PLIB,"Missing tlm container");
  ierr = PetscContainerGetPointer(c,(void**)&tlm);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)P);CHKERRQ(ierr);
  ierr = MatDestroy(&tlm->P);CHKERRQ(ierr);
  tlm->P = P;
  PetscFunctionReturn(0);
}

static PetscErrorCode TLMTSSetDesign(TS lts, Vec design)
{
  PetscContainer c;
  TLMTS_Ctx      *tlm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lts,TS_CLASSID,1);
  PetscValidHeaderSpecific(design,VEC_CLASSID,2);
  ierr = PetscObjectQuery((PetscObject)lts,"_ts_tlm_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)lts),PETSC_ERR_PLIB,"Missing tlm container");
  ierr = PetscContainerGetPointer(c,(void**)&tlm);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)design);CHKERRQ(ierr);
  ierr = VecDestroy(&tlm->design);CHKERRQ(ierr);
  tlm->design = design;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSCreatePropagatorMat_Private(TS ts, PetscReal t0, PetscReal dt, PetscReal tf, Vec x0, Vec design, Mat P, Mat *A)
{
  MatPropagator_Ctx *prop;
  PetscInt          M,N,m,n,rbs,cbs;
  Vec               X;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&prop);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)ts);CHKERRQ(ierr);
  prop->model = ts;
  if (P) {
    PetscBool   match;
    PetscLayout pmap,map;

    ierr = MatGetLayouts(P,NULL,&pmap);CHKERRQ(ierr);
    ierr = VecGetLayout(x0,&map);CHKERRQ(ierr);
    ierr = PetscLayoutCompare(map,pmap,&match);CHKERRQ(ierr);
    if (!match) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"The layouts of P and x0 do not match");
    ierr = MatGetSize(P,&M,NULL);CHKERRQ(ierr);
    ierr = MatGetLocalSize(P,&m,NULL);CHKERRQ(ierr);
    ierr = MatGetBlockSizes(P,&rbs,NULL);CHKERRQ(ierr);
  } else {
    ierr = VecGetSize(x0,&M);CHKERRQ(ierr);
    ierr = VecGetLocalSize(x0,&m);CHKERRQ(ierr);
    ierr = VecGetBlockSize(x0,&rbs);CHKERRQ(ierr);
  }
  if (!design) {
    if (prop->model->G_m) {
      ierr = MatCreateVecs(prop->model->G_m,&design,NULL);CHKERRQ(ierr);
    } else {
      ierr = VecDuplicate(x0,&design);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscObjectReference((PetscObject)design);CHKERRQ(ierr);
  }
  ierr = VecGetSize(design,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(design,&n);CHKERRQ(ierr);
  ierr = VecGetBlockSize(design,&cbs);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)ts),A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(*A,rbs,cbs);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSHELL);CHKERRQ(ierr);
  ierr = MatShellSetContext(*A,(void *)prop);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A,MATOP_MULT,(void (*)())MatMult_Propagator);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A,MATOP_MULT_TRANSPOSE,(void (*)())MatMultTranspose_Propagator);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A,MATOP_DESTROY,(void (*)())MatDestroy_Propagator);
  ierr = VecDuplicate(x0,&prop->x0);CHKERRQ(ierr);
  ierr = VecLockPush(prop->x0);CHKERRQ(ierr);     /* this vector is locked since it stores the initial conditions */
  ierr = MatPropagatorUpdate_Propagator(*A,t0,dt,tf,x0);CHKERRQ(ierr);
  ierr = MatSetUp(*A);CHKERRQ(ierr);

  /* creates the linear tangent model solver and its adjoint */
  ierr = TSCreateTLMTS(prop->model,&prop->lts);CHKERRQ(ierr);
  ierr = TLMTSSetDesign(prop->lts,design);CHKERRQ(ierr);
  if (P) {
    ierr = TLMTSSetProjection(prop->lts,P);CHKERRQ(ierr);
  }
  ierr = PetscObjectDereference((PetscObject)design);CHKERRQ(ierr);
  ierr = TSSetFromOptions(prop->lts);CHKERRQ(ierr);
  ierr = TSSetObjective(prop->lts,prop->tf,NULL,NULL,TLMTS_dummyRHS,NULL,NULL,NULL);CHKERRQ(ierr);
  if (prop->model->F_m) {
    ierr = TSSetEvalGradient(prop->lts,prop->model->F_m,prop->model->F_m_f,prop->model->F_m_ctx);CHKERRQ(ierr);
  }
  if (prop->model->G_m) {
    ierr = TSSetEvalICGradient(prop->lts,prop->model->G_x,prop->model->G_m,prop->model->Ggrad,prop->model->Ggrad_ctx);CHKERRQ(ierr);
  } else { /* we compute a linear dependence on u_0 by default */
    Mat      G_m;
    PetscInt m;

    ierr = VecGetLocalSize(x0,&m);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&G_m);CHKERRQ(ierr);
    ierr = MatSetSizes(G_m,m,m,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetType(G_m,MATAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(G_m,1,NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(G_m,1,NULL,0,NULL);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(G_m,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(G_m,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatShift(G_m,-1.0);CHKERRQ(ierr);
    ierr = TSSetEvalICGradient(prop->lts,NULL,G_m,NULL,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&G_m);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(prop->x0,&X);CHKERRQ(ierr);
  ierr = TSSetSolution(prop->lts,X);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = TSCreateAdjointTS(prop->lts,&prop->adjlts);CHKERRQ(ierr);
  ierr = TSSetFromOptions(prop->adjlts);CHKERRQ(ierr);
  ierr = VecDuplicate(x0,&X);CHKERRQ(ierr);
  ierr = TSSetSolution(prop->adjlts,X);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = AdjointTSSetTimeLimits(prop->adjlts,t0,tf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATPROPAGATOR - a matrix to be used for evaluating the linearized propagator of a time stepper.

   Level: developer

.seealso: TS, MATSHELL, TSSetEvalGradient(), TSSetEvalICGradient()
M*/

/*
   TSCreatePropagatorMat - Creates a Mat object that behaves like a linearized propagator of a time stepper on the time window [t0,tf].

   Logically Collective on TS

   Input Parameters:
+  ts     - the TS context
.  t0     - the initial time
.  dt     - the initial time step
.  tf     - the final time
.  x0     - the vector of initial conditions
.  design - the vector of design
-  P      - an optional projection

   Output Parameters:
.  A  - the Mat object

   Notes: Internally, the Mat object solves the Tangent Linear Model (TLM) during MatMult() and the adjoint of the TLM during MatMultTranspose().
          The design vector can be NULL if the Jacobians (wrt to the parameters) of the DAE and of the initial conditions does not explicitly depend on it.
          The projection P is intended to analyze problems in Generalized Stability Theory of the type

            argmax      ||P du_T||^2
           ||du_0||=1

          when one can be interested in the norm of the final state in a subspace.
          The projector is applied (via MatMult) on the final state computed by the forward Tangent Linear Model.
          The transposed action of P is instead used to initialize the adjoint of the Tangent Linear Model.
          Note that the role of P is somewhat different from that of the matrix representing the norm in the state variables.
          If P is provided, the row layout of A is the same of that of P. Otherwise, it is the same of that of x0.
          The column layout of A is the same of that of the design vector. If the latter is not provided, it is inherited from x0.
          Note that the column layout of P should be compatible with the that of x0.

   Level: developer

.seealso: TSSetEvalGradient(), TSSetEvalICGradient()
*/
PetscErrorCode TSCreatePropagatorMat(TS ts, PetscReal t0, PetscReal dt, PetscReal tf, Vec x0, Vec design, Mat P, Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,t0,2);
  PetscValidLogicalCollectiveReal(ts,dt,3);
  PetscValidLogicalCollectiveReal(ts,tf,4);
  PetscValidHeaderSpecific(x0,VEC_CLASSID,5);
  if (design) PetscValidHeaderSpecific(design,VEC_CLASSID,6);
  if (P) PetscValidHeaderSpecific(P,MAT_CLASSID,7);
  PetscValidPointer(A,8);
  ierr = TSCreatePropagatorMat_Private(ts,t0,dt,tf,x0,design,P,A);CHKERRQ(ierr);
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
$  f(TS ts,PetscReal t,Vec u,Vec m,Vec out,void *ctx);

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
  PetscValidLogicalCollectiveReal(ts,fixtime,2);
  if (f) PetscValidPointer(f,3);
  if (f_ctx) PetscValidPointer(f_ctx,4);
  if (f_x) PetscValidPointer(f_x,5);
  if (f_x_ctx) PetscValidPointer(f_x_ctx,6);
  if (f_m) PetscValidPointer(f_m,7);
  if (f_m_ctx) PetscValidPointer(f_m_ctx,8);
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
$  f(TS ts,PetscReal t,Vec u,Vec u_t,Vec m,Mat J,void *ctx);

+  t   - time at step/stage being solved
.  u   - state vector
.  u_t - time derivative of state vector
.  m   - design vector
.  J   - the jacobian
-  ctx - [optional] user-defined context

   Notes: The layout of the J matrix has to be compatible with that of the state vector.
          The matrix doesn't need to be in assembled form. For propagator computations, J needs to implement MatMult() and MatMultTranspose().
          For gradient computations, just its action via MatMultTranspose() is needed.

   Level: developer

.seealso: TSSetObjective(), TSEvaluateObjectiveGradient(), TSSetEvalICGradient(), TSCreatePropagatorMat(), MATSHELL, MATPROPAGATOR
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
   TSSetEvalICGradient - Sets the callback function to compute the matrices g_x(x0,m) and g_m(x0,m), if there is any dependence of the DAE initial conditions from the design parameters.

   Logically Collective on TS

   Input Parameters:
+  ts      - the TS context obtained from TSCreate()
.  J_x     - the Mat object to hold g_x(x0,m) (optional, if NULL identity is assumed)
.  J_m     - the Mat object to hold g_m(x0,m)
.  f       - the function evaluation routine
-  f_ctx   - user-defined context for private data for the function evaluation routine (may be NULL)

   Calling sequence of f:
$  f(TS ts,PetscReal t,Vec u,Vec m,Mat Gx,Mat Gm,void *ctx);

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
  if (J_m) PetscValidHeaderSpecific(J_m,MAT_CLASSID,3);

  ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradient_G",NULL);
  ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradient_GW",NULL);
  if (J_x) {
    ierr = PetscObjectReference((PetscObject)J_x);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&ts->G_x);CHKERRQ(ierr);
  ts->G_x = J_x;
  if (J_m) {
    ierr = PetscObjectReference((PetscObject)J_m);CHKERRQ(ierr);
  }
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

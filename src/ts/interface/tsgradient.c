/*
   This code is very much inspired to the papers
   [1] Cao, Li, Petzold. Adjoint sensitivity analysis for differential-algebraic equations: algorithms and software, JCAM 149, 2002.
   [2] Cao, Li, Petzold. Adjoint sensitivity analysis for differential-algebraic equations: the adjoint DAE system and its numerical solution, SISC 24, 2003.
   TODO: register citations
*/
#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petsc/private/snesimpl.h>

/* ------------------ Helper routines for PDE-constrained support to evaluate objective functions and their gradients ----------------------- */
/* TODO: remove TS argument from user functions? */

/* Evaluates objective functions of the type f(state,design,t) */
static PetscErrorCode TSEvaluateObjective(TS ts, PetscReal time, Vec state, Vec design, PetscReal *val)
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
    if (link->f && link->fixedtime <= PETSC_MIN_REAL) {
      PetscReal v;
      ierr = (*link->f)(ts,time,state,design,&v,link->f_ctx);CHKERRQ(ierr);
      *val += v;
    }
    link = link->next;
  }
  ierr = VecLockPop(state);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Evaluates objective functions of the type f(state,design,t = fixed) */
static PetscErrorCode TSEvaluateObjectiveFixed(TS ts, PetscReal time, Vec state, Vec design, PetscReal *val)
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
    if (link->f && time == link->fixedtime) {
      PetscReal v;
      ierr = (*link->f)(ts,link->fixedtime,state,design,&v,link->f_ctx);CHKERRQ(ierr);
      *val += v;
    }
    link = link->next;
  }
  ierr = VecLockPop(state);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Evaluates derivative (wrt the state) of objective functions of the type f(state,design,t) */
static PetscErrorCode TSEvaluateObjective_U(TS ts, PetscReal time, Vec state, Vec design, Vec work, PetscBool *has, Vec out)
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

/* Evaluates derivative (wrt the state) of objective functions of the type f(state,design,t = fixed)
   These may lead to Dirac's delta terms in the adjoint DAE if the fixed time is in between (t0,tf) */
static PetscErrorCode TSEvaluateObjective_UFixed(TS ts, PetscReal time, Vec state, Vec design, Vec work, PetscBool *has, Vec out)
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

/* Evaluates derivative (wrt the parameters) of objective functions of the type f(state,design,t) */
static PetscErrorCode TSEvaluateObjective_M(TS ts, PetscReal time, Vec state, Vec design, Vec work, PetscBool *has, Vec out)
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

/* Evaluates derivative (wrt the parameters) of objective functions of the type f(state,design,t = tfixed) */
static PetscErrorCode TSEvaluateObjective_MFixed(TS ts, PetscReal time, Vec state, Vec design, Vec work, PetscBool *has, Vec out)
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
    if (link->f_m && time == link->fixedtime) *has = PETSC_TRUE;
    link = link->next;
  }
  if (*has) {
    PetscBool firstdone = PETSC_FALSE;

    link = ts->funchead;
    ierr = VecLockPush(state);CHKERRQ(ierr);
    ierr = VecLockPush(design);CHKERRQ(ierr);
    while (link) {
      if (link->f_m && time == link->fixedtime) {
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

PETSC_UNUSED static PetscErrorCode TSGetNumObjectives(TS ts, PetscInt *n)
{
  ObjectiveLink link = ts->funchead;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(n,2);
  *n = 0;
  while (link) {
    (*n)++;
    link = link->next;
  }
  PetscFunctionReturn(0);
}

/*
   Apply "Jacobians" of initial conditions
   if transpose is false : y = G_x^-1 G_m x
   if transpose is true  : y = G_m^t G_x^-T x
   (x0,design) are the variables one needs to linearize against to get the partial Jacobians G_x and G_m
*/
static PetscErrorCode TSLinearizeICApply(TS ts, PetscReal t0, Vec x0, Vec design, Vec x, Vec y, PetscBool transpose)
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
  if (!ts->G_m) {
    ierr = VecSet(y,0.0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (ts->Ggrad) {
    ierr = (*ts->Ggrad)(ts,t0,x0,design,ts->G_x,ts->G_m,ts->Ggrad_ctx);CHKERRQ(ierr);
  }
  if (ts->G_x) { /* this is optional. If not provided, identity is assumed */
    ierr = PetscObjectQuery((PetscObject)ts,"_ts_gradient_G",(PetscObject*)&ksp);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)ts,"_ts_gradient_GW",(PetscObject*)&workvec);CHKERRQ(ierr);
    if (!ksp) {
      const char *prefix;
      ierr = KSPCreate(PetscObjectComm((PetscObject)ts),&ksp);CHKERRQ(ierr);
      ierr = KSPSetTolerances(ksp,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
      ierr = TSGetOptionsPrefix(ts,&prefix);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(ksp,prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ksp,"JacIC_");CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradient_G",(PetscObject)ksp);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)ksp);CHKERRQ(ierr);
    }
    if (!workvec) {
      ierr = MatCreateVecs(ts->G_m,NULL,&workvec);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradient_GW",(PetscObject)workvec);CHKERRQ(ierr);
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

/* Updates history vectors U and Udot for a given time, if they are present */
static PetscErrorCode TSTrajectoryUpdateHistoryVecs(TSTrajectory tj, TS ts, PetscReal time, Vec U, Vec Udot)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  PetscValidLogicalCollectiveReal(tj,time,3);
  if (U) PetscValidHeaderSpecific(U,VEC_CLASSID,4);
  if (Udot) PetscValidHeaderSpecific(Udot,VEC_CLASSID,5);
  if (U)    { ierr = VecLockPop(U);CHKERRQ(ierr); }
  if (Udot) { ierr = VecLockPop(Udot);CHKERRQ(ierr); }
  ierr = TSTrajectoryGetVecs(tj,ts,PETSC_DECIDE,&time,U,Udot);CHKERRQ(ierr);
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
  Mat              pJ_U;      /* Jacobian : F_U (U,Udot,t) (to be preconditioned) */
  Mat              pJ_Udot;   /* Jacobian : F_Udot(U,Udot,t) (to be preconditioned) */
} SplitJac;

static PetscErrorCode SplitJacDestroy_Private(void *ptr)
{
  SplitJac*      s = (SplitJac*)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&s->pJ_U);CHKERRQ(ierr);
  ierr = MatDestroy(&s->pJ_Udot);CHKERRQ(ierr);
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
static PetscErrorCode TSGetSplitJacobians(TS ts, Mat* JU, Mat* pJU, Mat *JUdot, Mat* pJUdot)
{
  PetscErrorCode ierr;
  PetscContainer c;
  Mat            A,B;
  SplitJac       *splitJ;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (JU) PetscValidPointer(JU,2);
  if (pJU) PetscValidPointer(pJU,3);
  if (JUdot) PetscValidPointer(JUdot,4);
  if (pJUdot) PetscValidPointer(pJUdot,5);
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_splitJac",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Missing splitJac container");
  ierr = PetscContainerGetPointer(c,(void**)&splitJ);CHKERRQ(ierr);
  ierr = TSGetIJacobian(ts,&A,&B,NULL,NULL);CHKERRQ(ierr);
  if (JU) {
    if (!splitJ->J_U) {
      ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&splitJ->J_U);CHKERRQ(ierr);
    }
    *JU = splitJ->J_U;
  }
  if (pJU) {
    if (!splitJ->pJ_U) {
      if (B && B != A) {
        ierr = MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&splitJ->pJ_U);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectReference((PetscObject)splitJ->J_U);CHKERRQ(ierr);
        splitJ->pJ_U = splitJ->J_U;
      }
    }
    *pJU = splitJ->pJ_U;
  }
  if (JUdot) {
    if (!splitJ->J_Udot) {
      ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&splitJ->J_Udot);CHKERRQ(ierr);
    }
    *JUdot = splitJ->J_Udot;
  }
  if (pJUdot) {
    if (!splitJ->pJ_Udot) {
      if (B && B != A) {
        ierr = MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&splitJ->pJ_Udot);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectReference((PetscObject)splitJ->J_Udot);CHKERRQ(ierr);
        splitJ->pJ_Udot = splitJ->J_Udot;
      }
    }
    *pJUdot = splitJ->pJ_Udot;
  }
  PetscFunctionReturn(0);
}

/* Updates F_Udot (splitJ->J_Udot) and F_U (splitJ->J_U) at a given time
   Updates U and Udot from history if needed */
static PetscErrorCode TSUpdateSplitJacobiansFromHistory(TS ts, PetscReal time, Vec U, Vec Udot)
{
  PetscContainer c;
  SplitJac       *splitJ;
  Mat            J_U = NULL,J_Udot = NULL,pJ_U = NULL,pJ_Udot = NULL;
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
  ierr = TSGetSplitJacobians(ts,&J_U,&pJ_U,&J_Udot,&pJ_Udot);CHKERRQ(ierr);
  ierr = TSComputeSplitJacobians(ts,time,U,Udot,J_U,pJ_U,J_Udot,pJ_Udot);CHKERRQ(ierr);
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
  ierr = MatCopy(splitJ->J_U,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(A,shift,splitJ->J_Udot,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PetscObjectStateGet((PetscObject)A,&splitJ->Astate);CHKERRQ(ierr);
  ierr = PetscObjectGetId((PetscObject)A,&splitJ->Aid);CHKERRQ(ierr);
  splitJ->shift = shift;
  if (B && A != B) {
    ierr = MatCopy(splitJ->pJ_U,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(B,shift,splitJ->pJ_Udot,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------ Wrappers for quadrature evaluation ----------------------- */

/* prototypes for cost integral evaluation */
typedef PetscErrorCode (*SQuadEval)(TS,Vec,PetscReal,PetscReal*,void*);
typedef PetscErrorCode (*VQuadEval)(TS,Vec,PetscReal,Vec,void*);

typedef struct {
  PetscErrorCode (*user)(TS); /* user post step method */
  PetscBool      userafter;   /* call user-defined poststep after quadrature evaluation */
  SQuadEval      seval;       /* scalar function to be evaluated */
  void           *seval_ctx;  /* context for scalar function */
  PetscReal      squad;       /* scalar function value */
  PetscReal      psquad;      /* previous scalar function value (for trapezoidal rule) */
  VQuadEval      veval;       /* vector function to be evaluated */
  void           *veval_ctx;  /* context for vector function */
  Vec            vquad;       /* used for vector quadrature */
  Vec            *wquad;      /* quadrature work vectors: 0 and 1 are used by the trapezoidal rule */
  PetscInt       cur,old;     /* pointers to current and old wquad vectors for trapezoidal rule */
} TSQuadratureCtx;

static PetscErrorCode TSQuadratureCtxDestroy_Private(void *ptr)
{
  TSQuadratureCtx* q = (TSQuadratureCtx*)ptr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(2,&q->wquad);CHKERRQ(ierr);
  ierr = PetscFree(q->veval_ctx);CHKERRQ(ierr);
  ierr = PetscFree(q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalQuadObj(TS ts, Vec U, PetscReal t, PetscReal *f, void* ctx)
{
  Vec            design = (Vec)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSEvaluateObjective(ts,t,U,design,f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalQuadObj_M(TS ts, Vec U, PetscReal t, Vec F, void* ctx)
{
  Vec            *v = (Vec*)ctx;
  PetscBool      has_m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSEvaluateObjective_M(ts,t,U,v[0],v[1],&has_m,F);CHKERRQ(ierr);
  if (!has_m) SETERRQ(PetscObjectComm((PetscObject)U),PETSC_ERR_PLIB,"This should not happen");
  PetscFunctionReturn(0);
}

/* private context for adjoint quadrature */
typedef struct {
  TS        fwdts;
  Vec       hist[2];
  Vec       design;
  PetscReal t0,tf;
} AdjEvalQuadCtx;

/* computes L^T H_M at backward time t */
static PetscErrorCode EvalQuadDAE_M(TS ts, Vec L, PetscReal t, Vec F, void* ctx)
{
  AdjEvalQuadCtx *q = (AdjEvalQuadCtx*)ctx;
  TS             fwdts = q->fwdts;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (fwdts->F_m_f) { /* non constant dependence */
    PetscReal fwdt = q->tf - t + q->t0;
    ierr = TSTrajectoryUpdateHistoryVecs(fwdts->trajectory,fwdts,fwdt,q->hist[0],q->hist[1]);CHKERRQ(ierr);
    ierr = (*fwdts->F_m_f)(fwdts,fwdt,q->hist[0],q->hist[1],q->design,fwdts->F_m,fwdts->F_m_ctx);CHKERRQ(ierr);
  }
  ierr = MatMultTranspose(fwdts->F_m,L,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSQuadrature_PostStep(TS ts)
{
  PetscContainer  container;
  Vec             solution;
  TSQuadratureCtx *qeval_ctx;
  PetscReal       squad = 0.0;
  PetscReal       dt,time,ptime;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->reason < 0) PetscFunctionReturn(0);
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_evaluate_quadrature",(PetscObject*)&container);CHKERRQ(ierr);
  if (!container) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Missing evaluate_quadrature container");
  ierr = PetscContainerGetPointer(container,(void**)&qeval_ctx);CHKERRQ(ierr);
  if (qeval_ctx->user && !qeval_ctx->userafter) {
    PetscStackPush("User post-step function");
    ierr = (*qeval_ctx->user)(ts);CHKERRQ(ierr);
    PetscStackPop;
  }

  ierr = TSGetSolution(ts,&solution);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&time);CHKERRQ(ierr);
  if (ts->reason == TS_CONVERGED_TIME) {
    ierr = TSGetMaxTime(ts,&time);CHKERRQ(ierr);
  }

  /* time step used */
  ierr = TSGetPrevTime(ts,&ptime);CHKERRQ(ierr);
  dt   = time - ptime;

  /* scalar quadrature (psquad have been initialized with the first function evaluation) */
  if (qeval_ctx->seval) {
    PetscStackPush("TS scalar quadrature function");
    ierr = (*qeval_ctx->seval)(ts,solution,time,&squad,qeval_ctx->seval_ctx);CHKERRQ(ierr);
    PetscStackPop;
    qeval_ctx->squad += dt*(squad+qeval_ctx->psquad)/2.0;
    qeval_ctx->psquad = squad;
  }

  /* scalar quadrature (qeval_ctx->wquad[qeval_ctx->old] have been initialized with the first function evaluation) */
  if (qeval_ctx->veval) {
    PetscScalar t[2];
    PetscInt    tmp;

    PetscStackPush("TS vector quadrature function");
    ierr = (*qeval_ctx->veval)(ts,solution,time,qeval_ctx->wquad[qeval_ctx->cur],qeval_ctx->veval_ctx);CHKERRQ(ierr);
    PetscStackPop;

    /* trapezoidal rule */
    t[0] = dt/2.0;
    t[1] = dt/2.0;
    ierr = VecMAXPY(qeval_ctx->vquad,2,t,qeval_ctx->wquad);CHKERRQ(ierr);

    /* swap pointers */
    tmp            = qeval_ctx->cur;
    qeval_ctx->cur = qeval_ctx->old;
    qeval_ctx->old = tmp;
  }
  if (qeval_ctx->user && qeval_ctx->userafter) {
    PetscStackPush("User post-step function");
    ierr = (*qeval_ctx->user)(ts);CHKERRQ(ierr);
    PetscStackPop;
  }
  PetscFunctionReturn(0);
}

/* ------------------ Routines for adjoints of DAE, namespaced with AdjointTS ----------------------- */

typedef struct {
  Vec       design;      /* design vector (fixed) */
  TS        fwdts;       /* forward solver */
  PetscReal t0,tf;       /* time limits, for forward time recovery */
  Vec       *W;          /* work vectors W[0] and W[1] always store U and Udot at a given time */
  Vec       gradient;    /* gradient we are evaluating */
  Vec       wgrad;       /* work vector */
  PetscBool dirac_delta; /* If true, means that a delta contribution needs to be added to lambda during the post step method */
} AdjointCtx;

static PetscErrorCode AdjointTSDestroy_Private(void *ptr)
{
  AdjointCtx*    adj = (AdjointCtx*)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&adj->design);CHKERRQ(ierr);
  ierr = VecDestroyVecs(4,&adj->W);CHKERRQ(ierr);
  ierr = VecDestroy(&adj->gradient);CHKERRQ(ierr);
  ierr = VecDestroy(&adj->wgrad);CHKERRQ(ierr);
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

/* The adjoint formulation used assumes the problem as H(U,Udot,t) = 0
   -> the forward DAE is Udot - G(U) = 0 ( -> H(U,Udot,t) := Udot - G(U) )
   -> the adjoint DAE is F - L^T * G_U - Ldot^T in backward time (F the derivative of the objective wrt U)
   -> the adjoint DAE is Ldot^T = L^T * G_U - F in forward time */
static PetscErrorCode AdjointTSRHSFunctionLinear(TS adjts, PetscReal time, Vec U, Vec F, void *ctx)
{
  AdjointCtx     *adj_ctx;
  PetscReal      fwdt;
  PetscBool      has;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - time + adj_ctx->t0;
  ierr = TSTrajectoryUpdateHistoryVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,fwdt,adj_ctx->W[0],NULL);CHKERRQ(ierr);
  ierr = TSEvaluateObjective_U(adj_ctx->fwdts,fwdt,adj_ctx->W[0],adj_ctx->design,adj_ctx->W[3],&has,F);CHKERRQ(ierr);
  ierr = TSComputeRHSJacobian(adjts,time,U,adjts->Arhs,adjts->Brhs);CHKERRQ(ierr);
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
   -> the adjoint DAE is : Ldot^T H_Udot + L^T * (H_U + d/dt H_Udot) + F = 0 (in forward time)
   TODO : add support for augmented system to avoid d/dt H_Udot (which is zero for most of the problems)
*/
static PetscErrorCode AdjointTSIFunctionLinear(TS adjts, PetscReal time, Vec U, Vec Udot, Vec F, void *ctx)
{
  AdjointCtx     *adj_ctx;
  Mat            J_U = NULL, J_Udot = NULL;
  PetscReal      fwdt;
  PetscBool      has;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - time + adj_ctx->t0;
  ierr = TSTrajectoryUpdateHistoryVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,fwdt,adj_ctx->W[0],NULL);CHKERRQ(ierr);
  ierr = TSEvaluateObjective_U(adj_ctx->fwdts,fwdt,adj_ctx->W[0],adj_ctx->design,adj_ctx->W[3],&has,F);CHKERRQ(ierr);
  ierr = TSUpdateSplitJacobiansFromHistory(adj_ctx->fwdts,fwdt,adj_ctx->W[0],adj_ctx->W[1]);CHKERRQ(ierr);
  ierr = TSGetSplitJacobians(adj_ctx->fwdts,&J_U,NULL,&J_Udot,NULL);CHKERRQ(ierr);
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

/* Handles the detection of Dirac's delta forcing terms (i.e. f_state(state,design,t = fixed)) in the adjoint equations */
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
static PetscErrorCode AdjointTSComputeInitialConditions(TS,PetscReal,Vec,PetscBool,PetscBool);

static PetscErrorCode AdjointTSPostEvent(TS adjts, PetscInt nevents, PetscInt event_list[], PetscReal t, Vec U, PetscBool forwardsolve, void* ctx)
{
  AdjointCtx     *adj_ctx;
  PetscReal      fwdt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - t + adj_ctx->t0;
  ierr = TSTrajectoryUpdateHistoryVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,fwdt,adj_ctx->W[0],NULL);CHKERRQ(ierr);
  /* just to double check that U is not changed here, as it is changed in AdjointTSPostStep */
  ierr = VecLockPush(U);CHKERRQ(ierr);
  ierr = AdjointTSComputeInitialConditions(adjts,t,adj_ctx->W[0],PETSC_FALSE,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecLockPop(U);CHKERRQ(ierr);
  adj_ctx->dirac_delta = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSPostStep(TS adjts)
{
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  if (adjts->reason < 0) PetscFunctionReturn(0);
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  /* We detected Dirac's delta terms -> add the increment here
     Re-evaluate L^T H_M and restart quadrature if needed */
  if (adj_ctx->dirac_delta) {
    PetscContainer  container;
    TSQuadratureCtx *qeval_ctx;
    Vec             lambda;

    ierr = TSGetSolution(adjts,&lambda);CHKERRQ(ierr);
    ierr = VecAXPY(lambda,1.0,adj_ctx->W[2]);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)adjts,"_ts_evaluate_quadrature",(PetscObject*)&container);CHKERRQ(ierr);
    if (container) {
      PetscReal t;

      ierr = TSGetTime(adjts,&t);CHKERRQ(ierr);
      ierr = PetscContainerGetPointer(container,(void**)&qeval_ctx);CHKERRQ(ierr);
      PetscStackPush("ADJTS vector quadrature function");
      ierr = (*qeval_ctx->veval)(adjts,lambda,t,qeval_ctx->wquad[qeval_ctx->old],qeval_ctx->veval_ctx);CHKERRQ(ierr);
      PetscStackPop;
    }
  }
  adj_ctx->dirac_delta = PETSC_FALSE;
  if (adjts->reason) { /* prevent from accumulation errors XXX */
    PetscReal time;

    ierr = TSGetTime(adjts,&time);CHKERRQ(ierr);
    adj_ctx->tf = time;
  }
  PetscFunctionReturn(0);
}

/* Creates the adjoint TS */
/* some comments by looking at [1] and [2]
   - Need to implement the augmented formulation (25) in [2] for implicit problems
   - Initial conditions for the adjoint variable are fine as they are now for the cases:
     - integrand terms : all but index-2 DAEs
     - g(x,T,p)        : all but index-2 DAEs
*/
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
  TSI2Function    i2func;
  TSType          type;
  TSEquationType  eqtype;
  const char      *prefix;
  PetscReal       atol,rtol,dtol;
  PetscInt        maxits;
  SplitJac        *splitJ;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
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

  /* setup callbacks for adjoint DAE: we reuse the same jacobian matrices of the forward solve */
  ierr = TSGetIFunction(ts,NULL,&ifunc,NULL);CHKERRQ(ierr);
  ierr = TSGetRHSFunction(ts,NULL,&rhsfunc,NULL);CHKERRQ(ierr);
  if (ifunc) {
    ierr = TSGetIJacobian(ts,&A,&B,NULL,NULL);CHKERRQ(ierr);
    ierr = TSSetIFunction(*adjts,NULL,AdjointTSIFunctionLinear,NULL);CHKERRQ(ierr);
    ierr = TSSetIJacobian(*adjts,A,B,AdjointTSIJacobian,NULL);CHKERRQ(ierr);
  } else {
    TSRHSJacobian rhsjacfunc;
    if (!rhsfunc) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"TSSetIFunction or TSSetRHSFunction not called");
    ierr = TSSetRHSFunction(*adjts,NULL,AdjointTSRHSFunctionLinear,NULL);CHKERRQ(ierr);
    ierr = TSGetRHSJacobian(ts,NULL,NULL,&rhsjacfunc,NULL);CHKERRQ(ierr);
    ierr = TSGetRHSMats_Private(ts,&A,&B);CHKERRQ(ierr);
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

  /* options specific to AdjointTS */
  ierr = TSGetOptionsPrefix(*adjts,&prefix);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)*adjts),prefix,"Adjoint options","TS");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-timeindependent","Whether or not the DAE Jacobians are time-independent",NULL,splitJ->timeindep,&splitJ->timeindep,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  splitJ->splitdone = PETSC_FALSE;

  /* the equation type is the same */
  ierr = TSSetEquationType(*adjts,eqtype);CHKERRQ(ierr);

  /* the adjoint DAE is linear */
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

  /* set special purpose post step method for handling of discontinuities */
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

  ierr = PetscObjectReference((PetscObject)gradient);CHKERRQ(ierr);
  ierr = VecDestroy(&adj_ctx->gradient);CHKERRQ(ierr);
  adj_ctx->gradient = gradient;
  PetscFunctionReturn(0);
}

/*
  Compute initial conditions for the adjoint DAE. It also initializes the quadrature (if needed).
  We use svec (instead of just loading from history inside the function), as the propagator Mat can use P*U
*/
static PetscErrorCode AdjointTSComputeInitialConditions(TS adjts, PetscReal time, Vec svec, PetscBool apply, PetscBool qinit)
{
  PetscReal      fwdt;
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
  fwdt = adj_ctx->tf - time + adj_ctx->t0;
  ierr = VecLockPop(adj_ctx->W[2]);CHKERRQ(ierr);
  ierr = VecSet(adj_ctx->W[2],0.0);CHKERRQ(ierr);
  ierr = TSEvaluateObjective_UFixed(adj_ctx->fwdts,fwdt,svec,adj_ctx->design,adj_ctx->W[3],&has_g,adj_ctx->W[2]);CHKERRQ(ierr);
  ierr = TSGetEquationType(adj_ctx->fwdts,&eqtype);CHKERRQ(ierr);
  ierr = TSGetIJacobian(adjts,NULL,NULL,&ijac,NULL);CHKERRQ(ierr);
  if (eqtype == TS_EQ_DAE_SEMI_EXPLICIT_INDEX1) { /* details in [1,Section 4.2] */
    KSP       kspM,kspD;
    Mat       M,B,C,D;
    IS        diff = NULL,alg = NULL;
    Vec       f_x;
    PetscBool has_f;

    ierr = VecDuplicate(adj_ctx->W[2],&f_x);CHKERRQ(ierr);
    ierr = TSEvaluateObjective_U(adj_ctx->fwdts,fwdt,svec,adj_ctx->design,adj_ctx->W[3],&has_f,f_x);CHKERRQ(ierr);
    if (!has_f && !has_g) {
      ierr = VecDestroy(&f_x);CHKERRQ(ierr);
      goto initialize;
    }
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
      Mat      J_U = NULL,J_Udot = NULL;
      PetscInt m,n,N;

      ierr = TSUpdateSplitJacobiansFromHistory(adj_ctx->fwdts,fwdt,adj_ctx->W[0],adj_ctx->W[1]);CHKERRQ(ierr);
      ierr = TSGetSplitJacobians(adj_ctx->fwdts,&J_U,NULL,&J_Udot,NULL);CHKERRQ(ierr);
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
            ierr = TSTrajectoryUpdateHistoryVecs(ts->trajectory,ts,fwdt,adj_ctx->W[0],adj_ctx->W[1]);CHKERRQ(ierr);
            ierr = (*ts->F_m_f)(ts,fwdt,adj_ctx->W[0],adj_ctx->W[1],adj_ctx->design,ts->F_m,ts->F_m_ctx);CHKERRQ(ierr);
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
#if 0
    {
      Mat J_U;
      Vec test,test_a;
      PetscReal norm;

      ierr = VecDuplicate(adj_ctx->W[2],&test);CHKERRQ(ierr);
      ierr = TSGetSplitJacobians(adj_ctx->fwdts,&J_U,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = MatMultTranspose(J_U,adj_ctx->W[2],test);CHKERRQ(ierr);
      ierr = VecGetSubVector(test,alg,&test_a);CHKERRQ(ierr);
      ierr = VecNorm(test_a,NORM_2,&norm);CHKERRQ(ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)test),"FINAL: This should be zero %1.16e\n",norm);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(test,alg,&test_a);CHKERRQ(ierr);
      ierr = VecDestroy(&test);CHKERRQ(ierr);
    }
#endif
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
        Mat       J_Udot = NULL;
        PetscReal rtol,atol;
        PetscInt  maxits;

        ierr = TSUpdateSplitJacobiansFromHistory(adj_ctx->fwdts,fwdt,adj_ctx->W[0],adj_ctx->W[1]);CHKERRQ(ierr);
        ierr = TSGetSNES(adjts,&snes);CHKERRQ(ierr);
        ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
        ierr = TSGetSplitJacobians(adj_ctx->fwdts,NULL,NULL,&J_Udot,NULL);CHKERRQ(ierr);
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
initialize:
  ierr = VecLockPush(adj_ctx->W[2]);CHKERRQ(ierr);
  if (apply) {
    Vec lambda;

    ierr = TSGetSolution(adjts,&lambda);CHKERRQ(ierr);
    ierr = VecCopy(adj_ctx->W[2],lambda);CHKERRQ(ierr);
  }
  if (qinit && adj_ctx->fwdts->F_m) { /* initialize quadrature */
    TS              ts = adj_ctx->fwdts;
    TSQuadratureCtx *qeval_ctx;
    AdjEvalQuadCtx  *adjq;
    PetscContainer  c;
    Vec             lambda;
    PetscReal       t0;

    if (!adj_ctx->gradient) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Missing gradient vector");
    if (!adj_ctx->design) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Missing design vector");
    ierr = PetscObjectQuery((PetscObject)adjts,"_ts_evaluate_quadrature",(PetscObject*)&c);CHKERRQ(ierr);
    if (!c) {
      ierr = PetscObjectQuery((PetscObject)adj_ctx->fwdts,"_ts_evaluate_quadrature",(PetscObject*)&c);CHKERRQ(ierr);
      if (!c) {
        ierr = PetscNew(&qeval_ctx);CHKERRQ(ierr);
        ierr = PetscContainerCreate(PetscObjectComm((PetscObject)adjts),&c);CHKERRQ(ierr);
        ierr = PetscContainerSetPointer(c,(void *)qeval_ctx);CHKERRQ(ierr);
        ierr = PetscContainerSetUserDestroy(c,TSQuadratureCtxDestroy_Private);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)adjts,"_ts_evaluate_quadrature",(PetscObject)c);CHKERRQ(ierr);
        ierr = PetscObjectDereference((PetscObject)c);CHKERRQ(ierr);
      } else {
        ierr = PetscObjectCompose((PetscObject)adjts,"_ts_evaluate_quadrature",(PetscObject)c);CHKERRQ(ierr);
      }
    }
    ierr = PetscContainerGetPointer(c,(void**)&qeval_ctx);CHKERRQ(ierr);
    qeval_ctx->user      = adjts->poststep;
    qeval_ctx->userafter = PETSC_TRUE;
    qeval_ctx->seval     = NULL;
    qeval_ctx->veval     = EvalQuadDAE_M;
    qeval_ctx->vquad     = adj_ctx->gradient;
    qeval_ctx->cur       = 0;
    qeval_ctx->old       = 1;
    if (!qeval_ctx->wquad) {
      ierr = VecDuplicateVecs(qeval_ctx->vquad,2,&qeval_ctx->wquad);CHKERRQ(ierr);
    }

    ierr = PetscFree(qeval_ctx->veval_ctx);CHKERRQ(ierr);
    ierr = PetscNew(&adjq);CHKERRQ(ierr);
    adjq->fwdts   = ts;
    adjq->t0      = adj_ctx->t0;
    adjq->tf      = adj_ctx->tf;
    adjq->design  = adj_ctx->design;
    adjq->hist[0] = adj_ctx->W[0];
    adjq->hist[1] = adj_ctx->W[1];
    qeval_ctx->veval_ctx = adjq;

    ierr = TSGetTime(adjts,&t0);CHKERRQ(ierr);
    ierr = TSGetSolution(adjts,&lambda);CHKERRQ(ierr);
    PetscStackPush("ADJTS vector quadrature function");
    ierr = (*qeval_ctx->veval)(adjts,lambda,t0,qeval_ctx->wquad[qeval_ctx->old],qeval_ctx->veval_ctx);CHKERRQ(ierr);
    PetscStackPop;
    ierr = TSSetPostStep(adjts,TSQuadrature_PostStep);CHKERRQ(ierr);
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

/* update time limits in the application context
   they are needed to recover the forward time from the backward */
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
  adj_ctx->tf = tf;
  adj_ctx->t0 = t0;
  PetscFunctionReturn(0);
}

/* event handler for Dirac's delta terms (if any) */
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
      Mat J_Udot = NULL;

      ierr = TSUpdateSplitJacobiansFromHistory(fwdts,adj_ctx->t0,adj_ctx->W[0],adj_ctx->W[1]);CHKERRQ(ierr);
      ierr = TSGetSplitJacobians(fwdts,NULL,NULL,&J_Udot,NULL);CHKERRQ(ierr);
      ierr = MatMultTranspose(J_Udot,lambda,adj_ctx->W[3]);CHKERRQ(ierr);
    }
    ierr = TSTrajectoryUpdateHistoryVecs(fwdts->trajectory,fwdts,adj_ctx->t0,adj_ctx->W[0],NULL);CHKERRQ(ierr);
    if (!adj_ctx->wgrad) {
      ierr = VecDuplicate(adj_ctx->gradient,&adj_ctx->wgrad);CHKERRQ(ierr);
    }
    ierr = TSLinearizeICApply(fwdts,adj_ctx->t0,adj_ctx->W[0],adj_ctx->design,adj_ctx->W[3],adj_ctx->wgrad,PETSC_TRUE);CHKERRQ(ierr);
    ierr = VecAXPY(adj_ctx->gradient,1.0,adj_ctx->wgrad);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEvaluateObjective_Private(TS ts, Vec X, Vec design, Vec gradient, PetscReal *val)
{
  Vec             U;
  PetscContainer  container;
  TSQuadratureCtx *qeval_ctx;
  PetscReal       t0,tf,tfup;
  PetscInt        tst;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (gradient) PetscValidHeaderSpecific(gradient,VEC_CLASSID,4);

  /* solution vector */
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  if (!U) {
    ierr = VecDuplicate(X,&U);CHKERRQ(ierr);
    ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)U);CHKERRQ(ierr);
  }
  ierr = VecCopy(X,U);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&tst);CHKERRQ(ierr);
  ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);

  /* set special purpose post step method for quadrature evaluation */
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_evaluate_quadrature",(PetscObject*)&container);CHKERRQ(ierr);
  if (!container) {
    ierr = PetscNew(&qeval_ctx);CHKERRQ(ierr);
    ierr = PetscContainerCreate(PetscObjectComm((PetscObject)ts),&container);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container,(void *)qeval_ctx);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container,TSQuadratureCtxDestroy_Private);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)ts,"_ts_evaluate_quadrature",(PetscObject)container);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)container);CHKERRQ(ierr);
  } else {
    ierr = PetscContainerGetPointer(container,(void**)&qeval_ctx);CHKERRQ(ierr);
  }
  qeval_ctx->user      = ts->poststep;
  qeval_ctx->userafter = PETSC_FALSE;
  qeval_ctx->seval     = val ? EvalQuadObj : NULL;
  qeval_ctx->seval_ctx = design;
  qeval_ctx->squad     = 0.0;
  qeval_ctx->psquad    = 0.0;
  qeval_ctx->veval     = gradient ? EvalQuadObj_M : NULL;
  qeval_ctx->vquad     = gradient;
  qeval_ctx->cur       = 0;
  qeval_ctx->old       = 1;

  ierr = TSSetPostStep(ts,TSQuadrature_PostStep);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);

  /* evaluate at initial time */
  ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
  if (qeval_ctx->seval) {
    PetscStackPush("TS scalar quadrature function");
    ierr = (*qeval_ctx->seval)(ts,U,t0,&qeval_ctx->psquad,qeval_ctx->seval_ctx);CHKERRQ(ierr);
    PetscStackPop;
  }
  if (qeval_ctx->veval) {
    PetscBool has;

    ierr = PetscFree(qeval_ctx->veval_ctx);CHKERRQ(ierr);
    if (!qeval_ctx->wquad) {
      ierr = VecDuplicateVecs(qeval_ctx->vquad,2,&qeval_ctx->wquad);CHKERRQ(ierr);
    }
    ierr = TSEvaluateObjective_M(ts,t0,U,design,qeval_ctx->wquad[qeval_ctx->cur],&has,qeval_ctx->wquad[qeval_ctx->old]);CHKERRQ(ierr);
    if (!has) { /* cost integrands not present */
      qeval_ctx->veval = NULL;
    } else { /* need the design vector and one work vector for the function evaluation */
      Vec *v,work;

      ierr = PetscMalloc1(2,&v);CHKERRQ(ierr);
      v[0] = design;
      ierr = PetscObjectQuery((PetscObject)ts,"_ts_quadwork_0",(PetscObject*)&work);CHKERRQ(ierr);
      if (!work) {
        ierr = VecDuplicate(qeval_ctx->vquad,&work);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)ts,"_ts_quadwork_0",(PetscObject)work);CHKERRQ(ierr);
        ierr = PetscObjectDereference((PetscObject)work);CHKERRQ(ierr);
      }
      v[1] = work;
      qeval_ctx->veval_ctx = v;
    }
  }

  /* forward solve */
  ierr = TSGetMaxTime(ts,&tf);CHKERRQ(ierr);
  do {
    ObjectiveLink link;

    /* determine if there are functionals and gradients wrt parameters of the type f(U,M,t=fixed) to be evaluated */
    /* we don't use events since there's no API to add new events to a pre-exiting set */
    tfup = tf;
    ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
    link = ts->funchead;
    while (link) {
      if (((link->f && qeval_ctx->seval) || (link->f_m && gradient)) && t0 < link->fixedtime && link->fixedtime <= tf) {
        tfup = PetscMin(tfup,link->fixedtime);
      }
      link = link->next;
    }
    ierr = TSSetMaxTime(ts,tfup);CHKERRQ(ierr);
    ierr = TSSolve(ts,NULL);CHKERRQ(ierr);

    if (qeval_ctx->seval) {
      PetscReal v;
      ierr = TSEvaluateObjectiveFixed(ts,tfup,U,design,&v);CHKERRQ(ierr);
      qeval_ctx->squad += v;
    }
    if (qeval_ctx->vquad) {
      Vec       work0,work1;
      PetscBool has;

      ierr = PetscObjectQuery((PetscObject)ts,"_ts_quadwork_0",(PetscObject*)&work0);CHKERRQ(ierr);
      if (!work0) {
        ierr = VecDuplicate(qeval_ctx->vquad,&work0);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)ts,"_ts_quadwork_0",(PetscObject)work0);CHKERRQ(ierr);
        ierr = PetscObjectDereference((PetscObject)work0);CHKERRQ(ierr);
      }
      ierr = PetscObjectQuery((PetscObject)ts,"_ts_quadwork_1",(PetscObject*)&work1);CHKERRQ(ierr);
      if (!work1) {
        ierr = VecDuplicate(qeval_ctx->vquad,&work1);CHKERRQ(ierr);
        ierr = PetscObjectCompose((PetscObject)ts,"_ts_quadwork_1",(PetscObject)work1);CHKERRQ(ierr);
        ierr = PetscObjectDereference((PetscObject)work1);CHKERRQ(ierr);
      }
      ierr = TSEvaluateObjective_MFixed(ts,tfup,U,design,work1,&has,work0);CHKERRQ(ierr);
      if (has) {
        ierr = VecAXPY(qeval_ctx->vquad,1.0,work0);CHKERRQ(ierr);
      }
    }
  } while (tfup < tf);

  /* restore */
  ierr = TSSetStepNumber(ts,tst);CHKERRQ(ierr);
  ierr = TSSetPostStep(ts,qeval_ctx->user);CHKERRQ(ierr);

  /* get back value */
  if (val) *val = qeval_ctx->squad;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEvaluateObjectiveAndGradient_Private(TS ts, Vec X, Vec design, Vec gradient, PetscReal *val)
{
  TSTrajectory   otrj = NULL;
  PetscReal      t0,tf,dt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (gradient) {
    ierr = VecSet(gradient,0.);CHKERRQ(ierr);
  }
  if (val) *val = 0.0;
  if (!ts->funchead) {
    PetscFunctionReturn(0);
  }
  if (!X) {
    ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
    if (!X) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Missing solution vector");
  }
  if (gradient) {
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
  }

  /* forward solve */
  ierr = TSEvaluateObjective_Private(ts,X,design,gradient,val);CHKERRQ(ierr);

  /* adjoint */
  if (gradient) {
    TS  adjts;
    Vec lambda,U;

    ierr = TSCreateAdjointTS(ts,&adjts);CHKERRQ(ierr);
    ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
    ierr = VecDuplicate(U,&lambda);CHKERRQ(ierr);
    ierr = TSSetSolution(adjts,lambda);CHKERRQ(ierr);
    ierr = TSGetTime(ts,&tf);CHKERRQ(ierr);
    ierr = TSGetPrevTime(ts,&dt);CHKERRQ(ierr);
    dt   = tf - dt;
    ierr = TSSetTimeStep(adjts,PetscMin(dt,tf-t0));CHKERRQ(ierr);
    ierr = AdjointTSSetTimeLimits(adjts,t0,tf);CHKERRQ(ierr);
    ierr = AdjointTSSetDesign(adjts,design);CHKERRQ(ierr);
    ierr = AdjointTSSetInitialGradient(adjts,gradient);CHKERRQ(ierr);
    ierr = AdjointTSComputeInitialConditions(adjts,t0,U,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
    ierr = AdjointTSEventHandler(adjts);CHKERRQ(ierr);
    ierr = TSSetFromOptions(adjts);CHKERRQ(ierr);
    ierr = AdjointTSSetTimeLimits(adjts,t0,tf);CHKERRQ(ierr);
    if (adjts->adapt) {
      PetscBool istr;

      ierr = PetscObjectTypeCompare((PetscObject)adjts->adapt,TSADAPTTRAJECTORY,&istr);CHKERRQ(ierr);
      ierr = TSAdaptTrajectorySetTrajectory(adjts->adapt,ts->trajectory,PETSC_TRUE);CHKERRQ(ierr);
      if (!istr) { /* indepently adapting the time step */
        ierr = TSSetMaxSteps(adjts,PETSC_MAX_INT);CHKERRQ(ierr);
        ierr = TSSetExactFinalTime(adjts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
      } else { /* follow trajectory -> fix number of time steps */
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
  }
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
  Mat            J_U = NULL,J_Udot = NULL,pJ_U = NULL,pJ_Udot = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A == B) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"A and B must be different matrices");
  if (pA == pB) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"pA and pB must be different matrices");
  ierr = TSGetApplicationContext(ts,(void*)&tlm_ctx);CHKERRQ(ierr);
  ierr = TSUpdateSplitJacobiansFromHistory(tlm_ctx->model,time,tlm_ctx->W[0],tlm_ctx->W[1]);CHKERRQ(ierr);
  ierr = TSGetSplitJacobians(tlm_ctx->model,&J_U,&pJ_U,&J_Udot,&pJ_Udot);CHKERRQ(ierr);
  if (A) { ierr = MatCopy(J_U,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  if (pA && pA != A) { ierr = MatCopy(pJ_U,pA,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  if (B) { ierr = MatCopy(J_Udot,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  if (pB && pB != B) { ierr = MatCopy(pJ_Udot,pB,SAME_NONZERO_PATTERN);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/* The TLM DAE is J_Udot * U_dot + J_U * U + f = 0, with f = dH/dm * deltam */
static PetscErrorCode TLMTSIFunctionLinear(TS lts, PetscReal time, Vec U, Vec Udot, Vec F, void *ctx)
{
  TLMTS_Ctx      *tlm_ctx;
  Mat            J_U = NULL, J_Udot = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(lts,(void*)&tlm_ctx);CHKERRQ(ierr);
  ierr = TSUpdateSplitJacobiansFromHistory(tlm_ctx->model,time,tlm_ctx->W[0],tlm_ctx->W[1]);CHKERRQ(ierr);
  ierr = TSGetSplitJacobians(tlm_ctx->model,&J_U,NULL,&J_Udot,NULL);CHKERRQ(ierr);
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
  TSProblemType  type;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr  = TSGetApplicationContext(lts,(void*)&tlm_ctx);CHKERRQ(ierr);
  model = tlm_ctx->model;
  ierr  = TSGetProblemType(model,&type);CHKERRQ(ierr);
  if (type > TS_LINEAR) {
    ierr = TSTrajectoryUpdateHistoryVecs(model->trajectory,model,time,tlm_ctx->W[0],tlm_ctx->W[1]);CHKERRQ(ierr);
  }
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
  ierr = TSComputeRHSJacobian(lts,time,U,lts->Arhs,lts->Brhs);CHKERRQ(ierr);
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
  TSProblemType  type;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(lts,(void*)&tlm_ctx);CHKERRQ(ierr);
  ierr = TSGetProblemType(tlm_ctx->model,&type);CHKERRQ(ierr);
  if (type > TS_LINEAR) {
    ierr = TSTrajectoryUpdateHistoryVecs(tlm_ctx->model->trajectory,tlm_ctx->model,time,tlm_ctx->W[0],NULL);CHKERRQ(ierr);
  }
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
    TSRHSJacobian rhsjacfunc;
    if (!rhsfunc) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"TSSetIFunction or TSSetRHSFunction not called");
    ierr = TSSetRHSFunction(*lts,NULL,TLMTSRHSFunctionLinear,NULL);CHKERRQ(ierr);
    ierr = TSGetRHSJacobian(ts,NULL,NULL,&rhsjacfunc,NULL);CHKERRQ(ierr);
    ierr = TSGetRHSMats_Private(ts,&A,&B);CHKERRQ(ierr);
    if (rhsjacfunc == TSComputeRHSJacobianConstant) {
      ierr = TSSetRHSJacobian(*lts,A,B,TSComputeRHSJacobianConstant,NULL);CHKERRQ(ierr);
    } else {
      ierr = TSSetRHSJacobian(*lts,A,B,TLMTSRHSJacobian,NULL);CHKERRQ(ierr);
    }
  }

  /* prefix */
  ierr = TSGetOptionsPrefix(ts,&prefix);CHKERRQ(ierr);
  ierr = TSSetOptionsPrefix(*lts,"tlm_");CHKERRQ(ierr);
  ierr = TSAppendOptionsPrefix(*lts,prefix);CHKERRQ(ierr);

  /* options specific to TLMTS */
  ierr = TSGetOptionsPrefix(*lts,&prefix);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)*lts),prefix,"Tangent Linear Model options","TS");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-userijacobian","Use the user-provided IJacobian routine, instead of the splits, to compute the Jacobian",NULL,tlm_ctx->userijac,&tlm_ctx->userijac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* the equation type is the same */
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
  PetscBool         istr;
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
  if (tlm->P) {
    ierr = MatMultTranspose(tlm->P,x,tlm->W[2]);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(x,tlm->W[2]);CHKERRQ(ierr);
  }
  ierr = AdjointTSComputeInitialConditions(prop->adjlts,prop->t0,tlm->W[2],PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSSetStepNumber(prop->adjlts,0);CHKERRQ(ierr);
  ierr = TSSetTime(prop->adjlts,prop->t0);CHKERRQ(ierr);
  ierr = TSHistoryGetTimeStep(prop->tj->tsh,PETSC_TRUE,0,&dt);CHKERRQ(ierr);
  ierr = TSSetTimeStep(prop->adjlts,dt);CHKERRQ(ierr);
  istr = PETSC_FALSE;
  if (prop->adjlts->adapt) {
    ierr = PetscObjectTypeCompare((PetscObject)prop->adjlts->adapt,TSADAPTTRAJECTORY,&istr);CHKERRQ(ierr);
    ierr = TSAdaptTrajectorySetTrajectory(prop->adjlts->adapt,prop->tj,PETSC_TRUE);CHKERRQ(ierr);
  }
  if (!istr) { /* if we don't follow the trajectory, we need to match the final time */
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
  PetscBool         istr;
  Vec               sol;
  TSTrajectory      otrj;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void **)&prop);CHKERRQ(ierr);
  otrj = prop->model->trajectory;
  prop->model->trajectory = prop->tj;
  istr = PETSC_FALSE;
  if (prop->lts->adapt) {
    ierr = PetscObjectTypeCompare((PetscObject)prop->lts->adapt,TSADAPTTRAJECTORY,&istr);CHKERRQ(ierr);
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
  ierr = TSLinearizeICApply(prop->lts,prop->t0,prop->x0,tlm->design,x,sol,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecScale(sol,-1.0);CHKERRQ(ierr);

  ierr = TSSetStepNumber(prop->lts,0);CHKERRQ(ierr);
  ierr = TSSetTime(prop->lts,prop->t0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(prop->lts,dt);CHKERRQ(ierr);
  ierr = TSSetMaxTime(prop->lts,prop->tf);CHKERRQ(ierr);
  if (istr) {
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

  /* Customize nonlinear model */
  ierr = TSSetStepNumber(prop->model,0);CHKERRQ(ierr);
  ierr = TSSetTime(prop->model,t0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(prop->model,dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(prop->model,PETSC_MAX_INT);CHKERRQ(ierr);
  ierr = TSSetMaxTime(prop->model,tf);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(prop->model,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

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
  if (!osol) {
    ierr = VecDuplicate(prop->x0,&osol);CHKERRQ(ierr);
    ierr = TSSetSolution(prop->model,osol);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)osol);CHKERRQ(ierr);
  }
  ierr = VecCopy(prop->x0,osol);CHKERRQ(ierr);
  ierr = TSSolve(prop->model,NULL);CHKERRQ(ierr);
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
  ierr = MatShellSetOperation(*A,MATOP_DESTROY,(void (*)())MatDestroy_Propagator);CHKERRQ(ierr);
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
  ierr = TSSetObjective(prop->lts,prop->tf,NULL,NULL,TLMTS_dummyRHS,NULL,NULL,NULL,
                        NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (prop->model->F_m) {
    ierr = TSSetGradientDAE(prop->lts,prop->model->F_m,prop->model->F_m_f,prop->model->F_m_ctx);CHKERRQ(ierr);
  }
  if (prop->model->G_m) {
    ierr = TSSetGradientIC(prop->lts,prop->model->G_x,prop->model->G_m,prop->model->Ggrad,prop->model->Ggrad_ctx);CHKERRQ(ierr);
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
    ierr = TSSetGradientIC(prop->lts,NULL,G_m,NULL,NULL);CHKERRQ(ierr);
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

.seealso: TS, MATSHELL, TSSetGradientDAE(), TSSetGradientIC()
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

.seealso: TSSetGradientDAE(), TSSetGradientIC()
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
    ierr = MatDestroy(&olink->f_XX);CHKERRQ(ierr);
    ierr = MatDestroy(&olink->f_XM);CHKERRQ(ierr);
    ierr = MatDestroy(&olink->f_MM);CHKERRQ(ierr);
    ierr = PetscFree(olink);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
   TSSetObjective - Sets a cost functional callback together with its gradient and hessian.

   Logically Collective on TS

   Input Parameters:
+  ts       - the TS context obtained from TSCreate()
.  fixtime  - the time at which the functional has to be evaluated (use PETSC_MIN_REAL for integrands)
.  f        - the function evaluation routine
.  f_ctx    - user-defined context for the function evaluation routine (can be NULL)
.  f_x      - the function evaluation routine for the derivativrt to the state variables (can be NULL)
.  f_x_ctx  - user-defined context for the function evaluation routine (can be NULL)
.  f_m      - the function evaluation routine for the derivative wrt the design variables (can be NULL)
.  f_m_ctx  - user-defined context for the function evaluation routine (can be NULL)
.  f_XX     - the Mat object to hold f_xx(x,m,t) (can be NULL)
.  f_xx     - the function evaluation routine for the second derivative wrt the state variables (can be NULL)
.  f_xx_ctx - user-defined context for the matrix evaluation routine (can be NULL)
.  f_XM     - the Mat object to hold f_xm(x,m,t) (can be NULL)
.  f_xm     - the function evaluation routine for the mixed derivative (can be NULL)
.  f_xm_ctx - user-defined context for the matrix evaluation routine (can be NULL)
.  f_MM     - the Mat object to hold f_mm(x,m,t) (can be NULL)
.  f_mm     - the function evaluation routine for the second derivative wrt the design variables (can be NULL)
-  f_mm_ctx - user-defined context for the matrix evaluation routine (can be NULL)

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

   Calling sequence of f_xx, f_xm and f_mm:
$  f(TS ts,PetscReal t,Vec u,Vec m,Mat A,void *ctx);

+  t   - time at step/stage being solved
.  u   - state vector
.  m   - design vector
.  A   - the output matrix
-  ctx - [optional] user-defined context

   Notes: the functions passed in are appended to a list. More functions can be passed by simply calling TSSetObjective multiple times.
          The functionals are intendended to be used as integrand terms of a time integration (if fixtime == PETSC_MIN_REAL) or as evaluation at a given specific time.
          Regularizers fall into the latter category: use f_x = NULL, and pass f and f_m with any time in between the half-open interval (t0, tf] (i.e. start and end of the forward solve).
          For f_x, the size of the output vector equals the size of the state vector; for f_m it equals the size of the design vector.
          The hessian matrices do not need to be in assembled form, just the MatMult() action is needed. If f_XM is present, the action of f_MX is obtained by calling MatMultTranspose().
          If any of the second derivative matrices are constant, the associated function pointers can be NULL.

   Level: developer

.seealso: TSSetGradientDAE(), TSEvaluateObjectiveAndGradient(), TSSetGradientIC()
*/
PetscErrorCode TSSetObjective(TS ts, PetscReal fixtime, TSEvalObjective f, void* f_ctx,
                              TSEvalObjectiveGradient f_x, void* f_x_ctx, TSEvalObjectiveGradient f_m, void* f_m_ctx,
                              Mat f_XX, TSEvalObjectiveHessian f_xx, void* f_xx_ctx,
                              Mat f_XM, TSEvalObjectiveHessian f_xm, void* f_xm_ctx,
                              Mat f_MM, TSEvalObjectiveHessian f_mm, void* f_mm_ctx)
{
  ObjectiveLink  link;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,fixtime,2);
  if (f_XX) PetscValidHeaderSpecific(f_XX,MAT_CLASSID,9);
  if (f_XM) PetscValidHeaderSpecific(f_XM,MAT_CLASSID,12);
  if (f_MM) PetscValidHeaderSpecific(f_MM,MAT_CLASSID,15);
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
  if (f_XX) {
    ierr = PetscObjectReference((PetscObject)f_XX);CHKERRQ(ierr);
    link->f_XX      = f_XX;
    link->f_xx      = f_xx;
    link->f_xx_ctx  = f_xx_ctx;
  }
  if (f_XM) {
    ierr = PetscObjectReference((PetscObject)f_XM);CHKERRQ(ierr);
    link->f_XM      = f_XM;
    link->f_xm      = f_xm;
    link->f_xm_ctx  = f_xm_ctx;
  }
  if (f_MM) {
    ierr = PetscObjectReference((PetscObject)f_MM);CHKERRQ(ierr);
    link->f_MM      = f_MM;
    link->f_mm      = f_mm;
    link->f_mm_ctx  = f_mm_ctx;
  }
  link->fixedtime = fixtime;
  PetscFunctionReturn(0);
}

/*
   TSSetGradientDAE - Sets the function for the evaluation of F_m(t,x(t),x_t(t);m).

   Logically Collective on TS

   Input Parameters:
+  ts      - the TS context obtained from TSCreate()
.  J       - the Mat object to hold F_m(t,x(t),x_t(t);m)
.  f       - the function evaluation routine
-  f_ctx   - user-defined context for the function evaluation routine (can be NULL)

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

.seealso: TSSetObjective(), TSEvaluateObjectiveAndGradient(), TSSetGradientIC(), TSCreatePropagatorMat(), MATSHELL, MATPROPAGATOR
*/
PetscErrorCode TSSetGradientDAE(TS ts, Mat J, TSEvalGradientDAE f, void *ctx)
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
   TSSetGradientIC - Sets the callback function to compute the matrices g_x(x0,m) and g_m(x0,m), if there is any dependence of the DAE initial conditions from the design parameters.

   Logically Collective on TS

   Input Parameters:
+  ts      - the TS context obtained from TSCreate()
.  J_x     - the Mat object to hold g_x(x0,m) (optional, if NULL identity is assumed)
.  J_m     - the Mat object to hold g_m(x0,m)
.  f       - the function evaluation routine
-  f_ctx   - user-defined context for the function evaluation routine (can be NULL)

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

.seealso: TSSetObjective(), TSSetGradientDAE(), TSEvaluateObjectiveAndGradient(), MATSHELL, MatMultTranspose()
*/
PetscErrorCode TSSetGradientIC(TS ts, Mat J_x, Mat J_m, TSEvalGradientIC f, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (J_x) PetscValidHeaderSpecific(J_x,MAT_CLASSID,2);
  if (J_m) PetscValidHeaderSpecific(J_m,MAT_CLASSID,3);

  ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradient_G",NULL);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradient_GW",NULL);CHKERRQ(ierr);
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
   TSEvaluateObjectiveAndGradient - Evaluates the objective functions set with TSSetObjective and the gradient.

   Logically Collective on TS

   Input Parameters:
+  ts     - the TS context
.  t0     - initial time
.  dt     - initial time step
.  tf     - final time
.  X      - the initial vector for the state (can be NULL)
-  design - current design vector

   Output Parameters:
+  gradient - the computed gradient (can be NULL)
-  obj      - the value of the objective function (can be NULL)

   Notes: If gradient is NULL, just a forward solve will be performed to compute the objective function. Otherwise, forward and backward solves are performed.
          The dt argument is ignored when smaller or equal to zero. If X is NULL, the initial state is given by the TS solution vector.

   Level: developer

.seealso: TSSetObjective(), TSSetGradientDAE(), TSSetGradientIC(), TSSetSolution()
*/
PetscErrorCode TSEvaluateObjectiveAndGradient(TS ts, PetscReal t0, PetscReal dt, PetscReal tf, Vec X, Vec design, Vec gradient, PetscReal *obj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,t0,2);
  PetscValidLogicalCollectiveReal(ts,dt,3);
  PetscValidLogicalCollectiveReal(ts,tf,4);
  if (X) PetscValidHeaderSpecific(X,VEC_CLASSID,5);
  PetscValidHeaderSpecific(design,VEC_CLASSID,6);
  if (gradient) PetscValidHeaderSpecific(gradient,VEC_CLASSID,7);
  if (obj) PetscValidPointer(obj,8);
  if (!gradient && !obj) PetscFunctionReturn(0);

  ierr = TSSetTime(ts,t0);CHKERRQ(ierr);
  if (dt > 0) {
    ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  }
  ierr = TSSetMaxTime(ts,tf);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,PETSC_MAX_INT);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = VecLockPush(design);CHKERRQ(ierr);
  ierr = TSEvaluateObjectiveAndGradient_Private(ts,X,design,gradient,obj);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

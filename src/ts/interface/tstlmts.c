/* ------------------ Routines for the TS representing the tangent linear model, namespaced by TLMTS ----------------------- */
#include <petsc/private/tstlmtsimpl.h>
#include <petsc/private/tssplitjacimpl.h>
#include <petsc/private/tshistoryimpl.h>
#include <petscdm.h>

typedef struct {
  TS        model;
  PetscBool userijac;
  Vec       workrhs;
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
  ierr = VecDestroy(&tlm->workrhs);CHKERRQ(ierr);
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
  ierr = TSUpdateSplitJacobiansFromHistory(tlm_ctx->model,time);CHKERRQ(ierr);
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
  ierr = TSUpdateSplitJacobiansFromHistory(tlm_ctx->model,time);CHKERRQ(ierr);
  ierr = TSGetSplitJacobians(tlm_ctx->model,&J_U,NULL,&J_Udot,NULL);CHKERRQ(ierr);
  ierr = MatMult(J_U,U,F);CHKERRQ(ierr);
  ierr = MatMultAdd(J_Udot,Udot,F,F);CHKERRQ(ierr);
  if (tlm_ctx->model->F_m) {
    TS ts = tlm_ctx->model;
    if (ts->F_m_f) { /* non constant dependence */
      Vec W[2];

      ierr = TSTrajectoryGetUpdatedHistoryVecs(ts->trajectory,ts,time,&W[0],&W[1]);CHKERRQ(ierr);
      ierr = (*ts->F_m_f)(ts,time,W[0],W[1],tlm_ctx->design,ts->F_m,ts->F_m_ctx);CHKERRQ(ierr);
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(ts->trajectory,&W[0],&W[1]);CHKERRQ(ierr);
      ierr = MatMult(ts->F_m,tlm_ctx->mdelta,tlm_ctx->workrhs);CHKERRQ(ierr);
    }
    ierr = VecAXPY(F,1.0,tlm_ctx->workrhs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TLMTSIJacobian(TS lts, PetscReal time, Vec U, Vec Udot, PetscReal shift, Mat A, Mat B, void *ctx)
{
  TLMTS_Ctx      *tlm_ctx;
  TS             model;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr  = TSGetApplicationContext(lts,(void*)&tlm_ctx);CHKERRQ(ierr);
  model = tlm_ctx->model;
  if (tlm_ctx->userijac) {
    Vec W[2];

    ierr = TSTrajectoryGetUpdatedHistoryVecs(model->trajectory,model,time,&W[0],&W[1]);CHKERRQ(ierr);
    ierr = TSComputeIJacobian(model,time,W[0],W[1],shift,A,B,PETSC_FALSE);CHKERRQ(ierr);
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(model->trajectory,&W[0],&W[1]);CHKERRQ(ierr);
  } else {
    ierr = TSComputeIJacobianWithSplits(model,time,NULL,NULL,shift,A,B,ctx);CHKERRQ(ierr);
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
  /* force recomputation of RHS Jacobian */
  tlm_ctx->model->rhsjacobian.time = PETSC_MIN_REAL;
  ierr = TSComputeRHSJacobian(lts,time,U,lts->Arhs,lts->Brhs);CHKERRQ(ierr);
  ierr = MatMult(lts->Arhs,U,F);CHKERRQ(ierr);
  if (tlm_ctx->model->F_m) {
    TS ts = tlm_ctx->model;
    if (ts->F_m_f) { /* non constant dependence */
      Vec W[2];

      ierr = TSTrajectoryGetUpdatedHistoryVecs(ts->trajectory,ts,time,&W[0],&W[1]);CHKERRQ(ierr);
      ierr = (*ts->F_m_f)(ts,time,W[0],W[1],tlm_ctx->design,ts->F_m,ts->F_m_ctx);CHKERRQ(ierr);
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(ts->trajectory,&W[0],&W[1]);CHKERRQ(ierr);
      ierr = MatMult(ts->F_m,tlm_ctx->mdelta,tlm_ctx->workrhs);CHKERRQ(ierr);
    }
    ierr = VecAXPY(F,-1.0,tlm_ctx->workrhs);CHKERRQ(ierr);
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
    ierr = TSTrajectoryGetUpdatedHistoryVecs(tlm_ctx->model->trajectory,tlm_ctx->model,time,&U,NULL);CHKERRQ(ierr);
  }
  /* force recomputation of RHS Jacobian: this is needed because this function can be called from within an adjoint solver */
  if (lts->rhsjacobian.time == PETSC_MIN_REAL) tlm_ctx->model->rhsjacobian.time = PETSC_MIN_REAL;
  ierr = TSComputeRHSJacobian(tlm_ctx->model,time,U,A,P);CHKERRQ(ierr);
  if (type > TS_LINEAR) {
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tlm_ctx->model->trajectory,&U,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TLMTSGetRHSVec(TS lts, Vec *rhs)
{
  PetscContainer c;
  TLMTS_Ctx      *tlm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lts,TS_CLASSID,1);
  PetscValidPointer(rhs,2);
  ierr = PetscObjectQuery((PetscObject)lts,"_ts_tlm_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)lts),PETSC_ERR_PLIB,"Missing tlm container");
  ierr = PetscContainerGetPointer(c,(void**)&tlm);CHKERRQ(ierr);
  if (!tlm->workrhs) {
    Vec U;

    ierr = TSGetSolution(lts,&U);CHKERRQ(ierr);
    ierr = VecDuplicate(U,&tlm->workrhs);CHKERRQ(ierr);
  }
  *rhs = tlm->workrhs;
  PetscFunctionReturn(0);
}

PetscErrorCode TLMTSSetPerturbationVec(TS lts, Vec mdelta)
{
  PetscContainer c;
  TLMTS_Ctx      *tlm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lts,TS_CLASSID,1);
  PetscValidHeaderSpecific(mdelta,VEC_CLASSID,2);
  ierr = PetscObjectQuery((PetscObject)lts,"_ts_tlm_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)lts),PETSC_ERR_PLIB,"Missing tlm container");
  ierr = PetscContainerGetPointer(c,(void**)&tlm);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)mdelta);CHKERRQ(ierr);
  ierr = VecDestroy(&tlm->mdelta);CHKERRQ(ierr);
  tlm->mdelta = mdelta;
  PetscFunctionReturn(0);
}

PetscErrorCode TLMTSSetDesignVec(TS lts, Vec design)
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

PetscErrorCode TLMTSGetDesignVec(TS lts, Vec* design)
{
  PetscContainer c;
  TLMTS_Ctx      *tlm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lts,TS_CLASSID,1);
  PetscValidPointer(design,2);
  ierr = PetscObjectQuery((PetscObject)lts,"_ts_tlm_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)lts),PETSC_ERR_PLIB,"Missing tlm container");
  ierr = PetscContainerGetPointer(c,(void**)&tlm);CHKERRQ(ierr);
  *design = tlm->design;
  PetscFunctionReturn(0);
}

PetscErrorCode TLMTSGetModelTS(TS lts, TS* ts)
{
  PetscContainer c;
  TLMTS_Ctx      *tlm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(lts,TS_CLASSID,1);
  PetscValidPointer(ts,2);
  ierr = PetscObjectQuery((PetscObject)lts,"_ts_tlm_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)lts),PETSC_ERR_PLIB,"Missing tlm container");
  ierr = PetscContainerGetPointer(c,(void**)&tlm);CHKERRQ(ierr);
  *ts = tlm->model;
  PetscFunctionReturn(0);
}

/* creates the TS for the tangent linear model */
PetscErrorCode TSCreateTLMTS(TS ts, TS* lts)
{
  SNES             snes;
  KSP              ksp;
  Mat              A,B;
  Vec              vatol,vrtol;
  PetscContainer   container;
  TLMTS_Ctx        *tlm_ctx;
  TSIFunction      ifunc;
  TSRHSFunction    rhsfunc;
  TSI2Function     i2func;
  TSType           type;
  TSEquationType   eqtype;
  const char       *prefix;
  PetscReal        atol,rtol;
  PetscInt         maxits;
  PetscBool        jcon,rksp;
  PetscErrorCode   ierr;

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
  if (ts->adapt) {
    ierr = TSAdaptCreate(PetscObjectComm((PetscObject)*lts),&(*lts)->adapt);CHKERRQ(ierr);
    ierr = TSAdaptSetType((*lts)->adapt,((PetscObject)ts->adapt)->type_name);CHKERRQ(ierr);
  }
  ierr = PetscObjectComposeFunction((PetscObject)(*lts),"TSComputeSplitJacobians_C",TLMTSComputeSplitJacobians);CHKERRQ(ierr);

  ierr = PetscNew(&tlm_ctx);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(*lts,(void *)tlm_ctx);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)ts);CHKERRQ(ierr);
  tlm_ctx->model = ts;

  /* wrap application context in a container, so that it will be destroyed when calling TSDestroy on lts */
  ierr = PetscContainerCreate(PetscObjectComm((PetscObject)(*lts)),&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,tlm_ctx);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(container,TLMTSDestroy_Private);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)(*lts),"_ts_tlm_ctx",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);

  /* TLMTS prefix: i.e. options called as -tlm_ts_monitor or -tlm_modelprefix_ts_monitor */
  ierr = TSGetOptionsPrefix(ts,&prefix);CHKERRQ(ierr);
  ierr = TSSetOptionsPrefix(*lts,"tlm_");CHKERRQ(ierr);
  ierr = TSAppendOptionsPrefix(*lts,prefix);CHKERRQ(ierr);

  /* options specific to TLMTS */
  ierr = TSGetOptionsPrefix(*lts,&prefix);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)*lts),prefix,"Tangent Linear Model options","TS");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-userijacobian","Use the user-provided IJacobian routine, instead of the splits, to compute the Jacobian",NULL,tlm_ctx->userijac,&tlm_ctx->userijac,NULL);CHKERRQ(ierr);
  jcon = PETSC_FALSE;
  ierr = PetscOptionsBool("-constjacobians","Whether or not the DAE Jacobians are constant",NULL,jcon,&jcon,NULL);CHKERRQ(ierr);
  rksp = PETSC_FALSE;
  ierr = PetscOptionsBool("-reuseksp","Reuse the KSP solver from the nonlinear model",NULL,rksp,&rksp,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* setup callbacks for the tangent linear model DAE: we reuse the same jacobian matrices of the forward model */
  ierr = TSGetIFunction(ts,NULL,&ifunc,NULL);CHKERRQ(ierr);
  ierr = TSGetRHSFunction(ts,NULL,&rhsfunc,NULL);CHKERRQ(ierr);
  if (ifunc) {
    TSSplitJacobians *splitJ;

    ierr = TSGetIJacobian(ts,&A,&B,NULL,NULL);CHKERRQ(ierr);
    ierr = TSSetIFunction(*lts,NULL,TLMTSIFunctionLinear,NULL);CHKERRQ(ierr);
    ierr = TSSetIJacobian(*lts,A,B,TLMTSIJacobian,NULL);CHKERRQ(ierr);
    /* caching to prevent from recomputation of Jacobians */
    ierr = PetscObjectQuery((PetscObject)ts,"_ts_splitJac",(PetscObject*)&container);CHKERRQ(ierr);
    if (container) {
      ierr = PetscContainerGetPointer(container,(void**)&splitJ);CHKERRQ(ierr);
    } else {
      ierr = PetscNew(&splitJ);CHKERRQ(ierr);
      splitJ->Astate = -1;
      splitJ->Aid    = PETSC_MIN_INT;
      splitJ->shift  = PETSC_MIN_REAL;
      ierr = PetscContainerCreate(PetscObjectComm((PetscObject)ts),&container);CHKERRQ(ierr);
      ierr = PetscContainerSetPointer(container,splitJ);CHKERRQ(ierr);
      ierr = PetscContainerSetUserDestroy(container,TSSplitJacobiansDestroy_Private);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)ts,"_ts_splitJac",(PetscObject)container);CHKERRQ(ierr);
      /* we can setup an AdjointTS from a TLMTS -> propagate splitJac to save memory */
      ierr = PetscObjectCompose((PetscObject)(*lts),"_ts_splitJac",(PetscObject)container);CHKERRQ(ierr);
      ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
    }
    splitJ->splitdone = PETSC_FALSE;
    splitJ->jacconsts = jcon;
  } else {
    TSRHSJacobian rhsjacfunc;

    if (!rhsfunc) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"TSSetIFunction or TSSetRHSFunction not called");
    ierr = TSSetRHSFunction(*lts,NULL,TLMTSRHSFunctionLinear,NULL);CHKERRQ(ierr);
    ierr = TSGetRHSJacobian(ts,NULL,NULL,&rhsjacfunc,NULL);CHKERRQ(ierr);
    ierr = TSGetRHSMats_Private(ts,&A,&B);CHKERRQ(ierr);
    if (rhsjacfunc == TSComputeRHSJacobianConstant) {
      ierr = TSSetRHSJacobian(*lts,A,B,TSComputeRHSJacobianConstant,NULL);CHKERRQ(ierr);
    } else if (jcon) { /* just to make sure we have a correct Jacobian */
      DM  dm;
      Vec U;

      ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dm,&U);CHKERRQ(ierr);
      ierr = TSComputeRHSJacobian(ts,PETSC_MIN_REAL,U,A,B);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dm,&U);CHKERRQ(ierr);
      ierr = TSSetRHSJacobian(*lts,A,B,TSComputeRHSJacobianConstant,NULL);CHKERRQ(ierr);
    } else {
      ierr = TSSetRHSJacobian(*lts,A,B,TLMTSRHSJacobian,NULL);CHKERRQ(ierr);
    }
  }

  /* the equation type is the same */
  ierr = TSGetEquationType(ts,&eqtype);CHKERRQ(ierr);
  ierr = TSSetEquationType(*lts,eqtype);CHKERRQ(ierr);

  /* tangent linear model DAE is linear */
  ierr = TSSetProblemType(*lts,TS_LINEAR);CHKERRQ(ierr);

  /* tangent linear model linear solver */
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  if (!rksp) { /* propagate KSP info of the forward model but use a different object */
    KSPType   ksptype;
    PetscReal atol,rtol,dtol;

    ierr = KSPGetType(ksp,&ksptype);CHKERRQ(ierr);
    ierr = KSPGetTolerances(ksp,&rtol,&atol,&dtol,&maxits);CHKERRQ(ierr);
    ierr = TSGetSNES(*lts,&snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    if (ksptype) { ierr = KSPSetType(ksp,ksptype);CHKERRQ(ierr); }
    ierr = KSPSetTolerances(ksp,rtol,atol,dtol,maxits);CHKERRQ(ierr);
  } else { /* reuse the same KSP */
    ierr = TSGetSNES(*lts,&snes);CHKERRQ(ierr);
    ierr = SNESSetKSP(snes,ksp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

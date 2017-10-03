/*
   This code is very much inspired to the papers
   [1] Cao, Li, Petzold. Adjoint sensitivity analysis for differential-algebraic equations: algorithms and software, JCAM 149, 2002.
   [2] Cao, Li, Petzold. Adjoint sensitivity analysis for differential-algebraic equations: the adjoint DAE system and its numerical solution, SISC 24, 2003.
   TODO: register citations
   TODO: add custom fortran wrappers
*/
#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petsc/private/tsobjimpl.h>
#include <petsc/private/tssplitjacimpl.h>
#include <petsc/private/snesimpl.h>
#include <petsc/private/kspimpl.h>
#include <petscdm.h>

/*
   Apply "Jacobians" of initial conditions
   if transpose is false : y =   G_x^-1 G_m x or G_x^-1 x if G_m == 0
   if transpose is true  : y = G_m^T G_x^-T x or G_x^-T x if G_m == 0
   (x0,design) are the variables one needs to linearize against to get the partial Jacobians G_x and G_m
   The case for useGm == PETSC_FALSE is a hack to reuse the same code for the second-order adjoint
*/
static PetscErrorCode TSLinearizedICApply(TS ts, PetscReal t0, Vec x0, Vec design, Vec x, Vec y, PetscBool transpose, PetscBool useGm)
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
  if (useGm && !ts->G_m) {
    ierr = VecSet(y,0.);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (ts->Ggrad) {
    ierr = (*ts->Ggrad)(ts,t0,x0,design,ts->G_x,ts->G_m,ts->Ggrad_ctx);CHKERRQ(ierr);
  }
  if (ts->G_x) { /* this is optional. If not provided, identity is assumed */
    ierr = PetscObjectQuery((PetscObject)ts,"_ts_gradientIC_G",(PetscObject*)&ksp);CHKERRQ(ierr);
    if (!ksp) {
      const char *prefix;
      ierr = KSPCreate(PetscObjectComm((PetscObject)ts),&ksp);CHKERRQ(ierr);
      ierr = KSPSetTolerances(ksp,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
      ierr = TSGetOptionsPrefix(ts,&prefix);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(ksp,prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ksp,"jactsic_");CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradientIC_G",(PetscObject)ksp);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)ksp);CHKERRQ(ierr);
    }
    ierr = KSPSetOperators(ksp,ts->G_x,ts->G_x);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)ts,"_ts_gradientIC_GW",(PetscObject*)&workvec);CHKERRQ(ierr);
    if (!workvec) {
      ierr = MatCreateVecs(ts->G_x,&workvec,NULL);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradientIC_GW",(PetscObject)workvec);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)workvec);CHKERRQ(ierr);
    }
  }
  if (transpose) {
    if (ksp) {
      if (useGm) {
        ierr = KSPSolveTranspose(ksp,x,workvec);CHKERRQ(ierr);
        ierr = MatMultTranspose(ts->G_m,workvec,y);CHKERRQ(ierr);
      } else {
        ierr = KSPSolveTranspose(ksp,x,y);CHKERRQ(ierr);
      }
    } else {
      if (useGm) {
        ierr = MatMultTranspose(ts->G_m,x,y);CHKERRQ(ierr);
      } else {
        ierr = VecCopy(x,y);CHKERRQ(ierr);
      }
    }
  } else {
    if (ksp) {
      if (useGm) {
        ierr = MatMult(ts->G_m,x,workvec);CHKERRQ(ierr);
        ierr = KSPSolve(ksp,workvec,y);CHKERRQ(ierr);
      } else {
        ierr = KSPSolve(ksp,x,y);CHKERRQ(ierr);
      }
    } else {
      if (useGm) {
        ierr = MatMult(ts->G_m,x,y);CHKERRQ(ierr);
      } else {
        ierr = VecCopy(x,y);CHKERRQ(ierr);
      }
    }
  }
  if (ksp) { /* destroy inner vectors to avoid ABA issues when destroying the DM */
    ierr = VecDestroy(&ksp->vec_rhs);CHKERRQ(ierr);
    ierr = VecDestroy(&ksp->vec_sol);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------ Wrappers for quadrature evaluation ----------------------- */

/* prototypes for cost integral evaluation */
typedef PetscErrorCode (*SQuadEval)(TSObj,Vec,PetscReal,PetscReal*,void*);
typedef PetscErrorCode (*VQuadEval)(TSObj,Vec,PetscReal,Vec,void*);

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
  Vec            *wquad;      /* quadrature work vectors used by the trapezoidal rule + 3 extra work vectors */
  PetscInt       cur,old;     /* pointers to current and old wquad vectors for trapezoidal rule */
} TSQuadratureCtx;

static PetscErrorCode TSQuadratureCtxDestroy_Private(void *ptr)
{
  TSQuadratureCtx* q = (TSQuadratureCtx*)ptr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(5,&q->wquad);CHKERRQ(ierr);
  ierr = PetscFree(q->veval_ctx);CHKERRQ(ierr);
  ierr = PetscFree(q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  XXX_FWD are evaluated during the forward run
  XXX_TLM are evaluated during the tangent linear model run within Hessian computations
  XXX_ADJ are evaluated during the adjoint run
*/
static PetscErrorCode EvalQuadObj_FWD(TSObj link, Vec U, PetscReal t, PetscReal *f, void* ctx)
{
  Vec            design = (Vec)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSObjEval(link,U,design,t,f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalQuadObjFixed_FWD(TSObj link, Vec U, PetscReal t, PetscReal *f, void* ctx)
{
  Vec            design = (Vec)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSObjEvalFixed(link,U,design,t,f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalQuadIntegrand_FWD(TSObj link, Vec U, PetscReal t, Vec F, void* ctx)
{
  Vec            *v = (Vec*)ctx;
  Vec            design = v[0], work = v[1];
  PetscBool      has_m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSObjEval_M(link,U,design,t,work,&has_m,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalQuadIntegrandFixed_FWD(TSObj link,Vec U, PetscReal t, Vec F, void* ctx)
{
  Vec            *v = (Vec*)ctx;
  Vec            design = v[0], work = v[1];
  PetscBool      has_m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSObjEvalFixed_M(link,U,design,t,work,&has_m,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  TS        fwdts;
  TS        adjts;
  TS        tlmts;
  PetscReal t0,tf;
  Vec       design;
  Vec       direction;
  Vec       work1;
  Vec       work2;
} TLMEvalQuadCtx;

/* computes d^2 f / dp^2 direction + d^2 f / dp dx U + (L^T \otimes I_M)(H_MM direction + H_MU U + H_MUdot Udot) durintg TLM runs */
static PetscErrorCode EvalQuadIntegrand_TLM(TSObj link, Vec U, PetscReal t, Vec F, void* ctx)
{
  TLMEvalQuadCtx *q = (TLMEvalQuadCtx*)ctx;
  TS             fwdts = q->fwdts;
  TS             adjts = q->adjts;
  TS             tlmts = q->tlmts;
  Vec            FWDH[2],FOAH;
  PetscReal      adjt  = q->tf - t + q->t0;
  PetscBool      AXPY;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,t,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
  ierr = TSObjEval_MU(fwdts->funchead,FWDH[0],q->design,t,U,q->work1,&AXPY,F);CHKERRQ(ierr);
  if (!AXPY) {
    ierr = TSObjEval_MM(fwdts->funchead,FWDH[0],q->design,t,q->direction,q->work1,&AXPY,F);CHKERRQ(ierr);
  } else {
    PetscBool has;

    ierr = TSObjEval_MM(fwdts->funchead,FWDH[0],q->design,t,q->direction,q->work1,&has,q->work2);CHKERRQ(ierr);
    if (has) {
      ierr = VecAXPY(F,1.0,q->work2);CHKERRQ(ierr);
    }
  }
  if (fwdts->HF[2][0] || fwdts->HF[2][1] || fwdts->HF[2][2]) {
    ierr = TSTrajectoryGetUpdatedHistoryVecs(adjts->trajectory,adjts,adjt,&FOAH,NULL);CHKERRQ(ierr);
  }
  if (fwdts->HF[2][2]) { /* (L^T \otimes I_M) H_MM direction */
    if (AXPY) {
      ierr = (*fwdts->HF[2][2])(fwdts,t,FWDH[0],FWDH[1],q->design,FOAH,q->direction,q->work1,fwdts->HFctx);CHKERRQ(ierr);
      ierr = VecAXPY(F,1.0,q->work1);CHKERRQ(ierr);
    } else {
      ierr = (*fwdts->HF[2][2])(fwdts,t,FWDH[0],FWDH[1],q->design,FOAH,q->direction,F,fwdts->HFctx);CHKERRQ(ierr);
      AXPY = PETSC_TRUE;
    }
  }
  if (fwdts->HF[2][0]) { /* (L^T \otimes I_M) H_MX \eta, \eta (=U) the TLM solution */
    if (AXPY) {
      ierr = (*fwdts->HF[2][0])(fwdts,t,FWDH[0],FWDH[1],q->design,FOAH,U,q->work1,fwdts->HFctx);CHKERRQ(ierr);
      ierr = VecAXPY(F,1.0,q->work1);CHKERRQ(ierr);
    } else {
      ierr = (*fwdts->HF[2][0])(fwdts,t,FWDH[0],FWDH[1],q->design,FOAH,U,F,fwdts->HFctx);CHKERRQ(ierr);
      AXPY = PETSC_TRUE;
    }
  }
  if (fwdts->HF[2][1]) { /* (L^T \otimes I_M) H_MXdot \etadot, \eta the TLM solution */
    Vec TLMHdot;

    ierr = TSTrajectoryGetUpdatedHistoryVecs(tlmts->trajectory,tlmts,t,NULL,&TLMHdot);CHKERRQ(ierr);
    if (AXPY) {
      ierr = (*fwdts->HF[2][1])(fwdts,t,FWDH[0],FWDH[1],q->design,FOAH,TLMHdot,q->work1,fwdts->HFctx);CHKERRQ(ierr);
      ierr = VecAXPY(F,1.0,q->work1);CHKERRQ(ierr);
    } else {
      ierr = (*fwdts->HF[2][1])(fwdts,t,FWDH[0],FWDH[1],q->design,FOAH,TLMHdot,F,fwdts->HFctx);CHKERRQ(ierr);
    }
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tlmts->trajectory,NULL,&TLMHdot);CHKERRQ(ierr);
  }
  ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
  if (fwdts->HF[2][0] || fwdts->HF[2][1] || fwdts->HF[2][2]) {
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(adjts->trajectory,&FOAH,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EvalQuadIntegrandFixed_TLM(TSObj link, Vec U, PetscReal t, Vec F, void* ctx)
{
  TLMEvalQuadCtx *q = (TLMEvalQuadCtx*)ctx;
  TS             fwdts = q->fwdts;
  PetscBool      has;
  Vec            FWDH;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,t,&FWDH,NULL);CHKERRQ(ierr);
  ierr = TSObjEvalFixed_MU(fwdts->funchead,FWDH,q->design,t,U,q->work1,&has,F);CHKERRQ(ierr);
  if (!has) {
    ierr = TSObjEvalFixed_MM(fwdts->funchead,FWDH,q->design,t,q->direction,q->work1,&has,F);CHKERRQ(ierr);
    if (!has) SETERRQ(PetscObjectComm((PetscObject)fwdts),PETSC_ERR_PLIB,"Point-form functionals not present");
  } else {
    ierr = TSObjEvalFixed_MM(fwdts->funchead,FWDH,q->design,t,q->direction,q->work1,&has,q->work2);CHKERRQ(ierr);
    if (has) {
      ierr = VecAXPY(F,1.0,q->work2);CHKERRQ(ierr);
    }
  }
  ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&FWDH,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  TS        fwdts;
  Vec       design;
  PetscReal t0,tf;
} AdjEvalQuadCtx;

static PetscErrorCode EvalQuadIntegrand_ADJ(TSObj link, Vec L, PetscReal t, Vec F, void* ctx)
{
  AdjEvalQuadCtx *q = (AdjEvalQuadCtx*)ctx;
  TS             fwdts = q->fwdts;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (fwdts->F_m_f) { /* non constant dependence */
    Vec       W[2];
    PetscReal fwdt = q->tf - t + q->t0;

    ierr = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,fwdt,&W[0],&W[1]);CHKERRQ(ierr);
    ierr = (*fwdts->F_m_f)(fwdts,fwdt,W[0],W[1],q->design,fwdts->F_m,fwdts->F_m_ctx);CHKERRQ(ierr);
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&W[0],&W[1]);CHKERRQ(ierr);
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
    ierr = (*qeval_ctx->seval)(ts->funchead,solution,time,&squad,qeval_ctx->seval_ctx);CHKERRQ(ierr);
    PetscStackPop;
    qeval_ctx->squad += dt*(squad+qeval_ctx->psquad)/2.0;
    qeval_ctx->psquad = squad;
  }

  /* scalar quadrature (qeval_ctx->wquad[qeval_ctx->old] have been initialized with the first function evaluation) */
  if (qeval_ctx->veval) {
    PetscScalar t[2];
    PetscInt    tmp;

    PetscStackPush("TS vector quadrature function");
    ierr = (*qeval_ctx->veval)(ts->funchead,solution,time,qeval_ctx->wquad[qeval_ctx->cur],qeval_ctx->veval_ctx);CHKERRQ(ierr);
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
  Vec       workinit;    /* work vector, used to initialize the adjoint variables and for Dirac's delta terms */
  Vec       gradient;    /* gradient we are evaluating */
  Vec       wgrad;       /* work vector */
  PetscBool dirac_delta; /* If true, means that a delta contribution needs to be added to lambda during the post step method */
  Vec       direction;   /* If present, it is a second-order adjoint */
  TS        tlmts;       /* Tangent Linear Model TS, used for Hessian matvecs */
  TS        foats;       /* First order adjoint TS, used for Hessian matvecs when solving for the second order adjoint */
} AdjointCtx;

static PetscErrorCode AdjointTSDestroy_Private(void *ptr)
{
  AdjointCtx*    adj = (AdjointCtx*)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&adj->design);CHKERRQ(ierr);
  ierr = VecDestroy(&adj->workinit);CHKERRQ(ierr);
  ierr = VecDestroy(&adj->gradient);CHKERRQ(ierr);
  ierr = VecDestroy(&adj->wgrad);CHKERRQ(ierr);
  ierr = TSDestroy(&adj->fwdts);CHKERRQ(ierr);
  ierr = TSDestroy(&adj->tlmts);CHKERRQ(ierr);
  ierr = TSDestroy(&adj->foats);CHKERRQ(ierr);
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
    ierr = TSTrajectoryGetUpdatedHistoryVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,ft,&U,NULL);CHKERRQ(ierr);
  }
  ierr = TSComputeRHSJacobian(adj_ctx->fwdts,ft,U,A,P);CHKERRQ(ierr);
  if (type > TS_LINEAR) {
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(adj_ctx->fwdts->trajectory,&U,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* The adjoint formulation used assumes the problem written as H(U,Udot,t) = 0

   -> the forward DAE is Udot - G(U) = 0 ( -> H(U,Udot,t) := Udot - G(U) )
      the first-order adjoint DAE is F - L^T * G_U - Ldot^T in backward time (F the derivative of the objective wrt U)
      the first-order adjoint DAE is Ldot^T = L^T * G_U - F in forward time
   -> the second-order adjoint differs only by the forcing term :
      F = O_UM * direction + O_UU * eta + (L \otimes I_N)(tH_UM * direction + tH_UU * eta + tH_UUdot * etadot)
      with eta the solution of the tangent linear model
*/
static PetscErrorCode AdjointTSRHSFunctionLinear(TS adjts, PetscReal time, Vec U, Vec F, void *ctx)
{
  AdjointCtx     *adj_ctx;
  PetscReal      fwdt;
  PetscBool      has;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - time + adj_ctx->t0;
  if (adj_ctx->direction) { /* second-order adjoint */
    TS        fwdts = adj_ctx->fwdts;
    TS        tlmts = adj_ctx->tlmts;
    TS        foats = adj_ctx->foats;
    DM        dm;
    Vec       soawork0,soawork1;
    Vec       FWDH,TLMH;
    PetscBool hast;

    ierr = VecSet(F,0.0);CHKERRQ(ierr);
    ierr = TSGetDM(fwdts,&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&soawork0);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&soawork1);CHKERRQ(ierr);
    ierr = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,fwdt,&FWDH,NULL);CHKERRQ(ierr);
    ierr = TSTrajectoryGetUpdatedHistoryVecs(tlmts->trajectory,tlmts,fwdt,&TLMH,NULL);CHKERRQ(ierr);
    ierr = TSObjEval_UU(fwdts->funchead,FWDH,adj_ctx->design,fwdt,TLMH,soawork0,&has,soawork1);CHKERRQ(ierr);
    if (has) {
      ierr = VecAXPY(F,1.0,soawork1);CHKERRQ(ierr);
    }
    ierr = TSObjEval_UM(fwdts->funchead,FWDH,adj_ctx->design,fwdt,adj_ctx->direction,soawork0,&hast,soawork1);CHKERRQ(ierr);
    if (hast) {
      ierr = VecAXPY(F,1.0,soawork1);CHKERRQ(ierr);
      has  = PETSC_TRUE;
    }
    if (fwdts->HF[0][0] || fwdts->HF[0][1] || fwdts->HF[0][2]) {
      Vec FWDHdot,FOAH;

      ierr = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,fwdt,NULL,&FWDHdot);CHKERRQ(ierr);
      ierr = TSTrajectoryGetUpdatedHistoryVecs(foats->trajectory,foats,time,&FOAH,NULL);CHKERRQ(ierr);
      if (fwdts->HF[0][0]) { /* (L^T \otimes I_N) H_XX \eta, \eta the TLM solution */
        ierr = (*fwdts->HF[0][0])(fwdts,fwdt,FWDH,FWDHdot,adj_ctx->design,FOAH,TLMH,soawork1,fwdts->HFctx);CHKERRQ(ierr);
        ierr = VecAXPY(F,1.0,soawork1);CHKERRQ(ierr);
        has  = PETSC_TRUE;
      }
      if (fwdts->HF[0][1]) { /* (L^T \otimes I_N) H_XXdot \etadot, \eta the TLM solution */
        Vec TLMHdot;

        ierr = TSTrajectoryGetUpdatedHistoryVecs(tlmts->trajectory,tlmts,fwdt,NULL,&TLMHdot);CHKERRQ(ierr);
        ierr = (*fwdts->HF[0][1])(fwdts,fwdt,FWDH,FWDHdot,adj_ctx->design,FOAH,TLMHdot,soawork1,fwdts->HFctx);CHKERRQ(ierr);
        ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tlmts->trajectory,NULL,&TLMHdot);CHKERRQ(ierr);
        ierr = VecAXPY(F,1.0,soawork1);CHKERRQ(ierr);
        has  = PETSC_TRUE;
      }
      if (fwdts->HF[0][2]) { /* (L^T \otimes I_N) H_XM direction */
        ierr = (*fwdts->HF[0][2])(fwdts,fwdt,FWDH,FWDHdot,adj_ctx->design,FOAH,adj_ctx->direction,soawork1,fwdts->HFctx);CHKERRQ(ierr);
        ierr = VecAXPY(F,1.0,soawork1);CHKERRQ(ierr);
        has  = PETSC_TRUE;
      }
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,NULL,&FWDHdot);CHKERRQ(ierr);
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(foats->trajectory,&FOAH,NULL);CHKERRQ(ierr);
    }
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&FWDH,NULL);CHKERRQ(ierr);
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tlmts->trajectory,&TLMH,NULL);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&soawork0);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&soawork1);CHKERRQ(ierr);
  } else {
    TS  fwdts = adj_ctx->fwdts;
    DM  dm;
    Vec FWDH,W;

    ierr = TSGetDM(fwdts,&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&W);CHKERRQ(ierr);
    ierr = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,fwdt,&FWDH,NULL);CHKERRQ(ierr);
    ierr = TSObjEval_U(fwdts->funchead,FWDH,adj_ctx->design,fwdt,W,&has,F);CHKERRQ(ierr);
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&FWDH,NULL);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&W);CHKERRQ(ierr);
  }
  /* force recomputation of RHS Jacobian */
  adj_ctx->fwdts->rhsjacobian.time = PETSC_MIN_REAL;
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

   -> the first-order adjoint DAE is : F - L^T * (H_U - d/dt H_Udot) - Ldot^T H_Udot = 0 (in backward time)
      the first-order adjoint DAE is : Ldot^T H_Udot + L^T * (H_U + d/dt H_Udot) + F = 0 (in forward time)
      with F = dObjectiveIntegrand/dU (O_U in short)
   -> the second-order adjoint DAE differs only by the forcing term :
      F = O_UM * direction + O_UU * eta + (L \otimes I_N)(tH_UM * direction + tH_UU * eta + tH_UUdot * etadot) +
                                        - (Ldot \otimes I_N)(tH_UdotM * direction + tH_UdotU * eta + tH_UdotUdot * etadot) +
      with eta the solution of the tangent linear model and tH_U = H_U + d/dt H_Udot

   TODO : add support for augmented system when d/dt H_Udot != 0 ?
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
  if (adj_ctx->direction) { /* second order adjoint */
    TS        fwdts = adj_ctx->fwdts;
    TS        tlmts = adj_ctx->tlmts;
    TS        foats = adj_ctx->foats;
    DM        dm;
    Vec       soawork0,soawork1;
    Vec       FWDH,TLMH;
    PetscBool hast;

    ierr = VecSet(F,0.0);CHKERRQ(ierr);
    ierr = TSGetDM(fwdts,&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&soawork0);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&soawork1);CHKERRQ(ierr);
    ierr = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,fwdt,&FWDH,NULL);CHKERRQ(ierr);
    ierr = TSTrajectoryGetUpdatedHistoryVecs(tlmts->trajectory,tlmts,fwdt,&TLMH,NULL);CHKERRQ(ierr);
    ierr = TSObjEval_UU(fwdts->funchead,FWDH,adj_ctx->design,fwdt,TLMH,soawork0,&has,soawork1);CHKERRQ(ierr);
    if (has) {
      ierr = VecAXPY(F,1.0,soawork1);CHKERRQ(ierr);
    }
    ierr = TSObjEval_UM(fwdts->funchead,FWDH,adj_ctx->design,fwdt,adj_ctx->direction,soawork0,&hast,soawork1);CHKERRQ(ierr);
    if (hast) {
      ierr = VecAXPY(F,1.0,soawork1);CHKERRQ(ierr);
      has  = PETSC_TRUE;
    }
    if (fwdts->HF[0][0] || fwdts->HF[0][1] || fwdts->HF[0][2]) {
      Vec FWDHdot,FOAH;

      ierr = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,fwdt,NULL,&FWDHdot);CHKERRQ(ierr);
      ierr = TSTrajectoryGetUpdatedHistoryVecs(foats->trajectory,foats,time,&FOAH,NULL);CHKERRQ(ierr);
      if (fwdts->HF[0][0]) { /* (L^T \otimes I_N) H_XX \eta, \eta the TLM solution */
        ierr = (*fwdts->HF[0][0])(fwdts,fwdt,FWDH,FWDHdot,adj_ctx->design,FOAH,TLMH,soawork1,fwdts->HFctx);CHKERRQ(ierr);
        ierr = VecAXPY(F,1.0,soawork1);CHKERRQ(ierr);
        has  = PETSC_TRUE;
      }
      if (fwdts->HF[0][1]) { /* (L^T \otimes I_N) H_XXdot \etadot, \eta the TLM solution */
        Vec TLMHdot;

        ierr = TSTrajectoryGetUpdatedHistoryVecs(tlmts->trajectory,tlmts,fwdt,NULL,&TLMHdot);CHKERRQ(ierr);
        ierr = (*fwdts->HF[0][1])(fwdts,fwdt,FWDH,FWDHdot,adj_ctx->design,FOAH,TLMHdot,soawork1,fwdts->HFctx);CHKERRQ(ierr);
        ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tlmts->trajectory,NULL,&TLMHdot);CHKERRQ(ierr);
        ierr = VecAXPY(F,1.0,soawork1);CHKERRQ(ierr);
        has  = PETSC_TRUE;
      }
      if (fwdts->HF[0][2]) { /* (L^T \otimes I_N) H_XM direction */
        ierr = (*fwdts->HF[0][2])(fwdts,fwdt,FWDH,FWDHdot,adj_ctx->design,FOAH,adj_ctx->direction,soawork1,fwdts->HFctx);CHKERRQ(ierr);
        ierr = VecAXPY(F,1.0,soawork1);CHKERRQ(ierr);
        has  = PETSC_TRUE;
      }
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,NULL,&FWDHdot);CHKERRQ(ierr);
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(foats->trajectory,&FOAH,NULL);CHKERRQ(ierr);
    }
    if (fwdts->HF[1][0] || fwdts->HF[1][1] || fwdts->HF[1][2]) {
      Vec FOAHdot,FWDHdot;

      ierr = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,fwdt,NULL,&FWDHdot);CHKERRQ(ierr);
      ierr = TSTrajectoryGetUpdatedHistoryVecs(foats->trajectory,foats,time,NULL,&FOAHdot);CHKERRQ(ierr);
      if (fwdts->HF[1][0]) { /* (Ldot^T \otimes I_N) H_XdotX \eta, \eta the TLM solution */
        ierr = (*fwdts->HF[1][0])(fwdts,fwdt,FWDH,FWDHdot,adj_ctx->design,FOAHdot,TLMH,soawork1,fwdts->HFctx);CHKERRQ(ierr);
        ierr = VecAXPY(F,-1.0,soawork1);CHKERRQ(ierr);
        has  = PETSC_TRUE;
      }
      if (fwdts->HF[1][1]) { /* (Ldot^T \otimes I_N) H_XdotXdot \etadot, \eta the TLM solution */
        Vec TLMHdot;

        ierr = TSTrajectoryGetUpdatedHistoryVecs(tlmts->trajectory,tlmts,fwdt,NULL,&TLMHdot);CHKERRQ(ierr);
        ierr = (*fwdts->HF[1][1])(fwdts,fwdt,FWDH,FWDHdot,adj_ctx->design,FOAHdot,TLMHdot,soawork1,fwdts->HFctx);CHKERRQ(ierr);
        ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tlmts->trajectory,NULL,&TLMHdot);CHKERRQ(ierr);
        ierr = VecAXPY(F,-1.0,soawork1);CHKERRQ(ierr);
        has  = PETSC_TRUE;
      }
      if (fwdts->HF[1][2]) { /* (L^T \otimes I_N) H_XdotM direction */
        ierr = (*fwdts->HF[1][2])(fwdts,fwdt,FWDH,FWDHdot,adj_ctx->design,FOAHdot,adj_ctx->direction,soawork1,fwdts->HFctx);CHKERRQ(ierr);
        ierr = VecAXPY(F,-1.0,soawork1);CHKERRQ(ierr);
        has  = PETSC_TRUE;
      }
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(foats->trajectory,NULL,&FOAHdot);CHKERRQ(ierr);
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,NULL,&FWDHdot);CHKERRQ(ierr);
    }
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&FWDH,NULL);CHKERRQ(ierr);
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tlmts->trajectory,&TLMH,NULL);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&soawork0);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&soawork1);CHKERRQ(ierr);
  } else {
    TS  fwdts = adj_ctx->fwdts;
    DM  dm;
    Vec FWDH,W;

    ierr = TSGetDM(fwdts,&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&W);CHKERRQ(ierr);
    ierr = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,fwdt,&FWDH,NULL);CHKERRQ(ierr);
    ierr = TSObjEval_U(fwdts->funchead,FWDH,adj_ctx->design,fwdt,W,&has,F);CHKERRQ(ierr);
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&FWDH,NULL);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&W);CHKERRQ(ierr);
  }
  ierr = TSUpdateSplitJacobiansFromHistory(adj_ctx->fwdts,fwdt);CHKERRQ(ierr);
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

/* Handles the detection of Dirac's delta forcing terms in the adjoint equations
    - first-order adjoint f_x(state,design,t = fixed)
    - second-order adjoint f_xx(state,design,t = fixed) or f_xm(state,design,t = fixed)
*/
static PetscErrorCode AdjointTSEventFunction(TS adjts, PetscReal t, Vec U, PetscScalar fvalue[], void *ctx)
{
  AdjointCtx     *adj_ctx;
  TSObj          link;
  TS             ts;
  PetscInt       cnt = 0;
  PetscReal      fwdt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
  fwdt = adj_ctx->tf - t + adj_ctx->t0;
  ts   = adj_ctx->fwdts;
  link = ts->funchead;
  if (adj_ctx->direction) { /* second-order adjoint */
    while (link) { fvalue[cnt++] = ((link->f_xx || link->f_xm) && link->fixedtime > PETSC_MIN_REAL) ?  link->fixedtime - fwdt : 1.0; link = link->next; }
  } else {
    while (link) { fvalue[cnt++] = (link->f_x && link->fixedtime > PETSC_MIN_REAL) ?  link->fixedtime - fwdt : 1.0; link = link->next; }
  }
  PetscFunctionReturn(0);
}

/* Dirac's delta integration H_Udot^T ( L(+) - L(-) )  = - f_U -> L(+) = - H_Udot^-T f_U + L(-)
   We store the increment - H_Udot^-T f_U in adj_ctx->workinit and apply it during the AdjointTSPostStep
   AdjointTSComputeInitialConditions supports index-1 DAEs too (singular H_Udot).
*/
static PetscErrorCode AdjointTSComputeInitialConditions(TS,PetscReal,Vec,PetscBool,PetscBool);

static PetscErrorCode AdjointTSPostEvent(TS adjts, PetscInt nevents, PetscInt event_list[], PetscReal t, Vec U, PetscBool forwardsolve, void* ctx)
{
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecLockPush(U);CHKERRQ(ierr);
  ierr = AdjointTSComputeInitialConditions(adjts,t,NULL,PETSC_FALSE,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecLockPop(U);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(adjts,(void*)&adj_ctx);CHKERRQ(ierr);
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
    ierr = VecAXPY(lambda,1.0,adj_ctx->workinit);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)adjts,"_ts_evaluate_quadrature",(PetscObject*)&container);CHKERRQ(ierr);
    if (container) {
      PetscReal t;

      ierr = TSGetTime(adjts,&t);CHKERRQ(ierr);
      ierr = PetscContainerGetPointer(container,(void**)&qeval_ctx);CHKERRQ(ierr);
      PetscStackPush("ADJTS vector quadrature function");
      ierr = (*qeval_ctx->veval)(NULL,lambda,t,qeval_ctx->wquad[qeval_ctx->old],qeval_ctx->veval_ctx);CHKERRQ(ierr);
      PetscStackPop;
    }
  }
  adj_ctx->dirac_delta = PETSC_FALSE;
  if (adjts->reason == TS_CONVERGED_TIME) {
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
  SNES             snes;
  KSP              ksp;
  Mat              A,B;
  Vec              vatol,vrtol;
  PetscContainer   container;
  AdjointCtx       *adj;
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
  ierr = PetscObjectReference((PetscObject)ts);CHKERRQ(ierr);
  adj->fwdts = ts;

  /* invalidate time limits, that need to be set by AdjointTSSetTimeLimits */
  adj->t0 = adj->tf = PETSC_MAX_REAL;

  /* wrap application context in a container, so that it will be destroyed when calling TSDestroy on adjts */
  ierr = PetscContainerCreate(PetscObjectComm((PetscObject)(*adjts)),&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,adj);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(container,AdjointTSDestroy_Private);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)(*adjts),"_ts_adjctx",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);

  /* AdjointTS prefix: i.e. options called as -adjoint_ts_monitor or -adjoint_fwdtsprefix_ts_monitor */
  ierr = TSGetOptionsPrefix(ts,&prefix);CHKERRQ(ierr);
  ierr = TSSetOptionsPrefix(*adjts,"adjoint_");CHKERRQ(ierr);
  ierr = TSAppendOptionsPrefix(*adjts,prefix);CHKERRQ(ierr);

  /* options specific to AdjointTS */
  ierr = TSGetOptionsPrefix(*adjts,&prefix);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)*adjts),prefix,"Adjoint options","TS");CHKERRQ(ierr);
  jcon = PETSC_FALSE;
  ierr = PetscOptionsBool("-constjacobians","Whether or not the DAE Jacobians are constant",NULL,jcon,&jcon,NULL);CHKERRQ(ierr);
  rksp = PETSC_FALSE;
  ierr = PetscOptionsBool("-reuseksp","Reuse the KSP solver from the nonlinear model",NULL,rksp,&rksp,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* setup callbacks for adjoint DAE: we reuse the same jacobian matrices of the forward solve */
  ierr = TSGetIFunction(ts,NULL,&ifunc,NULL);CHKERRQ(ierr);
  ierr = TSGetRHSFunction(ts,NULL,&rhsfunc,NULL);CHKERRQ(ierr);
  if (ifunc) {
    TSSplitJacobians *splitJ;

    ierr = TSGetIJacobian(ts,&A,&B,NULL,NULL);CHKERRQ(ierr);
    ierr = TSSetIFunction(*adjts,NULL,AdjointTSIFunctionLinear,NULL);CHKERRQ(ierr);
    ierr = TSSetIJacobian(*adjts,A,B,AdjointTSIJacobian,NULL);CHKERRQ(ierr);
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
      ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
    }
    splitJ->splitdone = PETSC_FALSE;
    splitJ->jacconsts = jcon;
  } else {
    TSRHSJacobian rhsjacfunc;

    if (!rhsfunc) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"TSSetIFunction or TSSetRHSFunction not called");
    ierr = TSSetRHSFunction(*adjts,NULL,AdjointTSRHSFunctionLinear,NULL);CHKERRQ(ierr);
    ierr = TSGetRHSJacobian(ts,NULL,NULL,&rhsjacfunc,NULL);CHKERRQ(ierr);
    ierr = TSGetRHSMats_Private(ts,&A,&B);CHKERRQ(ierr);
    if (rhsjacfunc == TSComputeRHSJacobianConstant) {
      ierr = TSSetRHSJacobian(*adjts,A,B,TSComputeRHSJacobianConstant,NULL);CHKERRQ(ierr);
    } else if (jcon) { /* just to make sure we have a correct Jacobian */
      DM  dm;
      Vec U;

      ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dm,&U);CHKERRQ(ierr);
      ierr = TSComputeRHSJacobian(ts,PETSC_MIN_REAL,U,A,B);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dm,&U);CHKERRQ(ierr);
      ierr = TSSetRHSJacobian(*adjts,A,B,TSComputeRHSJacobianConstant,NULL);CHKERRQ(ierr);
    } else {
      ierr = TSSetRHSJacobian(*adjts,A,B,AdjointTSRHSJacobian,NULL);CHKERRQ(ierr);
    }
  }

  /* the equation type is the same */
  ierr = TSSetEquationType(*adjts,eqtype);CHKERRQ(ierr);

  /* the adjoint DAE is linear */
  ierr = TSSetProblemType(*adjts,TS_LINEAR);CHKERRQ(ierr);

  /* use KSPSolveTranspose to solve the adjoint */
  ierr = TSGetSNES(*adjts,&snes);CHKERRQ(ierr);
  ierr = SNESKSPONLYSetUseTransposeSolve(snes,PETSC_TRUE);CHKERRQ(ierr);

  /* adjointTS linear solver */
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  if (!rksp) { /* propagate KSP info of the forward model but use a different object */
    KSPType   ksptype;
    PetscReal atol,rtol,dtol;

    ierr = KSPGetType(ksp,&ksptype);CHKERRQ(ierr);
    ierr = KSPGetTolerances(ksp,&rtol,&atol,&dtol,&maxits);CHKERRQ(ierr);
    ierr = TSGetSNES(*adjts,&snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,ksptype);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,rtol,atol,dtol,maxits);CHKERRQ(ierr);
  } else { /* reuse the same KSP */
    ierr = TSGetSNES(*adjts,&snes);CHKERRQ(ierr);
    ierr = SNESSetKSP(snes,ksp);CHKERRQ(ierr);
  }
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
  if (gradient) PetscValidHeaderSpecific(gradient,VEC_CLASSID,2);
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Missing adjoint container");
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  if (gradient) {
    ierr = PetscObjectReference((PetscObject)gradient);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&adj_ctx->gradient);CHKERRQ(ierr);
  adj_ctx->gradient = gradient;
  PetscFunctionReturn(0);
}

/*
  Compute initial conditions for the adjoint DAE. It also initializes the quadrature (if needed).
  We use svec (instead of just loading from history inside the function) since the propagator Mat can use P*U
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
  PetscBool      rsve = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(adjts,time,2);
  if (svec) PetscValidHeaderSpecific(svec,VEC_CLASSID,3);
  PetscValidLogicalCollectiveBool(adjts,apply,4);
  PetscValidLogicalCollectiveBool(adjts,qinit,5);
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Missing adjoint container");
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  if (adj_ctx->direction && !adj_ctx->tlmts) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_ORDER,"Missing TLMTS! You need to call AdjointTSSetTLMTSAndFOATS");
  if (adj_ctx->direction && !adj_ctx->foats) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_ORDER,"Missing FOATS! You need to call AdjointTSSetTLMTSAndFOATS");
  fwdt = adj_ctx->tf - time + adj_ctx->t0;
  if (!svec) {
    ierr = TSTrajectoryGetUpdatedHistoryVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,fwdt,&svec,NULL);CHKERRQ(ierr);
    rsve = PETSC_TRUE;
  }

  /* only AdjointTSPostEvent and AdjointTSComputeInitialConditions can modify workinit */
  if (!adj_ctx->workinit) {
    Vec lambda;

    ierr = TSGetSolution(adjts,&lambda);CHKERRQ(ierr);
    ierr = VecDuplicate(lambda,&adj_ctx->workinit);CHKERRQ(ierr);
    ierr = VecLockPush(adj_ctx->workinit);CHKERRQ(ierr);
  }
  ierr = VecLockPop(adj_ctx->workinit);CHKERRQ(ierr);
  ierr = VecSet(adj_ctx->workinit,0.0);CHKERRQ(ierr);

  if (adj_ctx->direction) {
    TS        fwdts = adj_ctx->fwdts;
    TS        tlmts = adj_ctx->tlmts;
    TS        foats = adj_ctx->foats;
    DM        dm;
    Vec       soawork0,soawork1,TLMH[2];
    PetscBool hast;

    ierr = TSGetDM(adj_ctx->fwdts,&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&soawork0);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&soawork1);CHKERRQ(ierr);
    ierr = TSTrajectoryGetUpdatedHistoryVecs(tlmts->trajectory,tlmts,fwdt,&TLMH[0],&TLMH[1]);CHKERRQ(ierr);
    ierr = TSObjEvalFixed_UU(fwdts->funchead,svec,adj_ctx->design,fwdt,TLMH[0],soawork0,&has_g,soawork1);CHKERRQ(ierr);
    if (has_g) {
      ierr  = VecAXPY(adj_ctx->workinit,1.0,soawork1);CHKERRQ(ierr);
    }
    ierr = TSObjEvalFixed_UM(fwdts->funchead,svec,adj_ctx->design,fwdt,adj_ctx->direction,soawork0,&hast,soawork1);CHKERRQ(ierr);
    if (hast) {
      ierr  = VecAXPY(adj_ctx->workinit,1.0,soawork1);CHKERRQ(ierr);
      has_g = PETSC_TRUE;
    }
    if (rsve) {
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&svec,NULL);CHKERRQ(ierr);
      rsve = PETSC_FALSE;
    }
    if (fwdts->HF[1][0] || fwdts->HF[1][1] || fwdts->HF[1][2]) {
      Vec FOAH,FWDH[2];

      ierr = TSTrajectoryGetUpdatedHistoryVecs(foats->trajectory,foats,time,&FOAH,NULL);CHKERRQ(ierr);
      if (fwdts->HF[1][0]) { /* (L^T \otimes I_N) H_XdotX \eta, \eta the TLM solution */
        ierr  = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,fwdt,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
        ierr  = (*fwdts->HF[1][0])(fwdts,fwdt,FWDH[0],FWDH[1],adj_ctx->design,FOAH,TLMH[0],soawork1,fwdts->HFctx);CHKERRQ(ierr);
        ierr  = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
        ierr  = VecAXPY(adj_ctx->workinit,1.0,soawork1);CHKERRQ(ierr);
        has_g = PETSC_TRUE;
      }
      if (fwdts->HF[1][1]) { /* (L^T \otimes I_N) H_XdotXdot \etadot, \eta the TLM solution */
        ierr  = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,fwdt,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
        ierr  = (*fwdts->HF[1][1])(fwdts,fwdt,FWDH[0],FWDH[1],adj_ctx->design,FOAH,TLMH[1],soawork1,fwdts->HFctx);CHKERRQ(ierr);
        ierr  = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
        ierr  = VecAXPY(adj_ctx->workinit,1.0,soawork1);CHKERRQ(ierr);
        has_g = PETSC_TRUE;
      }
      if (fwdts->HF[1][2]) { /* (L^T \otimes I_N) H_XdotM direction */
        ierr  = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,fwdt,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
        ierr  = (*fwdts->HF[1][2])(fwdts,fwdt,FWDH[0],FWDH[1],adj_ctx->design,FOAH,adj_ctx->direction,soawork1,fwdts->HFctx);CHKERRQ(ierr);
        ierr  = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
        ierr  = VecAXPY(adj_ctx->workinit,1.0,soawork1);CHKERRQ(ierr);
        has_g = PETSC_TRUE;
      }
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(foats->trajectory,&FOAH,NULL);CHKERRQ(ierr);
    }
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tlmts->trajectory,&TLMH[0],&TLMH[1]);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&soawork0);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&soawork1);CHKERRQ(ierr);
  } else {
    TS  fwdts = adj_ctx->fwdts;
    DM  dm;
    Vec W;

    ierr = TSGetDM(fwdts,&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&W);CHKERRQ(ierr);
    ierr = TSObjEvalFixed_U(fwdts->funchead,svec,adj_ctx->design,fwdt,W,&has_g,adj_ctx->workinit);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&W);CHKERRQ(ierr);
    if (rsve) {
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&svec,NULL);CHKERRQ(ierr);
      rsve = PETSC_FALSE;
    }
  }
  ierr = TSGetEquationType(adj_ctx->fwdts,&eqtype);CHKERRQ(ierr);
  ierr = TSGetIJacobian(adjts,NULL,NULL,&ijac,NULL);CHKERRQ(ierr);
  if (eqtype == TS_EQ_DAE_SEMI_EXPLICIT_INDEX1) { /* details in [1,Section 4.2] */
    KSP       kspM,kspD;
    Mat       M = NULL,B = NULL,C = NULL,D = NULL,pM = NULL,pD = NULL;
    Mat       J_U,J_Udot,pJ_U,pJ_Udot;
    PetscInt  m,n,N;
    DM        dm;
    IS        diff = NULL,alg = NULL;
    Vec       f_x,W;
    PetscBool has_f;

    if (adj_ctx->direction) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_SUP,"Second order adjoint for INDEX-1 DAE not yet coded");
    ierr = VecDuplicate(adj_ctx->workinit,&f_x);CHKERRQ(ierr);
    if (!svec) {
      ierr = TSTrajectoryGetUpdatedHistoryVecs(adj_ctx->fwdts->trajectory,adj_ctx->fwdts,fwdt,&svec,NULL);CHKERRQ(ierr);
      rsve = PETSC_TRUE;
    }
    ierr = TSGetDM(adj_ctx->fwdts,&dm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(dm,&W);CHKERRQ(ierr);
    ierr = TSObjEval_U(adj_ctx->fwdts->funchead,svec,adj_ctx->design,fwdt,W,&has_f,f_x);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm,&W);CHKERRQ(ierr);
    if (rsve) {
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(adj_ctx->fwdts->trajectory,&svec,NULL);CHKERRQ(ierr);
    }
    if (!has_f && !has_g) {
      ierr = VecDestroy(&f_x);CHKERRQ(ierr);
      goto initialize;
    }
    if (!ijac) SETERRQ(PetscObjectComm((PetscObject)adj_ctx->fwdts),PETSC_ERR_SUP,"IJacobian routine is missing");
    ierr = PetscObjectQuery((PetscObject)adj_ctx->fwdts,"_ts_algebraic_is",(PetscObject*)&alg);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)adj_ctx->fwdts,"_ts_differential_is",(PetscObject*)&diff);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)adj_ctx->fwdts,"_ts_dae_BMat",(PetscObject*)&B);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)adj_ctx->fwdts,"_ts_dae_CMat",(PetscObject*)&C);CHKERRQ(ierr);
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
    } else {
      ierr = KSPGetOperators(kspD,&D,&pD);CHKERRQ(ierr);
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
    } else {
      ierr = KSPGetOperators(kspM,&M,&pM);CHKERRQ(ierr);
    }
    ierr = TSUpdateSplitJacobiansFromHistory(adj_ctx->fwdts,fwdt);CHKERRQ(ierr);
    ierr = TSGetSplitJacobians(adj_ctx->fwdts,&J_U,&pJ_U,&J_Udot,&pJ_Udot);CHKERRQ(ierr);
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
    if (M) { ierr = PetscObjectReference((PetscObject)M);CHKERRQ(ierr); }
    if (B) { ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr); }
    if (C) { ierr = PetscObjectReference((PetscObject)C);CHKERRQ(ierr); }
    if (D) { ierr = PetscObjectReference((PetscObject)D);CHKERRQ(ierr); }
    ierr = MatCreateSubMatrix(J_Udot,diff,diff,M ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,&M);CHKERRQ(ierr);
    if (pJ_Udot != J_Udot) {
      if (pM) { ierr = PetscObjectReference((PetscObject)pM);CHKERRQ(ierr); }
      ierr = MatCreateSubMatrix(pJ_Udot,diff,diff,pM ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,&pM);CHKERRQ(ierr);
    } else {
      if (pM && pM != M) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Amat and Pmat don't match");
      ierr = PetscObjectReference((PetscObject)M);CHKERRQ(ierr);
      pM   = M;
    }
    ierr = MatCreateSubMatrix(J_U,diff,alg ,B ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(J_U,alg ,diff,C ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(J_U,alg ,alg ,D ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
    if (pJ_U != J_U) {
      if (pD) { ierr = PetscObjectReference((PetscObject)pD);CHKERRQ(ierr); }
      ierr = MatCreateSubMatrix(pJ_U,alg,alg,pD ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,&pD);CHKERRQ(ierr);
    } else {
      if (pD && pD != D) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Amat and Pmat don't match");
      ierr = PetscObjectReference((PetscObject)D);CHKERRQ(ierr);
      pD   = D;
    }
    ierr = PetscObjectCompose((PetscObject)adj_ctx->fwdts,"_ts_dae_BMat",(PetscObject)B);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)adj_ctx->fwdts,"_ts_dae_CMat",(PetscObject)C);CHKERRQ(ierr);

    /* we first compute the contribution of the g(x,T,p) terms,
       the initial conditions are consistent by construction with the adjointed algebraic constraints, i.e.
       B^T lambda_d + D^T lambda_a = 0 */
    if (has_g) {
      Vec       g_d,g_a;
      PetscReal norm;

      ierr = VecGetSubVector(adj_ctx->workinit,diff,&g_d);CHKERRQ(ierr);
      ierr = VecGetSubVector(adj_ctx->workinit,alg,&g_a);CHKERRQ(ierr);
      ierr = VecNorm(g_a,NORM_2,&norm);CHKERRQ(ierr);
      if (norm) {
        ierr = KSPSetOperators(kspD,D,pD);CHKERRQ(ierr);
        ierr = KSPSolveTranspose(kspD,g_a,g_a);CHKERRQ(ierr);
        ierr = VecScale(g_a,-1.0);CHKERRQ(ierr);
        ierr = MatMultTransposeAdd(C,g_a,g_d,g_d);CHKERRQ(ierr);
        if (adj_ctx->fwdts->F_m && adj_ctx->gradient) { /* add fixed term to the gradient */
          TS        ts = adj_ctx->fwdts;
          PetscBool hasop;

          if (ts->F_m_f) { /* non constant dependence */
            Vec FWDH[2];

            ierr = TSTrajectoryGetUpdatedHistoryVecs(ts->trajectory,ts,fwdt,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
            ierr = (*ts->F_m_f)(ts,fwdt,FWDH[0],FWDH[1],adj_ctx->design,ts->F_m,ts->F_m_ctx);CHKERRQ(ierr);
            ierr = TSTrajectoryRestoreUpdatedHistoryVecs(ts->trajectory,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
          }
          ierr = MatHasOperation(ts->F_m,MATOP_MULT_TRANSPOSE_ADD,&hasop);CHKERRQ(ierr);
          if (hasop) {
            ierr = MatMultTransposeAdd(ts->F_m,g_a,adj_ctx->gradient,adj_ctx->gradient);CHKERRQ(ierr);
          } else {
            Vec w;

            ierr = VecDuplicate(adj_ctx->gradient,&w);CHKERRQ(ierr);
            ierr = MatMultTranspose(ts->F_m,g_a,w);CHKERRQ(ierr);
            ierr = VecAXPY(adj_ctx->gradient,1.0,w);CHKERRQ(ierr);
            ierr = VecDestroy(&w);CHKERRQ(ierr);
          }
        }
      }
      ierr = KSPSetOperators(kspM,M,pM);CHKERRQ(ierr);
      ierr = KSPSolveTranspose(kspM,g_d,g_d);CHKERRQ(ierr);
      ierr = MatMultTranspose(B,g_d,g_a);CHKERRQ(ierr);
      ierr = KSPSetOperators(kspD,D,pD);CHKERRQ(ierr);
      ierr = KSPSolveTranspose(kspD,g_a,g_a);CHKERRQ(ierr);
      ierr = VecScale(g_d,-1.0);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(adj_ctx->workinit,diff,&g_d);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(adj_ctx->workinit,alg,&g_a);CHKERRQ(ierr);
#if 0
      {
        Mat J_U;
        Vec test,test_a;
        PetscReal norm;

        ierr = VecDuplicate(adj_ctx->workinit,&test);CHKERRQ(ierr);
        ierr = TSGetSplitJacobians(adj_ctx->fwdts,&J_U,NULL);CHKERRQ(ierr);
        ierr = MatMultTranspose(J_U,adj_ctx->workinit,test);CHKERRQ(ierr);
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
      ierr = VecGetSubVector(adj_ctx->workinit,alg,&lambda_a);CHKERRQ(ierr);
      ierr = KSPSetOperators(kspD,D,pD);CHKERRQ(ierr);
      ierr = KSPSolveTranspose(kspD,f_a,f_a);CHKERRQ(ierr);
      ierr = VecAXPY(lambda_a,-1.0,f_a);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(adj_ctx->workinit,alg,&lambda_a);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(f_x,alg,&f_a);CHKERRQ(ierr);
    }
#if 0
    {
      Mat J_U;
      Vec test,test_a;
      PetscReal norm;

      ierr = VecDuplicate(adj_ctx->workinit,&test);CHKERRQ(ierr);
      ierr = TSGetSplitJacobians(adj_ctx->fwdts,&J_U,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = MatMultTranspose(J_U,adj_ctx->workinit,test);CHKERRQ(ierr);
      ierr = VecGetSubVector(test,alg,&test_a);CHKERRQ(ierr);
      ierr = VecNorm(test_a,NORM_2,&norm);CHKERRQ(ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)test),"FINAL: This should be zero %1.16e\n",norm);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(test,alg,&test_a);CHKERRQ(ierr);
      ierr = VecDestroy(&test);CHKERRQ(ierr);
    }
#endif
    ierr = VecDestroy(&f_x);CHKERRQ(ierr);
    ierr = MatDestroy(&M);CHKERRQ(ierr);
    ierr = MatDestroy(&pM);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = MatDestroy(&C);CHKERRQ(ierr);
    ierr = MatDestroy(&D);CHKERRQ(ierr);
    ierr = MatDestroy(&pD);CHKERRQ(ierr);
  } else {
    if (has_g) {
      if (ijac) { /* lambda_T(T) = (J_Udot)^T D_x, D_x the gradients of the functionals that sample the solution at the final time */
        KSP       ksp;
        Mat       J_Udot, pJ_Udot;
        DM        dm;
        Vec       W;

        ierr = TSUpdateSplitJacobiansFromHistory(adj_ctx->fwdts,fwdt);CHKERRQ(ierr);
        ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjointinit_ksp",(PetscObject*)&ksp);CHKERRQ(ierr);
        if (!ksp) {
          SNES       snes;
          PC         pc;
          KSPType    ksptype;
          PCType     pctype;
          const char *prefix;

          ierr = TSGetSNES(adjts,&snes);CHKERRQ(ierr);
          ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
          ierr = KSPGetType(ksp,&ksptype);CHKERRQ(ierr);
          ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
          ierr = PCGetType(pc,&pctype);CHKERRQ(ierr);
          ierr = KSPCreate(PetscObjectComm((PetscObject)adjts),&ksp);CHKERRQ(ierr);
          ierr = KSPSetTolerances(ksp,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
          ierr = KSPSetType(ksp,ksptype);CHKERRQ(ierr);
          ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
          ierr = PCSetType(pc,pctype);CHKERRQ(ierr);
          ierr = TSGetOptionsPrefix(adjts,&prefix);CHKERRQ(ierr);
          ierr = KSPSetOptionsPrefix(ksp,prefix);CHKERRQ(ierr);
          ierr = KSPAppendOptionsPrefix(ksp,"initlambda_");CHKERRQ(ierr);
          ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
          ierr = PetscObjectCompose((PetscObject)adjts,"_ts_adjointinit_ksp",(PetscObject)ksp);CHKERRQ(ierr);
          ierr = PetscObjectDereference((PetscObject)ksp);CHKERRQ(ierr);
        }
        ierr = TSGetSplitJacobians(adj_ctx->fwdts,NULL,NULL,&J_Udot,&pJ_Udot);CHKERRQ(ierr);
        ierr = KSPSetOperators(ksp,J_Udot,pJ_Udot);CHKERRQ(ierr);
        ierr = TSGetDM(adjts,&dm);CHKERRQ(ierr);
        ierr = DMGetGlobalVector(dm,&W);CHKERRQ(ierr);
        ierr = VecCopy(adj_ctx->workinit,W);CHKERRQ(ierr);
        ierr = KSPSolveTranspose(ksp,W,adj_ctx->workinit);CHKERRQ(ierr);
        ierr = DMRestoreGlobalVector(dm,&W);CHKERRQ(ierr);
        /* destroy inner vectors to avoid ABA issues when destroying the DM */
        ierr = VecDestroy(&ksp->vec_rhs);CHKERRQ(ierr);
        ierr = VecDestroy(&ksp->vec_sol);CHKERRQ(ierr);
      }
      /* the lambdas we use are equivalent to -lambda_T in [1] */
      ierr = VecScale(adj_ctx->workinit,-1.0);CHKERRQ(ierr);
    }
  }
initialize:
  ierr = VecLockPush(adj_ctx->workinit);CHKERRQ(ierr);
  if (apply) {
    Vec lambda;

    ierr = TSGetSolution(adjts,&lambda);CHKERRQ(ierr);
    ierr = VecCopy(adj_ctx->workinit,lambda);CHKERRQ(ierr);
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
    qeval_ctx->user      = AdjointTSPostStep;
    qeval_ctx->userafter = PETSC_TRUE;
    qeval_ctx->seval     = NULL;
    qeval_ctx->veval     = EvalQuadIntegrand_ADJ;
    qeval_ctx->vquad     = adj_ctx->gradient;
    qeval_ctx->cur       = 0;
    qeval_ctx->old       = 1;
    if (!qeval_ctx->wquad) {
      ierr = VecDuplicateVecs(qeval_ctx->vquad,5,&qeval_ctx->wquad);CHKERRQ(ierr);
    }

    ierr = PetscFree(qeval_ctx->veval_ctx);CHKERRQ(ierr);
    ierr = PetscNew(&adjq);CHKERRQ(ierr);
    adjq->fwdts   = ts;
    adjq->t0      = adj_ctx->t0;
    adjq->tf      = adj_ctx->tf;
    adjq->design  = adj_ctx->design;
    qeval_ctx->veval_ctx = adjq;

    ierr = TSGetTime(adjts,&t0);CHKERRQ(ierr);
    ierr = TSGetSolution(adjts,&lambda);CHKERRQ(ierr);
    PetscStackPush("ADJTS vector quadrature function");
    ierr = (*qeval_ctx->veval)(NULL,lambda,t0,qeval_ctx->wquad[qeval_ctx->old],qeval_ctx->veval_ctx);CHKERRQ(ierr);
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
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Missing adjoint container");
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)design);CHKERRQ(ierr);
  ierr = VecDestroy(&adj_ctx->design);CHKERRQ(ierr);
  adj_ctx->design = design;
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSSetDirection(TS adjts, Vec direction)
{
  PetscContainer c;
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  if (direction) PetscValidHeaderSpecific(direction,VEC_CLASSID,2);
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Missing adjoint container");
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  if (direction) ierr = PetscObjectReference((PetscObject)direction);CHKERRQ(ierr);
  ierr = VecDestroy(&adj_ctx->direction);CHKERRQ(ierr);
  adj_ctx->direction = direction;
  PetscFunctionReturn(0);
}

static PetscErrorCode AdjointTSSetTLMTSAndFOATS(TS adjts, TS tlmts, TS foats)
{
  PetscContainer c;
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  PetscValidHeaderSpecific(tlmts,TS_CLASSID,2);
  PetscValidHeaderSpecific(foats,TS_CLASSID,2);
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Missing adjoint container");
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)tlmts);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)foats);CHKERRQ(ierr);
  ierr = TSDestroy(&adj_ctx->tlmts);CHKERRQ(ierr);
  ierr = TSDestroy(&adj_ctx->foats);CHKERRQ(ierr);
  adj_ctx->tlmts = tlmts;
  adj_ctx->foats = foats;
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
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
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
  PetscInt       cnt;
  PetscBool      has;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Missing adjoint container");
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  ierr = TSGetNumObjectives(adj_ctx->fwdts,&cnt);CHKERRQ(ierr);
  if (!cnt) PetscFunctionReturn(0);
  ierr = TSHasObjectiveFixed(adj_ctx->fwdts,adj_ctx->t0,adj_ctx->tf,NULL,&has,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  if (has) {
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
  PetscReal      tf;
  PetscContainer c;
  AdjointCtx     *adj_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adjts,TS_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)adjts,"_ts_adjctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_PLIB,"Missing adjoint container");
  ierr = PetscContainerGetPointer(c,(void**)&adj_ctx);CHKERRQ(ierr);
  if (!adj_ctx->gradient) SETERRQ(PetscObjectComm((PetscObject)adjts),PETSC_ERR_ORDER,"Missing gradient vector");
  ierr = TSGetTime(adjts,&tf);CHKERRQ(ierr);
  if (PetscAbsReal(tf - adj_ctx->tf) > PETSC_SMALL) SETERRQ1(PetscObjectComm((PetscObject)adjts),PETSC_ERR_ORDER,"Backward solve did not complete %1.14e",tf-adj_ctx->tf);

  /* initial condition contribution to the gradient */
  if (adj_ctx->fwdts->G_m) {
    TS          fwdts = adj_ctx->fwdts;
    Vec         lambda, FWDH[2], work;
    TSIJacobian ijacfunc;
    Mat         J_Udot = NULL;
    DM          adm;

    ierr = TSGetDM(adjts,&adm);CHKERRQ(ierr);
    ierr = DMGetGlobalVector(adm,&work);CHKERRQ(ierr);
    ierr = TSGetSolution(adjts,&lambda);CHKERRQ(ierr);
    ierr = TSGetIJacobian(adjts,NULL,NULL,&ijacfunc,NULL);CHKERRQ(ierr);
    if (!ijacfunc) {
      ierr = VecCopy(lambda,work);CHKERRQ(ierr);
    } else {
      ierr = TSUpdateSplitJacobiansFromHistory(fwdts,adj_ctx->t0);CHKERRQ(ierr);
      ierr = TSGetSplitJacobians(fwdts,NULL,NULL,&J_Udot,NULL);CHKERRQ(ierr);
      ierr = MatMultTranspose(J_Udot,lambda,work);CHKERRQ(ierr);
    }
    if (!adj_ctx->wgrad) {
      ierr = VecDuplicate(adj_ctx->gradient,&adj_ctx->wgrad);CHKERRQ(ierr);
    }
    if (!adj_ctx->direction) { /* first-order adjoint in gradient computations */
      ierr = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,adj_ctx->t0,&FWDH[0],NULL);CHKERRQ(ierr);
      ierr = TSLinearizedICApply(fwdts,adj_ctx->t0,FWDH[0],adj_ctx->design,work,adj_ctx->wgrad,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&FWDH[0],NULL);CHKERRQ(ierr);
      ierr = VecAXPY(adj_ctx->gradient,1.0,adj_ctx->wgrad);CHKERRQ(ierr);
    } else { /* second-order adjoint in Hessian computations */
      TS  foats = adj_ctx->foats;
      TS  tlmts = adj_ctx->tlmts;
      DM  dm;
      Vec soawork0,soawork1,FOAH,FWDH[2],TLMH,TLMHdot = NULL;

      ierr = TSGetDM(adj_ctx->fwdts,&dm);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dm,&soawork0);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dm,&soawork1);CHKERRQ(ierr);

      /* compute first order contribution */
      ierr = TSTrajectoryGetUpdatedHistoryVecs(foats->trajectory,foats,adj_ctx->tf,&FOAH,NULL);CHKERRQ(ierr);
      ierr = TSTrajectoryGetUpdatedHistoryVecs(tlmts->trajectory,tlmts,adj_ctx->t0,&TLMH,fwdts->HF[1][1] ? &TLMHdot : NULL);CHKERRQ(ierr);
      ierr = TSTrajectoryGetUpdatedHistoryVecs(fwdts->trajectory,fwdts,adj_ctx->t0,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
      if (!J_Udot) {
        ierr = VecCopy(FOAH,soawork1);CHKERRQ(ierr);
      } else {
        ierr = MatMultTranspose(J_Udot,FOAH,soawork1);CHKERRQ(ierr);
      }
      /* XXX Hack to just solve for G_x (if any) */
      ierr = TSLinearizedICApply(fwdts,adj_ctx->t0,FWDH[0],adj_ctx->design,soawork1,soawork0,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
      if (fwdts->HG[0][0]) { /* (\mu^T \otimes I_N) G_XX \eta, \eta the TLM solution */
        ierr = (*fwdts->HG[0][0])(fwdts,adj_ctx->t0,FWDH[0],adj_ctx->design,soawork0,TLMH,soawork1,fwdts->HGctx);CHKERRQ(ierr);
        ierr = VecAXPY(work,-1.0,soawork1);CHKERRQ(ierr);
      }
      if (fwdts->HG[0][1]) { /* (\mu^T \otimes I_N) G_XM direction */
        ierr = (*fwdts->HG[0][1])(fwdts,adj_ctx->t0,FWDH[0],adj_ctx->design,soawork0,adj_ctx->direction,soawork1,fwdts->HGctx);CHKERRQ(ierr);
        ierr = VecAXPY(work,-1.0,soawork1);CHKERRQ(ierr);
      }
      if (fwdts->HF[1][0]) { /* (L^T \otimes I_N) H_XdotX \eta, \eta the TLM solution */
        ierr = (*fwdts->HF[1][0])(fwdts,adj_ctx->t0,FWDH[0],FWDH[1],adj_ctx->design,FOAH,TLMH,soawork1,fwdts->HFctx);CHKERRQ(ierr);
        ierr = VecAXPY(work,1.0,soawork1);CHKERRQ(ierr);
      }
      if (fwdts->HF[1][1]) { /* (L^T \otimes I_N) H_XdotXdot \etadot, \eta the TLM solution */
        ierr = (*fwdts->HF[1][1])(fwdts,adj_ctx->t0,FWDH[0],FWDH[1],adj_ctx->design,FOAH,TLMHdot,soawork1,fwdts->HFctx);CHKERRQ(ierr);
        ierr = VecAXPY(work,1.0,soawork1);CHKERRQ(ierr);
      }
      if (fwdts->HF[1][2]) { /* (L^T \otimes I_N) H_XdotM direction */
        ierr = (*fwdts->HF[1][2])(fwdts,adj_ctx->t0,FWDH[0],FWDH[1],adj_ctx->design,FOAH,adj_ctx->direction,soawork1,fwdts->HFctx);CHKERRQ(ierr);
        ierr = VecAXPY(work,1.0,soawork1);CHKERRQ(ierr);
      }
      ierr = TSLinearizedICApply(fwdts,adj_ctx->t0,FWDH[0],adj_ctx->design,work,adj_ctx->wgrad,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
      ierr = VecAXPY(adj_ctx->gradient,1.0,adj_ctx->wgrad);CHKERRQ(ierr);
      if (fwdts->HG[1][1]) { /* (\mu^T \otimes I_M) G_MM direction */
        ierr = (*fwdts->HG[1][1])(fwdts,adj_ctx->t0,FWDH[0],adj_ctx->design,soawork0,adj_ctx->direction,adj_ctx->wgrad,fwdts->HGctx);CHKERRQ(ierr);
        ierr = VecAXPY(adj_ctx->gradient,1.0,adj_ctx->wgrad);CHKERRQ(ierr);
      }
      if (fwdts->HG[1][0]) { /* (\mu^T \otimes I_M) G_MX  \eta, \eta the TLM solution */
        ierr = (*fwdts->HG[1][0])(fwdts,adj_ctx->t0,FWDH[0],adj_ctx->design,soawork0,TLMH,adj_ctx->wgrad,fwdts->HGctx);CHKERRQ(ierr);
        ierr = VecAXPY(adj_ctx->gradient,1.0,adj_ctx->wgrad);CHKERRQ(ierr);
      }
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(foats->trajectory,&FOAH,NULL);CHKERRQ(ierr);
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(fwdts->trajectory,&FWDH[0],&FWDH[1]);CHKERRQ(ierr);
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tlmts->trajectory,&TLMH,TLMHdot ? &TLMHdot : NULL);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dm,&soawork0);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dm,&soawork1);CHKERRQ(ierr);
    }
    ierr = DMRestoreGlobalVector(adm,&work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------ Routines for the TS representing the tangent linear model, namespaced by TLMTS ----------------------- */

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

static PetscErrorCode TLMTSSetPerturbation(TS lts, Vec mdelta)
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

/* creates the TS for the tangent linear model */
static PetscErrorCode TSCreateTLMTS(TS ts, TS* lts)
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
  ierr = TSAdaptCreate(PetscObjectComm((PetscObject)*lts),&(*lts)->adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetType((*lts)->adapt,TSADAPTTRAJECTORY);CHKERRQ(ierr);
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
    ierr = KSPSetType(ksp,ksptype);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,rtol,atol,dtol,maxits);CHKERRQ(ierr);
  } else { /* reuse the same KSP */
    ierr = TSGetSNES(*lts,&snes);CHKERRQ(ierr);
    ierr = SNESSetKSP(snes,ksp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* auxiliary function to solve a forward model with a quadrature */
static PetscErrorCode TSFWDWithQuadrature_Private(TS ts, Vec X, Vec design, Vec direction, Vec quadvec, PetscReal *quadscalar)
{
  Vec             U;
  PetscContainer  container;
  TSQuadratureCtx *qeval_ctx;
  TLMTS_Ctx       *tlm = NULL;
  PetscReal       t0,tf,tfup,dt;
  PetscBool       fidt;
  SQuadEval       seval_fixed, seval;
  VQuadEval       veval_fixed, veval;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (direction) PetscValidHeaderSpecific(direction,VEC_CLASSID,4);
  if (quadvec)   PetscValidHeaderSpecific(quadvec,VEC_CLASSID,5);
  if (direction && !quadvec) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Cannot compute Hessian without quadrature vector");
  if (direction) {
    PetscContainer c;
    ierr = PetscObjectQuery((PetscObject)ts,"_ts_tlm_ctx",(PetscObject*)&c);CHKERRQ(ierr);
    if (!c) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Missing tlm container");
    ierr = PetscContainerGetPointer(c,(void**)&tlm);CHKERRQ(ierr);
    if (ts->funchead) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Objective functions linked to TLMTS");
    ts->funchead = tlm->model->funchead;
  }

  /* solution vector */
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  if (!U) {
    ierr = VecDuplicate(X,&U);CHKERRQ(ierr);
    ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)U);CHKERRQ(ierr);
  }
  ierr = VecCopy(X,U);CHKERRQ(ierr);

  /* quadrature evaluations */
  seval       = tlm ? NULL : (quadscalar ? EvalQuadObj_FWD      : NULL);
  seval_fixed = tlm ? NULL : (quadscalar ? EvalQuadObjFixed_FWD : NULL);
  veval       = quadvec ? (tlm ? EvalQuadIntegrand_TLM      : EvalQuadIntegrand_FWD)      : NULL;
  veval_fixed = quadvec ? (tlm ? EvalQuadIntegrandFixed_TLM : EvalQuadIntegrandFixed_FWD) : NULL;

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
  qeval_ctx->seval     = seval;
  qeval_ctx->seval_ctx = design;
  qeval_ctx->squad     = 0.0;
  qeval_ctx->psquad    = 0.0;
  qeval_ctx->veval     = veval;
  qeval_ctx->vquad     = quadvec;
  qeval_ctx->cur       = 0;
  qeval_ctx->old       = 1;

  ierr = TSSetPostStep(ts,TSQuadrature_PostStep);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);

  /* evaluate scalar function at initial time */
  ierr = TSGetMaxTime(ts,&tf);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
  if (qeval_ctx->seval) {
    PetscStackPush("TS scalar quadrature function");
    ierr = (*qeval_ctx->seval)(ts->funchead,U,t0,&qeval_ctx->psquad,qeval_ctx->seval_ctx);CHKERRQ(ierr);
    PetscStackPop;
  }
  ierr = PetscFree(qeval_ctx->veval_ctx);CHKERRQ(ierr);
  if (qeval_ctx->veval) {
    PetscBool has;

    if (tlm) { /* Hessian computations */
      PetscBool has1,has2;

      ierr = TSHasObjectiveIntegrand(ts,NULL,NULL,NULL,NULL,&has1,&has2);CHKERRQ(ierr);
      has  = (PetscBool)(has1 || has2);
      if (tlm->model->HF[2][0] || tlm->model->HF[2][1] || tlm->model->HF[2][2]) has = PETSC_TRUE;
    } else {
      ierr = TSHasObjectiveIntegrand(ts,NULL,NULL,&has,NULL,NULL,NULL);CHKERRQ(ierr);
    }
    if (!has) { /* cost integrands not present */
      qeval_ctx->veval = NULL;
    }
    if (!qeval_ctx->wquad) {
      ierr = VecDuplicateVecs(qeval_ctx->vquad,5,&qeval_ctx->wquad);CHKERRQ(ierr);
    }
    if (tlm) {
      PetscBool has1,has2;

      ierr = TSHasObjectiveFixed(ts,t0,tf,NULL,NULL,NULL,NULL,&has1,&has2,NULL);CHKERRQ(ierr);
      has  = (PetscBool)(has1 || has2);
    } else {
      ierr = TSHasObjectiveFixed(ts,t0,tf,NULL,NULL,&has,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    }
    if (!has) veval_fixed = NULL;
  }
  /* for gradient computations, we just need the design vector and one work vector for the function evaluation
     for Hessian computations, we need extra data */
  if (qeval_ctx->veval || veval_fixed) {
    if (!tlm) {
      Vec *v;

      ierr = PetscCalloc1(2,&v);CHKERRQ(ierr);
      v[0] = design;
      v[1] = qeval_ctx->wquad[2];
      qeval_ctx->veval_ctx = v;
    } else {
      TLMEvalQuadCtx* q;

      ierr = PetscNew(&q);CHKERRQ(ierr);
      qeval_ctx->veval_ctx = q;

      ierr = PetscObjectQuery((PetscObject)tlm->model,"_ts_hessian_foats",(PetscObject*)&q->adjts);CHKERRQ(ierr); /* XXX */
      if (!q->adjts) SETERRQ(PetscObjectComm((PetscObject)tlm->model),PETSC_ERR_PLIB,"Missing first-order adjoint");

      q->fwdts     = tlm->model;
      q->tlmts     = ts;
      q->t0        = t0;
      q->tf        = tf;
      q->design    = design;
      q->direction = direction;
      q->work1     = qeval_ctx->wquad[2];
      q->work2     = qeval_ctx->wquad[3];
    }

    /* initialize trapz rule */
    if (qeval_ctx->veval) {
      Vec sol;

      ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
      PetscStackPush("TS vector quadrature function");
      ierr = (*qeval_ctx->veval)(ts->funchead,sol,t0,qeval_ctx->wquad[qeval_ctx->old],qeval_ctx->veval_ctx);CHKERRQ(ierr);
      PetscStackPop;
    }
  }

  /* forward solve */
  fidt = PETSC_TRUE;
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  if (ts->adapt) {
    ierr = PetscObjectTypeCompare((PetscObject)ts->adapt,TSADAPTNONE,&fidt);CHKERRQ(ierr);
  }

  /* determine if there are functionals, gradients or Hessians wrt parameters of the type f(U,M,t=fixed) to be evaluated */
  /* we don't use events since there's no API to add new events to a pre-existing set */
  tfup = tf;
  do {
    PetscBool has_f = PETSC_FALSE, has_m = PETSC_FALSE;

    ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
    if (seval_fixed || veval_fixed) {
      if (direction && veval_fixed) {
        PetscBool has1,has2;

        ierr  = TSHasObjectiveFixed(ts,t0,tf,NULL,NULL,NULL,NULL,&has1,&has2,&tfup);CHKERRQ(ierr);
        has_m = (PetscBool)(has1 || has2);
      } else {
        ierr = TSHasObjectiveFixed(ts,t0,tf,seval_fixed ? &has_f : NULL,NULL,veval_fixed ? &has_m : NULL,NULL,NULL,NULL,&tfup);CHKERRQ(ierr);
      }
    }
    ierr = TSSetMaxTime(ts,tfup);CHKERRQ(ierr);
    ierr = TSSolve(ts,NULL);CHKERRQ(ierr);
    if (has_f) {
      Vec       sol;
      PetscReal v;

      ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
      PetscStackPush("TS scalar quadrature function (fixed time)");
      ierr = (*seval_fixed)(ts->funchead,sol,tfup,&v,qeval_ctx->seval_ctx);CHKERRQ(ierr);
      PetscStackPop;
      qeval_ctx->squad += v;
    }
    if (has_m) { /* we use wquad[4] since wquad[3] can be used by the TLM quadrature */
      Vec sol;

      ierr = TSGetSolution(ts,&sol);CHKERRQ(ierr);
      PetscStackPush("TS vector quadrature function (fixed time)");
      ierr = (*veval_fixed)(ts->funchead,sol,tfup,qeval_ctx->wquad[4],qeval_ctx->veval_ctx);CHKERRQ(ierr);
      PetscStackPop;
      ierr = VecAXPY(qeval_ctx->vquad,1.0,qeval_ctx->wquad[4]);CHKERRQ(ierr);
    }
    if (fidt) { /* restore fixed time step */
      ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
    }
  } while (tfup < tf);

  if (tlm) ts->funchead = NULL;

  /* restore user PostStep */
  ierr = TSSetPostStep(ts,qeval_ctx->user);CHKERRQ(ierr);

  /* get back scalar value */
  if (quadscalar) *quadscalar = qeval_ctx->squad;
  PetscFunctionReturn(0);
}

/* ------------------ Routines for the Mat that represents the linearized propagator ----------------------- */

typedef struct {
  TS           model;
  TS           lts;
  TS           adjlts;
  Vec          x0;
  Mat          P;
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
  ierr = MatDestroy(&prop->P);CHKERRQ(ierr);
  ierr = PetscFree(prop);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Dummy objective function to not have TSSetObjective complaining about a null objective */
static PetscErrorCode TLMTS_dummyOBJ(Vec U, Vec M, PetscReal time, PetscReal *f, void *ctx)
{
  PetscFunctionBegin;
  *f = 0.0;
  PetscFunctionReturn(0);
}

/* Just a silly function to pass information to initialize the adjoint variables */
static PetscErrorCode TLMTS_dummyRHS(Vec U, Vec M, PetscReal time, Vec grad, void *ctx)
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
  if (!tlm->workrhs) {
    ierr = VecDuplicate(prop->x0,&tlm->workrhs);CHKERRQ(ierr);
  }
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = AdjointTSSetDesign(prop->adjlts,tlm->design);CHKERRQ(ierr);
  ierr = AdjointTSSetTimeLimits(prop->adjlts,prop->t0,prop->tf);CHKERRQ(ierr);
  ierr = AdjointTSSetInitialGradient(prop->adjlts,y);CHKERRQ(ierr);
  /* Initialize adjoint variables using P^T x or x */
  if (prop->P) {
    ierr = MatMultTranspose(prop->P,x,tlm->workrhs);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(x,tlm->workrhs);CHKERRQ(ierr);
  }
  ierr = AdjointTSComputeInitialConditions(prop->adjlts,prop->t0,tlm->workrhs,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSSetStepNumber(prop->adjlts,0);CHKERRQ(ierr);
  ierr = TSRestartStep(prop->adjlts);CHKERRQ(ierr);
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
  ierr = TLMTSSetPerturbation(prop->lts,x);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(prop->lts,(void *)&tlm);CHKERRQ(ierr);
  if (!tlm->workrhs) {
    ierr = VecDuplicate(prop->x0,&tlm->workrhs);CHKERRQ(ierr);
  }

  /* initialize tlm->workrhs if needed */
  ierr = VecSet(tlm->workrhs,0.0);CHKERRQ(ierr);
  if (prop->model->F_m) {
    TS ts = prop->model;
    if (!ts->F_m_f) { /* constant dependence */
      ierr = MatMult(ts->F_m,x,tlm->workrhs);CHKERRQ(ierr);
    }
  }

  /* sample initial condition dependency
     we use prop->lts instead of prop->model since the MatPropagator tests
     for IC dependency even if the model does not have any IC gradient set */
  ierr = TSGetSolution(prop->lts,&sol);CHKERRQ(ierr);
  ierr = TSLinearizedICApply(prop->lts,prop->t0,prop->x0,tlm->design,x,sol,PETSC_FALSE,PETSC_TRUE);CHKERRQ(ierr);
  ierr = VecScale(sol,-1.0);CHKERRQ(ierr);

  ierr = TSSetStepNumber(prop->lts,0);CHKERRQ(ierr);
  ierr = TSRestartStep(prop->lts);CHKERRQ(ierr);
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
  if (prop->P) {
    ierr = MatMult(prop->P,sol,y);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(sol,y);CHKERRQ(ierr);
  }
  prop->tj = prop->model->trajectory;
  prop->model->trajectory = otrj;
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
  ierr = TSRestartStep(prop->model);CHKERRQ(ierr);
  ierr = TSSetTime(prop->model,t0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(prop->model,dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(prop->model,PETSC_MAX_INT);CHKERRQ(ierr);
  ierr = TSSetMaxTime(prop->model,tf);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(prop->model,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  /* Create trajectory object */
  otrj = prop->model->trajectory;
  ierr = TSTrajectoryCreate(PetscObjectComm((PetscObject)prop->model),&prop->model->trajectory);CHKERRQ(ierr);
  ierr = TSTrajectorySetType(prop->model->trajectory,prop->model,TSTRAJECTORYMEMORY);CHKERRQ(ierr);
  ierr = TSTrajectorySetSolutionOnly(prop->model->trajectory,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSTrajectorySetFromOptions(prop->model->trajectory,prop->model);CHKERRQ(ierr);
  /* we don't have an API for this right now */
  prop->model->trajectory->adjoint_solve_mode = PETSC_FALSE;

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

    ierr = PetscObjectReference((PetscObject)P);CHKERRQ(ierr);
    ierr = MatDestroy(&prop->P);CHKERRQ(ierr);
    prop->P = P;
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

  /* creates the tangent linear model solver and its adjoint */
  ierr = TSCreateTLMTS(prop->model,&prop->lts);CHKERRQ(ierr);
  ierr = TLMTSSetDesign(prop->lts,design);CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject)design);CHKERRQ(ierr);
  ierr = TSSetFromOptions(prop->lts);CHKERRQ(ierr);
  ierr = TSSetObjective(prop->lts,prop->tf,TLMTS_dummyOBJ,TLMTS_dummyRHS,NULL,
                        NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  /* we need to call this since we will then compute the adjoint of the TLM */
  ierr = TSSetGradientDAE(prop->lts,prop->model->F_m,prop->model->F_m_f,prop->model->F_m_ctx);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

/* ------------------ Routines for the Hessian matrix ----------------------- */

typedef struct {
  TS           model;    /* nonlinear DAE */
  TS           tlmts;    /* tangent linear model solver */
  TS           foats;    /* first-order adjoint solver */
  TS           soats;    /* second-order adjoint solver */
  Vec          x0;       /* initial conditions */
  PetscReal    t0,dt,tf;
  Vec          design;
  TSTrajectory modeltj;  /* nonlinear model trajectory */
} TSHessian;

static PetscErrorCode TSHessianReset_Private(void *ptr)
{
  TSHessian*     tshess = (TSHessian*)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSDestroy(&tshess->model);CHKERRQ(ierr);
  ierr = TSDestroy(&tshess->tlmts);CHKERRQ(ierr);
  ierr = TSDestroy(&tshess->foats);CHKERRQ(ierr);
  ierr = TSDestroy(&tshess->soats);CHKERRQ(ierr);
  ierr = VecDestroy(&tshess->x0);CHKERRQ(ierr);
  ierr = VecDestroy(&tshess->design);CHKERRQ(ierr);
  ierr = TSTrajectoryDestroy(&tshess->modeltj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSHessianDestroy_Private(void *ptr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSHessianReset_Private(ptr);CHKERRQ(ierr);
  ierr = PetscFree(ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_TSHessian(Mat H, Vec x, Vec y)
{
  PetscContainer c;
  TSHessian      *tshess;
  TLMTS_Ctx      *tlm;
  TSTrajectory   otrj;
  TSAdapt        adapt;
  Vec            eta,L;
  PetscReal      dt;
  PetscBool      istr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)H,"_ts_hessian_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) SETERRQ(PetscObjectComm((PetscObject)H),PETSC_ERR_PLIB,"Not a valid Hessian matrix");
  ierr = PetscContainerGetPointer(c,(void**)&tshess);CHKERRQ(ierr);
  if (!tshess->model->funchead) PetscFunctionReturn(0);

  otrj = tshess->model->trajectory;
  tshess->model->trajectory = tshess->modeltj;

  /* solve tangent linear model */
  ierr = TSTrajectoryDestroy(&tshess->tlmts->trajectory);CHKERRQ(ierr); /* XXX add Reset method to TSTrajectory */
  ierr = TSTrajectoryCreate(PetscObjectComm((PetscObject)tshess->tlmts),&tshess->tlmts->trajectory);CHKERRQ(ierr);
  ierr = TSTrajectorySetType(tshess->tlmts->trajectory,tshess->tlmts,TSTRAJECTORYMEMORY);CHKERRQ(ierr);
  ierr = TSTrajectorySetSolutionOnly(tshess->tlmts->trajectory,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSTrajectorySetFromOptions(tshess->tlmts->trajectory,tshess->tlmts);CHKERRQ(ierr);
  tshess->tlmts->trajectory->adjoint_solve_mode = PETSC_FALSE;

  ierr = TLMTSSetPerturbation(tshess->tlmts,x);CHKERRQ(ierr);
  ierr = TSGetSolution(tshess->tlmts,&eta);CHKERRQ(ierr);
  if (!eta) {
    Vec U;

    ierr = TSGetSolution(tshess->model,&U);CHKERRQ(ierr);
    ierr = VecDuplicate(U,&eta);CHKERRQ(ierr);
    ierr = TSSetSolution(tshess->tlmts,eta);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)eta);CHKERRQ(ierr);
  }
  ierr = TSLinearizedICApply(tshess->model,tshess->t0,tshess->x0,tshess->design,x,eta,PETSC_FALSE,PETSC_TRUE);CHKERRQ(ierr);
  ierr = VecScale(eta,-1.0);CHKERRQ(ierr);

  ierr = TSGetAdapt(tshess->tlmts,&adapt);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)adapt,TSADAPTTRAJECTORY,&istr);CHKERRQ(ierr);
  ierr = TSAdaptTrajectorySetTrajectory(adapt,tshess->modeltj,PETSC_FALSE);CHKERRQ(ierr);
  ierr = TSGetApplicationContext(tshess->tlmts,(void *)&tlm);CHKERRQ(ierr);
  if (!tlm->workrhs) {
    ierr = VecDuplicate(tshess->x0,&tlm->workrhs);CHKERRQ(ierr);
  }
  /* initialize tlm->workrhs if needed */
  ierr = VecSet(tlm->workrhs,0.0);CHKERRQ(ierr);
  if (tshess->model->F_m) {
    TS ts = tshess->model;
    if (!ts->F_m_f) { /* constant dependence */
      ierr = MatMult(ts->F_m,x,tlm->workrhs);CHKERRQ(ierr);
    }
  }

  ierr = TSSetStepNumber(tshess->tlmts,0);CHKERRQ(ierr);
  ierr = TSRestartStep(tshess->tlmts);CHKERRQ(ierr);
  ierr = TSSetTime(tshess->tlmts,tshess->t0);CHKERRQ(ierr);
  ierr = TSSetMaxTime(tshess->tlmts,tshess->tf);CHKERRQ(ierr);
  ierr = TSHistoryGetTimeStep(tshess->modeltj->tsh,PETSC_FALSE,0,&dt);CHKERRQ(ierr);
  ierr = TSSetTimeStep(tshess->tlmts,dt);CHKERRQ(ierr);
  if (istr) {
    ierr = TSSetMaxSteps(tshess->tlmts,tshess->modeltj->tsh->n-1);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(tshess->tlmts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  } else {
    ierr = TSSetMaxSteps(tshess->tlmts,PETSC_MAX_INT);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(tshess->tlmts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  }

  /* XXX should we add the AdjointTS to the TS private data? */
  ierr = PetscObjectCompose((PetscObject)tshess->model,"_ts_hessian_foats",(PetscObject)tshess->foats);CHKERRQ(ierr);
  ierr = TSFWDWithQuadrature_Private(tshess->tlmts,eta,tshess->design,x,y,NULL);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)tshess->model,"_ts_hessian_foats",NULL);CHKERRQ(ierr);

  /* second-order adjoint solve */
  ierr = TSGetSolution(tshess->soats,&L);CHKERRQ(ierr);
  if (!L) {
    Vec U;

    ierr = TSGetSolution(tshess->model,&U);CHKERRQ(ierr);
    ierr = VecDuplicate(U,&L);CHKERRQ(ierr);
    ierr = TSSetSolution(tshess->soats,L);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)L);CHKERRQ(ierr);
  }
  ierr = AdjointTSSetTimeLimits(tshess->soats,tshess->t0,tshess->tf);CHKERRQ(ierr);
  ierr = AdjointTSSetDirection(tshess->soats,x);CHKERRQ(ierr);
  ierr = AdjointTSSetTLMTSAndFOATS(tshess->soats,tshess->tlmts,tshess->foats);CHKERRQ(ierr);
  ierr = AdjointTSSetInitialGradient(tshess->soats,y);CHKERRQ(ierr);
  ierr = AdjointTSComputeInitialConditions(tshess->soats,tshess->t0,NULL,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSSetStepNumber(tshess->soats,0);CHKERRQ(ierr);
  ierr = TSRestartStep(tshess->soats);CHKERRQ(ierr);
  ierr = TSHistoryGetTimeStep(tshess->modeltj->tsh,PETSC_TRUE,0,&dt);CHKERRQ(ierr);
  ierr = TSSetTimeStep(tshess->soats,dt);CHKERRQ(ierr);
  ierr = TSGetAdapt(tshess->soats,&adapt);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)adapt,TSADAPTTRAJECTORY,&istr);CHKERRQ(ierr);
  if (!istr) {
    PetscBool isnone;

    ierr = PetscObjectTypeCompare((PetscObject)adapt,TSADAPTNONE,&isnone);CHKERRQ(ierr);
    if (isnone && tshess->dt > 0.0) {
      ierr = TSSetTimeStep(tshess->soats,tshess->dt);CHKERRQ(ierr);
    }
    ierr = TSSetMaxSteps(tshess->soats,PETSC_MAX_INT);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(tshess->soats,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  } else { /* follow trajectory -> fix number of time steps */
    PetscInt nsteps = tshess->modeltj->tsh->n;

    ierr = TSSetMaxSteps(tshess->soats,nsteps-1);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(tshess->soats,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  }
  ierr = TSSolve(tshess->soats,L);CHKERRQ(ierr);
  ierr = AdjointTSComputeFinalGradient(tshess->soats);CHKERRQ(ierr);

  ierr = AdjointTSSetInitialGradient(tshess->soats,NULL);CHKERRQ(ierr);
  ierr = AdjointTSSetDirection(tshess->soats,NULL);CHKERRQ(ierr);
  ierr = TSTrajectoryDestroy(&tshess->tlmts->trajectory);CHKERRQ(ierr); /* XXX add Reset method to TSTrajectory */
  tshess->model->trajectory = otrj;
  PetscFunctionReturn(0);
}

/* private functions for objective, gradient and Hessian evaluation */
static PetscErrorCode TSComputeObjectiveAndGradient_Private(TS ts, Vec X, Vec design, Vec gradient, PetscReal *val)
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
  }

  /* forward solve */
  ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
  ierr = TSFWDWithQuadrature_Private(ts,X,design,NULL,gradient,val);CHKERRQ(ierr);

  /* adjoint */
  if (gradient) {
    TS  adjts;
    Vec lambda,U;

    ierr = TSCreateAdjointTS(ts,&adjts);CHKERRQ(ierr);
    ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
    ierr = VecDuplicate(U,&lambda);CHKERRQ(ierr);
    ierr = TSSetSolution(adjts,lambda);CHKERRQ(ierr);
    ierr = TSHistoryGetTimeStep(ts->trajectory->tsh,PETSC_TRUE,0,&dt);CHKERRQ(ierr);
    ierr = TSGetTime(ts,&tf);CHKERRQ(ierr);
    ierr = TSSetTimeStep(adjts,dt);CHKERRQ(ierr);
    ierr = AdjointTSSetTimeLimits(adjts,t0,tf);CHKERRQ(ierr);
    ierr = AdjointTSSetDesign(adjts,design);CHKERRQ(ierr);
    ierr = AdjointTSSetInitialGradient(adjts,gradient);CHKERRQ(ierr);
    ierr = AdjointTSComputeInitialConditions(adjts,t0,NULL,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
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

    /* restore TS to its original state */
    ierr = TSTrajectoryDestroy(&ts->trajectory);CHKERRQ(ierr);
    ts->trajectory  = otrj;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeHessian_Private(TS ts, PetscReal t0, PetscReal dt, PetscReal tf, Vec X, Vec design, Mat H)
{
  PetscContainer c;
  TSHessian      *tshess;
  Vec            U,L;
  TSTrajectory   otrj;
  PetscInt       n,N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)H,"_ts_hessian_ctx",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c) {
    ierr = PetscNew(&tshess);CHKERRQ(ierr);
    ierr = PetscContainerCreate(PetscObjectComm((PetscObject)ts),&c);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(c,tshess);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(c,TSHessianDestroy_Private);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)H,"_ts_hessian_ctx",(PetscObject)c);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)c);CHKERRQ(ierr);
  }
  ierr = VecGetLocalSize(design,&n);CHKERRQ(ierr);
  ierr = VecGetSize(design,&N);CHKERRQ(ierr);
  ierr = MatSetSizes(H,n,n,N,N);CHKERRQ(ierr);
  ierr = MatSetType(H,MATSHELL);CHKERRQ(ierr);
  ierr = MatShellSetOperation(H,MATOP_MULT,(void (*)())MatMult_TSHessian);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(c,(void**)&tshess);CHKERRQ(ierr);

  /* nonlinear model */
  if (ts != tshess->model) {
    ierr = TSHessianReset_Private(tshess);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)ts);CHKERRQ(ierr);
    tshess->model = ts;
  }

  if (!X) {
    ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
    if (!X) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Missing solution vector");
  }
  if (!tshess->x0) {
    ierr = VecDuplicate(X,&tshess->x0);CHKERRQ(ierr);
  }
  ierr = VecCopy(X,tshess->x0);CHKERRQ(ierr);
  if (!tshess->design) {
    ierr = VecDuplicate(design,&tshess->design);CHKERRQ(ierr);
  }
  ierr = VecCopy(design,tshess->design);CHKERRQ(ierr);
  tshess->t0 = t0;
  tshess->dt = dt;
  tshess->tf = tf;

  /* tangent linear model solver */
  if (!tshess->tlmts) {
    const char* prefix;
    char        *prefix_cp;

    ierr = TSCreateTLMTS(tshess->model,&tshess->tlmts);CHKERRQ(ierr);
    ierr = TSGetOptionsPrefix(tshess->tlmts,&prefix);CHKERRQ(ierr);
    ierr = PetscStrallocpy(prefix,&prefix_cp);CHKERRQ(ierr);
    ierr = TSSetOptionsPrefix(tshess->tlmts,"hessian_");CHKERRQ(ierr);
    ierr = TSAppendOptionsPrefix(tshess->tlmts,prefix_cp);CHKERRQ(ierr);
    ierr = PetscFree(prefix_cp);CHKERRQ(ierr);
    ierr = TSSetFromOptions(tshess->tlmts);CHKERRQ(ierr);
    ierr = TSSetTime(tshess->tlmts,tshess->t0);CHKERRQ(ierr);
    ierr = TSSetMaxTime(tshess->tlmts,tshess->tf);CHKERRQ(ierr);
    ierr = TSSetMaxSteps(tshess->tlmts,PETSC_MAX_INT);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(tshess->tlmts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  }
  ierr = TLMTSSetDesign(tshess->tlmts,design);CHKERRQ(ierr);

  /* first-order adjoint solver */
  if (!tshess->foats) {
    const char* prefix;
    char        *prefix_cp;

    ierr = TSCreateAdjointTS(tshess->model,&tshess->foats);CHKERRQ(ierr);
    ierr = TSGetOptionsPrefix(tshess->foats,&prefix);CHKERRQ(ierr);
    ierr = PetscStrallocpy(prefix,&prefix_cp);CHKERRQ(ierr);
    ierr = TSSetOptionsPrefix(tshess->foats,"hessian_fo");CHKERRQ(ierr);
    ierr = TSAppendOptionsPrefix(tshess->foats,prefix_cp);CHKERRQ(ierr);
    ierr = PetscFree(prefix_cp);CHKERRQ(ierr);
    ierr = AdjointTSSetTimeLimits(tshess->foats,t0,tf);CHKERRQ(ierr);
    ierr = AdjointTSEventHandler(tshess->foats);CHKERRQ(ierr);
    ierr = TSSetFromOptions(tshess->foats);CHKERRQ(ierr);
  }
  ierr = AdjointTSSetTimeLimits(tshess->foats,t0,tf);CHKERRQ(ierr);
  ierr = AdjointTSSetDesign(tshess->foats,design);CHKERRQ(ierr);
  ierr = AdjointTSSetInitialGradient(tshess->foats,NULL);CHKERRQ(ierr);

  /* second-order adjoint solver */
  if (!tshess->soats) {
    const char* prefix;
    char        *prefix_cp;

    ierr = TSCreateAdjointTS(tshess->model,&tshess->soats);CHKERRQ(ierr);
    ierr = TSGetOptionsPrefix(tshess->soats,&prefix);CHKERRQ(ierr);
    ierr = PetscStrallocpy(prefix,&prefix_cp);CHKERRQ(ierr);
    ierr = TSSetOptionsPrefix(tshess->soats,"hessian_so");CHKERRQ(ierr);
    ierr = TSAppendOptionsPrefix(tshess->soats,prefix_cp);CHKERRQ(ierr);
    ierr = PetscFree(prefix_cp);CHKERRQ(ierr);
    ierr = AdjointTSSetTimeLimits(tshess->soats,t0,tf);CHKERRQ(ierr);
    ierr = AdjointTSEventHandler(tshess->soats);CHKERRQ(ierr);
    ierr = TSSetFromOptions(tshess->soats);CHKERRQ(ierr);
  }
  ierr = AdjointTSSetDesign(tshess->soats,design);CHKERRQ(ierr);
  ierr = AdjointTSSetInitialGradient(tshess->soats,NULL);CHKERRQ(ierr);

  /* sample nonlinear model */
  otrj = ts->trajectory;
  ierr = TSTrajectoryCreate(PetscObjectComm((PetscObject)ts),&ts->trajectory);CHKERRQ(ierr);
  ierr = TSTrajectorySetType(ts->trajectory,ts,TSTRAJECTORYMEMORY);CHKERRQ(ierr);
  ierr = TSTrajectorySetSolutionOnly(ts->trajectory,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSTrajectorySetFromOptions(ts->trajectory,ts);CHKERRQ(ierr);
  ts->trajectory->adjoint_solve_mode = PETSC_FALSE;
  ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
  ierr = TSRestartStep(ts);CHKERRQ(ierr);
  ierr = TSSetTime(ts,tshess->t0);CHKERRQ(ierr);
  if (tshess->dt > 0) {
    ierr = TSSetTimeStep(ts,tshess->dt);CHKERRQ(ierr);
  }
  ierr = TSSetMaxTime(ts,tshess->tf);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,PETSC_MAX_INT);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  if (!U) {
    ierr = VecDuplicate(tshess->x0,&U);CHKERRQ(ierr);
    ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)U);CHKERRQ(ierr);
  }
  ierr = VecCopy(tshess->x0,U);CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);
  ierr = TSTrajectoryDestroy(&tshess->modeltj);CHKERRQ(ierr);
  tshess->modeltj = ts->trajectory;

  /* sample first-order adjoint */
  ierr = TSTrajectoryDestroy(&tshess->foats->trajectory);CHKERRQ(ierr); /* XXX add Reset method to TSTrajectory */
  ierr = TSTrajectoryCreate(PetscObjectComm((PetscObject)tshess->foats),&tshess->foats->trajectory);CHKERRQ(ierr);
  ierr = TSTrajectorySetType(tshess->foats->trajectory,tshess->foats,TSTRAJECTORYMEMORY);CHKERRQ(ierr);
  ierr = TSTrajectorySetSolutionOnly(tshess->foats->trajectory,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSTrajectorySetFromOptions(tshess->foats->trajectory,tshess->foats);CHKERRQ(ierr);
  tshess->foats->trajectory->adjoint_solve_mode = PETSC_FALSE;
  ierr = TSGetSolution(tshess->foats,&L);CHKERRQ(ierr);
  if (!L) {
    ierr = VecDuplicate(tshess->x0,&L);CHKERRQ(ierr);
    ierr = TSSetSolution(tshess->foats,L);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)L);CHKERRQ(ierr);
  }
  ierr = TSSetStepNumber(tshess->foats,0);CHKERRQ(ierr);
  ierr = TSRestartStep(tshess->foats);CHKERRQ(ierr);
  ierr = TSHistoryGetTimeStep(tshess->modeltj->tsh,PETSC_TRUE,0,&dt);CHKERRQ(ierr);
  ierr = TSSetTimeStep(tshess->foats,dt);CHKERRQ(ierr);
  ierr = AdjointTSComputeInitialConditions(tshess->foats,tshess->t0,NULL,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
  ierr = TSSolve(tshess->foats,L);CHKERRQ(ierr);

  /* restore old TSTrajectory (if any) */
  ts->trajectory = otrj;
  ierr = MatSetUp(H);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------ Public API ----------------------- */

/*@
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

$           argmax ||P du_T||^2
$          ||du_0||=1

          when one can be interested in the norm of the final state in a subspace.
          The projector is applied (via MatMult) on the final state computed by the forward Tangent Linear Model.
          The transposed action of P is instead used to initialize the adjoint of the Tangent Linear Model.
          Note that the role of P is somewhat different from that of the matrix representing the norm in the state variables.
          If P is provided, the row layout of A is the same of that of P. Otherwise, it is the same of that of x0.
          The column layout of A is the same of that of the design vector. If the latter is not provided, it is inherited from x0.
          Note that the column layout of P should be compatible with that of x0.

   Level: developer

.seealso: TSSetGradientDAE(), TSSetGradientIC()
@*/
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

/*@C
   TSSetGradientDAE - Sets the callback for the evaluation of the Jacobian matrix F_m(t,x(t),x_t(t);m) of a parameter dependent DAE.

   Logically Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  J   - the Mat object to hold F_m(t,x(t),x_t(t);m)
.  f   - the function evaluation routine
-  ctx - user-defined context for the function evaluation routine (can be NULL)

   Calling sequence of f:
$  f(TS ts,PetscReal t,Vec u,Vec u_t,Vec m,Mat J,void *ctx);

+  t   - time at step/stage being solved
.  u   - state vector
.  u_t - time derivative of state vector
.  m   - design vector
.  J   - the jacobian
-  ctx - [optional] user-defined context

   Notes: The ij entry of F_m is given by \frac{\partial F_i}{\partial m_j}, where F_i is the i-th component of the DAE and m_j the j-th design variable.
          The row and column layouts of the J matrix have to be compatible with those of the state and design vector, respectively.
          The matrix doesn't need to be in assembled form. For propagator computations, J needs to implement MatMult() and MatMultTranspose().
          For gradient and Hessian computations, both MatMult() and MatMultTranspose() need to be implemented.
          Pass NULL for J if you want to cancel the DAE dependence on the parameters.

   Level: advanced

.seealso: TSSetObjective(), TSSetHessianDAE(), TSComputeObjectiveAndGradient(), TSSetGradientIC(), TSSetHessianIC(), TSCreatePropagatorMat()
@*/
PetscErrorCode TSSetGradientDAE(TS ts, Mat J, TSEvalGradientDAE f, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (J) {
    PetscValidHeaderSpecific(J,MAT_CLASSID,2);
    PetscCheckSameComm(ts,1,J,2);
    ierr = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
  }
  ierr        = MatDestroy(&ts->F_m);CHKERRQ(ierr);
  ts->F_m     = J;
  ts->F_m_f   = f;
  ts->F_m_ctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   TSSetHessianDAE - Sets the callbacks for the evaluation of Hessian terms of a parameter dependent DAE.

   Logically Collective on TS

   Input Parameters:
+  ts     - the TS context obtained from TSCreate()
.  f_xx   - the function evaluation routine for second order state derivative
.  f_xxt  - the function evaluation routine for second order mixed x,x_t derivative
.  f_xm   - the function evaluation routine for second order mixed state and parameter derivative
.  f_xtx  - the function evaluation routine for second order mixed x_t,x derivative
.  f_xtxt - the function evaluation routine for second order x_t,x_t derivative
.  f_xtm  - the function evaluation routine for second order mixed x_t and parameter derivative
.  f_mx   - the function evaluation routine for second order mixed m,x derivative
.  f_mxt  - the function evaluation routine for second order mixed m,x_t derivative
.  f_mm   - the function evaluation routine for second order parameter derivative
-  ctx    - user-defined context for the function evaluation routines (can be NULL)

   Calling sequence of each function evaluation routine:
$  f(TS ts,PetscReal t,Vec u,Vec u_t,Vec m,Vec L,Vec X,Vec Y,void *ctx);

+  t   - time at step/stage being solved
.  u   - state vector
.  u_t - time derivative of state vector
.  m   - design vector
.  L   - input vector (adjoint variable)
.  X   - input vector (state or parameter variable)
.  Y   - output vector (state or parameter variable)
-  ctx - [optional] user-defined context

   Notes: the callbacks need to return

$  f_xx   : Y = (L^T \otimes I_N)*F_UU*X
$  f_xxt  : Y = (L^T \otimes I_N)*F_UUdot*X
$  f_xm   : Y = (L^T \otimes I_N)*F_UM*X
$  f_xtx  : Y = (L^T \otimes I_N)*F_UdotU*X
$  f_xtxt : Y = (L^T \otimes I_N)*F_UdotUdot*X
$  f_xtm  : Y = (L^T \otimes I_N)*F_UdotM*X
$  f_mx   : Y = (L^T \otimes I_P)*F_MU*X
$  f_mxt  : Y = (L^T \otimes I_P)*F_MUdot*X
$  f_mm   : Y = (L^T \otimes I_P)*F_MM*X

   where L is a vector of size N (the number of DAE equations), I_x the identity matrix of size x, \otimes is the Kronecker product, X an input vector of appropriate size, and F_AB an N*size(A) x size(B) matrix given as

$            | F^1_AB |
$     F_AB = |   ...  |, A = {U|Udot|M}, B = {U|Udot|M}.
$            | F^N_AB |

   Each F^k_AB block term has dimension size(A) x size(B), with {F^k_AB}_ij = \frac{\partial^2 F_k}{\partial b_j \partial a_i}, where F_k is the k-th component of the DAE, a_i the i-th variable of A and b_j the j-th variable of B.
   For example, {F^k_UM}_ij = \frac{\partial^2 F_k}{\partial m_j \partial u_i}.
   Developing the Kronecker product, we get Y = (\sum_k L_k*F^k_AB)*X, with L_k the k-th entry of the adjoint variable L.
   Pass NULL if F_AB is zero for some A and B.

   Level: advanced

.seealso: TSSetObjective(), TSSetGradientDAE(), TSComputeObjectiveAndGradient(), TSSetGradientIC(), TSSetHessianIC()
@*/
PetscErrorCode TSSetHessianDAE(TS ts, TSEvalHessianDAE f_xx,  TSEvalHessianDAE f_xxt,  TSEvalHessianDAE f_xm,
                                      TSEvalHessianDAE f_xtx, TSEvalHessianDAE f_xtxt, TSEvalHessianDAE f_xtm,
                                      TSEvalHessianDAE f_mx,  TSEvalHessianDAE f_mxt,  TSEvalHessianDAE f_mm, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->HF[0][0] = f_xx;
  ts->HF[0][1] = f_xxt;
  ts->HF[0][2] = f_xm;
  ts->HF[1][0] = f_xtx;
  ts->HF[1][1] = f_xtxt;
  ts->HF[1][2] = f_xtm;
  ts->HF[2][0] = f_mx;
  ts->HF[2][1] = f_mxt;
  ts->HF[2][2] = f_mm;
  ts->HFctx    = ctx;
  PetscFunctionReturn(0);
}

/*@C
   TSSetGradientIC - Sets the callback to compute the Jacobian matrices G_x(x0,m) and G_m(x0,m), with parameter dependent initial conditions implicitly defined by the function G(x(0),m) = 0.

   Logically Collective on TS

   Input Parameters:
+  ts  - the TS context obtained from TSCreate()
.  J_x - the Mat object to hold G_x(x0,m) (optional, if NULL identity is assumed)
.  J_m - the Mat object to hold G_m(x0,m)
.  f   - the function evaluation routine
-  ctx - user-defined context for the function evaluation routine (can be NULL)

   Calling sequence of f:
$  f(TS ts,PetscReal t,Vec u,Vec m,Mat Gx,Mat Gm,void *ctx);

+  t   - initial time
.  u   - state vector (at initial time)
.  m   - design vector
.  Gx  - the Mat object to hold the Jacobian wrt the state variables
.  Gm  - the Mat object to hold the Jacobian wrt the design variables
-  ctx - [optional] user-defined context

   Notes: J_x is a square matrix of the same size of the state vector. J_m is a rectangular matrix with "state size" rows and "design size" columns.
          If f is not provided, J_x is assumed constant. The J_m matrix doesn't need to assembled; only MatMult() and MatMultTranspose() are needed.
          Currently, the initial condition vector should be computed by the user.
          Pass NULL for J_m if you want to cancel the initial condition dependency from the parameters.

   Level: advanced

.seealso: TSSetObjective(), TSSetGradientDAE(), TSSetHessianDAE(), TSSetHessianIC(), TSComputeObjectiveAndGradient(), MATSHELL
@*/
PetscErrorCode TSSetGradientIC(TS ts, Mat J_x, Mat J_m, TSEvalGradientIC f, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (J_x) {
    PetscValidHeaderSpecific(J_x,MAT_CLASSID,2);
    PetscCheckSameComm(ts,1,J_x,2);
  }
  if (J_m) {
    PetscValidHeaderSpecific(J_m,MAT_CLASSID,3);
    PetscCheckSameComm(ts,1,J_m,3);
  } else J_x = NULL;
  ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradientIC_G",NULL);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradientIC_GW",NULL);CHKERRQ(ierr);
  if (J_x) {
    ierr = PetscObjectReference((PetscObject)J_x);CHKERRQ(ierr);
  }
  ierr    = MatDestroy(&ts->G_x);CHKERRQ(ierr);
  ts->G_x = J_x;
  if (J_m) {
    ierr = PetscObjectReference((PetscObject)J_m);CHKERRQ(ierr);
  }
  ierr          = MatDestroy(&ts->G_m);CHKERRQ(ierr);
  ts->G_m       = J_m;
  ts->Ggrad     = f;
  ts->Ggrad_ctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   TSSetHessianIC - Sets the callback to compute the action of the Hessian matrices G_xx(x0,m), G_xm(x0,m), G_mx(x0,m) and G_mm(x0,m), with parameter dependent initial conditions implicitly defined by the function G(x(0),m) = 0.

   Logically Collective on TS

   Input Parameters:
+  ts   - the TS context obtained from TSCreate()
.  g_xx - the function evaluation routine for second order state derivative
.  g_xm - the function evaluation routine for second order mixed x,m derivative
.  g_mx - the function evaluation routine for second order mixed m,x derivative
.  g_mm - the function evaluation routine for second order parameter derivative
-  ctx  - user-defined context for the function evaluation routines (can be NULL)

   Calling sequence of each function evaluation routine:
$  f(TS ts,PetscReal t,Vec u,Vec m,Vec L,Vec X,Vec Y,void *ctx);

+  t   - time at step/stage being solved
.  u   - state vector
.  m   - design vector
.  L   - input vector (adjoint variable)
.  X   - input vector (state or parameter variable)
.  Y   - output vector (state or parameter variable)
-  ctx - [optional] user-defined context

   Notes: the callbacks need to return

$  g_xx   : Y = (L^T \otimes I_N)*G_UU*X
$  g_xm   : Y = (L^T \otimes I_N)*G_UM*X
$  g_mx   : Y = (L^T \otimes I_P)*G_MU*X
$  g_mm   : Y = (L^T \otimes I_P)*G_MM*X

   where L is a vector of size N (the number of DAE equations), I_x the identity matrix of size x, \otimes is the Kronecker product, X an input vector of appropriate size, and G_AB an N*size(A) x size(B) matrix given as

$            | G^1_AB |
$     G_AB = |   ...  | , A = {U|M}, B = {U|M}.
$            | G^N_AB |

   Each G^k_AB block term has dimension size(A) x size(B), with {G^k_AB}_ij = \frac{\partial^2 G_k}{\partial b_j \partial a_i}, where G_k is the k-th component of the implicit function G that determines the initial conditions, a_i the i-th variable of A and b_j the j-th variable of B.
   For example, {G^k_UM}_ij = \frac{\partial^2 G_k}{\partial m_j \partial u_i}.
   Developing the Kronecker product, we get Y = (\sum_k L_k*G^k_AB)*X, with L_k the k-th entry of the adjoint variable L.
   Pass NULL if G_AB is zero for some A and B.

   Level: advanced

.seealso: TSSetObjective(), TSSetGradientDAE(), TSSetHessianDAE(), TSComputeObjectiveAndGradient(), TSSetGradientIC()
@*/
PetscErrorCode TSSetHessianIC(TS ts, TSEvalHessianIC g_xx,  TSEvalHessianIC g_xm,  TSEvalHessianIC g_mx, TSEvalHessianIC g_mm, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->HG[0][0] = g_xx;
  ts->HG[0][1] = g_xm;
  ts->HG[1][0] = g_mx;
  ts->HG[1][1] = g_mm;
  ts->HGctx    = ctx;
  PetscFunctionReturn(0);
}

/*@
   TSComputeObjectiveAndGradient - Evaluates the objective functions set with TSSetObjective and their gradient with respect the parameters.

   Logically Collective on TS

   Input Parameters:
+  ts     - the TS context
.  t0     - initial time
.  dt     - initial time step
.  tf     - final time
.  X      - the initial vector for the state (can be NULL)
-  design - current design vector

   Output Parameters:
+  gradient - the computed gradient
-  obj      - the value of the objective function

   Notes: If gradient is NULL, just a forward solve will be performed to compute the objective function. Otherwise, forward and backward solves are performed.
          The dt argument is ignored when smaller or equal to zero. If X is NULL, the initial state is given by the current TS solution vector.

   Level: advanced

.seealso: TSSetObjective(), TSSetGradientDAE(), TSSetHessianDAE(), TSSetGradientIC(), TSSetSolution()
@*/
PetscErrorCode TSComputeObjectiveAndGradient(TS ts, PetscReal t0, PetscReal dt, PetscReal tf, Vec X, Vec design, Vec gradient, PetscReal *obj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,t0,2);
  PetscValidLogicalCollectiveReal(ts,dt,3);
  PetscValidLogicalCollectiveReal(ts,tf,4);
  if (X) {
    PetscValidHeaderSpecific(X,VEC_CLASSID,5);
    PetscCheckSameComm(ts,1,X,5);
  }
  PetscValidHeaderSpecific(design,VEC_CLASSID,6);
  PetscCheckSameComm(ts,1,design,6);
  if (gradient) {
    PetscValidHeaderSpecific(gradient,VEC_CLASSID,7);
    PetscCheckSameComm(ts,1,gradient,7);
  }
  if (obj) PetscValidPointer(obj,8);
  if (!gradient && !obj) PetscFunctionReturn(0);

  ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
  ierr = TSRestartStep(ts);CHKERRQ(ierr);
  ierr = TSSetTime(ts,t0);CHKERRQ(ierr);
  if (dt > 0) {
    ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  }
  ierr = TSSetMaxTime(ts,tf);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,PETSC_MAX_INT);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = VecLockPush(design);CHKERRQ(ierr);
  ierr = TSComputeObjectiveAndGradient_Private(ts,X,design,gradient,obj);CHKERRQ(ierr);
  ierr = VecLockPop(design);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSComputeHessian - Computes the Hessian matrix with respect to the parameters for the objective functions set with TSSetObjective.

   Logically Collective on TS

   Input Parameters:
+  ts     - the TS context
.  t0     - initial time
.  dt     - initial time step
.  tf     - final time
.  X      - the initial vector for the state (can be NULL)
-  design - current design vector

   Output Parameters:
.  H - the Hessian matrix

   Notes: The Hessian matrix is not computed explictly; the only operation implemented for H is MatMult().
          The dt argument is ignored when smaller or equal to zero. If X is NULL, the initial state is given by the current TS solution vector.

   Level: advanced

.seealso: TSSetObjective(), TSSetGradientDAE(), TSSetHessianDAE(), TSSetGradientIC(), TSSetSolution()
@*/
PetscErrorCode TSComputeHessian(TS ts, PetscReal t0, PetscReal dt, PetscReal tf, Vec X, Vec design, Mat H)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveReal(ts,t0,2);
  PetscValidLogicalCollectiveReal(ts,dt,3);
  PetscValidLogicalCollectiveReal(ts,tf,4);
  if (X) {
    PetscValidHeaderSpecific(X,VEC_CLASSID,5);
    PetscCheckSameComm(ts,1,X,5);
  }
  PetscValidHeaderSpecific(design,VEC_CLASSID,6);
  PetscCheckSameComm(ts,1,design,6);
  PetscValidHeaderSpecific(H,MAT_CLASSID,7);
  PetscCheckSameComm(ts,1,H,7);
  ierr = TSComputeHessian_Private(ts,t0,dt,tf,X,design,H);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

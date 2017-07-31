#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/

typedef struct {
  PetscErrorCode (*user)(TS); /* user post step method */
  Vec            design;      /* the design vector we are evaluating against */
  PetscReal      pdt;         /* previous time step (for trapz rule) */
  PetscScalar    obj;         /* objective function value */
} TSGradientPostStepCtx;

static PetscErrorCode TSGradientEvalFunctionals(TS ts, PetscReal time, Vec state, Vec design, PetscScalar *val)
{
  PetscErrorCode     ierr;
  CostFunctionalLink link = ts->funchead;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(state,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidPointer(val,4);
  *val = 0.0;
  while (link) {
    PetscScalar v = 0.0;
    if (link->fixedtime == PETSC_MIN_REAL) {
      ierr = (*link->f)(ts,time,state,design,&v,link->f_ctx);CHKERRQ(ierr);
    }
    *val += v;
    link = link->next;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGradientPostStep(TS ts)
{
  PetscContainer        container;
  Vec                   solution;
  TSGradientPostStepCtx *poststep_ctx;
  PetscScalar           val;
  PetscReal             dt,time;
  PetscInt              step;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (ts->reason < 0) PetscFunctionReturn(0);
  ierr = PetscObjectQuery((PetscObject)ts,"_ts_gradient_poststep",(PetscObject*)&container);CHKERRQ(ierr);
  if (!container) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Missing container");
  ierr = PetscContainerGetPointer(container,(void**)&poststep_ctx);CHKERRQ(ierr);
  if (poststep_ctx->user) {
    ierr = (*poststep_ctx->user)(ts);CHKERRQ(ierr);
  }
  ierr = TSGetSolution(ts,&solution);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&time);CHKERRQ(ierr);
  ierr = TSGradientEvalFunctionals(ts,time,solution,poststep_ctx->design,&val);CHKERRQ(ierr);

  /* trapezoidal rule */
  if (ts->reason) { /* last step */
    dt = 0.0;
  } else {
    ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  }
  /* first step: obj has been initialized with the first function evaluation at t0 */
  ierr = TSGetTimeStepNumber(ts,&step);CHKERRQ(ierr);
  if (step == 1) poststep_ctx->obj *= dt/2.0;
  poststep_ctx->obj += (dt + poststep_ctx->pdt)*val/2.0;
  poststep_ctx->pdt = dt;
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
.  J       - the Mat object to hold F_m(t,x(t),x_t(t);m) (optional)
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

   Notes:

   Level: developer

.seealso: TSSetCostFunctional(), TSEvaluateGradient(), TSSetEvalICGradient()
*/
PetscErrorCode TSSetEvalGradient(TS ts, Mat J, TSEvalGradient f, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (J) {
    PetscValidHeaderSpecific(J,MAT_CLASSID,2);
    ierr    = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
    ierr    = MatDestroy(&ts->F_m);CHKERRQ(ierr);
    ts->F_m = J;
  }
  ts->F_m_f   = f;
  ts->F_m_ctx = ctx;
  PetscFunctionReturn(0);
}

/*
   TSSetEvalICGradient - Sets the callback function to compute the matrices g_x(x0,m) and g_m(x0,m), if there is any dependence of the PDE initial conditions from the design parameters.

   Logically Collective on TS

   Input Parameters:
+  ts      - the TS context obtained from TSCreate()
.  J_x     - the Mat object to hold g_x(x0,m) (optional)
.  J_m     - the Mat object to hold g_m(x0,m) (optional)
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

   Notes: Gx is a square matrix of the same size of the state vector. Gp is a rectangular matrix with "state size" rows and "design size" columns.

   Level: developer

.seealso: TSSetCostFunctional(), TSSetEvalGradient(), TSEvaluateGradient()
*/
PetscErrorCode TSSetEvalICGradient(TS ts, Mat J_x, Mat J_m, TSEvalICGradient f, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (J_x) {
    PetscValidHeaderSpecific(J_x,MAT_CLASSID,2);
    ierr    = PetscObjectReference((PetscObject)J_x);CHKERRQ(ierr);
    ierr    = MatDestroy(&ts->G_x);CHKERRQ(ierr);
    ts->G_x = J_x;
  }
  if (J_m) {
    PetscValidHeaderSpecific(J_m,MAT_CLASSID,3);
    ierr    = PetscObjectReference((PetscObject)J_m);CHKERRQ(ierr);
    ierr    = MatDestroy(&ts->G_m);CHKERRQ(ierr);
    ts->G_m = J_m;
  }
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

.seealso: TSSetCostFunctional(), TSSetEvalGradient(), TSSetEvalICGradient(), TSEvaluateGradient()
*/
PetscErrorCode TSEvaluateCostFunctionals(TS ts, Vec X, Vec design, PetscScalar *val)
{
  Vec                   U;
  PetscContainer        container;
  TSGradientPostStepCtx poststep_ctx;
  PetscReal             t0;
  PetscInt              tst;
  PetscBool             destroyX;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (X) PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(design,VEC_CLASSID,3);
  PetscValidPointer(val,4);
  if (!ts->funchead) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Cost functional are missing");

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
  poststep_ctx.user   = ts->poststep;
  poststep_ctx.design = design;
  poststep_ctx.pdt    = 0.0;
  poststep_ctx.obj    = 0.0;
  ierr = PetscContainerCreate(PetscObjectComm((PetscObject)ts),&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,(void*)&poststep_ctx);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradient_poststep",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  ierr = TSSetPostStep(ts,TSGradientPostStep);CHKERRQ(ierr);

  /* evaluate at initial time */
  ierr = TSGetTime(ts,&t0);CHKERRQ(ierr);
  ierr = TSGradientEvalFunctionals(ts,t0,X,design,&poststep_ctx.obj);CHKERRQ(ierr);

  /* forward solve */
  ierr = TSSetUp(ts);CHKERRQ(ierr);
  tst  = ts->total_steps;
  ts->total_steps = 0;
  ierr = TSSolve(ts,NULL);CHKERRQ(ierr);

  /* restore */
  ierr = PetscObjectCompose((PetscObject)ts,"_ts_gradient_poststep",NULL);CHKERRQ(ierr);
  ierr = TSSetPostStep(ts,poststep_ctx.user);CHKERRQ(ierr);
  if (U) {
    ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
  }
  ts->total_steps = tst;
  if (destroyX) {
    ierr = VecDestroy(&X);CHKERRQ(ierr);
  }

  /* get back value */
  *val = poststep_ctx.obj;
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

.seealso: TSSetCostFunctional(), TSSetEvalGradient(), TSSetEvalICGradient(), TSEvaluateCostFunctionals()
*/
PetscErrorCode TSEvaluateGradient(TS ts, Vec X, Vec design, Vec gradient)
{
  TSTrajectory   otrj;
  PetscReal      val;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(design,VEC_CLASSID,2);
  PetscValidHeaderSpecific(gradient,VEC_CLASSID,3);
  if (!ts->funchead) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Cost functional are missing");

  /* trajectory */
  otrj = ts->trajectory;
  ierr = TSTrajectoryCreate(PetscObjectComm((PetscObject)ts),&ts->trajectory);CHKERRQ(ierr);
  ierr = TSTrajectorySetFromOptions(ts->trajectory,ts);CHKERRQ(ierr);

  /* forward solve */
  ierr = TSEvaluateCostFunctionals(ts,X,design,&val);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Objective function inside gradient: val %g\n",(double)PetscRealPart(val));CHKERRQ(ierr);

  /* adjoint */
  //ierr = TSGradientCreateAdjointTS(ts,design,gradient,&adjts);CHKERRQ(ierr);
  //ierr = TSSolve(adjts,NULL);CHKERRQ(ierr);

  /* restore */
  ierr = TSTrajectoryDestroy(&ts->trajectory);CHKERRQ(ierr);
  ts->trajectory  = otrj;
  PetscFunctionReturn(0);
}

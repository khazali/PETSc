/*
  Code for timestepping with implicit Runge-Kutta method

  Notes:
  The general system is written as

  G(t,U,Udot) = 0

*/
#include <petsc-private/tsimpl.h>                /*I   "petscts.h"   I*/
#include <petscdm.h>

static TSIRKType      TSIRKDefault = TSIRKRADAU23;
static PetscBool      TSIRKRegisterAllCalled;
static PetscBool      TSIRKPackageInitialized;
static PetscInt       explicit_stage_time_id;
static PetscErrorCode TSExtrapolate_IRK(TS,PetscReal,Vec);

typedef struct _IRKTableau *IRKTableau;
struct _IRKTableau {
  char      *name;
  PetscInt  order;                /* Classical approximation order of the method */
  PetscInt  s;                    /* Number of stages */
  PetscInt  pinterp;              /* Interpolation order */
  PetscReal *A,*b,*c;             /* Non-stiff tableau */
  PetscReal *bembed;              /* Embedded formula of order one less (order-1) */
  PetscReal *binterp;             /* Dense output formula */
  PetscReal ccfl;                 /* Placeholder for CFL coefficient relative to forward Euler */
};
typedef struct _IRKTableauLink *IRKTableauLink;
struct _IRKTableauLink {
  struct _IRKTableau tab;
  IRKTableauLink     next;
};
static IRKTableauLink IRKTableauList;

typedef struct {
  IRKTableau   tableau;
  Vec          *Y;               /* States computed during the step */
  Vec          *Y_prev;          /* States computed during the previous time step */
  Vec          *Ydot_prev;       /* Time derivatives for the previous time step*/
  Vec          Ydot;             /* Work vector holding Ydot during residual evaluation */
  Vec          Work;             /* Generic work vector */
  Vec          Z;                /* Ydot = shift(Y-Z) */
  PetscScalar  *work;            /* Scalar work */
  PetscReal    scoeff;           /* shift = scoeff/dt */
  PetscReal    stage_time;
  PetscBool    init_guess_extrp; /* Extrapolate initial guess from previous time-step stage values */
  TSStepStatus status;
} TS_IRK;

/*MC
     TSIRKGAUSS12 - Second order, 1-stage Gauss method.

     Level: advanced

.seealso: TSIRK
M*/
/*MC
     TSIRKGAUSS24 - Fourth order, 2-stage Gauss method.

     Level: advanced

.seealso: TSIRK
M*/
/*MC
     TSIRKRADAU11 - First order, 1-stage Radau method.

     Level: advanced

.seealso: TSIRK
M*/
/*MC
     TSIRKRADAU23 - Third order, 2-stage Radau method.

     Level: advanced

.seealso: TSIRK
M*/
/*MC
     TSIRKLOBATTO22 - Second order, 2-stage Lobatto method.

     Level: advanced

.seealso: TSIRK
M*/
/*MC
     TSIRKLOBATTO34 - Fourth order, 3-stage Lobatto method.

     Level: advanced

.seealso: TSIRK
M*/

#undef __FUNCT__
#define __FUNCT__ "TSIRKRegisterAll"
/*@C
  TSIRKRegisterAll - Registers all of the implicit Runge-Kutta methods in TSIRK

  Not Collective, but should be called by all processes which will need the schemes to be registered

  Level: advanced

.keywords: TS, TSIRK, register, all

.seealso:  TSIRKRegisterDestroy()
@*/
PetscErrorCode TSIRKRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSIRKRegisterAllCalled) PetscFunctionReturn(0);
  TSIRKRegisterAllCalled = PETSC_TRUE;

  {
    const PetscReal
      A[3][3] = {{0.0,0.0,0.0},
                 {0.0,0.0,0.0},
                 {0.0,0.5,0.0}},
      At[3][3] = {{1.0,0.0,0.0},
                  {0.0,0.5,0.0},
                  {0.0,0.5,0.5}},
      b[3]       = {0.0,0.5,0.5},
      bembedt[3] = {1.0,0.0,0.0};
    ierr = TSIRKRegister(TSIRKGAUSS12,2,3,&At[0][0],b,NULL,&A[0][0],b,NULL,bembedt,bembedt,1,b,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSIRKRegisterDestroy"
/*@C
   TSIRKRegisterDestroy - Frees the list of schemes that were registered by TSIRKRegister().

   Not Collective

   Level: advanced

.keywords: TSIRK, register, destroy
.seealso: TSIRKRegister(), TSIRKRegisterAll()
@*/
PetscErrorCode TSIRKRegisterDestroy(void)
{
  PetscErrorCode ierr;
  IRKTableauLink link;

  PetscFunctionBegin;
  while ((link = IRKTableauList)) {
    IRKTableau t = &link->tab;
    IRKTableauList = link->next;
    ierr = PetscFree3(t->A,t->b,t->c);CHKERRQ(ierr);
    ierr = PetscFree(t->bembed);CHKERRQ(ierr);
    ierr = PetscFree(t->binterp);CHKERRQ(ierr);
    ierr = PetscFree(t->name);CHKERRQ(ierr);
    ierr = PetscFree(link);CHKERRQ(ierr);
  }
  TSIRKRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSIRKInitializePackage"
/*@C
  TSIRKInitializePackage - This function initializes everything in the TSIRK package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to TSCreate_IRK()
  when using static libraries.

  Level: developer

.keywords: TS, TSIRK, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode TSIRKInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSIRKPackageInitialized) PetscFunctionReturn(0);
  TSIRKPackageInitialized = PETSC_TRUE;
  ierr = TSIRKRegisterAll();CHKERRQ(ierr);
  ierr = PetscObjectComposedDataRegister(&explicit_stage_time_id);CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(TSIRKFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSIRKFinalizePackage"
/*@C
  TSIRKFinalizePackage - This function destroys everything in the TSIRK package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package
.seealso: PetscFinalize()
@*/
PetscErrorCode TSIRKFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TSIRKPackageInitialized = PETSC_FALSE;
  ierr = TSIRKRegisterDestroy();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSIRKRegister"
/*@C
   TSIRKRegister - register an IRK scheme by providing the entries in the Butcher tableau and optionally embedded approximations and interpolation

   Not Collective, but the same schemes should be registered on all processes on which they will be used

   Input Parameters:
+  name - identifier for method
.  order - approximation order of method
.  s - number of stages, this is the dimension of the matrices below
.  A - Stage coefficients (dimension s*s, row-major)
.  b - Step completion table (dimension s; NULL to use last row of At)
.  c - Abscissa (dimension s; NULL to use row sums of A)
.  bembed - Completion table for embedded method (dimension s)
.  pinterp - Order of the interpolation scheme, equal to the number of columns of binterp
-  binterp - Coefficients of the interpolation formula (dimension s*pinterp)

   Notes:
   Several IRK methods are provided, this function is only needed to create new methods.

   Level: advanced

.keywords: TS, register

.seealso: TSIRK
@*/
PetscErrorCode TSIRKRegister(TSIRKType name,PetscInt order,PetscInt s,
                             const PetscReal A[],const PetscReal b[],const PetscReal c[],
                             const PetscReal bembed[],PetscInt pinterp,const PetscReal binterp[])
{
  PetscErrorCode ierr;
  IRKTableauLink link;
  IRKTableau     t;
  PetscInt       i,j;

  PetscFunctionBegin;
  ierr     = PetscMalloc(sizeof(*link),&link);CHKERRQ(ierr);
  ierr     = PetscMemzero(link,sizeof(*link));CHKERRQ(ierr);
  t        = &link->tab;
  ierr     = PetscStrallocpy(name,&t->name);CHKERRQ(ierr);
  t->order = order;
  t->s     = s;
  ierr     = PetscMalloc3(s*s,PetscReal,&t->A,s,PetscReal,&t->b,s,PetscReal,&t->c);CHKERRQ(ierr);
  ierr     = PetscMemcpy(t->A,A,s*s*sizeof(A[0]));CHKERRQ(ierr);
  if (b)  { ierr = PetscMemcpy(t->b,b,s*sizeof(b[0]));CHKERRQ(ierr); }
  else for (i=0; i<s; i++) t->b[i] = A[(s-1)*s+i];
  if (c)  { ierr = PetscMemcpy(t->c,c,s*sizeof(c[0]));CHKERRQ(ierr); }
  else for (i=0; i<s; i++) for (j=0,t->c[i]=0; j<s; j++) t->c[i] += A[i*s+j];
  if (bembed) {
    ierr = PetscMalloc(s,PetscReal,&t->bembed);CHKERRQ(ierr);
    ierr = PetscMemcpy(t->bembed,bembed,s*sizeof(bembed[0]));CHKERRQ(ierr);
  }

  t->pinterp     = pinterp;
  ierr           = PetscMalloc(s*pinterp,PetscReal,&t->binterp);CHKERRQ(ierr);
  ierr           = PetscMemcpy(t->binterp,binterp,s*pinterp*sizeof(binterp[0]));CHKERRQ(ierr);
  link->next     = IRKTableauList;
  IRKTableauList = link;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSEvaluateStep_IRK"
/*
 The step completion formula is

 x1 = x0 - h b^T Ydot 

 This function can be called before or after ts->vec_sol has been updated.
 Suppose we have a completion formula (bt,b) and an embedded formula (bet,be) of different order.
 We can write

 x1e = x0 - h be^T Ydot
     = x1 + h b^T Ydot - h be^T Ydot
     = x1 - h (be - b)^T Ydot

 so we can evaluate the method with different order even after the step has been optimistically completed.
*/
static PetscErrorCode TSEvaluateStep_IRK(TS ts,PetscInt order,Vec X,PetscBool *done)
{
  TS_IRK          *irk = (TS_IRK*)ts->data;
  IRKTableau      tab  = irk->tableau;
  PetscScalar     *w   = irk->work;
  PetscReal       h;
  PetscInt        s = tab->s,j;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  switch (irk->status) {
  case TS_STEP_INCOMPLETE:
  case TS_STEP_PENDING:
    h = ts->time_step; break;
  case TS_STEP_COMPLETE:
    h = ts->time_step_prev; break;
  default: SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Invalid TSStepStatus");
  }
  if (order == tab->order) {
    if (irk->status == TS_STEP_INCOMPLETE) {
      ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);
      for (j=0; j<s; j++) w[j] = h*tab->b[j];
      ierr = VecMAXPY(X,s,w,irk->Ydot);CHKERRQ(ierr);
    } else {ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);}
    if (done) *done = PETSC_TRUE;
    PetscFunctionReturn(0);
  } else if (order == tab->order-1) {
    if (!tab->bembed) goto unavailable;
    if (irk->status == TS_STEP_INCOMPLETE) { /* Complete with the embedded method (be) */
      ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);
      for (j=0; j<s; j++) w[j] = h*tab->bembed[j];
      ierr = VecMAXPY(X,s,w,irk->Ydot);CHKERRQ(ierr);
    } else {                    /* Rollback and re-complete using (be-b) */
      ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);
      for (j=0; j<s; j++) w[j] = h*(tab->bembed[j] - tab->b[j]);
      ierr = VecMAXPY(X,tab->s,w,irk->Ydot);CHKERRQ(ierr);
    }
    if (done) *done = PETSC_TRUE;
    PetscFunctionReturn(0);
  }
unavailable:
  if (done) *done = PETSC_FALSE;
  else SETERRQ3(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"IRK '%s' of order %D cannot evaluate step at order %D",tab->name,tab->order,order);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSStep_IRK"
static PetscErrorCode TSStep_IRK(TS ts)
{
  TS_IRK          *irk = (TS_IRK*)ts->data;
  IRKTableau      tab  = irk->tableau;
  const PetscInt  s    = tab->s;
  const PetscReal *A   = tab->A,*b = tab->b,*c = tab->c;
  PetscScalar     *w   = irk->work;
  Vec             *Y   = irk->Y,*Ydot = irk->Ydot,W = irk->Work,Z = irk->Z;
  PetscBool       init_guess_extrp = irk->init_guess_extrp;
  TSAdapt         adapt;
  SNES            snes;
  PetscInt        i,j,its,lits,reject,next_scheme;
  PetscReal       next_time_step;
  PetscReal       t;
  PetscBool       accept;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  ierr           = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  next_time_step = ts->time_step;
  t              = ts->ptime;
  accept         = PETSC_TRUE;
  irk->status    = TS_STEP_INCOMPLETE;


  for (reject=0; reject<ts->max_reject && !ts->reason; reject++,ts->reject++) {
    PetscReal h = ts->time_step;
    ierr = TSPreStep(ts);CHKERRQ(ierr);
    for (i=0; i<s; i++) {
      irk->stage_time = t + h*c[i];
      ierr            = TSPreStage(ts,irk->stage_time);CHKERRQ(ierr);
      if (At[i*s+i] == 0) {           /* This stage is explicit */
        ierr = VecCopy(ts->vec_sol,Y[i]);CHKERRQ(ierr);
        for (j=0; j<i; j++) w[j] = h*A[i*s+j];
        ierr = VecMAXPY(Y[i],i,w,Ydot);CHKERRQ(ierr);
      } else {
        irk->scoeff     = 1./A[i*s+i];
        /* Affine part */
        ierr = VecZeroEntries(W);CHKERRQ(ierr);
        for (j=0; j<i; j++) w[j] = h*A[i*s+j];
        ierr = VecMAXPY(W,i,w,YdotRHS);CHKERRQ(ierr);
        ierr = VecScale(W, irk->scoeff/h);CHKERRQ(ierr);

        /* Ydot = shift*(Y-Z) */
        ierr = VecCopy(ts->vec_sol,Z);CHKERRQ(ierr);
        for (j=0; j<i; j++) w[j] = h*At[i*s+j];
        ierr = VecMAXPY(Z,i,w,Ydot);CHKERRQ(ierr);

        if (init_guess_extrp && ts->steps) {
          /* Initial guess extrapolated from previous time step stage values */
          ierr        = TSExtrapolate_IRK(ts,c[i],Y[i]);CHKERRQ(ierr);
        } else {
          /* Initial guess taken from last stage */
          ierr        = VecCopy(i>0 ? Y[i-1] : ts->vec_sol,Y[i]);CHKERRQ(ierr);
        }
        ierr          = SNESSolve(snes,W,Y[i]);CHKERRQ(ierr);
        ierr          = (ts->ops->snesfunction)(snes,Y[i],W,ts);CHKERRQ(ierr);
        ierr          = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
        ierr          = SNESGetLinearSolveIterations(snes,&lits);CHKERRQ(ierr);
        ts->snes_its += its; ts->ksp_its += lits;
        ierr          = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
        ierr          = TSAdaptCheckStage(adapt,ts,&accept);CHKERRQ(ierr);
        if (!accept) goto reject_step;
      }
      ierr = TSPostStage(ts,irk->stage_time,i,Y); CHKERRQ(ierr);
      if (ts->equation_type>=TS_EQ_IMPLICIT) {
        if (i==0 && tab->explicit_first_stage) {
          ierr = VecCopy(Ydot0,Ydot[0]);CHKERRQ(ierr);
        } else {
          ierr = VecAXPBYPCZ(Ydot[i],-irk->scoeff/h,irk->scoeff/h,0,Z,Y[i]);CHKERRQ(ierr); /* Ydot = shift*(X-Z) */
        }
      } else {
        ierr = VecZeroEntries(Ydot);CHKERRQ(ierr);
        ierr = TSComputeIFunction(ts,t+h*ct[i],Y[i],Ydot,Ydot[i],irk->imex);CHKERRQ(ierr);
        ierr = VecScale(Ydot[i], -1.0);CHKERRQ(ierr);
        if (irk->imex) {
          ierr = TSComputeRHSFunction(ts,t+h*c[i],Y[i],YdotRHS[i]);CHKERRQ(ierr);
        } else {
          ierr = VecZeroEntries(YdotRHS[i]);CHKERRQ(ierr);
        }
      }
    }
    ierr = TSEvaluateStep(ts,tab->order,ts->vec_sol,NULL);CHKERRQ(ierr);
    irk->status = TS_STEP_PENDING;

    /* Register only the current method as a candidate because we're not supporting multiple candidates yet. */
    ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
    ierr = TSAdaptCandidatesClear(adapt);CHKERRQ(ierr);
    ierr = TSAdaptCandidateAdd(adapt,tab->name,tab->order,1,tab->ccfl,1.*tab->s,PETSC_TRUE);CHKERRQ(ierr);
    ierr = TSAdaptChoose(adapt,ts,ts->time_step,&next_scheme,&next_time_step,&accept);CHKERRQ(ierr);
    if (accept) {
      /* ignore next_scheme for now */
      ts->ptime    += ts->time_step;
      ts->time_step = next_time_step;
      ts->steps++;
      irk->status = TS_STEP_COMPLETE;
      /* Save the Y, Ydot for extrapolation initial guess */
      if (irk->init_guess_extrp) {
        for (i = 0; i<s; i++) {
          ierr = VecCopy(Y[i],irk->Y_prev[i]);CHKERRQ(ierr);
          ierr = VecCopy(Ydot[i],irk->Ydot_prev[i]);CHKERRQ(ierr);
        }
      }
      break;
    } else {                    /* Roll back the current step */
      for (j=0; j<s; j++) w[j] = -h*b[j];
      ierr = VecMAXPY(ts->vec_sol,s,w,irk->Ydot);CHKERRQ(ierr);
      ts->time_step = next_time_step;
      irk->status   = TS_STEP_INCOMPLETE;
    }
reject_step: continue;
  }
  if (irk->status != TS_STEP_COMPLETE && !ts->reason) ts->reason = TS_DIVERGED_STEP_REJECTED;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSInterpolate_IRK"
static PetscErrorCode TSInterpolate_IRK(TS ts,PetscReal itime,Vec X)
{
  TS_IRK          *irk = (TS_IRK*)ts->data;
  PetscInt        s    = irk->tableau->s,pinterp = irk->tableau->pinterp,i,j;
  PetscReal       h;
  PetscReal       tt,t;
  PetscScalar     *b;
  const PetscReal *B = irk->tableau->binterp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!B) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSIRK %s does not have an interpolation formula",irk->tableau->name);
  switch (irk->status) {
  case TS_STEP_INCOMPLETE:
  case TS_STEP_PENDING:
    h = ts->time_step;
    t = (itime - ts->ptime)/h;
    break;
  case TS_STEP_COMPLETE:
    h = ts->time_step_prev;
    t = (itime - ts->ptime)/h + 1; /* In the interval [0,1] */
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Invalid TSStepStatus");
  }
  ierr = PetscMalloc(s,PetscScalar,&b);CHKERRQ(ierr);
  for (i=0; i<s; i++) b[i] = 0;
  for (j=0,tt=t; j<pinterp; j++,tt*=t) for (i=0; i<s; i++) b[i]  += h * B[i*pinterp+j] * tt;
  ierr = VecCopy(irk->Y[0],X);CHKERRQ(ierr);
  ierr = VecMAXPY(X,s,b,irk->Ydot);CHKERRQ(ierr);
  ierr = PetscFree(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSExtrapolate_IRK"
static PetscErrorCode TSExtrapolate_IRK(TS ts,PetscReal c,Vec X)
{
  TS_IRK          *irk = (TS_IRK*)ts->data;
  PetscInt        s    = irk->tableau->s,pinterp = irk->tableau->pinterp,i,j;
  PetscReal       h;
  PetscReal       tt,t;
  PetscScalar     *b;
  const PetscReal *B = irk->tableau->binterp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!B) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSIRK %s does not have an interpolation formula",irk->tableau->name);
  t = 1.0 + (ts->time_step/ts->time_step_prev)*c;
  h = ts->time_step;
  ierr = PetscMalloc(s,PetscScalar,&b);CHKERRQ(ierr);
  for (i=0; i<s; i++) b[i] = 0;
  for (j=0,tt=t; j<pinterp; j++,tt*=t) for (i=0; i<s; i++) b[i]  += h * B[i*pinterp+j] * tt;
  ierr = VecCopy(irk->Y_prev[0],X);CHKERRQ(ierr);
  ierr = VecMAXPY(X,s,b,irk->Ydot_prev);CHKERRQ(ierr);
  ierr = PetscFree(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "TSReset_IRK"
static PetscErrorCode TSReset_IRK(TS ts)
{
  TS_IRK          *irk = (TS_IRK*)ts->data;
  PetscInt        s;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!irk->tableau) PetscFunctionReturn(0);
  s    = irk->tableau->s;
  ierr = VecDestroyVecs(s,&irk->Y);CHKERRQ(ierr);
  ierr = VecDestroyVecs(s,&irk->Ydot);CHKERRQ(ierr);
  if (&irk->init_guess_extrp) {
    ierr = VecDestroyVecs(s,&irk->Y_prev);CHKERRQ(ierr);
    ierr = VecDestroyVecs(s,&irk->Ydot_prev);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&irk->Work);CHKERRQ(ierr);
  ierr = VecDestroy(&irk->Z);CHKERRQ(ierr);
  ierr = PetscFree(irk->work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDestroy_IRK"
static PetscErrorCode TSDestroy_IRK(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_IRK(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSIRKGetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSIRKSetType_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSIRKGetVecs"
static PetscErrorCode TSIRKGetVecs(TS ts,DM dm,Vec *Z,Vec *Ydot)
{
  TS_IRK     *ax = (TS_IRK*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (Z) {
    if (dm && dm != ts->dm) {
      ierr = DMGetNamedGlobalVector(dm,"TSIRK_Z",Z);CHKERRQ(ierr);
    } else *Z = ax->Z;
  }
  if (Ydot) {
    if (dm && dm != ts->dm) {
      ierr = DMGetNamedGlobalVector(dm,"TSIRK_Ydot",Ydot);CHKERRQ(ierr);
    } else *Ydot = ax->Ydot;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSIRKRestoreVecs"
static PetscErrorCode TSIRKRestoreVecs(TS ts,DM dm,Vec *Z,Vec *Ydot)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (Z) {
    if (dm && dm != ts->dm) {
      ierr = DMRestoreNamedGlobalVector(dm,"TSIRK_Z",Z);CHKERRQ(ierr);
    }
  }
  if (Ydot) {
    if (dm && dm != ts->dm) {
      ierr = DMRestoreNamedGlobalVector(dm,"TSIRK_Ydot",Ydot);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
  This defines the nonlinear equation that is to be solved with SNES
  G(U) = F[t0+Theta*dt, U, (U-U0)*shift] = 0
*/
#undef __FUNCT__
#define __FUNCT__ "SNESTSFormFunction_IRK"
static PetscErrorCode SNESTSFormFunction_IRK(SNES snes,Vec X,Vec F,TS ts)
{
  TS_IRK          *irk = (TS_IRK*)ts->data;
  DM              dm,dmsave;
  Vec             Z,Ydot;
  PetscReal       shift = irk->scoeff / ts->time_step;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr   = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr   = TSIRKGetVecs(ts,dm,&Z,&Ydot);CHKERRQ(ierr);
  ierr   = VecAXPBYPCZ(Ydot,-shift,shift,0,Z,X);CHKERRQ(ierr); /* Ydot = shift*(X-Z) */
  dmsave = ts->dm;
  ts->dm = dm;

  ierr = TSComputeIFunction(ts,irk->stage_time,X,Ydot,F,irk->imex);CHKERRQ(ierr);

  ts->dm = dmsave;
  ierr   = TSIRKRestoreVecs(ts,dm,&Z,&Ydot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormJacobian_IRK"
static PetscErrorCode SNESTSFormJacobian_IRK(SNES snes,Vec X,Mat *A,Mat *B,MatStructure *str,TS ts)
{
  TS_IRK         *irk = (TS_IRK*)ts->data;
  DM             dm,dmsave;
  Vec            Ydot;
  PetscReal      shift = irk->scoeff / ts->time_step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = TSIRKGetVecs(ts,dm,NULL,&Ydot);CHKERRQ(ierr);
  /* irk->Ydot has already been computed in SNESTSFormFunction_IRK (SNES guarantees this) */
  dmsave = ts->dm;
  ts->dm = dm;

  ierr = TSComputeIJacobian(ts,irk->stage_time,X,Ydot,shift,A,B,str,irk->imex);CHKERRQ(ierr);

  ts->dm = dmsave;
  ierr   = TSIRKRestoreVecs(ts,dm,NULL,&Ydot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCoarsenHook_TSIRK"
static PetscErrorCode DMCoarsenHook_TSIRK(DM fine,DM coarse,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRestrictHook_TSIRK"
static PetscErrorCode DMRestrictHook_TSIRK(DM fine,Mat restrct,Vec rscale,Mat inject,DM coarse,void *ctx)
{
  TS             ts = (TS)ctx;
  PetscErrorCode ierr;
  Vec            Z,Z_c;

  PetscFunctionBegin;
  ierr = TSIRKGetVecs(ts,fine,&Z,NULL);CHKERRQ(ierr);
  ierr = TSIRKGetVecs(ts,coarse,&Z_c,NULL);CHKERRQ(ierr);
  ierr = MatRestrict(restrct,Z,Z_c);CHKERRQ(ierr);
  ierr = VecPointwiseMult(Z_c,rscale,Z_c);CHKERRQ(ierr);
  ierr = TSIRKRestoreVecs(ts,fine,&Z,NULL);CHKERRQ(ierr);
  ierr = TSIRKRestoreVecs(ts,coarse,&Z_c,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMSubDomainHook_TSIRK"
static PetscErrorCode DMSubDomainHook_TSIRK(DM dm,DM subdm,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSubDomainRestrictHook_TSIRK"
static PetscErrorCode DMSubDomainRestrictHook_TSIRK(DM dm,VecScatter gscat,VecScatter lscat,DM subdm,void *ctx)
{
  TS             ts = (TS)ctx;
  PetscErrorCode ierr;
  Vec            Z,Z_c;

  PetscFunctionBegin;
  ierr = TSIRKGetVecs(ts,dm,&Z,NULL);CHKERRQ(ierr);
  ierr = TSIRKGetVecs(ts,subdm,&Z_c,NULL);CHKERRQ(ierr);

  ierr = VecScatterBegin(gscat,Z,Z_c,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(gscat,Z,Z_c,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = TSIRKRestoreVecs(ts,dm,&Z,NULL);CHKERRQ(ierr);
  ierr = TSIRKRestoreVecs(ts,subdm,&Z_c,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetUp_IRK"
static PetscErrorCode TSSetUp_IRK(TS ts)
{
  TS_IRK          *irk = (TS_IRK*)ts->data;
  IRKTableau      tab;
  PetscInt        s;
  PetscErrorCode  ierr;
  DM              dm;

  PetscFunctionBegin;
  if (!irk->tableau) ierr = TSIRKSetType(ts,TSIRKDefault);CHKERRQ(ierr);
  tab  = irk->tableau;
  s    = tab->s;
  ierr = VecDuplicateVecs(ts->vec_sol,s,&irk->Y);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ts->vec_sol,s,&irk->Ydot);CHKERRQ(ierr);
  if (irk->init_guess_extrp) {
    ierr = VecDuplicateVecs(ts->vec_sol,s,&irk->Y_prev);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(ts->vec_sol,s,&irk->Ydot_prev);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(ts->vec_sol,&irk->Work);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&irk->Z);CHKERRQ(ierr);
  ierr = PetscMalloc(s*sizeof(irk->work[0]),&irk->work);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  if (dm) {
    ierr = DMCoarsenHookAdd(dm,DMCoarsenHook_TSIRK,DMRestrictHook_TSIRK,ts);CHKERRQ(ierr);
    ierr = DMSubDomainHookAdd(dm,DMSubDomainHook_TSIRK,DMSubDomainRestrictHook_TSIRK,ts);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_IRK"
static PetscErrorCode TSSetFromOptions_IRK(TS ts)
{
  TS_IRK          *irk = (TS_IRK*)ts->data;
  PetscErrorCode  ierr;
  char            irktype[256];

  PetscFunctionBegin;
  ierr = PetscOptionsHead("IRK ODE solver options");CHKERRQ(ierr);
  {
    IRKTableauLink link;
    PetscInt       count,choice;
    PetscBool      flg;
    const char     **namelist;
    ierr = PetscStrncpy(irktype,TSIRKDefault,sizeof(irktype));CHKERRQ(ierr);
    for (link=IRKTableauList,count=0; link; link=link->next,count++) ;
    ierr = PetscMalloc(count*sizeof(char*),&namelist);CHKERRQ(ierr);
    for (link=IRKTableauList,count=0; link; link=link->next,count++) namelist[count] = link->tab.name;
    ierr      = PetscOptionsEList("-ts_irk_type","Family of IRK method","TSIRKSetType",(const char*const*)namelist,count,irktype,&choice,&flg);CHKERRQ(ierr);
    ierr      = TSIRKSetType(ts,flg ? namelist[choice] : irktype);CHKERRQ(ierr);
    ierr      = PetscFree(namelist);CHKERRQ(ierr);
    irk->init_guess_extrp = PETSC_FALSE;
    ierr      = PetscOptionsBool("-ts_irk_initial_guess_extrapolate","Extrapolate the initial guess for the stage solution from stage values of the previous time step","",irk->init_guess_extrp,&irk->init_guess_extrp,NULL);CHKERRQ(ierr);
    ierr      = SNESSetFromOptions(ts->snes);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFormatRealArray"
static PetscErrorCode PetscFormatRealArray(char buf[],size_t len,const char *fmt,PetscInt n,const PetscReal x[])
{
  PetscErrorCode ierr;
  PetscInt       i;
  size_t         left,count;
  char           *p;

  PetscFunctionBegin;
  for (i=0,p=buf,left=len; i<n; i++) {
    ierr = PetscSNPrintfCount(p,left,fmt,&count,x[i]);CHKERRQ(ierr);
    if (count >= left) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Insufficient space in buffer");
    left -= count;
    p    += count;
    *p++  = ' ';
  }
  p[i ? 0 : -1] = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSView_IRK"
static PetscErrorCode TSView_IRK(TS ts,PetscViewer viewer)
{
  TS_IRK          *irk = (TS_IRK*)ts->data;
  IRKTableau      tab  = irk->tableau;
  PetscBool       iascii;
  PetscErrorCode  ierr;
  TSAdapt         adapt;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    TSIRKType irktype;
    char          buf[512];
    ierr = TSIRKGetType(ts,&irktype);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  IRK %s\n",irktype);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Abscissa     c = %s\n",buf);CHKERRQ(ierr);
    ierr = PetscFormatRealArray(buf,sizeof(buf),"% 8.6f",tab->s,tab->c);CHKERRQ(ierr);
  }
  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptView(adapt,viewer);CHKERRQ(ierr);
  ierr = SNESView(ts->snes,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSLoad_IRK"
static PetscErrorCode TSLoad_IRK(TS ts,PetscViewer viewer)
{
  PetscErrorCode ierr;
  SNES           snes;
  TSAdapt        tsadapt;

  PetscFunctionBegin;
  ierr = TSGetAdapt(ts,&tsadapt);CHKERRQ(ierr);
  ierr = TSAdaptLoad(tsadapt,viewer);CHKERRQ(ierr);
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESLoad(snes,viewer);CHKERRQ(ierr);
  /* function and Jacobian context for SNES when used with TS is always ts object */
  ierr = SNESSetFunction(snes,NULL,NULL,ts);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,NULL,NULL,NULL,ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSIRKSetType"
/*@C
  TSIRKSetType - Set the type of IRK scheme

  Logically collective

  Input Parameter:
+  ts - timestepping context
-  irktype - type of IRK-scheme

  Level: intermediate

.seealso: TSIRKGetType(), TSIRK, TSIRKGAUSS12, TSIRKGAUSS24, TSIRKRADAU11, TSIRKRADAU23, TSIRKLOBATTO22, TSIRKLOBATTO34
@*/
PetscErrorCode TSIRKSetType(TS ts,TSIRKType irktype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSIRKSetType_C",(TS,TSIRKType),(ts,irktype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSIRKGetType"
/*@C
  TSIRKGetType - Get the type of IRK scheme

  Logically collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  irktype - type of IRK-scheme

  Level: intermediate

.seealso: TSIRKSetType()
@*/
PetscErrorCode TSIRKGetType(TS ts,TSIRKType *irktype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscUseMethod(ts,"TSIRKGetType_C",(TS,TSIRKType*),(ts,irktype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSIRKGetType_IRK"
PetscErrorCode  TSIRKGetType_IRK(TS ts,TSIRKType *irktype)
{
  TS_IRK     *irk = (TS_IRK*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!irk->tableau) ierr = TSIRKSetType(ts,TSIRKDefault);CHKERRQ(ierr);
  *irktype = irk->tableau->name;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "TSIRKSetType_IRK"
PetscErrorCode  TSIRKSetType_IRK(TS ts,TSIRKType irktype)
{
  TS_IRK     *irk = (TS_IRK*)ts->data;
  PetscErrorCode ierr;
  PetscBool      match;
  IRKTableauLink link;

  PetscFunctionBegin;
  if (irk->tableau) {
    ierr = PetscStrcmp(irk->tableau->name,irktype,&match);CHKERRQ(ierr);
    if (match) PetscFunctionReturn(0);
  }
  for (link = IRKTableauList; link; link=link->next) {
    ierr = PetscStrcmp(link->tab.name,irktype,&match);CHKERRQ(ierr);
    if (match) {
      ierr = TSReset_IRK(ts);CHKERRQ(ierr);
      irk->tableau = &link->tab;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_UNKNOWN_TYPE,"Could not find '%s'",irktype);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
/*MC
      TSIRK - ODE and DAE solver using implicit Runge-Kutta schemes

  Notes:
  The default is TSIRKGAUSS12, it can be changed with TSIRKSetType() or -ts_irk_type

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSIRKSetType(), TSIRKGetType(), TSIRKSetFullyImplicit(), TSIRKGAUSS12, TSIRKGAUSS24, 
           TSIRKRADAU11, TSIRKRADAU23, TSIRKLOBATTO22, TSIRKLOBATTO34

M*/
#undef __FUNCT__
#define __FUNCT__ "TSCreate_IRK"
PETSC_EXTERN PetscErrorCode TSCreate_IRK(TS ts)
{
  TS_IRK          *th;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
  ierr = TSIRKInitializePackage();CHKERRQ(ierr);
#endif

  ts->ops->reset          = TSReset_IRK;
  ts->ops->destroy        = TSDestroy_IRK;
  ts->ops->view           = TSView_IRK;
  ts->ops->load           = TSLoad_IRK;
  ts->ops->setup          = TSSetUp_IRK;
  ts->ops->step           = TSStep_IRK;
  ts->ops->interpolate    = TSInterpolate_IRK;
  ts->ops->evaluatestep   = TSEvaluateStep_IRK;
  ts->ops->setfromoptions = TSSetFromOptions_IRK;
  ts->ops->snesfunction   = SNESTSFormFunction_IRK;
  ts->ops->snesjacobian   = SNESTSFormJacobian_IRK;

  ierr = PetscNewLog(ts,TS_IRK,&th);CHKERRQ(ierr);
  ts->data = (void*)th;

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSIRKGetType_C",TSIRKGetType_IRK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSIRKSetType_C",TSIRKSetType_IRK);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

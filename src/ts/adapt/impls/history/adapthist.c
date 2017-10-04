#include <petsc/private/tshistoryimpl.h>  /*I "petscts.h"  I*/

typedef struct {
  TSHistory hist;
  PetscBool bw;
} TSAdapt_History;

static PetscErrorCode TSAdaptChoose_History(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept,PetscReal *wlte,PetscReal *wltea,PetscReal *wlter)
{
  PetscErrorCode  ierr;
  PetscInt        step;
  TSAdapt_History *thadapt = (TSAdapt_History*)adapt->data;

  PetscFunctionBegin;
  if (!thadapt->hist) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_USER,"Need call TSAdaptHistorySetHistory()");
  ierr = TSGetStepNumber(ts,&step);CHKERRQ(ierr);
  ierr = TSHistoryGetTimeStep(thadapt->hist,thadapt->bw,step+1,next_h);CHKERRQ(ierr);
  *accept  = PETSC_TRUE;
  *next_sc = 0;
  *wlte    = -1;
  *wltea   = -1;
  *wlter   = -1;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdaptReset_History(TSAdapt adapt)
{
  TSAdapt_History *thadapt = (TSAdapt_History*)adapt->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = TSHistoryDestroy(&thadapt->hist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdaptDestroy_History(TSAdapt adapt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSAdaptReset_History(adapt);CHKERRQ(ierr);
  ierr = PetscFree(adapt->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* this is not public, as TSHistory is not a public object */
PetscErrorCode TSAdaptHistorySetTSHistory(TSAdapt adapt, TSHistory hist, PetscBool backward)
{
  TSAdapt_History *thadapt;
  PetscBool       flg;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidHeaderSpecific(hist,TSHISTORY_CLASSID,2);
  PetscValidLogicalCollectiveBool(adapt,backward,3);
  ierr = PetscObjectTypeCompare((PetscObject)adapt,TSADAPTHISTORY,&flg);CHKERRQ(ierr);
  if (!flg) PetscFunctionReturn(0);
  thadapt = (TSAdapt_History*)adapt->data;
  ierr = PetscObjectReference((PetscObject)hist);CHKERRQ(ierr);
  ierr = TSHistoryDestroy(&thadapt->hist);CHKERRQ(ierr);
  thadapt->hist = hist;
  thadapt->bw   = backward;
  PetscFunctionReturn(0);
}

/*@
   TSAdaptHistorySetHistory - Sets a time history in the adaptor

   Logically Collective on TSAdapt

   Input Parameters:
+  adapt    - the TSAdapt context
.  n        - size of the time history
.  hist     - the time history
-  backward - if the time history has to be followed backward

   Notes: The time history is internally copied, and the user can free the hist array.

   Level: advanced

.keywords: TSAdapt
.seealso: TSGetAdapt(), TSAdaptSetType(), TSADAPTHISTORY
@*/
PetscErrorCode TSAdaptHistorySetHistory(TSAdapt adapt, PetscInt n, PetscReal hist[], PetscBool backward)
{
  TSAdapt_History *thadapt;
  PetscBool       flg;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidLogicalCollectiveInt(adapt,n,2);
  PetscValidRealPointer(hist,3);
  PetscValidLogicalCollectiveBool(adapt,backward,4);
  ierr = PetscObjectTypeCompare((PetscObject)adapt,TSADAPTHISTORY,&flg);CHKERRQ(ierr);
  if (!flg) PetscFunctionReturn(0);
  thadapt = (TSAdapt_History*)adapt->data;
  ierr = TSHistoryDestroy(&thadapt->hist);CHKERRQ(ierr);
  ierr = TSHistoryCreate(PetscObjectComm((PetscObject)adapt),&thadapt->hist);CHKERRQ(ierr);
  ierr = TSHistorySetHistory(thadapt->hist,n,hist);CHKERRQ(ierr);
  thadapt->bw = backward;
  PetscFunctionReturn(0);
}

/*MC
   TSADAPTHISTORY - Time stepping controller that follows a given time history, used for Tangent Linear Model simulations

   Level: developer

.seealso: TS, TSAdapt, TSGetAdapt()
M*/
PETSC_EXTERN PetscErrorCode TSAdaptCreate_History(TSAdapt adapt)
{
  PetscErrorCode     ierr;
  TSAdapt_History *thadapt;

  PetscFunctionBegin;
  ierr = PetscNew(&thadapt);CHKERRQ(ierr);
  adapt->matchstepfac[0] = PETSC_SMALL; /* prevent from accumulation errors */
  adapt->matchstepfac[1] = 0.0;         /* we will always match the final step, prevent TSAdaptChoose to mess with it */
  adapt->data            = thadapt;

  adapt->ops->choose  = TSAdaptChoose_History;
  adapt->ops->reset   = TSAdaptReset_History;
  adapt->ops->destroy = TSAdaptDestroy_History;
  PetscFunctionReturn(0);
}

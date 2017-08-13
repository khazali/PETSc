#include <petsc/private/tsimpl.h> /*I "petscts.h" I*/

typedef struct {
  TSTrajectory tj;
  PetscBool    bw;
} TSAdapt_Trajectory;

static PetscErrorCode TSAdaptChoose_Trajectory(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept,PetscReal *wlte,PetscReal *wltea,PetscReal *wlter)
{
  PetscErrorCode     ierr;
  PetscInt           step;
  TSAdapt_Trajectory *tjadapt = (TSAdapt_Trajectory*)adapt->data;

  PetscFunctionBegin;
  if (!tjadapt->tj) SETERRQ(PetscObjectComm((PetscObject)adapt),PETSC_ERR_USER,"Need to attach a TSTrajectory object via TSAdaptTrajectorySetTrajectory()");
  ierr = TSGetStepNumber(ts,&step);CHKERRQ(ierr);
  ierr = TSHistoryGetTimeStep(tjadapt->tj->tsh,tjadapt->bw,step+1,next_h);CHKERRQ(ierr);
  *accept  = PETSC_TRUE;
  *next_sc = 0;
  *wlte    = -1;
  *wltea   = -1;
  *wlter   = -1;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdaptReset_Trajectory(TSAdapt adapt)
{
  TSAdapt_Trajectory  *tjadapt = (TSAdapt_Trajectory*)adapt->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSTrajectoryDestroy(&tjadapt->tj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdaptDestroy_Trajectory(TSAdapt adapt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSAdaptReset_Trajectory(adapt);CHKERRQ(ierr);
  ierr = PetscFree(adapt->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TSAdaptTrajectorySetTrajectory(TSAdapt adapt, TSTrajectory tj, PetscBool backward)
{
  TSAdapt_Trajectory *tjadapt;
  PetscBool          flg;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,2);
  PetscValidLogicalCollectiveBool(adapt,backward,3);
  ierr = PetscObjectTypeCompare((PetscObject)adapt,TSADAPTTRAJECTORY,&flg);CHKERRQ(ierr);
  if (!flg) PetscFunctionReturn(0); 
  tjadapt = (TSAdapt_Trajectory*)adapt->data;
  ierr = PetscObjectReference((PetscObject)tj);CHKERRQ(ierr);
  ierr = TSTrajectoryDestroy(&tjadapt->tj);CHKERRQ(ierr);
  tjadapt->tj = tj;
  tjadapt->bw = backward;
  PetscFunctionReturn(0);
}

/*MC
   TSADAPTTRAJECTORY - Time stepping controller that follows a given TSTrajectory, used for Tangent Linear Model simulations

   Level: developer

.seealso: TS, TSAdapt, TSSetAdapt()
M*/
PETSC_EXTERN PetscErrorCode TSAdaptCreate_Trajectory(TSAdapt adapt)
{
  PetscErrorCode     ierr;
  TSAdapt_Trajectory *tjadapt;

  PetscFunctionBegin;
  ierr = PetscNew(&tjadapt);CHKERRQ(ierr);
  adapt->matchstepfac[0] = PETSC_SMALL; /* prevent from accumulation errors */
  adapt->matchstepfac[1] = 0.0; /* we will always match the final step, prevent TSAdaptChoose to mess with it */
  adapt->data = tjadapt;
  adapt->ops->choose  = TSAdaptChoose_Trajectory;
  adapt->ops->reset   = TSAdaptReset_Trajectory;
  adapt->ops->destroy = TSAdaptDestroy_Trajectory;
  PetscFunctionReturn(0);
}

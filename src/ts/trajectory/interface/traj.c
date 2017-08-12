
#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/

PetscFunctionList TSTrajectoryList              = NULL;
PetscBool         TSTrajectoryRegisterAllCalled = PETSC_FALSE;
PetscClassId      TSTRAJECTORY_CLASSID;
PetscLogEvent     TSTrajectory_Set, TSTrajectory_Get, TSTrajectory_GetVecs;

/*@C
  TSTrajectoryRegister - Adds a way of storing trajectories to the TS package

  Not Collective

  Input Parameters:
+ name        - the name of a new user-defined creation routine
- create_func - the creation routine itself

  Notes:
  TSTrajectoryRegister() may be called multiple times to add several user-defined tses.

  Level: developer

.keywords: TS, trajectory, timestep, register

.seealso: TSTrajectoryRegisterAll()
@*/
PetscErrorCode TSTrajectoryRegister(const char sname[],PetscErrorCode (*function)(TSTrajectory,TS))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&TSTrajectoryList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  TSTrajectorySet - Sets a vector of state in the trajectory object

  Collective on TS

  Input Parameters:
+ tj      - the trajectory object
. ts      - the time stepper object (optional)
. stepnum - the step number
. time    - the current time
- X       - the current solution

  Level: developer

  Notes: Usually one does not call this routine, it is called automatically during TSSolve()

.keywords: TS, trajectory, create

.seealso: TSTrajectorySetUp(), TSTrajectoryDestroy(), TSTrajectorySetType(), TSTrajectorySetVariableNames(), TSGetTrajectory(), TSTrajectoryGet(), TSTrajectoryGetVecs()
*/
PetscErrorCode TSTrajectorySet(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tj) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  if (ts) PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  PetscValidLogicalCollectiveInt(tj,stepnum,3);
  PetscValidLogicalCollectiveReal(tj,time,4);
  PetscValidHeaderSpecific(X,VEC_CLASSID,5);
  if (!tj->ops->set) SETERRQ1(PetscObjectComm((PetscObject)tj),PETSC_ERR_SUP,"TSTrajectory type %s",((PetscObject)tj)->type_name);
  if (!tj->setupcalled) SETERRQ(PetscObjectComm((PetscObject)tj),PETSC_ERR_ORDER,"TSTrajectorySetUp should be called first");
  if (tj->monitor) {
    ierr = PetscViewerASCIIPrintf(tj->monitor,"TSTrajectorySet: stepnum %D, time %g (stages %D)\n",stepnum,(double)time,(PetscInt)!tj->solution_only);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(TSTrajectory_Set,tj,ts,0,0);CHKERRQ(ierr);
  ierr = (*tj->ops->set)(tj,ts,stepnum,time,X);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TSTrajectory_Set,tj,ts,0,0);CHKERRQ(ierr);
  ierr = TSHistoryUpdate(tj->tsh,stepnum,time);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  TSTrajectoryGet - Updates the solution vector of a time stepper object by inquiring the TSTrajectory

  Collective on TS

  Input Parameters:
+ tj      - the trajectory object
. ts      - the time stepper object (optional)
- stepnum - the step number

  Output Parameter:
. time    - the time associated with the step number

  Level: developer

  Notes: Usually one does not call this routine, it is called automatically during TSSolve()

.keywords: TS, trajectory, create

.seealso: TSTrajectorySetUp(), TSTrajectoryDestroy(), TSTrajectorySetType(), TSTrajectorySetVariableNames(), TSGetTrajectory(), TSTrajectorySet(), TSTrajectoryGetVecs()
*/
PetscErrorCode TSTrajectoryGet(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *time)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tj) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"TS solver did not save trajectory");
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  if (ts) PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  PetscValidLogicalCollectiveInt(tj,stepnum,3);
  PetscValidPointer(time,4);
  if (!tj->ops->get) SETERRQ1(PetscObjectComm((PetscObject)tj),PETSC_ERR_SUP,"TSTrajectory type %s",((PetscObject)tj)->type_name);
  if (!tj->setupcalled) SETERRQ(PetscObjectComm((PetscObject)tj),PETSC_ERR_ORDER,"TSTrajectorySetUp should be called first");
  if (tj->monitor) {
    ierr = PetscViewerASCIIPrintf(tj->monitor,"TSTrajectoryGet: stepnum %D, stages %D\n",stepnum,(PetscInt)!tj->solution_only);CHKERRQ(ierr);
    ierr = PetscViewerFlush(tj->monitor);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(TSTrajectory_Get,tj,ts,0,0);CHKERRQ(ierr);
  ierr = (*tj->ops->get)(tj,ts,stepnum,time);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(TSTrajectory_Get,tj,ts,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  TSTrajectoryGetVecs - Reconstructs the vectors of state and its time derivative using information from the TSTrajectory and, possibly, from the TS

  Collective on TS

  Input Parameters:
+ tj      - the trajectory object
. ts      - the time stepper object (optional)
- stepnum - the requested step number

  Input/Output Parameter:
. time    - the time associated with the step number

  Output Parameters:
+ U       - state vector (can be NULL)
- Udot    - time derivative of state vector (can be NULL)

  Level: developer

  Notes: If the step number is PETSC_DECIDE, the time is used to inquire the trajectory.
         Usually one does not call this routine, it is called during TSEvaluateGradient()

.keywords: TS, trajectory, create

.seealso: TSTrajectorySetUp(), TSTrajectoryDestroy(), TSTrajectorySetType(), TSTrajectorySetVariableNames(), TSGetTrajectory()
*/
PetscErrorCode TSTrajectoryGetVecs(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *time,Vec U,Vec Udot)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tj) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONGSTATE,"TS solver did not save trajectory");
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  if (ts) PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  PetscValidLogicalCollectiveInt(tj,stepnum,3);
  PetscValidPointer(time,4);
  if (U) PetscValidHeaderSpecific(U,VEC_CLASSID,5);
  if (Udot) PetscValidHeaderSpecific(Udot,VEC_CLASSID,6);
  if (!U && !Udot) PetscFunctionReturn(0);
  if (!tj->setupcalled) SETERRQ(PetscObjectComm((PetscObject)tj),PETSC_ERR_ORDER,"TSTrajectorySetUp should be called first");
  ierr = PetscLogEventBegin(TSTrajectory_GetVecs,tj,ts,0,0);CHKERRQ(ierr);
  if (tj->monitor) {
    PetscInt pU,pUdot;
    pU    = U ? 1 : 0;
    pUdot = Udot ? 1 : 0;
    ierr  = PetscViewerASCIIPrintf(tj->monitor,"Requested by GetVecs %D %D: stepnum %D, time %g\n",pU,pUdot,stepnum,(double)*time);CHKERRQ(ierr);
    ierr = PetscViewerFlush(tj->monitor);CHKERRQ(ierr);
  }
  if (U && tj->lag.caching) {
    PetscObjectId    id;
    PetscObjectState state;

    ierr = PetscObjectStateGet((PetscObject)U,&state);CHKERRQ(ierr);
    ierr = PetscObjectGetId((PetscObject)U,&id);CHKERRQ(ierr);
    if (stepnum == PETSC_DECIDE) {
      if (id == tj->lag.Ucached.id && *time == tj->lag.Ucached.time && state == tj->lag.Ucached.state) U = NULL;
    } else {
      if (id == tj->lag.Ucached.id && stepnum == tj->lag.Ucached.step && state == tj->lag.Ucached.state) U = NULL;
    }
    if (tj->monitor && !U) {
      ierr = PetscViewerASCIIPushTab(tj->monitor);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(tj->monitor,"State vector cached\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(tj->monitor);CHKERRQ(ierr);
      ierr = PetscViewerFlush(tj->monitor);CHKERRQ(ierr);
    }
  }
  if (Udot && tj->lag.caching) {
    PetscObjectId    id;
    PetscObjectState state;

    ierr = PetscObjectStateGet((PetscObject)Udot,&state);CHKERRQ(ierr);
    ierr = PetscObjectGetId((PetscObject)Udot,&id);CHKERRQ(ierr);
    if (stepnum == PETSC_DECIDE) {
      if (id == tj->lag.Udotcached.id && *time == tj->lag.Udotcached.time && state == tj->lag.Udotcached.state) Udot = NULL;
    } else {
      if (id == tj->lag.Udotcached.id && stepnum == tj->lag.Udotcached.step && state == tj->lag.Udotcached.state) Udot = NULL;
    }
    if (tj->monitor && !Udot) {
      ierr = PetscViewerASCIIPushTab(tj->monitor);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(tj->monitor,"Derivative vector cached\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(tj->monitor);CHKERRQ(ierr);
      ierr = PetscViewerFlush(tj->monitor);CHKERRQ(ierr);
    }
  }
  if (!U && !Udot) PetscFunctionReturn(0);

  if (stepnum == PETSC_DECIDE || Udot) { /* reverse search for requested time in TSHistory */
    if (tj->monitor) {
      ierr = PetscViewerASCIIPushTab(tj->monitor);CHKERRQ(ierr);
    }
    /* cached states will be updated in the function */
    ierr = TSTrajectoryReconstruct_Private(tj,ts,*time,U,Udot);CHKERRQ(ierr);
    if (tj->monitor) {
      ierr = PetscViewerASCIIPopTab(tj->monitor);CHKERRQ(ierr);
      ierr = PetscViewerFlush(tj->monitor);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }
  /* we were asked to load from stepnum, use TSTrajectoryGet */
  if (U) {
    Vec osol;

    ierr = TSGetSolution(ts,&osol);CHKERRQ(ierr);
    ierr = TSSetSolution(ts,U);CHKERRQ(ierr);
    ierr = TSTrajectoryGet(tj,ts,stepnum,time);CHKERRQ(ierr);
    if (osol) {
      ierr = TSSetSolution(ts,osol);CHKERRQ(ierr);
    }
    ierr = PetscObjectStateGet((PetscObject)U,&tj->lag.Ucached.state);CHKERRQ(ierr);
    ierr = PetscObjectGetId((PetscObject)U,&tj->lag.Ucached.id);CHKERRQ(ierr);
    tj->lag.Ucached.time = *time;
    tj->lag.Ucached.step = stepnum;
  }
  ierr = PetscLogEventEnd(TSTrajectory_GetVecs,tj,ts,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    TSTrajectoryView - Prints information about the trajectory object

    Collective on TSTrajectory

    Input Parameters:
+   tj - the TSTrajectory context obtained from TSTrajectoryCreate()
-   viewer - visualization context

    Options Database Key:
.   -ts_trajectory_view - calls TSTrajectoryView() at end of TSAdjointStep()

    Notes:
    The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

    The user can open an alternative visualization context with
    PetscViewerASCIIOpen() - output to a specified file.

    Level: developer

.keywords: TS, trajectory, timestep, view

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode  TSTrajectoryView(TSTrajectory tj,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)tj),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(tj,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)tj,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  total number of recomputations for adjoint calculation = %D\n",tj->recomps);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  disk checkpoint reads = %D\n",tj->diskreads);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  disk checkpoint writes = %D\n",tj->diskwrites);CHKERRQ(ierr);
    if (tj->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*tj->ops->view)(tj,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   TSTrajectorySetVariableNames - Sets the name of each component in the solution vector so that it may be saved with the trajectory

   Collective on TSTrajectory

   Input Parameters:
+  tr - the trajectory context
-  names - the names of the components, final string must be NULL

   Level: intermediate

.keywords: TS, TSTrajectory, vector, monitor, view

.seealso: TSTrajectory, TSGetTrajectory()
@*/
PetscErrorCode  TSTrajectorySetVariableNames(TSTrajectory ctx,const char * const *names)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx,TSTRAJECTORY_CLASSID,1);
  PetscValidPointer(names,2);
  ierr = PetscStrArrayDestroy(&ctx->names);CHKERRQ(ierr);
  ierr = PetscStrArrayallocpy(names,&ctx->names);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSTrajectorySetTransform - Solution vector will be transformed by provided function before being saved to disk

   Collective on TSLGCtx

   Input Parameters:
+  tj - the TSTrajectory context
.  transform - the transform function
.  destroy - function to destroy the optional context
-  ctx - optional context used by transform function

   Level: intermediate

.keywords: TSTrajectory,  vector, monitor, view

.seealso:  TSTrajectorySetVariableNames(), TSTrajectory, TSMonitorLGSetTransform()
@*/
PetscErrorCode  TSTrajectorySetTransform(TSTrajectory tj,PetscErrorCode (*transform)(void*,Vec,Vec*),PetscErrorCode (*destroy)(void*),void *tctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  tj->transform        = transform;
  tj->transformdestroy = destroy;
  tj->transformctx     = tctx;
  PetscFunctionReturn(0);
}

/*@C
  TSTrajectoryCreate - This function creates an empty trajectory object used to store the time dependent solution of an ODE/DAE

  Collective on MPI_Comm

  Input Parameter:
. comm - the communicator

  Output Parameter:
. tj   - the trajectory object

  Level: developer

  Notes: Usually one does not call this routine, it is called automatically when one calls TSSetSaveTrajectory().

.keywords: TS, trajectory, create

.seealso: TSTrajectorySetUp(), TSTrajectoryDestroy(), TSTrajectorySetType(), TSTrajectorySetVariableNames(), TSGetTrajectory()
@*/
PetscErrorCode  TSTrajectoryCreate(MPI_Comm comm,TSTrajectory *tj)
{
  TSTrajectory   t;
  TSHistory      tsh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(tj,2);
  *tj = NULL;
  ierr = TSInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(t,TSTRAJECTORY_CLASSID,"TSTrajectory","Time stepping","TS",comm,TSTrajectoryDestroy,TSTrajectoryView);CHKERRQ(ierr);
  t->setupcalled = PETSC_FALSE;

  ierr = PetscNew(&tsh);CHKERRQ(ierr);
  tsh->n      = 0;
  tsh->c      = 1024; /* capacity */
  tsh->s      = 1024; /* reallocation size */
  tsh->sorted = PETSC_TRUE;
  ierr = PetscMalloc1(tsh->c,&tsh->hist);CHKERRQ(ierr);
  ierr = PetscMalloc1(tsh->c,&tsh->hist_id);CHKERRQ(ierr);
  t->tsh = tsh;

  t->lag.order            = 1;
  t->lag.L                = NULL;
  t->lag.T                = NULL;
  t->lag.W                = NULL;
  t->lag.WW               = NULL;
  t->lag.TW               = NULL;
  t->lag.TT               = NULL;
  t->lag.caching          = PETSC_TRUE;
  t->lag.Ucached.id       = 0;
  t->lag.Ucached.state    = -1;
  t->lag.Ucached.time     = PETSC_MIN_REAL;
  t->lag.Ucached.step     = PETSC_MAX_INT;
  t->lag.Udotcached.id    = 0;
  t->lag.Udotcached.state = -1;
  t->lag.Udotcached.time  = PETSC_MIN_REAL;
  t->lag.Udotcached.step  = PETSC_MAX_INT;
  t->adjoint_solve_mode   = PETSC_TRUE;
  t->solution_only        = PETSC_FALSE;

  *tj = t;
  PetscFunctionReturn(0);
}

/*@C
  TSTrajectorySetType - Sets the storage method to be used as in a trajectory

  Collective on TS

  Input Parameters:
+ tj   - the TSTrajectory context
. ts   - the TS context
- type - a known method

  Options Database Command:
. -ts_trajectory_type <type> - Sets the method; use -help for a list of available methods (for instance, basic)

   Level: developer

.keywords: TS, trajectory, timestep, set, type

.seealso: TS, TSTrajectoryCreate(), TSTrajectorySetFromOptions(), TSTrajectoryDestroy()

@*/
PetscErrorCode  TSTrajectorySetType(TSTrajectory tj,TS ts,const TSTrajectoryType type)
{
  PetscErrorCode (*r)(TSTrajectory,TS);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)tj,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(TSTrajectoryList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TSTrajectory type: %s",type);
  if (tj->ops->destroy) {
    ierr = (*(tj)->ops->destroy)(tj);CHKERRQ(ierr);

    tj->ops->destroy = NULL;
  }
  ierr = PetscMemzero(tj->ops,sizeof(*tj->ops));CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)tj,type);CHKERRQ(ierr);
  ierr = (*r)(tj,ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Basic(TSTrajectory,TS);
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Singlefile(TSTrajectory,TS);
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Memory(TSTrajectory,TS);
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Visualization(TSTrajectory,TS);

/*@C
  TSTrajectoryRegisterAll - Registers all of the trajectory storage schecmes in the TS package.

  Not Collective

  Level: developer

.keywords: TS, trajectory, register, all

.seealso: TSTrajectoryRegister()
@*/
PetscErrorCode  TSTrajectoryRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSTrajectoryRegisterAllCalled) PetscFunctionReturn(0);
  TSTrajectoryRegisterAllCalled = PETSC_TRUE;

  ierr = TSTrajectoryRegister(TSTRAJECTORYBASIC,TSTrajectoryCreate_Basic);CHKERRQ(ierr);
  ierr = TSTrajectoryRegister(TSTRAJECTORYSINGLEFILE,TSTrajectoryCreate_Singlefile);CHKERRQ(ierr);
  ierr = TSTrajectoryRegister(TSTRAJECTORYMEMORY,TSTrajectoryCreate_Memory);CHKERRQ(ierr);
  ierr = TSTrajectoryRegister(TSTRAJECTORYVISUALIZATION,TSTrajectoryCreate_Visualization);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSTrajectoryDestroy - Destroys a trajectory context

   Collective on TSTrajectory

   Input Parameter:
.  tj - the TSTrajectory context obtained from TSTrajectoryCreate()

   Level: developer

.keywords: TS, trajectory, timestep, destroy

.seealso: TSTrajectoryCreate(), TSTrajectorySetUp()
@*/
PetscErrorCode  TSTrajectoryDestroy(TSTrajectory *tj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*tj) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*tj),TSTRAJECTORY_CLASSID,1);
  if (--((PetscObject)(*tj))->refct > 0) {*tj = 0; PetscFunctionReturn(0);}

  ierr = TSHistoryDestroy((*tj)->tsh);CHKERRQ(ierr);
  ierr = VecDestroyVecs((*tj)->lag.order+1,&(*tj)->lag.W);CHKERRQ(ierr);
  ierr = PetscFree5((*tj)->lag.L,(*tj)->lag.T,(*tj)->lag.WW,(*tj)->lag.TT,(*tj)->lag.TW);CHKERRQ(ierr);
  if ((*tj)->transformdestroy) {ierr = (*(*tj)->transformdestroy)((*tj)->transformctx);CHKERRQ(ierr);}
  if ((*tj)->ops->destroy) {ierr = (*(*tj)->ops->destroy)((*tj));CHKERRQ(ierr);}
  ierr = PetscStrArrayDestroy(&(*tj)->names);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(tj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  TSTrajectorySetTypeFromOptions_Private - Sets the type of ts from user options.

  Collective on TSTrajectory

  Input Parameter:
+ tj - the TSTrajectory context
- ts - the TS context

  Options Database Keys:
. -ts_trajectory_type <type> - TSTRAJECTORYBASIC, TSTRAJECTORYMEMORY, TSTRAJECTORYSINGLEFILE, TSTRAJECTORYVISUALIZATION

  Level: developer

.keywords: TS, trajectory, set, options, type

.seealso: TSTrajectorySetFromOptions(), TSTrajectorySetType()
*/
static PetscErrorCode TSTrajectorySetTypeFromOptions_Private(PetscOptionItems *PetscOptionsObject,TSTrajectory tj,TS ts)
{
  PetscBool      opt;
  const char     *defaultType;
  char           typeName[256];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (((PetscObject)tj)->type_name) defaultType = ((PetscObject)tj)->type_name;
  else defaultType = TSTRAJECTORYBASIC;

  ierr = TSTrajectoryRegisterAll();CHKERRQ(ierr);
  ierr = PetscOptionsFList("-ts_trajectory_type","TSTrajectory method"," TSTrajectorySetType",TSTrajectoryList,defaultType,typeName,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = TSTrajectorySetType(tj,ts,typeName);CHKERRQ(ierr);
  } else {
    ierr = TSTrajectorySetType(tj,ts,defaultType);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   TSTrajectorySetMonitor - Monitor the schedules generated by the checkpointing controller

   Collective on TSTrajectory

   Input Arguments:
+  tj - the TSTrajectory context
-  flg - PETSC_TRUE to active a monitor, PETSC_FALSE to disable

   Options Database Keys:
.  -ts_trajectory_monitor - print TSTrajectory information

   Level: developer

.keywords: TS, trajectory, set, monitor

.seealso: TSTrajectoryCreate(), TSTrajectoryDestroy(), TSTrajectorySetUp()
@*/
PetscErrorCode TSTrajectorySetMonitor(TSTrajectory tj,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  PetscValidLogicalCollectiveBool(tj,flg,2);
  if (flg) tj->monitor = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)tj));
  else tj->monitor = NULL;
  PetscFunctionReturn(0);
}

/*@
   TSTrajectorySetFromOptions - Sets various TSTrajectory parameters from user options.

   Collective on TSTrajectory

   Input Parameter:
+  tj - the TSTrajectory context obtained from TSTrajectoryCreate()
-  ts - the TS context

   Options Database Keys:
+  -ts_trajectory_type <type> - TSTRAJECTORYBASIC, TSTRAJECTORYMEMORY, TSTRAJECTORYSINGLEFILE, TSTRAJECTORYVISUALIZATION
-  -ts_trajectory_monitor - print TSTrajectory information

   Level: developer

   Notes: This is not normally called directly by users

.keywords: TS, trajectory, timestep, set, options, database

.seealso: TSSetSaveTrajectory(), TSTrajectorySetUp()
@*/
PetscErrorCode  TSTrajectorySetFromOptions(TSTrajectory tj,TS ts)
{
  PetscErrorCode ierr;
  PetscBool      set,flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  if (ts) PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  ierr = PetscObjectOptionsBegin((PetscObject)tj);CHKERRQ(ierr);
  ierr = TSTrajectorySetTypeFromOptions_Private(PetscOptionsObject,tj,ts);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_trajectory_monitor","Print checkpointing schedules","TSTrajectorySetMonitor",tj->monitor ? PETSC_TRUE:PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set) {ierr = TSTrajectorySetMonitor(tj,flg);CHKERRQ(ierr);}
  ierr = PetscOptionsInt("-ts_trajectory_reconstruction_order","Interpolation order for reconstruction",NULL,tj->lag.order,&tj->lag.order,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_trajectory_reconstruction_caching","Turn on caching of TSTrajectoryGetVecs input",NULL,tj->lag.caching,&tj->lag.caching,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ts_trajectory_adjointmode","Instruct the trajectory that will be used in a TSAdjointSolve()",NULL,tj->adjoint_solve_mode,&tj->adjoint_solve_mode,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ts_trajectory_solution_only","Checkpoint solution only","TSTrajectorySetSolutionOnly",tj->solution_only,&tj->solution_only,NULL);CHKERRQ(ierr);
  /* Handle specific TSTrajectory options */
  if (tj->ops->setfromoptions) {
    ierr = (*tj->ops->setfromoptions)(PetscOptionsObject,tj);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TSTrajectorySetUp - Sets up the internal data structures, e.g. stacks, for the later use
   of a TS trajectory.

   Collective on TS

   Input Parameter:
+  ts - the TS context obtained from TSCreate()
-  tj - the TS trajectory context

   Level: developer

.keywords: TS, trajectory, setup

.seealso: TSSetSaveTrajectory(), TSTrajectoryCreate(), TSTrajectoryDestroy()
@*/
PetscErrorCode  TSTrajectorySetUp(TSTrajectory tj,TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tj) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  if (ts) PetscValidHeaderSpecific(ts,TS_CLASSID,2);
  if (tj->setupcalled) PetscFunctionReturn(0);

  if (!((PetscObject)tj)->type_name) {
    ierr = TSTrajectorySetType(tj,ts,TSTRAJECTORYBASIC);CHKERRQ(ierr);
  }
  if (tj->ops->setup) {
    ierr = (*tj->ops->setup)(tj,ts);CHKERRQ(ierr);
  }

  tj->setupcalled = PETSC_TRUE;

  /* Set the counters to zero */
  tj->recomps    = 0;
  tj->diskreads  = 0;
  tj->diskwrites = 0;
  PetscFunctionReturn(0);
}

/*@
   TSTrajectorySetSolutionOnly - Tells the trajectory to store just the solution, and not any intermidiate stage also.

   Collective on TS

   Input Parameter:
+  tj  - the TS trajectory context
-  flg - the boolean flag

   Level: developer

.keywords: trajectory

.seealso: TSSetSaveTrajectory(), TSTrajectoryCreate(), TSTrajectoryDestroy()
@*/
PetscErrorCode TSTrajectorySetSolutionOnly(TSTrajectory tj,PetscBool solution_only)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tj,TSTRAJECTORY_CLASSID,1);
  PetscValidLogicalCollectiveBool(tj,solution_only,2);
  tj->solution_only = solution_only;
  PetscFunctionReturn(0);
}

/*
 Code for Timestepping with a multiscale method
 */
#include <petsc-private/tsimpl.h>                /*I   "petscts.h"   I*/

/* Once there are more solvers, this can be changed to only include data being used by a given solver */
typedef struct {
    TSMultiType     type_name;
    TS              ts, tsCoarse, tsFine;
    PetscReal       epsilon; /* A small parameter which characterizes the fine scale compared to an O(1) coarse scale */

    struct{
        PetscErrorCode (*precoarse)(TS ts);
        PetscErrorCode (*setup)(TS ts);
    } ops;
    
    Vec W; /* A work vector (not always used)  */ 
} TS_Multi;

PetscFunctionList TSMultiList_precoarse = 0;
PetscFunctionList TSMultiList_setup = 0;
static TSMultiType TSMultiDefault = TSMULTIFLAVORFE;
static PetscBool  TSMultiPackageInitialized;

/* -------------------------------------------------------------------------- */
/* Methods for TSMULTIFLAVORFE */

#undef __FUNCT__
#define __FUNCT__ "TakeFineStep_MultiFLAVORFE"
static PetscErrorCode TakeFineStep_MultiFLAVORFE(TS ts)
{
    PetscErrorCode ierr;
    TS_Multi       *multi = (TS_Multi*)ts->data;
    
    PetscFunctionBegin;
    ierr = TSStep(multi->tsFine);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetUp_MultiFLAVORFE"
static PetscErrorCode SetUp_MultiFLAVORFE(TS ts)
{
    PetscErrorCode ierr;
    TS_Multi       *multi = (TS_Multi*)ts->data;
    TSRHSFunction  rhsFuncSlow, rhsFuncFull;
    void           *ctx;
    DM             dm;
    
    PetscFunctionBegin;
    ierr = TSSetType(multi->tsCoarse,TSEULER);CHKERRQ(ierr);
    ierr = TSSetType(multi->tsFine,TSEULER);CHKERRQ(ierr);
    
    /* Here, we can simply use one vector for both timesteppers*/
    ierr = TSSetSolution(multi->tsCoarse,ts->vec_sol);CHKERRQ(ierr);
    ierr = TSSetSolution(multi->tsFine,ts->vec_sol);CHKERRQ(ierr);
    
    ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
    ierr = DMTSGetRHSPartitionFunction(dm,TS_MULTI_PARTITION,TS_MULTI_SLOW_SLOT,&rhsFuncSlow,&ctx);CHKERRQ(ierr); 
    ierr = TSSetRHSFunction(multi->tsCoarse,NULL,rhsFuncSlow,ctx);CHKERRQ(ierr);
    ierr = DMTSGetRHSPartitionFunction(dm,TS_MULTI_PARTITION,TS_MULTI_FULL_SLOT,&rhsFuncFull,&ctx);CHKERRQ(ierr);
    ierr = TSSetRHSFunction(multi->tsFine,NULL,rhsFuncFull,ctx);CHKERRQ(ierr);
    
    /* Time steps */
    /* For now, we use 'meso' steps which are 1/20 of the timestep set by the TS (under the assumption that those are 'macro'. Micro time steps are epsilon/20 */
    ierr = TSSetTimeStep(multi->tsCoarse, (ts->time_step * 0.05) - (multi->epsilon*0.05));CHKERRQ(ierr); 
    ierr = TSSetTimeStep(multi->tsFine,   multi->epsilon * 0.05);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/* Methods for BFHMMMFE */
#undef __FUNCT__
#define __FUNCT__ "Precoarse_MultiBFHMMFE"
static PetscErrorCode Precoarse_MultiBFHMMFE(TS ts)
{
  PetscErrorCode ierr;
  TS_Multi       *multi = (TS_Multi*)ts->data;
  TS             tsFine = multi->tsFine, tsCoarse = multi->tsCoarse;
  Vec            S = ts->vec_sol;
  DM             dm;
  void           *rhsctx;
  TSRHSFunction  rhsfunc;
  
  PetscFunctionBegin;

  /* Store state in the work vector, to start the upcoming second fine solve*/
  ierr = VecCopy(S,multi->W);CHKERRQ(ierr);

  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetRHSPartitionFunction(dm,TS_MULTI_PARTITION,TS_MULTI_FAST_SLOT,&rhsfunc,&rhsctx);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(tsFine,NULL,rhsfunc,rhsctx);CHKERRQ(ierr);
  ierr = TSSetTime(tsFine,0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(tsFine,0);CHKERRQ(ierr);

  ierr = TSSolve(tsFine,S);CHKERRQ(ierr);
  
  ierr = VecScale(S,1.-(tsCoarse->time_step/tsFine->max_time));

  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "FormRHSFunction_BFHMMFE"
static PetscErrorCode FormRHSFunction_BFHMMFE(TS tsCoarse, PetscReal t, Vec X, Vec F, void* ctx)
{
  PetscErrorCode ierr;  
  TS_Multi       *multi;
  DM             dm;
  void           *rhsctx;
  TSRHSFunction  rhsfunc;
  TS             tsFine;

  PetscFunctionBegin;  

  ierr = TSGetApplicationContext(tsCoarse,&multi);CHKERRQ(ierr);
  tsFine = multi->tsFine;
  
  ierr = TSGetDM(multi->ts,&dm);CHKERRQ(ierr); /*  We store a reference back to the main TS object in TS_Multi  */ 
  ierr = DMTSGetRHSPartitionFunction(dm,TS_MULTI_PARTITION,TS_MULTI_FULL_SLOT,&rhsfunc,&rhsctx);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(tsFine,NULL,rhsfunc,rhsctx);CHKERRQ(ierr);
  ierr = TSSetTime(tsFine,0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(tsFine,0);CHKERRQ(ierr);

  ierr = TSSolve(tsFine,multi->W);CHKERRQ(ierr);

  /* Scale and copy to F*/
 ierr = VecAXPBY(F,1./tsFine->max_time,0,multi->W);CHKERRQ(ierr); 

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetUp_MultiBFHMMFE"
static PetscErrorCode SetUp_MultiBFHMMFE(TS ts)
{
  PetscErrorCode ierr;
  TS_Multi       *multi = (TS_Multi*)ts->data;
  TS             tsCoarse = multi->tsCoarse,tsFine = multi->tsFine;

  PetscFunctionBegin;
  ierr = TSSetType(tsCoarse,TSEULER);CHKERRQ(ierr);
  ierr = TSSetType(tsFine,TSSSP);CHKERRQ(ierr); 
  /* [More sophisticated inner solvers can be chosen at will, and in the future should be user-settable]
  ierr = TSSSPSetNumStages(tsFine,25);CHKERRQ(ierr);    
  ierr = TSSSPSetType(tsFine,TSSSPRKS3);CHKERRQ(ierr);
 */
  /* Use this TS's work vec_sol for the coarse solver */
  ierr = TSSetSolution(tsCoarse,ts->vec_sol);CHKERRQ(ierr);
  
  ierr = TSSetRHSFunction(tsCoarse,NULL,FormRHSFunction_BFHMMFE,NULL);CHKERRQ(ierr);

  ierr = TSSetTimeStep(tsCoarse,ts->time_step);CHKERRQ(ierr);
  ierr = TSSetTimeStep(tsFine,multi->epsilon/10.);CHKERRQ(ierr); /* fine time step is epsilon/N for some moderately large N, arbitrarily */

  /* Set the fine TS to solve at an exact final time */
  ierr = TSSetExactFinalTime(tsFine,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr); 

  /* Fine scale solve time is 15 \epsilon, arbitrarily  (this is "delta") */
  ierr = TSSetDuration(tsFine,PETSC_MAX_INT,multi->epsilon * 15);CHKERRQ(ierr);

  ierr = VecDuplicate(ts->vec_sol,&multi->W);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "TSMultiInitializePackage"
/*@C
 TSMultiInitializePackage - This function initializes everything in the TSMulti package. It is called
 from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to TSCreate_Multi()
 when using static libraries.
 
 Level: developer
 
 .keywords: TS, TSMulti, initialize, package
 .seealso: PetscInitialize()
 @*/
PetscErrorCode TSMultiInitializePackage(void)
{
    PetscErrorCode ierr;
    
    PetscFunctionBegin;
    if (TSMultiPackageInitialized) PetscFunctionReturn(0);
    TSMultiPackageInitialized = PETSC_TRUE;
    ierr = PetscFunctionListAdd(&TSMultiList_precoarse,TSMULTIFLAVORFE, TakeFineStep_MultiFLAVORFE);CHKERRQ(ierr);
    ierr = PetscFunctionListAdd(&TSMultiList_setup,TSMULTIFLAVORFE, SetUp_MultiFLAVORFE);CHKERRQ(ierr);
    ierr = PetscFunctionListAdd(&TSMultiList_precoarse,TSMULTIBFHMMFE, Precoarse_MultiBFHMMFE);CHKERRQ(ierr);
    ierr = PetscFunctionListAdd(&TSMultiList_setup,TSMULTIBFHMMFE, SetUp_MultiBFHMMFE);CHKERRQ(ierr);
    ierr = PetscRegisterFinalize(TSMultiFinalizePackage);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSMultiFinalizePackage"
/*@C
 TSMultiFinalizePackage - This function destroys everything in the TSMulti package. It is
 called from PetscFinalize().
 
 Level: developer
 
 .keywords: Petsc, destroy, package
 .seealso: PetscFinalize()
 @*/
PetscErrorCode TSMultiFinalizePackage(void)
{
    PetscErrorCode ierr;
    
    PetscFunctionBegin;
    TSMultiPackageInitialized = PETSC_FALSE;
    ierr = PetscFunctionListDestroy(&TSMultiList_precoarse);CHKERRQ(ierr);
    ierr = PetscFunctionListDestroy(&TSMultiList_setup);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "TSMultiSetType"
/*@C
 TSMultiSetType - set the multiscale time integration scheme to use
 
 Logically Collective
 
 Input Arguments:
 ts - time stepping object
 type - type of scheme to use
 
 Level: beginner
 
 .seealso: TSMULTI, TSMultiGetType()
 @*/
PetscErrorCode TSMultiSetType(TS ts,TSMultiType type)
{
    PetscErrorCode ierr;
    
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ts,TS_CLASSID,1);
    ierr = PetscTryMethod(ts,"TSMultiSetType_C",(TS,TSMultiType),(ts,type));CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSMultiGetType"
/*@C
 TSMultiGetType - get the Multi time integration scheme
 
 Logically Collective
 
 Input Argument:
 ts - time stepping object
 
 Output Argument:
 type - type of scheme being used
 
 Level: beginner
 
 .seealso: TSMULTI, TSMultiSettype()
 @*/
PetscErrorCode TSMultiGetType(TS ts,TSMultiType *type)
{
    PetscErrorCode ierr;
    
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ts,TS_CLASSID,1);
    ierr = PetscTryMethod(ts,"TSMultiGetType_C",(TS,TSMultiType*),(ts,type));CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSMultiSetType_Multi"
PETSC_EXTERN PetscErrorCode TSMultiSetType_Multi(TS ts,TSMultiType type)
{
    PetscErrorCode ierr, (*precoarse)(TS), (*setup)(TS);
    TS_Multi             *multi = (TS_Multi*)ts->data;
    
    PetscFunctionBegin;
    ierr = PetscFunctionListFind(TSMultiList_precoarse,type,&precoarse);CHKERRQ(ierr);
    if (!precoarse) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TSMultiType %s given",type);
    ierr = PetscFunctionListFind(TSMultiList_setup,type,&setup);CHKERRQ(ierr);
    if (!setup) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown TSMultiType %s given",type);
    multi->ops.precoarse = precoarse;
    multi->ops.setup = setup;
    ierr = PetscFree(multi->type_name);CHKERRQ(ierr);
    ierr = PetscStrallocpy(type,&multi->type_name);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "TSMultiGetType_Multi"
PetscErrorCode TSMultiGetType_Multi(TS ts,TSMultiType *type)
{
    TS_Multi *multi = (TS_Multi*)ts->data;
    
    PetscFunctionBegin;
    *type = multi->type_name;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSMultiSetEpsilon"
PetscErrorCode TSMultiSetEpsilon(TS ts,PetscReal epsilon)
{
  TS_Multi            *multi = (TS_Multi*)ts->data;
  
  PetscFunctionBegin;
  multi->epsilon = epsilon;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSMultiGetEpsilon"
PetscErrorCode TSMultiGetEpsilon(TS ts,PetscReal *epsilon)
{
  TS_Multi            *multi = (TS_Multi*)ts->data;
  
  PetscFunctionBegin;
  *epsilon = multi->epsilon;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "TSStep_Multi"
static PetscErrorCode TSStep_Multi(TS ts)
{
    PetscErrorCode ierr;
    TS_Multi       *multi = (TS_Multi*)ts->data;
    
    PetscFunctionBegin;
    ierr = TSPreStep(ts);CHKERRQ(ierr);
    ierr = TSPreStage(ts,ts->ptime); CHKERRQ(ierr); 

    /* integrator-specific pre-coarse-step behavior */
    if(multi->ops.precoarse){
      ierr = multi->ops.precoarse(ts);CHKERRQ(ierr);
    }
    /*  Take a step with the coarse solver. 
        Various functions registered with tsCoarse (PreStep, PostStep, PreStagIFunctions, RHSFunctions) can also
         invoke tsFine. To do this, note that we put a pointer to this solver's ts->data in multi->tsCoarse->user */
    ierr = TSStep(multi->tsCoarse);CHKERRQ(ierr);
  
    /*  TSMULTI counts coarse solver steps, and accumulates time based on coarse solver steps automatically here. 
        Time can also be accumulated in an appropriate way after calls are made to the fine solver, for instance
        adding a small amount after FLAVOR fine steps or microsolves in HMM that change the coarse step starting point */
    ts->ptime += multi->tsCoarse->time_step;
    ts->steps++;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetUp_Multi"
static PetscErrorCode TSSetUp_Multi(TS ts)
{
    PetscErrorCode ierr;
    TS_Multi       *multi = (TS_Multi*)ts->data;
    
    PetscFunctionBegin;

    /* pass in the TS_Multi* pointer as the coarse solver's user context */
    ierr = TSSetApplicationContext(multi->tsCoarse, multi);CHKERRQ(ierr);

    /* integrator-specific setup */
    ierr =  multi->ops.setup(ts); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSReset_Multi"
static PetscErrorCode TSReset_Multi(TS ts)
{
    PetscErrorCode ierr;
    TS_Multi       *multi = (TS_Multi*)ts->data;
    
    PetscFunctionBegin;
    ierr = TSReset(multi->tsCoarse);CHKERRQ(ierr);
    ierr = TSReset(multi->tsFine);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDestroy_Multi"
static PetscErrorCode TSDestroy_Multi(TS ts)
{
    PetscErrorCode ierr;
    TS_Multi       *multi = (TS_Multi*)ts->data;
    
    PetscFunctionBegin;
    ierr = TSReset_Multi(ts);CHKERRQ(ierr);

    ierr = PetscObjectComposeFunction((PetscObject)ts,"TSMultiGetType_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)ts,"TSMultiSetType_C",NULL);CHKERRQ(ierr);
    
    ierr = TSDestroy(&multi->tsCoarse);CHKERRQ(ierr);
    ierr = TSDestroy(&multi->tsFine);CHKERRQ(ierr);
   
    ierr = VecDestroy(&multi->W);CHKERRQ(ierr); 

    PetscFree(multi->type_name);
    ierr = PetscFree(ts->data);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_Multi"
static PetscErrorCode TSSetFromOptions_Multi(TS ts)
{
    PetscFunctionBegin;
    char           tname[256];
    TS_Multi       *multi       = (TS_Multi*)ts->data;
    PetscErrorCode ierr;
    PetscBool      flg;

    PetscFunctionBegin;
    ierr = PetscOptionsHead("Multiscale ODE solver options");CHKERRQ(ierr);
    {
      ierr = PetscOptionsList("-ts_multi_type","Type of Multiscale method","TSMultiSetType",TSMultiList_precoarse,tname,tname,sizeof(tname),&flg);CHKERRQ(ierr); 
      if (flg) {
        ierr = TSMultiSetType(ts,tname);CHKERRQ(ierr);
      }
    }
    ierr = PetscOptionsTail();CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/*MC
 TSMULTI- Multiscale ODE Solver

  Provides implementations of integrators useful for problems with well-separated 'slow' and 'fast' scales, where only slow observables need be computed accurately, allowing for homegenization on the fast scale.
 
 Level: beginner
 
 .seealso:  TSCreate(), TS, TSSetType()
 
 M*/
#undef __FUNCT__
#define __FUNCT__ "TSCreate_Multi"
PETSC_EXTERN PetscErrorCode TSCreate_Multi(TS ts)
{
    TS_Multi        *multi;
    PetscErrorCode  ierr;
    MPI_Comm        comm;
    
    PetscFunctionBegin;
#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
    ierr = TSMultiInitializePackage();CHKERRQ(ierr);
#endif
    ts->ops->setup           = TSSetUp_Multi;
    ts->ops->step            = TSStep_Multi;
    ts->ops->reset           = TSReset_Multi;
    ts->ops->destroy         = TSDestroy_Multi;
    ts->ops->setfromoptions  = TSSetFromOptions_Multi;
    
    ierr = PetscNewLog(ts,TS_Multi,&multi);CHKERRQ(ierr);
    ts->data = multi;
    multi->ts = ts;
    
    ierr = PetscObjectComposeFunction((PetscObject)ts,"TSMultiGetType_C",TSMultiGetType_Multi);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)ts,"TSMultiSetType_C",TSMultiSetType_Multi);CHKERRQ(ierr);
    
    ierr = TSMultiSetType(ts,TSMultiDefault);CHKERRQ(ierr);
    
    ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
    ierr = TSCreate(comm,&multi->tsCoarse);CHKERRQ(ierr);
    ierr = TSCreate(comm,&multi->tsFine);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

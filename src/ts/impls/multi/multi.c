/*
 Code for Timestepping with a multiscale method
 */
#include <petsc-private/tsimpl.h>                /*I   "petscts.h"   I*/

/* Once there are more solvers, this can be changed to only include data being used by a given solver */
typedef struct {
    char            *type_name;
    TS              ts, tsCoarse, tsFine;
    PetscReal       window; 
    struct{
        PetscErrorCode (*precoarse)(TS ts);
        PetscErrorCode (*setup)(TS ts);
    } ops;
    
    PetscInt nwork;
    Vec *W; /* work vectors */ 
    Vec B; /* Basepoint to advance */
    PetscReal tRef; /* A Reference time used by filter functions */
} TS_Multi;

/* Methods for FHMMHEUN */
#undef __FUNCT__
#define __FUNCT__ "Precoarse_MultiFHMMHEUN"
static PetscErrorCode Precoarse_MultiFHMMHEUN(TS ts)
{
  PetscFunctionBegin;
  /* Empty. The precoarse function is mainly intended for FLAVORs, which alternate integrators instead of nesting them */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define  __FUNCT__ "ComputeFilter"
static PetscErrorCode ComputeFilter(PetscReal t, PetscReal *value)
{
  PetscFunctionBegin;
  
  /* This is a polynomial function with 5 vanishing derivatives at {0,1}, unit integral over its support
     of [0,1], and vanishing 'first moment' (symmetric around 1/2). Future improvements can replace this
     with a variable order function (or a non-analytic C^\infty type function) */
  PetscReal fac = t*(1-t);
  *value = 2772 * fac * fac * fac * fac * fac;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormRHSFunctionFullFiltered"
PetscErrorCode FormRHSFunctionFullFiltered(TS tsFine, PetscReal t, Vec X, Vec F, void *ctx)
{
  PetscErrorCode  ierr;
  TS_Multi        *multi;
  PetscReal       filt, relTime;
  TSRHSFunction   rhsfuncfast, rhsfuncslow;
  Vec             Ffast;
  void            *rhsctxfast, *rhsctxslow; 
  DM              dm;
  TS              ts;

  /* Note here and elsewhere that we do not set the corresponding Jacobian functions - this will be required before general inner solvers can be chosen by the user */

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(tsFine,&multi);CHKERRQ(ierr);
  ts = multi->ts;
  relTime = (t - multi->tRef)/multi->window; /* expects tRef to be set to the correct value */
  ierr = VecDuplicate(F,&Ffast);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetRHSPartitionFunction(dm,TS_MULTI_PARTITION,TS_MULTI_SLOW_SLOT,&rhsfuncslow,&rhsctxslow);CHKERRQ(ierr);
  rhsfuncslow(tsFine,t,X,F,rhsctxslow); CHKERRQ(ierr);
  ierr = ComputeFilter(relTime,&filt);CHKERRQ(ierr); /* Note that we expect to only integrate filtered quantities over periods of length multi->window */
  ierr = VecScale(F,filt);CHKERRQ(ierr);
  ierr = DMTSGetRHSPartitionFunction(dm,TS_MULTI_PARTITION,TS_MULTI_FAST_SLOT,&rhsfuncfast,&rhsctxfast);CHKERRQ(ierr);
  rhsfuncfast(tsFine,t,X,Ffast,rhsctxfast);CHKERRQ(ierr);
  ierr = VecAXPY(F,1.0,Ffast);
  ierr = VecDestroy(&Ffast);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormRHSFunction_FHMMHEUN"
static PetscErrorCode FormRHSFunction_FHMMHEUN(TS tsCoarse, PetscReal t, Vec X, Vec F, void* ctx)
{
  PetscErrorCode ierr;
  PetscReal      d, window;
  TS_Multi       *multi;
  TS             tsFine, ts;
  DM             dm;
  TSRHSFunction  rhsfuncfast;
  void           *rhsctxfast;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(tsCoarse,&multi);CHKERRQ(ierr);
  tsFine = multi->tsFine;
  ts = multi->ts;
  window = multi->window; /* This does not yet adapt the window */

  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetRHSPartitionFunction(dm,TS_MULTI_PARTITION,TS_MULTI_FAST_SLOT,&rhsfuncfast,&rhsctxfast);CHKERRQ(ierr);

  /* Copy X to F and W[1] */
  ierr = VecCopy(X,F);CHKERRQ(ierr);
  ierr = VecCopy(X,multi->W[1]);CHKERRQ(ierr);
  
  /* Advance W[1] by 2window, unperturbed ("fast")*/
  ierr = TSSetRHSFunction(tsFine,NULL,rhsfuncfast,rhsctxfast);CHKERRQ(ierr);
  ierr = TSSetTime(tsFine,t);CHKERRQ(ierr);
  ierr = TSSetDuration(tsFine,-1,t + 2*window);CHKERRQ(ierr); 
  ierr = TSSolve(tsFine,multi->W[1]);CHKERRQ(ierr); 

  /* Advance F by window, full filtered system */
  ierr = TSSetRHSFunction(tsFine,NULL,FormRHSFunctionFullFiltered,NULL);CHKERRQ(ierr);
  ierr = TSSetTime(tsFine,t);CHKERRQ(ierr);
  ierr = TSSetDuration(tsFine,-1,t + window);CHKERRQ(ierr);
  multi->tRef = t;
  ierr = TSSolve(tsFine,F);CHKERRQ(ierr); 
  
  ierr = VecCopy(F,multi->W[0]);CHKERRQ(ierr);

  /* Advance W[0] by window, unperturbed, from time t + window */
  ierr = TSSetRHSFunction(tsFine,NULL,rhsfuncfast,rhsctxfast);CHKERRQ(ierr);
  ierr = TSSetTime(tsFine,t + window);CHKERRQ(ierr); 
  ierr = TSSetDuration(tsFine,-1,t + 2*window);CHKERRQ(ierr);
  ierr = TSSolve(tsFine,multi->W[0]);CHKERRQ(ierr); 

  /* Advance F by Delta, full filtered system, from time t + window */
  ierr = TSSetRHSFunction(tsFine,NULL,FormRHSFunctionFullFiltered,NULL);CHKERRQ(ierr);
  ierr = TSSetTime(tsFine,t + multi->window);CHKERRQ(ierr); 
  ierr = TSSetDuration(tsFine,-1,t + 2*window);CHKERRQ(ierr);
  multi->tRef = t + window;
  ierr = TSSolve(tsFine,F);CHKERRQ(ierr); 

  /* Forward 2nd order accurate finite difference */
  d = 2.0*window;
  ierr = VecAXPBYPCZ(F,-3.0/d,4.0/d,-1.0/d,multi->W[1],multi->W[0]);CHKERRQ(ierr);

  /* Advance basepoint clock. The basepoint may or may not be the same as X */
  ierr = TSSetRHSFunction(tsFine,NULL,rhsfuncfast,rhsctxfast);CHKERRQ(ierr);
  ierr = TSSetTime(tsFine,t);CHKERRQ(ierr);
  ierr = TSSetDuration(tsFine,-1,t + 2*window);CHKERRQ(ierr);
  ierr = TSSolve(tsFine,multi->B);CHKERRQ(ierr); 

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetUp_MultiFHMMHEUN"
static PetscErrorCode SetUp_MultiFHMMHEUN(TS ts)
{
  PetscErrorCode ierr;
  TS_Multi       *multi = (TS_Multi*)ts->data;
  TS             tsCoarse = multi->tsCoarse,tsFine = multi->tsFine;
  TSAdapt        adapt;

  PetscFunctionBegin; 
  /* We use a very naive (new) coarse TS without any timestep adaptivity. Hopefully exisiting TS's (TSSSP, TSRK) can be used in the future, but steps must be taken to ensure
      that they respect the 'basepoint updating' required for this method (and higher order versions) */
  ierr = TSSetType(tsCoarse,TSNAIVEHEUN);CHKERRQ(ierr);

  ierr = TSSetType(tsFine,TSRK);CHKERRQ(ierr); /* to be made user-settable. In particular, exponential integrators could be very effective for oscillatory finescales */ 
  ierr = TSSetType(tsFine,TSRK);CHKERRQ(ierr);
  ierr = TSRKSetType(tsFine,TSRK5F);CHKERRQ(ierr); /*or TSRK3BS perhaps */
  ierr = TSGetAdapt(tsFine,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetType(adapt,TSADAPTBASIC);CHKERRQ(ierr);

  /* Use this TS's vec_sol for the coarse solver */
  ierr = TSSetSolution(tsCoarse,ts->vec_sol);CHKERRQ(ierr);
  
  ierr = TSSetRHSFunction(tsCoarse,NULL,FormRHSFunction_FHMMHEUN,NULL);CHKERRQ(ierr);
  
  /*  Set tolerances for fine solver (temporary - todo is to get them from the main TS)*/
  ierr = TSSetTolerances(tsFine,1e-8,NULL,1e-8,NULL);CHKERRQ(ierr);

  ierr = TSSetTimeStep(tsCoarse,ts->time_step);CHKERRQ(ierr);

  ierr = TSSetTimeStep(tsFine,multi->window/150.);CHKERRQ(ierr); 

  /* Set the fine TS to solve at an exact final time */
  ierr = TSSetExactFinalTime(tsFine,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr); 

  ierr = TSSetDuration(tsFine,PETSC_MAX_INT,PETSC_DEFAULT);CHKERRQ(ierr); /* solve times are set before each TSSolve call */

  /* Create work vectors */
  multi->nwork = 2;
  ierr = VecDuplicateVecs(ts->vec_sol,multi->nwork,&multi->W);CHKERRQ(ierr); /* Note that number of works vecs is hard coded here. Later, this will depend on the order of approximate required when computing F_HMM */

  /* Set basepoint */
  multi->B = ts->vec_sol; /* The assumption here is that this gets dealt with properly by the coarse integrator, which is the case for our hand-coded one but needs to be determined in general, as the 'moving base point' machinery here seems very fragile */

  PetscFunctionReturn(0);
}

/* Methods for FHMMMFE */
#undef __FUNCT__
#define __FUNCT__ "Precoarse_MultiFHMMFE"
static PetscErrorCode Precoarse_MultiFHMMFE(TS ts)
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
  ierr = VecCopy(S,multi->W[0]);CHKERRQ(ierr);

  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetRHSPartitionFunction(dm,TS_MULTI_PARTITION,TS_MULTI_FAST_SLOT,&rhsfunc,&rhsctx);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(tsFine,NULL,rhsfunc,rhsctx);CHKERRQ(ierr);
  ierr = TSSetTime(tsFine,ts->ptime);CHKERRQ(ierr); 
  ierr = TSSetDuration(tsFine,-1,ts->ptime + multi->window);CHKERRQ(ierr);
  ierr = TSSolve(tsFine,S);CHKERRQ(ierr);

  ierr = VecScale(S,1.-(tsCoarse->time_step/multi->window));

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormRHSFunction_FHMMFE"
static PetscErrorCode FormRHSFunction_FHMMFE(TS tsCoarse, PetscReal t, Vec X, Vec F, void* ctx)
{
  PetscErrorCode ierr;  
  TS_Multi       *multi;
  DM             dm;
  void           *rhsctx;
  TSRHSFunction  rhsfunc;
  TS             tsFine, ts;

  PetscFunctionBegin;  

  ierr = TSGetApplicationContext(tsCoarse,&multi);CHKERRQ(ierr);
  tsFine = multi->tsFine;
  ts = multi->ts;
  
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMTSGetRHSPartitionFunction(dm,TS_MULTI_PARTITION,TS_MULTI_FULL_SLOT,&rhsfunc,&rhsctx);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(tsFine,NULL,rhsfunc,rhsctx);CHKERRQ(ierr);
  ierr = TSSetTime(tsFine,ts->ptime);CHKERRQ(ierr);
  ierr = TSSetDuration(tsFine,-1,ts->ptime + multi->window);CHKERRQ(ierr);
  ierr = TSSolve(tsFine,multi->W[0]);CHKERRQ(ierr);

  /* Scale and copy to F*/
 ierr = VecAXPBY(F,1./multi->window,0,multi->W[0]);CHKERRQ(ierr); 

  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "SetUp_MultiFHMMFE"
static PetscErrorCode SetUp_MultiFHMMFE(TS ts)
{
  PetscErrorCode ierr;
  TS_Multi       *multi = (TS_Multi*)ts->data;
  TS             tsCoarse = multi->tsCoarse,tsFine = multi->tsFine;
  TSAdapt        adapt;

  PetscFunctionBegin;
  ierr = TSSetType(tsCoarse,TSEULER);CHKERRQ(ierr);

  /* Todo is to make this inner solver user-settable. Note that
   the integrator must support adaptivity to allow our 'matchstep' setting below. TSSSP for example does not currently do this. Here, this is fairly innoccuous since it has the effect of lengthening the 'window' slightly, but it can be catastrophic for higher order solvers */
  ierr = TSSetType(tsFine,TSRK);CHKERRQ(ierr);
  ierr = TSRKSetType(tsFine,TSRK5F);CHKERRQ(ierr); /*or TSRK3BS perhaps */
  ierr = TSGetAdapt(tsFine,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetType(adapt,TSADAPTBASIC);CHKERRQ(ierr);

  /* Use this TS's work vec_sol for the coarse solver */
  ierr = TSSetSolution(tsCoarse,ts->vec_sol);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(tsCoarse,NULL,FormRHSFunction_FHMMFE,NULL);CHKERRQ(ierr);
  ierr = TSSetTimeStep(tsCoarse,ts->time_step);CHKERRQ(ierr);
  ierr = TSSetTimeStep(tsFine,multi->window/150.);CHKERRQ(ierr); /* fine time step is set arbitrarily (it should be adaptive) */

  /* Set the fine TS to solve at an exact final time */
  ierr = TSSetExactFinalTime(tsFine,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr); 

  ierr = TSSetDuration(tsFine,PETSC_MAX_INT,PETSC_DEFAULT);CHKERRQ(ierr); /* no step limits */
  multi->nwork = 1;
  ierr = VecDuplicateVecs(ts->vec_sol,multi->nwork,&multi->W);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscFunctionList TSMultiList_precoarse = 0;
PetscFunctionList TSMultiList_setup = 0;
static TSMultiType TSMultiDefault = TSMULTIFHMMFE;
static PetscBool  TSMultiPackageInitialized;

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
    ierr = PetscFunctionListAdd(&TSMultiList_precoarse,TSMULTIFHMMFE, Precoarse_MultiFHMMFE);CHKERRQ(ierr);
    ierr = PetscFunctionListAdd(&TSMultiList_setup,TSMULTIFHMMFE, SetUp_MultiFHMMFE);CHKERRQ(ierr);
    ierr = PetscFunctionListAdd(&TSMultiList_precoarse,TSMULTIFHMMHEUN, Precoarse_MultiFHMMHEUN);CHKERRQ(ierr);
    ierr = PetscFunctionListAdd(&TSMultiList_setup,TSMULTIFHMMHEUN, SetUp_MultiFHMMHEUN);CHKERRQ(ierr);
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
PetscErrorCode TSMultiSetType_Multi(TS ts,TSMultiType type)
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
#define __FUNCT__ "TSMultiSetWindow"
/*@C
 TSMultiSetWindow - set the mesoscale time window to use 
 
 Set the mesoscale time window for a multiscale integrator. 
 Different implementations are free to interpet this parameter
 differently, but typically it represents a scale between the 
 timesteps of the fine and coarse integrators, describing
 an amount of time required for the fine scale to suitably
 average or homegenize.

 Input Arguments:
 ts - time stepping object
 window - length of mesoscale window
 
 Level: beginner
 
 .seealso: TSMULTI, TSMultiGetWindow()
 @*/
 PetscErrorCode TSMultiSetWindow(TS ts,PetscReal window)
{
    TS_Multi            *multi = (TS_Multi*)ts->data;
            
    PetscFunctionBegin;
    multi->window = window;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSMultiGetWindow"
/*@C
 TSMultiGetWindow - retrieve the mesoscale time window  
 
 Retrive the mesoscale time window for a multiscale integrator. 
 Different implementations are free to interpet this parameter
 differently, but typically it represents a scale between the 
 timesteps of the fine and coarse integrators, describing
 an amount of time required for the fine scale to suitably
 average or homegenize.

 Input Arguments:
 ts - time stepping object
 window - length of mesoscale window
 
 Level: beginner
 
 .seealso: TSMULTI, TSMultiGetWindow()
 @*/
PetscErrorCode TSMultiGetWindow(TS ts,PetscReal *window)
{
    TS_Multi            *multi = (TS_Multi*)ts->data;
          
    PetscFunctionBegin;
    *window = multi->window;
    PetscFunctionReturn(0);
}

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
        Various functions registered with tsCoarse (PreStep, PostStep, PreStage, IFunctions, RHSFunctions) can also
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

    /* pass in the TS_Multi* pointer as the sub-solvers' user contexts*/
    ierr = TSSetApplicationContext(multi->tsCoarse, multi);CHKERRQ(ierr);
    ierr = TSSetApplicationContext(multi->tsFine,   multi);CHKERRQ(ierr);

    /* todo - pass tolerances and other settings (since user may set them) */

    if(!multi->window){
    SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must call TSMultiSetWindow before TSSetUp");  
    }

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
   
    ierr = VecDestroyVecs(multi->nwork,&multi->W);CHKERRQ(ierr); 

    PetscFree(multi->type_name);
    ierr = PetscFree(ts->data);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_Multi"
static PetscErrorCode TSSetFromOptions_Multi(TS ts)
{
  PetscFunctionBegin;
  char           tname[256];
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

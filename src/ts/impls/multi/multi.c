/*
 Code for Timestepping with a multiscale method
 */
#include <petsc-private/tsimpl.h>                /*I   "petscts.h"   I*/

typedef struct {
    TS tsCoarse, tsFine;
    TSType TSCOARSE, TSFINE;
} TS_Multi;

/*<<< This is a function that later will be registered by a more specific solver  (something like FLAVOR_FE) */
#undef __FUNCT__
#define __FUNCT__ "TakeFineStep"
static PetscErrorCode TakeFineStep(TS ts)
{
  PetscErrorCode ierr;
  TS_Multi       *multi = (TS_Multi*)ts->user; 

  PetscFunctionBegin;
  ierr = TSStep(multi->tsFine);

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

    /*  Take a step with the coarse solver. 
        Various functions registered with tsCoarse (PreStep, PostStep, PreStagIFunctions, RHSFunctions) will invoke tsFine. 
        To do this, note that we put a pointer to this solver's ts->data in multi->tsCoarse->user */
    ierr = TSStep(multi->tsCoarse);
  
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
    TSRHSFunction  rhsFuncSlow, rhsFuncFull;
    void           *ctx;
    DM             dm; 
    
    PetscFunctionBegin;
    ierr = TSSetType(multi->tsCoarse,multi->TSCOARSE);CHKERRQ(ierr);
    ierr = TSSetType(multi->tsFine,multi->TSFINE);CHKERRQ(ierr);

    /* pass in the TS_Multi* pointer as the coarse solver's user context */
    ierr = TSSetApplicationContext(multi->tsCoarse, multi);CHKERRQ(ierr);

    /*<<< Here, we  set the coarse solver's vec_sol to the same vector as this solver's.
      This probably makes sense for the coarse solver, and in some cases (FLAVORS) it makes sense for the fine scale
      solver as well. For HMM we need to retain the state between coarse steps so the fine solver would need its own storage.*/
   ierr = TSSetSolution(multi->tsCoarse,ts->vec_sol);CHKERRQ(ierr);

   /*<<< we do the same for the fine solver. This should work for FLAVOR_FE, but in general the fine solver will
    want its own storage to work with */
   ierr = TSSetSolution(multi->tsFine,ts->vec_sol);CHKERRQ(ierr);

    /*<<<  Set the RHS and IFunctions for the coarse and fine solvers
    This is temporary - since the way this happens depends very heavily on teh type of solver,
    we will probaly have to pass this off to a method to be defined by registered subclasses, so here we'd call 
    multi->ops->setCoarseRHSFunction() etc., perhaps */
    ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
    ierr = DMTSGetRHSPartitionFunction(dm,TS_MULTI_PARTITION,TS_MULTI_SLOW_SLOT,&rhsFuncSlow,&ctx);CHKERRQ(ierr); /*<<< once we have a real package, this should be changed to TSMULTI_FLAVOR_SLOW */
    ierr = TSSetRHSFunction(multi->tsCoarse,NULL,rhsFuncSlow,ctx);CHKERRQ(ierr);
    ierr = DMTSGetRHSPartitionFunction(dm,TS_MULTI_PARTITION,TS_MULTI_FULL_SLOT,&rhsFuncFull,&ctx);CHKERRQ(ierr);
    ierr = TSSetRHSFunction(multi->tsFine,NULL,rhsFuncFull,ctx);CHKERRQ(ierr);

    /*<<< Another type-dependent operation: set the PreStep function to take a fine step: */
    ierr = TSSetPreStep(multi->tsCoarse,TakeFineStep);CHKERRQ(ierr);

    /*<<< Set More parameters for the coarse and fine solvers. Again, this behavior depends on the type of solverm
    so will have to be defined by an op which can be set */

    /*<<< Time steps
    <<< This is all hard coded to test. Really what needs to happen is the TS_Multi (or TS_Multi_FLAVOR) object should 
    contain enough information to allow the user to pick a good meso or macro timestep based on the stiffness parameter
    (and this stiffness parameter itself might be something the solver knows about) */
    ierr = TSSetTimeStep(multi->tsCoarse, ts->time_step * 0.995);CHKERRQ(ierr);
    ierr = TSSetTimeStep(multi->tsFine,   ts->time_step * 0.005);CHKERRQ(ierr);

    ierr = TSSetUp(multi->tsCoarse);CHKERRQ(ierr);
    ierr = TSSetUp(multi->tsFine);CHKERRQ(ierr);

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
    ierr = PetscFree(ts->data);CHKERRQ(ierr);
    
    ierr = TSDestroy(&multi->tsCoarse);CHKERRQ(ierr);
    ierr = TSDestroy(&multi->tsFine);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_Multi"
static PetscErrorCode TSSetFromOptions_Multi(TS ts)
{
    PetscFunctionBegin;
    
    /*<<< How options passed in should be propagated to the coarse and fine solvers might depend
    on which type of multiscale method we're using, and TSMULTI has its own options

    We will never expose the internal TS objects to the user. If they want to tweak
    beyond the provided flavors of TSMULTI (and the options they accept), we should provide the user
    with the ability to register new timesteppers in a TSMULTI package (and potential subpackages like TSFLAVOR, TSHMM) */
    
    PetscFunctionReturn(0);
}

/*MC
 TSMULTI- Multiscale ODE Solver
 
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
    ts->ops->setup           = TSSetUp_Multi;
    ts->ops->step            = TSStep_Multi;
    ts->ops->reset           = TSReset_Multi;
    ts->ops->destroy         = TSDestroy_Multi;
    ts->ops->setfromoptions  = TSSetFromOptions_Multi;
    
    ierr = PetscNewLog(ts,TS_Multi,&multi);CHKERRQ(ierr);
    ts->data = multi;
   
    multi->TSCOARSE = TSEULER; /*<<< default - other solvers will want to change this */
    multi->TSFINE   = TSEULER; /*<<< default */
    
    ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
    ierr = TSCreate(comm,&multi->tsCoarse);CHKERRQ(ierr);
    ierr = TSCreate(comm,&multi->tsFine);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

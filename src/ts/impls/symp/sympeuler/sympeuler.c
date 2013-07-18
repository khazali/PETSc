/*
  Code for timestepping with the Symplectic Euler method
*/
#include <petsc-private/tsimpl.h> /*I "petscts.h" I*/

typedef struct {
  Vec update; /* Work vector */
} TS_SympEuler;

#undef __FUNCT__
#define __FUNCT__ "TSStep_SympEuler"
static PetscErrorCode TSStep_SympEuler(TS ts)
{

  TS_SympEuler    *sympeuler = (TS_SympEuler*)ts->data;
  Vec            sol = ts->vec_sol,update = sympeuler->update;
  PetscErrorCode ierr;
  // ..
  PetscFunctionBegin;

  // Note -  there are currently no checks anywhere to ensure that you have actually set 
  //  the required RHS functions - rather the behavior is that if they don't exist, nothing happens.
  //  This might not be a good choice.

  // Note - there are two variants on symplectic euler ( which of p or q is treated as implicit)
  //   Here we choose one arbitrarily, but later when we construct a proper subpackage we can
  //   have two different integrators (of course, practically speaking, the user could just switch which variables are called P and which are called Q)

  // Implicit Step in Q
  //..
  // Explicit Step in P
  //..
  // DEBUG - take both explicit steps for now
  // (This works for separable mechanical Lagrangian type problems - the trick is to reproduce 
  //    the speed and simplicity for that system when we can, and deal with the actual implicit solve when we need to)
  //

  //? If the user specified an IFunction, this isn't going to pick it up I don't think. 
  //  I'm not certain if the euler method respects it either.

  ierr = TSPreStep(ts);CHKERRQ(ierr);
  ierr = TSPreStage(ts,ts->ptime);CHKERRQ(ierr);
  ierr = TSComputeRHSPartitionFunction(ts,SYMPLECTIC,SYMPLECTIC_P,ts->ptime,sol,update);CHKERRQ(ierr);
  ierr = VecAXPY(sol,ts->time_step,update);CHKERRQ(ierr);
  ierr = TSComputeRHSPartitionFunction(ts,SYMPLECTIC,SYMPLECTIC_Q,ts->ptime,sol,update);CHKERRQ(ierr);
  ierr = VecAXPY(sol,ts->time_step,update);CHKERRQ(ierr);
  ts->ptime += ts->time_step;
  ts->steps++;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetUp_SympEuler"
static PetscErrorCode TSSetUp_SympEuler(TS ts)
{
  TS_SympEuler       *sympeuler = (TS_SympEuler*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&sympeuler->update);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSReset_SympEuler"
static PetscErrorCode TSReset_SympEuler(TS ts)
{
  TS_SympEuler       *sympeuler = (TS_SympEuler*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&sympeuler->update);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDestroy_SympEuler"
static PetscErrorCode TSDestroy_SympEuler(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_SympEuler(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_SympEuler"
static PetscErrorCode TSSetFromOptions_SympEuler(TS ts)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSView_SympEuler"
static PetscErrorCode TSView_SympEuler(TS ts,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSInterpolate_SympEuler"
static PetscErrorCode TSInterpolate_SympEuler(TS ts,PetscReal t,Vec X)
{
  PetscReal      alpha = (ts->ptime - t)/ts->time_step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecAXPBY(ts->vec_sol,1.0-alpha,alpha,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSComputeLinearStability_SympEuler"
PetscErrorCode TSComputeLinearStability_SympEuler(TS ts,PetscReal xr,PetscReal xi,PetscReal *yr,PetscReal *yi)
{
  //!! This is just copied from euler !!
  PetscFunctionBegin;
  *yr = 1.0 + xr;
  *yi = xi;
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------ */

/*MC
      TSSYMPEULER - ODE solver using the explicit forward Euler method

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSEULER, TSBEULER

M*/
#undef __FUNCT__
#define __FUNCT__ "TSCreate_SympEuler"
PETSC_EXTERN PetscErrorCode TSCreate_SympEuler(TS ts)
{
  TS_SympEuler       *sympeuler;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ts->ops->setup           = TSSetUp_SympEuler;
  ts->ops->step            = TSStep_SympEuler;
  ts->ops->reset           = TSReset_SympEuler;
  ts->ops->destroy         = TSDestroy_SympEuler;
  ts->ops->setfromoptions  = TSSetFromOptions_SympEuler;
  ts->ops->view            = TSView_SympEuler;
  ts->ops->interpolate     = TSInterpolate_SympEuler;
  ts->ops->linearstability = TSComputeLinearStability_SympEuler;

  ierr = PetscNewLog(ts,TS_SympEuler,&sympeuler);CHKERRQ(ierr);
  ts->data = (void*)sympeuler;
  PetscFunctionReturn(0);
}

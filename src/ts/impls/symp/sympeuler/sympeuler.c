/*
  Code for timestepping with the Symplectic Euler method
*/
#include <petsc-private/tsimpl.h> /*I "petscts.h" I*/

typedef struct {
  Vec update;  /* Work vector for explicit steps (could use one of the ones below)*/
  Vec X0,Xdot; /* work vectors for implicit steps */
  PetscReal stage_time;
} TS_SympEuler;

#undef __FUNCT__
#define __FUNCT__ "TSStep_SympEuler"
static PetscErrorCode TSStep_SympEuler(TS ts)
{

  PetscErrorCode ierr;
  TS_SympEuler       *sympeuler = (TS_SympEuler*)ts->data;
  Vec                 sol = ts->vec_sol,update = sympeuler->update; // todo just use on of the other vectors for update
  PetscInt            its,lits;
  SNESConvergedReason snesreason;

  PetscFunctionBegin;

  // Note -  there are currently no checks anywhere to ensure that you have actually set 
  //  the required RHS functions - rather the behavior is that if they don't exist, nothing happens.
  //  This might not be a good choice.

  // Note - there are two variants on symplectic euler ( which of p or q is treated as implicit)
  // Here, we integrate explicitly in Q, but later on we could add an option to use the other slot's RHSfunction
  // (Practically speaking, the user can of course just switch the meanings of P and Q, but that is a bad interface)
  
  ierr = TSPreStep(ts);CHKERRQ(ierr);
  ierr = TSPreStage(ts,ts->ptime);CHKERRQ(ierr);

  sympeuler->stage_time = ts->ptime + ts->time_step; // full time step (no adaptation)

  // Implicit Step in P
  // Note that for separable systems, this can (and should) just be another euler step. This can hopefully
  //   be dealt with at the level of assigning an RHSFunction to a special value (as we can with linear or constant functions)
  /* Note that this choice is hard-coded below in SNESTSFormFunction_SympEuler and SNESTSFormJacobian_SympEuler */
  ierr = VecCopy(sol,sympeuler->X0);CHKERRQ(ierr);
  ierr = SNESSolve(ts->snes,NULL,sol);CHKERRQ(ierr); // Solve F(u,udot,t)==0 
  ierr = SNESGetIterationNumber(ts->snes,&its);CHKERRQ(ierr);
  ierr = SNESGetLinearSolveIterations(ts->snes,&lits);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(ts->snes,&snesreason);CHKERRQ(ierr);
  ts->snes_its += its; ts->ksp_its += lits;
  //..
  // Explicit Step in Q

  //? If the user specified a nontrivial IFunction, this implicit step isn't going to pick it up I don't think. 
  //  I'm not certain if the euler method respects it either.

  // DEBUG - take both explicit steps for now
  //ierr = TSComputeRHSPartitionFunction(ts,SYMPLECTIC,SYMPLECTIC_P,ts->ptime,sol,update);CHKERRQ(ierr);
  //ierr = VecAXPY(sol,ts->time_step,update);CHKERRQ(ierr);
  ierr = TSComputeRHSPartitionFunction(ts,SYMPLECTIC,SYMPLECTIC_Q,ts->ptime,sol,update);CHKERRQ(ierr);
  ierr = VecAXPY(sol,ts->time_step,update);CHKERRQ(ierr);
  ts->ptime += ts->time_step;
  ts->steps++;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormFunction_SympEuler"
static PetscErrorCode SNESTSFormFunction_SympEuler(SNES snes,Vec x,Vec y,TS ts)
{
  TS_SympEuler    *sympeuler = (TS_SympEuler*)ts->data;
  PetscErrorCode  ierr;
  DM              dm,dmsave; // just copying blindly wrt these
  Vec             X0 = sympeuler->X0,Xdot = sympeuler->Xdot;  
  PetscReal       shift = 1./(ts->time_step);

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = VecAXPBYPCZ(Xdot,-shift,shift,0,X0,x);CHKERRQ(ierr); // Xdot = shift(x-X0)

  /* DM monkey-business allows user code to call TSGetDM() inside of functions evaluated on levels of FAS */
  dmsave = ts->dm;
  ts->dm = dm;

  /* Note that the P step being implicit is hard-coded. This can/should change at some point to allow both sympeuler variants */

  ierr   = TSComputeIPartitionFunction(ts,SYMPLECTIC,SYMPLECTIC_P,sympeuler->stage_time,x,Xdot,y,PETSC_FALSE);CHKERRQ(ierr);
  ts->dm = dmsave;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormJacobian_SympEuler"
static PetscErrorCode SNESTSFormJacobian_SympEuler(SNES snes,Vec x,Mat *A,Mat *B,MatStructure *str,TS ts)
{
  TS_SympEuler   *sympeuler  = (TS_SympEuler*)ts->data;
  PetscErrorCode ierr;
  DM             dm,dmsave;
  PetscReal      shift = 1./(ts->time_step);
  Vec            Xdot = sympeuler->Xdot;  

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);

  /* th->Xdot has already been computed in SNESTSFormFunction_Theta (SNES guarantees this) */

  dmsave = ts->dm;
  ts->dm = dm;

  /* Note that the P step being implicit is hard-coded. This can/should change at some point to allow both sympeuler variants */
  ierr   = TSComputeIPartitionJacobian(ts,SYMPLECTIC,SYMPLECTIC_P,sympeuler->stage_time,x,Xdot,shift,A,B,str,PETSC_FALSE);CHKERRQ(ierr);
  ts->dm = dmsave;
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
  ierr = VecDuplicate(ts->vec_sol,&sympeuler->X0);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&sympeuler->Xdot);CHKERRQ(ierr);
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
  ierr = VecDestroy(&sympeuler->X0);CHKERRQ(ierr);
  ierr = VecDestroy(&sympeuler->Xdot);CHKERRQ(ierr);
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

/* ------------------------------------------------------------ */

/*MC
      TSSYMPEULER - ODE solver which takes Backwards Euler steps in the 'P' variables and Euler steps in the 'Q' variables

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
  ts->ops->snesfunction    = SNESTSFormFunction_SympEuler;
  ts->ops->snesjacobian    = SNESTSFormJacobian_SympEuler;

  ierr = PetscNewLog(ts,TS_SympEuler,&sympeuler);CHKERRQ(ierr);
  ts->data = (void*)sympeuler;
  PetscFunctionReturn(0);
}

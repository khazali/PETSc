/*
       Code for Timestepping with naive Heun method
*/
#include <petsc-private/tsimpl.h>                /*I   "petscts.h"   I*/

typedef struct {
  Vec update,k1;     
} TS_NaiveHeun;

#undef __FUNCT__
#define __FUNCT__ "TSStep_NaiveHeun"
static PetscErrorCode TSStep_NaiveHeun(TS ts)
{
  TS_NaiveHeun       *naiveheun = (TS_NaiveHeun*)ts->data;
  Vec            sol    = ts->vec_sol,update = naiveheun->update, k1 = naiveheun->k1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSPreStep(ts);CHKERRQ(ierr);
  ierr = TSPreStage(ts,ts->ptime);CHKERRQ(ierr);

  /* Note that crucially this allows for 'moving the basepoint' */
  ierr = TSComputeRHSFunction(ts,ts->ptime,sol,update);CHKERRQ(ierr);/* this could change sol */
  ierr = VecCopy(sol,k1);
  ierr = VecAXPY(k1,0.5*ts->time_step,update);CHKERRQ(ierr);
  ierr = TSComputeRHSFunction(ts,ts->ptime + 0.5*ts->time_step,k1,update); /* this could change sol again, if sol is designated as 'B' in a TS_Multi object */
  ierr = VecAXPY(sol,ts->time_step,update); 

  ts->ptime += ts->time_step;
  ts->steps++;
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TSSetUp_NaiveHeun"
static PetscErrorCode TSSetUp_NaiveHeun(TS ts)
{
  TS_NaiveHeun       *naiveheun = (TS_NaiveHeun*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&naiveheun->update);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&naiveheun->k1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSReset_NaiveHeun"
static PetscErrorCode TSReset_NaiveHeun(TS ts)
{
  TS_NaiveHeun       *naiveheun = (TS_NaiveHeun*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&naiveheun->update);CHKERRQ(ierr);
  ierr = VecDestroy(&naiveheun->k1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDestroy_NaiveHeun"
static PetscErrorCode TSDestroy_NaiveHeun(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_NaiveHeun(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_NaiveHeun"
static PetscErrorCode TSSetFromOptions_NaiveHeun(TS ts)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSView_NaiveHeun"
static PetscErrorCode TSView_NaiveHeun(TS ts,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */

/*MC
      TSnaiveheun - ODE solver using Heun's method (aka the Midpoint rule)
               
      Note: this is for debugging, a simple example, and as a temporary
       coarse solver for TSMULTIFHMMHEUN . See TSSSP, TSARKIMEX, TSRK for RK methods with adaptivity and other features

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSBnaiveheun

M*/
#undef __FUNCT__
#define __FUNCT__ "TSCreate_NaiveHeun"
PETSC_EXTERN PetscErrorCode TSCreate_NaiveHeun(TS ts)
{
  TS_NaiveHeun       *naiveheun;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ts->ops->setup           = TSSetUp_NaiveHeun;
  ts->ops->step            = TSStep_NaiveHeun;
  ts->ops->reset           = TSReset_NaiveHeun;
  ts->ops->destroy         = TSDestroy_NaiveHeun;
  ts->ops->setfromoptions  = TSSetFromOptions_NaiveHeun;
  ts->ops->view            = TSView_NaiveHeun;

  ierr = PetscNewLog(ts,TS_NaiveHeun,&naiveheun);CHKERRQ(ierr);
  ts->data = (void*)naiveheun;
  PetscFunctionReturn(0);
}

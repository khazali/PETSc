
#include <../src/snes/impls/newtonas/newtonasimpl.h> /*I "petscsnes.h" I*/
#include <petsc-private/kspimpl.h>
#include <petsc-private/matimpl.h>
#include <petsc-private/dmimpl.h>
#include <petsc-private/vecimpl.h>



#undef __FUNCT__
#define __FUNCT__ "SNESSolve_NEWTONAS_Primal"
PetscErrorCode SNESSolve_NEWTONAS_Primal(SNES snes)
{
  SNES_NEWTONAS      *newtas = (SNES_NEWTONAS*)snes->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* TODO: Set things up. */

  ierr = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);

  ierr = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);

  ierr = SNESVIProjectOntoBounds(snes,X);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }
  ierr = SNESVIComputeInactiveSetFnorm(snes,F,X,&fnorm);CHKERRQ(ierr);
  ierr = VecNormBegin(X,NORM_2,&xnorm);CHKERRQ(ierr);        /* xnorm <- ||x||  */
  ierr = VecNormEnd(X,NORM_2,&xnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");

  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  ierr       = SNESLogConvergenceHistory(snes,fnorm,0);CHKERRQ(ierr);
  ierr       = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);

  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);


  for (i=0; i<maxits; i++) {

    IS         IS_act,IS_inact; /* _act -> active set _inact -> inactive set */
    IS         IS_redact; /* redundant active set */
    VecScatter scat_act,scat_inact;
    PetscInt   nis_act,nis_inact;
    Vec        Y_act,Y_inact,F_inact;
    Mat        jac_inact_inact,prejac_inact_inact;
    PetscBool  isequal;

    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
    ierr = SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);


    /* Create active and inactive index sets */

    /*original
    ierr = SNESVICreateIndexSets_RS(snes,X,F,&IS_act,&IS_inact);CHKERRQ(ierr);
     */
    ierr = SNESVIGetActiveSetIS(snes,X,F,&IS_act);CHKERRQ(ierr);

    if (vi->checkredundancy) {
      (*vi->checkredundancy)(snes,IS_act,&IS_redact,vi->ctxP);CHKERRQ(ierr);
      if (IS_redact) {
        ierr = ISSort(IS_redact);CHKERRQ(ierr);
        ierr = ISComplement(IS_redact,X->map->rstart,X->map->rend,&IS_inact);CHKERRQ(ierr);
        ierr = ISDestroy(&IS_redact);CHKERRQ(ierr);
      } else {
        ierr = ISComplement(IS_act,X->map->rstart,X->map->rend,&IS_inact);CHKERRQ(ierr);
      }
    } else {
      ierr = ISComplement(IS_act,X->map->rstart,X->map->rend,&IS_inact);CHKERRQ(ierr);
    }


    /* Create inactive set submatrix */
    ierr = MatGetSubMatrix(snes->jacobian,IS_inact,IS_inact,MAT_INITIAL_MATRIX,&jac_inact_inact);CHKERRQ(ierr);

    if (0) {                    /* Dead code (temporary developer hack) */
      IS keptrows;
      ierr = MatFindNonzeroRows(jac_inact_inact,&keptrows);CHKERRQ(ierr);
      if (keptrows) {
        PetscInt       cnt,*nrows,k;
        const PetscInt *krows,*inact;
        PetscInt       rstart=jac_inact_inact->rmap->rstart;

        ierr = MatDestroy(&jac_inact_inact);CHKERRQ(ierr);
        ierr = ISDestroy(&IS_act);CHKERRQ(ierr);

        ierr = ISGetLocalSize(keptrows,&cnt);CHKERRQ(ierr);
        ierr = ISGetIndices(keptrows,&krows);CHKERRQ(ierr);
        ierr = ISGetIndices(IS_inact,&inact);CHKERRQ(ierr);
        ierr = PetscMalloc1(cnt,&nrows);CHKERRQ(ierr);
        for (k=0; k<cnt; k++) nrows[k] = inact[krows[k]-rstart];
        ierr = ISRestoreIndices(keptrows,&krows);CHKERRQ(ierr);
        ierr = ISRestoreIndices(IS_inact,&inact);CHKERRQ(ierr);
        ierr = ISDestroy(&keptrows);CHKERRQ(ierr);
        ierr = ISDestroy(&IS_inact);CHKERRQ(ierr);

        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)snes),cnt,nrows,PETSC_OWN_POINTER,&IS_inact);CHKERRQ(ierr);
        ierr = ISComplement(IS_inact,F->map->rstart,F->map->rend,&IS_act);CHKERRQ(ierr);
        ierr = MatGetSubMatrix(snes->jacobian,IS_inact,IS_inact,MAT_INITIAL_MATRIX,&jac_inact_inact);CHKERRQ(ierr);
      }
    }
    ierr = DMSetVI(snes->dm,IS_inact);CHKERRQ(ierr);
    /* remove later */

    /*
    ierr = VecView(vi->xu,PETSC_VIEWER_BINARY_(((PetscObject)(vi->xu))->comm));CHKERRQ(ierr);
    ierr = VecView(vi->xl,PETSC_VIEWER_BINARY_(((PetscObject)(vi->xl))->comm));CHKERRQ(ierr);
    ierr = VecView(X,PETSC_VIEWER_BINARY_(PetscObjectComm((PetscObject)X)));CHKERRQ(ierr);
    ierr = VecView(F,PETSC_VIEWER_BINARY_(PetscObjectComm((PetscObject)F)));CHKERRQ(ierr);
    ierr = ISView(IS_inact,PETSC_VIEWER_BINARY_(PetscObjectComm((PetscObject)IS_inact)));CHKERRQ(ierr);
     */

    /* Get sizes of active and inactive sets */
    ierr = ISGetLocalSize(IS_act,&nis_act);CHKERRQ(ierr);
    ierr = ISGetLocalSize(IS_inact,&nis_inact);CHKERRQ(ierr);

    /* Create active and inactive set vectors */
    ierr = SNESCreateSubVectors_VINEWTONRSLS(snes,nis_inact,&F_inact);CHKERRQ(ierr);
    ierr = SNESCreateSubVectors_VINEWTONRSLS(snes,nis_act,&Y_act);CHKERRQ(ierr);
    ierr = SNESCreateSubVectors_VINEWTONRSLS(snes,nis_inact,&Y_inact);CHKERRQ(ierr);

    /* Create scatter contexts */
    ierr = VecScatterCreate(Y,IS_act,Y_act,NULL,&scat_act);CHKERRQ(ierr);
    ierr = VecScatterCreate(Y,IS_inact,Y_inact,NULL,&scat_inact);CHKERRQ(ierr);

    /* Do a vec scatter to active and inactive set vectors */
    ierr = VecScatterBegin(scat_inact,F,F_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_inact,F,F_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    ierr = VecScatterBegin(scat_act,Y,Y_act,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_act,Y,Y_act,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    ierr = VecScatterBegin(scat_inact,Y,Y_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_inact,Y,Y_inact,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    /* Active set direction = 0 */
    ierr = VecSet(Y_act,0);CHKERRQ(ierr);
    if (snes->jacobian != snes->jacobian_pre) {
      ierr = MatGetSubMatrix(snes->jacobian_pre,IS_inact,IS_inact,MAT_INITIAL_MATRIX,&prejac_inact_inact);CHKERRQ(ierr);
    } else prejac_inact_inact = jac_inact_inact;

    ierr = ISEqual(vi->IS_inact_prev,IS_inact,&isequal);CHKERRQ(ierr);
    if (!isequal) {
      ierr = SNESVIResetPCandKSP(snes,jac_inact_inact,prejac_inact_inact);CHKERRQ(ierr);
    }

    /*      ierr = ISView(IS_inact,0);CHKERRQ(ierr); */
    /*      ierr = ISView(IS_act,0);CHKERRQ(ierr);*/
    /*      ierr = MatView(snes->jacobian_pre,0); */



    ierr = KSPSetOperators(snes->ksp,jac_inact_inact,prejac_inact_inact);CHKERRQ(ierr);
    ierr = KSPSetUp(snes->ksp);CHKERRQ(ierr);
    {
      PC        pc;
      PetscBool flg;
      ierr = KSPGetPC(snes->ksp,&pc);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)pc,PCFIELDSPLIT,&flg);CHKERRQ(ierr);
      if (flg) {
        KSP *subksps;
        ierr = PCFieldSplitGetSubKSP(pc,NULL,&subksps);CHKERRQ(ierr);
        ierr = KSPGetPC(subksps[0],&pc);CHKERRQ(ierr);
        ierr = PetscFree(subksps);CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)pc,PCBJACOBI,&flg);CHKERRQ(ierr);
        if (flg) {
          PetscInt       n,N = 101*101,j,cnts[3] = {0,0,0};
          const PetscInt *ii;

          ierr = ISGetSize(IS_inact,&n);CHKERRQ(ierr);
          ierr = ISGetIndices(IS_inact,&ii);CHKERRQ(ierr);
          for (j=0; j<n; j++) {
            if (ii[j] < N) cnts[0]++;
            else if (ii[j] < 2*N) cnts[1]++;
            else if (ii[j] < 3*N) cnts[2]++;
          }
          ierr = ISRestoreIndices(IS_inact,&ii);CHKERRQ(ierr);

          ierr = PCBJacobiSetTotalBlocks(pc,3,cnts);CHKERRQ(ierr);
        }
      }
    }

    ierr = KSPSolve(snes->ksp,F_inact,Y_inact);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(snes->ksp,&kspreason);CHKERRQ(ierr);
    if (kspreason < 0) {
      if (++snes->numLinearSolveFailures >= snes->maxLinearSolveFailures) {
        ierr         = PetscInfo2(snes,"iter=%D, number linear solve failures %D greater than current SNES allowed, stopping solve\n",snes->iter,snes->numLinearSolveFailures);CHKERRQ(ierr);
        snes->reason = SNES_DIVERGED_LINEAR_SOLVE;
        break;
      }
     }

    ierr = VecScatterBegin(scat_act,Y_act,Y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_act,Y_act,Y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(scat_inact,Y_inact,Y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat_inact,Y_inact,Y,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

    ierr = VecDestroy(&F_inact);CHKERRQ(ierr);
    ierr = VecDestroy(&Y_act);CHKERRQ(ierr);
    ierr = VecDestroy(&Y_inact);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scat_act);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scat_inact);CHKERRQ(ierr);
    ierr = ISDestroy(&IS_act);CHKERRQ(ierr);
    if (!isequal) {
      ierr = ISDestroy(&vi->IS_inact_prev);CHKERRQ(ierr);
      ierr = ISDuplicate(IS_inact,&vi->IS_inact_prev);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&IS_inact);CHKERRQ(ierr);
    ierr = MatDestroy(&jac_inact_inact);CHKERRQ(ierr);
    if (snes->jacobian != snes->jacobian_pre) {
      ierr = MatDestroy(&prejac_inact_inact);CHKERRQ(ierr);
    }
    ierr              = KSPGetIterationNumber(snes->ksp,&lits);CHKERRQ(ierr);
    snes->linear_its += lits;
    ierr              = PetscInfo2(snes,"iter=%D, linear solve iterations=%D\n",snes->iter,lits);CHKERRQ(ierr);
    /*
    if (snes->ops->precheck) {
      PetscBool changed_y = PETSC_FALSE;
      ierr = (*snes->ops->precheck)(snes,X,Y,snes->precheck,&changed_y);CHKERRQ(ierr);
    }

    if (PetscLogPrintInfo) {
      ierr = SNESVICheckResidual_Private(snes,snes->jacobian,F,Y,G,W);CHKERRQ(ierr);
    }
    */
    /* Compute a (scaled) negative update in the line search routine:
         Y <- X - lambda*Y
       and evaluate G = function(Y) (depends on the line search).
    */
    ierr  = VecCopy(Y,snes->vec_sol_update);CHKERRQ(ierr);
    ynorm = 1; gnorm = fnorm;
    ierr  = SNESLineSearchApply(snes->linesearch, X, F, &gnorm, Y);CHKERRQ(ierr);
    ierr  = SNESLineSearchGetSuccess(snes->linesearch, &lssucceed);CHKERRQ(ierr);
    ierr  = SNESLineSearchGetNorms(snes->linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);
    ierr  = PetscInfo4(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n",(double)fnorm,(double)gnorm,(double)ynorm,(int)lssucceed);CHKERRQ(ierr);
    if (snes->reason == SNES_DIVERGED_FUNCTION_COUNT) break;
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      ierr         = DMDestroyVI(snes->dm);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    if (!lssucceed) {
      if (++snes->numFailures >= snes->maxFailures) {
        PetscBool ismin;
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        ierr         = SNESVICheckLocalMin_Private(snes,snes->jacobian,F,X,gnorm,&ismin);CHKERRQ(ierr);
        if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        break;
      }
    }
    /* Update function and solution vectors */
    fnorm = gnorm;
    /* Monitor convergence */
    ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    ierr       = SNESLogConvergenceHistory(snes,snes->norm,lits);CHKERRQ(ierr);
    ierr       = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence, xnorm = || X || */
    if (snes->ops->converged != SNESConvergedSkip) { ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr); }
    ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;
  }
  ierr = DMDestroyVI(snes->dm);CHKERRQ(ierr);
  if (i == maxits) {
    ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",maxits);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}



#undef  __FUNCT__
#define __FUNCT__ "SNESNEWTONASSetType"
/*@C
  SNESNEWTONASSetType - set the type of the solver handling the saddle-point problem
  underlying the active-set Newton linesearch method solving the nonlinear constrained
  (variational inequality) problem.

  Logically collective on SNES.

  Input Parameters:
+ snes  - the SNES context
- type  - SNESNEWTONAS solver type

  Level: intermediate

  Options Database Keys: -snes_newtonas_type primal|dual|saddle

.keywords: SNES constraints, saddle-point

.seealso: SNESSetType(), SNESNEWTONASGetType()
@*/
PetscErrorCode SNESNEWTONASSetType(SNES snes,SNESNEWTONASType type)
{
  PetscErrorCode    ierr;
  SNES_NEWTONAS     *newtas = (SNES_NEWTONAS*) snes->data;

  PetscFunctionBegin;
  newtas->type = type;
  if (type == SNES_NEWTONAS_PRIMAL) snes->ops->solve = SNESSolve_NEWTONAS_Primal;
  else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"SNESNEWTONAS solver type not supported: %s",type);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SNESNEWTONASGetType"
/*@C
  SNESNEWTONASGetType - retrieve the type of the solver handling the saddle-point problem
  underlying the active-set Newton linesearch method solving the nonlinear constrained
  (variational inequality) problem.

  Not collective

  Input parameter:
. snes  - the SNES context

  Output parameter:
. type  - SNESNEWTONAS solver type

  Level: intermediate

 .keywords: SNES constraints, saddle-point

.seealso: SNESGetType(), SNESNEWTONASSetType()
@*/
PetscErrorCode SNESNEWTONASGetType(SNES snes,SNESNEWTONASType *type)
{
  PetscErrorCode    ierr;
  SNES_NEWTONAS     *newtas = (SNES_NEWTONAS*) snes->data;

  PetscFunctionBegin;
  if (type) *type = newtas->type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESActiveConstraints_Default"
PETSC_INTERN PetscErrorCode SNESActiveConstraints_Default(SNES snes,Vec x,IS *active,IS *basis,void *ctx)
{
  PetscErrorCode    ierr;
  SNES_NEWTONAS     *newtas = (SNES_NEWTONAS*) snes->data;

  PetscFunctionBegin;
  /* TODO: Apply QR or SVD to a redundant B? */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_NEWTONAS"
PETSC_INTERN PetscErrorCode SNESSetFromOptions_NEWTONAS(SNES snes)
{
  PetscErrorCode    ierr;
  SNES_NEWTONAS     *newtas = (SNES_NEWTONAS*) snes->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("NewtonAS solver options");CHKERRQ(ierr);
  as->type = SNES_NEWTONAS_PRIMAL;
  ierr = PetscOptionsBool("-snes_newtonas_type","Type of linear solver to use the saddle-point problem","SNESASSetType",newtas->type,&newtas->type,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_NEWTONAS"
PETSC_INTERN PetscErrorCode SNESSetUp_NEWTONAS(SNES snes)
{
  PetscErrorCode    ierr;
  SNES_VINEWTONRSLS *as = (SNES_NEWTONAS*) snes->data;

  PetscFunctionBegin;
  if (!as->type) {
    ierr = SNESNEWTONASSetType(snes,SNES_NEWTONAS_PRIMAL);CHKERRQ(ierr);
  }
  /* TODO: Set up the specific linear solvers, etc. */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_NEWTONAS"
PETSC_INTERN PetscErrorCode SNESDestroy_NEWTONAS(SNES snes)
{
  PetscErrorCode    ierr;
  SNES_VINEWTONRSLS *as = (SNES_NEWTONAS*) snes->data;

  PetscFunctionBegin;
  /* TODO: Tear things down. */
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
      SNESNEWTONAS - Augmented-space active-set linesearch-based Newton-like solver
      for nonlinear problems with constraints (variational inequalities or mixed
      complementarity problems).

   Options Database:
.   -snes_newtonas_type primal|dual|saddle


   Level: beginner

   References:
   - T. S. Munson, and S. Benson. Flexible Complementarity Solvers for Large-Scale
     Applications, Optimization Methods and Software, 21 (2006).

.seealso:  SNES, SNESCreate(), SNESSetType(), SNESLineSearchSet(),
           SNESSetFunction(), SNESSetJacobian(), SNESSetConstraintFunction(), SNESSetConstraintJacobian(),
           SNESSetActiveConstraints(), SNESSetProjectOntoConstraints(), SNESSetDistanceToConstraints(),
           SNESNEWTONASSetType()

M*/
#undef __FUNCT__
#define __FUNCT__ "SNESCreate_NEWTONAS"
PETSC_EXTERN PetscErrorCode SNESCreate_NEWTONAS(SNES snes)
{
  PetscErrorCode    ierr;
  SNES_NEWTONAS     *newtas;

  PetscFunctionBegin;
  snes->ops->setup          = SNESSetUp_NEWTONAS;
  snes->ops->solve          = SNESSolve_NEWTONAS;
  snes->ops->destroy        = SNESDestroy_NEWTONAS;
  snes->ops->setfromoptions = SNESSetFromOptions_NEWTONAS;

  ierr                = PetscNewLog(snes,&newtas);CHKERRQ(ierr);
  snes->data          = (void*)newtas;
  PetscFunctionReturn(0);
}

#include <petsc-private/linesearchimpl.h>
#include <petsc-private/snesimpl.h>

typedef struct {
  PetscReal *memory;

  PetscReal alpha;                      /* Initial reference factor >= 1 */
  PetscReal beta;                       /* Steplength determination < 1 */
  PetscReal beta_inf;           /* Steplength determination < 1 */
  PetscReal sigma;                      /* Acceptance criteria < 1) */
  PetscReal minimumStep;                /* Minimum step size */
  PetscReal lastReference;              /* Reference value of last iteration */

  PetscInt memorySize;          /* Number of functions kept in memory */
  PetscInt current;                     /* Current element for FIFO */
  PetscInt referencePolicy;             /* Integer for reference calculation rule */
  PetscInt replacementPolicy;   /* Policy for replacing values in memory */

  PetscBool nondescending;
  PetscBool memorySetup;


  Vec x;        /* Maintain reference to variable vector to check for changes */
  Vec work;
  Vec workstep;
} SNESLineSearch_BTSD;

#define REPLACE_FIFO 1
#define REPLACE_MRU  2

#define REFERENCE_MAX  1
#define REFERENCE_AVE  2
#define REFERENCE_MEAN 3

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchDestroy_BTSD"
static PetscErrorCode SNESLineSearchDestroy_BTSD(SNESLineSearch ls)
{
  SNESLineSearch_BTSD *btsdP = (SNESLineSearch_BTSD *)ls->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscFree(btsdP->memory);CHKERRQ(ierr);
  ierr = VecDestroy(&btsdP->x);CHKERRQ(ierr);
  ierr = VecDestroy(&btsdP->work);CHKERRQ(ierr);
  ierr = VecDestroy(&btsdP->workstep);CHKERRQ(ierr);
  ierr = PetscFree(ls->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchReset_BTSD"
static PetscErrorCode SNESLineSearchReset_BTSD(SNESLineSearch ls)
{
  SNESLineSearch_BTSD *btsdP = (SNESLineSearch_BTSD *)ls->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  if (btsdP->memory != NULL) {
    ierr = PetscFree(btsdP->memory);CHKERRQ(ierr);
  }
  btsdP->memorySetup = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetFromOptions_BTSD"
static PetscErrorCode SNESLineSearchSetFromOptions_BTSD(SNESLineSearch ls)
{
  SNESLineSearch_BTSD *btsdP = (SNESLineSearch_BTSD *)ls->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("BTSD linesearch options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_linesearch_btsd_alpha", "initial reference constant", "", btsdP->alpha, &btsdP->alpha, 0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_linesearch_btsd_beta_inf", "decrease constant one", "", btsdP->beta_inf, &btsdP->beta_inf, 0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_linesearch_btsd_beta", "decrease constant", "", btsdP->beta, &btsdP->beta, 0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_linesearch_btsd_sigma", "acceptance constant", "", btsdP->sigma, &btsdP->sigma, 0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_linesearch_btsd_memory_size", "number of historical elements", "", btsdP->memorySize, &btsdP->memorySize, 0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_linesearch_btsd_reference_policy", "policy for updating reference value", "", btsdP->referencePolicy, &btsdP->referencePolicy, 0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_linesearch_btsd_replacement_policy", "policy for updating memory", "", btsdP->replacementPolicy, &btsdP->replacementPolicy, 0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_linesearch_btsd_nondescending","Use nondescending btsd algorithm","",btsdP->nondescending,&btsdP->nondescending, 0);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchView_BTSD"
static PetscErrorCode SNESLineSearchView_BTSD(SNESLineSearch ls, PetscViewer pv)
{
  SNESLineSearch_BTSD *btsdP = (SNESLineSearch_BTSD *)ls->data;
  PetscBool            isascii;
  PetscErrorCode       ierr;


  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    //    ierr = PetscViewerASCIIPrintf(pv,"  maxf=%D, ftol=%g, gtol=%g\n",ls->max_funcs, (double)ls->rtol, (double)ls->ftol);CHKERRQ(ierr);
    ierr=PetscViewerASCIIPrintf(pv,"  BTSD linesearch",btsdP->alpha);CHKERRQ(ierr);
    if (btsdP->nondescending) {
      ierr = PetscViewerASCIIPrintf(pv, " (nondescending)");CHKERRQ(ierr);
    }

    ierr=PetscViewerASCIIPrintf(pv,": alpha=%g beta=%g ",(double)btsdP->alpha,(double)btsdP->beta);CHKERRQ(ierr);
    ierr=PetscViewerASCIIPrintf(pv,"sigma=%g ",(double)btsdP->sigma);CHKERRQ(ierr);
    ierr=PetscViewerASCIIPrintf(pv,"memsize=%D\n",btsdP->memorySize);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetup_BTSD"
PetscErrorCode SNESLineSearchSetup_BTSD (SNESLineSearch linesearch)
{
  /*  PetscErrorCode ierr;
  PetscFunctionBegin;
  // If this snes doesn't have an objective, then use the default 
  PetscErrorCode (*usermerit)(Vec,PetscReal*,
  if (!linesearch->ops->merit) {
    ierr = SNESGetObjective(snes,&usermerit);CHKERRQ(ierr);
    if (usermerit) {
      ierr = SNESLineSearchSetMerit(linesearch,SNESComputeObjective);CHKERRQ(ierr);
    } else {
      ierr = SNESLineSearchSetMerit(linesearch,SNESLineSearchDefaultMerit);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
   */
  return(0);
}
#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchApply_BTSD"
static PetscErrorCode  SNESLineSearchApply_BTSD(SNESLineSearch linesearch)
{
  PetscBool      changed_y, changed_w;
  SNESLineSearch_BTSD *btsdP;
  //Vec            X, F, Y, W;
  Vec            x,work,s,g,origs;
  SNES           snes;
  PetscReal      gnorm, xnorm, ynorm, lambda, minlambda, maxstep,f;
  PetscBool      domainerror;
  PetscReal      fact,ref,gdx,stol;
  PetscViewer    monitor;
  PetscInt       max_it;
  PetscInt       idx,i;
  PetscErrorCode (*merit)(SNES,Vec,PetscReal*);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  btsdP = (SNESLineSearch_BTSD*)linesearch->data;
  ierr = SNESLineSearchGetVecs(linesearch, &x, &g, &origs, &work, NULL);CHKERRQ(ierr);
  ierr = SNESLineSearchGetNorms(linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);
  ierr = SNESLineSearchGetLambda(linesearch, &lambda);CHKERRQ(ierr);
  ierr = SNESLineSearchGetSNES(linesearch, &snes);CHKERRQ(ierr);
  ierr = SNESLineSearchGetMonitor(linesearch, &monitor);CHKERRQ(ierr);
  ierr = SNESLineSearchGetTolerances(linesearch,&minlambda,&maxstep,NULL,NULL,NULL,&max_it);CHKERRQ(ierr);
  ierr = SNESGetTolerances(snes,NULL,NULL,&stol,NULL,NULL);CHKERRQ(ierr);
  if (!linesearch->ops->merit) {
    SNESLineSearchSetMerit(linesearch,SNESLineSearchDefaultMerit);CHKERRQ(ierr);
    f = gnorm;
  } else {
     ierr = SNESLineSearchComputeMerit(linesearch,work,&f);CHKERRQ(ierr);
  }
  merit = linesearch->ops->merit;

  if (!btsdP->work) {
    ierr = VecDuplicate(x,&btsdP->work);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&btsdP->workstep);CHKERRQ(ierr);
    btsdP->x = x;
    ierr = PetscObjectReference((PetscObject)btsdP->x);CHKERRQ(ierr);
  } else if (x != btsdP->x) {
    /* If x has changed, then recreate work */
    ierr = VecDestroy(&btsdP->work);CHKERRQ(ierr);
    ierr = VecDestroy(&btsdP->workstep);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&btsdP->work);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&btsdP->workstep);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)btsdP->x);CHKERRQ(ierr);
    btsdP->x = x;
    ierr = PetscObjectReference((PetscObject)btsdP->x);CHKERRQ(ierr);
  }

  ierr = SNESLineSearchSetSuccess(linesearch, PETSC_TRUE);CHKERRQ(ierr);

  /* precheck */
  ierr = SNESLineSearchPreCheck(linesearch,x,origs,&changed_y);CHKERRQ(ierr);

  /* update */

  /* ABOVE HERE FROM BASIC */
  /* Check to see of the memory has been allocated.  If not, allocate
     the historical array and populate it with the initial function
     values. */
  if (!btsdP->memory) {
    ierr = PetscMalloc1(btsdP->memorySize, &btsdP->memory );CHKERRQ(ierr);
  }

  if (!btsdP->memorySetup) {
    for (i = 0; i < btsdP->memorySize; i++) {
      btsdP->memory[i] = btsdP->alpha*(f);
    }

    btsdP->current = 0;
    btsdP->lastReference = btsdP->memory[0];
    btsdP->memorySetup=PETSC_TRUE;
  }

  /* Calculate reference value (MAX) */
  ref = btsdP->memory[0];
  idx = 0;

  for (i = 1; i < btsdP->memorySize; i++) {
    if (btsdP->memory[i] > ref) {
      ref = btsdP->memory[i];
      idx = i;
    }
  }

  if (btsdP->referencePolicy == REFERENCE_AVE) {
    ref = 0;
    for (i = 0; i < btsdP->memorySize; i++) {
      ref += btsdP->memory[i];
    }
    ref = ref / btsdP->memorySize;
    ref = PetscMax(ref, btsdP->memory[btsdP->current]);
  } else if (btsdP->referencePolicy == REFERENCE_MEAN) {
    ref = PetscMin(ref, 0.5*(btsdP->lastReference + btsdP->memory[btsdP->current]));
  }
  ierr = VecDotRealPart(g,origs,&gdx);CHKERRQ(ierr);

  if (PetscIsInfOrNanReal(gdx)) {
    ierr = PetscInfo1(linesearch,"Initial Line Search step * g is Inf or Nan (%g)\n",(double)gdx);CHKERRQ(ierr);
    ierr = SNESLineSearchSetSuccess(linesearch,PETSC_FALSE);CHKERRQ(ierr);
    snes->reason=SNES_DIVERGED_LINE_SEARCH;
    PetscFunctionReturn(0);
  }
  if (gdx >= 0.0) {
    ierr = PetscInfo1(linesearch,"Initial Line Search step is not descent direction (g's=%g), using -step\n",(double)gdx);CHKERRQ(ierr);
    gdx *= -1;
    ierr = VecCopy(origs,btsdP->workstep);CHKERRQ(ierr);
    ierr = VecScale(btsdP->workstep,-1.0);CHKERRQ(ierr);
    s = btsdP->workstep;
    /*
    ierr = SNESLineSearchSetSuccess(linesearch,PETSC_FALSE);CHKERRQ(ierr);
    snes->reason=SNES_DIVERGED_LINE_SEARCH;
     */
  } else {
    s = origs;
  }

  if (btsdP->nondescending) {
    fact = btsdP->sigma;
  } else {
    fact = btsdP->sigma * gdx;
  }
  while (lambda >= minlambda && (snes->nfuncs < snes->max_funcs)) {
    /* Calculate iterate */
    ierr = VecWAXPY(work,lambda,s,x);CHKERRQ(ierr);

    /* TODO: add projection here */
    
    /* Calculate function at new iterate */
    ierr = (*merit)(snes,work,&f);CHKERRQ(ierr);

    if (monitor) {
      ierr = PetscViewerASCIIAddTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(monitor,"    Line search:  step: %8.6g fmerit %14.12e\n", lambda, (double)f);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(monitor,((PetscObject)linesearch)->tablevel);CHKERRQ(ierr);
    }
    if (PetscIsInfOrNanReal(f)) {
      lambda *= btsdP->beta_inf;
    } else {
      /* Check descent condition */
      if (btsdP->nondescending && f <= ref - lambda*fact*ref)
        break;
      if (!btsdP->nondescending && f <= ref + lambda*fact) {
        break;
      }

      lambda *= btsdP->beta;
    }
  }

  /* Check termination */
  if (PetscIsInfOrNanReal(f)) {
    ierr = PetscInfo(linesearch, "Function is inf or nan.\n");CHKERRQ(ierr);
    ierr = SNESLineSearchSetSuccess(linesearch,PETSC_FALSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else if (lambda < minlambda) {
    ierr = PetscInfo(linesearch, "Step length is below tolerance.\n");CHKERRQ(ierr);
    ierr = SNESLineSearchSetSuccess(linesearch,PETSC_FALSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  } else if (snes->nfuncs >= snes->max_funcs) {
    ierr = PetscInfo2(linesearch, "Number of line search function evals (%D) > maximum allowed (%D)\n",snes->nfuncs, snes->max_funcs);CHKERRQ(ierr);
    ierr = SNESLineSearchSetSuccess(linesearch,PETSC_FALSE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* Successful termination, update memory */
  btsdP->lastReference = ref;
  if (btsdP->replacementPolicy == REPLACE_FIFO) {
    btsdP->memory[btsdP->current++] = f;
    if (btsdP->current >= btsdP->memorySize) {
      btsdP->current = 0;
    }
  } else {
    btsdP->current = idx;
    btsdP->memory[idx] = f;
  }






  /* BELOW HERE FROM BASIC */


  if (linesearch->ops->viproject) {
    ierr = (*linesearch->ops->viproject)(snes, work);CHKERRQ(ierr);
  }

  /* postcheck */
  ierr = SNESLineSearchPostCheck(linesearch,x,s,work,&changed_y,&changed_w);CHKERRQ(ierr);
  if (changed_y) {
    ierr = VecWAXPY(work,-lambda,s,x);CHKERRQ(ierr);
    if (linesearch->ops->viproject) {
      ierr = (*linesearch->ops->viproject)(snes, work);CHKERRQ(ierr);
    }
  }
  if (linesearch->norms || snes->iter < max_it-1) {
    ierr = (*linesearch->ops->snesfunc)(snes,work,g);CHKERRQ(ierr);
    ierr = SNESGetFunctionDomainError(snes, &domainerror);CHKERRQ(ierr);
    if (domainerror) {
      ierr = SNESLineSearchSetSuccess(linesearch, PETSC_FALSE);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }

  if (linesearch->norms) {
    if (!linesearch->ops->vinorm) VecNormBegin(g, NORM_2, &linesearch->fnorm);
    ierr = VecNormBegin(s, NORM_2, &linesearch->ynorm);CHKERRQ(ierr);
    ierr = VecNormBegin(work, NORM_2, &linesearch->xnorm);CHKERRQ(ierr);
    if (!linesearch->ops->vinorm) VecNormEnd(g, NORM_2, &linesearch->fnorm);
    ierr = VecNormEnd(s, NORM_2, &linesearch->ynorm);CHKERRQ(ierr);
    ierr = VecNormEnd(work, NORM_2, &linesearch->xnorm);CHKERRQ(ierr);

    if (linesearch->ops->vinorm) {
      linesearch->fnorm = gnorm;

      ierr = (*linesearch->ops->vinorm)(snes, g, work, &linesearch->fnorm);CHKERRQ(ierr);
    } else {
      ierr = VecNorm(g,NORM_2,&linesearch->fnorm);CHKERRQ(ierr);
    }
  }

  /* copy the solution over */
  ierr = VecCopy(work, x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchCreate_BTSD"
/*MC
   SNESLINESEARCHBTSD - The BackTracking using Sufficient Decrease line

   Options Database Keys:
+   -snes_linesearch_btsd_alpha<1.0> -  initial reference constant
.   -snes_linesearch_btsd_beta_inf<0.5> -  decrease constant one
.   -snes_linesearch_btsd_beta<0.5> - decrease constant
.   -snes_linesearch_btsd_sigma<1e-4> - acceptance constant
.   -snes_linesearch_btsd_memory_size<1> - number of historical elements
.   -snes_linesearch_btsd_reference_policy<1> - policy for updating reference value
.   -snes_linesearch_btsd_replacement_policy<2> - policy for updating memory
-   -snes_linesearch_btsd_nondescending<false> - Force nondescending

   Level: advanced

.keywords: SNES, SNESLineSearch, damping

.seealso: SNESLineSearchCreate(), SNESLineSearchSetType()
M*/
PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_BTSD(SNESLineSearch linesearch)
{
  SNESLineSearch_BTSD *btsdP;
  PetscErrorCode        ierr;
  PetscFunctionBegin;


  linesearch->ops->apply          = SNESLineSearchApply_BTSD;
  linesearch->ops->destroy        = SNESLineSearchDestroy_BTSD;
  linesearch->ops->setfromoptions = SNESLineSearchSetFromOptions_BTSD;
  linesearch->ops->reset          = SNESLineSearchReset_BTSD;
  linesearch->ops->view           = SNESLineSearchView_BTSD;
  linesearch->ops->setup          = SNESLineSearchSetup_BTSD;

  ierr = PetscNewLog(linesearch,&btsdP);CHKERRQ(ierr);
  linesearch->data = (void*)btsdP;
  btsdP->memory = NULL;
  btsdP->alpha = 1.0;
  btsdP->beta = 0.5;
  btsdP->beta_inf = 0.5;
  btsdP->sigma = 1e-4;
  btsdP->memorySize = 1;
  btsdP->referencePolicy = REFERENCE_MAX;
  btsdP->replacementPolicy = REPLACE_MRU;
  btsdP->nondescending=PETSC_FALSE;

  PetscFunctionReturn(0);
}

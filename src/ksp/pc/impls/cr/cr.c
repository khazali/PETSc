#include <petsc-private/pcimpl.h>     /*I "petscpc.h" I*/
#include <petscdm.h>

const char *const PCCRTrialTypes[] = {"CONSTANT","RANDOM","NULLSPACE","PCCRTrialType","PC_CR_TRIAL_",0};

typedef struct {
  Mat           inj;              /* injection matrix */
  Vec           injscale;         /* injection scaling for the Kaczmarz iteration */
  PC            fpc;              /* habituated PC */
  PetscInt      candidate_sweeps;
  PetscInt      candidate_trials;
  PetscBool     candidate_view;
  PetscBool     trydm;            /* try to coarsen the DM and create the injection if no injection is set */
  PetscReal     stol;             /* stagnation tolerance for convergence estimates */
  PCCRTrialType trial_type;
} PC_CR;

#undef __FUNCT__
#define __FUNCT__ "PCView_CR"
static PetscErrorCode PCView_CR(PC pc,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii,isdraw;
  PC_CR             *cr = (PC_CR*)pc->data;
  Vec               s;
  PetscInt          m,n;
  PetscReal         convfact;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) {
    if (cr->inj) {
      ierr = MatGetSize(cr->inj,&m,&n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"CR: %d fine and %d coarse variables\n",n,m);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"CR: no injection set\n");CHKERRQ(ierr);
    }
    if (cr->fpc) {
      ierr = PetscViewerASCIIPrintf(viewer,"Habituated preconditioner:");CHKERRQ(ierr);
      ierr = PCView(cr->fpc,viewer);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"No inner preconditioner setup");CHKERRQ(ierr);
    }
  }
  if (cr->candidate_view) {
    ierr = MatGetVecs(pc->pmat,&s,NULL);CHKERRQ(ierr);
    ierr = PCCRGetCandidateEstimates(pc,s,&convfact);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Candidate Fine-point convergence factor: %f\n",convfact);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Candidate Coarse-point Measures:\n");CHKERRQ(ierr);
    ierr = VecView(s,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecDestroy(&s);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCCRGetCandidateEstimates_CR"
PetscErrorCode  PCCRGetCandidateEstimates_CR(PC pc,Vec s,PetscReal *convest)
{
  PetscErrorCode ierr;
  Vec            x,e,eold,cvec;
  PetscRandom    rand;
  PetscBool      has_const = PETSC_FALSE;
  PetscInt       j,i,candidate_trials=0,nns;
  PetscReal      enorm,eoldnorm,trialconvest;
  PC_CR          *cr = (PC_CR*)pc->data;
  MatNullSpace   nullspace;
  const Vec      *nsvecs;

  PetscFunctionBegin;
  switch (cr->trial_type) {
  case (PC_CR_TRIAL_CONSTANT):
    candidate_trials=1;
    break;
  case (PC_CR_TRIAL_RANDOM):
    candidate_trials=cr->candidate_trials;
    break;
  case (PC_CR_TRIAL_NULLSPACE):
    ierr = MatGetNullSpace(pc->pmat,&nullspace);CHKERRQ(ierr);
    if (!nullspace) {
      ierr = MatGetNearNullSpace(pc->pmat,&nullspace);CHKERRQ(ierr);
    }
    if (!nullspace) {
      SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Matrix must have a set nullspace or near-nullspace to use PC_CR_TRIAL_NULLSPACE");
    }
    ierr = MatNullSpaceGetVecs(nullspace,&has_const,&nns,&nsvecs);CHKERRQ(ierr);
    candidate_trials=nns+(has_const ? 1 : 0);
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Unsupported trial type.");
    break;
  }
  ierr = VecDuplicate(s,&e);CHKERRQ(ierr);
  ierr = VecDuplicate(s,&eold);CHKERRQ(ierr);
  ierr = VecDuplicate(s,&x);CHKERRQ(ierr);
  if (cr->inj) {ierr = MatGetVecs(cr->inj,NULL,&cvec);CHKERRQ(ierr);}
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)pc),&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = VecSet(s,0.);CHKERRQ(ierr);

  for (i=0;i<candidate_trials;i++) {
    switch (cr->trial_type) {
    case (PC_CR_TRIAL_CONSTANT):
      ierr = VecSet(e,1.0);CHKERRQ(ierr);
      break;
    case (PC_CR_TRIAL_RANDOM):
      /* set the vector to be randomly between 1/2 and 1 */
      ierr = VecSetRandom(e,rand);CHKERRQ(ierr);
      ierr = VecScale(e,0.25);CHKERRQ(ierr);
      ierr = VecShift(e,0.75);CHKERRQ(ierr);
      break;
    case (PC_CR_TRIAL_NULLSPACE):
      if (i==nns && has_const) {
        ierr = VecSet(e,1.0);CHKERRQ(ierr);
      } else {
        ierr = VecCopy(nsvecs[i],e);CHKERRQ(ierr);
      }
      break;
    }
    if (cr->inj) {
      ierr = MatMult(cr->inj,e,cvec);CHKERRQ(ierr);
      ierr = MatMultTranspose(cr->inj,cvec,eold);CHKERRQ(ierr);
      ierr = VecPointwiseMult(eold,cr->injscale,eold);CHKERRQ(ierr);
      ierr = VecAXPY(e,-1.0,eold);CHKERRQ(ierr);
    }
    /* error propagation: e_{i+1} = (I - MA)e_i where M smooths only the fine subspace and e_0 has zero error on coarse and boundary nodes */
    for (j=0;j<cr->candidate_sweeps;j++) {
      ierr = MatMult(pc->pmat,e,x);CHKERRQ(ierr);
      ierr = VecCopy(e,eold);CHKERRQ(ierr);
      ierr = PCApply(pc,x,e);CHKERRQ(ierr);
      ierr = VecAYPX(e,-1.0,eold);CHKERRQ(ierr);
      if (convest && j == cr->candidate_sweeps-1) {
        ierr = VecNorm(eold,NORM_2,&eoldnorm);CHKERRQ(ierr);
        ierr = VecNorm(e,NORM_2,&enorm);CHKERRQ(ierr);
        if (eoldnorm > 0.) {
          trialconvest = enorm/eoldnorm;
        } else {
          trialconvest = 0.;
        }
        if (trialconvest > *convest) *convest = trialconvest;
      }
    }
    ierr = VecPointwiseMax(s,e,s);CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  if (cr->inj) {ierr = VecDestroy(&cvec);CHKERRQ(ierr);}
  ierr = VecDestroy(&e);CHKERRQ(ierr);
  ierr = VecDestroy(&eold);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCCRGetCandidateEstimates"
PetscErrorCode PCCRGetCandidateEstimates(PC pc,Vec s,PetscReal *convtest)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(s,VEC_CLASSID,2);
  ierr = VecSet(s,0.);CHKERRQ(ierr);
  if (convtest) *convtest = 0.;
  ierr = PetscTryMethod(pc,"PCCRGetCandidateEstimates_C",(PC,Vec,PetscReal*),(pc,s,convtest));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_CR"
static PetscErrorCode PCApply_CR(PC pc,Vec x,Vec y)
{
  PC_CR             *cr = (PC_CR*)pc->data;
  PetscErrorCode    ierr;
  Vec               cvec,w;

  PetscFunctionBegin;
  /* habituated CR */
  ierr = PCApply(cr->fpc,x,y);CHKERRQ(ierr);
  if (cr->inj) {
    ierr = MatGetVecs(cr->inj,&w,&cvec);CHKERRQ(ierr);
    ierr = MatMult(cr->inj,y,cvec);CHKERRQ(ierr);
    ierr = MatMultTranspose(cr->inj,cvec,w);CHKERRQ(ierr);
    ierr = VecPointwiseMult(w,cr->injscale,w);CHKERRQ(ierr);
    ierr = VecAXPY(y,-1.0,w);CHKERRQ(ierr);
    ierr = VecDestroy(&w);CHKERRQ(ierr);
    ierr = VecDestroy(&cvec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApplyTranspose_CR"
static PetscErrorCode PCApplyTranspose_CR(PC pc,Vec x,Vec y)
{
  PC_CR             *cr = (PC_CR*)pc->data;
  PetscErrorCode    ierr;
  Vec               cvec,w;

  PetscFunctionBegin;
   /* habituated CR */
  ierr = PCApplyTranspose(cr->fpc,x,y);CHKERRQ(ierr);
  if (cr->inj) {
    ierr = MatGetVecs(cr->inj,&w,&cvec);CHKERRQ(ierr);
    ierr = MatMult(cr->inj,y,cvec);CHKERRQ(ierr);
    ierr = MatMultTranspose(cr->inj,cvec,w);CHKERRQ(ierr);
    ierr = VecPointwiseMult(w,cr->injscale,w);CHKERRQ(ierr);
    ierr = VecAXPY(y,-1.0,w);CHKERRQ(ierr);
    ierr = VecDestroy(&w);CHKERRQ(ierr);
    ierr = VecDestroy(&cvec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCReset_CR"
static PetscErrorCode PCReset_CR(PC pc)
{
  PC_CR             *cr = (PC_CR*)pc->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&cr->inj);CHKERRQ(ierr);
  ierr = PCDestroy(&cr->fpc);CHKERRQ(ierr);
  ierr = VecDestroy(&cr->injscale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

 #undef __FUNCT__
#define __FUNCT__ "PCDestroy_CR"
static PetscErrorCode PCDestroy_CR(PC pc)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCRSetInjection_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCRGetInjection_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCRGetCandidateEstimates_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCRSetTrialType_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_CR"
static PetscErrorCode PCSetFromOptions_CR(PC pc)
{
  PetscErrorCode  ierr;
  PC_CR           *cr = (PC_CR*)pc->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Compatible Relaxation options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_cr_candidate_sweeps","Number of smoother sweeps in candidate measure computation","",cr->candidate_sweeps,&cr->candidate_sweeps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-pc_cr_trial_type","Type of trial vectors to use","PCCRTrialType",PCCRTrialTypes,(PetscEnum)cr->trial_type,(PetscEnum*)&cr->trial_type,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_cr_candidate_trials","Number of trial vectors in candidate measure computation","",cr->candidate_trials,&cr->candidate_trials,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_cr_candidate_view","Show candidate measures during PCView()","",cr->candidate_view,&cr->candidate_view,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_cr_use_dm","Show candidate measures during PCView()","",cr->trydm,&cr->trydm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_CR(PC pc);

#undef __FUNCT__
#define __FUNCT__ "PCCRSetInjection_CR"
static PetscErrorCode  PCCRSetInjection_CR(PC pc,Mat inj)
{
  PC_CR             *cr = (PC_CR*)pc->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (cr->inj) {
    ierr = MatDestroy(&cr->inj);CHKERRQ(ierr);
    ierr = VecDestroy(&cr->injscale);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject)inj);CHKERRQ(ierr);
  cr->inj = inj;
  ierr = PCSetUp_CR(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCCRGetInjection_CR"
PetscErrorCode PCCRGetInjection_CR(PC pc,Mat *inj)
{
  PC_CR             *cr = (PC_CR*)pc->data;

  PetscFunctionBegin;
  *inj = cr->inj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCCRSetTrialType_CR"
PetscErrorCode PCCRSetTrialType_CR(PC pc,PCCRTrialType tt)
{
  PC_CR          *cr = (PC_CR*)pc->data;

  PetscFunctionBegin;
  cr->trial_type = tt;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCCRSetInjection"
PetscErrorCode  PCCRSetInjection(PC pc,Mat inj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(inj,MAT_CLASSID,2);
  ierr = PetscTryMethod(pc,"PCCRSetInjection_C",(PC,Mat),(pc,inj));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCCRGetInjection"
PetscErrorCode  PCCRGetInjection(PC pc,Mat *inj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCCRGetInjection_C",(PC,Mat*),(pc,inj));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCCRSetTrialType"
PetscErrorCode  PCCRSetTrialType(PC pc,PCCRTrialType tt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCCRSetTrialType_C",(PC,PCCRTrialType),(pc,tt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_CR"
static PetscErrorCode PCSetUp_CR(PC pc)
{
  PetscErrorCode ierr;
  PC_CR          *cr = (PC_CR*)pc->data;
  DM             dm,dmc;
  Vec            fvec,cvec;
  VecScatter     inj;
  Mat            A,P;
  MatStructure   flg;
  const char     *prefix;
  PetscInt       fs,fe,fn,i;
  PetscScalar    *fvecarray;

  PetscFunctionBegin;
  if (!cr->inj) {
    ierr = PCGetDM(pc,&dm);CHKERRQ(ierr);
    if (dm && cr->trydm) {
      ierr = DMCoarsen(dm,PetscObjectComm((PetscObject)pc),&dmc);CHKERRQ(ierr);
      if (dmc) {
        ierr = DMCreateInjection(dmc,dm,&inj);CHKERRQ(ierr);
        ierr = MatCreateScatter(PetscObjectComm((PetscObject)pc),inj,&cr->inj);CHKERRQ(ierr);
        ierr = VecScatterDestroy(&inj);CHKERRQ(ierr);
      }
      ierr = DMDestroy(&dmc);CHKERRQ(ierr);
    }
  }
  /* form the scaling -- if it's not one-to-one one has to scale the transpose application going from coarse to fine */
  if (cr->inj && !cr->injscale) {
    ierr = MatGetVecs(cr->inj,&fvec,&cvec);CHKERRQ(ierr);
    ierr = VecSet(cvec,1.);CHKERRQ(ierr);
    ierr = MatMultTranspose(cr->inj,cvec,fvec);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(fvec,&fs,&fe);CHKERRQ(ierr);
    fn=fe-fs;
    ierr = VecGetArray(fvec,&fvecarray);CHKERRQ(ierr);
    for (i=0;i<fn;i++) {
      if (fvecarray[i] != 0.) {
        fvecarray[i] = 1./fvecarray[i];
      }
    }
    ierr = VecRestoreArray(fvec,&fvecarray);CHKERRQ(ierr);
    cr->injscale = fvec;
    ierr = VecDestroy(&cvec);CHKERRQ(ierr);
  }

  if (!cr->fpc) {
    ierr = PCCreate(PetscObjectComm((PetscObject)pc),&cr->fpc);CHKERRQ(ierr);
    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
    ierr = PCSetOptionsPrefix(cr->fpc,prefix);CHKERRQ(ierr);
    ierr = PCAppendOptionsPrefix(cr->fpc,"habituated_");CHKERRQ(ierr);
    ierr = PCGetOperators(pc,&A,&P,&flg);CHKERRQ(ierr);
    ierr = PCSetOperators(cr->fpc,A,P,flg);CHKERRQ(ierr);
    ierr = PCSetType(cr->fpc,PCSOR);CHKERRQ(ierr);
    ierr = PCSetFromOptions(cr->fpc);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------------------*/
/*MC

 PCCR - Compatible relaxation preconditioner created by using some smoothing method on a subset (the "fine" nodes) of
        the problem rather than the whole.  Uses the action of an inner preconditioner to smooth the nodes outside of the
        coarse set provided by the user.  The coarse set is held invariant by use of projection.

   References:

   Brandt, Achi. "General highly accurate algebraic coarsening."
   Electronic Transactions on Numerical Analysis 10.1 (2000): 1-20.

   Brannick, James J., and Robert D. Falgout. "Compatible relaxation and coarsening in algebraic multigrid."
   SIAM Journal on Scientific Computing 32.3 (2010): 1393-1416.

   Livne, O. E. "Coarsening by compatible relaxation."
   Numerical linear algebra with applications 11.2-3 (2004): 205-227.

   Level: intermediate

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCCRSetFineIS()
M*/

#undef __FUNCT__
#define __FUNCT__ "PCCreate_CR"
PETSC_EXTERN PetscErrorCode PCCreate_CR(PC pc)
{
  PetscErrorCode ierr;
  PC_CR          *cr;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,&cr);CHKERRQ(ierr);

  pc->data = (void*)cr;

  pc->ops->apply           = PCApply_CR;
  pc->ops->applytranspose  = PCApplyTranspose_CR;
  pc->ops->setup           = PCSetUp_CR;
  pc->ops->reset           = PCReset_CR;
  pc->ops->destroy         = PCDestroy_CR;
  pc->ops->setfromoptions  = PCSetFromOptions_CR;
  pc->ops->view            = PCView_CR;
  pc->ops->applyrichardson = 0;


  cr->inj                  = NULL;
  cr->fpc                  = NULL;

  cr->candidate_sweeps     = 1;
  cr->candidate_trials     = 5;
  cr->candidate_view       = PETSC_FALSE;

  cr->trydm                = PETSC_TRUE;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCRSetInjection_C",PCCRSetInjection_CR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCRGetInjection_C",PCCRGetInjection_CR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCRGetCandidateEstimates_C",PCCRGetCandidateEstimates_CR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCRSetTrialType_C",PCCRSetTrialType_CR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petsc-private/pcimpl.h>     /*I "petscpc.h" I*/
#include <petscdm.h>

typedef struct {
  IS       cis;   /* IS of fine nodes */
  PC       fpc;
} PC_CR;

#undef __FUNCT__
#define __FUNCT__ "PCView_CR"
static PetscErrorCode PCView_CR(PC pc,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii,isdraw;
  PetscInt          ncoarse;
  PC_CR             *cr = (PC_CR*)pc->data;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) {
    ncoarse = 0;
    if (cr->cis) {
      ierr = ISGetSize(cr->cis,&ncoarse);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"CR: number of coarse variables=%d\n",ncoarse);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"CR: no coarse unknowns set\n");CHKERRQ(ierr);
    }
    if (cr->fpc) {
      ierr = PetscViewerASCIIPrintf(viewer,"Habituated preconditioner:");CHKERRQ(ierr);
      ierr = PCView(cr->fpc,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_CR"
static PetscErrorCode PCApply_CR(PC pc,Vec x,Vec y)
{
  PC_CR             *cr = (PC_CR*)pc->data;
  PetscErrorCode    ierr;
  PetscInt          i;
  const PetscInt    *cisarray;
  PetscInt          ncoarse;

  PetscFunctionBegin;
  /* habituated CR */
  ierr = PCApply(cr->fpc,x,y);CHKERRQ(ierr);
  ierr = ISGetIndices(cr->cis,&cisarray);CHKERRQ(ierr);
  ierr = ISGetLocalSize(cr->cis,&ncoarse);CHKERRQ(ierr);
  for (i=0;i<ncoarse;i++) {
    ierr = VecSetValue(y,cisarray[i],0.,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(cr->cis,&cisarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);

}

#undef __FUNCT__
#define __FUNCT__ "PCApplyTranspose_CR"
static PetscErrorCode PCApplyTranspose_CR(PC pc,Vec x,Vec y)
{
  PC_CR             *cr = (PC_CR*)pc->data;
  PetscErrorCode    ierr;
  PetscInt          i;
  const PetscInt    *cisarray;
  PetscInt          ncoarse;

  PetscFunctionBegin;
    /* habituated CR */
  ierr = PCApplyTranspose(cr->fpc,x,y);CHKERRQ(ierr);
  ierr = ISGetIndices(cr->cis,&cisarray);CHKERRQ(ierr);
  ierr = ISGetLocalSize(cr->cis,&ncoarse);CHKERRQ(ierr);
  for (i=0;i<ncoarse;i++) {
    ierr = VecSetValue(y,cisarray[i],0.,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(cr->cis,&cisarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCReset_CR"
static PetscErrorCode PCReset_CR(PC pc)
{
  PC_CR             *cr = (PC_CR*)pc->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = ISDestroy(&cr->cis);CHKERRQ(ierr);
  ierr = PCDestroy(&cr->fpc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_CR"
static PetscErrorCode PCDestroy_CR(PC pc)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCRSetCoarseIS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCRGetCoarseIS_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_CR"
static PetscErrorCode PCSetFromOptions_CR(PC pc)
{
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsHead("Compatible Relaxation options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCCRSetCoarseIS_CR"
static PetscErrorCode  PCCRSetCoarseIS_CR(PC pc,IS is)
{
  PC_CR             *cr = (PC_CR*)pc->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)is);CHKERRQ(ierr);
  cr->cis = is;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCCRGetCoarseIS_CR"
PetscErrorCode PCCRGetCoarseIS_CR(PC pc,IS *is)
{
  PC_CR             *cr = (PC_CR*)pc->data;

  PetscFunctionBegin;
  *is = cr->cis;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCCRSetCoarseIS"
PetscErrorCode  PCCRSetCoarseIS(PC pc,IS is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  ierr = PetscTryMethod(pc,"PCCRSetFineIS_C",(PC,IS),(pc,is));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCCRGetCoarseIS"
PetscErrorCode  PCCRGetCoarseIS(PC pc,IS is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  ierr = PetscTryMethod(pc,"PCCRGetCoarseIS_C",(PC,IS),(pc,is));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_CR"
static PetscErrorCode PCSetUp_CR(PC pc)
{
  PetscErrorCode ierr;
  PC_CR          *cr = (PC_CR*)pc->data;
  DM             dm,dmc;
  VecScatter     inj;
  Vec            fv,cv;
  PetscInt       fs,fe;
  PetscInt       ncoarse,idx,i;
  PetscScalar    *fvarray;
  PetscInt       *carray;
  Mat            A,P;
  MatStructure   flg;
  const char     *prefix;

  PetscFunctionBegin;
  if (!cr->cis) {
    ierr = PCGetDM(pc,&dm);CHKERRQ(ierr);
    ierr = DMCoarsen(dm,PetscObjectComm((PetscObject)pc),&dmc);CHKERRQ(ierr);
    if (dmc) {
      ierr = DMCreateInjection(dmc,dm,&inj);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dm,&fv);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dmc,&cv);CHKERRQ(ierr);
      ierr = VecZeroEntries(fv);CHKERRQ(ierr);
      ierr = VecSet(cv,1.);CHKERRQ(ierr);
      ierr = VecScatterBegin(inj,cv,fv,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(inj,cv,fv,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(fv,&fs,&fe);CHKERRQ(ierr);
      ierr = VecGetArray(fv,&fvarray);CHKERRQ(ierr);
      ncoarse=0;
      for (i=0;i<fe-fs;i++) {
        if (fvarray[i] != 0.) {
          ncoarse++;
        }
      }
      ierr = PetscMalloc(ncoarse*sizeof(PetscInt),&carray);CHKERRQ(ierr);
      idx=0;
      for (i=0;i<fe-fs;i++) {
        if (fvarray[i] != 0.) {
          carray[idx] = i+fs;
          idx++;
        }
      }
      ierr = VecRestoreArray(fv,&fvarray);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)pc),ncoarse,carray,PETSC_OWN_POINTER,&cr->cis);CHKERRQ(ierr);
      ierr = VecScatterDestroy(&inj);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dm,&fv);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dmc,&cv);CHKERRQ(ierr);
      ierr = DMDestroy(&dmc);CHKERRQ(ierr);
    }
  }
  if (!cr->fpc) {
    ierr = PCCreate(PetscObjectComm((PetscObject)pc),&cr->fpc);CHKERRQ(ierr);
    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
    ierr = PCSetOptionsPrefix(cr->fpc,prefix);CHKERRQ(ierr);
    ierr = PCAppendOptionsPrefix(cr->fpc,"cr_");CHKERRQ(ierr);
    ierr = PCGetOperators(pc,&A,&P,&flg);CHKERRQ(ierr);
    ierr = PCSetOperators(cr->fpc,A,P,flg);CHKERRQ(ierr);
    ierr = PCSetType(cr->fpc,PCSOR);CHKERRQ(ierr);
    ierr = PCSetFromOptions(cr->fpc);CHKERRQ(ierr);
  }
  if (!cr->cis) {
    SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set the coarse-unknown IS in order to use PCCR");
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
/*MC

 PCCR - Compatible relaxation preconditioner created by using some smoothing method on a subset (the "fine" nodes) of
        the problem rather than the whole.  Uses the action of an inner preconditioner to smooth the nodes outside of the
        coarse set provided by the user.  The coarse set is held invariant.  This could be more general for non coarse-fine
        splittings, but for now it's very "classical".

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
  ierr = PetscNewLog(pc,PC_CR,&cr);CHKERRQ(ierr);

  pc->data = (void*)cr;

  pc->ops->apply           = PCApply_CR;
  pc->ops->applytranspose  = PCApplyTranspose_CR;
  pc->ops->setup           = PCSetUp_CR;
  pc->ops->reset           = PCReset_CR;
  pc->ops->destroy         = PCDestroy_CR;
  pc->ops->setfromoptions  = PCSetFromOptions_CR;
  pc->ops->view            = PCView_CR;
  pc->ops->applyrichardson = 0;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCRSetCoarseIS_C",PCCRSetCoarseIS_CR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCRGetCoarseIS_C",PCCRGetCoarseIS_CR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

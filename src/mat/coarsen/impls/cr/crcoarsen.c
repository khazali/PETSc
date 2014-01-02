#include <petsc-private/matimpl.h>    /*I "petscmat.h" I*/
#include <petscpc.h>

typedef struct {
  PetscReal entrythreshold;
  PetscReal convgoal;
  PetscReal abstol;
  PetscReal maxratio;
  PetscInt  maxsweeps;
  PetscBool verbose;
} MatCoarsen_CR;

#undef __FUNCT__
#define __FUNCT__ "MatCoarsenApply_CR"
static PetscErrorCode MatCoarsenApply_CR(MatCoarsen coarse)
{
  PetscErrorCode   ierr;
  MatCoarsen_CR    *cr = (MatCoarsen_CR*)coarse->subctx;
  Mat              mat = coarse->graph;
  PC               pccr;
  PetscInt         k,j,i,idx,maxsweeps=cr->maxsweeps,ms,me;
  Vec              s;
  PetscInt         *sperm,ncoarse;
  PetscScalar      *sarray;
  PetscReal        *sarray_real;
  PetscCoarsenData *agg_lists;
  PetscInt         *state;
  const PetscInt   *mcol;
  PetscInt         mncol;
  PetscInt         *cidx;
  IS               cis;
  VecScatter       cscatter;
  Vec              cvec;
  Mat              inj;
  PetscReal        smax;
  PetscReal        convfact,convgoal=cr->convgoal,abstol=cr->abstol,entrythreshold=cr->entrythreshold,maxratio=cr->maxratio;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse,MAT_COARSEN_CLASSID,1);

  /* create the smoother */
  ierr = PCCreate(PetscObjectComm((PetscObject)coarse),&pccr);CHKERRQ(ierr);
  ierr = PCSetType(pccr,PCCR);CHKERRQ(ierr);
  ierr = PCAppendOptionsPrefix(pccr,"coarsen_");CHKERRQ(ierr);
  ierr = PCSetOperators(pccr,mat,mat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = PCSetFromOptions(pccr);CHKERRQ(ierr);
  ierr = MatGetVecs(mat,&s,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(mat,&ms,&me);CHKERRQ(ierr);

  /* setup the space for the aggregation */
  ierr = PetscCDCreate(me-ms,&agg_lists);CHKERRQ(ierr);
  ierr = PetscMalloc1(me-ms,&sarray_real);CHKERRQ(ierr);
  ierr = PetscMalloc1(me-ms,&sperm);CHKERRQ(ierr);
  ierr = PetscMalloc1(me-ms,&state);CHKERRQ(ierr);
  ierr = PetscMalloc1(me-ms,&cidx);CHKERRQ(ierr);
  ncoarse = 0;

  for (i=0;i<me-ms;i++) {
    state[i] = -1;
  }
  idx=0;
  for (k=0;k<maxsweeps;k++) {
    /* run the smoother to get error propagation */
    ierr = PCCRGetCandidateEstimates(pccr,s,&convfact);CHKERRQ(ierr);
    ierr = VecMax(s,NULL,&smax);CHKERRQ(ierr);
    if (cr->verbose) {
      ierr = PetscPrintf(PetscObjectComm((PetscObject)coarse),"%d out of %d: convergence estimate: %f\n",idx,me-ms,convfact);CHKERRQ(ierr);
    }
    if (convfact < convgoal) break;
    if (ncoarse >= maxratio*(me-ms)) break;
    /* sort the indices by badness */
    ierr = VecGetArray(s,&sarray);CHKERRQ(ierr);
    for (i=0;i<me-ms;i++) {
      sarray_real[i] = PetscRealPart(sarray[i]);
      sperm[i] = i;
    }
    ierr = VecRestoreArray(s,&sarray);CHKERRQ(ierr);
    ierr = PetscSortRealWithPermutation(me-ms,sarray_real,sperm);CHKERRQ(ierr);

    /* add a local MIS worth of them to the mix */
    for (i=me-ms-1;i >= 0;i--) {
      idx=sperm[i];
      if (state[idx] == -1 && (PetscAbsReal(sarray_real[idx]) > entrythreshold*smax && (smax > abstol))) {
        state[idx] = -2; /* set it to be accepted forever and ever */
        ierr = MatGetRow(mat,idx+ms,&mncol,&mcol,NULL);CHKERRQ(ierr);
        for (j=0;j<mncol;j++) {
          if (mcol[j] >= ms && mcol[j] < me) {
            /* set the aggregation */
            if (state[mcol[j]-ms] == -1) {
              state[mcol[j]-ms] = idx;
            }
          }
        }
        ierr = MatRestoreRow(mat,idx+ms,&mncol,&mcol,NULL);CHKERRQ(ierr);
        ncoarse++;
        if (ncoarse >= maxratio*(me-ms)) {
          break;
        }
      }
    }
    idx=0;
    for (i=0;i<me-ms;i++) {
      if (state[i] == -2) {
        cidx[idx]=i+ms;
        idx++;
      }
    }
    /* create the scatter for this new set */
    ierr = VecCreate(PetscObjectComm((PetscObject)coarse),&cvec);CHKERRQ(ierr);
    ierr = VecSetSizes(cvec,idx,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(cvec);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)coarse),idx,cidx,PETSC_COPY_VALUES,&cis);CHKERRQ(ierr);
    ierr = VecScatterCreate(s,cis,cvec,NULL,&cscatter);CHKERRQ(ierr);
    ierr = MatCreateScatter(PetscObjectComm((PetscObject)coarse),cscatter,&inj);CHKERRQ(ierr);
    ierr = PCDestroy(&pccr);CHKERRQ(ierr);

    ierr = PCCreate(PetscObjectComm((PetscObject)coarse),&pccr);CHKERRQ(ierr);
    ierr = PCAppendOptionsPrefix(pccr,"mat_coarsen_cr_");CHKERRQ(ierr);
    ierr = PCSetType(pccr,PCCR);CHKERRQ(ierr);
    ierr = PCSetOperators(pccr,mat,mat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = PCCRSetInjection(pccr,inj);CHKERRQ(ierr);
    ierr = PCSetFromOptions(pccr);CHKERRQ(ierr);

    ierr = MatDestroy(&inj);CHKERRQ(ierr);
    ierr = VecDestroy(&cvec);CHKERRQ(ierr);
    ierr = ISDestroy(&cis);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&cscatter);CHKERRQ(ierr);

    /* reset for another round */
    if (k != maxsweeps-1) {
      for (i=0;i<me-ms;i++) {
        if (state[i] >= 0) {
          state[i] = -1;
        }
      }
    }
  }
  ierr = PCCRGetCandidateEstimates(pccr,s,&convfact);CHKERRQ(ierr);
  ierr = VecDestroy(&s);CHKERRQ(ierr);
  ierr = PCDestroy(&pccr);CHKERRQ(ierr);
  for (i=0;i<me-ms;i++) {
    if (state[i] == -2) {
      ierr = PetscCDAppendID(agg_lists,i,i);CHKERRQ(ierr);
    }
  }
  coarse->agg_lists = agg_lists;
  ierr = PetscFree(sarray_real);CHKERRQ(ierr);
  ierr = PetscFree(sperm);CHKERRQ(ierr);
  ierr = PetscFree(state);CHKERRQ(ierr);
  ierr = PetscFree(cidx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCoarsenView_CR"
PetscErrorCode MatCoarsenView_CR(MatCoarsen coarse,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse,MAT_COARSEN_CLASSID,1);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)coarse),&rank);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] CR aggregator\n",rank);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCoarsenDestroy_CR"
PetscErrorCode MatCoarsenDestroy_CR(MatCoarsen coarse)
{
  MatCoarsen_CR *cr = (MatCoarsen_CR*)coarse->subctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse,MAT_COARSEN_CLASSID,1);
  ierr = PetscFree(cr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCoarsenSetFromOptions_CR"
PetscErrorCode MatCoarsenSetFromOptions_CR(MatCoarsen coarse)
{
  MatCoarsen_CR *cr = (MatCoarsen_CR*)coarse->subctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("MatCoarsen CR options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_coarsen_cr_convgoal","Smoother convergence goal","MatCoarsen",cr->convgoal,&cr->convgoal,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_coarsen_cr_threshold","Relative threshold for adding an unknown to the coarse set","MatCoarsen",cr->entrythreshold,&cr->entrythreshold,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_coarsen_cr_abstol","Absolute threshold for considering unknown entries for the coarse set","MatCoarsen",cr->abstol,&cr->abstol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_coarsen_cr_maxsweeps","Maximum number of attempts at finding a coarse set that meets the convergence goals","MatCoarsen",cr->maxsweeps,&cr->maxsweeps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_coarsen_cr_maxratio","Maximum coarse nodes as a multiple of fine","MatCoarsen",cr->maxratio,&cr->maxratio,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_coarsen_cr_verbose","Verbose output for the coarsening","MatCoarsen",cr->verbose,&cr->verbose,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCoarsenCreate_CR"
PETSC_EXTERN PetscErrorCode MatCoarsenCreate_CR(MatCoarsen coarse)
{
  PetscErrorCode ierr;
  MatCoarsen_CR  *cr;

  PetscFunctionBegin;
  ierr           = PetscNewLog(coarse,&cr);CHKERRQ(ierr);
  coarse->subctx = (void*)cr;
  cr->convgoal                = 0.7;
  cr->entrythreshold          = 0.5;
  cr->abstol                  = 1e-6;
  cr->maxsweeps               = 5;
  cr->maxratio                = 0.5;
  cr->verbose                 = PETSC_FALSE;

  coarse->ops->apply          = MatCoarsenApply_CR;
  coarse->ops->view           = MatCoarsenView_CR;
  coarse->ops->destroy        = MatCoarsenDestroy_CR;
  coarse->ops->setfromoptions = MatCoarsenSetFromOptions_CR;
  PetscFunctionReturn(0);
}

#include <../src/ksp/pc/impls/gamg/gamg.h>        /*I "petscpc.h" I*/
#include <petsc-private/kspimpl.h>

typedef struct {
  PetscInt nv;   /* number of test vectors to use in the construction of the prolongator */
} PC_GAMG_Bootstrap;

typedef struct {
  PetscInt  nv;      /* the number of vectors in the space */
  Vec       *v;      /* the subspace */
} BootstrapTestSpace;

#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrapTestSpaceCreate"
PetscErrorCode PCGAMGBootstrapTestSpaceCreate(BootstrapTestSpace **bsts,Mat G,PetscInt n)
{
  BootstrapTestSpace *ts;
  PetscErrorCode ierr;
  Vec            v;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(BootstrapTestSpace),&ts);CHKERRQ(ierr);
  ierr = MatGetVecs(G,&v,NULL);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(v,n,&ts->v);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ts->nv = n;
  *bsts = ts;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrapTestSpaceDestroy"
PetscErrorCode PCGAMGBootstrapTestSpaceDestroy(BootstrapTestSpace *bsts)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"destroying test space\n");
  ierr = VecDestroyVecs(bsts->nv,&bsts->v);CHKERRQ(ierr);
  ierr = PetscFree(bsts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrapCreateTestSpace_Private"
PetscErrorCode PCGAMGBootstrapCreateTestSpace_Private(Mat G,PetscInt nv,Vec **v) {
  PetscErrorCode     ierr;
  PetscContainer     c;
  BootstrapTestSpace *bsts;

  PetscFunctionBegin;
  PetscPrintf(PETSC_COMM_WORLD,"creating test space\n");
  ierr = PetscContainerCreate(PetscObjectComm((PetscObject)G),&c);CHKERRQ(ierr);
  ierr = PCGAMGBootstrapTestSpaceCreate(&bsts,G,nv);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(c,bsts);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(c,(PetscErrorCode (*)(void *))PCGAMGBootstrapTestSpaceDestroy);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)G,"BootstrapTestSpace",(PetscObject)c);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&c);CHKERRQ(ierr);
  *v = bsts->v;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrapGetTestSpace_Private"
PetscErrorCode PCGAMGBootstrapGetTestSpace_Private(Mat G,PetscInt *nv,Vec **v)
{
  PetscErrorCode     ierr;
  PetscContainer     c;
  BootstrapTestSpace *bsts;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)G,"BootstrapTestSpace",(PetscObject *)&c);
  if (c) {
    ierr = PetscContainerGetPointer(c,(void**)&bsts);CHKERRQ(ierr);
    if (nv) *nv = bsts->nv;
    if (v) *v = bsts->v;
  } else {
    if (nv) *nv = 0;
    if (v) *v  = NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetupLevel_Bootstrap"
PetscErrorCode PCGAMGSetupLevel_Bootstrap(PC pc,Mat Gf,Mat P, Mat Gc)
{
  PetscErrorCode ierr;
  PC_MG             *mg          = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Bootstrap *pc_gamg_bs  = (PC_GAMG_Bootstrap*)pc_gamg->subctx;
  Vec               *vf;
  Vec               *vc;
  PetscInt          nv = pc_gamg_bs->nv,i;

  PetscFunctionBegin;
  ierr = PCGAMGBootstrapGetTestSpace_Private(Gf,NULL,&vf);CHKERRQ(ierr);
  if (!vf) {
    ierr = PCGAMGBootstrapCreateTestSpace_Private(Gf,nv,&vf);CHKERRQ(ierr);
    for (i=0;i<nv;i++) {
      ierr = VecSetRandom(vf[i],NULL);CHKERRQ(ierr);
    }
  }
  ierr = PCGAMGBootstrapCreateTestSpace_Private(Gc,nv,&vc);CHKERRQ(ierr);
  for (i=0;i<nv;i++) {
    ierr = MatRestrict(P,vf[i],vc[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrapCreateGhostVector_Private"
PetscErrorCode PCGAMGBootstrapCreateGhostVector_Private(Mat G,Vec *gvec,PetscInt **global)
{
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)G->data;
  PetscErrorCode ierr;
  PetscBool      isMPIAIJ;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)G, MATMPIAIJ, &isMPIAIJ); CHKERRQ(ierr);
  if (isMPIAIJ) {
    if (gvec)ierr = VecDuplicate(aij->lvec,gvec);CHKERRQ(ierr);
    if (global)*global = aij->garray;
  } else {
    /* no off-processor nodes */
    if (gvec)*gvec = NULL;
    if (global)*global = NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrapGraphSplitting_Private"
/*
 Split the relevant graph into diagonal and off-diagonal parts in local numbering; for now this
 a roundabout private interface to the mats' internal diag and offdiag mats.
 */
PetscErrorCode PCGAMGBootstrapGraphSplitting_Private(Mat G,Mat *Gd, Mat *Go)
{
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)G->data;
  PetscErrorCode ierr;
  PetscBool      isMPIAIJ;
  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)G, MATMPIAIJ, &isMPIAIJ ); CHKERRQ(ierr);
  if (isMPIAIJ) {
    *Gd = aij->A;
    *Go = aij->B;
  } else {
    *Gd = G;
    *Go = NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGGraph_Bootstrap"
PetscErrorCode PCGAMGGraph_Bootstrap(PC pc,const Mat A,Mat *G)
{
  PetscErrorCode            ierr;
  PC_MG                     *mg          = (PC_MG*)pc->data;
  PC_GAMG                   *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PetscInt                  verbose      = pc_gamg->verbose;
  PetscReal                 vfilter      = pc_gamg->threshold;
  Mat                       Gmat;

  PetscFunctionBegin;
  ierr = PCGAMGCreateGraph(A, &Gmat);CHKERRQ(ierr);
  ierr = PCGAMGFilterGraph(&Gmat,vfilter,PETSC_FALSE,verbose);CHKERRQ(ierr);
  *G = Gmat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGCoarsen_Bootstrap"
PetscErrorCode PCGAMGCoarsen_Bootstrap(PC pc,Mat *G,PetscCoarsenData **agg_lists)
{
  PetscErrorCode   ierr;
  MatCoarsen       crs;
  MPI_Comm         fcomm = ((PetscObject)pc)->comm;

  PetscFunctionBegin;

  /* construct the graph if necessary */
  if (!G) {
    SETERRQ(fcomm,PETSC_ERR_ARG_WRONGSTATE,"Must set Graph in PC in PCGAMG before coarsening");
  }

  ierr = MatCoarsenCreate(fcomm,&crs);CHKERRQ(ierr);
  ierr = MatCoarsenSetFromOptions(crs);CHKERRQ(ierr);
  ierr = MatCoarsenSetAdjacency(crs,*G);CHKERRQ(ierr);
  ierr = MatCoarsenSetStrictAggs(crs,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatCoarsenApply(crs);CHKERRQ(ierr);
  ierr = MatCoarsenGetData(crs,agg_lists);CHKERRQ(ierr);
  ierr = MatCoarsenDestroy(&crs);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrapGhost_Private"
/*
 Find all ghost nodes that are coarse and output the fine/coarse splitting for those as well

 Input:
 G - graph;
 gvec - Global Vector
 avec - Local part of the scattered vec
 bvec - Global part of the scattered vec

 Output:
 findx - indirection t

 */
PetscErrorCode PCGAMGBootstrapGhost_Private(Mat G,Vec v,Vec gv)
{
  PetscErrorCode ierr;
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)G->data;
  PetscBool      isMPIAIJ;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)G, MATMPIAIJ, &isMPIAIJ ); CHKERRQ(ierr);
  if (isMPIAIJ) {
    ierr = VecScatterBegin(aij->Mvctx,v,gv,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(aij->Mvctx,v,gv,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGProlongator_Bootstrap"
PetscErrorCode PCGAMGProlongator_Bootstrap(PC pc, const Mat A, const Mat G, PetscCoarsenData *agg_lists,Mat *P)
{
  PetscErrorCode    ierr;
  MPI_Comm          comm;
  Mat               lG,gG,lA,gA;     /* on and off diagonal matrices */
  PetscInt          fn;                        /* fine local blocked sizes */
  PetscInt          cn;                        /* coarse local blocked sizes */
  PetscInt          gn;                        /* size of the off-diagonal fine vector */
  PetscInt          fs,fe;                     /* fine (row) ownership range*/
  PetscInt          cs,ce;                     /* coarse (column) ownership range */
  PetscInt          i,j,k;                     /* indices! */
  PetscBool         iscoarse;                  /* flag for determining if a node is coarse */
  PetscInt          *lcid,*gcid;               /* on and off-processor coarse unknown IDs */
  PetscInt          *lsparse,*gsparse;         /* on and off-processor sparsity patterns for prolongator */
  PetscScalar       pij;
  const PetscScalar *rval;
  const PetscInt    *rcol;
  Vec               F;   /* vec of coarse size */
  Vec               C;   /* vec of fine size */
  Vec               gF;  /* vec of off-diagonal fine size */
  MatType           mtype;
  PetscInt          c_indx;
  const PetscScalar *vcols;
  const PetscInt    *icols;
  PetscScalar       c_scalar;
  PetscInt          ncols,col;
  PetscInt          row_f,row_c;
  PetscInt          cmax=0,ncolstotal,idx;
  PetscScalar       *pvals;
  PetscInt          *pcols;

  PetscFunctionBegin;
  comm = ((PetscObject)pc)->comm;
  ierr = MatGetOwnershipRange(A,&fs,&fe); CHKERRQ(ierr);
  fn = (fe - fs);

  ierr = MatGetVecs(A,&F,NULL);CHKERRQ(ierr);

  /* get the number of local unknowns and the indices of the local unknowns */

  ierr = PetscMalloc(sizeof(PetscInt)*fn,&lsparse);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*fn,&gsparse);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*fn,&lcid);CHKERRQ(ierr);

  /* count the number of coarse unknowns */
  cn = 0;
  for (i=0;i<fn;i++) {
    /* filter out singletons */
    ierr = PetscCDEmptyAt(agg_lists,i,&iscoarse); CHKERRQ(ierr);
    lcid[i] = -1;
    if (!iscoarse) {
      cn++;
    }
  }

   /* create the coarse vector */
  ierr = VecCreateMPI(comm,cn,PETSC_DECIDE,&C);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(C,&cs,&ce);CHKERRQ(ierr);

  /* construct a global vector indicating the global indices of the coarse unknowns */
  cn = 0;
  for (i=0;i<fn;i++) {
    ierr = PetscCDEmptyAt(agg_lists,i,&iscoarse); CHKERRQ(ierr);
    if (!iscoarse) {
      lcid[i] = cs+cn;
      cn++;
    } else {
      lcid[i] = -1;
    }
    c_scalar = (PetscScalar)lcid[i];
    c_indx = fs+i;
    ierr = VecSetValues(F,1,&c_indx,&c_scalar,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(F);CHKERRQ(ierr);

  /* split the graph into two */
  ierr = PCGAMGBootstrapGraphSplitting_Private(G,&lG,&gG);CHKERRQ(ierr);
  ierr = PCGAMGBootstrapGraphSplitting_Private(A,&lA,&gA);CHKERRQ(ierr);

  /* scatter to the ghost vector */
  ierr = PCGAMGBootstrapCreateGhostVector_Private(G,&gF,NULL);CHKERRQ(ierr);
  ierr = PCGAMGBootstrapGhost_Private(G,F,gF);CHKERRQ(ierr);

  if (gG) {
    ierr = VecGetSize(gF,&gn);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscInt)*gn,&gcid);CHKERRQ(ierr);
    for (i=0;i<gn;i++) {
      ierr = VecGetValues(gF,1,&i,&c_scalar);CHKERRQ(ierr);
      gcid[i] = (PetscInt)PetscRealPart(c_scalar);
    }
  }

  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = VecDestroy(&gF);CHKERRQ(ierr);
  ierr = VecDestroy(&C);CHKERRQ(ierr);

  /* count the on and off processor sparsity patterns for the prolongator */

  for (i=0;i<fn;i++) {
    /* on */
    ierr = MatGetRow(lG,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
    ncolstotal = ncols;
    lsparse[i] = 0;
    gsparse[i] = 0;
    if (lcid[i] >= 0) {
      lsparse[i] = 1;
      gsparse[i] = 0;
    } else {
      for (j = 0;j < ncols;j++) {
        col = icols[j];
        if (lcid[col] >= 0 && vcols[j] != 0.) {
          lsparse[i] += 1;
        }
      }
      ierr = MatRestoreRow(lG,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      ncolstotal += ncols;
      /* off */
      if (gG) {
        ierr = MatGetRow(gG,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
        for (j = 0; j < ncols; j++) {
          col = icols[j];
          if (gcid[col] >= 0 && vcols[j] != 0.) {
            gsparse[i] += 1;
          }
        }
        ierr = MatRestoreRow(gG,i,&ncols,NULL,NULL);CHKERRQ(ierr);
      }
      if (ncolstotal > cmax) cmax = ncolstotal;
    }
  }

  ierr = PetscMalloc(sizeof(PetscInt)*cmax,&pcols);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*cmax,&pvals);CHKERRQ(ierr);

  /* preallocate and create the prolongator */
  ierr = MatCreate(comm,P); CHKERRQ(ierr);
  ierr = MatGetType(G,&mtype);CHKERRQ(ierr);
  ierr = MatSetType(*P,mtype);CHKERRQ(ierr);

  ierr = MatSetSizes(*P,fn,cn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*P,0,lsparse,0,gsparse);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*P,0,lsparse);CHKERRQ(ierr);

  for (i = 0;i < fn;i++) {
    /* determine on or off */
    row_f = i + fs;
    row_c = lcid[i];
    if (row_c >= 0) {
      pij = 1.;
      ierr = MatSetValues(*P,1,&row_f,1,&row_c,&pij,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      PetscInt ntotal=0;
      /* local connections */
      ierr = MatGetRow(lG,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      for (k = 0; k < ncols; k++) {
        if (lcid[rcol[k]] >= 0) {
          ntotal++;
        }
      }
      ierr = MatRestoreRow(lG,i,&ncols,&rcol,&rval);CHKERRQ(ierr);

      /* ghosted connections */
      if (gG) {
        ierr = MatGetRow(gG,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
        for (k = 0; k < ncols; k++) {
          if (gcid[rcol[k]] >= 0) {
            ntotal++;
          }
        }
        ierr = MatRestoreRow(gG,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      }

      /* on */
      ierr = MatGetRow(lG,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      idx = 0;
      for (j = 0;j < ncols;j++) {
        col = icols[j];
        if (lcid[col] >= 0 && vcols[j] != 0.) {
          row_f = i + fs;
          row_c = lcid[col];
          pij = 1. / ntotal;
          pvals[idx] = pij;
          pcols[idx] = row_c;
          idx++;
        }
      }
      ierr = MatRestoreRow(lG,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      /* off */
      if (gG) {
        ierr = MatGetRow(gG,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
        for (j = 0; j < ncols; j++) {
          col = icols[j];
          if (gcid[col] >= 0 && vcols[j] != 0.) {
            row_f = i + fs;
            row_c = gcid[col];
            pij = 1./ntotal;
            pvals[idx] = pij;
            pcols[idx] = row_c;
            idx++;
          }
        }
        ierr = MatRestoreRow(gG,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      }
      ierr = MatSetValues(*P,1,&row_f,idx,pcols,pvals,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(*P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree(lsparse);CHKERRQ(ierr);
  ierr = PetscFree(gsparse);CHKERRQ(ierr);
  ierr = PetscFree(pcols);CHKERRQ(ierr);
  ierr = PetscFree(pvals);CHKERRQ(ierr);
  ierr = PetscFree(lcid);CHKERRQ(ierr);
  if (gG) {ierr = PetscFree(gcid);CHKERRQ(ierr);}

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGDestroy_Bootstrap"
PetscErrorCode PCGAMGDestroy_Bootstrap(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg          = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg     = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  ierr = PetscFree(pc_gamg->subctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetFromOptions_Bootstrap"
PetscErrorCode PCGAMGSetFromOptions_Bootstrap(PC pc)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetData_Bootstrap"
PetscErrorCode PCGAMGSetData_Bootstrap(PC pc, Mat A)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  /* no data for bootstrap AMG */
  pc_gamg->data           = NULL;
  pc_gamg->data_cell_cols = 1;
  pc_gamg->data_cell_rows = 1;
  pc_gamg->data_sz = 0;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "PCCreateGAMG_Bootstrap"
PetscErrorCode  PCCreateGAMG_Bootstrap(PC pc)
{
  PetscErrorCode ierr;
  PC_MG             *mg      = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Bootstrap *pc_gamg_bootstrap;

  PetscFunctionBegin;
  if (pc_gamg->subctx) {
    /* call base class */
    ierr = PCDestroy_GAMG(pc);CHKERRQ(ierr);
  }

  /* create sub context for SA */
  ierr = PetscNewLog(pc, PC_GAMG_Bootstrap, &pc_gamg_bootstrap);CHKERRQ(ierr);
  pc_gamg_bootstrap->nv = 10; 
  pc_gamg->subctx = pc_gamg_bootstrap;
  pc->ops->setfromoptions = PCGAMGSetFromOptions_Bootstrap;
  /* reset does not do anything; setup not virtual */

  /* set internal function pointers */
  pc_gamg->ops->destroy     = PCGAMGDestroy_Bootstrap;
  pc_gamg->ops->graph       = PCGAMGGraph_Bootstrap;
  pc_gamg->ops->coarsen     = PCGAMGCoarsen_Bootstrap;
  pc_gamg->ops->prolongator = PCGAMGProlongator_Bootstrap;
  pc_gamg->ops->optprol     = NULL;
  pc_gamg->ops->setuplevel  = PCGAMGSetupLevel_Bootstrap;

  pc_gamg->ops->createdefaultdata = PCGAMGSetData_Bootstrap;
  PetscFunctionReturn(0);
}

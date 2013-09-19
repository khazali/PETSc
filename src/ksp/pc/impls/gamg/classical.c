#include <../src/ksp/pc/impls/gamg/gamg.h>        /*I "petscpc.h" I*/
#include <petsc-private/kspimpl.h>
#include <petscblaslapack.h>

typedef struct {
  PetscReal dummy;
} PC_GAMG_Classical;

#undef __FUNCT__
#define __FUNCT__ "PCGAMGClassicalCreateGhostVector_Private"
PetscErrorCode PCGAMGClassicalCreateGhostVector_Private(Mat G,Vec *gvec,PetscInt **global)
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
#define __FUNCT__ "PCGAMGClassicalGraphSplitting_Private"
/*
 Split the relevant graph into diagonal and off-diagonal parts in local numbering; for now this
 a roundabout private interface to the mats' internal diag and offdiag mats.
 */
PetscErrorCode PCGAMGClassicalGraphSplitting_Private(Mat G,Mat *Gd, Mat *Go)
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
#define __FUNCT__ "PCGAMGGraph_Classical"
PetscErrorCode PCGAMGGraph_Classical(PC pc,const Mat A,Mat *G)
{
  PetscInt          s,f,idx;
  PetscInt          r,c,ncols;
  const PetscInt    *rcol;
  const PetscScalar *rval;
  PetscInt          *gcol;
  PetscScalar       *gval;
  PetscReal         rmax;
  PetscInt          ncolstotal,cmax = 0;
  PC_MG             *mg;
  PC_GAMG           *gamg;
  PetscErrorCode    ierr;
  PetscInt          *gsparse,*lsparse;
  PetscScalar       *Amax;
  Mat               lA,gA;
  MatType           mtype;

  PetscFunctionBegin;
  mg   = (PC_MG *)pc->data;
  gamg = (PC_GAMG *)mg->innerctx;

  ierr = MatGetOwnershipRange(A,&s,&f);CHKERRQ(ierr);

  ierr = PCGAMGClassicalGraphSplitting_Private(A,&lA,&gA);CHKERRQ(ierr);

  ierr = PetscMalloc(sizeof(PetscInt)*(f - s),&lsparse);CHKERRQ(ierr);
  if (gA) {ierr = PetscMalloc(sizeof(PetscInt)*(f - s),&gsparse);CHKERRQ(ierr);}
  else {
    gsparse = NULL;
  }
  ierr = PetscMalloc(sizeof(PetscScalar)*(f - s),&Amax);CHKERRQ(ierr);

  for (r = 0;r < f-s;r++) {
    lsparse[r] = 0;
    if (gsparse) gsparse[r] = 0;
  }

  for (r = 0;r < f-s;r++) {
    /* determine the maximum off-diagonal in each row */
    rmax = 0.;
    ierr = MatGetRow(lA,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
    ncolstotal = ncols;
    for (c = 0; c < ncols; c++) {
      if (PetscAbsScalar(rval[c]) > rmax && rcol[c] != r) {
        rmax = PetscAbsScalar(rval[c]);
      }
    }
    ierr = MatRestoreRow(lA,r,&ncols,&rcol,&rval);CHKERRQ(ierr);

    if (gA) {
      ierr = MatGetRow(gA,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
      ncolstotal += ncols;
      for (c = 0; c < ncols; c++) {
        if (PetscAbsScalar(rval[c]) > rmax) {
          rmax = PetscAbsScalar(rval[c]);
        }
      }
      ierr = MatRestoreRow(gA,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
    }
    Amax[r] = rmax;
    if (ncolstotal > cmax) cmax = ncolstotal;

    ierr = MatGetRow(lA,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
    idx = 0;

    /* create the local and global sparsity patterns */
    for (c = 0; c < ncols; c++) {
      if (PetscAbsScalar(rval[c]) > gamg->threshold*PetscRealPart(Amax[r])) {
        idx++;
      }
    }
    ierr = MatRestoreRow(lA,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
    lsparse[r] = idx;
    if (gA) {
      idx = 0;
      ierr = MatGetRow(gA,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
      for (c = 0; c < ncols; c++) {
        if (PetscAbsScalar(rval[c]) > gamg->threshold*PetscRealPart(Amax[r])) {
          idx++;
        }
      }
      ierr = MatRestoreRow(gA,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
      gsparse[r] = idx;
    }
  }
  ierr = PetscMalloc(sizeof(PetscScalar)*cmax,&gval);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*cmax,&gcol);CHKERRQ(ierr);

  ierr = MatCreate(PetscObjectComm((PetscObject)A),G); CHKERRQ(ierr);
  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = MatSetType(*G,mtype);CHKERRQ(ierr);
  ierr = MatSetSizes(*G,f-s,f-s,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*G,0,lsparse,0,gsparse);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*G,0,lsparse);CHKERRQ(ierr);
  for (r = s;r < f;r++) {
    ierr = MatGetRow(A,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
    idx = 0;
    for (c = 0; c < ncols; c++) {
      /* classical strength of connection */
      if (PetscAbsScalar(rval[c]) > gamg->threshold*PetscRealPart(Amax[r-s])) {
        gcol[idx] = rcol[c];
        gval[idx] = rval[c];
        idx++;
      }
    }
    ierr = MatSetValues(*G,1,&r,idx,gcol,gval,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,r,&ncols,&rcol,&rval);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*G, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*G, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree(gval);CHKERRQ(ierr);
  ierr = PetscFree(gcol);CHKERRQ(ierr);
  ierr = PetscFree(lsparse);CHKERRQ(ierr);
  ierr = PetscFree(gsparse);CHKERRQ(ierr);
  ierr = PetscFree(Amax);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCGAMGCoarsen_Classical"
PetscErrorCode PCGAMGCoarsen_Classical(PC pc,Mat *G,PetscCoarsenData **agg_lists)
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
#define __FUNCT__ "PCGAMGClassicalGhost_Private"
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
PetscErrorCode PCGAMGClassicalGhost_Private(Mat G,Vec v,Vec gv)
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
#define __FUNCT__ "PCGAMGProlongator_Classical"
PetscErrorCode PCGAMGProlongator_Classical(PC pc, const Mat A, const Mat G, PetscCoarsenData *agg_lists,Mat *P)
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
  PetscScalar       g_pos,g_neg,a_pos,a_neg,diag,invdiag,alpha,beta;
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
  ierr = PCGAMGClassicalGraphSplitting_Private(G,&lG,&gG);CHKERRQ(ierr);
  ierr = PCGAMGClassicalGraphSplitting_Private(A,&lA,&gA);CHKERRQ(ierr);

  /* scatter to the ghost vector */
  ierr = PCGAMGClassicalCreateGhostVector_Private(G,&gF,NULL);CHKERRQ(ierr);
  ierr = PCGAMGClassicalGhost_Private(G,F,gF);CHKERRQ(ierr);

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

  /* loop over local fine nodes -- get the diagonal, the sum of positive and negative strong and weak weights, and set up the row */
  for (i = 0;i < fn;i++) {
    /* determine on or off */
    row_f = i + fs;
    row_c = lcid[i];
    if (row_c >= 0) {
      pij = 1.;
      ierr = MatSetValues(*P,1,&row_f,1,&row_c,&pij,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      PetscInt nstrong=0,ntotal=0;
      g_pos = 0.;
      g_neg = 0.;
      a_pos = 0.;
      a_neg = 0.;
      diag = 0.;

      /* local strong connections */
      ierr = MatGetRow(lG,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      for (k = 0; k < ncols; k++) {
        if (lcid[rcol[k]] >= 0) {
          if (PetscRealPart(rval[k]) > 0) {
            g_pos += rval[k];
          } else {
            g_neg += rval[k];
          }
          nstrong++;
        }
      }
      ierr = MatRestoreRow(lG,i,&ncols,&rcol,&rval);CHKERRQ(ierr);

      /* ghosted strong connections */
      if (gG) {
        ierr = MatGetRow(gG,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
        for (k = 0; k < ncols; k++) {
          if (gcid[rcol[k]] >= 0) {
            if (PetscRealPart(rval[k]) > 0.) {
              g_pos += rval[k];
            } else {
              g_neg += rval[k];
            }
            nstrong++;
          }
        }
        ierr = MatRestoreRow(gG,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      }

      /* local all connections */
      ierr = MatGetRow(lA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      for (k = 0; k < ncols; k++) {
        if (rcol[k] != i) {
          if (PetscRealPart(rval[k]) > 0) {
            a_pos += rval[k];
          } else {
            a_neg += rval[k];
          }
          ntotal++;
        } else diag = rval[k];
      }
      ierr = MatRestoreRow(lA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);

      /* ghosted all connections */
      if (gA) {
        ierr = MatGetRow(gA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
        for (k = 0; k < ncols; k++) {
          if (PetscRealPart(rval[k]) > 0.) {
            a_pos += PetscRealPart(rval[k]);
          } else {
            a_neg += PetscRealPart(rval[k]);
          }
          ntotal++;
        }
        ierr = MatRestoreRow(gA,i,&ncols,&rcol,&rval);CHKERRQ(ierr);
      }

      if (g_neg == 0.) {
        alpha = 0.;
      } else {
        alpha = -a_neg/g_neg;
      }

      if (g_pos == 0.) {
        diag += a_pos;
        beta = 0.;
      } else {
        beta = -a_pos/g_pos;
      }
      if (diag == 0.) {
        invdiag = 0.;
      } else invdiag = 1. / diag;
      /* on */
      ierr = MatGetRow(lG,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      idx = 0;
      for (j = 0;j < ncols;j++) {
        col = icols[j];
        if (lcid[col] >= 0 && vcols[j] != 0.) {
          row_f = i + fs;
          row_c = lcid[col];
          /* set the values for on-processor ones */
          if (PetscRealPart(vcols[j]) < 0.) {
            pij = vcols[j]*alpha*invdiag;
          } else {
            pij = vcols[j]*beta*invdiag;
          }
          if (PetscAbsScalar(pij) != 0.) {
            pvals[idx] = pij;
            pcols[idx] = row_c;
            idx++;
          }
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
            /* set the values for on-processor ones */
            if (PetscRealPart(vcols[j]) < 0.) {
              pij = vcols[j]*alpha*invdiag;
            } else {
              pij = vcols[j]*beta*invdiag;
            }
            if (PetscAbsScalar(pij) != 0.) {
              pvals[idx] = pij;
              pcols[idx] = row_c;
              idx++;
            }
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
#define __FUNCT__ "PCGAMGDestroy_Classical"
PetscErrorCode PCGAMGDestroy_Classical(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg          = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg     = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  ierr = PetscFree(pc_gamg->subctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetFromOptions_Classical"
PetscErrorCode PCGAMGSetFromOptions_Classical(PC pc)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("GAMG Classical options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetData_Classical"
PetscErrorCode PCGAMGSetData_Classical(PC pc, Mat A)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  /* no data for classical AMG */
  pc_gamg->data           = NULL;
  pc_gamg->data_cell_cols = 1;
  pc_gamg->data_cell_rows = 1;
  pc_gamg->data_sz = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGClassicalInjection"
PetscErrorCode PCGAMGClassicalInjection(PC pc,Mat P,VecScatter *inj)
{
  PetscErrorCode    ierr;
  PetscInt          ncols,i,j,fe,fs,fn;
  const PetscInt    *icols;
  const PetscScalar *vcols;
  PetscInt          *fmap,*cmap;
  IS                fis,cis;
  Vec               wf,wc;
  MPI_Comm          comm;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)pc);
  ierr = MatGetVecs(P,&wc,&wf);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(wf,&fs,&fe);CHKERRQ(ierr);
  fn = fe-fs;
  /* build the scatter injection into the coarse space */
  ierr = PetscMalloc(fn*sizeof(PetscInt),&cmap);CHKERRQ(ierr);
  ierr = PetscMalloc(fn*sizeof(PetscInt),&fmap);CHKERRQ(ierr);
  j=0;
  for (i=0;i<fn;i++) fmap[i]=-1;
  for (i=0;i<fn;i++) cmap[i]=-1;
  for (i=fs;i<fe;i++) {
    ierr = MatGetRow(P,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
    if (ncols == 1) {
      if (vcols[0] == 1.) {
        fmap[j] = i;
        cmap[j] = icols[0];
        j++;
      }
    }
    ierr = MatRestoreRow(P,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
  }

  ierr = ISCreateGeneral(comm,j,fmap,PETSC_OWN_POINTER,&fis);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,j,cmap,PETSC_OWN_POINTER,&cis);CHKERRQ(ierr);
  ierr = VecScatterCreate(wf,fis,wc,cis,inj);CHKERRQ(ierr);
  ierr = ISDestroy(&fis);CHKERRQ(ierr);
  ierr = ISDestroy(&cis);CHKERRQ(ierr);
  ierr = VecDestroy(&wf);CHKERRQ(ierr);
  ierr = VecDestroy(&wc);CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGClassicalBootstrapProlongator"
PetscErrorCode PCGAMGClassicalBootstrapProlongator(PC pc,const Mat A,Mat *aP,PetscInt nv,Vec *vs)
{
  PetscErrorCode    ierr;
  Mat               P=*aP;
  Mat               Pnew;
  Mat               lP,gP;
  PetscInt          rs,re,rn;                  /* fine (row) ownership range*/
  PetscInt          i,j,k,l,iidx;              /* indices! */
  Vec               cgv,*cvs,*cgvs;        /* vec of off-diagonal fine size */
  const PetscScalar *vcols,*gvcols;
  const PetscInt    *icols,*gicols;
  PetscScalar       *wvcols;
  PetscInt          *wicols;
  PC_MG             *mg          = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PetscInt          ncols,gncols,ncolstotal,ncolsloc;
  PetscScalar       *a,*b;
  PetscScalar       *wts;
  Vec               wf,wc;                    /* fine and coarse work vectors */
  PetscInt          cn,cs,ce;
  PetscInt          *gidx;
  PetscScalar       *vsarray,*cvsarray,*cgvsarray;
  PetscScalar       denom;
  PetscInt          sz,ncolsmax=0;
  PetscBLASInt      nls,mls,nrhs,lda,ldb,lwork,info,rank;
  PetscScalar       *work;
  PetscReal         *s,rcond;
  VecScatter        inj;
#if defined(PETSC_USE_COMPLEX)
  PetscReal         *rwork;
#endif

  PetscFunctionBegin;
  /* split the prolongator into two */
  ierr = PCGAMGClassicalGraphSplitting_Private(P,&lP,&gP);CHKERRQ(ierr);
  ierr = MatDuplicate(P,MAT_SHARE_NONZERO_PATTERN,&Pnew);CHKERRQ(ierr);

  ierr = MatGetVecs(P,&wc,&wf);CHKERRQ(ierr);
  ierr = VecGetLocalSize(wc,&cn);CHKERRQ(ierr);
  ierr = VecGetLocalSize(wf,&rn);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(wc,&cs,&ce);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rs,&re);CHKERRQ(ierr);

  ierr = PCGAMGClassicalInjection(pc,P,&inj);CHKERRQ(ierr);

  ierr = PetscMalloc(sizeof(PetscScalar)*nv,&wts);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(wc,nv,&cvs);CHKERRQ(ierr);
  if (gP) {
    ierr = PCGAMGClassicalCreateGhostVector_Private(P,&cgv,&gidx);CHKERRQ(ierr);
  }
  if (gP) {
    ierr = VecDuplicateVecs(cgv,nv,&cgvs);CHKERRQ(ierr);
  }

  /* find the biggest column and allocate */
  for (i=rs;i<re;i++) {
    ierr = MatGetRow(P,i,&ncols,NULL,NULL);CHKERRQ(ierr);
    if (ncols > ncolsmax) ncolsmax=ncols;
    ierr = MatRestoreRow(P,i,&ncols,NULL,NULL);CHKERRQ(ierr);
  }
  ierr = PetscMalloc6(ncolsmax*ncolsmax,PetscScalar,&a,
                      ncolsmax,PetscScalar,&b,
                      12*ncolsmax,PetscScalar,&work,
                      ncolsmax,PetscScalar,&s,
                      ncolsmax,PetscScalar,&wvcols,
                      ncolsmax,PetscScalar,&wicols);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc(ncolsmax*sizeof(PetscScalar),&rwork);CHKERRQ(ierr);
#endif

  /* construct the weights by the interpolation Rayleigh quotient <Av,v>^-1<Pv,Pv> */
  for (j=0;j<nv;j++) {
    ierr = MatMult(A,vs[j],wf);CHKERRQ(ierr);
    ierr = MatRestrict(P,vs[j],wc);CHKERRQ(ierr);
    ierr = VecScatterBegin(inj,vs[j],cvs[j],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(inj,vs[j],cvs[j],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecDotBegin(wf,vs[j],&wts[j]);CHKERRQ(ierr);
    ierr = VecDotBegin(wc,wc,&denom);CHKERRQ(ierr);
    ierr = VecDotEnd(wf,vs[j],&wts[j]);CHKERRQ(ierr);
    ierr = VecDotEnd(wc,wc,&denom);CHKERRQ(ierr);
    if (PetscRealPart(wts[j]) > 0.) {
      wts[j] = denom / PetscRealPart(wts[j]);
    } else { /* nullspace */
      wts[j] = 1.;
    }
    ierr = VecGetSize(vs[j],&sz);CHKERRQ(ierr);
    if (gP) {
      ierr = PCGAMGClassicalGhost_Private(P,cvs[j],cgvs[j]);CHKERRQ(ierr);
    }
  }

  /* sanity check -- see how good the initial projection is */
  for (i=0;i<nv;i++) {
    PetscReal vsnrm,dnrm;
    ierr = MatMult(P,cvs[i],wf);CHKERRQ(ierr);
    ierr = VecAXPY(wf,-1.,vs[i]);CHKERRQ(ierr);
    /* have to zero singleton rows */
    for (j=rs;j<re;j++) {
      ierr = MatGetRow(P,j,&ncols,&icols,&vcols);CHKERRQ(ierr);
      if (ncols == 0) {
        ierr = VecSetValue(wf,j,0.,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(P,j,&ncols,&icols,&vcols);CHKERRQ(ierr);
    }
    ierr = VecNorm(vs[i],NORM_2,&vsnrm);CHKERRQ(ierr);
    ierr = VecNorm(wf,NORM_2,&dnrm);CHKERRQ(ierr);
    if (pc_gamg->verbose) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Vector %d; Norm: %f Rel. error: %f\n",i,vsnrm,dnrm/vsnrm);CHKERRQ(ierr);
    }
  }

  for (i=0;i<rn;i++) {
    iidx=i+rs;
    /* set up the least squares minimization problem */
    ierr = MatGetRow(lP,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
    ncolsloc = ncols;
    ncolstotal = ncols;
    ierr = MatRestoreRow(lP,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
    if (gP) {
      ierr = MatGetRow(gP,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      ncolstotal += ncols;
      ierr = MatRestoreRow(gP,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
    }
    if (ncolstotal > 0) {
      for (j=0;j<ncolsmax;j++) {
        b[j] = 0.;
        for (k=0;k<ncolsmax;k++) {
          a[k+ncolsmax*j] = 0.;
        }
      }
      ierr = MatGetRow(lP,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      if (gP) {
        ierr = MatGetRow(gP,i,&gncols,&gicols,&gvcols);CHKERRQ(ierr);
      }
      for (l=0;l<nv;l++) {
        ierr = VecGetArray(vs[l],&vsarray);CHKERRQ(ierr);
        ierr = VecGetArray(cvs[l],&cvsarray);CHKERRQ(ierr);
        if (gP) {
          ierr = VecGetArray(cgvs[l],&cgvsarray);CHKERRQ(ierr);
        }
        /* addition for on-processor entries */
        for (j=0;j<ncols;j++) {
          b[j] += wts[l]*cvsarray[icols[j]]*vsarray[i];
          for (k=0;k<ncols;k++) {
            a[k+j*ncolstotal] += wts[l]*cvsarray[icols[j]]*cvsarray[icols[k]];
          }
          if (gP) {
            for (k=0;k<gncols;k++) {
              a[k+ncolsloc+j*ncolstotal] += wts[l]*cvsarray[icols[j]]*cgvsarray[gicols[k]]*wts[l];
            }
          }
        }
        /* addition for off-processor entries */
        if (gP) {
          for (j=0;j<gncols;j++) {
            b[j+ncolsloc] += wts[l]*cgvsarray[gicols[j]]*vsarray[i];
            for (k=0;k<ncols;k++) {
              a[k+(j+ncolsloc)*ncolstotal] += wts[l]*cgvsarray[gicols[j]]*cvsarray[icols[k]];
            }
            for (k=0;k<gncols;k++) {
              a[k+ncolsloc+(j+ncolsloc)*ncolstotal] += wts[l]*cgvsarray[gicols[j]]*cgvsarray[gicols[k]];
            }
          }
        }
        ierr = VecRestoreArray(vs[l],&vsarray);CHKERRQ(ierr);
        ierr = VecRestoreArray(cvs[l],&cvsarray);CHKERRQ(ierr);
        if (gP) {
          ierr = VecRestoreArray(cgvs[l],&cgvsarray);CHKERRQ(ierr);
        }
      }
      ierr = MatRestoreRow(lP,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      if (gP) {
        ierr = MatRestoreRow(gP,i,&gncols,&gicols,&gvcols);CHKERRQ(ierr);
      }
      nls=ncolstotal;mls=ncolstotal;nrhs=1;lda=ncolstotal;ldb=ncolstotal;lwork=12*ncolsmax;info=0;
      /* solve the problem */
#if defined(PETSC_MISSING_LAPACK_GELSS)
      SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"LS solve in PCGAMGBootstrap requires the LAPACK GELSS routine.");
#else
      rcond         = -1.;
      ierr          = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&nls,&mls,&nrhs,a,&lda,b,&ldb,s,&rcond,&rank,work,&lwork,rwork,&info));
#else
      PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&nls,&mls,&nrhs,a,&lda,b,&ldb,s,&rcond,&rank,work,&lwork,&info));
#endif
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
      if (info < 0) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_LIB,"Bad argument to GELSS");
      if (info > 0) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_LIB,"SVD failed to converge");
#endif
      /* set the row to be the solution */
      ierr = MatGetRow(lP,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      for (j=0;j<ncols;j++) {
        wvcols[j] = b[j];
        wicols[j] = icols[j]+cs;
      }
      ierr = MatRestoreRow(lP,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      if (gP) {
        ierr = MatGetRow(gP,i,&gncols,&gicols,&gvcols);CHKERRQ(ierr);
        for (j=0;j<gncols;j++) {
          wvcols[j+ncolsloc] = b[j+ncolsloc];
          wicols[j+ncolsloc] = gidx[gicols[j]];
        }
        ierr = MatRestoreRow(gP,i,&gncols,&gicols,&gvcols);CHKERRQ(ierr);
      }
      ierr = MatSetValues(Pnew,1,&iidx,ncolstotal,wicols,wvcols,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      ierr = MatGetRow(P,iidx,&ncols,&icols,&vcols);CHKERRQ(ierr);
      ierr = MatSetValues(Pnew,1,&iidx,ncols,icols,vcols,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(P,iidx,&ncols,&icols,&vcols);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree6(a,b,work,s,wvcols,wicols);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#endif

  ierr = MatAssemblyBegin(Pnew, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Pnew, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* sanity check -- see how good the projection is */
  for (i=0;i<nv;i++) {
    PetscReal vsnrm,dnrm;
    ierr = MatMult(Pnew,cvs[i],wf);CHKERRQ(ierr);
    ierr = VecAXPY(wf,-1.,vs[i]);CHKERRQ(ierr);
    /* have to zero singleton rows */
    for (j=rs;j<re;j++) {
      ierr = MatGetRow(Pnew,j,&ncols,&icols,&vcols);CHKERRQ(ierr);
      if (ncols == 0) {
        ierr = VecSetValue(wf,j,0.,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(Pnew,j,&ncols,&icols,&vcols);CHKERRQ(ierr);
    }
    ierr = VecNorm(vs[i],NORM_2,&vsnrm);CHKERRQ(ierr);
    ierr = VecNorm(wf,NORM_2,&dnrm);CHKERRQ(ierr);
    if (pc_gamg->verbose) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Vector %d; Norm: %f Rel. error: %f\n",i,vsnrm,dnrm/vsnrm);CHKERRQ(ierr);
    }
  }

  ierr = MatView(Pnew,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(&wf);CHKERRQ(ierr);
  ierr = VecDestroy(&wc);CHKERRQ(ierr);
  ierr = VecDestroyVecs(nv,&cvs);CHKERRQ(ierr);
  if (gP) {
    ierr = VecDestroy(&cgv);CHKERRQ(ierr);
    ierr = VecDestroyVecs(nv,&cgvs);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&P);CHKERRQ(ierr);
  *aP = Pnew;
  ierr = VecScatterDestroy(&inj);CHKERRQ(ierr);
  ierr = PetscFree(wts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrap_Classical"
PetscErrorCode PCGAMGBootstrap_Classical(PC pc,PetscInt nlevels,Mat *A,Mat *P)
{
  PC_MG             *mg       = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg  = (PC_GAMG*)mg->innerctx;
  KSP               bootksp;
  PC                bootpc;
  MPI_Comm          comm;
  PetscErrorCode    ierr;
  PetscInt          i,j,k,nv=pc_gamg->bs_nv;
  Vec               w,v;
  const char        *prefix;
  Vec               **vs;
  PetscRandom       rand;
  PetscInt          rs,re,ncols,ncolsmax;

  PetscFunctionBegin;

  /* set up the bootstrap test space at each level */
  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
  comm = PetscObjectComm((PetscObject)pc);
  if (nv < 1) {
    /* decide nv adaptively as 2 times the widest row */
    ncolsmax=0;
    ierr = MatGetOwnershipRange(A[0],&rs,&re);CHKERRQ(ierr);
    for (i=rs;i<re;i++) {
      ierr = MatGetRow(A[0],i,&ncols,NULL,NULL);CHKERRQ(ierr);
      if (ncols > ncolsmax) ncolsmax=ncols;
      ierr = MatRestoreRow(A[0],i,&ncols,NULL,NULL);CHKERRQ(ierr);
    }
    nv = 2*ncolsmax;
  }

  ierr = PetscMalloc(sizeof(Vec*)*nlevels,&vs);CHKERRQ(ierr);
  for (i=0;i<nlevels;i++) {
    ierr = MatGetVecs(A[i],NULL,&v);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(v,nv,&vs[i]);CHKERRQ(ierr);
    ierr = VecDestroy(&v);CHKERRQ(ierr);
  }

  ierr = PetscRandomCreate(comm,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetType(rand,PETSCRAND48);CHKERRQ(ierr);
  for (i=0;i<nv;i++) {
    ierr = VecSetRandom(vs[0][i],rand);CHKERRQ(ierr);
  }
  for(k=0;k<pc_gamg->bs_sweeps;k++) {
    /* optimize the prolongators,then project the spaces up */
    for (i=0;i<nlevels;i++) {
      if (i != nlevels-1) {
        ierr = MatGetVecs(A[i],NULL,&w);CHKERRQ(ierr);
        ierr = KSPCreate(comm,&bootksp);CHKERRQ(ierr);
        ierr = KSPSetOptionsPrefix(bootksp,"boot_");CHKERRQ(ierr);
        ierr = KSPAppendOptionsPrefix(bootksp,prefix);CHKERRQ(ierr);
        ierr = KSPSetType(bootksp,KSPGMRES);CHKERRQ(ierr);
        ierr = KSPSetInitialGuessNonzero(bootksp,PETSC_TRUE);CHKERRQ(ierr);
        ierr = KSPSetTolerances(bootksp,bootksp->rtol,bootksp->abstol,bootksp->divtol,1);CHKERRQ(ierr);
        ierr = KSPSetOperators(bootksp,A[i],A[i],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
        ierr = KSPGetPC(bootksp,&bootpc);CHKERRQ(ierr);
        ierr = PCSetType(bootpc,PCSOR);CHKERRQ(ierr);
        ierr = KSPSetFromOptions(bootksp);CHKERRQ(ierr);
        ierr = VecSet(w,0.);CHKERRQ(ierr);
        for (j=0;j<nv;j++) {
          ierr = KSPSolve(bootksp,w,vs[i][j]);CHKERRQ(ierr);
        }
        ierr = VecDestroy(&w);CHKERRQ(ierr);
        ierr = KSPDestroy(&bootksp);CHKERRQ(ierr);
        ierr = PCGAMGClassicalBootstrapProlongator(pc,A[i],&P[i+1],nv,vs[i]);CHKERRQ(ierr);
        ierr = MatDestroy(&A[i+1]);CHKERRQ(ierr);
        ierr = MatPtAP(A[i],P[i+1],MAT_INITIAL_MATRIX,2.0,&A[i+1]);CHKERRQ(ierr);
        for (j=0;j<nv;j++) {
          ierr = MatRestrict(P[i+1],vs[i][j],vs[i+1][j]);CHKERRQ(ierr);
        }
      }
    }
  }
  for (i=0;i<nlevels;i++) {
    ierr = VecDestroyVecs(nv,&vs[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(vs);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = KSPDestroy(&bootksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
/*
   PCCreateGAMG_Classical

*/
#undef __FUNCT__
#define __FUNCT__ "PCCreateGAMG_Classical"
PetscErrorCode  PCCreateGAMG_Classical(PC pc)
{
  PetscErrorCode ierr;
  PC_MG             *mg      = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Classical *cls;

  PetscFunctionBegin;
  if (pc_gamg->subctx) {
    /* call base class */
    ierr = PCDestroy_GAMG(pc);CHKERRQ(ierr);
  }

  /* create sub context for SA */
  ierr = PetscNewLog(pc, PC_GAMG_Classical, &cls);CHKERRQ(ierr);
  pc_gamg->subctx = cls;

  /* set internal function pointers */
  pc_gamg->ops->setfromoptions = PCGAMGSetFromOptions_Classical;
  pc_gamg->ops->destroy     = PCGAMGDestroy_Classical;
  pc_gamg->ops->graph       = PCGAMGGraph_Classical;
  pc_gamg->ops->coarsen     = PCGAMGCoarsen_Classical;
  pc_gamg->ops->prolongator = PCGAMGProlongator_Classical;
  pc_gamg->ops->optprol     = NULL;
  pc_gamg->ops->bootstrap   = PCGAMGBootstrap_Classical;

  pc_gamg->ops->createdefaultdata = PCGAMGSetData_Classical;
  PetscFunctionReturn(0);
}

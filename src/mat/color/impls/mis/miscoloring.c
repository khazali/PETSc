#include <petsc-private/matimpl.h>      /*I "petscmat.h"  I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

PETSC_EXTERN PetscErrorCode MatColoringGetLocalGraph_Private(Mat G,PetscInt **ia,PetscInt **ja,PetscInt **iao,PetscInt **jao);
PETSC_EXTERN PetscErrorCode MatColoringGetDistanceKGraph(Mat M,PetscInt k,PetscInt **dia,Mat *Mk);

#undef __FUNCT__
#define __FUNCT__ "MISCreateWeights_Private"
PetscErrorCode MISCreateWeights_Private(MatColoring mc,PetscReal *weights,PetscInt *lperm)
{
  Mat            G=mc->graph;
  PetscErrorCode ierr;
  PetscInt       i,ncols,s,e,n;
  PetscRandom    rand;
  PetscReal      r;

  PetscFunctionBegin;
  /* each weight should be the degree plus a random perturbation */
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)mc),&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(G,&s,&e);CHKERRQ(ierr);
  n=e-s;
  for (i=s;i<e;i++) {
    ierr = MatGetRow(G,i,&ncols,NULL,NULL);CHKERRQ(ierr);
    ierr = PetscRandomGetValueReal(rand,&r);CHKERRQ(ierr);
    weights[i-s] = ncols + PetscAbsReal(r);
    lperm[i-s] = i-s;
    ierr = MatRestoreRow(G,i,&ncols,NULL,NULL);CHKERRQ(ierr);
  }
  ierr = PetscSortRealWithPermutation(n,weights,lperm);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringComputeMIS_Private"
PETSC_EXTERN PetscErrorCode MatColoringComputeMIS_Private(MatColoring mc,PetscInt *lperm,PetscReal *weights,Vec wtvec,ISColoringValue curcolor,ISColoringValue *colors,PetscInt *ncurcolor)
{
  Mat            G=mc->graph,Gd,Go;
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)G->data;
  PetscErrorCode ierr;
  PetscBool      isMPIAIJ,isSEQAIJ;
  PetscInt       idx,i,j,ms,me,mn,msize;
  PetscInt       nr,nr_global,nr_global_old;
  Vec            ovec;
  VecScatter     oscatter;
  PetscScalar    *wtarray,*owtarray;
  const PetscInt *cidx;
  PetscInt       ncols;
  PetscBool      isIn,isValid;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)G,MATMPIAIJ,&isMPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)G,MATSEQAIJ,&isSEQAIJ);CHKERRQ(ierr);
  if (isMPIAIJ) {
    Gd = aij->A;
    Go = aij->B;
    ovec = aij->lvec;
    oscatter = aij->Mvctx;
  } else if (isSEQAIJ) {
    Gd = G;
    Go = NULL;
    ovec = NULL;
    oscatter = NULL;
  } else {
    SETERRQ(PetscObjectComm((PetscObject)mc),PETSC_ERR_ARG_WRONGSTATE,"Only support for AIJ matrices");
  }
  /* distance one coloring -- super simple, just do it greedily based upon locally-known weights */
  ierr = MatGetOwnershipRange(G,&ms,&me);CHKERRQ(ierr);
  ierr = MatGetSize(G,&msize,NULL);CHKERRQ(ierr);
  mn = me-ms;
  ierr = VecGetArray(wtvec,&wtarray);CHKERRQ(ierr);
  for (i=0;i<mn;i++) {
    if (colors[i]==IS_COLORING_MAX || colors[i]==curcolor) {
      wtarray[i] = weights[i];
    } else {
      wtarray[i] = 0.;
    }
  }
  ierr = VecRestoreArray(wtvec,&wtarray);CHKERRQ(ierr);
  nr_global=0;
  nr_global_old=-1;
  nr=0;
  while (nr_global != nr_global_old) {
    nr_global_old=nr_global;
    nr_global=0;
    /* transfer weights over by the scatter */
    if (oscatter) {
      ierr = VecScatterBegin(oscatter,wtvec,ovec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(oscatter,wtvec,ovec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    }
    /* for each local row, check if it's the top weight in its neighbors, and alter the weights in the weight vector accordingly as things are eliminated*/
    ierr = VecGetArray(wtvec,&wtarray);CHKERRQ(ierr);
    if (ovec) {
      ierr = VecGetArray(ovec,&owtarray);CHKERRQ(ierr);
    }
    for (i=0;i<mn;i++) {
      idx = lperm[mn-i-1];
      if (PetscRealPart(wtarray[idx]) > 0. && colors[idx]==IS_COLORING_MAX) {
        isIn = PETSC_TRUE;
        isValid = PETSC_TRUE;
        ierr = MatGetRow(Gd,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) {
          if (PetscRealPart(wtarray[cidx[j]]) < 0) {
            isValid=PETSC_FALSE;
            isIn=PETSC_FALSE;
            break;
          }
          if (PetscRealPart(wtarray[cidx[j]]) > PetscRealPart(wtarray[idx])) {
            isIn = PETSC_FALSE;
            break;
          }
        }
        ierr = MatRestoreRow(Gd,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        if (Go) {
          ierr = MatGetRow(Go,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
          for (j=0;j<ncols;j++) {
            if (PetscRealPart(owtarray[cidx[j]]) < 0) {
              isValid=PETSC_FALSE;
              isIn=PETSC_FALSE;
              break;
            }
            if (PetscRealPart(owtarray[cidx[j]]) > PetscRealPart(wtarray[idx])) {
              isIn = PETSC_FALSE;
              break;
            }
          }
          ierr = MatRestoreRow(Go,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        }
        if (!isValid) {
          wtarray[idx]=0.;
        }
        if (isIn) {
          colors[idx]=curcolor;
          nr++;
          wtarray[idx]=-1;
        }
      }
    }
    ierr = VecRestoreArray(wtvec,&wtarray);CHKERRQ(ierr);
    if (ovec) {
      ierr = VecRestoreArray(ovec,&owtarray);CHKERRQ(ierr);
    }
    ierr = MPI_Allreduce(&nr,&nr_global,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
  }
  *ncurcolor=nr_global;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringApply_MIS"
PETSC_EXTERN PetscErrorCode MatColoringApply_MIS(MatColoring mc,ISColoring *iscoloring)
{
  PetscErrorCode  ierr;
  ISColoringValue curcolor,finalcolor;
  ISColoringValue *colors;
  PetscInt        i,n,s,e,nadded_total,ncolstotal,ncols,ncurcolor;
  Vec             work;
  PetscReal       *wts;
  PetscInt        *lperm;

  PetscFunctionBegin;
  nadded_total=0;
  ierr = MatGetSize(mc->graph,NULL,&ncolstotal);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mc->graph,NULL,&ncols);CHKERRQ(ierr);
  ierr = MatGetVecs(mc->graph,NULL,&work);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(mc->graph,&s,&e);CHKERRQ(ierr);
  n=e-s;
  ierr = PetscMalloc1(n,&wts);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&lperm);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&colors);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    colors[i] = IS_COLORING_MAX;
  }
  ierr = MISCreateWeights_Private(mc,wts,lperm);CHKERRQ(ierr);
  curcolor=0;
  for (i=0;(i<mc->maxcolors || mc->maxcolors == 0) && (nadded_total < ncolstotal);i++) {
    ierr = MatColoringComputeMIS_Private(mc,lperm,wts,work,curcolor,colors,&ncurcolor);CHKERRQ(ierr);
    nadded_total += ncurcolor;
    curcolor++;
  }
  finalcolor = curcolor;
  for (i=0;i<ncols;i++) {
    /* set up a dummy color if the coloring has been truncated */
    if (colors[i] == IS_COLORING_MAX) {
      colors[i] = curcolor;
      finalcolor = curcolor+1;
    }
  }
  ierr = ISColoringCreate(PetscObjectComm((PetscObject)mc),finalcolor,ncols,colors,iscoloring);CHKERRQ(ierr);
  ierr = PetscFree(wts);CHKERRQ(ierr);
  /* ierr = PetscFree(colors);CHKERRQ(ierr); */
  ierr = PetscFree(lperm);CHKERRQ(ierr);
  ierr = VecDestroy(&work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatColoringGetGraph_Private(MatColoring,Mat*);

#undef __FUNCT__
#define __FUNCT__ "MatColoringCreate_MIS"
/*MC
  MATCOLORINGMIS - Maximal Independent Set based Matrix Coloring

   Level: beginner

   Notes: This algorithm uses a Luby-type method to create a series of independent sets that may be combined into a
   maximal independent set.  This is repeated on the induced subgraph of uncolored vertices until every column of the
   matrix is assigned a color.  This algorithm supports arbitrary distance.  If the maximum number of colors is set to
   one, it will create a maximal independent set.

.seealso: MatColoringCreate(), MatColoring, MatColoringSetType()
M*/
PETSC_EXTERN PetscErrorCode MatColoringCreate_MIS(MatColoring mc)
{
  PetscFunctionBegin;
  mc->ops->apply          = MatColoringApply_MIS;
  mc->ops->view           = NULL;
  mc->ops->graph          = MatColoringGetGraph_Private;
  mc->ops->destroy        = NULL;
  mc->ops->setfromoptions = NULL;
  PetscFunctionReturn(0);
}

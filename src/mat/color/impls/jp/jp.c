#include <petsc-private/matimpl.h>      /*I "petscmat.h"  I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

PETSC_EXTERN PetscErrorCode MatColoringGetGraph_Private(MatColoring,Mat*);

#undef __FUNCT__
#define __FUNCT__ "JPCreateWeights_Private"
PetscErrorCode JPCreateWeights_Private(MatColoring mc,PetscReal *weights,PetscInt *lperm)
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
#define __FUNCT__ "MatColoringApply_JP"
PETSC_EXTERN PetscErrorCode MatColoringApply_JP(MatColoring mc,ISColoring *iscoloring)
{
  Mat            G=mc->graph,Gd,Go;
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)G->data;
  PetscErrorCode ierr;
  PetscBool      isMPIAIJ,isSEQAIJ;
  PetscInt       idx,i,j,ms,me,mn,msize,curncols,maxcolors,finalcolor,finalcolor_global;
  PetscInt       nr,nr_global;
  Vec            wtvec,colvec;
  Vec            owtvec,ocolvec;
  VecScatter     oscatter;
  PetscScalar    *wtarray,*owtarray,*colarray,*ocolarray;
  const PetscInt *cidx;
  PetscInt       ncols;
  PetscBool      isIn;
  PetscReal      *weights;
  PetscInt       *perm;
  ISColoringValue *colors;
  PetscBool       *mask;

  PetscFunctionBegin;
  finalcolor=0;
  ierr = PetscObjectTypeCompare((PetscObject)G,MATMPIAIJ,&isMPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)G,MATSEQAIJ,&isSEQAIJ);CHKERRQ(ierr);
  if (isMPIAIJ) {
    Gd = aij->A;
    Go = aij->B;
    owtvec = aij->lvec;
    oscatter = aij->Mvctx;
  } else if (isSEQAIJ) {
    Gd = G;
    Go = NULL;
    ocolvec = NULL;
    owtvec = NULL;
    oscatter = NULL;
  } else {
    SETERRQ(PetscObjectComm((PetscObject)mc),PETSC_ERR_ARG_WRONGSTATE,"Only support for AIJ matrices");
  }
  ierr = MatGetOwnershipRange(G,&ms,&me);CHKERRQ(ierr);
  ierr = MatGetSize(G,&msize,NULL);CHKERRQ(ierr);
  mn = me-ms;

  ierr = PetscMalloc1(mn,&perm);CHKERRQ(ierr);
  ierr = PetscMalloc1(mn,&weights);CHKERRQ(ierr);
  ierr = PetscMalloc1(mn,&colors);CHKERRQ(ierr);
  /* create weights and permutation */
  ierr = JPCreateWeights_Private(mc,weights,perm);CHKERRQ(ierr);
  ierr = MatGetVecs(G,NULL,&wtvec);CHKERRQ(ierr);
  ierr = VecDuplicate(wtvec,&colvec);CHKERRQ(ierr);
  if (owtvec) {
    ierr = VecDuplicate(owtvec,&ocolvec);CHKERRQ(ierr);
  }
  ierr = VecGetArray(wtvec,&wtarray);CHKERRQ(ierr);
  ierr = VecGetArray(colvec,&colarray);CHKERRQ(ierr);
  for (i=0;i<mn;i++) {
    colors[i] = IS_COLORING_MAX;
    colarray[i] = IS_COLORING_MAX;
    wtarray[i] = weights[i];
  }
  ierr = VecRestoreArray(wtvec,&wtarray);CHKERRQ(ierr);
  ierr = VecRestoreArray(colvec,&colarray);CHKERRQ(ierr);
  nr_global=0;
  nr=0;
 /* maxcolors = max(degree) + 1 or the user-set maximum colors */
  if (mc->maxcolors) {
    maxcolors=mc->maxcolors+1;
  } else {
    maxcolors=0;
    for (i=0;i<mn;i++) {
      ierr = MatGetRow(Gd,i,&ncols,NULL,NULL);CHKERRQ(ierr);
      curncols=ncols;
      ierr = MatRestoreRow(Gd,i,&ncols,NULL,NULL);CHKERRQ(ierr);
      if (Go) {
        ierr = MatGetRow(Go,i,&ncols,NULL,NULL);CHKERRQ(ierr);
        curncols+=ncols;
        ierr = MatRestoreRow(Go,i,&ncols,NULL,NULL);CHKERRQ(ierr);
      }
      if (curncols > maxcolors) maxcolors = curncols;
    }
    maxcolors++;
  }
  ierr = PetscMalloc1(maxcolors,&mask);CHKERRQ(ierr);

  while (nr_global < msize) {
    nr_global=0;
    /* transfer weights over by the scatter */
    if (oscatter) {
      ierr = VecScatterBegin(oscatter,wtvec,owtvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(oscatter,wtvec,owtvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterBegin(oscatter,colvec,ocolvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(oscatter,colvec,ocolvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    }
    /* for each local row, check if it's the top weight in its neighbors and assign it the lowest possible color if it is */
    ierr = VecGetArray(wtvec,&wtarray);CHKERRQ(ierr);
    ierr = VecGetArray(colvec,&colarray);CHKERRQ(ierr);
    if (owtvec) {
      ierr = VecGetArray(owtvec,&owtarray);CHKERRQ(ierr);
      ierr = VecGetArray(ocolvec,&ocolarray);CHKERRQ(ierr);
    }
    for (i=0;i<mn;i++) {
      idx = perm[mn-i-1];
      if (PetscRealPart(wtarray[idx]) > 0.) {
        for (j=0;j<maxcolors;j++) {
          mask[j]=PETSC_FALSE;
        }
        isIn = PETSC_TRUE;
        ierr = MatGetRow(Gd,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) {
          if (PetscRealPart(colarray[cidx[j]]) != IS_COLORING_MAX) {
            mask[(PetscInt)PetscRealPart(colarray[cidx[j]])] = PETSC_TRUE;
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
            if (PetscRealPart(ocolarray[cidx[j]]) != IS_COLORING_MAX) {
              mask[(PetscInt)PetscRealPart(ocolarray[cidx[j]])] = PETSC_TRUE;
            }
            if (PetscRealPart(owtarray[cidx[j]]) > PetscRealPart(wtarray[idx])) {
              isIn = PETSC_FALSE;
              break;
            }
          }
          ierr = MatRestoreRow(Go,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        }
        if (isIn) {
          /* you've filled the mask; set it to the lowest color */
          for (j=0;j<maxcolors;j++) {
            if (!mask[j]) {
              break;
            }
          }
          colarray[idx]=j;
          colors[idx]=j;
          if (j>finalcolor) finalcolor=j;
          nr++;
          wtarray[idx]=0.;
        }
      }
    }
    ierr = VecRestoreArray(wtvec,&wtarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(colvec,&colarray);CHKERRQ(ierr);
    if (owtvec) {
      ierr = VecRestoreArray(owtvec,&owtarray);CHKERRQ(ierr);
      ierr = VecRestoreArray(ocolvec,&ocolarray);CHKERRQ(ierr);
    }
    ierr = MPI_Allreduce(&nr,&nr_global,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
    finalcolor_global=0;
    ierr = MPI_Allreduce(&finalcolor,&finalcolor_global,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
  }
  ierr = ISColoringCreate(PetscObjectComm((PetscObject)mc),finalcolor_global+1,mn,colors,iscoloring);CHKERRQ(ierr);
  ierr = VecDestroy(&wtvec);CHKERRQ(ierr);
  ierr = VecDestroy(&colvec);CHKERRQ(ierr);
  if (ocolvec) {ierr = VecDestroy(&ocolvec);CHKERRQ(ierr);}
  ierr = PetscFree(mask);CHKERRQ(ierr);
  ierr = PetscFree(weights);CHKERRQ(ierr);
  ierr = PetscFree(perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringCreate_JP"
/*MC
  MATCOLORINGJP - Parallel Jones-Plassmann Coloring

   Level: beginner

   Notes: This method uses a parallel Luby-style coloring with with weights to choose an independent set of processor
   boundary vertices at each stage that may be assigned colors independently.

   References:
   M. Jones and P. Plassmann, “A parallel graph coloring heuristic,” SIAM Journal on Scientific Computing, vol. 14, no. 3,
   pp. 654–669, 1993.

.seealso: MatColoringCreate(), MatColoring, MatColoringSetType()
M*/
PETSC_EXTERN PetscErrorCode MatColoringCreate_JP(MatColoring mc)
{
  PetscFunctionBegin;
  mc->ops->apply          = MatColoringApply_JP;
  mc->ops->graph          = MatColoringGetGraph_Private;
  mc->ops->view           = NULL;
  mc->ops->destroy        = NULL;
  mc->ops->setfromoptions = NULL;
  PetscFunctionReturn(0);
}

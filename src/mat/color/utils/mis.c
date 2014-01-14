#include <petsc-private/matimpl.h>      /*I "petscmat.h"  I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

PETSC_EXTERN PetscErrorCode MatColoringComputeMISDistanceOne_Private(MatColoring mc,PetscReal *weights,IS *mis)
{
  Mat            M=mc->mat,Md,Mo;
  PetscInt       dist=mc->dist;
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)M->data;
  PetscErrorCode ierr;
  PetscBool      isMPIAIJ,isSEQAIJ;
  PetscInt       idx,i,j,ms,me,mn,*lperm,msize;
  PetscInt       nr,nr_global,nr_global_old;
  Vec            ovec,wtvec;
  VecScatter     oscatter;
  PetscScalar    *wtarray,*owtarray;
  const PetscInt *cidx;
  PetscInt       ncols,nmis=0;
  PetscBool      isIn;
  PetscInt       *isarray;
  PetscInt       *inarray;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)M,MATMPIAIJ,&isMPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)M,MATSEQAIJ,&isSEQAIJ);CHKERRQ(ierr);
  if (isMPIAIJ) {
    Md = aij->A;
    Mo = aij->B;
    ovec = aij->lvec;
    oscatter = aij->Mvctx;
  } else if (isSEQAIJ) {
    Md = M;
    Mo = NULL;
    ovec = NULL;
    oscatter = NULL;
  } else {
    SETERRQ(PetscObjectComm((PetscObject)mc),PETSC_ERR_ARG_WRONGSTATE,"Only support for AIJ matrices");
  }
  if (dist != 1) {
    SETERRQ(PetscObjectComm((PetscObject)mc),PETSC_ERR_ARG_WRONGSTATE,"Only distance one MIS supported at this time");
  }
  /* distance one coloring -- super simple, just do it greedily based upon locally-known weights */
  ierr = MatGetOwnershipRange(M,&ms,&me);CHKERRQ(ierr);
  ierr = MatGetVecs(M,NULL,&wtvec);CHKERRQ(ierr);
  ierr = MatGetSize(M,&msize,NULL);CHKERRQ(ierr);
  mn = me-ms;
  ierr = PetscMalloc1(mn,&lperm);CHKERRQ(ierr);
  ierr = PetscMalloc1(mn,&inarray);CHKERRQ(ierr);
  for (i=0;i<mn;i++) {
    lperm[i]=i;
    inarray[i]=0;
  }
  ierr = PetscSortRealWithPermutation(mn,weights,lperm);CHKERRQ(ierr);
  ierr = VecGetArray(wtvec,&wtarray);CHKERRQ(ierr);
  for (i=0;i<mn;i++) {
    wtarray[i] = weights[i];
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
    ierr = VecView(wtvec,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    /* for each local row, check if it's the top weight in its neighbors, and alter the weights in the weight vector accordingly as things are eliminated*/
    ierr = VecGetArray(wtvec,&wtarray);CHKERRQ(ierr);
    if (ovec) {
      ierr = VecGetArray(ovec,&owtarray);CHKERRQ(ierr);
    }
    for (i=0;i<mn;i++) {
      idx = lperm[mn-i-1];
      if (wtarray[idx] > 0. && inarray[idx]==0) {
        isIn = PETSC_TRUE;
        ierr = MatGetRow(Md,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) {
          if (PetscRealPart(wtarray[cidx[j]]) > PetscRealPart(wtarray[idx])) {
            isIn = PETSC_FALSE;
            break;
          }
        }
        ierr = MatRestoreRow(Md,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        if (Mo) {
          ierr = MatGetRow(Mo,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
          for (j=0;j<ncols;j++) {
            if (PetscRealPart(owtarray[cidx[j]]) > PetscRealPart(wtarray[idx])) {
              isIn = PETSC_FALSE;
              break;
            }
          }
          ierr = MatRestoreRow(Mo,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        }
        if (isIn) {
          inarray[idx]=1;
          nr++;
          /* add it by getting rid of the other ones */
          ierr = MatGetRow(Md,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
          for (j=0;j<ncols;j++) {
            /* invalidate local neighbors */
            if (PetscRealPart(wtarray[cidx[j]]) > 0. && idx != cidx[j]) {
              wtarray[cidx[j]] = 0.;
            }
          }
          ierr = MatRestoreRow(Md,idx,&ncols,&cidx,NULL);CHKERRQ(ierr);
        }
      }
    }
    ierr = VecRestoreArray(wtvec,&wtarray);CHKERRQ(ierr);
    if (ovec) {
      ierr = VecRestoreArray(ovec,&owtarray);CHKERRQ(ierr);
    }
    ierr = MPI_Allreduce(&nr,&nr_global,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Another round! %d %d\n",nr_global,nr_global_old);CHKERRQ(ierr);
  }
  ierr = VecGetArray(wtvec,&wtarray);CHKERRQ(ierr);
  for (i=0;i<mn;i++) {
    if (inarray[i]==1) {
      nmis++;
    }
  }
  ierr = PetscMalloc1(nmis,&isarray);CHKERRQ(ierr);
  idx=0;
  for (i=0;i<mn;i++) {
    if (inarray[i]==1) {
      isarray[idx] = i+ms;
      idx++;
    }
  }
  ierr = VecRestoreArray(wtvec,&wtarray);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)mc),nmis,isarray,PETSC_OWN_POINTER,mis);CHKERRQ(ierr);
  ierr = ISView(*mis,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscFree(lperm);CHKERRQ(ierr);
  ierr = PetscFree(inarray);CHKERRQ(ierr);
  ierr = VecDestroy(&wtvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

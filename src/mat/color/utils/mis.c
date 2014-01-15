#include <petsc-private/matimpl.h>      /*I "petscmat.h"  I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

#undef __FUNCT__
#define __FUNCT__ "MatColoringComputeMISDistanceOne_Private"
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

#undef __FUNCT__
#define __FUNCT__ "MatColoringComputeMISDistanceTwo_Private"
PETSC_EXTERN PetscErrorCode MatColoringComputeMISDistanceTwo_Private(MatColoring mc,PetscReal *weights,IS *mis)
{
  Mat            rM=mc->mat,rMd,rMo;
  Mat            cM,cMd,cMo;
  PetscInt       dist=mc->dist;
  Mat_MPIAIJ     *raij,*caij;
  PetscErrorCode ierr;
  PetscBool      isMPIAIJ,isSEQAIJ;
  PetscInt       idx,i,j,k,rms,rme,rmn,cms,cme,cmn,*cperm;
  PetscInt       nr,nr_global,nr_global_old;
  Vec            rovec,rwtvec;
  Vec            covec,cwtvec;
  VecScatter     roscatter;
  VecScatter     coscatter;
  PetscScalar    *rwtarray,*rowtarray;
  PetscScalar    *cwtarray,*cowtarray;
  const PetscInt *cidx,*ridx;
  PetscInt       ncols,nrows,nmis=0;
  PetscBool      isIn,isValid,*rtaken;
  PetscInt       *isarray;
  PetscInt       *inarray;

  PetscFunctionBegin;
  /* create the transpose if it's not made yet */
  if (!mc->matt) {
    ierr = MatTranspose(mc->mat,MAT_INITIAL_MATRIX,&mc->matt);CHKERRQ(ierr);
  }
  cM = mc->matt;
  /* get the communication structures for both */
  ierr = PetscObjectTypeCompare((PetscObject)cM,MATMPIAIJ,&isMPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)cM,MATSEQAIJ,&isSEQAIJ);CHKERRQ(ierr);
  raij=(Mat_MPIAIJ*)rM->data;
  caij=(Mat_MPIAIJ*)cM->data;
  if (isMPIAIJ) {
    rMd = raij->A;
    rMo = raij->B;
    covec = raij->lvec;
    coscatter = raij->Mvctx;
    cMd = caij->A;
    cMo = caij->B;
    rovec = caij->lvec;
    roscatter = caij->Mvctx;
  } else if (isSEQAIJ) {
    rMd = rM;
    rMo = NULL;
    rovec = NULL;
    roscatter = NULL;
    cMd = cM;
    cMo = NULL;
    covec = NULL;
    coscatter = NULL;
  } else {
    SETERRQ(PetscObjectComm((PetscObject)mc),PETSC_ERR_ARG_WRONGSTATE,"Only support for AIJ matrices");
  }
  ierr = MatGetOwnershipRange(cM,&cms,&cme);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(rM,&rms,&rme);CHKERRQ(ierr);
  ierr = MatGetVecs(cM,&cwtvec,&rwtvec);CHKERRQ(ierr);
  cmn = cme-cms;
  rmn = rme-rms;
  ierr = PetscMalloc1(cmn,&cperm);CHKERRQ(ierr);
  ierr = PetscMalloc1(cmn,&inarray);CHKERRQ(ierr);
  ierr = PetscMalloc1(rmn,&rtaken);CHKERRQ(ierr);
  for (i=0;i<cmn;i++) {
    cperm[i]=i;
    inarray[i]=0;
  }
  ierr = PetscSortRealWithPermutation(cmn,weights,cperm);CHKERRQ(ierr);
  ierr = VecGetArray(cwtvec,&cwtarray);CHKERRQ(ierr);
  for (i=0;i<cmn;i++) {
    cwtarray[i] = weights[i];
  }
  ierr = VecRestoreArray(cwtvec,&cwtarray);CHKERRQ(ierr);
  nr_global=0;
  nr_global_old=-1;
  nr=0;
  while (nr_global != nr_global_old) {
    nr_global_old=nr_global;
    nr_global=0;
    /* transfer column weights to row weights by the scatter */
    if (coscatter) {
      ierr = VecScatterBegin(coscatter,cwtvec,covec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(coscatter,cwtvec,covec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    }

    /* zero the row weight vector */
    ierr = VecGetArray(rwtvec,&rwtarray);CHKERRQ(ierr);
    for (i=0;i<rmn;i++) {
      rwtarray[i]=0.;
      rtaken[i]=NULL;
    }
    /* find the maximum column weight in each row and set the row weight to that */
    ierr = VecGetArray(cwtvec,&cwtarray);CHKERRQ(ierr);
    if (covec) {
      ierr = VecGetArray(covec,&cowtarray);CHKERRQ(ierr);
    }
    for (i=0;i<rmn;i++) {
      ierr = MatGetRow(rMd,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
      for (j=0;j<ncols;j++) {
        if (PetscRealPart(rwtarray[i])<PetscRealPart(cwtarray[cidx[j]])) {
          rwtarray[i]=cwtarray[cidx[j]];
        }
      }
      ierr = MatRestoreRow(rMd,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
      if (rMo) {
        ierr = MatGetRow(rMo,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
        for (j=0;j<ncols;j++) {
          if (PetscRealPart(rwtarray[i])<PetscRealPart(cowtarray[cidx[j]])) {
            rwtarray[i]=cowtarray[cidx[j]];
          }
        }
        ierr = MatRestoreRow(rMo,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
      }
    }
    ierr = VecRestoreArray(cwtvec,&cwtarray);CHKERRQ(ierr);
    if (covec) {
      ierr = VecRestoreArray(covec,&cowtarray);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(rwtvec,&rwtarray);CHKERRQ(ierr);

    /* scatter the max row weights to the local and global vectors */
    if (roscatter) {
      ierr = VecScatterBegin(roscatter,rwtvec,rovec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(roscatter,rwtvec,rovec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    }

    ierr = VecView(rwtvec,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    /* for each local row, check if it's the top weight in its neighbors, and alter the weights in the weight vector accordingly as things are eliminated*/
    ierr = VecGetArray(cwtvec,&cwtarray);CHKERRQ(ierr);
    if (covec) {
      ierr = VecGetArray(covec,&cowtarray);CHKERRQ(ierr);
    }
    ierr = VecGetArray(rwtvec,&rwtarray);CHKERRQ(ierr);
    if (rovec) {
      ierr = VecGetArray(rovec,&rowtarray);CHKERRQ(ierr);
    }
    /* for each local column, check to see if it's still available to be added, and if it's got the top weight in all its rows -- then recompute local row weights for neighbors */
    for (i=0;i<cmn;i++) {
      idx = cperm[cmn-i-1];
      if (cwtarray[idx] > 0. && inarray[idx]==0) {
        isIn = PETSC_TRUE;
        isValid = PETSC_TRUE;
        ierr = MatGetRow(cMd,idx,&nrows,&ridx,NULL);CHKERRQ(ierr);
        for (j=0;j<nrows;j++) {
          /* if this row is already locally taken, the column may be counted out; trigger local recalculation of the rows of this column's maximum */
          if (rtaken[ridx[j]]) {
            isIn=PETSC_FALSE;
            isValid=PETSC_FALSE;
            cwtarray[idx] = 0.;
            break;
          }
          if (PetscRealPart(rwtarray[ridx[j]]) > PetscRealPart(cwtarray[idx])) {
            isIn = PETSC_FALSE;
            break;
          }
        }
        ierr = MatRestoreRow(cMd,idx,&nrows,&ridx,NULL);CHKERRQ(ierr);
        if (cMo) {
          ierr = MatGetRow(cMo,idx,&nrows,&ridx,NULL);CHKERRQ(ierr);
          for (j=0;j<nrows;j++) {
            if (PetscRealPart(rowtarray[ridx[j]]) > PetscRealPart(cwtarray[idx])) {
              isIn = PETSC_FALSE;
              break;
            }
          }
          ierr = MatRestoreRow(cMo,idx,&nrows,&ridx,NULL);CHKERRQ(ierr);
        }
        if (!isValid) {
          /* recalculate the row maxes for the local rows of this column */
          ierr = MatGetRow(cMd,idx,&nrows,&ridx,NULL);CHKERRQ(ierr);
          for (j=0;j<nrows;j++) {
            ierr = MatGetRow(rMd,ridx[j],&ncols,&cidx,NULL);CHKERRQ(ierr);
            for (k=0;k<ncols;k++) {
              if (PetscRealPart(rwtarray[ridx[j]])<PetscRealPart(cwtarray[cidx[k]])) {
                rwtarray[ridx[j]]=cwtarray[cidx[k]];
              }
            }
            ierr = MatRestoreRow(rMd,ridx[j],&ncols,&cidx,NULL);CHKERRQ(ierr);
            if (rMo) {
              ierr = MatGetRow(rMo,ridx[j],&ncols,&cidx,NULL);CHKERRQ(ierr);
              for (k=0;k<ncols;k++) {
                if (PetscRealPart(rwtarray[ridx[j]])<PetscRealPart(cowtarray[cidx[k]])) {
                  rwtarray[ridx[j]]=cowtarray[cidx[k]];
                }
              }
              ierr = MatRestoreRow(rMo,ridx[j],&ncols,&cidx,NULL);CHKERRQ(ierr);
            }
          }
          ierr = MatRestoreRow(cMd,i,&ncols,&cidx,NULL);CHKERRQ(ierr);
        }
        if (isIn) {
          inarray[idx]=1;
          nr++;
          /* add it by getting rid of the other ones */
          ierr = MatGetRow(cMd,idx,&nrows,&ridx,NULL);CHKERRQ(ierr);
          for (j=0;j<nrows;j++) {
            rtaken[ridx[j]] = PETSC_TRUE;
          }
          ierr = MatRestoreRow(rMd,idx,&nrows,&ridx,NULL);CHKERRQ(ierr);
        }
      }
    }
    ierr = VecRestoreArray(cwtvec,&cwtarray);CHKERRQ(ierr);
    if (covec) {
      ierr = VecRestoreArray(covec,&cowtarray);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(rwtvec,&rwtarray);CHKERRQ(ierr);
    if (rovec) {
      ierr = VecRestoreArray(rovec,&rowtarray);CHKERRQ(ierr);
    }
    ierr = MPI_Allreduce(&nr,&nr_global,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
  }
  ierr = VecGetArray(cwtvec,&cwtarray);CHKERRQ(ierr);
  for (i=0;i<cmn;i++) {
    if (inarray[i]==1) {
      nmis++;
    }
  }
  ierr = PetscMalloc1(nmis,&isarray);CHKERRQ(ierr);
  idx=0;
  for (i=0;i<cmn;i++) {
    if (inarray[i]==1) {
      isarray[idx] = i+cms;
      idx++;
    }
  }
  ierr = VecRestoreArray(cwtvec,&cwtarray);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)mc),nmis,isarray,PETSC_OWN_POINTER,mis);CHKERRQ(ierr);
  ierr = ISView(*mis,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscFree(cperm);CHKERRQ(ierr);
  ierr = PetscFree(inarray);CHKERRQ(ierr);
  ierr = PetscFree(rtaken);CHKERRQ(ierr);
  ierr = VecDestroy(&cwtvec);CHKERRQ(ierr);
  ierr = VecDestroy(&rwtvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringComputeMIS"
PETSC_EXTERN PetscErrorCode MatColoringComputeMIS(MatColoring mc,PetscReal *weights,IS *mis)
{
  PetscInt       dist=mc->dist;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (dist == 1) {
    ierr = MatColoringComputeMISDistanceOne_Private(mc,weights,mis);CHKERRQ(ierr);
  } else if (dist == 2) {
    ierr = MatColoringComputeMISDistanceTwo_Private(mc,weights,mis);CHKERRQ(ierr);
  } else {
    SETERRQ(PetscObjectComm((PetscObject)mc),PETSC_ERR_ARG_WRONGSTATE,"Only support for distance 1 and 2");
  }
  PetscFunctionReturn(0);
}

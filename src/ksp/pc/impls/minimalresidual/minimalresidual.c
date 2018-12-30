
/*
   Include files needed for the selective minimal residual preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners
*/

#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/

/*
   Private context (data structure) for the selective minimal residual preconditioner preconditioner.
*/
typedef struct {
  Mat premr;
  PetscInt nnz;
  MatScalar *diag;
  PetscInt initer;
  PetscInt nMRrows,*MRrows,firstindex;
  Vec      vdiag;
} PC_MinimalResidual;


static PetscErrorCode PCApply_MinimalResidual(PC pc,Vec x,Vec y)
{
  PC_MinimalResidual      *jac = (PC_MinimalResidual*)pc->data;
  PetscErrorCode          ierr;
  //Vec                     w;
  
  PetscFunctionBegin;  
  //ierr = VecDuplicate(x,&w);CHKERRQ(ierr);
  ierr = VecPointwiseMult(x,x,jac->vdiag);CHKERRQ(ierr);
  ierr = MatMult(jac->premr,x,y);CHKERRQ(ierr);
 // ierr = VecDestroy(&w); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCMinimalResidualSetLines_MinimalResidual(PC pc,PetscInt nMRrows,PetscInt firstindex,PetscInt *MRrows)
{
  PC_MinimalResidual *j = (PC_MinimalResidual*)pc->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  j->nMRrows = nMRrows;
  j->firstindex = firstindex;
  ierr = PetscMalloc1(nMRrows,&j->MRrows);CHKERRQ(ierr);
  ierr = PetscMemcpy(j->MRrows,MRrows,nMRrows*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCMinimalResidualGetLines_MinimalResidual(PC pc,PetscInt *nMRrows,PetscInt *firstindex,const PetscInt **MRrows)
{
  PC_MinimalResidual *j = (PC_MinimalResidual*)pc->data;

  PetscFunctionBegin;
  *nMRrows = j->nMRrows;
  *MRrows  = j->MRrows;
  *firstindex = j->firstindex;
  PetscFunctionReturn(0);
}

PetscErrorCode  PCMinimalResidualSetLines(PC pc, PetscInt nMRrows,PetscInt firstindex,PetscInt *MRrows)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCMinimalResidualSetLines_C",(PC,PetscInt,PetscInt,PetscInt*),(pc,nMRrows,firstindex,MRrows));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  PCMinimalResidualGetLines(PC pc,PetscInt *nMRrows,PetscInt *firstindex,const PetscInt **MRrows)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCMinimalResidualGetLines_C",(PC,PetscInt*,PetscInt*,PetscInt**),(pc,nMRrows,firstindex,MRrows));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode  PCMinimalResidualSetInnerIterations_MinimalResidual(PC pc,PetscInt Inneriter)
{
  PC_MinimalResidual *j = (PC_MinimalResidual*)pc->data;

  PetscFunctionBegin;
  j->initer = Inneriter;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCMinimalResidualGetInnerIterations_MinimalResidual(PC pc,PetscInt *Inneriter)
{
  PC_MinimalResidual *j = (PC_MinimalResidual*)pc->data;

  PetscFunctionBegin;
  *Inneriter = j->initer;
  PetscFunctionReturn(0);
}

PetscErrorCode  PCMinimalResidualSetInnerIterations(PC pc,PetscInt Inneriter)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCMinimalResidualSetInnerIterations_C",(PC,PetscInt),(pc,Inneriter));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  PCMinimalResidualGetInnerIterations(PC pc,PetscInt *Inneriter)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCMinimalResidualGetInnerIterations_C",(PC,PetscInt*),(pc,Inneriter));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCMinimalResidualSetNNZ_MinimalResidual(PC pc,PetscInt nnz)
{
  PC_MinimalResidual *j = (PC_MinimalResidual*)pc->data;
  
  PetscFunctionBegin;
  j->nnz = nnz;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCMinimalResidualGetNNZ_MinimalResidual(PC pc,PetscInt *nnz)
{
  PC_MinimalResidual *j = (PC_MinimalResidual*)pc->data;

  PetscFunctionBegin;
  *nnz = j->nnz;
  PetscFunctionReturn(0);
}

PetscErrorCode  PCMinimalResidualSetNNZ(PC pc,PetscInt nnz)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCMinimalResidualSetNNZ_C",(PC,PetscInt),(pc,nnz));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  PCMinimalResidualGetNNZ(PC pc,PetscInt *nnz)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCMinimalResidualGetNNZ_C",(PC,PetscInt*),(pc,nnz));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_MinimalResidual(PC pc)
{
  PC_MinimalResidual    *jac = (PC_MinimalResidual*)pc->data;
  PetscErrorCode ierr;
  Mat            Acopy;
  PetscInt       nnz = jac->nnz;
  PetscInt       firstindex = jac->firstindex;
  MatFactorError err;
  PetscInt       i,j,k,n,m,col,nsize = 0,nrlocal,nclocal,nrglobal,ncglobal,startrow,endrow,bs,row,vecstart,vecend,rm,l,qq;
  PetscInt       nblocks;
  const PetscInt *bsizes;
  MPI_Comm       comm;
  MatScalar      *ptodiag, aalpha = -1;
  PetscInt       nMRrows = jac->nMRrows;
  const PetscInt *MRrows = jac->MRrows;
  PetscScalar    *workrow;
  PetscInt       *nncols;
  const PetscInt *pcols;
  Vec            workvec_s,workvec_r,workvec_e,workvec_z,workvec_q;
  PetscScalar    inprod1,inprod2,inq;
  const PetscScalar *vecpart;
  PetscInt       *dnnz,*onnz;
  

  PetscFunctionBegin;
  ierr = MatDuplicate(pc->pmat,MAT_COPY_VALUES,&Acopy);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Acopy,&startrow,&endrow);CHKERRQ(ierr);
  ierr = MatCreateVecs(Acopy,&(jac->vdiag),0);CHKERRQ(ierr);
  //ierr = VecDuplicate(workvec_d,&workvec_r); CHKERRQ(ierr);
  ierr = MatGetRowMaxAbs(Acopy,jac->vdiag,NULL);CHKERRQ(ierr);
  //ierr = VecCopy(workvec_d,workvec_r);
  ierr = VecReciprocal(jac->vdiag);CHKERRQ(ierr);
  ierr = VecGetLocalSize(jac->vdiag,&n);CHKERRQ(ierr);
  ierr = VecGetArray(jac->vdiag,&workrow);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (workrow[i] == 0.0) {
      workrow[i]     = 1.0;
    }
  }
  ierr = VecRestoreArray(jac->vdiag,&workrow);CHKERRQ(ierr);
  ierr = MatDiagonalScale(Acopy,jac->vdiag,NULL);CHKERRQ(ierr);
  //ierr = VecDestroy(&workvec_r); CHKERRQ(ierr);

  ierr = MatGetVariableBlockSizes(pc->pmat,&nblocks,&bsizes);CHKERRQ(ierr);
  ierr = MatGetLocalSize(pc->pmat,&nrlocal,&nclocal);CHKERRQ(ierr);
  ierr = MatGetSize(pc->pmat,&nrglobal,&ncglobal);CHKERRQ(ierr);
  if (nrlocal && !nblocks) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call MatSetVariableBlockSizes() before using PCMINIMALRESIDUAL");
  if (!jac->diag) {
    for (i=0; i<nblocks; i++) nsize += bsizes[i]*bsizes[i];
    ierr = PetscMalloc1(nsize,&jac->diag);CHKERRQ(ierr);
  }
  ierr = MatInvertVariableBlockDiagonal(Acopy,nblocks,bsizes,jac->diag);CHKERRQ(ierr);
  ierr = MatFactorGetError(Acopy,&err);CHKERRQ(ierr);
  if (err) pc->failedreason = (PCFailedReason)err;
  ierr = PetscObjectGetComm(((PetscObject) (pc->pmat)),&comm);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrlocal,&dnnz);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrlocal,&onnz);CHKERRQ(ierr);

  j = firstindex + 1;
  k = 0;
  n = 0;
  m = 0;
  bs = bsizes[0];
  rm = MRrows[firstindex];
  l = endrow-startrow;
  qq = ((l+nnz)>ncglobal) ? (ncglobal-l):nnz;  
  for (i=startrow; i<endrow; i++)
  {
    if (n == bs)
    {
      m++;
      bs = bsizes[m];
      n = 0;
    }
    if (i==rm)
    {
      if (j<nMRrows)
      {
        rm = MRrows[j];
        j++;
      }      
      onnz[k] = qq;
      dnnz[k] = l;
    }
    else
    {
      onnz[k] = 0;
      dnnz[k] = bs;
    }
    k++;
    n++;
  }

  ierr = MatCreateAIJ(comm, nrlocal,nclocal,nrglobal,ncglobal,0,dnnz,0,onnz, &(jac->premr));CHKERRQ(ierr);
  ierr = MatSetVariableBlockSizes(jac->premr,nblocks,(PetscInt*)bsizes);CHKERRQ(ierr);
  //ierr = MatSetOption(jac->premr,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);CHKERRQ(ierr);
  //ierr = MatSetOption(jac->premr,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
  
  j = 0;
  ptodiag=jac->diag;
  bs = bsizes[0];
  n = 0;
  m = startrow;
  l = firstindex+1;
  rm = MRrows[firstindex];
  for (i=startrow; i<endrow; i++)
  {
    if (n == bs)
    {
      ptodiag += bs*bs;
      m += bs;
      j++;
      bs = bsizes[j];
      n = 0;
    }
    if (i==rm)
    {
      if (l<nMRrows)
      {
        rm = MRrows[l];
        l++;
      }      
      inq = 0;
      qq = 0;
      for (k=0; k<bs; k++)
      {
        col = k+m;
        ierr = MatSetValues(jac->premr,1,&i,1,&col,ptodiag+n+k*bs,INSERT_VALUES);CHKERRQ(ierr);
      }
      for (k=0; ((k<m) && (qq<nnz)); k++)
      {        
        ierr = MatSetValues(jac->premr,1,&i,1,&k,&inq,INSERT_VALUES);CHKERRQ(ierr);
        qq++;
      }
      for (k=(m + bs); ((k<ncglobal) && (qq<nnz)); k++)
      {        
        ierr = MatSetValues(jac->premr,1,&i,1,&k,&inq,INSERT_VALUES);CHKERRQ(ierr);
        qq++;
      }
    }
    else
    {
      for (k=0; k<bs; k++)
      {
        col = k+m;
        ierr = MatSetValues(jac->premr,1,&i,1,&col,ptodiag+n+k*bs,INSERT_VALUES);CHKERRQ(ierr);
      }
    }    
    n++;
  }

  ierr = MatAssemblyBegin(jac->premr, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac->premr, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateVecs(jac->premr,NULL,&workvec_s);CHKERRQ(ierr);
  ierr = VecDuplicate(workvec_s,&workvec_r);CHKERRQ(ierr);
  ierr = VecDuplicate(workvec_s,&workvec_e);CHKERRQ(ierr);
  ierr = VecDuplicate(workvec_s,&workvec_z);CHKERRQ(ierr);
  ierr = VecDuplicate(workvec_s,&workvec_q);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(workvec_s,&vecstart,&vecend);CHKERRQ(ierr);    
  col = vecend-vecstart;  
  ierr = PetscMalloc1(col, &nncols);CHKERRQ(ierr);
  k = 0;
  for (i=vecstart; i<vecend; i++){
    nncols[k] = i;
    k++;
  }

  for (j=0; j<nMRrows; j++)
  {
	  row = MRrows[j];
    ierr = VecZeroEntries(workvec_s);CHKERRQ(ierr);
    if ((row>=startrow) && (row<endrow))
    {      
      ierr = MatGetRow(jac->premr,row,&qq,&pcols,&vecpart);CHKERRQ(ierr);
      ierr = VecSetValues(workvec_s,qq,pcols,vecpart,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(jac->premr,row,&qq,&pcols,&vecpart);CHKERRQ(ierr);
    }   
    ierr = VecAssemblyBegin(workvec_s);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(workvec_s);CHKERRQ(ierr);

    ierr = VecGetArray(workvec_e,&workrow);CHKERRQ(ierr);
    k = 0;
    for (i=vecstart; i<vecend; i++)
    {
      workrow[k] = (i==row) ? 1 : 0;
      k++;
    }
    ierr = VecRestoreArray(workvec_e,&workrow);CHKERRQ(ierr);

    ierr = VecAssemblyBegin(workvec_e);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(workvec_e);CHKERRQ(ierr);
    for (i=0;i<jac->initer;i++)
    {
      ierr = MatMultTranspose(Acopy,workvec_s,workvec_z);CHKERRQ(ierr);
      ierr = VecAYPX(workvec_z,aalpha,workvec_e);CHKERRQ(ierr);
      ierr = MatMultTranspose(jac->premr,workvec_z,workvec_q);CHKERRQ(ierr);
      ierr = MatMultTranspose(Acopy,workvec_q,workvec_r);CHKERRQ(ierr);
      ierr = VecDot(workvec_r, workvec_r, &inprod2);CHKERRQ(ierr);
      if (inprod2)
      {
        ierr = VecDot(workvec_z, workvec_r, &inprod1);CHKERRQ(ierr);
        inq = inprod1/inprod2;
        ierr = VecAXPY(workvec_s,inq,workvec_q);CHKERRQ(ierr);
      }
      else
      {
        inq = 1;
        ierr = VecAXPY(workvec_s,inq,workvec_q);CHKERRQ(ierr);
        break;
      }     
    }    
    ierr = VecGetArrayRead(workvec_s,&vecpart);CHKERRQ(ierr);
    ierr = MatSetValues(jac->premr,1,&row,col,nncols,vecpart,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(workvec_s,&vecpart);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(jac->premr, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(jac->premr, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  //ierr = MatDiagonalScale(jac->premr,NULL,workvec_d); CHKERRQ(ierr);

  ierr = PetscFree(nncols);CHKERRQ(ierr);
  ierr = PetscFree(dnnz);CHKERRQ(ierr);
  ierr = PetscFree(onnz);CHKERRQ(ierr);
  ierr = VecDestroy(&workvec_s);CHKERRQ(ierr);
  ierr = VecDestroy(&workvec_r);CHKERRQ(ierr);
  ierr = VecDestroy(&workvec_e);CHKERRQ(ierr);
  ierr = VecDestroy(&workvec_z);CHKERRQ(ierr);
  ierr = VecDestroy(&workvec_q);CHKERRQ(ierr);
  //ierr = VecDestroy(&workvec_d); CHKERRQ(ierr);
  ierr = MatDestroy(&Acopy);CHKERRQ(ierr);
  pc->ops->apply = PCApply_MinimalResidual;
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
static PetscErrorCode PCDestroy_MinimalResidual(PC pc)
{
  PC_MinimalResidual    *jac = (PC_MinimalResidual*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(jac->diag);CHKERRQ(ierr);
  ierr = PetscFree(jac->MRrows);CHKERRQ(ierr);
  ierr = VecDestroy(&(jac->vdiag));CHKERRQ(ierr);
  ierr = MatDestroy(&(jac->premr));CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


PETSC_EXTERN PetscErrorCode PCCreate_MinimalResidual(PC pc)
{
  PC_MinimalResidual   *jac;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  ierr     = PetscNewLog(pc,&jac);CHKERRQ(ierr);
  pc->data = (void*)jac;

 
  jac->diag = NULL;
  jac->nnz = 1;
  jac->premr = NULL;
  jac->initer = 10;

  pc->ops->apply               = PCApply_MinimalResidual;
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_MinimalResidual;
  pc->ops->destroy             = PCDestroy_MinimalResidual;
  pc->ops->setfromoptions      = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCMinimalResidualGetInnerIterations_C",PCMinimalResidualGetInnerIterations_MinimalResidual);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCMinimalResidualSetInnerIterations_C",PCMinimalResidualSetInnerIterations_MinimalResidual);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCMinimalResidualGetNNZ_C",PCMinimalResidualGetNNZ_MinimalResidual);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCMinimalResidualSetNNZ_C",PCMinimalResidualSetNNZ_MinimalResidual);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCMinimalResidualGetLines_C",PCMinimalResidualGetLines_MinimalResidual);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCMinimalResidualSetLines_C",PCMinimalResidualSetLines_MinimalResidual);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}



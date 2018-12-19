
/*
   Include files needed for the variable size block PBJacobi preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners
*/

#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/

/*
   Private context (data structure) for the MinimalResidual preconditioner.
*/
typedef struct {
  Mat premr;
  PetscInt nnz;
  MatScalar *diag;
  PetscInt initer;
  Vec diagforjacobi;
} PC_MinimalResidual;


static PetscErrorCode PCApply_MinimalResidual(PC pc,Vec x,Vec y)
{
  PC_MinimalResidual      *jac = (PC_MinimalResidual*)pc->data;
  PetscErrorCode    ierr;
  Vec w;
  
  PetscFunctionBegin;
  ierr = VecDuplicate(x,&w);CHKERRQ(ierr);
  ierr = VecPointwiseMult(w,x,jac->diagforjacobi);
  ierr = MatMult(jac->premr,w,y);CHKERRQ(ierr);
  ierr = VecDestroy(&w);CHKERRQ(ierr);
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



/* -------------------------------------------------------------------------- */
static PetscErrorCode PCSetUp_MinimalResidual(PC pc)
{
  PC_MinimalResidual    *jac = (PC_MinimalResidual*)pc->data;
  PetscErrorCode ierr;
  Mat            A = pc->pmat;
  MatFactorError err;
  PetscInt       i,j,k,n,m,col,nsize = 0,nrlocal,nclocal,nrglobal,ncglobal,startrow,endrow,bs,row,vecstart,vecend;
  PetscInt       nblocks;
  const PetscInt *bsizes;
  MPI_Comm       comm;
  MatScalar      *ptodiag, aalpha = -1;
  PetscInt       nMRrows;
  const PetscInt *MRrows;
  PetscScalar    *workrow;
  PetscInt       *nncols;
  Vec            workvec_s,workvec_r,workvec_e,workvec_z,workvec_q;
  PetscScalar    inprod1,inprod2,inq;
  const PetscScalar *vecpart;

  PetscFunctionBegin;
  ierr = MatCreateVecs(pc->pmat,&jac->diagforjacobi,0);CHKERRQ(ierr);
  ierr = MatGetRowMaxAbs(pc->pmat,jac->diagforjacobi,NULL);CHKERRQ(ierr);
  ierr = VecReciprocal(jac->diagforjacobi);CHKERRQ(ierr);
  ierr = VecGetLocalSize(jac->diagforjacobi,&n);CHKERRQ(ierr);
  ierr = VecGetArray(jac->diagforjacobi,&workrow);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (workrow[i] == 0.0) {
      workrow[i]     = 1.0;
    }
  }
  ierr = VecRestoreArray(jac->diagforjacobi,&workrow);CHKERRQ(ierr);
  ierr = MatDiagonalScale(pc->pmat, jac->diagforjacobi,NULL);CHKERRQ(ierr);



  ierr = MatGetVariableBlockSizes(pc->pmat,&nblocks,&bsizes);CHKERRQ(ierr);
  ierr = MatGetLocalSize(pc->pmat,&nrlocal,&nclocal);CHKERRQ(ierr);
  ierr = MatGetSize(pc->pmat,&nrglobal,&ncglobal);CHKERRQ(ierr);
  if (nrlocal && !nblocks) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call MatSetVariableBlockSizes() before using PCMINIMALRESIDUAL");
  if (!jac->diag) {
    for (i=0; i<nblocks; i++) nsize += bsizes[i]*bsizes[i];
    ierr = PetscMalloc1(nsize,&jac->diag);CHKERRQ(ierr);
  }
  ierr = MatInvertVariableBlockDiagonal(A,nblocks,bsizes,jac->diag);CHKERRQ(ierr);
  ierr = MatFactorGetError(A,&err);CHKERRQ(ierr);
  if (err) pc->failedreason = (PCFailedReason)err;
  ierr = PetscObjectGetComm(((PetscObject) (pc->pmat)),&comm);CHKERRQ(ierr);
  ierr = MatDuplicate(pc->pmat,MAT_DO_NOT_COPY_VALUES,&(jac->premr));CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(pc->pmat,&startrow,&endrow);CHKERRQ(ierr);

  j = 0;
  ptodiag=jac->diag;
  bs = bsizes[0];
  n = 0;
  m = startrow;
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
    for (k=0; k<bs; k++)
    {
      col = k+m;
      ierr = MatSetValues(jac->premr,1,&i,1,&col,&(ptodiag[n+k*bs]),INSERT_VALUES);CHKERRQ(ierr);
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
  ierr = MatGetMRLine(pc->pmat,&nMRrows,&MRrows);CHKERRQ(ierr);
  ierr = PetscMalloc1(ncglobal, &workrow);CHKERRQ(ierr);
  ierr = PetscMalloc1(ncglobal, &nncols);CHKERRQ(ierr);
  for (i=0; i<ncglobal; i++) nncols[i] = i;
  for (j=0; j<nMRrows; j++)
  {
    row = MRrows[j];       //global rows in MRrows
    ierr = MatGetValues(jac->premr,1,&row,ncglobal,nncols,workrow);CHKERRQ(ierr);
    ierr = VecSetValues(workvec_s,ncglobal,nncols,workrow,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(workvec_s);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(workvec_s);CHKERRQ(ierr);
    for (i=0; i<ncglobal; i++)
    {
      workrow[i] = (i==j) ? 1:0;
    }
    ierr = VecSetValues(workvec_e,ncglobal,nncols,workrow, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(workvec_e);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(workvec_e);CHKERRQ(ierr);
    for (i=0;i<jac->initer;i++)
    {
      ierr = MatMultTranspose(pc->pmat,workvec_s,workvec_z);CHKERRQ(ierr);
      ierr = VecAYPX(workvec_z,aalpha,workvec_e);CHKERRQ(ierr);
      ierr = MatMultTranspose(jac->premr,workvec_z,workvec_q);CHKERRQ(ierr);
      ierr = MatMultTranspose(pc->pmat,workvec_q,workvec_r);CHKERRQ(ierr);
      ierr = VecDot(workvec_z, workvec_r, &inprod1);CHKERRQ(ierr);
      ierr = VecDot(workvec_r, workvec_r, &inprod2);CHKERRQ(ierr);
      inq = inprod1/inprod2;
      ierr = VecAXPY(workvec_s,inq,workvec_q);CHKERRQ(ierr);
    }
    ierr = VecGetOwnershipRange(workvec_s,&vecstart,&vecend);CHKERRQ(ierr);    
    col = vecend-vecstart;
    for (i=vecstart; i<vecend; i++) nncols[i] = i;
    ierr = VecGetArrayRead(workvec_s,&vecpart);CHKERRQ(ierr);
    ierr = MatSetValues(jac->premr,1,&row,col,nncols,vecpart,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(workvec_s,&vecpart);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(jac->premr, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	  ierr = MatAssemblyEnd(jac->premr, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  ierr = PetscFree(workrow);CHKERRQ(ierr);
  ierr = PetscFree(nncols);CHKERRQ(ierr);
  ierr = VecDestroy(&workvec_s);CHKERRQ(ierr);
  ierr = VecDestroy(&workvec_r);CHKERRQ(ierr);
  ierr = VecDestroy(&workvec_e);CHKERRQ(ierr);
  ierr = VecDestroy(&workvec_z);CHKERRQ(ierr);
  ierr = VecDestroy(&workvec_q);CHKERRQ(ierr);
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
  ierr = VecDestroy(&jac->diagforjacobi);CHKERRQ(ierr);
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



  PetscFunctionReturn(0);
}



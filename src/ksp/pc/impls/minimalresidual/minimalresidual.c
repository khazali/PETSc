
/*
   Include files needed for the variable size block PBJacobi preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners
*/

#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/

/*
   Private context (data structure) for the VPBJacobi preconditioner.
*/
typedef struct {
  Mat premr;
  PetscInt nnz;
  MatScalar *diag;
  PetscInt initer;
} PC_MinimalResidual;


static PetscErrorCode PCApply_MinimalResidual(PC pc,Vec x,Vec y)
{
  PC_MinimalResidual      *jac = (PC_MinimalResidual*)pc->data;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatMult(jac->premr,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCMinimalResidualSetSetInnerIterations_MinimalResidual(PC pc,PetscInt Inneriter)
{
  PC_MinimalResidual *j = (PC_MinimalResidual*)pc->data;

  PetscFunctionBegin;
  j->initer = Inneriter;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCMinimalResidualGetSetInnerIterations_MinimalResidual(PC pc,PetscInt *Inneriter)
{
  PC_MinimalResidual *j = (PC_MinimalResidual*)pc->data;

  PetscFunctionBegin;
  *Inneriter = j->initer;
  PetscFunctionReturn(0);
}

PetscErrorCode  PCMinimalResidualSetSetInnerIterations(PC pc,PetscInt Inneriter)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCMinimalResidualSetSetInnerIterations_C",(PC,PetscInt),(pc,Inneriter));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  PCMinimalResidualGetSetInnerIterations(PC pc,PetscInt *Inneriter)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  //ierr = PetscUseMethod(pc,"PCJacobiGetUseAbs_C",(PC,PetscBool*),(pc,flg));CHKERRQ(ierr);
  ierr = PetscTryMethod(pc,"PCMinimalResidualGetSetInnerIterations_C",(PC,PetscInt*),(pc,Inneriter));CHKERRQ(ierr);
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
  ierr = MatGetVariableBlockSizes(pc->pmat,&nblocks,&bsizes);CHKERRQ(ierr);
  ierr = MatGetLocalSize(pc->pmat,&nrlocal,&nclocal);CHKERRQ(ierr);
  ierr = MatGetSize(pc->pmat,&nrglobal,&ncglobal);CHKERRQ(ierr);
  if (nrlocal && !nblocks) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call MatSetVariableBlockSizes() before using PCVPBJACOBI");
  if (!jac->diag) {
    for (i=0; i<nblocks; i++) nsize += bsizes[i]*bsizes[i];
    ierr = PetscMalloc1(nsize,&jac->diag);CHKERRQ(ierr);
  }
  ierr = MatInvertVariableBlockDiagonal(A,nblocks,bsizes,jac->diag);CHKERRQ(ierr);
  ierr = MatFactorGetError(A,&err);CHKERRQ(ierr);
  if (err) pc->failedreason = (PCFailedReason)err;
  ierr = PetscObjectGetComm(((PetscObject) (jac->premr)),&comm);CHKERRQ(ierr);
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
    row = startrow+MRrows[j];       //local rows in MRrows
    ierr = MatGetValues(jac->premr,1,&row,ncglobal,nncols,workrow);CHKERRQ(ierr);
    ierr = VecSetValues(workvec_s,ncglobal,nncols,workrow, INSERT_VALUES);CHKERRQ(ierr);
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
      ierr = VecAYPX(workvec_s,inq,workvec_q);CHKERRQ(ierr);
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
  ierr = MatDestroy(&(jac->premr));CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
     PCVPBJACOBI - Variable size point block Jacobi preconditioner


   Notes:
    See PCJACOBI for point Jacobi preconditioning, PCPBJACOBI for fixed point block size, and PCBJACOBI for large size blocks

   This works for AIJ matrices

   Uses dense LU factorization with partial pivoting to invert the blocks; if a zero pivot
   is detected a PETSc error is generated.

   One must call MatSetVariableBlockSizes() to use this preconditioner
   Developer Notes:
    This should support the PCSetErrorIfFailure() flag set to PETSC_TRUE to allow
   the factorization to continue even after a zero pivot is found resulting in a Nan and hence
   terminating KSP with a KSP_DIVERGED_NANORIF allowing
   a nonlinear solver/ODE integrator to recover without stopping the program as currently happens.

   Perhaps should provide an option that allows generation of a valid preconditioner
   even if a block is singular as the PCJACOBI does.

   Level: beginner

  Concepts: variable point block Jacobi

.seealso:  MatSetVariableBlockSizes(), PCCreate(), PCSetType(), PCType (for list of available types), PC, PCJACOBI

M*/

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

  /*
     Initialize the pointers to vectors to ZERO; these will be used to store
     diagonal entries of the matrix for fast preconditioner application.
  */
  jac->diag = NULL;
  jac->nnz = 1;
  jac->premr = NULL;
  jac->initer = 3;

  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_MinimalResidual;
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_MinimalResidual;
  pc->ops->destroy             = PCDestroy_MinimalResidual;
  pc->ops->setfromoptions      = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  PetscFunctionReturn(0);
}



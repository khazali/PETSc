
/*
  Defines the basic matrix operations for the TAIJ  matrix storage format.
  This format is used to evaluate matrices of the form:

    [S \otimes I + T \otimes A]

  where
    S is a dense (p \times q) matrix
    T is a dense (p \times q) matrix
    A is an AIJ  (n \times n) matrix
    I is the identity matrix

  The resulting matrix is (np \times nq)

  We provide:
     MatMult()
     MatMultAdd()
     MatInvertBlockDiagonal()
  and
     MatCreateTAIJ(Mat,p,q,,Mat*)

  This single directory handles both the sequential and parallel codes
*/

#include <../src/mat/impls/taij/taij.h> /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <petsc-private/vecimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatTAIJGetAIJ"
/*@C
   MatTAIJGetAIJ - Get the AIJ matrix describing the blockwise action of the TAIJ matrix

   Not Collective, but if the TAIJ matrix is parallel, the AIJ matrix is also parallel

   Input Parameter:
.  A - the TAIJ matrix

   Output Parameter:
.  B - the AIJ matrix

   Level: advanced

   Notes: The reference count on the AIJ matrix is not increased so you should not destroy it.

.seealso: MatCreateTAIJ()
@*/
PetscErrorCode  MatTAIJGetAIJ(Mat A,Mat *B)
{
  PetscErrorCode ierr;
  PetscBool      ismpitaij,isseqtaij;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPITAIJ,&ismpitaij);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQTAIJ,&isseqtaij);CHKERRQ(ierr);
  if (ismpitaij) {
    Mat_MPITAIJ *b = (Mat_MPITAIJ*)A->data;

    *B = b->A;
  } else if (isseqtaij) {
    Mat_SeqTAIJ *b = (Mat_SeqTAIJ*)A->data;

    *B = b->AIJ;
  } else {
    *B = A;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTAIJGetS"
/*@C
   MatTAIJGetS - Get the S matrix describing the shift action of the TAIJ matrix

   Not Collective, the entire S is stored and returned independently on all processes

   Input Parameter:
.  A - the TAIJ matrix

   Output Parameter:
.  S - the S matrix, in form of a scalar array in column-major format

   Level: advanced

   Notes: The reference count on the S matrix is not increased so you should not destroy it.

.seealso: MatCreateTAIJ()
@*/
PetscErrorCode  MatTAIJGetS(Mat A,const PetscScalar **S)
{
  Mat_SeqTAIJ *b = (Mat_SeqTAIJ*)A->data;
  PetscFunctionBegin;
  *S = b->S;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTAIJGetT"
/*@C
   MatTAIJGetT - Get the T matrix describing the shift action of the TAIJ matrix

   Not Collective, the entire T is stored and returned independently on all processes

   Input Parameter:
.  A - the TAIJ matrix

   Output Parameter:
.  T - the T matrix, in form of a scalar array in column-major format

   Level: advanced

   Notes: The reference count on the T matrix is not increased so you should not destroy it.

.seealso: MatCreateTAIJ()
@*/
PetscErrorCode  MatTAIJGetT(Mat A,const PetscScalar **T)
{
  Mat_SeqTAIJ *b = (Mat_SeqTAIJ*)A->data;
  PetscFunctionBegin;
  *T = b->T;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqTAIJ"
PetscErrorCode MatDestroy_SeqTAIJ(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqTAIJ    *b = (Mat_SeqTAIJ*)A->data;

  PetscFunctionBegin;
  ierr = MatDestroy(&b->AIJ);CHKERRQ(ierr);
  ierr = PetscFree3(b->S,b->T,b->ibdiag);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetUp_TAIJ"
PetscErrorCode MatSetUp_TAIJ(Mat A)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Must use MatCreateTAIJ() to create TAIJ matrices");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_SeqTAIJ"
PetscErrorCode MatView_SeqTAIJ(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  Mat            B;

  PetscFunctionBegin;
  ierr = MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = MatView(B,viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_MPITAIJ"
PetscErrorCode MatView_MPITAIJ(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  Mat            B;

  PetscFunctionBegin;
  ierr = MatConvert(A,MATMPIAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = MatView(B,viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_MPITAIJ"
PetscErrorCode MatDestroy_MPITAIJ(Mat A)
{
  PetscErrorCode ierr;
  Mat_MPITAIJ    *b = (Mat_MPITAIJ*)A->data;

  PetscFunctionBegin;
  ierr = MatDestroy(&b->AIJ);CHKERRQ(ierr);
  ierr = MatDestroy(&b->OAIJ);CHKERRQ(ierr);
  ierr = MatDestroy(&b->A);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&b->ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&b->w);CHKERRQ(ierr);
  ierr = PetscFree3(b->S,b->T,b->ibdiag);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*MC
  MATTAIJ - MATTAIJ = "taij" - A matrix type to be used to evaluate matrices of the following form:

    [S \otimes I + T \otimes A]

  where
    S is a dense (p \times q) matrix
    T is a dense (p \times q) matrix
    A is an AIJ  (n \times n) matrix
    I is the identity matrix
  The resulting matrix is (np \times nq)
  
  The matrix type is based on MATSEQAIJ for a sequential matrix A, and MATMPIAIJ for a distributed matrix A. 
  S and T are always stored independently on all processes as a PetscScalar array in column-major format.

  Operations provided:
. MatMult
. MatMultAdd
. MatInvertBlockDiagonal

  Level: advanced

.seealso: MatTAIJGetAIJ(), MatTAIJGetS(), MatTAIJGetT(), MatCreateTAIJ()
M*/

#undef __FUNCT__
#define __FUNCT__ "MatCreate_TAIJ"
PETSC_EXTERN PetscErrorCode MatCreate_TAIJ(Mat A)
{
  PetscErrorCode ierr;
  Mat_MPITAIJ    *b;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr     = PetscNewLog(A,Mat_MPITAIJ,&b);CHKERRQ(ierr);
  A->data  = (void*)b;

  ierr = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);

  A->ops->setup = MatSetUp_TAIJ;

  b->w    = 0;
  ierr    = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscObjectChangeTypeName((PetscObject)A,MATSEQTAIJ);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectChangeTypeName((PetscObject)A,MATMPITAIJ);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TAIJMultAdd_Seq"
/* zz = yy + Axx */
PetscErrorCode TAIJMultAdd_Seq(Mat A,Vec *xx,Vec *yy,Vec *zz)
{
  Mat_SeqTAIJ       *b = (Mat_SeqTAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *s = b->S, *t = b->T;
  const PetscScalar *x,*v,*bx;
  PetscScalar       *y,*sums;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          n,i,jrow,j,l,p=b->p,q=b->q,k;

  PetscFunctionBegin;

  if (!yy) {
    ierr = VecSet(*zz,0.0);CHKERRQ(ierr); 
  } else {
    ierr = VecCopy(*yy,*zz);CHKERRQ(ierr);
  }

  ierr = VecGetArrayRead(*xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(*zz,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sums = y + p*i;
    bx   = x + q*i;
    for (j=0; j<q; j++) {
      for (k=0; k<p; k++) {
        sums[k] += s[k+j*p]*bx[j];
      }
    }
    for (j=0; j<n; j++) {
      for (k=0; k<p; k++) {
        for (l=0; l<q; l++) {
          sums[k] += v[jrow+j]*t[k+l*p]*x[q*idx[jrow+j]+l];
        }
      }
    }
  }

  ierr = PetscLogFlops((2.0*p*q-p)*m+2*p*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(*xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(*zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_SeqTAIJ_N"
PetscErrorCode MatMult_SeqTAIJ_N(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TAIJMultAdd_Seq(A,&xx,PETSC_NULL,&yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_SeqTAIJ_N"
PetscErrorCode MatMultAdd_SeqTAIJ_N(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TAIJMultAdd_Seq(A,&xx,&yy,&zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petsc-private/kernels/blockinvert.h>

#undef __FUNCT__
#define __FUNCT__ "MatInvertBlockDiagonal_SeqTAIJ_N"
PetscErrorCode MatInvertBlockDiagonal_SeqTAIJ_N(Mat A,const PetscScalar **values)
{
  Mat_SeqTAIJ       *b  = (Mat_SeqTAIJ*)A->data;
  Mat_SeqAIJ        *a  = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *s  = b->S;
  const PetscScalar *v  = a->a;
  const PetscInt     p  = b->p, q = b->q, m = b->AIJ->rmap->n, *idx = a->j, *ii = a->i;
  PetscErrorCode    ierr;
  PetscInt          i,j,*v_pivots,dof,dof2;
  PetscScalar       *diag,aval,*v_work;

  PetscFunctionBegin;
  if (p != q) {
    SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Block size must be square to calculate inverse.");
  } else {
    dof  = p;
    dof2 = dof*dof;
  }
  if (b->ibdiagvalid) {
    if (values) *values = b->ibdiag;
    PetscFunctionReturn(0);
  }
  if (!b->ibdiag) {
    ierr = PetscMalloc(dof2*m*sizeof(PetscScalar),&b->ibdiag);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)A,dof2*m*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  if (values) *values = b->ibdiag;
  diag = b->ibdiag;

  ierr = PetscMalloc2(dof,PetscScalar,&v_work,dof,PetscInt,&v_pivots);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    ierr = PetscMemcpy(diag,s,dof2*sizeof(PetscScalar));CHKERRQ(ierr);
    aval = 0;
    for (j=ii[i]; j<ii[i+1]; j++) if (idx[j] == i) aval = v[j];
    for (j=0; j<dof; j++) diag[j+dof*j] += aval;
    ierr = PetscKernel_A_gets_inverse_A(dof,diag,v_pivots,v_work);CHKERRQ(ierr);
    diag += dof2;
  }
  ierr = PetscFree2(v_work,v_pivots);CHKERRQ(ierr);

  b->ibdiagvalid = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*===================================================================================*/

#undef __FUNCT__
#define __FUNCT__ "TAIJMultAdd_MPI"
PetscErrorCode TAIJMultAdd_MPI(Mat A,Vec *xx,Vec *yy,Vec *zz)
{
  Mat_MPITAIJ    *b = (Mat_MPITAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!yy) {
    ierr = VecSet(*zz,0.0);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(*yy,*zz);CHKERRQ(ierr);
  }
  /* start the scatter */
  ierr = VecScatterBegin(b->ctx,*xx,b->w,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*b->AIJ->ops->multadd)(b->AIJ,*xx,*zz,*zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(b->ctx,*xx,b->w,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*b->OAIJ->ops->multadd)(b->OAIJ,b->w,*zz,*zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_MPITAIJ_dof"
PetscErrorCode MatMult_MPITAIJ_dof(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TAIJMultAdd_MPI(A,&xx,PETSC_NULL,&yy);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_MPITAIJ_dof"
PetscErrorCode MatMultAdd_MPITAIJ_dof(Mat A,Vec xx,Vec yy, Vec zz)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = TAIJMultAdd_MPI(A,&xx,&yy,&zz);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatInvertBlockDiagonal_MPITAIJ_dof"
PetscErrorCode MatInvertBlockDiagonal_MPITAIJ_dof(Mat A,const PetscScalar **values)
{
  Mat_MPITAIJ     *b = (Mat_MPITAIJ*)A->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = (*b->AIJ->ops->invertblockdiagonal)(b->AIJ,values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqTAIJ_SeqAIJ"
PETSC_EXTERN PetscErrorCode MatConvert_SeqTAIJ_SeqAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat_SeqTAIJ    *b   = (Mat_SeqTAIJ*)A->data;
  Mat            a    = b->AIJ,B;
  Mat_SeqAIJ     *aij = (Mat_SeqAIJ*)a->data;
  PetscErrorCode ierr;
  PetscInt       m,n,i,ncols,*ilen,nmax = 0,*icols,j,k,l,p=b->p,q=b->q;
  PetscInt       *cols,*srow,*scol;
  PetscScalar    *vals,*s,*t;

  PetscFunctionBegin;
  ierr = MatGetSize(a,&m,&n);CHKERRQ(ierr);
  ierr = PetscMalloc(p*m*sizeof(PetscInt),&ilen);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    nmax = PetscMax(nmax,aij->ilen[i]);
    for (j=0; j<p; j++) ilen[p*i+j] = aij->ilen[i]*q;
  }
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,m*p,n*q,0,ilen,&B);CHKERRQ(ierr);
  ierr = PetscFree(ilen);CHKERRQ(ierr);
  ierr = PetscMalloc5(q,PetscInt,&icols,p*q,PetscScalar,&s,
                      p,PetscInt,&srow,q,PetscInt,&scol,
                      p*q,PetscScalar,&t);CHKERRQ(ierr);

  for (i=0; i<m; i++) {
    for (j=0; j<p; j++) srow[j] = i*p+j;
    for (k=0; k<q; k++) scol[k] = i*q+k;
    for (j=0; j<p; j++) {
      for (k=0; k<q; k++) {
        s[j*q+k] = b->S[j+k*p];
      }
    }
    ierr = MatSetValues_SeqAIJ(B,p,srow,q,scol,s,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatGetRow_SeqAIJ(a,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    for (k=0; k<ncols; k++) {
      for (j=0; j<q; j++) icols[j] = q*cols[k]+j;
      for (j=0; j<p; j++) {
        for (l=0; l<q; l++) {
          t[j*q+l] = b->T[j+l*p]*vals[k];
        }
      }
      ierr = MatSetValues_SeqAIJ(B,p,srow,q,icols,t,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow_SeqAIJ(a,i,&ncols,&cols,&vals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree5(icols,s,srow,scol,t);CHKERRQ(ierr);

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(A,B);CHKERRQ(ierr);
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}

#if 0

#include <../src/mat/impls/aij/mpi/mpiaij.h>

#undef __FUNCT__
#define __FUNCT__ "MatConvert_MPITAIJ_MPIAIJ"
PETSC_EXTERN PetscErrorCode MatConvert_MPITAIJ_MPIAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat_MPITAIJ    *taij   = (Mat_MPITAIJ*)A->data;
  Mat            MatAIJ  = ((Mat_SeqTAIJ*)taij->AIJ->data)->AIJ,B;
  Mat            MatOAIJ = ((Mat_SeqTAIJ*)taij->OAIJ->data)->AIJ;
  Mat_SeqAIJ     *AIJ    = (Mat_SeqAIJ*) MatAIJ->data;
  Mat_SeqAIJ     *OAIJ   =(Mat_SeqAIJ*) MatOAIJ->data;
  Mat_MPIAIJ     *mpiaij = (Mat_MPIAIJ*) taij->A->data;
  PetscInt       dof     = taij->dof,i,j,*dnz = NULL,*onz = NULL,nmax = 0,onmax = 0;
  PetscInt       *oicols = NULL,*icols = NULL,ncols,*cols = NULL,oncols,*ocols = NULL;
  PetscInt       rstart,cstart,*garray,ii,k;
  PetscErrorCode ierr;
  PetscScalar    *vals,*ovals;

  PetscFunctionBegin;
  ierr = PetscMalloc2(A->rmap->n,PetscInt,&dnz,A->rmap->n,PetscInt,&onz);CHKERRQ(ierr);
  for (i=0; i<A->rmap->n/dof; i++) {
    nmax  = PetscMax(nmax,AIJ->ilen[i]);
    onmax = PetscMax(onmax,OAIJ->ilen[i]);
    for (j=0; j<dof; j++) {
      dnz[dof*i+j] = AIJ->ilen[i];
      onz[dof*i+j] = OAIJ->ilen[i];
    }
  }
  ierr = MatCreateAIJ(PetscObjectComm((PetscObject)A),A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N,0,dnz,0,onz,&B);CHKERRQ(ierr);
  ierr = PetscFree2(dnz,onz);CHKERRQ(ierr);

  ierr   = PetscMalloc2(nmax,PetscInt,&icols,onmax,PetscInt,&oicols);CHKERRQ(ierr);
  rstart = dof*taij->A->rmap->rstart;
  cstart = dof*taij->A->cmap->rstart;
  garray = mpiaij->garray;

  ii = rstart;
  for (i=0; i<A->rmap->n/dof; i++) {
    ierr = MatGetRow_SeqAIJ(MatAIJ,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    ierr = MatGetRow_SeqAIJ(MatOAIJ,i,&oncols,&ocols,&ovals);CHKERRQ(ierr);
    for (j=0; j<dof; j++) {
      for (k=0; k<ncols; k++) {
        icols[k] = cstart + dof*cols[k]+j;
      }
      for (k=0; k<oncols; k++) {
        oicols[k] = dof*garray[ocols[k]]+j;
      }
      ierr = MatSetValues_MPIAIJ(B,1,&ii,ncols,icols,vals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues_MPIAIJ(B,1,&ii,oncols,oicols,ovals,INSERT_VALUES);CHKERRQ(ierr);
      ii++;
    }
    ierr = MatRestoreRow_SeqAIJ(MatAIJ,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    ierr = MatRestoreRow_SeqAIJ(MatOAIJ,i,&oncols,&ocols,&ovals);CHKERRQ(ierr);
  }
  ierr = PetscFree2(icols,oicols);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_REUSE_MATRIX) {
    PetscInt refct = ((PetscObject)A)->refct; /* save ((PetscObject)A)->refct */
    ((PetscObject)A)->refct = 1;

    ierr = MatHeaderReplace(A,B);CHKERRQ(ierr);

    ((PetscObject)A)->refct = refct; /* restore ((PetscObject)A)->refct */
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}

#endif

#undef __FUNCT__
#define __FUNCT__ "MatGetSubMatrix_TAIJ"
PetscErrorCode  MatGetSubMatrix_TAIJ(Mat mat,IS isrow,IS iscol,MatReuse cll,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            A;

  PetscFunctionBegin;
  ierr = MatConvert(mat,MATAIJ,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(A,isrow,iscol,cll,newmat);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "MatCreateTAIJ"
/*@C
  MatCreateTAIJ - Creates a matrix type to be used for matrices of the following form:

    [S \otimes I + T \otimes A]

  where
    S is a dense (p \times q) matrix
    T is a dense (p \times q) matrix
    A is an AIJ  (n \times n) matrix
    I is the identity matrix
  The resulting matrix is (np \times nq)
  
  The matrix type is based on MATSEQAIJ for a sequential matrix A, and MATMPIAIJ for a distributed matrix A. 
  S is always stored independently on all processes as a PetscScalar array in column-major format.
  
  Collective

  Input Parameters:
+ A - the AIJ matrix
+ S - the S matrix, stored as a PetscScalar array (column-major)
+ T - the T matrix, stored as a PetscScalar array (column-major)
- p - number of rows in S and T
- q - number of columns in S and T

  Output Parameter:
. taij - the new TAIJ matrix

  Operations provided:
+ MatMult
+ MatMultAdd
+ MatInvertBlockDiagonal
- MatView

  Level: advanced

.seealso: MatTAIJGetAIJ(), MATTAIJ
@*/
PetscErrorCode  MatCreateTAIJ(Mat A,PetscInt p,PetscInt q,const PetscScalar S[],const PetscScalar T[],Mat *taij)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n;
  Mat            B;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);

  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,p*A->rmap->n,q*A->cmap->n,p*A->rmap->N,q*A->cmap->N);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(B->rmap,p);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(B->cmap,q);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);

  B->assembled = PETSC_TRUE;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);

  if (size == 1) {
    Mat_SeqTAIJ *b;

    ierr = MatSetType(B,MATSEQTAIJ);CHKERRQ(ierr);

    B->ops->setup   = NULL;
    B->ops->destroy = MatDestroy_SeqTAIJ;
    B->ops->view    = MatView_SeqTAIJ;
    b               = (Mat_SeqTAIJ*)B->data;
    b->p            = p;
    b->q            = q;
    b->AIJ          = A;
    PetscMalloc2(p*q,PetscScalar,&b->S,p*q,PetscScalar,&b->T);CHKERRQ(ierr);
    if (S)  PetscMemcpy (b->S,S,p*q*sizeof(PetscScalar));
    else    PetscMemzero(b->S,p*q*sizeof(PetscScalar));
    if (T)  PetscMemcpy(b->T,T,p*q*sizeof(PetscScalar));
    else    PetscMemzero(b->T,p*q*sizeof(PetscScalar));

    B->ops->mult                = MatMult_SeqTAIJ_N;
    B->ops->multadd             = MatMultAdd_SeqTAIJ_N;
    B->ops->invertblockdiagonal = MatInvertBlockDiagonal_SeqTAIJ_N;

    ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqtaij_seqaij_C",MatConvert_SeqTAIJ_SeqAIJ);CHKERRQ(ierr);

  } else {
    Mat_MPIAIJ  *mpiaij = (Mat_MPIAIJ*)A->data;
    Mat_MPITAIJ *b;
    IS          from,to;
    Vec         gvec;

    ierr = MatSetType(B,MATMPITAIJ);CHKERRQ(ierr);

    B->ops->setup   = NULL;
    B->ops->destroy = MatDestroy_MPITAIJ;
    B->ops->view    = MatView_MPITAIJ;

    b      = (Mat_MPITAIJ*)B->data;
    b->p   = p;
    b->q   = q;
    b->A   = A;
    PetscMalloc2(p*q,PetscScalar,&b->S,p*q,PetscScalar,&b->T);CHKERRQ(ierr);
    if (S)  PetscMemcpy (b->S,S,p*q*sizeof(PetscScalar));
    else    PetscMemzero(b->S,p*q*sizeof(PetscScalar));
    if (T)  PetscMemcpy(b->T,T,p*q*sizeof(PetscScalar));
    else    PetscMemzero(b->T,p*q*sizeof(PetscScalar));

    ierr = MatCreateTAIJ(mpiaij->A,p,q,b->S,b->T,&b->AIJ);CHKERRQ(ierr); 
    ierr = MatCreateTAIJ(mpiaij->B,p,q,b->S,b->T,&b->OAIJ);CHKERRQ(ierr);

    ierr = VecGetSize(mpiaij->lvec,&n);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_SELF,&b->w);CHKERRQ(ierr);
    ierr = VecSetSizes(b->w,n*q,n*q);CHKERRQ(ierr);
    ierr = VecSetBlockSize(b->w,q);CHKERRQ(ierr);
    ierr = VecSetType(b->w,VECSEQ);CHKERRQ(ierr);

    /* create two temporary Index sets for build scatter gather */
    ierr = ISCreateBlock(PetscObjectComm((PetscObject)A),q,n,mpiaij->garray,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,n*q,0,1,&to);CHKERRQ(ierr);

    /* create temporary global vector to generate scatter context */
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)A),q,q*A->cmap->n,q*A->cmap->N,NULL,&gvec);CHKERRQ(ierr);

    /* generate the scatter context */
    ierr = VecScatterCreate(gvec,from,b->w,to,&b->ctx);CHKERRQ(ierr);

    ierr = ISDestroy(&from);CHKERRQ(ierr);
    ierr = ISDestroy(&to);CHKERRQ(ierr);
    ierr = VecDestroy(&gvec);CHKERRQ(ierr);

    B->ops->mult                = MatMult_MPITAIJ_dof;
    B->ops->multadd             = MatMultAdd_MPITAIJ_dof;
    B->ops->invertblockdiagonal = MatInvertBlockDiagonal_MPITAIJ_dof;

    //ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpitaij_mpiaij_C",MatConvert_MPITAIJ_MPIAIJ);CHKERRQ(ierr);
  }
  B->ops->getsubmatrix = MatGetSubMatrix_TAIJ;
  ierr  = MatSetUp(B);CHKERRQ(ierr);
  *taij = B;
  ierr  = MatViewFromOptions(B,NULL,"-mat_view");CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

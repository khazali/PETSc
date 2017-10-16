
/*
  Defines the basic matrix operations for the KAIJ  matrix storage format.
  This format is used to evaluate matrices of the form:

    [I \otimes S + A \otimes T]

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
     MatCreateKAIJ(Mat,PetscInt,PetscInt,const PetscScalar[],const PetscScalar[],Mat*)

  This single directory handles both the sequential and parallel codes
*/

#include <../src/mat/impls/kaij/kaij.h> /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <petsc/private/vecimpl.h>

/*@C
   MatKAIJGetAIJ - Get the AIJ matrix describing the blockwise action of the KAIJ matrix

   Not Collective, but if the KAIJ matrix is parallel, the AIJ matrix is also parallel

   Input Parameter:
.  A - the KAIJ matrix

   Output Parameter:
.  B - the AIJ matrix

   Level: advanced

   Notes: The reference count on the AIJ matrix is not increased so you should not destroy it.

.seealso: MatCreateKAIJ()
@*/
PetscErrorCode  MatKAIJGetAIJ(Mat A,Mat *B)
{
  PetscErrorCode ierr;
  PetscBool      ismpikaij,isseqkaij;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIKAIJ,&ismpikaij);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQKAIJ,&isseqkaij);CHKERRQ(ierr);
  if (ismpikaij) {
    Mat_MPIKAIJ *b = (Mat_MPIKAIJ*)A->data;

    *B = b->A;
  } else if (isseqkaij) {
    Mat_SeqKAIJ *b = (Mat_SeqKAIJ*)A->data;

    *B = b->AIJ;
  } else {
    *B = A;
  }
  PetscFunctionReturn(0);
}

/*@C
   MatKAIJGetS - Get the S matrix describing the shift action of the KAIJ matrix

   Not Collective, the entire S is stored and returned independently on all processes

   Input Parameter:
.  A - the KAIJ matrix

   Output Parameter:
.  S - the S matrix, in form of a scalar array in column-major format

   Level: advanced

   Notes: The reference count on the S matrix is not increased so you should not destroy it.

.seealso: MatCreateKAIJ()
@*/
PetscErrorCode  MatKAIJGetS(Mat A,const PetscScalar **S)
{
  Mat_SeqKAIJ *b = (Mat_SeqKAIJ*)A->data;
  PetscFunctionBegin;
  *S = b->S;
  PetscFunctionReturn(0);
}

/*@C
   MatKAIJGetT - Get the T matrix describing the shift action of the KAIJ matrix

   Not Collective, the entire T is stored and returned independently on all processes

   Input Parameter:
.  A - the KAIJ matrix

   Output Parameter:
.  T - the T matrix, in form of a scalar array in column-major format

   Level: advanced

   Notes: The reference count on the T matrix is not increased so you should not destroy it.

.seealso: MatCreateKAIJ()
@*/
PetscErrorCode  MatKAIJGetT(Mat A,const PetscScalar **T)
{
  Mat_SeqKAIJ *b = (Mat_SeqKAIJ*)A->data;
  PetscFunctionBegin;
  *T = b->T;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqKAIJ(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqKAIJ    *b = (Mat_SeqKAIJ*)A->data;

  PetscFunctionBegin;
  ierr = MatDestroy(&b->AIJ);CHKERRQ(ierr);
  ierr = PetscFree3(b->S,b->T,b->ibdiag);CHKERRQ(ierr);
  ierr = PetscFree5(b->sor.w,b->sor.y,b->sor.work,b->sor.t,b->sor.arr);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_KAIJ(Mat A)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Must use MatCreateKAIJ() to create KAIJ matrices");
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_SeqKAIJ(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  Mat            B;

  PetscFunctionBegin;
  ierr = MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = MatView(B,viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_MPIKAIJ(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  Mat            B;

  PetscFunctionBegin;
  ierr = MatConvert(A,MATMPIAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = MatView(B,viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIKAIJ(Mat A)
{
  PetscErrorCode ierr;
  Mat_MPIKAIJ    *b = (Mat_MPIKAIJ*)A->data;

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
  MATKAIJ - MATKAIJ = "kaij" - A matrix type to be used to evaluate matrices of the following form:

    [I \otimes S + A \otimes T]

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

.seealso: MatKAIJGetAIJ(), MatKAIJGetS(), MatKAIJGetT(), MatCreateKAIJ()
M*/

PETSC_EXTERN PetscErrorCode MatCreate_KAIJ(Mat A)
{
  PetscErrorCode ierr;
  Mat_MPIKAIJ    *b;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr     = PetscNewLog(A,&b);CHKERRQ(ierr);
  A->data  = (void*)b;

  ierr = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);

  A->ops->setup = MatSetUp_KAIJ;

  b->w    = 0;
  ierr    = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscObjectChangeTypeName((PetscObject)A,MATSEQKAIJ);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectChangeTypeName((PetscObject)A,MATMPIKAIJ);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

/* zz = yy + Axx */
PetscErrorCode KAIJMultAdd_Seq(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqKAIJ       *b = (Mat_SeqKAIJ*)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *s = b->S, *t = b->T;
  const PetscScalar *x,*v,*bx;
  PetscScalar       *y,*sums;
  PetscErrorCode    ierr;
  const PetscInt    m = b->AIJ->rmap->n,*idx,*ii;
  PetscInt          n,i,jrow,j,l,p=b->p,q=b->q,k;

  PetscFunctionBegin;

  if (!yy) {
    ierr = VecSet(zz,0.0);CHKERRQ(ierr); 
  } else {
    ierr = VecCopy(yy,zz);CHKERRQ(ierr);
  }
  if ((!s) && (!t) && (!b->isTI)) PetscFunctionReturn(0);

  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(zz,&y);CHKERRQ(ierr);
  idx  = a->j;
  v    = a->a;
  ii   = a->i;

  if (b->isTI) {
    for (i=0; i<m; i++) {
      jrow = ii[i];
      n    = ii[i+1] - jrow;
      sums = y + p*i;
      for (j=0; j<n; j++) {
        for (k=0; k<p; k++) {
          sums[k] += v[jrow+j]*x[q*idx[jrow+j]+k];
        }
      }
    }
  } else if (t) {
    for (i=0; i<m; i++) {
      jrow = ii[i];
      n    = ii[i+1] - jrow;
      sums = y + p*i;
      bx   = x + q*i;
      for (j=0; j<n; j++) {
        for (k=0; k<p; k++) {
          for (l=0; l<q; l++) {
            sums[k] += v[jrow+j]*t[k+l*p]*x[q*idx[jrow+j]+l];
          }
        }
      }
    }
  }
  if (s) {
    for (i=0; i<m; i++) {
      jrow = ii[i];
      n    = ii[i+1] - jrow;
      sums = y + p*i;
      bx   = x + q*i;
      if (i < b->AIJ->cmap->n) {
        for (j=0; j<q; j++) {
          for (k=0; k<p; k++) {
            sums[k] += s[k+j*p]*bx[j];
          }
        }
      }
    }
  }

  ierr = PetscLogFlops((2.0*p*q-p)*m+2*p*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(zz,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SeqKAIJ_N(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = KAIJMultAdd_Seq(A,xx,PETSC_NULL,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_SeqKAIJ_N(Mat A,Vec xx,Vec yy,Vec zz)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = KAIJMultAdd_Seq(A,xx,yy,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petsc/private/kernels/blockinvert.h>

PetscErrorCode MatInvertBlockDiagonal_SeqKAIJ_N(Mat A,const PetscScalar **values)
{
  Mat_SeqKAIJ       *b  = (Mat_SeqKAIJ*)A->data;
  Mat_SeqAIJ        *a  = (Mat_SeqAIJ*)b->AIJ->data;
  const PetscScalar *S  = b->S;
  const PetscScalar *T  = b->T;
  const PetscScalar *v  = a->a;
  const PetscInt     p  = b->p, q = b->q, m = b->AIJ->rmap->n, *idx = a->j, *ii = a->i;
  PetscErrorCode    ierr;
  PetscInt          i,j,*v_pivots,dof,dof2;
  PetscScalar       *diag,aval,*v_work;

  PetscFunctionBegin;
  if (p != q) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MATKAIJ: Block size must be square to calculate inverse.");
  if ((!S) && (!T) && (!b->isTI)) {
    SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MATKAIJ: Cannot invert a zero matrix.");
  }

  dof  = p;
  dof2 = dof*dof;

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

  ierr = PetscMalloc2(dof,&v_work,dof,&v_pivots);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    if (S) {
      ierr = PetscMemcpy(diag,S,dof2*sizeof(PetscScalar));CHKERRQ(ierr);
    } else {
      ierr = PetscMemzero(diag,dof2*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    if (b->isTI) {
      aval = 0;
      for (j=ii[i]; j<ii[i+1]; j++) if (idx[j] == i) aval = v[j];
      for (j=0; j<dof; j++) diag[j+dof*j] += aval;
    } else if (T) {
      aval = 0;
      for (j=ii[i]; j<ii[i+1]; j++) if (idx[j] == i) aval = v[j];
      for (j=0; j<dof2; j++) diag[j] += aval*T[j];
    }
    ierr = PetscKernel_A_gets_inverse_A(dof,diag,v_pivots,v_work,PETSC_FALSE,NULL);CHKERRQ(ierr);
    diag += dof2;
  }
  ierr = PetscFree2(v_work,v_pivots);CHKERRQ(ierr);

  b->ibdiagvalid = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonalBlock_MPIKAIJ(Mat A,Mat *B)
{
  Mat_MPIKAIJ *kaij = (Mat_MPIKAIJ*) A->data;

  PetscFunctionBegin;
  *B = kaij->AIJ;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSOR_SeqKAIJ(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  PetscErrorCode    ierr;
  Mat_SeqKAIJ       *kaij = (Mat_SeqKAIJ*) A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)kaij->AIJ->data;
  const PetscScalar *aa = a->a, *T = kaij->T, *v;
  const PetscInt    m  = kaij->AIJ->rmap->n, *ai=a->i, *aj=a->j, p = kaij->p, q = kaij->q, *diag, *vi;
  const PetscScalar *b, *xb, *idiag;
  PetscScalar       *x, *work, *workt, *w, *y, *arr, *t, *arrt;
  PetscInt          i, j, k, i2, bs, bs2, nz;


  PetscFunctionBegin;
  its = its*lits;
  if (flag & SOR_EISENSTAT) SETERRQ (PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for Eisenstat");
  if (its <= 0)             SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  if (fshift)               SETERRQ (PETSC_COMM_SELF,PETSC_ERR_SUP,"Sorry, no support for diagonal shift");
  if ((flag & SOR_APPLY_UPPER) || (flag & SOR_APPLY_LOWER)) 
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Sorry, no support for applying upper or lower triangular parts");
  if (p != q) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatSOR for KAIJ: Sorry, no support for non-square dense blocks");
  else        {bs = p; bs2 = bs*bs; }

  if (!m) PetscFunctionReturn(0);

  if (!kaij->ibdiagvalid) { ierr = MatInvertBlockDiagonal_SeqKAIJ_N(A,NULL);CHKERRQ(ierr); }
  idiag = kaij->ibdiag;
  diag  = a->diag;

  if (!kaij->sor.setup) {
    ierr = PetscMalloc5(bs,&kaij->sor.w,bs,&kaij->sor.y,m*bs,&kaij->sor.work,m*bs,&kaij->sor.t,m*bs2,&kaij->sor.arr);CHKERRQ(ierr);
    kaij->sor.setup = PETSC_TRUE;
  }
  y     = kaij->sor.y;
  w     = kaij->sor.w;
  work  = kaij->sor.work;
  t     = kaij->sor.t;
  arr   = kaij->sor.arr;

  ierr = VecGetArray(xx,&x);    CHKERRQ(ierr);
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);

  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      PetscKernel_w_gets_Ar_times_v(bs,bs,b,idiag,x);                            /* x[0:bs] <- D^{-1} b[0:bs] */
      ierr   =  PetscMemcpy(t,b,bs*sizeof(PetscScalar));CHKERRQ(ierr);
      i2     =  bs;
      idiag  += bs2;
      for (i=1; i<m; i++) {
        v  = aa + ai[i];
        vi = aj + ai[i];
        nz = diag[i] - ai[i];

        if (T) {                /* b - T (Arow * x) */
          for (k=0; k<bs; k++) w[k] = 0;
          for (j=0; j<nz; j++) {
            for (k=0; k<bs; k++) w[k] -= v[j] * x[vi[j]*bs+k];
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs,w,T,&t[i2]);
          for (k=0; k<bs; k++) t[i2+k] += b[i2+k];
        } else if (kaij->isTI) {
          for (k=0; k<bs; k++) t[i2+k] = b[i2+k];
          for (j=0; j<nz; j++) {
            for (k=0; k<bs; k++) t[i2+k] -= v[j] * x[vi[j]*bs+k];
          }
        } else {
          for (k=0; k<bs; k++) t[i2+k] = b[i2+k];
        }

        PetscKernel_w_gets_Ar_times_v(bs,bs,t+i2,idiag,y);
        for (j=0; j<bs; j++) x[i2+j] = omega * y[j];

        idiag += bs2;
        i2    += bs;
      }
      /* for logging purposes assume number of nonzero in lower half is 1/2 of total */
      ierr = PetscLogFlops(1.0*bs2*a->nz);CHKERRQ(ierr);
      xb = t;
    } else xb = b;
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      idiag = kaij->ibdiag+bs2*(m-1);
      i2    = bs * (m-1);
      ierr  = PetscMemcpy(w,xb+i2,bs*sizeof(PetscScalar));CHKERRQ(ierr);
      PetscKernel_w_gets_Ar_times_v(bs,bs,w,idiag,x+i2);
      i2    -= bs;
      idiag -= bs2;
      for (i=m-2; i>=0; i--) {
        v  = aa + diag[i] + 1 ;
        vi = aj + diag[i] + 1;
        nz = ai[i+1] - diag[i] - 1;

        if (T) {                /* FIXME: This branch untested */
          ierr = PetscMemcpy(w,xb+i2,bs*sizeof(PetscScalar));CHKERRQ(ierr);
          /* copy all rows of x that are needed into contiguous space */
          workt = work;
          for (j=0; j<nz; j++) {
            ierr   = PetscMemcpy(workt,x + bs*(*vi++),bs*sizeof(PetscScalar));CHKERRQ(ierr);
            workt += bs;
          }
          arrt = arr;
          for (j=0; j<nz; j++) {
            ierr  = PetscMemcpy(arrt,T,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
            for (k=0; k<bs2; k++) arrt[k] *= v[j];
            arrt += bs2;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
        } else if (kaij->isTI) {
          for (k=0; k<bs; k++) w[k] = t[i2+k];
          for (j=0; j<nz; j++) {
            for (k=0; k<bs; k++) w[k] -= v[j] * x[vi[j]*bs+k];
          }
        }

        PetscKernel_w_gets_Ar_times_v(bs,bs,w,idiag,y); /* RHS incorrect for omega != 1.0 */
        for (j=0; j<bs; j++) x[i2+j] = (1.0-omega) * x[i2+j] + omega * y[j];

        idiag -= bs2;
        i2    -= bs;
      }
      ierr = PetscLogFlops(1.0*bs2*(a->nz));CHKERRQ(ierr);
    }
    its--;
  }
  while (its--) {               /* FIXME: This branch not updated */
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      i2     =  0;
      idiag  = kaij->ibdiag;
      for (i=0; i<m; i++) {
        ierr = PetscMemcpy(w,b+i2,bs*sizeof(PetscScalar));CHKERRQ(ierr);

        v  = aa + ai[i];
        vi = aj + ai[i];
        nz = diag[i] - ai[i];
        workt = work;
        for (j=0; j<nz; j++) {
          ierr   = PetscMemcpy(workt,x + bs*(*vi++),bs*sizeof(PetscScalar));CHKERRQ(ierr);
          workt += bs;
        }
        arrt = arr;
        if (T) {
          for (j=0; j<nz; j++) {
            ierr  = PetscMemcpy(arrt,T,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
            for (k=0; k<bs2; k++) arrt[k] *= v[j];
            arrt += bs2;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
        } else if (kaij->isTI) {
          for (j=0; j<nz; j++) {
            ierr = PetscMemzero(arrt,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
            for (k=0; k<bs; k++) arrt[k+bs*k] = v[j];
            arrt += bs2;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
        }
        ierr = PetscMemcpy(t+i2,w,bs*sizeof(PetscScalar));CHKERRQ(ierr);

        v  = aa + diag[i] + 1;
        vi = aj + diag[i] + 1;
        nz = ai[i+1] - diag[i] - 1;
        workt = work;
        for (j=0; j<nz; j++) {
          ierr   = PetscMemcpy(workt,x + bs*(*vi++),bs*sizeof(PetscScalar));CHKERRQ(ierr);
          workt += bs;
        }
        arrt = arr;
        if (T) {
          for (j=0; j<nz; j++) {
            ierr  = PetscMemcpy(arrt,T,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
            for (k=0; k<bs2; k++) arrt[k] *= v[j];
            arrt += bs2;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
        } else if (kaij->isTI) {
          for (j=0; j<nz; j++) {
            ierr = PetscMemzero(arrt,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
            for (k=0; k<bs; k++) arrt[k+bs*k] = v[j];
            arrt += bs2;
          }
          PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
        }

        PetscKernel_w_gets_Ar_times_v(bs,bs,w,idiag,y);
        for (j=0; j<bs; j++) *(x+i2+j) = (1.0-omega) * *(x+i2+j) + omega * *(y+j);

        idiag += bs2;
        i2    += bs;
      }
      xb = t;
    }
    else xb = b;
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      idiag = kaij->ibdiag+bs2*(m-1);
      i2    = bs * (m-1);
      if (xb == b) {
        for (i=m-1; i>=0; i--) {
          ierr = PetscMemcpy(w,b+i2,bs*sizeof(PetscScalar));CHKERRQ(ierr);
  
          v  = aa + ai[i];
          vi = aj + ai[i];
          nz = diag[i] - ai[i];
          workt = work;
          for (j=0; j<nz; j++) {
            ierr   = PetscMemcpy(workt,x + bs*(*vi++),bs*sizeof(PetscScalar));CHKERRQ(ierr);
            workt += bs;
          }
          arrt = arr;
          if (T) {
            for (j=0; j<nz; j++) {
              ierr  = PetscMemcpy(arrt,T,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
              for (k=0; k<bs2; k++) arrt[k] *= v[j];
              arrt += bs2;
            }
            PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
          } else if (kaij->isTI) {
            for (j=0; j<nz; j++) {
              ierr = PetscMemzero(arrt,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
              for (k=0; k<bs; k++) arrt[k+bs*k] = v[j];
              arrt += bs2;
            }
            PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
          }

          v  = aa + diag[i] + 1;
          vi = aj + diag[i] + 1;
          nz = ai[i+1] - diag[i] - 1;
          workt = work;
          for (j=0; j<nz; j++) {
            ierr   = PetscMemcpy(workt,x + bs*(*vi++),bs*sizeof(PetscScalar));CHKERRQ(ierr);
            workt += bs;
          }
          arrt = arr;
          if (T) {
            for (j=0; j<nz; j++) {
              ierr  = PetscMemcpy(arrt,T,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
              for (k=0; k<bs2; k++) arrt[k] *= v[j];
              arrt += bs2;
            }
            PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
          } else if (kaij->isTI) {
            for (j=0; j<nz; j++) {
              ierr = PetscMemzero(arrt,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
              for (k=0; k<bs; k++) arrt[k+bs*k] = v[j];
              arrt += bs2;
            }
            PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
          }

          PetscKernel_w_gets_Ar_times_v(bs,bs,w,idiag,y);
          for (j=0; j<bs; j++) *(x+i2+j) = (1.0-omega) * *(x+i2+j) + omega * *(y+j);
        }
      } else {
        for (i=m-1; i>=0; i--) {
          ierr = PetscMemcpy(w,xb+i2,bs*sizeof(PetscScalar));CHKERRQ(ierr);
          v  = aa + diag[i] + 1;
          vi = aj + diag[i] + 1;
          nz = ai[i+1] - diag[i] - 1;
          workt = work;
          for (j=0; j<nz; j++) {
            ierr   = PetscMemcpy(workt,x + bs*(*vi++),bs*sizeof(PetscScalar));CHKERRQ(ierr);
            workt += bs;
          }
          arrt = arr;
          if (T) {
            for (j=0; j<nz; j++) {
              ierr  = PetscMemcpy(arrt,T,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
              for (k=0; k<bs2; k++) arrt[k] *= v[j];
              arrt += bs2;
            }
            PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
          } else if (kaij->isTI) {
            for (j=0; j<nz; j++) {
              ierr = PetscMemzero(arrt,bs2*sizeof(PetscScalar));CHKERRQ(ierr);
              for (k=0; k<bs; k++) arrt[k+bs*k] = v[j];
              arrt += bs2;
            }
            PetscKernel_w_gets_w_minus_Ar_times_v(bs,bs*nz,w,arr,work);
          }
          PetscKernel_w_gets_Ar_times_v(bs,bs,w,idiag,y);
          for (j=0; j<bs; j++) *(x+i2+j) = (1.0-omega) * *(x+i2+j) + omega * *(y+j);
        }
        idiag -= bs2;
        i2    -= bs;
      }
      ierr = PetscLogFlops(1.0*bs2*(a->nz));CHKERRQ(ierr);
    }
  }

  ierr = VecRestoreArray(xx,&x);    CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*===================================================================================*/

PetscErrorCode KAIJMultAdd_MPI(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIKAIJ    *b = (Mat_MPIKAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!yy) {
    ierr = VecSet(zz,0.0);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(yy,zz);CHKERRQ(ierr);
  }
  /* start the scatter */
  ierr = VecScatterBegin(b->ctx,xx,b->w,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*b->AIJ->ops->multadd)(b->AIJ,xx,zz,zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(b->ctx,xx,b->w,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*b->OAIJ->ops->multadd)(b->OAIJ,b->w,zz,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPIKAIJ_dof(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = KAIJMultAdd_MPI(A,xx,PETSC_NULL,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_MPIKAIJ_dof(Mat A,Vec xx,Vec yy, Vec zz)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = KAIJMultAdd_MPI(A,xx,yy,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatInvertBlockDiagonal_MPIKAIJ_dof(Mat A,const PetscScalar **values)
{
  Mat_MPIKAIJ     *b = (Mat_MPIKAIJ*)A->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = (*b->AIJ->ops->invertblockdiagonal)(b->AIJ,values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/

PetscErrorCode MatGetRow_SeqKAIJ(Mat A,PetscInt row,PetscInt *ncols,PetscInt **cols,PetscScalar **values)
{
  Mat_SeqKAIJ     *b    = (Mat_SeqKAIJ*) A->data;
  PetscErrorCode  ierr,diag;
  PetscInt        nzaij,nz,*colsaij,*idx,i,j,p=b->p,q=b->q,r=row/p,s=row%p,c;
  PetscScalar     *vaij,*v,*S=b->S,*T=b->T;

  PetscFunctionBegin;
  if (b->getrowactive) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Already active");
  b->getrowactive = PETSC_TRUE;
  if (row < 0 || row >= A->rmap->n) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %D out of range",row);

  if ((!S) && (!T) && (!b->isTI)) {
    if (ncols)    *ncols  = 0;
    if (cols)     *cols   = NULL;
    if (values)   *values = NULL;
    PetscFunctionReturn(0);
  }

  if (T || b->isTI) {
    ierr  = MatGetRow_SeqAIJ(b->AIJ,r,&nzaij,&colsaij,&vaij);CHKERRQ(ierr);
    diag  = PETSC_FALSE;
    c     = nzaij;
    for (i=0; i<nzaij; i++) {
      /* check if this row contains a diagonal entry */
      if (colsaij[i] == r) {
        diag = PETSC_TRUE;
        c = i;
      }
    }
  } else nzaij = c = 0;

  /* calculate size of row */
  nz = 0;
  if (S)            nz += q;
  if (T || b->isTI) nz += (diag && S ? (nzaij-1)*q : nzaij*q);

  if (cols || values) {
    ierr = PetscMalloc2(nz,&idx,nz,&v);CHKERRQ(ierr);
    if (b->isTI) {
      for (i=0; i<nzaij; i++) {
        for (j=0; j<q; j++) {
          idx[i*q+j] = colsaij[i]*q+j;
          v[i*q+j]   = (j==s ? vaij[i] : 0);
        }
      }
    } else if (T) {
      for (i=0; i<nzaij; i++) {
        for (j=0; j<q; j++) {
          idx[i*q+j] = colsaij[i]*q+j;
          v[i*q+j]   = vaij[i]*T[s+j*p];
        }
      }
    }
    if (S) {
      for (j=0; j<q; j++) {
        idx[c*q+j] = r*q+j; 
        v[c*q+j]  += S[s+j*p];
      }
    }
  }

  if (ncols)    *ncols  = nz;
  if (cols)     *cols   = idx;
  if (values)   *values = v;

  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreRow_SeqKAIJ(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFree2(*idx,*v);CHKERRQ(ierr);
  ((Mat_SeqKAIJ*)A->data)->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetRow_MPIKAIJ(Mat A,PetscInt row,PetscInt *ncols,PetscInt **cols,PetscScalar **values)
{
  Mat_MPIKAIJ     *b      = (Mat_MPIKAIJ*) A->data;
  Mat             MatAIJ  = ((Mat_SeqKAIJ*)b->AIJ->data)->AIJ;
  Mat             MatOAIJ = ((Mat_SeqKAIJ*)b->OAIJ->data)->AIJ;
  Mat             AIJ     = b->A;
  PetscErrorCode  ierr;
  const PetscInt  rstart=A->rmap->rstart,rend=A->rmap->rend,p=b->p,q=b->q,*garray;
  PetscInt        nz,*idx,ncolsaij,ncolsoaij,*colsaij,*colsoaij,r,s,c,i,j,lrow;
  PetscScalar     *v,*vals,*ovals,*S=b->S,*T=b->T;
  PetscBool       diag;

  PetscFunctionBegin;
  if (b->getrowactive) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Already active");
  b->getrowactive = PETSC_TRUE;
  if (row < rstart || row >= rend) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Only local rows");
  lrow = row - rstart;

  if ((!S) && (!T) && (!b->isTI)) {
    if (ncols)    *ncols  = 0;
    if (cols)     *cols   = NULL;
    if (values)   *values = NULL;
    PetscFunctionReturn(0);
  }

  r = lrow/p;
  s = lrow%p;

  if (T || b->isTI) {
    ierr = MatMPIAIJGetSeqAIJ(AIJ,NULL,NULL,&garray);
    ierr = MatGetRow_SeqAIJ(MatAIJ,lrow/p,&ncolsaij,&colsaij,&vals);CHKERRQ(ierr);
    ierr = MatGetRow_SeqAIJ(MatOAIJ,lrow/p,&ncolsoaij,&colsoaij,&ovals);CHKERRQ(ierr);

    diag  = PETSC_FALSE;
    c     = ncolsaij + ncolsoaij;
    for (i=0; i<ncolsaij; i++) {
      /* check if this row contains a diagonal entry */
      if (colsaij[i] == r) {
        diag = PETSC_TRUE;
        c = i;
      }
    }
  } else c = 0;

  /* calculate size of row */
  nz = 0;
  if (S)            nz += q;
  if (T || b->isTI) nz += (diag && S ? (ncolsaij+ncolsoaij-1)*q : (ncolsaij+ncolsoaij)*q);

  if (cols || values) {
    ierr = PetscMalloc2(nz,&idx,nz,&v);CHKERRQ(ierr);
    if (b->isTI) {
      for (i=0; i<ncolsaij; i++) {
        for (j=0; j<q; j++) {
          idx[i*q+j] = (colsaij[i]+rstart/p)*q+j;
          v[i*q+j]   = (j==s ? vals[i] : 0.0);
        }
      }
      for (i=0; i<ncolsoaij; i++) {
        for (j=0; j<q; j++) {
          idx[(i+ncolsaij)*q+j] = garray[colsoaij[i]]*q+j;
          v[(i+ncolsaij)*q+j]   = (j==s ? ovals[i]: 0.0);
        }
      }
    } else if (T) {
      for (i=0; i<ncolsaij; i++) {
        for (j=0; j<q; j++) {
          idx[i*q+j] = (colsaij[i]+rstart/p)*q+j;
          v[i*q+j]   = vals[i]*T[s+j*p];
        }
      }
      for (i=0; i<ncolsoaij; i++) {
        for (j=0; j<q; j++) {
          idx[(i+ncolsaij)*q+j] = garray[colsoaij[i]]*q+j;
          v[(i+ncolsaij)*q+j]   = ovals[i]*T[s+j*p];
        }
      }
    }
    if (S) {
      for (j=0; j<q; j++) {
        idx[c*q+j] = (r+rstart/p)*q+j; 
        v[c*q+j]  += S[s+j*p];
      }
    }
  }

  if (ncols)  *ncols  = nz;
  if (cols)   *cols   = idx;
  if (values) *values = v;
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreRow_MPIKAIJ(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFree2(*idx,*v);CHKERRQ(ierr);
  ((Mat_SeqKAIJ*)A->data)->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode  MatCreateSubMatrix_KAIJ(Mat mat,IS isrow,IS iscol,MatReuse cll,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            A;

  PetscFunctionBegin;
  ierr = MatConvert(mat,MATAIJ,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(A,isrow,iscol,cll,newmat);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------- */
/*@C
  MatCreateKAIJ - Creates a matrix type to be used for matrices of the following form:

    [I \otimes S + A \otimes T]

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
. S - the S matrix, stored as a PetscScalar array (column-major)
. T - the T matrix, stored as a PetscScalar array (column-major)
. p - number of rows in S and T
- q - number of columns in S and T

  Output Parameter:
. kaij - the new KAIJ matrix

  Operations provided:
+ MatMult
. MatMultAdd
. MatInvertBlockDiagonal
- MatView

  Level: advanced

.seealso: MatKAIJGetAIJ(), MATKAIJ
@*/
PetscErrorCode  MatCreateKAIJ(Mat A,PetscInt p,PetscInt q,const PetscScalar S[],const PetscScalar T[],Mat *kaij)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n,i,j;
  Mat            B;
  PetscBool      isTI = PETSC_FALSE;

  PetscFunctionBegin;

  /* check if T is an identity matrix */
  if (T && (p == q)) {
    isTI = PETSC_TRUE;
    for (i=0; i<p; i++) {
      for (j=0; j<q; j++) {
        if (i == j) {
          /* diagonal term must be 1 */
          if (T[i+j*p] != 1.0) isTI = PETSC_FALSE;
        } else {
          /* off-diagonal term must be 0 */
          if (T[i+j*p] != 0.0) isTI = PETSC_FALSE;
        }
      }
    }
  }

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
    Mat_SeqKAIJ *b;

    ierr = MatSetType(B,MATSEQKAIJ);CHKERRQ(ierr);

    B->ops->setup   = NULL;
    B->ops->destroy = MatDestroy_SeqKAIJ;
    B->ops->view    = MatView_SeqKAIJ;
    b               = (Mat_SeqKAIJ*)B->data;
    b->p            = p;
    b->q            = q;
    b->AIJ          = A;
    b->isTI         = isTI;
    if (S) {
      ierr = PetscMalloc(p*q*sizeof(PetscScalar),&b->S);CHKERRQ(ierr);
      ierr = PetscMemcpy(b->S,S,p*q*sizeof(PetscScalar));CHKERRQ(ierr);
    } else  b->S = NULL;
    if (T && (!isTI)) {
      ierr = PetscMalloc(p*q*sizeof(PetscScalar),&b->T);CHKERRQ(ierr);
      ierr = PetscMemcpy(b->T,T,p*q*sizeof(PetscScalar));CHKERRQ(ierr);
    } else b->T = NULL;

    B->ops->mult                = MatMult_SeqKAIJ_N;
    B->ops->multadd             = MatMultAdd_SeqKAIJ_N;
    B->ops->invertblockdiagonal = MatInvertBlockDiagonal_SeqKAIJ_N;
    B->ops->getrow              = MatGetRow_SeqKAIJ;
    B->ops->restorerow          = MatRestoreRow_SeqKAIJ;
    B->ops->sor                 = MatSOR_SeqKAIJ;

    /*
    This is commented out since MATKAIJ will use the basic MatConvert (which uses MatGetRow()).
    ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqkaij_seqaij_C",MatConvert_SeqKAIJ_SeqAIJ);CHKERRQ(ierr);
    */

  } else {
    Mat_MPIAIJ  *mpiaij = (Mat_MPIAIJ*)A->data;
    Mat_MPIKAIJ *b;
    IS          from,to;
    Vec         gvec;

    ierr = MatSetType(B,MATMPIKAIJ);CHKERRQ(ierr);

    B->ops->setup   = NULL;
    B->ops->destroy = MatDestroy_MPIKAIJ;
    B->ops->view    = MatView_MPIKAIJ;

    b       = (Mat_MPIKAIJ*)B->data;
    b->p    = p;
    b->q    = q;
    b->A    = A;
    b->isTI = isTI;
    if (S) {
      ierr = PetscMalloc(p*q*sizeof(PetscScalar),&b->S);CHKERRQ(ierr);
      ierr = PetscMemcpy(b->S,S,p*q*sizeof(PetscScalar));CHKERRQ(ierr);
    } else  b->S = NULL;
    if (T &&(!isTI)) {
      ierr = PetscMalloc(p*q*sizeof(PetscScalar),&b->T);CHKERRQ(ierr);
      ierr = PetscMemcpy(b->T,T,p*q*sizeof(PetscScalar));CHKERRQ(ierr);
    } else b->T = NULL;

    ierr = MatCreateKAIJ(mpiaij->A,p,q,S   ,T,&b->AIJ);CHKERRQ(ierr); 
    ierr = MatCreateKAIJ(mpiaij->B,p,q,NULL,T,&b->OAIJ);CHKERRQ(ierr);

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

    B->ops->mult                = MatMult_MPIKAIJ_dof;
    B->ops->multadd             = MatMultAdd_MPIKAIJ_dof;
    B->ops->invertblockdiagonal = MatInvertBlockDiagonal_MPIKAIJ_dof;
    B->ops->getrow              = MatGetRow_MPIKAIJ;
    B->ops->restorerow          = MatRestoreRow_MPIKAIJ;
    ierr = PetscObjectComposeFunction((PetscObject)B,"MatGetDiagonalBlock_C",MatGetDiagonalBlock_MPIKAIJ);CHKERRQ(ierr);
  }
  B->ops->createsubmatrix = MatCreateSubMatrix_KAIJ;
  B->assembled = PETSC_TRUE;
  ierr  = MatSetUp(B);CHKERRQ(ierr);
  *kaij = B;
  ierr  = MatViewFromOptions(B,NULL,"-mat_view");CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

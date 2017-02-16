#include <petscmat.h>
#include <petsc/private/matimpl.h>
#include <../src/mat/impls/elementlist/matelementlist.h>

/* This class implements a linear operator as a sum of element contibutions
   Element matrices are general, AIJ is supported to aggregate elements together */

static PetscErrorCode MatMultTranspose_Elementlist(Mat A, Vec x, Vec y)
{
  Mat_Elementlist     *elist = (Mat_Elementlist*)A->data;
  Mat_ElementlistLink next;
  const PetscScalar   *ax;
  PetscScalar         *ay;
  PetscInt            cc,rc;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecSet(elist->cv,0.0);CHKERRQ(ierr);
  ierr = VecSet(elist->rv,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(elist->rscctx,x,elist->rv,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(elist->rscctx,x,elist->rv,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  next = elist->root->next;
  ierr = VecGetArrayRead(elist->rv,&ax);CHKERRQ(ierr);
  ierr = VecGetArray(elist->cv,&ay);CHKERRQ(ierr);
  cc   = 0;
  rc   = 0;
  while (next) {
    ierr = VecPlaceArray(next->rv,ax+rc);CHKERRQ(ierr);
    ierr = VecPlaceArray(next->cv,ay+cc);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(next->elem,next->rv,next->cv,next->cv);CHKERRQ(ierr);
    ierr = VecResetArray(next->cv);CHKERRQ(ierr);
    ierr = VecResetArray(next->rv);CHKERRQ(ierr);
    cc  += next->elem->cmap->n; 
    rc  += next->elem->rmap->n; 
    next = next->next;
  }
  ierr = VecRestoreArray(elist->cv,&ay);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(elist->rv,&ax);CHKERRQ(ierr);
  ierr = VecScatterBegin(elist->cscctx,elist->cv,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(elist->cscctx,elist->cv,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_Elementlist(Mat A, Vec x, Vec y)
{
  Mat_Elementlist     *elist = (Mat_Elementlist*)A->data;
  Mat_ElementlistLink next;
  const PetscScalar   *ax;
  PetscScalar         *ay;
  PetscInt            cc,rc;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecSet(elist->cv,0.0);CHKERRQ(ierr);
  ierr = VecSet(elist->rv,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(elist->cscctx,x,elist->cv,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(elist->cscctx,x,elist->cv,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  next = elist->root->next;
  ierr = VecGetArrayRead(elist->cv,&ax);CHKERRQ(ierr);
  ierr = VecGetArray(elist->rv,&ay);CHKERRQ(ierr);
  cc   = 0;
  rc   = 0;
  while (next) {
    ierr = VecPlaceArray(next->cv,ax+cc);CHKERRQ(ierr);
    ierr = VecPlaceArray(next->rv,ay+rc);CHKERRQ(ierr);
    ierr = MatMultAdd(next->elem,next->cv,next->rv,next->rv);CHKERRQ(ierr);
    ierr = VecResetArray(next->cv);CHKERRQ(ierr);
    ierr = VecResetArray(next->rv);CHKERRQ(ierr);
    cc  += next->elem->cmap->n; 
    rc  += next->elem->rmap->n; 
    next = next->next;
  }
  ierr = VecRestoreArray(elist->rv,&ay);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(elist->cv,&ax);CHKERRQ(ierr);
  ierr = VecScatterBegin(elist->rscctx,elist->rv,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(elist->rscctx,elist->rv,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyBegin_Elementlist(Mat A, MatAssemblyType atype)
{
  PetscFunctionBegin;
  if (atype == MAT_FLUSH_ASSEMBLY) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Unsupported operation");
  /* here we can rearrange in subdomains */
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_Elementlist(Mat A, MatAssemblyType atype)
{
  Mat_Elementlist     *elist = (Mat_Elementlist*)A->data;
  Mat_ElementlistLink next;
  Vec                 rv,cv;
  IS                  ris,cis,rgis,cgis;
  PetscInt            *ridxs,*cidxs,*rp,*cp,cumr,cumc;
  PetscInt            tr,tc,M,N;
  PetscBool           rhasnull,chasnull;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (atype == MAT_FLUSH_ASSEMBLY) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Unsupported operation");
  /* setup matrix-vector multiply */
  if (!elist->rscctx) {
    next = elist->root->next;
    tr   = 0;
    tc   = 0;
    while (next) {
      PetscInt lr,lc;

      ierr = ISGetLocalSize(next->row,&lr);CHKERRQ(ierr);
      ierr = ISGetLocalSize(next->col,&lc);CHKERRQ(ierr);
      tr  += lr;
      tc  += lc;
      ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)next->elem),1,lr,PETSC_DECIDE,NULL,&next->rv);CHKERRQ(ierr);
      ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)next->elem),1,lc,PETSC_DECIDE,NULL,&next->cv);CHKERRQ(ierr);
      next = next->next;
    }
    ierr = VecCreateMPI(PetscObjectComm((PetscObject)A),tr,PETSC_DECIDE,&elist->rv);CHKERRQ(ierr);
    ierr = VecCreateMPI(PetscObjectComm((PetscObject)A),tc,PETSC_DECIDE,&elist->cv);CHKERRQ(ierr);

    ierr = PetscMalloc2(tr,&ridxs,tc,&cidxs);CHKERRQ(ierr);
    rp   = ridxs;
    cp   = cidxs;
    next = elist->root->next;
    while (next) {
      const PetscInt *idxs;
      PetscInt       i,ll;

      ierr = ISGetLocalSize(next->row,&ll);CHKERRQ(ierr);
      ierr = ISGetIndices(next->row,&idxs);CHKERRQ(ierr);
      for (i=0;i<ll;i++) *rp++ = idxs[i];
      ierr = ISRestoreIndices(next->row,&idxs);CHKERRQ(ierr);

      ierr = ISGetLocalSize(next->col,&ll);CHKERRQ(ierr);
      ierr = ISGetIndices(next->col,&idxs);CHKERRQ(ierr);
      for (i=0;i<ll;i++) *cp++ = idxs[i];
      ierr = ISRestoreIndices(next->col,&idxs);CHKERRQ(ierr);

      next = next->next;
    }
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)A),tr,ridxs,PETSC_OWN_POINTER,&ris);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)A),tc,cidxs,PETSC_OWN_POINTER,&cis);CHKERRQ(ierr);
    ierr = ISRenumber(ris,NULL,&M,&rgis);CHKERRQ(ierr);
    ierr = ISRenumber(cis,NULL,&N,&cgis);CHKERRQ(ierr);
    ierr = ISDestroy(&ris);CHKERRQ(ierr);
    ierr = ISDestroy(&cis);CHKERRQ(ierr);

    rhasnull = M != A->rmap->N ? PETSC_TRUE : PETSC_FALSE;
    chasnull = N != A->cmap->N ? PETSC_TRUE : PETSC_FALSE;
    if (rhasnull) {
      const PetscInt *idxs;
      PetscInt       *rgidxs,*rlidxs,i,ntr;

      ierr = PetscMalloc2(tr,&rgidxs,tr,&rlidxs);CHKERRQ(ierr);
      ierr = ISGetIndices(rgis,&idxs);CHKERRQ(ierr);
      for (i=0,ntr=0;i<tr;i++) {
        if (idxs[i] >= 0) {
          rgidxs[ntr] = idxs[i];
          rlidxs[ntr] = i;
          ntr++;
        }
      }
      ierr = ISRestoreIndices(rgis,&idxs);CHKERRQ(ierr);
      ierr = ISDestroy(&rgis);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)A),ntr,rgidxs,PETSC_OWN_POINTER,&rgis);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)A),ntr,rlidxs,PETSC_OWN_POINTER,&ris);CHKERRQ(ierr);
    }

    if (chasnull) {
      const PetscInt *idxs;
      PetscInt       *cgidxs,*clidxs,i,ntc;

      ierr = PetscMalloc2(tc,&cgidxs,tc,&clidxs);CHKERRQ(ierr);
      ierr = ISGetIndices(cgis,&idxs);CHKERRQ(ierr);
      for (i=0,ntc=0;i<tc;i++) {
        if (idxs[i] >= 0) {
          cgidxs[ntc] = idxs[i];
          clidxs[ntc] = i;
          ntc++;
        }
      }
      ierr = ISRestoreIndices(cgis,&idxs);CHKERRQ(ierr);
      ierr = ISDestroy(&cgis);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)A),ntc,cgidxs,PETSC_OWN_POINTER,&cgis);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)A),ntc,clidxs,PETSC_OWN_POINTER,&cis);CHKERRQ(ierr);
    }
    ierr = MatCreateVecs(A,&rv,&cv);CHKERRQ(ierr);
    ierr = VecScatterCreate(rv,rgis,elist->rv,ris,&elist->rscctx);CHKERRQ(ierr);
    ierr = VecScatterCreate(cv,cgis,elist->cv,cis,&elist->cscctx);CHKERRQ(ierr);
    ierr = VecDestroy(&rv);CHKERRQ(ierr);
    ierr = VecDestroy(&cv);CHKERRQ(ierr);
    ierr = ISDestroy(&rgis);CHKERRQ(ierr);
    ierr = ISDestroy(&cgis);CHKERRQ(ierr);
    ierr = ISDestroy(&ris);CHKERRQ(ierr);
    ierr = ISDestroy(&cis);CHKERRQ(ierr);
  }

  /* reset pointer for setvalues to the root of the list */
  elist->head = elist->root;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_Elementlist(Mat A)
{
  Mat_Elementlist     *elist = (Mat_Elementlist*)A->data;
  Mat_ElementlistLink next;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  next = elist->root->next;
  while (next) {
    ierr = MatZeroEntries(next->elem);CHKERRQ(ierr);
    next = next->next;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_Elementlist(Mat A, PetscViewer viewer)
{
  Mat_Elementlist     *elist = (Mat_Elementlist*)A->data;
  Mat_ElementlistLink next;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  next = elist->root->next;
  while (next) {
    ierr = MatView(next->elem,viewer);CHKERRQ(ierr);
    ierr = ISView(next->row,viewer);CHKERRQ(ierr);
    ierr = ISView(next->col,viewer);CHKERRQ(ierr);
    next = next->next;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_Elementlist(Mat A)
{
  Mat_Elementlist     *elist = (Mat_Elementlist*)A->data;
  Mat_ElementlistLink next;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  next = elist->root->next;
  while (next) {
    Mat_ElementlistLink tnext = next->next;

    ierr = MatDestroy(&next->elem);CHKERRQ(ierr);
    ierr = VecDestroy(&next->rv);CHKERRQ(ierr);
    ierr = VecDestroy(&next->cv);CHKERRQ(ierr);
    ierr = ISDestroy(&next->row);CHKERRQ(ierr);
    ierr = ISDestroy(&next->col);CHKERRQ(ierr);
    ierr = PetscFree(next);CHKERRQ(ierr);
    next = tnext;
  }
  ierr = PetscFree(elist->root);CHKERRQ(ierr);
  ierr = VecDestroy(&elist->rv);CHKERRQ(ierr);
  ierr = VecDestroy(&elist->cv);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&elist->rscctx);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&elist->cscctx);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValues_Elementlist(Mat A,PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols,const PetscScalar *values,InsertMode addv)
{
  PetscScalar         *delem;
  PetscErrorCode      ierr;
  Mat_Elementlist     *elist = (Mat_Elementlist*)A->data;
  Mat_ElementlistLink next,old;

  PetscFunctionBegin;
  if (addv == INSERT_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported operation");
  if (!elist->head->next) {
    ierr = PetscCalloc1(1,&next);CHKERRQ(ierr);
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,m,n,NULL,&next->elem);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,m,rows,PETSC_COPY_VALUES,&next->row);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n,cols,PETSC_COPY_VALUES,&next->col);CHKERRQ(ierr);
  } else {
#if defined(PETSC_USE_DEBUG)
    const PetscInt *idxs;
    PetscBool      ok;

    next = elist->head->next;
    if (m != next->elem->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only same sequence of element assembly is supported! row mismatch %D %D",m,next->elem->rmap->n);
    if (n != next->elem->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only same sequence of element assembly is supported! col mismatch %D %D",n,next->elem->cmap->n);
    ierr = ISGetIndices(next->row,&idxs);CHKERRQ(ierr);
    ierr = PetscMemcmp(idxs,rows,m,&ok);CHKERRQ(ierr);
    ierr = ISRestoreIndices(next->row,&idxs);CHKERRQ(ierr);
    if (!ok) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only same sequence of element assembly is supported! row indices mismatch");
    ierr = ISGetIndices(next->col,&idxs);CHKERRQ(ierr);
    ierr = PetscMemcmp(idxs,cols,n,&ok);CHKERRQ(ierr);
    ierr = ISRestoreIndices(next->col,&idxs);CHKERRQ(ierr);
    if (!ok) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only same sequence of element assembly is supported! col indices mismatch");
#else
    next = elist->head->next;
#endif
  }
  ierr = MatDenseGetArray(next->elem,&delem);CHKERRQ(ierr);
  if (elist->roworiented) {
    PetscInt i,j;
    for (i=0;i<m;i++) {
      for (j=0;j<n;j++) {
        delem[j*m+i] = values[i*n+j];
      }
    }
  } else {
    ierr = PetscMemcpy(delem,values,m*n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(next->elem,&delem);CHKERRQ(ierr);
  elist->head->next = next;
  elist->head = next;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_Elementlist(Mat A)
{
  PetscErrorCode  ierr;
  Mat_Elementlist *a;

  PetscFunctionBegin;
  ierr    = PetscNewLog(A,&a);CHKERRQ(ierr);
  A->data = (void*)a;

  /* root node of the list */
  ierr           = PetscCalloc1(1,&a->root);CHKERRQ(ierr);
  a->head        = a->root;
  a->roworiented = PETSC_TRUE;

  /* matrix ops */
  ierr = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
  A->ops->setvalues     = MatSetValues_Elementlist;
  A->ops->destroy       = MatDestroy_Elementlist;
  A->ops->view          = MatView_Elementlist;
  A->ops->zeroentries   = MatZeroEntries_Elementlist;
  A->ops->assemblyend   = MatAssemblyEnd_Elementlist;
  A->ops->assemblybegin = MatAssemblyBegin_Elementlist;
  A->ops->mult          = MatMult_Elementlist;
  A->ops->multtranspose = MatMultTranspose_Elementlist;

  /* TODO: add preallocation */
  A->preallocated = PETSC_TRUE;

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATELEMENTLIST);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

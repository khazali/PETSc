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
  Mat_Elementlist *elist = (Mat_Elementlist*)A->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (atype == MAT_FLUSH_ASSEMBLY) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Unsupported operation");
  {
    PetscBool lt = (PetscBool)(elist->unsym || (A->rmap->N != A->cmap->N));
    ierr = MPI_Allreduce(&lt,&elist->unsym,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
  }
  /* here we can rearrange in subdomains */
  PetscFunctionReturn(0);
}

static PetscErrorCode MatElementlistElVec_Private(Mat A)
{
  Mat_Elementlist     *elist = (Mat_Elementlist*)A->data;
  Mat_ElementlistLink next;
  PetscInt            tr,tc;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (elist->rv) PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
}

static PetscErrorCode MatElementlistHasneg_Private(Mat A)
{
  Mat_Elementlist     *elist = (Mat_Elementlist*)A->data;
  Mat_ElementlistLink next;
  PetscBool           lhasneg[2];
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (elist->hasneg[2]) PetscFunctionReturn(0);
  lhasneg[0] = PETSC_FALSE;
  lhasneg[1] = PETSC_FALSE;
  next       = elist->root->next;
  while (next && (!lhasneg[0] || !lhasneg[1])) {
    PetscInt m;
    ierr = ISGetMinMax(next->row,&m,NULL);CHKERRQ(ierr);
    if (m < 0) lhasneg[0] = PETSC_TRUE;
    ierr = ISGetMinMax(next->col,&m,NULL);CHKERRQ(ierr);
    if (m < 0) lhasneg[1] = PETSC_TRUE;
    next = next->next;
  }
  ierr = MPI_Allreduce(lhasneg,elist->hasneg,2,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
  elist->hasneg[2] = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatElementlistISes_Private(Mat A)
{
  Mat_Elementlist     *elist = (Mat_Elementlist*)A->data;
  Mat_ElementlistLink next;
  PetscInt            *ridxs,*cidxs,*rp,*cp;
  PetscInt            *rlidxs,*clidxs,*rlp,*clp;
  PetscInt            tr,tc,cst,rst,rcm,ccm;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (elist->rgis) PetscFunctionReturn(0);
  ierr = MatElementlistElVec_Private(A);CHKERRQ(ierr);
  ierr = MatElementlistHasneg_Private(A);CHKERRQ(ierr);
  ierr = VecGetLocalSize(elist->rv,&tr);CHKERRQ(ierr);
  ierr = VecGetLocalSize(elist->cv,&tc);CHKERRQ(ierr);
  ierr = PetscMalloc1(tr,&ridxs);CHKERRQ(ierr);
  if (elist->hasneg[0]) {
    ierr = PetscMalloc1(tr,&rlidxs);CHKERRQ(ierr);
  }
  rp  = ridxs;
  rlp = rlidxs;
  if (elist->unsym) {
    ierr = PetscMalloc1(tc,&cidxs);CHKERRQ(ierr);
    if (elist->hasneg[1]) {
      ierr = PetscMalloc1(tc,&clidxs);CHKERRQ(ierr);
    }
    cp  = cidxs;
    clp = clidxs;
  }
  next = elist->root->next;
  tr   = 0;
  tc   = 0;
  ierr = VecGetOwnershipRange(elist->rv,&rst,NULL);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(elist->cv,&cst,NULL);CHKERRQ(ierr);
  rcm  = 0;
  ccm  = 0;
  while (next) {
    const PetscInt *idxs;
    PetscInt       i,ll;

    ierr = ISGetLocalSize(next->row,&ll);CHKERRQ(ierr);
    ierr = ISGetIndices(next->row,&idxs);CHKERRQ(ierr);
    if (elist->hasneg[0]) {
      for (i=0;i<ll;i++) {
        if (idxs[i]>=0) {
          *rp++  = idxs[i];
          *rlp++ = i + rcm + rst;
          tr++;
        }
      }
    } else {
      for (i=0;i<ll;i++) *rp++ = idxs[i];
      tr += ll;
    }
    rcm += ll;
    ierr = ISRestoreIndices(next->row,&idxs);CHKERRQ(ierr);

    if (elist->unsym) {
      ierr = ISGetLocalSize(next->col,&ll);CHKERRQ(ierr);
      ierr = ISGetIndices(next->col,&idxs);CHKERRQ(ierr);
      if (elist->hasneg[1]) {
        for (i=0;i<ll;i++) {
          if (idxs[i]>=0) {
            *cp++  = idxs[i];
            *clp++ = i + ccm + cst;
            tc++;
          }
        }
      } else {
        for (i=0;i<ll;i++) *cp++ = idxs[i];
        tc += ll;
      }
      ccm += ll;
      ierr = ISRestoreIndices(next->col,&idxs);CHKERRQ(ierr);
    }
    next = next->next;
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)A),tr,ridxs,PETSC_OWN_POINTER,&elist->rgis);CHKERRQ(ierr);
  if (elist->hasneg[0]) {
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)A),tr,rlidxs,PETSC_OWN_POINTER,&elist->ris);CHKERRQ(ierr);
  }
  if (elist->unsym) {
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)A),tc,cidxs,PETSC_OWN_POINTER,&elist->cgis);CHKERRQ(ierr);
    if (elist->hasneg[1]) {
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)A),tc,clidxs,PETSC_OWN_POINTER,&elist->cis);CHKERRQ(ierr);
    }
  } else {
    ierr        = PetscObjectReference((PetscObject)elist->rgis);CHKERRQ(ierr);
    elist->cgis = elist->rgis;
    if (elist->hasneg[1]) {
      ierr       = PetscObjectReference((PetscObject)elist->ris);CHKERRQ(ierr);
      elist->cis = elist->ris;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_Elementlist(Mat A, MatAssemblyType atype)
{
  Mat_Elementlist *elist = (Mat_Elementlist*)A->data;

  PetscFunctionBegin;
  if (atype == MAT_FLUSH_ASSEMBLY) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Unsupported operation");
  /* setup matrix-vector multiply */
  if (!elist->rscctx) {
    Vec            rv,cv;
    PetscBool      congruent;
    PetscErrorCode ierr;

    /* setup work vectors */
    ierr = MatElementlistElVec_Private(A);CHKERRQ(ierr);

    /* determine if there are negative indices */
    ierr = MatElementlistHasneg_Private(A);CHKERRQ(ierr);

    /* compute indices for scatters */
    ierr = MatElementlistISes_Private(A);CHKERRQ(ierr);

    /* create scatters */
    ierr = MatCreateVecs(A,&rv,&cv);CHKERRQ(ierr);
    ierr = VecScatterCreate(rv,elist->rgis,elist->rv,elist->ris,&elist->rscctx);CHKERRQ(ierr);
    ierr = PetscLayoutCompare(A->rmap,A->cmap,&congruent);CHKERRQ(ierr);
    if (elist->unsym || !congruent) {
      ierr = VecScatterCreate(cv,elist->cgis,elist->cv,elist->cis,&elist->cscctx);CHKERRQ(ierr);
    } else {
      ierr          = PetscObjectReference((PetscObject)elist->rscctx);CHKERRQ(ierr);
      elist->cscctx = elist->rscctx;
    }
    ierr = VecDestroy(&rv);CHKERRQ(ierr);
    ierr = VecDestroy(&cv);CHKERRQ(ierr);
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
  MPI_Comm            scomm;
  PetscViewer         sviewer = NULL;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  next = elist->root->next;
  if (next) {
    ierr = PetscObjectGetComm((PetscObject)next->elem,&scomm);
    ierr = PetscViewerGetSubViewer(viewer,scomm,&sviewer);CHKERRQ(ierr);
  }
  while (next) {
    ierr = MatView(next->elem,sviewer);CHKERRQ(ierr);
    ierr = ISView(next->row,sviewer);CHKERRQ(ierr);
    ierr = ISView(next->col,sviewer);CHKERRQ(ierr);
    next = next->next;
  }
  if (sviewer) {
    ierr = PetscViewerRestoreSubViewer(viewer,scomm,&sviewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
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
  ierr = ISDestroy(&elist->rgis);CHKERRQ(ierr);
  ierr = ISDestroy(&elist->cgis);CHKERRQ(ierr);
  ierr = ISDestroy(&elist->ris);CHKERRQ(ierr);
  ierr = ISDestroy(&elist->cis);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValues_Elementlist(Mat A,PetscInt m,const PetscInt *rows, PetscInt n,const PetscInt *cols,const PetscScalar *values,InsertMode addv)
{
  PetscScalar         *delem;
  PetscErrorCode      ierr;
  Mat_Elementlist     *elist = (Mat_Elementlist*)A->data;
  Mat_ElementlistLink next;

  PetscFunctionBegin;
  if (addv == INSERT_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported operation");
  if (!elist->head->next) {
    ierr = PetscCalloc1(1,&next);CHKERRQ(ierr);
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,m,n,NULL,&next->elem);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,m,rows,PETSC_COPY_VALUES,&next->row);CHKERRQ(ierr);
    if (!elist->unsym && (m != n || rows != cols)) {
      PetscBool same = PETSC_FALSE;
      if (m == n) {
        ierr = PetscMemcmp(rows,cols,m,&same);CHKERRQ(ierr);
      }
      if (!same) {
        ierr = ISCreateGeneral(PETSC_COMM_SELF,n,cols,PETSC_COPY_VALUES,&next->col);CHKERRQ(ierr);
        elist->unsym = PETSC_TRUE;
      } else {
        ierr = PetscObjectReference((PetscObject)next->row);CHKERRQ(ierr);
        next->col = next->row;
      }
    } else {
      ierr = PetscObjectReference((PetscObject)next->row);CHKERRQ(ierr);
      next->col = next->row;
    }
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
  a->hasneg[2]   = PETSC_FALSE;

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

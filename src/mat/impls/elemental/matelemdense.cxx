#include <../src/mat/impls/elemental/matelemimpl.h> /*I "petscmat.h" I*/

/*
    The variable Petsc_Elemental_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a Mat_Elemental_Grid
*/
static PetscMPIInt Petsc_Elemental_keyval = MPI_KEYVAL_INVALID;

#undef __FUNCT__
#define __FUNCT__ "MatView_ElemDense"
static PetscErrorCode MatView_ElemDense(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  Mat_ElemDense  *a = (Mat_ElemDense*)A->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscViewerFormat format;
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      /* call elemental viewing function */
      ierr = PetscViewerASCIIPrintf(viewer,"ElemDense run parameters:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  allocated entries=%d\n",(*a->emat).AllocatedMemory());CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  grid height=%d, grid width=%d\n",(*a->emat).Grid().Height(),(*a->emat).Grid().Width());CHKERRQ(ierr);
      if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
        /* call elemental viewing function */
        ierr = PetscPrintf(PetscObjectComm((PetscObject)viewer),"test matview_elemental 2\n");CHKERRQ(ierr);
      }

    } else if (format == PETSC_VIEWER_DEFAULT) {
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
      El::Print( *a->emat, "ElemDense matrix (cyclic ordering)" );
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      if (A->factortype == MAT_FACTOR_NONE){
        Mat Adense;
        ierr = PetscPrintf(PetscObjectComm((PetscObject)viewer),"ElemDense matrix (explicit ordering)\n");CHKERRQ(ierr);
        ierr = MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&Adense);CHKERRQ(ierr);
        ierr = MatView(Adense,viewer);CHKERRQ(ierr);
        ierr = MatDestroy(&Adense);CHKERRQ(ierr);
      }
    } else SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Format");
  } else {
    /* convert to dense format and call MatView() */
    Mat Adense;
    ierr = PetscPrintf(PetscObjectComm((PetscObject)viewer),"ElemDense matrix (explicit ordering)\n");CHKERRQ(ierr);
    ierr = MatConvert(A,MATDENSE,MAT_INITIAL_MATRIX,&Adense);CHKERRQ(ierr);
    ierr = MatView(Adense,viewer);CHKERRQ(ierr);
    ierr = MatDestroy(&Adense);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetInfo_ElemDense"
static PetscErrorCode MatGetInfo_ElemDense(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_ElemDense  *a = (Mat_ElemDense*)A->data;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank);

  /* if (!rank) printf("          .........MatGetInfo_ElemDense ...\n"); */
  info->block_size     = 1.0;

  if (flag == MAT_LOCAL) {
    info->nz_allocated   = (double)(*a->emat).AllocatedMemory(); /* locally allocated */
    info->nz_used        = info->nz_allocated;
  } else if (flag == MAT_GLOBAL_MAX) {
    //ierr = MPI_Allreduce(isend,irecv,5,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)matin));CHKERRQ(ierr);
    /* see MatGetInfo_MPIAIJ() for getting global info->nz_allocated! */
    //SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP," MAT_GLOBAL_MAX not written yet");
  } else if (flag == MAT_GLOBAL_SUM) {
    //SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP," MAT_GLOBAL_SUM not written yet");
    info->nz_allocated   = (double)(*a->emat).AllocatedMemory(); /* locally allocated */
    info->nz_used        = info->nz_allocated; /* assume ElemDense does accurate allocation */
    //ierr = MPI_Allreduce(isend,irecv,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)A));CHKERRQ(ierr);
    //PetscPrintf(PETSC_COMM_SELF,"    ... [%d] locally allocated %g\n",rank,info->nz_allocated);
  }

  info->nz_unneeded       = 0.0;
  info->assemblies        = (double)A->num_ass;
  info->mallocs           = 0;
  info->memory            = ((PetscObject)A)->mem;
  info->fill_ratio_given  = 0; /* determined by Elemental */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetValues_ElemDense"
static PetscErrorCode MatSetValues_ElemDense(Mat A,PetscInt nr,const PetscInt *rows,PetscInt nc,const PetscInt *cols,const PetscScalar *vals,InsertMode imode)
{
  PetscErrorCode ierr;
  Mat_ElemDense  *a = (Mat_ElemDense*)A->data;
  PetscMPIInt    rank;
  PetscInt       i,j,rrank,ridx,crank,cidx;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank);CHKERRQ(ierr);

  const El::Grid &grid = a->emat->Grid();
  for (i=0; i<nr; i++) {
    PetscInt erow,ecol,elrow,elcol;
    if (rows[i] < 0) continue;
    P2RO(A,0,rows[i],&rrank,&ridx);
    RO2E(A,0,rrank,ridx,&erow);
    if (rrank < 0 || ridx < 0 || erow < 0) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Incorrect row translation");
    for (j=0; j<nc; j++) {
      if (cols[j] < 0) continue;
      P2RO(A,1,cols[j],&crank,&cidx);
      RO2E(A,1,crank,cidx,&ecol);
      if (crank < 0 || cidx < 0 || ecol < 0) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Incorrect col translation");
      if (erow % grid.MCSize() != grid.MCRank() || ecol % grid.MRSize() != grid.MRRank()){ /* off-proc entry */
        if (imode != ADD_VALUES) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only ADD_VALUES to off-processor entry is supported");
        /* PetscPrintf(PETSC_COMM_SELF,"[%D] add off-proc entry (%D,%D, %g) (%D %D)\n",rank,rows[i],cols[j],*(vals+i*nc),erow,ecol); */
        a->esubmat->Set(0,0, (PetscElemScalar)vals[i*nc+j]);
        a->interface->Axpy(1.0,*(a->esubmat),erow,ecol);
        continue;
      }
      elrow = erow / grid.MCSize();
      elcol = ecol / grid.MRSize();
      switch (imode) {
      case INSERT_VALUES: a->emat->SetLocal(elrow,elcol,(PetscElemScalar)vals[i*nc+j]); break;
      case ADD_VALUES: a->emat->UpdateLocal(elrow,elcol,(PetscElemScalar)vals[i*nc+j]); break;
      default: SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No support for InsertMode %d",(int)imode);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_ElemDense"
static PetscErrorCode MatMult_ElemDense(Mat A,Vec X,Vec Y)
{
  Mat_ElemDense         *a = (Mat_ElemDense*)A->data;
  PetscErrorCode        ierr;
  const PetscElemScalar *x;
  PetscElemScalar       *y;
  PetscElemScalar       one = 1,zero = 0;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecGetArray(Y,(PetscScalar **)&y);CHKERRQ(ierr);
  { /* Scoping so that constructor is called before pointer is returned */
    El::DistMatrix<PetscElemScalar,El::VC,El::STAR> xe, ye;
    xe.LockedAttach(A->cmap->N,1,*a->grid,0,0,x,A->cmap->n);
    ye.Attach(A->rmap->N,1,*a->grid,0,0,y,A->rmap->n);
    El::Gemv(El::NORMAL,one,*a->emat,xe,zero,ye);
  }
  ierr = VecRestoreArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,(PetscScalar **)&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_ElemDense"
static PetscErrorCode MatMultTranspose_ElemDense(Mat A,Vec X,Vec Y)
{
  Mat_ElemDense         *a = (Mat_ElemDense*)A->data;
  PetscErrorCode        ierr;
  const PetscElemScalar *x;
  PetscElemScalar       *y;
  PetscElemScalar       one = 1,zero = 0;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecGetArray(Y,(PetscScalar **)&y);CHKERRQ(ierr);
  { /* Scoping so that constructor is called before pointer is returned */
    El::DistMatrix<PetscElemScalar,El::VC,El::STAR> xe, ye;
    xe.LockedAttach(A->rmap->N,1,*a->grid,0,0,x,A->rmap->n);
    ye.Attach(A->cmap->N,1,*a->grid,0,0,y,A->cmap->n);
    El::Gemv(El::TRANSPOSE,one,*a->emat,xe,zero,ye);
  }
  ierr = VecRestoreArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,(PetscScalar **)&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_ElemDense"
static PetscErrorCode MatMultAdd_ElemDense(Mat A,Vec X,Vec Y,Vec Z)
{
  Mat_ElemDense         *a = (Mat_ElemDense*)A->data;
  PetscErrorCode        ierr;
  const PetscElemScalar *x;
  PetscElemScalar       *z;
  PetscElemScalar       one = 1;

  PetscFunctionBegin;
  if (Y != Z) {ierr = VecCopy(Y,Z);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecGetArray(Z,(PetscScalar **)&z);CHKERRQ(ierr);
  { /* Scoping so that constructor is called before pointer is returned */
    El::DistMatrix<PetscElemScalar,El::VC,El::STAR> xe, ze;
    xe.LockedAttach(A->cmap->N,1,*a->grid,0,0,x,A->cmap->n);
    ze.Attach(A->rmap->N,1,*a->grid,0,0,z,A->rmap->n);
    El::Gemv(El::NORMAL,one,*a->emat,xe,one,ze);
  }
  ierr = VecRestoreArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Z,(PetscScalar **)&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_ElemDense"
static PetscErrorCode MatMultTransposeAdd_ElemDense(Mat A,Vec X,Vec Y,Vec Z)
{
  Mat_ElemDense         *a = (Mat_ElemDense*)A->data;
  PetscErrorCode        ierr;
  const PetscElemScalar *x;
  PetscElemScalar       *z;
  PetscElemScalar       one = 1;

  PetscFunctionBegin;
  if (Y != Z) {ierr = VecCopy(Y,Z);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecGetArray(Z,(PetscScalar **)&z);CHKERRQ(ierr);
  { /* Scoping so that constructor is called before pointer is returned */
    El::DistMatrix<PetscElemScalar,El::VC,El::STAR> xe, ze;
    xe.LockedAttach(A->rmap->N,1,*a->grid,0,0,x,A->rmap->n);
    ze.Attach(A->cmap->N,1,*a->grid,0,0,z,A->cmap->n);
    El::Gemv(El::TRANSPOSE,one,*a->emat,xe,one,ze);
  }
  ierr = VecRestoreArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Z,(PetscScalar **)&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultNumeric_ElemDense"
static PetscErrorCode MatMatMultNumeric_ElemDense(Mat A,Mat B,Mat C)
{
  Mat_ElemDense    *a = (Mat_ElemDense*)A->data;
  Mat_ElemDense    *b = (Mat_ElemDense*)B->data;
  Mat_ElemDense    *c = (Mat_ElemDense*)C->data;
  PetscElemScalar  one = 1,zero = 0;

  PetscFunctionBegin;
  { /* Scoping so that constructor is called before pointer is returned */
    El::Gemm(El::NORMAL,El::NORMAL,one,*a->emat,*b->emat,zero,*c->emat);
  }
  C->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultSymbolic_ElemDense"
static PetscErrorCode MatMatMultSymbolic_ElemDense(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;
  Mat            Ce;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatCreate(comm,&Ce);CHKERRQ(ierr);
  ierr = MatSetSizes(Ce,A->rmap->n,B->cmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(Ce,MATELEMDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(Ce);CHKERRQ(ierr);
  *C = Ce;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_ElemDense"
static PetscErrorCode MatMatMult_ElemDense(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){
    ierr = PetscLogEventBegin(MAT_MatMultSymbolic,A,B,0,0);CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_ElemDense(A,B,1.0,C);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_MatMultSymbolic,A,B,0,0);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(MAT_MatMultNumeric,A,B,0,0);CHKERRQ(ierr);
  ierr = MatMatMultNumeric_ElemDense(A,B,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMultNumeric,A,B,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatTransposeMultNumeric_ElemDense"
static PetscErrorCode MatMatTransposeMultNumeric_ElemDense(Mat A,Mat B,Mat C)
{
  Mat_ElemDense      *a = (Mat_ElemDense*)A->data;
  Mat_ElemDense      *b = (Mat_ElemDense*)B->data;
  Mat_ElemDense      *c = (Mat_ElemDense*)C->data;
  PetscElemScalar    one = 1,zero = 0;

  PetscFunctionBegin;
  { /* Scoping so that constructor is called before pointer is returned */
    El::Gemm(El::NORMAL,El::TRANSPOSE,one,*a->emat,*b->emat,zero,*c->emat);
  }
  C->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatTransposeMultSymbolic_ElemDense"
static PetscErrorCode MatMatTransposeMultSymbolic_ElemDense(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;
  Mat            Ce;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatCreate(comm,&Ce);CHKERRQ(ierr);
  ierr = MatSetSizes(Ce,A->rmap->n,B->rmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(Ce,MATELEMDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(Ce);CHKERRQ(ierr);
  *C = Ce;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatTransposeMult_ElemDense"
static PetscErrorCode MatMatTransposeMult_ElemDense(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){
    ierr = PetscLogEventBegin(MAT_MatTransposeMultSymbolic,A,B,0,0);CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_ElemDense(A,B,1.0,C);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_MatTransposeMultSymbolic,A,B,0,0);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(MAT_MatTransposeMultNumeric,A,B,0,0);CHKERRQ(ierr);
  ierr = MatMatTransposeMultNumeric_ElemDense(A,B,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatTransposeMultNumeric,A,B,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_ElemDense"
static PetscErrorCode MatGetDiagonal_ElemDense(Mat A,Vec D)
{
  PetscInt        i,nrows,ncols,nD,rrank,ridx,crank,cidx;
  Mat_ElemDense   *a = (Mat_ElemDense*)A->data;
  PetscErrorCode  ierr;
  PetscElemScalar v;
  MPI_Comm        comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatGetSize(A,&nrows,&ncols);CHKERRQ(ierr);
  nD = nrows>ncols ? ncols : nrows;
  for (i=0; i<nD; i++) {
    PetscInt erow,ecol;
    P2RO(A,0,i,&rrank,&ridx);
    RO2E(A,0,rrank,ridx,&erow);
    if (rrank < 0 || ridx < 0 || erow < 0) SETERRQ(comm,PETSC_ERR_PLIB,"Incorrect row translation");
    P2RO(A,1,i,&crank,&cidx);
    RO2E(A,1,crank,cidx,&ecol);
    if (crank < 0 || cidx < 0 || ecol < 0) SETERRQ(comm,PETSC_ERR_PLIB,"Incorrect col translation");
    v = a->emat->Get(erow,ecol);
    ierr = VecSetValues(D,1,&i,(PetscScalar*)&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(D);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDiagonalScale_ElemDense"
static PetscErrorCode MatDiagonalScale_ElemDense(Mat X,Vec L,Vec R)
{
  Mat_ElemDense         *x = (Mat_ElemDense*)X->data;
  const PetscElemScalar *d;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (R) {
    ierr = VecGetArrayRead(R,(const PetscScalar **)&d);CHKERRQ(ierr);
    El::DistMatrix<PetscElemScalar,El::VC,El::STAR> de;
    de.LockedAttach(X->cmap->N,1,*x->grid,0,0,d,X->cmap->n);
    El::DiagonalScale(El::RIGHT,El::NORMAL,de,*x->emat);
    ierr = VecRestoreArrayRead(R,(const PetscScalar **)&d);CHKERRQ(ierr);
  }
  if (L) {
    ierr = VecGetArrayRead(L,(const PetscScalar **)&d);CHKERRQ(ierr);
    El::DistMatrix<PetscElemScalar,El::VC,El::STAR> de;
    de.LockedAttach(X->rmap->N,1,*x->grid,0,0,d,X->rmap->n);
    El::DiagonalScale(El::LEFT,El::NORMAL,de,*x->emat);
    ierr = VecRestoreArrayRead(L,(const PetscScalar **)&d);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatScale_ElemDense"
static PetscErrorCode MatScale_ElemDense(Mat X,PetscScalar a)
{
  Mat_ElemDense  *x = (Mat_ElemDense*)X->data;

  PetscFunctionBegin;
  El::Scale((PetscElemScalar)a,*x->emat);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAXPY_ElemDense"
static PetscErrorCode MatAXPY_ElemDense(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  Mat_ElemDense  *x = (Mat_ElemDense*)X->data;
  Mat_ElemDense  *y = (Mat_ElemDense*)Y->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  El::Axpy((PetscElemScalar)a,*x->emat,*y->emat);
  ierr = PetscObjectStateIncrease((PetscObject)Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCopy_ElemDense"
static PetscErrorCode MatCopy_ElemDense(Mat A,Mat B,MatStructure str)
{
  Mat_ElemDense *a=(Mat_ElemDense*)A->data;
  Mat_ElemDense *b=(Mat_ElemDense*)B->data;

  PetscFunctionBegin;
  El::Copy(*a->emat,*b->emat);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_ElemDense"
static PetscErrorCode MatDuplicate_ElemDense(Mat A,MatDuplicateOption op,Mat *B)
{
  Mat            Be;
  MPI_Comm       comm;
  Mat_ElemDense  *a=(Mat_ElemDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatCreate(comm,&Be);CHKERRQ(ierr);
  ierr = MatSetSizes(Be,A->rmap->n,A->cmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(Be,MATELEMDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(Be);CHKERRQ(ierr);
  *B = Be;
  if (op == MAT_COPY_VALUES) {
    Mat_ElemDense *b=(Mat_ElemDense*)Be->data;
    El::Copy(*a->emat,*b->emat);
  }
  Be->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTranspose_ElemDense"
static PetscErrorCode MatTranspose_ElemDense(Mat A,MatReuse reuse,Mat *B)
{
  Mat            Be = *B;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  Mat_ElemDense  *a = (Mat_ElemDense*)A->data, *b;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  /* Only out-of-place supported */
  if (reuse == MAT_INITIAL_MATRIX){
    ierr = MatCreate(comm,&Be);CHKERRQ(ierr);
    ierr = MatSetSizes(Be,A->cmap->n,A->rmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetType(Be,MATELEMDENSE);CHKERRQ(ierr);
    ierr = MatSetUp(Be);CHKERRQ(ierr);
    *B = Be;
  }
  b = (Mat_ElemDense*)Be->data;
  El::Transpose(*a->emat,*b->emat);
  Be->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConjugate_ElemDense"
static PetscErrorCode MatConjugate_ElemDense(Mat A)
{
  Mat_ElemDense  *a = (Mat_ElemDense*)A->data;

  PetscFunctionBegin;
  El::Conjugate(*a->emat);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatHermitianTranspose_ElemDense"
static PetscErrorCode MatHermitianTranspose_ElemDense(Mat A,MatReuse reuse,Mat *B)
{
  Mat            Be = *B;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  Mat_ElemDense  *a = (Mat_ElemDense*)A->data, *b;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  /* Only out-of-place supported */
  if (reuse == MAT_INITIAL_MATRIX){
    ierr = MatCreate(comm,&Be);CHKERRQ(ierr);
    ierr = MatSetSizes(Be,A->cmap->n,A->rmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetType(Be,MATELEMDENSE);CHKERRQ(ierr);
    ierr = MatSetUp(Be);CHKERRQ(ierr);
    *B = Be;
  }
  b = (Mat_ElemDense*)Be->data;
  El::Adjoint(*a->emat,*b->emat);
  Be->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolve_ElemDense"
static PetscErrorCode MatSolve_ElemDense(Mat A,Vec B,Vec X)
{
  Mat_ElemDense     *a = (Mat_ElemDense*)A->data;
  PetscErrorCode    ierr;
  PetscElemScalar   *x;

  PetscFunctionBegin;
  ierr = VecCopy(B,X);CHKERRQ(ierr);
  ierr = VecGetArray(X,(PetscScalar **)&x);CHKERRQ(ierr);
  El::DistMatrix<PetscElemScalar,El::VC,El::STAR> xe;
  xe.Attach(A->rmap->N,1,*a->grid,0,0,x,A->rmap->n);
  El::DistMatrix<PetscElemScalar,El::MC,El::MR> xer(xe);
  switch (A->factortype) {
  case MAT_FACTOR_LU:
    if ((*a->pivot).AllocatedMemory()) {
      El::lu::SolveAfter(El::NORMAL,*a->emat,*a->pivot,xer);
      El::Copy(xer,xe);
    } else {
      El::lu::SolveAfter(El::NORMAL,*a->emat,xer);
      El::Copy(xer,xe);
    }
    break;
  case MAT_FACTOR_CHOLESKY:
    El::cholesky::SolveAfter(El::UPPER,El::NORMAL,*a->emat,xer);
    El::Copy(xer,xe);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unfactored Matrix or Unsupported MatFactorType");
    break;
  }
  ierr = VecRestoreArray(X,(PetscScalar **)&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolveAdd_ElemDense"
static PetscErrorCode MatSolveAdd_ElemDense(Mat A,Vec B,Vec Y,Vec X)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatSolve_ElemDense(A,B,X);CHKERRQ(ierr);
  ierr = VecAXPY(X,1,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatSolve_ElemDense"
static PetscErrorCode MatMatSolve_ElemDense(Mat A,Mat B,Mat X)
{
  Mat_ElemDense *a=(Mat_ElemDense*)A->data;
  Mat_ElemDense *b=(Mat_ElemDense*)B->data;
  Mat_ElemDense *x=(Mat_ElemDense*)X->data;

  PetscFunctionBegin;
  El::Copy(*b->emat,*x->emat);
  switch (A->factortype) {
  case MAT_FACTOR_LU:
    if ((*a->pivot).AllocatedMemory()) {
      El::lu::SolveAfter(El::NORMAL,*a->emat,*a->pivot,*x->emat);
    } else {
      El::lu::SolveAfter(El::NORMAL,*a->emat,*x->emat);
    }
    break;
  case MAT_FACTOR_CHOLESKY:
    El::cholesky::SolveAfter(El::UPPER,El::NORMAL,*a->emat,*x->emat);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unfactored Matrix or Unsupported MatFactorType");
    break;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLUFactor_ElemDense"
static PetscErrorCode MatLUFactor_ElemDense(Mat A,IS row,IS col,const MatFactorInfo *info)
{
  Mat_ElemDense  *a = (Mat_ElemDense*)A->data;

  PetscFunctionBegin;
  if (info->dtcol){
    El::LU(*a->emat,*a->pivot);
  } else {
    El::LU(*a->emat);
  }
  A->factortype = MAT_FACTOR_LU;
  A->assembled  = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorNumeric_ElemDense"
static PetscErrorCode  MatLUFactorNumeric_ElemDense(Mat F,Mat A,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCopy(A,F,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatLUFactor_ElemDense(F,0,0,info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorSymbolic_ElemDense"
static PetscErrorCode  MatLUFactorSymbolic_ElemDense(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  /* F is create and allocated by MatGetFactor_elemental_petsc(), skip this routine. */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactor_ElemDense"
static PetscErrorCode MatCholeskyFactor_ElemDense(Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_ElemDense  *a = (Mat_ElemDense*)A->data;
  El::DistMatrix<PetscElemScalar,El::MC,El::STAR> d;

  PetscFunctionBegin;
  El::Cholesky(El::UPPER,*a->emat);
  A->factortype = MAT_FACTOR_CHOLESKY;
  A->assembled  = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorNumeric_ElemDense"
static PetscErrorCode MatCholeskyFactorNumeric_ElemDense(Mat F,Mat A,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCopy(A,F,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatCholeskyFactor_ElemDense(F,0,info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorSymbolic_ElemDense"
static PetscErrorCode MatCholeskyFactorSymbolic_ElemDense(Mat F,Mat A,IS perm,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  /* F is create and allocated by MatGetFactor_elemdense_petsc(), skip this routine. */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_elemdense_elemdense"
PetscErrorCode MatGetFactor_elemdense_elemdense(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(B,MATELEMDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  B->factortype = ftype;
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverPackage_C",MatFactorGetSolverPackage_elemental_elemental);CHKERRQ(ierr);
  *F            = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatNorm_ElemDense"
static PetscErrorCode MatNorm_ElemDense(Mat A,NormType type,PetscReal *nrm)
{
  Mat_ElemDense *a=(Mat_ElemDense*)A->data;

  PetscFunctionBegin;
  switch (type){
  case NORM_1:
    *nrm = El::OneNorm(*a->emat);
    break;
  case NORM_FROBENIUS:
    *nrm = El::FrobeniusNorm(*a->emat);
    break;
  case NORM_INFINITY:
    *nrm = El::InfinityNorm(*a->emat);
    break;
  default:
    printf("Error: unsupported norm type!\n");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatZeroEntries_ElemDense"
static PetscErrorCode MatZeroEntries_ElemDense(Mat A)
{
  Mat_ElemDense *a=(Mat_ElemDense*)A->data;

  PetscFunctionBegin;
  El::Zero(*a->emat);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetOwnershipIS_ElemDense"
static PetscErrorCode MatGetOwnershipIS_ElemDense(Mat A,IS *rows,IS *cols)
{
  Mat_ElemDense  *a = (Mat_ElemDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,m,shift,stride,*idx;

  PetscFunctionBegin;
  if (rows) {
    m = a->emat->LocalHeight();
    shift = a->emat->ColShift();
    stride = a->emat->ColStride();
    ierr = PetscMalloc1(m,&idx);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      PetscInt rank,offset;
      E2RO(A,0,shift+i*stride,&rank,&offset);
      RO2P(A,0,rank,offset,&idx[i]);
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,m,idx,PETSC_OWN_POINTER,rows);CHKERRQ(ierr);
  }
  if (cols) {
    m = a->emat->LocalWidth();
    shift = a->emat->RowShift();
    stride = a->emat->RowStride();
    ierr = PetscMalloc1(m,&idx);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      PetscInt rank,offset;
      E2RO(A,1,shift+i*stride,&rank,&offset);
      RO2P(A,1,rank,offset,&idx[i]);
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,m,idx,PETSC_OWN_POINTER,cols);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_ElemDense_Dense"
static PetscErrorCode MatConvert_ElemDense_Dense(Mat A,MatType newtype,MatReuse reuse,Mat *B)
{
  Mat                Bmpi;
  Mat_ElemDense      *a = (Mat_ElemDense*)A->data;
  MPI_Comm           comm;
  PetscErrorCode     ierr;
  PetscInt           rrank,ridx,crank,cidx,nrows,ncols,i,j;
  PetscElemScalar    v;
  PetscBool          s1,s2,s3;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = PetscStrcmp(newtype,MATDENSE,&s1);CHKERRQ(ierr);
  ierr = PetscStrcmp(newtype,MATSEQDENSE,&s2);CHKERRQ(ierr);
  ierr = PetscStrcmp(newtype,MATMPIDENSE,&s3);CHKERRQ(ierr);
  if (!s1 && !s2 && !s3) SETERRQ(comm,PETSC_ERR_SUP,"Unsupported New MatType: must be MATDENSE, MATSEQDENSE or MATMPIDENSE");
  ierr = MatCreate(comm,&Bmpi);CHKERRQ(ierr);
  ierr = MatSetSizes(Bmpi,A->rmap->n,A->cmap->n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(Bmpi,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(Bmpi);CHKERRQ(ierr);
  ierr = MatGetSize(A,&nrows,&ncols);CHKERRQ(ierr);
  for (i=0; i<nrows; i++) {
    PetscInt erow,ecol;
    P2RO(A,0,i,&rrank,&ridx);
    RO2E(A,0,rrank,ridx,&erow);
    if (rrank < 0 || ridx < 0 || erow < 0) SETERRQ(comm,PETSC_ERR_PLIB,"Incorrect row translation");
    for (j=0; j<ncols; j++) {
      P2RO(A,1,j,&crank,&cidx);
      RO2E(A,1,crank,cidx,&ecol);
      if (crank < 0 || cidx < 0 || ecol < 0) SETERRQ(comm,PETSC_ERR_PLIB,"Incorrect col translation");
      v = a->emat->Get(erow,ecol);
      ierr = MatSetValues(Bmpi,1,&i,1,&j,(PetscScalar *)&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Bmpi,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Bmpi,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(A,Bmpi);CHKERRQ(ierr);
  } else {
    *B = Bmpi;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqAIJ_ElemDense"
PETSC_EXTERN PetscErrorCode MatConvert_SeqAIJ_ElemDense(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat               mat_elemental;
  PetscErrorCode    ierr;
  PetscInt          M=A->rmap->N,N=A->cmap->N,row,ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A), &mat_elemental);CHKERRQ(ierr);
  ierr = MatSetSizes(mat_elemental,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetType(mat_elemental,MATELEMDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(mat_elemental);CHKERRQ(ierr);
  for (row=0; row<M; row++) {
    ierr = MatGetRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
    /* PETSc-Elemental interaface uses axpy for setting off-processor entries, only ADD_VALUES is allowed */
    ierr = MatSetValues(mat_elemental,1,&row,ncols,cols,vals,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat_elemental, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat_elemental, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(A,mat_elemental);CHKERRQ(ierr);
  } else {
    *newmat = mat_elemental;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvert_MPIAIJ_ElemDense"
PETSC_EXTERN PetscErrorCode MatConvert_MPIAIJ_ElemDense(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat               mat_elemental;
  PetscErrorCode    ierr;
  PetscInt          row,ncols,rstart=A->rmap->rstart,rend=A->rmap->rend,j;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A), &mat_elemental);CHKERRQ(ierr);
  ierr = MatSetSizes(mat_elemental,PETSC_DECIDE,PETSC_DECIDE,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(mat_elemental,MATELEMDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(mat_elemental);CHKERRQ(ierr);
  for (row=rstart; row<rend; row++) {
    ierr = MatGetRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
    for (j=0; j<ncols; j++) {
      /* PETSc-ElemDense interaface uses axpy for setting off-processor entries, only ADD_VALUES is allowed */
      ierr = MatSetValues(mat_elemental,1,&row,1,&cols[j],&vals[j],ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat_elemental, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat_elemental, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(A,mat_elemental);CHKERRQ(ierr);
  } else {
    *newmat = mat_elemental;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_ElemDense"
static PetscErrorCode MatDestroy_ElemDense(Mat A)
{
  Mat_ElemDense      *a = (Mat_ElemDense*)A->data;
  PetscErrorCode     ierr;
  Mat_ElemDense_Grid *commgrid;
  PetscBool          flg;
  MPI_Comm           icomm;

  PetscFunctionBegin;
  a->interface->Detach();
  delete a->interface;
  delete a->esubmat;
  delete a->emat;

  El::mpi::Comm cxxcomm(PetscObjectComm((PetscObject)A));
  ierr = PetscCommDuplicate(cxxcomm.comm,&icomm,NULL);CHKERRQ(ierr);
  ierr = MPI_Attr_get(icomm,Petsc_Elemental_keyval,(void**)&commgrid,(int*)&flg);CHKERRQ(ierr);
  if (--commgrid->grid_refct == 0) {
    delete commgrid->grid;
    ierr = PetscFree(commgrid);CHKERRQ(ierr);
  }
  ierr = PetscCommDestroy(&icomm);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatGetOwnershipIS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverPackage_C",NULL);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetUp_ElemDense"
PetscErrorCode MatSetUp_ElemDense(Mat A)
{
  Mat_ElemDense  *a = (Mat_ElemDense*)A->data;
  PetscErrorCode ierr;
  PetscMPIInt    rsize,csize;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);

  a->emat->Resize(A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  El::Zero(*a->emat);

  ierr = MPI_Comm_size(A->rmap->comm,&rsize);CHKERRQ(ierr);
  ierr = MPI_Comm_size(A->cmap->comm,&csize);CHKERRQ(ierr);
  if (csize != rsize) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Cannot use row and column communicators of different sizes");
  a->commsize = rsize;
  a->mr[0] = A->rmap->N % rsize; if (!a->mr[0]) a->mr[0] = rsize;
  a->mr[1] = A->cmap->N % csize; if (!a->mr[1]) a->mr[1] = csize;
  a->m[0]  = A->rmap->N / rsize + (a->mr[0] != rsize);
  a->m[1]  = A->cmap->N / csize + (a->mr[1] != csize);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyBegin_ElemDense"
PetscErrorCode MatAssemblyBegin_ElemDense(Mat A, MatAssemblyType type)
{
  Mat_ElemDense  *a = (Mat_ElemDense*)A->data;

  PetscFunctionBegin;
  a->interface->Detach();
  a->interface->Attach(El::LOCAL_TO_GLOBAL,*(a->emat));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_ElemDense"
PetscErrorCode MatAssemblyEnd_ElemDense(Mat A, MatAssemblyType type)
{
  PetscFunctionBegin;
  /* Currently does nothing */
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {
       MatSetValues_ElemDense,
       0,
       0,
       MatMult_ElemDense,
/* 4*/ MatMultAdd_ElemDense,
       MatMultTranspose_ElemDense,
       MatMultTransposeAdd_ElemDense,
       MatSolve_ElemDense,
       MatSolveAdd_ElemDense,
       0, //MatSolveTranspose_ElemDense,
/*10*/ 0, //MatSolveTransposeAdd_ElemDense,
       MatLUFactor_ElemDense,
       MatCholeskyFactor_ElemDense,
       0,
       MatTranspose_ElemDense,
/*15*/ MatGetInfo_ElemDense,
       0,
       MatGetDiagonal_ElemDense,
       MatDiagonalScale_ElemDense,
       MatNorm_ElemDense,
/*20*/ MatAssemblyBegin_ElemDense,
       MatAssemblyEnd_ElemDense,
       0, //MatSetOption_ElemDense,
       MatZeroEntries_ElemDense,
/*24*/ 0,
       MatLUFactorSymbolic_ElemDense,
       MatLUFactorNumeric_ElemDense,
       MatCholeskyFactorSymbolic_ElemDense,
       MatCholeskyFactorNumeric_ElemDense,
/*29*/ MatSetUp_ElemDense,
       0,
       0,
       0,
       0,
/*34*/ MatDuplicate_ElemDense,
       0,
       0,
       0,
       0,
/*39*/ MatAXPY_ElemDense,
       0,
       0,
       0,
       MatCopy_ElemDense,
/*44*/ 0,
       MatScale_ElemDense,
       0,
       0,
       0,
/*49*/ 0,
       0,
       0,
       0,
       0,
/*54*/ 0,
       0,
       0,
       0,
       0,
/*59*/ 0,
       MatDestroy_ElemDense,
       MatView_ElemDense,
       0,
       0,
/*64*/ 0,
       0,
       0,
       0,
       0,
/*69*/ 0,
       0,
       MatConvert_ElemDense_Dense,
       0,
       0,
/*74*/ 0,
       0,
       0,
       0,
       0,
/*79*/ 0,
       0,
       0,
       0,
       0,
/*84*/ 0,
       0,
       0,
       0,
       0,
/*89*/ MatMatMult_ElemDense,
       MatMatMultSymbolic_ElemDense,
       MatMatMultNumeric_ElemDense,
       0,
       0,
/*94*/ 0,
       MatMatTransposeMult_ElemDense,
       MatMatTransposeMultSymbolic_ElemDense,
       MatMatTransposeMultNumeric_ElemDense,
       0,
/*99*/ 0,
       0,
       0,
       MatConjugate_ElemDense,
       0,
/*104*/0,
       0,
       0,
       0,
       0,
/*109*/MatMatSolve_ElemDense,
       0,
       0,
       0,
       0,
/*114*/0,
       0,
       0,
       0,
       0,
/*119*/0,
       MatHermitianTranspose_ElemDense,
       0,
       0,
       0,
/*124*/0,
       0,
       0,
       0,
       0,
/*129*/0,
       0,
       0,
       0,
       0,
/*134*/0,
       0,
       0,
       0,
       0
};

/*MC
   MATELEMDENSE = "elemdense" - A matrix type for dense matrices using the Elemental package

   Options Database Keys:
+ -mat_type elemdense - sets the matrix type to "elemdense" during a call to MatSetFromOptions()
- -mat_elemdense_grid_height - sets Grid Height for 2D cyclic ordering of internal matrix

  Level: beginner

.seealso: MATDENSE
M*/

#undef __FUNCT__
#define __FUNCT__ "MatCreate_ElemDense"
PETSC_EXTERN PetscErrorCode MatCreate_ElemDense(Mat A)
{
  Mat_ElemDense      *a;
  PetscErrorCode     ierr;
  PetscBool          flg,flg1;
  Mat_ElemDense_Grid *commgrid;
  MPI_Comm           icomm;
  PetscInt           optv1;

  PetscFunctionBegin;
  ierr = PetscElementalInitializePackage();CHKERRQ(ierr);
  ierr = PetscMemcpy(A->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  A->insertmode = NOT_SET_VALUES;

  ierr = PetscNewLog(A,&a);CHKERRQ(ierr);
  A->data = (void*)a;

  /* Set up the elemental matrix */
  El::mpi::Comm cxxcomm(PetscObjectComm((PetscObject)A));

  /* Grid needs to be shared between multiple Mats on the same communicator, implement by attribute caching on the MPI_Comm */
  if (Petsc_Elemental_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Elemental_keyval,(void*)0);
  }
  ierr = PetscCommDuplicate(cxxcomm.comm,&icomm,NULL);CHKERRQ(ierr);
  ierr = MPI_Attr_get(icomm,Petsc_Elemental_keyval,(void**)&commgrid,(int*)&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscNewLog(A,&commgrid);CHKERRQ(ierr);

    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"Elemental Options","Mat");CHKERRQ(ierr);
    /* displayed default grid sizes (CommSize,1) are set by us arbitrarily until El::Grid() is called */
    ierr = PetscOptionsInt("-mat_elemdense_grid_height","Grid Height","None",El::mpi::Size(cxxcomm),&optv1,&flg1);CHKERRQ(ierr);
    if (flg1) {
      if (El::mpi::Size(cxxcomm) % optv1 != 0) {
        SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Grid Height %D must evenly divide CommSize %D",optv1,(PetscInt)El::mpi::Size(cxxcomm));
      }
      commgrid->grid = new El::Grid(cxxcomm,optv1); /* use user-provided grid height */
    } else {
      commgrid->grid = new El::Grid(cxxcomm); /* use Elemental default grid sizes */
    }
    commgrid->grid_refct = 1;
    ierr = MPI_Attr_put(icomm,Petsc_Elemental_keyval,(void*)commgrid);CHKERRQ(ierr);
    PetscOptionsEnd();
  } else {
    commgrid->grid_refct++;
  }
  ierr = PetscCommDestroy(&icomm);CHKERRQ(ierr);
  a->grid      = commgrid->grid;
  a->emat      = new El::DistMatrix<PetscElemScalar>(*a->grid);
  a->esubmat   = new El::Matrix<PetscElemScalar>(1,1);
  a->interface = new El::AxpyInterface<PetscElemScalar>;
  a->pivot     = new El::DistMatrix<PetscInt,El::VC,El::STAR>;

  /* build cache for off array entries formed */
  a->interface->Attach(El::LOCAL_TO_GLOBAL,*(a->emat));

  ierr = PetscObjectComposeFunction((PetscObject)A,"MatGetOwnershipIS_C",MatGetOwnershipIS_ElemDense);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATELEMDENSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

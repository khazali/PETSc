#include <../src/mat/impls/elemental/matelemsparseimpl.h> /*I "petscmat.h" I*/

/*
 Provides an interface to Elemental sparse solver
 This code has been copied and modified from the previous interface to Clique
*/

/*
  MatConvertToElemSparse: Convert Petsc aij matrix to ElemSparse matrix

  input:
+   A     - matrix in seqaij or mpiaij format
-   reuse - denotes if the destination matrix is to be created or reused. Currently
            MAT_REUSE_MATRIX is only supported for inplace conversion, otherwise use MAT_INITIAL_MATRIX.

  output:
.   cliq - ElemSparse context
*/
#undef __FUNCT__
#define __FUNCT__ "MatConvertToElemSparse"
PetscErrorCode MatConvertToElemSparse(Mat A,MatReuse reuse,Mat_ElemSparse *cliq)
{
  PetscErrorCode                        ierr;
  PetscInt                              i,j,rstart,rend,ncols;
  const PetscInt                        *cols;
  const PetscElemScalar                 *vals;
  El::DistSparseMatrix<PetscElemScalar> *cmat;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX){
    /* create ElemSparse matrix */
    cmat = new El::DistSparseMatrix<PetscElemScalar>(A->rmap->N,cliq->cliq_comm);
    cliq->cmat = cmat;
  } else {
    cmat = cliq->cmat;
  }
  /* fill matrix values */
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  const int firstLocalRow = cmat->FirstLocalRow();
  const int localHeight = cmat->LocalHeight();
  if (rstart != firstLocalRow || rend-rstart != localHeight) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"matrix rowblock distribution does not match");

  //cmat->Reserve( 7*localHeight ); ??? TODO PREALLOCATION IS MISSING
  for (i=rstart; i<rend; i++){
    ierr = MatGetRow(A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    for (j=0; j<ncols; j++){
      cmat->QueueLocalUpdate(i,cols[j],vals[j]);
    }
    ierr = MatRestoreRow(A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
  }
  cmat->MakeConsistent();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_ElemSparse"
static PetscErrorCode MatMult_ElemSparse(Mat A,Vec X,Vec Y)
{
  PetscErrorCode        ierr;
  PetscInt              i;
  const PetscElemScalar *x;
  Mat_ElemSparse            *cliq=(Mat_ElemSparse*)A->spptr;
  El::DistSparseMatrix<PetscElemScalar> *cmat=cliq->cmat;
  El::mpi::Comm cxxcomm(PetscObjectComm((PetscObject)A));

  PetscFunctionBegin;
  if (!cmat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"ElemSparse matrix cmat is not created yet");
  ierr = VecGetArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);

  El::DistMultiVec<PetscElemScalar> xc(A->cmap->N,1,cxxcomm);
  El::DistMultiVec<PetscElemScalar> yc(A->rmap->N,1,cxxcomm);
  for (i=0; i<A->cmap->n; i++) {
    xc.SetLocal(i,0,x[i]);
  }
  El::Multiply(El::NORMAL,1.0,*cmat,xc,0.0,yc);
  ierr = VecRestoreArrayRead(X,(const PetscScalar **)&x);CHKERRQ(ierr);

  for (i=0; i<A->cmap->n; i++) {
    ierr = VecSetValueLocal(Y,i,yc.GetLocal(i,0),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(Y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_ElemSparse"
PetscErrorCode MatView_ElemSparse(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscViewerFormat format;
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = PetscViewerASCIIPrintf(viewer,"ElemSparse run parameters:\n");CHKERRQ(ierr);
    } else if (format == PETSC_VIEWER_DEFAULT) { /* matrix A is factored matrix, remove this block */
      Mat Aaij;
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)viewer),"ElemSparse matrix\n");CHKERRQ(ierr);
      ierr = MatComputeExplicitOperator(A,&Aaij);CHKERRQ(ierr);
      ierr = MatView(Aaij,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = MatDestroy(&Aaij);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_ElemSparse"
PetscErrorCode MatDestroy_ElemSparse(Mat A)
{
  PetscErrorCode ierr;
  Mat_ElemSparse     *cliq=(Mat_ElemSparse*)A->spptr;

  PetscFunctionBegin;
  printf("MatDestroy_ElemSparse ...\n");
  if (cliq && cliq->CleanUpClique) {
    /* Terminate instance, deallocate memories */
    printf("MatDestroy_ElemSparse ... destroy clique struct \n");
    ierr = PetscCommDestroy(&(cliq->cliq_comm));CHKERRQ(ierr);
    // free cmat here
    delete cliq->cmat;
    delete cliq->frontTree;
    delete cliq->rhs;
    delete cliq->xNodal;
    delete cliq->info;
    delete cliq->inverseMap;
  }
  if (cliq && cliq->Destroy) {
    ierr = cliq->Destroy(A);CHKERRQ(ierr);
  }
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);

  /* clear composed functions */
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverPackage_C",NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolve_ElemSparse"
PetscErrorCode MatSolve_ElemSparse(Mat A,Vec B,Vec X)
{
  PetscErrorCode        ierr;
  PetscInt              i,rank;
  const PetscElemScalar *b;
  Mat_ElemSparse            *cliq=(Mat_ElemSparse*)A->spptr;
  El::DistMultiVec<PetscElemScalar> *bc=cliq->rhs;
  El::DistNodalMultiVec<PetscElemScalar> *xNodal=cliq->xNodal;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(B,(const PetscScalar **)&b);CHKERRQ(ierr);
  for (i=0; i<A->rmap->n; i++) {
    bc->SetLocal(i,0,b[i]);
  }
  ierr = VecRestoreArrayRead(B,(const PetscScalar **)&b);CHKERRQ(ierr);

  xNodal->Pull( *cliq->inverseMap, *cliq->info, *bc );
  El::Solve( *cliq->info, *cliq->frontTree, *xNodal);
  xNodal->Push( *cliq->inverseMap, *cliq->info, *bc );

  ierr = MPI_Comm_rank(cliq->cliq_comm,&rank);CHKERRQ(ierr);
  for (i=0; i<bc->LocalHeight(); i++) {
    ierr = VecSetValue(X,rank*bc->Blocksize()+i,bc->GetLocal(i,0),INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(X);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorNumeric_ElemSparse"
PetscErrorCode MatCholeskyFactorNumeric_ElemSparse(Mat F,Mat A,const MatFactorInfo *info)
{
  PetscErrorCode    ierr;
  Mat_ElemSparse        *cliq=(Mat_ElemSparse*)F->spptr;
  PETSC_UNUSED
  El::DistSparseMatrix<PetscElemScalar> *cmat;

  PetscFunctionBegin;
  cmat = cliq->cmat;
  if (cliq->matstruc == SAME_NONZERO_PATTERN){ /* successing numerical factorization */
    /* Update cmat */
    ierr = MatConvertToElemSparse(A,MAT_REUSE_MATRIX,cliq);CHKERRQ(ierr);
  }

  /* Numeric factorization */
  El::LDL( *cliq->info, *cliq->frontTree, El::LDL_1D);
  //L.frontType = cliq::SYMM_2D;

  // refactor
  //cliq::ChangeFrontType( *cliq->frontTree, cliq::LDL_2D );
  //*(cliq->frontTree.frontType) = cliq::LDL_2D;
  //cliq::LDL( *cliq->info, *cliq->frontTree, cliq::LDL_2D );

  cliq->matstruc = SAME_NONZERO_PATTERN;
  F->assembled   = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorSymbolic_ElemSparse"
PetscErrorCode MatCholeskyFactorSymbolic_ElemSparse(Mat F,Mat A,IS r,const MatFactorInfo *info)
{
  PetscErrorCode                        ierr;
  Mat_ElemSparse                        *Acliq=(Mat_ElemSparse*)F->spptr;
  El::DistSparseMatrix<PetscElemScalar> *cmat;
  El::DistSeparatorTree                 sepTree;
  El::DistMap                           map;
  El::BisectCtrl                        ctrl;

  PetscFunctionBegin;
  /* Convert A to Aclique */
  ierr = MatConvertToElemSparse(A,MAT_INITIAL_MATRIX,Acliq);CHKERRQ(ierr);
  cmat = Acliq->cmat;

  ctrl.sequential = PETSC_TRUE;
  ctrl.numSeqSeps = Acliq->numSeqSeps;
  ctrl.numDistSeps = Acliq->numDistSeps;
  ctrl.cutoff = Acliq->cutoff;
  El::NestedDissection( cmat->DistGraph(), map, sepTree, *Acliq->info, ctrl);
  map.FormInverse( *Acliq->inverseMap );
  //Acliq->frontTree = new El::DistSymmFrontTree<PetscElemScalar>( El::TRANSPOSE, *cmat, map, sepTree, *Acliq->info );
  Acliq->frontTree = new El::DistSymmFrontTree<PetscElemScalar>( *cmat, map, sepTree, *Acliq->info );

  Acliq->matstruc      = DIFFERENT_NONZERO_PATTERN;
  Acliq->CleanUpClique = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*MC
     MATSOLVERCLIQUE  - A solver package providing direct solvers for distributed
  and sequential matrices via the external package Elemental.

  Options Database Keys:
+ -mat_clique_    -
- -mat_clique_ <integer> -

  Level: beginner

.seealso: PCFactorSetMatSolverPackage(), MatSolverPackage

M*/

#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_aij_clique"
PETSC_EXTERN PetscErrorCode MatGetFactor_aij_clique(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  Mat_ElemSparse     *cliq;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  if (ftype == MAT_FACTOR_CHOLESKY){
    B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_ElemSparse;
    B->ops->choleskyfactornumeric  = MatCholeskyFactorNumeric_ElemSparse;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported");

  ierr = PetscNewLog(B,&cliq);CHKERRQ(ierr);
  B->spptr            = (void*)cliq;
  El::mpi::Comm cxxcomm(PetscObjectComm((PetscObject)A));
  ierr = PetscCommDuplicate(cxxcomm.comm,&(cliq->cliq_comm),NULL);CHKERRQ(ierr);
  cliq->rhs           = new El::DistMultiVec<PetscElemScalar>(A->rmap->N,1,cliq->cliq_comm);
  cliq->xNodal        = new El::DistNodalMultiVec<PetscElemScalar>();
  cliq->info          = new El::DistSymmInfo;
  cliq->inverseMap    = new El::DistMap;
  cliq->CleanUpClique = PETSC_FALSE;
  cliq->Destroy       = B->ops->destroy;

  B->ops->view    = MatView_ElemSparse;
  B->ops->mult    = MatMult_ElemSparse; /* for cliq->cmat */
  B->ops->solve   = MatSolve_ElemSparse;

  B->ops->destroy = MatDestroy_ElemSparse;
  B->factortype   = ftype;
  B->assembled    = PETSC_FALSE;

  /* Set Clique options */
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"ElemSparse Options","Mat");CHKERRQ(ierr);
  cliq->cutoff      = 128;  /* maximum size of leaf node */
  cliq->numDistSeps = 1;    /* number of distributed separators to try */
  cliq->numSeqSeps  = 1;    /* number of sequential separators to try */
  PetscOptionsEnd();

  *F = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolverPackageRegister_ElemSparse"
PETSC_EXTERN PetscErrorCode MatSolverPackageRegister_ElemSparse(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverPackageRegister(MATSOLVERCLIQUE,MATMPIAIJ,        MAT_FACTOR_LU,MatGetFactor_aij_clique);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

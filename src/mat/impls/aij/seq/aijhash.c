#include <../src/mat/impls/aij/seq/aij.h>
#include <petsc/private/hash.h>

typedef struct {
  PetscInt row,col;
} IJKey;

#define IJKey_hash_equal(a,b) ((a).row == (b).row && (a).col == (b).col)
PETSC_STATIC_INLINE khint32_t IJKey_hash_func(IJKey k) {
  PetscInt64 x = k.row,y = k.col,cantor = (x+y)*(x+y+1)/2 + y;
  return kh_int64_hash_func((khint64_t)cantor);
}

KHASH_INIT(hij, IJKey, MatScalar*, 1, IJKey_hash_func, IJKey_hash_equal)

struct Mat_AIJHash {
  PetscSegBuffer segvalues;
  khash_t(hij) *hash;
};

PetscErrorCode MatCreateAIJHash(Mat A,MatAIJHash *aijhash)
{
  PetscErrorCode ierr;
  PetscInt rbs,cbs,m;

  PetscFunctionBegin;
  if (*aijhash) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Hash not empty");
  ierr = PetscMalloc1(1,aijhash);CHKERRQ(ierr);
  ierr = MatGetBlockSizes(A,&rbs,&cbs);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscSegBufferCreate(rbs*cbs*sizeof(MatScalar),m,&(*aijhash)->segvalues);CHKERRQ(ierr);
  (*aijhash)->hash = kh_init(hij);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetValuesAIJHash(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode imode,MatAIJHash aijhash)
{
  PetscErrorCode ierr;
  PetscInt i,j,rbs,cbs;

  PetscFunctionBegin;
  ierr = MatGetBlockSizes(A,&rbs,&cbs);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      IJKey key = {im[i]/rbs, in[j]/cbs};
      khiter_t iter;
      MatScalar *values;
      int ret;
      iter = kh_put(hij,aijhash->hash,key,&ret);
      if (ret) {                /* New entry */
        ierr = PetscSegBufferGet(aijhash->segvalues,1,&values);CHKERRQ(ierr);
        ierr = PetscMemzero(values,rbs*cbs*sizeof(values[0]));CHKERRQ(ierr);
        kh_val(aijhash->hash,iter) = values;
      } else {
        values = kh_val(aijhash->hash,iter);
      }
      if (imode == INSERT_VALUES) {
        values[(im[i]%rbs)*rbs + in[j]%cbs] = v[i*m+j];
      } else {
        values[(im[i]%rbs)*rbs + in[j]%cbs] += v[i*m+j];
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatUnpackAIJHash(Mat A,MatAssemblyType amode,MatAIJHash *aijhash)
{
  PetscErrorCode ierr;
  MatAIJHash aijh = *aijhash;
  khash_t(hij) *h = aijh->hash;
  khiter_t iter;
  PetscInt *nnz,m,rbs,cbs;

  PetscFunctionBegin;
  *aijhash = NULL;
  ierr = MatAssemblyEnd(A,amode);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = MatGetBlockSizes(A,&rbs,&cbs);CHKERRQ(ierr);
  ierr = PetscCalloc1(m/rbs,&nnz);CHKERRQ(ierr);
  for (iter=kh_begin(h); iter!=kh_end(h); iter++) {
    if (kh_exist(h,iter)) nnz[kh_key(h,iter).row]++;
  }
  ierr = PetscIntView(m/rbs,nnz,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatXAIJSetPreallocation(A,rbs,nnz,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  for (iter=kh_begin(h); iter!=kh_end(h); iter++) {
    if (kh_exist(h,iter)) {
      IJKey b = kh_key(h,iter);
      const MatScalar *values = kh_value(h,iter);
      // ierr = MatSetValues(A,1,&b.row,1,&b.col,values,A->insertmode);CHKERRQ(ierr);
      ierr = MatSetValuesBlocked(A,1,&b.row,1,&b.col,values,A->insertmode);CHKERRQ(ierr);
    }
  }
  kh_destroy(hij,h);
  ierr = PetscSegBufferDestroy(&aijh->segvalues);CHKERRQ(ierr);
  ierr = PetscFree(aijh);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,amode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,amode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

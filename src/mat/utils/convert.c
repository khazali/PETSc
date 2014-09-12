
#include <petsc-private/matimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatConvert_Basic"
/*
  MatConvert_Basic - Converts from any input format to another format. For
  parallel formats, the new matrix distribution is determined by PETSc.

  Does not do preallocation so in general will be slow
 */
PetscErrorCode MatConvert_Basic(Mat mat, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat               M;
  const PetscScalar *vwork;
  PetscErrorCode    ierr;
  PetscInt          i,j,nz,m,n,rstart,rend,lm,ln,prbs,pcbs,cstart,cend,*dnz,*onz;
  const PetscInt    *cwork;
  PetscBool         isseqsbaij,ismpisbaij,isseqbaij,ismpibaij,isseqdense,ismpidense;

  PetscFunctionBegin;
  ierr = MatGetSize(mat,&m,&n);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mat,&lm,&ln);CHKERRQ(ierr);

  if (ln == n) ln = PETSC_DECIDE; /* try to preserve column ownership */

  ierr = MatCreate(PetscObjectComm((PetscObject)mat),&M);CHKERRQ(ierr);
  ierr = MatSetSizes(M,lm,ln,m,n);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(M,mat,mat);CHKERRQ(ierr);
  ierr = MatSetType(M,newtype);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(mat,&rstart,&rend);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)M,MATSEQSBAIJ,&isseqsbaij);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)M,MATMPISBAIJ,&ismpisbaij);CHKERRQ(ierr);
  if (isseqsbaij || ismpisbaij) {ierr = MatSetOption(M,MAT_IGNORE_LOWER_TRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);}
  ierr = PetscObjectTypeCompare((PetscObject)M,MATSEQBAIJ,&isseqbaij);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)M,MATMPIBAIJ,&ismpibaij);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)M,MATSEQDENSE,&isseqdense);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)M,MATMPIDENSE,&ismpidense);CHKERRQ(ierr);

  if (isseqdense) {
    ierr = MatSeqDenseSetPreallocation(M,NULL);CHKERRQ(ierr);
  } else if (ismpidense) {
    ierr = MatMPIDenseSetPreallocation(M,NULL);CHKERRQ(ierr);
  } else {
    PetscSegBuffer seg;
    PetscInt cnt;
    /* Preallocation block sizes.  (S)BAIJ matrices will have one index per block. */
    prbs = (isseqbaij || ismpibaij || isseqsbaij || ismpisbaij) ? PetscAbs(M->rmap->bs) : 1;
    pcbs = (isseqbaij || ismpibaij || isseqsbaij || ismpisbaij) ? PetscAbs(M->cmap->bs) : 1;

    ierr = PetscMalloc2(lm/prbs,&dnz,lm/prbs,&onz);CHKERRQ(ierr);
    ierr = MatGetOwnershipRangeColumn(mat,&cstart,&cend);CHKERRQ(ierr);
    ierr = PetscSegBufferCreate(sizeof(PetscInt),100,&seg);CHKERRQ(ierr);
    for (i=0; i*prbs<lm; i++,cnt=0) {
      PetscInt ii,*bcols;
      for (ii=i*prbs; ii<PetscMin((i+1)*prbs,lm); ii++) {
        ierr = MatGetRow(mat,rstart+ii,&nz,&cwork,NULL);CHKERRQ(ierr);
        ierr = PetscSegBufferGet(seg,nz,&bcols);CHKERRQ(ierr);
        for (j=0; j<nz; j++) {
          bcols[cnt++] = cwork[j] / pcbs;
        }
        ierr = MatRestoreRow(mat,rstart+ii,&nz,&cwork,NULL);CHKERRQ(ierr);
      }
      ierr = PetscSegBufferExtractInPlace(seg,&bcols);CHKERRQ(ierr);
      ierr = PetscSortRemoveDupsInt(&cnt,bcols);CHKERRQ(ierr);
      dnz[i] = 0;
      onz[i] = 0;
      for (j=0; j<cnt; j++) {
        if (cstart <= bcols[j]*pcbs && bcols[j]*pcbs < cend) dnz[i]++;
        else                                                 onz[i]++;
      }
    }
    ierr = PetscSegBufferDestroy(&seg);CHKERRQ(ierr);
    ierr = MatXAIJSetPreallocation(M,PETSC_DECIDE,dnz,onz,dnz,onz);CHKERRQ(ierr);
    ierr = PetscFree2(dnz,onz);CHKERRQ(ierr);
  }

  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(mat,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
    ierr = MatSetValues(M,1,&i,nz,cwork,vwork,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(mat,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(mat,M);CHKERRQ(ierr);
  } else {
    *newmat = M;
  }
  PetscFunctionReturn(0);
}

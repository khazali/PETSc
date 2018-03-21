
/*
   Support for the parallel AIJ matrix vector multiply
*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/isimpl.h>    /* needed because accesses data structure of ISLocalToGlobalMapping directly */

/* Do node-reorganization of an MPI matrix, which moves on-node nonzeros of off-diag B to diag A.
  Inpute Parameter:
  +mat       - the matrix
  .bnodecols - number of on-node nonzero columns of B
  -neworder  - [] imagine nonzero columns of B are re-ordered such that the first nodecols columns are on node,
               and the remaining are off-node, then neworder[] gives the ordering. The caller ensures neworder[]
               has a proper size.
  Notes:
   One can think nodecols and neworder[] specify a node-reorg plan

  .seealso PetscLayoutNodeReorderIndices()
 */
PetscErrorCode MatNodeReorg_MPIAIJ_Private(Mat mat,PetscInt bnodecols,const PetscInt *neworder)
{
  PetscErrorCode ierr;
  Mat            Anew,Bnew;
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)mat->data;
  Mat_SeqAIJ     *A   = (Mat_SeqAIJ*)(aij->A->data);
  Mat_SeqAIJ     *B   = (Mat_SeqAIJ*)(aij->B->data);
  PetscInt       m = mat->rmap->n,an,bn,*onnodenz,*offnodenz,newpos;
  PetscInt       i,j,col,*Ai = A->i,*Aj = A->j,*Ailen = A->ilen,*Bi=B->i,*Bj = B->j,*Bilen = B->ilen;
  PetscScalar    v,*Aa = A->a,*Ba = B->a;

  PetscFunctionBegin;
  /* return if none of B's columns is on node */
  if (!bnodecols) PetscFunctionReturn(0);

  /* count on/off-node nonzeros of rows of B, to be used in new matrices' preallocation */
  ierr = PetscCalloc2(m,&onnodenz,m,&offnodenz);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    for (j=0; j<Bilen[i]; j++) {
      if (neworder[Bj[Bi[i]]+j] < bnodecols) onnodenz[i]++;
      else offnodenz[i]++;
    }
  }

  /* create new A and new B */
  for (i=0; i<m; i++) onnodenz[i] += Ailen[i]; /* add original nonzeros in A */
  an   = aij->A->cmap->n + bnodecols;
  ierr = MatCreate(PETSC_COMM_SELF,&Anew);CHKERRQ(ierr);
  ierr = MatSetSizes(Anew,m,an,m,an);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(Anew,mat,mat);CHKERRQ(ierr);
  ierr = MatSetType(Anew,((PetscObject)aij->A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Anew,0,onnodenz);CHKERRQ(ierr);

  bn   = aij->B->cmap->n - bnodecols;
  ierr = MatCreate(PETSC_COMM_SELF,&Bnew);CHKERRQ(ierr);
  ierr = MatSetSizes(Bnew,m,bn,m,bn);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(Bnew,mat,mat);CHKERRQ(ierr);
  ierr = MatSetType(Bnew,((PetscObject)aij->B)->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Bnew,0,offnodenz);CHKERRQ(ierr);

  ((Mat_SeqAIJ*)Anew->data)->nonew = A->nonew;
  ((Mat_SeqAIJ*)Bnew->data)->nonew = B->nonew;
  Anew->nonzerostate               = aij->A->nonzerostate;
  Bnew->nonzerostate               = aij->B->nonzerostate;

  /* copy A to Anew */
  for (i=0; i<m; i++) { ierr = MatSetValues(Anew,1,&i,Ailen[i],&Aj[Ai[i]],&Aa[Ai[i]],INSERT_VALUES);CHKERRQ(ierr); }

  /* copy B's nonzeros to Anew or Bnew based on their new index */
  for (i=0; i<m; i++) {
    for (j=0; j<Bilen[i]; j++) {
      newpos = neworder[Bj[Bi[i]+j]]; /* new order of the old column index in B */
      v = Ba[Bi[i]+j];
      if (newpos < bnodecols) {
        col  = aij->A->cmap->n + newpos;
        ierr = MatSetValues(Anew,1,&i,1,&col,&v,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        col  = newpos - bnodecols;
        ierr = MatSetValues(Bnew,1,&i,1,&col,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  /* destroy old and assign new */
  ierr = PetscFree2(onnodenz,offnodenz);CHKERRQ(ierr);
  ierr = MatDestroy(&aij->A);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)Anew);CHKERRQ(ierr);
  ierr = MatDestroy(&aij->B);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)Bnew);CHKERRQ(ierr);

  aij->A             = Anew;
  aij->B             = Bnew;
  aij->nodereorged   = PETSC_TRUE;
  aij->nodecols      = bnodecols;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUpMultiply_MPIAIJ(Mat mat)
{
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)mat->data;
  Mat_SeqAIJ     *B   = (Mat_SeqAIJ*)(aij->B->data);
  PetscErrorCode ierr;
  PetscInt       i,j,*aj = B->j,bcols = 0,*earray,*neworder,acols,bnodecols,boffnodecols;
  IS             from,to;
  Vec            gvec;
  PetscBool      matmult_nodereorg = PETSC_FALSE;
#if defined(PETSC_USE_CTABLE)
  PetscTable         gid1_lid1;
  PetscTablePosition tpos;
  PetscInt           gid,lid;
#else
  PetscInt N = mat->cmap->N,*indices;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_CTABLE)
  /* use a table */
  ierr = PetscTableCreate(aij->B->rmap->n,mat->cmap->N+1,&gid1_lid1);CHKERRQ(ierr);
  for (i=0; i<aij->B->rmap->n; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      PetscInt data,gid1 = aj[B->i[i] + j] + 1;
      ierr = PetscTableFind(gid1_lid1,gid1,&data);CHKERRQ(ierr);
      if (!data) {
        /* one based table */
        ierr = PetscTableAdd(gid1_lid1,gid1,++bcols,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  /* form array of columns we need */
  ierr = PetscMalloc1(bcols+1,&earray);CHKERRQ(ierr);
  ierr = PetscTableGetHeadPosition(gid1_lid1,&tpos);CHKERRQ(ierr);
  while (tpos) {
    ierr = PetscTableGetNext(gid1_lid1,&tpos,&gid,&lid);CHKERRQ(ierr);
    gid--;
    lid--;
    earray[lid] = gid;
  }
  ierr = PetscSortInt(bcols,earray);CHKERRQ(ierr); /* sort, and rebuild */
  ierr = PetscTableRemoveAll(gid1_lid1);CHKERRQ(ierr);
  for (i=0; i<bcols; i++) {
    ierr = PetscTableAdd(gid1_lid1,earray[i]+1,i+1,INSERT_VALUES);CHKERRQ(ierr);
  }
  /* compact out the extra columns in B */
  for (i=0; i<aij->B->rmap->n; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      PetscInt gid1 = aj[B->i[i] + j] + 1;
      ierr = PetscTableFind(gid1_lid1,gid1,&lid);CHKERRQ(ierr);
      lid--;
      aj[B->i[i] + j] = lid;
    }
  }
  aij->B->cmap->n  = aij->B->cmap->N = bcols;
  aij->B->cmap->bs = 1;

  ierr = PetscLayoutSetUp((aij->B->cmap));CHKERRQ(ierr);
  ierr = PetscTableDestroy(&gid1_lid1);CHKERRQ(ierr);
#else
  /* Make an array as long as the number of columns */
  /* mark those columns that are in aij->B */
  ierr = PetscCalloc1(N+1,&indices);CHKERRQ(ierr);
  for (i=0; i<aij->B->rmap->n; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      if (!indices[aj[B->i[i] + j]]) bcols++;
      indices[aj[B->i[i] + j]] = 1;
    }
  }

  /* form array of columns we need */
  ierr  = PetscMalloc1(bcols+1,&earray);CHKERRQ(ierr);
  bcols = 0;
  for (i=0; i<N; i++) {
    if (indices[i]) earray[bcols++] = i;
  }

  /* make indices now point into garray */
  for (i=0; i<bcols; i++) {
    indices[earray[i]] = i;
  }

  /* compact out the extra columns in B */
  for (i=0; i<aij->B->rmap->n; i++) {
    for (j=0; j<B->ilen[i]; j++) {
      aj[B->i[i] + j] = indices[aj[B->i[i] + j]];
    }
  }
  aij->B->cmap->n  = aij->B->cmap->N = bcols;
  aij->B->cmap->bs = 1;

  ierr = PetscLayoutSetUp((aij->B->cmap));CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
#endif

  /* node-reorder earray and then use the output plan to node-reorg mat */
  bnodecols         = 0; /* B's on-node cols */
  matmult_nodereorg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-matmult_nodereorg",&matmult_nodereorg,NULL);CHKERRQ(ierr);
  if (matmult_nodereorg) {
    ierr = PetscMalloc1(bcols,&neworder);CHKERRQ(ierr);
    ierr = PetscLayoutNodeReorderIndices(PetscObjectComm((PetscObject)mat),mat->cmap,bcols,earray,&bnodecols,neworder);CHKERRQ(ierr);
    ierr = MatNodeReorg_MPIAIJ_Private(mat,bnodecols,neworder);CHKERRQ(ierr);
    ierr = PetscFree(neworder);CHKERRQ(ierr);
  }

  /* set up farray & garray from earray */
  acols        = mat->cmap->n;      /* A's cols (diag part) */
  boffnodecols = bcols - bnodecols; /* B's off-node cols */

  ierr = PetscMalloc1(acols+bnodecols+boffnodecols,&aij->farray);CHKERRQ(ierr);
  aij->garray = aij->farray + acols + bnodecols;

  for (i=0; i<acols; i++)        aij->farray[i]       = mat->cmap->rstart + i;
  for (i=0; i<bnodecols; i++)    aij->farray[acols+i] = earray[i];
  for (i=0; i<boffnodecols; i++) aij->garray[i]       = earray[bnodecols+i];

  if (!aij->lvec) {
    /* create local vector that is used to scatter into */
    ierr = VecCreateSeq(PETSC_COMM_SELF,boffnodecols,&aij->lvec);CHKERRQ(ierr);
  } else {
    ierr = VecGetSize(aij->lvec,&boffnodecols);CHKERRQ(ierr);
  }

  /* build the on-node vecscatter context by building a node-ghosted vector */
  if (matmult_nodereorg) {
    ierr = VecDestroy(&aij->ngvec);CHKERRQ(ierr);
    ierr = VecCreateGhost(PetscObjectComm((PetscObject)mat),mat->cmap->n,mat->cmap->N,bnodecols,&aij->farray[acols],&aij->ngvec);CHKERRQ(ierr);
  }

  /* build the off-node vecscatter context */
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)mat),boffnodecols,earray,PETSC_COPY_VALUES,&from);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,boffnodecols,0,1,&to);CHKERRQ(ierr);

  /* create temporary global vector to generate scatter context */
  /* This does not allocate the array's memory so is efficient */
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)mat),1,mat->cmap->n,mat->cmap->N,NULL,&gvec);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&aij->Mvctx);CHKERRQ(ierr);
  ierr = VecScatterCreate(gvec,from,aij->lvec,to,&aij->Mvctx);CHKERRQ(ierr);

  if (matmult_nodereorg) aij->Mvctx->is_nodereorged_Mvctx = PETSC_TRUE;

  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)aij->Mvctx);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)aij->lvec);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)aij->ngvec);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)mat,bcols*sizeof(PetscInt));CHKERRQ(ierr);

  ierr = ISDestroy(&from);CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = VecDestroy(&gvec);CHKERRQ(ierr);
  ierr = PetscFree(earray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Takes the local part of an already assembled MPIAIJ matrix
   and disassembles it. This is to allow new nonzeros into the matrix
   that require more communication in the matrix vector multiply.
   Thus certain data-structures must be rebuilt.

   Kind of slow! But that's what application programmers get when
   they are sloppy.
*/
PetscErrorCode MatDisAssemble_MPIAIJ(Mat mat)
{
  Mat_MPIAIJ     *aij  = (Mat_MPIAIJ*)mat->data;
  Mat            A     = aij->A,B = aij->B,Bnew;
  Mat_SeqAIJ     *Aaij = (Mat_SeqAIJ*)A->data,*Baij = (Mat_SeqAIJ*)B->data;
  PetscErrorCode ierr;
  PetscInt       i,j,m = B->rmap->n,n = mat->cmap->N,col,ct = 0,*farray = aij->farray,*garray = aij->garray,*nz,ec,loc;
  PetscScalar    v;

  PetscFunctionBegin;
  /* free stuff related to matrix-vec multiply */
  ierr = VecGetSize(aij->lvec,&ec);CHKERRQ(ierr); /* needed for PetscLogObjectMemory below */
  ierr = VecDestroy(&aij->lvec);CHKERRQ(ierr);
  if (aij->colmap) {
#if defined(PETSC_USE_CTABLE)
    ierr = PetscTableDestroy(&aij->colmap);CHKERRQ(ierr);
#else
    ierr = PetscFree(aij->colmap);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)mat,-aij->B->cmap->n*sizeof(PetscInt));CHKERRQ(ierr);
#endif
  }

  /* disassemble B first. Make sure that B is assembled so we can access its values */
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* calculate new B's nz per row */
  ierr = PetscMalloc1(m+1,&nz);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    nz[i] = Baij->i[i+1] - Baij->i[i];
  }

  /* also count off-diag nonzeros of A when mat is nodereorged */
  if (aij->nodecols) {
    for (i=0; i<m; i++) {
      ierr = PetscFindInt(mat->cmap->n,Aaij->ilen[i],&Aaij->j[Aaij->i[i]],&loc);CHKERRQ(ierr); /* find loc splitting diag & off-diag */
      if (loc < 0) loc = -loc - 1;
      nz[i] += Aaij->ilen[i] - loc;
    }
  }

  ierr = MatCreate(PETSC_COMM_SELF,&Bnew);CHKERRQ(ierr);
  ierr = MatSetSizes(Bnew,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(Bnew,mat,mat);CHKERRQ(ierr);
  ierr = MatSetType(Bnew,((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Bnew,0,nz);CHKERRQ(ierr);
  ierr = PetscFree(nz);CHKERRQ(ierr);

  /* move over B's stuff to new B */
  ((Mat_SeqAIJ*)Bnew->data)->nonew = Baij->nonew; /* Inherit insertion error options. */

  /*
   Ensure that B's nonzerostate is monotonically increasing.
   Or should this follow the MatSetValues() loop to preserve B's nonzerstate across a MatDisAssemble() call?
   */
  Bnew->nonzerostate = B->nonzerostate;

  for (i=0; i<m; i++) {
    for (j=Baij->i[i]; j<Baij->i[i+1]; j++) {
      col  = garray[Baij->j[ct]];
      v    = Baij->a[ct++];
      ierr = MatSetValues(Bnew,1,&i,1,&col,&v,B->insertmode);CHKERRQ(ierr);
    }
  }

  /* disassemble A when needed. Do not need to create a new A. Moving off-diag & on-node part of A to B is enough */
  if (aij->nodecols) { /* may be true when aij is nodereorged */
    for (i=0; i<m; i++) {
      ierr = PetscFindInt(mat->cmap->n,Aaij->ilen[i],&Aaij->j[Aaij->i[i]],&loc);CHKERRQ(ierr);
      for (j=loc; j<Aaij->ilen[i]; j++) { /* move off-diag segment of the row to Bnew */
        col  = farray[Aaij->j[Aaij->i[i]+j]];
        v    = Aaij->a[Aaij->i[i]+j];
        ierr = MatSetValues(Bnew,1,&i,1,&col,&v,B->insertmode);CHKERRQ(ierr);
      }
      Aaij->ilen[i] = loc; /* shrink the row */
    }
  }

  ierr = PetscFree(aij->farray);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)mat,-(mat->cmap->n+aij->nodecols+ec)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)Bnew);CHKERRQ(ierr);

  aij->nodecols    = 0;
  aij->nodereorged = PETSC_FALSE;
  aij->B           = Bnew;
  A->was_assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*      ugly stuff added for Glenn someday we should fix this up */

static PetscInt *auglyrmapd = 0,*auglyrmapo = 0; /* mapping from the local ordering to the "diagonal" and "off-diagonal" parts of the local matrix */
static Vec auglydd          = 0,auglyoo     = 0; /* work vectors used to scale the two parts of the local matrix */


PetscErrorCode MatMPIAIJDiagonalScaleLocalSetUp(Mat inA,Vec scale)
{
  Mat_MPIAIJ     *ina = (Mat_MPIAIJ*) inA->data; /*access private part of matrix */
  PetscErrorCode ierr;
  PetscInt       i,n,nt,cstart,cend,no,*garray = ina->garray,*lindices;
  PetscInt       *r_rmapd,*r_rmapo;

  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(inA,&cstart,&cend);CHKERRQ(ierr);
  ierr = MatGetSize(ina->A,NULL,&n);CHKERRQ(ierr);
  ierr = PetscCalloc1(inA->rmap->mapping->n+1,&r_rmapd);CHKERRQ(ierr);
  nt   = 0;
  for (i=0; i<inA->rmap->mapping->n; i++) {
    if (inA->rmap->mapping->indices[i] >= cstart && inA->rmap->mapping->indices[i] < cend) {
      nt++;
      r_rmapd[i] = inA->rmap->mapping->indices[i] + 1;
    }
  }
  if (nt != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Hmm nt %D n %D",nt,n);
  ierr = PetscMalloc1(n+1,&auglyrmapd);CHKERRQ(ierr);
  for (i=0; i<inA->rmap->mapping->n; i++) {
    if (r_rmapd[i]) {
      auglyrmapd[(r_rmapd[i]-1)-cstart] = i;
    }
  }
  ierr = PetscFree(r_rmapd);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&auglydd);CHKERRQ(ierr);

  ierr = PetscCalloc1(inA->cmap->N+1,&lindices);CHKERRQ(ierr);
  for (i=0; i<ina->B->cmap->n; i++) {
    lindices[garray[i]] = i+1;
  }
  no   = inA->rmap->mapping->n - nt;
  ierr = PetscCalloc1(inA->rmap->mapping->n+1,&r_rmapo);CHKERRQ(ierr);
  nt   = 0;
  for (i=0; i<inA->rmap->mapping->n; i++) {
    if (lindices[inA->rmap->mapping->indices[i]]) {
      nt++;
      r_rmapo[i] = lindices[inA->rmap->mapping->indices[i]];
    }
  }
  if (nt > no) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Hmm nt %D no %D",nt,n);
  ierr = PetscFree(lindices);CHKERRQ(ierr);
  ierr = PetscMalloc1(nt+1,&auglyrmapo);CHKERRQ(ierr);
  for (i=0; i<inA->rmap->mapping->n; i++) {
    if (r_rmapo[i]) {
      auglyrmapo[(r_rmapo[i]-1)] = i;
    }
  }
  ierr = PetscFree(r_rmapo);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,nt,&auglyoo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMPIAIJDiagonalScaleLocal(Mat A,Vec scale)
{
  /* This routine should really be abandoned as it duplicates MatDiagonalScaleLocal */
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(A,"MatDiagonalScaleLocal_C",(Mat,Vec),(A,scale));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatDiagonalScaleLocal_MPIAIJ(Mat A,Vec scale)
{
  Mat_MPIAIJ        *a = (Mat_MPIAIJ*) A->data; /*access private part of matrix */
  PetscErrorCode    ierr;
  PetscInt          n,i;
  PetscScalar       *d,*o;
  const PetscScalar *s;

  PetscFunctionBegin;
  if (!auglyrmapd) {
    ierr = MatMPIAIJDiagonalScaleLocalSetUp(A,scale);CHKERRQ(ierr);
  }

  ierr = VecGetArrayRead(scale,&s);CHKERRQ(ierr);

  ierr = VecGetLocalSize(auglydd,&n);CHKERRQ(ierr);
  ierr = VecGetArray(auglydd,&d);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    d[i] = s[auglyrmapd[i]]; /* copy "diagonal" (true local) portion of scale into dd vector */
  }
  ierr = VecRestoreArray(auglydd,&d);CHKERRQ(ierr);
  /* column scale "diagonal" portion of local matrix */
  ierr = MatDiagonalScale(a->A,NULL,auglydd);CHKERRQ(ierr);

  ierr = VecGetLocalSize(auglyoo,&n);CHKERRQ(ierr);
  ierr = VecGetArray(auglyoo,&o);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    o[i] = s[auglyrmapo[i]]; /* copy "off-diagonal" portion of scale into oo vector */
  }
  ierr = VecRestoreArrayRead(scale,&s);CHKERRQ(ierr);
  ierr = VecRestoreArray(auglyoo,&o);CHKERRQ(ierr);
  /* column scale "off-diagonal" portion of local matrix */
  ierr = MatDiagonalScale(a->B,NULL,auglyoo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

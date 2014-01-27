/*
 GAMG geometric-algebric multigrid PC - Mark Adams 2011
 */
#include "petsc-private/matimpl.h"
#include <../src/ksp/pc/impls/gamg/gamg.h>           /*I "petscpc.h" I*/
#include <petsc-private/kspimpl.h>

/* -------------------------------------------------------------------------- */
/*
   PCGAMGCreateGraph - create simple scaled scalar graph from matrix

 Input Parameter:
 . Amat - matrix
 Output Parameter:
 . a_Gmaat - eoutput scalar graph (symmetric?)
 */
#undef __FUNCT__
#define __FUNCT__ "PCGAMGCreateGraph"
PetscErrorCode PCGAMGCreateGraph(const Mat Amat, Mat *a_Gmat)
{
  PetscErrorCode ierr;
  PetscInt       Istart,Iend,Ii,jj,kk,ncols,nloc,NN,MM,bs;
  PetscMPIInt    rank, size;
  MPI_Comm       comm;
  Mat            Gmat;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)Amat,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Amat, &Istart, &Iend);CHKERRQ(ierr);
  ierr = MatGetSize(Amat, &MM, &NN);CHKERRQ(ierr);
  ierr = MatGetBlockSize(Amat, &bs);CHKERRQ(ierr);
  nloc = (Iend-Istart)/bs;

#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventBegin(petsc_gamg_setup_events[GRAPH],0,0,0,0);CHKERRQ(ierr);
#endif
  if (bs > 1) {
    const PetscScalar *vals;
    const PetscInt    *idx;
    PetscInt          *d_nnz, *o_nnz;
    /* count nnz, there is sparcity in here so this might not be enough */
    ierr = PetscMalloc1(nloc, &d_nnz);CHKERRQ(ierr);
    ierr = PetscMalloc1(nloc, &o_nnz);CHKERRQ(ierr);
    for (Ii = Istart, jj = 0; Ii < Iend; Ii += bs, jj++) {
      d_nnz[jj] = 0;
      for (kk=0; kk<bs; kk++) {
        ierr = MatGetRow(Amat,Ii+kk,&ncols,0,0);CHKERRQ(ierr);
        if (ncols > d_nnz[jj]) {
          d_nnz[jj] = ncols; /* very pessimistic but could be too low in theory */
          o_nnz[jj] = ncols;
          if (d_nnz[jj] > nloc) d_nnz[jj] = nloc;
          if (o_nnz[jj] > (NN/bs-nloc)) o_nnz[jj] = NN/bs-nloc;
        }
        ierr = MatRestoreRow(Amat,Ii+kk,&ncols,0,0);CHKERRQ(ierr);
      }
    }

    /* get scalar copy (norms) of matrix -- AIJ specific!!! */
    ierr = MatCreateAIJ(comm, nloc, nloc, PETSC_DETERMINE, PETSC_DETERMINE,0, d_nnz, 0, o_nnz, &Gmat);CHKERRQ(ierr);

    ierr = PetscFree(d_nnz);CHKERRQ(ierr);
    ierr = PetscFree(o_nnz);CHKERRQ(ierr);

    for (Ii = Istart; Ii < Iend; Ii++) {
      PetscInt dest_row = Ii/bs;
      ierr = MatGetRow(Amat,Ii,&ncols,&idx,&vals);CHKERRQ(ierr);
      for (jj=0; jj<ncols; jj++) {
        PetscInt    dest_col = idx[jj]/bs;
        PetscScalar sv       = PetscAbs(PetscRealPart(vals[jj]));
        ierr = MatSetValues(Gmat,1,&dest_row,1,&dest_col,&sv,ADD_VALUES);CHKERRQ(ierr);
      }
      ierr = MatRestoreRow(Amat,Ii,&ncols,&idx,&vals);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(Gmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Gmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  } else {
    /* just copy scalar matrix - abs() not taken here but scaled later */
    ierr = MatDuplicate(Amat, MAT_COPY_VALUES, &Gmat);CHKERRQ(ierr);
  }

#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventEnd(petsc_gamg_setup_events[GRAPH],0,0,0,0);CHKERRQ(ierr);
#endif

  *a_Gmat = Gmat;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGFilterGraph - filter graph and symetrize if needed

 Input Parameter:
 . vfilter - threshold paramter [0,1)
 . symm - symetrize?
 In/Output Parameter:
 . a_Gmat - original graph
 */
#undef __FUNCT__
#define __FUNCT__ "PCGAMGFilterGraph"
PetscErrorCode PCGAMGFilterGraph(Mat *a_Gmat,const PetscReal vfilter,const PetscBool symm,const PetscInt verbose)
{
  PetscErrorCode    ierr;
  PetscInt          Istart,Iend,Ii,jj,ncols,nnz0,nnz1, NN, MM, nloc;
  PetscMPIInt       rank, size;
  Mat               Gmat  = *a_Gmat, tGmat, matTrans;
  MPI_Comm          comm;
  const PetscScalar *vals;
  const PetscInt    *idx;
  PetscInt          *d_nnz, *o_nnz;
  Vec               diag;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)Gmat,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Gmat, &Istart, &Iend);CHKERRQ(ierr);
  nloc = Iend - Istart;
  ierr = MatGetSize(Gmat, &MM, &NN);CHKERRQ(ierr);
#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventBegin(petsc_gamg_setup_events[GRAPH],0,0,0,0);CHKERRQ(ierr);
#endif
  /* scale Gmat so filter works */
  ierr = MatGetVecs(Gmat, &diag, 0);CHKERRQ(ierr);
  ierr = MatGetDiagonal(Gmat, diag);CHKERRQ(ierr);
  ierr = VecReciprocal(diag);CHKERRQ(ierr);
  ierr = VecSqrtAbs(diag);CHKERRQ(ierr);
  ierr = MatDiagonalScale(Gmat, diag, diag);CHKERRQ(ierr);
  ierr = VecDestroy(&diag);CHKERRQ(ierr);

  if (symm) {
    ierr = MatTranspose(Gmat, MAT_INITIAL_MATRIX, &matTrans);CHKERRQ(ierr);
  }

  /* filter - dup zeros out matrix */
  ierr = PetscMalloc1(nloc, &d_nnz);CHKERRQ(ierr);
  ierr = PetscMalloc1(nloc, &o_nnz);CHKERRQ(ierr);
  for (Ii = Istart, jj = 0; Ii < Iend; Ii++, jj++) {
    ierr      = MatGetRow(Gmat,Ii,&ncols,NULL,NULL);CHKERRQ(ierr);
    d_nnz[jj] = ncols;
    o_nnz[jj] = ncols;
    ierr      = MatRestoreRow(Gmat,Ii,&ncols,NULL,NULL);CHKERRQ(ierr);
    if (symm) {
      ierr       = MatGetRow(matTrans,Ii,&ncols,NULL,NULL);CHKERRQ(ierr);
      d_nnz[jj] += ncols;
      o_nnz[jj] += ncols;
      ierr       = MatRestoreRow(matTrans,Ii,&ncols,NULL,NULL);CHKERRQ(ierr);
    }
    if (d_nnz[jj] > nloc) d_nnz[jj] = nloc;
    if (o_nnz[jj] > (MM-nloc)) o_nnz[jj] = MM - nloc;
  }
  ierr = MatCreateAIJ(comm, nloc, nloc, MM, MM, 0, d_nnz, 0, o_nnz, &tGmat);CHKERRQ(ierr);
  ierr = PetscFree(d_nnz);CHKERRQ(ierr);
  ierr = PetscFree(o_nnz);CHKERRQ(ierr);
  if (symm) {
    ierr = MatDestroy(&matTrans);CHKERRQ(ierr);
  }

  for (Ii = Istart, nnz0 = nnz1 = 0; Ii < Iend; Ii++) {
    ierr = MatGetRow(Gmat,Ii,&ncols,&idx,&vals);CHKERRQ(ierr);
    for (jj=0; jj<ncols; jj++,nnz0++) {
      PetscScalar sv = PetscAbs(PetscRealPart(vals[jj]));
      if (PetscRealPart(sv) > vfilter) {
        nnz1++;
        if (symm) {
          sv  *= 0.5;
          ierr = MatSetValues(tGmat,1,&Ii,1,&idx[jj],&sv,ADD_VALUES);CHKERRQ(ierr);
          ierr = MatSetValues(tGmat,1,&idx[jj],1,&Ii,&sv,ADD_VALUES);CHKERRQ(ierr);
        } else {
          ierr = MatSetValues(tGmat,1,&Ii,1,&idx[jj],&sv,ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
    ierr = MatRestoreRow(Gmat,Ii,&ncols,&idx,&vals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(tGmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(tGmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventEnd(petsc_gamg_setup_events[GRAPH],0,0,0,0);CHKERRQ(ierr);
#endif

  if (verbose) {
    if (verbose == 1) {
      ierr = PetscPrintf(comm,"\t[%d]%s %g%% nnz after filtering, with threshold %g, %g nnz ave. (N=%d)\n",rank,__FUNCT__,
                         100.*(double)nnz1/(double)nnz0,vfilter,(double)nnz0/(double)nloc,MM);CHKERRQ(ierr);
    } else {
      PetscInt nnz[2],out[2];
      nnz[0] = nnz0; nnz[1] = nnz1;
      ierr   = MPI_Allreduce(nnz, out, 2, MPIU_INT, MPI_SUM, comm);CHKERRQ(ierr);
      ierr   = PetscPrintf(comm,"\t[%d]%s %g%% nnz after filtering, with threshold %g, %g nnz ave. (N=%d)\n",rank,__FUNCT__,
                           100.*(double)out[1]/(double)out[0],vfilter,(double)out[0]/(double)MM,MM);CHKERRQ(ierr);
    }
  }
  ierr    = MatDestroy(&Gmat);CHKERRQ(ierr);
  *a_Gmat = tGmat;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGGetDataWithGhosts - hacks into Mat MPIAIJ so this must have > 1 pe

   Input Parameter:
   . Gmat - MPIAIJ matrix for scattters
   . data_sz - number of data terms per node (# cols in output)
   . data_in[nloc*data_sz] - column oriented data
   Output Parameter:
   . a_stride - numbrt of rows of output
   . a_data_out[stride*data_sz] - output data with ghosts
*/
#undef __FUNCT__
#define __FUNCT__ "PCGAMGGetDataWithGhosts"
PetscErrorCode PCGAMGGetDataWithGhosts(const Mat Gmat,const PetscInt data_sz,const PetscReal data_in[],PetscInt *a_stride,PetscReal **a_data_out)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  MPI_Comm       comm;
  Vec            tmp_crds;
  Mat_MPIAIJ     *mpimat = (Mat_MPIAIJ*)Gmat->data;
  PetscInt       nnodes,num_ghosts,dir,kk,jj,my0,Iend,nloc;
  PetscScalar    *data_arr;
  PetscReal      *datas;
  PetscBool      isMPIAIJ;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)Gmat,&comm);CHKERRQ(ierr);
  ierr      = PetscObjectTypeCompare((PetscObject)Gmat, MATMPIAIJ, &isMPIAIJ);CHKERRQ(ierr);
  ierr      = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr      = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr      = MatGetOwnershipRange(Gmat, &my0, &Iend);CHKERRQ(ierr);
  nloc      = Iend - my0;
  ierr      = VecGetLocalSize(mpimat->lvec, &num_ghosts);CHKERRQ(ierr);
  nnodes    = num_ghosts + nloc;
  *a_stride = nnodes;
  ierr      = MatGetVecs(Gmat, &tmp_crds, 0);CHKERRQ(ierr);

  ierr = PetscMalloc1(data_sz*nnodes, &datas);CHKERRQ(ierr);
  for (dir=0; dir<data_sz; dir++) {
    /* set local, and global */
    for (kk=0; kk<nloc; kk++) {
      PetscInt    gid = my0 + kk;
      PetscScalar crd = (PetscScalar)data_in[dir*nloc + kk]; /* col oriented */
      datas[dir*nnodes + kk] = PetscRealPart(crd);

      ierr = VecSetValues(tmp_crds, 1, &gid, &crd, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(tmp_crds);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(tmp_crds);CHKERRQ(ierr);
    /* get ghost datas */
    ierr = VecScatterBegin(mpimat->Mvctx,tmp_crds,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(mpimat->Mvctx,tmp_crds,mpimat->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(mpimat->lvec, &data_arr);CHKERRQ(ierr);
    for (kk=nloc,jj=0;jj<num_ghosts;kk++,jj++) datas[dir*nnodes + kk] = PetscRealPart(data_arr[jj]);
    ierr = VecRestoreArray(mpimat->lvec, &data_arr);CHKERRQ(ierr);
  }
  ierr        = VecDestroy(&tmp_crds);CHKERRQ(ierr);
  *a_data_out = datas;
  PetscFunctionReturn(0);
}


/* hash table stuff - simple, not dymanic, key >= 0, has table
 *
 *  GAMGTableCreate
 */
/* avoid overflow */
#define GAMG_HASH(key) ((7*key)%a_tab->size)
#undef __FUNCT__
#define __FUNCT__ "GAMGTableCreate"
PetscErrorCode GAMGTableCreate(PetscInt a_size, GAMGHashTable *a_tab)
{
  PetscErrorCode ierr;
  PetscInt       kk;

  PetscFunctionBegin;
  a_tab->size = a_size;

  ierr = PetscMalloc1(a_size, &a_tab->table);CHKERRQ(ierr);
  ierr = PetscMalloc1(a_size, &a_tab->data);CHKERRQ(ierr);
  for (kk=0; kk<a_size; kk++) a_tab->table[kk] = -1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GAMGTableDestroy"
PetscErrorCode GAMGTableDestroy(GAMGHashTable *a_tab)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(a_tab->table);CHKERRQ(ierr);
  ierr = PetscFree(a_tab->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GAMGTableAdd"
PetscErrorCode GAMGTableAdd(GAMGHashTable *a_tab, PetscInt a_key, PetscInt a_data)
{
  PetscInt kk,idx;

  PetscFunctionBegin;
  if (a_key<0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Negative key %d.",a_key);
  for (kk = 0, idx = GAMG_HASH(a_key);
       kk < a_tab->size;
       kk++, idx = (idx==(a_tab->size-1)) ? 0 : idx + 1) {

    if (a_tab->table[idx] == a_key) {
      /* exists */
      a_tab->data[idx] = a_data;
      break;
    } else if (a_tab->table[idx] == -1) {
      /* add */
      a_tab->table[idx] = a_key;
      a_tab->data[idx]  = a_data;
      break;
    }
  }
  if (kk==a_tab->size) {
    /* this is not to efficient, waiting until completely full */
    PetscInt       oldsize = a_tab->size, new_size = 2*a_tab->size + 5, *oldtable = a_tab->table, *olddata = a_tab->data;
    PetscErrorCode ierr;

    a_tab->size = new_size;

    ierr = PetscMalloc1(a_tab->size, &a_tab->table);CHKERRQ(ierr);
    ierr = PetscMalloc1(a_tab->size, &a_tab->data);CHKERRQ(ierr);

    for (kk=0;kk<a_tab->size;kk++) a_tab->table[kk] = -1;
    for (kk=0;kk<oldsize;kk++) {
      if (oldtable[kk] != -1) {
        ierr = GAMGTableAdd(a_tab, oldtable[kk], olddata[kk]);CHKERRQ(ierr);
       }
    }
    ierr = PetscFree(oldtable);CHKERRQ(ierr);
    ierr = PetscFree(olddata);CHKERRQ(ierr);
    ierr = GAMGTableAdd(a_tab, a_key, a_data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GAMGTableFind"
PetscErrorCode GAMGTableFind(GAMGHashTable *a_tab, PetscInt a_key, PetscInt *a_data)
{
  PetscInt kk,idx;

  PetscFunctionBegin;
  if (a_key<0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Negative key %d.",a_key);
  for (kk = 0, idx = GAMG_HASH(a_key); kk < a_tab->size; kk++, idx = (idx==(a_tab->size-1)) ? 0 : idx + 1) {
    if (a_tab->table[idx] == a_key) {
      *a_data = a_tab->data[idx];
      break;
    } else if (a_tab->table[idx] == -1) {
      /* not here */
      *a_data = -1;
      break;
    }
  }
  if (kk==a_tab->size) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"key %d not found in table",a_key);
  PetscFunctionReturn(0);
}

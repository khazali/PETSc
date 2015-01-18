
#include <petsc-private/matimpl.h>               /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

/* Logging support */
PetscClassId MAT_COARSEN_CLASSID;

PetscFunctionList MatCoarsenList              = 0;
PetscBool         MatCoarsenRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "MatCoarsenRegister"
/*@C
   MatCoarsenRegister - Adds a new sparse matrix coarsen to the  matrix package.

   Not Collective

   Input Parameters:
+  sname - name of coarsen (for example MATCOARSENMIS)
-  function - function pointer that creates the coarsen type

   Level: developer

   Sample usage:
.vb
   MatCoarsenRegister("my_agg",MyAggCreate);
.ve

   Then, your aggregator can be chosen with the procedural interface via
$     MatCoarsenSetType(agg,"my_agg")
   or at runtime via the option
$     -mat_coarsen_type my_agg

.keywords: matrix, coarsen, register

.seealso: MatCoarsenRegisterDestroy(), MatCoarsenRegisterAll()
@*/
PetscErrorCode  MatCoarsenRegister(const char sname[],PetscErrorCode (*function)(MatCoarsen))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&MatCoarsenList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCoarsenGetType"
/*@C
   MatCoarsenGetType - Gets the Coarsen method type and name (as a string)
        from the coarsen context.

   Not collective

   Input Parameter:
.  coarsen - the coarsen context

   Output Parameter:
.  type - aggregator type

   Level: intermediate

   Not Collective

.keywords: Coarsen, get, method, name, type
@*/
PetscErrorCode  MatCoarsenGetType(MatCoarsen coarsen,MatCoarsenType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarsen,MAT_COARSEN_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)coarsen)->type_name;
  PetscFunctionReturn(0);
}

typedef PetscInt NState;
static const NState NOT_DONE=-2;
static const NState DELETED =-1;
static const NState REMOVED =-3;
#define IS_SELECTED(s) (s!=DELETED && s!=NOT_DONE && s!=REMOVED)

/* -------------------------------------------------------------------------- */
/*
   smoothAggs - greedy grab of with G1 (unsquared graph) -- AIJ specific
     - AGG-MG specific: clears singletons out of 'selected_2'

   Input Parameter:
   . Gmat_2 - glabal matrix of graph (data not defined)
   . Gmat_1 - base graph to grab with
   Input/Output Parameter:
   . aggs_2 - linked list of aggs with gids)
*/
#undef __FUNCT__
#define __FUNCT__ "smoothAggs"
static PetscErrorCode smoothAggs(const Mat Gmat_2, /* base (squared) graph */
                                 const Mat Gmat_1,  /* base graph */
                                 /* const IS selected_2, [nselected local] selected vertices */
                                 PetscCoarsenData *aggs_2)  /* [nselected local] global ID of aggregate */
{
  PetscErrorCode ierr;
  PetscBool      isMPI;
  Mat_SeqAIJ     *matA_1, *matB_1=0, *matA_2, *matB_2=0;
  MPI_Comm       comm;
  PetscMPIInt    rank,size;
  PetscInt       lid,*ii,*idx,ix,Iend,my0,kk,n,j;
  Mat_MPIAIJ     *mpimat_2 = 0, *mpimat_1=0;
  const PetscInt nloc      = Gmat_2->rmap->n;
  const PetscInt nglob     = Gmat_2->rmap->N;
  PetscScalar    *cpcol_1_state,*cpcol_2_state,*cpcol_2_par_orig,*lid_parent_gid;
  PetscInt       *lid_cprowID_1;
  NState         *lid_state;
  Vec            ghost_par_orig2;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)Gmat_2,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Gmat_1,&my0,&Iend);CHKERRQ(ierr);

  if (PETSC_FALSE) {
    PetscViewer viewer; char fname[32]; static int llev=0;
    sprintf(fname,"Gmat2_%d.m",llev++);
    PetscViewerASCIIOpen(comm,fname,&viewer);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = MatView(Gmat_2, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
  }

  /* get submatrices */
  ierr = PetscObjectTypeCompare((PetscObject)Gmat_1, MATMPIAIJ, &isMPI);CHKERRQ(ierr);
  if (isMPI) {
    /* grab matrix objects */
    mpimat_2 = (Mat_MPIAIJ*)Gmat_2->data;
    mpimat_1 = (Mat_MPIAIJ*)Gmat_1->data;
    matA_1   = (Mat_SeqAIJ*)mpimat_1->A->data;
    matB_1   = (Mat_SeqAIJ*)mpimat_1->B->data;
    matA_2   = (Mat_SeqAIJ*)mpimat_2->A->data;
    matB_2   = (Mat_SeqAIJ*)mpimat_2->B->data;

    /* force compressed row storage for B matrix in AuxMat */
    ierr = MatCheckCompressedRow(mpimat_1->B,matB_1->nonzerorowcnt,&matB_1->compressedrow,matB_1->i,Gmat_1->rmap->n,-1.0);CHKERRQ(ierr);

    ierr = PetscMalloc1(nloc, &lid_cprowID_1);CHKERRQ(ierr);
    for (lid = 0; lid < nloc; lid++) lid_cprowID_1[lid] = -1;
    for (ix=0; ix<matB_1->compressedrow.nrows; ix++) {
      PetscInt lid = matB_1->compressedrow.rindex[ix];
      lid_cprowID_1[lid] = ix;
    }
  } else {
    matA_1        = (Mat_SeqAIJ*)Gmat_1->data;
    matA_2        = (Mat_SeqAIJ*)Gmat_2->data;
    lid_cprowID_1 = NULL;
  }
  if (nloc>0) {
    if (!(matA_1 && !matA_1->compressedrow.use)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"!(matA_1 && !matA_1->compressedrow.use)");
    if (!(matB_1==0 || matB_1->compressedrow.use)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"!(matB_1==0 || matB_1->compressedrow.use)");
    if (!(matA_2 && !matA_2->compressedrow.use)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"!(matA_2 && !matA_2->compressedrow.use)");
    if (!(matB_2==0 || matB_2->compressedrow.use)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"!(matB_2==0 || matB_2->compressedrow.use)");
  }
  /* get state of locals and selected gid for deleted */
  ierr = PetscMalloc1(nloc, &lid_state);CHKERRQ(ierr);
  ierr = PetscMalloc1(nloc, &lid_parent_gid);CHKERRQ(ierr);
  for (lid = 0; lid < nloc; lid++) {
    lid_parent_gid[lid] = -1.0;
    lid_state[lid]      = DELETED;
  }

  /* set lid_state */
  for (lid = 0; lid < nloc; lid++) {
    PetscCDPos pos;
    ierr = PetscCDGetHeadPos(aggs_2,lid,&pos);CHKERRQ(ierr);
    if (pos) {
      PetscInt gid1;

      ierr = PetscLLNGetID(pos, &gid1);CHKERRQ(ierr);
      if (gid1 != lid+my0) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"gid1 %D != lid %D + my0 %D",gid1,lid,my0);
      lid_state[lid] = gid1;
    }
  }

  /* map local to selected local, DELETED means a ghost owns it */
  for (lid=kk=0; lid<nloc; lid++) {
    NState state = lid_state[lid];
    if (IS_SELECTED(state)) {
      PetscCDPos pos;
      ierr = PetscCDGetHeadPos(aggs_2,lid,&pos);CHKERRQ(ierr);
      while (pos) {
        PetscInt gid1;
        ierr = PetscLLNGetID(pos, &gid1);CHKERRQ(ierr);
        ierr = PetscCDGetNextPos(aggs_2,lid,&pos);CHKERRQ(ierr);

        if (gid1 >= my0 && gid1 < Iend) lid_parent_gid[gid1-my0] = (PetscScalar)(lid + my0);
      }
    }
  }
  /* get 'cpcol_1/2_state' & cpcol_2_par_orig - uses mpimat_1/2->lvec for temp space */
  if (isMPI) {
    Vec tempVec;
    /* get 'cpcol_1_state' */
    ierr = MatCreateVecs(Gmat_1, &tempVec, 0);CHKERRQ(ierr);
    for (kk=0,j=my0; kk<nloc; kk++,j++) {
      PetscScalar v = (PetscScalar)lid_state[kk];
      ierr = VecSetValues(tempVec, 1, &j, &v, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(tempVec);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(tempVec);CHKERRQ(ierr);
    ierr = VecScatterBegin(mpimat_1->Mvctx,tempVec, mpimat_1->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat_1->Mvctx,tempVec, mpimat_1->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(mpimat_1->lvec, &cpcol_1_state);CHKERRQ(ierr);
    /* get 'cpcol_2_state' */
    ierr = VecScatterBegin(mpimat_2->Mvctx,tempVec, mpimat_2->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat_2->Mvctx,tempVec, mpimat_2->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(mpimat_2->lvec, &cpcol_2_state);CHKERRQ(ierr);
    /* get 'cpcol_2_par_orig' */
    for (kk=0,j=my0; kk<nloc; kk++,j++) {
      PetscScalar v = (PetscScalar)lid_parent_gid[kk];
      ierr = VecSetValues(tempVec, 1, &j, &v, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(tempVec);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(tempVec);CHKERRQ(ierr);
    ierr = VecDuplicate(mpimat_2->lvec, &ghost_par_orig2);CHKERRQ(ierr);
    ierr = VecScatterBegin(mpimat_2->Mvctx,tempVec, ghost_par_orig2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat_2->Mvctx,tempVec, ghost_par_orig2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(ghost_par_orig2, &cpcol_2_par_orig);CHKERRQ(ierr);

    ierr = VecDestroy(&tempVec);CHKERRQ(ierr);
  } /* ismpi */

  /* doit */
  for (lid=0; lid<nloc; lid++) {
    NState state = lid_state[lid];
    if (IS_SELECTED(state)) {
      /* steal locals */
      ii  = matA_1->i; n = ii[lid+1] - ii[lid];
      idx = matA_1->j + ii[lid];
      for (j=0; j<n; j++) {
        PetscInt lidj   = idx[j], sgid;
        NState   statej = lid_state[lidj];
        if (statej==DELETED && (sgid=(PetscInt)PetscRealPart(lid_parent_gid[lidj])) != lid+my0) { /* steal local */
          lid_parent_gid[lidj] = (PetscScalar)(lid+my0); /* send this if sgid is not local */
          if (sgid >= my0 && sgid < Iend) {       /* I'm stealing this local from a local sgid */
            PetscInt   hav=0,slid=sgid-my0,gidj=lidj+my0;
            PetscCDPos pos,last=NULL;
            /* looking for local from local so id_llist_2 works */
            ierr = PetscCDGetHeadPos(aggs_2,slid,&pos);CHKERRQ(ierr);
            while (pos) {
              PetscInt gid;
              ierr = PetscLLNGetID(pos, &gid);CHKERRQ(ierr);
              if (gid == gidj) {
                if (!last) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"last cannot be null");
                ierr = PetscCDRemoveNextNode(aggs_2, slid, last);CHKERRQ(ierr);
                ierr = PetscCDAppendNode(aggs_2, lid, pos);CHKERRQ(ierr);
                hav  = 1;
                break;
              } else last = pos;

              ierr = PetscCDGetNextPos(aggs_2,slid,&pos);CHKERRQ(ierr);
            }
            if (hav!=1) {
              if (!hav) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"failed to find adj in 'selected' lists - structurally unsymmetric matrix");
              SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"found node %d times???",hav);
            }
          } else {            /* I'm stealing this local, owned by a ghost */
            if (sgid != -1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Have un-symmetric graph (apparently). Use '-pc_gamg_sym_graph true' to symetrize the graph or '-pc_gamg_threshold 0.0' if the matrix is structurally symmetric.");
            ierr = PetscCDAppendID(aggs_2, lid, lidj+my0);CHKERRQ(ierr);
          }
        }
      } /* local neighbors */
    } else if (state == DELETED && lid_cprowID_1) {
      PetscInt sgidold = (PetscInt)PetscRealPart(lid_parent_gid[lid]);
      /* see if I have a selected ghost neighbor that will steal me */
      if ((ix=lid_cprowID_1[lid]) != -1) {
        ii  = matB_1->compressedrow.i; n = ii[ix+1] - ii[ix];
        idx = matB_1->j + ii[ix];
        for (j=0; j<n; j++) {
          PetscInt cpid   = idx[j];
          NState   statej = (NState)PetscRealPart(cpcol_1_state[cpid]);
          if (IS_SELECTED(statej) && sgidold != (PetscInt)statej) { /* ghost will steal this, remove from my list */
            lid_parent_gid[lid] = (PetscScalar)statej; /* send who selected */
            if (sgidold>=my0 && sgidold<Iend) { /* this was mine */
              PetscInt   hav=0,oldslidj=sgidold-my0;
              PetscCDPos pos,last=NULL;
              /* remove from 'oldslidj' list */
              ierr = PetscCDGetHeadPos(aggs_2,oldslidj,&pos);CHKERRQ(ierr);
              while (pos) {
                PetscInt gid;
                ierr = PetscLLNGetID(pos, &gid);CHKERRQ(ierr);
                if (lid+my0 == gid) {
                  /* id_llist_2[lastid] = id_llist_2[flid];   /\* remove lid from oldslidj list *\/ */
                  if (!last) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"last cannot be null");
                  ierr = PetscCDRemoveNextNode(aggs_2, oldslidj, last);CHKERRQ(ierr);
                  /* ghost (PetscScalar)statej will add this later */
                  hav = 1;
                  break;
                } else last = pos;

                ierr = PetscCDGetNextPos(aggs_2,oldslidj,&pos);CHKERRQ(ierr);
              }
              if (hav!=1) {
                if (hav==0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"failed to find adj in 'selected' lists - structurally unsymmetric matrix");
                SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"found node %d times???",hav);
              }
            } else {
              /* ghosts remove this later */
            }
          }
        }
      }
    } /* selected/deleted */
  } /* node loop */

  if (isMPI) {
    PetscScalar   *cpcol_2_parent,*cpcol_2_gid;
    Vec           tempVec,ghostgids2,ghostparents2;
    PetscInt      cpid,nghost_2;
    PetscTable    gid_cpid;

    ierr = VecGetSize(mpimat_2->lvec, &nghost_2);CHKERRQ(ierr);
    ierr = MatCreateVecs(Gmat_2, &tempVec, 0);CHKERRQ(ierr);

    /* get 'cpcol_2_parent' */
    for (kk=0,j=my0; kk<nloc; kk++,j++) {
      ierr = VecSetValues(tempVec, 1, &j, &lid_parent_gid[kk], INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(tempVec);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(tempVec);CHKERRQ(ierr);
    ierr = VecDuplicate(mpimat_2->lvec, &ghostparents2);CHKERRQ(ierr);
    ierr = VecScatterBegin(mpimat_2->Mvctx,tempVec, ghostparents2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(mpimat_2->Mvctx,tempVec, ghostparents2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(ghostparents2, &cpcol_2_parent);CHKERRQ(ierr);

    /* get 'cpcol_2_gid' */
    for (kk=0,j=my0; kk<nloc; kk++,j++) {
      PetscScalar v = (PetscScalar)j;
      ierr = VecSetValues(tempVec, 1, &j, &v, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin(tempVec);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(tempVec);CHKERRQ(ierr);
    ierr = VecDuplicate(mpimat_2->lvec, &ghostgids2);CHKERRQ(ierr);
    ierr = VecScatterBegin(mpimat_2->Mvctx,tempVec, ghostgids2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr =   VecScatterEnd(mpimat_2->Mvctx,tempVec, ghostgids2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(ghostgids2, &cpcol_2_gid);CHKERRQ(ierr);

    ierr = VecDestroy(&tempVec);CHKERRQ(ierr);

    /* look for deleted ghosts and add to table */
    ierr = PetscTableCreate(nghost_2,nglob,&gid_cpid);CHKERRQ(ierr);
    for (cpid = 0; cpid < nghost_2; cpid++) {
      NState state = (NState)PetscRealPart(cpcol_2_state[cpid]);
      if (state==DELETED) {
        PetscInt sgid_new = (PetscInt)PetscRealPart(cpcol_2_parent[cpid]);
        PetscInt sgid_old = (PetscInt)PetscRealPart(cpcol_2_par_orig[cpid]);
        if (sgid_old == -1 && sgid_new != -1) {
          PetscInt gid = (PetscInt)PetscRealPart(cpcol_2_gid[cpid]);
          /* PetscTable keys are 1:N inclusive, for some reason */
          ierr = PetscTableAdd(gid_cpid, gid + 1, cpid + 1, INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }

    /* look for deleted ghosts and see if they moved - remove it */
    for (lid=0; lid<nloc; lid++) {
      NState state = lid_state[lid];
      if (IS_SELECTED(state)) {
        PetscCDPos pos,last=NULL;
        /* look for deleted ghosts and see if they moved */
        ierr = PetscCDGetHeadPos(aggs_2,lid,&pos);CHKERRQ(ierr);
        while (pos) {
          PetscInt gid;
          ierr = PetscLLNGetID(pos, &gid);CHKERRQ(ierr);

          if (gid < my0 || gid >= Iend) {
            /* PetscTable keys are 1:N inclusive, for some reason */
            ierr = PetscTableFind(gid_cpid, gid + 1,&cpid);CHKERRQ(ierr);
            cpid--;
            if (cpid != -1) {
              /* a moved ghost - */
              /* id_llist_2[lastid] = id_llist_2[flid];    /\* remove 'flid' from list *\/ */
              ierr = PetscCDRemoveNextNode(aggs_2, lid, last);CHKERRQ(ierr);
            } else last = pos;
          } else last = pos;

          ierr = PetscCDGetNextPos(aggs_2,lid,&pos);CHKERRQ(ierr);
        } /* loop over list of deleted */
      } /* selected */
    }
    ierr = PetscTableDestroy(&gid_cpid);CHKERRQ(ierr);

    /* look at ghosts, see if they changed - and it */
    for (cpid = 0; cpid < nghost_2; cpid++) {
      PetscInt sgid_new = (PetscInt)PetscRealPart(cpcol_2_parent[cpid]);
      if (sgid_new >= my0 && sgid_new < Iend) { /* this is mine */
        PetscInt   gid     = (PetscInt)PetscRealPart(cpcol_2_gid[cpid]);
        PetscInt   slid_new=sgid_new-my0,hav=0;
        PetscCDPos pos;
        /* search for this gid to see if I have it */
        ierr = PetscCDGetHeadPos(aggs_2,slid_new,&pos);CHKERRQ(ierr);
        while (pos) {
          PetscInt gidj;
          ierr = PetscLLNGetID(pos, &gidj);CHKERRQ(ierr);
          ierr = PetscCDGetNextPos(aggs_2,slid_new,&pos);CHKERRQ(ierr);

          if (gidj == gid) { hav = 1; break; }
        }
        if (hav != 1) {
          /* insert 'flidj' into head of llist */
          ierr = PetscCDAppendID(aggs_2, slid_new, gid);CHKERRQ(ierr);
        }
      }
    }

    ierr = VecRestoreArray(mpimat_1->lvec, &cpcol_1_state);CHKERRQ(ierr);
    ierr = VecRestoreArray(mpimat_2->lvec, &cpcol_2_state);CHKERRQ(ierr);
    ierr = VecRestoreArray(ghostparents2, &cpcol_2_parent);CHKERRQ(ierr);
    ierr = VecRestoreArray(ghostgids2, &cpcol_2_gid);CHKERRQ(ierr);
    ierr = PetscFree(lid_cprowID_1);CHKERRQ(ierr);
    ierr = VecDestroy(&ghostgids2);CHKERRQ(ierr);
    ierr = VecDestroy(&ghostparents2);CHKERRQ(ierr);
    ierr = VecDestroy(&ghost_par_orig2);CHKERRQ(ierr);
  }

  ierr = PetscFree(lid_parent_gid);CHKERRQ(ierr);
  ierr = PetscFree(lid_state);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCoarsenApply"
/*@
   MatCoarsenApply - Gets a coarsen for a matrix.

   Collective on Mat

   Input Parameters:
.  matp - the matrix coarsen object

   Output Parameters:
.   coarsen - the coarsen. For each local node this tells the aggregate
                   number that that node is assigned to.

   Options Database Keys:
   To specify the coarsen through the options database, use one of
   the following
$    -mat_coarsen_type mis
   To see the coarsen result
$    -mat_coarsen_view

   Level: beginner

   The user can define additional coarsens; see MatCoarsenRegister().

.keywords: matrix, get, coarsen

.seealso:  MatCoarsenRegister(), MatCoarsenCreate(),
           MatCoarsenDestroy(), MatCoarsenSetAdjacency(), ISCoarsenToNumbering(),
           ISCoarsenCount()
@*/
PetscErrorCode  MatCoarsenApply(MatCoarsen coarser)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarser,MAT_COARSEN_CLASSID,1);
  PetscValidPointer(coarser,2);
  if (!coarser->graph->assembled) SETERRQ(PetscObjectComm((PetscObject)coarser),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (coarser->graph->factortype) SETERRQ(PetscObjectComm((PetscObject)coarser),PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  if (!coarser->ops->apply) SETERRQ(PetscObjectComm((PetscObject)coarser),PETSC_ERR_ARG_WRONGSTATE,"Must set type with MatCoarsenSetFromOptions() or MatCoarsenSetType()");
  ierr = PetscLogEventBegin(MAT_Coarsen,coarser,0,0,0);CHKERRQ(ierr);
  ierr = (*coarser->ops->apply)(coarser);CHKERRQ(ierr);
  if (coarser->smoothgraph && coarser->smoothgraph != coarser->graph) {
    ierr = smoothAggs(coarser->graph, coarser->smoothgraph, coarser->agg_lists);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(MAT_Coarsen,coarser,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCoarsenSetAdjacency"
/*@
   MatCoarsenSetAdjacency - Sets the adjacency graph (matrix) of the thing to be
      partitioned.

   Collective on MatCoarsen and Mat

   Input Parameters:
+  agg - the coarsen context
-  adj - the adjacency matrix

   Level: beginner

.keywords: Coarsen, adjacency

.seealso: MatCoarsenCreate()
@*/
PetscErrorCode  MatCoarsenSetAdjacency(MatCoarsen agg, Mat adj)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(agg,MAT_COARSEN_CLASSID,1);
  PetscValidHeaderSpecific(adj,MAT_CLASSID,2);
  agg->graph = adj;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCoarsenSetStrictAggs"
/*@
   MatCoarsenSetStrictAggs -

   Not Collective on MatCoarsen and Mat

   Input Parameters:
+  agg - the coarsen context
-  str - the adjacency matrix

   Level: beginner

.keywords: Coarsen, adjacency

.seealso: MatCoarsenCreate()
@*/
PetscErrorCode MatCoarsenSetStrictAggs(MatCoarsen agg, PetscBool str)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(agg,MAT_COARSEN_CLASSID,1);
  agg->strict_aggs = str;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCoarsenSetVerbose"
/*@
   MatCoarsenSetVerbose -

   Not Collective on MatCoarsen and Mat

   Input Parameters:
+  agg - the coarsen context
-  str - the adjacency matrix

   Level: beginner

.keywords: Coarsen, adjacency

.seealso: MatCoarsenCreate()
@*/
PetscErrorCode MatCoarsenSetVerbose(MatCoarsen agg, PetscInt vv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(agg,MAT_COARSEN_CLASSID,1);
  agg->verbose = vv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCoarsenDestroy"
/*@
   MatCoarsenDestroy - Destroys the coarsen context.

   Collective on Coarsen

   Input Parameters:
.  agg - the coarsen context

   Level: beginner

.keywords: Coarsen, destroy, context

.seealso: MatCoarsenCreate()
@*/
PetscErrorCode  MatCoarsenDestroy(MatCoarsen *agg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*agg) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*agg),MAT_COARSEN_CLASSID,1);
  if (--((PetscObject)(*agg))->refct > 0) {*agg = 0; PetscFunctionReturn(0);}

  if ((*agg)->ops->destroy) {
    ierr = (*(*agg)->ops->destroy)((*agg));CHKERRQ(ierr);
  }

  if ((*agg)->agg_lists) {
    ierr = PetscCDDestroy((*agg)->agg_lists);CHKERRQ(ierr);
  }

  ierr = PetscHeaderDestroy(agg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCoarsenCreate"
/*@
   MatCoarsenCreate - Creates a coarsen context.

   Collective on MPI_Comm

   Input Parameter:
.   comm - MPI communicator

   Output Parameter:
.  newcrs - location to put the context

   Level: beginner

.keywords: Coarsen, create, context

.seealso: MatCoarsenSetType(), MatCoarsenApply(), MatCoarsenDestroy(),
          MatCoarsenSetAdjacency()

@*/
PetscErrorCode  MatCoarsenCreate(MPI_Comm comm, MatCoarsen *newcrs)
{
  MatCoarsen     agg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *newcrs = 0;

  ierr = MatInitializePackage();CHKERRQ(ierr);
  ierr = PetscHeaderCreate(agg, _p_MatCoarsen, struct _MatCoarsenOps, MAT_COARSEN_CLASSID,"MatCoarsen","Matrix/graph coarsen",
                           "MatCoarsen", comm, MatCoarsenDestroy, MatCoarsenView);CHKERRQ(ierr);

  *newcrs = agg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCoarsenView"
/*@C
   MatCoarsenView - Prints the coarsen data structure.

   Collective on MatCoarsen

   Input Parameters:
.  agg - the coarsen context
.  viewer - optional visualization context

   Level: intermediate

   Note:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

   The user can open alternative visualization contexts with
.     PetscViewerASCIIOpen() - output to a specified file

.keywords: Coarsen, view

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode  MatCoarsenView(MatCoarsen agg,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(agg,MAT_COARSEN_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)agg),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(agg,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectPrintClassNamePrefixType((PetscObject)agg,viewer);CHKERRQ(ierr);
  if (agg->ops->view) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = (*agg->ops->view)(agg,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCoarsenSetType"
/*@C
   MatCoarsenSetType - Sets the type of aggregator to use

   Collective on MatCoarsen

   Input Parameter:
.  coarser - the coarsen context.
.  type - a known method

   Options Database Command:
$  -mat_coarsen_type  <type>
$      Use -help for a list of available methods
$      (for instance, mis)

   Level: intermediate

.keywords: coarsen, set, method, type

.seealso: MatCoarsenCreate(), MatCoarsenApply(), MatCoarsenType

@*/
PetscErrorCode  MatCoarsenSetType(MatCoarsen coarser, MatCoarsenType type)
{
  PetscErrorCode ierr,(*r)(MatCoarsen);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarser,MAT_COARSEN_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)coarser,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (coarser->setupcalled) {
    ierr =  (*coarser->ops->destroy)(coarser);CHKERRQ(ierr);

    coarser->ops->destroy = NULL;
    coarser->subctx       = 0;
    coarser->setupcalled  = 0;
  }

  ierr =  PetscFunctionListFind(MatCoarsenList,type,&r);CHKERRQ(ierr);

  if (!r) SETERRQ1(PetscObjectComm((PetscObject)coarser),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown coarsen type %s",type);

  coarser->ops->destroy = (PetscErrorCode (*)(MatCoarsen)) 0;
  coarser->ops->view    = (PetscErrorCode (*)(MatCoarsen,PetscViewer)) 0;

  ierr = (*r)(coarser);CHKERRQ(ierr);

  ierr = PetscFree(((PetscObject)coarser)->type_name);CHKERRQ(ierr);
  ierr = PetscStrallocpy(type,&((PetscObject)coarser)->type_name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCoarsenSetGreedyOrdering"
/*@C
   MatCoarsenSetGreedyOrdering - Sets the weights for vertices for a coarsen.

   Logically Collective on Coarsen

   Input Parameters:
+  coarser - the coarsen context
-  perm - vertex ordering of (greedy) algorithm

   Level: beginner

   Notes:
      The IS weights is freed by PETSc, so user has given this to us

.keywords: Coarsen

.seealso: MatCoarsenCreate(), MatCoarsenSetType()
@*/
PetscErrorCode MatCoarsenSetGreedyOrdering(MatCoarsen coarser, const IS perm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarser,MAT_COARSEN_CLASSID,1);
  coarser->perm = perm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCoarsenGetData"
/*@C
   MatCoarsenGetData - Sets the weights for vertices for a coarsen.

   Logically Collective on Coarsen

   Input Parameters:
+  coarser - the coarsen context
-  mis - pointer into 'llist'
-  llist - linked list of aggregates

   Level: beginner

   Notes:

.keywords: Coarsen

.seealso: MatCoarsenCreate(), MatCoarsenSetType()
@*/
PetscErrorCode MatCoarsenGetData(MatCoarsen coarser, PetscCoarsenData **llist)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarser,MAT_COARSEN_CLASSID,1);
  if (!coarser->agg_lists) SETERRQ(PetscObjectComm((PetscObject)coarser),PETSC_ERR_ARG_WRONGSTATE,"No linked list - generate it or call ApplyCoarsen");
  *llist             = coarser->agg_lists;
  coarser->agg_lists = 0; /* giving up ownership */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCoarsenSetFromOptions"
/*@
   MatCoarsenSetFromOptions - Sets various coarsen options from the
        options database.

   Collective on MatCoarsen

   Input Parameter:
.  coarser - the coarsen context.

   Options Database Command:
$  -mat_coarsen_type  <type>
$      Use -help for a list of available methods
$      (for instance, mis)

   Level: beginner

.keywords: coarsen, set, method, type
@*/
PetscErrorCode MatCoarsenSetFromOptions(MatCoarsen coarser)
{
  PetscErrorCode ierr;
  PetscBool      flag;
  char           type[256];
  const char     *def;

  PetscFunctionBegin;
  ierr = PetscObjectOptionsBegin((PetscObject)coarser);CHKERRQ(ierr);
  if (!((PetscObject)coarser)->type_name) {
    def = MATCOARSENMIS;
  } else {
    def = ((PetscObject)coarser)->type_name;
  }

  ierr = PetscOptionsFList("-mat_coarsen_type","Type of aggregator","MatCoarsenSetType",MatCoarsenList,def,type,256,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = MatCoarsenSetType(coarser,type);CHKERRQ(ierr);
  }
  /*
   Set the type if it was never set.
   */
  if (!((PetscObject)coarser)->type_name) {
    ierr = MatCoarsenSetType(coarser,def);CHKERRQ(ierr);
  }

  if (coarser->ops->setfromoptions) {
    ierr = (*coarser->ops->setfromoptions)(coarser);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = MatCoarsenViewFromOptions(coarser,NULL,"-mat_coarsen_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCoarsenSetSmoothingAdjacency"
/*@
   MatCoarsenSetSmoothingAdjacency - Sets the smoothing adjacency graph (matrix) of the thing to be
      partitioned.  Edges in the smoothing adjacency graph should be a subset of the edges in the adjacency graph.

   Collective on MatCoarsen and Mat

   Input Parameters:
+  agg - the coarsen context
-  adj - the smoothing adjacency matrix

   Level: intermediate

.keywords: Coarsen, adjacency

.seealso: MatCoarsenCreate()
@*/
PetscErrorCode MatCoarsenSetSmoothingAdjacency(MatCoarsen agg, Mat adj)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(agg,MAT_COARSEN_CLASSID,1);
  PetscValidHeaderSpecific(adj,MAT_CLASSID,2);
  agg->smoothgraph = adj;
  PetscFunctionReturn(0);
}






#include <petsc-private/matimpl.h>      /*I "petscmat.h"  I*/
#include <zoltan.h>

static int get_number_of_rows(void *ctx,int *ierr)
{
  PetscInt    s,e,n;
  MatColoring mc = (MatColoring)ctx;
  Mat         m = mc->mat;
  MatGetOwnershipRange(m,&s,&e);
  n=e-s;
  *ierr = ZOLTAN_OK;
  return (int)n;
}

static void get_row_list(void *ctx,int n,int nl,ZOLTAN_ID_PTR gID,ZOLTAN_ID_PTR lID,int nwts,float *wts,int *ierr)
{
  PetscInt    i,s,e;
  MatColoring mc = (MatColoring)ctx;
  Mat         m = mc->mat;
  MatGetOwnershipRange(m,&s,&e);
  for (i=s;i<e;i++){
    gID[i-s] = i;
    lID[i-s] = i-s;
  }
  *ierr = ZOLTAN_OK;
}

static void get_num_edges_list(void *ctx,int sizeGID,int sizeLID,int nrows,
             ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
             int *numEdges, int *ierr)
{
  PetscInt    i,s,e,n,ncols;
  MatColoring mc=(MatColoring)ctx;
  Mat         mat=mc->mat;
  MatGetOwnershipRange(mat,&s,&e);
  n=e-s;
  if ( (sizeGID != 1) || (sizeLID != 1) || (nrows != n)){
    *ierr = ZOLTAN_FATAL;
    return;
  }
  MatGetOwnershipRange(mat,&s,&e);
  for (i=s;i<e;i++){
    MatGetRow(mat,i,&ncols,NULL,NULL);
    numEdges[i-s] = (int)ncols;
    MatRestoreRow(mat,i,&ncols,NULL,NULL);
  }
  *ierr = ZOLTAN_OK;
  return;
}

static void get_edge_list(void *ctx,int sizeGID,int sizeLID,int num_obj, ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,int *num_edges,ZOLTAN_ID_PTR nborGID, int *nborProc,int wgt_dim, float *ewgts, int *ierr)
{
  PetscInt       i,j,s,e,n,idx,ncols,rnk;
  const PetscInt *cidx;
  MatColoring    mc=(MatColoring)ctx;
  Mat            mat=mc->mat;
  PetscLayout    cmap;
  if ((sizeGID!=1)||(sizeLID!=1)||(wgt_dim != 0)) {
    *ierr = ZOLTAN_FATAL;
    return;
  }
  MatGetOwnershipRange(mat,&s,&e);
  n=e-s;
  MatGetLayouts(mat,NULL,&cmap);
  /* printf("querying %d vertices\n", num_obj); */
  idx=0;
  for (i=s;i<e;i++) {
    MatGetRow(mat,i,&ncols,&cidx,NULL);
    for (j=0;j<ncols;j++) {
      PetscLayoutFindOwner(cmap,cidx[j],&rnk);
      nborGID[idx]=(int)cidx[j];
      nborProc[idx]=(int)rnk;
      idx++;
    }
    MatRestoreRow(mat,i,&ncols,&cidx,NULL);
  }
  *ierr = ZOLTAN_OK;
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringApply_Zoltan"
PETSC_EXTERN PetscErrorCode MatColoringApply_Zoltan(MatColoring mc,ISColoring *iscoloring)
{
  PetscErrorCode       ierr;
  struct Zoltan_Struct *zz;
  Mat                  mat=mc->mat;
  PetscInt             i,s,e,n;
  int                  *zcol;
  ZOLTAN_ID_PTR        gids;
  ISColoringValue      *colors;
  PetscInt             maxcolor,maxcolor_global;
  MPI_Comm             comm_zoltan=PetscObjectComm((PetscObject)mc);
  float                version;

  PetscFunctionBegin;
  ierr = MatGetOwnershipRange(mat,&s,&e);CHKERRQ(ierr);
  n=e-s;
  ierr = PetscMalloc(sizeof(ZOLTAN_ID_TYPE)*n,&gids);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(int)*n,&zcol);CHKERRQ(ierr);
  for (i=s;i<e;i++) {
    gids[i-s]=i;
  }
  Zoltan_Initialize(0, NULL, &version);
  zz = Zoltan_Create(comm_zoltan);
  Zoltan_Set_Param(zz,"DEBUG_LEVEL","0");
  Zoltan_Set_Param(zz,"NUM_GID_ENTRIES","1");
  Zoltan_Set_Param(zz,"NUM_LID_ENTRIES","1");
  Zoltan_Set_Param(zz,"OBJ_WEIGHT_DIM","0");
  /* distance */
  if (mc->dist==1) {
    Zoltan_Set_Param(zz,"COLORING_PROBLEM","distance-1");
  } else if (mc->dist==2) {
    Zoltan_Set_Param(zz,"COLORING_PROBLEM","distance-2");

  }
  Zoltan_Set_Num_Obj_Fn(zz,get_number_of_rows,mc);
  Zoltan_Set_Obj_List_Fn(zz,get_row_list,mc);
  Zoltan_Set_Num_Edges_Multi_Fn(zz,get_num_edges_list,mc);
  Zoltan_Set_Edge_List_Multi_Fn(zz,get_edge_list,mc);
  Zoltan_Color(zz,1,(int)n,gids,zcol);
  ierr = PetscMalloc1(n,&colors);CHKERRQ(ierr);
  maxcolor=0;
  for (i=0;i<n;i++) {
    colors[i]=(ISColoringValue)zcol[i];
    if (zcol[i]>maxcolor) maxcolor=zcol[i];
  }
  ierr = MPI_Allreduce(&maxcolor,&maxcolor_global,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)mc));CHKERRQ(ierr);
  ierr = ISColoringCreate(PetscObjectComm((PetscObject)mc),maxcolor_global+1,n,colors,iscoloring);CHKERRQ(ierr);
  Zoltan_Destroy(&zz);
  ierr = PetscFree(gids);CHKERRQ(ierr);
  ierr = PetscFree(zcol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatColoringCreate_Zoltan"
PETSC_EXTERN PetscErrorCode MatColoringCreate_Zoltan(MatColoring mc)
{
    PetscFunctionBegin;
    mc->data                = NULL;
    mc->ops->apply          = MatColoringApply_Zoltan;
    mc->ops->view           = NULL;
    mc->ops->destroy        = NULL;
    mc->ops->setfromoptions = NULL;
    PetscFunctionReturn(0);
}

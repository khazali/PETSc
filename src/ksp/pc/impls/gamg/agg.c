/*
 GAMG geometric-algebric multiogrid PC - Mark Adams 2011
 */

#include <../src/ksp/pc/impls/gamg/gamg.h>        /*I "petscpc.h" I*/
#include <petsc-private/kspimpl.h>

#include <petscblaslapack.h>

typedef struct {
  PetscInt  nsmooths;
  PetscBool sym_graph;
  PetscBool square_graph;
} PC_GAMG_AGG;

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetNSmooths"
/*@
   PCGAMGSetNSmooths - Set number of smoothing steps (1 is typical)

   Not Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_gamg_agg_nsmooths

   Level: intermediate

   Concepts: Aggregation AMG preconditioner

.seealso: ()
@*/
PetscErrorCode PCGAMGSetNSmooths(PC pc, PetscInt n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGSetNSmooths_C",(PC,PetscInt),(pc,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetNSmooths_GAMG"
PetscErrorCode PCGAMGSetNSmooths_GAMG(PC pc, PetscInt n)
{
  PC_MG       *mg          = (PC_MG*)pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;

  PetscFunctionBegin;
  pc_gamg_agg->nsmooths = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetSymGraph"
/*@
   PCGAMGSetSymGraph -

   Not Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_gamg_sym_graph

   Level: intermediate

   Concepts: Aggregation AMG preconditioner

.seealso: ()
@*/
PetscErrorCode PCGAMGSetSymGraph(PC pc, PetscBool n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGSetSymGraph_C",(PC,PetscBool),(pc,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetSymGraph_GAMG"
PetscErrorCode PCGAMGSetSymGraph_GAMG(PC pc, PetscBool n)
{
  PC_MG       *mg          = (PC_MG*)pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;

  PetscFunctionBegin;
  pc_gamg_agg->sym_graph = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetSquareGraph"
/*@
   PCGAMGSetSquareGraph -

   Not Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_gamg_square_graph

   Level: intermediate

   Concepts: Aggregation AMG preconditioner

.seealso: ()
@*/
PetscErrorCode PCGAMGSetSquareGraph(PC pc, PetscBool n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGAMGSetSquareGraph_C",(PC,PetscBool),(pc,n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetSquareGraph_GAMG"
PetscErrorCode PCGAMGSetSquareGraph_GAMG(PC pc, PetscBool n)
{
  PC_MG       *mg          = (PC_MG*)pc->data;
  PC_GAMG     *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;

  PetscFunctionBegin;
  pc_gamg_agg->square_graph = n;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCSetFromOptions_GAMG_AGG

  Input Parameter:
   . pc -
*/
#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_GAMG_AGG"
PetscErrorCode PCSetFromOptions_GAMG_AGG(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg          = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG    *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("GAMG-AGG options");CHKERRQ(ierr);
  {
    /* -pc_gamg_agg_nsmooths */
    pc_gamg_agg->nsmooths = 1;

    ierr = PetscOptionsInt("-pc_gamg_agg_nsmooths","smoothing steps for smoothed aggregation, usually 1","PCGAMGSetNSmooths",pc_gamg_agg->nsmooths,&pc_gamg_agg->nsmooths,NULL);CHKERRQ(ierr);

    /* -pc_gamg_sym_graph */
    pc_gamg_agg->sym_graph = PETSC_FALSE;

    ierr = PetscOptionsBool("-pc_gamg_sym_graph","Set for asymmetric matrices","PCGAMGSetSymGraph",pc_gamg_agg->sym_graph,&pc_gamg_agg->sym_graph,NULL);CHKERRQ(ierr);

    /* -pc_gamg_square_graph */
    pc_gamg_agg->square_graph = PETSC_TRUE;

    ierr = PetscOptionsBool("-pc_gamg_square_graph","For faster coarsening and lower coarse grid complexity","PCGAMGSetSquareGraph",pc_gamg_agg->square_graph,&pc_gamg_agg->square_graph,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCDestroy_AGG

  Input Parameter:
   . pc -
*/
#undef __FUNCT__
#define __FUNCT__ "PCDestroy_GAMG_AGG"
PetscErrorCode PCDestroy_GAMG_AGG(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg          = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg     = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  ierr = PetscFree(pc_gamg->subctx);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCSetCoordinates_AGG
     - collective

   Input Parameter:
   . pc - the preconditioner context
   . ndm - dimesion of data (used for dof/vertex for Stokes)
   . a_nloc - number of vertices local
   . coords - [a_nloc][ndm] - interleaved coordinate data: {x_0, y_0, z_0, x_1, y_1, ...}
*/

#undef __FUNCT__
#define __FUNCT__ "PCSetCoordinates_AGG"
static PetscErrorCode PCSetCoordinates_AGG(PC pc, PetscInt ndm, PetscInt a_nloc, PetscReal *coords)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscErrorCode ierr;
  PetscInt       arrsz,kk,ii,jj,nloc,ndatarows,ndf;
  Mat            mat = pc->pmat;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  nloc = a_nloc;

  /* SA: null space vectors */
  ierr = MatGetBlockSize(mat, &ndf);CHKERRQ(ierr); /* this does not work for Stokes */
  if (coords && ndf==1) pc_gamg->data_cell_cols = 1; /* scalar w/ coords and SA (not needed) */
  else if (coords) {
    if (ndm > ndf) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"degrees of motion %d > block size %d",ndm,ndf);
    pc_gamg->data_cell_cols = (ndm==2 ? 3 : 6); /* displacement elasticity */
    if (ndm != ndf) {
      if (pc_gamg->data_cell_cols != ndf) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Don't know how to create null space for ndm=%d, ndf=%d.  Use MatSetNearNullSpace.",ndm,ndf);
    }
  } else pc_gamg->data_cell_cols = ndf; /* no data, force SA with constant null space vectors */
  pc_gamg->data_cell_rows = ndatarows = ndf;
  if (pc_gamg->data_cell_cols <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"pc_gamg->data_cell_cols %D <= 0",pc_gamg->data_cell_cols);
  arrsz = nloc*pc_gamg->data_cell_rows*pc_gamg->data_cell_cols;

  /* create data - syntactic sugar that should be refactored at some point */
  if (pc_gamg->data==0 || (pc_gamg->data_sz != arrsz)) {
    ierr = PetscFree(pc_gamg->data);CHKERRQ(ierr);
    ierr = PetscMalloc1(arrsz+1, &pc_gamg->data);CHKERRQ(ierr);
  }
  /* copy data in - column oriented */
  for (kk=0; kk<nloc; kk++) {
    const PetscInt M     = nloc*pc_gamg->data_cell_rows; /* stride into data */
    PetscReal      *data = &pc_gamg->data[kk*ndatarows]; /* start of cell */
    if (pc_gamg->data_cell_cols==1) *data = 1.0;
    else {
      /* translational modes */
      for (ii=0;ii<ndatarows;ii++) {
        for (jj=0;jj<ndatarows;jj++) {
          if (ii==jj)data[ii*M + jj] = 1.0;
          else data[ii*M + jj] = 0.0;
        }
      }

      /* rotational modes */
      if (coords) {
        if (ndm == 2) {
          data   += 2*M;
          data[0] = -coords[2*kk+1];
          data[1] =  coords[2*kk];
        } else {
          data   += 3*M;
          data[0] = 0.0;             data[M+0] =  coords[3*kk+2]; data[2*M+0] = -coords[3*kk+1];
          data[1] = -coords[3*kk+2]; data[M+1] = 0.0;             data[2*M+1] =  coords[3*kk];
          data[2] =  coords[3*kk+1]; data[M+2] = -coords[3*kk];   data[2*M+2] = 0.0;
        }
      }
    }
  }

  pc_gamg->data_sz = arrsz;
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */
/*
   PCSetData_AGG - called if data is not set with PCSetCoordinates.
      Looks in Mat for near null space.
      Does not work for Stokes

  Input Parameter:
   . pc -
   . a_A - matrix to get (near) null space out of.
*/
#undef __FUNCT__
#define __FUNCT__ "PCSetData_AGG"
PetscErrorCode PCSetData_AGG(PC pc, Mat a_A)
{
  PetscErrorCode ierr;
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  MatNullSpace   mnull;

  PetscFunctionBegin;
  ierr = MatGetNearNullSpace(a_A, &mnull);CHKERRQ(ierr);
  if (!mnull) {
    PetscInt bs,NN,MM;
    ierr = MatGetBlockSize(a_A, &bs);CHKERRQ(ierr);
    ierr = MatGetLocalSize(a_A, &MM, &NN);CHKERRQ(ierr);
    if (MM % bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MM %D must be divisible by bs %D",MM,bs);
    ierr = PCSetCoordinates_AGG(pc, bs, MM/bs, NULL);CHKERRQ(ierr);
  } else {
    PetscReal *nullvec;
    PetscBool has_const;
    PetscInt  i,j,mlocal,nvec,bs;
    const Vec *vecs; const PetscScalar *v;
    ierr = MatGetLocalSize(a_A,&mlocal,NULL);CHKERRQ(ierr);
    ierr = MatNullSpaceGetVecs(mnull, &has_const, &nvec, &vecs);CHKERRQ(ierr);
    pc_gamg->data_sz = (nvec+!!has_const)*mlocal;
    ierr = PetscMalloc1((nvec+!!has_const)*mlocal,&nullvec);CHKERRQ(ierr);
    if (has_const) for (i=0; i<mlocal; i++) nullvec[i] = 1.0;
    for (i=0; i<nvec; i++) {
      ierr = VecGetArrayRead(vecs[i],&v);CHKERRQ(ierr);
      for (j=0; j<mlocal; j++) nullvec[(i+!!has_const)*mlocal + j] = PetscRealPart(v[j]);
      ierr = VecRestoreArrayRead(vecs[i],&v);CHKERRQ(ierr);
    }
    pc_gamg->data           = nullvec;
    pc_gamg->data_cell_cols = (nvec+!!has_const);

    ierr = MatGetBlockSize(a_A, &bs);CHKERRQ(ierr);

    pc_gamg->data_cell_rows = bs;
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
 formProl0

   Input Parameter:
   . agg_llists - list of arrays with aggregates
   . bs - block size
   . nSAvec - column bs of new P
   . my0crs - global index of start of locals
   . data_stride - bs*(nloc nodes + ghost nodes)
   . data_in[data_stride*nSAvec] - local data on fine grid
   . flid_fgid[data_stride/bs] - make local to global IDs, includes ghosts in 'locals_llist'
  Output Parameter:
   . a_data_out - in with fine grid data (w/ghosts), out with coarse grid data
   . a_Prol - prolongation operator
*/
#undef __FUNCT__
#define __FUNCT__ "formProl0"
static PetscErrorCode formProl0(const PetscCoarsenData *agg_llists, /* list from selected vertices of aggregate unselected vertices */
                                const PetscInt bs,          /* (row) block size */
                                const PetscInt nSAvec,      /* column bs */
                                const PetscInt my0crs,      /* global index of start of locals */
                                const PetscInt data_stride, /* (nloc+nghost)*bs */
                                PetscReal      data_in[],   /* [data_stride][nSAvec] */
                                const PetscInt flid_fgid[], /* [data_stride/bs] */
                                PetscReal **a_data_out,
                                Mat a_Prol) /* prolongation operator (output)*/
{
  PetscErrorCode ierr;
  PetscInt       Istart,my0,Iend,nloc,clid,flid,aggID,kk,jj,ii,mm,ndone,nSelected,minsz,nghosts,out_data_stride;
  MPI_Comm       comm;
  PetscMPIInt    rank, size;
  PetscReal      *out_data;
  PetscCDPos     pos;
  GAMGHashTable  fgid_flid;

/* #define OUT_AGGS */
#if defined(OUT_AGGS)
  static PetscInt llev = 0; char fname[32]; FILE *file = NULL; PetscInt pM;
#endif

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)a_Prol,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(a_Prol, &Istart, &Iend);CHKERRQ(ierr);
  nloc = (Iend-Istart)/bs; my0 = Istart/bs;
  if ((Iend-Istart) % bs) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Iend %D - Istart %d must be divisible by bs %D",Iend,Istart,bs);
  Iend   /= bs;
  nghosts = data_stride/bs - nloc;

  ierr = GAMGTableCreate(2*nghosts+1, &fgid_flid);CHKERRQ(ierr);
  for (kk=0; kk<nghosts; kk++) {
    ierr = GAMGTableAdd(&fgid_flid, flid_fgid[nloc+kk], nloc+kk);CHKERRQ(ierr);
  }

#if defined(OUT_AGGS)
  sprintf(fname,"aggs_%d_%d.m",llev++,rank);
  if (llev==1) file = fopen(fname,"w");
  MatGetSize(a_Prol, &pM, &jj);
#endif

  /* count selected -- same as number of cols of P */
  for (nSelected=mm=0; mm<nloc; mm++) {
    PetscBool ise;
    ierr = PetscCDEmptyAt(agg_llists, mm, &ise);CHKERRQ(ierr);
    if (!ise) nSelected++;
  }
  ierr = MatGetOwnershipRangeColumn(a_Prol, &ii, &jj);CHKERRQ(ierr);
  if ((ii/nSAvec) != my0crs) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ii %D /nSAvec %D  != my0crs %D",ii,nSAvec,my0crs);
  if (nSelected != (jj-ii)/nSAvec) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"nSelected %D != (jj %D - ii %D)/nSAvec %D",nSelected,jj,ii,nSAvec);

  /* aloc space for coarse point data (output) */
  out_data_stride = nSelected*nSAvec;

  ierr = PetscMalloc1(out_data_stride*nSAvec, &out_data);CHKERRQ(ierr);
  for (ii=0;ii<out_data_stride*nSAvec;ii++) out_data[ii]=PETSC_MAX_REAL;
  *a_data_out = out_data; /* output - stride nSelected*nSAvec */

  /* find points and set prolongation */
  minsz = 100;
  ndone = 0;
  for (mm = clid = 0; mm < nloc; mm++) {
    ierr = PetscCDSizeAt(agg_llists, mm, &jj);CHKERRQ(ierr);
    if (jj > 0) {
      const PetscInt lid = mm, cgid = my0crs + clid;
      PetscInt       cids[100]; /* max bs */
      PetscBLASInt   asz  =jj,M=asz*bs,N=nSAvec,INFO;
      PetscBLASInt   Mdata=M+((N-M>0) ? N-M : 0),LDA=Mdata,LWORK=N*bs;
      PetscScalar    *qqc,*qqr,*TAU,*WORK;
      PetscInt       *fids;
      PetscReal      *data;
      /* count agg */
      if (asz<minsz) minsz = asz;

      /* get block */
      ierr = PetscMalloc1(Mdata*N, &qqc);CHKERRQ(ierr);
      ierr = PetscMalloc1(M*N, &qqr);CHKERRQ(ierr);
      ierr = PetscMalloc1(N, &TAU);CHKERRQ(ierr);
      ierr = PetscMalloc1(LWORK, &WORK);CHKERRQ(ierr);
      ierr = PetscMalloc1(M, &fids);CHKERRQ(ierr);

      aggID = 0;
      ierr  = PetscCDGetHeadPos(agg_llists,lid,&pos);CHKERRQ(ierr);
      while (pos) {
        PetscInt gid1;
        ierr = PetscLLNGetID(pos, &gid1);CHKERRQ(ierr);
        ierr = PetscCDGetNextPos(agg_llists,lid,&pos);CHKERRQ(ierr);

        if (gid1 >= my0 && gid1 < Iend) flid = gid1 - my0;
        else {
          ierr = GAMGTableFind(&fgid_flid, gid1, &flid);CHKERRQ(ierr);
          if (flid < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot find gid1 in table");
        }
        /* copy in B_i matrix - column oriented */
        data = &data_in[flid*bs];
        for (kk = ii = 0; ii < bs; ii++) {
          for (jj = 0; jj < N; jj++) {
            PetscReal d = data[jj*data_stride + ii];
            qqc[jj*Mdata + aggID*bs + ii] = d;
          }
        }
#if defined(OUT_AGGS)
        if (llev==1) {
          char     str[] = "plot(%e,%e,'r*'), hold on,\n", col[] = "rgbkmc", sim[] = "*os+h>d<vx^";
          PetscInt MM,pi,pj;
          str[12] = col[(clid+17*rank)%6]; str[13] = sim[(clid+17*rank)%11];
          MM      = (PetscInt)(PetscSqrtReal((PetscReal)pM));
          pj      = gid1/MM; pi = gid1%MM;
          fprintf(file,str,(double)pi,(double)pj);
          /* fprintf(file,str,data[2*data_stride+1],-data[2*data_stride]); */
        }
#endif
        /* set fine IDs */
        for (kk=0; kk<bs; kk++) fids[aggID*bs + kk] = flid_fgid[flid]*bs + kk;

        aggID++;
      }

      /* pad with zeros */
      for (ii = asz*bs; ii < Mdata; ii++) {
        for (jj = 0; jj < N; jj++, kk++) {
          qqc[jj*Mdata + ii] = .0;
        }
      }

      ndone += aggID;
      /* QR */
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
      PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&Mdata, &N, qqc, &LDA, TAU, WORK, &LWORK, &INFO));
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
      if (INFO != 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"xGEQRF error");
      /* get R - column oriented - output B_{i+1} */
      {
        PetscReal *data = &out_data[clid*nSAvec];
        for (jj = 0; jj < nSAvec; jj++) {
          for (ii = 0; ii < nSAvec; ii++) {
            if (data[jj*out_data_stride + ii] != PETSC_MAX_REAL) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"data[jj*out_data_stride + ii] != %e",PETSC_MAX_REAL);
           if (ii <= jj) data[jj*out_data_stride + ii] = PetscRealPart(qqc[jj*Mdata + ii]);
           else data[jj*out_data_stride + ii] = 0.;
          }
        }
      }

      /* get Q - row oriented */
      PetscStackCallBLAS("LAPACKungqr",LAPACKungqr_(&Mdata, &N, &N, qqc, &LDA, TAU, WORK, &LWORK, &INFO));
      if (INFO != 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"xORGQR error arg %d",-INFO);

      for (ii = 0; ii < M; ii++) {
        for (jj = 0; jj < N; jj++) {
          qqr[N*ii + jj] = qqc[jj*Mdata + ii];
        }
      }

      /* add diagonal block of P0 */
      for (kk=0; kk<N; kk++) {
        cids[kk] = N*cgid + kk; /* global col IDs in P0 */
      }
      ierr = MatSetValues(a_Prol,M,fids,N,cids,qqr,INSERT_VALUES);CHKERRQ(ierr);

      ierr = PetscFree(qqc);CHKERRQ(ierr);
      ierr = PetscFree(qqr);CHKERRQ(ierr);
      ierr = PetscFree(TAU);CHKERRQ(ierr);
      ierr = PetscFree(WORK);CHKERRQ(ierr);
      ierr = PetscFree(fids);CHKERRQ(ierr);
      clid++;
    } /* coarse agg */
  } /* for all fine nodes */
  ierr = MatAssemblyBegin(a_Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(a_Prol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

/* ierr = MPI_Allreduce(&ndone, &ii, 1, MPIU_INT, MPIU_SUM, comm); */
/* MatGetSize(a_Prol, &kk, &jj); */
/* ierr = MPI_Allreduce(&minsz, &jj, 1, MPIU_INT, MPIU_MIN, comm); */
/* PetscPrintf(comm," **** [%d]%s %d total done, %d nodes (%d local done), min agg. size = %d\n",rank,__FUNCT__,ii,kk/bs,ndone,jj); */

#if defined(OUT_AGGS)
  if (llev==1) fclose(file);
#endif
  ierr = GAMGTableDestroy(&fgid_flid);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGgraph_AGG

  Input Parameter:
   . pc - this
   . Amat - matrix on this fine level
  Output Parameter:
   . a_Gmat -
*/
#undef __FUNCT__
#define __FUNCT__ "PCGAMGgraph_AGG"
PetscErrorCode PCGAMGgraph_AGG(PC pc,const Mat Amat,Mat *a_Gmat)
{
  PetscErrorCode            ierr;
  PC_MG                     *mg          = (PC_MG*)pc->data;
  PC_GAMG                   *pc_gamg     = (PC_GAMG*)mg->innerctx;
  const PetscInt            verbose      = pc_gamg->verbose;
  const PetscReal           vfilter      = pc_gamg->threshold;
  PC_GAMG_AGG               *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;
  PetscMPIInt               rank,size;
  Mat                       Gmat;
  MPI_Comm                  comm;
  PetscBool /* set,flg , */ symm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)Amat,&comm);CHKERRQ(ierr);
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(PC_GAMGGgraph_AGG,0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);

  /* ierr = MatIsSymmetricKnown(Amat, &set, &flg);CHKERRQ(ierr); || !(set && flg) -- this causes lot of symm calls */
  symm = (PetscBool)(pc_gamg_agg->sym_graph); /* && !pc_gamg_agg->square_graph; */

  ierr = PCGAMGCreateGraph(Amat, &Gmat);CHKERRQ(ierr);
  ierr = PCGAMGFilterGraph(&Gmat, vfilter, symm, verbose);CHKERRQ(ierr);

  *a_Gmat = Gmat;

  if (Amat != Gmat) {
    ierr = PetscObjectCompose((PetscObject)(*a_Gmat),"PCGAMGgraph_AGG_Amat",(PetscObject)Amat);CHKERRQ(ierr);
  }

#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(PC_GAMGGgraph_AGG,0,0,0,0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGCoarsen_AGG

  Input Parameter:
   . a_pc - this
  Input/Output Parameter:
   . a_Gmat1 - graph on this fine level - coarsening can change this (squares it)
  Output Parameter:
   . agg_lists - list of aggregates
*/
#undef __FUNCT__
#define __FUNCT__ "PCGAMGCoarsen_AGG"
PetscErrorCode PCGAMGCoarsen_AGG(PC a_pc,Mat *a_Gmat1,PetscCoarsenData **agg_lists)
{
  PetscErrorCode ierr;
  PC_MG          *mg          = (PC_MG*)a_pc->data;
  PC_GAMG        *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG    *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;
  Mat            mat,Gmat2, Gmat1 = *a_Gmat1;  /* squared graph */
  const PetscInt verbose = pc_gamg->verbose;
  IS             perm;
  PetscInt       Ii,nloc,bs,n,m;
  PetscInt       *permute;
  PetscBool      *bIndexSet;
  MatCoarsen     crs;
  MPI_Comm       comm;
  PetscMPIInt    rank,size;

  PetscFunctionBegin;
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(PC_GAMGCoarsen_AGG,0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = PetscObjectGetComm((PetscObject)Gmat1,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MatGetLocalSize(Gmat1, &n, &m);CHKERRQ(ierr);
  ierr = MatGetBlockSize(Gmat1, &bs);CHKERRQ(ierr);
  if (bs != 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"bs %D must be 1",bs);
  nloc = n/bs;

  if (pc_gamg_agg->square_graph) {
    PetscObject Amat_obj;

    if (verbose > 1) PetscPrintf(comm,"[%d]%s square graph\n",rank,__FUNCT__);
    /* ierr = MatMatTransposeMult(Gmat1, Gmat1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Gmat2); */
    ierr = MatTransposeMatMult(Gmat1, Gmat1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Gmat2);CHKERRQ(ierr);
    if (verbose > 2) {
      ierr = PetscPrintf(comm,"[%d]%s square graph done\n",rank,__FUNCT__);CHKERRQ(ierr);
      /* check for symetry */
      if (verbose > 4) {

      }
    }
    ierr = PetscObjectQuery((PetscObject)Gmat1,"PCGAMGgraph_AGG_Amat",&Amat_obj);CHKERRQ(ierr);
    if (Amat_obj) {
      ierr = PetscObjectCompose((PetscObject)Gmat2,"PCGAMGgraph_AGG_Amat",Amat_obj);CHKERRQ(ierr);
    }
    else {
      ierr = PetscObjectCompose((PetscObject)Gmat2,"PCGAMGgraph_AGG_Amat",(PetscObject)Gmat1);CHKERRQ(ierr);
    }
  } else Gmat2 = Gmat1;

  /* get MIS aggs */
  /* randomize */
  ierr = PetscMalloc1(nloc, &permute);CHKERRQ(ierr);
  ierr = PetscMalloc1(nloc, &bIndexSet);CHKERRQ(ierr);
  for (Ii = 0; Ii < nloc; Ii++) {
    bIndexSet[Ii] = PETSC_FALSE;
    permute[Ii]   = Ii;
  }
  srand(1); /* make deterministic */
  for (Ii = 0; Ii < nloc; Ii++) {
    PetscInt iSwapIndex = rand()%nloc;
    if (!bIndexSet[iSwapIndex] && iSwapIndex != Ii) {
      PetscInt iTemp = permute[iSwapIndex];
      permute[iSwapIndex]   = permute[Ii];
      permute[Ii]           = iTemp;
      bIndexSet[iSwapIndex] = PETSC_TRUE;
    }
  }
  ierr = PetscFree(bIndexSet);CHKERRQ(ierr);

  if (verbose > 1) PetscPrintf(comm,"[%d]%s coarsen graph\n",rank,__FUNCT__);

  ierr = ISCreateGeneral(PETSC_COMM_SELF, nloc, permute, PETSC_USE_POINTER, &perm);CHKERRQ(ierr);
#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET4],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = MatCoarsenCreate(comm, &crs);CHKERRQ(ierr);
  /* ierr = MatCoarsenSetType(crs, MATCOARSENMIS);CHKERRQ(ierr); */
  ierr = MatCoarsenSetFromOptions(crs);CHKERRQ(ierr);
  ierr = MatCoarsenSetGreedyOrdering(crs, perm);CHKERRQ(ierr);
  ierr = MatCoarsenSetAdjacency(crs, Gmat2);CHKERRQ(ierr);
  ierr = MatCoarsenSetVerbose(crs, pc_gamg->verbose);CHKERRQ(ierr);
  ierr = MatCoarsenSetStrictAggs(crs, PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatCoarsenApply(crs);CHKERRQ(ierr);
  ierr = MatCoarsenGetData(crs, agg_lists);CHKERRQ(ierr); /* output */
  ierr = MatCoarsenDestroy(&crs);CHKERRQ(ierr);

  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = PetscFree(permute);CHKERRQ(ierr);
#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET4],0,0,0,0);CHKERRQ(ierr);
#endif
  if (verbose > 2) PetscPrintf(comm,"[%d]%s coarsen graph done\n",rank,__FUNCT__);

  /* smooth aggs */
  if (Gmat2 != Gmat1) {
    const PetscCoarsenData *llist = *agg_lists;
    ierr     = MatDestroy(&Gmat1);CHKERRQ(ierr);
    *a_Gmat1 = Gmat2; /* output */
    ierr     = PetscCDGetMat(llist, &mat);CHKERRQ(ierr);
    if (mat) SETERRQ(comm,PETSC_ERR_ARG_WRONG, "Auxilary matrix with squared graph????");
  } else {
    const PetscCoarsenData *llist = *agg_lists;
    /* see if we have a matrix that takes precedence (returned from MatCoarsenAppply) */
    ierr = PetscCDGetMat(llist, &mat);CHKERRQ(ierr);
    if (mat) {
      ierr     = MatDestroy(&Gmat1);CHKERRQ(ierr);
      *a_Gmat1 = mat; /* output */
    }
  }
#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(PC_GAMGCoarsen_AGG,0,0,0,0);CHKERRQ(ierr);
#endif
  if (verbose > 2) PetscPrintf(comm,"[%d]%s PCGAMGCoarsen_AGG done\n",rank,__FUNCT__);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
 PCGAMGProlongator_AGG

 Input Parameter:
 . pc - this
 . Amat - matrix on this fine level
 . Graph - used to get ghost data for nodes in
 . agg_lists - list of aggregates
 Output Parameter:
 . a_P_out - prolongation operator to the next level
 */
#undef __FUNCT__
#define __FUNCT__ "PCGAMGProlongator_AGG"
PetscErrorCode PCGAMGProlongator_AGG(PC pc,const Mat Amat,const Mat Gmat,PetscCoarsenData *agg_lists,Mat *a_P_out)
{
  PC_MG          *mg       = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg  = (PC_GAMG*)mg->innerctx;
  const PetscInt verbose   = pc_gamg->verbose;
  const PetscInt data_cols = pc_gamg->data_cell_cols;
  PetscErrorCode ierr;
  PetscInt       Istart,Iend,nloc,ii,jj,kk,my0,nLocalSelected,bs;
  Mat            Prol;
  PetscMPIInt    rank, size;
  MPI_Comm       comm;
  const PetscInt col_bs = data_cols;
  PetscReal      *data_w_ghost;
  PetscInt       myCrs0, nbnodes=0, *flid_fgid;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)Amat,&comm);CHKERRQ(ierr);
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(PC_GAMGProlongator_AGG,0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Amat, &Istart, &Iend);CHKERRQ(ierr);
  ierr = MatGetBlockSize(Amat, &bs);CHKERRQ(ierr);
  nloc = (Iend-Istart)/bs; my0 = Istart/bs;
  if ((Iend-Istart) % bs) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"(Iend %D - Istart %D) not divisible by bs %D",Iend,Istart,bs);

  /* get 'nLocalSelected' */
  for (ii=0, nLocalSelected = 0; ii < nloc; ii++) {
    PetscBool ise;
    /* filter out singletons 0 or 1? */
    ierr = PetscCDEmptyAt(agg_lists, ii, &ise);CHKERRQ(ierr);
    if (!ise) nLocalSelected++;
  }

  /* create prolongator, create P matrix */
  ierr = MatCreate(comm, &Prol);CHKERRQ(ierr);
  ierr = MatSetSizes(Prol,nloc*bs,nLocalSelected*col_bs,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(Prol, bs, col_bs);CHKERRQ(ierr);
  ierr = MatSetType(Prol, MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Prol, data_cols, NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Prol,data_cols, NULL,data_cols, NULL);CHKERRQ(ierr);
  /* nloc*bs, nLocalSelected*col_bs, */
  /* PETSC_DETERMINE, PETSC_DETERMINE, */
  /* data_cols, NULL, data_cols, NULL, */
  /* &Prol); */

  /* can get all points "removed" */
  ierr =  MatGetSize(Prol, &kk, &ii);CHKERRQ(ierr);
  if (ii==0) {
    if (verbose > 0) PetscPrintf(comm,"[%d]%s no selected points on coarse grid\n",rank,__FUNCT__);
    ierr = MatDestroy(&Prol);CHKERRQ(ierr);
    *a_P_out = NULL;  /* out */
    PetscFunctionReturn(0);
  }
  if (verbose > 0) PetscPrintf(comm,"\t\t[%d]%s New grid %d nodes\n",rank,__FUNCT__,ii/col_bs);
  ierr = MatGetOwnershipRangeColumn(Prol, &myCrs0, &kk);CHKERRQ(ierr);

  if ((kk-myCrs0) % col_bs) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"(kk %D -myCrs0 %D) not divisible by col_bs %D",kk,myCrs0,col_bs);
  myCrs0 = myCrs0/col_bs;
  if ((kk/col_bs-myCrs0) != nLocalSelected) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"(kk %D/col_bs %D - myCrs0 %D) != nLocalSelected %D)",kk,col_bs,myCrs0,nLocalSelected);

  /* create global vector of data in 'data_w_ghost' */
#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET7],0,0,0,0);CHKERRQ(ierr);
#endif
  if (size > 1) { /*  */
    PetscReal *tmp_gdata,*tmp_ldata,*tp2;
    ierr = PetscMalloc1(nloc, &tmp_ldata);CHKERRQ(ierr);
    for (jj = 0; jj < data_cols; jj++) {
      for (kk = 0; kk < bs; kk++) {
        PetscInt        ii,stride;
        const PetscReal *tp = pc_gamg->data + jj*bs*nloc + kk;
        for (ii = 0; ii < nloc; ii++, tp += bs) tmp_ldata[ii] = *tp;

        ierr = PCGAMGGetDataWithGhosts(Gmat, 1, tmp_ldata, &stride, &tmp_gdata);CHKERRQ(ierr);

        if (jj==0 && kk==0) { /* now I know how many todal nodes - allocate */
          ierr    = PetscMalloc1(stride*bs*data_cols, &data_w_ghost);CHKERRQ(ierr);
          nbnodes = bs*stride;
        }
        tp2 = data_w_ghost + jj*bs*stride + kk;
        for (ii = 0; ii < stride; ii++, tp2 += bs) *tp2 = tmp_gdata[ii];
        ierr = PetscFree(tmp_gdata);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(tmp_ldata);CHKERRQ(ierr);
  } else {
    nbnodes      = bs*nloc;
    data_w_ghost = (PetscReal*)pc_gamg->data;
  }

  /* get P0 */
  if (size > 1) {
    PetscReal *fid_glid_loc,*fiddata;
    PetscInt  stride;

    ierr = PetscMalloc1(nloc, &fid_glid_loc);CHKERRQ(ierr);
    for (kk=0; kk<nloc; kk++) fid_glid_loc[kk] = (PetscReal)(my0+kk);
    ierr = PCGAMGGetDataWithGhosts(Gmat, 1, fid_glid_loc, &stride, &fiddata);CHKERRQ(ierr);
    ierr = PetscMalloc1(stride, &flid_fgid);CHKERRQ(ierr);
    for (kk=0; kk<stride; kk++) flid_fgid[kk] = (PetscInt)fiddata[kk];
    ierr = PetscFree(fiddata);CHKERRQ(ierr);

    if (stride != nbnodes/bs) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"stride %D != nbnodes %D/bs %D",stride,nbnodes,bs);
    ierr = PetscFree(fid_glid_loc);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc1(nloc, &flid_fgid);CHKERRQ(ierr);
    for (kk=0; kk<nloc; kk++) flid_fgid[kk] = my0 + kk;
  }
#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET7],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET8],0,0,0,0);CHKERRQ(ierr);
#endif
  {
    PetscReal *data_out = NULL;
    ierr = formProl0(agg_lists, bs, data_cols, myCrs0, nbnodes,data_w_ghost, flid_fgid, &data_out, Prol);CHKERRQ(ierr);
    ierr = PetscFree(pc_gamg->data);CHKERRQ(ierr);

    pc_gamg->data           = data_out;
    pc_gamg->data_cell_rows = data_cols;
    pc_gamg->data_sz        = data_cols*data_cols*nLocalSelected;
  }
#if defined PETSC_GAMG_USE_LOG
  ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET8],0,0,0,0);CHKERRQ(ierr);
#endif
  if (size > 1) ierr = PetscFree(data_w_ghost);CHKERRQ(ierr);
  ierr = PetscFree(flid_fgid);CHKERRQ(ierr);

  *a_P_out = Prol;  /* out */

#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(PC_GAMGProlongator_AGG,0,0,0,0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCGAMGOptprol_AGG

  Input Parameter:
   . pc - this
   . Amat - matrix on this fine level
 In/Output Parameter:
   . a_P_out - prolongation operator to the next level
*/
#undef __FUNCT__
#define __FUNCT__ "PCGAMGOptprol_AGG"
PetscErrorCode PCGAMGOptprol_AGG(PC pc,const Mat Amat,Mat *a_P)
{
  PetscErrorCode ierr;
  PC_MG          *mg          = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg     = (PC_GAMG*)mg->innerctx;
  const PetscInt verbose      = pc_gamg->verbose;
  PC_GAMG_AGG    *pc_gamg_agg = (PC_GAMG_AGG*)pc_gamg->subctx;
  PetscInt       jj;
  PetscMPIInt    rank,size;
  Mat            Prol  = *a_P;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)Amat,&comm);CHKERRQ(ierr);
#if defined PETSC_USE_LOG
  ierr = PetscLogEventBegin(PC_GAMGOptprol_AGG,0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);

  /* smooth P0 */
  for (jj = 0; jj < pc_gamg_agg->nsmooths; jj++) {
    Mat       tMat;
    Vec       diag;
    PetscReal alpha, emax, emin;
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventBegin(petsc_gamg_setup_events[SET9],0,0,0,0);CHKERRQ(ierr);
#endif
    if (jj == 0) {
      KSP eksp;
      Vec bb, xx;
      PC  epc;
      ierr = MatCreateVecs(Amat, &bb, 0);CHKERRQ(ierr);
      ierr = MatCreateVecs(Amat, &xx, 0);CHKERRQ(ierr);
      {
        PetscRandom rctx;
        ierr = PetscRandomCreate(comm,&rctx);CHKERRQ(ierr);
        ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
        ierr = VecSetRandom(bb,rctx);CHKERRQ(ierr);
        ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
      }

      /* zeroing out BC rows -- needed for crazy matrices */
      {
        PetscInt    Istart,Iend,ncols,jj,Ii;
        PetscScalar zero = 0.0;
        ierr = MatGetOwnershipRange(Amat, &Istart, &Iend);CHKERRQ(ierr);
        for (Ii = Istart, jj = 0; Ii < Iend; Ii++, jj++) {
          ierr = MatGetRow(Amat,Ii,&ncols,0,0);CHKERRQ(ierr);
          if (ncols <= 1) {
            ierr = VecSetValues(bb, 1, &Ii, &zero, INSERT_VALUES);CHKERRQ(ierr);
          }
          ierr = MatRestoreRow(Amat,Ii,&ncols,0,0);CHKERRQ(ierr);
        }
        ierr = VecAssemblyBegin(bb);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(bb);CHKERRQ(ierr);
      }

      ierr = KSPCreate(comm,&eksp);CHKERRQ(ierr);
      ierr = KSPSetTolerances(eksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,10);CHKERRQ(ierr);
      ierr = KSPSetNormType(eksp, KSP_NORM_NONE);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(eksp,((PetscObject)pc)->prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(eksp, "gamg_est_");CHKERRQ(ierr);
      ierr = KSPSetFromOptions(eksp);CHKERRQ(ierr);

      ierr = KSPSetInitialGuessNonzero(eksp, PETSC_FALSE);CHKERRQ(ierr);
      ierr = KSPSetOperators(eksp, Amat, Amat);CHKERRQ(ierr);
      ierr = KSPSetComputeSingularValues(eksp,PETSC_TRUE);CHKERRQ(ierr);

      ierr = KSPGetPC(eksp, &epc);CHKERRQ(ierr);
      ierr = PCSetType(epc, PCJACOBI);CHKERRQ(ierr);  /* smoother in smoothed agg. */

      /* solve - keep stuff out of logging */
      ierr = PetscLogEventDeactivate(KSP_Solve);CHKERRQ(ierr);
      ierr = PetscLogEventDeactivate(PC_Apply);CHKERRQ(ierr);
      ierr = KSPSolve(eksp, bb, xx);CHKERRQ(ierr);
      ierr = PetscLogEventActivate(KSP_Solve);CHKERRQ(ierr);
      ierr = PetscLogEventActivate(PC_Apply);CHKERRQ(ierr);

      ierr = KSPComputeExtremeSingularValues(eksp, &emax, &emin);CHKERRQ(ierr);
      if (verbose > 0) {
        PetscPrintf(comm,"\t\t\t%s smooth P0: max eigen=%e min=%e PC=%s\n",
                    __FUNCT__,emax,emin,PCJACOBI);
      }
      ierr = VecDestroy(&xx);CHKERRQ(ierr);
      ierr = VecDestroy(&bb);CHKERRQ(ierr);
      ierr = KSPDestroy(&eksp);CHKERRQ(ierr);

      if (pc_gamg->emax_id == -1) {
        ierr = PetscObjectComposedDataRegister(&pc_gamg->emax_id);CHKERRQ(ierr);
        if (pc_gamg->emax_id == -1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"pc_gamg->emax_id == -1");
      }
      ierr = PetscObjectComposedDataSetScalar((PetscObject)Amat, pc_gamg->emax_id, emax);CHKERRQ(ierr);
    }

    /* smooth P1 := (I - omega/lam D^{-1}A)P0 */
    ierr  = MatMatMult(Amat, Prol, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tMat);CHKERRQ(ierr);
    ierr  = MatCreateVecs(Amat, &diag, 0);CHKERRQ(ierr);
    ierr  = MatGetDiagonal(Amat, diag);CHKERRQ(ierr); /* effectively PCJACOBI */
    ierr  = VecReciprocal(diag);CHKERRQ(ierr);
    ierr  = MatDiagonalScale(tMat, diag, 0);CHKERRQ(ierr);
    ierr  = VecDestroy(&diag);CHKERRQ(ierr);
    alpha = -1.4/emax;
    ierr  = MatAYPX(tMat, alpha, Prol, SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr  = MatDestroy(&Prol);CHKERRQ(ierr);
    Prol  = tMat;
#if defined PETSC_GAMG_USE_LOG
    ierr = PetscLogEventEnd(petsc_gamg_setup_events[SET9],0,0,0,0);CHKERRQ(ierr);
#endif
  }
#if defined PETSC_USE_LOG
  ierr = PetscLogEventEnd(PC_GAMGOptprol_AGG,0,0,0,0);CHKERRQ(ierr);
#endif
  *a_P = Prol;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCCreateGAMG_AGG

  Input Parameter:
   . pc -
*/
#undef __FUNCT__
#define __FUNCT__ "PCCreateGAMG_AGG"
PetscErrorCode  PCCreateGAMG_AGG(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_AGG    *pc_gamg_agg;

  PetscFunctionBegin;
  /* create sub context for SA */
  ierr            = PetscNewLog(pc,&pc_gamg_agg);CHKERRQ(ierr);
  pc_gamg->subctx = pc_gamg_agg;

  pc_gamg->ops->setfromoptions = PCSetFromOptions_GAMG_AGG;
  pc_gamg->ops->destroy        = PCDestroy_GAMG_AGG;
  /* reset does not do anything; setup not virtual */

  /* set internal function pointers */
  pc_gamg->ops->graph       = PCGAMGgraph_AGG;
  pc_gamg->ops->coarsen     = PCGAMGCoarsen_AGG;
  pc_gamg->ops->prolongator = PCGAMGProlongator_AGG;
  pc_gamg->ops->optprol     = PCGAMGOptprol_AGG;

  pc_gamg->ops->createdefaultdata = PCSetData_AGG;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",PCSetCoordinates_AGG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* M. Adams, August 2016 */

static char help[] = "X2: A partical in cell code for slab plasmas using PICell.\n";

#ifdef H5PART
#include <H5Part.h>
#endif
#include <petsc/private/dmpicellimpl.h>    /*I   "petscdmpicell.h"   I*/
#include <assert.h>
#include <petscds.h>

PetscLogEvent s_events[22];
static const int diag_event_id = sizeof(s_events)/sizeof(s_events[0])-1;

/* copy to fit with viz outpout in array class */
#define cylindricalToCart( __R,  __Z,  __phi, __cart) \
{ \
 __cart[0] = __R;  \
 __cart[1] = __Z;  \
 __cart[2] = __phi;\
}

#include "x2_particle_array.h"
#include "x2_physics.h"

typedef struct {
  /* particle grid, flux tube, sizes */
  PetscInt ft_np[3];
  PetscInt ft_rank[3];
  /* solver grid sizes */
  PetscInt solver_np[3];
  PetscInt solver_proc_idx[3];
  /* geometry  */
  PetscReal dom_lo[3], dom_hi[3];
  PetscReal b0[3];
  /* context */
  void *ctx;
} X2GridParticle;
typedef enum {X2_DUMMY} runType;

#include "x2_ctx.h"

#define X2_IDX(i,j,k,np)  (np[1]*np[2]*i + np[2]*j + k)
#define X2_IDX3(ii,np)  (np[1]*np[2]*ii[0] + np[2]*ii[1] + ii[2])
#define X2_IDX_X(rank,np) (rank/(np[1]*np[2]))
#define X2_IDX_Y(rank,np) (rank%(np[1]*np[2])/np[2])
#define X2_IDX_Z(rank,np) (rank%np[2])

static PetscInt s_debug;
static PetscInt s_rank;
static int s_fluxtubeelem=0;
#define X2PROCLISTSIZE 256

/* X2GridSolverLocatePoints: find processor and element in solver grid that this one point is in
    Input:
     - dm: solver dm
     - ctx: context
     - xvec: Cartesian coordinates (native data), clobbers with fake local
   Output:
     - pes: process IDs
     - elemIDs: element IDs
*/
/*
  dm - The DM
  x - Cartesian coordinate

  pe - Rank of process owning the grid cell containing the particle, -1 if not found
  elemID - Local cell number on rank pe containing the particle, -1 if not found
*/
#undef __FUNCT__
#define __FUNCT__ "X2GridSolverLocatePoints"
PetscErrorCode X2GridSolverLocatePoints(DM dm,  Vec xvec, const X2Ctx *ctx,  IS *pes, IS *elemIDs)
{
  PetscInt       i,idxs[3],n,nn,dim,ii;
  const PetscInt *np = ctx->particleGrid.solver_np;
  const PetscReal *dlo = ctx->particleGrid.dom_lo, *dhi = ctx->particleGrid.dom_hi;
  PetscErrorCode ierr;
  PetscScalar *xx,*xx0;
  PetscInt *peidxs,*elemidxs;
  PetscSF            cellSF = NULL;
  const PetscSFNode *foundCells;
  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(xvec, VEC_CLASSID, 2);
  PetscValidPointer(pes, 2);
  PetscValidPointer(elemIDs, 3);
  /* find processor */
  ierr = DMGetCoordinateDim(dm, &dim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(xvec,&n);CHKERRQ(ierr);
  if (n%dim) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER, "n%dim n=%D",n);
  nn = n/dim;
  ierr = PetscMalloc2(nn,&peidxs,nn,&elemidxs);CHKERRQ(ierr);
  ierr = VecGetArray(xvec,&xx0);CHKERRQ(ierr); xx = xx0;
  for (ii=0;ii<nn;ii++,xx+=dim) {
    if (xx[0]<dlo[0] || xx[0]>dhi[0] || xx[1]<dlo[1] ||
        xx[1]>dhi[1] || xx[2]<dlo[2] || xx[2]>dhi[2])
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"point out of bounds %g %g %g",xx[0],xx[1],xx[2]);
    for (i=0;i<3;i++) {
      idxs[i] = (PetscInt)((xx[i]-dlo[i])/(dhi[i]-dlo[i])*(double)np[i]);
    }
    peidxs[ii] = X2_IDX(idxs[0],idxs[1],idxs[2],np); /* get pe */
    for (i=0;i<3;i++) { /* make fake coord for find */
      PetscReal dx = (dhi[i]-dlo[i])/np[i];
      PetscReal x0 = xx[i] - dlo[i] - idxs[i]*dx; assert(x0 >= 0 && x0 <= dx); /* local coord */
      xx[i] = x0 + dlo[i] + ctx->particleGrid.solver_proc_idx[i]*dx; /* my fake coordinate */
    }
  }
  ierr = VecRestoreArray(xvec,&xx0);CHKERRQ(ierr);
  ierr = DMLocatePoints(dm, xvec, DM_POINTLOCATION_NONE, &cellSF);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(cellSF, NULL, NULL, NULL, &foundCells);CHKERRQ(ierr);
  for (ii=0;ii<nn;ii++) {
    elemidxs[ii] = foundCells[ii].index;
    /* *pe = foundCells[0].rank; */
    if (elemidxs[ii]<0) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "missed point");
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nn,peidxs,PETSC_COPY_VALUES,pes);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nn,elemidxs,PETSC_COPY_VALUES,elemIDs);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&cellSF);CHKERRQ(ierr);
  ierr = PetscFree2(peidxs,elemidxs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include "x2_common.h"

/*
   ProcessOptions: set parameters from input, setup w/o allocation, called first, no DM here
*/
#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions( X2Ctx *ctx )
{
  PetscErrorCode ierr,k;
  PetscBool chunkFlag,npflag;
  PetscInt three = 3;
  PetscFunctionBeginUser;
  /* general */
  ierr = MPI_Comm_rank(ctx->wComm, &ctx->rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(ctx->wComm, &ctx->npe);CHKERRQ(ierr);
  s_rank = ctx->rank;
  /* physics */
  ctx->massAu=2;  /* mass ratio to proton */
  /* ctx->eMassAu=2e-2; /\* mass of electron?? *\/ */
  ctx->chargeEu=1;    /* charge number */
  ctx->eChargeEu=-1;  /* negative electron */

  ctx->species[1].mass=ctx->massAu*x2ProtMass;
  ctx->species[1].charge=ctx->chargeEu*x2ECharge;
  ctx->species[0].mass=x2ElecMass/* ctx->eMassAu*x2ProtMass */;
  ctx->species[0].charge=ctx->eChargeEu*x2ECharge;

  /* mesh */
  ctx->particleGrid.ft_np[0]  = 1;
  ctx->particleGrid.ft_np[1]  = 1;
  ctx->particleGrid.ft_np[2]  = 1;
  ctx->particleGrid.solver_np[0]  = 1;
  ctx->particleGrid.solver_np[1]  = 1;
  ctx->particleGrid.solver_np[2]  = 1;
  ctx->particleGrid.dom_hi[0]  = 1;
  ctx->particleGrid.dom_hi[1]  = 1;
  ctx->particleGrid.dom_hi[2]  = 1;
  ctx->particleGrid.dom_lo[0]  = 0;
  ctx->particleGrid.dom_lo[1]  = 0;
  ctx->particleGrid.dom_lo[2]  = 0;
  ctx->particleGrid.b0[0]  = .1;
  ctx->particleGrid.b0[1]  = .2;
  ctx->particleGrid.b0[2]  =  1; /* mostly in z */

  ctx->tablecount = 0;

  ierr = PetscOptionsBegin(ctx->wComm, "", "Poisson Problem Options", "X2");CHKERRQ(ierr);
  /* general options */
  s_debug = 0;
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex2.c", s_debug, &s_debug, NULL);CHKERRQ(ierr);
  ctx->plot = PETSC_TRUE;
  ierr = PetscOptionsBool("-plot", "Write plot files (particles)", "ex2.c", ctx->plot, &ctx->plot, NULL);CHKERRQ(ierr);
  ctx->chunksize = X2_V_LEN; /* too small */
  ierr = PetscOptionsInt("-chunksize", "Size of particle list to chunk sends", "ex2.c", ctx->chunksize, &ctx->chunksize,&chunkFlag);CHKERRQ(ierr);
  if (chunkFlag) ctx->chunksize = X2_V_LEN*(ctx->chunksize/X2_V_LEN);
  if (ctx->chunksize<=0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB," invalid chuck size = %D",ctx->chunksize);
  ctx->use_bsp = 0;
  ierr = PetscOptionsInt("-use_bsp", "Size of chucks for PETSc's TwoSide communication (0 to use 'nonblocking consensus')", "ex2.c", ctx->use_bsp, &ctx->use_bsp, NULL);CHKERRQ(ierr);
  if (ctx->use_bsp<0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB," invalid BSP chuck size = %D",ctx->use_bsp);
  ctx->proc_send_table_size = ((ctx->npe>10) ? 10 + (ctx->npe-10)/10 : ctx->npe) + 1; /* hash table size of processors to send to */
  ierr = PetscOptionsInt("-proc_send_table_size", "Size of hash table proc->send_list", "ex2.c",ctx->proc_send_table_size, &ctx->proc_send_table_size, NULL);CHKERRQ(ierr);

  /* Domain and mesh definition */
  ierr = PetscOptionsRealArray("-dom_hi", "Domain size", "ex2.c", ctx->particleGrid.dom_hi, &three, NULL);CHKERRQ(ierr);
  three = 3;
  ierr = PetscOptionsRealArray("-dom_lo", "Domain size", "ex2.c", ctx->particleGrid.dom_lo, &three, NULL);CHKERRQ(ierr);
  three = 3;
  ierr = PetscOptionsIntArray("-ft_np", "Number of (flux tube) processor in each dimension", "ex2.c", ctx->particleGrid.ft_np, &three, &npflag);CHKERRQ(ierr);
  if ( (k=ctx->particleGrid.ft_np[0]*ctx->particleGrid.ft_np[1]*ctx->particleGrid.ft_np[2]) != ctx->npe) { /* recover from inconsistant grid/procs */
    if (npflag && three==3) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"over constrained number of particle processes npe (%D) != %D",ctx->npe,k);
    if (npflag && three==2) {
      ctx->particleGrid.ft_np[2] = ctx->npe/(ctx->particleGrid.ft_np[0]*ctx->particleGrid.ft_np[1]);
      if ( (k=ctx->particleGrid.ft_np[0]*ctx->particleGrid.ft_np[1]*ctx->particleGrid.ft_np[2]) != ctx->npe) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"over constrained number of particle processes npe (%D) != %D",ctx->npe,k);
    }
    else if (npflag) {
      if (ctx->npe%ctx->particleGrid.ft_np[0]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"over constrained number of particle processes npe (%D) MOD %D",ctx->npe,ctx->particleGrid.ft_np[0]);
      k = ctx->npe/ctx->particleGrid.ft_np[0];
      k = (int)pow((double)k,0.5);
      ctx->particleGrid.ft_np[1] = ctx->particleGrid.ft_np[2] = k;
    }
    else {
      k = (int)pow((double)ctx->npe,0.33334);
      ctx->particleGrid.ft_np[0] = ctx->particleGrid.ft_np[1] = ctx->particleGrid.ft_np[2] = k;
      if ( (k=ctx->particleGrid.ft_np[0]*ctx->particleGrid.ft_np[1]*ctx->particleGrid.ft_np[2]) != ctx->npe) {
        k = (int)pow((double)ctx->npe,0.5);
        if ( ctx->npe%(k*k)==0 ) {
          ctx->particleGrid.ft_np[0] = ctx->particleGrid.ft_np[1] = k;
          ctx->particleGrid.ft_np[2] = ctx->npe/(k*k);
        }
      }
    }
    if ( (k=ctx->particleGrid.ft_np[0]*ctx->particleGrid.ft_np[1]*ctx->particleGrid.ft_np[2]) != ctx->npe) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"particle grids do not work npe (%D) != %D",ctx->npe,k);
  }
  {
    PetscInt i=X2_IDX_X(s_rank,ctx->particleGrid.ft_np),j=X2_IDX_Y(s_rank,ctx->particleGrid.ft_np),k=X2_IDX_Z(s_rank,ctx->particleGrid.ft_np);
    PetscInt rank = X2_IDX(i,j,k,ctx->particleGrid.ft_np);
    if (rank!=s_rank) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB," index maps not correct X2_IDX = %D, rank = %D",rank,s_rank);
  }
  /* solver grid */
  three = 3;
  ierr = PetscOptionsIntArray("-solver_np", "Number of (solver) processor in each dimension", "ex2.c", ctx->particleGrid.solver_np, &three, &npflag);CHKERRQ(ierr);
  if (!npflag) {
    ctx->particleGrid.solver_np[0] = ctx->particleGrid.ft_np[0];
    ctx->particleGrid.solver_np[1] = ctx->particleGrid.ft_np[1];
    ctx->particleGrid.solver_np[2] = ctx->particleGrid.ft_np[2];
  }
  else if ( (k=ctx->particleGrid.solver_np[0]*ctx->particleGrid.solver_np[1]*ctx->particleGrid.solver_np[2]) != ctx->npe) { /* recover from inconsistant grid/procs */
    if (npflag && three==3) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"over constrained number of particle processes npe (%D) != %D",ctx->npe,k);
    if (npflag && three==2) {
      ctx->particleGrid.solver_np[2] = ctx->npe/(ctx->particleGrid.solver_np[0]*ctx->particleGrid.solver_np[1]);
      if ( (k=ctx->particleGrid.solver_np[0]*ctx->particleGrid.solver_np[1]*ctx->particleGrid.solver_np[2]) != ctx->npe) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"over constrained number of particle processes npe (%D) != %D",ctx->npe,k);
    }
    else if (npflag) {
      if (ctx->npe%ctx->particleGrid.solver_np[0]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"over constrained number of particle processes npe (%D) MOD %D",ctx->npe,ctx->particleGrid.solver_np[0]);
      k = ctx->npe/ctx->particleGrid.solver_np[0];
      k = (int)pow((double)k,0.5);
      ctx->particleGrid.solver_np[1] = ctx->particleGrid.solver_np[2] = k;
    }
    else {
      k = (int)pow((double)ctx->npe,0.33334);
      ctx->particleGrid.solver_np[0] = ctx->particleGrid.solver_np[1] = ctx->particleGrid.solver_np[2] = k;
    }
    if ( (k=ctx->particleGrid.solver_np[0]*ctx->particleGrid.solver_np[1]*ctx->particleGrid.solver_np[2]) != ctx->npe) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"particle grids do not work npe (%D) != %D",ctx->npe,k);
  }
  { /* solver_proc_idx */
    int i,j,k;
    ctx->particleGrid.solver_proc_idx[0] = -1;
    for (i=0;i<ctx->particleGrid.solver_np[0] && ctx->particleGrid.solver_proc_idx[0] == -1;i++) {
    for (j=0;j<ctx->particleGrid.solver_np[1] && ctx->particleGrid.solver_proc_idx[0] == -1;j++) {
    for (k=0;k<ctx->particleGrid.solver_np[2] && ctx->particleGrid.solver_proc_idx[0] == -1;k++) {
      if (X2_IDX(i,j,k,ctx->particleGrid.solver_np) == ctx->rank) {
        ctx->particleGrid.solver_proc_idx[0] = i;
        ctx->particleGrid.solver_proc_idx[1] = j;
        ctx->particleGrid.solver_proc_idx[2] = k;
      }
    }}}
    assert(ctx->particleGrid.solver_proc_idx[0] != -1);
    assert(X2_IDX_X(s_rank,ctx->particleGrid.solver_np) == ctx->particleGrid.solver_proc_idx[0]);
    assert(X2_IDX_Y(s_rank,ctx->particleGrid.solver_np) == ctx->particleGrid.solver_proc_idx[1]);
    assert(X2_IDX_Z(s_rank,ctx->particleGrid.solver_np) == ctx->particleGrid.solver_proc_idx[2]);
  }
  three = 3;
  ierr = PetscOptionsRealArray("-b0", "B_0 vector", "ex2.c", ctx->particleGrid.b0, &three, NULL);CHKERRQ(ierr);
  {
    PetscReal len = ctx->particleGrid.b0[0]*ctx->particleGrid.b0[0] + ctx->particleGrid.b0[1]*ctx->particleGrid.b0[1] + ctx->particleGrid.b0[2]*ctx->particleGrid.b0[2];
    len = sqrt(len);
    if (len==0) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bad B_0 vector length %g %g %g",ctx->particleGrid.b0[0],ctx->particleGrid.b0[1],ctx->particleGrid.b0[2]);
    ctx->particleGrid.b0[0] /= len; ctx->particleGrid.b0[1] /= len; ctx->particleGrid.b0[2] /= len;
  }
  /* time integrator */
  ctx->msteps = 1;
  ierr = PetscOptionsInt("-mstep", "Maximum number of time steps", "ex2.c", ctx->msteps, &ctx->msteps, NULL);CHKERRQ(ierr);
  ctx->maxTime = 1000000000.;
  ierr = PetscOptionsReal("-maxTime", "Maximum time", "ex2.c",ctx->maxTime,&ctx->maxTime,NULL);CHKERRQ(ierr);
  ctx->dt = 1.;
  ierr = PetscOptionsReal("-dt","Time step","ex2.c",ctx->dt,&ctx->dt,NULL);CHKERRQ(ierr);
  /* particles */
  ctx->num_particles_proc = X2_V_LEN;
  ierr = PetscOptionsInt("-num_particles_proc", "Number of particles local (flux tube cell)", "ex2.c", ctx->num_particles_proc, &ctx->num_particles_proc, NULL);CHKERRQ(ierr);
  if (ctx->num_particles_proc<0) {
    if (ctx->rank==ctx->npe-1) ctx->num_particles_proc = -ctx->num_particles_proc;
    else ctx->num_particles_proc = 0;
  }
  if (!chunkFlag) ctx->chunksize = X2_V_LEN*((ctx->num_particles_proc/80+1)/X2_V_LEN+1); /* an intelegent message chunk size */

  if (s_debug>0) PetscPrintf(ctx->wComm,"npe=%D; %D x %D x %D flux tube grid; mpi_send size (chunksize) equal %d particles. %s. %s\n",
                             ctx->npe,ctx->particleGrid.solver_np[0],ctx->particleGrid.solver_np[1],ctx->particleGrid.solver_np[2],ctx->chunksize,
#ifdef X2_S_OF_V
			     "Use struct of arrays",
#else
			     "Use of array structs",
#endif
                             ctx->use_bsp ? "BSP communication" : "Non-blocking consensus communication");
  ctx->collision_period = 10000;
  ierr = PetscOptionsInt("-collision_period", "Period between collision operators", "ex2.c", ctx->collision_period, &ctx->collision_period, NULL);CHKERRQ(ierr);
  ctx->use_electrons = PETSC_FALSE; /* need neutral because periodic domain */
  ierr = PetscOptionsBool("-use_electrons", "Include electrons", "ex2.c", ctx->use_electrons, &ctx->use_electrons, NULL);CHKERRQ(ierr);
  ctx->max_vpar = 1.;
  ierr = PetscOptionsReal("-max_vpar", "Maximum parallel velocity", "ex2.c",ctx->max_vpar,&ctx->max_vpar,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* X2GridFluxTubeLocatePoint: find flux tube (pe:element=0) for this point
    Input:
     - grid: the particle grid
     - x:
   Output:
    - pe: process ID
    - elem: element ID
*/
#undef __FUNCT__
#define __FUNCT__ "X2GridFluxTubeLocatePoint"
PetscErrorCode X2GridFluxTubeLocatePoint( const X2GridParticle *grid, PetscReal x[3], PetscMPIInt *pe, PetscInt *elem)
{
  PetscInt       i,ii[3];
  X2Ctx *ctx = (X2Ctx*)grid->ctx;
  const PetscInt *np = ctx->particleGrid.solver_np;
  const PetscReal *dlo = ctx->particleGrid.dom_lo, *dhi = ctx->particleGrid.dom_hi;
  /* PetscErrorCode ierr; */
  PetscFunctionBeginUser;
  PetscValidPointer(x, 2);
  PetscValidPointer(pe, 3);
  PetscValidPointer(elem, 4);
  /* find processor */
  if (x[0]<dlo[0] || x[0]>dhi[0] || x[1]<dlo[1] ||
      x[1]>dhi[1] || x[2]<dlo[2] || x[2]>dhi[2])
    SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"point out of bounds %g %g %g",x[0],x[1],x[2]);
  for (i=0;i<3;i++) {
    ii[i] = (PetscInt)((x[i]-dlo[i])/(dhi[i]-dlo[i])*(double)np[i]);
  }
  *pe = X2_IDX3(ii,np);
  *elem = s_fluxtubeelem; /* 0 */
  PetscFunctionReturn(0);
}

#ifdef H5PART
/* add corners to get bounding box */
static void prewrite(X2Ctx *ctx, X2PList *l, X2PListPos *ppos1,  X2PListPos *ppos2)
{
  if (ctx->rank==0) {
    X2Particle part;
    PetscReal r,z,phi;
    r   = ctx->particleGrid.dom_lo[0];
    z   = ctx->particleGrid.dom_lo[1];
    phi = ctx->particleGrid.dom_lo[2];
    X2ParticleCreate(&part,1,r,z,phi,0.);
    X2PListAdd(l,&part,ppos2);
    r   = ctx->particleGrid.dom_hi[0];
    z   = ctx->particleGrid.dom_hi[1];
    phi = ctx->particleGrid.dom_hi[2];
    X2ParticleCreate(&part,1,r,z,phi,0.);
    X2PListAdd(l,&part,ppos1);
  }
}
static void postwrite(X2Ctx *ctx, X2PList *l, X2PListPos *ppos1,  X2PListPos *ppos2)
{
  if (ctx->rank==0) {
    X2PListRemoveAt(l,*ppos2);
    X2PListRemoveAt(l,*ppos1);
  }
}
#endif
/* processParticle: move particles if (sendListTable) , push if (irk>=0)
    Input:
     - dt: time step
     - tag: MPI tag to send with
     - irk: RK stage (<0 for send and deposit only)
     - solver: use solver partitioning to get processor of point?
   Input/Output:
     - ctx: global data
     - lists: list of particle lists
   Output:
     - sendListTable: send list hash table
*/
#undef __FUNCT__
#define __FUNCT__ "processParticles"
static PetscErrorCode processParticles( X2Ctx *ctx, const PetscReal dt, X2PSendList *sendListTable, const PetscMPIInt tag,
					const int irk, const int istep, PetscBool solver)
{
  X2GridParticle *grid = &ctx->particleGrid;
  DM_PICell *dmpi = (DM_PICell *) ctx->dm->data;     assert(solver || irk<0); /* don't push flux tubes */
  PetscMPIInt pe,hash,ii;
  X2Particle  part;
  X2PListPos  pos;
  PetscErrorCode ierr;
  const int part_dsize = sizeof(X2Particle)/sizeof(double);
  Vec          jetVec,xVec,vVec;
  PetscScalar *xx=0,*jj=0,*vv=0,*xx0=0,*jj0=0,*vv0=0;
  PetscInt isp,order=1,nslist,nlistsTot,elid,elid2,one=1,three=3,ndeposit;
  int origNlocal,nmoved;
  X2ISend slist[X2PROCLISTSIZE];
  IS pes,elems;
  const PetscInt *cpeidxs,*celemidxs;
  PetscInt *peidxs,*elemidxs;
  PetscFunctionBeginUser;
  MPI_Barrier(ctx->wComm);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(ctx->events[1],0,0,0,0);CHKERRQ(ierr);
#endif
  if (!dmpi) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"DM_PICell data not created");
  if (solver) {
    ierr = VecZeroEntries(dmpi->rho);CHKERRQ(ierr); /* zero density to get ready for this deposition */
  }
  /* push particles, if necessary, and make send lists */
  for (isp=ctx->use_electrons ? 0 : 1, ndeposit = 0, nslist = 0, nmoved = 0, nlistsTot = 0, origNlocal = 0;
       isp <= X2_NION ;
       isp++) {
    /* loop over element particle lists */
    for (elid=0;elid<ctx->nElems;elid++) {
      X2PList *list = &ctx->partlists[isp][elid];
      origNlocal += X2PListSize(list);
      /* get Cartesian coordinates (not used for flux tube move) */
      ierr = X2PListCompress(list);CHKERRQ(ierr); /* allows for simpler vectorization */
      if (solver) {
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventBegin(ctx->events[7],0,0,0,0);CHKERRQ(ierr); /* timer on particle list */
#endif
        /* make vectors for this element */
        ierr = VecCreateSeq(PETSC_COMM_SELF,three*list->vec_top, &xVec);CHKERRQ(ierr);
        ierr = VecCreateSeq(PETSC_COMM_SELF,three*list->vec_top, &jetVec);CHKERRQ(ierr);
        ierr = VecSetBlockSize(xVec,three);CHKERRQ(ierr);
        ierr = VecSetBlockSize(jetVec,three);CHKERRQ(ierr);
        /* make coordinates array to get gradients */
        ierr = VecGetArray(xVec,&xx0);CHKERRQ(ierr); xx = xx0;
/* #pragma simd vectorlengthfor(PetscScalar) */
	for (pos=0 ; pos < list->vec_top ; pos++, xx += 3) {
#ifdef X2_S_OF_V
          xx[0] = list->data_v.r[pos], xx[1] = list->data_v.z[pos], xx[2] = list->data_v.phi[pos];
#else
          xx[0] = list->data[pos].r,   xx[1] = list->data[pos].z,   xx[2] = list->data[pos].phi;
#endif
        }
        ierr = VecRestoreArray(xVec,&xx0);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventEnd(ctx->events[7],0,0,0,0);CHKERRQ(ierr);
#endif
      }
      if (solver) {
        /* push, and collect x */
        PetscReal l[] = { ctx->particleGrid.dom_hi[0]-ctx->particleGrid.dom_lo[0],
                          ctx->particleGrid.dom_hi[1]-ctx->particleGrid.dom_lo[1],
                          ctx->particleGrid.dom_hi[2]-ctx->particleGrid.dom_lo[2]};
        const PetscReal *dlo = ctx->particleGrid.dom_lo, *b0 = ctx->particleGrid.b0;
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventBegin(ctx->events[8],0,0,0,0);CHKERRQ(ierr); /* timer on particle list */
#endif
        if (irk>=0) {
          /* get E, should set size of vecs for true size? */
          ierr = DMPICellGetJet(dmpi->dmplex, xVec, order, jetVec, elid);CHKERRQ(ierr);
          ierr = VecGetArray(jetVec,&jj0);CHKERRQ(ierr); jj = jj0;
        }
        /* vectorize (todo) push: theta = theta + q*dphi .... grad not used */
        ierr = VecGetArray(xVec,&xx0);CHKERRQ(ierr); xx = xx0;
        for (pos=0 ; pos < list->vec_top ; pos++, xx += 3, jj += 3 ) {
	  /* push particle, real data, could do it on copy for non-final stage of TS */
          if (irk>=0) {
#ifdef X2_S_OF_V
            PetscReal r = dt*list->data_v.vpar[pos];
            for(ii=0;ii<3;ii++) { /* we could use jet here */
              list->data_v.x[ii][pos] += r*b0[ii];
              xx[ii] = list->data_v.x[ii][pos] = dlo[ii] + fmod(list->data_v.x[ii][pos] - dlo[ii] + 20.*l[ii], l[ii]);
            }
#else
            PetscReal r = dt*list->data[pos].vpar;
            for(ii=0;ii<3;ii++) {
              list->data[pos].x[ii] += r*b0[ii];
              xx[ii] = list->data[pos].x[ii] = dlo[ii] +  fmod(list->data[pos].x[ii]   - dlo[ii] + 10.*l[ii], l[ii]);
            }
#endif
          } else {
#ifdef X2_S_OF_V
            xx[2] = list->data_v.phi[pos];
            xx[0] = list->data_v.r[pos];
            xx[1] = list->data_v.z[pos];
#else
            xx[2] = list->data[pos].x[2];
            xx[0] = list->data[pos].x[0];
            xx[1] = list->data[pos].x[1];
#endif
          }
        }
        ierr = VecRestoreArray(xVec,&xx0);
        if (irk>=0) {
          ierr = VecRestoreArray(jetVec,&jj0);
        }
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventEnd(ctx->events[8],0,0,0,0);CHKERRQ(ierr);
#endif
      }
      /* collect pes to communicate - not vectorizable */
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventBegin(ctx->events[5],0,0,0,0);CHKERRQ(ierr);
#endif
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventBegin(s_events[9],0,0,0,0);CHKERRQ(ierr);
#endif
      /* get pe & element id - vectorize!!! */
      if (solver) {
        /* see if need communication? no: add density, yes: add to communication list */
        ierr = X2GridSolverLocatePoints(dmpi->dmplex, xVec, ctx, &pes, &elems);CHKERRQ(ierr);
      } else {
        ierr = PetscMalloc2(list->vec_top,&peidxs,list->vec_top,&elemidxs);CHKERRQ(ierr);
	for (pos=0 ; pos < list->vec_top ; pos++ ) {
#ifdef X2_S_OF_V
	  X2V2P((&part),list->data_v,pos); /* return copy */
#else
	  part = list->data[pos]; /* return copy */
#endif
	  ierr = X2GridFluxTubeLocatePoint(grid, part.x, &pe, &elid2);CHKERRQ(ierr);
	  peidxs[pos] = pe;
	  elemidxs[pos] = elid2;
	}
        ierr = ISCreateGeneral(PETSC_COMM_SELF,list->vec_top,peidxs,PETSC_COPY_VALUES,&pes);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PETSC_COMM_SELF,list->vec_top,elemidxs,PETSC_COPY_VALUES,&elems);CHKERRQ(ierr);
        ierr = PetscFree2(peidxs,elemidxs);CHKERRQ(ierr);
      }
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventEnd(s_events[9],0,0,0,0);CHKERRQ(ierr);
#endif
      /* move particles, not vectorizable */
      ierr = ISGetIndices(pes,&cpeidxs);CHKERRQ(ierr);
      ierr = ISGetIndices(elems,&celemidxs);CHKERRQ(ierr);
      for (pos=0 ; pos < list->top ; pos++ ) {
	pe = cpeidxs[pos];
	elid2 = celemidxs[pos]; /* we can not just move to new list because it might get pushed again, so use MPI (local) buffer pretty big as a cache */
	if (pe==ctx->rank && elid2==elid) continue; /* don't move */
#ifdef X2_S_OF_V
	X2V2P((&part),list->data_v,pos); /* return copy */
#else
	part = list->data[pos]; /* return copy */
#endif
	/* add to list to send, find list with table lookup, send full lists - no vectorization */
	hash = (pe*593)%ctx->proc_send_table_size; /* hash */
	for (ii=0;ii<ctx->proc_send_table_size;ii++){
	  if (sendListTable[hash].data_size==0) {
            int np = ctx->num_particles_proc, cs = ctx->chunksize; /* lots of local movement */
            if (pe==ctx->rank) {
              ierr = X2PSendListCreate(&sendListTable[hash], cs*(np/cs + np/cs/64 + 1) );CHKERRQ(ierr);
            } else {
              ierr = X2PSendListCreate(&sendListTable[hash], cs*(np/cs/ctx->nElems + np/cs/ctx->nElems/8 + 1));CHKERRQ(ierr);
            }
	    sendListTable[hash].proc = pe;
	    ctx->tablecount++;
	    if (ctx->tablecount==ctx->proc_send_table_size) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Table too small (%D)",ctx->proc_send_table_size);
	  }
	  if (sendListTable[hash].proc==pe) { /* found hash table entry */
	    if (X2PSendListSize(&sendListTable[hash])==X2PSendListMaxSize(&sendListTable[hash]) && !ctx->use_bsp) { /* list is full, send and recreate */
	      MPI_Datatype mtype;
#if defined(PETSC_USE_LOG)
	      ierr = PetscLogEventBegin(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
	      PetscDataTypeToMPIDataType(PETSC_REAL,&mtype);
	      /* send and reset - we can just send this because it is dense, but no species data */
	      if (nslist==X2PROCLISTSIZE) {
		SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"process send table too small (%D) == snlist(%D)",nslist,(PetscInt)X2PROCLISTSIZE);
	      }
	      slist[nslist].data = sendListTable[hash].data; /* cache data */
	      slist[nslist].proc = pe;
	      ierr = MPI_Isend((void*)slist[nslist].data,ctx->chunksize*part_dsize,mtype,pe,tag+isp,ctx->wComm,&slist[nslist].request);
	      CHKERRQ(ierr);
	      nslist++;
	      /* ready for next round, save meta-data  */
	      ierr = X2PSendListClear(&sendListTable[hash]);CHKERRQ(ierr);
	      assert(sendListTable[hash].data_size == ctx->chunksize);
	      sendListTable[hash].data = 0;
	      ierr = PetscMalloc1(ctx->chunksize, &sendListTable[hash].data);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
	      ierr = PetscLogEventEnd(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
	    }
	    /* add to list - pass this in as a function to a function? */
	    ierr = X2PSendListAdd(&sendListTable[hash],&part);CHKERRQ(ierr); /* not vectorizable */
	    ierr = X2PListRemoveAt(list,pos);CHKERRQ(ierr); /* not vectorizable */
	    if (pe!=ctx->rank) nmoved++;
	    break;
	  }
	  if (++hash == ctx->proc_send_table_size) hash=0;
	}
	assert(ii!=ctx->proc_send_table_size);
      }
      ierr = ISRestoreIndices(pes,&cpeidxs);CHKERRQ(ierr);
      ierr = ISRestoreIndices(elems,&celemidxs);CHKERRQ(ierr);
      ierr = ISDestroy(&pes);CHKERRQ(ierr);
      ierr = ISDestroy(&elems);CHKERRQ(ierr);
      if (solver) {
	/* done with these, need new ones after communication */
	ierr = VecDestroy(&xVec);CHKERRQ(ierr);
	ierr = VecDestroy(&jetVec);CHKERRQ(ierr);
      }
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventEnd(ctx->events[5],0,0,0,0);CHKERRQ(ierr);
#endif
    } /* element list */

    /* finish sends and receive new particles for this species */
    ierr = shiftParticles(ctx, sendListTable, &nslist, ctx->partlists[isp], slist, tag+isp, solver);CHKERRQ(ierr);

#ifdef PETSC_USE_DEBUG
    { /* debug */
      PetscMPIInt flag,sz; MPI_Status  status; MPI_Datatype mtype;
      ierr = MPI_Iprobe(MPI_ANY_SOURCE, tag+isp, ctx->wComm, &flag, &status);CHKERRQ(ierr);
      if (flag) {
        PetscDataTypeToMPIDataType(PETSC_REAL,&mtype);
        MPI_Get_count(&status, mtype, &sz); assert(sz%part_dsize==0);
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"found %D extra particles from %d",sz/part_dsize,status.MPI_SOURCE);
      }
      MPI_Barrier(ctx->wComm);
    }
#endif
    nlistsTot += nslist;
    /* add density (while in cache, by species at least) */
    if (solver) {
      Vec locrho;        assert(solver);
      ierr = DMGetLocalVector(dmpi->dmplex, &locrho);CHKERRQ(ierr);
      ierr = VecSet(locrho, 0.0);CHKERRQ(ierr);
      for (elid=0;elid<ctx->nElems;elid++) {
        X2PList *list = &ctx->partlists[isp][elid];
        if (X2PListSize(list)==0) continue;
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventBegin(ctx->events[7],0,0,0,0);CHKERRQ(ierr); /* timer on particle list */
#endif
        ierr = X2PListCompress(list);CHKERRQ(ierr); /* allows for simpler vectorization */
        /* make vectors for this element */
        ierr = VecCreateSeq(PETSC_COMM_SELF,three*list->vec_top, &xVec);CHKERRQ(ierr);
        ierr = VecCreateSeq(PETSC_COMM_SELF,one*list->vec_top, &vVec);CHKERRQ(ierr);
        ierr = VecSetBlockSize(xVec,three);CHKERRQ(ierr);
        ierr = VecSetBlockSize(vVec,one);CHKERRQ(ierr);
        /* make coordinates array and density */
        ierr = VecGetArray(xVec,&xx0);CHKERRQ(ierr); xx = xx0;
        ierr = VecGetArray(vVec,&vv0);CHKERRQ(ierr); vv = vv0;
        /* ierr = X2PListGetHead( list, &part, &pos );CHKERRQ(ierr); */
        /* do { */
        for (pos=0 ; pos < list->vec_top ; pos++, xx += 3, vv++) { /* this has holes, but few and zero weight - vectorizable */
#ifdef X2_S_OF_V
          xx[0]=list->data_v.r[pos];
          xx[1]=list->data_v.z[pos];
          xx[2]=list->data_v.phi[pos];
          *vv = list->data_v.w0[pos]*ctx->species[isp].charge;
#else
          xx[0]=list->data[pos].r;
          xx[1]=list->data[pos].z;
          xx[2]=list->data[pos].phi;
          *vv = list->data[pos].w0*ctx->species[isp].charge;
#endif
          ndeposit++;
        }
        /* } while ( !X2PListGetNext(list, &part, &pos) ); */
        ierr = VecRestoreArray(xVec,&xx0);CHKERRQ(ierr);
        ierr = VecRestoreArray(vVec,&vv0);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventEnd(ctx->events[7],0,0,0,0);CHKERRQ(ierr);
#endif
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventBegin(ctx->events[6],0,0,0,0);CHKERRQ(ierr); /* timer on particle list */
#endif
        ierr = DMPICellAddSource(ctx->dm, xVec, vVec, elid, locrho);CHKERRQ(ierr);
        ierr = VecDestroy(&xVec);CHKERRQ(ierr);
        ierr = VecDestroy(&vVec);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventEnd(ctx->events[6],0,0,0,0);CHKERRQ(ierr);
#endif
      }
      ierr = DMLocalToGlobalBegin(dmpi->dmplex, locrho, ADD_VALUES, dmpi->rho);CHKERRQ(ierr);
      ierr = DMLocalToGlobalEnd(dmpi->dmplex, locrho, ADD_VALUES, dmpi->rho);CHKERRQ(ierr);
      ierr = DMRestoreLocalVector(dmpi->dmplex, &locrho);CHKERRQ(ierr);
    }
  } /* isp */
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(ctx->events[1],0,0,0,0);CHKERRQ(ierr);
#endif
  /* diagnostics */
  if (dmpi->debug>0) {
    MPI_Datatype mtype;
    PetscInt rb1[4], rb2[4], sb[4], nloc;
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(ctx->events[diag_event_id],0,0,0,0);CHKERRQ(ierr);
#endif
    /* count particles */
    for (isp=ctx->use_electrons ? 0 : 1, nloc = 0 ; isp <= X2_NION ; isp++) {
      for (elid=0;elid<ctx->nElems;elid++) {
        nloc += X2PListSize(&ctx->partlists[isp][elid]);
      }
    }
    sb[0] = origNlocal;
    sb[1] = nmoved;
    sb[2] = nlistsTot;
    sb[3] = nloc;
    PetscDataTypeToMPIDataType(PETSC_INT,&mtype);
    ierr = MPI_Allreduce(sb, rb1, 4, mtype, MPI_SUM, ctx->wComm);CHKERRQ(ierr);
    ierr = MPI_Allreduce(sb, rb2, 4, mtype, MPI_MAX, ctx->wComm);CHKERRQ(ierr);
    PetscPrintf(ctx->wComm,
                "%d) %s %D local particles, %D/%D global, %g %% total particles moved in %D messages total (to %D/%D processors local), %g load imbalance factor\n",
                istep+1,irk<0 ? "processed" : "pushed", origNlocal, rb1[0], rb1[3], 100.*(double)rb1[1]/(double)rb1[0], rb1[2], ctx->tablecount,ctx->proc_send_table_size,(double)rb2[3]/((double)rb1[3]/(double)ctx->npe));
    if (rb1[0] != rb1[3]) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Number of partilces %D --> %D",rb1[0],rb1[3]);
#ifdef H5PART
    if (irk>=0 && ctx->plot) {
      for (isp=ctx->use_electrons ? 0 : 1 ; isp <= X2_NION ; isp++ ) {
        char  fname1[256],fname2[256];
        X2PListPos pos1,pos2;
        /* hdf5 output */
        sprintf(fname1,"ex2_particles_sp%d_time%05d.h5part",(int)isp,(int)istep+1);
        sprintf(fname2,"ex2_sub_rank_particles_sp%d_time%05d.h5part",(int)isp,(int)istep+1);
        /* write */
        prewrite(ctx, &ctx->partlists[isp][s_fluxtubeelem], &pos1, &pos2);
        ierr = X2PListWrite(ctx->partlists[isp], ctx->nElems, ctx->rank, ctx->npe, ctx->wComm, fname1, fname2);CHKERRQ(ierr);
        postwrite(ctx, &ctx->partlists[isp][s_fluxtubeelem], &pos1, &pos2);
      }
    }
#endif
#if defined(PETSC_USE_LOG)
    MPI_Barrier(ctx->wComm);
    ierr = PetscLogEventEnd(ctx->events[diag_event_id],0,0,0,0);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}
#define X2NDIG 100000
  /* create particles in flux tube, create particle lists, move particles to flux tube element list */
#undef __FUNCT__
#define __FUNCT__ "createParticles"
static PetscErrorCode createParticles(X2Ctx *ctx)
{
  PetscErrorCode ierr;
  PetscInt isp,gid,i,dim,cStart,cEnd,elid;
  const PetscInt  *np = ctx->particleGrid.ft_np;
  const PetscReal *dlo = ctx->particleGrid.dom_lo, *dhi = ctx->particleGrid.dom_hi;
  const PetscReal dx=(dhi[0]-dlo[0])/(PetscReal)np[0];
  const PetscReal x1=dlo[0] + dx*X2_IDX_X(ctx->rank,np);
  const PetscReal dy=(dhi[1]-dlo[1])/(PetscReal)np[1];
  const PetscReal y1=dlo[1] + dy*X2_IDX_Y(ctx->rank,np);
  const PetscReal dz=(dhi[2]-dlo[2])/(PetscReal)np[2];
  const PetscReal z1=dlo[2] + dz*X2_IDX_Z(ctx->rank,np);
  X2Particle particle;
  DM_PICell *dmpi;
  PetscFunctionBeginUser;

  /* Create vector and get pointer to data space */
  dmpi = (DM_PICell *) ctx->dm->data;
  ierr = DMGetDimension(dmpi->dmplex, &dim);CHKERRQ(ierr);
  if (dim!=3) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"wrong dimension (3) = %D",dim);
  ierr = DMGetCellChart(dmpi->dmplex, &cStart, &cEnd);CHKERRQ(ierr);
  ctx->nElems = PetscMax(1,cEnd-cStart);CHKERRQ(ierr);
  /* setup particles - lexicographic partition of -- flux tube -- cells */
  gid = ctx->rank*ctx->num_particles_proc;

  /* my first cell index */
  srand(ctx->rank);
  for (isp=ctx->use_electrons ? 0 : 1 ; isp <= X2_NION ; isp++ ) {
    const PetscReal maxe=ctx->max_vpar*ctx->max_vpar,mass=ctx->species[isp].mass,charge=ctx->species[isp].charge;
    ierr = PetscMalloc1(ctx->nElems,&ctx->partlists[isp]);CHKERRQ(ierr);
    /* create list for element 0 and add all to it */
    ierr = X2PListCreate(&ctx->partlists[isp][s_fluxtubeelem],
                         ctx->chunksize*(ctx->num_particles_proc/ctx->chunksize + ctx->num_particles_proc/ctx->chunksize/64 + 1));
    CHKERRQ(ierr);
    /* create each particle */
    for (i=0 ; i<ctx->num_particles_proc; i++ ) {
      PetscReal xx[] = { x1 + (PetscReal)(rand()%X2NDIG+1)/(PetscReal)(X2NDIG+1)*dx,
                         y1 + (PetscReal)(rand()%X2NDIG+1)/(PetscReal)(X2NDIG+1)*dy,
                         z1 + (PetscReal)(rand()%X2NDIG+1)/(PetscReal)(X2NDIG+1)*dz };
      PetscReal zmax,v,zdum,vpar;
      /* v_parallel from random number */
      zmax = 1.0 - exp(-maxe);
      zdum = zmax*(PetscReal)(rand()%X2NDIG)/(PetscReal)X2NDIG;
      v= sqrt(-2.0/mass*log(1.0-zdum));
      v= v*cos(M_PI*(PetscReal)(rand()%X2NDIG)/(PetscReal)X2NDIG);
      /* vshift= v + up ! shift of velocity */
      vpar = v*mass/charge;
      ierr = X2ParticleCreate(&particle,++gid,xx[0],xx[1],xx[2],vpar);CHKERRQ(ierr); /* only time this is called! */
      ierr = X2PListAdd(&ctx->partlists[isp][s_fluxtubeelem],&particle, NULL);CHKERRQ(ierr);
      /* debug, particles are created in a flux tube */
#ifdef PETSC_USE_DEBUG
      {
        PetscMPIInt pe; PetscInt id;
        ierr = X2GridFluxTubeLocatePoint(&ctx->particleGrid,xx,&pe,&id);CHKERRQ(ierr);
        if(pe != ctx->rank){
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB," created particle for proc %D",pe);
        }
      }
#endif
    }
    /* finish off list creates for rest of elements */
    for (elid=0;elid<ctx->nElems;elid++) {
      if (elid!=s_fluxtubeelem) {
        int np = ctx->num_particles_proc, cs = ctx->chunksize, b = cs*(np/cs/ctx->nElems + np/cs/ctx->nElems/8 + 1);
        ierr = X2PListCreate(&ctx->partlists[isp][elid],b);CHKERRQ(ierr); /* this will get expanded, chunksize used for message chunk size and initial list size! */
      }
    }
  } /* species */
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
static PetscErrorCode CreateMesh(X2Ctx *ctx)
{
  PetscErrorCode ierr;
  PetscInt dimEmbed, i;
  PetscInt nCoords,dim;
  PetscScalar *coords;
  Vec coordinates;
  DM_PICell *dmpi = (DM_PICell *) ctx->dm->data;
  PetscInt  *sizes = NULL;
  PetscInt  *points = NULL;
  PetscPartitioner part;
  const char *prefix;
  DM dm;
  DMLabel label;
  PetscFunctionBeginUser;

  /* setup solver grid */
  dim = 3;
  ierr = DMPlexCreateHexBoxMesh(ctx->wComm, dim, ctx->particleGrid.solver_np, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, &dmpi->dmplex);CHKERRQ(ierr);
  /* set domain size */
  ierr = DMGetCoordinatesLocal(dmpi->dmplex,&coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dmpi->dmplex,&dimEmbed);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordinates,&nCoords);CHKERRQ(ierr);
  if (nCoords % dimEmbed) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Coordinate vector the wrong size");CHKERRQ(ierr);
  ierr = VecGetArray(coordinates,&coords);CHKERRQ(ierr);
  for (i = 0; i < nCoords; i += dimEmbed) {
    PetscInt j;
    PetscScalar *coord = &coords[i];
    for (j = 0; j < dimEmbed; j++) {
      coord[j] = ctx->particleGrid.dom_lo[j] + coord[j] * (ctx->particleGrid.dom_hi[j] - ctx->particleGrid.dom_lo[j]);
    }
  }
  ierr = VecRestoreArray(coordinates,&coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dmpi->dmplex,coordinates);CHKERRQ(ierr);

  ierr = DMSetApplicationContext(dmpi->dmplex, &ctx);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) dmpi->dmplex, "x2_");CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) ctx->dm, "x2_");CHKERRQ(ierr);
  ierr = PetscMalloc(1 * sizeof(PetscErrorCode (*)(PetscInt,const PetscReal [],PetscInt,PetscScalar*,void*)),&ctx->BCFuncs);CHKERRQ(ierr);
  ctx->BCFuncs[0] = zero;
  /* add BCs */
  ierr = DMCreateLabel(dmpi->dmplex, "boundary");CHKERRQ(ierr);
  ierr = DMGetLabel(dmpi->dmplex, "boundary", &label);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dmpi->dmplex, label);CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(dmpi->dmplex, label);CHKERRQ(ierr);
  if (1) {
    PetscInt id = 1;
    ierr = DMAddBoundary(dmpi->dmplex, PETSC_TRUE, "wall", "boundary", 0, 0, NULL, (void (*)()) ctx->BCFuncs[0], 1, &id, &ctx);CHKERRQ(ierr);
  }
  if (sizeof(long long)!=sizeof(PetscReal)) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "sizeof(long long)!=sizeof(PetscReal)");
  /* set a simple partitioner */
  if (!ctx->rank) {
    PetscInt cEnd,c,*cs,*cp,ii[3],i;
    const PetscReal *dlo = ctx->particleGrid.dom_lo, *dhi = ctx->particleGrid.dom_hi;
    const PetscInt *np = ctx->particleGrid.solver_np;
    ierr = DMPlexGetHeightStratum(dmpi->dmplex, 0, NULL, &cEnd);CHKERRQ(ierr); /* DMGetCellChart */
    if (cEnd && cEnd!=ctx->npe) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_USER, "cEnd=%d != %d",cEnd,ctx->npe);
    if (cEnd!=ctx->npe) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_USER, "cEnd=%d != %d",cEnd,ctx->npe);
    ierr = PetscMalloc2(ctx->npe, &sizes, ctx->npe, &points);CHKERRQ(ierr);
    for (c=0,cs=sizes,cp=points;c<ctx->npe;c++,cs++,cp++) {
      PetscReal x[3];
      *cs = 1; // points[c] = c;
      ierr = DMPlexComputeCellGeometryFVM(dmpi->dmplex, c, NULL, x, NULL);CHKERRQ(ierr);
      for(i=0;i<3;i++) {
        ii[i] = (PetscInt)((x[i]-dlo[i])/(dhi[i]-dlo[i])*(double)np[i]);
      }
      *cp = X2_IDX3(ii,np);
    }
  }
  ierr = DMPlexGetPartitioner(dmpi->dmplex, &part);CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(part, PETSCPARTITIONERSHELL);CHKERRQ(ierr);
  ierr = PetscPartitionerShellSetPartition(part, ctx->npe, sizes, points);CHKERRQ(ierr);
  if (sizes) {
    ierr = PetscFree2(sizes,points);CHKERRQ(ierr);
  }
  ierr = PetscObjectGetOptionsPrefix((PetscObject)dmpi->dmplex,&prefix);CHKERRQ(ierr);
  ierr = DMPlexDistribute(dmpi->dmplex, 0, NULL, &dm);CHKERRQ(ierr);
  if (dm) {
    ierr = PetscObjectSetOptionsPrefix((PetscObject)dm,prefix);CHKERRQ(ierr);
    ierr = DMDestroy(&dmpi->dmplex);CHKERRQ(ierr);
    dmpi->dmplex = dm;
  }
  else assert(ctx->npe==1);
  /* set from options: refinement done here */
  ierr = DMSetFromOptions( ctx->dm );CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "SetupDiscretization"
static PetscErrorCode SetupDiscretization(X2Ctx *ctx)
{
  PetscErrorCode ierr;
  DM_PICell *dmpi = (DM_PICell *) ctx->dm->data;
  PetscDS    prob;
  PetscInt   dim;
  PetscSection   s;
  PetscFunctionBeginUser;

  ierr = DMGetDimension(dmpi->dmplex, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dmpi->dmplex, dim, 1, PETSC_FALSE, NULL, 1, &dmpi->fem);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dmpi->fem, "poisson");CHKERRQ(ierr);
  /* FEM prob */
  ierr = DMGetDS(dmpi->dmplex, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) dmpi->fem);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 0, f0_u, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
  if (0) {
    PetscInt id = 1;
    ierr = PetscDSAddBoundary(prob, PETSC_TRUE, "wall", "boundary", 0, 0, NULL, (void (*)())  ctx->BCFuncs[0], 1, &id, &ctx);CHKERRQ(ierr);
  }
  ierr = DMSetUp( ctx->dm );CHKERRQ(ierr); /* create vectors */

  /* diagnostics */
  ierr = DMGetDefaultSection(dmpi->dmplex, &s);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dmpi->dmplex, &s);CHKERRQ(ierr);
  if (!s) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "DMGetDefaultSection return NULL");

  ierr = PetscSectionViewFromOptions(s, NULL, "-section_view");CHKERRQ(ierr);
  if (dmpi->debug>3) { /* this shows a bug with crap in the section */
    ierr = PetscSectionView(s,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  if (dmpi->debug>2) {
    ierr = DMView(dmpi->dmplex,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  {
    PetscInt n,cStart,cEnd;
    ierr = VecGetSize(dmpi->rho,&n);CHKERRQ(ierr);
    if (!n) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "No dofs");
    ierr = DMPlexGetHeightStratum(dmpi->dmplex, 0, &cStart, &cEnd);CHKERRQ(ierr); /* DMGetCellChart */
    if (cStart) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER, "cStart != 0. %D",cStart);
    if (dmpi->debug>0 && !cEnd) {
      ierr = PetscPrintf((dmpi->debug>1 || !cEnd) ? PETSC_COMM_SELF : ctx->wComm,"[%D] ERROR %D global equations, %d local cells, (cEnd=%d), debug=%D\n",ctx->rank,n,cEnd-cStart,cEnd,dmpi->debug);
    }
    if (!cEnd) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "No cells");
    }
    s_fluxtubeelem = cEnd/2; /* could just be 0 */
    if (dmpi->debug>0) PetscPrintf(ctx->wComm,"%D equations on %D processors, %D local cells, (element %D used for flux tube list)\n",
                                   n,ctx->npe,cEnd,s_fluxtubeelem);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  X2Ctx          ctx; /* user-defined work context */
  PetscErrorCode ierr;
  DM_PICell      *dmpi;
  PetscInt idx,isp;
  Mat            J;
  PetscFunctionBeginUser;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ctx.events = s_events;
  ctx.particleGrid.ctx = &ctx;
#if defined(PETSC_USE_LOG)
  {
    PetscInt currevent = 0;
    PetscLogStage  setup_stage;
    ierr = PetscLogEventRegister("X2Setup", DM_CLASSID, &ctx.events[currevent++]);CHKERRQ(ierr); /* 0 */
    ierr = PetscLogEventRegister("X2Process parts",0,&ctx.events[currevent++]);CHKERRQ(ierr); /* 1 */
    ierr = PetscLogEventRegister(" -shiftParticles",0,&ctx.events[currevent++]);CHKERRQ(ierr); /* 2 */
    ierr = PetscLogEventRegister("  =Non-block con",0,&ctx.events[currevent++]);CHKERRQ(ierr); /* 3 */
    ierr = PetscLogEventRegister("  =Part. Send", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 4 */
    ierr = PetscLogEventRegister(" -Move parts", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 5 */
    ierr = PetscLogEventRegister(" -AddSource", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 6 */
    ierr = PetscLogEventRegister(" -Pre Push", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 7 */
    ierr = PetscLogEventRegister(" -Push (Jet)", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 8 */
    ierr = PetscLogEventRegister("  =Point Locate", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 9 */
    ierr = PetscLogEventRegister(" -create Particles", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 10 */
    ierr = PetscLogEventRegister("X2Poisson Solve", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 11 */
    ierr = PetscLogEventRegister("X2Part AXPY", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 12 */
    ierr = PetscLogEventRegister("X2Compress array", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 13 */
    ierr = PetscLogEventRegister("X2Diagnostics", 0, &ctx.events[diag_event_id]);CHKERRQ(ierr); /* N-1 */
    assert(sizeof(s_events)/sizeof(s_events[0]) > currevent);
/*     ierr = PetscLogStageRegister("Setup", &setup_stage);CHKERRQ(ierr); */
/*     ierr = PetscLogStagePush(setup_stage);CHKERRQ(ierr); */
  }
#endif

  ierr = PetscCommDuplicate(PETSC_COMM_WORLD,&ctx.wComm,NULL);CHKERRQ(ierr);
  ierr = ProcessOptions( &ctx );CHKERRQ(ierr);

  /* construct DMs */
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(ctx.events[0],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = DMCreate(ctx.wComm, &ctx.dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(ctx.dm, &ctx);CHKERRQ(ierr);
  ierr = DMSetType(ctx.dm, DMPICELL);CHKERRQ(ierr); /* creates (DM_PICell *) dm->data */
  dmpi = (DM_PICell *) ctx.dm->data; assert(dmpi);
  dmpi->debug = s_debug;

  ierr = CreateMesh(&ctx);CHKERRQ(ierr);

  /* setup Discretization */
  ierr = SetupDiscretization(&ctx);CHKERRQ(ierr);

  // dmpi->dmplex = dmpi->dmplex; /* done with setup but some methods in picell.c (still) need a plex DM */

  /* create SNESS */
  ierr = SNESCreate( ctx.wComm, &dmpi->snes);CHKERRQ(ierr);
  ierr = SNESSetDM( dmpi->snes, dmpi->dmplex);CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dmpi->dmplex,&ctx,&ctx,&ctx);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dmpi->dmplex, &J);CHKERRQ(ierr);
  ierr = SNESSetJacobian(dmpi->snes, J, J, NULL, NULL);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(dmpi->snes);CHKERRQ(ierr);
  if (dmpi->debug>3) {
    PetscViewer viewer;
    PetscViewerASCIIOpen(ctx.wComm, "Amat.m", &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
    MatView(J,viewer);
    PetscViewerPopFormat(viewer);
    PetscViewerDestroy(&viewer);
  }
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(ctx.events[10],0,0,0,0);CHKERRQ(ierr);
#endif
  /* setup particles */
  ierr = createParticles( &ctx );CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(ctx.events[10],0,0,0,0);CHKERRQ(ierr);
#endif

  /* init send tables */
  ierr = PetscMalloc1(ctx.proc_send_table_size,&ctx.sendListTable);CHKERRQ(ierr);
  for (idx=0;idx<ctx.proc_send_table_size;idx++) {
    for (isp=ctx.use_electrons ? 0 : 1 ; isp <= X2_NION ; isp++) {
      ctx.sendListTable[idx].data_size = 0; /* init */
    }
  }
  /* hdf5 output - init */
#ifdef H5PART
  if (ctx.plot) {
    for (isp=ctx.use_electrons ? 0 : 1 ; isp <= X2_NION ; isp++) { // for each species
      char  fname1[256],fname2[256];
      X2PListPos pos1,pos2;
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventBegin(ctx.events[diag_event_id],0,0,0,0);CHKERRQ(ierr);
#endif
      sprintf(fname1,"ex2_particles_sp%d_time%05d_fluxtube.h5part",(int)isp,0);
      sprintf(fname2,"ex2_sub_rank_particles_sp%d_time%05d_fluxtube.h5part",(int)isp,0);
      /* write */
      prewrite(&ctx, &ctx.partlists[isp][s_fluxtubeelem], &pos1, &pos2);
      ierr = X2PListWrite(ctx.partlists[isp], ctx.nElems, ctx.rank, ctx.npe, ctx.wComm, fname1, fname2);CHKERRQ(ierr);
      postwrite(&ctx, &ctx.partlists[isp][s_fluxtubeelem], &pos1, &pos2);
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventEnd(ctx.events[diag_event_id],0,0,0,0);CHKERRQ(ierr);
#endif
    }
  }
#endif

  /* move back to solver space and make density vector */
  ierr = processParticles(&ctx, 0.0, ctx.sendListTable, 99, -1, -1, PETSC_TRUE);CHKERRQ(ierr);

  /* setup solver, dummy solve to really setup */
  {
    KSP ksp; PetscReal krtol,katol,kdtol; PetscInt kmit,one=1;
    ierr = SNESGetKSP(dmpi->snes, &ksp);CHKERRQ(ierr);
    ierr = KSPGetTolerances(ksp,&krtol,&katol,&kdtol,&kmit);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,krtol,katol,kdtol,one);CHKERRQ(ierr);
    ierr = DMPICellSolve( ctx.dm );CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,krtol,katol,kdtol,kmit);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(ctx.events[0],0,0,0,0);CHKERRQ(ierr);
  /* ierr = PetscLogStagePop();CHKERRQ(ierr); */
#endif

  /* do it */
  ierr = go( &ctx );CHKERRQ(ierr);

  if (dmpi->debug>0) PetscPrintf(ctx.wComm,"[%D] done - cleanup\n",ctx.rank);
  /* Particle STREAM test */
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(ctx.events[12],0,0,0,0);CHKERRQ(ierr); /* timer on particle list */
#endif
  {
    int isp,elid; X2PListPos  pos; X2Particle  part;
    ierr = X2ParticleCreate(&part,777777,0,0,0,0);CHKERRQ(ierr);
    for (isp=ctx.use_electrons ? 0 : 1 ; isp <= X2_NION ; isp++) {
      for (elid=0;elid<ctx.nElems;elid++) {
        X2PList *list = &ctx.partlists[isp][elid];
        if (X2PListSize(list)==0) continue;
        ierr = X2PListCompress(list);CHKERRQ(ierr);
        for (pos=0 ; pos < list->vec_top ; pos++) {
          X2PAXPY(1.0,list,part,pos);
        }
      }
    }
  }
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(ctx.events[12],0,0,0,0);CHKERRQ(ierr);
#endif
  /* Cleanup */
  for (idx=0;idx<ctx.proc_send_table_size;idx++) {
    if (ctx.sendListTable[idx].data_size != 0) {
      ierr = X2PSendListDestroy( &ctx.sendListTable[idx] );CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(ctx.sendListTable);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&dmpi->fem);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = destroyParticles(&ctx);CHKERRQ(ierr);
  ierr = DMDestroy(&ctx.dm);CHKERRQ(ierr);
  ierr = PetscFree(ctx.BCFuncs);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&ctx.wComm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* M. Adams, August 2016 */

static char help[] = "X2: A partical in cell code for slab plasmas using PICell.\n";

#ifdef H5PART
#include <H5Part.h>
#endif
#include <petsc/private/dmpicellimpl.h>    /*I   "petscdmpicell.h"   I*/
#include <assert.h>
#include <petscds.h>

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
  PetscReal domain_lo[3], domain_hi[3];
  PetscReal b0[3];
  /* context */
  void *ctx;
} X2Grid;
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
PetscErrorCode X2GridSolverLocatePoints(DM dm, Vec xvec, const X2Ctx *ctx,  IS *pes, IS *elemIDs)
{
  PetscInt         i,idxs[3],n,nn,dim,ii;
  const PetscInt   *np = ctx->grid.solver_np;
  const PetscReal  *dlo = ctx->grid.domain_lo, *dhi = ctx->grid.domain_hi;
  PetscErrorCode   ierr;
  PetscScalar      *xx,*xx0;
  PetscInt         *peidxs,*elemidxs;
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
      xx[i] = x0 + dlo[i] + ctx->grid.solver_proc_idx[i]*dx; /* my fake coordinate */
    }
  }
  ierr = VecRestoreArray(xvec,&xx0);CHKERRQ(ierr);
  if (ctx->nElems==1&&0) {
    for (ii=0;ii<nn;ii++) elemidxs[ii] = 0;
  } else {
    PetscSF            cellSF = NULL;
    const PetscSFNode *foundCells;
    ierr = DMLocatePoints(dm, xvec, DM_POINTLOCATION_NONE, &cellSF);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(cellSF, NULL, NULL, NULL, &foundCells);CHKERRQ(ierr);
    for (ii=0;ii<nn;ii++) {
      elemidxs[ii] = foundCells[ii].index; /* asssumes all processors have same element layout */
      /* *pe = foundCells[0].rank; */
      if (elemidxs[ii]<0) {
        ierr = VecGetArray(xvec,&xx);CHKERRQ(ierr);
        PetscPrintf(PETSC_COMM_SELF,"\t[%D]%s ERROR miseed index %D/%D x = %g %g %g elem id = %D\n",s_rank,__FUNCT__,ii+1,nn,xx[3*ii+0],xx[3*ii+1],xx[3*ii+2],elemidxs[ii]);
        ierr = VecRestoreArray(xvec,&xx);CHKERRQ(ierr);
        PetscSleep(1);
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "missed point");
      }
    }
    ierr = PetscSFDestroy(&cellSF);CHKERRQ(ierr);
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nn,peidxs,PETSC_COPY_VALUES,pes);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nn,elemidxs,PETSC_COPY_VALUES,elemIDs);CHKERRQ(ierr);
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
  PetscBool      chunkFlag,npflag1,npflag2;
  PetscInt       three = 3,i1,i2;
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
  ctx->grid.ft_np[0]  = 1;
  ctx->grid.ft_np[1]  = 1;
  ctx->grid.ft_np[2]  = 1;
  ctx->grid.solver_np[0]  = 1;
  ctx->grid.solver_np[1]  = 1;
  ctx->grid.solver_np[2]  = 1;
  ctx->grid.domain_hi[0]  = 1;
  ctx->grid.domain_hi[1]  = 1;
  ctx->grid.domain_hi[2]  = 1;
  ctx->grid.domain_lo[0]  = -1;
  ctx->grid.domain_lo[1]  = -1;
  ctx->grid.domain_lo[2]  = -1;
  ctx->grid.b0[0]  = .1;
  ctx->grid.b0[1]  = .2;
  ctx->grid.b0[2]  =  1; /* mostly in z */

  ctx->tablecount = 0;

  ierr = PetscOptionsBegin(ctx->wComm, "", "Poisson Problem Options", "X2");CHKERRQ(ierr);
  /* general options */
  s_debug = 0;
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex2.c", s_debug, &s_debug, NULL);CHKERRQ(ierr);
  ctx->plot = PETSC_TRUE;
  ierr = PetscOptionsBool("-plot", "Write plot files", "ex2.c", ctx->plot, &ctx->plot, NULL);CHKERRQ(ierr);
  ctx->chunksize = X2_V_LEN; /* too small */
  ierr = PetscOptionsInt("-chunksize", "Size of particle list to chunk sends", "ex2.c", ctx->chunksize, &ctx->chunksize,&chunkFlag);CHKERRQ(ierr);
  if (chunkFlag) ctx->chunksize = X2_V_LEN*(ctx->chunksize/X2_V_LEN);
  if (ctx->chunksize<=0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB," invalid chuck size = %D",ctx->chunksize);
  ctx->use_bsp = 0;
  ierr = PetscOptionsInt("-use_bsp", "Size of chucks for PETSc's TwoSide communication (0 to use 'non-blocking consensus')", "ex2.c", ctx->use_bsp, &ctx->use_bsp, NULL);CHKERRQ(ierr);
  if (ctx->use_bsp<0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB," invalid BSP chuck size = %D",ctx->use_bsp);
  ctx->proc_send_table_size = ((ctx->npe>10) ? 10 + (ctx->npe-10)/10 : ctx->npe) + 1; /* hash table size of processors to send to */
  ierr = PetscOptionsInt("-proc_send_table_size", "Size of hash table proc->send_list", "ex2.c",ctx->proc_send_table_size, &ctx->proc_send_table_size, NULL);CHKERRQ(ierr);

  /* Domain and mesh definition */
  ierr = PetscOptionsRealArray("-domain_hi", "Domain size", "ex2.c", ctx->grid.domain_hi, &three, NULL);CHKERRQ(ierr);
  three = 3;
  ierr = PetscOptionsRealArray("-domain_lo", "Domain size", "ex2.c", ctx->grid.domain_lo, &three, NULL);CHKERRQ(ierr);
  i1 = 3;
  ierr = PetscOptionsIntArray("-ft_np", "Number of (flux tube) processor in each dimension", "ex2.c", ctx->grid.ft_np, &i1, &npflag1);CHKERRQ(ierr);
  i2 = 3;
  ierr = PetscOptionsIntArray("-solver_np", "Number of (solver) processor in each dimension", "ex2.c", ctx->grid.solver_np, &i2, &npflag2);CHKERRQ(ierr);
  if ( (k=ctx->grid.ft_np[0]*ctx->grid.ft_np[1]*ctx->grid.ft_np[2]) != ctx->npe && !npflag2) { /* recover from inconsistant grid/procs */
    if (npflag1 && i1==3 && k != ctx->npe) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"over constrained number of particle processes npe (%D) != %D",ctx->npe,k);
    else if (npflag1 && i1==2) {
      ctx->grid.ft_np[2] = ctx->npe/(ctx->grid.ft_np[0]*ctx->grid.ft_np[1]);
      if ( (k=ctx->grid.ft_np[0]*ctx->grid.ft_np[1]*ctx->grid.ft_np[2]) != ctx->npe) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"particle grid not working processes npe (%D) != %D",ctx->npe,k);
    }
    else if (npflag1) {
      if (ctx->npe%ctx->grid.ft_np[0]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"over constrained number of particle processes npe (%D) MOD %D",ctx->npe,ctx->grid.ft_np[0]);
      k = ctx->npe/ctx->grid.ft_np[0];
      k = (int)pow((double)k,0.5);
      ctx->grid.ft_np[1] = ctx->grid.ft_np[2] = k;
    }
    else {
      k = (int)pow((double)ctx->npe,0.33334);
      ctx->grid.ft_np[0] = ctx->grid.ft_np[1] = ctx->grid.ft_np[2] = k;
      if ( (k=ctx->grid.ft_np[0]*ctx->grid.ft_np[1]*ctx->grid.ft_np[2]) != ctx->npe) {
        k = (int)pow((double)ctx->npe,0.5);
        if ( ctx->npe%(k*k)==0 ) {
          ctx->grid.ft_np[0] = ctx->grid.ft_np[1] = k;
          ctx->grid.ft_np[2] = ctx->npe/(k*k);
        }
      }
    }
    /* solver process grid */
    ctx->grid.solver_np[0] = ctx->grid.ft_np[0];
    ctx->grid.solver_np[1] = ctx->grid.ft_np[1];
    ctx->grid.solver_np[2] = ctx->grid.ft_np[2];
    if ( (k=ctx->grid.ft_np[0]*ctx->grid.ft_np[1]*ctx->grid.ft_np[2]) != ctx->npe) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"particle grids do not work npe (%D) != %D",ctx->npe,k);
  }
  if (npflag2 && (k=ctx->grid.solver_np[0]*ctx->grid.solver_np[1]*ctx->grid.solver_np[2]) != ctx->npe) { /* solver flag set */
    if (i2==3) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"over constrained number of particle processes npe (%D) != %D",ctx->npe,k);
    else if (i2==2) {
      ctx->grid.solver_np[2] = ctx->npe/(ctx->grid.solver_np[0]*ctx->grid.solver_np[1]);
      if ( (k=ctx->grid.solver_np[0]*ctx->grid.solver_np[1]*ctx->grid.solver_np[2]) != ctx->npe) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"over constrained number of particle processes npe (%D) != %D",ctx->npe,k);
    }
    else { /* have one entry */
      k = ctx->npe/ctx->grid.solver_np[0];
      k = (int)pow((double)k,0.5);
      ctx->grid.solver_np[1] = ctx->grid.solver_np[2] = k;
    }
    if ( (k=ctx->grid.solver_np[0]*ctx->grid.solver_np[1]*ctx->grid.solver_np[2]) != ctx->npe) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"particle grids do not work npe (%D) != %D",ctx->npe,k);
  }
  if ((k=ctx->grid.ft_np[0]*ctx->grid.ft_np[1]*ctx->grid.ft_np[2]) != ctx->npe && npflag2) {
    /* flux tube process grid */
    ctx->grid.ft_np[0] = ctx->grid.solver_np[0];
    ctx->grid.ft_np[1] = ctx->grid.solver_np[1];
    ctx->grid.ft_np[2] = ctx->grid.solver_np[2];
  }
  else if (!npflag2) { /* ft was good but solver needs it */
    /* solver process grid */
    ctx->grid.solver_np[0] = ctx->grid.ft_np[0];
    ctx->grid.solver_np[1] = ctx->grid.ft_np[1];
    ctx->grid.solver_np[2] = ctx->grid.ft_np[2];
  }
  if ( (k=ctx->grid.solver_np[0]*ctx->grid.solver_np[1]*ctx->grid.solver_np[2]) != ctx->npe) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"solver grids do not work npe (%D) != %D",ctx->npe,k);
  if ( (k=ctx->grid.ft_np[0]*ctx->grid.ft_np[1]*ctx->grid.ft_np[2]) != ctx->npe)             SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"particle flux tube grids do not work npe (%D) != %D",ctx->npe,k);
  { /* debug */
    PetscInt i=X2_IDX_X(s_rank,ctx->grid.ft_np),j=X2_IDX_Y(s_rank,ctx->grid.ft_np),k=X2_IDX_Z(s_rank,ctx->grid.ft_np);
    PetscInt rank = X2_IDX(i,j,k,ctx->grid.ft_np);
    if (rank!=s_rank) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB," index maps not correct X2_IDX = %D, rank = %D",rank,s_rank);
  }
  { /* solver_proc_idx */
    int i,j,k;
    ctx->grid.solver_proc_idx[0] = -1;
    for (i=0;i<ctx->grid.solver_np[0] && ctx->grid.solver_proc_idx[0] == -1;i++) {
    for (j=0;j<ctx->grid.solver_np[1] && ctx->grid.solver_proc_idx[0] == -1;j++) {
    for (k=0;k<ctx->grid.solver_np[2] && ctx->grid.solver_proc_idx[0] == -1;k++) {
      if (X2_IDX(i,j,k,ctx->grid.solver_np) == ctx->rank) {
        ctx->grid.solver_proc_idx[0] = i;
        ctx->grid.solver_proc_idx[1] = j;
        ctx->grid.solver_proc_idx[2] = k;
      }
    }}}
    assert(ctx->grid.solver_proc_idx[0] != -1);
    assert(X2_IDX_X(s_rank,ctx->grid.solver_np) == ctx->grid.solver_proc_idx[0]);
    assert(X2_IDX_Y(s_rank,ctx->grid.solver_np) == ctx->grid.solver_proc_idx[1]);
    assert(X2_IDX_Z(s_rank,ctx->grid.solver_np) == ctx->grid.solver_proc_idx[2]);
  }
  three = 3;
  ierr = PetscOptionsRealArray("-b0", "B_0 vector", "ex2.c", ctx->grid.b0, &three, NULL);CHKERRQ(ierr);
  {
    PetscReal *b0 = ctx->grid.b0, len = b0[0]*b0[0] + b0[1]*b0[1] + b0[2]*b0[2];
    if (b0[2]==0.) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bad B_0 vector: must have z(2) component: %g %g %g",b0[0],b0[1],b0[2]);
    len = sqrt(len);
    if (len==0) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bad B_0 vector length %g %g %g",b0[0],b0[1],b0[2]);
    for (k=0;k<3;k++) b0[k] /= len;
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
    if (ctx->rank==0) ctx->num_particles_proc = -ctx->num_particles_proc;
    else ctx->num_particles_proc = 0;
  }
  if (!chunkFlag) ctx->chunksize = X2_V_LEN*((ctx->num_particles_proc/80+1)/X2_V_LEN+1); /* an intelegent message chunk size */

  ctx->collision_period = 10000;
  ierr = PetscOptionsInt("-collision_period", "Period between collision operators", "ex2.c", ctx->collision_period, &ctx->collision_period, NULL);CHKERRQ(ierr);
  ctx->use_electrons = PETSC_TRUE; /* need neutral because periodic domain */
  ierr = PetscOptionsBool("-use_electrons", "Include electrons", "ex2.c", ctx->use_electrons, &ctx->use_electrons, NULL);CHKERRQ(ierr);
  ctx->max_vpar = 1.;
  ierr = PetscOptionsReal("-max_vpar", "Maximum parallel velocity", "ex2.c",ctx->max_vpar,&ctx->max_vpar,NULL);CHKERRQ(ierr);
  npflag2 = PETSC_FALSE;
  ierr = PetscOptionsBool("-periodic_domain", "Periodic domain", "ex2.c", npflag2, &npflag2, &npflag1);CHKERRQ(ierr);
  if (npflag1 && npflag2) ctx->dtype = X2_PERIODIC;
  else ctx->dtype = X2_DIRI;
  ctx->use_mms = PETSC_FALSE;
  ierr = PetscOptionsBool("-use_mms", "Us a manufactured RHS for particle weight", "ex2.c", ctx->use_mms, &ctx->use_mms, NULL);CHKERRQ(ierr);
  ctx->use_vel_update = PETSC_TRUE;
  ierr = PetscOptionsBool("-use_vel_update", "Update the particle velocity with the E field", "ex2.c", ctx->use_vel_update, &ctx->use_vel_update, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (s_debug>0) PetscPrintf(ctx->wComm,"npe=%D; %Dx%Dx%D solver grid; %Dx%Dx%D particle grid grid; mpi_send size (chunksize) equal %d particles. %s, %s, %s\n",
                             ctx->npe,ctx->grid.solver_np[0],ctx->grid.solver_np[1],ctx->grid.solver_np[2],ctx->grid.ft_np[0],
                             ctx->grid.ft_np[1],ctx->grid.ft_np[2],ctx->chunksize,
                             ctx->use_electrons ? "use electrons" : "ions only", ctx->use_bsp ? "BSP communication" : "Non-blocking consensus communication",
#ifdef X2_S_OF_V
			     "Use struct of arrays"
#else
			     "Use of array structs"
#endif
                             );

  PetscFunctionReturn(0);
}

/* X2GridFluxTubeLocatePoint: find z(2) flux tube for this point (should vectorize)
    Input:
     - grid: the particle grid
     - x:
   Output:
    - pe: process ID
    - elem: element ID
*/
#undef __FUNCT__
#define __FUNCT__ "X2GridFluxTubeLocatePoint"
PetscErrorCode X2GridFluxTubeLocatePoint( const X2Grid *grid, PetscReal x[3], PetscMPIInt *pe, PetscInt *elem)
{
  PetscInt        i,ii[3];
  X2Ctx           *ctx = (X2Ctx*)grid->ctx;
  const PetscInt  *np = ctx->grid.ft_np;
  const PetscReal *dlo = ctx->grid.domain_lo, *dhi = ctx->grid.domain_hi, *b0 = ctx->grid.b0;
  PetscReal       dx[3],xstar[3],deltaz;
  /* PetscErrorCode ierr; */
  PetscFunctionBeginUser;
  PetscValidPointer(x, 2);
  PetscValidPointer(pe, 3);
  PetscValidPointer(elem, 4);
  if (x[0]<dlo[0] || x[0]>dhi[0] || x[1]<dlo[1] ||
      x[1]>dhi[1] || x[2]<dlo[2] || x[2]>dhi[2]) {
    SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"point out of bounds %g %g %g",x[0],x[1],x[2]);
  }
  /* find processor's flux tube for x */
  for (i=0;i<3;i++) dx[i] = (dhi[i]-dlo[i])/(PetscReal)np[i];
  for (i=0;i<3;i++) ii[i] = (PetscInt)((x[i]-dlo[i])/dx[i]);
  deltaz = x[2] - dlo[2] - ii[2]*dx[2];
  for (i=0;i<3;i++) xstar[i] = x[i] - b0[i]/b0[2]*deltaz + 1.e-14; /* keep in same z plane */
  for (i=0;i<2;i++) {
    while (xstar[i] >= dhi[i]) xstar[i] -= (dhi[i]-dlo[i]);
    while (xstar[i] <  dlo[i]) xstar[i] += (dhi[i]-dlo[i]);
  }
  for (i=0;i<3;i++) ii[i] = (PetscInt)((xstar[i]-dlo[i])/dx[i]);
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
    r   = ctx->grid.domain_lo[0];
    z   = ctx->grid.domain_lo[1];
    phi = ctx->grid.domain_lo[2];
    X2ParticleCreate(&part,1,r,z,phi,0.);
    X2PListAdd(l,&part,ppos2);
    r   = ctx->grid.domain_hi[0];
    z   = ctx->grid.domain_hi[1];
    phi = ctx->grid.domain_hi[2];
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

#undef __FUNCT__
#define __FUNCT__ "u_x4_op"
PetscErrorCode u_x4_op(PetscInt dim, PetscReal time, const PetscReal xx[], PetscInt Nf, PetscScalar *u, void *a_ctx)
{
  X2Ctx *ctx = (X2Ctx*)a_ctx;
  PetscInt comp,i;
  const PetscReal  *dlo = ctx->grid.domain_lo, *dhi = ctx->grid.domain_hi;
  PetscReal dlen[3];
  PetscFunctionBeginUser;
  for(i=0;i<dim;i++) dlen[i] = dhi[i] - dlo[i];
  for (comp = 0; comp < Nf; ++comp) {
    u[comp] = 1;
    for (i = 0; i < dim; ++i) {
      PetscReal x = (xx[i]-dlo[i])/dlen[i];
      u[comp] *= (x*x - x*x*x*x);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "u_sinz_op"
PetscErrorCode u_sinz_op(PetscInt dim, PetscReal time, const PetscReal xx[], PetscInt Nf, PetscScalar *u, void *a_ctx)
{
  X2Ctx *ctx = (X2Ctx*)a_ctx;
  PetscInt comp,i;
  const PetscReal  *dlo = ctx->grid.domain_lo, *dhi = ctx->grid.domain_hi;
  PetscReal dlen[3];
  PetscFunctionBeginUser;
  for(i=0;i<dim;i++) dlen[i] = dhi[i] - dlo[i];
  for (comp = 0; comp < Nf; ++comp) {
    u[comp] = 1;
    /* for (i = 0; i < dim; ++i) { */
    /* PetscReal x = (xx[i]-dlo[i])/dlen[i]; */
    /* u[comp] *= (x*x - x*x*x*x); */
    i = 2;
    u[comp] *= sin(2*M_PI*(xx[i]-dlo[i])/dlen[i]);
  }
  PetscFunctionReturn(0);
}

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
#define X2PROCLISTSIZE 256
#define X2_HASH(__n) ((__n*593)%ctx->proc_send_table_size)
#undef __FUNCT__
#define __FUNCT__ "processParticles"
static PetscErrorCode processParticles( X2Ctx *ctx, const PetscReal dt, X2PSendList **sendListTable_in, const PetscMPIInt tag,
					const int irk, const int istep, PetscBool solver)
{
  X2Grid           *grid = &ctx->grid;
  X2PSendList      *sendListTable = *sendListTable_in;
  DM_PICell        *dmpi = (DM_PICell *) ctx->dm->data;     assert(!(!solver && irk>=0)); /* don't push flux tubes */
  PetscMPIInt      pe;
  X2Particle       part;
  X2PListPos       pos;
  PetscErrorCode   ierr;
  const int        part_dsize = sizeof(X2Particle)/sizeof(double);
  Vec              jVec,xVec,vVec;
  PetscScalar      *xx=0,*jj=0,*vv=0,*xx0=0,*jj0=0,*vv0=0;
  PetscInt         nslist,nlistsTot,elid,elid2,one=1,three=3,ndeposit;
  long int         hash;
  int              ii,isp,origNlocal,nmoved;
  IS               pes,elems;
  const PetscInt   *cpeidxs,*celemidxs;
  PetscInt         *peidxs,*elemidxs;
  const PetscReal *dlo = ctx->grid.domain_lo, *dhi = ctx->grid.domain_hi, *b0 = ctx->grid.b0;
  PetscReal dlen[3];
  X2ISend          slist[X2PROCLISTSIZE];

  PetscFunctionBeginUser;
  MPI_Barrier(ctx->wComm);
  for(ii=0;ii<3;ii++) dlen[ii] = dhi[ii] - dlo[ii];
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
    const PetscReal mass = ctx->species[isp].mass;
    const PetscReal charge = ctx->species[isp].charge;
    /* loop over element particle lists */
    for (elid=0;elid<ctx->nElems;elid++) {
      X2PList *list = &ctx->partlists[isp][elid];
      if (X2PListSize(list)==0) continue;
      origNlocal += X2PListSize(list);
      /* get Cartesian coordinates (not used for flux tube move) */
      ierr = X2PListCompress(list);CHKERRQ(ierr); /* allows for simpler vectorization */
      if (solver) {
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventBegin(ctx->events[7],0,0,0,0);CHKERRQ(ierr); /* timer on particle list */
#endif
        /* make vectors for this element */
        ierr = VecCreateSeq(PETSC_COMM_SELF,three*list->vec_top, &xVec);CHKERRQ(ierr);
        ierr = VecSetBlockSize(xVec,three);CHKERRQ(ierr);
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
        ierr = PetscLogEventBegin(ctx->events[8],0,0,0,0);CHKERRQ(ierr); /* timer on particle list */
#endif
        /* push, and collect x */
        if (irk>=0) {
          Vec locphi;
          ierr = DMGetLocalVector(dmpi->dm, &locphi);CHKERRQ(ierr);
          ierr = DMGlobalToLocalBegin(dmpi->dm, dmpi->phi, INSERT_VALUES, locphi);CHKERRQ(ierr);
          /* PetscPrintf(PETSC_COMM_SELF,"\t\t\t[%D]%s call DMGlobalToLocalEnd\n",s_rank,__FUNCT__); */ /* CODE HANGS HERE */
          ierr = DMGlobalToLocalEnd(dmpi->dm, dmpi->phi, INSERT_VALUES, locphi);CHKERRQ(ierr);
          /* PetscPrintf(PETSC_COMM_SELF,"\t\t\t\t[%D]%s DMGlobalToLocalEnd DONE\n",s_rank,__FUNCT__); */
          /* get E, should set size of vecs for true size? */
          ierr = VecCreateSeq(PETSC_COMM_SELF,three*list->vec_top,&jVec);CHKERRQ(ierr);
          ierr = DMPICellGetJet(ctx->dm, xVec, locphi, elid, jVec);CHKERRQ(ierr);
          ierr = DMRestoreLocalVector(dmpi->dm, &locphi);CHKERRQ(ierr);
          ierr = VecGetArray(jVec,&jj0);CHKERRQ(ierr); jj = jj0;
        }
        /* vectorize (todo) push: theta = theta + q*dphi .... grad not used */
        ierr = VecGetArray(xVec,&xx0);CHKERRQ(ierr); xx = xx0;
        for (pos=0 ; pos < list->vec_top ; pos++, xx += 3, jj += 3 ) {
	  /* push particle, real data, could do it on copy for non-final stage of TS */
          if (irk>=0) {
            PetscReal r, b0dotgrad = jj[0]*b0[0] + jj[1]*b0[1] + jj[2]*b0[2];
#ifdef X2_S_OF_V
            if (ctx->use_vel_update) list->data_v.vpar[pos] += -dt*b0dotgrad*charge/mass;
            r = dt*list->data_v.vpar[pos]; /* could use average of this and the updated velocity */
            for(ii=0;ii<3;ii++) {
              list->data_v.x[ii][pos] += r*b0[ii];
              while (list->data_v.x[ii][pos] >= dhi[ii]) list->data_v.x[ii][pos] -= dlen[ii];
              while (list->data_v.x[ii][pos] <  dlo[ii]) list->data_v.x[ii][pos] += dlen[ii];
              xx[ii] = list->data_v.x[ii][pos];
            }
#else
            if (ctx->use_vel_update) list->data[pos].vpar += -dt*b0dotgrad*charge/mass;
            r = dt*list->data[pos].vpar; /* could use average of this and the updated velocity */
            for(ii=0;ii<3;ii++) {
              list->data[pos].x[ii] += r*b0[ii];
              while (list->data[pos].x[ii] >= dhi[ii]) list->data[pos].x[ii] -= dlen[ii];
              while (list->data[pos].x[ii] <  dlo[ii]) list->data[pos].x[ii] += dlen[ii];
              xx[ii] = list->data[pos].x[ii];
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
        ierr = VecRestoreArray(xVec,&xx0);CHKERRQ(ierr);
        if (irk>=0) {
          ierr = VecRestoreArray(jVec,&jj0);CHKERRQ(ierr);
          ierr = VecDestroy(&jVec);CHKERRQ(ierr);
        }
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventEnd(ctx->events[8],0,0,0,0);CHKERRQ(ierr);
#endif
      }
      /* collect pes to communicate - not vectorizable */
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventBegin(ctx->events[9],0,0,0,0);CHKERRQ(ierr);
#endif
      /* get pe & element id */
      if (solver) {
        /* see if need communication? no: add density, yes: add to communication list */
        ierr = X2GridSolverLocatePoints(dmpi->dm, xVec, ctx, &pes, &elems);CHKERRQ(ierr);
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
      ierr = PetscLogEventEnd(ctx->events[9],0,0,0,0);CHKERRQ(ierr);
      ierr = PetscLogEventBegin(ctx->events[5],0,0,0,0);CHKERRQ(ierr);
#endif
      /* move particles, not vectorizable */
      ierr = ISGetIndices(pes,&cpeidxs);CHKERRQ(ierr);
      ierr = ISGetIndices(elems,&celemidxs);CHKERRQ(ierr);
      for (pos=0 ; pos < list->top ; pos++ ) {
	pe = cpeidxs[pos]; assert(pe<ctx->npe);
	elid2 = celemidxs[pos]; /* we can not just move to new list because it might get pushed again, so use MPI (local) buffer as a cache */
	if (pe==ctx->rank && elid2==elid) continue; /* don't move */
        /* rehash if needed */
        if (ctx->tablecount >= (7*ctx->proc_send_table_size)/8) { /* rehash */
          /* need to rehash */
          X2PSendList *newdata;
          int idx,jjj,iii,oldsize = ctx->proc_send_table_size;
          ctx->proc_send_table_size *= 2;
          ierr = PetscMalloc1(ctx->proc_send_table_size, &newdata);CHKERRQ(ierr);
          for (idx=0;idx<ctx->proc_send_table_size;idx++) {
            newdata[idx].data_size = 0; /* init */
          }
          /* copy over old lists */
          for (jjj=0;jjj<oldsize;jjj++) {
            if (sendListTable[jjj].data_size) { /* an entry */
              PetscMPIInt pe2 = sendListTable[jjj].proc;
              long int hash2 = X2_HASH(pe2); /* new hash */
              for (iii=0;iii<ctx->proc_send_table_size;iii++){
                if (newdata[hash2].data_size==0) {
                  newdata[hash2].data      = sendListTable[jjj].data;
                  newdata[hash2].size      = sendListTable[jjj].size;
                  newdata[hash2].data_size = sendListTable[jjj].data_size;
                  newdata[hash2].proc      = sendListTable[jjj].proc;
                  break;
                }
                if (++hash2 == ctx->proc_send_table_size) hash2=0;
              }
              if (iii==ctx->proc_send_table_size) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Failed to find? (%D)",ctx->proc_send_table_size);
            }
          }
          ierr = PetscFree(sendListTable);CHKERRQ(ierr);
          sendListTable = newdata;
          *sendListTable_in = newdata;
        }
#ifdef X2_S_OF_V
	X2V2P((&part),list->data_v,pos); /* return copy */
#else
	part = list->data[pos]; /* return copy */
#endif
	/* add to list to send, find list with table lookup, send full lists - no vectorization */
	hash = X2_HASH(pe); /* hash */
	for (ii=0;ii<ctx->proc_send_table_size;ii++){
	  if (sendListTable[hash].data_size==0) { /* found entry but need to create it */
            int np = ctx->num_particles_proc, cs = ctx->chunksize; /* lots of local movement */
            if (pe==ctx->rank) {
              ierr = X2PSendListCreate(&sendListTable[hash], cs*(np/cs + np/cs/64 + 1) );CHKERRQ(ierr);
            } else {
              ierr = X2PSendListCreate(&sendListTable[hash], cs*(np/cs/ctx->nElems + np/cs/ctx->nElems/8 + 1));CHKERRQ(ierr);
            }
	    sendListTable[hash].proc = pe;
	    ctx->tablecount++;
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
	      ierr = MPI_Isend((void*)slist[nslist].data,X2PSendListSize(&sendListTable[hash])*part_dsize,mtype,pe,tag+isp,ctx->wComm,&slist[nslist].request);
	      CHKERRQ(ierr);
	      nslist++;
	      /* ready for next round, save meta-data  */
	      ierr = X2PSendListClear(&sendListTable[hash]);CHKERRQ(ierr);
	      sendListTable[hash].data = 0;
	      ierr = PetscMalloc1(sendListTable[hash].data_size, &sendListTable[hash].data);CHKERRQ(ierr);
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
      }
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventEnd(ctx->events[5],0,0,0,0);CHKERRQ(ierr);
#endif
    } /* element list */

    /* finish sends and receive new particles for this species */
    ierr = shiftParticles(ctx, sendListTable, ctx->partlists[isp], &nslist, X2PROCLISTSIZE, slist, tag+isp, solver);CHKERRQ(ierr);

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
    /* add density (while in cache, by species at least) irk<0 first time to get rho */
    if (solver) {
      Vec locrho;
      ierr = DMGetLocalVector(dmpi->dm, &locrho);CHKERRQ(ierr);
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
          /* method of manufactured solution, scale weight by f(x) */
          if (ctx->use_mms) {
            PetscScalar fact;
            *vv *= (double)ctx->nElems/(double)ctx->num_particles_proc; /* normalize */
            if (ctx->dtype == X2_PERIODIC) {
              /* ii = 2; */
              ierr = u_sinz_op(3,0.0,xx,1,&fact,ctx);CHKERRQ(ierr);
              /* *vv *= sin(2*M_PI*(xx[ii]-dlo[ii])/dlen[ii]); */
            } else {
              ierr = u_x4_op(3,0.0,xx,1,&fact,ctx);CHKERRQ(ierr);
              /* for(ii=0;ii<3;ii++) { */
              /*   PetscReal x = (xx[ii]-dlo[ii])/dlen[ii]; */
              /*   *vv *= (x*x - x*x*x*x); */
              /* } */
            }
            *vv *= fact;
          }
          ndeposit++;
        }
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
      ierr = DMLocalToGlobalBegin(dmpi->dm, locrho, ADD_VALUES, dmpi->rho);CHKERRQ(ierr);
      ierr = DMLocalToGlobalEnd(dmpi->dm, locrho, ADD_VALUES, dmpi->rho);CHKERRQ(ierr);
      ierr = DMRestoreLocalVector(dmpi->dm, &locrho);CHKERRQ(ierr);
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
        if (!isp) {
          sprintf(fname1,         "ex2_particles_electrons_time%05d.h5part",istep+1);
          sprintf(fname2,"ex2_sub_rank_particles_electrons_time%05d.h5part",istep+1);
        } else {
          sprintf(fname1,         "ex2_particles_sp%d_time%05d.h5part",isp,istep+1);
          sprintf(fname2,"ex2_sub_rank_particles_sp%d_time%05d.h5part",isp,istep+1);
        }
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
  PetscErrorCode  ierr;
  PetscInt        isp,gid,i,j,dim,cStart,cEnd,elid;
  const PetscInt  *np = ctx->grid.ft_np;
  const PetscReal *dlo = ctx->grid.domain_lo, *dhi = ctx->grid.domain_hi, *b0 = ctx->grid.b0;
  const PetscReal dx=(dhi[0]-dlo[0])/(PetscReal)np[0];
  const PetscReal x1=dlo[0] + dx*X2_IDX_X(ctx->rank,np);
  const PetscReal dy=(dhi[1]-dlo[1])/(PetscReal)np[1];
  const PetscReal y1=dlo[1] + dy*X2_IDX_Y(ctx->rank,np);
  const PetscReal dz=(dhi[2]-dlo[2])/(PetscReal)np[2];
  const PetscReal z1=dlo[2] + dz*X2_IDX_Z(ctx->rank,np);
  X2Particle      particle;
  DM_PICell       *dmpi;
  PetscFunctionBeginUser;

  /* Create vector and get pointer to data space */
  dmpi = (DM_PICell *) ctx->dm->data;
  ierr = DMGetDimension(dmpi->dm, &dim);CHKERRQ(ierr);
  if (dim!=3) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"wrong dimension (3) = %D",dim);
  ierr = DMGetCellChart(dmpi->dm, &cStart, &cEnd);CHKERRQ(ierr);
  ctx->nElems = PetscMax(1,cEnd-cStart);CHKERRQ(ierr);
  /* setup particles - lexicographic partition of -- flux tube -- cells */
  gid = ctx->rank*ctx->num_particles_proc;

  /* my first cell index */
  srand(ctx->rank);
  for (isp=ctx->use_electrons ? 0 : 1 ; isp <= X2_NION ; isp++ ) {
    const PetscReal maxe=ctx->max_vpar*ctx->max_vpar,mass=ctx->species[isp].mass,charge=ctx->species[isp].charge,zmax=1.0 - exp(-maxe);
    ierr = PetscMalloc1(ctx->nElems,&ctx->partlists[isp]);CHKERRQ(ierr);
    /* create list for element 0 and add all to it */
    ierr = X2PListCreate(&ctx->partlists[isp][s_fluxtubeelem],
                         ctx->chunksize*(ctx->num_particles_proc/ctx->chunksize + ctx->num_particles_proc/ctx->chunksize/64 + 1));
    CHKERRQ(ierr);
    /* create each particle */
    for (i=0 ; i<ctx->num_particles_proc; i++ ) {
      PetscReal xx[3],deltaz;
      PetscReal v,zdum,vpar;
      deltaz = (PetscReal)(rand()%X2NDIG+1)/(PetscReal)(X2NDIG+1)*dz;
      xx[2] = z1 + deltaz;
      xx[0] = x1 + (PetscReal)(rand()%X2NDIG+1)/(PetscReal)(X2NDIG+1)*dx + b0[0]/b0[2]*deltaz;
      xx[1] = y1 + (PetscReal)(rand()%X2NDIG+1)/(PetscReal)(X2NDIG+1)*dy + b0[1]/b0[2]*deltaz;
      for (j=0;j<2;j++) {
        while (xx[j] >= dhi[j]) xx[j] -= (dhi[j]-dlo[j]);
        while (xx[j] <  dlo[j]) xx[j] += (dhi[j]-dlo[j]);
      }
#ifdef PETSC_USE_DEBUG
      {
        PetscMPIInt pe; PetscInt id;
        ierr = X2GridFluxTubeLocatePoint(&ctx->grid,xx,&pe,&id);CHKERRQ(ierr);
        if(pe != ctx->rank){
          SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB," created particle for proc %D, %g %g %g",pe,xx[0],xx[1],xx[2]);
        }
      }
#endif
      /* v_parallel from random number */
      zdum = zmax*(PetscReal)(rand()%X2NDIG)/(PetscReal)X2NDIG;
      v = sqrt(-2.0/mass*log(1.0-zdum));
      v = v*cos(M_PI*(PetscReal)(rand()%X2NDIG)/(PetscReal)X2NDIG);
      /* vshift= v + up ! shift of velocity */
      vpar = v*mass/charge;
      ierr = X2ParticleCreate(&particle,++gid,xx[0],xx[1],xx[2],vpar);CHKERRQ(ierr); /* only time this is called! */
      ierr = X2PListAdd(&ctx->partlists[isp][s_fluxtubeelem],&particle, NULL);CHKERRQ(ierr);
      /* debug, particles are created in a flux tube */
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
#define __FUNCT__ "setupDiscretization"
static PetscErrorCode setupDiscretization(X2Ctx *ctx, PetscInt dim)
{
  PetscErrorCode ierr;
  DM_PICell      *dmpi = (DM_PICell *) ctx->dm->data;
  PetscDS        prob;
  PetscFunctionBeginUser;
  /* fem */
  ierr = PetscFEDestroy(&dmpi->fem);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dmpi->dm, dim, 1, PETSC_FALSE, NULL, PETSC_DECIDE, &dmpi->fem);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dmpi->fem, "poisson");CHKERRQ(ierr);
  /* FEM prob */
  ierr = DMGetDS(dmpi->dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) dmpi->fem);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 0, f0_u, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
static PetscErrorCode CreateMesh(X2Ctx *ctx)
{
  PetscErrorCode   ierr;
  PetscInt         dimEmbed, i, id;
  PetscInt         nCoords,dim;
  PetscScalar      *coords;
  Vec              coordinates;
  DM_PICell        *dmpi = (DM_PICell *) ctx->dm->data;
  DM               dm;
  DMLabel          label;
  PetscInt         *sizes = NULL, *points = NULL, * counts = NULL, cells[] = {2,2,2};
  PetscPartitioner part;
  PetscFunctionBeginUser;

  /* setup solver grid */
  dim = 3;
  if (ctx->dtype == X2_PERIODIC) {
    if (ctx->npe==1) {
      for (i = 0; i < dim; i++) cells[i] = 4; /* periodic on one proc needs help */
      ierr = DMPlexCreateHexBoxMesh(ctx->wComm, dim, cells, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, &dmpi->dm);CHKERRQ(ierr);
    } else {
      /* periodic wants 2^D cells per processor */
      for (i = 0; i < dim; i++) cells[i] *= ctx->grid.solver_np[i];
      ierr = DMPlexCreateHexBoxMesh(ctx->wComm, dim, cells, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, &dmpi->dm);CHKERRQ(ierr);
    }
    ierr = DMLocalizeCoordinates(dmpi->dm);CHKERRQ(ierr);
  }
  else {
    /* one cell per processor */
    for (i = 0; i < dim; i++) cells[i] = ctx->grid.solver_np[i];
    ierr = DMPlexCreateHexBoxMesh(ctx->wComm, dim, cells, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, &dmpi->dm);CHKERRQ(ierr);
  }
  /* set domain size */
  ierr = DMGetCoordinatesLocal(dmpi->dm,&coordinates);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dmpi->dm,&dimEmbed);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordinates,&nCoords);CHKERRQ(ierr);
  if (nCoords % dimEmbed) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Coordinate vector the wrong size");CHKERRQ(ierr);
  ierr = VecGetArray(coordinates,&coords);CHKERRQ(ierr);
  for (i = 0; i < nCoords; i += dimEmbed) {
    PetscInt j;
    PetscScalar *coord = &coords[i];
    for (j = 0; j < dimEmbed; j++) {
      coord[j] = ctx->grid.domain_lo[j] + coord[j] * (ctx->grid.domain_hi[j] - ctx->grid.domain_lo[j]);
    }
  }
  if (ctx->dtype == X2_PERIODIC) { /* modify periodicity to match new bounds */
    PetscReal            *maxCell, *L;
    const PetscReal      *maxCellOrig, *Lorig;
    DMBoundaryType       *bdtype;
    const DMBoundaryType *bdtypeOrig;
    PetscInt              i;

    ierr = PetscMalloc3(dim,&maxCell,dim,&L,dim,&bdtype);CHKERRQ(ierr);
    ierr = DMGetPeriodicity(dmpi->dm,&maxCellOrig,&Lorig,&bdtypeOrig);CHKERRQ(ierr);
    for (i = 0; i < dim; i++) {
      maxCell[i] = maxCellOrig[i] * (ctx->grid.domain_hi[i] - ctx->grid.domain_lo[i]);
      L[i]       = Lorig[i] * (ctx->grid.domain_hi[i] - ctx->grid.domain_lo[i]);
      bdtype[i]  = bdtypeOrig[i];
    }
    ierr = DMSetPeriodicity(dmpi->dm,maxCell,L,bdtype);CHKERRQ(ierr);
    ierr = PetscFree3(maxCell,L,bdtype);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coordinates,&coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dmpi->dm,coordinates);CHKERRQ(ierr);

  ierr = DMSetApplicationContext(dmpi->dm, &ctx);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) dmpi->dm, "x2_");CHKERRQ(ierr);
  ierr = PetscMalloc(1 * sizeof(PetscErrorCode (*)(PetscInt,const PetscReal [],PetscInt,PetscScalar*,void*)),&ctx->BCFuncs);
  CHKERRQ(ierr);
  ctx->BCFuncs[0] = zero;
  /* add BCs */
  ierr = DMCreateLabel(dmpi->dm, "boundary");CHKERRQ(ierr);
  ierr = DMGetLabel(dmpi->dm, "boundary", &label);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dmpi->dm, label);CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(dmpi->dm, label);CHKERRQ(ierr);
  id = 1;
  ierr = DMAddBoundary(dmpi->dm, DM_BC_ESSENTIAL, "wall", "boundary", 0, 0, NULL, (void (*)()) ctx->BCFuncs[0], 1, &id, &ctx);CHKERRQ(ierr);
  if (sizeof(long long)!=sizeof(PetscReal)) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "sizeof(long long)!=sizeof(PetscReal)");

  ierr = setupDiscretization( ctx, dim );CHKERRQ(ierr);

  /* set a simple partitioner - needed to make my indexing work for point locate */
  if (!ctx->rank && ctx->npe>1) {
    PetscInt cEnd,c,*cs,idx[3],i;
    PetscInt *offsets;
    const PetscReal *dlo = ctx->grid.domain_lo, *dhi = ctx->grid.domain_hi;
    const PetscInt locN = ctx->dtype == X2_PERIODIC ? 2 : 1, c_proc = pow(locN,dim);
    ierr = DMPlexGetHeightStratum(dmpi->dm, 0, NULL, &cEnd);CHKERRQ(ierr); /* DMGetCellChart */
    if (cEnd && cEnd!=ctx->npe*c_proc) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_USER, "cEnd=%d != %d",cEnd,ctx->npe*c_proc);
    if (cEnd!=ctx->npe*c_proc) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_USER, "cEnd=%d != %d",cEnd,ctx->npe*c_proc);
    ierr = PetscMalloc2(ctx->npe, &sizes, ctx->npe*c_proc, &points);CHKERRQ(ierr);
    ierr = PetscMalloc1(ctx->npe, &offsets);CHKERRQ(ierr);
    for (c=0,cs=sizes;c<ctx->npe;c++,cs++) *cs = c_proc;
    offsets[0] = 0;
    for (c = 1; c < ctx->npe; c++) {offsets[c] = offsets[c - 1] + sizes[c];}
    for (c=0;c<ctx->npe*c_proc;c++) {
      PetscInt  proc;
      PetscReal v0[81],detJ[27];
      ierr = DMPlexComputeCellGeometryFEM(dmpi->dm, c, NULL, v0, NULL, NULL, detJ);CHKERRQ(ierr);
      for(i=0;i<3;i++) {
        idx[i] = (PetscInt)((v0[i]-dlo[i]+1.e-12)/(dhi[i]-dlo[i])*(double)ctx->grid.solver_np[i]);
      }
      proc = X2_IDX3(idx,ctx->grid.solver_np);
      points[offsets[proc]++] = c;
    }
    ierr = PetscFree(offsets);CHKERRQ(ierr);
  }
  if (ctx->npe>1) {
    ierr = DMPlexGetPartitioner(dmpi->dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetType(part, PETSCPARTITIONERSHELL);CHKERRQ(ierr);
    ierr = PetscPartitionerShellSetPartition(part, ctx->npe, sizes, points);CHKERRQ(ierr);
    if (sizes) ierr = PetscFree3(sizes,counts,points);CHKERRQ(ierr);
  }
  /* distribute */
  ierr = DMPlexDistribute(dmpi->dm, 0, NULL, &dm);CHKERRQ(ierr);
  if (dm) {
    ierr = PetscObjectSetOptionsPrefix((PetscObject)dm,"x2_");CHKERRQ(ierr);
    ierr = DMDestroy(&dmpi->dm);CHKERRQ(ierr);
    dmpi->dm = dm;
  }
  else assert(ctx->npe==1);

  /* set from options: refinement done here */
  ierr = DMSetFromOptions( ctx->dm );CHKERRQ(ierr);
  ierr = setupDiscretization( ctx, dim );CHKERRQ(ierr);
  ierr = DMSetUp( ctx->dm );CHKERRQ(ierr); /* create vectors */

  if (dmpi->debug>3) { /* this shows a bug with crap in the section */
    PetscSection     sec;
    ierr = DMGetDefaultGlobalSection(dmpi->dm, &sec);CHKERRQ(ierr);
    /* diagnostics */
    if (!sec) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "DMGetDefaultSection return NULL");
    ierr = PetscSectionView(sec,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  if (dmpi->debug>2) {
    ierr = DMView(dmpi->dm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  {
    PetscInt n,cStart,cEnd;
    ierr = VecGetSize(dmpi->rho,&n);CHKERRQ(ierr);
    if (!n) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "No dofs");
    ierr = DMPlexGetHeightStratum(dmpi->dm, 0, &cStart, &cEnd);CHKERRQ(ierr); /* DMGetCellChart */
    if (cStart) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER, "cStart != 0. %D",cStart);
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
  PetscInt       idx,isp;
  Mat            J;
  PetscFunctionBeginUser;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ctx.events = s_events;
  ctx.grid.ctx = &ctx;
#if defined(PETSC_USE_LOG)
  {
    PetscInt currevent = 0;
    /* PetscLogStage  setup_stage; */
    ierr = PetscLogEventRegister("X2Setup*", DM_CLASSID, &ctx.events[currevent++]);CHKERRQ(ierr); /* 0 */
    ierr = PetscLogEventRegister("X2Process parts",0,&ctx.events[currevent++]);CHKERRQ(ierr); /* 1 */
    ierr = PetscLogEventRegister(" -shiftParticles",0,&ctx.events[currevent++]);CHKERRQ(ierr); /* 2 */
    ierr = PetscLogEventRegister("  =N-blk consensus",0,&ctx.events[currevent++]);CHKERRQ(ierr); /* 3 */
    ierr = PetscLogEventRegister("  =Part. Send/BSP", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 4 */
    ierr = PetscLogEventRegister(" -Move parts", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 5 */
    ierr = PetscLogEventRegister(" -AddSource", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 6 */
    ierr = PetscLogEventRegister(" -Pre Push", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 7 */
    ierr = PetscLogEventRegister(" -Push (Jet)", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 8 */
    ierr = PetscLogEventRegister(" -Point Locate", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 9 */
    ierr = PetscLogEventRegister(" *create Particles", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 10 */
    ierr = PetscLogEventRegister("X2Poisson Solve", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 11 */
    ierr = PetscLogEventRegister("X2Part AXPY", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 12 */
    ierr = PetscLogEventRegister("X2Compress array", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 13 */
    ierr = PetscLogEventRegister("X2Diagnostics", 0, &ctx.events[diag_event_id]);CHKERRQ(ierr); /* N-1 */
    assert(sizeof(s_events)/sizeof(s_events[0]) > currevent);
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

  /* create SNESS */
  ierr = SNESCreate( ctx.wComm, &dmpi->snes);CHKERRQ(ierr);
  ierr = SNESSetDM( dmpi->snes, dmpi->dm);CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dmpi->dm,&ctx,&ctx,&ctx);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dmpi->dm, &J);CHKERRQ(ierr);
  if (ctx.dtype == X2_PERIODIC) {
    MatNullSpace nullsp;
    ierr = MatNullSpaceCreate(ctx.wComm, PETSC_TRUE, 0, NULL, &nullsp);CHKERRQ(ierr);
    ierr = MatSetNullSpace(J, nullsp);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
  }
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
    ctx.sendListTable[idx].data_size = 0; /* init */
    ctx.sendListTable[idx].size = 0;
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
      if (!isp) {
        sprintf(fname1,         "particles_electrons_time%05d_fluxtube.h5part",0);
        sprintf(fname2,"sub_rank_particles_electrons_time%05d_fluxtube.h5part",0);
      } else {
        sprintf(fname1,         "particles_sp%d_time%05d_fluxtube.h5part",(int)isp,0);
        sprintf(fname2,"sub_rank_particles_sp%d_time%05d_fluxtube.h5part",(int)isp,0);
      }
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

  /* setup solver, dummy solve to really setup */
  {
    ierr = VecZeroEntries(dmpi->rho);CHKERRQ(ierr); /* zero density to make solver do nothing */
    ierr = DMPICellSolve( ctx.dm );CHKERRQ(ierr);
  }

  /* move back to solver space and make density vector */
  ierr = processParticles(&ctx, 0.0, &ctx.sendListTable, 99, -1, -1, PETSC_TRUE);CHKERRQ(ierr);

  if (ctx.use_mms) {
    X2Ctx             *ctxArray[1];
    PetscErrorCode    (**exactFuncs)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);
    PetscViewer       viewer = NULL;
    PetscBool         flg;
    PetscViewerFormat fmt;
    PetscReal         norm,norm1,norm2;
    /* put exact function in dmpi->phi */
    ierr = PetscMalloc(sizeof(PetscErrorCode (*)(PetscInt, PetscReal,const PetscReal[],PetscInt,PetscScalar*,void*)),&exactFuncs);
    CHKERRQ(ierr);
    ctxArray[0] = &ctx;
    if (ctx.dtype == X2_PERIODIC) exactFuncs[0] = u_sinz_op;
    else exactFuncs[0] = u_x4_op;
    ierr = DMProjectFunction(dmpi->dm, 0.0, exactFuncs, (void **)ctxArray, INSERT_ALL_VALUES, dmpi->phi);CHKERRQ(ierr);
    ierr = PetscFree(exactFuncs);CHKERRQ(ierr);
    ierr = DMViewFromOptions(dmpi->dm,NULL,"-dm_view");CHKERRQ(ierr);
    ierr = PetscOptionsGetViewer(ctx.wComm,NULL,"-x2_vec_view",&viewer,&fmt,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerPushFormat(viewer,fmt);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) dmpi->phi,"exact-RHS");CHKERRQ(ierr);
      ierr = VecView(dmpi->phi,viewer);CHKERRQ(ierr);
      ierr = VecNorm(dmpi->phi,NORM_INFINITY,&norm2);CHKERRQ(ierr);
      /* ierr = PetscObjectSetName((PetscObject) dmpi->phi,"phi");CHKERRQ(ierr); */
      ierr = PetscObjectSetName((PetscObject) dmpi->rho,"pic-RHS");CHKERRQ(ierr);
      ierr = VecView(dmpi->rho,viewer);CHKERRQ(ierr);
      ierr = VecNorm(dmpi->rho,NORM_1,&norm1);CHKERRQ(ierr);
      /* ierr = PetscObjectSetName((PetscObject) dmpi->rho,"rho");CHKERRQ(ierr); */
      ierr = VecAXPY(dmpi->rho,-1.,dmpi->phi);CHKERRQ(ierr); /* error */
      ierr = PetscObjectSetName((PetscObject) dmpi->rho,"error-RHS");CHKERRQ(ierr);
      ierr = VecView(dmpi->rho,viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      ierr = VecNorm(dmpi->rho,NORM_INFINITY,&norm);CHKERRQ(ierr);
      PetscPrintf(ctx.wComm,"\tDeposition error |exact rho - rho|_inf/|rho|_inf = %g, |rho|_1 = %g\n",norm/norm2,norm1);
    }
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ctx.use_mms = PETSC_FALSE; /* just do this once */
    ctx.plot = PETSC_FALSE; /* don't bother printing */
  }

#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(ctx.events[0],0,0,0,0);CHKERRQ(ierr);
#endif

  /* do it */
  ierr = go( &ctx );CHKERRQ(ierr);

  if (dmpi->debug>0) PetscPrintf(ctx.wComm,"[%D] done - cleanup\n",ctx.rank);
#if defined(PETSC_HAVE_MEMALIGN) && (PETSC_MEMALIGN==64)
  PetscPrintf(ctx.wComm,"[%D] defined(PETSC_HAVE_MEMALIGN) && (PETSC_MEMALIGN==64)\n",ctx.rank);
#endif
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
  ierr = SNESDestroy(&dmpi->snes);CHKERRQ(ierr);
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

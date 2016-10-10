/* M. Adams, April 2015 */

static char help[] = "X2: A partical in cell code for tokamak plasmas using PICell.\n";

#ifdef H5PART
#include <H5Part.h>
#endif
#include <petsc/private/dmpicellimpl.h>    /*I   "petscdmpicell.h"   I*/
#include <assert.h>
#include <petscds.h>
#include <petscdmforest.h>

/* coordinate transformation - simple radial coordinates. Not really cylindrical as r_Minor is radius from plane axis */
#define cylindricalToPolPlane(__rMinor,__Z,__psi,__theta) { \
    __psi = sqrt((__rMinor)*(__rMinor) + (__Z)*(__Z));	    \
  if (__psi==0.) __theta = 0.; \
  else { \
    __theta = (__Z) > 0. ? asin((__Z)/__psi) : -asin(-(__Z)/__psi);	\
    if ((__rMinor) < 0) __theta = M_PI - __theta;			\
    else if (__theta < 0.) __theta = __theta + 2.*M_PI; \
  } \
}

/* q: safty factor */
#define qsafty(psi) (3.*pow(psi,2.0))

/* b0 dot jj at (psi,theta,phi) -- todo */
#define getB0DotX(__R,__psi,__theta,__phi,__jj,__b0dotx)                  \
{                                                                     \
  __b0dotx = -(__R)*sin(__phi)*qsafty(__psi)*__jj[0]*1.e-10 + (__R)*cos(__phi)*qsafty(__psi)*1.e-10; \
}

#define polPlaneToCylindrical( __psi, __theta, __rMinor, __Z) \
{\
  __rMinor = (__psi)*cos(__theta);		\
  __Z = (__psi)*sin(__theta);			\
}

#define cylindricalToCart( __R,  __Z,  __phi, __cart) \
{ \
 __cart[0] = (__R)*cos(__phi);			\
 __cart[1] = (__R)*sin(__phi);			\
 __cart[2] = __Z;				\
}

#include "x2_particle_array.h"
#include "x2_physics.h"

typedef enum {X2_ITER,X2_TORUS,X2_BOXTORUS} runType;
typedef struct {
  /* particle grid sizes */
  PetscInt np_radius;
  PetscInt np_theta;
  PetscInt np_phi; /* toroidal direction */
  /* tokamak geometry  */
  PetscReal  radius_major;
  PetscReal  radius_minor;
  PetscInt   num_phi_cells; /* number of cells per major circle in the torus */
  PetscReal  inner_mult; /* (0,1) percent of the total radius taken by the inner square */
  PetscReal  section_phi; /* *PI = size of section around torus (0,2] */
} X2Grid;

#include "x2_ctx.h"

#define X2_WALL_ARRAY_MAX 68 /* ITER file is 67 */
/* static vars for callbacks without a context */
static float s_wallVtx[X2_WALL_ARRAY_MAX][2];
static int s_numWallPtx;
static int s_numQuads;
static int s_quad_vertex[X2_WALL_ARRAY_MAX][9];
static PetscInt s_debug;
static PetscInt s_rank;
static int s_fluxtubeelem=5000;
static PetscReal s_rminor_inflate = 1.7;
static PetscReal s_section_phi;

/* X2GridSolverLocatePoints: find processor and element in solver grid that this point is in
    Input:
     - dm: solver dm
     - xvec: Cylindrical coordinates (native data), transformed to Cartesian!!!
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
PetscErrorCode X2GridSolverLocatePoints(DM dm, Vec xvec, const void *ctx_dum, IS *pes, IS *elemIDs)
{
  PetscErrorCode ierr;
  PetscBool      isForest;
  PetscMPIInt    npe,rank;
  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(xvec, VEC_CLASSID, 2);
  PetscValidPointer(pes, 4);
  PetscValidPointer(elemIDs, 5);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &npe);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(s_events[9],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = DMIsForest(dm,&isForest);CHKERRQ(ierr);
  if (isForest) {
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
    SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_USER, "P4est does not supporting locate\n");
  } else {
    PetscSF        cellSF = NULL;
    const PetscSFNode *foundCells;
    PetscInt       dim,n,nn,ii,jj;
    PetscScalar *xx,*xx0;
    PetscInt *peidxs,*elemidxs;
    ierr = DMGetCoordinateDim(dm, &dim);CHKERRQ(ierr);
    ierr = VecGetLocalSize(xvec,&n);CHKERRQ(ierr);
    if (n%dim) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER, "n%dim n=%D",n);
    nn = n/dim;
    ierr = VecGetArray(xvec,&xx0);CHKERRQ(ierr); xx = xx0;
    for (ii=0;ii<nn;ii++,xx+=dim) {
      PetscScalar x[3];
      cylindricalToCart(xx[0], xx[1], xx[2], x); /* get into Cartesion coords */
      for (jj=0;jj<dim;jj++) xx[jj] = x[jj];
    }
    ierr = VecRestoreArray(xvec,&xx0);CHKERRQ(ierr);
    ierr = DMLocatePoints(dm, xvec, DM_POINTLOCATION_NONE, &cellSF);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(cellSF, NULL, NULL, NULL, &foundCells);CHKERRQ(ierr);
    ierr = PetscMalloc2(nn,&peidxs,nn,&elemidxs);CHKERRQ(ierr);
    for (ii=0;ii<nn;ii++) {
      elemidxs[ii] = foundCells[ii].index;
      if (elemidxs[ii] == -1 && npe==1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "We are not supporting out of domain points.");
      else if (elemidxs[ii] < 0) elemidxs[ii] = 0; /* not working in parallel */
      /* peidxs[ii] = foundCells[ii].rank; */
      peidxs[ii] = rank; /* dummy - no move until have global search */
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,nn,peidxs,PETSC_COPY_VALUES,pes);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,nn,elemidxs,PETSC_COPY_VALUES,elemIDs);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&cellSF);CHKERRQ(ierr);
    ierr = PetscFree2(peidxs,elemidxs);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(s_events[9],0,0,0,0);CHKERRQ(ierr);
#endif
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
  PetscErrorCode ierr,isp,k,sz;
  FILE           *fp;
  PetscBool      phiFlag,radFlag,thetaFlag,flg,chunkFlag,secFlg,rMajFlg;
  char           str[256],str2[256],fname[256];
  PetscReal t1;
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
  ctx->grid.radius_major = 5.; /* 6.2 m of ITER */
  ctx->grid.radius_minor = 1.; /* 2.0 m of ITER */
  ctx->grid.section_phi  = 2.0; /* 2 pi, whole torus */
  ctx->grid.np_phi  = 1;
  ctx->grid.np_radius = 1;
  ctx->grid.np_theta  = 1;
  ctx->grid.num_phi_cells = 4; /* number of poloidal planes (before refinement) */
  ctx->grid.inner_mult= M_SQRT2 - 1.;

  ctx->tablecount = 0;

  ierr = PetscOptionsBegin(ctx->wComm, "", "Poisson Problem Options", "X2");CHKERRQ(ierr);
  /* general options */
  s_debug = 0;
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex1.c", s_debug, &s_debug, NULL);CHKERRQ(ierr);
  ctx->plot = PETSC_TRUE;
  ierr = PetscOptionsBool("-plot", "Write plot files (particles)", "ex1.c", ctx->plot, &ctx->plot, NULL);CHKERRQ(ierr);
  ctx->chunksize = X2_V_LEN; /* too small */
  ierr = PetscOptionsInt("-chunksize", "Size of particle list to chunk sends", "ex1.c", ctx->chunksize, &ctx->chunksize,&chunkFlag);CHKERRQ(ierr);
  if (chunkFlag) ctx->chunksize = X2_V_LEN*(ctx->chunksize/X2_V_LEN);
  if (ctx->chunksize<=0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER," invalid chuck size = %D",ctx->chunksize);
  ctx->use_bsp = 0;
  ierr = PetscOptionsInt("-use_bsp", "Size of chucks for PETSc's TwoSide communication (0 to use 'nonblocking consensus')", "ex1.c", ctx->use_bsp, &ctx->use_bsp, NULL);CHKERRQ(ierr);
  if (ctx->use_bsp<0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER," invalid BSP chuck size = %D",ctx->use_bsp);
  ctx->proc_send_table_size = ((ctx->npe>100) ? 100 + (ctx->npe-100)/10 : ctx->npe) + 1; /* hash table size of processors to send to */
  ierr = PetscOptionsInt("-proc_send_table_size", "Size of hash table proc->send_list", "ex1.c",ctx->proc_send_table_size, &ctx->proc_send_table_size, NULL);CHKERRQ(ierr);

  /* Domain and mesh definition */
  ierr = PetscOptionsReal("-radius_major", "Major radius of torus", "ex1.c", ctx->grid.radius_major, &ctx->grid.radius_major, &rMajFlg);CHKERRQ(ierr);
    /* section of the torus, adjust major radius as needed */
  ierr = PetscOptionsReal("-section_phi", "Number of pi radians of torus section, phi direction (0 < section_phi <= 2) ", "ex1.c",ctx->grid.section_phi,&ctx->grid.section_phi,&secFlg);CHKERRQ(ierr);
  if (ctx->grid.section_phi <= 0 || ctx->grid.section_phi > 2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"invalid -section_phi %g: (0,2]",ctx->grid.section_phi);
  t1 = 1.;
  ierr = PetscOptionsReal("-section_length", "Length if section (cylinder)", "ex1.c", t1, &t1,&flg);CHKERRQ(ierr);
  if (flg || secFlg) {
    if(secFlg && flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "-section_length with -section_phi not allowed, over constrained specification");
    else if(secFlg) {
      /* section of torus */
    } else {
      /* cylindrical flux tube, with L (section_length) specified */
      ctx->grid.section_phi = t1/(ctx->grid.radius_major*M_PI);
    }
  }
  s_section_phi = ctx->grid.section_phi;
  ierr = PetscOptionsReal("-radius_minor", "Minor radius of torus", "ex1.c", ctx->grid.radius_minor, &ctx->grid.radius_minor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_phi_cells", "Number of cells per major circle", "ex1.c", ctx->grid.num_phi_cells, &ctx->grid.num_phi_cells, NULL);CHKERRQ(ierr);
  if (ctx->grid.num_phi_cells<3 && ctx->grid.section_phi==2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER," Not enough phi cells %D",ctx->grid.num_phi_cells);
  ierr = PetscOptionsReal("-inner_mult", "Fraction of minor radius taken by inner square", "ex1.c", ctx->grid.inner_mult, &ctx->grid.inner_mult, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-np_phi", "Number of planes for particle mesh", "ex1.c", ctx->grid.np_phi, &ctx->grid.np_phi, &phiFlag);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-np_radius", "Number of radial cells for particle mesh", "ex1.c", ctx->grid.np_radius, &ctx->grid.np_radius, &radFlag);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-np_theta", "Number of theta cells for particle mesh", "ex1.c", ctx->grid.np_theta, &ctx->grid.np_theta, &thetaFlag);CHKERRQ(ierr);
  ctx->npe_particlePlane = -1;
  if (ctx->grid.np_phi*ctx->grid.np_radius*ctx->grid.np_theta != ctx->npe) { /* recover from inconsistant grid/procs */
    if (thetaFlag && radFlag && phiFlag) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"over constrained number of particle processes npe (%D) != %D",ctx->npe,ctx->grid.np_phi*ctx->grid.np_radius*ctx->grid.np_theta);

    if (!thetaFlag && radFlag && phiFlag) ctx->grid.np_theta = ctx->npe/(ctx->grid.np_phi*ctx->grid.np_radius);
    else if (thetaFlag && !radFlag && phiFlag) ctx->grid.np_radius = ctx->npe/(ctx->grid.np_phi*ctx->grid.np_theta);
    else if (thetaFlag && radFlag && !phiFlag) ctx->grid.np_phi = ctx->npe/(ctx->grid.np_radius*ctx->grid.np_theta);
    else if (!thetaFlag && !radFlag && !phiFlag) {
      ctx->npe_particlePlane = (int)pow((double)ctx->npe,0.6667);
      ctx->grid.np_phi = ctx->npe/ctx->npe_particlePlane;
      ctx->grid.np_radius = (int)(sqrt((double)ctx->npe_particlePlane)+0.5);
      ctx->grid.np_theta = ctx->npe_particlePlane/ctx->grid.np_radius;
      if (ctx->grid.np_phi*ctx->grid.np_radius*ctx->grid.np_theta != ctx->npe) {
	ctx->grid.np_phi = ctx->npe;
      }
    }
    else if (ctx->grid.np_phi*ctx->grid.np_radius*ctx->grid.np_theta != ctx->npe) { /* recover */
      if (!ctx->npe%ctx->grid.np_phi) {
	ctx->npe_particlePlane = ctx->npe/ctx->grid.np_phi;
	ctx->grid.np_radius = (int)(sqrt((double)ctx->npe_particlePlane)+0.5);
	ctx->grid.np_theta = ctx->npe_particlePlane/ctx->grid.np_radius;
      }
      else {
      }
    }
    if (ctx->grid.np_phi*ctx->grid.np_radius*ctx->grid.np_theta != ctx->npe) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"particle grids do not work npe (%D) != %D",ctx->npe,ctx->grid.np_phi*ctx->grid.np_radius*ctx->grid.np_theta);
  }

  /* particle grids: <= npe, <= num solver planes */
  if (ctx->npe < ctx->grid.np_phi) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"num particle planes np_phi (%D) > npe (%D)",ctx->grid.np_phi,ctx->npe);
  if (ctx->npe%ctx->grid.np_phi) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"np=%D not divisible by number of particle planes (np_phi) %D",ctx->npe,ctx->grid.np_phi);

  if (ctx->npe_particlePlane == -1) ctx->npe_particlePlane = ctx->npe/ctx->grid.np_phi;
  if (ctx->npe_particlePlane != ctx->npe/ctx->grid.np_phi) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Inconsistant number planes (%D), pes (%D), and pe/plane (%D) requested",ctx->grid.np_phi,ctx->npe,ctx->npe_particlePlane);

  if (ctx->grid.np_theta*ctx->grid.np_radius != ctx->npe_particlePlane) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"%D particle cells/plane != %D pe/plane",ctx->grid.np_theta*ctx->grid.np_radius,ctx->npe_particlePlane);
  if (ctx->grid.np_theta*ctx->grid.np_radius*ctx->grid.np_phi != ctx->npe) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"%D particle cells != %D npe",ctx->grid.np_theta*ctx->grid.np_radius*ctx->grid.np_phi,ctx->npe);
  ctx->ParticlePlaneIdx = ctx->rank/ctx->npe_particlePlane;
  ctx->particlePlaneRank = ctx->rank%ctx->npe_particlePlane;

  /* PetscPrintf(PETSC_COMM_SELF,"[%D] pe/plane=%D, my plane=%D, my local rank=%D, np_phi=%D\n",ctx->rank,ctx->npe_particlePlane,ctx->ParticlePlaneIdx,ctx->particlePlaneRank,ctx->grid.np_phi);    */

  /* time integrator */
  ctx->msteps = 1;
  ierr = PetscOptionsInt("-mstep", "Maximum number of time steps", "ex1.c", ctx->msteps, &ctx->msteps, NULL);CHKERRQ(ierr);
  ctx->maxTime = 1000000000.;
  ierr = PetscOptionsReal("-maxTime", "Maximum time", "ex1.c",ctx->maxTime,&ctx->maxTime,NULL);CHKERRQ(ierr);
  ctx->dt = 1.;
  ierr = PetscOptionsReal("-dt","Time step","ex1.c",ctx->dt,&ctx->dt,NULL);CHKERRQ(ierr);
  /* particles */
  ctx->num_particles_proc = 10;
  ierr = PetscOptionsInt("-num_particles_proc", "Number of particles local (flux tube cell)", "ex1.c", ctx->num_particles_proc, &ctx->num_particles_proc, NULL);CHKERRQ(ierr);
  if (ctx->num_particles_proc<0) {
    if (!ctx->rank) ctx->num_particles_proc = -ctx->num_particles_proc;
    else  ctx->num_particles_proc = 0;
  }
  if (!chunkFlag) ctx->chunksize = X2_V_LEN*((ctx->num_particles_proc/80+1)/X2_V_LEN + 1); /* an intelegent message chunk size */
  if (ctx->chunksize<64 && !chunkFlag) ctx->chunksize = 64; /* 4K messages minumum */

  if (ctx->npe>1) PetscPrintf(ctx->wComm,"[%D] **** Warning ****, no global point location. multiple processors (%D) not supported\n",ctx->rank,ctx->npe);

  ctx->collision_period = 10000;
  ierr = PetscOptionsInt("-collision_period", "Period between collision operators", "ex1.c", ctx->collision_period, &ctx->collision_period, NULL);CHKERRQ(ierr);
  ctx->use_electrons = PETSC_FALSE;
  ierr = PetscOptionsBool("-use_electrons", "Include electrons", "ex1.c", ctx->use_electrons, &ctx->use_electrons, NULL);CHKERRQ(ierr);
  ctx->max_vpar = 30.;
  ierr = PetscOptionsReal("-max_vpar", "Maximum parallel velocity", "ex1.c",ctx->max_vpar,&ctx->max_vpar,NULL);CHKERRQ(ierr);

  ierr = PetscStrcpy(fname,"iter");CHKERRQ(ierr);
  ierr = PetscOptionsString("-run_type", "Type of run (iter or torus)", "ex1.c", fname, fname, sizeof(fname)/sizeof(fname[0]), NULL);CHKERRQ(ierr);
  PetscStrcmp("iter",fname,&flg);
  if (flg) { /* ITER */
    if (ctx->grid.section_phi != 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"section of ITER not supported");
    ctx->run_type = X2_ITER;
    ierr = PetscStrcpy(fname,"ITER-51vertex-quad.txt");CHKERRQ(ierr);
    ierr = PetscOptionsString("-iter_vertex_file", "Name of vertex .txt file of ITER vertices", "ex1.c", fname, fname, sizeof(fname)/sizeof(fname[0]), NULL);CHKERRQ(ierr);
    fp = fopen(fname, "r");
    if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"ITER file %s not found, use -fname FILE_NAME",fname);
    if (!fgets(str,256,fp)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Error reading ITER file");
    k = sscanf(str,"%d\n",&sz);
    if (k<1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Error reading ITER file %d words",k);
    if (sz>X2_WALL_ARRAY_MAX) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Too Many vertices %d > %d",sz,X2_WALL_ARRAY_MAX);
    for (isp=0;isp<sz;isp++) {
      if (!fgets(str,256,fp)) break;
      k = sscanf(str,"%e %e %s\n",&s_wallVtx[isp][0],&s_wallVtx[isp][1],str2);
      if (k<2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Error reading ITER file %d words",k);
      s_wallVtx[isp][0] -= ctx->grid.radius_major;
    }
    s_numWallPtx = isp;
    if (s_numWallPtx!=sz) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Error reading ITER file, %d lines",s_numWallPtx);
    /* cell ids */
    for (isp=0;isp<1000;isp++) {
      if (!fgets(str,256,fp)) break;
      k = sscanf(str,"%d %d %d %d %d %d %d %d %d %s\n",
                 &s_quad_vertex[isp][0],&s_quad_vertex[isp][1],&s_quad_vertex[isp][2],
                 &s_quad_vertex[isp][3],&s_quad_vertex[isp][4],&s_quad_vertex[isp][5],
                 &s_quad_vertex[isp][6],&s_quad_vertex[isp][7],&s_quad_vertex[isp][8],
                 str2);
      if (k==-1) break;
      if (k<9) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Error reading ITER file, read %d terms != 9",k);
      if (isp>X2_WALL_ARRAY_MAX) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Too Many Elements %d",X2_WALL_ARRAY_MAX);
    }
    fclose(fp);
    s_numQuads = isp;
    if (s_debug>0) PetscPrintf(PETSC_COMM_WORLD,"ProcessOptions:  numQuads=%d, numWallPtx=%d\n",s_numQuads,s_numWallPtx);
    s_rminor_inflate = 1; /* not used for ITER */
  }
  else {
    char      convType[256];
    PetscReal ntheta_total = 4, radius, nmajor_total;
    PetscInt  idx;
    /* hack to get grid expansion factor for torus */
    ierr = PetscOptionsFList("-x2_dm_type","Convert DMPlex to another format","ex1.c",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    nmajor_total = ctx->grid.num_phi_cells;
    if (flg) {
      ierr = PetscOptionsGetInt(NULL,"x2_","-dm_forest_initial_refinement", &idx, &flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "-x2_dm_forest_initial_refinement not found?");
      nmajor_total *= pow(2,idx);  /* plex does not get the curve */
    }
    else {
      ierr = PetscOptionsGetInt(NULL,"x2_","-dm_refinement", &idx, &flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "-x2_dm_refinement not found?");
    }
    if (idx<1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "refine must be greater than 0");
    ntheta_total *= pow(2,idx);
    /* inflate for corners in plane */
    radius = ctx->grid.radius_minor;
    s_rminor_inflate = 1.01/cos(M_PI/ntheta_total);
    /* inflate for corners around torus */
    radius = ctx->grid.radius_minor + ctx->grid.radius_major;
    /* s_rminor_inflate *= 1.01/cos(.5*ctx->grid.section_phi*M_PI/nmajor_total); */
    s_rminor_inflate *= 1.01*((radius) / cos(.5*ctx->grid.section_phi*M_PI/nmajor_total) - ctx->grid.radius_major) / ctx->grid.radius_minor;
    if (ntheta_total < 10) s_rminor_inflate *= 2.; /* hack to fix, this is not working for on level of refinement */
    if (ctx->grid.radius_minor*s_rminor_inflate >= ctx->grid.radius_major) {
      s_rminor_inflate = 0.95*ctx->grid.radius_major/ctx->grid.radius_minor;
    }

    PetscStrcmp("torus",fname,&flg);
    if (flg) ctx->run_type = X2_TORUS;
    else {
      PetscStrcmp("boxtorus",fname,&flg);
      if (flg) ctx->run_type = X2_BOXTORUS;
      else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Unknown run type %s",fname);
    }
  }
  ierr = PetscOptionsEnd();

  if (s_debug>0) PetscPrintf(ctx->wComm,"[%D] npe=%D; %D x %D x %D flux tube grid; mpi_send size (chunksize) has %d particles. %s. %s, %s, phi section = %g, inflate = %g\n",ctx->rank,ctx->npe,ctx->grid.np_phi,ctx->grid.np_theta,ctx->grid.np_radius,ctx->chunksize,
#ifdef X2_S_OF_V
			     "Use struct of arrays"
#else
			     "Use of array structs"
#endif
                             , ctx->use_electrons ? "use electrons" : "ions only", ctx->use_bsp ? "BSP communication" : "Non-blocking consensus communication",ctx->grid.section_phi*M_PI,s_rminor_inflate);

  PetscFunctionReturn(0);
}

/* X2GridFluxTubeLocatePoint: find processor and local flux tube that this point is in
    Input:
     - grid: the particle grid
     - psi: r in [r,theta] coordinates (poliodal coordinates)
     - theta:
     - phi: toroidal angle
   Output:
    - pe: process ID
    - elem: element ID
*/
#undef __FUNCT__
#define __FUNCT__ "X2GridFluxTubeLocatePoint"
PetscErrorCode X2GridFluxTubeLocatePoint( const X2Grid *grid, PetscReal x[3],
                                          PetscMPIInt *pe, PetscInt *elem)
{
  const PetscReal rminor = grid->radius_minor;
  const PetscReal dphi   = s_section_phi*M_PI/(PetscReal)grid->np_phi;
  const PetscReal dth    = 2.*M_PI/(PetscReal)grid->np_theta;
  PetscReal psi = x[0], theta = x[1], phi = x[2];
  PetscMPIInt planeIdx,irs,iths;
  PetscFunctionBeginUser;
/* #if defined(PETSC_USE_LOG) */
/*   ierr = PetscLogEventBegin(s_events[10],0,0,0,0);CHKERRQ(ierr); */
/* #endif */
  theta = fmod(theta - qsafty(psi/rminor)*phi + 20.*M_PI, 2.*M_PI);  /* pull back to reference grid */
  planeIdx = (PetscMPIInt)(phi/dphi)*grid->np_radius*grid->np_theta; /* assuming one particle cell per PE */
  iths = (PetscMPIInt)(theta/dth);                               assert(iths<grid->np_theta);
  irs = (PetscMPIInt)((PetscReal)grid->np_radius*psi*psi/(rminor*rminor)); assert(irs<grid->np_radius);
  *pe = planeIdx + irs*grid->np_theta + iths;
  *elem = s_fluxtubeelem; /* only one cell per process */
/* #if defined(PETSC_USE_LOG) */
/*     ierr = PetscLogEventEnd(s_events[10],0,0,0,0);CHKERRQ(ierr); */
/* #endif */
  PetscFunctionReturn(0);
}

#ifdef H5PART
/* add corners to get bounding box */
static void prewrite(X2Ctx *ctx, X2PList *l, X2PListPos *ppos1,  X2PListPos *ppos2)
{
  if (ctx->rank==0) {
    X2Particle part;
    PetscReal r,z,phi;
    PetscErrorCode ierr;
    z = ctx->grid.radius_minor;
    if (ctx->grid.section_phi == 2) {
      r = 1.414213562373095*(ctx->grid.radius_major + ctx->grid.radius_minor);
      phi = M_PI/4.;
    } else {
      r = ctx->grid.radius_major + ctx->grid.radius_minor;
      phi = 0.;
    }
    X2ParticleCreate(&part,1,r,z,phi,0.);
    ierr = X2PListAdd(l,&part,ppos1); assert(!ierr);
    z = -z;
    if (ctx->grid.section_phi == 2) {
      phi += M_PI;
    } else {
      r = ctx->grid.radius_major - ctx->grid.radius_minor;
      phi = ctx->grid.section_phi*M_PI;
    }
    X2ParticleCreate(&part,2,r,z,phi,0.);
    ierr = X2PListAdd(l,&part,ppos2); assert(!ierr);
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
     - irk: RK stage (<0 for send only)
     - solver: use solver partitioning to get processor of point?
   Input/Output:
     - ctx: global data
     - lists: list of particle lists
   Output:
     - sendListTable: send list hash table, null if not sending (irk==0)
*/
#define X2PROCLISTSIZE 256
#undef __FUNCT__
#define __FUNCT__ "processParticles"
static PetscErrorCode processParticles( X2Ctx *ctx, const PetscReal dt, X2PSendList **sendListTable_in, const PetscMPIInt tag,
					const int irk, const int istep, PetscBool solver)
{
  X2Grid *grid = &ctx->grid;
  X2PSendList *sendListTable = *sendListTable_in;
  DM_PICell *dmpi = (DM_PICell *) ctx->dm->data;     assert(solver || irk<0); /* don't push flux tubes */
  PetscReal   psi,theta,dphi,rmaj=grid->radius_major,rminor=grid->radius_minor;
  PetscMPIInt pe,hash,ii;
  X2Particle  part;
  X2PListPos  pos;
  PetscErrorCode ierr;
  const int part_dsize = sizeof(X2Particle)/sizeof(double);
  Vec          jVec,xVec,vVec;
  PetscScalar *xx=0,*jj=0,*vv=0,*xx0=0,*jj0=0,*vv0=0;
  PetscInt isp,nslist,nlistsTot,elid,elid2,one=1,three=3,ndeposit;
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
  if (!dmpi) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"DM_PICell data not created");
  if (solver) {
    ierr = VecZeroEntries(dmpi->rho);CHKERRQ(ierr); /* zero density to get ready for next deposition */
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
      if (solver) {
        ierr = X2PListCompress(list);CHKERRQ(ierr); /* allows for simpler vectorization */
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventBegin(ctx->events[7],0,0,0,0);CHKERRQ(ierr); /* timer on particle list */
#endif
        /* make vectors for this element */
        ierr = VecCreateSeq(PETSC_COMM_SELF,three*list->vec_top, &xVec);CHKERRQ(ierr);
        ierr = VecSetBlockSize(xVec,three);CHKERRQ(ierr);
        /* make coordinates array to get gradients */
        ierr = VecGetArray(xVec,&xx0);CHKERRQ(ierr); xx = xx0;
#pragma simd vectorlengthfor(PetscScalar)
	for (pos=0 ; pos < list->vec_top ; pos++, xx += 3) {
#ifdef X2_S_OF_V
	  PetscReal r=list->data_v.r[pos], z=list->data_v.z[pos], phi=list->data_v.phi[pos];
#else
          PetscReal r=list->data[pos].r, z=list->data[pos].z, phi=list->data[pos].phi;
#endif
	  cylindricalToCart(r, z, phi, xx);
        }
        ierr = VecRestoreArray(xVec,&xx0);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventEnd(ctx->events[7],0,0,0,0);CHKERRQ(ierr);
#endif
      }
      if (solver) {
        /* push, and collect x */
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventBegin(ctx->events[8],0,0,0,0);CHKERRQ(ierr); /* timer on particle list */
#endif
        /* get E, should set size of vecs for true size? */
        if (irk>=0) {
          Vec locphi;
          ierr = DMGetLocalVector(dmpi->dmplex, &locphi);CHKERRQ(ierr);
          ierr = DMGlobalToLocalBegin(dmpi->dmplex, dmpi->phi, INSERT_VALUES, locphi);CHKERRQ(ierr);
          ierr = DMGlobalToLocalEnd(dmpi->dmplex, dmpi->phi, INSERT_VALUES, locphi);CHKERRQ(ierr);
          /* get E, should set size of vecs for true size? */
          ierr = VecCreateSeq(PETSC_COMM_SELF,three*list->vec_top,&jVec);CHKERRQ(ierr);
          ierr = DMPICellGetJet(ctx->dm, xVec, locphi, elid, jVec);CHKERRQ(ierr);
          ierr = DMRestoreLocalVector(dmpi->dmplex, &locphi);CHKERRQ(ierr);
          ierr = VecGetArray(jVec,&jj0);CHKERRQ(ierr); jj = jj0;
        }
        /* vectorize (todo) push: theta = theta + q*dphi .... grad not used */
        ierr = VecGetArray(xVec,&xx0);CHKERRQ(ierr); xx = xx0;
        for (pos=0 ; pos < list->vec_top ; pos++, xx += 3, jj += 3 ) {
	  /* push particle, real data, could do it on copy for non-final stage of TS, copy new coordinate to xx */
          if (irk>=0) {
            PetscReal b0dotgrad;
#ifdef X2_S_OF_V
            PetscReal r=list->data_v.r[pos] - rmaj, z=list->data_v.z[pos];
            cylindricalToPolPlane(r, z, psi, theta );
            getB0DotX( list->data_v.r[pos], psi, theta, list->data_v.phi[pos], jj, b0dotgrad );
            list->data_v.vpar[pos] += -dt*b0dotgrad*charge/mass;
            dphi = (dt*list->data_v.vpar[pos])/list->data_v.r[pos];  /* toroidal step */
            list->data_v.phi[pos] += dphi;
            xx[2] = list->data_v.phi[pos] = fmod(list->data_v.phi[pos] + 100.*s_section_phi*M_PI,s_section_phi*M_PI);
            theta += qsafty(psi/rminor)*dphi;  /* twist */
            theta = fmod( theta + 20.*M_PI, 2.*M_PI);
            polPlaneToCylindrical( psi, theta, r, z); /* time spent here */
            xx[0] = list->data_v.r[pos] = rmaj + r;
            xx[1] = list->data_v.z[pos] = z;
#else
            X2Particle *ppart = &list->data[pos];
            PetscReal r = ppart->r - rmaj, z = ppart->z;
            cylindricalToPolPlane( r, z, psi, theta );
            getB0DotX( psi, theta, ppart->phi, jj, b0 );
            ppart->vpar += -dt*b0dotgrad*charge/mass;
            dphi = (dt*ppart->vpar)/ppart->r;  /* toroidal step */
            ppart->phi += dphi;
            xx[2] = ppart->phi = fmod(ppart->phi + 100.*s_section_phi*M_PI, s_section_phi*M_PI);
            theta += qsafty(psi/rminor)*dphi;  /* twist */
            theta = fmod( theta + 20.*M_PI, 2.*M_PI);
            polPlaneToCylindrical( psi, theta, r, z); /* time spent here */
            xx[0] = ppart->r = rmaj + r;
            xx[1] = ppart->z = z;
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
          ierr = VecRestoreArray(jVec,&jj0);
          ierr = VecDestroy(&jVec);CHKERRQ(ierr);
        }
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventEnd(ctx->events[8],0,0,0,0);CHKERRQ(ierr);
#endif
      } /* if solver */
      /* move */
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventBegin(ctx->events[5],0,0,0,0);CHKERRQ(ierr);
#endif
      /* get pe & element id */
      if (solver) {
        /* see if need communication? no: add density, yes: add to communication list */
        ierr = X2GridSolverLocatePoints(dmpi->dmplex, xVec, ctx, &pes, &elems);CHKERRQ(ierr);
      } else {
        PetscReal r, x[3];
        PetscInt idx;
        ierr = PetscMalloc2(list->vec_top,&peidxs,list->vec_top,&elemidxs);CHKERRQ(ierr);
        for (pos=0 ; pos < list->vec_top ; pos++, xx += 3, jj += 3 ) {
#ifdef X2_S_OF_V
	  X2V2P((&part),list->data_v,pos); /* return copy */
#else
	  part = list->data[pos]; /* return copy */
#endif
          r = part.r - rmaj;
          cylindricalToPolPlane( r, part.z, x[0], x[1]);
          x[2] = part.phi;
          ierr = X2GridFluxTubeLocatePoint(grid, x, &pe, &idx);CHKERRQ(ierr);
          peidxs[pos] = pe;
          elemidxs[pos] = idx;
        }
        ierr = ISCreateGeneral(PETSC_COMM_SELF,list->vec_top,peidxs,PETSC_COPY_VALUES,&pes);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PETSC_COMM_SELF,list->vec_top,elemidxs,PETSC_COPY_VALUES,&elems);CHKERRQ(ierr);
        ierr = PetscFree2(peidxs,elemidxs);CHKERRQ(ierr);
      }
      /* move particles - not vectorizable */
      ierr = ISGetIndices(pes,&cpeidxs);CHKERRQ(ierr);
      ierr = ISGetIndices(elems,&celemidxs);CHKERRQ(ierr);
      for (pos=0 ; pos < list->top ; pos++ ) {
        pe = cpeidxs[pos];
        elid2 = celemidxs[pos];
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
              PetscInt hash2 = (pe2*593)%(ctx->proc_send_table_size); /* new hash */
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
        /* move particles - not vectorizable */
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
            ierr = X2PSendListCreate(&sendListTable[hash],ctx->chunksize);CHKERRQ(ierr);
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
                SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"process send table too small (%D) == snlist(%D)",nslist,(PetscInt)X2PROCLISTSIZE);
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
      }
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventEnd(ctx->events[5],0,0,0,0);CHKERRQ(ierr);
#endif
    } /* element list */
    /* finish sends and receive new particles for this species */
    ierr = shiftParticles(ctx, sendListTable, ctx->partlists[isp], &nslist, X2PROCLISTSIZE, slist, tag+isp, solver );
    CHKERRQ(ierr);
#ifdef PETSC_USE_DEBUG
    { /* debug */
      PetscMPIInt flag,sz; MPI_Status  status; MPI_Datatype mtype;
      ierr = MPI_Iprobe(MPI_ANY_SOURCE, tag+isp, ctx->wComm, &flag, &status);CHKERRQ(ierr);
      if (flag) {
        PetscDataTypeToMPIDataType(PETSC_REAL,&mtype);
        MPI_Get_count(&status, mtype, &sz); assert(sz%part_dsize==0);
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"found %D extra particles from %d",sz/part_dsize,status.MPI_SOURCE);
      }
      MPI_Barrier(ctx->wComm);
    }
#endif
    nlistsTot += nslist;
    /* add density (while in cache, by species at least) */
    if (solver) {
      Vec locrho;
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
          PetscReal r=list->data_v.r[pos], z=list->data_v.z[pos], phi=list->data_v.phi[pos];
#else
          PetscReal r=list->data[pos].r, z=list->data[pos].z, phi=list->data[pos].phi;
#endif
          cylindricalToCart(r, z, phi, xx);
#ifdef X2_S_OF_V
          *vv = list->data_v.w0[pos]*ctx->species[isp].charge;
#else
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
                "%d) %s %D local particles, %D/%D global, %g %% total particles moved in %D messages total (to %D processors local), %g load imbalance factor\n",
                istep+1,irk<0 ? "processed" : "pushed", origNlocal, rb1[0], rb1[3], 100.*(double)rb1[1]/(double)rb1[0], rb1[2], ctx->tablecount,(double)rb2[3]/((double)rb1[3]/(double)ctx->npe));
    if (rb1[0] != rb1[3]) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"Number of partilces %D --> %D",rb1[0],rb1[3]);
#ifdef H5PART
    if (irk>=0 && ctx->plot) {
      for (isp=ctx->use_electrons ? 0 : 1 ; isp <= X2_NION ; isp++ ) {
        char  fname1[256],fname2[256];
        X2PListPos pos1,pos2;
        /* hdf5 output */
        if (!isp) {
          sprintf(fname1,         "x2_particles_electrons_time%05d.h5part",(int)istep+1);
          sprintf(fname2,"x2_sub_rank_particles_electrons_time%05d.h5part",(int)istep+1);
        } else {
          sprintf(fname1,         "x2_particles_sp%d_time%05d.h5part",(int)isp,(int)istep+1);
          sprintf(fname2,"x2_sub_rank_particles_sp%d_time%05d.h5part",(int)isp,(int)istep+1);
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
  PetscInt        isp,nCellsLoc,my0,irs,iths,gid,ii,np,dim,cStart,cEnd,elid;
  const PetscReal rmin = ctx->grid.radius_minor;
  const PetscReal dth  = 2*M_PI/(PetscReal)ctx->grid.np_theta;
  const PetscReal dphi = ctx->grid.section_phi*M_PI/(PetscReal)ctx->grid.np_phi; /* rmin for particles < rmin */
  const PetscReal phi1 = (PetscReal)ctx->ParticlePlaneIdx*dphi + 1.e-8,rmaj=ctx->grid.radius_major;
  const PetscInt  nPartProcss_plane = ctx->grid.np_theta*ctx->grid.np_radius; /* nPartProcss_plane == ctx->npe_particlePlane */
  const PetscReal dx = pow( (M_PI*rmin*rmin/4.0) * rmaj*ctx->grid.section_phi*M_PI / (PetscReal)(ctx->npe*ctx->num_particles_proc), 0.333); /* ~length of a particle */
  X2Particle particle;
  DM dm;
  DM_PICell *dmpi;
  PetscFunctionBeginUser;

  /* Create vector and get pointer to data space */
  dmpi = (DM_PICell *) ctx->dm->data;
  dm = dmpi->dmgrid;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim!=3) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"wrong dimension (3) = %D",dim);
  ierr = DMGetCellChart(dm, &cStart, &cEnd);CHKERRQ(ierr);
  ctx->nElems = PetscMax(1,cEnd-cStart);CHKERRQ(ierr);

  /* setup particles - lexicographic partition of -- flux tube -- cells */
  nCellsLoc = nPartProcss_plane/ctx->npe_particlePlane; /* = 1; nPartProcss_plane == ctx->npe_particlePlane */
  my0 = ctx->particlePlaneRank*nCellsLoc;              /* cell index in plane == particlePlaneRank */
  gid = (my0 + ctx->ParticlePlaneIdx*nPartProcss_plane)*ctx->num_particles_proc; /* based particle ID */
  if (ctx->ParticlePlaneIdx == ctx->npe_particlePlane-1){
    nCellsLoc = nPartProcss_plane - nCellsLoc*(ctx->npe_particlePlane-1);
  }
  assert(nCellsLoc==1);

  /* my first cell index */
  srand(ctx->rank);
  for (isp=ctx->use_electrons ? 0 : 1 ; isp <= X2_NION ; isp++ ) {
    iths = my0%ctx->grid.np_theta;
    irs = my0/ctx->grid.np_theta;
    ierr = PetscMalloc1(ctx->nElems,&ctx->partlists[isp]);CHKERRQ(ierr);
    {
      const PetscReal r1 = sqrt(((PetscReal)irs      /(PetscReal)ctx->grid.np_radius)*rmin*rmin) +       1.e-12*rmin;
      const PetscReal dr = sqrt((((PetscReal)irs+1.0)/(PetscReal)ctx->grid.np_radius)*rmin*rmin) - (r1 - 1.e-12*rmin);
      const PetscReal th1 = (PetscReal)iths*dth + 1.e-12*dth;
      const PetscReal maxe=ctx->max_vpar*ctx->max_vpar,mass=ctx->species[isp].mass,charge=ctx->species[isp].charge;
      /* create list for element 0 and add all to it */
      ierr = X2PListCreate(&ctx->partlists[isp][s_fluxtubeelem],ctx->chunksize);CHKERRQ(ierr);
      /* create each particle */
      for (np=0 ; np<ctx->num_particles_proc; /* void */ ) {
	PetscReal theta0,r,z;
	const PetscReal psi = r1 + (PetscReal)(rand()%X2NDIG+1)/(PetscReal)(X2NDIG+1)*dr;
	const PetscReal qsaf = qsafty(psi/ctx->grid.radius_minor);
	const PetscInt  NN = (PetscInt)(dth*psi/dx) + 1;
	const PetscReal dth2 = dth/(PetscReal)NN - 1.e-12*dth;
	for ( ii = 0, theta0 = th1 + (PetscReal)(rand()%X2NDIG)/(PetscReal)X2NDIG*dth2;
	      ii < NN && np<ctx->num_particles_proc;
	      ii++, theta0 += dth2, np++ ) {
	  PetscReal       zmax,zdum,v,vpar;
          const PetscReal phi = phi1 + (PetscReal)(rand()%X2NDIG)/(PetscReal)X2NDIG*dphi;
	  const PetscReal thetap = theta0 + qsaf*phi; /* push forward to follow field-lines */
	  polPlaneToCylindrical(psi, thetap, r, z);
	  r += rmaj;
	  /* v_parallel from random number */
	  zmax = 1.0 - exp(-maxe);
	  zdum = zmax*(PetscReal)(rand()%X2NDIG)/(PetscReal)X2NDIG;
	  v= sqrt(-2.0/mass*log(1.0-zdum));
	  v= v*cos(M_PI*(PetscReal)(rand()%X2NDIG)/(PetscReal)X2NDIG);
	  /* vshift= v + up ! shift of velocity */
	  vpar = v*mass/charge;
          ierr = X2ParticleCreate(&particle,++gid,r,z,phi,vpar);CHKERRQ(ierr); /* only time this is called! */
	  ierr = X2PListAdd(&ctx->partlists[isp][s_fluxtubeelem],&particle, NULL);CHKERRQ(ierr);
          /* debug, particles are created in a flux tube */
#ifdef PETSC_USE_DEBUG
          {
            PetscMPIInt pe; PetscInt id;
            PetscReal xx[] = {psi,thetap,phi};
            ierr = X2GridFluxTubeLocatePoint(&ctx->grid,xx,&pe,&id);CHKERRQ(ierr);
            if(pe != ctx->rank){
              PetscPrintf(PETSC_COMM_SELF,"[%D] ERROR particle in proc %d r=%e:%e:%e theta=%e:%e:%e phi=%e:%e:%e\n",ctx->rank,pe,r1,psi,r1+dr,th1,thetap,th1+dth,phi1,phi,phi1+dphi);
              SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER," created particle for proc %D",pe);
            }
          }
#endif
	} /* theta */
      }
      iths++;
      if (iths==ctx->grid.np_theta) { iths = 0; irs++; }
    } /* cells */
    /* finish off list creates for rest of elements */
    for (elid=0;elid<ctx->nElems;elid++) {
      if (elid!=s_fluxtubeelem) //
        ierr = X2PListCreate(&ctx->partlists[isp][elid],ctx->chunksize);CHKERRQ(ierr); /* this will get expanded, chunksize used for message chunk size and initial list size! */
    }
  } /* species */
  PetscFunctionReturn(0);
}

/* == Defining a base plex for ITER, which looks like a rectilinear (sort of) donut */

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreatePICellITER"
static PetscErrorCode DMPlexCreatePICellITER(MPI_Comm comm, X2Grid *params, DM *dm)
{
  PetscMPIInt    rank;
  PetscInt       numCells = 0;
  PetscInt       numVerts = 0;
  PetscReal      radius_major   = params->radius_major;
  PetscInt       num_phi_cells = params->num_phi_cells;
  int           *flatCells = NULL;
  double        *flatCoords = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    const int numQuads = s_numQuads;
    const int numQuadVtx = 2*numQuads + 2; /* kind of a hack to get the number of vertices in the quads */
    numCells = num_phi_cells * numQuads;
    numVerts = num_phi_cells * numQuadVtx;
    ierr = PetscMalloc2(numCells * 8,&flatCells,numVerts * 3,&flatCoords);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"DMPlexCreatePICellITER: numCells=%d\n",numCells);
    {
      double (*coords)[numQuadVtx][3] = (double (*) [numQuadVtx][3]) flatCoords;
      PetscInt i;

      for (i = 0; i < num_phi_cells; i++) {
        PetscInt j;
        double cosphi, sinphi;

        cosphi = cos(2 * M_PI * i / num_phi_cells);
        sinphi = sin(2 * M_PI * i / num_phi_cells);

        for (j = 0; j < numQuadVtx; j++) {
          double r, z;

          r = radius_major + s_wallVtx[j][0];
          z =          s_wallVtx[j][1];

          coords[i][j][0] = cosphi * r;
          coords[i][j][1] = sinphi * r;
          coords[i][j][2] = z;
        }
      }
    }
    {
      int (*cells)[numQuads][8] = (int (*) [numQuads][8]) flatCells;
      PetscInt i;

      for (i = 0; i < num_phi_cells; i++) {
        PetscInt j;

        for (j = 0; j < numQuads; j++) {
          PetscInt k;
          for (k = 0; k < 8; k++) {
            PetscInt l = k % 4, off = k/4;
            if (i==num_phi_cells-1 && off) off = 1-num_phi_cells;
            cells[i][j][k] = i*numQuadVtx + off*numQuadVtx + (s_quad_vertex[j][l]-1);
          }
          {
            PetscInt swap = cells[i][j][1];
            cells[i][j][1] = cells[i][j][3];
            cells[i][j][3] = swap;
          }
        }
      }
    }
  }
  ierr = DMPlexCreateFromCellList(comm,3,numCells,numVerts,8,PETSC_TRUE,flatCells,3,flatCoords,dm);CHKERRQ(ierr);
  ierr = PetscFree2(flatCells,flatCoords);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "ITER");CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
/* PetscErrorCode(*)(DM,PetscInt,PetscInt,const PetscReal[],PetscReal[],void*) */
#undef __FUNCT__
#define __FUNCT__ "GeometryPICellITER"
static PetscErrorCode GeometryPICellITER(DM base, PetscInt point, PetscInt dim, const PetscReal abc[], PetscReal xyz[], void *a_ctx)
{
  X2Ctx *ctx = (X2Ctx*)a_ctx;
  X2Grid *params = &ctx->grid;
  PetscReal radius_major    = params->radius_major;
  PetscInt  num_phi_cells  = params->num_phi_cells;
  PetscInt  i,idxPhi;
  PetscReal a, b, z;
  PetscReal inPhi, outPhi;
  PetscReal midPhi, leftPhi;
  PetscReal cosOutPhi, sinOutPhi;
  PetscReal cosMidPhi, sinMidPhi;
  PetscReal cosLeftPhi, sinLeftPhi;
  PetscReal secHalf;
  PetscReal r, rhat, dist, fulldist;
  PetscFunctionBegin;
  a = abc[0];
  b = abc[1];
  z = abc[2];
  inPhi = atan2(b,a);
  inPhi = (inPhi < 0.) ? (inPhi + 2. * M_PI) : inPhi;
  idxPhi = (inPhi * num_phi_cells) / (2. * M_PI);
  idxPhi = PetscMin(idxPhi,num_phi_cells - 1);
  leftPhi =  (idxPhi *        2. * M_PI) / num_phi_cells;
  midPhi  = ((idxPhi + 0.5) * 2. * M_PI) / num_phi_cells;
  cosMidPhi  = cos(midPhi);
  sinMidPhi  = sin(midPhi);
  cosLeftPhi = cos(leftPhi);
  sinLeftPhi = sin(leftPhi);
  secHalf    = 1. / cos(M_PI / num_phi_cells);
  rhat = (a * cosMidPhi + b * sinMidPhi);
  r    = secHalf * rhat;
  dist = secHalf * (-a * sinLeftPhi + b * cosLeftPhi);
  fulldist = 2. * sin(M_PI / num_phi_cells) * r;
  outPhi = ((idxPhi + (dist/fulldist)) * 2. * M_PI) / num_phi_cells;
  /* solve r * (cosLeftPhi * _i + sinLeftPhi * _j) + dist * (nx * _i + ny * _j) = a * _i + b * _j;
   *
   * (r * cosLeftPhi + dist * nx) = a;
   * (r * sinLeftPhi + dist * ny) = b;
   *
   * r    = idet * ( a * ny         - b * nx);
   * dist = idet * (-a * sinLeftPhi + b * cosLeftPhi);
   * idet = 1./(cosLeftPhi * ny - sinLeftPhi * nx) = sec(Pi/num_phi_cells);
   */
  r -= radius_major; /* now centered inside torus */
  {
    PetscInt crossQuad = point % s_numQuads;
    int       vertices[9];
    PetscReal eta[2] = {-1., -1.};
    PetscReal vertCoords[9][2];
    PetscReal shape[9];

    for (i = 0; i < 9; i++) {
      vertices[i] = s_quad_vertex[crossQuad][i] -1;
    }
    for (i = 0; i < 4; i++) { /* read in corners of quad */
      vertCoords[i][0] = s_wallVtx[vertices[i]][0];
      vertCoords[i][1] = s_wallVtx[vertices[i]][1];
    }
    for (; i < 8; i++) { /* read in mid edge vertices: if not present, average */
      if (vertices[i] >= 0) {
        vertCoords[i][0] = s_wallVtx[vertices[i]][0];
        vertCoords[i][1] = s_wallVtx[vertices[i]][1];
      }
      else {
        vertCoords[i][0] = (vertCoords[i - 4][0] + vertCoords[(i - 4 + 1) % 4][0])/2.;
        vertCoords[i][1] = (vertCoords[i - 4][1] + vertCoords[(i - 4 + 1) % 4][1])/2.;
        PetscPrintf(PETSC_COMM_SELF,"edge vertCoords=%g %g\n",vertCoords[i][0]+radius_major,vertCoords[i][1]);
      }
    }
    for (; i < 9; i++) { /* read in middle vertex: if not present, average edge vertices*/
      if (vertices[i] >= 0) {
        vertCoords[i][0] = s_wallVtx[vertices[i]][0];
        vertCoords[i][1] = s_wallVtx[vertices[i]][1];
      }
      else {
        vertCoords[i][0] = (vertCoords[i - 4][0] + vertCoords[i - 3][0] + vertCoords[i - 2][0] + vertCoords[i - 1][0])/4.;
        vertCoords[i][1] = (vertCoords[i - 4][1] + vertCoords[i - 3][1] + vertCoords[i - 2][1] + vertCoords[i - 1][1])/4.;
        PetscPrintf(PETSC_COMM_SELF,"corner vertCoords=%g %g\n",vertCoords[i][0]+radius_major,vertCoords[i][1]);
      }
    }

    /* convert (r,z) to eta (i.e., refToReal for bilinear elements) */
    {
      PetscReal v0[2] = {vertCoords[0][0], vertCoords[0][1]};
      PetscReal v2[2] = {vertCoords[2][0], vertCoords[2][1]};
      PetscReal J0[2][2] = {{vertCoords[1][0], vertCoords[3][0]},{vertCoords[1][1], vertCoords[3][1]}};
      PetscReal J0det;
      PetscReal J0inv[2][2];
      PetscReal x[2], y[2];
      PetscReal a, b, c;

      J0[0][0] -= v0[0];
      J0[0][1] -= v0[0];
      J0[1][0] -= v0[1];
      J0[1][1] -= v0[1];
      J0det = J0[0][0] * J0[1][1] - J0[0][1] * J0[1][0];
      J0inv[0][0] = J0[1][1] / J0det;
      J0inv[1][1] = J0[0][0] / J0det;
      J0inv[1][0] = -J0[1][0] / J0det;
      J0inv[0][1] = -J0[0][1] / J0det;
      x[0] = (r - v0[0]) * J0inv[0][0] + (z - v0[1]) * J0inv[0][1];
      x[1] = (r - v0[0]) * J0inv[1][0] + (z - v0[1]) * J0inv[1][1];
      y[0] = (v2[0] - v0[0]) * J0inv[0][0] + (v2[1] - v0[1]) * J0inv[0][1];
      y[1] = (v2[0] - v0[0]) * J0inv[1][0] + (v2[1] - v0[1]) * J0inv[1][1];
      a = (1. - y[1]);
      b = (x[1] - x[0] + x[0]*y[1] - x[1]*y[0] - 1.);
      c = x[0];
      if (fabs(a) > PETSC_SMALL) {
        eta[0] = (-b + PetscSqrtReal(b*b-4.* a * c)) / (2. * a);
        if (eta[0] < 0. || eta[0] > 1.) {
          eta[0] = (-b - PetscSqrtReal(b*b-4.* a * c)) / (2. * a);
        }
      }
      else {
        eta[0] = -c / b;
      }
      eta[1] = x[1] / (1. - eta[0] + eta[0] * y[1]);
    }
    /* evaluate quadratic functions at eta */
    for (i = 0; i < 4; i++) {
      shape[i] = ((i == 0 || i == 3) ? (2. * (0.5 - eta[0]) * (1. - eta[0])) : (2. * eta[0] * (eta[0] - 0.5))) *
                 ((i == 0 || i == 1) ? (2. * (0.5 - eta[1]) * (1. - eta[1])) : (2. * eta[1] * (eta[1] - 0.5)));
    }
    for (; i < 8; i++) {
      shape[i] = ((i == 4 || i == 6) ? (4. * eta[0] * (1. - eta[0])) : ((i == 5) ? (2. * eta[0] * (eta[0] - 0.5)) : (2. * (0.5 - eta[0]) * (1. - eta[0])))) *
                 ((i == 5 || i == 7) ? (4. * eta[1] * (1. - eta[1])) : ((i == 4) ? (2. * (0.5 - eta[1]) * (1. - eta[1])) : (2. * eta[1] * (eta[1] - 0.5))));
    }
    shape[8] = 16. * eta[0] * (1. - eta[0]) * eta[1] * (1. - eta[1]);
    r = 0.;
    z = 0.;
    for (i = 0; i < 9; i++) {
      r += shape[i] * s_wallVtx[vertices[i]][0];
      z += shape[i] * s_wallVtx[vertices[i]][1];
    }
  }
  r += radius_major; /* centered back at the origin */

  cosOutPhi = cos(outPhi);
  sinOutPhi = sin(outPhi);
  xyz[0] = r * cosOutPhi;
  xyz[1] = r * sinOutPhi;
  xyz[2] = z;

  PetscFunctionReturn(0);
}

/* == Defining a base plex for a torus, which looks like a rectilinear donut */
#undef __FUNCT__
#define __FUNCT__ "DMLocalizeCoordinate_Torus"
/*
  DMLocalizeCoordinate_Internal - If a mesh is periodic, and the input point is far from the anchor, pick the coordinate sheet of the torus which moves it closer.

  Input Parameters:
+ dm     - The DM
. dim    - The spatial dimension
. anchor - The anchor point, the input point can be no more than maxCell away from it
. ctx    - Optional context to pass
- in     - The input coordinate point (dim numbers)

  Output Parameter:
. out - The localized coordinate point

  Level: developer

  Note: This is meant to get a set of coordinates close to each other, as in a cell. The anchor is usually the one of the vertices on a containing cell

.seealso: DMLocalizeCoordinates(), DMLocalizeAddCoordinate()
*/
PetscErrorCode DMLocalizeCoordinate_Torus(DM dm, PetscInt dim, const PetscScalar anchor[], const PetscScalar in[], PetscScalar out[], void *actx)
{
  PetscInt d;
  X2Ctx    *ctx = (X2Ctx*)actx;
  PetscFunctionBegin;
  if (!dm->maxCell) {
    for (d = 0; d < dim; ++d) out[d] = in[d];
  } else {
    if (!ctx) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "No context?");
    for (d = 0; d < dim; ++d) {
      PetscPrintf(PETSC_COMM_SELF,"[%D]%s %D) anchor=%g in=%g maxCell=%g\n",ctx->rank,__FUNCT__,d,anchor[d],in[d],dm->maxCell[d]);
      if (PetscAbsScalar(anchor[d] - in[d]) > dm->maxCell[d]) {
        out[d] = PetscRealPart(anchor[d]) > PetscRealPart(in[d]) ? dm->L[d] + in[d] : in[d] - dm->L[d];
        PetscPrintf(PETSC_COMM_SELF,"\t\t[%D]%s Found a periodic point: %D) anchor=%g in=%g maxCell=%g\n",ctx->rank,__FUNCT__,d,anchor[d],in[d],dm->maxCell[d]);
      } else {
        out[d] = in[d];
      }
    }
  }
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "DMPlexCreatePICellBoxTorus"
static PetscErrorCode DMPlexCreatePICellBoxTorus(MPI_Comm comm, X2Grid *params, DM *dm)
{
  PetscMPIInt    rank;
  PetscInt       numCells = 0;
  PetscInt       numVerts = 0;
  PetscReal      radius_major   = params->radius_major;
  const PetscInt num_phi_cells = params->num_phi_cells, nplains = (s_section_phi == 2) ? num_phi_cells : num_phi_cells+1;;
  int           *flatCells = NULL;
  double         *flatCoords = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    numCells = num_phi_cells * 1;
    numVerts = nplains * 4;
    ierr = PetscMalloc2(numCells * 8,&flatCells,numVerts * 3,&flatCoords);CHKERRQ(ierr);
    {
      double (*coords)[4][3] = (double (*) [4][3]) flatCoords;
      PetscInt i;

      for (i = 0; i < nplains; i++) {
        PetscInt j;
        double cosphi, sinphi;

        cosphi = cos(2 * M_PI * i / num_phi_cells);
        sinphi = sin(2 * M_PI * i / num_phi_cells);

        for (j = 0; j < 4; j++) {
          double r, z;

          r = radius_major + params->radius_minor*s_rminor_inflate * ( (j==1 || j==2)        ? -1. :  1.);
          z =  params->radius_minor * ( (j < 2) ?  1. : -1. );

          coords[i][j][0] = cosphi * r;
          coords[i][j][1] = sinphi * r;
          coords[i][j][2] = z;
        }
      }
    }
    {
      int (*cells)[1][8] = (int (*) [1][8]) flatCells;
      PetscInt k, i, j = 0;

      for (i = 0; i < num_phi_cells; i++) {
        for (k = 0; k < 8; k++) {
          PetscInt l = k % 4;

          cells[i][j][k] = (4 * ((k < 4) ? i : (i + 1)) + (3 - l)) % numVerts;
        }
        {
          PetscInt swap = cells[i][j][1];
          cells[i][j][1] = cells[i][j][3];
          cells[i][j][3] = swap;
        }
      }
    }
  }

  ierr = DMPlexCreateFromCellList(comm,3,numCells,numVerts,8,PETSC_TRUE,flatCells,3,flatCoords,dm);CHKERRQ(ierr);
  ierr = PetscFree2(flatCells,flatCoords);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "torus");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* == Defining a base plex for a torus, which looks like a rectilinear donut, and a mapping that turns it into a conventional round donut == */

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreatePICellTorus"
static PetscErrorCode DMPlexCreatePICellTorus(MPI_Comm comm, X2Grid *params, DM *dm)
{
  PetscMPIInt    rank;
  const PetscInt num_phi_cells = params->num_phi_cells, nplains = (s_section_phi == 2) ? num_phi_cells : num_phi_cells+1;
  PetscInt       numCells = 0;
  PetscInt       numVerts = 0;
  PetscReal      radius_major = params->radius_major;
  PetscReal      radius_minor = params->radius_minor*s_rminor_inflate;
  PetscReal      inner_mult = params->inner_mult;
  int            *flatCells = NULL;
  double         *flatCoords = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    numCells = num_phi_cells * 5;
    numVerts = nplains * 8;
    ierr = PetscMalloc2(numCells * 8,&flatCells,numVerts * 3,&flatCoords);CHKERRQ(ierr);
    {
      double (*coords)[8][3] = (double (*) [8][3]) flatCoords;
      PetscInt i;

      for (i = 0; i < nplains; i++) {
        PetscInt j;
        double cosphi, sinphi;

        cosphi = cos(s_section_phi * M_PI * i / num_phi_cells);
        sinphi = sin(s_section_phi * M_PI * i / num_phi_cells);

        for (j = 0; j < 8; j++) {
          double r, z;
          double mult = (j < 4) ? inner_mult : 1.;

          r = radius_major + mult * radius_minor * cos(j * M_PI_2);
          z = mult * radius_minor * sin(j * M_PI_2);

          coords[i][j][0] = cosphi * r;
          coords[i][j][1] = sinphi * r;
          coords[i][j][2] = z;
        }
      }
    }
    {
      int (*cells)[5][8] = (int (*) [5][8]) flatCells;
      PetscInt i;

      for (i = 0; i < num_phi_cells; i++) {
        PetscInt j;

        for (j = 0; j < 5; j++) {
          PetscInt k;

          if (j < 4) {
            for (k = 0; k < 8; k++) {
              PetscInt l = k % 4;

              cells[i][j][k] = (8 * ((k < 4) ? i : (i + 1)) + ((l % 3) ? 0 : 4) + ((l < 2) ? j : ((j + 1) % 4))) % numVerts;
            }
          }
          else {
            for (k = 0; k < 8; k++) {
              PetscInt l = k % 4;

              cells[i][j][k] = (8 * ((k < 4) ? i : (i + 1)) + (3 - l)) % numVerts;
            }
          }
          {
            PetscInt swap = cells[i][j][1];

            cells[i][j][1] = cells[i][j][3];
            cells[i][j][3] = swap;
          }
        }
      }
    }
  }

  ierr = DMPlexCreateFromCellList(comm,3,numCells,numVerts,8,PETSC_TRUE,flatCells,3,flatCoords,dm);CHKERRQ(ierr);
  ierr = PetscFree2(flatCells,flatCoords);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "torus");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static void PICellCircleInflate(PetscReal r, PetscReal inner_mult, PetscReal x, PetscReal y,
                                PetscReal *outX, PetscReal *outY)
{
  PetscReal l       = x + y;
  PetscReal rfrac   = l / r;

  if (rfrac >= inner_mult) {
    PetscReal phifrac = l ? (y / l) : 0.5;
    PetscReal phi     = phifrac * M_PI_2;
    PetscReal cosphi  = cos(phi);
    PetscReal sinphi  = sin(phi);
    PetscReal isect   = inner_mult / (cosphi + sinphi);
    PetscReal outfrac = (1. - rfrac) / (1. - inner_mult);

    rfrac = pow(inner_mult,outfrac);

    outfrac = (1. - rfrac) / (1. - inner_mult);

    *outX = r * (outfrac * isect + (1. - outfrac)) * cosphi;
    *outY = r * (outfrac * isect + (1. - outfrac)) * sinphi;
  }
  else {
    PetscReal halfdiffl  = (r * inner_mult - l) / 2.;
    PetscReal phifrac    = (y + halfdiffl) / (r * inner_mult);
    PetscReal phi        = phifrac * M_PI_2;
    PetscReal m          = y - x;
    PetscReal halfdiffm  = (r * inner_mult - m) / 2.;
    PetscReal thetafrac  = (y + halfdiffm) / (r * inner_mult);
    PetscReal theta      = thetafrac * M_PI_2;
    PetscReal cosphi     = cos(phi);
    PetscReal sinphi     = sin(phi);
    PetscReal ymxcoord   = sinphi / (cosphi + sinphi);
    PetscReal costheta   = cos(theta);
    PetscReal sintheta   = sin(theta);
    PetscReal xpycoord   = sintheta / (costheta + sintheta);

    *outX = r * inner_mult * (xpycoord - ymxcoord);
    *outY = r * inner_mult * (ymxcoord + xpycoord - 1.);
  }
}

#undef __FUNCT__
#define __FUNCT__ "GeometryPICellTorus"
static PetscErrorCode GeometryPICellTorus(DM base, PetscInt point, PetscInt dim, const PetscReal abc[], PetscReal xyz[], void *a_ctx)
{
  X2Ctx     *ctx = (X2Ctx*)a_ctx;
  X2Grid    *grid = &ctx->grid;
  PetscReal radius_major = grid->radius_major;
  PetscReal radius_minor = grid->radius_minor*s_rminor_inflate;
  PetscReal inner_mult = grid->inner_mult;
  PetscInt  num_phi_cells  = grid->num_phi_cells;
  PetscInt  i;
  PetscReal a, b, z;
  PetscReal inPhi, outPhi;
  PetscReal midPhi, leftPhi;
  PetscReal cosOutPhi, sinOutPhi;
  PetscReal cosMidPhi, sinMidPhi;
  PetscReal cosLeftPhi, sinLeftPhi;
  PetscReal secHalf;
  PetscReal r, rhat, dist, fulldist;

  PetscFunctionBegin;
  a = abc[0];
  b = abc[1];
  z = abc[2];
  inPhi = atan2(b,a);
  inPhi = (inPhi < 0.) ? (inPhi + s_section_phi * M_PI) : inPhi;
  i = (inPhi * num_phi_cells) / (s_section_phi * M_PI);
  i = PetscMin(i,num_phi_cells - 1);
  leftPhi =  (i *        s_section_phi * M_PI) / num_phi_cells;
  midPhi  = ((i + 0.5) * s_section_phi * M_PI) / num_phi_cells;
  cosMidPhi  = cos(midPhi);
  sinMidPhi  = sin(midPhi);
  cosLeftPhi = cos(leftPhi);
  sinLeftPhi = sin(leftPhi);
  secHalf    = 1. / cos(s_section_phi*M_PI / (2*num_phi_cells) );

  rhat = (a * cosMidPhi + b * sinMidPhi);
  r    = secHalf * rhat;
  dist = secHalf * (-a * sinLeftPhi + b * cosLeftPhi);
  fulldist = 2. * sin(s_section_phi*M_PI / (2*num_phi_cells)) * r;
  outPhi = ((i + (dist/fulldist)) * s_section_phi * M_PI) / num_phi_cells;
  /* solve r * (cosLeftPhi * _i + sinLeftPhi * _j) + dist * (nx * _i + ny * _j) = a * _i + b * _j;
   *
   * (r * cosLeftPhi + dist * nx) = a;
   * (r * sinLeftPhi + dist * ny) = b;
   *
   * r    = idet * ( a * ny         - b * nx);
   * dist = idet * (-a * sinLeftPhi + b * cosLeftPhi);
   * idet = 1./(cosLeftPhi * ny - sinLeftPhi * nx) = sec(Pi/num_phi_cells);
   */
  r -= radius_major; /* now centered inside torus */
  if (ctx->inflate_torus) {
    PetscReal absR, absZ;

    absR = PetscAbsReal(r);
    absZ = PetscAbsReal(z);
    PICellCircleInflate(radius_minor,inner_mult,absR,absZ,&absR,&absZ);
    r = (r > 0) ? absR : -absR;
    z = (z > 0) ? absZ : -absZ;
  }
  r += radius_major; /* centered back at the origin */

  cosOutPhi = cos(outPhi);
  sinOutPhi = sin(outPhi);
  xyz[0] = r * cosOutPhi;
  xyz[1] = r * sinOutPhi;
  xyz[2] = z;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  X2Ctx          ctx; /* user-defined work context */
  PetscErrorCode ierr;
  DM_PICell      *dmpi;
  PetscInt       dim,idx,isp;
  Mat            J;
  DMLabel        label;
  PetscDS        prob;
  PetscSection   section;
  PetscLogStage  setup_stage;
  PetscFunctionBeginUser;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ctx.events = s_events;
#if defined(PETSC_USE_LOG)
  {
    PetscInt currevent = 0;
    ierr = PetscLogEventRegister("X2CreateMesh", DM_CLASSID, &ctx.events[currevent++]);CHKERRQ(ierr); /* 0 */
    ierr = PetscLogEventRegister("X2Process parts",0,&ctx.events[currevent++]);CHKERRQ(ierr); /* 1 */
    ierr = PetscLogEventRegister(" -shiftParticles",0,&ctx.events[currevent++]);CHKERRQ(ierr); /* 2 */
    ierr = PetscLogEventRegister("   =Non-block con",0,&ctx.events[currevent++]);CHKERRQ(ierr); /* 3 */
    ierr = PetscLogEventRegister("     *Part. Send", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 4 */
    ierr = PetscLogEventRegister(" -Move parts", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 5 */
    ierr = PetscLogEventRegister(" -AddSource", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 6 */
    ierr = PetscLogEventRegister(" -Pre Push", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 7 */
    ierr = PetscLogEventRegister(" -Push (Jet)", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 8 */
    ierr = PetscLogEventRegister("   =Part find (s)", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 9 */
    ierr = PetscLogEventRegister("   =Part find (p)", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 10 */
    ierr = PetscLogEventRegister("X2Poisson Solve", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 11 */
    ierr = PetscLogEventRegister("X2Part AXPY", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 12 */
    ierr = PetscLogEventRegister("X2Compress array", 0, &ctx.events[currevent++]);CHKERRQ(ierr); /* 13 */
    ierr = PetscLogEventRegister("X2Diagnostics", 0, &ctx.events[diag_event_id]);CHKERRQ(ierr); /* N-1 */
    assert(sizeof(s_events)/sizeof(s_events[0]) > currevent);
    ierr = PetscLogStageRegister("Setup", &setup_stage);CHKERRQ(ierr);
    ierr = PetscLogStagePush(setup_stage);CHKERRQ(ierr);
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
  /* setup solver grid */
  if (ctx.run_type == X2_ITER) {
    ierr = DMPlexCreatePICellITER(ctx.wComm,&ctx.grid,&dmpi->dmplex);CHKERRQ(ierr);
  }
  else {
    if (ctx.run_type == X2_TORUS) {
      ierr = DMPlexCreatePICellTorus(ctx.wComm,&ctx.grid,&dmpi->dmplex);CHKERRQ(ierr);
      ctx.inflate_torus = PETSC_TRUE;
    }
    else {
      ierr = DMPlexCreatePICellBoxTorus(ctx.wComm,&ctx.grid,&dmpi->dmplex);CHKERRQ(ierr);
      ctx.inflate_torus = PETSC_FALSE;
      assert(ctx.run_type == X2_BOXTORUS);
    }
    if (s_section_phi != 2) {
      PetscInt i;
      DMBoundaryType bd[3] = {DM_BOUNDARY_NONE,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_NONE};
      const PetscReal L[3] = {1.e10, (radius_major-radius_minor)*s_section_phi*M_PI, 1.e10};
      PetscReal maxCell[3];
      if (s_section_phi > -.1) PetscPrintf(comm, "%s Warning section_phi is large %g, L=%g %g %g",__FUNCT__,s_section_phi,L[0],L[1],L[2]);CHKERRQ(ierr);
      for (i = 0; i < 3; i++) maxCell[i] =  (radius_major-radius_minor)/3;
      ierr = DMSetPeriodicity(*dm, maxCell, L, bd, DMLocalizeCoordinate_Torus);CHKERRQ(ierr);
    }
  }
  ierr = DMSetApplicationContext(dmpi->dmplex, &ctx);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) dmpi->dmplex, "x2_");CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) ctx.dm, "x2_");CHKERRQ(ierr);
  ierr = PetscMalloc(1 * sizeof(PetscErrorCode (*)(PetscInt,const PetscReal [],PetscInt,PetscScalar*,void*)),&ctx.BCFuncs);CHKERRQ(ierr);
  ctx.BCFuncs[0] = zero;
  /* add BCs */
  {
    PetscInt id = 1;
    ierr = DMCreateLabel(dmpi->dmplex, "boundary");CHKERRQ(ierr);
    ierr = DMGetLabel(dmpi->dmplex, "boundary", &label);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(dmpi->dmplex, label);CHKERRQ(ierr);
    ierr = DMAddBoundary(dmpi->dmplex, PETSC_TRUE, "wall", "boundary", 0, 0, NULL, (void (*)()) ctx.BCFuncs[0], 1, &id, &ctx);CHKERRQ(ierr);
  }
  { /* convert to p4est */
    char convType[256];
    PetscBool flg;
    ierr = PetscOptionsBegin(ctx.wComm, "", "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-x2_dm_type","Convert DMPlex to another format (should not be Plex!)","ex1.c",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
    if (flg) {
      ierr = DMConvert(dmpi->dmplex,convType,&dmpi->dmgrid);CHKERRQ(ierr);
      if (dmpi->dmgrid) {
        const char *prefix;
        PetscBool isForest;
        ierr = PetscObjectGetOptionsPrefix((PetscObject)dmpi->dmplex,&prefix);CHKERRQ(ierr);
        ierr = PetscObjectSetOptionsPrefix((PetscObject)dmpi->dmgrid,prefix);CHKERRQ(ierr);
        ierr = DMIsForest(dmpi->dmgrid,&isForest);CHKERRQ(ierr);
        if (isForest) {
          if (ctx.run_type == X2_ITER) {
            ierr = DMForestSetBaseCoordinateMapping(dmpi->dmgrid,GeometryPICellITER,&ctx);CHKERRQ(ierr);
          }
          else if (ctx.run_type == X2_TORUS) {
            ierr = DMForestSetBaseCoordinateMapping(dmpi->dmgrid,GeometryPICellTorus,&ctx);CHKERRQ(ierr);
          }
          else {
            ierr = DMForestSetBaseCoordinateMapping(dmpi->dmgrid,GeometryPICellTorus,&ctx);CHKERRQ(ierr);
            assert(ctx.run_type == X2_BOXTORUS);
          }
        }
        else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Converted to non Forest?");
      }
      else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Convert failed?");
    }
    else {
      if (dmpi->debug>0) PetscPrintf(ctx.wComm,"[%D] No p4est\n",ctx.rank);
      dmpi->dmgrid = dmpi->dmplex;
    }
  }
  if (sizeof(long long)!=sizeof(PetscReal)) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "sizeof(long long)!=sizeof(PetscReal)");

  /* setup DM, refine, distribute */
  ierr = DMSetFromOptions( ctx.dm );CHKERRQ(ierr); /* refinement done here */
  if (dmpi->dmgrid == dmpi->dmplex) { /* not using p4est, distribute */
    const char *prefix;
    DM dm;
    ierr = PetscObjectGetOptionsPrefix((PetscObject)dmpi->dmplex,&prefix);CHKERRQ(ierr);
    /* plex does not distribute by implicitly, so do it. But p4est partitioning is different. if != should get distribution from p4est */
    ierr = DMPlexDistribute(dmpi->dmplex, 0, NULL, &dm);CHKERRQ(ierr);
    if (dm) {
      ierr = PetscObjectSetOptionsPrefix((PetscObject)dm,prefix);CHKERRQ(ierr);
      ierr = DMDestroy(&dmpi->dmplex);CHKERRQ(ierr);
      dmpi->dmplex = dmpi->dmgrid = dm;
    }
  }

  /* setup Discretization */
  ierr = DMGetDimension(dmpi->dmgrid, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dmpi->dmgrid, dim, 1, PETSC_FALSE, NULL, PETSC_DECIDE, &dmpi->fem);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dmpi->fem, "poisson");CHKERRQ(ierr);
  /* FEM prob */
  ierr = DMGetDS(dmpi->dmgrid, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) dmpi->fem);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 0, 0, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);

  /* convert and get section */
  if (dmpi->dmgrid == dmpi->dmplex) {
    ierr = DMGetDefaultSection(dmpi->dmplex, &section);CHKERRQ(ierr);
    if (!section) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "DMGetDefaultSection return NULL");
    ierr = DMGetDefaultGlobalSection(dmpi->dmgrid, &section);CHKERRQ(ierr);
    if (!section) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "DMGetDefaultSection return NULL");
  } else { /* convert forest to plex - original plex not refined with -x2_dm_forest_initial_refinement */
    ierr = DMDestroy(&dmpi->dmplex);CHKERRQ(ierr);
    ierr = DMSetUp(dmpi->dmgrid);CHKERRQ(ierr);
    ierr = DMConvert(dmpi->dmgrid,DMPLEX,&dmpi->dmplex);CHKERRQ(ierr); /* low overhead, cached */
    /* get section */
    ierr = DMGetDefaultGlobalSection(dmpi->dmgrid, &section);CHKERRQ(ierr);
  }
  ierr = PetscSectionViewFromOptions(section, NULL, "-section_view");CHKERRQ(ierr);
  if (dmpi->debug>3) { /* this shows a bug with crap in the section */
    ierr = PetscSectionView(section,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  if (dmpi->debug>2) {
    ierr = DMView(dmpi->dmplex,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = DMSetUp( ctx.dm );CHKERRQ(ierr);

  {
    PetscInt n,cStart,cEnd;
    ierr = VecGetSize(dmpi->rho,&n);CHKERRQ(ierr);
    if (!n) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "No dofs");
    ierr = DMPlexGetHeightStratum(dmpi->dmplex, 0, &cStart, &cEnd);CHKERRQ(ierr);
    if (cStart) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER, "cStart != 0. %D",cStart);
    if (dmpi->debug>0 && !cEnd) {
      ierr = PetscPrintf((dmpi->debug>1 || !cEnd) ? PETSC_COMM_SELF : ctx.wComm,"[%D] ERROR %D global equations, %d local cells, (cEnd=%d), debug=%D\n",ctx.rank,n,cEnd-cStart,cEnd,dmpi->debug);
    }
    if (!cEnd) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "No cells");
    }
    s_fluxtubeelem = cEnd/2;
    if (dmpi->debug>0) PetscPrintf(ctx.wComm,"[%D] %D equations on %D processors, %D local cells, (element %D used for flux tube list)\n",
                                   ctx.rank,n,ctx.npe,cEnd,s_fluxtubeelem);
  }

  /* create SNESS */
  ierr = SNESCreate( ctx.wComm, &dmpi->snes);CHKERRQ(ierr);
  ierr = SNESSetDM( dmpi->snes, dmpi->dmgrid);CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dmpi->dmgrid,&ctx,&ctx,&ctx);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dmpi->dmgrid, &J);CHKERRQ(ierr);
  ierr = SNESSetJacobian(dmpi->snes, J, J, NULL, NULL);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(dmpi->snes);CHKERRQ(ierr);

  /* setup particles */
  ierr = createParticles( &ctx );CHKERRQ(ierr);

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
      if (!isp) {
        sprintf(fname1,         "particles_electrons_time%05d_fluxtube.h5part",(int)0);
        sprintf(fname2,"sub_rank_particles_electrons_time%05d_fluxtube.h5part",(int)0);
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
  if (ctx.plot) { /* debug */
    PetscViewer       viewer = NULL;
    PetscBool         flg;
    PetscViewerFormat fmt;
    DM_PICell      *dmpi = (DM_PICell *) ctx.dm->data;
    if (dmpi->dmgrid) {
      ierr = DMViewFromOptions(dmpi->dmgrid,NULL,"-dm_view");CHKERRQ(ierr);
    }
    else {
      ierr = DMViewFromOptions(dmpi->dmplex,NULL,"-dm_view");CHKERRQ(ierr);
    }
    ierr = PetscOptionsGetViewer(ctx.wComm,NULL,"-x2_vec_view",&viewer,&fmt,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerPushFormat(viewer,fmt);CHKERRQ(ierr);
      ierr = VecView(dmpi->phi,viewer);CHKERRQ(ierr);
      ierr = VecView(dmpi->rho,viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  /* setup solver, dummy solve to really setup */
  {
    ierr = VecZeroEntries(dmpi->rho);CHKERRQ(ierr); /* zero density to make solver do nothing */
    ierr = DMPICellSolve( ctx.dm );CHKERRQ(ierr);
  }

  /* move back to solver space and make density vector */
  ierr = processParticles(&ctx, 0.0, &ctx.sendListTable, 99, -1, -1, PETSC_TRUE);CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(ctx.events[0],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
#endif
  /* do it */
  ierr = go( &ctx );CHKERRQ(ierr);

  if (dmpi->debug>3) {
    PetscViewer viewer;
    PetscViewerASCIIOpen(ctx.wComm, "Amat.m", &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
    MatView(J,viewer);
    PetscViewerPopFormat(viewer);
    PetscViewerDestroy(&viewer);
  }
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

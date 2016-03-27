/* M. Adams, April 2015 */

static char help[] = "X2: A partical in cell code for tokamak plasmas using PICell.\n";

#ifdef H5PART
#include <H5Part.h>
#endif
#include <petsc/private/dmpicellimpl.h>    /*I   "petscdmpicell.h"   I*/
#include <assert.h>
#include <petscds.h>
#include <petscdmforest.h>
/* #include <petscoptions.h> */

typedef struct {
  /* particle grid sizes */
  PetscInt nradius;
  PetscInt ntheta;
  PetscInt nphi; /* toroidal direction */
  /* tokamak geometry  */
  PetscReal  rMajor;
  PetscReal  rMinor;
  PetscInt   numMajor; /* number of cells per major circle in the torus */
  PetscReal  innerMult; /* (0,1) percent of the total radius taken by the inner square */
} X2GridParticle;
/* X2Species */
#define X2_NION 1
typedef struct {
  PetscReal mass;
  PetscReal charge;
} X2Species;

/* X2Particle */
typedef struct { /* ptl_type */
  /* phase (4D) */
  PetscReal r;   /* r from center */
  PetscReal z;   /* vertical coordinate */
  PetscReal phi; /* toroidal coordinate */
  PetscReal vpar; /* toroidal velocity */
  /* const */
  PetscReal mu; /* 5th D */
  PetscReal w0;
  PetscReal f0;
  long long gid; /* diagnostic */
} X2Particle;
/* X2PList */
typedef PetscInt X2PListPos;
typedef struct {
  X2Particle *data;
  PetscInt    data_size, size, hole, top;
} X2PList;
/* send particle list */
typedef struct X2SendList_TAG{
  X2PList plist[X2_NION+1];
  PetscMPIInt proc;
} X2SendList;
/* MPI Isend particle list */
typedef struct X2ISend_TAG{
  X2Particle *data;
  PetscMPIInt proc;
  MPI_Request request;
} X2ISend;
/*
  General parameters and context
*/
typedef struct {
  PetscInt      debug;   /* The debugging level, not used */
  PetscLogEvent events[12];
  PetscInt      currevent;
  PetscInt      bsp_chuncksize;
  /* MPI parallel data */
  MPI_Comm      particlePlaneComm,wComm;
  PetscMPIInt   rank,npe,npe_particlePlane,particlePlaneRank,ParticlePlaneIdx;
  /* grids & solver */
  DM             dm;
  X2GridParticle particleGrid;
  /* time */
  PetscInt  msteps;
  PetscReal maxTime;
  PetscReal dt;
  /* physics */
  PetscErrorCode (**BCFuncs)(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  PetscReal massAu; /* =2D0  !mass ratio to proton */
  PetscReal eMassAu; /* =2D-2 */
  PetscReal chargeEu; /* =1D0  ! charge number */
  PetscReal eChargeEu; /* =-1D0 */
  /* particles */
  PetscInt  npart_cell;
  PetscInt  partBuffSize;
  PetscBool useElectrons;
  PetscInt  collisionPeriod;
  PetscReal max_vpar;
  X2PList   partlists[X2_NION+1]; /* 0: electron, 1:N ions */
  X2Species species[X2_NION+1]; /* 0: electron, 1:N ions */
  PetscInt  tablesize,tablecount[X2_NION+1]; /* hash table meta-data for proc-send list table */
  char      wall_file[128];
#define X2_WALL_ARRAY_MAX 68 /* ITER file is 67 */
  float wallEdges[X2_WALL_ARRAY_MAX][2];
  int   numWallEdges;
} X2Ctx;

static const PetscReal x2ECharge=1.6022e-19;  /* electron charge (MKS) */
/* static const PetscReal x2Epsilon0=8.8542e-12; /\* permittivity of free space (MKS) *\/ */
static const PetscReal x2ProtMass=1.6720e-27; /* proton mass (MKS) */
/* static const PetscReal x2ElecMass=9.1094e-31; /\* electron mass (MKS) *\/ */
/* particle */
PetscErrorCode X2ParticleCreate(X2Particle *p, PetscInt gid, PetscReal r, PetscReal z, PetscReal phi, PetscReal vpar)
{
  if (gid <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"X2ParticleCreate: gid <= 0");
  p->r = r;
  p->z = z;
  p->phi = phi;
  p->vpar = vpar;
  p->gid = gid;
  p->mu = 0;
  p->w0 = 1.;
  p->f0 = 0;
  return 0;
}
PetscErrorCode X2ParticleCopy(X2Particle *p, X2Particle p2)
{
  if (p2.gid <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"X2ParticleCreate: gid <= 0");
  p->r = p2.r;
  p->z = p2.z;
  p->phi = p2.phi;
  p->vpar = p2.vpar;
  p->gid = p2.gid;
  p->mu = 0;
  p->w0 = 1.;
  p->f0 = 0;
  return 0;
}
PetscErrorCode X2ParticleRead(X2Particle *p, void *buf)
{
  PetscInt *ip;
  PetscReal *rp = (PetscReal*)buf;
  p->r = *rp++;
  p->z = *rp++;
  p->phi = *rp++;
  p->vpar = *rp++;
  p->mu = *rp++;
  p->w0 = *rp++;
  p->f0 = *rp++;
  ip =  (PetscInt*)rp;
  p->gid = *ip++;
  buf = (void*)ip;
  return 0;
}
PetscErrorCode X2ParticleWrite(X2Particle *p, void *buf)
{
  PetscInt *ip;
  PetscReal *rp = (PetscReal*)buf;
  *rp++ = p->r;
  *rp++ = p->z;
  *rp++ = p->phi;
  *rp++ = p->vpar;
  *rp++ = p->mu;
  *rp++ = p->w0;
  *rp++ = p->f0;
  ip =  (PetscInt*)rp;
  *ip++ = p->gid;
  buf = (void*)ip;
  return 0;
}

/* particle list */
PetscErrorCode X2PListCreate(X2PList *l, PetscInt msz)
{
  PetscErrorCode ierr;
  l->size=0;
  l->top=0;
  l->hole=-1;
  l->data_size=msz;
  ierr = PetscMalloc1(msz, &l->data);
  return ierr;
}
PetscErrorCode X2PListClear(X2PList *l)
{
  l->size=0; /* keep memory but kill data */
  l->top=0;
  l->hole=-1;
  return 0;
}

PetscErrorCode X2PListSetAt(X2PList *l, X2PListPos pos, X2Particle *part)
{
  l->data[pos] = *part;
  return 0;
}

PetscErrorCode X2PListCompress(X2PList *l)
{
  PetscInt ii;
  /* fill holes with end of list */
  for ( ii = 0 ; ii < l->top && l->top > l->size ; ii++) {
    if (l->data[ii].gid <= 0) {
      l->top--; /* maybe data */
      if (ii == l->top) /* pop */ ;
      else {
	while (l->data[l->top].gid <= 0) l->top--; /* data */
	l->data[ii] = l->data[l->top]; /* now above */
	/* PetscPrintf(PETSC_COMM_SELF,"\tfilled ii %d, with %d from top (%d), size=%d\n",ii,l->data[ii].gid,l->top,l->size); */
      }
    }
  }
  l->hole = -1;
  assert(l->top==l->size);
  return 0;
}

/* keep list of hols after removal so that we can iterate over a list and remove as we go
  gid < 0 : hole : -gid - 1 == -(gid+1) is next index
  gid = 0 : sentinal
  gid > 0 : real
*/
PetscErrorCode X2PListGetHead(X2PList *l, X2PListPos *pos)
{
  if (l->size==0) {
    *pos = 0; /* past end */
  }
  else {
    X2PListPos idx=0;
    while (l->data[idx].gid <= 0) idx++;

    *pos = idx - 1; /* eg -1 */
  }
  return 0;
}

PetscErrorCode X2PListGetNext(X2PList *l, X2Particle *p, X2PListPos *pos)
{
  /* l->size == 0 can happen on empty list */
  (*pos)++; /* get next position */
  if (*pos >= l->data_size || *pos >= l->top) return 1; /* hit end, can go past if list is just drained */
  while(l->data[*pos].gid <= 0 && *pos < l->data_size && *pos < l->top) (*pos)++; /* skip holes */
  *p = l->data[*pos]; /* return copy */

  return 0;
}

PetscErrorCode X2PListAdd( X2PList *l, X2Particle *p)
{
  if (l->size==l->data_size) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"X2PListAdd: list full (%D) - todo",l->size);
  if (l->hole != -1) { /* have a hole - fill it */
    X2PListPos idx = l->hole; assert(idx<l->data_size);
    if (l->data[idx].gid == 0) l->hole = -1; /* filled last hole */
    else if (l->data[idx].gid>=0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"X2PListAdd: hole with non-neg gid!!!",l->data[idx].gid);
    else l->hole = (X2PListPos)(-l->data[idx].gid - 1); /* use gid as pointer */
    l->data[idx] = *p; /* struct copy */
  }
  else {
    l->data[l->top++] = *p; /* struct copy */
  }
  l->size++;
  assert(l->top >= l->size);
  return 0;
}
PetscErrorCode X2PListRemoveAt( X2PList *l, X2PListPos pos)
{
  if(pos >= l->data_size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"X2PListRemoveAt past end of data %d %d",pos,l->data_size);
  if(pos >= l->top) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"X2PListRemoveAt past end of top pointer %d %d",pos,l->top);

  if (pos == l->top-1) {
    l->top--; /* simple pop */
  }
  else {
    if (l->hole==-1) l->data[pos].gid = 0; /* sentinal */
    else l->data[pos].gid = -(l->hole + 1); /* hole >= 0 */
    l->hole = pos; /* head of linked list of holes */
  }
  l->size--;
  if (!l->size) { /* lets reset if we drained the list */
    l->hole = -1;
    l->top = 0;
  }
  assert(l->top >= l->size);
  return 0;
}

PetscInt X2PListMaxSize(X2PList *l) {
  return l->data_size;
}

PetscInt X2PListSize(X2PList *l) {
  return l->size;
}
PetscErrorCode X2PListDestroy(X2PList *l)
{
  PetscErrorCode ierr;
  ierr = PetscFree(l->data);CHKERRQ(ierr);
  l->size=0;
  l->top=0;
  l->hole=-1;
  l->data = 0;
  l->data_size = 0;
  return ierr;
}

#define X2PROCLISTSIZE 4096
static PetscInt s_chunksize = 32768;
/*
   ProcessOptions: set parameters from input, setup w/o allocation, called first, no DM here
*/
#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions( X2Ctx *ctx )
{
  PetscErrorCode ierr,isp;
  PetscBool phiFlag,radFlag,thetaFlag;
  /* static char fname[] = "B0.eqd"; */

  PetscFunctionBeginUser;
  /* general */
  ierr = MPI_Comm_rank(ctx->wComm, &ctx->rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(ctx->wComm, &ctx->npe);CHKERRQ(ierr);

  /* physics */
  ctx->massAu=2;  /* mass ratio to proton */
  ctx->eMassAu=2e-2; /* mass of electron?? */
  ctx->chargeEu=1;   /* charge number */
  ctx->eChargeEu=-1;

  ctx->species[1].mass=ctx->massAu*x2ProtMass;
  ctx->species[1].charge=ctx->chargeEu*x2ECharge;
  ctx->species[0].mass=ctx->eMassAu*x2ProtMass;
  ctx->species[0].charge=ctx->eChargeEu*x2ECharge;

 /* mesh */
  ctx->particleGrid.rMajor = 6.2; /* cm of ITER */
  ctx->particleGrid.rMinor = 2.0; /* cm of ITER */
  ctx->particleGrid.nphi = 1;
  ctx->particleGrid.nradius    = 1;
  ctx->particleGrid.ntheta     = 1;
  ctx->particleGrid.numMajor   = 4;
  ctx->particleGrid.innerMult  = M_SQRT2 - 1.;

  ierr = PetscOptionsBegin(ctx->wComm, "", "Poisson Problem Options", "X2");CHKERRQ(ierr);
  /* general options */
  ctx->debug = 0;
  ierr = PetscOptionsInt("-debug", "The debugging level", "x2.c", ctx->debug, &ctx->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-chuncksize", "Size of particle list to chunk sends", "x2.c", s_chunksize, &s_chunksize, NULL);CHKERRQ(ierr);
  ctx->bsp_chuncksize = 0; /* 32768; */
  ierr = PetscOptionsInt("-bsp_chuncksize", "Size of chucks for PETSc's TwoSide communication (0 to use 'nonblocking consensus')", "x2.c", ctx->bsp_chuncksize, &ctx->bsp_chuncksize, NULL);CHKERRQ(ierr);
  if (ctx->bsp_chuncksize<0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB," invalid chuck size = %D",ctx->bsp_chuncksize);
  ctx->tablesize = 200; /* hash table size of processors to send to */
  ierr = PetscOptionsInt("-proc_send_table_size", "Size of hash table proc->send_list", "x2.c",ctx->tablesize = 200, &ctx->tablesize, NULL);CHKERRQ(ierr);

  /* Domain and mesh definition */
  ierr = PetscOptionsReal("-rMajor", "Major radius of torus", "x2.c", ctx->particleGrid.rMajor, &ctx->particleGrid.rMajor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rMinor", "Minor radius of torus", "x2.c", ctx->particleGrid.rMinor, &ctx->particleGrid.rMinor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-numMajor", "Number of cells per major circle", "x2.c", ctx->particleGrid.numMajor, &ctx->particleGrid.numMajor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-innerMult", "Percent of minor radius taken by inner square", "x2.c", ctx->particleGrid.innerMult, &ctx->particleGrid.innerMult, NULL);CHKERRQ(ierr);

  ierr = PetscOptionsInt("-nphi_particles", "Number of planes for particle mesh", "x2.c", ctx->particleGrid.nphi, &ctx->particleGrid.nphi, &phiFlag);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nradius_particles", "Number of radial cells for particle mesh", "x2.c", ctx->particleGrid.nradius, &ctx->particleGrid.nradius, &radFlag);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ntheta_particles", "Number of theta cells for particle mesh", "x2.c", ctx->particleGrid.ntheta, &ctx->particleGrid.ntheta, &thetaFlag);CHKERRQ(ierr);
  ctx->npe_particlePlane = -1;
  if (ctx->particleGrid.nphi*ctx->particleGrid.nradius*ctx->particleGrid.ntheta != ctx->npe) { /* recover from inconsistant grid/procs */
    if (thetaFlag && radFlag && phiFlag) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"over constrained number of particle processes npe (%D) != %D",ctx->npe,ctx->particleGrid.nphi*ctx->particleGrid.nradius*ctx->particleGrid.ntheta);

    if (!thetaFlag && radFlag && phiFlag) ctx->particleGrid.ntheta = ctx->npe/(ctx->particleGrid.nphi*ctx->particleGrid.nradius);
    else if (thetaFlag && !radFlag && phiFlag) ctx->particleGrid.nradius = ctx->npe/(ctx->particleGrid.nphi*ctx->particleGrid.ntheta);
    else if (thetaFlag && radFlag && !phiFlag) ctx->particleGrid.nphi = ctx->npe/(ctx->particleGrid.nradius*ctx->particleGrid.ntheta);
    else if (!thetaFlag && !radFlag && !phiFlag) {
      ctx->npe_particlePlane = (int)pow((double)ctx->npe,0.6667);
      ctx->particleGrid.nphi = ctx->npe/ctx->npe_particlePlane;
      ctx->particleGrid.nradius = (int)(sqrt((double)ctx->npe_particlePlane)+0.5);
      ctx->particleGrid.ntheta = ctx->npe_particlePlane/ctx->particleGrid.nradius;
      if (ctx->particleGrid.nphi*ctx->particleGrid.nradius*ctx->particleGrid.ntheta != ctx->npe) {
	ctx->particleGrid.nphi = ctx->npe;
      }
    }
    else if (ctx->particleGrid.nphi*ctx->particleGrid.nradius*ctx->particleGrid.ntheta != ctx->npe) { /* recover */
      if (!ctx->npe%ctx->particleGrid.nphi) {
	ctx->npe_particlePlane = ctx->npe/ctx->particleGrid.nphi;
	ctx->particleGrid.nradius = (int)(sqrt((double)ctx->npe_particlePlane)+0.5);
	ctx->particleGrid.ntheta = ctx->npe_particlePlane/ctx->particleGrid.nradius;
      }
      else {
      }
    }
    if (ctx->particleGrid.nphi*ctx->particleGrid.nradius*ctx->particleGrid.ntheta != ctx->npe) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"particle grids do not work npe (%D) != %D",ctx->npe,ctx->particleGrid.nphi*ctx->particleGrid.nradius*ctx->particleGrid.ntheta);
  }

  PetscPrintf(ctx->wComm,"[%D] npe=%D part=[%D,%D,%D]\n",ctx->rank,ctx->npe,ctx->particleGrid.nphi,ctx->particleGrid.ntheta,ctx->particleGrid.nradius);

  /* particle grids: <= npe, <= num solver planes */
  if (ctx->npe < ctx->particleGrid.nphi) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"num particle planes nphi (%D) > npe (%D)",ctx->particleGrid.nphi,ctx->npe);
  if (ctx->npe%ctx->particleGrid.nphi) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"np=%D not divisible by number of particle planes (nphi) %D",ctx->npe,ctx->particleGrid.nphi);

  if (ctx->npe_particlePlane == -1) ctx->npe_particlePlane = ctx->npe/ctx->particleGrid.nphi;
  if (ctx->npe_particlePlane != ctx->npe/ctx->particleGrid.nphi) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inconsistant number planes (%D), pes (%D), and pe/plane (%D) requested",ctx->particleGrid.nphi,ctx->npe,ctx->npe_particlePlane);

  if (ctx->particleGrid.ntheta*ctx->particleGrid.nradius != ctx->npe_particlePlane) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"%D particle cells/plane != %D pe/plane",ctx->particleGrid.ntheta*ctx->particleGrid.nradius,ctx->npe_particlePlane);
  if (ctx->particleGrid.ntheta*ctx->particleGrid.nradius*ctx->particleGrid.nphi != ctx->npe) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"%D particle cells != %D npe",ctx->particleGrid.ntheta*ctx->particleGrid.nradius*ctx->particleGrid.nphi,ctx->npe);
  ctx->ParticlePlaneIdx = ctx->rank/ctx->npe_particlePlane;
  ctx->particlePlaneRank = ctx->rank%ctx->npe_particlePlane;

  /* PetscPrintf(PETSC_COMM_SELF,"[%D] pe/plane=%D, my plane=%D, my local rank=%D, nphi=%D\n",ctx->rank,ctx->npe_particlePlane,ctx->ParticlePlaneIdx,ctx->particlePlaneRank,ctx->particleGrid.nphi);    */

  /* time integrator */
  ctx->msteps = 1;
  ierr = PetscOptionsInt("-mstep", "Maximum number of time steps", "x2.c", ctx->msteps, &ctx->msteps, NULL);CHKERRQ(ierr);
  ctx->maxTime = 1000000000.;
  ierr = PetscOptionsReal("-maxTime", "Maximum time", "x2.c",ctx->maxTime,&ctx->maxTime,NULL);CHKERRQ(ierr);
  ctx->dt = 1.;
  ierr = PetscOptionsReal("-dt","Time step","x2.c",ctx->dt,&ctx->dt,NULL);CHKERRQ(ierr);
  /* particles */
  ctx->npart_cell = 10; ctx->partBuffSize = 15;
  ierr = PetscOptionsInt("-npart_cell", "Number of particles local (not cell!!!)", "x2.c", ctx->npart_cell, &ctx->npart_cell, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-partBuffSize", "Number of z cells for solver mesh", "x2.c", ctx->partBuffSize, &ctx->partBuffSize, NULL);CHKERRQ(ierr);
  if (ctx->partBuffSize<10*ctx->npart_cell/9) ctx->partBuffSize = 3*ctx->npart_cell/2; /* hack */
  ctx->collisionPeriod = 10;
  ierr = PetscOptionsInt("-collisionPeriod", "Period between collision operators", "x2.c", ctx->collisionPeriod, &ctx->collisionPeriod, NULL);CHKERRQ(ierr);
  if (ctx->partBuffSize < ctx->npart_cell) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"partBuffSize (%D) < npart_cell (%D)",ctx->partBuffSize,ctx->npart_cell);
  ctx->useElectrons = PETSC_FALSE;
  ierr = PetscOptionsBool("-use_electrons", "Include electrons", "x2.c", ctx->useElectrons, &ctx->useElectrons, NULL);CHKERRQ(ierr);
  ctx->max_vpar = 1.;
  ierr = PetscOptionsReal("-max_vpar", "Maximum parallel velocity", "x2.c",ctx->max_vpar,&ctx->max_vpar,NULL);CHKERRQ(ierr);

  {
    size_t sz; FILE *fp;
    ierr = PetscStrcpy(ctx->wall_file,"ITER-wall-geo-67.txt");CHKERRQ(ierr);
    sz = (sizeof(ctx->wall_file)/sizeof((ctx->wall_file)[0]));
    ierr = PetscOptionsString("-wall_file", "Name of wall .txt file", "x2.c", ctx->wall_file, ctx->wall_file, sz, NULL);CHKERRQ(ierr);
    fp = fopen(ctx->wall_file, "r");
    if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wall file %s not found, use -wall_file FILE_NAME",ctx->wall_file);
    for (sz=0;sz<X2_WALL_ARRAY_MAX;sz++) {
      int c = fscanf(fp,"%e %e\n",&ctx->wallEdges[sz][0],&ctx->wallEdges[sz][1]);
      if (c==EOF) break;
    }
    if (sz==X2_WALL_ARRAY_MAX) PetscPrintf(ctx->wComm,"Warning: wall file too large %d, maybe\n",sz);
    ctx->numWallEdges = sz;
  }

  for (isp = ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++ ) ctx->tablecount[isp] = 0;

  ierr = PetscOptionsEnd();

  PetscFunctionReturn(0);
}

/* q: safty factor, should be parameterized */
static PetscReal qsafty(const PetscReal psi)
{
  return 3.*pow(psi,2.0); /* simple q(r) = 3*r^2 */
  /* return 1.; */
}

/* coordinate transformation - simple radial coordinates. Not really cylindrical as r_Minor is radius from plane axis */
#undef __FUNCT__
#define __FUNCT__ "cylindricalToPolPlane"
PetscErrorCode cylindricalToPolPlane(PetscReal a_rMinor, PetscReal a_Z, PetscReal *a_psi, PetscReal *a_theta)
{
  PetscReal psi,theta;
  PetscFunctionBeginUser;

  psi = sqrt(a_rMinor*a_rMinor + a_Z*a_Z);
  if (psi==0.) theta = 0.;
  else {
    theta = a_Z > 0. ? asin(a_Z/psi) : -asin(-a_Z/psi);
    if (a_rMinor < 0) theta = M_PI - theta;
    else if (theta < 0.) theta = theta + 2.*M_PI;
  }
  *a_psi = psi;
  *a_theta = theta;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "polPlaneToCylindrical"
PetscErrorCode polPlaneToCylindrical(PetscReal a_psi, PetscReal a_theta, PetscReal *a_rMinor, PetscReal *a_Z)
{
  PetscFunctionBeginUser;
  *a_rMinor = a_psi*cos(a_theta);
  *a_Z = a_psi*sin(a_theta);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "cylindricalToCart"
PetscErrorCode cylindricalToCart(PetscReal a_R, PetscReal a_Z, PetscReal a_phi, PetscReal a_cart[3])
{
  PetscFunctionBeginUser;
  a_cart[0] = a_R*cos(a_phi);
  a_cart[1] = a_R*sin(a_phi);
  a_cart[2] = a_Z;
  PetscFunctionReturn(0);
}

/* X2GridParticleGetProc_FluxTube: find processor and local flux tube that this point is in
    Input:
     - grid: the particle grid
     - psi: r in [r,theta] coordinates
     - theta:
     - phi: toroidal angle
   Output:
    - pe: process ID
    - elem: element ID
*/
#undef __FUNCT__
#define __FUNCT__ "X2GridParticleGetProc_FluxTube"
PetscErrorCode X2GridParticleGetProc_FluxTube(DM d, const X2GridParticle *grid, PetscReal psi, PetscReal theta, PetscReal phi, PetscReal c[], PetscMPIInt *pe, PetscInt *elem)
{
  const PetscReal rminor=grid->rMinor;
  const PetscReal dphi=2.*M_PI/(PetscReal)grid->nphi;
  const PetscReal dth=2.*M_PI/(PetscReal)grid->ntheta;
  PetscMPIInt planeIdx,irs,iths;
  PetscFunctionBeginUser;

  PetscFunctionBeginUser;
  theta = fmod( theta - qsafty(psi/grid->rMinor)*phi + 20.*M_PI, 2.*M_PI);  /* pull back to referance grid */
  planeIdx = (PetscMPIInt)(phi/dphi)*grid->nradius*grid->ntheta; /* assumeing one particle cell per PE */
  iths = (PetscMPIInt)(theta/dth);                               assert(iths<grid->ntheta);
  irs = (PetscMPIInt)((PetscReal)grid->nradius*psi*psi/(rminor*rminor));assert(irs<grid->nradius);
  *pe = planeIdx + irs*grid->ntheta + iths;
  *elem = 0;
  PetscFunctionReturn(0);
}

/* X2GridParticleGetProc_Solver: find processor and element in solver grid that this point is in
    Input:
     - dm: solver dm
     - coord: Cartesian coordinates
   Output:
     - pe: process ID
     - elem: element ID
*/
#undef __FUNCT__
#define __FUNCT__ "X2GridParticleGetProc_Solver"
PetscErrorCode X2GridParticleGetProc_Solver(DM dm, const X2GridParticle *d, PetscReal d1, PetscReal d2, PetscReal d3, PetscReal coord[], PetscMPIInt *pe, PetscInt *elem)
{
  PetscMPIInt rank;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  /* Matt do your thing here */
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);
  *pe = rank; /* noop */
  *elem = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode zero(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  int i;
  for (i = 0 ; i < dim ; i++) u[i] = 0.;
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "destroyParticles"
static PetscErrorCode destroyParticles(X2Ctx *ctx)
{
  PetscErrorCode ierr;
  PetscInt       isp;
  PetscFunctionBeginUser;
  /* idiom for iterating over particle lists */
  for (isp = ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++ ) { // for each species
    ierr = X2PListDestroy(&ctx->partlists[isp]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#ifdef H5PART
#undef __FUNCT__
#define __FUNCT__ "X2PListWrite"
PetscErrorCode X2PListWrite( X2PList *l, int rank, int npe, MPI_Comm comm, char fname1[], char fname2[])
{
  /* PetscErrorCode ierr; */
  double *x=0,*y=0,*z=0,*v=0;
  h5part_int64_t *id=0,nparticles;
  X2PListPos     pos;
  X2Particle     part;
  PetscErrorCode ierr;
  H5PartFile    *file1,*file2;
  PetscFunctionBeginUser;

  nparticles = X2PListSize(l);
  if (nparticles && (fname1 || fname2)) {
    x=(double*)malloc(nparticles*sizeof(double));
    y=(double*)malloc(nparticles*sizeof(double));
    z=(double*)malloc(nparticles*sizeof(double));
    v=(double*)malloc(nparticles*sizeof(double));
    id=(h5part_int64_t*)malloc(nparticles*sizeof(h5part_int64_t));
  }
  if (fname1) {
    file1 = H5PartOpenFileParallel(fname1,H5PART_WRITE,comm);assert(file1);
    ierr = H5PartFileIsValid(file1);CHKERRQ(ierr);
    ierr = H5PartSetStep(file1, 0);CHKERRQ(ierr);
    nparticles = 0;
    ierr = X2PListGetHead( l, &pos );CHKERRQ(ierr);
    while ( !X2PListGetNext( l, &part, &pos) ) {
      x[nparticles] = part.r*cos(part.phi);
      y[nparticles] = part.r*sin(part.phi);
      z[nparticles] = part.z;
      v[nparticles] = part.vpar;
      id[nparticles] = part.gid;
      nparticles++;
    }
    ierr = H5PartSetNumParticles(file1, nparticles);assert(ierr==H5PART_SUCCESS);
    ierr = H5PartWriteDataFloat64(file1, "x", x);assert(ierr==H5PART_SUCCESS);
    ierr = H5PartWriteDataFloat64(file1, "y", y);assert(ierr==H5PART_SUCCESS);
    ierr = H5PartWriteDataFloat64(file1, "z", z);assert(ierr==H5PART_SUCCESS);
    ierr = H5PartWriteDataInt64(file1, "gid", id);assert(ierr==H5PART_SUCCESS);
    ierr = H5PartCloseFile(file1);assert(ierr==H5PART_SUCCESS);
  }
  if (fname2) {
    file2 = H5PartOpenFileParallel(fname2,H5PART_WRITE,comm);assert(file2);
    ierr = H5PartFileIsValid(file2);CHKERRQ(ierr);assert(ierr==H5PART_SUCCESS);
    ierr = H5PartSetStep(file2, 0);CHKERRQ(ierr);assert(ierr==H5PART_SUCCESS);
    // if (rank!=npe-1 && rank!=npe-2) nparticles = 0; /* just write last (two) proc(s) */
    if (rank>=(npe+1)/2) nparticles = 0; /* just write last (two) proc(s) */
    else {
      nparticles = 0;
      ierr = X2PListGetHead( l, &pos );CHKERRQ(ierr);
      while ( !X2PListGetNext( l, &part, &pos) ) {
        x[nparticles] = part.r*cos(part.phi);
        y[nparticles] = part.r*sin(part.phi);
        z[nparticles] = part.z;
        v[nparticles] = part.vpar;
        id[nparticles] = rank;
        nparticles++;
      }
    }
    ierr = H5PartSetNumParticles( file2, nparticles);assert(ierr==H5PART_SUCCESS);
    ierr = H5PartWriteDataFloat64(file2, "x", x);assert(ierr==H5PART_SUCCESS);
    ierr = H5PartWriteDataFloat64(file2, "y", y);assert(ierr==H5PART_SUCCESS);
    ierr = H5PartWriteDataFloat64(file2, "z", z);assert(ierr==H5PART_SUCCESS);
    ierr = H5PartWriteDataFloat64(file2, "v", v);assert(ierr==H5PART_SUCCESS);
    ierr = H5PartWriteDataInt64(file2, "rank", id);assert(ierr==H5PART_SUCCESS);
    ierr = H5PartCloseFile(file2);assert(ierr==H5PART_SUCCESS);
  }
  if (x) {
    free(x); free(y); free(z); free(id); free(v);
  }

  PetscFunctionReturn(0);
}
#endif

/* shiftParticles: send particles
    Input:
     - ctx: global data
     - isp: species index into sendListTable
     - irk: flag for deposit charge (>=0), or just move (<0)
     - tag: MPI tag to send with
   Input/Output:
     - nIsend: number of sends so far
     - sendListTable: send list hash table array, emptied but meta-data kept
     - particlelist: the list of particle lists to add to
     - slists: array of non-blocking send caches (!ctx->bsp_chuncksize only), cleared
   Output:
*/
#undef __FUNCT__
#define __FUNCT__ "shiftParticles"
PetscErrorCode shiftParticles( const X2Ctx *ctx, X2SendList *sendListTable, const PetscInt isp, const PetscInt irk, PetscInt *const nIsend,
                               X2PList *particlelist, X2ISend slist[], PetscInt tag )
{
  PetscErrorCode ierr;
  const int part_dsize = sizeof(X2Particle)/sizeof(double);
  int  sz,ii,jj,kk,mm,idx;
  Vec vVec,xVec;
  PetscScalar x[3],rone=1.;
  DM dm;
  DM_PICell *dmpi;
  PetscInt dim,one=1;
  PetscFunctionBeginUser;

  dmpi = (DM_PICell *) ctx->dm->data;
  dm = dmpi->dmplex;
  if (irk>=0) {
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,one,&vVec);CHKERRQ(ierr);
    ierr = VecSet(vVec,rone);CHKERRQ(ierr); /* dummy density now */
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,one,dim,x,   &xVec);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(ctx->events[1],0,0,0,0);CHKERRQ(ierr);
#endif
  if ( ctx->bsp_chuncksize ) { /* use BSP */
    PetscMPIInt  nto,*fromranks,chucksz=s_chunksize;
    PetscMPIInt *toranks;
    X2Particle  *fromdata,*todata,*pp;
    PetscMPIInt  nfrom;

    /* count send  */
    for (ii=0,nto=0;ii<ctx->tablesize;ii++) {
      if (sendListTable[ii].plist[isp].data_size != 0) {
	if ((sz=X2PListSize(&sendListTable[ii].plist[isp])) > 0) {
	  for (jj=0 ; jj<sz ; jj += chucksz) nto++;
	}
      }
    }
    /* make to ranks & data */
    ierr = PetscMalloc1(nto, &toranks);CHKERRQ(ierr);
    ierr = PetscMalloc1(s_chunksize*nto, &todata);CHKERRQ(ierr);
    for (ii=0,nto=0,pp=todata;ii<ctx->tablesize;ii++) {
      if (sendListTable[ii].plist[isp].data_size != 0) {
	if ((sz=X2PListSize(&sendListTable[ii].plist[isp])) > 0) {
	  /* empty list */
	  for (jj=0, mm=0 ; jj<sz ; jj += chucksz) {
	    toranks[nto++] = sendListTable[ii].proc;
	    for (kk=0 ; kk<chucksz && mm < sz; kk++, mm++) {
	      *pp++ = sendListTable[ii].plist[isp].data[mm];
	    }
	  }
	  assert(mm==sz);
	  while (kk++ < chucksz) { /* pad with zeros */
	    pp->gid = 0;
	    pp++;
	  }
          /* get ready for next round */
	  ierr = X2PListClear( &sendListTable[ii].plist[isp] );CHKERRQ(ierr);
          assert(X2PListSize(&sendListTable[ii].plist[isp])==0);
          assert(sendListTable[ii].plist[isp].data_size);
	} /* a list */
      }
    }

    /* do it */
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
    ierr = PetscCommBuildTwoSided( ctx->wComm, chucksz*part_dsize, MPI_DOUBLE, nto, toranks, (double*)todata,
				   &nfrom, &fromranks, &fromdata);
    CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
    for (ii=0, pp = fromdata ; ii<nfrom ; ii++) {
      for (jj=0 ; jj<chucksz ; jj++, pp++) {
	if (pp->gid > 0) {
	  ierr = X2PListAdd( particlelist, pp);CHKERRQ(ierr);
	  /* add density to RHS */
          if (irk>=0) {
            ierr = cylindricalToCart(pp->r, pp->z, pp->phi, x);CHKERRQ(ierr);
            ierr = DMPICellAddSource(ctx->dm, xVec, vVec);CHKERRQ(ierr);
          }
        }
      }
    }

    ierr = PetscFree(todata);CHKERRQ(ierr);
    ierr = PetscFree(fromranks);CHKERRQ(ierr);
    ierr = PetscFree(fromdata);CHKERRQ(ierr);
    ierr = PetscFree(toranks);CHKERRQ(ierr);
  }
  else { /* non-blocking consensus, buggy on my OSX */
    X2Particle *data;
    PetscBool   done=PETSC_FALSE,bar_act=PETSC_FALSE;
    MPI_Request ib_request;
    PetscInt    numSent;
    MPI_Status  status;
    int flag;
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
    /* send lists */
    for (ii=0;ii<ctx->tablesize;ii++) {
      if (sendListTable[ii].plist[isp].data_size != 0) {
	if ((sz=X2PListSize(&sendListTable[ii].plist[isp])) > 0) {
	  if (*nIsend==X2PROCLISTSIZE) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"process send table too small (%D)",X2PROCLISTSIZE);
	  slist[*nIsend].proc = sendListTable[ii].proc;
	  slist[*nIsend].data = sendListTable[ii].plist[isp].data; /* cache data */
	  /* send and reset - we can just send this because it is dense */
	  ierr = MPI_Isend((void*)slist[*nIsend].data,sz*part_dsize,MPI_DOUBLE,slist[*nIsend].proc,tag,ctx->wComm,&slist[*nIsend].request);
	  CHKERRQ(ierr);
	  (*nIsend)++;
          /* ready for next round, save meta-data  */
	  ierr = X2PListClear( &sendListTable[ii].plist[isp] );CHKERRQ(ierr);
	  assert(sendListTable[ii].plist[isp].data_size == s_chunksize);
	  ierr = PetscMalloc1(s_chunksize, &sendListTable[ii].plist[isp].data);CHKERRQ(ierr);
	  assert(!(sendListTable[ii].plist[isp].data_size != 0 && (sz=X2PListSize(&sendListTable[ii].plist[isp]) ) > 0));
	}
      }
      /* else - empty list  */
    }
    numSent = *nIsend; /* size of send array */
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
    /* process receives - non-blocking consensus */
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(ctx->events[2],0,0,0,0);CHKERRQ(ierr);
#endif
    /* process recieves - non-blocking consensus */
    ierr = PetscMalloc1(s_chunksize, &data);CHKERRQ(ierr);
    while (!done) {
      /* probe for incoming */
      do {
	ierr = MPI_Iprobe(MPI_ANY_SOURCE, tag, ctx->wComm, &flag, &status);CHKERRQ(ierr);
	if (flag) {
	  MPI_Get_count(&status, MPI_DOUBLE, &sz); assert(sz<=s_chunksize*part_dsize);
	  ierr = MPI_Recv((void*)data,sz,MPI_DOUBLE,status.MPI_SOURCE,tag,ctx->wComm,&status);CHKERRQ(ierr);
	  MPI_Get_count(&status, MPI_DOUBLE, &sz);
	  sz = sz/part_dsize;
	  for (jj=0;jj<sz;jj++) {
            ierr = X2PListAdd( particlelist, &data[jj]);CHKERRQ(ierr);
	    /* add density to RHS */
            if (irk>=0) {
              ierr = cylindricalToCart(data[jj].r, data[jj].z, data[jj].phi, x);CHKERRQ(ierr);
              ierr = DMPICellAddSource(ctx->dm, xVec, vVec);CHKERRQ(ierr);
            }
          }
	}
      } while (flag);

      if (bar_act) {
	ierr = MPI_Test(&ib_request, &flag, &status);CHKERRQ(ierr);
	if (flag) done = PETSC_TRUE;
      }
      else {
	/* test for sends */
	for (idx=0;idx<numSent;idx++){
	  if (slist[idx].data) {
	    ierr = MPI_Test( &slist[idx].request, &flag, &status);CHKERRQ(ierr);
	    if (flag) {
	      ierr = PetscFree(slist[idx].data);CHKERRQ(ierr);
	      slist[idx].data = 0;
	    }
	    else break; /* not done yet */
	  }
	}
	if (idx==numSent) {
	  bar_act = PETSC_TRUE;
	  ierr = MPI_Ibarrier(ctx->wComm, &ib_request);CHKERRQ(ierr);
	}
      }
    } /* non-blocking consensus */
    ierr = PetscFree(data);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(ctx->events[2],0,0,0,0);CHKERRQ(ierr);
#endif
  } /* switch for BPS */
  if (irk>=0) {
    VecDestroy(&vVec);
    VecDestroy(&xVec);
  }
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(ctx->events[1],0,0,0,0);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

/* processParticle: move particles if (sendListTable) , push if (irk>=0)
    Input:
     - dt: time step
     - tag: MPI tag to send with
     - irk: RK stage (<0 for send only)
     - get_proc: function to get processor of point
   Input/Output:
     - ctx: global data
     - lists: list of particle lists
   Output:
     - sendListTable: send list hash table, null if not sending (irk==0)
*/
#undef __FUNCT__
#define __FUNCT__ "processParticles"
static PetscErrorCode processParticles( X2Ctx *ctx, const PetscReal dt, X2SendList *sendListTable, const PetscInt tag,
					const int irk, const int istep, PetscErrorCode (*get_proc)(DM, const X2GridParticle *, PetscReal, PetscReal, PetscReal, PetscReal[], PetscMPIInt *, PetscInt *))
{
  X2GridParticle *grid = &ctx->particleGrid;
  DM_PICell *dmpi = (DM_PICell *) ctx->dm->data;
  DM dm = dmpi->dmplex;
  PetscReal   r,z,psi,theta,dphi,rmaj=grid->rMajor,rminor=grid->rMinor;
  PetscMPIInt pe,hash,ii;
  X2Particle  part;
  X2PListPos  pos;
  PetscErrorCode ierr;
  const int part_dsize = sizeof(X2Particle)/sizeof(double);
  Vec gradVec,xVec,vVec;
  PetscScalar x[3],grad[3],rone=1.;
  PetscInt isp,order=1,nslist,nlistsTot,idx,dim,one=1;
  int origNlocal,nmoved;
  X2ISend slist[X2PROCLISTSIZE];
  PetscFunctionBeginUser;

  nslist = 0;
  nmoved = 0;

  /* ierr = X2PListCompress( list );CHKERRQ(ierr); */

  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,one,dim,grad,&gradVec);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,one,dim,x,   &xVec);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,one,&vVec);CHKERRQ(ierr);
  ierr = VecSet(vVec,rone);CHKERRQ(ierr); /* dummy density now */

  nlistsTot = origNlocal = 0;
  for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++) { // for each species
    X2PList *list = &ctx->partlists[isp];
    origNlocal += X2PListSize(list);
    /* for (pos=0 ; pos < list->size ; pos++) { */
    /* part = list->data[pos]; */
    ierr = X2PListGetHead( list, &pos );CHKERRQ(ierr);
    while ( !X2PListGetNext(list, &part, &pos) ) {
      ierr = cylindricalToCart(part.r, part.z, part.phi, x);CHKERRQ(ierr);
      if (irk>=0) { /* push */
        /* get E */
        ierr = DMPICellGetJet(ctx->dm, xVec, order, gradVec);CHKERRQ(ierr); /* not used */
        /* simple push: theta = theta + q*dphi .... grad not used */
        ierr = cylindricalToPolPlane( part.r - rmaj, part.z, &psi, &theta );CHKERRQ(ierr);
        dphi = (dt*part.vpar)/(2.*M_PI*part.r);  /* toroidal step */
        part.phi += dphi;
        part.phi = fmod( part.phi + 20.*M_PI, 2.*M_PI);
        theta += qsafty(psi/rminor)*dphi;  /* twist */
        theta = fmod( theta + 20.*M_PI, 2.*M_PI);
        ierr = polPlaneToCylindrical( psi, theta, &r, &z);CHKERRQ(ierr); /* time spent here */
        part.r = rmaj + r;
        part.z = z;
      }
      else { /* irk < 0: just moving particles, no push */
        ierr = cylindricalToPolPlane( part.r - rmaj, part.z, &psi, &theta );CHKERRQ(ierr); /* not needed for solver move */
      }
      /* else -- just communicate */
      /* see if need communication, add density if not, add to communication list if so */
      ierr = get_proc(dm, grid, psi, theta, part.phi, x, &pe, &idx);CHKERRQ(ierr);
      if (!sendListTable || pe==ctx->rank) { /* don't move */
        /* add density to RHS */
        if (irk>=0) {
          ierr = cylindricalToCart(part.r, part.z, part.phi, x);CHKERRQ(ierr);
          ierr = DMPICellAddSource(ctx->dm, xVec, vVec);CHKERRQ(ierr);
          if (sendListTable) {
            ierr = X2PListSetAt( list, pos, &part );CHKERRQ(ierr); /* not moved and final step so write back */
          }
        }
      }
      else { /* move: sendListTable && off proc */
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventBegin(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
        /* add to list to send, find list with manual has table lookup, send full lists */
        hash = (pe*593)%ctx->tablesize; /* hash function */
        for (ii=0;ii<ctx->tablesize;ii++){
          if (sendListTable[hash].plist[isp].data_size==0) {
            ierr = X2PListCreate(&sendListTable[hash].plist[isp],s_chunksize);CHKERRQ(ierr);
            sendListTable[hash].proc = pe;
            ctx->tablecount[isp]++;
            if (ctx->tablecount[isp]==ctx->tablesize) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Hash table too small (%D) - need to rehash!!!",ctx->tablesize);
          }
          if (sendListTable[hash].proc==pe) { /* found hash table entry */
            if (X2PListSize(&sendListTable[hash].plist[isp])==s_chunksize) {
              if (ctx->bsp_chuncksize) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"cache too small (%D) for BSP TwoSided communication",s_chunksize);
              /* send and reset - we can just send this because it is dense, but no species data */
              if (nslist==X2PROCLISTSIZE) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"process send table too small (%D)",X2PROCLISTSIZE);
              slist[nslist].data = sendListTable[hash].plist[isp].data; /* cache data */
              sendListTable[hash].plist[isp].data = 0; /* clear for safty, make ready for more */
              slist[nslist].proc = pe;
              ierr = MPI_Isend( (void*)slist[nslist].data,s_chunksize*part_dsize,MPI_DOUBLE,pe,tag+isp,ctx->wComm,&slist[nslist].request);
              CHKERRQ(ierr);
              nslist++;
              /* ready for more */
              ierr = X2PListCreate(&sendListTable[hash].plist[isp],s_chunksize);CHKERRQ(ierr);
              assert(sendListTable[hash].plist[isp].data_size == s_chunksize);
            }
            /* add to list - pass this in as a function to a function? */
            ierr = X2PListAdd(&sendListTable[hash].plist[isp],&part);CHKERRQ(ierr);assert(part.gid>0);
            ierr = X2PListRemoveAt(list,pos);CHKERRQ(ierr);
            nmoved++;
            break;
          }
          if (++hash == ctx->tablesize) hash=0;
        }
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventEnd(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
      }
    }
    /* finish sends */
    if (sendListTable) {
      /* send recv */
      ierr = shiftParticles(ctx, sendListTable, isp, irk, &nslist, &ctx->partlists[isp], slist, tag+isp );CHKERRQ(ierr);
    }

    nlistsTot += nslist;
    nslist = 0;
  } /* isp */
  /* diagnostics */
  if (sendListTable) {
    MPI_Datatype mtype;
    PetscInt rb[3], sb[3] = {origNlocal, nmoved, nlistsTot};
    PetscDataTypeToMPIDataType(PETSC_INT,&mtype);
    ierr = MPI_Allreduce(sb, rb, 3, mtype, MPI_SUM, ctx->wComm);CHKERRQ(ierr);
    PetscPrintf(ctx->wComm,
                "%d) %s %D local particles, %D global, %g %% total particles moved in %D messages total (to %D processors local)\n",
                istep+1,irk<0 ? "processed" : "pushed", origNlocal, rb[0], 100.*(double)rb[1]/(double)rb[0], rb[2], ctx->tablecount[1]);
#ifdef H5PART
    if (irk>=0) {
      if (ctx->debug>1) {
        for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++ ) {
          char  fname1[256],fname2[256];
          /* hdf5 output */
          sprintf(fname1,"particles_sp%d_time%05d.h5part",isp,istep+1);
          sprintf(fname2,"sub_rank_particles_sp%d_time%05d.h5part",isp,istep+1);
          /* write */
          ierr = X2PListWrite(&ctx->partlists[isp], ctx->rank, ctx->npe, ctx->wComm, fname1, fname2);CHKERRQ(ierr);
        }
      }
    }
#endif
  }
  else PetscPrintf(ctx->wComm,"%d) %s %D local particles\n",istep+1,irk<0 ? "processed" : "pushed",origNlocal);

  VecDestroy(&gradVec);
  VecDestroy(&xVec);
  VecDestroy(&vVec);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "createParticles"
static PetscErrorCode createParticles(X2Ctx *ctx)
{
  PetscErrorCode ierr;
  PetscInt isp,nCellsLoc,my0,irs,iths,gid,ii,np,dim,one=1;
  const PetscReal dth=(2.*M_PI)/(PetscReal)ctx->particleGrid.ntheta,rone=1.;
  const PetscReal dphi=2.*M_PI/(PetscReal)ctx->particleGrid.nphi,rmin=ctx->particleGrid.rMinor; /* rmin for particles < rmin */
  const PetscReal phi1 = (PetscReal)ctx->ParticlePlaneIdx*dphi + 1.e-8,rmaj=ctx->particleGrid.rMajor;
  const PetscInt  nPartCells_plane = ctx->particleGrid.ntheta*ctx->particleGrid.nradius; /* nPartCells_plane == ctx->npe_particlePlane */
  const PetscReal dx = pow( (M_PI*rmin*rmin/4.0) * rmaj*2.*M_PI / (PetscReal)(ctx->npe*ctx->npart_cell), 0.333); /* lenth of a particle, approx. */
  X2Particle particle;
  Vec vVec,xVec;
  PetscScalar x[3];
  DM dm;
  DM_PICell *dmpi;
  PetscFunctionBeginUser;

  /* Create vector and get pointer to data space */
  dmpi = (DM_PICell *) ctx->dm->data;
  dm = dmpi->dmplex;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim!=3) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"wrong dimension (3) = %D",dim);
  ierr = VecCreateSeq(PETSC_COMM_SELF,one,&vVec);CHKERRQ(ierr);
  ierr = VecSet(vVec,rone);CHKERRQ(ierr); /* dummy density now */
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,one,dim,x,   &xVec);CHKERRQ(ierr);

  /* setup particles - lexigraphic partition of cells (np local cells) */
  nCellsLoc = nPartCells_plane/ctx->npe_particlePlane; /* = 1; nPartCells_plane == ctx->npe_particlePlane */
  my0 = ctx->particlePlaneRank*nCellsLoc;              /* cell index in plane == particlePlaneRank */
  gid = (my0 + ctx->ParticlePlaneIdx*nPartCells_plane)*ctx->npart_cell; /* based particle ID */
  if (ctx->ParticlePlaneIdx == ctx->npe_particlePlane-1){
    nCellsLoc = nPartCells_plane - nCellsLoc*(ctx->npe_particlePlane-1);
  }
  assert(nCellsLoc==1);
  /* my first cell index */
  srand(ctx->rank);
  for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++ ) {
    iths = my0%ctx->particleGrid.ntheta;
    irs = my0/ctx->particleGrid.ntheta;
    {
      const PetscReal r1 = sqrt(((PetscReal)irs      /(PetscReal)ctx->particleGrid.nradius)*rmin*rmin) +       1.e-12*rmin;
      const PetscReal dr = sqrt((((PetscReal)irs+1.0)/(PetscReal)ctx->particleGrid.nradius)*rmin*rmin) - (r1 - 1.e-12*rmin);
      const PetscReal th1 = (PetscReal)iths*dth + 1.e-12*dth;
      /* create list */
      ierr = X2PListCreate(&ctx->partlists[isp],ctx->partBuffSize);CHKERRQ(ierr);
#define X2NDIG 100000
      /* create each particle */
      //for (int i=0;i<ctx->npart_cell;i++) {
      for (np=0 ; np<ctx->npart_cell; /* void */ ) {
	PetscReal theta0,r,z;
	const PetscReal psi = r1 + (PetscReal)(rand()%X2NDIG+1)/(PetscReal)(X2NDIG+1)*dr;
	const PetscReal qsaf = qsafty(psi/ctx->particleGrid.rMinor);
	const PetscInt NN = (PetscInt)(dth*psi/dx) + 1;
	const PetscReal dth2 = dth/(PetscReal)NN - 1.e-12*dth;
	for ( ii = 0, theta0 = th1 + (PetscReal)(rand()%X2NDIG)/(PetscReal)X2NDIG*dth2;
	      ii < NN && np<ctx->npart_cell ;
	      ii++, theta0 += dth2, np++ ) {
	  PetscReal zmax,maxe=ctx->max_vpar*ctx->max_vpar,zdum,mass=1.,b=1.,charge=1.,t=1.;
          PetscScalar v=1.,vpar;
          const PetscReal phi = phi1 + (PetscReal)(rand()%X2NDIG)/(PetscReal)X2NDIG*dphi;
	  const PetscReal thetap = theta0 + qsaf*phi; /* push forward to follow fieldlines */
	  ierr = polPlaneToCylindrical(psi, thetap, &r, &z);CHKERRQ(ierr);
	  r += rmaj;

	  /* v_parallel from random number */
	  zmax = 1.0 - exp(-maxe);
	  zdum = zmax*(PetscReal)(rand()%X2NDIG)/(PetscReal)X2NDIG;
	  v= sqrt(-2.0/mass*log(1.0-zdum)*t);
	  v= v*cos(M_PI*(PetscReal)(rand()%X2NDIG)/(PetscReal)X2NDIG);
	  /* vshift= v + up ! shift of velocity */
	  vpar = v/b*mass/charge;
	  vpar *= 208.3333; /* fudge factor to get it to fit input */
	  ierr = X2ParticleCreate(&particle,++gid,r,z,phi,vpar);CHKERRQ(ierr);
	  ierr = X2PListAdd(&ctx->partlists[isp],&particle);CHKERRQ(ierr);
          /* debug, particles are created in a flux tube */
          {
            PetscMPIInt pe; PetscInt id; PetscReal dummy[3];
            ierr = X2GridParticleGetProc_FluxTube(NULL,&ctx->particleGrid,psi,thetap,phi,dummy,&pe,&id);CHKERRQ(ierr);
            if(pe != ctx->rank){
              PetscPrintf(PETSC_COMM_SELF,"[%D] ERROR particle in proc %d r=%e:%e:%e theta=%e:%e:%e phi=%e:%e:%e\n",ctx->rank,pe,r1,psi,r1+dr,th1,thetap,th1+dth,phi1,phi,phi1+dphi);
              SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB," created particle for proc %D",pe);
            }
          }
	} /* theta */
      }
      iths++;
      if (iths==ctx->particleGrid.ntheta) { iths = 0; irs++; }
    } /* cells */
  } /* species */
  /* move back to solver space and make density vector */
  {
    PetscInt tag = 90, istep=-1, idx;
    X2SendList *sendListTable;
    /* init send tables */
    ierr = PetscMalloc1(ctx->tablesize,&sendListTable);CHKERRQ(ierr);
    for (idx=0;idx<ctx->tablesize;idx++) {
      for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++) {
        sendListTable[idx].plist[isp].data_size = 0; /* init */
      }
    }
    /* fake time step will add density to RHS */
    ierr = processParticles(ctx, 0.0, sendListTable, tag, 0, istep, X2GridParticleGetProc_Solver);
    CHKERRQ(ierr);
    ierr = PetscFree(sendListTable);CHKERRQ(ierr);
  }
  VecDestroy(&vVec);
  VecDestroy(&xVec);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "go"
PetscErrorCode go( X2Ctx *ctx )
{
  PetscErrorCode ierr;
  PetscInt       istep,tag;
  int            irk,idx,isp;
  PetscReal      time,dt;
  X2SendList    *sendListTable;
  PetscFunctionBeginUser;

  /* init send tables */
  ierr = PetscMalloc1(ctx->tablesize,&sendListTable);CHKERRQ(ierr);
  for (idx=0;idx<ctx->tablesize;idx++) {
    for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++) {
      sendListTable[idx].plist[isp].data_size = 0; /* init */
    }
  }

  /* hdf5 output - init */
#ifdef H5PART
  if (ctx->debug>1) {
    for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++) { // for each species
      char  fname1[256],fname2[256];
      sprintf(fname1,"particles_sp%d_time%05d.h5part",isp,0);
      sprintf(fname2,"sub_rank_particles_sp%d_time%05d.h5part",isp,0);
      /* write */
      ierr = X2PListWrite(&ctx->partlists[isp], ctx->rank, ctx->npe, ctx->wComm, fname1, fname2);CHKERRQ(ierr);
    }
  }
#endif
  /* main time step loop */
  ierr = MPI_Barrier(ctx->wComm);CHKERRQ(ierr);
  for ( istep=0, time=0., tag = 100;
	istep < ctx->msteps && time < ctx->maxTime;
	istep++, time += ctx->dt, tag += X2_NION + (ctx->useElectrons ? 1 : 0) ) {

    /* do collisions */
    if (((istep+1)%ctx->collisionPeriod)==0) {
      /* move to flux tube space */
      ierr = processParticles(ctx, 0.0, sendListTable, tag,                                         -1, istep, X2GridParticleGetProc_FluxTube);
      CHKERRQ(ierr);
      /* call collision method - todo */
#ifdef H5PART
      if (ctx->debug>0) {
        for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++) { // for each species
          char  fname1[256], fname2[256];
          sprintf(fname1,         "particles_sp%d_time%05d_fluxtube.h5part",isp,istep);
          sprintf(fname2,"sub_rank_particles_sp%d_time%05d_fluxtube.h5part",isp,istep);
          /* write */
          ierr = X2PListWrite(&ctx->partlists[isp], ctx->rank, ctx->npe, ctx->wComm, fname1, fname2);CHKERRQ(ierr);
        }
      }
#endif
      /* move back to solver space */
      ierr = processParticles(ctx, 0.0, sendListTable, tag + X2_NION + (ctx->useElectrons ? 1 : 0), -1, istep, X2GridParticleGetProc_Solver);
      CHKERRQ(ierr);
    }

    /* very crude explicit RK */
    irk=0;
    dt = ctx->dt;

    /* solve for potential, density being assembled is an invariant */
    ierr = DMPICellSolve( ctx->dm );CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(ctx->events[0],0,0,0,0);CHKERRQ(ierr);
#endif
    /* process particles: push, move */
    ierr = processParticles(ctx, dt, sendListTable, tag, irk, istep, X2GridParticleGetProc_Solver);
    CHKERRQ(ierr);
    tag += X2_NION + (ctx->useElectrons ? 1 : 0);
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(ctx->events[0],0,0,0,0);CHKERRQ(ierr);
#endif
  } /* time step */

  /* clean up */
  for (idx=0;idx<ctx->tablesize;idx++) {
    for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++) {
      if (sendListTable[idx].plist[isp].data_size != 0 ) {
	ierr = X2PListDestroy( &sendListTable[idx].plist[isp] );CHKERRQ(ierr);
      }
    }
  }

  ierr = PetscFree(sendListTable);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* == Defining a base plex for a torus, which looks like a rectilinear donut, and a mapping that turns it into a conventional round donut == */

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreatePICellTorus"
static PetscErrorCode DMPlexCreatePICellTorus (MPI_Comm comm, X2GridParticle *params, DM *dm)
{
  PetscMPIInt    rank;
  PetscInt       numCells = 0;
  PetscInt       numVerts = 0;
  PetscReal      rMajor   = params->rMajor;
  PetscReal      rMinor   = params->rMinor;
  PetscReal      innerMult = params->innerMult;
  PetscInt       numMajor = params->numMajor;
  int           *flatCells = NULL;
  double         *flatCoords = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    numCells = numMajor * 5;
    numVerts = numMajor * 8;
    ierr = PetscMalloc2(numCells * 8,&flatCells,numVerts * 3,&flatCoords);CHKERRQ(ierr);
    {
      double (*coords)[8][3] = (double (*) [8][3]) flatCoords;
      PetscInt i;

      for (i = 0; i < numMajor; i++) {
        PetscInt j;
        double cosphi, sinphi;

        cosphi = cos(2 * M_PI * i / numMajor);
        sinphi = sin(2 * M_PI * i / numMajor);

        for (j = 0; j < 8; j++) {
          double r, z;
          double mult = (j < 4) ? innerMult : 1.;

          r = rMajor + mult * rMinor * cos(j * M_PI_2);
          z = mult * rMinor * sin(j * M_PI_2);

          coords[i][j][0] = cosphi * r;
          coords[i][j][1] = sinphi * r;
          coords[i][j][2] = z;
        }
      }
    }
    {
      int (*cells)[5][8] = (int (*) [5][8]) flatCells;
      PetscInt i;

      for (i = 0; i < numMajor; i++) {
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
  ierr = PetscObjectSetOptionsPrefix((PetscObject) *dm, "torus_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscReal findWallPoint(const float wallEdges[][2], const int numWallEdges)
{
  PetscReal rt = 2;


  return rt;
}

static void PICellCircleInflate(PetscReal r, PetscReal innerMult, PetscReal x, PetscReal y,
                                PetscReal *outX, PetscReal *outY, PetscReal rMinor, const float wallEdges[][2], const int numWallEdges)
{
  PetscReal l       = x + y;
  PetscReal rfrac   = l / r;

  if (rfrac >= innerMult) {
    PetscReal phifrac = l ? (y / l) : 0.5;
    PetscReal phi     = phifrac * M_PI_2;
    PetscReal cosphi  = cos(phi);
    PetscReal sinphi  = sin(phi);
    /* get real rt = d/rMinor from phi tokamak */
    PetscReal rt = findWallPoint(wallEdges, numWallEdges)/rMinor;
    PetscReal isect   = innerMult / (cosphi + sinphi);
    PetscReal outfrac = (1. - rfrac) / (1. - innerMult);

    rfrac = pow(innerMult,outfrac);

    outfrac = (1. - rfrac) / (1. - innerMult);

    /* *outX = r * (outfrac * isect + (1. - outfrac)) * cosphi; */
    /* *outY = r * (outfrac * isect + (1. - outfrac)) * sinphi; */
    *outX = r * (outfrac * isect + (1. - outfrac)*rt) * cosphi;
    *outY = r * (outfrac * isect + (1. - outfrac)*rt) * sinphi;
  }
  else {
    PetscReal halfdiffl  = (r * innerMult - l) / 2.;
    PetscReal phifrac    = (y + halfdiffl) / (r * innerMult);
    PetscReal phi        = phifrac * M_PI_2;
    PetscReal m          = y - x;
    PetscReal halfdiffm  = (r * innerMult - m) / 2.;
    PetscReal thetafrac  = (y + halfdiffm) / (r * innerMult);
    PetscReal theta      = thetafrac * M_PI_2;
    PetscReal cosphi     = cos(phi);
    PetscReal sinphi     = sin(phi);
    PetscReal ymxcoord   = sinphi / (cosphi + sinphi);
    PetscReal costheta   = cos(theta);
    PetscReal sintheta   = sin(theta);
    PetscReal xpycoord   = sintheta / (costheta + sintheta);

    *outX = r * innerMult * (xpycoord - ymxcoord);
    *outY = r * innerMult * (ymxcoord + xpycoord - 1.);
  }
}

#undef __FUNCT__
#define __FUNCT__ "GeometryPICellTorus"
static PetscErrorCode GeometryPICellTorus(PetscInt dim, const PetscReal abc[], PetscReal xyz[], void *a_ctx)
{
  X2Ctx *ctx = (X2Ctx*)a_ctx;
  X2GridParticle *params = &ctx->particleGrid;
  PetscReal rMajor    = params->rMajor;
  PetscReal rMinor    = params->rMinor;
  PetscReal innerMult = params->innerMult;
  PetscInt  numMajor  = params->numMajor;
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
  inPhi = (inPhi < 0.) ? (inPhi + 2. * M_PI) : inPhi;
  i = (inPhi * numMajor) / (2. * M_PI);
  i = PetscMin(i,numMajor - 1);
  leftPhi =  (i *        2. * M_PI) / numMajor;
  midPhi  = ((i + 0.5) * 2. * M_PI) / numMajor;
  cosMidPhi  = cos(midPhi);
  sinMidPhi  = sin(midPhi);
  cosLeftPhi = cos(leftPhi);
  sinLeftPhi = sin(leftPhi);
  secHalf    = 1. / cos(M_PI / numMajor);

  rhat = (a * cosMidPhi + b * sinMidPhi);
  r    = secHalf * rhat;
  dist = secHalf * (-a * sinLeftPhi + b * cosLeftPhi);
  fulldist = 2. * sin(M_PI / numMajor) * r;
  outPhi = ((i + (dist/fulldist)) * 2. * M_PI) / numMajor;
  /* solve r * (cosLeftPhi * _i + sinLeftPhi * _j) + dist * (nx * _i + ny * _j) = a * _i + b * _j;
   *
   * (r * cosLeftPhi + dist * nx) = a;
   * (r * sinLeftPhi + dist * ny) = b;
   *
   * r    = idet * ( a * ny         - b * nx);
   * dist = idet * (-a * sinLeftPhi + b * cosLeftPhi);
   * idet = 1./(cosLeftPhi * ny - sinLeftPhi * nx) = sec(Pi/numMajor);
   */
  r -= rMajor; /* now centered inside torus */
  {
    PetscReal absR, absZ;

    absR = PetscAbsReal(r);
    absZ = PetscAbsReal(z);
    PICellCircleInflate(rMinor,innerMult,absR,absZ,&absR,&absZ,rMinor,ctx->wallEdges,ctx->numWallEdges);
    r = (r > 0) ? absR : -absR;
    z = (z > 0) ? absZ : -absZ;
  }
  r += rMajor; /* centered back at the origin */

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
  DM             dm;
  PetscInt       dim;
  PetscFE        fe; /* FV might be better */
  Mat            J;
  PetscFunctionBeginUser;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ctx.currevent = 0;
  ierr = PetscLogEventRegister("Particle proc",0,&ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 0 */
  ierr = PetscLogEventRegister("Part. Send",0,&ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 1 */
  ierr = PetscLogEventRegister("Part. Recv",0,&ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 2 */
  ierr = PetscLogEventRegister("CreateMesh", DM_CLASSID, &ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 3 */
  ierr = PetscLogEventRegister("TwoSides", 0, &ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 4 */
  ierr = PetscLogEventRegister("Aux", 0, &ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 5 */
#endif

  ierr = PetscCommDuplicate(PETSC_COMM_WORLD,&ctx.wComm,NULL);CHKERRQ(ierr);
  ierr = ProcessOptions( &ctx );CHKERRQ(ierr);

  /* construct DMs */
  ierr = PetscLogEventBegin(ctx.events[3],0,0,0,0);CHKERRQ(ierr);
  ierr = DMCreate(ctx.wComm, &ctx.dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(ctx.dm, &ctx);CHKERRQ(ierr);
  ierr = DMSetType(ctx.dm, DMPICELL);CHKERRQ(ierr); /* creates (DM_PICell *) dm->data */
  /* setup solver grid, this should go in DMCreate_PICell? */
  ierr = DMPlexCreatePICellTorus(ctx.wComm,&ctx.particleGrid,&dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  /* setup Discretization */
  ierr = PetscMalloc(1 * sizeof(PetscErrorCode (*)(PetscInt,const PetscReal [],PetscInt,PetscScalar*,void*)),&ctx.BCFuncs);
  CHKERRQ(ierr);
  ctx.BCFuncs[0] = zero;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, 1, PETSC_FALSE, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "potential");CHKERRQ(ierr);
  {
    DMLabel label;
    PetscDS prob;
    PetscInt id = 1;
    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
    ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
    ierr = DMCreateLabel(dm, "boundary");CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "boundary", &label);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(dm, label);CHKERRQ(ierr);
    ierr = DMAddBoundary(dm, PETSC_TRUE, "wall", "boundary", 0, 0, NULL, (void (*)()) ctx.BCFuncs, 1, &id, &ctx);
    CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  {
    char      convType[256];
    PetscBool flg;
    ierr = PetscOptionsBegin(ctx.wComm, "", "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-torus_dm_type","Convert DMPlex to another format","x2.c",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
    if (flg) {
      DM dmConv;
      ierr = DMConvert(dm,convType,&dmConv);CHKERRQ(ierr);
      if (dmConv) {
        const char *prefix;
        PetscBool isForest;
        ierr = PetscObjectGetOptionsPrefix((PetscObject)dm,&prefix);CHKERRQ(ierr);
        ierr = PetscObjectSetOptionsPrefix((PetscObject)dmConv,prefix);CHKERRQ(ierr);
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        dm   = dmConv;
        ierr = DMIsForest(dm,&isForest);CHKERRQ(ierr);
        if (isForest) {
          ierr = DMForestSetBaseCoordinateMapping(dm,GeometryPICellTorus,&ctx);CHKERRQ(ierr);
        }
      }
    }
  }
  dmpi = (DM_PICell *) ctx.dm->data;
  /* setup DM */
  dmpi->dmplex = dm;
  ierr = DMSetFromOptions( ctx.dm );CHKERRQ(ierr);
  ierr = DMSetUp( ctx.dm );CHKERRQ(ierr); /* set all up & build initial grid */
  /* create SNESS */
  ierr = SNESCreate( ctx.wComm, &dmpi->snes);CHKERRQ(ierr);
  ierr = SNESSetDM( dmpi->snes, dm);CHKERRQ(ierr);
  ierr = DMSNESSetFunctionLocal(dm,  (PetscErrorCode (*)(DM,Vec,Vec,void*))DMPlexSNESComputeResidualFEM,&ctx);CHKERRQ(ierr);
  ierr = DMSNESSetJacobianLocal(dm,  (PetscErrorCode (*)(DM,Vec,Mat,Mat,void*))DMPlexSNESComputeJacobianFEM,&ctx);CHKERRQ(ierr);
  ierr = DMSetMatType(dm,MATAIJ);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(dmpi->snes);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &J);CHKERRQ(ierr);
  ierr = SNESSetJacobian(dmpi->snes, J, J, NULL, NULL);CHKERRQ(ierr);
  ierr = SNESSetUp( dmpi->snes );CHKERRQ(ierr);
  /* setup particles */
  ierr = createParticles( &ctx );CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ctx.events[3],0,0,0,0);CHKERRQ(ierr);
  {
    PetscViewer    viewer = NULL;
    PetscBool      flg;
    ierr = PetscOptionsGetViewer(ctx.wComm,NULL,"-torus_dm_view",&viewer,NULL,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = DMView(dm,viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* do it */
  ierr = go( &ctx );CHKERRQ(ierr);

  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = destroyParticles(&ctx);CHKERRQ(ierr);
  ierr = DMDestroy(&ctx.dm);CHKERRQ(ierr);
  ierr = PetscFree(ctx.BCFuncs);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&ctx.wComm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

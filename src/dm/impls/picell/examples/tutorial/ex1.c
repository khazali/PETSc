/* M. Adams, April 2015 */

static char help[] = "X2: A partical in cell code for tokamac plasmas using PICell.\n";

#ifdef H5PART
#include <H5Part.h>
#endif
#include <petsc/private/dmpicellimpl.h>    /*I   "petscdmpicell.h"   I*/
#include <assert.h>
/* #include <petscdmforest.h> */
/* #include <petscoptions.h> */

/* particle grid, not PETSc ? */
typedef struct {
  PetscInt nradius;
  PetscInt ntheta;
  PetscInt nphi;
  /* tokamac geometry  */
  PetscReal  rMajor;
  PetscReal  rMinor;
} X2GridParticle;
PetscMPIInt X2GridParticleGetProc(X2GridParticle *, PetscReal, PetscReal, PetscReal);
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
  PetscReal z;   /* vertical coorinate */
  PetscReal phi; /* toroidal coordinate */
  PetscReal vpar; /* toroidal velocity */
  /* const */
  PetscReal mu; /* 5th D */
  PetscReal w0;
  PetscReal f0;
  long long gid; /* diagnostic */
} X2Particle;
/* PetscErrorCode X2ParticleCreate(X2Particle *p, PetscInt gid, PetscReal r, PetscReal z, PetscReal phi, PetscReal rho); */
/* PetscErrorCode X2ParticleCopy(X2Particle *p, X2Particle); */
/* PetscErrorCode X2ParticleRead(X2Particle*, void*); */
/* PetscErrorCode X2ParticleWrite(X2Particle*, void*); */
/* X2PList */
typedef PetscInt X2PListPos;
typedef struct {
  X2Particle *data;
  PetscInt    data_size, size, hole, top;
} X2PList;
/* PetscErrorCode X2PListCreate(X2PList *l, PetscInt msz); */
/* PetscErrorCode X2PListDestroy(X2PList *l); */
/* PetscErrorCode X2PListAdd( X2PList*, X2Particle*); /\* copy data *\/ */
/* PetscErrorCode X2PListGetHead(X2PList*, X2PListPos*); */
/* PetscErrorCode X2PListGetNext(X2PList*, X2Particle *, X2PListPos*); /\* get pointer to data in list *\/ */
/* PetscErrorCode X2PListRemoveAt( X2PList*, X2PListPos); */
/* PetscErrorCode X2PListSetAt(X2PList*, X2PListPos, X2Particle *); */
/* PetscErrorCode X2PListClear( X2PList*); */
/* PetscErrorCode X2PListCompress( X2PList*); */
/* PetscInt X2PListMaxSize(X2PList *l); */
/* PetscInt X2PListSize(X2PList *l); */
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
  PetscInt      debug;   /* The debugging level */
  PetscLogEvent events[10];
  PetscInt      currevent;
  PetscInt      tablesize,tablecount;
  PetscInt      useBSPchuncksize;
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
  /* char *B0eqFilename[256]; */
  PetscReal massAu; /* =2D0  !mass ratio to proton */
  PetscReal eMassAu; /* =2D-2 */
  PetscReal chargeEu; /* =1D0  ! charge number */
  PetscReal eChargeEu; /* =-1D0 */
  X2Species species[X2_NION+1]; /* 0: electron, 1:N ions */
  /* particles */
  PetscInt  npart_cell;
  PetscInt  partBuffSize;
  PetscBool useElectrons;
  PetscInt  collisionPeriod;
  PetscReal maxparvel;
  X2PList   partlists[X2_NION+1]; /* 0: electron, 1:N ions */
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
  l->size=0;
  l->top=0;
  l->hole=-1;
  /* l->data_size=msz; */
  /* ierr = PetscMalloc1(msz, &l->data);CHKERRQ(ierr); */
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
  if (l->size==l->data_size) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"X2PListAdd: list full - todo");
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

#include <petscdmforest.h>
#include <petscoptions.h>
#ifdef H5PART
#include <H5Part.h>
#endif


#define X2PROCLISTSIZE 4096
static PetscInt s_chunksize = 32768;
/*
   ProcessOptions: set parameters from input, setup w/o allocation, called first, no DM here
*/
#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions( X2Ctx *ctx )
{
  PetscErrorCode ierr;
  PetscBool phiFlag,radFlag,thetaFlag;
  PetscReal refinement_limit = 1.;
  /* static char fname[] = "B0.eqd"; */

  PetscFunctionBeginUser;
  /* general */
  ierr = MPI_Comm_rank(ctx->wComm, &ctx->rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(ctx->wComm, &ctx->npe);CHKERRQ(ierr);
  ctx->debug     = 0;
  ctx->tablesize = 197;
  ctx->tablecount = 0;

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
  ctx->particleGrid.rMajor = 621.0; /* cm of ITER */
  ctx->particleGrid.rMinor = 200.0; /* cm of ITER */
  ctx->particleGrid.nphi = 1;
  ctx->particleGrid.nradius    = 1;
  ctx->particleGrid.ntheta     = 1;

  ierr = PetscOptionsBegin(ctx->wComm, "", "Poisson Problem Options", "XGC2");CHKERRQ(ierr);
  /* general options */
  ierr = PetscOptionsInt("-debug", "The debugging level", "xgc2.c", ctx->debug, &ctx->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-chuncksize", "Size of particle list to chunk sends", "xgc2.c", s_chunksize, &s_chunksize, NULL);CHKERRQ(ierr);
  ctx->useBSPchuncksize = 32768;
  ierr = PetscOptionsInt("-useBSPchuncksize", "Size of chucks for PETSc's TwoSide communication (0 for use non-BSP)", "xgc2.c", ctx->useBSPchuncksize, &ctx->useBSPchuncksize, NULL);CHKERRQ(ierr);
  if (ctx->useBSPchuncksize<0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB," invalid chuck size = %D",ctx->useBSPchuncksize);

  /* run  = ctx->runType; */
  /* ierr = PetscOptionsEList("-run_type", "The run type", "xgc2.c", runTypes, 3, runTypes[ctx->runType], &run, NULL);CHKERRQ(ierr); */
  /* ctx->runType = (RunType) run; */

  /* Domain and mesh definition */
  ierr = PetscOptionsReal("-rMajor", "Major radius of torus", "xgc2.c", ctx->particleGrid.rMajor, &ctx->particleGrid.rMajor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rMinor", "Minor radius of torus", "xgc2.c", ctx->particleGrid.rMinor, &ctx->particleGrid.rMinor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "xgc2.c", refinement_limit, &refinement_limit, NULL); /* not used!!! */
  CHKERRQ(ierr);

  ierr = PetscOptionsInt("-nphi_particles", "Number of planes for particle mesh", "xgc2.c", ctx->particleGrid.nphi, &ctx->particleGrid.nphi, &phiFlag);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nradius_particles", "Number of radial cells for particle mesh", "xgc2.c", ctx->particleGrid.nradius, &ctx->particleGrid.nradius, &radFlag);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ntheta_particles", "Number of theta cells for particle mesh", "xgc2.c", ctx->particleGrid.ntheta, &ctx->particleGrid.ntheta, &thetaFlag);CHKERRQ(ierr);
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
  ierr = PetscOptionsInt("-mstep", "Maximum number of time steps", "xgc2.c", ctx->msteps, &ctx->msteps, NULL);CHKERRQ(ierr);
  ctx->maxTime = 1000000000.;
  ierr = PetscOptionsReal("-maxTime", "Maximum time", "xgc2.c",ctx->maxTime,&ctx->maxTime,NULL);CHKERRQ(ierr);
  ctx->dt = 1.;
  ierr = PetscOptionsReal("-dt","Time step","xgc2.c",ctx->dt,&ctx->dt,NULL);CHKERRQ(ierr);
  /* particles */
  ctx->npart_cell = 10; ctx->partBuffSize = 15;
  ierr = PetscOptionsInt("-npart_cell", "Number of particles local (not cell!!!)", "xgc2.c", ctx->npart_cell, &ctx->npart_cell, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-partBuffSize", "Number of z cells for solver mesh", "xgc2.c", ctx->partBuffSize, &ctx->partBuffSize, NULL);CHKERRQ(ierr);
  if (ctx->partBuffSize<10*ctx->npart_cell/9) ctx->partBuffSize = 3*ctx->npart_cell/2; /* hack */
  ctx->collisionPeriod = 10;
  ierr = PetscOptionsInt("-collisionPeriod", "Period between collision operators", "xgc2.c", ctx->collisionPeriod, &ctx->collisionPeriod, NULL);CHKERRQ(ierr);
  if (ctx->partBuffSize < ctx->npart_cell) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"partBuffSize (%D) < npart_cell (%D)",ctx->partBuffSize,ctx->npart_cell);
  ctx->useElectrons = PETSC_FALSE;
  ierr = PetscOptionsBool("-useElectrons", "Include electrons", "xgc2.c", ctx->useElectrons, &ctx->useElectrons, NULL);CHKERRQ(ierr);
  ctx->maxparvel = 1.;
  ierr = PetscOptionsReal("-maxparvel", "Maximum parallel velocity", "xgc2.c",ctx->maxparvel,&ctx->maxparvel,NULL);CHKERRQ(ierr);

  /* model/equations */
  /* (sizeof(a)/sizeof((a)[0])) */
  /* run = (sizeof(ctx->B0eqFilename)/sizeof((ctx->B0eqFilename)[0])); */
  /* ierr = PetscOptionsStringArray("-B0eqFilename", "Name of equilibrium B field file", "xgc2.c", ctx->B0eqFilename, &run, &flg);CHKERRQ(ierr); */
  /* if (!flg) *ctx->B0eqFilename = fname; */

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
#define __FUNCT__ "cylindrical2polPlane"
PetscErrorCode cylindrical2polPlane(PetscReal a_rMinor, PetscReal a_Z, PetscReal *a_psi, PetscReal *a_theta)
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
#define __FUNCT__ "polPlane2cylindrical"
PetscErrorCode polPlane2cylindrical(PetscReal a_psi, PetscReal a_theta, PetscReal *a_rMinor, PetscReal *a_Z)
{
  PetscFunctionBeginUser;
  *a_rMinor = a_psi*cos(a_theta);
  *a_Z = a_psi*sin(a_theta);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "cylindrical2Cart"
PetscErrorCode cylindrical2Cart(PetscReal a_R, PetscReal a_Z, PetscReal a_phi, PetscReal a_cart[3])
{
  PetscFunctionBeginUser;
  a_cart[0] = a_R*cos(a_phi);
  a_cart[1] = a_R*sin(a_phi);
  a_cart[2] = a_Z;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "createParticles"
static PetscErrorCode createParticles(X2Ctx *ctx)
{
  PetscErrorCode ierr;
  PetscInt isp,nCellsLoc,my0,irs,iths,gid,ii,np,j,dim,one=1;
  const PetscReal dth=(2.*M_PI)/(PetscReal)ctx->particleGrid.ntheta,rone=1.;
  const PetscReal dphi=2.*M_PI/(PetscReal)ctx->particleGrid.nphi,rmin=ctx->particleGrid.rMinor;
  const PetscReal phi1 = (PetscReal)ctx->ParticlePlaneIdx*dphi + 1.e-8,rmaj=ctx->particleGrid.rMajor;
  const PetscInt  nPartCells_plane = ctx->particleGrid.ntheta*ctx->particleGrid.nradius; /* nPartCells_plane == ctx->npe_particlePlane */
  const PetscReal dx = pow( (M_PI*rmin*rmin/4.0) * rmaj*2.*M_PI / (PetscReal)(ctx->npe*ctx->npart_cell), 0.333); /* lenth of a particle, approx. */
  X2Particle particle;
  Vec vVec,xVec;
  PetscScalar *x;
  DM dm;
  DM_PICell *dmpi;
  PetscFunctionBeginUser;

  /* Create vector and get pointer to data space */
  dmpi = (DM_PICell *) ctx->dm->data;
  dm = dmpi->dmplex;
  /* ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr); */dim = 3;
  if (dim!=3) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"wrong dimension (3) = %D",dim);
  ierr = VecCreateSeq(PETSC_COMM_SELF,one,&vVec);CHKERRQ(ierr);
  ierr = VecSet(vVec,rone);CHKERRQ(ierr); /* dummy density now */
  ierr = VecCreateSeq(PETSC_COMM_SELF,dim,&xVec);CHKERRQ(ierr);
  ierr = VecGetArray(xVec,&x);CHKERRQ(ierr);

  /* setup particles - lexigraphic partition of cells (np local cells) */
  nCellsLoc = nPartCells_plane/ctx->npe_particlePlane; /* = 1; nPartCells_plane == ctx->npe_particlePlane */
  my0 = ctx->particlePlaneRank*nCellsLoc;              /* cell index in plane == particlePlaneRank */
  gid = (my0 + ctx->ParticlePlaneIdx*nPartCells_plane)*ctx->npart_cell; /* based particle ID */
  if (ctx->ParticlePlaneIdx == ctx->npe_particlePlane-1){
    nCellsLoc = nPartCells_plane - nCellsLoc*(ctx->npe_particlePlane-1);
  }
  assert(nCellsLoc==1);
  /* my first cell idex */
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
	  PetscReal zmax,maxe=ctx->maxparvel*ctx->maxparvel,zdum,mass=1.,b=1.,charge=1.,t=1.;
          PetscScalar v=1.,vpar;
          const PetscReal phi = phi1 + (PetscReal)(rand()%X2NDIG)/(PetscReal)X2NDIG*dphi;
	  const PetscReal thetap = theta0 + qsaf*phi; /* push forward to follow fieldlines */
	  ierr = polPlane2cylindrical(psi, thetap, &r, &z);CHKERRQ(ierr);
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
	  /* add density to RHS */
	  ierr = cylindrical2Cart(particle.r, particle.z, particle.phi, x);CHKERRQ(ierr);
	  ierr = DMPICellAddSource(ctx->dm, xVec, vVec);CHKERRQ(ierr);
	  if((j=X2GridParticleGetProc(&ctx->particleGrid,psi,thetap,phi)) != ctx->rank){
	    PetscPrintf(PETSC_COMM_SELF,"[%D] ERROR proc %d r=%e:%e:%e theta=%e:%e:%e phi=%e:%e:%e\n",ctx->rank,j,r1,psi,r1+dr,th1,thetap,th1+dth,phi1,phi,phi1+dphi);
	    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB," created particle for proc %D",j);
	  }
	} /* theta */
      }
      iths++;
      if (iths==ctx->particleGrid.ntheta) { iths = 0; irs++; }
    } /* cells */
  } /* species */
  VecDestroy(&vVec);
  VecRestoreArray(xVec,&x);
  VecDestroy(&xVec);
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

#undef __FUNCT__
#define __FUNCT__ "CtxDestroy"
PetscErrorCode CtxDestroy(X2Ctx *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ierr = DMDestroy(&ctx->dm);CHKERRQ(ierr);

  ierr = destroyParticles(ctx);CHKERRQ(ierr);

  ierr = PetscFree(ctx->BCFuncs);CHKERRQ(ierr);

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

  file1 = H5PartOpenFileParallel(fname1,H5PART_WRITE,comm);
  file2 = H5PartOpenFileParallel(fname2,H5PART_WRITE,comm);
  ierr = H5PartFileIsValid(file1);CHKERRQ(ierr);
  ierr = H5PartSetStep(file1, 0);CHKERRQ(ierr);
  ierr = H5PartFileIsValid(file2);CHKERRQ(ierr);
  ierr = H5PartSetStep(file2, 0);CHKERRQ(ierr);

  nparticles = X2PListSize(l);
  if (nparticles) {
    x=(double*)malloc(nparticles*sizeof(double));
    y=(double*)malloc(nparticles*sizeof(double));
    z=(double*)malloc(nparticles*sizeof(double));
    v=(double*)malloc(nparticles*sizeof(double));
    id=(h5part_int64_t*)malloc(nparticles*sizeof(h5part_int64_t));
    nparticles = 0;
    ierr = X2PListGetHead( l, &pos );CHKERRQ(ierr);
    while ( !X2PListGetNext( l, &part, &pos) ) {
      x[nparticles] = part.r*cos(part.phi);
      y[nparticles] = part.r*sin(part.phi);
      z[nparticles] = part.z;
      v[nparticles] = part.vpar;
      id[nparticles] = rank; // part.id;
      nparticles++;
    }
  }
  ierr = H5PartSetNumParticles(file1, nparticles);
  ierr = H5PartWriteDataFloat64(file1, "x", x);
  ierr = H5PartWriteDataFloat64(file1, "y", y);
  ierr = H5PartWriteDataFloat64(file1, "z", z);
  ierr = H5PartWriteDataInt64(file1, "id", id);

  if (rank!=npe-1 && rank!=npe-2) nparticles = 0; /* just write last proc(s) */
  /* if (rank!=1 && rank!=0) nparticles = 0; */ /* just write first proc(s) */
  ierr = H5PartSetNumParticles(file2, nparticles);
  ierr = H5PartWriteDataFloat64(file2, "x", x);
  ierr = H5PartWriteDataFloat64(file2, "y", y);
  ierr = H5PartWriteDataFloat64(file2, "z", z);
  ierr = H5PartWriteDataFloat64(file2, "v", v);
  ierr = H5PartWriteDataInt64(file2, "rank", id);

  if (nparticles) {
    free(x); free(y); free(z); free(id); free(v);
  }
  ierr = H5PartCloseFile(file1);
  ierr = H5PartCloseFile(file2);

  PetscFunctionReturn(0);
}
#endif

/* X2GridParticleGetProc: find processor that this point belongs
    Input:
     - grid: the particle grid
     - radius: r in r,theta coordinates
     - theta:
     _ phi: toroidal angle
   Output:
     - return proccess ID
*/
#undef __FUNCT__
#define __FUNCT__ "X2GridParticleGetProc"
PetscMPIInt X2GridParticleGetProc(X2GridParticle *grid, PetscReal psi, PetscReal theta, PetscReal phi)
{
  const PetscReal rminor=grid->rMinor;
  const PetscReal dphi=2.*M_PI/(PetscReal)grid->nphi;
  const PetscReal dth=2.*M_PI/(PetscReal)grid->ntheta;

  PetscMPIInt pe,planeIdx,irs,iths;

  PetscFunctionBeginUser;
  theta = fmod( theta - qsafty(psi/rminor)*phi + 20.*M_PI, 2.*M_PI);  /* pull back to referance grid */
  planeIdx = (PetscMPIInt)(phi/dphi)*grid->nradius*grid->ntheta; /* assumeing one particle cell per PE */
  iths = (PetscMPIInt)(theta/dth);                               assert(iths<grid->ntheta);
  irs = (PetscMPIInt)((PetscReal)grid->nradius*psi*psi/(rminor*rminor)); assert(irs<grid->nradius);
  pe = planeIdx + irs*grid->ntheta + iths;
  PetscFunctionReturn(pe);
}

/* processParticle: */
#undef __FUNCT__
#define __FUNCT__ "processParticles"
static PetscErrorCode processParticles( X2Ctx *ctx, X2PList *list, X2GridParticle *grid, PetscInt isp,
					PetscReal dt, X2SendList *sendLTabPtr, PetscInt tag,
					X2ISend slist[], int irk, int *slistIdx, int *nmoved )
{
  PetscReal   r,z,psi,theta,dphi,rmaj=grid->rMajor,rminor=grid->rMinor;
  PetscMPIInt pe,hash,ii;
  X2Particle  part;
  X2PListPos  pos;
  PetscErrorCode ierr;
  const int part_dsize = sizeof(X2Particle)/sizeof(double);
  DM_PICell *dmpi;
  DM dm;
  Vec gradVec,xVec,vVec;
  PetscScalar *x,*grad,rone=1.;
  PetscInt dim,order=1,one=1;
  PetscFunctionBeginUser;

  /* ierr = X2PListCompress( list );CHKERRQ(ierr); */
  dmpi = (DM_PICell *) ctx->dm->data;
  dm = dmpi->dmplex;
  /* ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr); */dim = 3;
  ierr = VecCreateSeq(PETSC_COMM_SELF,dim,&gradVec);CHKERRQ(ierr);
  ierr = VecGetArray(gradVec,&grad);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,dim,&xVec);CHKERRQ(ierr);
  ierr = VecGetArray(xVec,&x);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,one,&vVec);CHKERRQ(ierr);
  ierr = VecSet(vVec,rone);CHKERRQ(ierr); /* dummy density now */

  /* for (pos=0 ; pos < list->size ; pos++) { */
  /* part = list->data[pos]; */
  ierr = X2PListGetHead( list, &pos );CHKERRQ(ierr);
  while ( !X2PListGetNext(list, &part, &pos) ) {
    /* get E */
    ierr = cylindrical2Cart(part.r, part.z, part.phi, x);CHKERRQ(ierr);
    ierr = DMPICellGetJet(ctx->dm, xVec, order, gradVec);CHKERRQ(ierr); /* not used */

    /* simple push: theta = theta + q*dphi .... grad not used */
    ierr = cylindrical2polPlane( part.r - rmaj, part.z, &psi, &theta );CHKERRQ(ierr);

    dphi = (dt*part.vpar)/(2.*M_PI*part.r);  /* toroidal step */
    part.phi += dphi;
    part.phi = fmod( part.phi + 20.*M_PI, 2.*M_PI);

    theta += qsafty(psi/rminor)*dphi;  /* twist */
    theta = fmod( theta + 20.*M_PI, 2.*M_PI);

    ierr = polPlane2cylindrical( psi, theta, &r, &z);CHKERRQ(ierr); /* time spent here */
    part.r = rmaj + r;
    part.z = z;

    /* see if need communication, add density if not, add to communication list if so */
    if (irk==0 || !sendLTabPtr || (pe=X2GridParticleGetProc(grid, psi, theta, part.phi))==ctx->rank) { /* don't move */
      /* add density to RHS */
      ierr = cylindrical2Cart(part.r, part.z, part.phi, x);CHKERRQ(ierr);
      ierr = DMPICellAddSource(ctx->dm, xVec, vVec);CHKERRQ(ierr);
      if (irk==1) {
	ierr = X2PListSetAt( list, pos, &part );CHKERRQ(ierr); /* not moved and final step so write back */
	/* deposit to collision RHS, Amper'es ... */
      }
    }
    else { /* move: irk==1 && off proc */
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventBegin(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
      /* add to list to send, find list with manual has table lookup, send full lists */
      hash = (pe*(17*ctx->tablesize+3))%ctx->tablesize; /* hash function */
      for (ii=0;ii<ctx->tablesize;ii++){
	if (sendLTabPtr[hash].plist[isp].data_size==0) {
	  ierr = X2PListCreate(&sendLTabPtr[hash].plist[isp],s_chunksize);CHKERRQ(ierr);
	  sendLTabPtr[hash].proc = pe;
	  ctx->tablecount++;
	}
	if (sendLTabPtr[hash].proc==pe) { /* found hash table entry */
	  if (X2PListSize(&sendLTabPtr[hash].plist[isp])==s_chunksize) {
	    if (ctx->useBSPchuncksize) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"cache too small (%D) for BSP TwoSided communication",s_chunksize);
	    /* send and reset - we can just send this because it is dense, but no species data */
	    if (*slistIdx==X2PROCLISTSIZE) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"process send table too small (%D)",X2PROCLISTSIZE);

	    slist[*slistIdx].data = sendLTabPtr[hash].plist[isp].data; /* cache data */
	    sendLTabPtr[hash].plist[isp].data = 0; /* clear for safty, make ready for more */
	    slist[*slistIdx].proc = pe;
	    ierr = MPI_Isend( (void*)slist[*slistIdx].data,s_chunksize*part_dsize,MPI_DOUBLE,pe,tag,ctx->wComm,&slist[*slistIdx].request );
	    CHKERRQ(ierr);
	    /* ready for more */
	    ierr = X2PListCreate(&sendLTabPtr[hash].plist[isp],s_chunksize);CHKERRQ(ierr);

	    (*slistIdx)++;
	  }
	  /* add to list - pass this in as a function to a function? */
	  ierr = X2PListAdd(&sendLTabPtr[hash].plist[isp],&part);CHKERRQ(ierr);assert(part.gid>0);
	  ierr = X2PListRemoveAt(list,pos);CHKERRQ(ierr);
	  (*nmoved)++;
	  break;
	}
	if (++hash == ctx->tablesize) hash=0;
      }
      if (ii==ctx->tablesize) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"processor table (%D) not large enough, need to alloc",ctx->tablesize);
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventEnd(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
    }
  }
  VecRestoreArray(gradVec,&grad);
  VecDestroy(&gradVec);
  VecRestoreArray(xVec,&x);
  VecDestroy(&xVec);
  VecDestroy(&vVec);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "shiftParticles"
PetscErrorCode shiftParticles(X2Ctx *ctx, X2SendList *sendLTabPtr, PetscInt isp,
			      PetscInt numGlobalISends, PetscInt nIsend,
			      X2PList *particlelist, X2ISend slist[], PetscInt tag )
{
  PetscErrorCode ierr;
  const int part_dsize = sizeof(X2Particle)/sizeof(double);
  int  sz,ii,jj,kk,mm,idx;
  Vec vVec,xVec;
  PetscScalar *x,rone=1.;
  DM dm;
  DM_PICell *dmpi;
  PetscInt dim,one=1;
  PetscFunctionBeginUser;

  dmpi = (DM_PICell *) ctx->dm->data;
  dm = dmpi->dmplex;
  /* ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr); */ dim = 3;
  ierr = VecCreateSeq(PETSC_COMM_SELF,one,&vVec);CHKERRQ(ierr);
  ierr = VecSet(vVec,rone);CHKERRQ(ierr); /* dummy density now */
  ierr = VecCreateSeq(PETSC_COMM_SELF,dim,&xVec);CHKERRQ(ierr);
  ierr = VecGetArray(xVec,&x);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(ctx->events[1],0,0,0,0);CHKERRQ(ierr);
#endif

  if ( ctx->useBSPchuncksize ) { /* use BSP */
    PetscMPIInt  nto,*fromranks,chucksz=s_chunksize;
    PetscMPIInt *toranks;
    X2Particle  *fromdata,*todata,*pp;
    PetscInt     nfrom;

    /* count send  */
    for (ii=0,nto=0;ii<ctx->tablesize;ii++) {
      if (sendLTabPtr[ii].plist[isp].data_size != 0) {
	if ((sz=X2PListSize(&sendLTabPtr[ii].plist[isp])) > 0) {
	  for (jj=0 ; jj<sz ; jj += chucksz) nto++;
	}
      }
    }

    /* make toranks & data */
    ierr = PetscMalloc1(nto, &toranks);CHKERRQ(ierr);
    ierr = PetscMalloc1(s_chunksize*nto, &todata);CHKERRQ(ierr);
    for (ii=0,nto=0,pp=todata;ii<ctx->tablesize;ii++) {
      if (sendLTabPtr[ii].plist[isp].data_size != 0) {
	if ((sz=X2PListSize(&sendLTabPtr[ii].plist[isp])) > 0) {
	  /* empty list */
	  for (jj=0, mm=0 ; jj<sz ; jj += chucksz) {
	    toranks[nto++] = sendLTabPtr[ii].proc;
	    for (kk=0 ; kk<chucksz && mm < sz; kk++, mm++) {
	      *pp++ = sendLTabPtr[ii].plist[isp].data[mm];assert(sendLTabPtr[ii].plist[isp].data[mm].gid>0);
	    }
	  }
	  assert(mm==sz);
	  while (kk++ < chucksz) { /* pad with zeros */
	    pp->gid = 0;
	    pp++;
	  }
	  /* get ready for next round */
	  ierr = X2PListDestroy( &sendLTabPtr[ii].plist[isp] );CHKERRQ(ierr);
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
	  ierr = cylindrical2Cart(pp->r, pp->z, pp->phi, x);CHKERRQ(ierr);
	  ierr = DMPICellAddSource(ctx->dm, xVec, vVec);CHKERRQ(ierr);
	}
      }
    }

    ierr = PetscFree(todata);CHKERRQ(ierr);
    ierr = PetscFree(fromranks);CHKERRQ(ierr);
    ierr = PetscFree(fromdata);CHKERRQ(ierr);
    ierr = PetscFree(toranks);CHKERRQ(ierr);
  }
  else {
    X2Particle *data;
    PetscBool   done=PETSC_FALSE,bar_act=PETSC_FALSE;
    MPI_Request ib_request;
    PetscInt    numSent/* , numIrecv=nIsend,Irecvcount=nIsend */;
    MPI_Status  status;
    int flag;

    /* send lists */
    for (ii=0;ii<ctx->tablesize;ii++) {
      if (sendLTabPtr[ii].plist[isp].data_size != 0) {
	if ((sz=X2PListSize(&sendLTabPtr[ii].plist[isp])) > 0) {
	  if (nIsend==X2PROCLISTSIZE) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"process send table too small (%D)",X2PROCLISTSIZE);
	  slist[nIsend].proc = sendLTabPtr[ii].proc;
	  slist[nIsend].data = sendLTabPtr[ii].plist[isp].data; /* cache data */
	  /* send and reset - we can just send this because it is dense */
	  ierr = MPI_Isend((void*)slist[nIsend].data,sz*part_dsize,MPI_DOUBLE,slist[nIsend].proc,tag,ctx->wComm,&slist[nIsend].request);
	  CHKERRQ(ierr);
	  nIsend++;

	  ierr = X2PListClear( &sendLTabPtr[ii].plist[isp] );CHKERRQ(ierr); /* ready for next round */
	  assert(sendLTabPtr[ii].plist[isp].data_size == s_chunksize);
	  ierr = PetscMalloc1(s_chunksize, &sendLTabPtr[ii].plist[isp].data);CHKERRQ(ierr);
	  assert(!(sendLTabPtr[ii].plist[isp].data_size != 0 && (sz=X2PListSize(&sendLTabPtr[ii].plist[isp]) ) > 0));
	}
      }
      /* else - empty list  */
    }
    numSent = nIsend; /* size of send array */
    /* process recieves - nonblocking consensus */
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(ctx->events[2],0,0,0,0);CHKERRQ(ierr);
#endif
    /* process recieves - nonblocking consensus */
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
	    ierr = cylindrical2Cart(data[jj].r, data[jj].z, data[jj].phi, x);CHKERRQ(ierr);
	    ierr = DMPICellAddSource(ctx->dm, xVec, vVec);CHKERRQ(ierr);
	    /* deposit to collision RHS, Amper'es ... */
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
    } /* nonblocking consensus */
    ierr = PetscFree(data);CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(ctx->events[2],0,0,0,0);CHKERRQ(ierr);
#endif
  } /* switch for BPS */
  VecDestroy(&vVec);
  VecRestoreArray(xVec,&x);
  VecDestroy(&xVec);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(ctx->events[1],0,0,0,0);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "go"
PetscErrorCode go(X2Ctx *ctx)
{
  PetscErrorCode ierr;
  PetscInt       istep,isp,tag;
  int            irk,nmoved,idx;
  PetscReal      time,dt;
#ifdef H5PART
  char           fname1[256],fname2[256];
#endif
  X2SendList    *sendListTable,*sendLTabPtr;
  X2ISend        slist[X2PROCLISTSIZE];
  PetscFunctionBeginUser;

  ierr = PetscMalloc1(ctx->tablesize,&sendListTable);CHKERRQ(ierr);

  for (idx=0;idx<ctx->tablesize;idx++) {
    for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++) {
      sendListTable[idx].plist[isp].data_size = 0; /* init */
    }
  }

  /* hdf5 output - init */
#ifdef H5PART
  for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++) { // for each species
    sprintf(fname1,"particles_sp%d_time%05d.h5part",isp,0);
    sprintf(fname2,"sub_rank_particles_sp%d_time%05d.h5part",isp,0);
    /* write */
    ierr = X2PListWrite(&ctx->partlists[isp], ctx->rank, ctx->npe, ctx->wComm, fname1, fname2);CHKERRQ(ierr);
  }
#endif
  /* main time step loop */
  ierr = MPI_Barrier(ctx->wComm);CHKERRQ(ierr);
  for ( istep=0, time=0., tag = 100;
	istep < ctx->msteps && time < ctx->maxTime;
	istep++, time += ctx->dt, tag += X2_NION + (ctx->useElectrons ? 1 : 0) ) {

    for (irk=0;irk<2;irk++){
      sendLTabPtr = &sendListTable[0];
      if (((istep+1)%ctx->collisionPeriod)) sendLTabPtr = NULL;
      if (irk==0) dt = ctx->dt/2.;
      else dt = ctx->dt;

      /* solve for potential, desity being assembled is an invarient */
      ierr = DMPICellSolve(ctx->dm);CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventBegin(ctx->events[0],0,0,0,0);CHKERRQ(ierr);
#endif
      for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++) { // for each species
	const PetscInt origNlocal = X2PListSize(&ctx->partlists[isp]);
	/* fused loop for particle processing: */
	idx = 0;
	nmoved = 0;
	processParticles( ctx, &ctx->partlists[isp], &ctx->particleGrid, isp, dt, sendLTabPtr, tag+isp,
			  slist, irk, &idx, &nmoved);
	assert(idx==0 || irk==1);
	/* finish sends */
	if (irk==1 && sendLTabPtr) {
	  /* test */
	  MPI_Datatype mtype;
	  PetscInt rb[3], sb[3] = {origNlocal, nmoved, idx};
	  PetscDataTypeToMPIDataType(PETSC_INT,&mtype);
	  ierr = MPI_Allreduce(sb, rb, 3, mtype, MPI_SUM, ctx->wComm);CHKERRQ(ierr);
	  PetscPrintf(ctx->wComm,"%d.%d) go: %s processed %D local particles, %D global, %g %% particles moved, %D MPI_Isends total (to %d processors local) \n",
		      istep,irk,isp==0?"electrons":"ions",origNlocal,rb[0],100.*(double)rb[1]/(double)rb[0], rb[2],ctx->tablecount);

	  /* send recv */
	  ierr = shiftParticles(ctx, sendLTabPtr, isp, rb[2], idx, &ctx->partlists[isp], slist, tag+isp );CHKERRQ(ierr);

#ifdef H5PART
	  /* hdf5 output */
	  sprintf(fname1,"particles_sp%d_time%05d.h5part",isp,istep+1);
	  sprintf(fname2,"sub_rank_particles_sp%d_time%05d.h5part",isp,istep+1);
	  /* write */
	  ierr = X2PListWrite(&ctx->partlists[isp], ctx->rank, ctx->npe, ctx->wComm, fname1, fname2);CHKERRQ(ierr);
#endif
	} /* final RK stage send: irk==1 */
	else PetscPrintf(ctx->wComm,"%d.%d) go: %s processed %D local particles\n",istep,irk,isp==0?"electrons":"ions",X2PListSize(&ctx->partlists[isp]));
      } /* species */
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventEnd(ctx->events[0],0,0,0,0);CHKERRQ(ierr);
#endif
    } /* rk */
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

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  X2Ctx          ctx; /* user-defined work context */
  PetscErrorCode ierr;
  char           typeString[256] = {'\0'};
  PetscViewer    viewer = NULL;
  PetscBool      flg;
  DM_PICell      *dmpi;
  DM             dm,cdm;
  PetscInt       dim;
  PetscFE        fe; /* FV might be better */
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
  /* setup solver grid */
  dmpi = (DM_PICell *) ctx.dm->data;
  ierr = DMCreate(ctx.wComm, &dmpi->dmplex);CHKERRQ(ierr);
  dm   = dmpi->dmplex;
  ierr = DMSetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  ierr = PetscStrncpy(typeString,DMFOREST,256);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"DM Forest example options",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-dm_type","The type of the dm",NULL,DMFOREST,typeString,sizeof(typeString),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = DMSetType(dm,(DMType) typeString);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr); /* get file name from -dm_forest_topology */
  ierr = PetscObjectSetName((PetscObject) dm, "Mesh");CHKERRQ(ierr);
  ierr = DMSetUp(ctx.dm);CHKERRQ(ierr); /* set all up & build initial grid */

  /* setup Discretization */
  ierr = PetscMalloc(1 * sizeof(PetscErrorCode (*)(PetscInt,const PetscReal [],PetscInt,PetscScalar*,void*)),&ctx.BCFuncs);
  CHKERRQ(ierr);
  ctx.BCFuncs[0] = zero;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  /* ierr = PetscFECreateDefault(dm, dim, 1, PETSC_FALSE, NULL, -1, &fe);CHKERRQ(ierr); */
  /* ierr = PetscObjectSetName((PetscObject) fe, "potential");CHKERRQ(ierr); */
  /* cdm = dm; */
  /* while (cdm) { */
  /*   DMLabel label; */
  /*   PetscDS prob; */
  /*   PetscInt id = 1; */
  /*   ierr = DMGetDS(cdm, &prob);CHKERRQ(ierr); */
  /*   ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr); */
  /*   ierr = DMCreateLabel(cdm, "boundary");CHKERRQ(ierr); */
  /*   ierr = DMGetLabel(cdm, "boundary", &label);CHKERRQ(ierr); */
  /*   ierr = DMPlexMarkBoundaryFaces(cdm, label);CHKERRQ(ierr); */
  /*   ierr = DMAddBoundary(cdm, PETSC_TRUE, "wall", "boundary", 0, 0, NULL, (void (*)()) ctx.BCFuncs, 1, &id, &ctx); */
  /*   CHKERRQ(ierr); */
  /*   ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr); */
  /* } */
  /* ierr = PetscFEDestroy(&fe);CHKERRQ(ierr); */
   /* setup particles */
  ierr = createParticles( &ctx );CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ctx.events[3],0,0,0,0);CHKERRQ(ierr);

  ierr = PetscOptionsGetViewer(PETSC_COMM_WORLD,NULL,"-dm_view",&viewer,NULL,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMView(dm,viewer);CHKERRQ(ierr);
  }
  /* if (!ctx.rank) ierr = test_list();CHKERRQ(ierr); */

  ierr = go( &ctx );CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&ctx.wComm);CHKERRQ(ierr);
  ierr = CtxDestroy(&ctx);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

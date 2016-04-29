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
#define X2_WALL_ARRAY_MAX 68 /* ITER file is 67 */
static float s_wallVtx[X2_WALL_ARRAY_MAX][2];
static int s_numWallPtx;
static int s_numQuads;
static int s_quad_vertex[X2_WALL_ARRAY_MAX][9];
typedef enum {X2_ITER,X2_TORUS,X2_BOXTORUS} runType;
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
#define X2_V_LEN 8
/* #define X2_S_OF_V */
typedef struct { /* ptl_type */
  /* phase (4D) */
  PetscReal *r;   /* r from center */
  PetscReal *z;   /* vertical coordinate */
  PetscReal *phi; /* toroidal coordinate */
  PetscReal *vpar; /* toroidal velocity */
  /* const */
  PetscReal *mu; /* 5th D */
  PetscReal *w0;
  PetscReal *f0;
  long long *gid; /* diagnostic */
} X2Particle_v;
/* X2PList */
typedef PetscInt X2PListPos;
typedef struct {
#ifdef X2_S_OF_V
  X2Particle_v data_v;
#else
  X2Particle *data; /* make this arrays of X2Particle members for struct-of-arrays */
#endif
  PetscInt    data_size, size, hole, top;
} X2PList;
/* send particle list */
typedef struct X2SendList_TAG{
  X2PList plist[X2_NION+1]; /* make this just one list!!! */
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
  PetscLogEvent events[12];
  PetscInt      currevent;
  PetscInt      bsp_chuncksize;
  runType       run_type;
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
  PetscInt  npart_flux_tube;
  PetscInt  partBuffSize;
  PetscBool useElectrons;
  PetscInt  collisionPeriod;
  PetscReal max_vpar;
  PetscInt  nElems; /* size of array of particle lists */
  X2PList  *partlists[X2_NION+1]; /* 0: electron, 1:N ions */
  X2Species species[X2_NION+1]; /* 0: electron, 1:N ions */
  PetscInt  tablesize,tablecount[X2_NION+1]; /* hash table meta-data for proc-send list table */
} X2Ctx;

/* dummy DMPlexFindLocalCellID */
PetscErrorCode DMPlexFindLocalCellID(DM dm, PetscReal x[], PetscInt *elemID)
{
  *elemID = 0;
  return 0;
}

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
  p->mu = 0;  /* perp velocity - not used */
  p->w0 = 1.; /* just a default weight for now */
  p->f0 = 0;  /* not used */
  return 0;
}
PetscErrorCode X2ParticleCopy(X2Particle *p, X2Particle p2)
{
  if (p2.gid <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"X2ParticleCopy: gid <= 0");
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
  l->data_size = (X2_V_LEN*msz)/X2_V_LEN;
  ierr = PetscMalloc1(l->data_size, &l->data);CHKERRQ(ierr); /* malloc each for struct-of-arrays */
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
  l->data[pos] = *part; /* copy to parts for struct-of-arrays */
  return 0;
}

PetscErrorCode X2PListCompress(X2PList *l)
{
  PetscInt ii;
  /* fill holes with end of list */
  for ( ii = 0 ; ii < l->top && l->top > l->size ; ii++) {
    if (l->data[ii].gid <= 0) {
      l->top--; /* maybe data to move */
      if (ii == l->top) /* just pop hole at end */ ;
      else {
	while (l->data[l->top].gid <= 0) l->top--; /* get real data */
	l->data[ii] = l->data[l->top]; /* now above, copy! */
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
  *p = l->data[*pos]; /* return copy! */

  return 0;
}

PetscErrorCode X2PListAdd( X2PList *l, X2Particle *p)
{
  if (l->size==l->data_size) {
    X2Particle *data2; /* make this arrays of X2Particle members for struct-of-arrays */
    int i;PetscErrorCode ierr;
    PetscPrintf(PETSC_COMM_SELF,"X2PListAdd expanded list %d --> %d%d\n",l->data_size,2*l->data_size);
    l->data_size *= 2;
    ierr = PetscMalloc1(l->data_size, &data2);CHKERRQ(ierr);
    for (i=0;i<l->size;i++) data2[i] = l->data[i];
    ierr = PetscFree(l->data);CHKERRQ(ierr);
    l->data = data2;
    assert(l->hole == -1);
  }
  if (l->hole != -1) { /* have a hole - fill it */
    X2PListPos idx = l->hole; assert(idx<l->data_size);
    if (l->data[idx].gid == 0) l->hole = -1; /* filled last hole */
    else if (l->data[idx].gid>=0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"X2PListAdd: hole with non-neg gid!!!",l->data[idx].gid);
    else l->hole = (X2PListPos)(-l->data[idx].gid - 1); /* use gid as pointer */
    l->data[idx] = *p; /* struct copy! */
  }
  else {
    l->data[l->top++] = *p; /* struct copy! */
  }
  l->size++;
  assert(l->top >= l->size); /* no holes? */
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
  ierr = PetscFree(l->data);CHKERRQ(ierr); /* free each for struct-of-arrays */
  l->size=0;
  l->top=0;
  l->hole=-1;
  l->data = 0;
  l->data_size = 0;
  return ierr;
}

#define X2PROCLISTSIZE 256
static PetscInt s_chunksize = 1*X2_V_LEN;
/*
   ProcessOptions: set parameters from input, setup w/o allocation, called first, no DM here
*/
#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions( X2Ctx *ctx )
{
  PetscErrorCode ierr,isp,k,sz;
  FILE *fp;
  PetscBool phiFlag,radFlag,thetaFlag,flg;
  char str[256],str2[256],fname[256];

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
  ctx->particleGrid.rMajor = 6.2; /* m of ITER */
  ctx->particleGrid.rMinor = 2.0; /* m of ITER */
  ctx->particleGrid.nphi = 1;
  ctx->particleGrid.nradius    = 1;
  ctx->particleGrid.ntheta     = 1;
  ctx->particleGrid.numMajor   = 4;
  ctx->particleGrid.innerMult  = M_SQRT2 - 1.;

  ierr = PetscOptionsBegin(ctx->wComm, "", "Poisson Problem Options", "X2");CHKERRQ(ierr);
  /* general options */
  ctx->debug = 0;
  ierr = PetscOptionsInt("-debug", "The debugging level", "x2.c", ctx->debug, &ctx->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-chuncksize", "Size of particle list to chunk sends", "x2.c", s_chunksize, &s_chunksize,&flg);CHKERRQ(ierr);
  if (flg) s_chunksize = (X2_V_LEN*s_chunksize)/X2_V_LEN;
  ctx->bsp_chuncksize = 0; /* 32768; */
  ierr = PetscOptionsInt("-bsp_chuncksize", "Size of chucks for PETSc's TwoSide communication (0 to use 'nonblocking consensus')", "x2.c", ctx->bsp_chuncksize, &ctx->bsp_chuncksize, NULL);CHKERRQ(ierr);
  if (ctx->bsp_chuncksize<0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB," invalid chuck size = %D",ctx->bsp_chuncksize);
  ctx->tablesize = 256; /* hash table size of processors to send to */
  ierr = PetscOptionsInt("-proc_send_table_size", "Size of hash table proc->send_list", "x2.c",ctx->tablesize, &ctx->tablesize, NULL);CHKERRQ(ierr);

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
  ctx->npart_flux_tube = 10; ctx->partBuffSize = 15;
  ierr = PetscOptionsInt("-npart_flux_tube", "Number of particles local (flux tube cell)", "x2.c", ctx->npart_flux_tube, &ctx->npart_flux_tube, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-partBuffSize", "Number of z cells for solver mesh", "x2.c", ctx->partBuffSize, &ctx->partBuffSize, NULL);CHKERRQ(ierr);
  if (ctx->partBuffSize<3*ctx->npart_flux_tube/2+10) ctx->partBuffSize = 3*ctx->npart_flux_tube/2+10; /* hack */
  ctx->collisionPeriod = 10;
  ierr = PetscOptionsInt("-collisionPeriod", "Period between collision operators", "x2.c", ctx->collisionPeriod, &ctx->collisionPeriod, NULL);CHKERRQ(ierr);
  if (ctx->partBuffSize < ctx->npart_flux_tube) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"partBuffSize (%D) < npart_flux_tube (%D)",ctx->partBuffSize,ctx->npart_flux_tube);
  ctx->useElectrons = PETSC_FALSE;
  ierr = PetscOptionsBool("-use_electrons", "Include electrons", "x2.c", ctx->useElectrons, &ctx->useElectrons, NULL);CHKERRQ(ierr);
  ctx->max_vpar = .1;
  ierr = PetscOptionsReal("-max_vpar", "Maximum parallel velocity", "x2.c",ctx->max_vpar,&ctx->max_vpar,NULL);CHKERRQ(ierr);

  ierr = PetscStrcpy(fname,"iter");CHKERRQ(ierr);
  ierr = PetscOptionsString("-run_type", "Type of run (iter or torus)", "x2.c", fname, fname, sizeof(fname)/sizeof(fname[0]), NULL);CHKERRQ(ierr);
  PetscStrcmp("iter",fname,&flg);
  if (flg) { /* ITER */
    ctx->run_type = X2_ITER;
    ierr = PetscStrcpy(fname,"ITER-51vertex-quad.txt");CHKERRQ(ierr);
    ierr = PetscOptionsString("-iter_vertex_file", "Name of vertex .txt file of ITER vertices", "x2.c", fname, fname, sizeof(fname)/sizeof(fname[0]), NULL);CHKERRQ(ierr);
    fp = fopen(fname, "r");
    if (!fp) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ITER file %s not found, use -fname FILE_NAME",fname);
    if (!fgets(str,256,fp)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error reading ITER file");
    k = sscanf(str,"%d\n",&sz);
    if (k<1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error reading ITER file %d words",k);
    if (sz>X2_WALL_ARRAY_MAX) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Too Many vertices %d > %d",sz,X2_WALL_ARRAY_MAX);
    for (isp=0;isp<sz;isp++) {
      if (!fgets(str,256,fp)) break;
      k = sscanf(str,"%e %e %s\n",&s_wallVtx[isp][0],&s_wallVtx[isp][1],str2);
      if (k<2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error reading ITER file %d words",k);
      s_wallVtx[isp][0] -= ctx->particleGrid.rMajor;
    }
    s_numWallPtx = isp;
    if (s_numWallPtx!=sz) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error reading ITER file, %d lines",s_numWallPtx);
    /* cell ids */
    for (isp=0;isp<1000;isp++) {
      if (!fgets(str,256,fp)) break;
      k = sscanf(str,"%d %d %d %d %d %d %d %d %d %s\n",
                 &s_quad_vertex[isp][0],&s_quad_vertex[isp][1],&s_quad_vertex[isp][2],
                 &s_quad_vertex[isp][3],&s_quad_vertex[isp][4],&s_quad_vertex[isp][5],
                 &s_quad_vertex[isp][6],&s_quad_vertex[isp][7],&s_quad_vertex[isp][8],
                 str2);
      if (k==-1) break;
      if (k<9) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error reading ITER file, read %d terms != 9",k);
      if (isp>X2_WALL_ARRAY_MAX) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Too Many Elements %d",X2_WALL_ARRAY_MAX);
    }
    fclose(fp);
    s_numQuads = isp;
    PetscPrintf(PETSC_COMM_WORLD,"ProcessOptions:  numQuads=%d, numWallPtx=%d\n",s_numQuads,s_numWallPtx);
  }
  else {
    PetscStrcmp("torus",fname,&flg);
    if (flg) ctx->run_type = X2_TORUS;
    else {
      PetscStrcmp("boxtorus",fname,&flg);
      if (flg) ctx->run_type = X2_BOXTORUS;
      else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unknown run type %s",fname);
    }
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
PetscErrorCode X2GridParticleGetProc_FluxTube( DM d, const X2GridParticle *grid, /* X2Particle *part, */
                                               PetscReal psi, PetscReal theta, PetscReal phi,
                                               PetscMPIInt *pe, PetscInt *elem)
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
  *elem = 0; /* only one cell per process */
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
/*
  dm - The DM
  x - Cartesian coordinate

  pe - Rank of process owning the grid cell containing the particle, -1 if not found
  elem - Local cell number on rank pe containing the particle, -1 if not found
*/
#undef __FUNCT__
#define __FUNCT__ "X2GridParticleGetProc_Solver"
PetscErrorCode X2GridParticleGetProc_Solver(DM dm, PetscReal coord[], PetscMPIInt *pe, PetscInt *elem)
{
  PetscMPIInt rank;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  /* Matt do your thing here */
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);
  *pe = rank; /* noop -- need to add a local lookup for 'elem' if (*pe == rank) */
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
  PetscInt       isp,elid;
  PetscFunctionBeginUser;
  /* idiom for iterating over particle lists */
  for (isp = ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++ ) { // for each species
    for (elid=0;elid<ctx->nElems;elid++) {
      ierr = X2PListDestroy(&ctx->partlists[isp][elid]);CHKERRQ(ierr);
    }
    ierr = PetscFree(ctx->partlists[isp]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#ifdef H5PART
#undef __FUNCT__
#define __FUNCT__ "X2PListWrite"
PetscErrorCode X2PListWrite( X2PList l[], PetscInt nLists, PetscMPIInt rank, PetscMPIInt npe, MPI_Comm comm, char fname1[], char fname2[])
{
  /* PetscErrorCode ierr; */
  double *x=0,*y=0,*z=0,*v=0;
  h5part_int64_t *id=0,nparticles;
  X2PListPos     pos;
  X2Particle     part;
  PetscErrorCode ierr;
  H5PartFile    *file1,*file2;
  PetscInt       elid;
  PetscFunctionBeginUser;
  for (nparticles=0,elid=0;elid<ctx->nElems;elid++) {
    nparticles += X2PListSize(&l[elid]);
  }
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
    for (nparticles=0,elid=0;elid<ctx->nElems;elid++) {
      ierr = X2PListGetHead( &l[elid], &pos );CHKERRQ(ierr);
      while ( !X2PListGetNext( &l[elid], &part, &pos) ) {
        x[nparticles] = part.r*cos(part.phi);
        y[nparticles] = part.r*sin(part.phi);
        z[nparticles] = part.z;
        v[nparticles] = part.vpar;
        id[nparticles] = part.gid;
        nparticles++;
      }
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
      for (nparticles=0,elid=0;elid<ctx->nElems;elid++) {
        ierr = X2PListGetHead( &l[elid], &pos );CHKERRQ(ierr);
        while ( !X2PListGetNext( &l[elid], &part, &pos) ) {
          x[nparticles] = part.r*cos(part.phi);
          y[nparticles] = part.r*sin(part.phi);
          z[nparticles] = part.z;
          v[nparticles] = part.vpar;
          id[nparticles] = rank;
          nparticles++;
        }
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
     - isp: species index into sendListTable -- remove this and make sendListTable not have list for each species!!!
     - irk: flag for deposit charge (>=0), or just move (<0)
     - tag: MPI tag to send with
     - solver: use solver partitioning to get processor of point?
   Input/Output:
     - nIsend: number of sends so far
     - sendListTable: send list hash table array, emptied but meta-data kept
     - particlelist: array of the lists of particle lists to add to
     - slists: array of non-blocking send caches (!ctx->bsp_chuncksize only), cleared
   Output:
*/
#undef __FUNCT__
#define __FUNCT__ "shiftParticles"
PetscErrorCode shiftParticles( const X2Ctx *ctx, X2SendList *sendListTable, const PetscInt isp, const PetscInt irk, PetscInt *const nIsend,
                               X2PList particlelist[], X2ISend slist[], PetscInt tag, PetscBool solver )
{
  PetscErrorCode ierr;
  const int part_dsize = sizeof(X2Particle)/sizeof(double);
  PetscInt sz,ii,jj,kk,mm,idx,elid;
  DM dm;
  DM_PICell *dmpi;
  PetscFunctionBeginUser;
  dmpi = (DM_PICell *) ctx->dm->data;
  dm = dmpi->dmgrid;
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
          if (solver) {
            PetscReal x[3];
            ierr = cylindricalToCart(pp->r, pp->z, pp->phi, x);CHKERRQ(ierr);
            ierr = DMPlexFindLocalCellID(dm, x, &elid);CHKERRQ(ierr);
          }
          else elid = 0;
	  ierr = X2PListAdd( &particlelist[elid], pp);CHKERRQ(ierr);
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
            if (solver) {
              PetscReal x[3];
              ierr = cylindricalToCart(data[jj].r, data[jj].z, data[jj].phi, x);CHKERRQ(ierr);
              ierr = DMPlexFindLocalCellID(dm, x, &elid);CHKERRQ(ierr);
            }
            else elid = 0;
            ierr = X2PListAdd( &particlelist[elid], &data[jj]);CHKERRQ(ierr);
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
     - solver: use solver partitioning to get processor of point?
   Input/Output:
     - ctx: global data
     - lists: list of particle lists
   Output:
     - sendListTable: send list hash table, null if not sending (irk==0)
*/
#undef __FUNCT__
#define __FUNCT__ "processParticles"
static PetscErrorCode processParticles( X2Ctx *ctx, const PetscReal dt, X2SendList *sendListTable, const PetscInt tag,
					const int irk, const int istep, PetscBool solver)
{
  X2GridParticle *grid = &ctx->particleGrid;         assert(sendListTable); /* always used now */
  DM_PICell *dmpi = (DM_PICell *) ctx->dm->data;     assert(solver || irk<0); /* don't push flux tubes */
  DM dm = dmpi->dmgrid;
  PetscReal   r,z,psi,theta,dphi,rmaj=grid->rMajor,rminor=grid->rMinor;
  PetscMPIInt pe,hash,ii;
  X2Particle  part;
  X2PListPos  pos;
  PetscErrorCode ierr;
  const int part_dsize = sizeof(X2Particle)/sizeof(double);
  Vec          jetVec,xVec,vVec,tjetVec,txVec,tvVec;
  PetscScalar *xx,*jj,*vv,*xx0,*jj0,*vv0;
  PetscInt isp,order=1,nslist,nlistsTot,vecsz,elid,idx,maxsz,one=1,three=3;
  int origNlocal,nmoved;
  X2ISend slist[X2PROCLISTSIZE];
  PetscFunctionBeginUser;
  nslist = 0;
  nmoved = 0;
  nlistsTot = origNlocal = 0;
  /* push particles, if necc., and make send lists */
  for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++) {
    if (solver) {
      /* count max */
      for (maxsz=elid=0;elid<ctx->nElems;elid++) {
        X2PList *list = &ctx->partlists[isp][elid];
        if (X2PListSize(list)>maxsz) maxsz = X2PListSize(list);
        ierr = X2PListCompress(list);CHKERRQ(ierr);
      }
      /* allocate vectors */
      ierr = DMGetDimension(dm, &vecsz);CHKERRQ(ierr); assert(vecsz==3);
      vecsz *= maxsz;
      ierr = VecCreateSeq(PETSC_COMM_SELF,vecsz,&jetVec);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF,vecsz,&xVec);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF,maxsz,&vVec);CHKERRQ(ierr);
    }
    /* loop over element particle lists */
    for (elid=0;elid<ctx->nElems;elid++) {
      X2PList *list = &ctx->partlists[isp][elid];
      if (X2PListSize(list)==0)continue;
      origNlocal += X2PListSize(list);
      /* get Cartesian coordinates (not used for flux tube move) */
      if (solver) {
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventBegin(ctx->events[7],0,0,0,0);CHKERRQ(ierr); /* timer on particle list */
#endif
        /* make vectors for this element */
        ierr = VecGetArray(xVec,&xx);
        ierr = VecGetArray(jetVec,&jj);
        ierr = VecGetArray(vVec,&vv);
        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,three,three*X2PListSize(list), xx, &txVec);CHKERRQ(ierr);
        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,three,three*X2PListSize(list), jj, &tjetVec);CHKERRQ(ierr);
        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,one,one*X2PListSize(list), vv, &tvVec);CHKERRQ(ierr);
        ierr = VecRestoreArray(xVec,&xx);
        ierr = VecRestoreArray(jetVec,&jj);
        ierr = VecRestoreArray(vVec,&vv);
        /* make coordinates array to get gradients */
        ierr = VecGetArray(txVec,&xx0);CHKERRQ(ierr); xx = xx0;
        ierr = X2PListGetHead( list, &pos );CHKERRQ(ierr);
        while ( !X2PListGetNext(list, &part, &pos) ) {
          /* for (pos=0 ; pos < list->size ; pos++, xvec += 3) { */
          /*   X2Particle *part = &list->data[pos]; */
          ierr = cylindricalToCart(part.r, part.z, part.phi, xx);CHKERRQ(ierr);
          xx += 3;
        }
        ierr = VecRestoreArray(txVec,&xx0);
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventEnd(ctx->events[7],0,0,0,0);CHKERRQ(ierr);
#endif
      }
      if (irk>=0) {
        assert(solver);
        /* push */
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventBegin(ctx->events[8],0,0,0,0);CHKERRQ(ierr); /* timer on particle list */
#endif
        /* get E, should set size of vecs for true size? */
        ierr = DMPICellGetJet(ctx->dm, txVec, order, tjetVec, elid);CHKERRQ(ierr); /* not used */
        /* vectorize (todo) push: theta = theta + q*dphi .... grad not used */
        ierr = VecGetArray(txVec,&xx0);CHKERRQ(ierr); xx = xx0;
        ierr = VecGetArray(tjetVec,&jj0);CHKERRQ(ierr); jj = jj0;
        for (pos=0 ; pos < list->size ; pos++, xx += 3, jj += 3 ) {
          X2Particle *part = &list->data[pos]; /* need to vectorize - using raw data, not a copy */
          ierr = cylindricalToPolPlane( part->r - rmaj, part->z, &psi, &theta );CHKERRQ(ierr);
          dphi = (dt*part->vpar)/(2.*M_PI*part->r);  /* toroidal step */
          part->phi += dphi;
          part->phi = fmod( part->phi + 20.*M_PI, 2.*M_PI);
          theta += qsafty(psi/rminor)*dphi;  /* twist */
          theta = fmod( theta + 20.*M_PI, 2.*M_PI);
          ierr = polPlaneToCylindrical( psi, theta, &r, &z);CHKERRQ(ierr); /* time spent here */
          part->r = rmaj + r;
          part->z = z;
        }
        ierr = VecRestoreArray(txVec,&xx0);
        ierr = VecRestoreArray(tjetVec,&jj0);
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventEnd(ctx->events[8],0,0,0,0);CHKERRQ(ierr);
#endif
      }
      /* move */
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventBegin(ctx->events[5],0,0,0,0);CHKERRQ(ierr);
#endif
      /* this has to be refactored with new vector X2GridParticleGetProc_Solver method  */
      if (solver) {
        ierr = VecGetArray(txVec,&xx0);CHKERRQ(ierr); xx = xx0;
      }
      ierr = X2PListGetHead( list, &pos );CHKERRQ(ierr);
      while ( !X2PListGetNext(list, &part, &pos) ) {
        /* see if need communication, add density if not, add to communication list if so */
        if (solver) {
          ierr = X2GridParticleGetProc_Solver(dm, xx, &pe, &idx);CHKERRQ(ierr);
        }
        else {
          ierr = cylindricalToPolPlane( part.r - rmaj, part.z, &psi, &theta );CHKERRQ(ierr);
          ierr = X2GridParticleGetProc_FluxTube(dm, grid, psi, theta, part.phi, &pe, &idx);CHKERRQ(ierr);
          assert(idx==0);
        }
        /* ierr = get_proc(dm, grid, &partpsi, theta, part.phi, x, &pe, &idx);CHKERRQ(ierr); */
        if (pe==ctx->rank && idx==elid) { /* don't move and don't add */
          /* ierr = X2PListSetAt(list, pos, &part );CHKERRQ(ierr); */ /* not moved and final step so write back */
        }
        else { /* move: sendListTable && off proc */
          /* add to list to send, find list with table lookup, send full lists */
          hash = (pe*593)%ctx->tablesize; /* hash */
          for (ii=0;ii<ctx->tablesize;ii++){
            if (sendListTable[hash].plist[isp].data_size==0) {
              ierr = X2PListCreate(&sendListTable[hash].plist[isp],s_chunksize);CHKERRQ(ierr);
              sendListTable[hash].proc = pe;
              ctx->tablecount[isp]++;
              if (ctx->tablecount[isp]==ctx->tablesize) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Table too small (%D)",ctx->tablesize);
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
        }
        if (solver) xx += 3;
      }
      if (solver) {
        ierr = VecRestoreArray(txVec,&xx0);
        /* done with these, need new ones after communication */
        ierr = VecDestroy(&tjetVec);CHKERRQ(ierr);
        ierr = VecDestroy(&txVec);CHKERRQ(ierr);
        ierr = VecDestroy(&tvVec);CHKERRQ(ierr);
      }
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventEnd(ctx->events[5],0,0,0,0);CHKERRQ(ierr);
#endif
    } /* particle lists */
    if (solver) {
      ierr = VecDestroy(&jetVec);CHKERRQ(ierr);
      ierr = VecDestroy(&xVec);CHKERRQ(ierr);
      ierr = VecDestroy(&vVec);CHKERRQ(ierr);
    }
    /* finish sends and receive new particles for this species */
    ierr = shiftParticles(ctx, sendListTable, isp, irk, &nslist, ctx->partlists[isp], slist, tag+isp, solver );CHKERRQ(ierr);
    nlistsTot += nslist;
    nslist = 0;
    /* add density (while in cache, by species at least) */
    if (irk>=0) {
      assert(solver);
      /* count max */
      for (maxsz=elid=0;elid<ctx->nElems;elid++) {
        X2PList *list = &ctx->partlists[isp][elid];
        if (X2PListSize(list)>maxsz) maxsz = X2PListSize(list);
        ierr = X2PListCompress(list);CHKERRQ(ierr);
      }
      /* allocate vectors */
      ierr = DMGetDimension(dm, &vecsz);CHKERRQ(ierr); assert(vecsz==3);
      vecsz *= maxsz;
      ierr = VecCreateSeq(PETSC_COMM_SELF,vecsz,&xVec);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF,maxsz,&vVec);CHKERRQ(ierr);
      for (elid=0;elid<ctx->nElems;elid++) {
        X2PList *list = &ctx->partlists[isp][elid];
        if (X2PListSize(list)==0)continue;
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventBegin(ctx->events[7],0,0,0,0);CHKERRQ(ierr); /* timer on particle list */
#endif
        /* make vectors for this element */
        ierr = VecGetArray(xVec,&xx);CHKERRQ(ierr);
        ierr = VecGetArray(vVec,&vv);CHKERRQ(ierr);
        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,three,three*X2PListSize(list), xx, &txVec);CHKERRQ(ierr);
        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,one,one*X2PListSize(list), vv, &tvVec);CHKERRQ(ierr);
        ierr = VecRestoreArray(xVec,&xx);CHKERRQ(ierr);
        ierr = VecRestoreArray(vVec,&vv);CHKERRQ(ierr);
        /* make coordinates array and desity */
        ierr = VecGetArray(txVec,&xx0);CHKERRQ(ierr); xx = xx0;
        ierr = VecGetArray(tvVec,&vv0);CHKERRQ(ierr); vv = vv0;
        ierr = X2PListGetHead( list, &pos );CHKERRQ(ierr);
        while ( !X2PListGetNext(list, &part, &pos) ) {
          /* for (pos=0 ; pos < list->size ; pos++, xx += 3) { */
          /*   X2Particle *part = &list->data[pos]; */
          ierr = cylindricalToCart(part.r, part.z, part.phi, xx);CHKERRQ(ierr);
          xx += 3;
          *vv = part.w0;
          vv++;
        }
PetscPrintf(PETSC_COMM_SELF,"[%d] processParticles 4444 %d %d\n",ctx->rank,vv-vv0,maxsz);        
        ierr = VecRestoreArray(txVec,&xx0);CHKERRQ(ierr);
        ierr = VecRestoreArray(tvVec,&vv0);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventEnd(ctx->events[7],0,0,0,0);CHKERRQ(ierr);
#endif
PetscPrintf(PETSC_COMM_SELF,"[%d] processParticles 5555elid=%d\n",ctx->rank,elid);        
        ierr = DMPICellAddSource(ctx->dm, txVec, tvVec, elid);CHKERRQ(ierr);
PetscPrintf(PETSC_COMM_SELF,"[%d] processParticles 6666\n",ctx->rank);        
        ierr = VecDestroy(&txVec);CHKERRQ(ierr);
        ierr = VecDestroy(&tvVec);CHKERRQ(ierr);
      }
      if (solver) {
        ierr = VecDestroy(&xVec);CHKERRQ(ierr);
        ierr = VecDestroy(&vVec);CHKERRQ(ierr);
      }
    }
  } /* isp */
  /* diagnostics */
  {
    MPI_Datatype mtype;
    PetscInt rb[3], sb[3] = {origNlocal, nmoved, nlistsTot};
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(ctx->events[6],0,0,0,0);CHKERRQ(ierr);
#endif
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
          ierr = X2PListWrite( &ctx->partlists[isp], ctx->nElems, ctx->rank, ctx->npe, ctx->wComm, fname1, fname2);CHKERRQ(ierr);
        }
      }
    }
#endif
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(ctx->events[6],0,0,0,0);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

/* create particles in flux tubes, create particle lists, move particles to element lists */
#undef __FUNCT__
#define __FUNCT__ "createParticles"
static PetscErrorCode createParticles(X2Ctx *ctx)
{
  PetscErrorCode ierr;
  PetscInt isp,nCellsLoc,my0,irs,iths,gid,ii,np,dim,cStart,cEnd,elid;
  const PetscReal dth=(2.*M_PI)/(PetscReal)ctx->particleGrid.ntheta;
  const PetscReal dphi=2.*M_PI/(PetscReal)ctx->particleGrid.nphi,rmin=ctx->particleGrid.rMinor; /* rmin for particles < rmin */
  const PetscReal phi1 = (PetscReal)ctx->ParticlePlaneIdx*dphi + 1.e-8,rmaj=ctx->particleGrid.rMajor;
  const PetscInt  nPartCells_plane = ctx->particleGrid.ntheta*ctx->particleGrid.nradius; /* nPartCells_plane == ctx->npe_particlePlane */
  const PetscReal dx = pow( (M_PI*rmin*rmin/4.0) * rmaj*2.*M_PI / (PetscReal)(ctx->npe*ctx->npart_flux_tube), 0.333); /* lenth of a particle, approx. */
  X2Particle particle;
  DM dm;
  DM_PICell *dmpi;
  PetscFunctionBeginUser;

  /* Create vector and get pointer to data space */
  dmpi = (DM_PICell *) ctx->dm->data;
  dm = dmpi->dmgrid;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim!=3) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"wrong dimension (3) = %D",dim);
  ierr = DMGetCellChart(dm, &cStart, &cEnd);CHKERRQ(ierr);
  ctx->nElems = PetscMax(1,cEnd-cStart);CHKERRQ(ierr);

  /* setup particles - lexigraphic partition of -- flux tube -- cells */
  nCellsLoc = nPartCells_plane/ctx->npe_particlePlane; /* = 1; nPartCells_plane == ctx->npe_particlePlane */
  my0 = ctx->particlePlaneRank*nCellsLoc;              /* cell index in plane == particlePlaneRank */
  gid = (my0 + ctx->ParticlePlaneIdx*nPartCells_plane)*ctx->npart_flux_tube; /* based particle ID */
  if (ctx->ParticlePlaneIdx == ctx->npe_particlePlane-1){
    nCellsLoc = nPartCells_plane - nCellsLoc*(ctx->npe_particlePlane-1);
  }
  assert(nCellsLoc==1);

  /* my first cell index */
  srand(ctx->rank);
  for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++ ) {
    iths = my0%ctx->particleGrid.ntheta;
    irs = my0/ctx->particleGrid.ntheta;
    ierr = PetscMalloc1(ctx->nElems,&ctx->partlists[isp]);CHKERRQ(ierr);
    {
      const PetscReal r1 = sqrt(((PetscReal)irs      /(PetscReal)ctx->particleGrid.nradius)*rmin*rmin) +       1.e-12*rmin;
      const PetscReal dr = sqrt((((PetscReal)irs+1.0)/(PetscReal)ctx->particleGrid.nradius)*rmin*rmin) - (r1 - 1.e-12*rmin);
      const PetscReal th1 = (PetscReal)iths*dth + 1.e-12*dth;
      /* create list */
      ierr = X2PListCreate(&ctx->partlists[isp][0], ctx->partBuffSize);CHKERRQ(ierr);
#define X2NDIG 100000
      /* create each particle */
      //for (int i=0;i<ctx->npart_flux_tube;i++) {
      for (np=0 ; np<ctx->npart_flux_tube; /* void */ ) {
	PetscReal theta0,r,z;
	const PetscReal psi = r1 + (PetscReal)(rand()%X2NDIG+1)/(PetscReal)(X2NDIG+1)*dr;
	const PetscReal qsaf = qsafty(psi/ctx->particleGrid.rMinor);
	const PetscInt NN = (PetscInt)(dth*psi/dx) + 1;
	const PetscReal dth2 = dth/(PetscReal)NN - 1.e-12*dth;
	for ( ii = 0, theta0 = th1 + (PetscReal)(rand()%X2NDIG)/(PetscReal)X2NDIG*dth2;
	      ii < NN && np<ctx->npart_flux_tube;
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
	  ierr = X2ParticleCreate(&particle,++gid,r,z,phi,vpar);CHKERRQ(ierr); /* only time this is called! */
	  ierr = X2PListAdd(&ctx->partlists[isp][0],&particle);CHKERRQ(ierr);
          /* debug, particles are created in a flux tube */
          {
            PetscMPIInt pe; PetscInt id;
            ierr = X2GridParticleGetProc_FluxTube(NULL,&ctx->particleGrid,psi,thetap,phi,&pe,&id);CHKERRQ(ierr);
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
    for (elid=1;elid<ctx->nElems;elid++) { /* finish off list creates */
      ii = 100; /* this will get enlarged dynamically */
      ierr = X2PListCreate(&ctx->partlists[isp][elid], ii);CHKERRQ(ierr);
    }
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
    /* fake time step (irk>=0) will add density to RHS */
    ierr = processParticles(ctx, 0.0, sendListTable, tag, 0, istep, PETSC_TRUE);
    CHKERRQ(ierr);
    ierr = PetscFree(sendListTable);CHKERRQ(ierr);
  }
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
      ierr = X2PListWrite(ctx->partlists[isp], ctx->nElems, ctx->rank, ctx->npe, ctx->wComm, fname1, fname2);CHKERRQ(ierr);
    }
  }
#endif
  /* main time step loop */
  ierr = MPI_Barrier(ctx->wComm);CHKERRQ(ierr);
  for ( istep=0, time=0., tag = 100;
	istep < ctx->msteps && time < ctx->maxTime;
	istep++, time += ctx->dt, tag += 2*(X2_NION + (ctx->useElectrons ? 1 : 0)) ) {

    /* do collisions */
    if (((istep+1)%ctx->collisionPeriod)==0) {
      /* move to flux tube space */
      ierr = processParticles(ctx, 0.0, sendListTable, tag, -1, istep, PETSC_FALSE);
      CHKERRQ(ierr);
      /* call collision method */
#ifdef H5PART
      if (ctx->debug>0) {
        for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++) { // for each species
          char  fname1[256], fname2[256];
          sprintf(fname1,         "particles_sp%d_time%05d_fluxtube.h5part",isp,istep);
          sprintf(fname2,"sub_rank_particles_sp%d_time%05d_fluxtube.h5part",isp,istep);
          /* write */
          ierr = X2PListWrite(ctx->partlists[isp], ctx->nElems, ctx->rank, ctx->npe, ctx->wComm, fname1, fname2);CHKERRQ(ierr);
        }
      }
#endif
      /* move back to solver space */
      ierr = processParticles(ctx, 0.0, sendListTable, tag + X2_NION + (ctx->useElectrons ? 1 : 0), -1, istep, PETSC_TRUE);
      CHKERRQ(ierr);
    }

    /* very crude explicit RK */
    dt = ctx->dt;

    /* solve for potential, density being assembled is an invariant */
    ierr = DMPICellSolve( ctx->dm );CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(ctx->events[0],0,0,0,0);CHKERRQ(ierr);
#endif
    /* process particles: push, move */
    irk=0;
    ierr = processParticles(ctx, dt, sendListTable, tag, irk, istep, PETSC_TRUE);
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

/* == Defining a base plex for ITER, which looks like a rectilinear (sort of) donut */

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreatePICellITER"
static PetscErrorCode DMPlexCreatePICellITER (MPI_Comm comm, X2GridParticle *params, DM *dm)
{
  PetscMPIInt    rank;
  PetscInt       numCells = 0;
  PetscInt       numVerts = 0;
  PetscReal      rMajor   = params->rMajor;
  PetscInt       numMajor = params->numMajor;
  int           *flatCells = NULL;
  double        *flatCoords = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    const int numQuads = s_numQuads;
    const int numQuadVtx = 2*numQuads + 2; /* kind of a hack to get the number of vertices in the quads */
    numCells = numMajor * numQuads;
    numVerts = numMajor * numQuadVtx;
    ierr = PetscMalloc2(numCells * 8,&flatCells,numVerts * 3,&flatCoords);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"DMPlexCreatePICellITER: numCells=%d\n",numCells);
    {
      double (*coords)[numQuadVtx][3] = (double (*) [numQuadVtx][3]) flatCoords;
      PetscInt i;

      for (i = 0; i < numMajor; i++) {
        PetscInt j;
        double cosphi, sinphi;

        cosphi = cos(2 * M_PI * i / numMajor);
        sinphi = sin(2 * M_PI * i / numMajor);

        for (j = 0; j < numQuadVtx; j++) {
          double r, z;

          r = rMajor + s_wallVtx[j][0];
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

      for (i = 0; i < numMajor; i++) {
        PetscInt j;

        for (j = 0; j < numQuads; j++) {
          PetscInt k;
          for (k = 0; k < 8; k++) {
            PetscInt l = k % 4, off = k/4;
            if (i==numMajor-1 && off) off = 1-numMajor;
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
  X2GridParticle *params = &ctx->particleGrid;
  PetscReal rMajor    = params->rMajor;
  PetscInt  numMajor  = params->numMajor;
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
  idxPhi = (inPhi * numMajor) / (2. * M_PI);
  idxPhi = PetscMin(idxPhi,numMajor - 1);
  leftPhi =  (idxPhi *        2. * M_PI) / numMajor;
  midPhi  = ((idxPhi + 0.5) * 2. * M_PI) / numMajor;
  cosMidPhi  = cos(midPhi);
  sinMidPhi  = sin(midPhi);
  cosLeftPhi = cos(leftPhi);
  sinLeftPhi = sin(leftPhi);
  secHalf    = 1. / cos(M_PI / numMajor);
  rhat = (a * cosMidPhi + b * sinMidPhi);
  r    = secHalf * rhat;
  dist = secHalf * (-a * sinLeftPhi + b * cosLeftPhi);
  fulldist = 2. * sin(M_PI / numMajor) * r;
  outPhi = ((idxPhi + (dist/fulldist)) * 2. * M_PI) / numMajor;
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
        PetscPrintf(PETSC_COMM_SELF,"edge vertCoords=%g %g\n",vertCoords[i][0]+rMajor,vertCoords[i][1]);
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
        PetscPrintf(PETSC_COMM_SELF,"corner vertCoords=%g %g\n",vertCoords[i][0]+rMajor,vertCoords[i][1]);
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
  r += rMajor; /* centered back at the origin */

  cosOutPhi = cos(outPhi);
  sinOutPhi = sin(outPhi);
  xyz[0] = r * cosOutPhi;
  xyz[1] = r * sinOutPhi;
  xyz[2] = z;

  PetscFunctionReturn(0);
}

/* == Defining a base plex for a torus, which looks like a rectilinear donut */

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreatePICellBoxTorus"
static PetscErrorCode DMPlexCreatePICellBoxTorus (MPI_Comm comm, X2GridParticle *params, DM *dm)
{
  PetscMPIInt    rank;
  PetscInt       numCells = 0;
  PetscInt       numVerts = 0;
  PetscReal      rMajor   = params->rMajor;
  PetscReal      rMinor   = params->rMinor;
  PetscInt       numMajor = params->numMajor;
  int           *flatCells = NULL;
  double         *flatCoords = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    numCells = numMajor * 1;
    numVerts = numMajor * 4;
    ierr = PetscMalloc2(numCells * 8,&flatCells,numVerts * 3,&flatCoords);CHKERRQ(ierr);
    {
      double (*coords)[4][3] = (double (*) [4][3]) flatCoords;
      PetscInt i;

      for (i = 0; i < numMajor; i++) {
        PetscInt j;
        double cosphi, sinphi;

        cosphi = cos(2 * M_PI * i / numMajor);
        sinphi = sin(2 * M_PI * i / numMajor);

        for (j = 0; j < 4; j++) {
          double r, z;

          r = rMajor + rMinor * ( (j==1 || j==2)        ? -1. :  1.);
          z =          rMinor * ( (j < 2) ?  1. : -1. );

          coords[i][j][0] = cosphi * r;
          coords[i][j][1] = sinphi * r;
          coords[i][j][2] = z;
        }
      }
    }
    {
      int (*cells)[1][8] = (int (*) [1][8]) flatCells;
      PetscInt k, i, j = 0;

      for (i = 0; i < numMajor; i++) {
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
          z = mult * rMinor * sin(j * M_PI_2)         ;

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
  PetscFunctionReturn(0);
}

static void PICellCircleInflate(PetscReal r, PetscReal innerMult, PetscReal x, PetscReal y,
                                PetscReal *outX, PetscReal *outY)
{
  PetscReal l       = x + y;
  PetscReal rfrac   = l / r;

  if (rfrac >= innerMult) {
    PetscReal phifrac = l ? (y / l) : 0.5;
    PetscReal phi     = phifrac * M_PI_2;
    PetscReal cosphi  = cos(phi);
    PetscReal sinphi  = sin(phi);
    PetscReal isect   = innerMult / (cosphi + sinphi);
    PetscReal outfrac = (1. - rfrac) / (1. - innerMult);

    rfrac = pow(innerMult,outfrac);

    outfrac = (1. - rfrac) / (1. - innerMult);

    *outX = r * (outfrac * isect + (1. - outfrac)) * cosphi;
    *outY = r * (outfrac * isect + (1. - outfrac)) * sinphi;
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
static PetscErrorCode GeometryPICellTorus(DM base, PetscInt point, PetscInt dim, const PetscReal abc[], PetscReal xyz[], void *a_ctx)
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
    PICellCircleInflate(rMinor,innerMult,absR,absZ,&absR,&absZ);
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
  PetscInt       dim = 3;
  Mat            J;
  PetscFunctionBeginUser;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ctx.currevent = 0;
  ierr = PetscLogEventRegister("Process parts",0,&ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 0 */
  ierr = PetscLogEventRegister("Part. Send",0,&ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 1 */
  ierr = PetscLogEventRegister("Part. Recv",0,&ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 2 */
  ierr = PetscLogEventRegister("CreateMesh", DM_CLASSID, &ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 3 */
  ierr = PetscLogEventRegister("TwoSides", 0, &ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 4 */
  ierr = PetscLogEventRegister("Move parts", 0, &ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 5 */
  ierr = PetscLogEventRegister("Diagnostics", 0, &ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 6 */
  ierr = PetscLogEventRegister("Pre Push", 0, &ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 7 */
  ierr = PetscLogEventRegister("Push", 0, &ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 8 */
  assert(sizeof(ctx.events)/sizeof(ctx.events[0]) >= ctx.currevent);
#endif

  ierr = PetscCommDuplicate(PETSC_COMM_WORLD,&ctx.wComm,NULL);CHKERRQ(ierr);
  ierr = ProcessOptions( &ctx );CHKERRQ(ierr);

  /* construct DMs */
  ierr = PetscLogEventBegin(ctx.events[3],0,0,0,0);CHKERRQ(ierr);
  ierr = DMCreate(ctx.wComm, &ctx.dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(ctx.dm, &ctx);CHKERRQ(ierr);
  ierr = DMSetType(ctx.dm, DMPICELL);CHKERRQ(ierr); /* creates (DM_PICell *) dm->data */
  dmpi = (DM_PICell *) ctx.dm->data; assert(dmpi);
  /* setup solver grid */
  if (ctx.run_type == X2_ITER) {
    ierr = DMPlexCreatePICellITER(ctx.wComm,&ctx.particleGrid,&dmpi->dmgrid);CHKERRQ(ierr);
  }
  else if (ctx.run_type == X2_TORUS) {
    ierr = DMPlexCreatePICellTorus(ctx.wComm,&ctx.particleGrid,&dmpi->dmgrid);CHKERRQ(ierr);
  }
  else {
    ierr = DMPlexCreatePICellBoxTorus(ctx.wComm,&ctx.particleGrid,&dmpi->dmgrid);CHKERRQ(ierr);
    assert(ctx.run_type == X2_BOXTORUS);
  }
  ierr = DMSetApplicationContext(dmpi->dmgrid, &ctx);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) dmpi->dmgrid, "x2_");CHKERRQ(ierr);
  /* setup Discretization */
  ierr = PetscMalloc(1 * sizeof(PetscErrorCode (*)(PetscInt,const PetscReal [],PetscInt,PetscScalar*,void*)),&ctx.BCFuncs);
  CHKERRQ(ierr);
  ctx.BCFuncs[0] = zero;
  ierr = DMGetDimension(dmpi->dmgrid, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dmpi->dmgrid, dim, 1, PETSC_FALSE, NULL, -1, &dmpi->fem);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dmpi->fem, "potential");CHKERRQ(ierr);
  {
    DMLabel label;
    PetscDS prob;
    PetscInt id = 1;
    ierr = DMGetDS(dmpi->dmgrid, &prob);CHKERRQ(ierr);
    ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) dmpi->fem);CHKERRQ(ierr);
    ierr = DMCreateLabel(dmpi->dmgrid, "boundary");CHKERRQ(ierr);
    ierr = DMGetLabel(dmpi->dmgrid, "boundary", &label);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(dmpi->dmgrid, label);CHKERRQ(ierr);
    ierr = DMAddBoundary(dmpi->dmgrid, PETSC_TRUE, "wall", "boundary", 0, 0, NULL, (void (*)()) ctx.BCFuncs, 1, &id, &ctx);
    CHKERRQ(ierr);
  }
  {
    char      convType[256];
    PetscBool flg;
    ierr = PetscOptionsBegin(ctx.wComm, "", "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-x2_dm_type","Convert DMPlex to another format","x2.c",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
    if (flg) {
      DM dmConv;
      ierr = DMConvert(dmpi->dmgrid,convType,&dmConv);CHKERRQ(ierr);
      if (dmConv) {
        const char *prefix;
        PetscBool isForest;
        ierr = PetscObjectGetOptionsPrefix((PetscObject)dmpi->dmgrid,&prefix);CHKERRQ(ierr);
        ierr = PetscObjectSetOptionsPrefix((PetscObject)dmConv,prefix);CHKERRQ(ierr);
        ierr = DMDestroy(&dmpi->dmgrid);CHKERRQ(ierr);
        dmpi->dmgrid   = dmConv;
        ierr = DMIsForest(dmpi->dmgrid,&isForest);CHKERRQ(ierr);
        if (isForest) {
          if (ctx.run_type == X2_ITER) {
            ierr = DMForestSetBaseCoordinateMapping(dmpi->dmgrid,GeometryPICellITER,&ctx);CHKERRQ(ierr);
          }
          else if (ctx.run_type == X2_TORUS) {
            ierr = DMForestSetBaseCoordinateMapping(dmpi->dmgrid,GeometryPICellTorus,&ctx);CHKERRQ(ierr);
          }
          else {
            ierr = DMForestSetBaseCoordinateMapping(dmpi->dmgrid,NULL,&ctx);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  /* setup DM */
  ierr = DMSetFromOptions( ctx.dm );CHKERRQ(ierr);
  ierr = DMSetUp( ctx.dm );CHKERRQ(ierr); /* set all up & build initial grid */
  /* create SNESS */
  ierr = SNESCreate( ctx.wComm, &dmpi->snes);CHKERRQ(ierr);
  ierr = SNESSetDM( dmpi->snes, dmpi->dmgrid);CHKERRQ(ierr);
  ierr = DMSetMatType(dmpi->dmgrid,MATAIJ);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(dmpi->snes);CHKERRQ(ierr);
  ierr = DMSNESSetFunctionLocal(dmpi->dmgrid,  (PetscErrorCode (*)(DM,Vec,Vec,void*))DMPlexSNESComputeResidualFEM,&ctx);CHKERRQ(ierr);
  ierr = DMSNESSetJacobianLocal(dmpi->dmgrid,  (PetscErrorCode (*)(DM,Vec,Mat,Mat,void*))DMPlexSNESComputeJacobianFEM,&ctx);CHKERRQ(ierr);
  ierr = SNESSetUp( dmpi->snes );CHKERRQ(ierr);
  ierr = DMCreateMatrix(dmpi->dmgrid, &J);CHKERRQ(ierr);
  ierr = DMPlexSNESComputeJacobianFEM(dmpi->dmgrid, dmpi->phi, J, J, (void*)&ctx);CHKERRQ(ierr);
  ierr = SNESSetJacobian(dmpi->snes, J, J, NULL, NULL);CHKERRQ(ierr);
  ierr = MatView(J,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* setup particles */
  ierr = createParticles( &ctx );CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ctx.events[3],0,0,0,0);CHKERRQ(ierr);
  {
    PetscViewer    viewer = NULL;
    PetscBool      flg;
    ierr = PetscOptionsGetViewer(ctx.wComm,NULL,"-x2_dm_view",&viewer,NULL,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = DMView(dmpi->dmgrid,viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* do it */
  ierr = go( &ctx );CHKERRQ(ierr);
  PetscPrintf(ctx.wComm,"[%D] done - cleanup\n",ctx.rank);
  /* Cleanup */
  ierr = PetscFEDestroy(&dmpi->fem);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = destroyParticles(&ctx);CHKERRQ(ierr);
  ierr = DMDestroy(&ctx.dm);CHKERRQ(ierr);
  ierr = PetscFree(ctx.BCFuncs);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&ctx.wComm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

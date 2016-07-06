/* M. Adams, April 2015 */

static char help[] = "X2: A partical in cell code for tokamak plasmas using PICell.\n";

#ifdef X2_HAVE_ADVISOR
#include "advisor-annotate.h"  // Add to each module that contains Intel Advisor annotations
#endif
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
static PetscInt s_debug;
typedef enum {X2_ITER,X2_TORUS,X2_BOXTORUS} runType;
typedef struct {
  /* particle grid sizes */
  PetscInt npradius;
  PetscInt nptheta;
  PetscInt npphi; /* toroidal direction */
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
  long long gid; /* diagnostic, should be size of double */
} X2Particle;
#define X2_V_LEN 4
#define X2PROCLISTSIZE 256
#define X2_S_OF_V
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
  X2Particle *data;
#endif
  PetscInt    data_size, size, hole, top;
} X2PList;
/* send particle list */
typedef struct X2PSendList_TAG{
  X2Particle *data;
  PetscInt    data_size, size;
  PetscMPIInt proc;
} X2PSendList;
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
  PetscLogEvent events[12];
  PetscInt      currevent;
  PetscInt      bsp_chunksize;
  PetscInt      chunksize;
  runType       run_type;
  PetscBool     plot;
  /* MPI parallel data */
  MPI_Comm      particlePlaneComm,wComm;
  PetscMPIInt   rank,npe,npe_particlePlane,particlePlaneRank,ParticlePlaneIdx;
  /* grids & solver */
  DM             dm;
  X2GridParticle particleGrid;
  PetscBool      inflate_torus;
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
  PetscBool useElectrons;
  PetscInt  collisionPeriod;
  PetscReal max_vpar;
  PetscInt  nElems; /* size of array of particle lists */
  X2PList  *partlists[X2_NION+1]; /* 0: electron, 1:N ions */
  X2Species species[X2_NION+1]; /* 0: electron, 1:N ions */
  PetscInt  tablesize,tablecount[X2_NION+1]; /* hash table meta-data for proc-send list table */
} X2Ctx;

/* DMPlexFindLocalCellID */
#undef __FUNCT__
#define __FUNCT__ "DMPlexFindLocalCellID"
PetscErrorCode DMPlexFindLocalCellID(DM dm, PetscReal x[], PetscInt *elemID)
{
  PetscErrorCode ierr;
  PetscBool isForest;
  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscLogEventBegin(DMPICell_LocateProcess,dm,0,0,0);CHKERRQ(ierr);

  ierr = DMIsForest(dm,&isForest);CHKERRQ(ierr);
  if (isForest) {

  }
  else {
    /* Matt */
    *elemID = 0;
  }
  ierr = PetscLogEventEnd(DMPICell_LocateProcess,dm,0,0,0);CHKERRQ(ierr);
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
#define X2FREEV(d) {                                           \
    ierr = PetscFree2( d.gid,  d.w0 );CHKERRQ(ierr);           \
    ierr = PetscFree6( d.r,    d.z,  d.phi,                    \
                       d.vpar, d.mu, d.f0  );CHKERRQ(ierr);    \
  }
#define X2ALLOCV(s,d) {                                           \
    ierr = PetscMalloc2( s,&d.gid, s,&d.w0);CHKERRQ(ierr);      \
    ierr = PetscMalloc6( s,&d.r,   s,&d.z, s,&d.phi,                    \
                         s,&d.vpar,s,&d.mu,s,&d.f0  );CHKERRQ(ierr);    \
  }
#define X2P2V(p,d,i)         { d.r[i] = p->r;         d.z[i] = p->z;         d.phi[i] = p->phi;         d.vpar[i] = p->vpar;         d.mu[i] = p->mu;         d.w0[i] = p->w0;         d.f0[i] = p->f0;         d.gid[i] = p->gid;}
#define X2V2V(src,dst,is,id) { dst.r[id] = src.r[is]; dst.z[id] = src.z[is]; dst.phi[id] = src.phi[is]; dst.vpar[id] = src.vpar[is]; dst.mu[id] = src.mu[is]; dst.w0[id] = src.w0[is]; dst.f0[id] = src.f0[is]; dst.gid[id] = src.gid[is];}
#define X2V2P(p,d,i)         { p->r = d.r[i];         p->z = d.z[i];         p->phi = d.phi[i];         p->vpar = d.vpar[i];         p->mu = d.mu[i];         p->w0 = d.w0[i];         p->f0 = d.f0[i];         p->gid = d.gid[i];}

/* particle list */
#undef __FUNCT__
#define __FUNCT__ "X2PListCreate"
PetscErrorCode X2PListCreate(X2PList *l, PetscInt msz)
{
  PetscErrorCode ierr;
  l->size=0;
  l->top=0;
  l->hole=-1;
  l->data_size = X2_V_LEN*(msz/X2_V_LEN);
#ifdef X2_S_OF_V
  X2ALLOCV(l->data_size,l->data_v);
#else
  ierr = PetscMalloc1(l->data_size, &l->data);CHKERRQ(ierr);
#endif
  return ierr;
}
PetscErrorCode X2PListClear(X2PList *l)
{
  l->size=0; /* keep memory but kill data */
  l->top=0;
  l->hole=-1;
  return 0;
}
#undef __FUNCT__
#define __FUNCT__ "X2PListDestroy"
PetscErrorCode X2PListDestroy(X2PList *l)
{
  PetscErrorCode ierr;
#ifdef X2_S_OF_V
  X2FREEV(l->data_v);
#else
  ierr = PetscFree(l->data);CHKERRQ(ierr);
  l->data = 0;
#endif
  l->size=0;
  l->top=0;
  l->hole=-1;
  l->data_size = 0;
  return 0;
}
#undef __FUNCT__
#define __FUNCT__ "X2PListAdd"
PetscErrorCode X2PListAdd( X2PList *l, X2Particle *p, X2PListPos *ppos)
{
  PetscFunctionBeginUser;
  if (l->size==l->data_size) {
#ifdef X2_S_OF_V
    X2Particle_v data2;
#else
    X2Particle *data2; /* make this arrays of X2Particle members for struct-of-arrays */
#endif
    int i;
    PetscErrorCode ierr;
    l->data_size *= 2;
#ifdef X2_S_OF_V
    X2ALLOCV(l->data_size,data2);
#pragma simd vectorlengthfor(PetscScalar)
    for (i=0;i<l->size;i++) {
      X2V2V(l->data_v,data2,i,i);
    }
    X2FREEV(l->data_v);
    l->data_v = data2;
#else
    ierr = PetscMalloc1(l->data_size, &data2);CHKERRQ(ierr);
    for (i=0;i<l->size;i++) data2[i] = l->data[i];
    ierr = PetscFree(l->data);CHKERRQ(ierr);
    l->data = data2;
#endif
    assert(l->hole == -1);
  }
  if (l->hole != -1) { /* have a hole - fill it */
    X2PListPos idx = l->hole; assert(idx<l->data_size);
#ifdef X2_S_OF_V
    if (l->data_v.gid[idx] == 0) l->hole = -1; /* filled last hole */
    else if (l->data_v.gid[idx]>=0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"X2PListAdd: hole with non-neg gid!!!",l->data_v.gid[idx]);
    else l->hole = (X2PListPos)(-l->data_v.gid[idx] - 1); /* use gid as pointer */
    X2P2V(p,l->data_v,idx);
#else
    if (l->data[idx].gid == 0) l->hole = -1; /* filled last hole */
    else if (l->data[idx].gid>=0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"X2PListAdd: hole with non-neg gid!!!",l->data[idx].gid);
    else l->hole = (X2PListPos)(-l->data[idx].gid - 1); /* use gid as pointer */
    l->data[idx] = *p;
#endif
    if (ppos) *ppos = idx;
  }
  else {
    X2PListPos i = l->top++;
#ifdef X2_S_OF_V
    X2P2V(p,l->data_v,i);
#else
    l->data[i] = *p;
#endif
    if (ppos) *ppos = i;
  }
  l->size++;
  assert(l->top >= l->size);
  PetscFunctionReturn(0);
}
PetscErrorCode X2PListSetAt(X2PList *l, X2PListPos pos, X2Particle *part)
{
#ifdef X2_S_OF_V
  X2P2V(part,l->data_v,pos);
#else
  l->data[pos] = *part;
#endif
  return 0;
}

PetscErrorCode X2PListCompress(X2PList *l)
{
  PetscInt ii;
  /* fill holes with end of list */
  for ( ii = 0 ; ii < l->top && l->top > l->size ; ii++) {
#ifdef X2_S_OF_V
    if (l->data_v.gid[ii] <= 0)
#else
    if (l->data[ii].gid <= 0)
#endif
    {
      l->top--; /* maybe data to move */
      if (ii == l->top) /* just pop hole at end */ ;
      else {
#ifdef X2_S_OF_V
        while (l->data_v.gid[l->top] <= 0) l->top--; /* get real data */
        X2V2V(l->data_v,l->data_v,l->top,ii);
#else
        while (l->data[l->top].gid <= 0) l->top--; /* get real data */
        l->data[ii] = l->data[l->top]; /* now above */
#endif
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
#undef __FUNCT__
#define __FUNCT__ "X2PListGetHead"
PetscErrorCode X2PListGetHead(X2PList *l, X2PListPos *pos)
{
 PetscFunctionBeginUser;
 if (l->size==0) {
    *pos = 0; /* past end */
  }
  else {
    X2PListPos idx=0;
#ifdef X2_S_OF_V
    while (l->data_v.gid[idx] <= 0) idx++;
#else
    while (l->data[idx].gid <= 0) idx++;
#endif
    *pos = idx - 1; /* eg -1 */
  }
  PetscFunctionReturn(0);
}

PetscErrorCode X2PListGetNext(X2PList *l, X2Particle *p, X2PListPos *pos)
{
  /* l->size == 0 can happen on empty list */
  (*pos)++; /* get next position */
  if (*pos >= l->data_size || *pos >= l->top) return 1; /* hit end, can go past if list is just drained */
#ifdef X2_S_OF_V
  while(l->data_v.gid[*pos] <= 0 && *pos < l->data_size && *pos < l->top) (*pos)++; /* skip holes */
  X2V2P(p,l->data_v,*pos); /* return copy */
#else
  while(l->data[*pos].gid <= 0 && *pos < l->data_size && *pos < l->top) (*pos)++; /* skip holes */
  *p = l->data[*pos]; /* return copy */
#endif
  return 0;
}
#undef __FUNCT__
#define __FUNCT__ "X2PListRemoveAt"
PetscErrorCode X2PListRemoveAt( X2PList *l, X2PListPos pos)
{
  PetscFunctionBeginUser;
  if(pos >= l->data_size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"X2PListRemoveAt past end of data %d %d",pos,l->data_size);
  if(pos >= l->top) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"X2PListRemoveAt past end of top pointer %d %d",pos,l->top);

  if (pos == l->top-1) {
    l->top--; /* simple pop */
  }
  else {
#ifdef X2_S_OF_V
    if (l->hole==-1) l->data_v.gid[pos] = 0; /* sentinal */
    else l->data_v.gid[pos] = -(l->hole + 1); /* hole >= 0 */
#else
    if (l->hole==-1) l->data[pos].gid = 0; /* sentinal */
    else l->data[pos].gid = -(l->hole + 1); /* hole >= 0 */
#endif
    l->hole = pos; /* head of linked list of holes */
  }
  l->size--;
  if (!l->size) { /* lets reset if we drained the list */
    l->hole = -1;
    l->top = 0;
  }
  assert(l->top >= l->size);
  PetscFunctionReturn(0);
}

PetscInt X2PListMaxSize(X2PList *l) {
  return l->data_size;
}

PetscInt X2PListSize(X2PList *l) {
  return l->size;
}

/* particle send list, non-vector simple array list */
PetscInt X2PSendListSize(X2PSendList *l) {
  return l->size;
}
#undef __FUNCT__
#define __FUNCT__ "X2PSendListCreate"
PetscErrorCode X2PSendListCreate(X2PSendList *l, PetscInt msz)
{
  PetscErrorCode ierr;
  l->size=0;
  l->data_size = msz;
  ierr = PetscMalloc1(l->data_size, &l->data);CHKERRQ(ierr);
  return ierr;
}
PetscErrorCode X2PSendListClear(X2PSendList *l)
{
  l->size=0; /* keep memory but kill data */
  return 0;
}
#undef __FUNCT__
#define __FUNCT__ "X2PSendListDestroy"
PetscErrorCode X2PSendListDestroy(X2PSendList *l)
{
  PetscErrorCode ierr;
  ierr = PetscFree(l->data);CHKERRQ(ierr);
  l->data = 0;
  l->size=0;
  l->data_size = 0;
  return ierr;
}
#undef __FUNCT__
#define __FUNCT__ "X2PSendListAdd"
PetscErrorCode X2PSendListAdd( X2PSendList *l, X2Particle *p)
{
  PetscFunctionBeginUser;
  if (l->size==l->data_size) {
    X2Particle *data2; /* make this arrays of X2Particle members for struct-of-arrays */
    int i;PetscErrorCode ierr;
    PetscPrintf(PETSC_COMM_SELF," *** X2PSendListAdd expanded list %d --> %d%d\n",l->data_size,2*l->data_size);
    l->data_size *= 2;
    ierr = PetscMalloc1(l->data_size, &data2);CHKERRQ(ierr);
    for (i=0;i<l->size;i++) data2[i] = l->data[i];
    ierr = PetscFree(l->data);CHKERRQ(ierr);
    l->data = data2;
  }
  l->data[l->size++] = *p;
  PetscFunctionReturn(0);
}
/*
   ProcessOptions: set parameters from input, setup w/o allocation, called first, no DM here
*/
#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions( X2Ctx *ctx )
{
  PetscErrorCode ierr,isp,k,sz;
  FILE *fp;
  PetscBool phiFlag,radFlag,thetaFlag,flg,chunkFlag;
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
  ctx->particleGrid.npphi  = 1;
  ctx->particleGrid.npradius = 1;
  ctx->particleGrid.nptheta  = 1;
  ctx->particleGrid.numMajor = 4; /* number of poloidal planes (before refinement) */
  ctx->particleGrid.innerMult= M_SQRT2 - 1.;

  ierr = PetscOptionsBegin(ctx->wComm, "", "Poisson Problem Options", "X2");CHKERRQ(ierr);
  /* general options */
  s_debug = 0;
  ierr = PetscOptionsInt("-debug", "The debugging level", "x2.c", s_debug, &s_debug, NULL);CHKERRQ(ierr);
  ctx->plot = PETSC_TRUE;
  ierr = PetscOptionsBool("-plot", "Write plot files (particles)", "x2.c", ctx->plot, &ctx->plot, NULL);CHKERRQ(ierr);
  ctx->chunksize = X2_V_LEN; /* too small */
  ierr = PetscOptionsInt("-chunksize", "Size of particle list to chunk sends", "x2.c", ctx->chunksize, &ctx->chunksize,&chunkFlag);CHKERRQ(ierr);
  if (chunkFlag) ctx->chunksize = X2_V_LEN*(ctx->chunksize/X2_V_LEN);
  if (ctx->chunksize<=0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB," invalid chuck size = %D",ctx->chunksize);
  ctx->bsp_chunksize = 0; /* 32768; */
  ierr = PetscOptionsInt("-bsp_chunksize", "Size of chucks for PETSc's TwoSide communication (0 to use 'nonblocking consensus')", "x2.c", ctx->bsp_chunksize, &ctx->bsp_chunksize, NULL);CHKERRQ(ierr);
  if (ctx->bsp_chunksize<0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB," invalid BSP chuck size = %D",ctx->bsp_chunksize);
  ctx->tablesize = 256; /* hash table size of processors to send to */
  ierr = PetscOptionsInt("-proc_send_table_size", "Size of hash table proc->send_list", "x2.c",ctx->tablesize, &ctx->tablesize, NULL);CHKERRQ(ierr);

  /* Domain and mesh definition */
  ierr = PetscOptionsReal("-rMajor", "Major radius of torus", "x2.c", ctx->particleGrid.rMajor, &ctx->particleGrid.rMajor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rMinor", "Minor radius of torus", "x2.c", ctx->particleGrid.rMinor, &ctx->particleGrid.rMinor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-numMajor", "Number of cells per major circle", "x2.c", ctx->particleGrid.numMajor, &ctx->particleGrid.numMajor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-innerMult", "Percent of minor radius taken by inner square", "x2.c", ctx->particleGrid.innerMult, &ctx->particleGrid.innerMult, NULL);CHKERRQ(ierr);

  ierr = PetscOptionsInt("-npphi_particles", "Number of planes for particle mesh", "x2.c", ctx->particleGrid.npphi, &ctx->particleGrid.npphi, &phiFlag);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-npradius_particles", "Number of radial cells for particle mesh", "x2.c", ctx->particleGrid.npradius, &ctx->particleGrid.npradius, &radFlag);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nptheta_particles", "Number of theta cells for particle mesh", "x2.c", ctx->particleGrid.nptheta, &ctx->particleGrid.nptheta, &thetaFlag);CHKERRQ(ierr);
  ctx->npe_particlePlane = -1;
  if (ctx->particleGrid.npphi*ctx->particleGrid.npradius*ctx->particleGrid.nptheta != ctx->npe) { /* recover from inconsistant grid/procs */
    if (thetaFlag && radFlag && phiFlag) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"over constrained number of particle processes npe (%D) != %D",ctx->npe,ctx->particleGrid.npphi*ctx->particleGrid.npradius*ctx->particleGrid.nptheta);

    if (!thetaFlag && radFlag && phiFlag) ctx->particleGrid.nptheta = ctx->npe/(ctx->particleGrid.npphi*ctx->particleGrid.npradius);
    else if (thetaFlag && !radFlag && phiFlag) ctx->particleGrid.npradius = ctx->npe/(ctx->particleGrid.npphi*ctx->particleGrid.nptheta);
    else if (thetaFlag && radFlag && !phiFlag) ctx->particleGrid.npphi = ctx->npe/(ctx->particleGrid.npradius*ctx->particleGrid.nptheta);
    else if (!thetaFlag && !radFlag && !phiFlag) {
      ctx->npe_particlePlane = (int)pow((double)ctx->npe,0.6667);
      ctx->particleGrid.npphi = ctx->npe/ctx->npe_particlePlane;
      ctx->particleGrid.npradius = (int)(sqrt((double)ctx->npe_particlePlane)+0.5);
      ctx->particleGrid.nptheta = ctx->npe_particlePlane/ctx->particleGrid.npradius;
      if (ctx->particleGrid.npphi*ctx->particleGrid.npradius*ctx->particleGrid.nptheta != ctx->npe) {
	ctx->particleGrid.npphi = ctx->npe;
      }
    }
    else if (ctx->particleGrid.npphi*ctx->particleGrid.npradius*ctx->particleGrid.nptheta != ctx->npe) { /* recover */
      if (!ctx->npe%ctx->particleGrid.npphi) {
	ctx->npe_particlePlane = ctx->npe/ctx->particleGrid.npphi;
	ctx->particleGrid.npradius = (int)(sqrt((double)ctx->npe_particlePlane)+0.5);
	ctx->particleGrid.nptheta = ctx->npe_particlePlane/ctx->particleGrid.npradius;
      }
      else {
      }
    }
    if (ctx->particleGrid.npphi*ctx->particleGrid.npradius*ctx->particleGrid.nptheta != ctx->npe) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"particle grids do not work npe (%D) != %D",ctx->npe,ctx->particleGrid.npphi*ctx->particleGrid.npradius*ctx->particleGrid.nptheta);
  }

  /* particle grids: <= npe, <= num solver planes */
  if (ctx->npe < ctx->particleGrid.npphi) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"num particle planes npphi (%D) > npe (%D)",ctx->particleGrid.npphi,ctx->npe);
  if (ctx->npe%ctx->particleGrid.npphi) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"np=%D not divisible by number of particle planes (npphi) %D",ctx->npe,ctx->particleGrid.npphi);

  if (ctx->npe_particlePlane == -1) ctx->npe_particlePlane = ctx->npe/ctx->particleGrid.npphi;
  if (ctx->npe_particlePlane != ctx->npe/ctx->particleGrid.npphi) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inconsistant number planes (%D), pes (%D), and pe/plane (%D) requested",ctx->particleGrid.npphi,ctx->npe,ctx->npe_particlePlane);

  if (ctx->particleGrid.nptheta*ctx->particleGrid.npradius != ctx->npe_particlePlane) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"%D particle cells/plane != %D pe/plane",ctx->particleGrid.nptheta*ctx->particleGrid.npradius,ctx->npe_particlePlane);
  if (ctx->particleGrid.nptheta*ctx->particleGrid.npradius*ctx->particleGrid.npphi != ctx->npe) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"%D particle cells != %D npe",ctx->particleGrid.nptheta*ctx->particleGrid.npradius*ctx->particleGrid.npphi,ctx->npe);
  ctx->ParticlePlaneIdx = ctx->rank/ctx->npe_particlePlane;
  ctx->particlePlaneRank = ctx->rank%ctx->npe_particlePlane;

  /* PetscPrintf(PETSC_COMM_SELF,"[%D] pe/plane=%D, my plane=%D, my local rank=%D, npphi=%D\n",ctx->rank,ctx->npe_particlePlane,ctx->ParticlePlaneIdx,ctx->particlePlaneRank,ctx->particleGrid.npphi);    */

  /* time integrator */
  ctx->msteps = 1;
  ierr = PetscOptionsInt("-mstep", "Maximum number of time steps", "x2.c", ctx->msteps, &ctx->msteps, NULL);CHKERRQ(ierr);
  ctx->maxTime = 1000000000.;
  ierr = PetscOptionsReal("-maxTime", "Maximum time", "x2.c",ctx->maxTime,&ctx->maxTime,NULL);CHKERRQ(ierr);
  ctx->dt = 1.;
  ierr = PetscOptionsReal("-dt","Time step","x2.c",ctx->dt,&ctx->dt,NULL);CHKERRQ(ierr);
  /* particles */
  ctx->npart_flux_tube = 10;
  ierr = PetscOptionsInt("-npart_flux_tube", "Number of particles local (flux tube cell)", "x2.c", ctx->npart_flux_tube, &ctx->npart_flux_tube, NULL);CHKERRQ(ierr);
  if (!chunkFlag) ctx->chunksize = X2_V_LEN*((ctx->npart_flux_tube/80+1)/X2_V_LEN); /* an intelegent message chunk size */
  if (ctx->chunksize<64 && !chunkFlag) ctx->chunksize = 64; /* 4K messages minumum */

  if (s_debug>0) PetscPrintf(ctx->wComm,"[%D] npe=%D part=[%D,%D,%D], send size (chunksize) is %d particles.\n",ctx->rank,ctx->npe,ctx->particleGrid.npphi,ctx->particleGrid.nptheta,ctx->particleGrid.npradius,ctx->chunksize);

  ctx->collisionPeriod = 10;
  ierr = PetscOptionsInt("-collisionPeriod", "Period between collision operators", "x2.c", ctx->collisionPeriod, &ctx->collisionPeriod, NULL);CHKERRQ(ierr);
  ctx->useElectrons = PETSC_FALSE;
  ierr = PetscOptionsBool("-use_electrons", "Include electrons", "x2.c", ctx->useElectrons, &ctx->useElectrons, NULL);CHKERRQ(ierr);
  ctx->max_vpar = .03;
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
#define qsafty(psi) (3.*pow(psi,2.0))

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
PetscErrorCode X2GridParticleGetProc_FluxTube( const X2GridParticle *grid, /* X2Particle *part, */
                                               PetscReal psi, PetscReal theta, PetscReal phi,
                                               PetscMPIInt *pe, PetscInt *elem)
{
  const PetscReal rminor=grid->rMinor;
  const PetscReal dphi=2.*M_PI/(PetscReal)grid->npphi;
  const PetscReal dth=2.*M_PI/(PetscReal)grid->nptheta;
  PetscMPIInt planeIdx,irs,iths;
  PetscFunctionBeginUser;

  theta = fmod( theta - qsafty(psi/grid->rMinor)*phi + 20.*M_PI, 2.*M_PI);  /* pull back to reference grid */
  planeIdx = (PetscMPIInt)(phi/dphi)*grid->npradius*grid->nptheta; /* assumeing one particle cell per PE */
  iths = (PetscMPIInt)(theta/dth);                               assert(iths<grid->nptheta);
  irs = (PetscMPIInt)((PetscReal)grid->npradius*psi*psi/(rminor*rminor));assert(irs<grid->npradius);
  *pe = planeIdx + irs*grid->nptheta + iths;
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
  PetscBool isForest;
  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  /* ierr = PetscLogEventBegin(DMPICell_GetJet,dm,0,0,0);CHKERRQ(ierr); */

  ierr = DMIsForest(dm,&isForest);CHKERRQ(ierr);
  if (isForest) {
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);
    *pe = rank; /* noop -- need to add a local lookup for 'elem' if (*pe == rank) */
    *elem = 0;
  }
  else {
    /* Matt do your thing here */
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);
    *pe = rank; /* noop -- need to add a local lookup for 'elem' if (*pe == rank) */
    *elem = 0;
  }
  /* ierr = PetscLogEventEnd(DMPICell_GetJet,dm,0,0,0);CHKERRQ(ierr); */
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
PetscErrorCode X2PListWrite(X2PList l[], PetscInt nLists, PetscMPIInt rank, PetscMPIInt npe, MPI_Comm comm, char fname1[], char fname2[])
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
  for (nparticles=0,elid=0;elid<nLists;elid++) {
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
    for (nparticles=0,elid=0;elid<nLists;elid++) {
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
      for (nparticles=0,elid=0;elid<nLists;elid++) {
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
     - slists: array of non-blocking send caches (!ctx->bsp_chunksize only), cleared
   Output:
*/
#undef __FUNCT__
#define __FUNCT__ "shiftParticles"
PetscErrorCode shiftParticles( const X2Ctx *ctx, X2PSendList *sendListTable, const PetscInt isp, const PetscInt irk, PetscInt *const nIsend,
                               X2PList particlelist[], X2ISend slist[], PetscMPIInt tag, PetscBool solver )
{
  PetscErrorCode ierr;
  const int part_dsize = sizeof(X2Particle)/sizeof(double); assert(sizeof(X2Particle)%sizeof(double)==0);
  PetscInt ii,jj,kk,mm,idx,elid;
  DM dm;
  DM_PICell *dmpi;
  MPI_Datatype mtype;

  PetscFunctionBeginUser;
  PetscDataTypeToMPIDataType(PETSC_REAL,&mtype);
  dmpi = (DM_PICell *) ctx->dm->data;
  dm = dmpi->dmgrid;
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(ctx->events[2],0,0,0,0);CHKERRQ(ierr);
#endif
  if ( ctx->bsp_chunksize ) { /* use BSP */
    PetscMPIInt  nto,*fromranks;
    PetscMPIInt *toranks;
    X2Particle  *fromdata,*todata,*pp;
    PetscMPIInt  nfrom;
    int sz;
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
    /* count send  */
    for (ii=0,nto=0;ii<ctx->tablesize;ii++) {
      if (sendListTable[ii].data_size != 0) {
	sz = X2PSendListSize(&sendListTable[ii]);
	for (jj=0 ; jj<sz ; jj += ctx->chunksize) nto++; /* can just figure this out */
      }
    }
    /* make to ranks & data */
    ierr = PetscMalloc1(nto, &toranks);CHKERRQ(ierr);
    ierr = PetscMalloc1(ctx->chunksize*nto, &todata);CHKERRQ(ierr);
    for (ii=0,nto=0,pp=todata;ii<ctx->tablesize;ii++) {
      if (sendListTable[ii].data_size) {
	if ((sz=X2PSendListSize(&sendListTable[ii])) > 0) {
	  /* empty the list */
	  for (jj=0, mm=0 ; jj<sz ; jj += ctx->chunksize) {
	    toranks[nto++] = sendListTable[ii].proc;
	    for (kk=0 ; kk<ctx->chunksize && mm < sz; kk++, mm++) {
	      *pp++ = sendListTable[ii].data[mm];
	    }
	  }
	  assert(mm==sz);
	  while (kk++ < ctx->chunksize) { /* pad with zeros (gid is 1-based) */
	    pp->gid = 0;
	    pp++;
	  }
          /* get ready for next round */
	  ierr = X2PSendListClear(&sendListTable[ii]);CHKERRQ(ierr);
          assert(X2PSendListSize(&sendListTable[ii])==0);
          assert(sendListTable[ii].data_size);
	} /* a list */
      }
    }

    /* do it */
    ierr = PetscCommBuildTwoSided( ctx->wComm, ctx->chunksize*part_dsize, mtype, nto, toranks, (double*)todata,
				   &nfrom, &fromranks, &fromdata);
    CHKERRQ(ierr);
    for (ii=0, pp = fromdata ; ii<nfrom ; ii++) {
      for (jj=0 ; jj<ctx->chunksize ; jj++, pp++) {
	if (pp->gid > 0) {
          if (solver) {
            PetscReal x[3];
	    cylindricalToCart(pp->r, pp->z, pp->phi, x);
            ierr = DMPlexFindLocalCellID(dm, x, &elid);CHKERRQ(ierr);
          }
          else elid = 0; /* non-solvers just put in element 0's list */
	  ierr = X2PListAdd( &particlelist[elid], pp, NULL);CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscFree(todata);CHKERRQ(ierr);
    ierr = PetscFree(fromranks);CHKERRQ(ierr);
    ierr = PetscFree(fromdata);CHKERRQ(ierr);
    ierr = PetscFree(toranks);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
  }
  else { /* non-blocking consensus */
    X2Particle *data;
    PetscBool   done=PETSC_FALSE,bar_act=PETSC_FALSE;
    MPI_Request ib_request;
    PetscInt    numSent;
    MPI_Status  status;
    PetscMPIInt flag,sz,sz1;
    /* send lists */
    for (ii=0;ii<ctx->tablesize;ii++) {
      if (sendListTable[ii].data_size != 0) {
	if ((sz=X2PSendListSize(&sendListTable[ii])) > 0) {
	  if (*nIsend==X2PROCLISTSIZE) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"process send table too small (%D)",X2PROCLISTSIZE);
#if defined(PETSC_USE_LOG)
	  ierr = PetscLogEventBegin(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
	  slist[*nIsend].proc = sendListTable[ii].proc;
          slist[*nIsend].data = sendListTable[ii].data; /* cache data */
          /* send and reset - we can just send this because it is dense */
	  ierr = MPI_Isend((void*)slist[*nIsend].data,sz*part_dsize,mtype,slist[*nIsend].proc,tag,ctx->wComm,&slist[*nIsend].request);
	  CHKERRQ(ierr);
          /* PetscPrintf(PETSC_COMM_SELF,"\t[%D] (1) send proc %d, %d particles\n",ctx->rank,slist[*nIsend].proc,sz); */
	  (*nIsend)++;
          /* ready for next round, save meta-data  */
	  ierr = X2PSendListClear( &sendListTable[ii] );CHKERRQ(ierr);
	  assert(sendListTable[ii].data_size == ctx->chunksize);
	  ierr = PetscMalloc1(ctx->chunksize, &sendListTable[ii].data);CHKERRQ(ierr);
	  assert(!(sendListTable[ii].data_size && X2PSendListSize(&sendListTable[ii])));
#if defined(PETSC_USE_LOG)
	  ierr = PetscLogEventEnd(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
	}
      }
      /* else - an empty list */
    }
    numSent = *nIsend; /* size of send array */
    /* process receives - non-blocking consensus */
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(ctx->events[3],0,0,0,0);CHKERRQ(ierr);
#endif
    /* process receives - non-blocking consensus */
    ierr = PetscMalloc1(ctx->chunksize, &data);CHKERRQ(ierr);
    while (!done) {
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
      /* probe for incoming */
      do {
	ierr = MPI_Iprobe(MPI_ANY_SOURCE, tag, ctx->wComm, &flag, &status);CHKERRQ(ierr);
	if (flag) {
	  MPI_Get_count(&status, mtype, &sz); assert(sz<=ctx->chunksize*part_dsize && sz%part_dsize==0);
	  ierr = MPI_Recv((void*)data,sz,mtype,status.MPI_SOURCE,tag,ctx->wComm,&status);CHKERRQ(ierr);
	  MPI_Get_count(&status, mtype, &sz1); assert(sz1<=ctx->chunksize*part_dsize && sz1%part_dsize==0); assert(sz==sz1);
	  sz = sz/part_dsize;
	  for (jj=0;jj<sz;jj++) {
            if (solver) {
              PetscReal x[3];
	      cylindricalToCart(data[jj].r, data[jj].z, data[jj].phi, x);
              ierr = DMPlexFindLocalCellID(dm, x, &elid);CHKERRQ(ierr);
            }
            else elid = 0; /* non-solvers just put in element 0's list */
            ierr = X2PListAdd( &particlelist[elid], &data[jj], NULL);CHKERRQ(ierr);
          }
	}
      } while (flag);
    } /* non-blocking consensus */
    ierr = PetscFree(data);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(ctx->events[3],0,0,0,0);CHKERRQ(ierr);
#endif
  } /* switch for BPS */
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(ctx->events[2],0,0,0,0);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

/* add corners to get bounding box */
static void prewrite(X2Ctx *ctx, X2PList *l, X2PListPos *ppos1,  X2PListPos *ppos2)
{
  X2Particle part;
  PetscReal r,z,phi;
  if (ctx->rank==0) {
    r = 1.414213562373095*(ctx->particleGrid.rMajor + ctx->particleGrid.rMinor);
    z = ctx->particleGrid.rMinor;
    phi = M_PI/4.;
    X2ParticleCreate(&part,1,r,z,phi,0.);
    X2PListAdd(l,&part,ppos1);
    z = -z;
    phi += M_PI;
    X2ParticleCreate(&part,2,r,z,phi,0.);
    X2PListAdd(l,&part,ppos2);
  }
}
static void postwrite(X2Ctx *ctx, X2PList *l, X2PListPos *ppos1,  X2PListPos *ppos2)
{
  if (ctx->rank==0) {
    X2PListRemoveAt(l,*ppos2);
    X2PListRemoveAt(l,*ppos1);
  }
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
static PetscErrorCode processParticles( X2Ctx *ctx, const PetscReal dt, X2PSendList *sendListTable, const PetscMPIInt tag,
					const int irk, const int istep, PetscBool solver)
{
  X2GridParticle *grid = &ctx->particleGrid;         assert(sendListTable); /* always used now */
  DM_PICell *dmpi = (DM_PICell *) ctx->dm->data;     assert(solver || irk<0); /* don't push flux tubes */
  PetscReal   psi,theta,dphi,rmaj=grid->rMajor,rminor=grid->rMinor;
  PetscMPIInt pe,hash,ii;
  X2Particle  part;
  X2PListPos  pos;
  PetscErrorCode ierr;
  const int part_dsize = sizeof(X2Particle)/sizeof(double);
  Vec          jetVec,xVec,vVec;
  PetscScalar *xx=0,*jj=0,*vv=0,*xx0=0,*jj0=0,*vv0=0;
  PetscInt isp,order=1,nslist,nlistsTot,elid,idx,one=1,three=3;
  int origNlocal,nmoved;
  X2ISend slist[X2PROCLISTSIZE];
  PetscFunctionBeginUser;
  MPI_Barrier(ctx->wComm);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(ctx->events[1],0,0,0,0);CHKERRQ(ierr);
#endif
  nslist = 0;
  nmoved = 0;
  nlistsTot = origNlocal = 0;
  if (!dmpi) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"DM_PICell data not created");
  if (irk>=0) {
    ierr = VecZeroEntries(dmpi->rho);CHKERRQ(ierr); /* zero density to get ready for next deposition */
  }

  /* push particles, if necc., and make send lists */
  for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++) {
    /* loop over element particle lists */
    for (elid=0;elid<ctx->nElems;elid++) {
      X2PList *list = &ctx->partlists[isp][elid];
      if (X2PListSize(list)==0)continue;
      ierr = X2PListCompress(list);CHKERRQ(ierr);
      origNlocal += X2PListSize(list);

      /* get Cartesian coordinates (not used for flux tube move) */
      if (solver) {
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventBegin(ctx->events[7],0,0,0,0);CHKERRQ(ierr); /* timer on particle list */
#endif
        /* make vectors for this element */
        ierr = VecCreateSeq(PETSC_COMM_SELF,three*X2PListSize(list), &xVec);CHKERRQ(ierr);
        ierr = VecCreateSeq(PETSC_COMM_SELF,three*X2PListSize(list), &jetVec);CHKERRQ(ierr);
        ierr = VecSetBlockSize(xVec,three);CHKERRQ(ierr);
        ierr = VecSetBlockSize(jetVec,three);CHKERRQ(ierr);
        /* make coordinates array to get gradients */
        ierr = VecGetArray(xVec,&xx0);CHKERRQ(ierr); xx = xx0;
#ifdef X2_S_OF_V
#pragma simd vectorlengthfor(PetscScalar)
	for (pos=0 ; pos < list->size ; pos++, xx += 3) {
	  PetscReal r=list->data_v.r[pos], z=list->data_v.z[pos], phi=list->data_v.phi[pos];
	  cylindricalToCart(r, z, phi, xx);
	}
#else
	ierr = X2PListGetHead( list, &pos );CHKERRQ(ierr);
	while ( !X2PListGetNext(list, &part, &pos) ) {
	  cylindricalToCart(part.r, part.z, part.phi, xx);
          xx += 3;
        }
#endif
        ierr = VecRestoreArray(xVec,&xx0);CHKERRQ(ierr);
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
        ierr = DMPICellGetJet(dmpi->dmgrid, xVec, order, jetVec, elid);CHKERRQ(ierr);
        /* vectorize (todo) push: theta = theta + q*dphi .... grad not used */
        ierr = VecGetArray(xVec,&xx0);CHKERRQ(ierr); xx = xx0;
        ierr = VecGetArray(jetVec,&jj0);CHKERRQ(ierr); jj = jj0;
        for (pos=0 ; pos < list->size ; pos++, xx += 3, jj += 3 ) {
	  /* push particle, real data, could do it on copy for non-final stage of TS */
#ifdef X2_S_OF_V
	  PetscReal r=list->data_v.r[pos] - rmaj, z=list->data_v.z[pos];
          cylindricalToPolPlane(r, z, psi, theta );
          dphi = (dt*list->data_v.vpar[pos])/(2.*M_PI*list->data_v.r[pos]);  /* toroidal step */
          list->data_v.phi[pos] += dphi;
          list->data_v.phi[pos] = fmod( list->data_v.phi[pos] + 20.*M_PI, 2.*M_PI);
          theta += qsafty(psi/rminor)*dphi;  /* twist */
          theta = fmod( theta + 20.*M_PI, 2.*M_PI);
          polPlaneToCylindrical( psi, theta, r, z); /* time spent here */
          list->data_v.r[pos] = rmaj + r;
          list->data_v.z[pos] = z;
#else
          X2Particle *ppart = &list->data[pos];
          PetscReal r = ppart->r - rmaj, z = ppart->z;
          cylindricalToPolPlane( r, z, psi, theta );
          dphi = (dt*ppart->vpar)/(2.*M_PI*ppart->r);  /* toroidal step */
          ppart->phi += dphi;
          ppart->phi = fmod( ppart->phi + 20.*M_PI, 2.*M_PI);
          theta += qsafty(psi/rminor)*dphi;  /* twist */
          theta = fmod( theta + 20.*M_PI, 2.*M_PI);
          polPlaneToCylindrical( psi, theta, r, z); /* time spent here */
          ppart->r = rmaj + r;
          ppart->z = z;
#endif
        }
        ierr = VecRestoreArray(xVec,&xx0);
        ierr = VecRestoreArray(jetVec,&jj0);
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventEnd(ctx->events[8],0,0,0,0);CHKERRQ(ierr);
#endif
      }
      /* move */
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventBegin(ctx->events[5],0,0,0,0);CHKERRQ(ierr);
#endif
      if (solver) {
        ierr = VecGetArray(xVec,&xx0);CHKERRQ(ierr); xx = xx0;
      }
#ifdef X2_S_OF_V
      for (pos=0 ; pos < list->size ; pos++) {
	X2V2P((&part),list->data_v,pos); /* copy */
#else
      ierr = X2PListGetHead( list, &pos );CHKERRQ(ierr);
      while ( !X2PListGetNext(list, &part, &pos) ) {
#endif
        /* see if need communication? no: add density, yes: add to communication list */
        if (solver) {
          ierr = X2GridParticleGetProc_Solver(dmpi->dmgrid, xx, &pe, &idx);CHKERRQ(ierr);
        }
        else {
          PetscReal r = part.r - rmaj;
          cylindricalToPolPlane( r, part.z, psi, theta );
          ierr = X2GridParticleGetProc_FluxTube(grid, psi, theta, part.phi, &pe, &idx);CHKERRQ(ierr);
          assert(idx==0);
        }
	/* do something with the particle */
        if (pe==ctx->rank && idx==elid) { /* don't move and don't add */
          /* noop */
	} else { /* move: sendListTable && off proc, send to self for particles that move elements */
          /* add to list to send, find list with table lookup, send full lists - no vectorization */
          hash = (pe*593)%ctx->tablesize; /* hash */
          for (ii=0;ii<ctx->tablesize;ii++){
            if (sendListTable[hash].data_size==0) {
              ierr = X2PSendListCreate(&sendListTable[hash],ctx->chunksize);CHKERRQ(ierr);
              sendListTable[hash].proc = pe;
              ctx->tablecount[isp]++;
              if (ctx->tablecount[isp]==ctx->tablesize) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Table too small (%D)",ctx->tablesize);
            }
            if (sendListTable[hash].proc==pe) { /* found hash table entry */
              if (X2PSendListSize(&sendListTable[hash])==ctx->chunksize) {
		MPI_Datatype mtype;
#if defined(PETSC_USE_LOG)
		ierr = PetscLogEventBegin(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
		PetscDataTypeToMPIDataType(PETSC_REAL,&mtype);
                if (ctx->bsp_chunksize) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"cache too small (%D) for BSP TwoSided communication",ctx->chunksize);
                /* send and reset - we can just send this because it is dense, but no species data */
                if (nslist==X2PROCLISTSIZE) {
		  SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"process send table too small (%D) == snlist(%D)",nslist,(PetscInt)X2PROCLISTSIZE);
		}
                slist[nslist].data = sendListTable[hash].data; /* cache data */
                sendListTable[hash].data = 0; /* clear for safty, make ready for more */
                slist[nslist].proc = pe;
                ierr = MPI_Isend( (void*)slist[nslist].data,ctx->chunksize*part_dsize,mtype,pe,tag+isp,ctx->wComm,&slist[nslist].request);
                CHKERRQ(ierr);
                nslist++;
                /* ready for more */
                ierr = X2PSendListCreate(&sendListTable[hash],ctx->chunksize);CHKERRQ(ierr);
                assert(sendListTable[hash].data_size == ctx->chunksize);
#if defined(PETSC_USE_LOG)
		ierr = PetscLogEventEnd(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
              }
              /* add to list - pass this in as a function to a function? */
              ierr = X2PSendListAdd(&sendListTable[hash],&part);CHKERRQ(ierr);assert(part.gid>0);
              ierr = X2PListRemoveAt(list,pos);CHKERRQ(ierr);
              nmoved++;
              break;
            }
            if (++hash == ctx->tablesize) hash=0;
          }
          assert(ii!=ctx->tablesize);
        }
        if (solver) xx += 3;
      }
      if (solver) {
        ierr = VecRestoreArray(xVec,&xx0);
        /* done with these, need new ones after communication */
        ierr = VecDestroy(&xVec);CHKERRQ(ierr);
        ierr = VecDestroy(&jetVec);CHKERRQ(ierr);
      }
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventEnd(ctx->events[5],0,0,0,0);CHKERRQ(ierr);
#endif
      } /* particle lists */
    /* finish sends and receive new particles for this species */
    ierr = shiftParticles(ctx, sendListTable, isp, irk, &nslist, ctx->partlists[isp], slist, tag+isp, solver );CHKERRQ(ierr);
    if (0) { /* debug */
      PetscMPIInt flag,sz; MPI_Status  status; MPI_Datatype mtype;
      ierr = MPI_Iprobe(MPI_ANY_SOURCE, tag+isp, ctx->wComm, &flag, &status);CHKERRQ(ierr);
      if (flag) {
	PetscDataTypeToMPIDataType(PETSC_REAL,&mtype);
	MPI_Get_count(&status, mtype, &sz); assert(sz%part_dsize==0);
	SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"found %D extra particles from %d",sz/part_dsize,status.MPI_SOURCE);
      }
      MPI_Barrier(ctx->wComm);
    }

    nlistsTot += nslist;
    nslist = 0;
    /* add density (while in cache, by species at least) */
    if (irk>=0) {
      Vec locrho;

      assert(solver);
      ierr = DMGetLocalVector(dmpi->dmplex, &locrho);CHKERRQ(ierr);
      ierr = VecSet(locrho, 0.0);CHKERRQ(ierr);
      for (elid=0;elid<ctx->nElems;elid++) {
        X2PList *list = &ctx->partlists[isp][elid];
        if (X2PListSize(list)==0)continue;
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventBegin(ctx->events[7],0,0,0,0);CHKERRQ(ierr); /* timer on particle list */
#endif
        /* make vectors for this element */
        ierr = VecCreateSeq(PETSC_COMM_SELF,three*X2PListSize(list), &xVec);CHKERRQ(ierr);
        ierr = VecCreateSeq(PETSC_COMM_SELF,one*X2PListSize(list), &vVec);CHKERRQ(ierr);
        ierr = VecSetBlockSize(xVec,three);CHKERRQ(ierr);
        ierr = VecSetBlockSize(vVec,one);CHKERRQ(ierr);
        /* make coordinates array and density */
        ierr = VecGetArray(xVec,&xx0);CHKERRQ(ierr); xx = xx0;
        ierr = VecGetArray(vVec,&vv0);CHKERRQ(ierr); vv = vv0;
        ierr = X2PListGetHead( list, &pos );CHKERRQ(ierr);
        while ( !X2PListGetNext(list, &part, &pos) ) {
          /* for (pos=0 ; pos < list->size ; pos++, xx += 3) { */
          /*   X2Particle *part = &list->data[pos]; */
	  cylindricalToCart(part.r, part.z, part.phi, xx);
          xx += 3;
          *vv = part.w0;
          vv++;
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
    ierr = PetscLogEventBegin(ctx->events[9],0,0,0,0);CHKERRQ(ierr);
#endif
    /* count particles */
    for (isp=ctx->useElectrons ? 0 : 1, nloc = 0 ; isp <= X2_NION ; isp++) {
      for (elid=0;elid<ctx->nElems;elid++) {
	X2PList *list = &ctx->partlists[isp][elid];
	nloc += X2PListSize(list);
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
                istep+1,irk<0 ? "processed" : "pushed", origNlocal, rb1[0], rb1[3], 100.*(double)rb1[1]/(double)rb1[0], rb1[2], ctx->tablecount[1],(double)rb2[3]/((double)rb1[3]/(double)ctx->npe));
		if (rb1[0] != rb1[3]) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Number of partilces %D --> %D",rb1[0],rb1[3]);
#ifdef H5PART
    if (irk>=0) {
      if (ctx->plot) {
        for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++ ) {
          char  fname1[256],fname2[256];
          X2PListPos pos1,pos2;
          /* hdf5 output */
          sprintf(fname1,"particles_sp%d_time%05d.h5part",(int)isp,(int)istep+1);
          sprintf(fname2,"sub_rank_particles_sp%d_time%05d.h5part",(int)isp,(int)istep+1);
          /* write */
          prewrite(ctx, &ctx->partlists[isp][0], &pos1, &pos2);
          //ierr = X2PListWrite(ctx->partlists[isp], ctx->nElems, ctx->rank, ctx->npe, ctx->wComm, fname1, fname2);CHKERRQ(ierr);
          postwrite(ctx, &ctx->partlists[isp][0], &pos1, &pos2);
        }
      }
    }
#endif
#if defined(PETSC_USE_LOG)
    MPI_Barrier(ctx->wComm);
    ierr = PetscLogEventEnd(ctx->events[9],0,0,0,0);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

#define X2NDIG 100000
/* create particles in flux tubes, create particle lists, move particles to element lists */
#undef __FUNCT__
#define __FUNCT__ "createParticles"
static PetscErrorCode createParticles(X2Ctx *ctx)
{
  PetscErrorCode ierr;
  PetscInt isp,nCellsLoc,my0,irs,iths,gid,ii,np,dim,cStart,cEnd,elid;
  const PetscReal dth=(2.*M_PI)/(PetscReal)ctx->particleGrid.nptheta;
  const PetscReal dphi=2.*M_PI/(PetscReal)ctx->particleGrid.npphi,rmin=ctx->particleGrid.rMinor; /* rmin for particles < rmin */
  const PetscReal phi1 = (PetscReal)ctx->ParticlePlaneIdx*dphi + 1.e-8,rmaj=ctx->particleGrid.rMajor;
  const PetscInt  nPartProcss_plane = ctx->particleGrid.nptheta*ctx->particleGrid.npradius; /* nPartProcss_plane == ctx->npe_particlePlane */
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
  nCellsLoc = nPartProcss_plane/ctx->npe_particlePlane; /* = 1; nPartProcss_plane == ctx->npe_particlePlane */
  my0 = ctx->particlePlaneRank*nCellsLoc;              /* cell index in plane == particlePlaneRank */
  gid = (my0 + ctx->ParticlePlaneIdx*nPartProcss_plane)*ctx->npart_flux_tube; /* based particle ID */
  if (ctx->ParticlePlaneIdx == ctx->npe_particlePlane-1){
    nCellsLoc = nPartProcss_plane - nCellsLoc*(ctx->npe_particlePlane-1);
  }
  assert(nCellsLoc==1);

  /* my first cell index */
  srand(ctx->rank);
  for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++ ) {
    iths = my0%ctx->particleGrid.nptheta;
    irs = my0/ctx->particleGrid.nptheta;
    ierr = PetscMalloc1(ctx->nElems,&ctx->partlists[isp]);CHKERRQ(ierr);
    {
      const PetscReal r1 = sqrt(((PetscReal)irs      /(PetscReal)ctx->particleGrid.npradius)*rmin*rmin) +       1.e-12*rmin;
      const PetscReal dr = sqrt((((PetscReal)irs+1.0)/(PetscReal)ctx->particleGrid.npradius)*rmin*rmin) - (r1 - 1.e-12*rmin);
      const PetscReal th1 = (PetscReal)iths*dth + 1.e-12*dth;
      /* create list for element 0 and add all to it */
      ierr = X2PListCreate(&ctx->partlists[isp][0],ctx->chunksize);CHKERRQ(ierr);
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
	  polPlaneToCylindrical(psi, thetap, r, z);
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
	  ierr = X2PListAdd(&ctx->partlists[isp][0],&particle, NULL);CHKERRQ(ierr);
          /* debug, particles are created in a flux tube */
          if (0) {
            PetscMPIInt pe; PetscInt id;
            ierr = X2GridParticleGetProc_FluxTube(&ctx->particleGrid,psi,thetap,phi,&pe,&id);CHKERRQ(ierr);
            if(pe != ctx->rank){
              PetscPrintf(PETSC_COMM_SELF,"[%D] ERROR particle in proc %d r=%e:%e:%e theta=%e:%e:%e phi=%e:%e:%e\n",ctx->rank,pe,r1,psi,r1+dr,th1,thetap,th1+dth,phi1,phi,phi1+dphi);
              SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB," created particle for proc %D",pe);
            }
          }
	} /* theta */
      }
      iths++;
      if (iths==ctx->particleGrid.nptheta) { iths = 0; irs++; }
    } /* cells */
    /* finish off list creates for rest of elements */
    for (elid=1;elid<ctx->nElems;elid++) {
      ierr = X2PListCreate(&ctx->partlists[isp][elid],ctx->chunksize);CHKERRQ(ierr); /* this will get expanded, chunksize used for message chunk size and initial list size! */
    }
  } /* species */
  /* move back to solver space and make density vector */
  {
    PetscMPIInt tag = 99, istep=-1, idx;
    X2PSendList *sendListTable;
    /* init send tables */
    ierr = PetscMalloc1(ctx->tablesize,&sendListTable);CHKERRQ(ierr);
    for (idx=0;idx<ctx->tablesize;idx++) {
      for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++) {
        sendListTable[idx].data_size = 0; /* init */
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
  PetscInt       istep;
  PetscMPIInt    tag;
  int            irk,idx,isp;
  PetscReal      time,dt;
  X2PSendList    *sendListTable;
  DM_PICell      *dmpi = (DM_PICell *) ctx->dm->data;
  PetscFunctionBeginUser;

  /* init send tables */
  ierr = PetscMalloc1(ctx->tablesize,&sendListTable);CHKERRQ(ierr);
  for (idx=0;idx<ctx->tablesize;idx++) {
    for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++) {
      sendListTable[idx].data_size = 0; /* init */
    }
  }

  /* hdf5 output - init */
#ifdef H5PART
  if (ctx->plot) {
    for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++) { // for each species
      char  fname1[256],fname2[256];
      X2PListPos pos1,pos2;
      sprintf(fname1,"particles_sp%d_time%05d.h5part",(int)isp,0);
      sprintf(fname2,"sub_rank_particles_sp%d_time%05d.h5part",(int)isp,0);
      /* write */
      prewrite(ctx, &ctx->partlists[isp][0], &pos1, &pos2);
      // ierr = X2PListWrite(ctx->partlists[isp], ctx->nElems, ctx->rank, ctx->npe, ctx->wComm, fname1, fname2);CHKERRQ(ierr);
      postwrite(ctx, &ctx->partlists[isp][0], &pos1, &pos2);
    }
  }
#endif
  /* main time step loop */
  ierr = PetscCommGetNewTag(ctx->wComm,&tag);CHKERRQ(ierr);
  for ( istep=0, time=0.;
	istep < ctx->msteps && time < ctx->maxTime;
	istep++, time += ctx->dt, tag += 3*(X2_NION + 1) ) {

    /* do collisions */
    if (((istep+1)%ctx->collisionPeriod)==0) {
      /* move to flux tube space */
      ierr = processParticles(ctx, 0.0, sendListTable, tag, -1, istep, PETSC_FALSE);
      CHKERRQ(ierr);
      /* call collision method */
#ifdef H5PART
      if (ctx->plot) {
        for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++) { // for each species
          char fname1[256], fname2[256];
          X2PListPos pos1,pos2;
          sprintf(fname1,         "particles_sp%d_time%05d_fluxtube.h5part",(int)isp,(int)istep);
          sprintf(fname2,"sub_rank_particles_sp%d_time%05d_fluxtube.h5part",(int)isp,(int)istep);
          /* write */
          prewrite(ctx, &ctx->partlists[isp][0], &pos1, &pos2);
          ierr = X2PListWrite(ctx->partlists[isp], ctx->nElems, ctx->rank, ctx->npe, ctx->wComm, fname1, fname2);CHKERRQ(ierr);
          postwrite(ctx, &ctx->partlists[isp][0], &pos1, &pos2);
        }
      }
#endif
      /* move back to solver space */
      ierr = processParticles(ctx, 0.0, sendListTable, tag + X2_NION + 1, -1, istep, PETSC_TRUE);
      CHKERRQ(ierr);
    }

    /* very crude explicit RK */
    dt = ctx->dt;

    /* solve for potential, density being assembled is an invariant */
    ierr = DMPICellSolve( ctx->dm );CHKERRQ(ierr);

    if (dmpi->debug>1) {
      PetscViewer       viewer = NULL;
      PetscBool         flg;
      PetscViewerFormat fmt;

      ierr = DMViewFromOptions(dmpi->dmgrid,NULL,"-dm_view");CHKERRQ(ierr);
      ierr = PetscOptionsGetViewer(ctx->wComm,NULL,"-x2_vec_view",&viewer,&fmt,&flg);CHKERRQ(ierr);
      if (flg) {
        ierr = PetscViewerPushFormat(viewer,fmt);CHKERRQ(ierr);
        ierr = VecView(dmpi->phi,viewer);CHKERRQ(ierr);
        ierr = VecView(dmpi->rho,viewer);CHKERRQ(ierr);
        ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }

    /* process particles: push, move */
    irk=0;
    ierr = processParticles(ctx, dt, sendListTable, tag + 2*(X2_NION + 1), irk, istep, PETSC_TRUE);
    CHKERRQ(ierr);
  } /* time step */

  /* clean up */
  for (idx=0;idx<ctx->tablesize;idx++) {
    for (isp=ctx->useElectrons ? 0 : 1 ; isp <= X2_NION ; isp++) {
      if (sendListTable[idx].data_size != 0) {
	ierr = X2PSendListDestroy( &sendListTable[idx] );CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFree(sendListTable);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* == Defining a base plex for ITER, which looks like a rectilinear (sort of) donut */

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreatePICellITER"
static PetscErrorCode DMPlexCreatePICellITER(MPI_Comm comm, X2GridParticle *params, DM *dm)
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
static PetscErrorCode DMPlexCreatePICellBoxTorus(MPI_Comm comm, X2GridParticle *params, DM *dm)
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
static PetscErrorCode DMPlexCreatePICellTorus(MPI_Comm comm, X2GridParticle *params, DM *dm)
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
  X2GridParticle *grid = &ctx->particleGrid;
  PetscReal rMajor    = grid->rMajor;
  PetscReal rMinor    = grid->rMinor;
  PetscReal innerMult = grid->innerMult;
  PetscInt  numMajor  = grid->numMajor;
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
  if (ctx->inflate_torus) {
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

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = 1.0;
}
void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  f0[0] = 4.;
}
/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  X2Ctx          ctx; /* user-defined work context */
  PetscErrorCode ierr;
  DM_PICell      *dmpi;
  PetscInt       dim;
  Mat            J;
  DMLabel        label;
  PetscDS        prob;
  PetscSection   s;
  PetscFunctionBeginUser;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ctx.currevent = 0;
  ierr = PetscLogEventRegister("+CreateMesh", DM_CLASSID, &ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 0 */
  ierr = PetscLogEventRegister("+Process parts",0,&ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 1 */
  ierr = PetscLogEventRegister(" -shiftParticles",0,&ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 2 */
  ierr = PetscLogEventRegister("   *N.B. consensus",0,&ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 3 */
  ierr = PetscLogEventRegister("     +Part. Send", 0, &ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 4 */
  ierr = PetscLogEventRegister(" -Move parts", 0, &ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 5 */
  ierr = PetscLogEventRegister(" -AddSource", 0, &ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 6 */
  ierr = PetscLogEventRegister(" -Pre Push", 0, &ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 7 */
  ierr = PetscLogEventRegister(" -Push (Jet)", 0, &ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 8 */
  ierr = PetscLogEventRegister("+Diagnostics", 0, &ctx.events[ctx.currevent++]);CHKERRQ(ierr); /* 9 */
  assert(sizeof(ctx.events)/sizeof(ctx.events[0]) >= ctx.currevent);
#endif

  ierr = PetscCommDuplicate(PETSC_COMM_WORLD,&ctx.wComm,NULL);CHKERRQ(ierr);
  ierr = ProcessOptions( &ctx );CHKERRQ(ierr);

  /* construct DMs */
  ierr = PetscLogEventBegin(ctx.events[0],0,0,0,0);CHKERRQ(ierr);
  ierr = DMCreate(ctx.wComm, &ctx.dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(ctx.dm, &ctx);CHKERRQ(ierr);
  ierr = DMSetType(ctx.dm, DMPICELL);CHKERRQ(ierr); /* creates (DM_PICell *) dm->data */
  dmpi = (DM_PICell *) ctx.dm->data; assert(dmpi);
  dmpi->debug = s_debug;
  /* setup solver grid */
  if (ctx.run_type == X2_ITER) {
    ierr = DMPlexCreatePICellITER(ctx.wComm,&ctx.particleGrid,&dmpi->dmplex);CHKERRQ(ierr);
  }
  else if (ctx.run_type == X2_TORUS) {
    ierr = DMPlexCreatePICellTorus(ctx.wComm,&ctx.particleGrid,&dmpi->dmplex);CHKERRQ(ierr);
    ctx.inflate_torus = PETSC_TRUE;
  }
  else {
    ierr = DMPlexCreatePICellBoxTorus(ctx.wComm,&ctx.particleGrid,&dmpi->dmplex);CHKERRQ(ierr);
    ctx.inflate_torus = PETSC_FALSE;
    assert(ctx.run_type == X2_BOXTORUS);
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
    char      convType[256];
    PetscBool flg;
    ierr = PetscOptionsBegin(ctx.wComm, "", "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-x2_dm_type","Convert DMPlex to another format (should not be Plex!)","x2.c",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
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
      dmpi->dmgrid = dmpi->dmplex;
    }
  }
  if (sizeof(long long)!=sizeof(PetscReal)) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "sizeof(long long)!=sizeof(PetscReal)");
  /* setup DM */
  ierr = DMSetFromOptions( ctx.dm );CHKERRQ(ierr); /* refinement done here */
  if (dmpi->dmgrid == dmpi->dmplex && ctx.npe > 1) {
    /* plex does not distribute by implicitly, so do it */
    if (dmpi->debug>0) PetscPrintf(ctx.wComm,"[%D] No p4est\n",ctx.rank);
    ierr = DMPlexDistribute(dmpi->dmplex, 0, NULL, &dmpi->dmgrid);CHKERRQ(ierr);
    ierr = DMDestroy(&dmpi->dmplex);CHKERRQ(ierr);
    dmpi->dmplex = dmpi->dmgrid;
  }
  /* setup Discretization */
  ierr = DMGetDimension(dmpi->dmgrid, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dmpi->dmgrid, dim, 1, PETSC_FALSE, NULL, 1, &dmpi->fem);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dmpi->fem, "poisson");CHKERRQ(ierr);
  /* FEM prob */
  ierr = DMGetDS(dmpi->dmgrid, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) dmpi->fem);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 0, f0_u, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
  ierr = DMSetUp( ctx.dm );CHKERRQ(ierr);
  if (dmpi->dmgrid == dmpi->dmplex) {
    ierr = DMGetDefaultSection(dmpi->dmplex, &s);CHKERRQ(ierr);
    ierr = DMGetDefaultGlobalSection(dmpi->dmgrid, &s);CHKERRQ(ierr);
    if (!s) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "DMGetDefaultSection return NULL");
  }
  else { /* convert to plex - using the origina plex has a problem */
    ierr = DMDestroy(&dmpi->dmplex);CHKERRQ(ierr);
    ierr = DMConvert(dmpi->dmgrid,DMPLEX,&dmpi->dmplex);CHKERRQ(ierr); /* low overhead, cached */
    /* get section */
    ierr = DMGetDefaultGlobalSection(dmpi->dmgrid, &s);CHKERRQ(ierr);
  }

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
    ierr = DMPlexGetHeightStratum(dmpi->dmplex, 0, &cStart, &cEnd);CHKERRQ(ierr);
    if (dmpi->debug>0 && !cEnd) {
      ierr = PetscPrintf((dmpi->debug>1 || !cEnd) ? PETSC_COMM_SELF : ctx.wComm,"[%D] ERROR %D global equations, %d local cells, (cEnd=%d), debug=%D\n",ctx.rank,n,cEnd-cStart,cEnd,dmpi->debug);
    }
    if (!cEnd) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "No cells");
    }
  }

  /* create SNESS */
  ierr = SNESCreate( ctx.wComm, &dmpi->snes);CHKERRQ(ierr);
  ierr = SNESSetDM( dmpi->snes, dmpi->dmgrid);CHKERRQ(ierr);
  ierr = DMSetMatType(dmpi->dmgrid,MATAIJ);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(dmpi->snes);CHKERRQ(ierr);
  ierr = DMSNESSetFunctionLocal(dmpi->dmgrid,  (PetscErrorCode (*)(DM,Vec,Vec,void*))DMPlexSNESComputeResidualFEM,&ctx);CHKERRQ(ierr);
  ierr = DMSNESSetJacobianLocal(dmpi->dmgrid,  (PetscErrorCode (*)(DM,Vec,Mat,Mat,void*))DMPlexSNESComputeJacobianFEM,&ctx);CHKERRQ(ierr);
  ierr = SNESSetUp( dmpi->snes );CHKERRQ(ierr);
  ierr = DMCreateMatrix(dmpi->dmgrid, &J);CHKERRQ(ierr);
  ierr = SNESSetJacobian(dmpi->snes, J, J, NULL, NULL);CHKERRQ(ierr);
  /* setup particles */
  ierr = createParticles( &ctx );CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ctx.events[0],0,0,0,0);CHKERRQ(ierr);

  /* do it */
  ierr = go( &ctx );CHKERRQ(ierr);

  if (dmpi->debug>3) {
    ierr = MatView(J,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  if (dmpi->debug>0) PetscPrintf(ctx.wComm,"[%D] done - cleanup\n",ctx.rank);
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

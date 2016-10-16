/* M. Adams, Oct 2016 */

static char help[] = "X3: MHD with partical in cell for tokamak plasmas using PICell.\n";

#ifdef H5PART
#include <H5Part.h>
#endif
#include <petsc/private/dmpicellimpl.h>    /*I   "petscdmpicell.h"   I*/
#include <assert.h>
#include <petscts.h>
#include <petscdmforest.h>

#define ALEN(a) (sizeof(a)/sizeof((a)[0]))

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
#if defined(PETSC_USE_LOG)
PetscLogEvent s_events[22];
static const int diag_event_id = sizeof(s_events)/sizeof(s_events[0])-1;
#endif
#define X3_V_LEN 1
#define X3_S_OF_V
typedef struct { /* array of struct */
  /* phase (4D) */
  PetscReal *x[0]; /* array of coordinate arrays */
  PetscReal *r;    /* r from center */
  PetscReal *z;    /* vertical coordinate */
  PetscReal *phi;  /* toroidal coordinate */
  PetscReal *vpar; /* toroidal velocity */
  PetscReal *mu;   /* 5th D */
  PetscReal *w0;
  PetscReal *f0;
  long long *gid;
} X3Particle_v;
/* X3Particle */
typedef struct { /* struct */
  /* phase (4D) */
  PetscReal x[0]; /* array of coordinates */
  PetscReal r;   /* r from center */
  PetscReal z;   /* vertical coordinate */
  PetscReal phi; /* toroidal coordinate */
  PetscReal vpar; /* toroidal velocity */
  PetscReal mu; /* 5th D */
  PetscReal w0;
  PetscReal f0;
  long long gid;
} X3Particle;

/* X3PList */
typedef PetscInt X3PListPos;
typedef struct {
#ifdef X3_S_OF_V
  X3Particle_v data_v;
#else
  X3Particle *data;
#endif
  PetscInt    data_size;
  PetscInt    size;
  PetscInt    hole;
  PetscInt    top;
  PetscInt    vec_top;
} X3PList;

/* particle */
#undef __FUNCT__
#define __FUNCT__ "X3ParticleCreate"
PetscErrorCode X3ParticleCreate(X3Particle *p, PetscInt gid, PetscReal r, PetscReal z, PetscReal phi, PetscReal vpar)
{
  if (gid <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"gid <= 0");
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
PetscErrorCode X3ParticleCopy(X3Particle *p, X3Particle p2)
{
  if (p2.gid <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"X3ParticleCopy: gid <= 0");
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
PetscErrorCode X3ParticleRead(X3Particle *p, void *buf)
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
PetscErrorCode X3ParticleWrite(X3Particle *p, void *buf)
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
#define X3FREEV(d) {                                           \
    ierr = PetscFree2( d.gid,  d.w0 );CHKERRQ(ierr);           \
    ierr = PetscFree6( d.r,    d.z,  d.phi,                    \
                       d.vpar, d.mu, d.f0  );CHKERRQ(ierr);    \
  }
#define X3ALLOCV(s,d) {                                           \
    ierr = PetscMalloc2( s,&d.gid, s,&d.w0);CHKERRQ(ierr);      \
    ierr = PetscMalloc6( s,&d.r,   s,&d.z, s,&d.phi,                    \
                         s,&d.vpar,s,&d.mu,s,&d.f0  );CHKERRQ(ierr);    \
  }
#define X3P2V(p,d,i)         { d.r[i] = p->r;         d.z[i] = p->z;         d.phi[i] = p->phi;         d.vpar[i] = p->vpar;         d.mu[i] = p->mu;         d.w0[i] = p->w0;         d.f0[i] = p->f0;         d.gid[i] = p->gid;}
#define X3V2V(src,dst,is,id) { dst.r[id] = src.r[is]; dst.z[id] = src.z[is]; dst.phi[id] = src.phi[is]; dst.vpar[id] = src.vpar[is]; dst.mu[id] = src.mu[is]; dst.w0[id] = src.w0[is]; dst.f0[id] = src.f0[is]; dst.gid[id] = src.gid[is];}
#define X3V2P(p,d,i)         { p->r = d.r[i];         p->z = d.z[i];         p->phi = d.phi[i];         p->vpar = d.vpar[i];         p->mu = d.mu[i];         p->w0 = d.w0[i];         p->f0 = d.f0[i];         p->gid = d.gid[i];}
#ifdef X3_S_OF_V
#define X3PAXPY(a,s,d,i) { d.r += a*s->data_v.r[i]; d.z += a*s->data_v.z[i]; d.phi += a*s->data_v.phi[i]; d.vpar += a*s->data_v.vpar[i]; d.mu += a*s->data_v.mu[i]; d.w0 += a*s->data_v.w0[i]; d.f0 += a*s->data_v.f0[i]; /* d.gid += s->data_v.gid[i]; */}
#else
#define X3PAXPY(a,s,d,i) { d.r += a*s->data[i].r;   d.z += a*s->data[i].z;   d.phi += a*s->data[i].phi;   d.vpar += a*s->data[i].vpar;   d.mu += a*s->data[i].mu;   d.w0 += a*s->data[i].w0 ;  d.f0 += a*s->data[i].f0;   /* d.gid += a*s->data[i].gid; */}
#endif

/* particle list */
#undef __FUNCT__
#define __FUNCT__ "X3PListCreate"
PetscErrorCode X3PListCreate(X3PList *l, PetscInt msz)
{
  PetscErrorCode ierr;
  if (msz <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"msz <= 0");
  l->size=0;
  l->top=0;
  l->vec_top=0;
  l->hole=-1;
  l->data_size = X3_V_LEN*(msz/X3_V_LEN + 1);
#ifdef X3_S_OF_V
  X3ALLOCV(l->data_size,l->data_v);
#else
  ierr = PetscMalloc1(l->data_size, &l->data);CHKERRQ(ierr);
#endif
  return ierr;
}
PetscErrorCode X3PListClear(X3PList *l)
{
  l->size=0; /* keep memory but kill data */
  l->top=0;
  l->hole=-1;
  return 0;
}
#undef __FUNCT__
#define __FUNCT__ "X3PListDestroy"
PetscErrorCode X3PListDestroy(X3PList *l)
{
  PetscErrorCode ierr;
#ifdef X3_S_OF_V
  X3FREEV(l->data_v);
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
#define __FUNCT__ "X3PListAdd"
PETSC_STATIC_INLINE PetscErrorCode X3PListAdd( X3PList *l, X3Particle *p, X3PListPos *ppos)
{
  PetscFunctionBeginUser;
  if (!l->data_size) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"List not created?");
  if (l->size==l->data_size) {
#ifdef X3_S_OF_V
    X3Particle_v data2;
#else
    X3Particle *data2;
#endif
    int i;
    PetscErrorCode ierr;
    l->data_size *= 2;
#ifdef X3_S_OF_V
    X3ALLOCV(l->data_size,data2);
    for (i=0;i<l->size;i++) {
      X3V2V(l->data_v,data2,i,i);
    }
    X3FREEV(l->data_v);
    l->data_v = data2;
#else
    ierr = PetscMalloc1(l->data_size, &data2);CHKERRQ(ierr);
    for (i=0;i<l->size;i++) data2[i] = l->data[i];
    ierr = PetscFree(l->data);CHKERRQ(ierr);
    l->data = data2;
#endif
    if (l->hole != -1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"rehash and l->hole != -1, size %d",l->size);
    assert(l->hole == -1);
  }
  if (l->hole != -1) { /* have a hole - fill it */
    X3PListPos i = l->hole; assert(i<l->data_size);
#ifdef X3_S_OF_V
    if (l->data_v.gid[i] == 0) l->hole = -1; /* filled last hole */
    else if (l->data_v.gid[i]>=0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"hole with non-neg gid %d",l->data_v.gid[i]);
    else l->hole = (X3PListPos)(-l->data_v.gid[i] - 1); /* use gid as pointer */
    X3P2V(p,l->data_v,i);
#else
    if (l->data[i].gid == 0) l->hole = -1; /* filled last hole */
    else if (l->data[i].gid>=0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"hole with non-neg gid %d",l->data[i].gid);
    else l->hole = (X3PListPos)(-l->data[i].gid - 1); /* use gid as pointer */
    l->data[i] = *p;
#endif
    if (ppos) *ppos = i;
  }
  else {
    X3PListPos i = l->top++;
#ifdef X3_S_OF_V
    X3P2V(p,l->data_v,i);
#else
    l->data[i] = *p;
#endif
    if (ppos) *ppos = i;
  }
  if (l->size == l->vec_top) { l->vec_top += X3_V_LEN; assert(l->vec_top<=l->data_size); }
  l->size++;
  assert(l->top >= l->size);
  PetscFunctionReturn(0);
}
PetscErrorCode X3PListSetAt(X3PList *l, X3PListPos pos, X3Particle *part)
{
#ifdef X3_S_OF_V
  X3P2V(part,l->data_v,pos);
#else
  l->data[pos] = *part;
#endif
  return 0;
}

PetscErrorCode X3PListCompress(X3PList *l)
{
  PetscInt ii;
  PetscErrorCode ierr;
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(s_events[13],0,0,0,0);CHKERRQ(ierr);
#endif
  /* fill holes with end of list */
  for ( ii = 0 ; ii < l->top && l->top > l->size ; ii++) {
#ifdef X3_S_OF_V
    if (l->data_v.gid[ii] <= 0)
#else
    if (l->data[ii].gid <= 0)
#endif
    {
      l->top--; /* index of data at end to move to hole */
      if (ii == l->top) /* just pop hole at end */ ;
      else {
#ifdef X3_S_OF_V
        while (l->data_v.gid[l->top] <= 0)  l->top--; /* get real data */
        if (l->top > ii) X3V2V(l->data_v,l->data_v,l->top,ii);
#else
        while (l->data[l->top].gid <= 0)  l->top--; /* get real data */
        if (l->top > ii) l->data[ii] = l->data[l->top]; /* now above */
#endif
      }
    }
  }
  l->hole = -1;
  l->top = l->size;
  /* pad end for vectorization */
  if (l->top%X3_V_LEN==0) l->vec_top=l->top;
  else {
    PetscInt vtop = (l->top/X3_V_LEN + 1)*X3_V_LEN;
    l->vec_top = vtop;
    for ( ii = l->top ; ii < vtop ; ii++) {
#ifdef X3_S_OF_V
      X3V2V(l->data_v,l->data_v,l->top-1,ii); /* use any valid coordinate */
      l->data_v.w0[ii] = 0; /* zero weight so it does nothing in deposition, etc */
#else
      l->data[ii] = l->data[l->top-1];
      l->data[ii].w0 = 0;
#endif
    }
  }
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(s_events[13],0,0,0,0);CHKERRQ(ierr);
#endif
#ifdef PETSC_USE_DEBUG
#ifdef X3_S_OF_V
  for (ii=0;ii<l->size;ii++) assert(l->data_v.gid[ii]>0);
#else
  for (ii=0;ii<l->size;ii++) assert(l->data[ii].gid>0);
#endif
#endif
  return 0;
}

/* keep list of hols after removal so that we can iterate over a list and remove as we go
  gid < 0 : hole : -gid - 1 == -(gid+1) is next index
  gid = 0 : sentinal
  gid > 0 : real
*/
#undef __FUNCT__
#define __FUNCT__ "X3PListGetHead"
PetscErrorCode X3PListGetHead(X3PList *l, X3Particle *p, X3PListPos *pos)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  PetscValidPointer(l, 1);
  PetscValidPointer(p, 2);
  PetscValidPointer(pos, 3);
  if (l->size==0) {
    ierr = 1; /* empty list */
  }
  else {
    *pos = 0; /* go past any holes */
#ifdef X3_S_OF_V
    while (l->data_v.gid[*pos] <= 0) (*pos)++;
    X3V2P(p,l->data_v,*pos); /* return copy */
#else
    while (l->data[*pos].gid <= 0) (*pos)++;
    *p = l->data[*pos]; /* return copy */
#endif
    ierr = 0;
    assert(*pos<l->data_size);
  }
  PetscFunctionReturn(ierr);
}

/* increment pos and get val, return if at end of list */
#undef __FUNCT__
#define __FUNCT__ "X3PListGetNext"
PetscErrorCode X3PListGetNext(X3PList *l, X3Particle *p, X3PListPos *pos)
{
  PetscFunctionBeginUser;
  /* l->size == 0 can happen on empty list */
  (*pos)++; /* get next position */
  if (*pos >= l->data_size || *pos >= l->top) PetscFunctionReturn(1); /* hit end, can go past if list is just drained */
#ifdef X3_S_OF_V
  while(*pos < l->top && l->data_v.gid[*pos] <= 0) (*pos)++; /* skip holes */
#else
  while(*pos < l->top && l->data[*pos].gid <= 0) (*pos)++; /* skip holes */
#endif
  assert(*pos<=l->top); assert(l->top<=l->data_size);
  if (*pos==l->top) PetscFunctionReturn(1); /* hit end with holes */
#ifdef X3_S_OF_V
  X3V2P(p,l->data_v,*pos); /* return copy */
#else
  *p = l->data[*pos]; /* return copy */
#endif
  assert(p->gid>0);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "X3PListRemoveAt"
PETSC_STATIC_INLINE PetscErrorCode X3PListRemoveAt( X3PList *l, const X3PListPos pos)
{
  PetscFunctionBeginUser;
#ifdef PETSC_USE_DEBUG
  if(pos >= l->data_size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"X3PListRemoveAt past end of data %d %d",pos,l->data_size);
  if(pos >= l->top) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"X3PListRemoveAt past end of top pointer %d %d",pos,l->top);
#endif
  if (l->hole==-1 && pos==l->top-1 && 0) l->top--; /* just pop */
  else {
#ifdef X3_S_OF_V
    if (l->hole==-1) l->data_v.gid[pos] = 0; /* sentinel */
    else l->data_v.gid[pos] = -(l->hole + 1); /* hole >= 0 */
#else
    if (l->hole==-1) l->data[pos].gid = 0; /* sentinel */
    else l->data[pos].gid = -(l->hole + 1); /* hole >= 0 */
#endif
  }
  l->size--;
  if (!l->size) { /* lets reset if we drained the list */
    l->hole = -1;
    l->top = 0;
  }
  else l->hole = pos; /* head of linked list of holes */
#ifdef X3_S_OF_V
  l->data_v.w0[pos] = 0; /* zero out so we can vectorize deposition with holes */
#else
  l->data[pos].w0 = 0;
#endif
  PetscFunctionReturn(0);
}
PetscInt X3PListMaxSize(X3PList *l) {
  return l->data_size;
}
PetscInt X3PListSize(X3PList *l) {
  return l->size;
}
PetscBool X3PListIsEmpty(X3PList *l) {
  return (PetscBool)(l->size==0);
}
#ifdef H5PART
#undef __FUNCT__
#define __FUNCT__ "X3PListWrite"
PetscErrorCode X3PListWrite(X3PList l[], PetscInt nLists, PetscMPIInt rank, PetscMPIInt npe, MPI_Comm comm, char fname1[], char fname2[])
{
  double         *x=0,*y=0,*z=0,*v=0;
  h5part_int64_t *id=0,nparticles;
  X3PListPos     pos;
  X3Particle     part;
  PetscErrorCode ierr;
  H5PartFile     *file1,*file2;
  PetscInt       elid;
  PetscFunctionBeginUser;
  for (nparticles=0,elid=0;elid<nLists;elid++) {
    nparticles += X3PListSize(&l[elid]);
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
      ierr = X3PListGetHead( &l[elid], &part, &pos );
      if (!ierr) {
        do {
          if (part.gid > 0) {
            PetscReal xx[3];
            cylindricalToCart(part.r, part.z, part.phi, xx);
            x[nparticles] = xx[0];
            y[nparticles] = xx[1];
            z[nparticles] = xx[2];
            v[nparticles] = part.vpar;
            id[nparticles] = part.gid;
            nparticles++;
          }
        } while ( !X3PListGetNext( &l[elid], &part, &pos) );
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
        ierr = X3PListGetHead( &l[elid], &part, &pos );
        if (!ierr) {
          do {
            if (part.gid > 0) {
              PetscReal xx[3];
              cylindricalToCart(part.r, part.z, part.phi, xx);
              x[nparticles] = xx[0];
              y[nparticles] = xx[1];
              z[nparticles] = xx[2];
              v[nparticles] = part.vpar;
              id[nparticles] = rank;
              nparticles++;
            }
          } while ( !X3PListGetNext( &l[elid], &part, &pos) );
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

/* send particle list */
typedef struct X3PSendList_TAG{
  X3Particle *data;
  PetscInt    data_size, size;
  PetscMPIInt proc;
} X3PSendList;
/* MPI Isend particle list */
typedef struct X3ISend_TAG{
  X3Particle *data;
  PetscMPIInt proc;
  MPI_Request request;
} X3ISend;
/* particle send list, non-vector simple array list */
PetscInt X3PSendListSize(X3PSendList *l) {
  return l->size;
}
PetscInt X3PSendListMaxSize(X3PSendList *l) {
  return l->data_size;
}
#undef __FUNCT__
#define __FUNCT__ "X3PSendListCreate"
PetscErrorCode X3PSendListCreate(X3PSendList *l, PetscInt msz)
{
  PetscErrorCode ierr;
  l->size=0;
  l->data_size = msz;
  ierr = PetscMalloc1(l->data_size, &l->data);CHKERRQ(ierr);
  return ierr;
}
PetscErrorCode X3PSendListClear(X3PSendList *l)
{
  l->size=0; /* keep memory but kill data */
  return 0;
}
#undef __FUNCT__
#define __FUNCT__ "X3PSendListDestroy"
PetscErrorCode X3PSendListDestroy(X3PSendList *l)
{
  PetscErrorCode ierr;
  ierr = PetscFree(l->data);CHKERRQ(ierr);
  l->data = 0;
  l->size = 0;
  l->data_size = 0;
  return ierr;
}
#undef __FUNCT__
#define __FUNCT__ "X3PSendListAdd"
PetscErrorCode X3PSendListAdd( X3PSendList *l, X3Particle *p)
{
  PetscFunctionBeginUser;
  if (l->size==l->data_size) {
    X3Particle *data2; /* make this arrays of X3Particle members for struct-of-arrays */
    int i;PetscErrorCode ierr;
    l->data_size *= 2;
    ierr = PetscMalloc1(l->data_size, &data2);CHKERRQ(ierr);
    for (i=0;i<l->size;i++) data2[i] = l->data[i];
    ierr = PetscFree(l->data);CHKERRQ(ierr);
    l->data = data2;
  }
  l->data[l->size++] = *p;
  PetscFunctionReturn(0);
}
#define X3_NION 1
typedef struct {
  PetscReal mass;
  PetscReal charge;
} X3Species;
/* static const PetscReal x3ECharge=1.6022e-19;  /\* electron charge (MKS) *\/ */
/* static const PetscReal x3Epsilon0=8.8542e-12; /\* permittivity of free space (MKS) *\/ */
/* static const PetscReal x3ProtMass=1.6720e-27; /\* proton mass (MKS) *\/ */
/* static const PetscReal x3ElecMass=9.1094e-31; /\* electron mass (MKS) *\/ */

static const PetscReal x3ECharge=1.;  /* electron charge */
static const PetscReal x3ProtMass=1.; /* proton mass */
static const PetscReal x3ElecMass=0.01; /* electron mass */

typedef enum {X3_TORUS,X3_BOXTORUS} runType;
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
} X3Grid;
/*
  General parameters and context
*/
typedef enum {X3_PERIODIC,X3_DIRI} domainType;
typedef struct {
  PetscLogEvent *events;
  PetscInt      use_bsp;
  PetscInt      chunksize;
  runType       run_type;
  PetscBool     plot;
  PetscBool     plot_amr_initial;
  PetscBool     use_amr;
  PetscReal     refine_tol;
  PetscReal     coarsen_tol;
  /* MPI parallel data */
  MPI_Comm      particlePlaneComm,wComm;
  PetscMPIInt   rank,npe,npe_particlePlane,particlePlaneRank,ParticlePlaneIdx;
  /* grids & solver */
  DM            dmpic;
  X3Grid        grid;
  PetscBool     inflate_torus;
  /* time */
  PetscInt      msteps;
  PetscReal     maxTime;
  PetscReal     dt;
  /* physics */
  PetscReal     massAu; /* =2D0  !mass ratio to proton */
  PetscReal     chargeEu; /* =1D0  ! charge number */
  PetscReal     eChargeEu; /* =-1D0 */
  /* MHD physics */
  PetscReal     rhor;
  PetscReal     amach;
  PetscReal     gamma;
  PetscReal     cfl;
  PetscReal     maxspeed;
  PetscInt      ndof;
  PetscInt      nfields;
  /* particles */
  PetscInt      num_particles_proc;
  PetscInt      num_particles_total;
  PetscBool     use_electrons;
  PetscReal     max_vpar;
  PetscInt      nElems; /* size of array of particle lists */
  X3PList       *partlists[X3_NION+1]; /* 0: electron, 1:N ions */
  X3Species     species[X3_NION+1]; /* 0: electron, 1:N ions */
  /* hash table meta-data for proc-send list table - should just be an object */
  PetscInt      proc_send_table_size, tablecount;
  X3PSendList   *sendListTable;
  /* prob type  */
  domainType    dtype;
  PetscBool     use_mms;
  PetscBool     use_vel_update;
} X3Ctx;
/* static vars for lack of context in all callbacks */
static PetscInt  s_debug;
static PetscInt  s_rank;
static PetscInt  s_fluxtubeelem; /* could just hardwire to 0 */
static PetscReal s_rminor_inflate;
static PetscReal s_section_phi;

/* X3GridSolverLocatePoints: find processor and element in solver grid that this point is in
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
#define __FUNCT__ "X3GridSolverLocatePoints"
PetscErrorCode X3GridSolverLocatePoints(DM dm, Vec xvec, IS *pes, IS *elemIDs)
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
    PetscSF           cellSF = NULL;
    const PetscSFNode *foundCells;
    PetscInt          dim,n,nn,ii,jj;
    PetscScalar       *xx,*xx0;
    PetscInt          *peidxs,*elemidxs;
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
      if (elemidxs[ii] < 0 && npe==1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "We are not supporting out of domain points.");
      if (elemidxs[ii] < 0) elemidxs[ii] = 0; /* not working in parallel */
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

/* FE point function */
PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  int i;
  for (i = 0 ; i < Nf ; i++) u[i] = 0.;
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "destroyParticles"
static PetscErrorCode destroyParticles(X3Ctx *ctx)
{
  PetscErrorCode ierr;
  PetscInt       isp,elid;
  PetscFunctionBeginUser;
  /* idiom for iterating over particle lists */
  for (isp = ctx->use_electrons ? 0 : 1 ; isp <= X3_NION ; isp++ ) { // for each species
    for (elid=0;elid<ctx->nElems;elid++) {
      ierr = X3PListDestroy(&ctx->partlists[isp][elid]);CHKERRQ(ierr);
    }
    ierr = PetscFree(ctx->partlists[isp]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* shiftParticles: send particles
    Input:
     - ctx: global data
     - tag: MPI tag to send with
     - solver: use solver partitioning to get processor of point?
   Input/Output:
     - nIsend: number of sends so far
     - sendListTable: send list hash table array, emptied but meta-data kept
     - particlelist: array of the lists of particle lists to add to
     - slists: array of non-blocking send caches (not used when ctx->use_bsp), cleared
   Output:
*/
#undef __FUNCT__
#define __FUNCT__ "shiftParticles"
PetscErrorCode shiftParticles( const X3Ctx *ctx, X3PSendList *sendListTable, X3PList particlelist[],
                               PetscInt *const nIsend, const PetscInt slist_size, X3ISend slist[], PetscMPIInt tag, PetscBool solver)
{
  PetscErrorCode ierr;
  const int part_dsize = sizeof(X3Particle)/sizeof(PetscReal); assert(sizeof(X3Particle)%sizeof(PetscReal)==0);
  PetscInt ii,jj,kk,mm,idx;
  DM_PICell *dmpi;
  MPI_Datatype real_type;

  PetscFunctionBeginUser;
  PetscDataTypeToMPIDataType(PETSC_REAL,&real_type);
  dmpi = (DM_PICell *) ctx->dmpic->data;
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(ctx->events[2],0,0,0,0);CHKERRQ(ierr);
#endif
  if ( ctx->use_bsp ) { /* use BSP */
    PetscMPIInt  nto,*fromranks;
    PetscMPIInt *toranks;
    X3Particle  *fromdata,*todata,*pp;
    PetscMPIInt  nfrom,pe;
    int sz;
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
    /* count send  */
    for (ii=0,nto=0;ii<ctx->proc_send_table_size;ii++) {
      if (sendListTable[ii].data_size != 0) {
	sz = X3PSendListSize(&sendListTable[ii]);
	for (jj=0 ; jj<sz ; jj += ctx->chunksize) nto++; /* can just figure this out */
      }
    }
    /* make to ranks & data */
    ierr = PetscMalloc1(nto, &toranks);CHKERRQ(ierr);
    ierr = PetscMalloc1(ctx->chunksize*nto, &todata);CHKERRQ(ierr);
    for (ii=0,nto=0,pp=todata;ii<ctx->proc_send_table_size;ii++) {
      if (sendListTable[ii].data_size) {
	if ((sz=X3PSendListSize(&sendListTable[ii])) > 0) {
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
	  ierr = X3PSendListClear(&sendListTable[ii]);CHKERRQ(ierr);
          assert(X3PSendListSize(&sendListTable[ii])==0);
          assert(sendListTable[ii].data_size);
	} /* a list */
      }
    }

    /* do it */
    ierr = PetscCommBuildTwoSided( ctx->wComm, ctx->chunksize*part_dsize, real_type, nto, toranks, (double*)todata,
				   &nfrom, &fromranks, &fromdata);
    CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(ctx->events[2],0,0,0,0);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(ctx->events[9],0,0,0,0);CHKERRQ(ierr);
#endif
    for (ii=0, pp = fromdata ; ii<nfrom ; ii++) {
      for (jj=0 ; jj<ctx->chunksize ; jj++, pp++) {
	if (pp->gid > 0) {
          PetscInt elid, dim=3;
          if (solver) { /* should vectorize this */
            Vec vec;
            IS pes,elems;
            const PetscInt *peidxs,*elemidxs;
            PetscScalar xx[3];
            for (kk=0;kk<dim;kk++) xx[kk] = pp->x[kk];
            ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, dim, dim, xx, &vec);CHKERRQ(ierr);
            ierr = X3GridSolverLocatePoints(dmpi->dm, vec, &pes, &elems);CHKERRQ(ierr);
            ierr = VecDestroy(&vec);CHKERRQ(ierr);
            ierr = ISGetIndices(pes,&peidxs);CHKERRQ(ierr);
            ierr = ISGetIndices(elems,&elemidxs);CHKERRQ(ierr);
            pe = peidxs[0];
            elid = elemidxs[0];
            if (pe!=ctx->rank) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not local (pe=%D)",pe);
            if (elid<0) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_USER, "No element found %d, pe=%d",elid,pe);
            ierr = ISRestoreIndices(pes,&peidxs);CHKERRQ(ierr);
            ierr = ISRestoreIndices(elems,&elemidxs);CHKERRQ(ierr);
            ierr = ISDestroy(&pes);CHKERRQ(ierr);
            ierr = ISDestroy(&elems);CHKERRQ(ierr);
          }
          else elid = s_fluxtubeelem; /* non-solvers just put in element 0's list */
	  ierr = X3PListAdd( &particlelist[elid], pp, NULL);CHKERRQ(ierr);
        }
      }
    }
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(ctx->events[9],0,0,0,0);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(ctx->events[2],0,0,0,0);CHKERRQ(ierr);
#endif
    ierr = PetscFree(todata);CHKERRQ(ierr);
    ierr = PetscFree(fromranks);CHKERRQ(ierr);
    ierr = PetscFree(fromdata);CHKERRQ(ierr);
    ierr = PetscFree(toranks);CHKERRQ(ierr);
  }
  else { /* non-blocking consensus */
    PetscBool   done=PETSC_FALSE,bar_act=PETSC_FALSE;
    MPI_Request ib_request;
    PetscInt    numSent;
    MPI_Status  status;
    PetscMPIInt flag,sz,pe;
    /* send lists */
    for (ii=0;ii<ctx->proc_send_table_size;ii++) {
      if (sendListTable[ii].data_size != 0 && (sz=X3PSendListSize(&sendListTable[ii])) > 0) {
	if (*nIsend==slist_size) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"process send table too small (%D)",slist_size);
#if defined(PETSC_USE_LOG)
	ierr = PetscLogEventBegin(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
	slist[*nIsend].proc = sendListTable[ii].proc;
	slist[*nIsend].data = sendListTable[ii].data; /* cache data */
	/* send and reset - we can just send this because it is dense */
      	ierr = MPI_Isend((void*)slist[*nIsend].data,sz*part_dsize,real_type,slist[*nIsend].proc,tag,ctx->wComm,&slist[*nIsend].request);
        if (ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MPI_Isend error (%D)",ierr);
	CHKERRQ(ierr);
	(*nIsend)++;
	/* ready for next round, save meta-data  */
	ierr = X3PSendListClear( &sendListTable[ii] );CHKERRQ(ierr);
	sendListTable[ii].data = 0;
	ierr = PetscMalloc1(sendListTable[ii].data_size, &sendListTable[ii].data);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
	ierr = PetscLogEventEnd(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
      }
      /* else - an empty list */
    }
    numSent = *nIsend; /* size of send array */
    /* process receives - non-blocking consensus */
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(ctx->events[3],0,0,0,0);CHKERRQ(ierr);
#endif
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
          X3Particle *data;
#if defined(PETSC_USE_LOG)
          ierr = PetscLogEventEnd(ctx->events[3],0,0,0,0);CHKERRQ(ierr);
          ierr = PetscLogEventEnd(ctx->events[2],0,0,0,0);CHKERRQ(ierr);
          ierr = PetscLogEventBegin(ctx->events[9],0,0,0,0);CHKERRQ(ierr);
#endif
	  MPI_Get_count(&status, real_type, &sz); assert(sz%part_dsize==0);
          ierr = PetscMalloc1(sz, &data);CHKERRQ(ierr);
          ierr = MPI_Recv((void*)data,sz,real_type,status.MPI_SOURCE,tag,ctx->wComm,&status);CHKERRQ(ierr);
	  sz = sz/part_dsize;
	  for (jj=0;jj<sz;jj++) {
            PetscInt elid, dim=3;
            if (solver) {
              Vec vec;
              IS pes,elems;
              const PetscInt *peidxs,*elemidxs;
              PetscScalar xx[3];
              for (kk=0;kk<dim;kk++) xx[kk] = data[jj].x[kk];
              ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, dim, dim, xx, &vec);CHKERRQ(ierr);
              ierr = X3GridSolverLocatePoints(dmpi->dm, vec, &pes, &elems);CHKERRQ(ierr);
              ierr = VecDestroy(&vec);CHKERRQ(ierr);
              ierr = ISGetIndices(pes,&peidxs);CHKERRQ(ierr);
              ierr = ISGetIndices(elems,&elemidxs);CHKERRQ(ierr);
              pe = peidxs[0];
              elid = elemidxs[0];
              if (pe!=ctx->rank) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not local (pe=%D)",pe);
              if (elid<0) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_USER, "No element found %d, pe=%d",elid,pe);
              ierr = ISRestoreIndices(pes,&peidxs);CHKERRQ(ierr);
              ierr = ISRestoreIndices(elems,&elemidxs);CHKERRQ(ierr);
              ierr = ISDestroy(&pes);CHKERRQ(ierr);
              ierr = ISDestroy(&elems);CHKERRQ(ierr);
            }
            else elid = s_fluxtubeelem; /* non-solvers just put in element 0's list */
            ierr = X3PListAdd( &particlelist[elid], &data[jj], NULL);CHKERRQ(ierr);
          }
          ierr = PetscFree(data);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
          ierr = PetscLogEventEnd(ctx->events[9],0,0,0,0);CHKERRQ(ierr);
          ierr = PetscLogEventBegin(ctx->events[2],0,0,0,0);CHKERRQ(ierr);
          ierr = PetscLogEventBegin(ctx->events[3],0,0,0,0);CHKERRQ(ierr);
#endif
	}
      } while (flag);
    } /* non-blocking consensus */
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(ctx->events[3],0,0,0,0);CHKERRQ(ierr);
#endif
  } /* switch for BPS */
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(ctx->events[2],0,0,0,0);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

#define x3_coef(x) (1.0)

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscScalar g3[])
{
  PetscInt d;
  PetscScalar coef = x3_coef(x);
  for (d = 0; d < dim; ++d) g3[d*dim+d] = coef;
}
void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  PetscBool in = PETSC_TRUE;
  PetscInt i;
  for (i=0;i<dim;i++) {
    if (x[i] < -0.5 || x[i] > 0.5) in = PETSC_FALSE;
  }
  if (in && PETSC_FALSE) f0[0] = 1.0;
  else    f0[0] = .0;
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

static PetscErrorCode processParticles( X3Ctx *ctx, const PetscReal dt, X3PSendList **sendListTable, const PetscMPIInt tag,
                                        const int irk, const int istep, PetscBool solver);
#ifdef H5PART
static void prewrite(X3Ctx *ctx, X3PList *l, X3PListPos *ppos1,  X3PListPos *ppos2);
static void postwrite(X3Ctx *ctx, X3PList *l, X3PListPos *ppos1,  X3PListPos *ppos2);
#endif

#ifdef H5PART
/* add corners to get bounding box */
static void prewrite(X3Ctx *ctx, X3PList *l, X3PListPos *ppos1,  X3PListPos *ppos2)
{
  if (ctx->rank==0) {
    X3Particle part;
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
    X3ParticleCreate(&part,1,r,z,phi,0.);
    ierr = X3PListAdd(l,&part,ppos1); assert(!ierr);
    z = -z;
    if (ctx->grid.section_phi == 2) {
      phi += M_PI;
    } else {
      r = ctx->grid.radius_major - ctx->grid.radius_minor;
      phi = ctx->grid.section_phi*M_PI;
    }
    X3ParticleCreate(&part,2,r,z,phi,0.);
    ierr = X3PListAdd(l,&part,ppos2); assert(!ierr);
  }
}
static void postwrite(X3Ctx *ctx, X3PList *l, X3PListPos *ppos1,  X3PListPos *ppos2)
{
  if (ctx->rank==0) {
    X3PListRemoveAt(l,*ppos2);
    X3PListRemoveAt(l,*ppos1);
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
#define X3PROCLISTSIZE 256
#undef __FUNCT__
#define __FUNCT__ "processParticles"
static PetscErrorCode processParticles( X3Ctx *ctx, const PetscReal dt, X3PSendList **sendListTable_in, const PetscMPIInt tag,
					const int irk, const int istep, PetscBool solver)
{
  X3Grid         *grid = &ctx->grid;
  X3PSendList    *sendListTable = *sendListTable_in;
  DM_PICell      *dmpi = (DM_PICell *) ctx->dmpic->data;     assert(solver || irk<0); /* don't push flux tubes */
  PetscReal      psi,theta,dphi,rmaj=grid->radius_major,rminor=grid->radius_minor;
  PetscMPIInt    pe,hash,ii;
  X3Particle     part;
  X3PListPos     pos;
  PetscErrorCode ierr;
  const int      part_dsize = sizeof(X3Particle)/sizeof(double);
  Vec            jVec,xVec,vVec;
  PetscScalar    *xx=0,*jj=0,*vv=0,*xx0=0,*jj0=0,*vv0=0;
  PetscInt       isp,nslist,nlistsTot,elid,elid2,one=1,three=3,ndeposit;
  int            origNlocal,nmoved;
  X3ISend        slist[X3PROCLISTSIZE];
  IS             pes,elems;
  const PetscInt *cpeidxs,*celemidxs;
  PetscFunctionBeginUser;
  if (!ctx->num_particles_total) PetscFunctionReturn(0);
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
       isp <= X3_NION ;
       isp++) {
    const PetscReal mass = ctx->species[isp].mass;
    const PetscReal charge = ctx->species[isp].charge;
    /* loop over element particle lists */
    for (elid=0;elid<ctx->nElems;elid++) {
      X3PList *list = &ctx->partlists[isp][elid];
      if (X3PListSize(list)==0) continue;
      origNlocal += X3PListSize(list);

      /* get Cartesian coordinates (not used for flux tube move) */
      if (solver) {
        ierr = X3PListCompress(list);CHKERRQ(ierr); /* allows for simpler vectorization */
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
#ifdef X3_S_OF_V
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
          ierr = DMGetLocalVector(dmpi->dm, &locphi);CHKERRQ(ierr);
          ierr = DMGlobalToLocalBegin(dmpi->dm, dmpi->phi, INSERT_VALUES, locphi);CHKERRQ(ierr);
          ierr = DMGlobalToLocalEnd(dmpi->dm, dmpi->phi, INSERT_VALUES, locphi);CHKERRQ(ierr);
          /* get E, should set size of vecs for true size? */
          ierr = VecCreateSeq(PETSC_COMM_SELF,three*list->vec_top,&jVec);CHKERRQ(ierr);
          ierr = DMPICellGetJet(ctx->dmpic, xVec, locphi, elid, jVec);CHKERRQ(ierr);
          ierr = DMRestoreLocalVector(dmpi->dm, &locphi);CHKERRQ(ierr);
          ierr = VecGetArray(jVec,&jj0);CHKERRQ(ierr); jj = jj0;
        }
        /* vectorize (todo) push: theta = theta + q*dphi .... grad not used */
        ierr = VecGetArray(xVec,&xx0);CHKERRQ(ierr); xx = xx0;
        for (pos=0 ; pos < list->vec_top ; pos++, xx += 3, jj += 3 ) {
	  /* push particle, real data, could do it on copy for non-final stage of TS, copy new coordinate to xx */
          if (irk>=0) {
            PetscReal b0dotgrad;
#ifdef X3_S_OF_V
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
            X3Particle *ppart = &list->data[pos];
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
#ifdef X3_S_OF_V
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
        ierr = X3GridSolverLocatePoints(dmpi->dm, xVec, &pes, &elems);CHKERRQ(ierr);
      } else {
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"should not be here.");
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
          X3PSendList *newdata;
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
#ifdef X3_S_OF_V
	X3V2P((&part),list->data_v,pos); /* return copy */
#else
	part = list->data[pos]; /* return copy */
#endif
        /* add to list to send, find list with table lookup, send full lists - no vectorization */
        hash = (pe*593)%ctx->proc_send_table_size; /* hash */
        for (ii=0;ii<ctx->proc_send_table_size;ii++){
          if (sendListTable[hash].data_size==0) {
            ierr = X3PSendListCreate(&sendListTable[hash],ctx->chunksize);CHKERRQ(ierr);
            sendListTable[hash].proc = pe;
            ctx->tablecount++;
          }
          if (sendListTable[hash].proc==pe) { /* found hash table entry */
            if (X3PSendListSize(&sendListTable[hash])==X3PSendListMaxSize(&sendListTable[hash]) && !ctx->use_bsp) { /* list is full, send and recreate */
              MPI_Datatype mtype;
#if defined(PETSC_USE_LOG)
              ierr = PetscLogEventBegin(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
              PetscDataTypeToMPIDataType(PETSC_REAL,&mtype);
              /* send and reset - we can just send this because it is dense, but no species data */
              if (nslist==X3PROCLISTSIZE) {
                SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"process send table too small (%D) == snlist(%D)",nslist,(PetscInt)X3PROCLISTSIZE);
              }
              slist[nslist].data = sendListTable[hash].data; /* cache data */
              slist[nslist].proc = pe;
              ierr = MPI_Isend((void*)slist[nslist].data,ctx->chunksize*part_dsize,mtype,pe,tag+isp,ctx->wComm,&slist[nslist].request);
              CHKERRQ(ierr);
              nslist++;
              /* ready for next round, save meta-data  */
              ierr = X3PSendListClear(&sendListTable[hash]);CHKERRQ(ierr);
              assert(sendListTable[hash].data_size == ctx->chunksize);
              sendListTable[hash].data = 0;
              ierr = PetscMalloc1(ctx->chunksize, &sendListTable[hash].data);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
              ierr = PetscLogEventEnd(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
            }
            /* add to list - pass this in as a function to a function? */
            ierr = X3PSendListAdd(&sendListTable[hash],&part);CHKERRQ(ierr); /* not vectorizable */
            ierr = X3PListRemoveAt(list,pos);CHKERRQ(ierr); /* not vectorizable */
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
    ierr = shiftParticles(ctx, sendListTable, ctx->partlists[isp], &nslist, X3PROCLISTSIZE, slist, tag+isp, solver );
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
      ierr = DMGetLocalVector(dmpi->dm, &locrho);CHKERRQ(ierr);
      ierr = VecSet(locrho, 0.0);CHKERRQ(ierr);
      for (elid=0;elid<ctx->nElems;elid++) {
        X3PList *list = &ctx->partlists[isp][elid];
        if (X3PListSize(list)==0) continue;
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventBegin(ctx->events[7],0,0,0,0);CHKERRQ(ierr); /* timer on particle list */
#endif
        ierr = X3PListCompress(list);CHKERRQ(ierr); /* allows for simpler vectorization */
        /* make vectors for this element */
        ierr = VecCreateSeq(PETSC_COMM_SELF,three*list->vec_top, &xVec);CHKERRQ(ierr);
        ierr = VecCreateSeq(PETSC_COMM_SELF,one*list->vec_top, &vVec);CHKERRQ(ierr);
        ierr = VecSetBlockSize(xVec,three);CHKERRQ(ierr);
        ierr = VecSetBlockSize(vVec,one);CHKERRQ(ierr);
        /* make coordinates array and density */
        ierr = VecGetArray(xVec,&xx0);CHKERRQ(ierr); xx = xx0;
        ierr = VecGetArray(vVec,&vv0);CHKERRQ(ierr); vv = vv0;
        /* ierr = X3PListGetHead( list, &part, &pos );CHKERRQ(ierr); */
        /* do { */
        for (pos=0 ; pos < list->vec_top ; pos++, xx += 3, vv++) { /* this has holes, but few and zero weight - vectorizable */
#ifdef X3_S_OF_V
          PetscReal r=list->data_v.r[pos], z=list->data_v.z[pos], phi=list->data_v.phi[pos];
#else
          PetscReal r=list->data[pos].r, z=list->data[pos].z, phi=list->data[pos].phi;
#endif
          cylindricalToCart(r, z, phi, xx);
#ifdef X3_S_OF_V
          *vv = list->data_v.w0[pos]*ctx->species[isp].charge;
#else
          *vv = list->data[pos].w0*ctx->species[isp].charge;
#endif
          ndeposit++;
        }
        /* } while ( !X3PListGetNext(list, &part, &pos) ); */
        ierr = VecRestoreArray(xVec,&xx0);CHKERRQ(ierr);
        ierr = VecRestoreArray(vVec,&vv0);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventEnd(ctx->events[7],0,0,0,0);CHKERRQ(ierr);
#endif
#if defined(PETSC_USE_LOG)
        ierr = PetscLogEventBegin(ctx->events[6],0,0,0,0);CHKERRQ(ierr); /* timer on particle list */
#endif
        ierr = DMPICellAddSource(ctx->dmpic, xVec, vVec, elid, locrho);CHKERRQ(ierr);
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
    for (isp=ctx->use_electrons ? 0 : 1, nloc = 0 ; isp <= X3_NION ; isp++) {
      for (elid=0;elid<ctx->nElems;elid++) {
        nloc += X3PListSize(&ctx->partlists[isp][elid]);
      }
    }
    sb[0] = origNlocal;
    sb[1] = nmoved;
    sb[2] = nlistsTot;
    sb[3] = nloc;
    PetscDataTypeToMPIDataType(PETSC_INT,&mtype);
    ierr = MPI_Allreduce(sb, rb1, 4, mtype, MPI_SUM, ctx->wComm);CHKERRQ(ierr);
    ierr = MPI_Allreduce(sb, rb2, 4, mtype, MPI_MAX, ctx->wComm);CHKERRQ(ierr);
    if (rb1[3]) PetscPrintf(ctx->wComm,
                            "%d) %s %D local particles, %D/%D global, %g %% total particles moved in %D messages total (to %D processors local), %g load imbalance factor\n",
                            istep+1,irk<0 ? "processed" : "pushed", origNlocal, rb1[0], rb1[3], 100.*(double)rb1[1]/(double)rb1[0], rb1[2], ctx->tablecount,(double)rb2[3]/((double)rb1[3]/(double)ctx->npe));
    if (rb1[0] != rb1[3]) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"Number of partilces %D --> %D",rb1[0],rb1[3]);
#ifdef H5PART
    if (irk>=0 && ctx->plot) {
      for (isp=ctx->use_electrons ? 0 : 1 ; isp <= X3_NION ; isp++ ) {
        char  fname1[256],fname2[256];
        X3PListPos pos1,pos2;
        /* hdf5 output */
        if (!isp) {
          sprintf(fname1,         "x3_particles_electrons_time%05d.h5part",(int)istep+1);
          sprintf(fname2,"x3_sub_rank_particles_electrons_time%05d.h5part",(int)istep+1);
        } else {
          sprintf(fname1,         "x3_particles_sp%d_time%05d.h5part",(int)isp,(int)istep+1);
          sprintf(fname2,"x3_sub_rank_particles_sp%d_time%05d.h5part",(int)isp,(int)istep+1);
        }
        /* write */
        prewrite(ctx, &ctx->partlists[isp][s_fluxtubeelem], &pos1, &pos2);
        ierr = X3PListWrite(ctx->partlists[isp], ctx->nElems, ctx->rank, ctx->npe, ctx->wComm, fname1, fname2);CHKERRQ(ierr);
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
#define X3NDIG 100000
/* create particles in flux tube, create particle lists, move particles to flux tube element list */
#undef __FUNCT__
#define __FUNCT__ "createParticles"
static PetscErrorCode createParticles(X3Ctx *ctx)
{
  PetscErrorCode  ierr;
  PetscInt        isp,nCellsLoc,my0,irs,iths,gid,ii,np,dim,cStart,cEnd,elid,idx;
  const PetscReal rmin = ctx->grid.radius_minor;
  const PetscReal dth  = 2*M_PI/(PetscReal)ctx->grid.np_theta;
  const PetscReal dphi = ctx->grid.section_phi*M_PI/(PetscReal)ctx->grid.np_phi; /* rmin for particles < rmin */
  const PetscReal phi1 = (PetscReal)ctx->ParticlePlaneIdx*dphi + 1.e-8,rmaj=ctx->grid.radius_major;
  const PetscInt  nPartProcss_plane = ctx->grid.np_theta*ctx->grid.np_radius; /* nPartProcss_plane == ctx->npe_particlePlane */
  const PetscReal dx = pow( (M_PI*rmin*rmin/4.0) * rmaj*ctx->grid.section_phi*M_PI / (PetscReal)(ctx->npe*ctx->num_particles_proc), 0.333); /* ~length of a particle */
  X3Particle particle;
  DM dm;
  DM_PICell *dmpi;
  PetscFunctionBeginUser;
  if (!ctx->num_particles_total) {
    ctx->sendListTable = NULL;
    ctx->proc_send_table_size = 0;
    for (isp=ctx->use_electrons ? 0 : 1 ; isp <= X3_NION ; isp++ ) ctx->partlists[isp] = NULL;
    PetscFunctionReturn(0);
  }
  /* Create vector and get pointer to data space */
  dmpi = (DM_PICell *) ctx->dmpic->data;
  dm = dmpi->dm;
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
  for (isp=ctx->use_electrons ? 0 : 1 ; isp <= X3_NION ; isp++ ) {
    iths = my0%ctx->grid.np_theta;
    irs = my0/ctx->grid.np_theta;
    ierr = PetscMalloc1(ctx->nElems,&ctx->partlists[isp]);CHKERRQ(ierr);
    {
      const PetscReal r1 = sqrt(((PetscReal)irs      /(PetscReal)ctx->grid.np_radius)*rmin*rmin) +       1.e-12*rmin;
      const PetscReal dr = sqrt((((PetscReal)irs+1.0)/(PetscReal)ctx->grid.np_radius)*rmin*rmin) - (r1 - 1.e-12*rmin);
      const PetscReal th1 = (PetscReal)iths*dth + 1.e-12*dth;
      const PetscReal maxe=ctx->max_vpar*ctx->max_vpar,mass=ctx->species[isp].mass,charge=ctx->species[isp].charge;
      /* create list for element 0 and add all to it */
      ierr = X3PListCreate(&ctx->partlists[isp][s_fluxtubeelem],ctx->chunksize);CHKERRQ(ierr);
      /* create each particle */
      for (np=0 ; np<ctx->num_particles_proc; /* void */ ) {
	PetscReal theta0,r,z;
	const PetscReal psi = r1 + (PetscReal)(rand()%X3NDIG+1)/(PetscReal)(X3NDIG+1)*dr;
	const PetscReal qsaf = qsafty(psi/ctx->grid.radius_minor);
	const PetscInt  NN = (PetscInt)(dth*psi/dx) + 1;
	const PetscReal dth2 = dth/(PetscReal)NN - 1.e-12*dth;
	for ( ii = 0, theta0 = th1 + (PetscReal)(rand()%X3NDIG)/(PetscReal)X3NDIG*dth2;
	      ii < NN && np<ctx->num_particles_proc;
	      ii++, theta0 += dth2, np++ ) {
	  PetscReal       zmax,zdum,v,vpar;
          const PetscReal phi = phi1 + (PetscReal)(rand()%X3NDIG)/(PetscReal)X3NDIG*dphi;
	  const PetscReal thetap = theta0 + qsaf*phi; /* push forward to follow field-lines */
	  polPlaneToCylindrical(psi, thetap, r, z);
	  r += rmaj;
	  /* v_parallel from random number */
	  zmax = 1.0 - exp(-maxe);
	  zdum = zmax*(PetscReal)(rand()%X3NDIG)/(PetscReal)X3NDIG;
	  v= sqrt(-2.0/mass*log(1.0-zdum));
	  v= v*cos(M_PI*(PetscReal)(rand()%X3NDIG)/(PetscReal)X3NDIG);
	  /* vshift= v + up ! shift of velocity */
	  vpar = v*mass/charge;
          ierr = X3ParticleCreate(&particle,++gid,r,z,phi,vpar);CHKERRQ(ierr); /* only time this is called! */
	  ierr = X3PListAdd(&ctx->partlists[isp][s_fluxtubeelem],&particle, NULL);CHKERRQ(ierr);
	} /* theta */
      }
      iths++;
      if (iths==ctx->grid.np_theta) { iths = 0; irs++; }
    } /* cells */
    /* finish off list creates for rest of elements */
    for (elid=0;elid<ctx->nElems;elid++) {
      if (elid!=s_fluxtubeelem) //
        ierr = X3PListCreate(&ctx->partlists[isp][elid],ctx->chunksize);CHKERRQ(ierr); /* this will get expanded, chunksize used for message chunk size and initial list size! */
    }
  } /* species */
  /* init send tables */
  ierr = PetscMalloc1(ctx->proc_send_table_size,&ctx->sendListTable);CHKERRQ(ierr);
  for (idx=0;idx<ctx->proc_send_table_size;idx++) {
    for (isp=ctx->use_electrons ? 0 : 1 ; isp <= X3_NION ; isp++) {
      ctx->sendListTable[idx].data_size = 0; /* init */
    }
  }
  PetscFunctionReturn(0);
}

/* == Defining a base plex for a torus, which looks like a rectilinear donut */

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreatePICellBoxTorus"
static PetscErrorCode DMPlexCreatePICellBoxTorus(MPI_Comm comm, X3Grid *params, DM *dm)
{
  PetscMPIInt    rank;
  PetscInt       numCells = 0;
  PetscInt       numVerts = 0;
  PetscReal      radius_major  = params->radius_major;
  const PetscInt num_phi_cells = params->num_phi_cells, nplains = (s_section_phi == 2) ? num_phi_cells : num_phi_cells+1;;
  int            *flatCells = NULL;
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
  ierr = PetscObjectSetName((PetscObject) *dm, "boxtorus");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* == Defining a base plex for a torus, which looks like a rectilinear donut, and a mapping that turns it into a conventional round donut == */

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreatePICellTorus"
static PetscErrorCode DMPlexCreatePICellTorus(MPI_Comm comm, X3Grid *params, DM *dm)
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
        double cosphi, sinphi, r;

        if (s_section_phi == 2) {
          cosphi = cos(s_section_phi * M_PI * i / num_phi_cells);
          sinphi = sin(s_section_phi * M_PI * i / num_phi_cells);
        }
        for (j = 0; j < 8; j++) {
          double z;
          double mult = (j < 4) ? inner_mult : 1.;
          z = mult * radius_minor * sin(j * M_PI_2);
          coords[i][j][2] = z;
          r = radius_major + mult * radius_minor * cos(j * M_PI_2);
          if (s_section_phi == 2) {
            coords[i][j][0] = cosphi * r;
            coords[i][j][1] = sinphi * r;
          } else {
            coords[i][j][0] = r;
            coords[i][j][1] = (params->radius_major+params->radius_minor) * tan(s_section_phi*M_PI)*(double)i/(double)num_phi_cells; /* height of cylinder */
          }
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
  X3Ctx     *ctx = (X3Ctx*)a_ctx;
  X3Grid    *grid = &ctx->grid;
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
  z = abc[2];
  a = abc[0];
  b = abc[1];
  if (ctx->grid.section_phi!=2) b = 0;
  inPhi = atan2(b,a);
  inPhi = (inPhi < 0.) ? (inPhi + ctx->grid.section_phi * M_PI) : inPhi;
  i = (inPhi * num_phi_cells) / (ctx->grid.section_phi * M_PI);
  i = PetscMin(i,num_phi_cells - 1);
  leftPhi =  (i *        ctx->grid.section_phi * M_PI) / num_phi_cells;
  midPhi  = ((i + 0.5) * ctx->grid.section_phi * M_PI) / num_phi_cells;
  cosMidPhi  = cos(midPhi);
  sinMidPhi  = sin(midPhi);
  cosLeftPhi = cos(leftPhi);
  sinLeftPhi = sin(leftPhi);
  secHalf    = 1. / cos(ctx->grid.section_phi*M_PI / (2*num_phi_cells) );
  rhat = (a * cosMidPhi + b * sinMidPhi);
  r    = secHalf * rhat;
  dist = secHalf * (-a * sinLeftPhi + b * cosLeftPhi);
  fulldist = 2. * sin(ctx->grid.section_phi*M_PI / (2*num_phi_cells)) * r;
  outPhi = ((i + (dist/fulldist)) * ctx->grid.section_phi * M_PI) / num_phi_cells;
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
  if (ctx->grid.section_phi!=2) xyz[1] = abc[1];
  else xyz[1] = r * sinOutPhi;
  xyz[2] = z;
  PetscFunctionReturn(0);
}

/******************* MHD ********************/
typedef struct {
  PetscScalar vals[0];
  PetscScalar r;
  PetscScalar ru[3];
  PetscScalar b[3];
  PetscScalar e;
} MHDNode;
#define DOT3(__x,__y,__r) {int i;for(i=0,__r=0;i<3;i++) __r += __x[i] * __y[i];}
#define MATVEC3(__a,__x,__p) {int i,j; for (i=0.; i<3; i++) {__p[i] = 0; for (j=0.; j<3; j++) __p[i] += __a[i][j]*__x[j]; }}
#define MATTRANPOSEVEC3(__a,__x,__p) {int i,j; for (i=0.; i<3; i++) {__p[i] = 0; for (j=0.; j<3; j++) __p[i] += __a[j][i]*__x[j]; }}

#undef __FUNCT__
#define __FUNCT__ "PhysicsBoundary_MHD_Wall"
static PetscErrorCode PhysicsBoundary_MHD_Wall(PetscReal time, const PetscReal *c, const PetscReal n[], const PetscScalar *axI, PetscScalar *axG, void *a_ctx)
{
  PetscInt      i;
  const MHDNode *xI = (const MHDNode*)axI;
  MHDNode       *xG = (MHDNode*)axG;
  PetscFunctionBeginUser;
  /* PetscPrintf(PETSC_COMM_WORLD,"%s: xI=%16.8e %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e, coord = %16.8e %16.8e %16.8e normal = %16.8e %16.8e %16.8e\n",__FUNCT__,xI->vals[0],xI->vals[1],xI->vals[2],xI->vals[3],xI->vals[4],xI->vals[5],xI->vals[6],xI->vals[7],c[0],c[1],c[2],n[0],n[1],n[2]); */
  /* if (xI->r<=0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER," bad density = %g",xI->r); */
  xG->r = xI->r; /* ghost cell density - same */
  xG->e = xI->e; /* ghost cell energy - same */
  for (i=0; i<3; i++) xG->b[i]  = -xI->b[i];  /* zero BC, negative */
  for (i=0; i<3; i++) xG->ru[i] = -xI->ru[i]; /* no flow? */
  /* PetscPrintf(PETSC_COMM_WORLD,"%s: G = %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e\n",__FUNCT__,xG->vals[0],xG->vals[1],xG->vals[2],xG->vals[3],xG->vals[4],xG->vals[5],xG->vals[6],xG->vals[7]); */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetDurl"
static void SetDurl(MHDNode *durl, const MHDNode *uL, const MHDNode *uR, const PetscReal gamma)
{
  PetscReal press;
  PetscFunctionBeginUser;
  durl->r=uR->r-uL->r;
  durl->ru[0]=uR->r*uR->ru[0]-uL->r*uL->ru[0];
  durl->ru[1]=uR->r*uR->ru[1]-uL->r*uL->ru[1];
  durl->ru[2]=uR->r*uR->ru[2]-uL->r*uL->ru[2];
  durl->b[0]=uR->b[0]-uL->b[0];
  durl->b[1]=uR->b[1]-uL->b[1];
  durl->b[2]=uR->b[2]-uL->b[2];
  press=uL->e;
  durl->e=press/(gamma-1.0)+0.5*uL->r*(uL->ru[0]*uL->ru[0]+uL->ru[1]*uL->ru[1]+uL->ru[2]*uL->ru[2])+0.5*(uL->b[0]*uL->b[0]+uL->b[1]*uL->b[1]+uL->b[2]*uL->b[2]);
  press=uR->e;
  durl->e=press/(gamma-1.0)+0.5*uR->r*(uR->ru[0]*uR->ru[0]+uR->ru[1]*uR->ru[1]+uR->ru[2]*uR->ru[2])+0.5*(uR->b[0]*uR->b[0]+uR->b[1]*uR->b[1]+uR->b[2]*uR->b[2])-durl->e;
  PetscFunctionReturnVoid();
}

#undef __FUNCT__
#define __FUNCT__ "SetEigenValues"
static void SetEigenValues(MHDNode *utilde, PetscReal alamda[], const PetscReal gamma)
{
  PetscReal rhoInv,Asq,axsq,csndsq,cfast,tmp,btlocalx=0,btlocaly=0,btlocalz=0;

  PetscFunctionBeginUser;
  rhoInv=1/utilde->r;
  axsq=(utilde->b[0]+btlocalx)*(utilde->b[0]+btlocalx)*rhoInv;
  Asq=((utilde->b[0]+btlocalx)*(utilde->b[0]+btlocalx)+(utilde->b[1]+btlocaly)*(utilde->b[1]+btlocaly)+(utilde->b[2]+btlocalz)*(utilde->b[2]+btlocalz))*rhoInv;
  csndsq=PetscMax(gamma*(utilde->e)*rhoInv,1.e-4);
  tmp=PetscSqrtReal(PetscMax(((csndsq+Asq)*(csndsq+Asq) - 4.*csndsq*axsq),0));
  cfast=PetscSqrtReal(0.5*(csndsq+Asq+tmp));
  alamda[0]=(utilde->ru[0]+cfast);
  alamda[1]=(utilde->ru[0]-cfast);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__
#define __FUNCT__ "MHDFlux"
static void MHDFlux(const MHDNode *vl, const MHDNode *vr, PetscReal area, X3Ctx *ctx, MHDNode *flux)
{
  int       i;
  MHDNode   finvl,finvr,durl,utilde;
  PetscReal alamdaL[2],alamdaR[2],rho,u,v,w,bx,by,bz,eint,press,bxsq,lamdaMax=0,lamdaMin=0,bysq,bzsq;

  PetscFunctionBeginUser;
  for (i=0; i<2; i++) alamdaL[i] = 0;
  for (i=0; i<2; i++) alamdaR[i] = 0;

  utilde = *vl;
  SetEigenValues(&utilde,alamdaL,ctx->gamma);
  utilde = *vr;
  SetEigenValues(&utilde,alamdaR,ctx->gamma);
  SetDurl(&durl,vl,vr,ctx->gamma);
  for (i=0;i<2;i++) lamdaMin=PetscMin(lamdaMin,PetscMin(alamdaL[i],alamdaR[i]));
  for (i=0;i<2;i++) lamdaMax=PetscMax(lamdaMax,PetscMax(alamdaL[i],alamdaR[i]));
  rho=vl->r;
  u=vl->ru[0]; v=vl->ru[1]; w=vl->ru[2];
  bx=vl->b[0]; by=vl->b[1]; bz=vl->b[2];
  bxsq=bx*bx;
  bysq=by*by;
  bzsq=bz*bz;
  press=vl->e;
  finvl.r=rho*u;
  finvl.ru[0]=rho*u*u +press+0.5*(bysq+bzsq-bxsq);
  finvl.ru[1]=rho*u*v-bx*by;
  finvl.ru[2]=rho*u*w-bx*bz;
  eint=(press)/(ctx->gamma-1);
  finvl.b[0]=0;
  finvl.b[1]=u*by-v*bx;
  finvl.b[2]=u*(bz)-w*bx;
  finvl.e=(0.5*rho*(u*u+v*v+w*w)+eint+(press)+(bx*bx+by*by+bz*bz))*vl->ru[0]-bx*(u*bx+v*by+w*bz);
  rho=vr->r;
  u=vr->ru[0]; v=vr->ru[1]; w=vr->ru[2];
  bx=vr->b[0]; by=vr->b[1]; bz=vr->b[2];
  bxsq=bx*bx;
  bysq=by*by;
  bzsq=bz*bz;
  press=vr->e;
  finvr.r=rho*u;
  finvr.ru[0]=rho*u*u +press+0.5*(bysq+bzsq-bxsq);
  finvr.ru[1]=rho*u*v-bx*by;
  finvr.ru[2]=rho*u*w-bx*bz;
  eint=(press)/(ctx->gamma-1);
  finvr.b[0]=0;
  finvr.b[1]=u*by-v*bx;
  finvr.b[2]=u*(bz)-w*bx;
  finvr.e=(0.5*rho*(u*u+v*v+w*w)+eint+press+(bx*bx+by*by+bz*bz))*vr->ru[0]-bx*(u*bx+v*by+w*bz);
  for (i=0;i<ctx->ndof;i++) flux->vals[i] = (lamdaMax*finvl.vals[i]-lamdaMin*finvr.vals[i]+lamdaMin*lamdaMax*durl.vals[i])/(lamdaMax-lamdaMin);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__
#define __FUNCT__ "PhysicsRiemann_MHD"
static void PhysicsRiemann_MHD( PetscInt dim, PetscInt Nf, const PetscReal x[], const PetscReal n[],
                                const PetscScalar *xL, const PetscScalar *xR, PetscScalar *aflux, void *a_ctx)
{
  X3Ctx           *ctx = (X3Ctx*)a_ctx;
  PetscReal       nn[3],t,c,v[3],R[3][3],a,ai;
  PetscInt        i,j;
  const MHDNode   *uL = (const MHDNode*)xL,*uR = (const MHDNode*)xR;
  MHDNode         *retflux = (MHDNode*)aflux, flux;
  MHDNode          luR, luL;
  /* MHDNode         fL,fR; */
  PetscFunctionBeginUser;

  for (i=0,a=0; i<3; i++) {
    nn[i] = n[i];
    a += nn[i]*nn[i];
  }
  ai = 1/PetscSqrtReal(a); /* area inverse */
  for (i=0; i<3; i++) nn[i] *= ai; /* |nn|==1 */
  if (nn[0] < -0.999999) { /* == -1 */
    /* PetscPrintf(PETSC_COMM_WORLD," ***** %s: Nf=%D, area=%g, Have -1 normal n = %g %g %g nn = %g %g %g\n",__FUNCT__,Nf,PetscSqrtReal(a),n[0],n[1],n[2],nn[0],nn[1],nn[2]); */
    for (i=0; i<3; i++) for (j=0; j<3; j++) R[i][j] = (i==j) ? -1 : 0; /* I * -cos(theta) = -I */
    /* R[1][1] = 1; */ /* rotation about y axis 180 degrees */
    R[2][2] = 1; /* rotation about z axis 180 degrees */
  } else {
    /* rotation matrix to put vectors on x axis, v = n X e_1 */
    v[0] = 0;
    v[1] = nn[2];
    v[2] = -nn[1];
    c = nn[0];
    R[0][0] = 1;     R[0][1] = -v[2]; R[0][2] =  v[1]; /* I + v cross */
    R[1][0] =  v[2]; R[1][1] = 1;     R[1][2] = -v[0];
    R[2][0] = -v[1]; R[2][1] =  v[0]; R[2][2] = 1;
    t = 1/(1+c); /* + (v cross)^2 / (1+c) */
    R[0][0] -= t*(v[2]*v[2] + v[1]*v[1]); R[0][1] += t*v[0]*v[1];               R[0][2] += t*v[2]*v[0];
    R[1][0] += t*v[0]*v[1];               R[1][1] -= t*(v[2]*v[2] + v[0]*v[0]); R[1][2] += t*v[1]*v[2];
    R[2][0] += t*v[2]*v[0];               R[2][1] += t*v[1]*v[2];               R[2][2] -= t*(v[1]*v[1] + v[0]*v[0]);
  }
  luL.r = uL->r; /* copy states and rotate to local coordinate, nn = (1,0,0) */
  luL.e = uL->e;
  luR.r = uR->r;
  luR.e = uR->e;
  MATVEC3(R,uR->ru,luR.ru);
  MATVEC3(R,uL->ru,luL.ru);
  MATVEC3(R,uR->b, luR.b);
  MATVEC3(R,uL->b, luL.b);
  /* PetscPrintf(PETSC_COMM_WORLD,"%s: uR.ru = %g %g %g, local uR.ru = %g %g %g, n = %g %g %g \n",__FUNCT__,uR->ru[0],uR->ru[1],uR->ru[2],luR.ru[0],luR.ru[1],luR.ru[2],n[0],n[1],n[2]); */
  /* compute flux */
  MHDFlux(&luL, &luR, a, ctx, &flux);
  /* rotate fluxes back to original coordinate system */
  MATTRANPOSEVEC3(R,flux.ru,retflux->ru);
  MATTRANPOSEVEC3(R,flux.b,retflux->b);
  retflux->r = flux.r;
  retflux->e = flux.e;
  PetscFunctionReturnVoid();
}

#undef __FUNCT__
#define __FUNCT__ "SolutionFunctional"
/* put the solution callback into a functional callback */
static PetscErrorCode SolutionFunctional(PetscInt dim, PetscReal time, const PetscReal xxx[], PetscInt Nf, PetscScalar *u, void *modctx)
{
  X3Ctx           *ctx = (X3Ctx*)modctx;
  PetscInt        i;
  MHDNode         *uu  = (MHDNode*)u;
  PetscScalar     c,t;
  PetscFunctionBegin;
  if (time != 0.0) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No solution known for time %g",(double)time);
  for (i=0; i<3; i++) uu->ru[i] = 0.0; /* zero out initial velocity */
  for (i=0; i<3; i++) uu->b[i] = 0.0;  /* zero out B */
  uu->ru[1] = xxx[1]; /* flow down tube pulling away */
  /* set E and rho */
  uu->r = 1.;
  uu->e = 10./(ctx->gamma-1.);
  for (i=0,c=0; i<3; i++) {
    t = uu->ru[i]/uu->r;
    c += t*t;
  }
  c = PetscSqrtReal(c);
  if (c > ctx->maxspeed) ctx->maxspeed = c;
  PetscPrintf(PETSC_COMM_WORLD,"%s: uu=%16.8e %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e %16.8e, coord = %16.8e %16.8e %16.8e\n",__FUNCT__,uu->vals[0],uu->vals[1],uu->vals[2],uu->vals[3],uu->vals[4],uu->vals[5],uu->vals[6],uu->vals[7],xxx[0],xxx[1],xxx[2]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetInitialCondition"
PetscErrorCode SetInitialCondition(DM dm, Vec X, X3Ctx *ctx)
{
  PetscErrorCode     (*func[1]) (PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  void               *ctxa[1];
  PetscErrorCode     ierr;
  PetscFunctionBeginUser;
  func[0] = SolutionFunctional;
  ctxa[0] = (void *) ctx;
  ierr = DMProjectFunction(dm,0.0,func,ctxa,INSERT_ALL_VALUES,X);CHKERRQ(ierr);
  /* ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ErrorIndicator_Simple"
static PetscErrorCode ErrorIndicator_Simple(PetscInt dim, PetscReal volume, PetscInt numComps, const PetscScalar u[], const PetscScalar grad[], PetscReal *error, void *ctx)
{
  PetscReal      err = 0.;
  PetscInt       i, j;

  PetscFunctionBeginUser;
  for (i = 0; i < numComps; i++) {
    for (j = 0; j < dim; j++) {
      err += PetscSqr(PetscRealPart(grad[i * dim + j]));
    }
  }
  *error = volume * err;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "initializeTS"
static PetscErrorCode initializeTS(DM dm, X3Ctx *ctx, TS *ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSCreate(ctx->wComm, ts);CHKERRQ(ierr);
  ierr = TSSetType(*ts, TSSSP);CHKERRQ(ierr);
  ierr = TSSetDM(*ts, dm);CHKERRQ(ierr);
  ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, ctx);CHKERRQ(ierr);
  ierr = DMTSSetRHSFunctionLocal(dm, DMPlexTSComputeRHSFunctionFVM, ctx);CHKERRQ(ierr);
  ierr = TSSetDuration(*ts,ctx->msteps,1.e12);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(*ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "adaptToleranceFVM"
static PetscErrorCode adaptToleranceFVM(PetscFV fvm, TS ts, Vec sol, PetscReal refineTol, PetscReal coarsenTol, X3Ctx *ctx, TS *tsNew, Vec *solNew)
{
  DM                dm, gradDM, plex, cellDM, adaptedDM = NULL;
  Vec               cellGeom, faceGeom;
  PetscBool         isForest, computeGradient;
  Vec               grad, locGrad, locX;
  PetscInt          cStart, cEnd, cEndInterior, c, dim;
  PetscReal         minMaxInd[2] = {PETSC_MAX_REAL, PETSC_MIN_REAL}, minMaxIndGlobal[2], minInd, maxInd, time;
  const PetscScalar *pointVals;
  const PetscScalar *pointGrads;
  const PetscScalar *pointGeom;
  DMLabel           adaptLabel = NULL;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = TSGetTime(ts,&time);CHKERRQ(ierr);
  ierr = VecGetDM(sol, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = PetscFVGetComputeGradients(fvm,&computeGradient);CHKERRQ(ierr);
  ierr = PetscFVSetComputeGradients(fvm,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMIsForest(dm, &isForest);CHKERRQ(ierr);
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMPlexGetDataFVM(plex, fvm, &cellGeom, &faceGeom, &gradDM);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(plex,&locX);CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(plex, PETSC_TRUE, locX, 0.0, faceGeom, cellGeom, NULL);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(plex, sol, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (plex, sol, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(gradDM, &grad);CHKERRQ(ierr);
  ierr = DMPlexReconstructGradientsFVM(plex, locX, grad);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(gradDM, &locGrad);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(gradDM, grad, INSERT_VALUES, locGrad);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(gradDM, grad, INSERT_VALUES, locGrad);CHKERRQ(ierr);
  ierr = VecDestroy(&grad);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(plex,&cEndInterior,NULL,NULL,NULL);CHKERRQ(ierr);
  cEnd = (cEndInterior < 0) ? cEnd : cEndInterior;

  ierr = VecGetArrayRead(locGrad,&pointGrads);CHKERRQ(ierr);
  ierr = VecGetArrayRead(cellGeom,&pointGeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(locX,&pointVals);CHKERRQ(ierr);
  ierr = VecGetDM(cellGeom,&cellDM);CHKERRQ(ierr);
  ierr = DMLabelCreate("adapt",&adaptLabel);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; c++) {
    PetscReal             errInd = 0.;
    const PetscScalar     *pointGrad;
    const PetscScalar     *pointVal;
    const PetscFVCellGeom *cg;

    ierr = DMPlexPointLocalRead(gradDM,c,pointGrads,&pointGrad);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(cellDM,c,pointGeom,&cg);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(plex,c,pointVals,&pointVal);CHKERRQ(ierr);

    ierr = ErrorIndicator_Simple(dim,cg->volume,ctx->ndof,pointVal,pointGrad,&errInd,ctx);CHKERRQ(ierr);
    minMaxInd[0] = PetscMin(minMaxInd[0],errInd);
    minMaxInd[1] = PetscMax(minMaxInd[1],errInd);
    if (errInd > refineTol)  {ierr = DMLabelSetValue(adaptLabel,c,DM_ADAPT_REFINE);CHKERRQ(ierr);}
    if (errInd < coarsenTol) {ierr = DMLabelSetValue(adaptLabel,c,DM_ADAPT_COARSEN);CHKERRQ(ierr);}
  }
  ierr = VecRestoreArrayRead(locX,&pointVals);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(cellGeom,&pointGeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(locGrad,&pointGrads);CHKERRQ(ierr);
  ierr = VecDestroy(&locGrad);CHKERRQ(ierr);
  ierr = VecDestroy(&locX);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  ierr = PetscFVSetComputeGradients(fvm,computeGradient);CHKERRQ(ierr);
  minMaxInd[1] = -minMaxInd[1];
  ierr = MPI_Allreduce(minMaxInd,minMaxIndGlobal,2,MPIU_REAL,MPI_MIN,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  minInd = minMaxIndGlobal[0];
  maxInd = -minMaxIndGlobal[1];
  ierr = PetscInfo2(ts, "error indicator range (%E, %E)\n", minInd, maxInd);CHKERRQ(ierr);
  if (maxInd > refineTol || minInd < coarsenTol) { /* at least one cell is over the refinement threshold */
    ierr = DMAdaptLabel(dm,adaptLabel,&adaptedDM);CHKERRQ(ierr);
  }
  else if (maxInd < coarsenTol) { /* all cells are under the coarsening threshold */
    ierr = DMCoarsen(dm,PetscObjectComm((PetscObject)dm),&adaptedDM);CHKERRQ(ierr);
  }
  else if (minInd > refineTol) { /* all cells are over the refinement threshold */
    ierr = DMRefine(dm,PetscObjectComm((PetscObject)dm),&adaptedDM);CHKERRQ(ierr);
  }
  ierr = DMLabelDestroy(&adaptLabel);CHKERRQ(ierr);
  if (adaptedDM) {
    if (tsNew) {
      ierr = initializeTS(adaptedDM,ctx,tsNew);CHKERRQ(ierr);
    }
    if (solNew) {
      ierr = DMCreateGlobalVector(adaptedDM, solNew);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) *solNew, "mhd");CHKERRQ(ierr);
      ierr = DMForestTransferVec(dm, sol, adaptedDM, *solNew, PETSC_TRUE, time);CHKERRQ(ierr);
    }
    if (isForest) {ierr = DMForestSetAdaptivityForest(adaptedDM,NULL);CHKERRQ(ierr);} /* clear internal references to the previous dm */
    ierr = DMDestroy(&adaptedDM);CHKERRQ(ierr);
  }
  else {
    if (tsNew) *tsNew = NULL;
    if (solNew) *solNew = NULL;
  }
  PetscFunctionReturn(0);
}
/*
   X3: ProcessOptions: set parameters from input, setup w/o allocation, called first, no DM here
*/
#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions( X3Ctx *ctx )
{
  PetscErrorCode ierr;
  PetscBool      phiFlag,radFlag,thetaFlag,flg,chunkFlag,secFlg,rMajFlg;
  char           fname[256];
  PetscReal      t1;
  PetscFunctionBeginUser;
  ctx->events = s_events;
#if defined(PETSC_USE_LOG)
  {
    PetscInt currevent = 0;
    PetscLogStage  setup_stage;
    ierr = PetscLogEventRegister("X3CreateMesh", DM_CLASSID, &ctx->events[currevent++]);CHKERRQ(ierr); /* 0 */
    ierr = PetscLogEventRegister("X3Process parts",0,&ctx->events[currevent++]);CHKERRQ(ierr); /* 1 */
    ierr = PetscLogEventRegister(" -shiftParticles",0,&ctx->events[currevent++]);CHKERRQ(ierr); /* 2 */
    ierr = PetscLogEventRegister("   =Non-block con",0,&ctx->events[currevent++]);CHKERRQ(ierr); /* 3 */
    ierr = PetscLogEventRegister("     *Part. Send", 0, &ctx->events[currevent++]);CHKERRQ(ierr); /* 4 */
    ierr = PetscLogEventRegister(" -Move parts", 0, &ctx->events[currevent++]);CHKERRQ(ierr); /* 5 */
    ierr = PetscLogEventRegister(" -AddSource", 0, &ctx->events[currevent++]);CHKERRQ(ierr); /* 6 */
    ierr = PetscLogEventRegister(" -Pre Push", 0, &ctx->events[currevent++]);CHKERRQ(ierr); /* 7 */
    ierr = PetscLogEventRegister(" -Push (Jet)", 0, &ctx->events[currevent++]);CHKERRQ(ierr); /* 8 */
    ierr = PetscLogEventRegister("   =Part find (s)", 0, &ctx->events[currevent++]);CHKERRQ(ierr); /* 9 */
    ierr = PetscLogEventRegister("   =Part find (p)", 0, &ctx->events[currevent++]);CHKERRQ(ierr); /* 10 */
    ierr = PetscLogEventRegister("X3Poisson Solve", 0, &ctx->events[currevent++]);CHKERRQ(ierr); /* 11 */
    ierr = PetscLogEventRegister("X3Part AXPY", 0, &ctx->events[currevent++]);CHKERRQ(ierr); /* 12 */
    ierr = PetscLogEventRegister("X3Compress array", 0, &ctx->events[currevent++]);CHKERRQ(ierr); /* 13 */
    ierr = PetscLogEventRegister("X3Diagnostics", 0, &ctx->events[diag_event_id]);CHKERRQ(ierr); /* N-1 */
    assert(sizeof(s_events)/sizeof(s_events[0]) > currevent);
    ierr = PetscLogStageRegister("Setup", &setup_stage);CHKERRQ(ierr);
    ierr = PetscLogStagePush(setup_stage);CHKERRQ(ierr);
  }
#endif
  /* general */
  ierr = MPI_Comm_rank(ctx->wComm, &ctx->rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(ctx->wComm, &ctx->npe);CHKERRQ(ierr);
  s_rank = ctx->rank;
  /* physics */
  ctx->massAu = 2;  /* mass ratio to proton */
  /* ctx->eMassAu=2e-2; /\* mass of electron?? *\/ */
  ctx->chargeEu = 1;    /* charge number */
  ctx->eChargeEu = -1;  /* negative electron */
  ctx->maxspeed = 0;
  ctx->species[1].mass=ctx->massAu*x3ProtMass;
  ctx->species[1].charge=ctx->chargeEu*x3ECharge;
  ctx->species[0].mass=x3ElecMass/* ctx->eMassAu*x3ProtMass */;
  ctx->species[0].charge=ctx->eChargeEu*x3ECharge;

  /* mesh */
  ctx->grid.radius_major = 5.;
  ctx->grid.radius_minor = 1.;
  ctx->grid.section_phi  = 2.0; /* 2 pi, whole torus */
  ctx->grid.np_phi  = 1;
  ctx->grid.np_radius = 1;
  ctx->grid.np_theta  = 1;
  ctx->grid.num_phi_cells = 4; /* number of poloidal planes (before refinement) */
  ctx->grid.inner_mult= M_SQRT2 - 1.;

  ctx->tablecount = 0;

  ierr = PetscOptionsBegin(ctx->wComm, "", "Poisson Problem Options", "X3");CHKERRQ(ierr);
  /* general options */
  s_debug = 0;
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex3.c", s_debug, &s_debug, NULL);CHKERRQ(ierr);
  ctx->plot = PETSC_TRUE;
  ierr = PetscOptionsBool("-plot", "Write plot files", "ex3.c", ctx->plot, &ctx->plot, NULL);CHKERRQ(ierr);
  ctx->plot_amr_initial = PETSC_FALSE;
  ierr = PetscOptionsBool("-plot_amr_initial", "Write plot files for initial AMR grid", "ex3.c", ctx->plot_amr_initial, &ctx->plot_amr_initial, NULL);CHKERRQ(ierr);
  ctx->use_amr = PETSC_FALSE;
  ierr = PetscOptionsBool("-use_amr", "Use adaptive mesh refinement (for MHD)", "ex3.c", ctx->use_amr, &ctx->use_amr, NULL);CHKERRQ(ierr);
  ctx->refine_tol = PETSC_MAX_REAL;
  ierr = PetscOptionsReal("-refine_tol", "Tolerance for refinement", "ex3.c", ctx->refine_tol, &ctx->refine_tol, NULL);CHKERRQ(ierr);
  ctx->coarsen_tol = 0;
  ierr = PetscOptionsReal("-coarsen_tol", "Tolerance for coarsening", "ex3.c", ctx->coarsen_tol, &ctx->coarsen_tol, NULL);CHKERRQ(ierr);
  ctx->chunksize = X3_V_LEN; /* too small */
  ierr = PetscOptionsInt("-chunksize", "Size of particle list to chunk sends", "ex3.c", ctx->chunksize, &ctx->chunksize,&chunkFlag);CHKERRQ(ierr);
  if (chunkFlag) ctx->chunksize = X3_V_LEN*(ctx->chunksize/X3_V_LEN);
  if (ctx->chunksize<=0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER," invalid chuck size = %D",ctx->chunksize);
  ctx->use_bsp = 0;
  ierr = PetscOptionsInt("-use_bsp", "Size of chucks for PETSc's TwoSide communication (0 to use 'nonblocking consensus')", "ex3.c", ctx->use_bsp, &ctx->use_bsp, NULL);CHKERRQ(ierr);
  if (ctx->use_bsp<0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER," invalid BSP chuck size = %D",ctx->use_bsp);
  ctx->proc_send_table_size = ((ctx->npe>100) ? 100 + (ctx->npe-100)/10 : ctx->npe) + 1; /* hash table size of processors to send to */
  ierr = PetscOptionsInt("-proc_send_table_size", "Size of hash table proc->send_list", "ex3.c",ctx->proc_send_table_size, &ctx->proc_send_table_size, NULL);CHKERRQ(ierr);

  /* Domain and mesh definition */
  ierr = PetscOptionsReal("-radius_minor", "Minor radius of torus", "ex3.c", ctx->grid.radius_minor, &ctx->grid.radius_minor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-radius_major", "Major radius of torus", "ex3.c", ctx->grid.radius_major, &ctx->grid.radius_major, &rMajFlg);CHKERRQ(ierr);
    /* section of the torus, adjust major radius as needed */
  ierr = PetscOptionsReal("-section_phi", "Number of pi radians of torus section, phi direction (0 < section_phi <= 2) ", "ex3.c",ctx->grid.section_phi,&ctx->grid.section_phi,&secFlg);CHKERRQ(ierr);
  if (ctx->grid.section_phi <= 0 || ctx->grid.section_phi > 2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"invalid -section_phi %g: (0,2]",ctx->grid.section_phi);
  t1 = 1.;
  ierr = PetscOptionsReal("-section_length", "Length if section (cylinder)", "ex3.c", t1, &t1,&flg);CHKERRQ(ierr);
  if (flg || secFlg) {
    if(secFlg && flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "-section_length with -section_phi not allowed, over constrained specification");
    else if(secFlg) {
      /* section of torus */
    } else {
      /* cylindrical flux tube, with L (section_length) specified */
      ctx->grid.section_phi = atan2( t1, ctx->grid.radius_major + ctx->grid.radius_minor ) / M_PI;
      PetscPrintf(PETSC_COMM_WORLD,"%s: L = %16.8e %16.8e\n",__FUNCT__,t1,(ctx->grid.radius_major + ctx->grid.radius_minor)*ctx->grid.section_phi*M_PI);
    }
  }
  s_section_phi = ctx->grid.section_phi;

  ierr = PetscOptionsInt("-num_phi_cells", "Number of cells per major circle", "ex3.c", ctx->grid.num_phi_cells, &ctx->grid.num_phi_cells, NULL);CHKERRQ(ierr);
  if (ctx->grid.num_phi_cells<3 && ctx->grid.section_phi==2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER," Not enough phi cells %D",ctx->grid.num_phi_cells);
  ierr = PetscOptionsReal("-inner_mult", "Fraction of minor radius taken by inner square", "ex3.c", ctx->grid.inner_mult, &ctx->grid.inner_mult, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-np_phi", "Number of planes for particle mesh", "ex3.c", ctx->grid.np_phi, &ctx->grid.np_phi, &phiFlag);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-np_radius", "Number of radial cells for particle mesh", "ex3.c", ctx->grid.np_radius, &ctx->grid.np_radius, &radFlag);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-np_theta", "Number of theta cells for particle mesh", "ex3.c", ctx->grid.np_theta, &ctx->grid.np_theta, &thetaFlag);CHKERRQ(ierr);
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
  ierr = PetscOptionsInt("-mstep", "Maximum number of time steps", "ex3.c", ctx->msteps, &ctx->msteps, NULL);CHKERRQ(ierr);
  ctx->maxTime = 1000000000.;
  ierr = PetscOptionsReal("-maxTime", "Maximum time", "ex3.c",ctx->maxTime,&ctx->maxTime,NULL);CHKERRQ(ierr);
  ctx->dt = 0.;
  ierr = PetscOptionsReal("-dt","Time step","ex3.c",ctx->dt,&ctx->dt,NULL);CHKERRQ(ierr);
  /* particles */
  ctx->num_particles_proc = 10;
  ierr = PetscOptionsInt("-num_particles_proc", "Number of particles local (flux tube cell)", "ex3.c", ctx->num_particles_proc, &ctx->num_particles_proc, NULL);CHKERRQ(ierr);
  if (ctx->num_particles_proc<0) {
    ctx->num_particles_total = -ctx->num_particles_proc;
    if (!ctx->rank) ctx->num_particles_proc = -ctx->num_particles_proc;
    else ctx->num_particles_proc = 0;
  }
  else ctx->num_particles_total = ctx->num_particles_proc*ctx->npe;
  if (ctx->npe>1 && ctx->num_particles_total) {
    PetscPrintf(ctx->wComm,"[%D] **** Warning ****, no global point location. multiple processors (%D) not supported\n",ctx->rank,ctx->npe);
  }
  if (!chunkFlag) ctx->chunksize = X3_V_LEN*((ctx->num_particles_proc/80+1)/X3_V_LEN + 1); /* an intelegent message chunk size */
  if (ctx->chunksize<64 && !chunkFlag) ctx->chunksize = 64; /* 4K messages minimum */

  ctx->use_electrons = PETSC_FALSE;
  ierr = PetscOptionsBool("-use_electrons", "Include electrons", "ex3.c", ctx->use_electrons, &ctx->use_electrons, NULL);CHKERRQ(ierr);
  ctx->max_vpar = 30.;
  ierr = PetscOptionsReal("-max_vpar", "Maximum parallel velocity", "ex3.c",ctx->max_vpar,&ctx->max_vpar,NULL);CHKERRQ(ierr);

  ierr = PetscStrcpy(fname,"torus");CHKERRQ(ierr);
  ierr = PetscOptionsString("-run_type", "Type of run (torus/cylinder)", "ex3.c", fname, fname, sizeof(fname)/sizeof(fname[0]), NULL);CHKERRQ(ierr);
  PetscStrcmp("torus",fname,&flg);
  if (flg) { /* Torus/cylinder */
    char      convType[256];
    PetscReal ntheta_total = 4, radius, nmajor_total;
    PetscInt  idx;
    /* hack to get grid expansion factor for torus */
    ierr = PetscOptionsFList("-x3_dm_type","Convert DMPlex to another format","ex3.c",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    nmajor_total = ctx->grid.num_phi_cells;
    if (flg) {
      ierr = PetscOptionsGetInt(NULL,"x3_","-dm_forest_initial_refinement", &idx, &flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "-x3_dm_forest_initial_refinement not found?");
      nmajor_total *= pow(2,idx);  /* plex does not get the curve */
    }
    else {
      ierr = PetscOptionsGetInt(NULL,"x3_","-dm_refine", &idx, &flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "-x3_dm_refinement not found?");
    }
    if (idx<1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "refine must be greater than 0 (Q2 would be legal)");
    ntheta_total *= pow(2,idx);
    /* inflate for corners in plane */
    radius = ctx->grid.radius_minor;
    s_rminor_inflate = 1.02/cos(M_PI/ntheta_total);
    /* inflate for corners around torus */
    radius = ctx->grid.radius_minor + ctx->grid.radius_major;
    /* s_rminor_inflate *= 1.01/cos(.5*ctx->grid.section_phi*M_PI/nmajor_total); */
    s_rminor_inflate *= 1.02*((radius) / cos(.5*ctx->grid.section_phi*M_PI/nmajor_total) - ctx->grid.radius_major) / ctx->grid.radius_minor;
    if (ntheta_total < 10) s_rminor_inflate *= 2.; /* hack to fix, this is not working for one level of refinement */
    if (ctx->grid.radius_minor*s_rminor_inflate >= ctx->grid.radius_major && ctx->grid.section_phi==2) {
      s_rminor_inflate = 0.95*ctx->grid.radius_major/ctx->grid.radius_minor;
    }

    PetscStrcmp("torus",fname,&flg);
    if (flg) ctx->run_type = X3_TORUS;
    else {
      PetscStrcmp("boxtorus",fname,&flg);
      if (flg) ctx->run_type = X3_BOXTORUS;
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Unknown run type %s",fname);
    }
  }
  /* MHD */
  {
    ctx->gamma = 1.4;
    ctx->amach = 2.02;
    ctx->rhor = 3.0;
    ctx->cfl = 0.5;
    ierr = PetscOptionsReal("-gamma","Heat capacity ratio","",ctx->gamma,&ctx->gamma,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-amach","Shock speed (Mach)","",ctx->amach,&ctx->amach,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-rho2","Density right of discontinuity","",ctx->rhor,&ctx->rhor,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-cfl", "CFL factor", "ex3.c", ctx->cfl, &ctx->cfl, NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();

  if (s_debug>0 && ctx->num_particles_total) {
    PetscPrintf(ctx->wComm,"[%D] npe=%D; %D x %D x %D flux tube grid; mpi_send size (chunksize) has %d particles. %s. %s, %s, phi section = %g, inflate = %g\n",ctx->rank,ctx->npe,ctx->grid.np_phi,ctx->grid.np_theta,ctx->grid.np_radius,ctx->chunksize,
#ifdef X3_S_OF_V
                "Use struct of arrays"
#else
                "Use of array structs"
#endif
                , ctx->use_electrons ? "use electrons" : "ions only", ctx->use_bsp ? "BSP communication" : "Non-blocking consensus communication",ctx->grid.section_phi*M_PI,s_rminor_inflate);
  } else {
    PetscPrintf(ctx->wComm,"[%D] npe=%D; phi=%D x theta=%D x r=%D grid; phi section=%g, inflate=%g\n",ctx->rank,ctx->npe,ctx->grid.np_phi,ctx->grid.np_theta,ctx->grid.np_radius,ctx->grid.section_phi*M_PI,s_rminor_inflate);
  }
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "poststep"
PetscErrorCode poststep(TS ts)
{
  PetscErrorCode ierr;
  X3Ctx          *ctx;
  PetscFunctionBegin;
  ierr = TSGetApplicationContext(ts, &ctx);CHKERRQ(ierr);
  if (ctx->num_particles_total) {
    int            irk=0;
    PetscMPIInt    tag;
    PetscReal      dt;
    PetscInt       istep;
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(ctx->events[11],0,0,0,0);CHKERRQ(ierr);
#endif
    /* solve for potential, density being assembled is an invariant */
    ierr = DMPICellSolve( ctx->dmpic );CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(ctx->events[11],0,0,0,0);CHKERRQ(ierr);
#endif
    ierr = PetscCommGetNewTag(ctx->wComm,&tag);CHKERRQ(ierr);
    ierr = TSGetTimeStepNumber(ts,&istep);CHKERRQ(ierr);
    ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
    ierr = processParticles(ctx, dt, &ctx->sendListTable, tag + 2*(X3_NION + 1), irk, istep, PETSC_TRUE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

struct FieldDescription {
  const char *name;
  PetscInt dof;
};
 /* 8 field MHD: r, ru, B, e */
static const struct FieldDescription PhysicsFields_MHD[] = {{"Density",1},{"Momentum",3},{"B",3},{"Energy",1},{NULL,0}};

#undef __FUNCT__
#define __FUNCT__ "SetupDMs"
static PetscErrorCode SetupDMs(X3Ctx *ctx, DM *admmhd, PetscFV *afvm)
{
  PetscErrorCode ierr;
  DM             dmmhd;
  PetscFV        fvm;
  PetscDS        prob,probmhd;
  DMLabel        label;
  DM_PICell      *dmpi;
  PetscInt       dim,i,id=1;
  PetscFunctionBegin;
  /* construct DMs */
  ierr = DMCreate(ctx->wComm, &ctx->dmpic);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(ctx->dmpic, ctx);CHKERRQ(ierr);
  ierr = DMSetType(ctx->dmpic, DMPICELL);CHKERRQ(ierr); /* creates (DM_PICell *) dm->data */
  dmpi = (DM_PICell *) ctx->dmpic->data; assert(dmpi);
  dmpi->debug = s_debug;
  /* setup solver grid */
  if (ctx->run_type == X3_TORUS) {
    ierr = DMPlexCreatePICellTorus(ctx->wComm,&ctx->grid,&dmpi->dm);CHKERRQ(ierr);
    ctx->inflate_torus = PETSC_TRUE;
  }
  else {
    ierr = DMPlexCreatePICellBoxTorus(ctx->wComm,&ctx->grid,&dmpi->dm);CHKERRQ(ierr);
    ctx->inflate_torus = PETSC_FALSE;
  }
  ierr = DMGetDimension(dmpi->dm, &dim);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dmpi->dm, ctx);CHKERRQ(ierr);
  /* mark BCs, how does this work with periodic? */
  ierr = DMCreateLabel(dmpi->dm, "boundary");CHKERRQ(ierr);
  ierr = DMGetLabel(dmpi->dm, "boundary", &label);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dmpi->dm, label);CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(dmpi->dm, label);CHKERRQ(ierr);
  if (s_section_phi != 2) {
    DMBoundaryType bd[3] = {DM_BOUNDARY_NONE,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_NONE};
    const PetscReal L2 = (ctx->grid.radius_major+ctx->grid.radius_minor)*tan(s_section_phi*M_PI);
    const PetscReal L[3] = {0, L2, 0};
    PetscReal maxCell[3];
    for (i = 0; i < 3; i++) maxCell[i] =  1e100;
    maxCell[1] = L2/3;
    if (s_section_phi > .1) {
      ierr = PetscPrintf(ctx->wComm, "\t\t%s WARNING section_phi is large %g, L = %16.8e, %16.8g, %16.8e maxCell = %16.8e, %16.8e, %16.8e\n",
                         __FUNCT__,s_section_phi,L[0],L[1],L[2],maxCell[0],maxCell[1],maxCell[2]);CHKERRQ(ierr);
    }
    /* ierr = DMSetPeriodicity(dmpi->dm, maxCell, L, bd);CHKERRQ(ierr); */
    /* ierr = DMLocalizeCoordinates(dmpi->dm);CHKERRQ(ierr); */
  }
  /* clone DM for MHD with ghosts and setup */
  ierr = DMClone(dmpi->dm, &dmmhd);CHKERRQ(ierr);

  { /* setup MHD DM */
    DM dm2;
    ierr = DMPlexSetAdjacencyUseCone(dmmhd, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMPlexSetAdjacencyUseClosure(dmmhd, PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMPlexDistribute(dmmhd, 1, NULL, &dm2);CHKERRQ(ierr);
    if (dm2) {
      ierr = DMDestroy(&dmmhd);CHKERRQ(ierr);
      dmmhd   = dm2;
    }
    ierr = PetscObjectSetOptionsPrefix((PetscObject) dmmhd, "mhd_");CHKERRQ(ierr);
    ierr = DMSetFromOptions(dmmhd);CHKERRQ(ierr);
    ierr = DMPlexConstructGhostCells(dmmhd, NULL, NULL, &dm2);CHKERRQ(ierr);
    ierr = DMDestroy(&dmmhd);CHKERRQ(ierr);
    dmmhd = dm2;
    /* setup PIC solver */
    ierr = DMPlexDistribute(dmpi->dm, 0, NULL, &dm2);CHKERRQ(ierr);
    if (dm2) {
      ierr = DMDestroy(&dmpi->dm);CHKERRQ(ierr);
      dmpi->dm = dm2;
    }
    ierr = PetscObjectSetOptionsPrefix((PetscObject) dmpi->dm, "x3_");CHKERRQ(ierr);
    ierr = DMSetFromOptions(dmpi->dm);CHKERRQ(ierr);
  }

  ierr = PetscDSCreate(ctx->wComm,&probmhd);CHKERRQ(ierr);
  ierr = PetscFVCreate(ctx->wComm, &fvm);CHKERRQ(ierr);
  ierr = PetscFVSetFromOptions(fvm);CHKERRQ(ierr);
  /* Count number of fields and dofs */
  for (ctx->ndof=0,i=0; PhysicsFields_MHD[i].name; i++) ctx->ndof += PhysicsFields_MHD[i].dof;
  ierr = PetscFVSetNumComponents(fvm, ctx->ndof);CHKERRQ(ierr);
  ierr = PetscFVSetSpatialDimension(fvm, dim);CHKERRQ(ierr);assert(dim==3);
  ierr = PetscObjectSetName((PetscObject) fvm,"");CHKERRQ(ierr);
  /* Count number of fields and dofs */
  for (ctx->ndof=0,ctx->nfields=0; PhysicsFields_MHD[ctx->nfields].name; ctx->nfields++) {
    for (i = 0; i < PhysicsFields_MHD[ctx->nfields].dof ; i++, ctx->ndof++) {
      char compName[256]  = "Unknown";
      ierr = PetscSNPrintf(compName,sizeof(compName),"%s_%d",PhysicsFields_MHD[ctx->nfields].name,i);CHKERRQ(ierr);
      ierr = PetscFVSetComponentName(fvm,ctx->ndof,compName);CHKERRQ(ierr); /* this does not work */
    }
  }
  /* FV is now structured with one field having all physics as components */
  ierr = PetscDSAddDiscretization(probmhd, (PetscObject) fvm);CHKERRQ(ierr);
  ierr = PetscDSSetRiemannSolver(probmhd, 0, PhysicsRiemann_MHD);CHKERRQ(ierr);
  ierr = PetscDSSetContext(probmhd, 0, ctx);CHKERRQ(ierr);
  /* add BCs, how does this work with periodic? */
  ierr = PetscDSAddBoundary(probmhd, PETSC_FALSE, "wall", "boundary", 0, 0, NULL, (void (*)()) PhysicsBoundary_MHD_Wall, 1, &id, ctx);CHKERRQ(ierr);
  /* ierr = PetscObjectSetOptionsPrefix((PetscObject)probmhd, "mhd_");CHKERRQ(ierr); */
  ierr = PetscDSSetFromOptions(probmhd);CHKERRQ(ierr);
  ierr = DMSetDS(dmmhd,probmhd);CHKERRQ(ierr);
  ierr = PetscDSDestroy(&probmhd);CHKERRQ(ierr);
  /* ierr = DMAddBoundary(dmmhd, PETSC_TRUE, "wall", "boundary", 0, 0, NULL, (void (*)()) PhysicsBoundary_MHD_Wall, 1, &id, ctx);CHKERRQ(ierr); */

  /* setup PIC FEM Poison Discretization */
  ierr = PetscFECreateDefault(dmpi->dm, dim, 1, PETSC_FALSE, NULL, PETSC_DECIDE, &dmpi->fem);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dmpi->fem, "potential");CHKERRQ(ierr);
  /* FEM prob */
  ierr = DMGetDS(dmpi->dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) dmpi->fem);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 0, 0, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
  ierr = PetscDSSetContext(prob, 0, ctx);CHKERRQ(ierr);
  /* add BCs, how does this work with periodic? */
  ierr = PetscDSAddBoundary(prob, PETSC_TRUE, "wall", "boundary", 0, 0, NULL, (void (*)()) zero, 1, &id, ctx);CHKERRQ(ierr);
  /* ierr = PetscObjectSetOptionsPrefix((PetscObject)prob, "poisson_");CHKERRQ(ierr); */
  ierr = PetscDSSetFromOptions(prob);CHKERRQ(ierr);
  { /* convert to plex */
    char       convType[256];
    PetscBool  flg;
    DM         dm2;
    const char *prefix;
    PetscBool  isForest;
    ierr = PetscOptionsBegin(ctx->wComm, "", "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-x3_dm_type","Convert DMPlex to another format (should not be Plex!)","ex3.c",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
    if (flg && ctx->num_particles_total) { /* convert (to p4est) */
      /* convert PIC */
      ierr = DMConvert(dmpi->dm,convType,&dm2);CHKERRQ(ierr);
      if (dm2) {
        ierr = PetscObjectGetOptionsPrefix((PetscObject)dmpi->dm,&prefix);CHKERRQ(ierr);
        ierr = PetscObjectSetOptionsPrefix((PetscObject)dm2,prefix);CHKERRQ(ierr);
        ierr = DMDestroy(&dmpi->dm);CHKERRQ(ierr);
        dmpi->dm = dm2;
        ierr = DMSetFromOptions(dmpi->dm);CHKERRQ(ierr);
        ierr = DMIsForest(dmpi->dm,&isForest);CHKERRQ(ierr);
        if (isForest && ctx->run_type == X3_TORUS) {
          ierr = DMForestSetBaseCoordinateMapping(dmpi->dm,GeometryPICellTorus,ctx);CHKERRQ(ierr);
        } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Converted to non Forest?");
      } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Convert failed?");
    }
    if ( flg ) {
      PetscPrintf(PETSC_COMM_WORLD,"%s: converting to Forest\n",__FUNCT__);
      ierr = DMConvert(dmmhd,convType,&dm2);CHKERRQ(ierr);
      if (dm2) {
        /* ierr = PetscObjectGetOptionsPrefix((PetscObject)dmmhd,&prefix);CHKERRQ(ierr); */
        /* ierr = PetscObjectSetOptionsPrefix((PetscObject)dm2,prefix);CHKERRQ(ierr); */
        ierr = DMDestroy(&dmmhd);CHKERRQ(ierr);
        dmmhd = dm2;
        ierr = DMSetFromOptions(dmmhd);CHKERRQ(ierr);
        ierr = DMIsForest(dmmhd,&isForest);CHKERRQ(ierr);
        if (isForest && ctx->run_type == X3_TORUS) {
          ierr = DMForestSetBaseCoordinateMapping(dmmhd,GeometryPICellTorus,ctx);CHKERRQ(ierr);
        } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Converted to non Forest?");
      } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Convert failed?");
    }
  }

  { /* print */
    PetscInt n,cStart,cEnd,bs;
    Vec X;
    ierr = DMCreateGlobalVector(dmmhd, &X);CHKERRQ(ierr);
    ierr = VecGetSize(X,&n);CHKERRQ(ierr);
    ierr = VecGetBlockSize(X,&bs);CHKERRQ(ierr);
    ierr = VecDestroy(&X);CHKERRQ(ierr);
    if (!n) SETERRQ(ctx->wComm, PETSC_ERR_USER, "No dofs");
    ierr = DMPlexGetHeightStratum(dmpi->dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    if (cStart) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER, "cStart != 0. %D",cStart);
    if (dmpi->debug>0 && !cEnd) {
      ierr = PetscPrintf((dmpi->debug>1 || !cEnd) ? PETSC_COMM_SELF : ctx->wComm,"[%D] ERROR %D global equations, %d local cells, (cEnd=%d), debug=%D\n",ctx->rank,n,cEnd-cStart,cEnd,dmpi->debug);
    }
    if (!cEnd) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "No cells");
    }
    s_fluxtubeelem = cEnd/2;
    if (dmpi->debug>0) PetscPrintf( ctx->wComm,"[%D] %D equations (block size %D) on %D processors, %D local cells\n",
                                    ctx->rank,n,bs,ctx->npe,cEnd);
  }
  if (dmpi->debug>2) {
    ierr = DMView(dmpi->dm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = DMView(dmmhd,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  *admmhd = dmmhd;
  *afvm = fvm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  X3Ctx          actx,*ctx=&actx; /* work context */
  PetscErrorCode ierr;
  DM_PICell      *dmpi = NULL;
  PetscInt       idx,isp,nsteps,adaptInterval=1;
  Mat            J;
  TS             ts;
  DM             dmmhd;
  PetscFV        fvm;
  PetscReal      ftime,maxspeed,minRadius;
  Vec            X;
  PetscLimiter   limiter = NULL, noneLimiter = NULL;
  TSConvergedReason reason;
  PetscFunctionBeginUser;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(ctx->events[0],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = PetscCommDuplicate(PETSC_COMM_WORLD,&ctx->wComm,NULL);CHKERRQ(ierr);
  ierr = ProcessOptions( ctx );CHKERRQ(ierr);
  ierr = SetupDMs( ctx, &dmmhd, &fvm );CHKERRQ(ierr);
  dmpi = (DM_PICell *) ctx->dmpic->data; assert(dmpi);

  /* create Poisson SNES, should put this and destroy in PICell */
  if (ctx->num_particles_total) {
    ierr = SNESCreate(ctx->wComm, &dmpi->snes);CHKERRQ(ierr);
    ierr = SNESSetDM(dmpi->snes, dmpi->dm);CHKERRQ(ierr);
    ierr = DMPlexSetSNESLocalFEM(dmpi->dm,ctx,ctx,ctx);CHKERRQ(ierr);
    ierr = DMCreateMatrix(dmpi->dm, &J);CHKERRQ(ierr);
    ierr = SNESSetJacobian(dmpi->snes, J, J, NULL, NULL);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(dmpi->snes);CHKERRQ(ierr);
  }

  /* setup particles, if used */
  ierr = createParticles(ctx);CHKERRQ(ierr);

  if (ctx->plot && ctx->num_particles_total) { /* plot initial particle distribution (flux tubes) */
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(ctx->events[diag_event_id],0,0,0,0);CHKERRQ(ierr);
#endif
    /* hdf5 output - init */
#ifdef H5PART
    for (isp=ctx->use_electrons ? 0 : 1 ; isp <= X3_NION ; isp++) { // for each species
      char  fname1[256],fname2[256];
      X3PListPos pos1,pos2;
      if (!isp) {
        sprintf(fname1,         "particles_electrons_time%05d_fluxtube.h5part",(int)0);
        sprintf(fname2,"sub_rank_particles_electrons_time%05d_fluxtube.h5part",(int)0);
      } else {
        sprintf(fname1,         "particles_sp%d_time%05d_fluxtube.h5part",(int)isp,0);
        sprintf(fname2,"sub_rank_particles_sp%d_time%05d_fluxtube.h5part",(int)isp,0);
      }
      /* write */
      prewrite(ctx, &ctx->partlists[isp][s_fluxtubeelem], &pos1, &pos2);
      ierr = X3PListWrite(ctx->partlists[isp], ctx->nElems, ctx->rank, ctx->npe, ctx->wComm, fname1, fname2);CHKERRQ(ierr);
      postwrite(ctx, &ctx->partlists[isp][s_fluxtubeelem], &pos1, &pos2);
    }
#endif
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(ctx->events[diag_event_id],0,0,0,0);CHKERRQ(ierr);
#endif
  }

  /* initialize TS */
  ierr = initializeTS(dmmhd, ctx, &ts);CHKERRQ(ierr);

  /* setup solver, dummy solve to really setup */
  if (ctx->num_particles_total) {
    ierr = VecZeroEntries(dmpi->rho);CHKERRQ(ierr); /* zero density to make solver do nothing */
    ierr = DMPICellSolve( ctx->dmpic );CHKERRQ(ierr);
    /* move back to solver space and make density vector */
    ierr = processParticles(ctx, 0.0, &ctx->sendListTable, 99, -1, -1, PETSC_TRUE);CHKERRQ(ierr);
    /* setup particle pushing after each TS step */
    ierr = TSSetPostStep(ts, poststep);CHKERRQ(ierr);
  }

  ierr = DMCreateGlobalVector(dmmhd, &X);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) X, "mhd");CHKERRQ(ierr);
  ierr = SetInitialCondition(dmmhd, X, ctx);CHKERRQ(ierr);
  if (ctx->use_amr) {
    PetscInt adaptIter;
    /* use no limiting when reconstructing gradients for adaptivity */
    ierr = PetscFVGetLimiter(fvm, &limiter);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)limiter);CHKERRQ(ierr);

    ierr = PetscLimiterCreate(PetscObjectComm((PetscObject)fvm),&noneLimiter);CHKERRQ(ierr);
    ierr = PetscLimiterSetType(noneLimiter,PETSCLIMITERNONE);CHKERRQ(ierr);

    ierr = PetscFVSetLimiter(fvm,noneLimiter);CHKERRQ(ierr);
    for (adaptIter = 0;;adaptIter++) {
      PetscLogDouble bytes;
      TS             tsNew = NULL;
      if (ctx->plot_amr_initial) {
        PetscViewer viewer;
        char        buf[256];
        PetscBool   isHDF5;

        ierr = PetscViewerCreate(ctx->wComm,&viewer);CHKERRQ(ierr);
        ierr = PetscViewerSetType(viewer,PETSCVIEWERHDF5);CHKERRQ(ierr);
        ierr = PetscViewerSetOptionsPrefix(viewer,"initial_");CHKERRQ(ierr);
        ierr = PetscViewerSetFromOptions(viewer);CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&isHDF5);CHKERRQ(ierr);
        if (isHDF5) {
          ierr = PetscSNPrintf(buf, 256, "ex3-initial-%d.h5", adaptIter);CHKERRQ(ierr);
        }
        ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
        ierr = PetscViewerFileSetName(viewer,buf);CHKERRQ(ierr);
        if (isHDF5) {
          ierr = DMView(dmmhd,viewer);CHKERRQ(ierr);
          ierr = PetscViewerFileSetMode(viewer,FILE_MODE_UPDATE);CHKERRQ(ierr);
        }
        ierr = VecView(X,viewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      }

      ierr = PetscMemoryGetCurrentUsage(&bytes);CHKERRQ(ierr);
      ierr = PetscInfo2(ts, "refinement loop %D: memory used %g\n", adaptIter, bytes);CHKERRQ(ierr);
      ierr = adaptToleranceFVM(fvm, ts, X, ctx->refine_tol, ctx->coarsen_tol, ctx, &tsNew, NULL);CHKERRQ(ierr);
      if (!tsNew) {
        break;
      } else {
        ierr = DMDestroy(&dmmhd);CHKERRQ(ierr);
        ierr = VecDestroy(&X);CHKERRQ(ierr);
        ierr = TSDestroy(&ts);CHKERRQ(ierr);
        ts   = tsNew;
        ierr = TSGetDM(ts,&dmmhd);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)dmmhd);CHKERRQ(ierr);
        ierr = DMCreateGlobalVector(dmmhd,&X);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject) X, "mhd");CHKERRQ(ierr);
        ierr = SetInitialCondition(dmmhd, X, ctx);CHKERRQ(ierr);
      }
    }
    /* restore original limiter */
    ierr = PetscFVSetLimiter(fvm,limiter);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(ctx->events[0],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
#endif

  /* collect max maxspeed from all processes */
  ierr = DMPlexTSGetGeometryFVM(dmmhd, NULL, NULL, &minRadius);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&ctx->maxspeed,&maxspeed,1,MPIU_REAL,MPI_MAX,ctx->wComm);CHKERRQ(ierr);
  if (maxspeed <= 0) SETERRQ(ctx->wComm,PETSC_ERR_ARG_WRONGSTATE,"Physics did not set maxspeed");
  ctx->maxspeed = maxspeed;
  if (ctx->dt==0) ctx->dt = ctx->cfl * minRadius / ctx->maxspeed;
  ierr = TSSetInitialTimeStep(ts,0.0,ctx->dt);CHKERRQ(ierr);
  if (ctx->dt == ctx->cfl * minRadius / ctx->maxspeed) ctx->dt = 0;
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  if (!ctx->use_amr) {
    ierr = TSSolve(ts,X);CHKERRQ(ierr);
    ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
    ierr = TSGetTimeStepNumber(ts,&nsteps);CHKERRQ(ierr);
  } else {
    PetscReal finalTime;
    PetscInt  adaptIter;
    TS        tsNew = NULL;
    Vec       solNew = NULL;
    PetscInt  incSteps;
    ierr   = TSGetDuration(ts,NULL,&finalTime);CHKERRQ(ierr);
    ierr   = TSSetDuration(ts,adaptInterval,finalTime);CHKERRQ(ierr);
    ierr   = TSSolve(ts,X);CHKERRQ(ierr);
    ierr   = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
    ierr   = TSGetTimeStepNumber(ts,&nsteps);CHKERRQ(ierr);
    for (adaptIter = 0;ftime < finalTime;adaptIter++) {
      PetscLogDouble bytes;

      ierr = PetscMemoryGetCurrentUsage(&bytes);CHKERRQ(ierr);
      ierr = PetscInfo2(ts, "AMR time step loop %D: memory used %g\n", adaptIter, bytes);CHKERRQ(ierr);
      ierr = PetscFVSetLimiter(fvm,noneLimiter);CHKERRQ(ierr);
      ierr = adaptToleranceFVM(fvm,ts,X,ctx->refine_tol, ctx->coarsen_tol,ctx,&tsNew,&solNew);CHKERRQ(ierr);
      ierr = PetscFVSetLimiter(fvm,limiter);CHKERRQ(ierr);
      if (tsNew) {
        ierr = PetscInfo(ts, "AMR used\n");CHKERRQ(ierr);
        ierr = DMDestroy(&dmmhd);CHKERRQ(ierr);
        ierr = VecDestroy(&X);CHKERRQ(ierr);
        ierr = TSDestroy(&ts);CHKERRQ(ierr);
        ts   = tsNew;
        X    = solNew;
        ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
        ierr = VecGetDM(X,&dmmhd);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)dmmhd);CHKERRQ(ierr);
        ierr = DMPlexTSGetGeometryFVM(dmmhd, NULL, NULL, &minRadius);CHKERRQ(ierr);
        ierr = MPI_Allreduce(&ctx->maxspeed,&maxspeed,1,MPIU_REAL,MPI_MAX,ctx->wComm);CHKERRQ(ierr);
        ctx->maxspeed = maxspeed;
        if (ctx->dt==0) ctx->dt = ctx->cfl * minRadius / maxspeed;
        ierr = TSSetInitialTimeStep(ts,ftime,ctx->dt);CHKERRQ(ierr);
        if (ctx->dt == ctx->cfl * minRadius / maxspeed) ctx->dt = 0;
      }
      else {
        ierr = PetscInfo(ts, "AMR not used\n");CHKERRQ(ierr);
      }
      ierr    = TSSetDuration(ts,adaptInterval,finalTime);CHKERRQ(ierr);
      ierr    = TSSolve(ts,X);CHKERRQ(ierr);
      ierr    = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
      ierr    = TSGetTimeStepNumber(ts,&incSteps);CHKERRQ(ierr);
      nsteps += incSteps;
    }
  }
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %g after %D steps\n",TSConvergedReasons[reason],(double)ftime,nsteps);CHKERRQ(ierr);

  if (ctx->plot) {
    PetscViewer       viewer = NULL;
    PetscBool         flg;
    PetscViewerFormat fmt;
    DM_PICell      *dmpi = (DM_PICell *) ctx->dmpic->data;
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(ctx->events[diag_event_id],0,0,0,0);CHKERRQ(ierr);
#endif
    if (ctx->num_particles_total) {
      ierr = DMViewFromOptions(dmpi->dm,NULL,"-dm_view");CHKERRQ(ierr);
      ierr = PetscOptionsGetViewer(ctx->wComm,NULL,"-x3_vec_view",&viewer,&fmt,&flg);CHKERRQ(ierr);
      if (flg) {
        ierr = PetscViewerPushFormat(viewer,fmt);CHKERRQ(ierr);
        ierr = VecView(dmpi->phi,viewer);CHKERRQ(ierr);
        ierr = VecView(dmpi->rho,viewer);CHKERRQ(ierr);
        ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    ierr = DMViewFromOptions(dmmhd,NULL,"-dm_view");CHKERRQ(ierr);
    ierr = PetscOptionsGetViewer(ctx->wComm,"mhd_","-vec_view",&viewer,&fmt,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerPushFormat(viewer,fmt);CHKERRQ(ierr);
      ierr = VecView(X,viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(ctx->events[diag_event_id],0,0,0,0);CHKERRQ(ierr);
#endif
  }
  if (dmpi->debug>0) PetscPrintf(ctx->wComm,"[%D] done - cleanup\n",ctx->rank);
  /* Cleanup */
  if (ctx->num_particles_total) {
    for (idx=0;idx<ctx->proc_send_table_size;idx++) {
      if (ctx->sendListTable[idx].data_size != 0) {
        ierr = X3PSendListDestroy( &ctx->sendListTable[idx] );CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(ctx->sendListTable);CHKERRQ(ierr);
    ierr = destroyParticles(ctx);CHKERRQ(ierr);
    ierr = MatDestroy(&J);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = PetscLimiterDestroy(&limiter);CHKERRQ(ierr);
  ierr = PetscLimiterDestroy(&noneLimiter);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&dmmhd);CHKERRQ(ierr);
  ierr = PetscFVDestroy(&fvm);CHKERRQ(ierr);
  ierr = SNESDestroy(&dmpi->snes);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&dmpi->fem);CHKERRQ(ierr);
  ierr = DMDestroy(&ctx->dmpic);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&ctx->wComm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

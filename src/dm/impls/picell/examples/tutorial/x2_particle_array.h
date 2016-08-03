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
#define X2_V_LEN 32
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
#ifdef X2_S_OF_V
#define X2PAXPY(a,s,d,i) { d.r += a*s->data_v.r[i]; d.z += a*s->data_v.z[i]; d.phi += a*s->data_v.phi[i]; d.vpar += a*s->data_v.vpar[i]; d.mu += a*s->data_v.mu[i]; d.w0 += a*s->data_v.w0[i]; d.f0 += a*s->data_v.f0[i]; d.gid += s->data_v.gid[i];}
#else
#define X2PAXPY(a,s,d,i) { d.r += a*s->data[i].r;   d.z += a*s->data[i].z;   d.phi += a*s->data[i].phi;   d.vpar += a*s->data[i].vpar;   d.mu += a*s->data[i].mu;   d.w0 += a*s->data[i].w0 ;  d.f0 += a*s->data[i].f0;   d.gid += s[i].gid;}
#endif

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
  if (!l->data_size) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"List not created?");
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
    if (l->hole != -1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"rehash and l->hole != -1, size %d",l->size);
    assert(l->hole == -1);
  }
  if (l->hole != -1) { /* have a hole - fill it */
    X2PListPos idx = l->hole; assert(idx<l->data_size);
#ifdef X2_S_OF_V
    if (l->data_v.gid[idx] == 0) l->hole = -1; /* filled last hole */
    else if (l->data_v.gid[idx]>=0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"hole with non-neg gid %d",l->data_v.gid[idx]);
    else l->hole = (X2PListPos)(-l->data_v.gid[idx] - 1); /* use gid as pointer */
    X2P2V(p,l->data_v,idx);
#else
    if (l->data[idx].gid == 0) l->hole = -1; /* filled last hole */
    else if (l->data[idx].gid>=0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"hole with non-neg gid %d",l->data[idx].gid);
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
  PetscErrorCode ierr;
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(s_events[13],0,0,0,0);CHKERRQ(ierr);
#endif
  /* fill holes with end of list */
  for ( ii = 0 ; ii < l->top && l->top > l->size ; ii++) {
#ifdef X2_S_OF_V
    if (l->data_v.gid[ii] <= 0)
#else
    if (l->data[ii].gid <= 0)
#endif
    {
      l->top--; /* index of data at end to move to hole */
      if (ii == l->top) /* just pop hole at end */ ;
      else {
#ifdef X2_S_OF_V
        while (l->data_v.gid[l->top] <= 0)  l->top--; /* get real data */
        if (l->top > ii) X2V2V(l->data_v,l->data_v,l->top,ii);
#else
        while (l->data[l->top].gid <= 0)  l->top--; /* get real data */
        if (l->top > ii) l->data[ii] = l->data[l->top]; /* now above */
#endif
      }
    }
  }
  l->hole = -1;
  l->top = l->size;
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(s_events[13],0,0,0,0);CHKERRQ(ierr);
#endif
#ifdef PETSC_USE_DEBUG
  for (ii=0;ii<l->size;ii++) assert(l->data_v.gid[ii]>0);
#endif
  return 0;
}

/* keep list of hols after removal so that we can iterate over a list and remove as we go
  gid < 0 : hole : -gid - 1 == -(gid+1) is next index
  gid = 0 : sentinal
  gid > 0 : real
*/
#undef __FUNCT__
#define __FUNCT__ "X2PListGetHead"
PetscErrorCode X2PListGetHead(X2PList *l, X2Particle *p, X2PListPos *pos)
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
#ifdef X2_S_OF_V
    while (l->data_v.gid[*pos] <= 0) (*pos)++;
    X2V2P(p,l->data_v,*pos); /* return copy */
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
#define __FUNCT__ "X2PListGetNext"
PetscErrorCode X2PListGetNext(X2PList *l, X2Particle *p, X2PListPos *pos)
{
  PetscFunctionBeginUser;
  /* l->size == 0 can happen on empty list */
  (*pos)++; /* get next position */
  if (*pos >= l->data_size || *pos >= l->top) PetscFunctionReturn(1); /* hit end, can go past if list is just drained */
#ifdef X2_S_OF_V
  while(*pos < l->top && l->data_v.gid[*pos] <= 0) (*pos)++; /* skip holes */
#else
  while(*pos < l->top && l->data[*pos].gid <= 0) (*pos)++; /* skip holes */
#endif
  assert(*pos<=l->top); assert(l->top<=l->data_size);
  if (*pos==l->top) PetscFunctionReturn(1); /* hit end with holes */
#ifdef X2_S_OF_V
  X2V2P(p,l->data_v,*pos); /* return copy */
#else
  *p = l->data[*pos]; /* return copy */
#endif
  assert(p->gid>0);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "X2PListRemoveAt"
PetscErrorCode X2PListRemoveAt( X2PList *l, const X2PListPos pos)
{
  PetscFunctionBeginUser;
#ifdef PETSC_USE_DEBUG
  if(pos >= l->data_size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"X2PListRemoveAt past end of data %d %d",pos,l->data_size);
  if(pos >= l->top) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"X2PListRemoveAt past end of top pointer %d %d",pos,l->top);
#endif
  if (l->hole==-1 && pos==l->top-1 && 0) l->top--; /* just pop */
  else {
#ifdef X2_S_OF_V
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
  assert(l->top >= l->size);
#ifdef X2_S_OF_V
  l->data_v.w0[pos] = 0; /* zero to add in vector */
#else
  l->data[pos].w0 = 0;
#endif
  PetscFunctionReturn(0);
}

PetscInt X2PListMaxSize(X2PList *l) {
  return l->data_size;
}

PetscInt X2PListSize(X2PList *l) {
  return l->size;
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
      ierr = X2PListGetHead( &l[elid], &part, &pos );
      if (!ierr) {
        do {
          if (part.gid > 0) {
            x[nparticles] = part.r*cos(part.phi);
            y[nparticles] = part.r*sin(part.phi);
            z[nparticles] = part.z;
            v[nparticles] = part.vpar;
            id[nparticles] = part.gid;
            nparticles++;
          }
        } while ( !X2PListGetNext( &l[elid], &part, &pos) );
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
        ierr = X2PListGetHead( &l[elid], &part, &pos );
        if (!ierr) {
          do {
            if (part.gid > 0) {
              x[nparticles] = part.r*cos(part.phi);
              y[nparticles] = part.r*sin(part.phi);
              z[nparticles] = part.z;
              v[nparticles] = part.vpar;
              id[nparticles] = rank;
              nparticles++;
            }
          } while ( !X2PListGetNext( &l[elid], &part, &pos) );
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
  l->size = 0;
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

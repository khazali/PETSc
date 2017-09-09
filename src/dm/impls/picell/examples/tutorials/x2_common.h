/* M. Adams, August 2016 */

#include <petscviewerhdf5.h>

PETSC_STATIC_INLINE PetscErrorCode zero(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
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
  for (isp = ctx->use_electrons ? 0 : 1 ; isp <= X2_NION ; isp++ ) { // for each species
    for (elid=0;elid<ctx->nElems;elid++) {
      ierr = X2PListDestroy(&ctx->partlists[isp][elid]);CHKERRQ(ierr);
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
PetscErrorCode shiftParticles( const X2Ctx *ctx, X2PSendList *sendListTable, X2PList particlelist[],
                               PetscInt *const nIsend, const PetscInt slist_size, X2ISend slist[], PetscMPIInt tag, PetscBool solver)
{
  PetscErrorCode ierr;
  const int part_dsize = sizeof(X2Particle)/sizeof(PetscReal); assert(sizeof(X2Particle)%sizeof(PetscReal)==0);
  PetscInt ii,jj,kk,mm,idx;
  DM_PICell *dmpi;
  MPI_Datatype real_type;

  PetscFunctionBeginUser;
  PetscDataTypeToMPIDataType(PETSC_REAL,&real_type);
  dmpi = (DM_PICell *) ctx->dm->data;
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(ctx->events[2],0,0,0,0);CHKERRQ(ierr);
#endif
  if ( ctx->use_bsp ) { /* use BSP */
    PetscMPIInt  nto,*fromranks;
    PetscMPIInt *toranks;
    X2Particle  *fromdata,*todata,*pp;
    PetscMPIInt  nfrom,pe;
    int sz;
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(ctx->events[4],0,0,0,0);CHKERRQ(ierr);
#endif
    /* count send  */
    for (ii=0,nto=0;ii<ctx->proc_send_table_size;ii++) {
      if (sendListTable[ii].data_size != 0) {
	sz = X2PSendListSize(&sendListTable[ii]);
	for (jj=0 ; jj<sz ; jj += ctx->chunksize) nto++; /* can just figure this out */
      }
    }
    /* make to ranks & data */
    ierr = PetscMalloc1(nto, &toranks);CHKERRQ(ierr);
    ierr = PetscMalloc1(ctx->chunksize*nto, &todata);CHKERRQ(ierr);
    for (ii=0,nto=0,pp=todata;ii<ctx->proc_send_table_size;ii++) {
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
            ierr = X2GridSolverLocatePoints(dmpi->dm, vec, ctx, &pes, &elems);CHKERRQ(ierr);
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
	  ierr = X2PListAdd( &particlelist[elid], pp, NULL);CHKERRQ(ierr);
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
      if (sendListTable[ii].data_size != 0 && (sz=X2PSendListSize(&sendListTable[ii])) > 0) {
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
	ierr = X2PSendListClear( &sendListTable[ii] );CHKERRQ(ierr);
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
          X2Particle *data;
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
              ierr = X2GridSolverLocatePoints(dmpi->dm, vec, ctx, &pes, &elems);CHKERRQ(ierr);
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
            ierr = X2PListAdd( &particlelist[elid], &data[jj], NULL);CHKERRQ(ierr);
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

#define x2_coef(x) (1.0)

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  PetscScalar coef = x2_coef(x);
  for (d = 0; d < dim; ++d) g3[d*dim+d] = coef;
}
void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
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
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static PetscErrorCode processParticles( X2Ctx *ctx, const PetscReal dt, X2PSendList **sendListTable, const PetscMPIInt tag,
                                        const int irk, const int istep, PetscBool solver);

#undef __FUNCT__
#define __FUNCT__ "go"
PetscErrorCode go( X2Ctx *ctx )
{
  PetscErrorCode ierr;
  int            istep;
  PetscMPIInt    tag;
  PetscReal      time,dt;
  PetscFunctionBeginUser;
  /* main time step loop */
  ierr = PetscCommGetNewTag(ctx->wComm,&tag);CHKERRQ(ierr);
  for ( istep=0, time=0.;
	istep < ctx->msteps && time < ctx->maxTime;
	istep++, time += ctx->dt, tag += 3*(X2_NION + 1) ) {
    int irk;
    /* do collisions */
    if (((istep+1)%ctx->collision_period)==0) {
      /* move to flux tube space */
      ierr = processParticles(ctx, 0.0, &ctx->sendListTable, tag, -1, istep, PETSC_FALSE);CHKERRQ(ierr);
      /* move back to solver space, and deposit (deposit not needed, should add a flag, could check for no change) */
      ierr = processParticles(ctx, 0.0, &ctx->sendListTable, tag + X2_NION + 1, -1, istep, PETSC_TRUE);CHKERRQ(ierr);
    }
    /* crude TS */
    dt = ctx->dt;
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(ctx->events[11],0,0,0,0);CHKERRQ(ierr);
#endif
    /* solve for potential, density being assembled is an invariant */
    ierr = DMPICellSolve( ctx->dm );CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(ctx->events[11],0,0,0,0);CHKERRQ(ierr);
#endif
    /* process particles: push, move */
    irk=0;
    ierr = processParticles(ctx, dt, &ctx->sendListTable, tag + 2*(X2_NION + 1), irk, istep, PETSC_TRUE);CHKERRQ(ierr);

    if (ctx->plot) {
      PetscViewer       viewer = NULL;
      PetscBool         flg;
      PetscViewerFormat fmt;
      DM_PICell         *dmpi = (DM_PICell *) ctx->dm->data;
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventBegin(ctx->events[14],0,0,0,0);CHKERRQ(ierr);
#endif
      //ierr = DMViewFromOptions(dmpi->dm,NULL,"-dm_view");CHKERRQ(ierr);
      ierr = DMViewFromOptions(ctx->dm,NULL,"-dm_view");CHKERRQ(ierr);
      ierr = PetscOptionsGetViewer(ctx->wComm,NULL,"-x2_vec_view",&viewer,&fmt,&flg);CHKERRQ(ierr);
      if (flg) {
        ierr = PetscViewerPushFormat(viewer,fmt);CHKERRQ(ierr);
        ierr = VecView(dmpi->phi,viewer);CHKERRQ(ierr);
        ierr = VecView(dmpi->rho,viewer);CHKERRQ(ierr);
        ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
      ierr = PetscLogEventEnd(ctx->events[14],0,0,0,0);CHKERRQ(ierr);
#endif
    }
  } /* time step */

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMView_PICell_part_private"
PetscErrorCode DMView_PICell_part_private(DM dm, PetscViewer viewer)
{
  DM_PICell        *dmpi = (DM_PICell *) dm->data;
  PetscInt          low;
  PetscInt          ii,elid,nloc,isp;
  char              vecname[] = "xcoord", pre[] = "xyzvi";
  PetscErrorCode    ierr;
  X2Ctx             *ctx;
#define X_NCOORD 4
  Vec               vec[X_NCOORD+1];
  PetscScalar       *xarr[X_NCOORD+1];
  PetscFunctionBegin;
  /* count particles */
  ctx = (X2Ctx*)dmpi->data; assert(ctx);
  for (isp=ctx->use_electrons ? 0 : 1, nloc = 0 ; isp <= X2_NION ; isp++) {
    for (elid=0;elid<ctx->nElems;elid++) {
      nloc += X2PListSize(&ctx->partlists[isp][elid]);
    }
    break; /* debug, do one species only!!! */
  }
  /* Select hyperslab in the file */
  for (ii=0;ii<X_NCOORD+1;ii++) {
    ierr = VecCreate(PetscObjectComm((PetscObject) dm),&vec[ii]);CHKERRQ(ierr);
    ierr = VecSetSizes(vec[ii],nloc,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(vec[ii]);CHKERRQ(ierr); /* should we call this? */
  }
  ierr = VecGetOwnershipRange(vec[0], &low, NULL);CHKERRQ(ierr);
  for (ii=0;ii<X_NCOORD+1;ii++) {
    ierr = VecGetArray(vec[ii], &xarr[ii]);CHKERRQ(ierr);
  }
  /* To write dataset independently use H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT) */
  for (isp=ctx->use_electrons ? 0 : 1, nloc = 0 ; isp <= X2_NION ; isp++) {
    X2PList     *l = ctx->partlists[isp];
    X2Particle  part;
    X2PListPos  pos;
    for (nloc=0,elid=0;elid<ctx->nElems;elid++) {
      ierr = X2PListGetHead( &l[elid], &part, &pos );
      if (!ierr) {
        do {
          if (part.gid > 0) {
            PetscReal xx[3];
            cylindricalToCart(part.r, part.z, part.phi, xx);
            xarr[0][nloc] = xx[0];
            xarr[1][nloc] = xx[1];
            xarr[2][nloc] = xx[2];
            xarr[3][nloc] = part.vpar;
            /* add more vel coords to index X_NCOORD-1 */
            xarr[X_NCOORD][nloc] = (PetscScalar)part.gid;
            nloc++;
          }
          else assert(0);
        } while ( !X2PListGetNext( &l[elid], &part, &pos) );
      }
    }
    break; /* debug, do one species only!!! */
  }
  for (ii=0;ii<X_NCOORD+1;ii++) {
    ierr = VecRestoreArray(vec[ii], &xarr[ii]);CHKERRQ(ierr);
    vecname[0] = pre[ii];
    ierr = PetscObjectSetName((PetscObject) vec[ii], vecname);CHKERRQ(ierr);
PetscPrintf(PETSC_COMM_WORLD,"%s: call VecView with %s\n",__FUNCT__,vecname);
    ierr = VecView(vec[ii],viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&vec[ii]);CHKERRQ(ierr);
  }
  ierr   = PetscInfo4(dm,"Wrote DMPICell object with name %s %s %s %s ...\n",vecname[0],vecname[1],vecname[2],vecname[3]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

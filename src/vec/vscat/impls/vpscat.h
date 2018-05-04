
/*
     Defines the methods VecScatterBegin/End_1,2,......
     This is included by vpscat.c with different values for BS

     This is a terrible way of doing "templates" in C.
*/
#define PETSCMAP1_a(a,b)  a ## _ ## b
#define PETSCMAP1_b(a,b)  PETSCMAP1_a(a,b)
#define PETSCMAP1(a)      PETSCMAP1_b(a,BS)

PetscErrorCode PETSCMAP1(VecScatterBegin)(VecScatter ctx,Vec xin,Vec yin,InsertMode addv,ScatterMode mode)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *xv,*yv,*svalues;
  MPI_Request            *rwaits,*swaits;
  PetscErrorCode         ierr;
  PetscInt               i,*indices,*sstarts,nrecvs,nsends,bs;
#if defined(PETSC_HAVE_VIENNACL)
  PetscBool              is_viennacltype = PETSC_FALSE;
#endif

  PetscFunctionBegin;
  if (mode & SCATTER_REVERSE) {
    to     = (VecScatter_MPI_General*)ctx->fromdata;
    from   = (VecScatter_MPI_General*)ctx->todata;
    rwaits = from->rev_requests;
    swaits = to->rev_requests;
  } else {
    to     = (VecScatter_MPI_General*)ctx->todata;
    from   = (VecScatter_MPI_General*)ctx->fromdata;
    rwaits = from->requests;
    swaits = to->requests;
  }
  bs      = to->bs;
  svalues = to->values;
  nrecvs  = from->n;
  nsends  = to->n;
  indices = to->indices;
  sstarts = to->starts;
#if defined(PETSC_HAVE_VIENNACL)
  ierr = PetscObjectTypeCompareAny((PetscObject)xin,&is_viennacltype,VECSEQVIENNACL,VECMPIVIENNACL,VECVIENNACL,"");CHKERRQ(ierr);
  if (is_viennacltype) {
    ierr = VecGetArrayRead(xin,(const PetscScalar**)&xv);CHKERRQ(ierr);
  } else
#endif
#if defined(PETSC_HAVE_VECCUDA)
  {
    VecCUDAAllocateCheckHost(xin);
    if (xin->valid_GPU_array == PETSC_OFFLOAD_GPU) {
      if (xin->spptr && ctx->spptr) {
        ierr = VecCUDACopyFromGPUSome_Public(xin,(PetscCUDAIndices)ctx->spptr);CHKERRQ(ierr);
      } else {
        ierr = VecCUDACopyFromGPU(xin);CHKERRQ(ierr);
      }
    }
    xv = *((PetscScalar**)xin->data);
  }
#else
  {
    ierr = VecGetArrayRead(xin,(const PetscScalar**)&xv);CHKERRQ(ierr);
  }
#endif

  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);}
  else yv = xv;

  if (!(mode & SCATTER_LOCAL)) {
    if (!from->use_readyreceiver && !to->sendfirst && !to->use_alltoallv && !to->use_window) {
      /* post receives since they were not previously posted    */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }

#if defined(PETSC_HAVE_MPI_ALLTOALLW)  && !defined(PETSC_USE_64BIT_INDICES)
    if (to->use_alltoallw && addv == INSERT_VALUES) {
      ierr = MPI_Alltoallw(xv,to->wcounts,to->wdispls,to->types,yv,from->wcounts,from->wdispls,from->types,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
    } else
#endif
    if (ctx->packtogether || to->use_alltoallv || to->use_window) {
      /* this version packs all the messages together and sends, when -vecscatter_packtogether used */
      PETSCMAP1(Pack)(sstarts[nsends],indices,xv,svalues,bs);
      if (to->use_alltoallv) {
        ierr = MPI_Alltoallv(to->values,to->counts,to->displs,MPIU_SCALAR,from->values,from->counts,from->displs,MPIU_SCALAR,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      } else if (to->use_window) {
        PetscInt cnt;

        ierr = MPI_Win_fence(0,from->window);CHKERRQ(ierr);
        for (i=0; i<nsends; i++) {
          cnt  = bs*(to->starts[i+1]-to->starts[i]);
          ierr = MPI_Put(to->values+bs*to->starts[i],cnt,MPIU_SCALAR,to->procs[i],bs*to->winstarts[i],cnt,MPIU_SCALAR,from->window);CHKERRQ(ierr);
        }
      } else if (nsends) {
        ierr = MPI_Startall_isend(to->starts[to->n]*bs,nsends,swaits);CHKERRQ(ierr);
      }
    } else {
      if (to->sharedspace) {
        /* Pack the send data into my shared memory buffer  --- this is the normal forward scatter */
        PETSCMAP1(Pack)(to->sharedcnt,to->sharedspaceindices,xv,to->sharedspace,bs);
      } else {
        /* Pack the send data into receivers shared memory buffer -- this is the normal backward scatter */
        for (i=0; i<to->msize; i++) {
          if (to->sharedspacesoffset && to->sharedspacesoffset[i] > -1) {
            PETSCMAP1(Pack)(to->sharedspacestarts[i+1] - to->sharedspacestarts[i],to->sharedspaceindices + to->sharedspacestarts[i],xv,&to->sharedspaces[i][bs*to->sharedspacesoffset[i]],bs);
          }
        }
      }
      /* this version packs and sends one at a time */
      for (i=0; i<nsends; i++) {
        PETSCMAP1(Pack)(sstarts[i+1]-sstarts[i],indices + sstarts[i],xv,svalues + bs*sstarts[i],bs);
        ierr = MPI_Start_isend((sstarts[i+1]-sstarts[i])*bs,swaits+i);CHKERRQ(ierr);
      }
    }

    if (!from->use_readyreceiver && to->sendfirst && !to->use_alltoallv && !to->use_window) {
      /* post receives since they were not previously posted   */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }
  }

  /* take care of local scatters */
  if (to->local.n) {
    if (to->local.made_of_copies && addv == INSERT_VALUES) {
      /* do copy when it is not a self-to-self copy */
      if (!(yv == xv && to->local.same_copy_starts)) {
        for (i=0; i<to->local.n_copies; i++) {
          /* Do we need to take care of overlaps? We could but overlaps sound more like a bug than a requirement,
             so I just leave it and let PetscMemcpy detect this bug.
           */
          ierr = PetscMemcpy(yv + from->local.copy_starts[i],xv + to->local.copy_starts[i],to->local.copy_lengths[i]);CHKERRQ(ierr);
        }
      }
    } else {
      if (xv == yv && addv == INSERT_VALUES && to->local.nonmatching_computed) {
        /* only copy entries that do not share identical memory locations */
        ierr = PETSCMAP1(Scatter)(to->local.n_nonmatching,to->local.slots_nonmatching,xv,from->local.slots_nonmatching,yv,addv,bs);CHKERRQ(ierr);
      } else {
        ierr = PETSCMAP1(Scatter)(to->local.n,to->local.vslots,xv,from->local.vslots,yv,addv,bs);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecRestoreArrayRead(xin,(const PetscScalar**)&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

PetscErrorCode PETSCMAP1(VecScatterEnd)(VecScatter ctx,Vec xin,Vec yin,InsertMode addv,ScatterMode mode)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *rvalues,*yv;
  PetscErrorCode         ierr;
  PetscInt               nrecvs,nsends,*indices,count,*rstarts,bs;
  PetscMPIInt            imdex;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             xrstatus,*rstatus,*sstatus;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);
  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  to      = (VecScatter_MPI_General*)ctx->todata;
  from    = (VecScatter_MPI_General*)ctx->fromdata;
  rwaits  = from->requests;
  swaits  = to->requests;
  sstatus = to->sstatus;    /* sstatus and rstatus are always stored in to */
  rstatus = to->rstatus;
  if (mode & SCATTER_REVERSE) {
    to     = (VecScatter_MPI_General*)ctx->fromdata;
    from   = (VecScatter_MPI_General*)ctx->todata;
    rwaits = from->rev_requests;
    swaits = to->rev_requests;
  }
  bs      = from->bs;
  rvalues = from->values;
  nrecvs  = from->n;
  nsends  = to->n;
  indices = from->indices;
  rstarts = from->starts;

  if (ctx->packtogether || (to->use_alltoallw && (addv != INSERT_VALUES)) || (to->use_alltoallv && !to->use_alltoallw) || to->use_window) {
    if (to->use_window) {ierr = MPI_Win_fence(0,from->window);CHKERRQ(ierr);}
    else if (nrecvs && !to->use_alltoallv) {ierr = MPI_Waitall(nrecvs,rwaits,rstatus);CHKERRQ(ierr);}
    ierr = PETSCMAP1(UnPack)(from->starts[from->n],from->values,indices,yv,addv,bs);CHKERRQ(ierr);
  } else if (!to->use_alltoallw) {
    PetscMPIInt i;
    ierr = MPI_Barrier(PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);

    /* unpack one at a time */
    count = nrecvs;
    while (count) {
      if (ctx->reproduce) {
        imdex = count - 1;
        ierr  = MPI_Wait(rwaits+imdex,&xrstatus);CHKERRQ(ierr);
      } else {
        ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&xrstatus);CHKERRQ(ierr);
      }
      /* unpack receives into our local space */
      ierr = PETSCMAP1(UnPack)(rstarts[imdex+1] - rstarts[imdex],rvalues + bs*rstarts[imdex],indices + rstarts[imdex],yv,addv,bs);CHKERRQ(ierr);
      count--;
    }
    /* handle processes that share the same shared memory communicator */
    if (from->sharedspace) {
      /* unpack the data from my shared memory buffer  --- this is the normal backward scatter */
      PETSCMAP1(UnPack)(from->sharedcnt,from->sharedspace,from->sharedspaceindices,yv,addv,bs);
    } else {
      /* unpack the data from each of my sending partners shared memory buffers --- this is the normal forward scatter */
      for (i=0; i<from->msize; i++) {
        if (from->sharedspacesoffset && from->sharedspacesoffset[i] > -1) {
          ierr = PETSCMAP1(UnPack)(from->sharedspacestarts[i+1] - from->sharedspacestarts[i],&from->sharedspaces[i][bs*from->sharedspacesoffset[i]],from->sharedspaceindices + from->sharedspacestarts[i],yv,addv,bs);CHKERRQ(ierr);
        }
      }
    }
    ierr = MPI_Barrier(PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
  }
  if (from->use_readyreceiver) {
    if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*bs,nrecvs,rwaits);CHKERRQ(ierr);}
    ierr = MPI_Barrier(PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
  }

  /* wait on sends */
  if (nsends  && !to->use_alltoallv  && !to->use_window) {ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);}
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------- */
#include <../src/vec/vec/impls/node/vecnodeimpl.h>

PetscErrorCode PETSCMAP1(VecScatterBeginMPI3Node)(VecScatter ctx,Vec xin,Vec yin,InsertMode addv,ScatterMode mode)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *xv,*yv,*svalues;
  MPI_Request            *rwaits,*swaits;
  PetscErrorCode         ierr;
  PetscInt               i,*indices,*sstarts,nrecvs,nsends,bs;

  PetscFunctionBegin;
  if (mode & SCATTER_REVERSE) {
    to     = (VecScatter_MPI_General*)ctx->fromdata;
    from   = (VecScatter_MPI_General*)ctx->todata;
    rwaits = from->rev_requests;
    swaits = to->rev_requests;
  } else {
    to     = (VecScatter_MPI_General*)ctx->todata;
    from   = (VecScatter_MPI_General*)ctx->fromdata;
    rwaits = from->requests;
    swaits = to->requests;
  }
  bs      = to->bs;
  svalues = to->values;
  nrecvs  = from->n;
  nsends  = to->n;
  indices = to->indices;
  sstarts = to->starts;

  ierr = VecGetArrayRead(xin,(const PetscScalar**)&xv);CHKERRQ(ierr);

  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);}
  else yv = xv;

  if (!(mode & SCATTER_LOCAL)) {
    if (!from->use_readyreceiver && !to->sendfirst && !to->use_alltoallv  & !to->use_window) {
      /* post receives since they were not previously posted    */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }

#if defined(PETSC_HAVE_MPI_ALLTOALLW)  && !defined(PETSC_USE_64BIT_INDICES)
    if (to->use_alltoallw && addv == INSERT_VALUES) {
      ierr = MPI_Alltoallw(xv,to->wcounts,to->wdispls,to->types,yv,from->wcounts,from->wdispls,from->types,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
    } else
#endif
    if (ctx->packtogether || to->use_alltoallv || to->use_window) {
      /* this version packs all the messages together and sends, when -vecscatter_packtogether used */
      PETSCMAP1(Pack)(sstarts[nsends],indices,xv,svalues,bs);
      if (to->use_alltoallv) {
        ierr = MPI_Alltoallv(to->values,to->counts,to->displs,MPIU_SCALAR,from->values,from->counts,from->displs,MPIU_SCALAR,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      } else if (to->use_window) {
        PetscInt cnt;

        ierr = MPI_Win_fence(0,from->window);CHKERRQ(ierr);
        for (i=0; i<nsends; i++) {
          cnt  = bs*(to->starts[i+1]-to->starts[i]);
          ierr = MPI_Put(to->values+bs*to->starts[i],cnt,MPIU_SCALAR,to->procs[i],bs*to->winstarts[i],cnt,MPIU_SCALAR,from->window);CHKERRQ(ierr);
        }
      } else if (nsends) {
        ierr = MPI_Startall_isend(to->starts[to->n]*bs,nsends,swaits);CHKERRQ(ierr);
      }
    } else {
      /* this version packs and sends one at a time */
      for (i=0; i<nsends; i++) {
        PETSCMAP1(Pack)(sstarts[i+1]-sstarts[i],indices + sstarts[i],xv,svalues + bs*sstarts[i],bs);
        ierr = MPI_Start_isend((sstarts[i+1]-sstarts[i])*bs,swaits+i);CHKERRQ(ierr);
      }
    }

    if (!from->use_readyreceiver && to->sendfirst && !to->use_alltoallv && !to->use_window) {
      /* post receives since they were not previously posted   */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }
  }

  /* take care of local scatters */
  if (to->local.n) {
    if (to->local.made_of_copies && addv == INSERT_VALUES) {
      /* do copy when it is not a self-to-self copy */
      if (!(yv == xv && to->local.same_copy_starts)) {
        for (i=0; i<to->local.n_copies; i++) {
          /* Do we need to take care of overlaps? We could but overlaps sound more like a bug than a requirement,
             so I just leave it and let PetscMemcpy detect this bug.
           */
          ierr = PetscMemcpy(yv + from->local.copy_starts[i],xv + to->local.copy_starts[i],to->local.copy_lengths[i]);CHKERRQ(ierr);
        }
      }
    } else {
      if (xv == yv && addv == INSERT_VALUES && to->local.nonmatching_computed) {
        /* only copy entries that do not share identical memory locations */
        ierr = PETSCMAP1(Scatter)(to->local.n_nonmatching,to->local.slots_nonmatching,xv,from->local.slots_nonmatching,yv,addv,bs);CHKERRQ(ierr);
      } else {
        ierr = PETSCMAP1(Scatter)(to->local.n,to->local.vslots,xv,from->local.vslots,yv,addv,bs);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecRestoreArrayRead(xin,(const PetscScalar**)&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#define PETSC_MEMSHARE_SAFE
PetscErrorCode PETSCMAP1(VecScatterEndMPI3Node)(VecScatter ctx,Vec xin,Vec yin,InsertMode addv,ScatterMode mode)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *rvalues,*yv;
  const PetscScalar      *xv;
  PetscErrorCode         ierr;
  PetscInt               nrecvs,nsends,*indices,count,*rstarts,bs;
  PetscMPIInt            imdex;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             xrstatus,*rstatus,*sstatus;
  Vec_Node               *vnode;
  PetscInt               cnt,*idx,*idy;
  MPI_Comm               comm,mscomm,veccomm;
  PetscCommShared        scomm;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);

  ierr = PetscObjectGetComm((PetscObject)ctx,&comm);CHKERRQ(ierr);
  ierr = PetscCommSharedGet(comm,&scomm);CHKERRQ(ierr);
  ierr = PetscCommSharedGetComm(scomm,&mscomm);CHKERRQ(ierr);

  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  to      = (VecScatter_MPI_General*)ctx->todata;
  from    = (VecScatter_MPI_General*)ctx->fromdata;
  rwaits  = from->requests;
  swaits  = to->requests;
  sstatus = to->sstatus;    /* sstatus and rstatus are always stored in to */
  rstatus = to->rstatus;
  if (mode & SCATTER_REVERSE) {
    to     = (VecScatter_MPI_General*)ctx->fromdata;
    from   = (VecScatter_MPI_General*)ctx->todata;
    rwaits = from->rev_requests;
    swaits = to->rev_requests;
  }
  bs      = from->bs;
  rvalues = from->values;
  nrecvs  = from->n;
  nsends  = to->n;
  indices = from->indices;
  rstarts = from->starts;

  if (ctx->packtogether || (to->use_alltoallw && (addv != INSERT_VALUES)) || (to->use_alltoallv && !to->use_alltoallw) || to->use_window) {
    if (to->use_window) {ierr = MPI_Win_fence(0,from->window);CHKERRQ(ierr);}
    else if (nrecvs && !to->use_alltoallv) {ierr = MPI_Waitall(nrecvs,rwaits,rstatus);CHKERRQ(ierr);}
    ierr = PETSCMAP1(UnPack)(from->starts[from->n],from->values,indices,yv,addv,bs);CHKERRQ(ierr);
  } else if (!to->use_alltoallw) {
    PetscMPIInt i,xsize;
    PetscInt    k,k1;
    PetscScalar *sharedspace;

    /* unpack one at a time */
    count = nrecvs;
    while (count) {
      if (ctx->reproduce) {
        imdex = count - 1;
        ierr  = MPI_Wait(rwaits+imdex,&xrstatus);CHKERRQ(ierr);
      } else {
        ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&xrstatus);CHKERRQ(ierr);
      }
      /* unpack receives into our local space */
      ierr = PETSCMAP1(UnPack)(rstarts[imdex+1] - rstarts[imdex],rvalues + bs*rstarts[imdex],indices + rstarts[imdex],yv,addv,bs);CHKERRQ(ierr);
      count--;
    }

    /* handle processes that share the same shared memory communicator */
#if defined(PETSC_MEMSHARE_SAFE)
    ierr = MPI_Barrier(mscomm);CHKERRQ(ierr);
#endif

    /* check if xin is sequential */
    ierr = PetscObjectGetComm((PetscObject)xin,&veccomm);CHKERRQ(ierr);
    ierr = MPI_Comm_size(veccomm,&xsize);CHKERRQ(ierr);

    if (xsize == 1 || from->sharedspace) { /* 'from->sharedspace' indicates this core's shared memory will be written */
      /* StoP: read sequential local xvalues, then write to shared yvalues */
      PetscInt notdone = to->notdone;
      vnode = (Vec_Node*)yin->data;
      if (!vnode->win) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"vector y must have type VECNODE with shared memory");
      if (ctx->is_duplicate) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Duplicate index is not supported");
      ierr  = VecGetArrayRead(xin,&xv);CHKERRQ(ierr);

      i = 0;
      while (notdone) {
        while (i < to->msize) {
          if (to->sharedspacesoffset && to->sharedspacesoffset[i] > -1) {
            cnt = to->sharedspacestarts[i+1] - to->sharedspacestarts[i];
            idx = to->sharedspaceindices + to->sharedspacestarts[i];
            idy = idx + to->sharedcnt;

            sharedspace = vnode->winarray[i];

            if (sharedspace[-1] != yv[-1]) {
              if (PetscRealPart(sharedspace[-1] - yv[-1]) > 0.0) {
                PetscMPIInt msrank;
                ierr = MPI_Comm_rank(mscomm,&msrank);CHKERRQ(ierr);
                SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"[%d] statecnt %g > [%d] my_statecnt %g",i,PetscRealPart(sharedspace[-1]),msrank,PetscRealPart(yv[-1]));
              }
              /* i-the core has not reached the current object statecnt yet, wait ... */
              continue;
            }

            if (addv == ADD_VALUES) {
              for (k= 0; k<cnt; k++) {
                for (k1=0; k1<bs; k1++) sharedspace[idy[k]+k1] += xv[idx[k]+k1];
              }
            } else if (addv == INSERT_VALUES) {
              for (k= 0; k<cnt; k++) {
                for (k1=0; k1<bs; k1++) sharedspace[idy[k]+k1] = xv[idx[k]+k1];
              }
            } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %D", addv);
            notdone--;
          }
          i++;
        }
      }
      ierr = VecRestoreArrayRead(xin,&xv);CHKERRQ(ierr);
    } else {
      /* PtoS: read shared xvalues, then write to sequential local yvalues */
      PetscInt notdone = from->notdone;

      vnode = (Vec_Node*)xin->data;
      if (!vnode->win && notdone) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"vector x must have type VECNODE with shared memory");
      ierr  = VecGetArrayRead(xin,&xv);CHKERRQ(ierr);

      i = 0;
      while (notdone) {
        while (i < from->msize) {
          if (from->sharedspacesoffset && from->sharedspacesoffset[i] > -1) {
            cnt = from->sharedspacestarts[i+1] - from->sharedspacestarts[i];
            idy = from->sharedspaceindices + from->sharedspacestarts[i]; /* recv local y indices */
            idx = idy + from->sharedcnt;

            sharedspace = vnode->winarray[i];

            if (sharedspace[-1] != xv[-1]) {
              if (PetscRealPart(sharedspace[-1] - xv[-1]) > 0.0) {
                PetscMPIInt msrank;
                ierr = MPI_Comm_rank(mscomm,&msrank);CHKERRQ(ierr);
                SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"[%d] statecnt %g > [%d] my_statecnt %g",i,PetscRealPart(sharedspace[-1]),msrank,PetscRealPart(xv[-1]));
              }
              /* i-the core has not reached the current object state cnt yet, wait ... */
              continue;
            }

            if (addv==ADD_VALUES) {
              for (k=0; k<cnt; k++) {
                for (k1=0; k1<bs; k1++) yv[idy[k]+k1] += sharedspace[idx[k]+k1]; /* read x shared values */
              }
            } else if (addv==INSERT_VALUES){
              for (k=0; k<cnt; k++) {
                for (k1=0; k1<bs; k1++) yv[idy[k]+k1] = sharedspace[idx[k]+k1]; /* read x shared values */
              }
            } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %D", addv);

            notdone--;
          }
          i++;
        }
      }
      ierr = VecRestoreArrayRead(xin,&xv);CHKERRQ(ierr);
    }

    /* output y is parallel, ensure it is done -- would lose performance */
    ierr = MPI_Barrier(mscomm);CHKERRQ(ierr);
  }
  if (from->use_readyreceiver) {
    if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*bs,nrecvs,rwaits);CHKERRQ(ierr);}
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  }

  /* wait on sends */
  if (nsends  && !to->use_alltoallv  && !to->use_window) {ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);}
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef PETSCMAP1_a
#undef PETSCMAP1_b
#undef PETSCMAP1
#undef BS

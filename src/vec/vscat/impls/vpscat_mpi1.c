
/*
    Defines parallel vector scatters using MPI1.
*/

#include <../src/vec/vec/impls/dvecimpl.h>         /*I "petscvec.h" I*/
#include <../src/vec/vec/impls/mpi/pvecimpl.h>
#include <petscsf.h>

PetscErrorCode VecScatterView_MPI_MPI1(VecScatter ctx,PetscViewer viewer)
{
  VecScatter_MPI_General *to  =(VecScatter_MPI_General*)ctx->todata;
  VecScatter_MPI_General *from=(VecScatter_MPI_General*)ctx->fromdata;
  PetscErrorCode         ierr;
  PetscInt               i;
  PetscMPIInt            rank;
  PetscViewerFormat      format;
  PetscBool              iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)ctx),&rank);CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format ==  PETSC_VIEWER_ASCII_INFO) {
      PetscInt nsend_max,nrecv_max,lensend_max,lenrecv_max,alldata,itmp;

      ierr = MPI_Reduce(&to->n,&nsend_max,1,MPIU_INT,MPI_MAX,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      ierr = MPI_Reduce(&from->n,&nrecv_max,1,MPIU_INT,MPI_MAX,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      itmp = to->starts[to->n+1];
      ierr = MPI_Reduce(&itmp,&lensend_max,1,MPIU_INT,MPI_MAX,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      itmp = from->starts[from->n+1];
      ierr = MPI_Reduce(&itmp,&lenrecv_max,1,MPIU_INT,MPI_MAX,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      ierr = MPI_Reduce(&itmp,&alldata,1,MPIU_INT,MPI_SUM,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"VecScatter statistics\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum number sends %D\n",nsend_max);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum number receives %D\n",nrecv_max);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum data sent %D\n",(int)(lensend_max*to->bs*sizeof(PetscScalar)));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum data received %D\n",(int)(lenrecv_max*to->bs*sizeof(PetscScalar)));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Total data sent %D\n",(int)(alldata*to->bs*sizeof(PetscScalar)));CHKERRQ(ierr);

    } else {
      ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number sends = %D; Number to self = %D\n",rank,to->n,to->local.n);CHKERRQ(ierr);
      if (to->n) {
        for (i=0; i<to->n; i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]   %D length = %D to whom %d\n",rank,i,to->starts[i+1]-to->starts[i],to->procs[i]);CHKERRQ(ierr);
          if (to->memcpy_plan.optimized[i]) { ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  is optimized with %D memcpy's in Pack\n",to->memcpy_plan.copy_offsets[i+1]-to->memcpy_plan.copy_offsets[i]);CHKERRQ(ierr); }
        }
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Now the indices for all remote sends (in order by process sent to)\n");CHKERRQ(ierr);
        for (i=0; i<to->starts[to->n]; i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D \n",rank,to->indices[i]);CHKERRQ(ierr);
        }
      }

      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number receives = %D; Number from self = %D\n",rank,from->n,from->local.n);CHKERRQ(ierr);
      if (from->n) {
        for (i=0; i<from->n; i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D length %D from whom %d\n",rank,i,from->starts[i+1]-from->starts[i],from->procs[i]);CHKERRQ(ierr);
          if (from->memcpy_plan.optimized[i]) { ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  is optimized with %D memcpy's in Unpack\n",to->memcpy_plan.copy_offsets[i+1]-to->memcpy_plan.copy_offsets[i]);CHKERRQ(ierr); }
        }

        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Now the indices for all remote receives (in order by process received from)\n");CHKERRQ(ierr);
        for (i=0; i<from->starts[from->n]; i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D \n",rank,from->indices[i]);CHKERRQ(ierr);
        }
      }
      if (to->local.n) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Indices for local part of scatter\n",rank);CHKERRQ(ierr);
        if (to->local.memcpy_plan.optimized[0]) {
          ierr = PetscViewerASCIIPrintf(viewer,"Local part of the scatter is made of %D copies\n",to->local.memcpy_plan.copy_offsets[1]);CHKERRQ(ierr);
        }
        for (i=0; i<to->local.n; i++) {  /* the to and from have the opposite meaning from what you would expect */
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] From %D to %D \n",rank,to->local.vslots[i],from->local.vslots[i]);CHKERRQ(ierr);
        }
      }

      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"Method used to implement the VecScatter: ");CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPI_ALLTOALLW)  && !defined(PETSC_USE_64BIT_INDICES)
      if (to->use_alltoallw) {
        ierr = PetscViewerASCIIPrintf(viewer,"Uses MPI_alltoallw if INSERT_MODE\n");CHKERRQ(ierr);
      } else
#endif
      if (ctx->packtogether || to->use_alltoallv || to->use_window) {
        if (to->use_alltoallv) {
          ierr = PetscViewerASCIIPrintf(viewer,"Uses MPI MPI_alltoallv\n");CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPI_WIN_CREATE_FEATURE)
        } else if (to->use_window) {
          ierr = PetscViewerASCIIPrintf(viewer,"Uses MPI window\n");CHKERRQ(ierr);
#endif
        } else {
          ierr = PetscViewerASCIIPrintf(viewer,"Packs all messages and then sends them\n");CHKERRQ(ierr);
        }
      }  else {
        ierr = PetscViewerASCIIPrintf(viewer,"Packs and sends messages one at a time\n");CHKERRQ(ierr);
      }

#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
      if (to->use_intranodeshm) {
        ierr = PetscViewerASCIIPrintf(viewer,"Uses MPI-3.0 process shared-memory for intra-node communication\n");CHKERRQ(ierr);
      }
#endif
    }
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
PetscErrorCode VecScatterDestroy_PtoP_MPI1(VecScatter ctx)
{
  VecScatter_MPI_General *to   = (VecScatter_MPI_General*)ctx->todata;
  VecScatter_MPI_General *from = (VecScatter_MPI_General*)ctx->fromdata;
  PetscErrorCode         ierr;
  PetscInt               i;

  PetscFunctionBegin;
  if (to->use_readyreceiver) {
    /*
       Since we have already posted sends we must cancel them before freeing
       the requests
    */
    for (i=0; i<from->n; i++) {
      ierr = MPI_Cancel(from->requests+i);CHKERRQ(ierr);
    }
    for (i=0; i<to->n; i++) {
      ierr = MPI_Cancel(to->rev_requests+i);CHKERRQ(ierr);
    }
    ierr = MPI_Waitall(from->n,from->requests,to->rstatus);CHKERRQ(ierr);
    ierr = MPI_Waitall(to->n,to->rev_requests,to->rstatus);CHKERRQ(ierr);
  }

#if defined(PETSC_HAVE_MPI_ALLTOALLW) && !defined(PETSC_USE_64BIT_INDICES)
  if (to->use_alltoallw) {
    for (i=0; i<to->n; i++) {
      ierr = MPI_Type_free(to->types+to->procs[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree3(to->wcounts,to->wdispls,to->types);CHKERRQ(ierr);
    if (!from->contiq) {
      for (i=0; i<from->n; i++) {
        ierr = MPI_Type_free(from->types+from->procs[i]);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree3(from->wcounts,from->wdispls,from->types);CHKERRQ(ierr);
  }
#endif

#if defined(PETSC_HAVE_MPI_WIN_CREATE_FEATURE)
  if (to->use_window) {
    ierr = MPI_Win_free(&from->window);CHKERRQ(ierr);
    ierr = MPI_Win_free(&to->window);CHKERRQ(ierr);
    ierr = PetscFree(from->winstarts);CHKERRQ(ierr);
    ierr = PetscFree(to->winstarts);CHKERRQ(ierr);
  }
#endif

  if (to->use_alltoallv) {
    ierr = PetscFree2(to->counts,to->displs);CHKERRQ(ierr);
    ierr = PetscFree2(from->counts,from->displs);CHKERRQ(ierr);
  }

  /* release MPI resources obtained with MPI_Send_init() and MPI_Recv_init() */
  /*
     IBM's PE version of MPI has a bug where freeing these guys will screw up later
     message passing.
  */
#if !defined(PETSC_HAVE_BROKEN_REQUEST_FREE)
  if (!to->use_alltoallv && !to->use_window) {   /* currently the to->requests etc are ALWAYS allocated even if not used */
    if (to->requests) {
      for (i=0; i<to->n; i++) {
        ierr = MPI_Request_free(to->requests + i);CHKERRQ(ierr);
      }
    }
    if (to->rev_requests) {
      for (i=0; i<to->n; i++) {
        ierr = MPI_Request_free(to->rev_requests + i);CHKERRQ(ierr);
      }
    }
  }
  /*
      MPICH could not properly cancel requests thus with ready receiver mode we
    cannot free the requests. It may be fixed now, if not then put the following
    code inside a if (!to->use_readyreceiver) {
  */
  if (!to->use_alltoallv && !to->use_window) {    /* currently the from->requests etc are ALWAYS allocated even if not used */
    if (from->requests) {
      for (i=0; i<from->n; i++) {
        ierr = MPI_Request_free(from->requests + i);CHKERRQ(ierr);
      }
    }

    if (from->rev_requests) {
      for (i=0; i<from->n; i++) {
        ierr = MPI_Request_free(from->rev_requests + i);CHKERRQ(ierr);
      }
    }
  }
#endif

#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  if (to->use_intranodeshm) {
#if !defined(PETSC_WRITE_MEMORY_BARRIER)
    ierr = MPI_Win_unlock_all(to->shmwin);CHKERRQ(ierr); /* see comments in VecScatterBegin for why this call */
#endif
    ierr = MPI_Win_free(&to->shmwin);CHKERRQ(ierr); /* free shmwin and the associated shared memory. from->shmwin = to->shmwin, so do not free the window twice */
    ierr = PetscFree5(to->shmprocs,to->shmspaces,to->shmstates,to->shmstarts,to->shmindices);CHKERRQ(ierr);
    ierr = PetscFree5(from->shmprocs,from->shmspaces,from->shmstates,from->shmstarts,from->shmindices);CHKERRQ(ierr);
  }
#endif

  ierr = PetscFree(to->local.vslots);CHKERRQ(ierr);
  ierr = PetscFree(from->local.vslots);CHKERRQ(ierr);
  ierr = PetscFree2(to->counts,to->displs);CHKERRQ(ierr);
  ierr = PetscFree2(from->counts,from->displs);CHKERRQ(ierr);
  ierr = PetscFree(to->local.slots_nonmatching);CHKERRQ(ierr);
  ierr = PetscFree(from->local.slots_nonmatching);CHKERRQ(ierr);
  ierr = PetscFree(to->rev_requests);CHKERRQ(ierr);
  ierr = PetscFree(from->rev_requests);CHKERRQ(ierr);
  ierr = PetscFree(to->requests);CHKERRQ(ierr);
  ierr = PetscFree(from->requests);CHKERRQ(ierr);
  ierr = PetscFree4(to->values,to->indices,to->starts,to->procs);CHKERRQ(ierr);
  ierr = PetscFree2(to->sstatus,to->rstatus);CHKERRQ(ierr);
  ierr = PetscFree4(from->values,from->indices,from->starts,from->procs);CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanDestroy_PtoP(to,from);CHKERRQ(ierr);
  ierr = PetscFree(from);CHKERRQ(ierr);
  ierr = PetscFree(to);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

PetscErrorCode VecScatterCopy_PtoP_MPI1(VecScatter in,VecScatter out)
{
  VecScatter_MPI_General *in_to   = (VecScatter_MPI_General*)in->todata,*out_to;
  VecScatter_MPI_General *in_from = (VecScatter_MPI_General*)in->fromdata,*out_from;
  PetscErrorCode         ierr;
  PetscInt               ny,bs = in_from->bs;
  PetscMPIInt            size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)in),&size);CHKERRQ(ierr);

  out->ops->begin   = in->ops->begin;
  out->ops->end     = in->ops->end;
  out->ops->copy    = in->ops->copy;
  out->ops->destroy = in->ops->destroy;
  out->ops->view    = in->ops->view;

  ierr = PetscNewLog(out,&out_to);CHKERRQ(ierr);
  ierr = PetscNewLog(out,&out_from);CHKERRQ(ierr);

  out->todata   = (void*)out_to;
  out->fromdata = (void*)out_from;

  /* copy the local scatter (on-processor) part of the context */
  out_to->local.n                      = in_to->local.n;
  out_to->local.nonmatching_computed   = PETSC_FALSE;
  out_to->local.n_nonmatching          = 0;
  out_to->local.slots_nonmatching      = 0;

  out_from->local.n                    = in_from->local.n;
  out_from->local.nonmatching_computed = PETSC_FALSE;
  out_from->local.n_nonmatching        = 0;
  out_from->local.slots_nonmatching    = 0;

  if (in_to->local.n) {
    ierr = PetscMalloc1(in_to->local.n,&out_to->local.vslots);CHKERRQ(ierr);
    ierr = PetscMalloc1(in_from->local.n,&out_from->local.vslots);CHKERRQ(ierr);
    ierr = PetscMemcpy(out_to->local.vslots,in_to->local.vslots,in_to->local.n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(out_from->local.vslots,in_from->local.vslots,in_from->local.n*sizeof(PetscInt));CHKERRQ(ierr);
  }

  /* copy the remote scatter (off-processor) part of the context */
  ny                  = in_to->starts[in_to->n];
  out_to->n           = in_to->n;
  out_to->format      = in_to->format;
  out_to->sendfirst   = in_to->sendfirst;
  out_to->bs          = bs;

  ierr = PetscMalloc1(out_to->n,&out_to->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(bs*ny,&out_to->values,ny,&out_to->indices,out_to->n+1,&out_to->starts,out_to->n,&out_to->procs);CHKERRQ(ierr);
  ierr = PetscMalloc2(PetscMax(in_to->n,in_from->n),&out_to->sstatus,PetscMax(in_to->n,in_from->n),&out_to->rstatus);CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->indices,in_to->indices,ny*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->starts,in_to->starts,(out_to->n+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->procs,in_to->procs,(out_to->n)*sizeof(PetscMPIInt));CHKERRQ(ierr);

  out_from->format    = in_from->format;
  ny                  = in_from->starts[in_from->n];
  out_from->n         = in_from->n;
  out_from->sendfirst = in_from->sendfirst;
  out_from->bs        = bs;

  ierr = PetscMalloc1(out_from->n,&out_from->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(ny*bs,&out_from->values,ny,&out_from->indices,out_from->n+1,&out_from->starts,out_from->n,&out_from->procs);CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->indices,in_from->indices,ny*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->starts,in_from->starts,(out_from->n+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->procs,in_from->procs,(out_from->n)*sizeof(PetscMPIInt));CHKERRQ(ierr);

  if (in_to->use_alltoallv) {
    out_to->use_alltoallv = out_from->use_alltoallv = PETSC_TRUE;

    ierr = PetscMalloc2(size,&out_to->counts,size,&out_to->displs);CHKERRQ(ierr);
    ierr = PetscMemcpy(out_to->counts,in_to->counts,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(out_to->displs,in_to->displs,size*sizeof(PetscMPIInt));CHKERRQ(ierr);

    ierr = PetscMalloc2(size,&out_from->counts,size,&out_from->displs);CHKERRQ(ierr);
    ierr = PetscMemcpy(out_from->counts,in_from->counts,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(out_from->displs,in_from->displs,size*sizeof(PetscMPIInt));CHKERRQ(ierr);

#if defined(PETSC_HAVE_MPI_ALLTOALLW)  && !defined(PETSC_USE_64BIT_INDICES)
    if (in_to->use_alltoallw) {
      /* It is not worth the trouble to implement alltoallw-context copying because 1) alltoallv/w
         is non-scalable and no one uses it; 2) indexed MPI data types created by
         MPI_Type_create_indexed_block for alltoallw are not easy to handle. If we duplicate those
         data types here, then in VecScatterRemap() we have to rebuild them since indices are changed
         and the old data types become invalid. So the best approach is to leave it and inform the user.
       */
      out_to->use_alltoallw = out_from->use_alltoallw = PETSC_FALSE;
      ierr = PetscInfo(out,"Copying a vecscatter context of type -vecscatter_alltoall -vecscatter_nopack is not implemented and we will copy it without -vecscatter_nopack\n");CHKERRQ(ierr);
    }
#endif
#if defined(PETSC_HAVE_MPI_WIN_CREATE_FEATURE)
  } else if (in_to->use_window) {
    PetscMPIInt winsize;
    MPI_Comm comm = PetscObjectComm((PetscObject)out);

    out_to->use_window = out_from->use_window = PETSC_TRUE;

    winsize = (out_to->n ? out_to->starts[out_to->n] : 0)*bs*sizeof(PetscScalar);
    ierr = MPI_Win_create(out_to->values ? out_to->values : MPI_BOTTOM,winsize,sizeof(PetscScalar),MPI_INFO_NULL,comm,&out_to->window);CHKERRQ(ierr);
    ierr = PetscMalloc1(out_to->n,&out_to->winstarts);CHKERRQ(ierr);
    ierr = PetscMemcpy(out_to->winstarts,in_to->winstarts,out_to->n*sizeof(PetscInt));CHKERRQ(ierr);

    winsize = (out_from->n ? out_from->starts[out_from->n] : 0)*bs*sizeof(PetscScalar);
    ierr = MPI_Win_create(out_from->values ? out_from->values : MPI_BOTTOM,winsize,sizeof(PetscScalar),MPI_INFO_NULL,comm,&out_from->window);CHKERRQ(ierr);
    ierr = PetscMalloc1(out_from->n,&out_from->winstarts);CHKERRQ(ierr);
    ierr = PetscMemcpy(out_from->winstarts,in_from->winstarts,out_from->n*sizeof(PetscInt));CHKERRQ(ierr);
 #endif
  } else {
    /* set up the request arrays for use with isend_init() and irecv_init() */
    PetscMPIInt tag;
    MPI_Comm    comm;
    PetscInt    *sstarts = out_to->starts,  *rstarts = out_from->starts;
    PetscMPIInt *sprocs  = out_to->procs,   *rprocs  = out_from->procs;
    PetscInt    i;
    PetscBool   flg;
    MPI_Request *swaits   = out_to->requests,*rwaits  = out_from->requests;
    MPI_Request *rev_swaits,*rev_rwaits;
    PetscScalar *Ssvalues = out_to->values, *Srvalues = out_from->values;

    ierr = PetscMalloc1(in_to->n,&out_to->rev_requests);CHKERRQ(ierr);
    ierr = PetscMalloc1(in_from->n,&out_from->rev_requests);CHKERRQ(ierr);

    rev_rwaits = out_to->rev_requests;
    rev_swaits = out_from->rev_requests;

    tag          = ((PetscObject)out)->tag;
    ierr         = PetscObjectGetComm((PetscObject)out,&comm);CHKERRQ(ierr);

    /* Register the receives that you will use later (sends for scatter reverse) */
    for (i=0; i<out_from->n; i++) {
      ierr = MPI_Recv_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);CHKERRQ(ierr);
      ierr = MPI_Send_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,comm,rev_swaits+i);CHKERRQ(ierr);
    }

    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_rsend",&flg,NULL);CHKERRQ(ierr);
    if (flg) {
      out_to->use_readyreceiver   = PETSC_TRUE;
      out_from->use_readyreceiver = PETSC_TRUE;
      for (i=0; i<out_to->n; i++) {
        ierr = MPI_Rsend_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
      }
      if (out_from->n) {ierr = MPI_Startall_irecv(out_from->starts[out_from->n]*out_from->bs,out_from->n,out_from->requests);CHKERRQ(ierr);}
      ierr = MPI_Barrier(comm);CHKERRQ(ierr);
      ierr = PetscInfo(in,"Using VecScatter ready receiver mode\n");CHKERRQ(ierr);
    } else {
      out_to->use_readyreceiver   = PETSC_FALSE;
      out_from->use_readyreceiver = PETSC_FALSE;
      flg                         = PETSC_FALSE;
      ierr                        = PetscOptionsGetBool(NULL,NULL,"-vecscatter_ssend",&flg,NULL);CHKERRQ(ierr);
      if (flg) {
        ierr = PetscInfo(in,"Using VecScatter Ssend mode\n");CHKERRQ(ierr);
      }
      for (i=0; i<out_to->n; i++) {
        if (!flg) {
          ierr = MPI_Send_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
        } else {
          ierr = MPI_Ssend_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
        }
      }
    }
    /* Register receives for scatter reverse */
    for (i=0; i<out_to->n; i++) {
      ierr = MPI_Recv_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,rev_rwaits+i);CHKERRQ(ierr);
    }
  }

#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  out_to->use_intranodeshm   = in_to->use_intranodeshm;
  out_from->use_intranodeshm = in_from->use_intranodeshm;
  if (in_to->use_intranodeshm) {
    /* to better understand how we copy a vecscatter context with shared memory support, please see comments
       in VecScatterCreate_PtoS. The shared memory allocation, and state address and data address calculation
       code is almost the same.
     */
    PetscInt         i,nn;
    MPI_Info         info;
    PetscObjectState state;
    char             *shmspace;
    MPI_Comm         comm;
    PetscShmComm     pshmcomm;
    MPI_Aint         sz;
    MPI_Request      *reqs = NULL;
    struct {PetscInt j,m,offset;} jmo,*triples = NULL;

    ierr = PetscObjectStateGet((PetscObject)out,&state);CHKERRQ(ierr);
    ierr = PetscObjectGetComm((PetscObject)out,&comm);CHKERRQ(ierr);
    ierr = PetscShmCommGet(comm,&pshmcomm);CHKERRQ(ierr);

    /* copy the to parts for intranode shared memory communication */
    out_to->shmn    = in_to->shmn;
    out_to->shmcomm = in_to->shmcomm; /* shmcomm is destroyed only when the outer PETSc communicator is freed, so a simple copy (other than MPI_Comm_dup) is fine */
    /* out_to->shmwin will be re-created and we can not copy in_to->shmwin, since we need new shared memory */

    ierr = PetscMalloc5(out_to->shmn,&out_to->shmprocs,out_to->shmn,&out_to->shmspaces,out_to->shmn,&out_to->shmstates,out_to->shmn+1,&out_to->shmstarts,in_to->shmstarts[in_to->shmn],&out_to->shmindices);CHKERRQ(ierr);
    ierr = PetscMemcpy(out_to->shmprocs,in_to->shmprocs,sizeof(PetscMPIInt)*out_to->shmn);CHKERRQ(ierr);
    ierr = PetscMemcpy(out_to->shmstarts,in_to->shmstarts,sizeof(PetscInt)*(out_to->shmn+1));CHKERRQ(ierr);
    ierr = PetscMemcpy(out_to->shmindices,in_to->shmindices,sizeof(PetscInt)*in_to->shmstarts[in_to->shmn]);CHKERRQ(ierr);
    /* allocate sharem memory in to, which has shmspace[] and ignores shmspaces[] */
    ierr = MPI_Info_create(&info);CHKERRQ(ierr);
    ierr = MPI_Info_set(info, "alloc_shared_noncontig", "true");CHKERRQ(ierr);
    sz   = bs*out_to->shmstarts[out_to->shmn]*sizeof(PetscScalar) + (out_to->shmn+1)*PETSC_LEVEL1_DCACHE_LINESIZE;
    ierr = MPI_Win_allocate_shared(sz,sizeof(PetscScalar),info,out_to->shmcomm,&shmspace,&out_to->shmwin);CHKERRQ(ierr);
#if !defined(PETSC_WRITE_MEMORY_BARRIER)
    ierr = MPI_Win_lock_all(MPI_MODE_NOCHECK,out_to->shmwin);CHKERRQ(ierr); /* see comments in VecScatterBegin for why this call */
#endif
    ierr = MPI_Info_free(&info);CHKERRQ(ierr);
    shmspace = (char*)((((PETSC_UINTPTR_T)(shmspace))+(PETSC_LEVEL1_DCACHE_LINESIZE-1)) & ~(PETSC_LEVEL1_DCACHE_LINESIZE-1));
    for (i=0; i<out_to->shmn; i++) {
       out_to->shmstates[i] = (PetscObjectState*)(shmspace + i*PETSC_LEVEL1_DCACHE_LINESIZE);
      *out_to->shmstates[i] = state; /* init the flag to empty(0) */
    }
    shmspace += out_to->shmn*PETSC_LEVEL1_DCACHE_LINESIZE;
    for (i=0; i<out_to->shmn; i++) out_to->shmspaces[i] = (PetscScalar*)shmspace + bs*out_to->shmstarts[i];

    /* copy the from parts for intranode shared memory communication */
    out_from->shmn    = in_from->shmn;
    out_from->shmcomm = in_from->shmcomm;
    out_from->shmwin  = out_to->shmwin; /* shmwin is the same between to/from */

    nn   = out_from->shmn;
    ierr = PetscMalloc5(nn,&out_from->shmprocs,nn,&out_from->shmspaces,nn,&out_from->shmstates,nn+1,&out_from->shmstarts,in_from->shmstarts[in_from->shmn],&out_from->shmindices);CHKERRQ(ierr);
    ierr = PetscMemcpy(out_from->shmprocs,in_from->shmprocs,sizeof(PetscMPIInt)*out_from->shmn);CHKERRQ(ierr);
    ierr = PetscMemcpy(out_from->shmstarts,in_from->shmstarts,sizeof(PetscInt)*(out_from->shmn+1));CHKERRQ(ierr);
    ierr = PetscMemcpy(out_from->shmindices,in_from->shmindices,sizeof(PetscInt)*in_from->shmstarts[in_from->shmn]);CHKERRQ(ierr);

    ierr = PetscMalloc2(out_from->shmn,&reqs,out_from->shmn,&triples);CHKERRQ(ierr);
    for (i=0; i<out_from->shmn; i++) { ierr = MPI_Irecv(triples+i,3,MPIU_INT,out_from->shmprocs[i],0/*tag*/,comm,reqs+i);CHKERRQ(ierr); }
    for (i=0; i<out_to->shmn; i++) {
      jmo.j = i;
      jmo.m = out_to->shmn;
      jmo.offset = out_to->shmstarts[i];
      ierr = MPI_Send(&jmo,3,MPIU_INT,out_to->shmprocs[i],0/*tag*/,comm);CHKERRQ(ierr);
    }
    ierr = MPI_Waitall(out_from->shmn,reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);

    /* figure out state addresses and data addresses aimed to me */
    for (i=0; i<out_from->shmn; i++) {
      MPI_Aint    size;
      PetscMPIInt disp_unit,lrank;
      ierr = PetscShmCommGlobalToLocal(pshmcomm,out_from->shmprocs[i],&lrank);CHKERRQ(ierr);
      ierr = MPI_Win_shared_query(out_from->shmwin,lrank,&size,&disp_unit,&out_from->shmspaces[i]);CHKERRQ(ierr);
      out_from->shmspaces[i]  = (PetscScalar*)((((PETSC_UINTPTR_T)(out_from->shmspaces[i]))+(PETSC_LEVEL1_DCACHE_LINESIZE-1)) & ~(PETSC_LEVEL1_DCACHE_LINESIZE-1));
      out_from->shmstates[i]  = (PetscObjectState*)((char*)out_from->shmspaces[i] + triples[i].j*PETSC_LEVEL1_DCACHE_LINESIZE); /* get address of the j-th state */
      out_from->shmspaces[i]  = (PetscScalar*)((char*)out_from->shmspaces[i] + triples[i].m*PETSC_LEVEL1_DCACHE_LINESIZE); /* skip the state area */
      out_from->shmspaces[i] += triples[i].offset*bs;/* and then add the offset to point to where my expected data lives */
    }

    ierr = PetscFree2(reqs,triples);CHKERRQ(ierr);
  }
#endif

  ierr = VecScatterMemcpyPlanCopy_PtoP(in_to,in_from,out_to,out_from);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Optimize a parallel vector to parallel vector vecscatter with memory copies */
PetscErrorCode VecScatterMemcpyPlanCreate_PtoP(VecScatter_MPI_General *to,VecScatter_MPI_General *from)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterMemcpyPlanCreate_Index(to->n,to->starts,to->indices,to->bs,&to->memcpy_plan);CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanCreate_Index(from->n,from->starts,from->indices,to->bs,&from->memcpy_plan);CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanCreate_SGToSG(to->bs,&to->local,&from->local);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  if (to->use_intranodeshm) {
    ierr = VecScatterMemcpyPlanCreate_Index(to->shmn,to->shmstarts,to->shmindices,to->bs,&to->shm_memcpy_plan);CHKERRQ(ierr);
    ierr = VecScatterMemcpyPlanCreate_Index(from->shmn,from->shmstarts,from->shmindices,to->bs,&from->shm_memcpy_plan);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterMemcpyPlanCopy_PtoP(const VecScatter_MPI_General *in_to,const VecScatter_MPI_General *in_from,VecScatter_MPI_General *out_to,VecScatter_MPI_General *out_from)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterMemcpyPlanCopy(&in_to->memcpy_plan,&out_to->memcpy_plan);CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanCopy(&in_from->memcpy_plan,&out_from->memcpy_plan);CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanCopy(&in_to->local.memcpy_plan,&out_to->local.memcpy_plan);CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanCopy(&in_from->local.memcpy_plan,&out_from->local.memcpy_plan);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  if (in_to->use_intranodeshm) {
    /* also copy the optimization plan */
    ierr = VecScatterMemcpyPlanCopy(&in_to->shm_memcpy_plan,&out_to->shm_memcpy_plan);CHKERRQ(ierr);
    ierr = VecScatterMemcpyPlanCopy(&in_from->shm_memcpy_plan,&out_from->shm_memcpy_plan);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterMemcpyPlanDestroy_PtoP(VecScatter_MPI_General *to,VecScatter_MPI_General *from)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterMemcpyPlanDestroy(&to->memcpy_plan);CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanDestroy(&from->memcpy_plan);CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanDestroy(&to->local.memcpy_plan);CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanDestroy(&from->local.memcpy_plan);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  if (to->use_intranodeshm) {
    ierr = VecScatterMemcpyPlanDestroy(&to->shm_memcpy_plan);CHKERRQ(ierr);
    ierr = VecScatterMemcpyPlanDestroy(&from->shm_memcpy_plan);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

/* Given a PtoP VecScatter context, return number of procs and vector entries as if the communication can be done solely in send/recv

  Input Parameters:
+ ctx - the context
- to  - true to select the todata, otherwise to select the fromdata

  Output parameters:
+ num_procs   - number of remote processors
- num_entries - number of vector entries to send/recv

 */
PetscErrorCode VecScatterGetNumberOfOffProcsAndEntries_Private(const VecScatter ctx,PetscBool to,PetscInt *num_procs,PetscInt *num_entries)
{
  VecScatter_MPI_General *vs = (VecScatter_MPI_General*)(to ? ctx->todata : ctx->fromdata);

  PetscFunctionBegin;
  if (vs->format != VEC_SCATTER_MPI_GENERAL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"ctx must be a VecScatter_MPI_General");
  *num_procs   = vs->n;
  *num_entries = vs->starts[vs->n];
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  if (vs->use_intranodeshm) {
    *num_procs   += vs->shmn;
    *num_entries += vs->shmstarts[vs->shmn];
  }
#endif
  PetscFunctionReturn(0);
}

/* Given a VecScatter_MPI_General, return a communication plan that is good for use with MPI send/recv.
   Any output parameter can be NULL.

  Input Parameters:
+ ctx - the context
- to  - true to select the todata, otherwise to select the fromdata

  Output parameters:
+ n        - number of remote processors
. starts   - starting point in indices for each proc
. indices  - indices of entries to send/recv
. procs    - remote processors
. requests - MPI_requets
- bs       - block size

  Notes:
   Sometimes PETSc needs to use the matrix-vector-multiply vecscatter context for other purposes. The client code
   usually only uses MPI_Send/Recv. This subroutine returns a communication plan suitable for such uses.

   I do not output the MPI_Status field (sstatus) of VecScatter_MPI_General, because 1) PETSc has weird design
   for this field for reason I dont know (e.g., only todata sets this field, and fromdata does not); 2) the field
   is not important since PETSc does not check return values in it, and one can use MPI_STATUSES_IGNORE instead.
 */
PetscErrorCode VecScatterGetOffProcCommunicationPlan_Private(VecScatter ctx,PetscBool to,PetscInt *n,PetscInt **starts,PetscInt **indices,PetscMPIInt **procs,MPI_Request **requests,PetscInt *bs)
{
  PetscErrorCode  ierr;
  VecScatter_MPI_General *vs = (VecScatter_MPI_General*)(to ? ctx->todata : ctx->fromdata);

  PetscFunctionBegin;
  if (vs->format != VEC_SCATTER_MPI_GENERAL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"ctx must be a VecScatter_MPI_General");
  vs->work_starts   = NULL;
  vs->work_indices  = NULL;
  vs->work_procs    = NULL;
  vs->work_requests = NULL;

#ifdef PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY
  if (vs->use_intranodeshm && vs->shmn) {
    PetscInt i;
    if (n) *n  = vs->n + vs->shmn;

    if (starts) { /* concat starts */
      ierr     = PetscMalloc1(vs->n+vs->shmn+1,&vs->work_starts);CHKERRQ(ierr);
      ierr     = PetscMemcpy(vs->work_starts,vs->starts,sizeof(PetscInt)*(vs->n+1));CHKERRQ(ierr);
      for (i=0; i<vs->shmn; i++) vs->work_starts[vs->n+i+1] = vs->work_starts[vs->n] + vs->shmstarts[i+1];
      *starts  = vs->work_starts;
    }

    if (indices) { /* concat indices */
      ierr     = PetscMalloc1(vs->starts[vs->n]+vs->shmstarts[vs->shmn],&vs->work_indices);CHKERRQ(ierr);
      ierr     = PetscMemcpy(vs->work_indices,vs->indices,sizeof(PetscInt)*vs->starts[vs->n]);CHKERRQ(ierr);
      ierr     = PetscMemcpy(vs->work_indices+vs->starts[vs->n],vs->shmindices,sizeof(PetscInt)*vs->shmstarts[vs->shmn]);CHKERRQ(ierr);
      *indices = vs->work_indices;
    }

    if (procs) { /* concat procs */
      ierr   = PetscMalloc1(vs->n+vs->shmn,&vs->work_procs);CHKERRQ(ierr);
      ierr   = PetscMemcpy(vs->work_procs,vs->procs,sizeof(PetscMPIInt)*vs->n);CHKERRQ(ierr);
      ierr   = PetscMemcpy(vs->work_procs+vs->n,vs->shmprocs,sizeof(PetscMPIInt)*vs->shmn);CHKERRQ(ierr);
      *procs = vs->work_procs;
    }

    if (requests) {
      ierr      = PetscMalloc1(vs->n+vs->shmn,&vs->work_requests);CHKERRQ(ierr);
      *requests = vs->work_requests;
    }
  } else
#endif
  {
    if (n)        *n        = vs->n;
    if (indices)  *indices  = vs->indices;
    if (starts)   *starts   = vs->starts;
    if (procs)    *procs    = vs->procs;
    if (requests) *requests = vs->requests;
  }
  if (bs) *bs = vs->bs;
  PetscFunctionReturn(0);
}

/* Like VecScatterGetOffProcCommunicationPlan_Private, except procs returned by this routine is sorted. starts, indices are also adapted for the sorted procs.
 */
PetscErrorCode VecScatterGetOffProcCommunicationPlanWithSortedProcs_Private(VecScatter ctx,PetscBool to,PetscInt *n,PetscInt **starts,PetscInt **indices,PetscMPIInt **procs,MPI_Request **requests,PetscInt *bs)
{
  PetscErrorCode  ierr;
  VecScatter_MPI_General *vs = (VecScatter_MPI_General*)(to ? ctx->todata : ctx->fromdata);

  PetscFunctionBegin;
  if (vs->format != VEC_SCATTER_MPI_GENERAL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"ctx must be a VecScatter_MPI_General");
  vs->work_starts   = NULL;
  vs->work_indices  = NULL;
  vs->work_procs    = NULL;
  vs->work_requests = NULL;

#ifdef PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY
  if (vs->use_intranodeshm && vs->shmn) {
    PetscInt i,j,k;
    if (n) *n  = vs->n + vs->shmn;

    /* merge sort procs with starts. Always allocate work_starts/procs (which are much smaller than work_indices) to ease the sorting */
    ierr   = PetscMalloc1(vs->n+vs->shmn+1,&vs->work_starts);CHKERRQ(ierr);
    ierr   = PetscMalloc1(vs->n+vs->shmn,&vs->work_procs);CHKERRQ(ierr);

    vs->work_starts[0] = 0;
    for (i=0,j=0,k=0; i<vs->n && j <vs->shmn; k++) {
      if (vs->procs[i] < vs->shmprocs[j]) {
        vs->work_procs[k]    = vs->procs[i];
        vs->work_starts[k+1] = vs->work_starts[k] + vs->starts[i+1] - vs->starts[i];
        i++;
      } else {
        vs->work_procs[k]    = vs->shmprocs[j];
        vs->work_starts[k+1] = vs->work_starts[k] + vs->shmstarts[j+1] - vs->shmstarts[j];
        j++;
      }
    }

    for (; i<vs->n; i++,k++) {
      vs->work_procs[k]    = vs->procs[i];
      vs->work_starts[k+1] = vs->work_starts[k] + vs->starts[i+1] - vs->starts[i];
    }

    for (; j<vs->shmn; j++,k++) {
      vs->work_procs[k]    = vs->shmprocs[j];
      vs->work_starts[k+1] = vs->work_starts[k] + vs->shmstarts[j+1] - vs->shmstarts[j];
    }

    if (starts) *starts = vs->work_starts;
    if (procs)  *procs  = vs->work_procs;

    /* conditionally fill in indices[] */
    if (indices) {
      ierr = PetscMalloc1(vs->work_starts[vs->n+vs->shmn],&vs->work_indices);CHKERRQ(ierr);
      for (i=0; i<vs->n; i++) {
        ierr = PetscFindInt(vs->procs[i],vs->n+vs->shmn,vs->work_procs,&j);CHKERRQ(ierr);
        ierr = PetscMemcpy(vs->work_indices+vs->work_starts[j],vs->indices+vs->starts[i],sizeof(PetscInt)*(vs->starts[i+1]-vs->starts[i]));CHKERRQ(ierr);
      }
      for (i=0; i<vs->shmn; i++) {
        ierr = PetscFindInt(vs->shmprocs[i],vs->n+vs->shmn,vs->work_procs,&j);CHKERRQ(ierr);
        ierr = PetscMemcpy(vs->work_indices+vs->work_starts[j],vs->shmindices+vs->shmstarts[i],sizeof(PetscInt)*(vs->shmstarts[i+1]-vs->shmstarts[i]));CHKERRQ(ierr);
      }
      *indices = vs->work_indices;
    }

    if (requests) {
      ierr      = PetscMalloc1(vs->n+vs->shmn,&vs->work_requests);CHKERRQ(ierr);
      *requests = vs->work_requests;
    }
  } else
#endif
  {
    if (n)        *n        = vs->n;
    if (indices)  *indices  = vs->indices;
    if (starts)   *starts   = vs->starts;
    if (procs)    *procs    = vs->procs;
    if (requests) *requests = vs->requests;
  }
  if (bs) *bs = vs->bs;
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterRestoreOffProcCommunicationPlan_Private(VecScatter ctx,PetscBool to,PetscInt *n,PetscInt **starts,PetscInt **indices,PetscMPIInt **procs,MPI_Request **requests,PetscInt *bs)
{
  PetscErrorCode  ierr;
  VecScatter_MPI_General *vs = (VecScatter_MPI_General*)(to ? ctx->todata : ctx->fromdata);

  PetscFunctionBegin;
  if (vs->format != VEC_SCATTER_MPI_GENERAL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"ctx must be a VecScatter_MPI_General");
  ierr = PetscFree(vs->work_starts);CHKERRQ(ierr);
  ierr = PetscFree(vs->work_indices);CHKERRQ(ierr);
  ierr = PetscFree(vs->work_procs);CHKERRQ(ierr);
  ierr = PetscFree(vs->work_requests);CHKERRQ(ierr);

  if (starts)   *starts   = NULL;
  if (indices)  *indices  = NULL;
  if (procs)    *procs    = NULL;
  if (requests) *requests = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterRestoreOffProcCommunicationPlanWithSortedProcs_Private(VecScatter ctx,PetscBool to,PetscInt *n,PetscInt **starts,PetscInt **indices,PetscMPIInt **procs,MPI_Request **requests,PetscInt *bs)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = VecScatterRestoreOffProcCommunicationPlan_Private(ctx,to,n,starts,indices,procs,requests,bs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------------------
    Packs and unpacks the message data into send or from receive buffers.

    These could be generated automatically.

    Fortran kernels etc. could be used.
*/
PETSC_STATIC_INLINE void Pack_MPI1_1(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i;
  for (i=0; i<n; i++) y[i] = x[indicesx[i]];
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_MPI1_1(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) y[indicesy[i]] = x[i];
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) y[indicesy[i]] += x[i];
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) y[indicesy[i]] = PetscMax(y[indicesy[i]],x[i]);
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_MPI1_1(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) y[indicesy[i]] = x[indicesx[i]];
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) y[indicesy[i]] += x[indicesx[i]];
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) y[indicesy[i]] = PetscMax(y[indicesy[i]],x[indicesx[i]]);
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_MPI1_2(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y   += 2;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_MPI1_2(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      x       += 2;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      x        += 2;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      x       += 2;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_MPI1_2(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_MPI1_3(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y[2] = x[idx+2];
    y   += 3;
  }
}
PETSC_STATIC_INLINE PetscErrorCode UnPack_MPI1_3(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      y[idy+2] = x[2];
      x       += 3;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      y[idy+2] += x[2];
      x        += 3;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      y[idy+2] = PetscMax(y[idy+2],x[2]);
      x       += 3;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_MPI1_3(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
      y[idy+2] = x[idx+2];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
      y[idy+2] += x[idx+2];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2] = PetscMax(y[idy+2],x[idx+2]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_MPI1_4(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y[2] = x[idx+2];
    y[3] = x[idx+3];
    y   += 4;
  }
}
PETSC_STATIC_INLINE PetscErrorCode UnPack_MPI1_4(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      y[idy+2] = x[2];
      y[idy+3] = x[3];
      x       += 4;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      y[idy+2] += x[2];
      y[idy+3] += x[3];
      x        += 4;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      y[idy+2] = PetscMax(y[idy+2],x[2]);
      y[idy+3] = PetscMax(y[idy+3],x[3]);
      x       += 4;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_MPI1_4(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
      y[idy+2] = x[idx+2];
      y[idy+3] = x[idx+3];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
      y[idy+2] += x[idx+2];
      y[idy+3] += x[idx+3];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2] = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3] = PetscMax(y[idy+3],x[idx+3]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_MPI1_5(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y[2] = x[idx+2];
    y[3] = x[idx+3];
    y[4] = x[idx+4];
    y   += 5;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_MPI1_5(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      y[idy+2] = x[2];
      y[idy+3] = x[3];
      y[idy+4] = x[4];
      x       += 5;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      y[idy+2] += x[2];
      y[idy+3] += x[3];
      y[idy+4] += x[4];
      x        += 5;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      y[idy+2] = PetscMax(y[idy+2],x[2]);
      y[idy+3] = PetscMax(y[idy+3],x[3]);
      y[idy+4] = PetscMax(y[idy+4],x[4]);
      x       += 5;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_MPI1_5(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
      y[idy+2] = x[idx+2];
      y[idy+3] = x[idx+3];
      y[idy+4] = x[idx+4];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
      y[idy+2] += x[idx+2];
      y[idy+3] += x[idx+3];
      y[idy+4] += x[idx+4];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2] = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3] = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4] = PetscMax(y[idy+4],x[idx+4]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_MPI1_6(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y[2] = x[idx+2];
    y[3] = x[idx+3];
    y[4] = x[idx+4];
    y[5] = x[idx+5];
    y   += 6;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_MPI1_6(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      y[idy+2] = x[2];
      y[idy+3] = x[3];
      y[idy+4] = x[4];
      y[idy+5] = x[5];
      x       += 6;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      y[idy+2] += x[2];
      y[idy+3] += x[3];
      y[idy+4] += x[4];
      y[idy+5] += x[5];
      x        += 6;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      y[idy+2] = PetscMax(y[idy+2],x[2]);
      y[idy+3] = PetscMax(y[idy+3],x[3]);
      y[idy+4] = PetscMax(y[idy+4],x[4]);
      y[idy+5] = PetscMax(y[idy+5],x[5]);
      x       += 6;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_MPI1_6(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
      y[idy+2] = x[idx+2];
      y[idy+3] = x[idx+3];
      y[idy+4] = x[idx+4];
      y[idy+5] = x[idx+5];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
      y[idy+2] += x[idx+2];
      y[idy+3] += x[idx+3];
      y[idy+4] += x[idx+4];
      y[idy+5] += x[idx+5];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2] = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3] = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4] = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5] = PetscMax(y[idy+5],x[idx+5]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_MPI1_7(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y[2] = x[idx+2];
    y[3] = x[idx+3];
    y[4] = x[idx+4];
    y[5] = x[idx+5];
    y[6] = x[idx+6];
    y   += 7;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_MPI1_7(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      y[idy+2] = x[2];
      y[idy+3] = x[3];
      y[idy+4] = x[4];
      y[idy+5] = x[5];
      y[idy+6] = x[6];
      x       += 7;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      y[idy+2] += x[2];
      y[idy+3] += x[3];
      y[idy+4] += x[4];
      y[idy+5] += x[5];
      y[idy+6] += x[6];
      x        += 7;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      y[idy+2] = PetscMax(y[idy+2],x[2]);
      y[idy+3] = PetscMax(y[idy+3],x[3]);
      y[idy+4] = PetscMax(y[idy+4],x[4]);
      y[idy+5] = PetscMax(y[idy+5],x[5]);
      y[idy+6] = PetscMax(y[idy+6],x[6]);
      x       += 7;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_MPI1_7(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
      y[idy+2] = x[idx+2];
      y[idy+3] = x[idx+3];
      y[idy+4] = x[idx+4];
      y[idy+5] = x[idx+5];
      y[idy+6] = x[idx+6];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
      y[idy+2] += x[idx+2];
      y[idy+3] += x[idx+3];
      y[idy+4] += x[idx+4];
      y[idy+5] += x[idx+5];
      y[idy+6] += x[idx+6];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2] = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3] = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4] = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5] = PetscMax(y[idy+5],x[idx+5]);
      y[idy+6] = PetscMax(y[idy+6],x[idx+6]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_MPI1_8(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y[2] = x[idx+2];
    y[3] = x[idx+3];
    y[4] = x[idx+4];
    y[5] = x[idx+5];
    y[6] = x[idx+6];
    y[7] = x[idx+7];
    y   += 8;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_MPI1_8(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      y[idy+2] = x[2];
      y[idy+3] = x[3];
      y[idy+4] = x[4];
      y[idy+5] = x[5];
      y[idy+6] = x[6];
      y[idy+7] = x[7];
      x       += 8;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      y[idy+2] += x[2];
      y[idy+3] += x[3];
      y[idy+4] += x[4];
      y[idy+5] += x[5];
      y[idy+6] += x[6];
      y[idy+7] += x[7];
      x        += 8;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      y[idy+2] = PetscMax(y[idy+2],x[2]);
      y[idy+3] = PetscMax(y[idy+3],x[3]);
      y[idy+4] = PetscMax(y[idy+4],x[4]);
      y[idy+5] = PetscMax(y[idy+5],x[5]);
      y[idy+6] = PetscMax(y[idy+6],x[6]);
      y[idy+7] = PetscMax(y[idy+7],x[7]);
      x       += 8;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_MPI1_8(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
      y[idy+2] = x[idx+2];
      y[idy+3] = x[idx+3];
      y[idy+4] = x[idx+4];
      y[idy+5] = x[idx+5];
      y[idy+6] = x[idx+6];
      y[idy+7] = x[idx+7];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
      y[idy+2] += x[idx+2];
      y[idy+3] += x[idx+3];
      y[idy+4] += x[idx+4];
      y[idy+5] += x[idx+5];
      y[idy+6] += x[idx+6];
      y[idy+7] += x[idx+7];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2] = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3] = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4] = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5] = PetscMax(y[idy+5],x[idx+5]);
      y[idy+6] = PetscMax(y[idy+6],x[idx+6]);
      y[idy+7] = PetscMax(y[idy+7],x[idx+7]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE void Pack_MPI1_9(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx   = *indicesx++;
    y[0]  = x[idx];
    y[1]  = x[idx+1];
    y[2]  = x[idx+2];
    y[3]  = x[idx+3];
    y[4]  = x[idx+4];
    y[5]  = x[idx+5];
    y[6]  = x[idx+6];
    y[7]  = x[idx+7];
    y[8]  = x[idx+8];
    y    += 9;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_MPI1_9(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = x[0];
      y[idy+1]  = x[1];
      y[idy+2]  = x[2];
      y[idy+3]  = x[3];
      y[idy+4]  = x[4];
      y[idy+5]  = x[5];
      y[idy+6]  = x[6];
      y[idy+7]  = x[7];
      y[idy+8]  = x[8];
      x        += 9;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy        = *indicesy++;
      y[idy]    += x[0];
      y[idy+1]  += x[1];
      y[idy+2]  += x[2];
      y[idy+3]  += x[3];
      y[idy+4]  += x[4];
      y[idy+5]  += x[5];
      y[idy+6]  += x[6];
      y[idy+7]  += x[7];
      y[idy+8]  += x[8];
      x         += 9;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[0]);
      y[idy+1]  = PetscMax(y[idy+1],x[1]);
      y[idy+2]  = PetscMax(y[idy+2],x[2]);
      y[idy+3]  = PetscMax(y[idy+3],x[3]);
      y[idy+4]  = PetscMax(y[idy+4],x[4]);
      y[idy+5]  = PetscMax(y[idy+5],x[5]);
      y[idy+6]  = PetscMax(y[idy+6],x[6]);
      y[idy+7]  = PetscMax(y[idy+7],x[7]);
      y[idy+8]  = PetscMax(y[idy+8],x[8]);
      x        += 9;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_MPI1_9(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = x[idx];
      y[idy+1]  = x[idx+1];
      y[idy+2]  = x[idx+2];
      y[idy+3]  = x[idx+3];
      y[idy+4]  = x[idx+4];
      y[idy+5]  = x[idx+5];
      y[idy+6]  = x[idx+6];
      y[idy+7]  = x[idx+7];
      y[idy+8]  = x[idx+8];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx        = *indicesx++;
      idy        = *indicesy++;
      y[idy]    += x[idx];
      y[idy+1]  += x[idx+1];
      y[idy+2]  += x[idx+2];
      y[idy+3]  += x[idx+3];
      y[idy+4]  += x[idx+4];
      y[idy+5]  += x[idx+5];
      y[idy+6]  += x[idx+6];
      y[idy+7]  += x[idx+7];
      y[idy+8]  += x[idx+8];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[idx]);
      y[idy+1]  = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2]  = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3]  = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4]  = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5]  = PetscMax(y[idy+5],x[idx+5]);
      y[idy+6]  = PetscMax(y[idy+6],x[idx+6]);
      y[idy+7]  = PetscMax(y[idy+7],x[idx+7]);
      y[idy+8]  = PetscMax(y[idy+8],x[idx+8]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE void Pack_MPI1_10(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx   = *indicesx++;
    y[0]  = x[idx];
    y[1]  = x[idx+1];
    y[2]  = x[idx+2];
    y[3]  = x[idx+3];
    y[4]  = x[idx+4];
    y[5]  = x[idx+5];
    y[6]  = x[idx+6];
    y[7]  = x[idx+7];
    y[8]  = x[idx+8];
    y[9]  = x[idx+9];
    y    += 10;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_MPI1_10(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = x[0];
      y[idy+1]  = x[1];
      y[idy+2]  = x[2];
      y[idy+3]  = x[3];
      y[idy+4]  = x[4];
      y[idy+5]  = x[5];
      y[idy+6]  = x[6];
      y[idy+7]  = x[7];
      y[idy+8]  = x[8];
      y[idy+9]  = x[9];
      x        += 10;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy        = *indicesy++;
      y[idy]    += x[0];
      y[idy+1]  += x[1];
      y[idy+2]  += x[2];
      y[idy+3]  += x[3];
      y[idy+4]  += x[4];
      y[idy+5]  += x[5];
      y[idy+6]  += x[6];
      y[idy+7]  += x[7];
      y[idy+8]  += x[8];
      y[idy+9]  += x[9];
      x         += 10;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[0]);
      y[idy+1]  = PetscMax(y[idy+1],x[1]);
      y[idy+2]  = PetscMax(y[idy+2],x[2]);
      y[idy+3]  = PetscMax(y[idy+3],x[3]);
      y[idy+4]  = PetscMax(y[idy+4],x[4]);
      y[idy+5]  = PetscMax(y[idy+5],x[5]);
      y[idy+6]  = PetscMax(y[idy+6],x[6]);
      y[idy+7]  = PetscMax(y[idy+7],x[7]);
      y[idy+8]  = PetscMax(y[idy+8],x[8]);
      y[idy+9]  = PetscMax(y[idy+9],x[9]);
      x        += 10;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_MPI1_10(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = x[idx];
      y[idy+1]  = x[idx+1];
      y[idy+2]  = x[idx+2];
      y[idy+3]  = x[idx+3];
      y[idy+4]  = x[idx+4];
      y[idy+5]  = x[idx+5];
      y[idy+6]  = x[idx+6];
      y[idy+7]  = x[idx+7];
      y[idy+8]  = x[idx+8];
      y[idy+9]  = x[idx+9];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx        = *indicesx++;
      idy        = *indicesy++;
      y[idy]    += x[idx];
      y[idy+1]  += x[idx+1];
      y[idy+2]  += x[idx+2];
      y[idy+3]  += x[idx+3];
      y[idy+4]  += x[idx+4];
      y[idy+5]  += x[idx+5];
      y[idy+6]  += x[idx+6];
      y[idy+7]  += x[idx+7];
      y[idy+8]  += x[idx+8];
      y[idy+9]  += x[idx+9];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[idx]);
      y[idy+1]  = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2]  = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3]  = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4]  = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5]  = PetscMax(y[idy+5],x[idx+5]);
      y[idy+6]  = PetscMax(y[idy+6],x[idx+6]);
      y[idy+7]  = PetscMax(y[idy+7],x[idx+7]);
      y[idy+8]  = PetscMax(y[idy+8],x[idx+8]);
      y[idy+9]  = PetscMax(y[idy+9],x[idx+9]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE void Pack_MPI1_11(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx   = *indicesx++;
    y[0]  = x[idx];
    y[1]  = x[idx+1];
    y[2]  = x[idx+2];
    y[3]  = x[idx+3];
    y[4]  = x[idx+4];
    y[5]  = x[idx+5];
    y[6]  = x[idx+6];
    y[7]  = x[idx+7];
    y[8]  = x[idx+8];
    y[9]  = x[idx+9];
    y[10] = x[idx+10];
    y    += 11;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_MPI1_11(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = x[0];
      y[idy+1]  = x[1];
      y[idy+2]  = x[2];
      y[idy+3]  = x[3];
      y[idy+4]  = x[4];
      y[idy+5]  = x[5];
      y[idy+6]  = x[6];
      y[idy+7]  = x[7];
      y[idy+8]  = x[8];
      y[idy+9]  = x[9];
      y[idy+10] = x[10];
      x        += 11;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy        = *indicesy++;
      y[idy]    += x[0];
      y[idy+1]  += x[1];
      y[idy+2]  += x[2];
      y[idy+3]  += x[3];
      y[idy+4]  += x[4];
      y[idy+5]  += x[5];
      y[idy+6]  += x[6];
      y[idy+7]  += x[7];
      y[idy+8]  += x[8];
      y[idy+9]  += x[9];
      y[idy+10] += x[10];
      x         += 11;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[0]);
      y[idy+1]  = PetscMax(y[idy+1],x[1]);
      y[idy+2]  = PetscMax(y[idy+2],x[2]);
      y[idy+3]  = PetscMax(y[idy+3],x[3]);
      y[idy+4]  = PetscMax(y[idy+4],x[4]);
      y[idy+5]  = PetscMax(y[idy+5],x[5]);
      y[idy+6]  = PetscMax(y[idy+6],x[6]);
      y[idy+7]  = PetscMax(y[idy+7],x[7]);
      y[idy+8]  = PetscMax(y[idy+8],x[8]);
      y[idy+9]  = PetscMax(y[idy+9],x[9]);
      y[idy+10] = PetscMax(y[idy+10],x[10]);
      x        += 11;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_MPI1_11(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = x[idx];
      y[idy+1]  = x[idx+1];
      y[idy+2]  = x[idx+2];
      y[idy+3]  = x[idx+3];
      y[idy+4]  = x[idx+4];
      y[idy+5]  = x[idx+5];
      y[idy+6]  = x[idx+6];
      y[idy+7]  = x[idx+7];
      y[idy+8]  = x[idx+8];
      y[idy+9]  = x[idx+9];
      y[idy+10] = x[idx+10];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx        = *indicesx++;
      idy        = *indicesy++;
      y[idy]    += x[idx];
      y[idy+1]  += x[idx+1];
      y[idy+2]  += x[idx+2];
      y[idy+3]  += x[idx+3];
      y[idy+4]  += x[idx+4];
      y[idy+5]  += x[idx+5];
      y[idy+6]  += x[idx+6];
      y[idy+7]  += x[idx+7];
      y[idy+8]  += x[idx+8];
      y[idy+9]  += x[idx+9];
      y[idy+10] += x[idx+10];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[idx]);
      y[idy+1]  = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2]  = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3]  = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4]  = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5]  = PetscMax(y[idy+5],x[idx+5]);
      y[idy+6]  = PetscMax(y[idy+6],x[idx+6]);
      y[idy+7]  = PetscMax(y[idy+7],x[idx+7]);
      y[idy+8]  = PetscMax(y[idy+8],x[idx+8]);
      y[idy+9]  = PetscMax(y[idy+9],x[idx+9]);
      y[idy+10] = PetscMax(y[idy+10],x[idx+10]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_MPI1_12(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx   = *indicesx++;
    y[0]  = x[idx];
    y[1]  = x[idx+1];
    y[2]  = x[idx+2];
    y[3]  = x[idx+3];
    y[4]  = x[idx+4];
    y[5]  = x[idx+5];
    y[6]  = x[idx+6];
    y[7]  = x[idx+7];
    y[8]  = x[idx+8];
    y[9]  = x[idx+9];
    y[10] = x[idx+10];
    y[11] = x[idx+11];
    y    += 12;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_MPI1_12(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = x[0];
      y[idy+1]  = x[1];
      y[idy+2]  = x[2];
      y[idy+3]  = x[3];
      y[idy+4]  = x[4];
      y[idy+5]  = x[5];
      y[idy+6]  = x[6];
      y[idy+7]  = x[7];
      y[idy+8]  = x[8];
      y[idy+9]  = x[9];
      y[idy+10] = x[10];
      y[idy+11] = x[11];
      x        += 12;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy        = *indicesy++;
      y[idy]    += x[0];
      y[idy+1]  += x[1];
      y[idy+2]  += x[2];
      y[idy+3]  += x[3];
      y[idy+4]  += x[4];
      y[idy+5]  += x[5];
      y[idy+6]  += x[6];
      y[idy+7]  += x[7];
      y[idy+8]  += x[8];
      y[idy+9]  += x[9];
      y[idy+10] += x[10];
      y[idy+11] += x[11];
      x         += 12;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[0]);
      y[idy+1]  = PetscMax(y[idy+1],x[1]);
      y[idy+2]  = PetscMax(y[idy+2],x[2]);
      y[idy+3]  = PetscMax(y[idy+3],x[3]);
      y[idy+4]  = PetscMax(y[idy+4],x[4]);
      y[idy+5]  = PetscMax(y[idy+5],x[5]);
      y[idy+6]  = PetscMax(y[idy+6],x[6]);
      y[idy+7]  = PetscMax(y[idy+7],x[7]);
      y[idy+8]  = PetscMax(y[idy+8],x[8]);
      y[idy+9]  = PetscMax(y[idy+9],x[9]);
      y[idy+10] = PetscMax(y[idy+10],x[10]);
      y[idy+11] = PetscMax(y[idy+11],x[11]);
      x        += 12;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_MPI1_12(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = x[idx];
      y[idy+1]  = x[idx+1];
      y[idy+2]  = x[idx+2];
      y[idy+3]  = x[idx+3];
      y[idy+4]  = x[idx+4];
      y[idy+5]  = x[idx+5];
      y[idy+6]  = x[idx+6];
      y[idy+7]  = x[idx+7];
      y[idy+8]  = x[idx+8];
      y[idy+9]  = x[idx+9];
      y[idy+10] = x[idx+10];
      y[idy+11] = x[idx+11];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx        = *indicesx++;
      idy        = *indicesy++;
      y[idy]    += x[idx];
      y[idy+1]  += x[idx+1];
      y[idy+2]  += x[idx+2];
      y[idy+3]  += x[idx+3];
      y[idy+4]  += x[idx+4];
      y[idy+5]  += x[idx+5];
      y[idy+6]  += x[idx+6];
      y[idy+7]  += x[idx+7];
      y[idy+8]  += x[idx+8];
      y[idy+9]  += x[idx+9];
      y[idy+10] += x[idx+10];
      y[idy+11] += x[idx+11];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[idx]);
      y[idy+1]  = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2]  = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3]  = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4]  = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5]  = PetscMax(y[idy+5],x[idx+5]);
      y[idy+6]  = PetscMax(y[idy+6],x[idx+6]);
      y[idy+7]  = PetscMax(y[idy+7],x[idx+7]);
      y[idy+8]  = PetscMax(y[idy+8],x[idx+8]);
      y[idy+9]  = PetscMax(y[idy+9],x[idx+9]);
      y[idy+10] = PetscMax(y[idy+10],x[idx+10]);
      y[idy+11] = PetscMax(y[idy+11],x[idx+11]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_MPI1_bs(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt       i,idx;

  for (i=0; i<n; i++) {
    idx   = *indicesx++;
    PetscMemcpy(y,x + idx,bs*sizeof(PetscScalar));
    y    += bs;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_MPI1_bs(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy,j;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      PetscMemcpy(y + idy,x,bs*sizeof(PetscScalar));
      x        += bs;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy        = *indicesy++;
      for (j=0; j<bs; j++) y[idy+j] += x[j];
      x         += bs;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy = *indicesy++;
      for (j=0; j<bs; j++) y[idy+j] = PetscMax(y[idy+j],x[j]);
      x  += bs;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_MPI1_bs(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy,j;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      PetscMemcpy(y + idy, x + idx,bs*sizeof(PetscScalar));
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx        = *indicesx++;
      idy        = *indicesy++;
      for (j=0; j<bs; j++ )  y[idy+j] += x[idx+j];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      for (j=0; j<bs; j++ )  y[idy+j] = PetscMax(y[idy+j],x[idx+j]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

/* Create the VecScatterBegin/End_P for our chosen block sizes */
#define BS 1
#include <../src/vec/vscat/impls/vpscat_mpi1.h>
#define BS 2
#include <../src/vec/vscat/impls/vpscat_mpi1.h>
#define BS 3
#include <../src/vec/vscat/impls/vpscat_mpi1.h>
#define BS 4
#include <../src/vec/vscat/impls/vpscat_mpi1.h>
#define BS 5
#include <../src/vec/vscat/impls/vpscat_mpi1.h>
#define BS 6
#include <../src/vec/vscat/impls/vpscat_mpi1.h>
#define BS 7
#include <../src/vec/vscat/impls/vpscat_mpi1.h>
#define BS 8
#include <../src/vec/vscat/impls/vpscat_mpi1.h>
#define BS 9
#include <../src/vec/vscat/impls/vpscat_mpi1.h>
#define BS 10
#include <../src/vec/vscat/impls/vpscat_mpi1.h>
#define BS 11
#include <../src/vec/vscat/impls/vpscat_mpi1.h>
#define BS 12
#include <../src/vec/vscat/impls/vpscat_mpi1.h>
#define BS bs
#include <../src/vec/vscat/impls/vpscat_mpi1.h>

/*
   bs indicates how many elements there are in each block. Normally this would be 1.
*/
PetscErrorCode VecScatterCreateCommon_PtoS_MPI1(VecScatter_MPI_General *from,VecScatter_MPI_General *to,VecScatter ctx)
{
  MPI_Comm       comm;
  PetscMPIInt    tag  = ((PetscObject)ctx)->tag, tagr;
  PetscInt       bs   = to->bs;
  PetscMPIInt    size;
  PetscInt       i, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ctx,&comm);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)ctx,&tagr);CHKERRQ(ierr);
  ctx->ops->destroy = VecScatterDestroy_PtoP_MPI1;
  ctx->ops->copy    = VecScatterCopy_PtoP_MPI1;

  ctx->reproduce = PETSC_FALSE;
  to->sendfirst  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_reproduce",&ctx->reproduce,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_sendfirst",&to->sendfirst,NULL);CHKERRQ(ierr);
  from->sendfirst = to->sendfirst;

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  /* check if the receives are ALL going into contiguous locations; if so can skip indexing */
  to->contiq = PETSC_FALSE;
  n = from->starts[from->n];
  from->contiq = PETSC_TRUE;
  for (i=1; i<n; i++) {
    if (from->indices[i] != from->indices[i-1] + bs) {
      from->contiq = PETSC_FALSE;
      break;
    }
  }

  to->use_alltoallv = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_alltoall",&to->use_alltoallv,NULL);CHKERRQ(ierr);
  from->use_alltoallv = to->use_alltoallv;
  if (from->use_alltoallv) {ierr = PetscInfo(ctx,"Using MPI_Alltoallv() for scatter\n");CHKERRQ(ierr);}
#if defined(PETSC_HAVE_MPI_ALLTOALLW)  && !defined(PETSC_USE_64BIT_INDICES)
  if (to->use_alltoallv) {
    to->use_alltoallw = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_nopack",&to->use_alltoallw,NULL);CHKERRQ(ierr);
  }
  from->use_alltoallw = to->use_alltoallw;
  if (from->use_alltoallw) {ierr = PetscInfo(ctx,"Using MPI_Alltoallw() for scatter\n");CHKERRQ(ierr);}
#endif

#if defined(PETSC_HAVE_MPI_WIN_CREATE_FEATURE)
  to->use_window = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_window",&to->use_window,NULL);CHKERRQ(ierr);
  from->use_window = to->use_window;
#endif

  if (to->use_alltoallv) {
    ierr       = PetscMalloc2(size,&to->counts,size,&to->displs);CHKERRQ(ierr);
    ierr       = PetscMemzero(to->counts,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
    for (i=0; i<to->n; i++) to->counts[to->procs[i]] = bs*(to->starts[i+1] - to->starts[i]);

    to->displs[0] = 0;
    for (i=1; i<size; i++) to->displs[i] = to->displs[i-1] + to->counts[i-1];

    ierr       = PetscMalloc2(size,&from->counts,size,&from->displs);CHKERRQ(ierr);
    ierr       = PetscMemzero(from->counts,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
    for (i=0; i<from->n; i++) from->counts[from->procs[i]] = bs*(from->starts[i+1] - from->starts[i]);
    from->displs[0] = 0;
    for (i=1; i<size; i++) from->displs[i] = from->displs[i-1] + from->counts[i-1];

#if defined(PETSC_HAVE_MPI_ALLTOALLW) && !defined(PETSC_USE_64BIT_INDICES)
    if (to->use_alltoallw) {
      PetscMPIInt mpibs, mpilen;

      ctx->packtogether = PETSC_FALSE;
      ierr = PetscMPIIntCast(bs,&mpibs);CHKERRQ(ierr);
      ierr = PetscMalloc3(size,&to->wcounts,size,&to->wdispls,size,&to->types);CHKERRQ(ierr);
      ierr = PetscMemzero(to->wcounts,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
      ierr = PetscMemzero(to->wdispls,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
      for (i=0; i<size; i++) to->types[i] = MPIU_SCALAR;

      for (i=0; i<to->n; i++) {
        to->wcounts[to->procs[i]] = 1;
        ierr = PetscMPIIntCast(to->starts[i+1]-to->starts[i],&mpilen);CHKERRQ(ierr);
        ierr = MPI_Type_create_indexed_block(mpilen,mpibs,to->indices+to->starts[i],MPIU_SCALAR,to->types+to->procs[i]);CHKERRQ(ierr);
        ierr = MPI_Type_commit(to->types+to->procs[i]);CHKERRQ(ierr);
      }
      ierr       = PetscMalloc3(size,&from->wcounts,size,&from->wdispls,size,&from->types);CHKERRQ(ierr);
      ierr       = PetscMemzero(from->wcounts,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
      ierr       = PetscMemzero(from->wdispls,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
      for (i=0; i<size; i++) from->types[i] = MPIU_SCALAR;

      if (from->contiq) {
        ierr = PetscInfo(ctx,"Scattered vector entries are stored contiguously, taking advantage of this with -vecscatter_alltoall\n");CHKERRQ(ierr);
        for (i=0; i<from->n; i++) from->wcounts[from->procs[i]] = bs*(from->starts[i+1] - from->starts[i]);

        if (from->n) from->wdispls[from->procs[0]] = sizeof(PetscScalar)*from->indices[0];
        for (i=1; i<from->n; i++) from->wdispls[from->procs[i]] = from->wdispls[from->procs[i-1]] + sizeof(PetscScalar)*from->wcounts[from->procs[i-1]];

      } else {
        for (i=0; i<from->n; i++) {
          from->wcounts[from->procs[i]] = 1;
          ierr = PetscMPIIntCast(from->starts[i+1]-from->starts[i],&mpilen);CHKERRQ(ierr);
          ierr = MPI_Type_create_indexed_block(mpilen,mpibs,from->indices+from->starts[i],MPIU_SCALAR,from->types+from->procs[i]);CHKERRQ(ierr);
          ierr = MPI_Type_commit(from->types+from->procs[i]);CHKERRQ(ierr);
        }
      }
    }

#else
    to->use_alltoallw   = PETSC_FALSE;
    from->use_alltoallw = PETSC_FALSE;
#endif
#if defined(PETSC_HAVE_MPI_WIN_CREATE_FEATURE)
  } else if (to->use_window) {
    PetscMPIInt temptag,winsize;
    MPI_Request *request;
    MPI_Status  *status;

    ierr = PetscObjectGetNewTag((PetscObject)ctx,&temptag);CHKERRQ(ierr);
    winsize = (to->n ? to->starts[to->n] : 0)*bs*sizeof(PetscScalar);
    ierr = MPI_Win_create(to->values ? to->values : MPI_BOTTOM,winsize,sizeof(PetscScalar),MPI_INFO_NULL,comm,&to->window);CHKERRQ(ierr);
    ierr = PetscMalloc1(to->n,&to->winstarts);CHKERRQ(ierr);
    ierr = PetscMalloc2(to->n,&request,to->n,&status);CHKERRQ(ierr);
    for (i=0; i<to->n; i++) {
      ierr = MPI_Irecv(to->winstarts+i,1,MPIU_INT,to->procs[i],temptag,comm,request+i);CHKERRQ(ierr);
    }
    for (i=0; i<from->n; i++) {
      ierr = MPI_Send(from->starts+i,1,MPIU_INT,from->procs[i],temptag,comm);CHKERRQ(ierr);
    }
    ierr = MPI_Waitall(to->n,request,status);CHKERRQ(ierr);
    ierr = PetscFree2(request,status);CHKERRQ(ierr);

    winsize = (from->n ? from->starts[from->n] : 0)*bs*sizeof(PetscScalar);
    ierr = MPI_Win_create(from->values ? from->values : MPI_BOTTOM,winsize,sizeof(PetscScalar),MPI_INFO_NULL,comm,&from->window);CHKERRQ(ierr);
    ierr = PetscMalloc1(from->n,&from->winstarts);CHKERRQ(ierr);
    ierr = PetscMalloc2(from->n,&request,from->n,&status);CHKERRQ(ierr);
    for (i=0; i<from->n; i++) {
      ierr = MPI_Irecv(from->winstarts+i,1,MPIU_INT,from->procs[i],temptag,comm,request+i);CHKERRQ(ierr);
    }
    for (i=0; i<to->n; i++) {
      ierr = MPI_Send(to->starts+i,1,MPIU_INT,to->procs[i],temptag,comm);CHKERRQ(ierr);
    }
    ierr = MPI_Waitall(from->n,request,status);CHKERRQ(ierr);
    ierr = PetscFree2(request,status);CHKERRQ(ierr);
#endif
  } else {
    PetscBool   use_rsend = PETSC_FALSE, use_ssend = PETSC_FALSE;
    PetscInt    *sstarts  = to->starts,  *rstarts = from->starts;
    PetscMPIInt *sprocs   = to->procs,   *rprocs  = from->procs;
    MPI_Request *swaits   = to->requests,*rwaits  = from->requests;
    MPI_Request *rev_swaits,*rev_rwaits;
    PetscScalar *Ssvalues = to->values, *Srvalues = from->values;

    /* allocate additional wait variables for the "reverse" scatter */
    ierr = PetscMalloc1(to->n,&rev_rwaits);CHKERRQ(ierr);
    ierr = PetscMalloc1(from->n,&rev_swaits);CHKERRQ(ierr);
    to->rev_requests   = rev_rwaits;
    from->rev_requests = rev_swaits;

    /* Register the receives that you will use later (sends for scatter reverse) */
    ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_rsend",&use_rsend,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_ssend",&use_ssend,NULL);CHKERRQ(ierr);
    if (use_rsend) {
      ierr = PetscInfo(ctx,"Using VecScatter ready receiver mode\n");CHKERRQ(ierr);
      to->use_readyreceiver   = PETSC_TRUE;
      from->use_readyreceiver = PETSC_TRUE;
    } else {
      to->use_readyreceiver   = PETSC_FALSE;
      from->use_readyreceiver = PETSC_FALSE;
    }
    if (use_ssend) {
      ierr = PetscInfo(ctx,"Using VecScatter Ssend mode\n");CHKERRQ(ierr);
    }

    for (i=0; i<from->n; i++) {
      if (use_rsend) {
        ierr = MPI_Rsend_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tagr,comm,rev_swaits+i);CHKERRQ(ierr);
      } else if (use_ssend) {
        ierr = MPI_Ssend_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tagr,comm,rev_swaits+i);CHKERRQ(ierr);
      } else {
        ierr = MPI_Send_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tagr,comm,rev_swaits+i);CHKERRQ(ierr);
      }
    }

    for (i=0; i<to->n; i++) {
      if (use_rsend) {
        ierr = MPI_Rsend_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
      } else if (use_ssend) {
        ierr = MPI_Ssend_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
      } else {
        ierr = MPI_Send_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
      }
    }
    /* Register receives for scatter and reverse */
    for (i=0; i<from->n; i++) {
      ierr = MPI_Recv_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);CHKERRQ(ierr);
    }
    for (i=0; i<to->n; i++) {
      ierr = MPI_Recv_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tagr,comm,rev_rwaits+i);CHKERRQ(ierr);
    }
    if (use_rsend) {
      if (to->n)   {ierr = MPI_Startall_irecv(to->starts[to->n]*to->bs,to->n,to->rev_requests);CHKERRQ(ierr);}
      if (from->n) {ierr = MPI_Startall_irecv(from->starts[from->n]*from->bs,from->n,from->requests);CHKERRQ(ierr);}
      ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    }
  }
  ierr = PetscInfo1(ctx,"Using blocksize %D scatter\n",bs);CHKERRQ(ierr);

#if defined(PETSC_USE_DEBUG)
  ierr = MPIU_Allreduce(&bs,&i,1,MPIU_INT,MPI_MIN,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&bs,&n,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
  if (bs!=i || bs!=n) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Blocks size %D != %D or %D",bs,i,n);
#endif

  switch (bs) {
  case 12:
    ctx->ops->begin = VecScatterBeginMPI1_12;
    ctx->ops->end   = VecScatterEndMPI1_12;
    break;
  case 11:
    ctx->ops->begin = VecScatterBeginMPI1_11;
    ctx->ops->end   = VecScatterEndMPI1_11;
    break;
  case 10:
    ctx->ops->begin = VecScatterBeginMPI1_10;
    ctx->ops->end   = VecScatterEndMPI1_10;
    break;
  case 9:
    ctx->ops->begin = VecScatterBeginMPI1_9;
    ctx->ops->end   = VecScatterEndMPI1_9;
    break;
  case 8:
    ctx->ops->begin = VecScatterBeginMPI1_8;
    ctx->ops->end   = VecScatterEndMPI1_8;
    break;
  case 7:
    ctx->ops->begin = VecScatterBeginMPI1_7;
    ctx->ops->end   = VecScatterEndMPI1_7;
    break;
  case 6:
    ctx->ops->begin = VecScatterBeginMPI1_6;
    ctx->ops->end   = VecScatterEndMPI1_6;
    break;
  case 5:
    ctx->ops->begin = VecScatterBeginMPI1_5;
    ctx->ops->end   = VecScatterEndMPI1_5;
    break;
  case 4:
    ctx->ops->begin = VecScatterBeginMPI1_4;
    ctx->ops->end   = VecScatterEndMPI1_4;
    break;
  case 3:
    ctx->ops->begin = VecScatterBeginMPI1_3;
    ctx->ops->end   = VecScatterEndMPI1_3;
    break;
  case 2:
    ctx->ops->begin = VecScatterBeginMPI1_2;
    ctx->ops->end   = VecScatterEndMPI1_2;
    break;
  case 1:
    ctx->ops->begin = VecScatterBeginMPI1_1;
    ctx->ops->end   = VecScatterEndMPI1_1;
    break;
  default:
    ctx->ops->begin = VecScatterBeginMPI1_bs;
    ctx->ops->end   = VecScatterEndMPI1_bs;

  }
  ctx->ops->view = VecScatterView_MPI_MPI1;
  /* try to optimize PtoP vecscatter with memcpy's */
  ierr = VecScatterMemcpyPlanCreate_PtoP(to,from);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   create parallel to sequential scatter context.
   bs indicates how many elements there are in each block. Normally this would be 1.

   contains check that PetscMPIInt can handle the sizes needed
*/
PetscErrorCode VecScatterCreateLocal_PtoS_MPI1(PetscInt nx,const PetscInt *inidx,PetscInt ny,const PetscInt *inidy,Vec xin,Vec yin,PetscInt bs,VecScatter ctx)
{
  VecScatter_MPI_General *from,*to;
  PetscMPIInt            nprocs,myrank,tag;
  PetscMPIInt            *recvfrom = NULL,*rlens = NULL,rlenlocal,rlentotal,rlenshm,nrecvs;
  PetscMPIInt            *sendto = NULL,*slens = NULL,slentotal,slenshm,nsends,nsendsshm;
  PetscInt               *range = NULL,i,j;
  PetscInt               *rstarts = NULL,count;
  PetscInt               *rindices,*sindices,*sindices2;
  MPI_Request            *sreqs = NULL,*rreqs = NULL;
  PetscErrorCode         ierr;
  PetscInt               *idxbs_sorted = NULL,*idybs_sorted = NULL;
  PetscShmComm           pshmcomm;
  MPI_Comm               comm; /* the outer communicator */
  PetscInt               it,first,step,lblocal,ublocal;
  PetscBool              use_intranodeshm;
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  PetscMPIInt            jj;
  MPI_Info               info;
#endif

  PetscFunctionBegin;
  ierr  = PetscObjectGetNewTag((PetscObject)ctx,&tag);CHKERRQ(ierr);
  ierr  = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr  = MPI_Comm_rank(comm,&myrank);CHKERRQ(ierr);
  ierr  = MPI_Comm_size(comm,&nprocs);CHKERRQ(ierr);
  range = xin->map->range;

  use_intranodeshm = PETSC_TRUE;
  ierr  = PetscOptionsGetBool(NULL,NULL,"-vecscatter_useintranodeshm",&use_intranodeshm,NULL);CHKERRQ(ierr);
  if (use_intranodeshm) { ierr = PetscShmCommGet(comm,&pshmcomm);CHKERRQ(ierr); }

  /*=========================================================================
            sort indices and locate my segment
    =========================================================================*/

  /* idxbs_sorted - [nx] inidx[]*bs and then sorted
     idybs_sorted - [ny] inidy[]*bs and then sorted
     lblocal      - lower & upper bound such that indices in ..
     ublocal      - idxbs_sorted[lblocal..ublocal) are owned by me
   */

  /* Sorted indices make code simpler, faster and also help getting rid of
     many O(P) arrays, which hurt scalability at large scale.
   */
  ierr = PetscMalloc2(nx,&idxbs_sorted,ny,&idybs_sorted);CHKERRQ(ierr);
  if (bs == 1) { /* accelerate the common case */
    ierr = PetscMemcpy(idxbs_sorted,inidx,sizeof(PetscInt)*nx);CHKERRQ(ierr);
    ierr = PetscMemcpy(idybs_sorted,inidy,sizeof(PetscInt)*ny);CHKERRQ(ierr);
  } else {
    for (i=0; i<nx; i++) idxbs_sorted[i] = inidx[i]*bs;
    for (i=0; i<ny; i++) idybs_sorted[i] = inidy[i]*bs;
  }

  ierr = PetscSortIntWithArray(nx,idxbs_sorted,idybs_sorted);CHKERRQ(ierr);

  /* search idxbs_sorted[] to locate my segment of indices. If exist, they are
     in idxbs_sorted[lblocal..ublocal), otherwise, lblocal=ublocal=0 or nx,
     depending on whether idxbs_sorted[0] > range[myrank] or not.
   */
  first = 0; count = nx; /* find first element in idxbs_sorted[] that is not less than range[myrank] */
  while (count > 0) {
    it = first; step = count/2; it += step;
    if (idxbs_sorted[it] < range[myrank]) { first  = ++it; count -= step + 1; }
    else { count = step; }
  }
  lblocal = first;

  first = 0; count = nx; /* do it again for range[myrank+1] */
  while (count > 0) {
    it = first; step = count/2; it += step;
    if (idxbs_sorted[it] < range[myrank+1]) { first  = ++it; count -= step + 1; }
    else { count = step; }
  }
  ublocal = first;

  /*=========================================================================
           collect info about messages I want to receive
    =========================================================================*/

  /* nrecvs    - number of non-empty messages, excluding the message from myself.
     recvfrom  - [nrecvs] processors I will receive messages from, excluding myself
     rindices  - [rlentotal] indices of entries I will receive
     rstarts   - [nrecvs+1] rstarts[i] is the starting index of rindices[] I expect from processor recvfrom[i]
     rlens     - [nprocs] I want to receive rlens[i] entries from processor i.
     rlentotal - total number of entries I will receive, excluding entries from myself
     rlenlocal - number of entries from myself

     Attention: rlens[] is of O(P) storage. It is the only one of this large in this function.
   */

  /* get rlens, nrecvs */
  ierr = PetscCalloc1(nprocs,&rlens);CHKERRQ(ierr);

  i = j = nrecvs = 0;
  while (i < nx) {
    if (idxbs_sorted[i] >= range[j+1]) { /* if i-th index is out of processor j's bound */
      do { j++; } while (idxbs_sorted[i] >= range[j+1] && j < nprocs); /* boost j until it falls in processor j's bound */
      if (j == nprocs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"entry %D not owned by any process, upper bound %D",idxbs_sorted[i],range[nprocs]);
    }

    if (j == myrank) { i = ublocal; } /* skip local indices, which are likely a lot */
    else { i++; if (!rlens[j]++) nrecvs++; }
  }

  rlenlocal = ublocal - lblocal;
  rlentotal = nx - rlenlocal;

  /* get rstarts, recvfrom, rindices and rreqs once we know nrecvs, rlens*/
  ierr = PetscMalloc4(rlentotal,&rindices,nrecvs+1,&rstarts,nrecvs,&recvfrom,nrecvs,&rreqs);CHKERRQ(ierr);

  j = rstarts[0] = 0;
  for (i=0; i<nprocs; i++) {
    if (rlens[i]) { recvfrom[j] = i; rstarts[j+1] = rstarts[j] + rlens[i]; j++; }
  }

  ierr = PetscMemcpy(rindices,idxbs_sorted,sizeof(PetscInt)*lblocal);CHKERRQ(ierr); /* two copies to skip indices in [lblocal,ublocal) */
  ierr = PetscMemcpy(&rindices[lblocal],idxbs_sorted+ublocal,sizeof(PetscInt)*(nx-ublocal));CHKERRQ(ierr);

  /*=========================================================================
           compute the reverse info about messages I need to send
    =========================================================================*/

  /* nsends    - number of (non-empty) messages I need to send
     sendto    - [nsends] processors I send to
     slens     - [nsends] I will send slens[i] entries to processor sendto[i]
     slentotal - sum of slens[]
     sindices  - [] store indices of entries I need to send
     sreqs     - [nsends] MPI requests
   */
  ierr = PetscGatherNumberOfMessages(comm,NULL,rlens,&nsends);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths(comm,nrecvs,nsends,rlens,&sendto,&slens);CHKERRQ(ierr);
  ierr = PetscSortMPIIntWithArray(nsends,sendto,slens);CHKERRQ(ierr);
  slentotal = 0; for (i=0; i<nsends; i++) slentotal += slens[i];

  ierr = PetscFree(rlens);CHKERRQ(ierr);

  /* communicate with processors in sendto[] to populate sindices[].
     Post irecvs first and then isends. It is funny (but correct) that we
     temporarily use send stuff in MPI_Irecv and recv stuff in MPI_Isends.
   */
  ierr  = PetscMalloc2(slentotal,&sindices,nsends,&sreqs);CHKERRQ(ierr);

  count = 0;
  for (i=0; i<nsends; i++) {
    ierr   = MPI_Irecv(sindices+count,slens[i],MPIU_INT,sendto[i],tag,comm,sreqs+i);CHKERRQ(ierr);
    count += slens[i];
  }

  for (i=0; i<nrecvs; i++) { ierr = MPI_Isend(rindices+rstarts[i],rstarts[i+1]-rstarts[i],MPIU_INT,recvfrom[i],tag,comm,rreqs+i);CHKERRQ(ierr); }

  /* wait on irecvs and if supported, figure out which sendto[] processors are in the shared memory communicator
     nsendsshm - number of sendto[] processors in the shared memory communicator
     slenshm   - total number of entries sent to shared memory partners
   */
  nsendsshm = 0;
  slenshm   = 0;
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  if (use_intranodeshm) {
    MPI_Status send_status;
    PetscMPIInt index,n;
    for (i=0; i<nsends; i++) {
      ierr = MPI_Waitany(nsends,sreqs,&index,&send_status);CHKERRQ(ierr);
      ierr = MPI_Get_count(&send_status,MPIU_INT,&n);CHKERRQ(ierr);
      ierr = PetscShmCommGlobalToLocal(pshmcomm,sendto[index],&jj);CHKERRQ(ierr);
      if (jj != MPI_PROC_NULL) { nsendsshm++; slenshm += n; }
    }
  } else
#endif
  { ierr = MPI_Waitall(nsends,sreqs,MPI_STATUS_IGNORE);CHKERRQ(ierr); }

  ierr = MPI_Waitall(nrecvs,rreqs,MPI_STATUS_IGNORE);CHKERRQ(ierr);

  /*=========================================================================
         allocate entire send scatter context
    =========================================================================*/
  ierr                   = PetscNewLog(ctx,&to);CHKERRQ(ierr); /* calloc to */
  ctx->todata            = (void*)to;
  to->n                  = nsends-nsendsshm;
  to->use_intranodeshm   = use_intranodeshm;

  ierr = PetscMalloc1(to->n,&to->requests);CHKERRQ(ierr);
  ierr = PetscMalloc2(PetscMax(to->n,nrecvs),&to->sstatus,PetscMax(to->n,nrecvs),&to->rstatus);CHKERRQ(ierr);
  ierr = PetscMalloc4(bs*(slentotal-slenshm),&to->values,slentotal-slenshm,&to->indices,to->n+1,&to->starts,to->n,&to->procs);CHKERRQ(ierr);

  sindices2     = sindices;
  to->n         = 0;
  to->starts[0] = 0;

#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  if (use_intranodeshm) {
    to->shmn         = nsendsshm;
    /* shmspaces[] contains pointers to spaces where other processors put data to be read from this processor.
       These spaces are allocated by other processors. In this sense, shmspaces[] is totally a fromdata thing
       and we need not to allocate memory for it in todata. However, we do the allocation because one may call
       VecScatterCreate_PtoS through VecScatterCreate_StoP by swapping todata and fromdata. By allocating
       memory symmetrically in todata and fromdata (i.e., both use PetscMalloc5), we can free memory symmetrically
       when destroying a vecscatter and do not care whether todata/fromdata in the vecscatter were swapped or not,
       thus avoid potential memory leaks.
    */
    ierr             = PetscMalloc5(to->shmn,&to->shmprocs,to->shmn,&to->shmspaces,to->shmn,&to->shmstates,to->shmn+1,&to->shmstarts,slenshm,&to->shmindices);CHKERRQ(ierr);
    to->shmn         = 0;
    to->shmstarts[0] = 0;
  }
#endif

  for (i=0; i<nsends; i++) {
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
    if (use_intranodeshm) {
      ierr = PetscShmCommGlobalToLocal(pshmcomm,sendto[i],&jj);CHKERRQ(ierr);
    }

    if (use_intranodeshm && jj != MPI_PROC_NULL) { /* sendto[i] is a shared memory partner and jj is its rank in shmcomm */
      to->shmstarts[to->shmn+1] = to->shmstarts[to->shmn] + slens[i];
      to->shmprocs[to->shmn]    = sendto[i]; /* use rank in the outer comm */
      for (j=0; j<slens[i]; j++) to->shmindices[to->shmstarts[to->shmn]+j] = sindices2[j] - range[myrank];
      to->shmn++;
    } else
#endif
    {
      to->starts[to->n+1] = to->starts[to->n] + slens[i];
      to->procs[to->n]    = sendto[i];
      for (j=0; j<slens[i]; j++) to->indices[to->starts[to->n]+j] = sindices2[j] - range[myrank];
      to->n++;
    }
    sindices2 += slens[i];
  }

  /* free send stuffs */
  ierr = PetscFree(slens);CHKERRQ(ierr);
  ierr = PetscFree(sendto);CHKERRQ(ierr);
  ierr = PetscFree2(sindices,sreqs);CHKERRQ(ierr);

#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  /* allocate the shared memory region at the sender side. The region is padded with a header
     containing sync flags. Flags are on different cache lines to avoid false sharing.
   */
  if (use_intranodeshm) {
    char             *shmspace;
    PetscObjectState state;
    MPI_Aint         sz;
    ierr = PetscObjectStateGet((PetscObject)ctx,&state);CHKERRQ(ierr);
    ierr = PetscShmCommGetMpiShmComm(pshmcomm,&to->shmcomm);CHKERRQ(ierr);
    ierr = MPI_Info_create(&info);CHKERRQ(ierr);
    ierr = MPI_Info_set(info, "alloc_shared_noncontig", "true");CHKERRQ(ierr);
    sz   = bs*to->shmstarts[to->shmn]*sizeof(PetscScalar) + (to->shmn+1)*PETSC_LEVEL1_DCACHE_LINESIZE; /* add an extra cacheline for alignment purpose */
    ierr = MPI_Win_allocate_shared(sz,sizeof(PetscScalar),info,to->shmcomm,&shmspace,&to->shmwin);CHKERRQ(ierr);
#if !defined(PETSC_WRITE_MEMORY_BARRIER)
    ierr = MPI_Win_lock_all(MPI_MODE_NOCHECK,to->shmwin);CHKERRQ(ierr); /* see comments in VecScatterBegin for why this call */
#endif
    ierr = MPI_Info_free(&info);CHKERRQ(ierr);

    /* Align the returned shared memory address to cacheline, where the state area
       starts. Each state takes one cacheline to avoid false sharing. Note we used
       alloc_shared_noncontig in shared memory allocation. The returned shared memory
       address on each process is expected to be page-aligned (and cacheline-aligned).
       However, for some unknown reason, I found it is not necessarily true on a Cray
       machine NERSC Cori (Cray ignored the info hints?). So I allocate an extra
       cacheline and do the alignment myself.
     */
    shmspace = (char*)((((PETSC_UINTPTR_T)(shmspace))+(PETSC_LEVEL1_DCACHE_LINESIZE-1)) & ~(PETSC_LEVEL1_DCACHE_LINESIZE-1));
    for (i=0; i<to->shmn; i++) {
       to->shmstates[i] = (PetscObjectState*)(shmspace + i*PETSC_LEVEL1_DCACHE_LINESIZE);
      *to->shmstates[i] = state; /* init the flag to empty (0) to say sender can write the buffer */
    }
    shmspace += to->shmn*PETSC_LEVEL1_DCACHE_LINESIZE; /* point the pointer to the data area */
    for (i=0; i<to->shmn; i++) to->shmspaces[i] = (PetscScalar*)shmspace + bs*to->shmstarts[i];
  }
#endif

  /*=========================================================================
         allocate entire receive scatter context
    =========================================================================*/
  ierr                     = PetscNewLog(ctx,&from);CHKERRQ(ierr);
  ctx->fromdata            = (void*)from;
  from->use_intranodeshm   = use_intranodeshm;

  /* compute rlenshm, from->n and from->shmn first to facilitate mallocs */
  rlenshm = 0;
  from->n = nrecvs;
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  if (use_intranodeshm) {
    from->n       = 0;
    from->shmn    = 0;
    from->shmwin  = to->shmwin;
    from->shmcomm = to->shmcomm;
    for (i=0; i<nrecvs; i++) {
      ierr = PetscShmCommGlobalToLocal(pshmcomm,recvfrom[i],&jj);CHKERRQ(ierr);
      if (jj != MPI_PROC_NULL) { from->shmn++; rlenshm += rstarts[i+1] - rstarts[i]; }
      else { from->n++; }
    }

    ierr = PetscMalloc5(from->shmn,&from->shmprocs,from->shmn,&from->shmspaces,from->shmn,&from->shmstates,from->shmn+1,&from->shmstarts,rlenshm,&from->shmindices);CHKERRQ(ierr);
  }
#endif

  ierr = PetscMalloc1(from->n,&from->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(bs*(ny-rlenshm),&from->values,ny-rlenshm,&from->indices,from->n+1,&from->starts,from->n,&from->procs);CHKERRQ(ierr);

  /* move data into receive scatter */
  from->n               = 0;
  from->starts[0]       = 0;
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  if (use_intranodeshm) {
    from->shmn          = 0;
    from->shmstarts[0]  = 0;
  }
#endif

  for (i=0; i<nrecvs; i++) {
    PetscInt len    = rstarts[i+1] - rstarts[i]; /* len works for both x and y */
    PetscInt ystart = rstarts[i] + (recvfrom[i] > myrank ? rlenlocal : 0); /* rstarts[] are offsets for x with locals removed. To use for y, one has to remedy them */

#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
    if (use_intranodeshm) {
      ierr = PetscShmCommGlobalToLocal(pshmcomm,recvfrom[i],&jj);CHKERRQ(ierr);
    }

    if (use_intranodeshm && jj != MPI_PROC_NULL) { /* recvfrom[i] is a shared memory partner and jj is its rank in shmcomm*/
      from->shmprocs[from->shmn]    = recvfrom[i];
      from->shmstarts[from->shmn+1] = from->shmstarts[from->shmn] + len;
      ierr = PetscMemcpy(&from->shmindices[from->shmstarts[from->shmn]],&idybs_sorted[ystart],sizeof(PetscInt)*len);CHKERRQ(ierr);
      from->shmn++;
    } else
#endif
    {
      from->procs[from->n]    = recvfrom[i];
      from->starts[from->n+1] = from->starts[from->n] + len;
      ierr = PetscMemcpy(&from->indices[from->starts[from->n]],&idybs_sorted[ystart],sizeof(PetscInt)*len);CHKERRQ(ierr);
      from->n++;
    }
  }

  /* free recv stuffs */
  ierr = PetscFree4(rindices,rstarts,recvfrom,rreqs);CHKERRQ(ierr);

  /* query addresses of the shared memory regions of my partners. I also need
     to know offsets to those regions from where I can directly read data I need.
     We use send/recv within shmcomm to get the offsets. One could also get the
     offsets through shared memory. But since it is only used once, it is not
     worth the trouble.
   */
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  if (use_intranodeshm) {
    MPI_Request *reqs = NULL;
    struct {
      PetscInt j,m,offset; /* from my partner's view, I am its j-th partner out of its m partners, and I should read from this offset */
    } jmo,*triples = NULL;

    /* get the above info from my partners */
    ierr = PetscMalloc2(from->shmn,&reqs,from->shmn,&triples);CHKERRQ(ierr);
    for (i=0; i<from->shmn; i++) { ierr = MPI_Irecv(triples+i,3,MPIU_INT,from->shmprocs[i],0/*tag*/,comm,reqs+i);CHKERRQ(ierr); }
    for (i=0; i<to->shmn; i++) {
      jmo.j = i;
      jmo.m = to->shmn;
      jmo.offset = to->shmstarts[i];
      ierr = MPI_Send(&jmo,3,MPIU_INT,to->shmprocs[i],0/*tag*/,comm);CHKERRQ(ierr);
    }
    ierr = MPI_Waitall(from->shmn,reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);

    /* figure out flag addresses and data addresses aimed to me */
    for (i=0; i<from->shmn; i++) {
      MPI_Aint    size;
      PetscMPIInt disp_unit,lrank;
      ierr = PetscShmCommGlobalToLocal(pshmcomm,from->shmprocs[i],&lrank);CHKERRQ(ierr);
      ierr = MPI_Win_shared_query(from->shmwin,lrank,&size,&disp_unit,&from->shmspaces[i]);CHKERRQ(ierr);
      from->shmspaces[i]  = (PetscScalar*)((((PETSC_UINTPTR_T)(from->shmspaces[i]))+(PETSC_LEVEL1_DCACHE_LINESIZE-1)) & ~(PETSC_LEVEL1_DCACHE_LINESIZE-1));
      from->shmstates[i]  = (PetscObjectState*)((char*)from->shmspaces[i] + triples[i].j*PETSC_LEVEL1_DCACHE_LINESIZE); /* get address of the j-th state */
      from->shmspaces[i]  = (PetscScalar*)((char*)from->shmspaces[i] + triples[i].m*PETSC_LEVEL1_DCACHE_LINESIZE); /* skip the state area */
      from->shmspaces[i] += triples[i].offset*bs; /* and then add the offset to point to where my expected data lives */
    }

    ierr = PetscFree2(reqs,triples);CHKERRQ(ierr);
  }
#endif

  /*=========================================================================
       handle the scatter to myself
    =========================================================================*/
  if (rlenlocal) {
    to->local.n   = rlenlocal;
    from->local.n = rlenlocal;
    ierr = PetscMalloc1(to->local.n,&to->local.vslots);CHKERRQ(ierr);
    ierr = PetscMalloc1(from->local.n,&from->local.vslots);CHKERRQ(ierr);
    for (i=lblocal; i<ublocal; i++) to->local.vslots[i-lblocal] = idxbs_sorted[i] - range[myrank];
    ierr = PetscMemcpy(from->local.vslots,&idybs_sorted[lblocal],sizeof(PetscInt)*rlenlocal);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)ctx,2*rlenlocal*sizeof(PetscInt));CHKERRQ(ierr);
  } else {
    to->local.n        = 0;
    to->local.vslots   = 0;
    from->local.n      = 0;
    from->local.vslots = 0;
  }

  ierr = PetscFree2(idxbs_sorted,idybs_sorted);CHKERRQ(ierr);

  from->local.nonmatching_computed = PETSC_FALSE;
  from->local.n_nonmatching        = 0;
  from->local.slots_nonmatching    = 0;
  to->local.nonmatching_computed   = PETSC_FALSE;
  to->local.n_nonmatching          = 0;
  to->local.slots_nonmatching      = 0;

  from->format = VEC_SCATTER_MPI_GENERAL;
  to->format   = VEC_SCATTER_MPI_GENERAL;
  from->bs     = bs;
  to->bs       = bs;

  ierr = VecScatterCreateCommon_PtoS_MPI1(from,to,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* ------------------------------------------------------------------------------------*/
/*
         Scatter from local Seq vectors to a parallel vector.
         Reverses the order of the arguments, calls VecScatterCreateLocal_PtoS() then
         reverses the result.
*/
PetscErrorCode VecScatterCreateLocal_StoP_MPI1(PetscInt nx,const PetscInt *inidx,PetscInt ny,const PetscInt *inidy,Vec xin,Vec yin,PetscInt bs,VecScatter ctx)
{
  PetscErrorCode         ierr;
  MPI_Request            *waits;
  VecScatter_MPI_General *to,*from;

  PetscFunctionBegin;
  ierr          = VecScatterCreateLocal_PtoS_MPI1(ny,inidy,nx,inidx,yin,xin,bs,ctx);CHKERRQ(ierr);
  to            = (VecScatter_MPI_General*)ctx->fromdata;
  from          = (VecScatter_MPI_General*)ctx->todata;
  ctx->todata   = (void*)to;
  ctx->fromdata = (void*)from;
  /* these two are special, they are ALWAYS stored in to struct */
  to->sstatus   = from->sstatus;
  to->rstatus   = from->rstatus;

  from->sstatus = 0;
  from->rstatus = 0;
  waits              = from->rev_requests;
  from->rev_requests = from->requests;
  from->requests     = waits;
  waits              = to->rev_requests;
  to->rev_requests   = to->requests;
  to->requests       = waits;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/
PetscErrorCode VecScatterCreateLocal_PtoP_MPI1(PetscInt nx,const PetscInt *inidx,PetscInt ny,const PetscInt *inidy,Vec xin,Vec yin,PetscInt bs,VecScatter ctx)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,tag,imdex,n;
  PetscInt       *owners = xin->map->range;
  PetscMPIInt    *nprocs = NULL;
  PetscInt       i,j,idx,nsends,*local_inidx = NULL,*local_inidy = NULL;
  PetscMPIInt    *owner   = NULL;
  PetscInt       *starts  = NULL,count,slen;
  PetscInt       *rvalues = NULL,*svalues = NULL,base,*values = NULL,*rsvalues,recvtotal,lastidx;
  PetscMPIInt    *onodes1,*olengths1,nrecvs;
  MPI_Comm       comm;
  MPI_Request    *send_waits = NULL,*recv_waits = NULL;
  MPI_Status     recv_status,*send_status = NULL;
  PetscBool      duplicate = PETSC_FALSE;
#if defined(PETSC_USE_DEBUG)
  PetscBool      found = PETSC_FALSE;
#endif

  PetscFunctionBegin;
  ierr = PetscObjectGetNewTag((PetscObject)ctx,&tag);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (size == 1) {
    ierr = VecScatterCreateLocal_StoP_MPI1(nx,inidx,ny,inidy,xin,yin,bs,ctx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /*
     Each processor ships off its inidx[j] and inidy[j] to the appropriate processor
     They then call the StoPScatterCreate()
  */
  /*  first count number of contributors to each processor */
  ierr = PetscMalloc3(size,&nprocs,nx,&owner,(size+1),&starts);CHKERRQ(ierr);
  ierr = PetscMemzero(nprocs,size*sizeof(PetscMPIInt));CHKERRQ(ierr);

  lastidx = -1;
  j       = 0;
  for (i=0; i<nx; i++) {
    /* if indices are NOT locally sorted, need to start search at the beginning */
    if (lastidx > (idx = bs*inidx[i])) j = 0;
    lastidx = idx;
    for (; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++;
        owner[i] = j;
#if defined(PETSC_USE_DEBUG)
        found = PETSC_TRUE;
#endif
        break;
      }
    }
#if defined(PETSC_USE_DEBUG)
    if (!found) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index %D out of range",idx);
    found = PETSC_FALSE;
#endif
  }
  nsends = 0;
  for (i=0; i<size; i++) nsends += (nprocs[i] > 0);

  /* inform other processors of number of messages and max length*/
  ierr = PetscGatherNumberOfMessages(comm,NULL,nprocs,&nrecvs);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths(comm,nsends,nrecvs,nprocs,&onodes1,&olengths1);CHKERRQ(ierr);
  ierr = PetscSortMPIIntWithArray(nrecvs,onodes1,olengths1);CHKERRQ(ierr);
  recvtotal = 0; for (i=0; i<nrecvs; i++) recvtotal += olengths1[i];

  /* post receives:   */
  ierr = PetscMalloc5(2*recvtotal,&rvalues,2*nx,&svalues,nrecvs,&recv_waits,nsends,&send_waits,nsends,&send_status);CHKERRQ(ierr);

  count = 0;
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv((rvalues+2*count),2*olengths1[i],MPIU_INT,onodes1[i],tag,comm,recv_waits+i);CHKERRQ(ierr);
    count += olengths1[i];
  }
  ierr = PetscFree(onodes1);CHKERRQ(ierr);

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to
         the ith processor
  */
  starts[0]= 0;
  for (i=1; i<size; i++) starts[i] = starts[i-1] + nprocs[i-1];
  for (i=0; i<nx; i++) {
    svalues[2*starts[owner[i]]]       = bs*inidx[i];
    svalues[1 + 2*starts[owner[i]]++] = bs*inidy[i];
  }

  starts[0] = 0;
  for (i=1; i<size+1; i++) starts[i] = starts[i-1] + nprocs[i-1];
  count = 0;
  for (i=0; i<size; i++) {
    if (nprocs[i]) {
      ierr = MPI_Isend(svalues+2*starts[i],2*nprocs[i],MPIU_INT,i,tag,comm,send_waits+count);CHKERRQ(ierr);
      count++;
    }
  }
  ierr = PetscFree3(nprocs,owner,starts);CHKERRQ(ierr);

  /*  wait on receives */
  count = nrecvs;
  slen  = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr  = MPI_Get_count(&recv_status,MPIU_INT,&n);CHKERRQ(ierr);
    slen += n/2;
    count--;
  }
  if (slen != recvtotal) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Total message lengths %D not as expected %D",slen,recvtotal);

  ierr     = PetscMalloc2(slen,&local_inidx,slen,&local_inidy);CHKERRQ(ierr);
  base     = owners[rank];
  count    = 0;
  rsvalues = rvalues;
  for (i=0; i<nrecvs; i++) {
    values    = rsvalues;
    rsvalues += 2*olengths1[i];
    for (j=0; j<olengths1[i]; j++) {
      local_inidx[count]   = values[2*j] - base;
      local_inidy[count++] = values[2*j+1];
    }
  }
  ierr = PetscFree(olengths1);CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);}
  ierr = PetscFree5(rvalues,svalues,recv_waits,send_waits,send_status);CHKERRQ(ierr);

  /*
     should sort and remove duplicates from local_inidx,local_inidy
  */
#if defined(do_it_slow)
  /* sort on the from index */
  ierr  = PetscSortIntWithArray(slen,local_inidx,local_inidy);CHKERRQ(ierr);
  start = 0;
  while (start < slen) {
    count = start+1;
    last  = local_inidx[start];
    while (count < slen && last == local_inidx[count]) count++;
    if (count > start + 1) { /* found 2 or more same local_inidx[] in a row */
      /* sort on to index */
      ierr = PetscSortInt(count-start,local_inidy+start);CHKERRQ(ierr);
    }
    /* remove duplicates; not most efficient way, but probably good enough */
    i = start;
    while (i < count-1) {
      if (local_inidy[i] != local_inidy[i+1]) i++;
      else { /* found a duplicate */
        duplicate = PETSC_TRUE;
        for (j=i; j<slen-1; j++) {
          local_inidx[j] = local_inidx[j+1];
          local_inidy[j] = local_inidy[j+1];
        }
        slen--;
        count--;
      }
    }
    start = count;
  }
#endif
  if (duplicate) {
    ierr = PetscInfo(ctx,"Duplicate from to indices passed in VecScatterCreate(), they are ignored\n");CHKERRQ(ierr);
  }
  ierr = VecScatterCreateLocal_StoP_MPI1(slen,local_inidx,slen,local_inidy,xin,yin,bs,ctx);CHKERRQ(ierr);
  ierr = PetscFree2(local_inidx,local_inidy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSFCreateFromZero - Create a PetscSF that maps a Vec from sequential to distributed

  Input Parameters:
. gv - A distributed Vec

  Output Parameters:
. sf - The SF created mapping a sequential Vec to gv

  Level: developer

.seealso: DMPlexDistributedToSequential()
@*/
PetscErrorCode PetscSFCreateFromZero(MPI_Comm comm, Vec gv, PetscSF *sf)
{
  PetscSFNode   *remotenodes;
  PetscInt      *localnodes;
  PetscInt       N, n, start, numroots, l;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFCreate(comm, sf);CHKERRQ(ierr);
  ierr = VecGetSize(gv, &N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(gv, &n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(gv, &start, NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, &localnodes);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, &remotenodes);CHKERRQ(ierr);
  if (!rank) numroots = N;
  else       numroots = 0;
  for (l = 0; l < n; ++l) {
    localnodes[l]        = l;
    remotenodes[l].rank  = 0;
    remotenodes[l].index = l+start;
  }
  ierr = PetscSFSetGraph(*sf, numroots, n, localnodes, PETSC_OWN_POINTER, remotenodes, PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

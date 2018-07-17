
/*
     Defines the methods VecScatterBegin/End_1,2,......
     This is included by vpscat.c with different values for BS

     This is a terrible way of doing "templates" in C.
*/
#define PETSCMAP1_a(a,b)  a ## _ ## b
#define PETSCMAP1_b(a,b)  PETSCMAP1_a(a,b)
#define PETSCMAP1(a)      PETSCMAP1_b(a,BS)

PetscErrorCode PETSCMAP1(VecScatterBeginMPI1)(VecScatter ctx,Vec xin,Vec yin,InsertMode addv,ScatterMode mode)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *xv,*yv;
  MPI_Request            *rwaits,*swaits;
  PetscErrorCode         ierr;
  PetscInt               i,bs;
#if defined(PETSC_HAVE_VIENNACL)
  PetscBool              is_viennacltype = PETSC_FALSE;
#endif
  PetscBool              packtogether,have_multiple_requests;

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
  bs       = to->bs;

  packtogether = (ctx->packtogether || to->use_alltoallv || to->use_window || to->use_neighborhood) ? PETSC_TRUE : PETSC_FALSE; /* pack all messages together before sending */
  have_multiple_requests = (!to->use_alltoallv && !to->use_window && !to->use_neighborhood) ? PETSC_TRUE : PETSC_FALSE; /* will generate multiple MPI requests */

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
    if (!from->use_readyreceiver && !to->sendfirst && have_multiple_requests) {
      /* post receives since they were not previously posted    */
      if (from->n) { ierr = MPI_Startall_irecv(from->starts[from->n]*bs,from->n,rwaits);CHKERRQ(ierr); }
    }

#if defined(PETSC_HAVE_MPI_ALLTOALLW)  && !defined(PETSC_USE_64BIT_INDICES)
    if (to->use_alltoallw && addv == INSERT_VALUES) {
      ierr = MPI_Alltoallw(xv,to->wcounts,to->wdispls,to->types,yv,from->wcounts,from->wdispls,from->types,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
    } else
#endif
    if (packtogether) {
      for (i=0; i<to->n; i++) {
        if (to->memcpy_plan.optimized[i]) { ierr = VecScatterMemcpyPlanExecute_Pack(i,xv,&to->memcpy_plan,to->values+bs*to->starts[i],INSERT_VALUES,bs);CHKERRQ(ierr); }
        else { PETSCMAP1(Pack_MPI1)(to->starts[i+1]-to->starts[i],to->indices+to->starts[i],xv,to->values+bs*to->starts[i],bs); }
      }

      if (to->use_alltoallv) {
        ierr = MPI_Alltoallv(to->values,to->counts,to->displs,MPIU_SCALAR,from->values,from->counts,from->displs,MPIU_SCALAR,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      }
#if defined(PETSC_HAVE_MPI_WIN_CREATE_FEATURE)
      else if (to->use_window) {
        PetscInt cnt;
        ierr = MPI_Win_fence(0,from->window);CHKERRQ(ierr);
        for (i=0; i<to->n; i++) {
          cnt  = bs*(to->starts[i+1]-to->starts[i]);
          ierr = MPI_Put(to->values+bs*to->starts[i],cnt,MPIU_SCALAR,to->procs[i],bs*to->winstarts[i],cnt,MPIU_SCALAR,from->window);CHKERRQ(ierr);
        }
      }
#endif
#if defined(PETSC_HAVE_MPI_NEIGHBORHOOD_COLLECTIVE)
      else if (to->use_neighborhood) {
        ierr = MPI_Start_ineighbor_alltoallv(to->n,from->n,to->values,to->neigh_counts,to->neigh_displs,MPIU_SCALAR,from->values,from->neigh_counts,from->neigh_displs,MPIU_SCALAR,to->comm_dist_graph,&to->neigh_request);CHKERRQ(ierr);
      }
#endif
      else if (to->n) { ierr = MPI_Startall_isend(to->starts[to->n]*bs,to->n,swaits);CHKERRQ(ierr); }
    } else {
      /* this version packs and sends one at a time */
      for (i=0; i<to->n; i++) {
        if (to->memcpy_plan.optimized[i]) { /* use memcpy instead of indivisual load/store */
          ierr = VecScatterMemcpyPlanExecute_Pack(i,xv,&to->memcpy_plan,to->values+bs*to->starts[i],INSERT_VALUES,bs);CHKERRQ(ierr);
        } else {
          PETSCMAP1(Pack_MPI1)(to->starts[i+1]-to->starts[i],to->indices+to->starts[i],xv,to->values + bs*to->starts[i],bs);
        }
        ierr = MPI_Start_isend((to->starts[i+1]-to->starts[i])*bs,swaits+i);CHKERRQ(ierr);
      }
    }

#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
    /* intranode shared memory communication is orthogonal to the above communication approaches,
       whatever they are, alltoallv, alltoallw or one-sided. They just handle internode communications
       if intranode shared memory communication is enabled.

       Vecscatter is logically collective since all processors have to call VecScatterBegin/End even though
       some of them may not actually send/recv any messages. Think VecScatterBegin/End as transcations. In
       VecScatterBegin we pack and sent data; whereas in VecScatterEnd we receive and unpack data. We increase
       the state of a VecScatter object as we leave VecScatterBegin/End so that different calls of VecScatter do
       not mix. Then we add another state in the shared memory buffer. Only when this state matches the object
       state, we can read/write the shared memory.

       Shared memory sender-receiver sync protocal:

       Sender (to)                        Receiver (from)
       ---------------------              ---------------
       while (shmstate!=state);           while (shmstate!=state);
       write buffer;                      read buffer;
       write_memory_barrier;
       shmstate=state+1;                  shmstate=state+1;
       state++;                           state++;

       On the sender side, wirte_memory_barrier ensures if the new shmstate is perceived by the receiver,
       then memory writes before the barrier, i.e., data written to the buffer could also be perceived by the
       receiver. Therefore, the receiver can get up-to-date data from the buffer.

       On the reciever side, if shmstate=state+1 is perceived by sender (as a result, sender can begin the next
       vecscatter transaction, and overwrite the buffer), that means the instruction shmstate=state+1 has been
       committed. Since 'shmstate=state+1' comes after 'read buffer', 'read buffer' must also have been committed.
       Therefore, sender can safely overwrite the buffer. shmstate is declared as volatile, so that compilers will not
       reorder 'read buffer and 'shmstate=state+1', and also will not put shmstate in registers, which otherwise makes
       the while(shmstate!=state) loop run forever.

       One may wonder why we used an integer state instead of a simpler boolean flag to do the synchronization.
       With flag, a possible design would be:

       Sender (to)                        Receiver (from)
       ---------------------              ---------------
       while (shmflag);                   while (!shmflag);
       write buffer;                      read buffer;
       write_memory_barrier;
       shmflag=1;                         shmflag=0;

       The design is wrong because PETSc supports both SCATTER_FORWARD and SCATTER_REVERSE. In SCATTER_REVERSE, receiver
       sends data to sender. Suppose one does a SCATTER_FORWARD immediately after a SCATTER_REVERSE on the same vecscatter
       context (it does happen in real codes). In SCATTER_REVERSE, receiver writes the data and sets flag=1, and enters
       SCATTER_FORWARD immediately. Suppose at this moment the data has not yet been read and the flag has not yet been
       reset by sender. Receiver finds the flag is 1 and gladly reads the (wrong) data and resets the flag to 0, making
       sender wait for its data forever. An integer state separates different vescatter transcations and avoids this bug.

       On x86, wirte_memory_barrier is a store fence instruction (sfence). One could also use MPI_Win_sync(shmwin) as the
       barrier. But MPI_Win_sync has additional requirements. To avoid the complexity and be efficient, we do the barrier
       on our own when PETSC_WRITE_MEMORY_BARRIER is defined (see below)
     */

    if (to->use_intranodeshm) {
      /* Pack the send data into shared memory buffers (potentially in other NUMA domains) */
      PetscObjectState state;
      ierr = PetscObjectStateGet((PetscObject)ctx,&state);CHKERRQ(ierr);
      for (i=0; i<to->shmn; i++) {
        while(*to->shmstates[i] != state); /* wait shmsate matches object state before write */
        if (to->shm_memcpy_plan.optimized[i]) {
          ierr = VecScatterMemcpyPlanExecute_Pack(i,xv,&to->shm_memcpy_plan,to->shmspaces[i],INSERT_VALUES,bs);CHKERRQ(ierr);
        } else {
          PETSCMAP1(Pack_MPI1)(to->shmstarts[i+1]-to->shmstarts[i],to->shmindices+to->shmstarts[i],xv,to->shmspaces[i],bs);
        }
#if defined(PETSC_WRITE_MEMORY_BARRIER)
        PETSC_WRITE_MEMORY_BARRIER();
#else
        /* MPI_Win_sync() synchronizes the private and public window copies of shmwin. It is a memory barrier implemented by
           MPI. MPI standard states that calls of MPI_Win_sync must be within passive target epochs (see MPI3.1 Chapter 11.5.4).
           We use the cheapest MPI_Win_lock_all(MPI_MODE_NOCHECK,..)/MPI_Win_unlock_all() to create such an epoch beginned at
           shmwin creation and ended at shmwin destroy. We actually do not do any MPI RMA operations inside the epoch.
        */
        ierr = MPI_Win_sync(to->shmwin);CHKERRQ(ierr);
#endif
        *to->shmstates[i] = state+1; /* update shmstate after write */
      }
      ierr = PetscObjectStateIncrease((PetscObject)ctx);CHKERRQ(ierr); /* finish a VecScatterBegin transcation */
    }
#endif

    if (!from->use_readyreceiver && to->sendfirst && !to->use_alltoallv && !to->use_window) {
      /* post receives since they were not previously posted   */
      if (from->n) { ierr = MPI_Startall_irecv(from->starts[from->n]*bs,from->n,rwaits);CHKERRQ(ierr); }
    }
  }

  /* take care of local scatters */
  if (to->local.n) {
    if (to->local.memcpy_plan.optimized[0] && addv == INSERT_VALUES) {
      /* do copy when it is not a self-to-self copy */
      if (!(xv == yv && to->local.memcpy_plan.same_copy_starts)) { ierr = VecScatterMemcpyPlanExecute_Scatter(0,xv,&to->local.memcpy_plan,yv,&from->local.memcpy_plan,addv);CHKERRQ(ierr); }
    } else if (to->local.memcpy_plan.optimized[0]) {
      ierr = VecScatterMemcpyPlanExecute_Scatter(0,xv,&to->local.memcpy_plan,yv,&from->local.memcpy_plan,addv);CHKERRQ(ierr);
    } else {
      if (xv == yv && addv == INSERT_VALUES && to->local.nonmatching_computed) {
        /* only copy entries that do not share identical memory locations */
        ierr = PETSCMAP1(Scatter_MPI1)(to->local.n_nonmatching,to->local.slots_nonmatching,xv,from->local.slots_nonmatching,yv,addv,bs);CHKERRQ(ierr);
      } else {
        ierr = PETSCMAP1(Scatter_MPI1)(to->local.n,to->local.vslots,xv,from->local.vslots,yv,addv,bs);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecRestoreArrayRead(xin,(const PetscScalar**)&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

PetscErrorCode PETSCMAP1(VecScatterEndMPI1)(VecScatter ctx,Vec xin,Vec yin,InsertMode addv,ScatterMode mode)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *yv;
  PetscErrorCode         ierr;
  PetscInt               i,count,bs;
  PetscMPIInt            index;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             xrstatus,*rstatus,*sstatus;
  PetscBool              packtogether,have_multiple_requests;

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
  bs       = from->bs;

  /* unpack messages after all have been received, which happens when any of the condition is met. Note that when use_alltoallw && addv != INSERT_VALUES,
     we have to use MPI_Alltoallv instead (therefore, need pack/unpapck together). When use_alltoallw but addv == INSERT_VALUES, there is no need
     of unpacking (since we used MPI datatypes); When use_alltoallw is true, use_alltoallv is always true. Therefore, (to->use_alltoallv && !to->use_alltoallw)
     means use alltoallv alone.
   */
  packtogether = (ctx->packtogether || (to->use_alltoallw && addv != INSERT_VALUES) || (to->use_alltoallv && !to->use_alltoallw) || to->use_window || to->use_neighborhood) ? PETSC_TRUE : PETSC_FALSE;
  have_multiple_requests = (!to->use_alltoallv && !to->use_window && !to->use_neighborhood) ? PETSC_TRUE : PETSC_FALSE; /* generated multiple MPI requests */

#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  if (from->use_intranodeshm) {
    PetscObjectState state;
    ierr = PetscObjectStateGet((PetscObject)ctx,&state);CHKERRQ(ierr);
    /* unpack the recv data from shared memory buffers */
    for (i=0; i<from->shmn; i++) {
      while(*from->shmstates[i] != state); /* wait shmsate matches object state before read */
      if (from->shm_memcpy_plan.optimized[i]) { ierr = VecScatterMemcpyPlanExecute_Unpack(i,from->shmspaces[i],yv,&from->shm_memcpy_plan,addv,bs);CHKERRQ(ierr); }
      else { ierr = PETSCMAP1(UnPack_MPI1)(from->shmstarts[i+1]-from->shmstarts[i],from->shmspaces[i],from->shmindices+from->shmstarts[i],yv,addv,bs);CHKERRQ(ierr);  }
      *from->shmstates[i] = state+1; /* update shmstate after read */
    }
    ierr = PetscObjectStateIncrease((PetscObject)ctx);CHKERRQ(ierr); /* finish a VecScatterEnd transcation */
  }
#endif

  /* if use shared memory for intranode is enabled, then the following is just for inter-node */
  if (packtogether) {
#if defined(PETSC_HAVE_MPI_WIN_CREATE_FEATURE)
    if (to->use_window) { ierr = MPI_Win_fence(0,from->window);CHKERRQ(ierr); }
    else
#endif
#if defined(PETSC_HAVE_MPI_NEIGHBORHOOD_COLLECTIVE)
    if (to->use_neighborhood) { ierr = MPI_Wait(&to->neigh_request,MPI_STATUS_IGNORE);CHKERRQ(ierr); }
    else
#endif
    if (!to->use_alltoallv) { ierr = MPI_Waitall(from->n,rwaits,rstatus);CHKERRQ(ierr); } /* alltoallv/w is blocking, therefore no MPI_Wait */

    for (i=0; i<from->n; i++) {
      if (from->memcpy_plan.optimized[i]) { ierr = VecScatterMemcpyPlanExecute_Unpack(i,from->values+bs*from->starts[i],yv,&from->memcpy_plan,addv,bs);CHKERRQ(ierr); }
      else { ierr = PETSCMAP1(UnPack_MPI1)(from->starts[i+1]-from->starts[i],from->values+bs*from->starts[i],from->indices+from->starts[i],yv,addv,bs);CHKERRQ(ierr); }
    }
  } else if (!to->use_alltoallw) { /* use_alltoallw implies no unpacking */
    /* unpack one at a time */
    count = from->n;
    while (count) {
      if (ctx->reproduce) {
        index = count - 1;
        ierr  = MPI_Wait(rwaits+index,&xrstatus);CHKERRQ(ierr);
      } else {
        ierr = MPI_Waitany(from->n,rwaits,&index,&xrstatus);CHKERRQ(ierr);
      }
      /* unpack receives into our local space */
      if (from->memcpy_plan.optimized[index]) { ierr = VecScatterMemcpyPlanExecute_Unpack(index,from->values+bs*from->starts[index],yv,&from->memcpy_plan,addv,bs);CHKERRQ(ierr); }
      else { ierr = PETSCMAP1(UnPack_MPI1)(from->starts[index+1]-from->starts[index],from->values+bs*from->starts[index],from->indices+from->starts[index],yv,addv,bs);CHKERRQ(ierr);  }
      count--;
    }
  }

  if (from->use_readyreceiver) {
    if (from->n) { ierr = MPI_Startall_irecv(from->starts[from->n]*bs,from->n,rwaits);CHKERRQ(ierr); }
    ierr = MPI_Barrier(PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
  }

  /* wait on sends */
  if (have_multiple_requests) {ierr = MPI_Waitall(to->n,swaits,sstatus);CHKERRQ(ierr);}
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef PETSCMAP1_a
#undef PETSCMAP1_b
#undef PETSCMAP1
#undef BS

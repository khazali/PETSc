#include <../src/sys/threadcomm/impls/pthread/tcpthreadimpl.h>

#if defined PETSC_HAVE_MALLOC_H
#include <malloc.h>
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInitialize_PThread"
PetscErrorCode PetscThreadCommInitialize_PThread(PetscThreadComm tcomm)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscThreadComm_PThread ptcomm=(PetscThreadComm_PThread)tcomm->data;
  PetscThreadCommJobQueue jobqueue=tcomm->pool->jobqueue;

  PetscFunctionBegin;
  /* Create threads */
  for (i=tcomm->pool->thread_num_start; i < tcomm->nworkThreads; i++) {
    printf("Creating thread=%d\n",i);
    jobqueue->tinfo[i]->status = THREAD_CREATED;
    jobqueue->tinfo[i]->rank = tcomm->pool->granks[i];
    jobqueue->tinfo[i]->tcomm = tcomm;
    ierr = pthread_create(&ptcomm->tid[i],&ptcomm->attr[i],&PetscThreadPoolFunc,&jobqueue->tinfo[i]);CHKERRQ(ierr);
  }

  if (tcomm->pool->ismainworker) jobqueue->tinfo[0]->status = THREAD_INITIALIZED;

  PetscInt threads_initialized=0;
  /* Wait till all threads have been initialized */
  while (threads_initialized != tcomm->nworkThreads) {
    threads_initialized=0;
    for (i=0; i<tcomm->nworkThreads; i++) {
      if (!jobqueue->tinfo[tcomm->pool->granks[i]]->status) break;
      threads_initialized++;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommFinalize_PThread"
PetscErrorCode PetscThreadCommFinalize_PThread(PetscThreadComm tcomm)
{
  PetscErrorCode          ierr;
  void                    *jstatus;
  PetscThreadComm_PThread ptcomm=(PetscThreadComm_PThread)tcomm->data;
  PetscThreadCommJobQueue jobqueue=tcomm->pool->jobqueue;
  PetscInt                i;

  PetscFunctionBegin;
  ierr = PetscThreadPoolBarrier(tcomm);CHKERRQ(ierr);
  for (i=tcomm->pool->thread_num_start; i < tcomm->nworkThreads; i++) {
    printf("Terminating thread=%d\n",i);
    jobqueue->tinfo[i]->status = THREAD_TERMINATE;
    ierr = pthread_join(ptcomm->tid[i],&jstatus);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

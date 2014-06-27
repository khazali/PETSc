/* Define feature test macros to make sure CPU_SET and other functions are available
 */
#define PETSC_DESIRE_FEATURE_TEST_MACROS

#include <../src/sys/threadcomm/impls/pthread/tcpthreadimpl.h>

#if defined PETSC_HAVE_MALLOC_H
#include <malloc.h>
#endif

#if defined(PETSC_PTHREAD_LOCAL)
PETSC_PTHREAD_LOCAL PetscInt PetscPThreadRank;
#else
pthread_key_t PetscPThreadRankkey;
#endif

PetscErrorCode PetscThreadCommGetRank_PThread(PetscInt *trank)
{
#if defined(PETSC_PTHREAD_LOCAL)
  *trank = PetscPThreadRank;
#else
  *trank = *((PetscInt*)pthread_getspecific(PetscPThreadRankkey));
#endif
  return 0;
}

/* Sets the attributes for threads */
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommSetAffinity_PThread"
PetscErrorCode PetscThreadCommSetAffinity_PThread(PetscThreadPool pool)
{
  PetscErrorCode      ierr;
  PetscThread_PThread ptcomm;
  PetscBool           set;
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  cpu_set_t           *cpuset;
#endif
  PetscInt            i;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  /* Set affinity for workers */
  ierr = PetscMalloc1(pool->npoolthreads,&cpuset);CHKERRQ(ierr);
  for (i=1; i<pool->npoolthreads; i++) {
    ptcomm = (PetscThread_PThread)pool->poolthreads[i]->data;
    ierr = pthread_attr_init(&ptcomm->attr);CHKERRQ(ierr);
    PetscThreadPoolSetAffinity(pool,&cpuset[i],i,&set);
    if(set) pthread_attr_setaffinity_np(&ptcomm->attr,sizeof(cpu_set_t),&cpuset[i]);
  }

  /* Set affinity for main thread */
  //if (pool->ismainworker) {
    PetscThreadPoolSetAffinity(pool,&cpuset[0],0,&set);
    sched_setaffinity(0,sizeof(cpu_set_t),&cpuset[0]);
    //}
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommDestroy_PThread"
PetscErrorCode PetscThreadCommDestroy_PThread(PetscThreadComm tcomm)
{
  PetscThreadComm_PThread ptcomm = (PetscThreadComm_PThread)tcomm->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  if (!ptcomm) PetscFunctionReturn(0);
  /* Terminate the thread pool */
  if(tcomm->model==THREAD_MODEL_LOOP) {
    ierr = PetscThreadCommFinalize_PThread(tcomm);CHKERRQ(ierr);
  }
  if(tcomm->model==THREAD_MODEL_USER) {
    ierr = pthread_barrier_destroy(&ptcomm->barr);CHKERRQ(ierr);
    ierr = pthread_mutex_destroy(&ptcomm->threadmutex);CHKERRQ(ierr);
  }
  ierr = PetscFree(ptcomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadDestroy_PThread"
PetscErrorCode PetscThreadDestroy_PThread(PetscThread thread)
{
  PetscThread_PThread ptcomm = (PetscThread_PThread)thread->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if(!ptcomm) PetscFunctionReturn(0);
  ierr = PetscFree(ptcomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCreate_PThread"
PETSC_EXTERN PetscErrorCode PetscThreadCreate_PThread(PetscThread thread)
{
  PetscThread_PThread ptcomm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&ptcomm);
  thread->data = (void*)ptcomm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_PThreadLoop"
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_PThreadLoop(PetscThreadComm tcomm)
{
  PetscThreadComm_PThread ptcomm;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  printf("Creating PThread Loop\n");
  ierr = PetscStrcpy(tcomm->type,PTHREAD);CHKERRQ(ierr);
  ierr = PetscNew(&ptcomm);CHKERRQ(ierr);

  pthread_barrier_init(&ptcomm->barr,NULL,tcomm->ncommthreads);
  pthread_mutex_init(&ptcomm->threadmutex,NULL);

  tcomm->data               = (void*)ptcomm;
  tcomm->ops->destroy       = PetscThreadCommDestroy_PThread;
  tcomm->ops->runkernel     = PetscThreadCommRunKernel_PThread;
  tcomm->ops->kernelbarrier = PetscThreadPoolBarrier;
  tcomm->ops->getrank       = PetscThreadCommGetRank_PThread;

  if (tcomm->ismainworker) {
#if defined(PETSC_PTHREAD_LOCAL)
    PetscPThreadRank=0; /* Main thread rank */
#else
    ierr = pthread_key_create(&PetscPThreadRankkey,NULL);CHKERRQ(ierr);
    ierr = pthread_setspecific(PetscPThreadRankkey,&tcomm->commthreads[0]->grank);CHKERRQ(ierr);
#endif
  }

  //ierr = PetscThreadCommSetAffinity_PThread(pool);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_PThreadAuto"
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_PThreadAuto(PetscThreadComm tcomm)
{
  PetscThreadComm_PThread ptcomm;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  printf("Creating PThread Auto\n");
  ierr = PetscStrcpy(tcomm->type,PTHREAD);CHKERRQ(ierr);
  ierr = PetscNew(&ptcomm);CHKERRQ(ierr);

  pthread_barrier_init(&ptcomm->barr,NULL,tcomm->ncommthreads);
  pthread_mutex_init(&ptcomm->threadmutex,NULL);

  tcomm->data                 = (void*)ptcomm;
  tcomm->ops->destroy         = PetscThreadCommDestroy_PThread;
  tcomm->ops->runkernel       = PetscThreadCommRunKernel_PThread;
  tcomm->ops->kernelbarrier   = PetscThreadPoolBarrier;
  tcomm->ops->globalbarrier   = PetscThreadCommBarrier_PThread;
  tcomm->ops->atomicincrement = PetscThreadCommAtomicIncrement_PThread;
  tcomm->ops->getrank         = PetscThreadCommGetRank_PThread;
  tcomm->ops->createthreads   = PetscThreadCommInitialize_PThreadUser;
  tcomm->ops->destroythreads  = PetscThreadCommFinalize_PThread;

  if (tcomm->ismainworker) {
#if defined(PETSC_PTHREAD_LOCAL)
    PetscPThreadRank=0; /* Main thread rank */
#else
    ierr = pthread_key_create(&PetscPThreadRankkey,NULL);CHKERRQ(ierr);
    ierr = pthread_setspecific(PetscPThreadRankkey,&tcomm->commthreads[0]->grank);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_PThreadUser"
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_PThreadUser(PetscThreadComm tcomm)
{
  PetscThreadComm_PThread ptcomm;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  printf("Creating PThread User\n");
  ierr = PetscStrcpy(tcomm->type,PTHREAD);CHKERRQ(ierr);
  ierr = PetscNew(&ptcomm);CHKERRQ(ierr);

  pthread_barrier_init(&ptcomm->barr,NULL,tcomm->ncommthreads);
  pthread_mutex_init(&ptcomm->threadmutex,NULL);

  tcomm->data                 = (void*)ptcomm;
  tcomm->ops->destroy         = PetscThreadCommDestroy_PThread;
  tcomm->ops->runkernel       = PetscThreadCommRunKernel_PThread;
  tcomm->ops->kernelbarrier   = PetscThreadPoolBarrier;
  tcomm->ops->globalbarrier   = PetscThreadCommBarrier_PThread;
  tcomm->ops->atomicincrement = PetscThreadCommAtomicIncrement_PThread;
  tcomm->ops->getrank         = PetscThreadCommGetRank_PThread;
  tcomm->ops->createthreads   = PetscThreadCommInitialize_PThreadUser;
  tcomm->ops->destroythreads  = PetscThreadCommFinalize_PThread;

  if (tcomm->ismainworker) {
#if defined(PETSC_PTHREAD_LOCAL)
    PetscPThreadRank=0; /* Main thread rank */
#else
    ierr = pthread_key_create(&PetscPThreadRankkey,NULL);CHKERRQ(ierr);
    ierr = pthread_setspecific(PetscPThreadRankkey,&tcomm->commthreads[0]->grank);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_PThread"
PetscErrorCode PetscThreadCommRunKernel_PThread(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  printf("rank=%d running kernel\n",0);
  if (tcomm->ismainworker) {
    job->job_status   = THREAD_JOB_RECIEVED;
    tcomm->commthreads[0]->jobdata = job;
    PetscRunKernel(job->commrank,job->nargs, tcomm->commthreads[0]->jobdata);
    job->job_status   = THREAD_JOB_COMPLETED;
  }
  if (tcomm->syncafter) {
    ierr = (*tcomm->ops->kernelbarrier)(tcomm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInitialize_PThreadUser"
PetscErrorCode PetscThreadCommInitialize_PThreadUser(PetscThreadComm tcomm)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscThread_PThread ptcomm;

  PetscFunctionBegin;
  /* Init thread structs */
  for (i=0; i<tcomm->ncommthreads; i++) {
    printf("Creating thread=%d\n",i);
    tcomm->commthreads[i]->status = THREAD_CREATED;
    tcomm->commthreads[i]->tcomm = tcomm;
  }

  /* Create threads */
  for (i=1; i<tcomm->ncommthreads; i++) {
    ptcomm = (PetscThread_PThread)tcomm->commthreads[i]->data;
    ierr = pthread_create(&ptcomm->tid,&ptcomm->attr,&PetscThreadPoolFunc,&tcomm->commthreads[i]);CHKERRQ(ierr);
  }

  if (tcomm->ismainworker) tcomm->commthreads[0]->status = THREAD_INITIALIZED;

  PetscInt threads_initialized=0;
  /* Wait till all threads have been initialized */
  while (threads_initialized != tcomm->ncommthreads) {
    threads_initialized=0;
    for (i=0; i<tcomm->ncommthreads; i++) {
      if (!tcomm->commthreads[i]->status) break;
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
  PetscThread_PThread ptcomm;
  PetscInt                i;

  PetscFunctionBegin;
  ierr = (*tcomm->ops->kernelbarrier)(tcomm);CHKERRQ(ierr);
  for (i=0; i<tcomm->ncommthreads; i++) {
    printf("Terminating thread=%d\n",i);
    tcomm->commthreads[i]->status = THREAD_TERMINATE;
  }
  for (i=1; i<tcomm->ncommthreads; i++) {
    ptcomm = (PetscThread_PThread)tcomm->commthreads[i]->data;
    ierr = pthread_join(ptcomm->tid,&jstatus);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommBarrier_PThread"
PetscErrorCode PetscThreadCommBarrier_PThread(PetscThreadComm tcomm)
{
  PetscThreadComm_PThread ptcomm = (PetscThreadComm_PThread)tcomm->data;

  PetscFunctionBegin;
  pthread_barrier_wait(&ptcomm->barr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommAtomicIncrement_PThread"
PetscErrorCode PetscThreadCommAtomicIncrement_PThread(PetscThreadComm tcomm,PetscInt *val,PetscInt inc)
{
  PetscThreadComm_PThread ptcomm = (PetscThreadComm_PThread)tcomm->data;

  PetscFunctionBegin;
  pthread_mutex_lock(&ptcomm->threadmutex);
  (*val)+=inc;
  pthread_mutex_unlock(&ptcomm->threadmutex);
  PetscFunctionReturn(0);
}

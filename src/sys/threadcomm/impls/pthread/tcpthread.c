/* Define feature test macros to make sure CPU_SET and other functions are available */
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

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetRank_PThread"
/*
   PetscThreadCommGetRank_PThread - Get the rank of the calling thread

   Not Collective

   Output Parameters:
.  trank - Rank of calling thread

   Level: developer

*/
PetscErrorCode PetscThreadCommGetRank_PThread(PetscInt *trank)
{
#if defined(PETSC_PTHREAD_LOCAL)
  *trank = PetscPThreadRank;
#else
  *trank = *((PetscInt*)pthread_getspecific(PetscPThreadRankkey));
#endif
  return 0;
}

/*
   PetscThreadCommSetAffinity_PThread - Set affinity for a PThread thread

   Not Collective

   Input Parameters:
+  pool   - Threadpool with thread model settings
-  thread - Thread whose affinity is being set

   Level: developer

   Notes:
   Sets the affinity for the thread by creating and setting a pthread attribute. When a
   pthread is created, it uses this attribute to set the affinity of the thread.

*/
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommSetAffinity_PThread"
PetscErrorCode PetscThreadCommSetAffinity_PThread(PetscThreadPool pool,PetscThread thread)
{
  PetscErrorCode      ierr;
  PetscThread_PThread ptcomm;
  PetscBool           set;
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  cpu_set_t           cpuset;
#endif

  PetscFunctionBegin;
  printf("in setaff_pthread\n");
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  /* Set affinity for workers */
  ptcomm = (PetscThread_PThread)thread->data;
  ierr = pthread_attr_init(&ptcomm->attr);CHKERRQ(ierr);
  PetscThreadPoolSetAffinity(pool,&cpuset,thread->affinity,&set);
  if (set) pthread_attr_setaffinity_np(&ptcomm->attr,sizeof(cpu_set_t),&cpuset);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommDestroy_PThread"
/*
   PetscThreadCommDestroy_PThread - Destroy the PThread specific threadcomm struct

   Not Collective

   Input Parameters:
.  tcomm - Threadcomm to destroy

   Level: developer

   Notes:
   Destroys the barrier and mutex for the threadcomm.

*/
PetscErrorCode PetscThreadCommDestroy_PThread(PetscThreadComm tcomm)
{
  PetscThreadComm_PThread ptcomm = (PetscThreadComm_PThread)tcomm->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  if (!ptcomm) PetscFunctionReturn(0);
  /* Destroy pthread threadcomm data */
  if (tcomm->model==THREAD_MODEL_USER || tcomm->model==THREAD_MODEL_AUTO) {
    ierr = pthread_barrier_destroy(&ptcomm->barr);CHKERRQ(ierr);
  }
  ierr = PetscFree(ptcomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolDestroy_PThread"
/*
   PetscThreadPoolDestroy_PThread - Destroy the PThread specific threadpool structures and
                                    terminate any threads created by Petsc

   Not Collective

   Input Parameters:
.  pool - Threadpool with input settings

   Level: developer

   Notes:
   In order to destroy a threadpool, the associated threadcomms must be destroyed.
   Threadcomm destroy has a barrier to verify that all threads have finished their jobs.
   This routine terminates the thread in the threadpool and then joins the thread.
   Then PThreads specific structs are then destroyed.

*/
PetscErrorCode PetscThreadPoolDestroy_PThread(PetscThreadPool pool)
{
  PetscThread_PThread ptcomm;
  PetscInt            i;
  PetscErrorCode      ierr;
  void                *status;

  PetscFunctionBegin;
  /* Terminate the thread pool */
  if (pool->model==THREAD_MODEL_LOOP || pool->model==THREAD_MODEL_AUTO) {
    for (i=0; i<pool->npoolthreads; i++) {
      printf("Terminating thread=%d\n",i);
      pool->poolthreads[i]->status = THREAD_TERMINATE;
    }
    for (i=1; i<pool->npoolthreads; i++) {
      ptcomm = (PetscThread_PThread)pool->poolthreads[i]->data;
      ierr = pthread_join(ptcomm->tid,&status);CHKERRQ(ierr);
    }
  }

  /* Destroy pthread thread data */
  for (i=0; i<pool->npoolthreads; i++) {
    ptcomm = (PetscThread_PThread)pool->poolthreads[i]->data;
    pthread_attr_destroy(&ptcomm->attr);
    ierr = PetscFree(ptcomm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCreate_PThread"
/*
   PetscThreadCreate_PThread - Create a PThread data struct

   Not Collective

   Input Parameters:
.  thread - Thread to create a PThread struct for

   Level: developer

*/
PETSC_EXTERN PetscErrorCode PetscThreadCreate_PThread(PetscThread thread)
{
  PetscThread_PThread ptcomm;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  printf("Create pthread\n");
  ierr = PetscNew(&ptcomm);
  thread->data = (void*)ptcomm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadInit_PThread"
PETSC_EXTERN PetscErrorCode PetscThreadInit_PThread()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ThreadType = THREAD_TYPE_PTHREAD;
  ierr = PetscThreadLockInitialize_PThread();CHKERRQ(ierr);
  PetscThreadLockAcquire = PetscThreadLockAcquire_PThread;
  PetscThreadLockRelease = PetscThreadLockRelease_PThread;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolInit_PThread"
/*
   PetscThreadPoolInit_PThread - Initialize PThread specific threadpool variables

   Not Collective

   Input Parameters:
.  pool - Threadpool to initialize

   Level: developer

*/
PETSC_EXTERN PetscErrorCode PetscThreadPoolInit_PThread(PetscThreadPool pool)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("Init PThread\n");
  ierr                     = PetscStrcpy(pool->type,PTHREAD);CHKERRQ(ierr);
  pool->threadtype         = THREAD_TYPE_PTHREAD;
  pool->ops->createthread  = PetscThreadCreate_PThread;
  pool->ops->startthreads  = PetscThreadCommInitialize_PThread;
  pool->ops->setaffinities = PetscThreadCommSetAffinity_PThread;
  pool->ops->pooldestroy   = PetscThreadPoolDestroy_PThread;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInit_PThread"
/*
   PetscThreadCommInit_PThread - Initialize PThread specific threadcomm variables

   Not Collective

   Input Parameters:
.  tcomm - Threadcomm to initialize

   Level: developer

   Notes:
   Initializes the PThread specific threadcomm structs, creates a pthread barrier, and
   creates a pthread mutex.

*/
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_PThread(PetscThreadComm tcomm)
{
  PetscThreadComm_PThread ptcomm;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  printf("Creating PThread\n");
  ierr = PetscNew(&ptcomm);CHKERRQ(ierr);

  ierr = pthread_barrier_init(&ptcomm->barr,PETSC_NULL,tcomm->ncommthreads);CHKERRQ(ierr);

  tcomm->data             = (void*)ptcomm;
  tcomm->ops->commdestroy = PetscThreadCommDestroy_PThread;
  tcomm->ops->runkernel   = PetscThreadCommRunKernel_PThread;
  tcomm->ops->barrier     = PetscThreadCommBarrier_PThread;
  tcomm->ops->getrank     = PetscThreadCommGetRank_PThread;

  if (tcomm->ismainworker) {
#if defined(PETSC_PTHREAD_LOCAL)
    PetscPThreadRank=0; /* Main thread rank */
#else
    ierr = pthread_key_create(&PetscPThreadRankkey,PETSC_NULL);CHKERRQ(ierr);
    ierr = pthread_setspecific(PetscPThreadRankkey,&tcomm->commthreads[0]->grank);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_PThread"
/*
   PetscThreadCommRunKernel_PThread - Run a job on a threadcomm

   Not Collective

   Input Parameters:
+  tcomm - Threadcomm to run kernel on
-  job   - Job to run on threadcomm

   Level: developer

   Notes:
   Runs the job for the threadcomm. Synchronization after running the kernel is user optional.
   If user does not synchronize here, then the user must use synchronization in their code.

*/
PetscErrorCode PetscThreadCommRunKernel_PThread(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscThreadCommJobQueue jobqueue;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  printf("rank=%d running kernel\n",0);
  // Do work for main thread
  if (tcomm->ismainworker) {
    job->job_status   = THREAD_JOB_RECEIVED;
    tcomm->commthreads[0]->jobdata = job;
    ierr = PetscRunKernel(job->commrank,job->nargs, tcomm->commthreads[0]->jobdata);CHKERRQ(ierr);
    job->job_status = THREAD_JOB_COMPLETED;
    jobqueue = tcomm->commthreads[tcomm->lleader]->jobqueue;
    jobqueue->current_job_index = (jobqueue->current_job_index+1)%tcomm->nkernels;
    jobqueue->completed_jobs_ctr++;
  }
  // Synchronize
  if (tcomm->syncafter) {
    ierr = PetscThreadCommJobBarrier(tcomm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInitialize_PThread"
/*
   PetscThreadCommInitialize_PThread - Initialize and create the threads in the threadpool

   Not Collective

   Input Parameters:
.  pool - Threadpool to initialize

   Level: developer

   Notes:
   Initializes the threads, creates the pthreads, and verifies that each thread is ready
   to receive work.

*/
PetscErrorCode PetscThreadCommInitialize_PThread(PetscThreadPool pool)
{
  PetscErrorCode      ierr;
  PetscInt            i,threads_initialized=0;
  PetscThread_PThread ptcomm;

  PetscFunctionBegin;
  /* Init thread structs */
  for (i=0; i<pool->npoolthreads; i++) {
    printf("Creating thread=%d\n",i);
    pool->poolthreads[i]->status = THREAD_CREATED;
    pool->poolthreads[i]->pool = pool;
  }

  /* Create threads */
  for (i=pool->thread_start; i<pool->npoolthreads; i++) {
    printf("Creating thread %d\n",i);
    ptcomm = (PetscThread_PThread)pool->poolthreads[i]->data;
    ierr = pthread_create(&ptcomm->tid,&ptcomm->attr,&PetscThreadPoolFunc,&pool->poolthreads[i]);CHKERRQ(ierr);
  }

  if (pool->ismainworker) pool->poolthreads[0]->status = THREAD_INITIALIZED;

  /* Wait till all threads have been initialized */
  while (threads_initialized != pool->npoolthreads) {
    threads_initialized=0;
    for (i=0; i<pool->npoolthreads; i++) {
      if (!pool->poolthreads[i]->status) break;
      threads_initialized++;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommBarrier_PThread"
/*
   PetscThreadCommBarrier_PThread - PThreads barrier for threadcomm

   Collective on threadcomm

   Input Parameters:
.  tcomm - Threadcomm to use barrier on

   Level: developer

   Notes:
   Must be called by all threads.

*/
PetscErrorCode PetscThreadCommBarrier_PThread(PetscThreadComm tcomm)
{
  PetscThreadComm_PThread ptcomm = (PetscThreadComm_PThread)tcomm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_Barrier,0,0,0,0);CHKERRQ(ierr);
  pthread_barrier_wait(&ptcomm->barr);
  ierr = PetscLogEventEnd(ThreadComm_Barrier,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadLockInitialize_PThread"
PetscErrorCode PetscThreadLockInitialize_PThread(void)
{
  PetscThreadLock_PThread ptlock;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscLocks) PetscFunctionReturn(0);

  ierr = PetscNew(&PetscLocks);CHKERRQ(ierr);
  ierr = PetscNew(&ptlock);CHKERRQ(ierr);
  ierr = pthread_mutex_init(ptlock,PETSC_NULL);CHKERRQ(ierr);
  PetscLocks->trmalloc_lock = (void*)ptlock;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadLockAcquire_PThread"
PetscErrorCode PetscThreadLockAcquire_PThread(void *lock)
{
  PetscThreadLock_PThread ptlock;

  PetscFunctionBegin;
  ptlock = (PetscThreadLock_PThread)lock;
  pthread_mutex_lock(ptlock);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadLockRelease_PThread"
PetscErrorCode PetscThreadLockRelease_PThread(void *lock)
{
  PetscThreadLock_PThread ptlock;

  PetscFunctionBegin;
  ptlock = (PetscThreadLock_PThread)lock;
  pthread_mutex_unlock(ptlock);
  PetscFunctionReturn(0);
}

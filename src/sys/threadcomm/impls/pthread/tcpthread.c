/* Define feature test macros to make sure CPU_SET and other functions are available
 */
#define PETSC_DESIRE_FEATURE_TEST_MACROS

#include <../src/sys/threadcomm/impls/pthread/tcpthreadimpl.h>

#if defined(PETSC_PTHREAD_LOCAL)
PETSC_PTHREAD_LOCAL PetscInt PetscPThreadRank;
#else
pthread_key_t PetscPThreadRankkey;
#endif

static PetscBool PetscPThreadCommInitializeCalled = PETSC_FALSE;

static PetscInt ptcommcrtct = 0; /* PThread communicator creation count. Incremented whenever a pthread
                                    communicator is created and decremented when it is destroyed. On the
                                    last pthread communicator destruction, the thread pool is also terminated
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

/* Sets the attributes for threads */
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommSetAffinity_PThread"
PetscErrorCode PetscThreadCommSetAffinity_PThread(PetscThreadComm tcomm)
{
  PetscErrorCode          ierr;
  PetscThreadComm_PThread ptcomm=(PetscThreadComm_PThread)tcomm->data;
  pthread_attr_t          *attr =ptcomm->attr;
  PetscBool               set;
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  cpu_set_t               *cpuset;
#endif
  PetscInt                i;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  /* Set affinity for workers */
  ierr = PetscMalloc1(tcomm->nworkThreads,&cpuset);CHKERRQ(ierr);
  for (i=tcomm->pool->thread_num_start; i<tcomm->nworkThreads; i++) {
    ierr = pthread_attr_init(&attr[i]);CHKERRQ(ierr);
    PetscThreadPoolSetAffinity(tcomm,&cpuset[i],i,&set);
    if(set) pthread_attr_setaffinity_np(&attr[i],sizeof(cpu_set_t),&cpuset[i]);
  }

  /* Set affinity for main thread */
  if (tcomm->pool->ismainworker) {
    PetscThreadPoolSetAffinity(tcomm,&cpuset[0],0,&set);
    sched_setaffinity(0,sizeof(cpu_set_t),&cpuset[0]);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommDestroy_PThread"
PetscErrorCode PetscThreadCommDestroy_PThread(PetscThreadComm tcomm)
{
  PetscThreadComm_PThread ptcomm=(PetscThreadComm_PThread)tcomm->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  if (!ptcomm) PetscFunctionReturn(0);
  ptcommcrtct--;
  if (!ptcommcrtct) {
    /* Terminate the thread pool */
    ierr = PetscThreadCommFinalize_PThread(tcomm);CHKERRQ(ierr);
    ierr = PetscFree(ptcomm->tid);CHKERRQ(ierr);
    ierr = PetscFree(ptcomm->attr);CHKERRQ(ierr);
    PetscPThreadCommInitializeCalled = PETSC_FALSE;
  }
  ierr = PetscFree(ptcomm);CHKERRQ(ierr);
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
  ptcommcrtct++;
  ierr = PetscStrcpy(tcomm->type,PTHREAD);CHKERRQ(ierr);
  ierr = PetscNew(&ptcomm);CHKERRQ(ierr);

  tcomm->data              = (void*)ptcomm;
  tcomm->ops->destroy      = PetscThreadCommDestroy_PThread;
  tcomm->ops->runkernel    = PetscThreadPoolRunKernel_PThread;
  tcomm->ops->barrier      = PetscThreadPoolBarrier;
  tcomm->ops->getrank      = PetscThreadCommGetRank_PThread;

  if (!PetscPThreadCommInitializeCalled) { /* Only done for PETSC_THREAD_COMM_WORLD */
    PetscPThreadCommInitializeCalled = PETSC_TRUE;

    if (tcomm->pool->ismainworker) {
#if defined(PETSC_PTHREAD_LOCAL)
      PetscPThreadRank=0; /* Main thread rank */
#else
      ierr = pthread_key_create(&PetscPThreadRankkey,NULL);CHKERRQ(ierr);
      ierr = pthread_setspecific(PetscPThreadRankkey,&tcomm->pool->granks[0]);CHKERRQ(ierr);
#endif
    }

    /* Create array holding pthread ids */
    ierr = PetscMalloc1(tcomm->nworkThreads,&ptcomm->tid);CHKERRQ(ierr);
    /* Create thread attributes */
    ierr = PetscMalloc1(tcomm->nworkThreads,&ptcomm->attr);CHKERRQ(ierr);
    ierr = PetscThreadCommSetAffinity_PThread(tcomm);CHKERRQ(ierr);

    /* Initialize thread pool */
    ierr = PetscThreadCommInitialize_PThread(tcomm);CHKERRQ(ierr);

  } else {
    PetscThreadComm         gtcomm;
    PetscThreadComm_PThread gptcomm;
    PetscInt                *gaffinities;

    ierr        = PetscCommGetThreadComm(PETSC_COMM_WORLD,&gtcomm);CHKERRQ(ierr);
    gaffinities = gtcomm->affinities;
    gptcomm     = (PetscThreadComm_PThread)tcomm->data;
    /* Copy over the data from the global thread communicator structure */
    tcomm->ops->runkernel    = gtcomm->ops->runkernel;
    tcomm->ops->barrier      = gtcomm->ops->barrier;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_PThreadUser"
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_PThreadUser(PetscThreadComm tcomm)
{
  PetscFunctionBegin;
  printf("Creating PThread User\n");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolInitialize_PThread"
PetscErrorCode PetscThreadPoolInitialize_PThread(PetscThreadComm tcomm)
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
#define __FUNCT__ "PetscThreadPoolRunKernel_PThread"
PetscErrorCode PetscThreadPoolRunKernel_PThread(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscErrorCode          ierr;
  PetscThreadComm_PThread ptcomm;
  PetscThreadCommJobQueue jobqueue=tcomm->pool->jobqueue;

  PetscFunctionBegin;
  printf("rank=%d running kernel\n",0);
  ptcomm = (PetscThreadComm_PThread)tcomm->data;
  if (tcomm->pool->ismainworker) {
    job->job_status[0]   = THREAD_JOB_RECIEVED;
    jobqueue->tinfo[0]->data = job;
    PetscRunKernel(0,job->nargs, jobqueue->tinfo[0]->data);
    job->job_status[0]   = THREAD_JOB_COMPLETED;
  }
  if (tcomm->pool->synchronizeafter) {
    ierr = (*tcomm->ops->barrier)(tcomm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

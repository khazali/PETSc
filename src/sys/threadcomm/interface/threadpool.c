/* Define feature test macros to make sure CPU_SET and other functions are available */
#define PETSC_DESIRE_FEATURE_TEST_MACROS

#include <petscthreadcomm.h>
#include <petsc-private/threadcommimpl.h>

PetscInt          N_CORES                                = -1;
PetscBool         PetscThreadCommRegisterAllModelsCalled = PETSC_FALSE;
PetscBool         PetscThreadCommRegisterAllTypesCalled  = PETSC_FALSE;

/*
  PetscPThreadCommAffinityPolicy - Core affinity policy for pthreads

$ THREADAFFPOLICY_ALL     - threads can run on any core. OS decides thread scheduling
$ THREADAFFPOLICY_ONECORE - threads can run on only one core.
$ THREADAFFPOLICY_NONE    - No set affinity policy. OS decides thread scheduling
*/
const char *const PetscThreadCommAffPolicyTypes[] = {"ALL","ONECORE","NONE","PetscPThreadCommAffinityPolicyType","THREADAFFPOLICY_",0};

#undef __FUNCT__
#define __FUNCT__ "PetscGetNCores"
/*@
   PetscGetNCores - Gets the number of available cores on the system

   Not Collective

   Output Parameters:
.  ncores - The number of available cores

   Level: developer

   Notes:
   Defaults to 1 if the available core count cannot be found.

@*/
PetscErrorCode PetscGetNCores(PetscInt *ncores)
{
  PetscFunctionBegin;
  if (N_CORES == -1) {
    N_CORES = 1; /* Default value if number of cores cannot be found out */

#if defined(PETSC_HAVE_SYS_SYSINFO_H) && (PETSC_HAVE_GET_NPROCS) /* Linux */
    N_CORES = get_nprocs();
#elif defined(PETSC_HAVE_SYS_SYSCTL_H) && (PETSC_HAVE_SYSCTLBYNAME) /* MacOS, BSD */
    {
      PetscErrorCode ierr;
      size_t         len = sizeof(N_CORES);
      ierr = sysctlbyname("hw.activecpu",&N_CORES,&len,PETSC_NULL,0); /* osx preferes activecpu over ncpu */
      if (ierr) { /* freebsd check ncpu */
        sysctlbyname("hw.ncpu",&N_CORES,&len,PETSC_NULL,0);
        /* continue even if there is an error */
      }
    }
#elif defined(PETSC_HAVE_WINDOWS_H)   /* Windows */
    {
      SYSTEM_INFO sysinfo;
      GetSystemInfo(&sysinfo);
      N_CORES = sysinfo.dwNumberOfProcessors;
    }
#endif
  }
  if (ncores) *ncores = N_CORES;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolGetPool"
/*@
   PetscThreadPoolGetPool - Get the threadpool for this MPI_Comm

   Not Collective

   Input Parameters:
.  comm - MPI communicator

   Output Parameters:
.  pool - Threadpool

   Level: developer

   Notes:
   Returns a threadpool if it exists, returns PETSC_NULL if not.

@*/
PetscErrorCode PetscThreadPoolGetPool(MPI_Comm comm,PetscThreadPool *pool)
{
  PetscThreadComm tcomm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);CHKERRQ(ierr);
  if (tcomm) {
    *pool = tcomm->pool;
  } else {
    *pool = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolAlloc"
/*@
   PetscThreadPoolAlloc - Allocate threadpool object

   Not Collective

   Input Parameters:
.  pool - Threadpool to allocate

   Level: developer

   Notes:
   Allocates a threadpool and sets initial values for each variable.
   Variables are set to indiciate that the threadpool has not been initialized.

@*/
PetscErrorCode PetscThreadPoolAlloc(PetscThreadPool *pool)
{
  PetscErrorCode  ierr;
  PetscThreadPool poolout;

  PetscFunctionBegin;
  *pool                 = PETSC_NULL;
  ierr                  = PetscNew(&poolout);CHKERRQ(ierr);

  poolout->refct        = 0;
  poolout->npoolthreads = -1;
  poolout->poolthreads  = PETSC_NULL;

  poolout->model        = THREAD_MODEL_LOOP;
  poolout->threadtype   = THREAD_TYPE_NOTHREAD;
  poolout->aff          = THREADAFFPOLICY_ONECORE;
  poolout->nkernels     = 16;
  poolout->thread_start = -1;
  poolout->ismainworker = PETSC_TRUE;
  ierr                  = PetscNew(&poolout->ops);CHKERRQ(ierr);

  *pool = poolout;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCreateJobQueue"
/*@
   PetscThreadCreateJobQueue - Create a job queue for a thread

   Not Collective

   Input Parameters:
+  thread - Thread struct to create a jobqueue for
-  pool   - Threadpool with threadcomm settings

   Level: developer

   Notes:
   Creates an empty jobqueue based on threadpool settings.

@*/
PetscErrorCode PetscThreadCreateJobQueue(PetscThread thread,PetscThreadPool pool)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Allocate queue
  ierr = PetscNew(&thread->jobqueue);CHKERRQ(ierr);

  // Create job contexts
  ierr = PetscMalloc1(pool->nkernels,&thread->jobqueue->jobs);CHKERRQ(ierr);
  for (i=0; i<pool->nkernels; i++) {
    thread->jobqueue->jobs[i].job_status = THREAD_JOB_NONE;
  }

  // Set queue variables
  thread->jobqueue->next_job_index     = 0;
  thread->jobqueue->total_jobs_ctr     = 0;
  thread->jobqueue->newest_job_index   = 0;
  thread->jobqueue->current_job_index  = 0;
  thread->jobqueue->completed_jobs_ctr = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolInitialize"
/*@
   PetscThreadPoolInitialize - Initialize a threadpool

   Not Collective

   Input Parameters:
+  pool     - Threadpool
-  nthreads - Number of threads to put in threadpool

   Options Database keys:
   -threadcomm_nkernels <nkernels>

   Level: developer

   Notes:
   Initializes threadpool and creates and initializes each thread.
   PETSc sets the global ranks from 0 to nthreads-1.
@*/
PetscErrorCode PetscThreadPoolInitialize(PetscThreadPool pool,PetscInt nthreads)
{
  PetscInt       i,ncores;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("Creating thread pool\n");
  // Set threadpool variables
  pool->model = ThreadModel;
  ierr = PetscThreadPoolSetNThreads(pool,nthreads);CHKERRQ(ierr);
  ierr = PetscThreadPoolSetType(pool,NOTHREAD);CHKERRQ(ierr);

  if (pool->model == THREAD_MODEL_LOOP) {
    pool->ismainworker = PETSC_TRUE;
    pool->thread_start = 1;
  } else if (pool->model == THREAD_MODEL_AUTO) {
    pool->ismainworker = PETSC_FALSE;
    pool->thread_start = 0;
  } else if (pool->model == THREAD_MODEL_USER) {
    pool->ismainworker = PETSC_TRUE;
    pool->thread_start = 1;
  }

  // Get option settings from command line
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Threadcomm options",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-threadcomm_nkernels","number of kernels that can be launched simultaneously","",16,&pool->nkernels,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  // Create thread structs for pool
  ierr = PetscGetNCores(&ncores);CHKERRQ(ierr);
  ierr = PetscMalloc1(pool->npoolthreads,&pool->poolthreads);CHKERRQ(ierr);
  for (i=0; i<pool->npoolthreads; i++) {
    ierr = PetscNew(&pool->poolthreads[i]);CHKERRQ(ierr);
    pool->poolthreads[i]->prank = i % ncores;
    printf("Pool index=%d prank=%d\n",i,pool->poolthreads[i]->prank);
    pool->poolthreads[i]->pool     = PETSC_NULL;
    pool->poolthreads[i]->status   = 0;
    pool->poolthreads[i]->jobdata  = PETSC_NULL;
    pool->poolthreads[i]->affinity = i % ncores;
    pool->poolthreads[i]->jobqueue = PETSC_NULL;
    pool->poolthreads[i]->data     = PETSC_NULL;

    ierr = PetscThreadCreateJobQueue(pool->poolthreads[i],pool);CHKERRQ(ierr);
    if (pool->threadtype == THREAD_TYPE_PTHREAD) {
      ierr = pool->ops->createthread(pool->poolthreads[i]);CHKERRQ(ierr);
    }
  }

  printf("Initialized pool with %d threads\n",pool->npoolthreads);
  pool->refct++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolSetType"
/*@
   PetscThreadPoolSetType - Sets the threading type for the thread communicator

   Logically Collective

   Input Parameters:
+  tcomm - the thread communicator
-  type  - the type of thread model needed

   Options Database keys:
   -threadcomm_type <type>

   Available types
   See "petsc/include/petscthreadcomm.h" for available types

   Level: developer

   Notes:
   Sets type of threadpool by checking thread type function list and calling
   the appropriate function.

@*/
PetscErrorCode PetscThreadPoolSetType(PetscThreadPool pool,PetscThreadCommType type)
{
  PetscBool      flg;
  PetscErrorCode ierr,(*r)(PetscThreadPool);

  PetscFunctionBegin;
  PetscValidCharPointer(type,2);
  if (!PetscThreadCommRegisterAllTypesCalled) { ierr = PetscThreadCommRegisterAllTypes();CHKERRQ(ierr);}

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Threadcomm type - setting threading type",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-threadcomm_type","Threadcomm type","PetscThreadCommSetType",PetscThreadCommTypeList,type,pool->type,256,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  // Find and call threadcomm init function
  if (flg) {
    ierr = PetscFunctionListFind(PetscThreadPoolTypeList,pool->type,&r);CHKERRQ(ierr);
    if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested threadpool type %s",pool->type);
    ierr = (*r)(pool);CHKERRQ(ierr);
  } else PetscStrcpy(pool->type,NOTHREAD);

  // Find threadcomm create function
  ierr = PetscFunctionListFind(PetscThreadCommTypeList,pool->type,&pool->ops->tcomminit);CHKERRQ(ierr);
  if (!pool->ops->tcomminit) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested threadcomm type %s",pool->type);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolCreate"
/*@
   PetscThreadPoolCreate - Creates a threadpool for a threadcomm

   Not collective

   Input Parameters:
+  tcomm      - Threadcomm to create threadpool for
.  nthreads   - Number of threads to create in pool or PETSC_DECIDE
-  affinities - Core affinities for each thread in pool or PETSC_DECIDE

   Level: developer

   Notes:
   Allocates and initializes a threadpool. Can create threads and add them to a threadpool.
   Passing PETSC_DECIDE for nthreads gets the number of threads from user input option
   or uses the number of cores available.
   Passing PETSC_DECIDE for affinities initializes the affinities
   from 0 to nthreads-1.

@*/
PetscErrorCode PetscThreadPoolCreate(PetscThreadComm tcomm,PetscInt nthreads,PetscInt *affinities)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("Creating ThreadPool\n");
  ierr = PetscThreadPoolAlloc(&tcomm->pool);CHKERRQ(ierr);
  ierr = PetscThreadPoolInitialize(tcomm->pool,nthreads);CHKERRQ(ierr);
  printf("Setting affinities in threadpool\n");

  // Set thread affinities in thread struct
  ierr = PetscThreadPoolSetAffinities(tcomm->pool,affinities);CHKERRQ(ierr);

  // Create threads and put in pool
  if (tcomm->pool->threadtype == THREAD_TYPE_PTHREAD && (tcomm->pool->model == THREAD_MODEL_AUTO || tcomm->pool->model == THREAD_MODEL_LOOP)) {
    ierr = (*tcomm->pool->ops->startthreads)(tcomm->pool);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolSetNThreads"
/*@
   PetscThreadCommSetNThreads - Set the thread count for the thread communicator

   Not collective

   Input Parameters:
+  tcomm    - Thread communicator
-  nthreads - Number of threads

   Options Database keys:
   -threadcomm_nthreads <nthreads> Number of threads to use

   Level: developer

   Notes:
   Defaults to using 1 thread.
   Use nthreads = PETSC_DECIDE or -threadcomm_nthreads PETSC_DECIDE for PETSc to decide
   the number of threads.

.seealso: PetscThreadCommGetNThreads()
@*/
PetscErrorCode PetscThreadPoolSetNThreads(PetscThreadPool pool,PetscInt nthreads)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscInt       nthr;

  PetscFunctionBegin;
  // Set number of threads to 1 if not using nothreads
  if (pool->type == THREAD_TYPE_NOTHREAD) {
    pool->npoolthreads = 1;
    PetscFunctionReturn(0);
  }

  // Check input options for number of threads
  if (nthreads == PETSC_DECIDE) {
    pool->npoolthreads = 1;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Thread comm - setting number of threads",PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-threadcomm_nthreads","number of threads to use in the thread communicator","PetscThreadPoolSetNThreads",1,&nthr,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (flg) {
      if (nthr == PETSC_DECIDE) pool->npoolthreads = N_CORES;
      else pool->npoolthreads = nthr;
    }
  } else pool->npoolthreads = nthreads;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolGetNThreads"
/*@C
   PetscThreadPoolGetNThreads - Gets the thread count from the thread communicator
                                associated with the MPI communicator

   Not collective

   Input Parameters:
.  comm - the MPI communicator

   Output Parameters:
.  nthreads - number of threads

   Level: developer

   Notes:
   Return -1 if the threadpool has not been created yet.

.seealso: PetscThreadCommSetNThreads()
@*/
PetscErrorCode PetscThreadPoolGetNThreads(MPI_Comm comm,PetscInt *nthreads)
{
  PetscErrorCode  ierr;
  PetscThreadPool pool;

  PetscFunctionBegin;
  ierr = PetscThreadPoolGetPool(comm,&pool);CHKERRQ(ierr);
  if (!pool) {
    *nthreads = pool->npoolthreads;
  } else {
    *nthreads = -1;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolSetAffinities"
/*
   PetscThreadPoolSetAffinities - Sets the core affinity for threads
                                  (which threads run on which cores)

   Not collective

   Input Parameters:
+  pool       - the threadpool
-  affinities - array of core affinity for threads

   Options Database keys:
+  -threadpool_affpolicy  <affinity policy>
-  -threadpool_affinities <list of thread affinities>

   Level: developer

   Notes:
   Use affinities = PETSC_NULL for PETSc to decide the affinities.
   If PETSc decides affinities, then each thread has affinity to
   a unique core with the main thread on Core 0, thread0 on core 1,
   and so on. If the thread count is more the number of available
   cores then multiple threads share a core.

   The first value is the affinity for the main thread

   The affinity list can be passed as
   a comma seperated list:                                 0,1,2,3,4,5,6,7
   a range (start-end+1):                                  0-8
   a range with given increment (start-end+1:inc):         0-7:2
   a combination of values and ranges seperated by commas: 0,1-8,8-15:2

   There must be no intervening spaces between the values.

.seealso: PetscThreadCommGetAffinities(), PetscThreadCommSetNThreads()
*/
PetscErrorCode PetscThreadPoolSetAffinities(PetscThreadPool pool,const PetscInt affinities[])
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscInt       i,*affopt,nmax=pool->npoolthreads;

  PetscFunctionBegin;
  printf("In poolsetaffinities\n");
  /* Do not need to set thread pool affinities if no threads */
  if (pool->threadtype == THREAD_TYPE_NOTHREAD) PetscFunctionReturn(0);

  /* If user did not pass in affinity settings */
  if (!affinities) {

    /* Check if option is present in the options database */
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Thread comm - setting thread affinities",PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-threadcomm_affpolicy","Thread affinity policy"," ",PetscThreadCommAffPolicyTypes,(PetscEnum)pool->aff,(PetscEnum*)&pool->aff,&flg);CHKERRQ(ierr);
    ierr = PetscMalloc1(pool->npoolthreads,&affopt);
    ierr = PetscOptionsIntArray("-threadcomm_affinities","Set core affinities of threads","PetscThreadCommSetAffinities",affopt,&nmax,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    /* If user passes in array from command line, use those affinities */
    if (flg) {
      if (nmax != pool->npoolthreads) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Must set affinities for all threads, Threads = %D, Core affinities set = %D",pool->npoolthreads,nmax);
      for (i=0; i<pool->npoolthreads; i++) pool->poolthreads[i]->affinity = affopt[i];
      pool->aff = THREADAFFPOLICY_ONECORE;
    }
    PetscFree(affopt);
  } else {
    /* Use affinities from input parameter */
    for (i=0; i<pool->npoolthreads; i++) pool->poolthreads[i]->affinity = affinities[i];
    pool->aff = THREADAFFPOLICY_ONECORE;
  }
  /* Set affinities based on thread policy and settings of each threads affinities */
  if (pool->threadtype == THREAD_TYPE_PTHREAD) {
    for (i=0; i<pool->npoolthreads; i++) {
      ierr = (*pool->ops->setaffinities)(pool,pool->poolthreads[i]);CHKERRQ(ierr);
    }
  }
  if (pool->threadtype == THREAD_TYPE_OPENMP && pool->model == THREAD_MODEL_LOOP) {
    ierr = (*pool->ops->setaffinities)(pool,PETSC_NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolSetAffinity"
/*
   PetscThreadPoolSetAffinity - Set affinity for a thread

   Not collective

   Input Parameters:
+  pool     - Threadpool with settings
.  cpuset   - Pointer to cpuset to set
-  affinity - Affinity of thread to set

   Output Parameters:
.  set      - True if affinity was set, false if not

   Level: developer

   Notes:
   Sets affinity for a thread based on the thread policy for the threadpool.
*/
PetscErrorCode PetscThreadPoolSetAffinity(PetscThreadPool pool,cpu_set_t *cpuset,PetscInt affinity,PetscBool *set)
{
  PetscInt       ncores,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("in poolsetaff\n");
  ierr = PetscGetNCores(&ncores);CHKERRQ(ierr);
  switch (pool->aff) {
  case THREADAFFPOLICY_ONECORE:
    CPU_ZERO(cpuset);
    printf("Setting thread affinity to core %d\n",affinity%ncores);
    CPU_SET(affinity%ncores,cpuset);
    *set = PETSC_TRUE;
    break;
  case THREADAFFPOLICY_ALL:
    printf("Setting affinity to all\n");
    CPU_ZERO(cpuset);
    for (j=0; j<ncores; j++) {
      CPU_SET(j,cpuset);
    }
    *set = PETSC_TRUE;
    break;
  case THREADAFFPOLICY_NONE:
    printf("Setting affinity to none\n");
    *set = PETSC_FALSE;
    break;
  }
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolFunc"
/*
   PetscThreadPoolFunc - Spin loop where worker threads can wait for jobs

   Not collective

   Input Paramers:
.  arg - Pointer to a PetscThread

   Level: developer

   Notes:
   Worker threads call this function and wait in a spin loop.
   Once the master thread adds a job to the threads job queue, the worker
   threads will call the kernel function and complete the work.
   Creates a threadcomm stack while worker thread is in loop.
   For OpenMP threads, this routine sets the core affinity of the thread
   at beginning of function.
*/
void* PetscThreadPoolFunc(void *arg)
{
  PetscInt                trank;
  PetscThread             thread;
  PetscThreadPool         pool;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  thread   = *(PetscThread*)arg;
  trank    = thread->prank;
  pool     = thread->pool;
  jobqueue = pool->poolthreads[trank]->jobqueue;
  printf("rank=%d in ThreadPoolFunc\n",trank);

  if (pool->model == THREAD_MODEL_LOOP || pool->model == THREAD_MODEL_AUTO) {
    ierr = PetscThreadInitialize();CHKERRCONTINUE(ierr);
  }

  thread->jobdata = 0;
  thread->status = THREAD_INITIALIZED;

  /* Spin loop */
  while (PetscReadOnce(int,thread->status) != THREAD_TERMINATE) {
    if (jobqueue->completed_jobs_ctr < jobqueue->total_jobs_ctr) {
      job = &jobqueue->jobs[jobqueue->current_job_index];
      pool->poolthreads[trank]->jobdata = job;
      /* Do own job */
      printf("Running job for commrank=%d\n",job->commrank);
      PetscRunKernel(job->commrank,thread->jobdata->nargs,thread->jobdata);
      /* Post job completed status */
      job->job_status = THREAD_JOB_COMPLETED;
      jobqueue->current_job_index = (jobqueue->current_job_index+1)%pool->nkernels;
      jobqueue->completed_jobs_ctr++;
    }
    PetscCPURelax();
  }

  if (pool->model == THREAD_MODEL_LOOP || pool->model == THREAD_MODEL_AUTO) {
    ierr = PetscThreadFinalize();CHKERRCONTINUE(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolDestroy"
/*@
   PetscThreadPoolDestroy - Destroys threadpool and threads in the pool

   Not Collective

   Input Parameters:
.  pool - Threadpool to destroy

   Level: developer

   Notes:
   Reduces the reference count for this pool. Once there are no more references
   to this pool, the pool is destroyed along with all threads in the pool.
@*/
PetscErrorCode PetscThreadPoolDestroy(PetscThreadPool pool)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("In ThreadPoolDestroy refct=%d\n",pool->refct);
  if (!pool) PetscFunctionReturn(0);
  if (!--pool->refct) {
    printf("Destroying ThreadPool\n");
    /* Destroy implementation specific structs */
    if (pool->threadtype != THREAD_TYPE_NOTHREAD) {
      ierr = (*pool->ops->pooldestroy)(pool);CHKERRQ(ierr);
    }
    /* Destroy thread structs in threadpool */
    for (i=0; i<pool->npoolthreads; i++) {
      ierr = PetscFree(pool->poolthreads[i]->jobqueue);CHKERRQ(ierr);
      ierr = PetscFree(pool->poolthreads[i]);CHKERRQ(ierr);
    }
    /* Destroy threadpool */
    ierr = PetscFree(pool->poolthreads);CHKERRQ(ierr);
    ierr = PetscFree(pool->ops);CHKERRQ(ierr);
    ierr = PetscFree(pool);CHKERRQ(ierr);
  }
  pool = PETSC_NULL;
  PetscFunctionReturn(0);
}

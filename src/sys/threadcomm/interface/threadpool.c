#define PETSC_DESIRE_FEATURE_TEST_MACROS
#include <petscthreadcomm.h>
#include <petsc-private/threadcommimpl.h>

static PetscInt   N_CORES                 = -1;
PetscBool         PetscThreadCommRegisterAllModelsCalled = PETSC_FALSE;
PetscBool         PetscThreadCommRegisterAllTypesCalled  = PETSC_FALSE;

extern PetscErrorCode PetscThreadPoolFunc_User(PetscThread thread);

/*
  PetscPThreadCommAffinityPolicy - Core affinity policy for pthreads

$ PTHREADAFFPOLICY_ALL     - threads can run on any core. OS decides thread scheduling
$ PTHREADAFFPOLICY_ONECORE - threads can run on only one core.
$ PTHREADAFFPOLICY_NONE    - No set affinity policy. OS decides thread scheduling
*/
const char *const PetscThreadCommAffPolicyTypes[] = {"ALL","ONECORE","NONE","PetscPThreadCommAffinityPolicyType","PTHREADAFFPOLICY_",0};

PETSC_EXTERN PetscErrorCode PetscThreadDestroy_PThread(PetscThread thread);

#undef __FUNCT__
#define __FUNCT__ "PetscGetNCores"
/*@
  PetscGetNCores - Gets the number of available cores on the system

  Not Collective

  Level: developer

  Notes
  Defaults to 1 if the available core count cannot be found

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
      ierr = sysctlbyname("hw.activecpu",&N_CORES,&len,NULL,0); /* osx preferes activecpu over ncpu */
      if (ierr) { /* freebsd check ncpu */
        sysctlbyname("hw.ncpu",&N_CORES,&len,NULL,0);
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
PetscErrorCode PetscThreadPoolGetPool(MPI_Comm comm,PetscThreadPool *pool)
{
  PetscThreadComm tcomm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCommGetThreadComm(comm,&tcomm);
  *pool = tcomm->pool;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolAlloc"
PetscErrorCode PetscThreadPoolAlloc(PetscThreadPool *pool)
{
  PetscErrorCode ierr;
  PetscThreadPool poolout;

  PetscFunctionBegin;
  *pool = NULL;
  ierr = PetscNew(&poolout);CHKERRQ(ierr);

  poolout->refct          = 0;
  poolout->npoolthreads   = -1;
  poolout->poolthreads    = NULL;

  poolout->model          = THREAD_MODEL_LOOP;
  poolout->threadtype     = THREAD_TYPE_NOTHREAD;
  poolout->aff            = PTHREADAFFPOLICY_ONECORE;
  poolout->nkernels       = 16;
  poolout->thread_start   = -1;
  poolout->ismainworker   = PETSC_TRUE;
  ierr                    = PetscNew(&poolout->ops);CHKERRQ(ierr);

  *pool = poolout;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCreateJobQueue"
PetscErrorCode PetscThreadCreateJobQueue(PetscThread thread,PetscThreadPool pool)
{
  PetscInt i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Allocate queue
  ierr = PetscNew(&thread->jobqueue);

  // Create job contexts
  ierr = PetscMalloc1(pool->nkernels,&thread->jobqueue->jobs);CHKERRQ(ierr);
  for (i=0; i<pool->nkernels; i++) {
    thread->jobqueue->jobs[i].job_status = THREAD_JOB_NONE;
  }

  // Set queue variables
  thread->jobqueue->next_job_index = 0;
  thread->jobqueue->total_jobs_ctr = 0;
  thread->jobqueue->newest_job_index = 0;
  thread->jobqueue->current_job_index = 0;
  thread->jobqueue->completed_jobs_ctr = 0;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolInitialize"
PetscErrorCode PetscThreadPoolInitialize(PetscThreadPool pool, PetscInt nthreads)
{
  PetscInt        i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  printf("Creating thread pool\n");

  // Set threadpool variables
  printf("Setting model\n");
  ierr = PetscThreadPoolSetModel(pool,LOOP);
  printf("Setting type\n");
  ierr = PetscThreadPoolSetType(pool,NOTHREAD);
  printf("Setting nthreads=%d\n",nthreads);
  ierr = PetscThreadPoolSetNThreads(pool,nthreads);

  if(pool->model==THREAD_MODEL_LOOP) {
    pool->ismainworker = PETSC_TRUE;
    pool->thread_start = 1;
  } else if(pool->model==THREAD_MODEL_AUTO) {
    pool->ismainworker = PETSC_FALSE;
    pool->thread_start = 0;
  } else if(pool->model==THREAD_MODEL_USER) {
    pool->ismainworker = PETSC_TRUE;
    pool->thread_start = 1;
  }

  // Get option settings from command line
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Threadcomm options",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-threadcomm_nkernels","number of kernels that can be launched simultaneously","",16,&pool->nkernels,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  // Create thread structs for pool
  ierr = PetscMalloc1(pool->npoolthreads,&pool->poolthreads);CHKERRQ(ierr);
  for(i=0; i<pool->npoolthreads; i++) {
    ierr = PetscNew(&pool->poolthreads[i]);CHKERRQ(ierr);
    pool->poolthreads[i]->grank = i;
    pool->poolthreads[i]->pool = NULL;
    pool->poolthreads[i]->status = 0;

    ierr = PetscThreadCreateJobQueue(pool->poolthreads[i],pool);
    if(pool->threadtype==THREAD_TYPE_PTHREAD) {
      ierr = pool->ops->createthread(pool->poolthreads[i]);
    }
  }

  printf("Setting affinities in threadpool\n");
  ierr = PetscThreadPoolSetAffinities(pool,NULL);CHKERRQ(ierr);

  printf("Initialized pool with %d threads\n",pool->npoolthreads);
  pool->refct++;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolSetType"
/*
   PetscThreadPoolSetType - Sets the threading model for the thread communicator

   Logically collective

   Input Parameters:
+  tcomm - the thread communicator
-  type  - the type of thread model needed


   Options Database keys:
   -threadcomm_type <type>

   Available types
   See "petsc/include/petscthreadcomm.h" for available types
*/
PetscErrorCode PetscThreadPoolSetType(PetscThreadPool pool,PetscThreadCommType type)
{
  PetscBool      flg;
  PetscErrorCode ierr,(*r)(PetscThreadPool);

  PetscFunctionBegin;
  PetscValidCharPointer(type,2);
  if (!PetscThreadCommRegisterAllTypesCalled) { ierr = PetscThreadCommRegisterAllTypes(pool);CHKERRQ(ierr);}

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Threadcomm type - setting threading type",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-threadcomm_type","Threadcomm type","PetscThreadCommSetType",PetscThreadCommTypeList,type,pool->type,256,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  // Find and call threadcomm init function
  if(flg) {
    ierr = PetscFunctionListFind(PetscThreadCommInitTypeList,pool->type,&r);CHKERRQ(ierr);
    if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested threadcomm type %s",pool->type);
    ierr = (*r)(pool);CHKERRQ(ierr);
  } else PetscStrcpy(pool->type,NOTHREAD);

  // Find threadcomm create function
  ierr = PetscFunctionListFind(PetscThreadCommTypeList,pool->type,&pool->ops->tcomminit);CHKERRQ(ierr);
  if (!pool->ops->tcomminit) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested threadcomm type %s",pool->type);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolSetModel"
/*
   PetscThreadPoolSetModel - Sets the threading model for the thread communicator

   Logically collective

   Input Parameters:
+  tcomm - the thread communicator
-  model  - the type of thread model needed


   Options Database keys:
   -threadcomm_model <type>

   Available models
   See "petsc/include/petscthreadcomm.h" for available types
*/
PetscErrorCode PetscThreadPoolSetModel(PetscThreadPool pool,PetscThreadCommModel model)
{
  PetscErrorCode ierr,(*r)(PetscThreadPool);
  char           smodel[256];
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidCharPointer(model,2);
  if (!PetscThreadCommRegisterAllModelsCalled) { ierr = PetscThreadCommRegisterAllModels();CHKERRQ(ierr);}

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Threadcomm model - setting threading model",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-threadcomm_model","Threadcomm model","PetscThreadCommSetModel",PetscThreadCommModelList,model,smodel,256,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (!flg) ierr = PetscStrcpy(smodel,model);CHKERRQ(ierr);
  ierr = PetscFunctionListFind(PetscThreadCommModelList,smodel,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested Threadcomm model %s",smodel);
  ierr = (*r)(pool);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolCreate"
PetscErrorCode PetscThreadPoolCreate(PetscThreadComm tcomm, PetscInt *nthreads)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("Creating ThreadPool\n");
  ierr = PetscThreadPoolAlloc(&tcomm->pool);
  ierr = PetscThreadPoolInitialize(tcomm->pool,*nthreads);

  // Create threads and put in pool
  if(tcomm->pool->threadtype==THREAD_TYPE_PTHREAD && (tcomm->pool->model==THREAD_MODEL_AUTO || tcomm->pool->model==THREAD_MODEL_LOOP)) {
    ierr = (*tcomm->pool->ops->startthreads)(tcomm->pool);
  }
  // Return number of threads in pool
  *nthreads = tcomm->pool->npoolthreads;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolSetNThreads"
/*
   PetscThreadCommSetNThreads - Set the thread count for the thread communicator

   Not collective

   Input Parameters:
+  tcomm - the thread communicator
-  nthreads - Number of threads

   Options Database keys:
   -threadcomm_nthreads <nthreads> Number of threads to use

   Level: developer

   Notes:
   Defaults to using 1 thread.

   Use nthreads = PETSC_DECIDE or -threadcomm_nthreads PETSC_DECIDE for PETSc to decide the number of threads.


.seealso: PetscThreadCommGetNThreads()
*/
PetscErrorCode PetscThreadPoolSetNThreads(PetscThreadPool pool,PetscInt nthreads)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscInt       nthr;

  PetscFunctionBegin;
  // Set number of threads to 1 if not using nothreads
  if(pool->type==THREAD_TYPE_NOTHREAD) {
    pool->npoolthreads = 1;
    PetscFunctionReturn(0);
  }
  // Check input options for number of threads
  if (nthreads == PETSC_DECIDE) {
    pool->npoolthreads = 1;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Thread comm - setting number of threads",NULL);CHKERRQ(ierr);
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

.seealso: PetscThreadCommSetNThreads()
@*/
PetscErrorCode PetscThreadPoolGetNThreads(MPI_Comm comm,PetscInt *nthreads)
{
  PetscErrorCode  ierr;
  PetscThreadPool pool;

  PetscFunctionBegin;
  ierr      = PetscThreadPoolGetPool(comm,&pool);CHKERRQ(ierr);
  *nthreads = pool->npoolthreads;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolSetAffinities"
/*
   PetscThreadPoolSetAffinities - Sets the core affinity for threads
                                  (which threads run on which cores)

   Not collective

   Input Parameters:
+  pool - the threadpool
-  affinities - array of core affinity for threads

   Options Database keys:
.  -threadpool_affinities <list of thread affinities>

   Level: developer

   Notes:
   Use affinities = NULL for PETSc to decide the affinities.
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
  PetscInt       i, *affopt, nmax=pool->npoolthreads;

  PetscFunctionBegin;
  printf("In poolsetaffinities\n");
  /* Do not need to set thread pool affinities if no threads */
  if(pool->threadtype==THREAD_TYPE_NOTHREAD) PetscFunctionReturn(0);

  /* If user did not pass in affinity settings */
  if (!affinities) {

    /* Check if option is present in the options database */
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Thread comm - setting thread affinities",NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-threadcomm_affpolicy","Thread affinity policy"," ",PetscThreadCommAffPolicyTypes,(PetscEnum)pool->aff,(PetscEnum*)&pool->aff,&flg);CHKERRQ(ierr);
    ierr = PetscMalloc1(pool->npoolthreads,&affopt);
    ierr = PetscOptionsIntArray("-threadcomm_affinities","Set core affinities of threads","PetscThreadCommSetAffinities",affopt,&nmax,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    /* If user passes in array from command line, use those affinities */
    if (flg) {
      if (nmax != pool->npoolthreads) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Must set affinities for all threads, Threads = %D, Core affinities set = %D",pool->npoolthreads,nmax);
      for(i=0; i<pool->npoolthreads; i++) pool->poolthreads[i]->affinity = affopt[i];
    } else {
      /* Set affinities based on affinity policy */
      ierr = (*pool->ops->setaffinities)(pool);CHKERRQ(ierr);
    }
    PetscFree(affopt);
  } else {
    /* Use affinities from input parameter */
    for(i=0; i<pool->npoolthreads; i++) pool->poolthreads[i]->affinity = affinities[i];
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolSetAffinity"
PetscErrorCode PetscThreadPoolSetAffinity(PetscThreadPool pool,cpu_set_t *cpuset,PetscInt trank,PetscBool *set)
{
  PetscInt ncores,j;

  PetscFunctionBegin;
  printf("in poolsetaff\n");
  PetscGetNCores(&ncores);
  switch (pool->aff) {
  case PTHREADAFFPOLICY_ONECORE:
    CPU_ZERO(cpuset);
    CPU_SET(pool->poolthreads[trank]->affinity%ncores,cpuset);
    *set = PETSC_TRUE;
    break;
  case PTHREADAFFPOLICY_ALL:
    CPU_ZERO(cpuset);
    for (j=0; j<ncores; j++) {
      CPU_SET(j,cpuset);
    }
    *set = PETSC_TRUE;
    break;
  case PTHREADAFFPOLICY_NONE:
    if(pool->model==THREAD_TYPE_NOTHREAD && trank==0) {
      CPU_ZERO(cpuset);
      CPU_SET(pool->poolthreads[0]->affinity%ncores,cpuset);
      *set = PETSC_TRUE;
    } else {
      *set = PETSC_FALSE;
    }
    break;
  }
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolJoin"
PetscErrorCode PetscThreadPoolJoin(MPI_Comm *comm,PetscInt ncomms,PetscInt trank,PetscInt *prank)
{
  PetscInt i, j;
  PetscThreadComm *tcomm;
  PetscErrorCode ierr;
  PetscInt comm_index = -1, local_index = -1;

  PetscFunctionBegin;

  printf("rank=%d joining thread pool\n",trank);
  PetscMalloc1(ncomms,&tcomm);
  // Determine which threadcomm and local thread this thread belongs to
  for(i=0; i<ncomms; i++) {
    ierr = PetscCommGetThreadComm(comm[i],&tcomm[i]);

    // Check if this thread is in threadcomm
    for(j=0; j<tcomm[i]->ncommthreads; j++) {
      if(tcomm[i]->commthreads[j]->grank==trank) {
        comm_index = i;
        local_index = j;
      }
    }
  }
  printf("trank=%d comm_index=%d local_index=%d leader=%d\n",trank,comm_index,local_index,tcomm[comm_index]->leader);

  // Make sure all threads have reached this routine
  ierr = PetscThreadCommLocalBarrier(tcomm[comm_index]);

  // Initialize thread and join threadpool if a worker thread
  if(trank==tcomm[comm_index]->leader) {
    tcomm[comm_index]->active = PETSC_TRUE;
    *prank = comm_index;
  } else {
    tcomm[comm_index]->commthreads[local_index]->status = THREAD_INITIALIZED;
    tcomm[comm_index]->commthreads[local_index]->jobdata = 0;
    tcomm[comm_index]->commthreads[local_index]->pool = tcomm[comm_index]->pool;
    *prank = -1;
  }
  ierr = (*tcomm[comm_index]->ops->atomicincrement)(tcomm[comm_index],&tcomm[comm_index]->nthreads,1);

  // Make sure all threads have initialized threadcomm
  ierr = PetscThreadCommLocalBarrier(tcomm[comm_index]);

  if(*prank==-1) {
    // Join thread pool if not leader thread
    PetscThreadPoolFunc((void*)&tcomm[comm_index]->commthreads[local_index]);
  }

  PetscFunctionReturn(0);
}

/* Checks whether this thread is a member of tcomm */
PetscBool CheckThreadCommMembership(PetscInt myrank,PetscThreadPool pool)
{
  PetscInt i;

  for (i=0;i<pool->npoolthreads;i++) {
    if (myrank == pool->poolthreads[i]->grank) return PETSC_TRUE;
  }
  return PETSC_FALSE;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolFunc"
void* PetscThreadPoolFunc(void *arg)
{
  PetscInt trank;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx job;
  PetscThread thread;
  PetscThreadPool pool;

  PetscFunctionBegin;
  thread = *(PetscThread*)arg;
  trank = thread->grank;
  pool = thread->pool;
  jobqueue = pool->poolthreads[trank]->jobqueue;
  printf("rank=%d in ThreadPoolFunc\n",trank);

  //Create thread stack
  PetscThreadCommStackCreate(trank);

  thread->jobdata = 0;
  thread->status = THREAD_INITIALIZED;

  /* Spin loop */
  while (PetscReadOnce(int,thread->status) != THREAD_TERMINATE) {
    if (jobqueue->completed_jobs_ctr < jobqueue->total_jobs_ctr) {
      job = &jobqueue->jobs[jobqueue->current_job_index];
      /* Check that this thread is in threadcomm */
      if (CheckThreadCommMembership(trank,pool)) {
        pool->poolthreads[trank]->jobdata = job;
        /* Do own job */
        PetscRunKernel(job->commrank,thread->jobdata->nargs,thread->jobdata);
        /* Post job completed status */
        job->job_status = THREAD_JOB_COMPLETED;
      }
      jobqueue->current_job_index = (jobqueue->current_job_index+1)%pool->nkernels;
      jobqueue->completed_jobs_ctr++;
    }
    PetscCPURelax();
  }
  // Destroy thread stack
  PetscThreadCommStackDestroy(trank);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolReturn"
PetscErrorCode PetscThreadPoolReturn(MPI_Comm *comm,PetscInt ncomms,PetscInt trank,PetscInt *prank)
{
  PetscThreadComm *tcomm;
  PetscInt i, j, comm_index=-1;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  PetscMalloc1(ncomms,&tcomm);
  // Determine which threadcomm and local thread this thread belongs to
  for(i=0; i<ncomms; i++) {
    ierr = PetscCommGetThreadComm(comm[i],&tcomm[i]);

    // Check if this thread is in threadcomm
    for(j=0; j<tcomm[i]->ncommthreads; j++) {
      if(tcomm[i]->commthreads[j]->grank==trank) comm_index = i;
    }
  }

  // Master threads terminate worker threads
  if(*prank>=0) {
    printf("Returning all threads\n");
    ierr = PetscThreadCommJobBarrier(tcomm[comm_index]);
    for(i=0; i<tcomm[comm_index]->ncommthreads; i++) {
      tcomm[comm_index]->commthreads[i]->status = THREAD_TERMINATE;
    }
  }

  // Make sure all threads have reached this point
  ierr = (*tcomm[comm_index]->ops->atomicincrement)(tcomm[comm_index],&tcomm[comm_index]->nthreads,-1);
  ierr = PetscThreadCommLocalBarrier(tcomm[comm_index]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolBarrier"
PetscErrorCode PetscThreadPoolBarrier(PetscThreadPool pool)
{
  PetscInt                active_threads=0,i;
  PetscBool               wait          =PETSC_TRUE;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;
  PetscInt                job_status;

  PetscFunctionBegin;
  if (pool->npoolthreads == 1 && pool->ismainworker) PetscFunctionReturn(0);

  /* Loop till all threads signal that they have done their job */
  while (wait) {
    for (i=0; i<pool->npoolthreads; i++) {
      jobqueue = pool->poolthreads[i]->jobqueue;
      job = &jobqueue->jobs[jobqueue->newest_job_index];
      job_status      = job->job_status;
      active_threads += job_status;
    }
    if (PetscReadOnce(int,active_threads) > 0) active_threads = 0;
    else wait=PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolDestroy"
PetscErrorCode PetscThreadPoolDestroy(PetscThreadPool pool)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("In ThreadPoolDestroy refct=%d\n",pool->refct);
  if(!pool) PetscFunctionReturn(0);
  if(!--pool->refct) {
    printf("Destroying ThreadPool\n");
    /* Destroy pthreads structs and join pthreads */
    if(pool->threadtype==THREAD_TYPE_PTHREAD) {
      ierr = (*pool->ops->pooldestroy)(pool);
    }
    /* Destroy thread structs in threadpool */
    for(i=0; i<pool->npoolthreads; i++) {
      ierr = PetscFree(pool->poolthreads[i]->jobqueue);CHKERRQ(ierr);
      ierr = PetscFree(pool->poolthreads[i]);CHKERRQ(ierr);
    }
    /* Destroy threadpool */
    ierr = PetscFree(pool->poolthreads);CHKERRQ(ierr);
    ierr = PetscFree(pool->ops);CHKERRQ(ierr);
    ierr = PetscFree(pool);CHKERRQ(ierr);
  }
  pool = NULL;
  PetscFunctionReturn(0);
}

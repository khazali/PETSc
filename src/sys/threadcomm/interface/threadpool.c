#define PETSC_DESIRE_FEATURE_TEST_MACROS
#include <petscthreadcomm.h>
#include <petsc-private/threadcommimpl.h>

static PetscInt N_CORES                 = -1;
PetscMPIInt     Petsc_ThreadPool_keyval = MPI_KEYVAL_INVALID;
PetscThreadPool PETSC_THREAD_POOL       = NULL;

const char *const PetscThreadPoolSparkTypes[] = {"SELF","PetscThreadPoolSparkType","PTHREADPOOLSPARK_",0};

/*
  PetscPThreadCommAffinityPolicy - Core affinity policy for pthreads

$ PTHREADAFFPOLICY_ALL     - threads can run on any core. OS decides thread scheduling
$ PTHREADAFFPOLICY_ONECORE - threads can run on only one core.
$ PTHREADAFFPOLICY_NONE    - No set affinity policy. OS decides thread scheduling
*/
const char *const PetscPThreadCommAffinityPolicyTypes[] = {"ALL","ONECORE","NONE","PetscPThreadCommAffinityPolicyType","PTHREADAFFPOLICY_",0};

PETSC_EXTERN PetscErrorCode PetscThreadPoolFunc_User(PetscThreadInfo tinfo);

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
  PetscErrorCode ierr;
  PetscMPIInt    flg;
  void           *ptr;

  PetscFunctionBegin;
  ierr = MPI_Attr_get(comm,Petsc_ThreadPool_keyval,(PetscThreadPool*)&ptr,&flg);CHKERRQ(ierr);
  printf("In getpool flg=%d\n",flg);
  if (!flg) {
    if (!PETSC_THREAD_POOL) {
      ierr = PetscThreadPoolInitialize();CHKERRQ(ierr);
    }
    *pool = PETSC_THREAD_POOL;
  } else *pool = (PetscThreadPool)ptr;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolCreate"
PetscErrorCode PetscThreadPoolCreate(PetscThreadPool *pool)
{
  PetscErrorCode ierr;
  PetscThreadPool poolout;

  PetscFunctionBegin;
  *pool = NULL;
  ierr = PetscNew(&poolout);CHKERRQ(ierr);

  poolout->refct = 0;
  ierr = PetscNew(&poolout->ops);CHKERRQ(ierr);

  poolout->model = 0;
  poolout->spark = PTHREADPOOLSPARK_SELF;
  poolout->aff = PTHREADAFFPOLICY_ONECORE;
  poolout->synchronizeafter = PETSC_TRUE;
  poolout->ismainworker = PETSC_TRUE;
  poolout->nkernels = 16;

  poolout->npoolthreads = -1;
  poolout->granks = NULL;
  poolout->affinities = NULL;

  *pool = poolout;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolInitialize"
PetscErrorCode PetscThreadPoolInitialize(void)
{
  PetscInt i;
  PetscBool flg;
  PetscErrorCode ierr;
  PetscThreadPool pool;

  PetscFunctionBegin;
  printf("Creating thread pool\n");
  ierr = PetscThreadPoolCreate(&PETSC_THREAD_POOL);
  pool = PETSC_THREAD_POOL;

  // Set threadpool variables
  ierr = PetscThreadPoolSetNThreads(pool,PETSC_DECIDE);
  ierr = PetscMalloc1(pool->npoolthreads,&pool->affinities);CHKERRQ(ierr);
  ierr = PetscThreadPoolSetAffinities(pool,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(pool->npoolthreads,&pool->granks);CHKERRQ(ierr);

  // Get option settings from command line
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Thread pool options",NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-threadcomm_nkernels","number of kernels that can be launched simultaneously","",16,&pool->nkernels,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-threadcomm_pool_main_is_worker","Main thread is also a worker thread",NULL,PETSC_TRUE,&pool->ismainworker,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-threadcomm_pool_affpolicy","Thread affinity policy"," ",PetscPThreadCommAffinityPolicyTypes,(PetscEnum)pool->aff,(PetscEnum*)&pool->aff,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-threadcomm_pool_spark","Thread pool spark type"," ",PetscThreadPoolSparkTypes,(PetscEnum)pool->spark,(PetscEnum*)&pool->spark,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-threadcomm_pool_synchronizeafter","Puts a barrier after every kernel call",NULL,PETSC_TRUE,&pool->synchronizeafter,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Set up thread ranks */
  for (i=0; i<pool->npoolthreads; i++) pool->granks[i] = i;

  printf("Initialized pool with %d threads\n",pool->npoolthreads);
  pool->refct++;
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
  if (nthreads == PETSC_DECIDE) {
    pool->npoolthreads = 1;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Thread pool - setting number of threads",NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-threadpool_nthreads","number of threads to use in the thread communicator","PetscThreadPoolSetNThreads",1,&nthr,&flg);CHKERRQ(ierr);
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
  PetscInt       nmax=pool->npoolthreads;

  PetscFunctionBegin;
  if (!affinities) {
    /* Check if option is present in the options database */
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Thread comm - setting thread affinities",NULL);CHKERRQ(ierr);
    ierr = PetscOptionsIntArray("-threadcomm_affinities","Set core affinities of threads","PetscThreadCommSetAffinities",pool->affinities,&nmax,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (flg) {
      if (nmax != pool->npoolthreads) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Must set affinities for all threads, Threads = %D, Core affinities set = %D",pool->npoolthreads,nmax);
    } else {
      /* PETSc default affinities */
      PetscInt i;
      for (i=0; i<pool->npoolthreads; i++) pool->affinities[i] = i%N_CORES;
    }
  } else {
    ierr = PetscMemcpy(pool->affinities,affinities,pool->npoolthreads*sizeof(PetscInt));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolDetach"
PetscErrorCode PetscThreadPoolDetach(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscMPIInt    flg;
  void           *ptr;

  PetscFunctionBegin;
  printf("Detaching thread pool\n");
  ierr = MPI_Attr_get(comm,Petsc_ThreadPool_keyval,&ptr,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MPI_Attr_delete(comm,Petsc_ThreadPool_keyval);CHKERRQ(ierr);
    //pool->refct--;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolAttach"
PetscErrorCode PetscThreadPoolAttach(MPI_Comm comm,PetscThreadPool pool)
{
  PetscErrorCode ierr;
  PetscMPIInt    flg;
  void           *ptr;

  PetscFunctionBegin;
  ierr = MPI_Attr_get(comm,Petsc_ThreadPool_keyval,&ptr,&flg);CHKERRQ(ierr);
  if (!flg) {
    pool->refct++;
    printf("attaching pool\n");
    ierr = MPI_Attr_put(comm,Petsc_ThreadPool_keyval,pool);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolJoin"
PetscErrorCode PetscThreadPoolJoin(MPI_Comm comm,PetscInt trank,PetscInt *prank)
{
  PetscThreadComm tcomm;
  PetscThreadPool pool;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("rank=%d joined thread pool\n",trank);
  ierr = PetscCommGetThreadComm(comm,&tcomm);
  ierr = PetscThreadPoolGetPool(comm,&pool);

  ierr = (*pool->ops->atomicincrement)(tcomm,&tcomm->nthreads,1);

  printf("adding thread nthreads=%d\n",tcomm->nthreads);
  ierr = (*pool->ops->globalbarrier)(tcomm);
  if(trank==0) {
    *prank = 0;
  } else {
    tcomm->jobqueue->tinfo[trank]->status = THREAD_INITIALIZED;
    tcomm->jobqueue->tinfo[trank]->rank = pool->granks[trank];
    tcomm->jobqueue->tinfo[trank]->data = 0;
    tcomm->jobqueue->tinfo[trank]->tcomm = tcomm;
    *prank = -1;
  }
  ierr = (*pool->ops->globalbarrier)(tcomm);

  if(trank>0) {
    PetscThreadInfo tinfo = tcomm->jobqueue->tinfo[trank];
    ierr = PetscThreadPoolFunc_User(tinfo);
  }
  PetscFunctionReturn(0);
}

/* Checks whether this thread is a member of tcomm */
PetscBool CheckThreadCommMembership(PetscInt myrank,PetscThreadComm tcomm)
{
  PetscInt i;
  PetscThreadPool pool = PETSC_THREAD_POOL;

  for (i=0;i<tcomm->ncommthreads;i++) {
    if (myrank == pool->granks[i]) return PETSC_TRUE;
  }
  return PETSC_FALSE;
}

void SparkThreads(PetscInt myrank,PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscThreadPool pool = PETSC_THREAD_POOL;

  switch (pool->spark) {
  case PTHREADPOOLSPARK_SELF:
    if (CheckThreadCommMembership(myrank,tcomm)) {
      tcomm->jobqueue->tinfo[myrank]->data = job;
      job->job_status[myrank] = THREAD_JOB_RECIEVED;
    }
    break;
  }
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolFunc"
void* PetscThreadPoolFunc(void *arg)
{
  PetscInt trank,my_job_counter = 0,my_kernel_ctr=0,glob_kernel_ctr;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx job;
  PetscThreadComm tcomm;
  PetscThreadInfo tinfo;
  PetscThreadPool pool = PETSC_THREAD_POOL;

  PetscFunctionBegin;
  tinfo = *(PetscThreadInfo*)arg;
  trank = tinfo->rank;
  tcomm = tinfo->tcomm;
  jobqueue = tcomm->jobqueue;
  printf("rank=%d in ThreadPoolFunc_Loop\n",trank);

  tinfo->data = 0;
  tinfo->status = THREAD_INITIALIZED;

  /* Spin loop */
  while (PetscReadOnce(int,tinfo->status) != THREAD_TERMINATE) {
    glob_kernel_ctr = PetscReadOnce(int,jobqueue->kernel_ctr);
    if (my_kernel_ctr < glob_kernel_ctr) {
      job = &jobqueue->jobs[my_job_counter];
      /* Spark the thread pool */
      SparkThreads(trank,tcomm,job);
      if (job->job_status[trank] == THREAD_JOB_RECIEVED) {
        /* Do own job */
        PetscRunKernel(trank,tinfo->data->nargs,tinfo->data);
        /* Post job completed status */
        job->job_status[trank] = THREAD_JOB_COMPLETED;
      }
      my_job_counter = (my_job_counter+1)%pool->nkernels;
      my_kernel_ctr++;
    }
    PetscCPURelax();
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolFunc_User"
PetscErrorCode PetscThreadPoolFunc_User(PetscThreadInfo tinfo)
{
  PetscInt trank;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx job;
  PetscThreadComm tcomm;
  PetscThreadPool pool = PETSC_THREAD_POOL;

  PetscFunctionBegin;
  trank = tinfo->rank;
  tcomm = tinfo->tcomm;
  jobqueue = tcomm->jobqueue;
  printf("rank=%d in ThreadPoolFunc_User\n",trank);

  /* Spin loop */
  while (PetscReadOnce(int,tinfo->status) != THREAD_TERMINATE) {
    tcomm->glob_kernel_ctr[trank] = PetscReadOnce(int,jobqueue->kernel_ctr);
    if (tcomm->my_kernel_ctr[trank] < tcomm->glob_kernel_ctr[trank]) {
      job = &jobqueue->jobs[tcomm->my_job_counter[trank]];
      /* Spark the thread pool */
      SparkThreads(trank,tcomm,job);
      if (job->job_status[trank] == THREAD_JOB_RECIEVED) {
        /* Do own job */
        PetscRunKernel(trank,tinfo->data->nargs,tinfo->data);
        /* Post job completed status */
        job->job_status[trank] = THREAD_JOB_COMPLETED;
      }
      tcomm->my_job_counter[trank] = (tcomm->my_job_counter[trank]+1)%pool->nkernels;
      tcomm->my_kernel_ctr[trank]++;
    }
    PetscCPURelax();
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolReturn"
PetscErrorCode PetscThreadPoolReturn(MPI_Comm comm,PetscInt *prank)
{
  PetscThreadComm tcomm;
  PetscThreadPool pool;
  PetscInt i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if(*prank>=0) {
    printf("Returning all threads\n");
    ierr = PetscCommGetThreadComm(comm,&tcomm);
    ierr = PetscThreadPoolGetPool(comm,&pool);
    for(i=tcomm->thread_start; i<tcomm->ncommthreads; i++) {
      printf("terminate thread %d\n",i);
      tcomm->jobqueue->tinfo[i]->status = THREAD_TERMINATE;
    }
  }
  ierr = (*pool->ops->globalbarrier)(tcomm);
  ierr = (*pool->ops->atomicincrement)(tcomm,&tcomm->nthreads,-1);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolBarrier"
PetscErrorCode PetscThreadPoolBarrier(PetscThreadComm tcomm)
{
  PetscInt                active_threads=0,i;
  PetscBool               wait          =PETSC_TRUE;
  PetscThreadPool pool = PETSC_THREAD_POOL;
  PetscThreadCommJobQueue jobqueue      =tcomm->jobqueue;
  PetscThreadCommJobCtx   job           =&jobqueue->jobs[tcomm->job_ctr];
  PetscInt                job_status;

  PetscFunctionBegin;
  printf("In PetscThreadPoolBarrier job_ctr=%d\n",tcomm->job_ctr);
  if (tcomm->ncommthreads == 1 && pool->ismainworker) PetscFunctionReturn(0);

  /* Loop till all threads signal that they have done their job */
  while (wait) {
    for (i=0; i<tcomm->ncommthreads; i++) {
      job_status      = job->job_status[pool->granks[i]];
      active_threads += job_status;
    }
    if (PetscReadOnce(int,active_threads) > 0) active_threads = 0;
    else wait=PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolDestroy"
PetscErrorCode PetscThreadPoolDestroy(PetscThreadPool *pool)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("Destroying thread pool refct=%d\n",(*pool)->refct);
  if(!(*pool)) PetscFunctionReturn(0);
  if(!--(*pool)->refct) {
    ierr = PetscFree((*pool)->granks);CHKERRQ(ierr);
    ierr = PetscFree((*pool)->affinities);CHKERRQ(ierr);
    ierr = PetscFree((*pool)->ops);CHKERRQ(ierr);
    ierr = PetscFree(*pool);CHKERRQ(ierr);
  }
  pool = NULL;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolSetAffinity"
PetscErrorCode PetscThreadPoolSetAffinity(PetscThreadPool pool,cpu_set_t *cpuset,PetscInt trank,PetscBool *set)
{
  PetscInt ncores,j;

  PetscFunctionBegin;
  PetscGetNCores(&ncores);
  switch (pool->aff) {
  case PTHREADAFFPOLICY_ONECORE:
    CPU_ZERO(cpuset);
    CPU_SET(pool->affinities[trank]%ncores,cpuset);
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
    if(pool->ismainworker && trank==0) {
      CPU_ZERO(cpuset);
      CPU_SET(pool->affinities[0]%ncores,cpuset);
      *set = PETSC_TRUE;
    } else {
      *set = PETSC_FALSE;
    }
    break;
  }
  PetscFunctionReturn(0);
}
#endif

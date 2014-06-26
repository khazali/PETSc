#define PETSC_DESIRE_FEATURE_TEST_MACROS
#include <petscthreadcomm.h>
#include <petsc-private/threadcommimpl.h>

PetscBool         PetscThreadPoolRegisterAllModelsCalled = PETSC_FALSE;
PetscBool         PetscThreadPoolRegisterAllTypesCalled  = PETSC_FALSE;
PetscFunctionList PetscThreadPoolTypeList                    = NULL;
PetscFunctionList PetscThreadPoolModelList                   = NULL;
static PetscInt   N_CORES                 = -1;

const char *const PetscThreadPoolSparkTypes[] = {"SELF","PetscThreadPoolSparkType","PTHREADPOOLSPARK_",0};

extern PetscErrorCode PetscThreadPoolFunc_User(PetscThread thread);

/*
  PetscPThreadCommAffinityPolicy - Core affinity policy for pthreads

$ PTHREADAFFPOLICY_ALL     - threads can run on any core. OS decides thread scheduling
$ PTHREADAFFPOLICY_ONECORE - threads can run on only one core.
$ PTHREADAFFPOLICY_NONE    - No set affinity policy. OS decides thread scheduling
*/
const char *const PetscPThreadCommAffinityPolicyTypes[] = {"ALL","ONECORE","NONE","PetscPThreadCommAffinityPolicyType","PTHREADAFFPOLICY_",0};

//PETSC_EXTERN PetscErrorCode PetscThreadPoolFunc_User(PetscThread thread);

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
PetscErrorCode PetscThreadPoolSetModel(PetscThreadPool pool,PetscThreadPoolModel model)
{
  PetscErrorCode ierr,(*r)(PetscThreadPool);
  char           smodel[256];
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidCharPointer(model,2);
  if (!PetscThreadPoolRegisterAllModelsCalled) { ierr = PetscThreadPoolRegisterAllModels();CHKERRQ(ierr);}

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Threadpool model - setting threading model",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-threadpool_model","Threadpool model","PetscThreadPoolSetModel",PetscThreadPoolModelList,model,smodel,256,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (!flg) ierr = PetscStrcpy(smodel,model);CHKERRQ(ierr);
  ierr = PetscFunctionListFind(PetscThreadPoolModelList,smodel,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested PetscThreadPool model %s",smodel);
  ierr = (*r)(pool);CHKERRQ(ierr);
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
PetscErrorCode PetscThreadPoolSetType(PetscThreadPool pool,PetscThreadPoolType type)
{
  PetscErrorCode ierr,(*r)(PetscThreadPool);
  char           ttype[256];
  PetscBool      tflg;

  PetscFunctionBegin;
  PetscValidCharPointer(type,2);
  if (!PetscThreadPoolRegisterAllTypesCalled) { ierr = PetscThreadPoolRegisterAllTypes(pool);CHKERRQ(ierr);}

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Threadpool type - setting threading type",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-threadpool_type","Thread pool type","PetscThreadPoolSetType",PetscThreadPoolTypeList,type,ttype,256,&tflg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (!tflg) {
    ierr = PetscStrcpy(ttype,type);CHKERRQ(ierr);
  }
  ierr = PetscFunctionListFind(PetscThreadPoolTypeList,ttype,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested PetscThreadPool type %s",ttype);
  ierr = (*r)(pool);CHKERRQ(ierr);
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
  ierr                    = PetscNew(&poolout->ops);CHKERRQ(ierr);

  poolout->model            = 0;
  poolout->spark            = PTHREADPOOLSPARK_SELF;
  poolout->aff              = PTHREADAFFPOLICY_ONECORE;
  poolout->synchronizeafter = PETSC_TRUE;
  poolout->ismainworker     = PETSC_TRUE;
  poolout->nkernels         = 16;
  poolout->createdthreads   = PETSC_FALSE;

  poolout->npoolthreads   = -1;
  poolout->affinities     = NULL;
  poolout->data           = NULL;
  poolout->poolthreads    = NULL;

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
  thread->jobqueue->ctr = 0;
  thread->jobqueue->kernel_ctr = 0;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolInitialize"
PetscErrorCode PetscThreadPoolInitialize(PetscThreadPool pool, PetscInt nthreads)
{
  PetscInt        i;
  PetscBool       flg;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  printf("Creating thread pool\n");

  // Set threadpool variables
  ierr = PetscThreadPoolSetNThreads(pool,nthreads);
  ierr = PetscMalloc1(pool->npoolthreads,&pool->affinities);CHKERRQ(ierr);
  ierr = PetscThreadPoolSetAffinities(pool,NULL);CHKERRQ(ierr);

  // Create thread structs for pool
  ierr = PetscMalloc1(pool->npoolthreads,&pool->poolthreads);CHKERRQ(ierr);
  for(i=0; i<pool->npoolthreads; i++) {
    ierr = PetscNew(&pool->poolthreads[i]);CHKERRQ(ierr);
    pool->poolthreads[i]->grank = i;
    pool->poolthreads[i]->tcomm = NULL;
    pool->poolthreads[i]->status = 0;

    ierr = PetscThreadCreateJobQueue(pool->poolthreads[i],pool);
    pool->poolthreads[i]->job_ctr = 0;
    pool->poolthreads[i]->my_job_counter = 0;
    pool->poolthreads[i]->my_kernel_ctr = 0;
    pool->poolthreads[i]->glob_kernel_ctr = 0;
  }

  // Get option settings from command line
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Thread pool options",NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-threadcomm_nkernels","number of kernels that can be launched simultaneously","",16,&pool->nkernels,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-threadcomm_pool_main_is_worker","Main thread is also a worker thread",NULL,PETSC_TRUE,&pool->ismainworker,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-threadcomm_pool_affpolicy","Thread affinity policy"," ",PetscPThreadCommAffinityPolicyTypes,(PetscEnum)pool->aff,(PetscEnum*)&pool->aff,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-threadcomm_pool_spark","Thread pool spark type"," ",PetscThreadPoolSparkTypes,(PetscEnum)pool->spark,(PetscEnum*)&pool->spark,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-threadcomm_pool_synchronizeafter","Puts a barrier after every kernel call",NULL,PETSC_TRUE,&pool->synchronizeafter,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Set up thread ranks */
  //for (i=0; i<pool->npoolthreads; i++) pool->granks[i] = i;

  ierr = PetscThreadPoolSetModel(pool,LOOP);CHKERRQ(ierr);
  ierr = PetscThreadPoolSetType(pool,NOTHREAD);CHKERRQ(ierr);

  printf("Initialized pool with %d threads\n",pool->npoolthreads);
  pool->refct++;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolCreate"
PetscErrorCode PetscThreadPoolCreate(PetscThreadComm tcomm, PetscInt nthreads, PetscBool createthreads)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscThreadPoolAlloc(&tcomm->pool);
  ierr = PetscThreadPoolInitialize(tcomm->pool,nthreads);

  // Create threads and put in pool
  tcomm->pool->createdthreads = createthreads;
  if(tcomm->pool->createdthreads) {
    ierr = (*tcomm->pool->ops->createthreads)(tcomm);
  }
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
#define __FUNCT__ "PetscThreadPoolCreateModel_Loop"
PetscErrorCode PetscThreadPoolCreateModel_Loop(PetscThreadPool pool)
{
  PetscFunctionBegin;
  printf("Creating Loop Model\n");
  pool->model = THREAD_MODEL_LOOP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolCreateModel_Auto"
PetscErrorCode PetscThreadPoolCreateModel_Auto(PetscThreadPool pool)
{
  PetscFunctionBegin;
  printf("Creating Auto Model\n");
  pool->model = THREAD_MODEL_AUTO;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolCreateModel_User"
PetscErrorCode PetscThreadPoolCreateModel_User(PetscThreadPool pool)
{
  PetscFunctionBegin;
  printf("Creating User Model\n");
  pool->model = THREAD_MODEL_USER;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolTypeRegister"
/*@C
  PetscThreadPoolTypeRegister -

  Level: advanced
@*/
PetscErrorCode  PetscThreadPoolTypeRegister(const char sname[],PetscErrorCode (*function)(PetscThreadPool))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscThreadPoolTypeList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolModelRegister"
/*@C
  PetscThreadPoolModelRegister -

  Level: advanced
@*/
PetscErrorCode  PetscThreadPoolModelRegister(const char sname[],PetscErrorCode (*function)(PetscThreadPool))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscThreadPoolModelList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Reusable barrier that can block threads in one threadcomm while threads
 in other threadcomms continue executing. May need to make implementation specific
 so that it can use critical sections */
#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommLocalBarrier"
PetscErrorCode PetscThreadCommLocalBarrier(PetscThreadComm tcomm)
{
  PetscThreadPool pool = tcomm->pool;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  // Make sure all threads increment counter
  ierr = (*pool->ops->atomicincrement)(tcomm,&tcomm->barrier_threads,1);
  while(tcomm->wait1) {
    if(PetscReadOnce(int,tcomm->barrier_threads) == tcomm->ncommthreads) {
      tcomm->wait2 = PETSC_TRUE;
      tcomm->wait1 = PETSC_FALSE;
    }
  }

  // Make sure all threads exited previous while loop so we can reset wait1
  ierr = (*pool->ops->atomicincrement)(tcomm,&tcomm->barrier_threads,-1);
  while(tcomm->wait2) {
    if(PetscReadOnce(int,tcomm->barrier_threads) == 0) {
      tcomm->wait1 = PETSC_TRUE;
      tcomm->wait2 = PETSC_FALSE;
    }
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

  printf("adding thread nthreads=%d\n",tcomm->nthreads);
  ierr = PetscThreadCommLocalBarrier(tcomm);
  if(trank==tcomm->leader) {
    tcomm->active = PETSC_TRUE;
    *prank = 0;
  } else {
    pool->poolthreads[trank]->status = THREAD_INITIALIZED;
    pool->poolthreads[trank]->jobdata = 0;
    pool->poolthreads[trank]->tcomm = tcomm;
    *prank = -1;
  }
  ierr = (*pool->ops->atomicincrement)(tcomm,&tcomm->nthreads,1);
  ierr = PetscThreadCommLocalBarrier(tcomm);

  if(trank!=tcomm->leader) {
    ierr = PetscThreadPoolFunc_User(pool->poolthreads[trank]);
  }
  PetscFunctionReturn(0);
}

/* Checks whether this thread is a member of tcomm */
PetscBool CheckThreadCommMembership(PetscInt myrank,PetscThreadComm tcomm)
{
  PetscInt i;
  PetscThreadPool pool = tcomm->pool;

  for (i=0;i<tcomm->ncommthreads;i++) {
    if (myrank == pool->poolthreads[i]->grank) return PETSC_TRUE;
  }
  return PETSC_FALSE;
}

void SparkThreads(PetscInt myrank,PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscThreadPool pool = tcomm->pool;

  switch (pool->spark) {
  case PTHREADPOOLSPARK_SELF:
    if (CheckThreadCommMembership(myrank,tcomm)) {
      tcomm->commthreads[myrank]->jobdata = job;
      job->job_status = THREAD_JOB_RECIEVED;
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
  PetscThread thread;
  PetscThreadPool pool;

  PetscFunctionBegin;
  thread = *(PetscThread*)arg;
  trank = thread->grank;
  tcomm = thread->tcomm;
  pool = tcomm->pool;
  jobqueue = pool->poolthreads[trank]->jobqueue;
  printf("rank=%d in ThreadPoolFunc_Loop\n",trank);

  thread->jobdata = 0;
  thread->status = THREAD_INITIALIZED;

  /* Spin loop */
  while (PetscReadOnce(int,thread->status) != THREAD_TERMINATE) {
    glob_kernel_ctr = PetscReadOnce(int,jobqueue->kernel_ctr);
    if (my_kernel_ctr < glob_kernel_ctr) {
      job = &jobqueue->jobs[my_job_counter];
      /* Spark the thread pool */
      SparkThreads(trank,tcomm,job);
      if (job->job_status == THREAD_JOB_RECIEVED) {
        /* Do own job */
        PetscRunKernel(job->commrank,thread->jobdata->nargs,thread->jobdata);
        /* Post job completed status */
        job->job_status = THREAD_JOB_COMPLETED;
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
PetscErrorCode PetscThreadPoolFunc_User(PetscThread thread)
{
  PetscInt trank;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx job;
  PetscThreadComm tcomm;
  PetscThreadPool pool;

  PetscFunctionBegin;
  trank = thread->grank;
  tcomm = thread->tcomm;
  pool = tcomm->pool;
  jobqueue = pool->poolthreads[trank]->jobqueue;
  printf("rank=%d in ThreadPoolFunc_User\n",trank);

  /* Spin loop */
  while (PetscReadOnce(int,thread->status) != THREAD_TERMINATE) {
    thread->glob_kernel_ctr = PetscReadOnce(int,jobqueue->kernel_ctr);
    if (thread->my_kernel_ctr < thread->glob_kernel_ctr) {
      job = &jobqueue->jobs[thread->my_job_counter];
      /* Spark the thread pool */
      SparkThreads(trank,tcomm,job);
      if (job->job_status == THREAD_JOB_RECIEVED) {
        /* Do own job */
        PetscRunKernel(job->commrank,thread->jobdata->nargs,thread->jobdata);
        /* Post job completed status */
        job->job_status = THREAD_JOB_COMPLETED;
      }
      thread->my_job_counter = (thread->my_job_counter+1)%pool->nkernels;
      thread->my_kernel_ctr++;
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
  ierr = PetscCommGetThreadComm(comm,&tcomm);
  ierr = PetscThreadPoolGetPool(comm,&pool);
  if(*prank>=0) {
    printf("Returning all threads\n");
    for(i=tcomm->thread_start; i<tcomm->ncommthreads; i++) {
      printf("terminate thread %d\n",i);
      tcomm->commthreads[i]->status = THREAD_TERMINATE;
    }
  }
  ierr = (*pool->ops->atomicincrement)(tcomm,&tcomm->nthreads,-1);
  ierr = PetscThreadCommLocalBarrier(tcomm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolBarrier"
PetscErrorCode PetscThreadPoolBarrier(PetscThreadComm tcomm)
{
  PetscInt                active_threads=0,i;
  PetscBool               wait          =PETSC_TRUE;
  PetscThreadPool         pool          =tcomm->pool;
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   job;
  PetscInt                job_status;

  PetscFunctionBegin;
  printf("In PetscThreadPoolBarrier job_ctr=%d\n",tcomm->commthreads[0]->job_ctr);
  if (tcomm->ncommthreads == 1 && pool->ismainworker) PetscFunctionReturn(0);

  /* Loop till all threads signal that they have done their job */
  while (wait) {
    for (i=0; i<tcomm->ncommthreads; i++) {
      jobqueue = tcomm->commthreads[i]->jobqueue;
      job = &jobqueue->jobs[tcomm->commthreads[i]->job_ctr];
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
PetscErrorCode PetscThreadPoolDestroy(PetscThreadComm tcomm)
{
  PetscThreadPool pool = tcomm->pool;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("In ThreadPoolDestroy refct=%d\n",pool->refct);
  if(!pool) PetscFunctionReturn(0);
  if(!--pool->refct) {
    printf("Destroying ThreadPool\n");
    if(pool->createdthreads) {
      ierr = (*pool->ops->destroythreads)(tcomm);
    }
    ierr = PetscFree(pool->affinities);CHKERRQ(ierr);
    ierr = PetscFree(pool->ops);CHKERRQ(ierr);
    for(i=0; i<pool->npoolthreads; i++) {
      ierr = PetscFree(pool->poolthreads[i]->jobqueue);CHKERRQ(ierr);
      ierr = PetscFree(pool->poolthreads[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(pool->poolthreads);CHKERRQ(ierr);
    ierr = PetscFree(pool);CHKERRQ(ierr);
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

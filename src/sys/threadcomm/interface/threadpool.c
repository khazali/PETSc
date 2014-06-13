#define PETSC_DESIRE_FEATURE_TEST_MACROS
#include <petscthreadcomm.h>
#include <petsc-private/threadcommimpl.h>

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
#define __FUNCT__ "PetscCommGetPool"
PetscErrorCode PetscCommGetPool(MPI_Comm comm,PetscThreadPool *pool)
{
  PetscThreadComm tcomm=NULL;

  PetscFunctionBegin;
  PetscCommGetThreadComm(comm,&tcomm);
  *pool = tcomm->pool;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolCreate"
PetscErrorCode PetscThreadPoolCreate(PetscThreadComm tcomm)
{
  PetscInt i;
  PetscBool flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("Creating thread pool\n");
  // Allocate pool
  ierr = PetscNew(&tcomm->pool);

  // Set ThreadPool variables
  tcomm->pool->nthreads = 0;
  tcomm->pool->maxthreads = tcomm->nworkThreads;
  ierr = PetscMalloc1(tcomm->pool->maxthreads,&tcomm->pool->granks);

  // Initialize options
  tcomm->pool->aff = PTHREADAFFPOLICY_ONECORE;
  tcomm->pool->spark = PTHREADPOOLSPARK_SELF;
  tcomm->ismainworker = PETSC_TRUE;
  tcomm->pool->synchronizeafter = PETSC_TRUE;

  // Get option settings from command line
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Thread pool options",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-threadcomm_pool_main_is_worker","Main thread is also a worker thread",NULL,PETSC_TRUE,&tcomm->ismainworker,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-threadcomm_pool_affpolicy","Thread affinity policy"," ",PetscPThreadCommAffinityPolicyTypes,(PetscEnum)tcomm->pool->aff,(PetscEnum*)&tcomm->pool->aff,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-threadcomm_pool_spark","Thread pool spark type"," ",PetscThreadPoolSparkTypes,(PetscEnum)tcomm->pool->spark,(PetscEnum*)&tcomm->pool->spark,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-threadcomm_pool_synchronizeafter","Puts a barrier after every kernel call",NULL,PETSC_TRUE,&tcomm->pool->synchronizeafter,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (tcomm->ismainworker) {
    tcomm->pool->thread_num_start = 1;
  } else {
    tcomm->pool->thread_num_start = 0;
  }

  /* Set up thread ranks */
  for (i=0; i<tcomm->nworkThreads; i++) tcomm->pool->granks[i] = i;

  /* Set the leader thread rank */
  if (tcomm->pool->nthreads) {
    if (tcomm->ismainworker) tcomm->leader = tcomm->pool->granks[1];
    else tcomm->leader = tcomm->pool->granks[0];
  }
  printf("Initialized pool with %d threads\n",tcomm->pool->nthreads);
  // Create job queue
  ierr = PetscThreadPoolCreateJobQueue(tcomm,tcomm->pool);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolCreateJobQueue"
PetscErrorCode PetscThreadPoolCreateJobQueue(PetscThreadComm tcomm,PetscThreadPool pool)
{
  PetscInt i,j;
  PetscThreadCommJobQueue jobqueue;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Allocate queue
  ierr = PetscNew(&pool->jobqueue);
  jobqueue = pool->jobqueue;

  // Create job contexts
  ierr = PetscMalloc1(tcomm->nkernels,&jobqueue->jobs);CHKERRQ(ierr);
  ierr = PetscMalloc1(tcomm->nworkThreads*tcomm->nkernels,&jobqueue->jobs[0].job_status);CHKERRQ(ierr);
  for (i=0; i<tcomm->nkernels; i++) {
    jobqueue->jobs[i].job_status = jobqueue->jobs[0].job_status + i*tcomm->nworkThreads;
    for (j=0; j<tcomm->nworkThreads; j++) jobqueue->jobs[i].job_status[j] = THREAD_JOB_NONE;
  }

  // Set queue variables
  pool->jobqueue->ctr = 0;
  pool->jobqueue->kernel_ctr = 0;
  tcomm->job_ctr = 0;

  // Create thread info
  ierr = PetscMalloc1(tcomm->nworkThreads,&jobqueue->tinfo);CHKERRQ(ierr);
  for(i=0; i<tcomm->nworkThreads; i++) {
    ierr = PetscNew(&jobqueue->tinfo[i]);CHKERRQ(ierr);
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
  ierr = PetscCommGetPool(comm,&pool);

  ierr = (*tcomm->ops->atomicincrement)(tcomm,&tcomm->pool->nthreads,1);

  printf("adding thread nthreads=%d\n",tcomm->pool->nthreads);
  ierr = (*tcomm->ops->globalbarrier)(tcomm);
  if(trank==0) {
    *prank = 0;
  } else {
    pool->jobqueue->tinfo[trank]->status = THREAD_INITIALIZED;
    pool->jobqueue->tinfo[trank]->rank = pool->granks[trank];
    pool->jobqueue->tinfo[trank]->data = 0;
    pool->jobqueue->tinfo[trank]->tcomm = tcomm;
    *prank = -1;
  }
  ierr = (*tcomm->ops->globalbarrier)(tcomm);

  if(trank>0) {
    PetscThreadInfo tinfo = pool->jobqueue->tinfo[trank];
    ierr = PetscThreadPoolFunc_User(tinfo);
  }
  PetscFunctionReturn(0);
}

/* Checks whether this thread is a member of tcomm */
PetscBool CheckThreadCommMembership(PetscInt myrank,PetscThreadComm tcomm)
{
  PetscInt i;

  for (i=0;i<tcomm->nworkThreads;i++) {
    if (myrank == tcomm->pool->granks[i]) return PETSC_TRUE;
  }
  return PETSC_FALSE;
}

void SparkThreads(PetscInt myrank,PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  switch (tcomm->pool->spark) {
  case PTHREADPOOLSPARK_SELF:
    if (CheckThreadCommMembership(myrank,tcomm)) {
      tcomm->pool->jobqueue->tinfo[myrank]->data = job;
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

  PetscFunctionBegin;
  tinfo = *(PetscThreadInfo*)arg;
  trank = tinfo->rank;
  tcomm = tinfo->tcomm;
  jobqueue = tcomm->pool->jobqueue;
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
      my_job_counter = (my_job_counter+1)%job->tcomm->nkernels;
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

  PetscFunctionBegin;
  trank = tinfo->rank;
  tcomm = tinfo->tcomm;
  jobqueue = tcomm->pool->jobqueue;
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
      tcomm->my_job_counter[trank] = (tcomm->my_job_counter[trank]+1)%job->tcomm->nkernels;
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
    ierr = PetscCommGetPool(comm,&pool);
    for(i=pool->thread_num_start; i<pool->nthreads; i++) {
      printf("terminate thread %d\n",i);
      pool->jobqueue->tinfo[i]->status = THREAD_TERMINATE;
    }
  }
  ierr = (*tcomm->ops->globalbarrier)(tcomm);
  ierr = (*tcomm->ops->atomicincrement)(tcomm,&pool->nthreads,-1);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolBarrier"
PetscErrorCode PetscThreadPoolBarrier(PetscThreadComm tcomm)
{
  PetscInt                active_threads=0,i;
  PetscBool               wait          =PETSC_TRUE;
  PetscThreadCommJobQueue jobqueue      =tcomm->pool->jobqueue;
  PetscThreadCommJobCtx   job           =&jobqueue->jobs[tcomm->job_ctr];
  PetscInt                job_status;

  PetscFunctionBegin;
  printf("In PetscThreadPoolBarrier job_ctr=%d\n",tcomm->job_ctr);
  if (tcomm->nworkThreads == 1 && tcomm->ismainworker) PetscFunctionReturn(0);

  /* Loop till all threads signal that they have done their job */
  while (wait) {
    for (i=0; i<tcomm->nworkThreads; i++) {
      job_status      = job->job_status[tcomm->pool->granks[i]];
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if(!pool) PetscFunctionReturn(0);

  ierr = PetscFree(pool->granks);CHKERRQ(ierr);
  ierr = PetscFree(pool->jobqueue->jobs[0].job_status);CHKERRQ(ierr);
  ierr = PetscFree(pool->jobqueue->jobs);CHKERRQ(ierr);
  ierr = PetscFree(pool->jobqueue->tinfo);CHKERRQ(ierr);
  ierr = PetscFree(pool->jobqueue);CHKERRQ(ierr);
  ierr = PetscFree(pool);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolSetAffinity"
PetscErrorCode PetscThreadPoolSetAffinity(PetscThreadComm tcomm,cpu_set_t *cpuset,PetscInt trank,PetscBool *set)
{
  PetscInt ncores,j;

  PetscFunctionBegin;
  PetscGetNCores(&ncores);
  switch (tcomm->pool->aff) {
  case PTHREADAFFPOLICY_ONECORE:
    CPU_ZERO(cpuset);
    CPU_SET(tcomm->affinities[trank]%ncores,cpuset);
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
    if(tcomm->ismainworker && trank==0) {
      CPU_ZERO(cpuset);
      CPU_SET(tcomm->affinities[0]%ncores,cpuset);
      *set = PETSC_TRUE;
    } else {
      *set = PETSC_FALSE;
    }
    break;
  }
  PetscFunctionReturn(0);
}
#endif

#include <../src/sys/threadcomm/impls/pthread/tcpthreadimpl.h>

#define THREAD_TERMINATE      0
#define THREAD_INITIALIZED    1
#define THREAD_CREATED        0
#if defined PETSC_HAVE_MALLOC_H
#include <malloc.h>
#endif



/* Checks whether this thread is a member of tcomm */
PetscBool CheckThreadCommMembership(PetscInt myrank,PetscThreadComm tcomm)
{
  PetscInt                i;
  PetscThreadComm_PThread ptcomm;

  ptcomm = (PetscThreadComm_PThread)tcomm->data;

  for (i=0;i<tcomm->nworkThreads;i++) {
    if (myrank == ptcomm->granks[i]) return PETSC_TRUE;
  }
  return PETSC_FALSE;
}

void SparkThreads_LockFree(PetscInt myrank,PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscThreadComm_PThread ptcomm;

  ptcomm = (PetscThreadComm_PThread)tcomm->data;
  switch (ptcomm->spark) {
  case PTHREADPOOLSPARK_SELF:
    if (CheckThreadCommMembership(myrank,tcomm)) {
      tcomm->jobqueue->current_data[myrank] = job;
      job->job_status[myrank] = THREAD_JOB_RECIEVED;
    }
    break;
  }
}

void *PetscPThreadCommFunc_LockFree(void *arg)
{
  PetscInt              my_job_counter = 0,my_kernel_ctr=0,glob_kernel_ctr;
  PetscThreadCommJobCtx job;
  PetscThreadCommJobQueue jobqueue;

  PetscThreadInfo tinfo = *(PetscThreadInfo*)arg;
#if defined(PETSC_PTHREAD_LOCAL)
  PetscPThreadRank = tinfo->rank;
#else
  PetscInt PetscPThreadRank=tinfo->rank;
  pthread_setspecific(PetscPThreadRankkey,&PetscPThreadRank);
#endif
  PetscInt trank = tinfo->rank;
  PetscThreadComm tcomm = tinfo->tcomm;
  jobqueue = tcomm->jobqueue;

  printf("In threadcommfunc rank=%d nworkThreads=%d\n",PetscPThreadRank,tcomm->nworkThreads);

  jobqueue->current_data[PetscPThreadRank] = NULL;
  jobqueue->current_status[PetscPThreadRank] = THREAD_INITIALIZED;

  /* Spin loop */
  while (PetscReadOnce(int,jobqueue->current_status[PetscPThreadRank]) != THREAD_TERMINATE) {
    glob_kernel_ctr = PetscReadOnce(int,jobqueue->kernel_ctr);
    if (my_kernel_ctr < glob_kernel_ctr) {
      printf("rank=%d in first if status=%d\n",trank,jobqueue->current_status[trank]);
      job = &jobqueue->jobs[my_job_counter];
      /* Spark the thread pool */
      SparkThreads_LockFree(PetscPThreadRank,tcomm,job);
      printf("rank=%d job status=%d\n",trank,job->job_status[PetscPThreadRank]);
      if (job->job_status[PetscPThreadRank] == THREAD_JOB_RECIEVED) {
        printf("rank=%d running job\n",PetscPThreadRank);
        /* Do own job */
        PetscRunKernel(PetscPThreadRank,jobqueue->current_data[PetscPThreadRank]->nargs,jobqueue->current_data[PetscPThreadRank]);
        /* Post job completed status */
        job->job_status[PetscPThreadRank] = THREAD_JOB_COMPLETED;
      }
      my_job_counter = (my_job_counter+1)%tcomm->nkernels;
      my_kernel_ctr++;
    }
    PetscCPURelax();
  }

  return NULL;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommBarrier_PThread_LockFree"
PetscErrorCode PetscThreadCommBarrier_PThread_LockFree(PetscThreadComm tcomm)
{
  PetscInt                active_threads=0,i;
  PetscBool               wait          =PETSC_TRUE;
  PetscThreadComm_PThread ptcomm        =(PetscThreadComm_PThread)tcomm->data;
  PetscThreadCommJobQueue jobqueue      =tcomm->jobqueue;
  PetscThreadCommJobCtx   job           =&jobqueue->jobs[tcomm->job_ctr];
  PetscInt                job_status;

  PetscFunctionBegin;
  if (tcomm->nworkThreads == 1 && ptcomm->ismainworker) PetscFunctionReturn(0);

  /* Loop till all threads signal that they have done their job */
  while (wait) {
    for (i=0; i<tcomm->nworkThreads; i++) {
      job_status      = job->job_status[ptcomm->granks[i]];
      active_threads += job_status;
    }
    if (PetscReadOnce(int,active_threads) > 0) active_threads = 0;
    else wait=PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscPThreadCommInitialize_LockFree"
PetscErrorCode PetscPThreadCommInitialize_LockFree(PetscThreadComm tcomm)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscThreadComm_PThread ptcomm=(PetscThreadComm_PThread)tcomm->data;
  PetscThreadCommJobQueue jobqueue=tcomm->jobqueue;

  PetscFunctionBegin;
  /* Create threads */
  for (i=ptcomm->thread_num_start; i < tcomm->nworkThreads; i++) {
    printf("Creating thread=%d\n",i);
    //ierr = PetscNew(&jobqueue->tinfo[i]);
    jobqueue->current_status[i] = THREAD_CREATED;
    jobqueue->tinfo[i]->rank = ptcomm->granks[i];
    jobqueue->tinfo[i]->tcomm = tcomm;
    ierr = pthread_create(&ptcomm->tid[i],&ptcomm->attr[i],&PetscPThreadCommFunc_LockFree,&jobqueue->tinfo);CHKERRQ(ierr);
  }

  if (ptcomm->ismainworker) jobqueue->current_status[0] = THREAD_INITIALIZED;

  PetscInt threads_initialized=0;
  /* Wait till all threads have been initialized */
  while (threads_initialized != tcomm->nworkThreads) {
    threads_initialized=0;
    for (i=0; i<tcomm->nworkThreads; i++) {
      if (!jobqueue->current_status[ptcomm->granks[i]]) break;
      threads_initialized++;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscPThreadCommFinalize_LockFree"
PetscErrorCode PetscPThreadCommFinalize_LockFree(PetscThreadComm tcomm)
{
  PetscErrorCode          ierr;
  void                    *jstatus;
  PetscThreadComm_PThread ptcomm=(PetscThreadComm_PThread)tcomm->data;
  PetscThreadCommJobQueue jobqueue=tcomm->jobqueue;
  PetscInt                i;

  PetscFunctionBegin;
  ierr = PetscThreadCommBarrier_PThread_LockFree(tcomm);CHKERRQ(ierr);
  for (i=ptcomm->thread_num_start; i < tcomm->nworkThreads; i++) {
    printf("Terminating thread=%d\n",i);
    jobqueue->current_status[i] = THREAD_TERMINATE;
    ierr = pthread_join(ptcomm->tid[i],&jstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(jobqueue->current_status);CHKERRQ(ierr);
  ierr = PetscFree(jobqueue->current_data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_PThread_LockFree"
PetscErrorCode PetscThreadCommRunKernel_PThread_LockFree(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscErrorCode          ierr;
  PetscThreadComm_PThread ptcomm;
  PetscThreadCommJobQueue jobqueue=tcomm->jobqueue;

  PetscFunctionBegin;
  printf("rank=%d running kernel\n",0);
  ptcomm = (PetscThreadComm_PThread)tcomm->data;
  if (ptcomm->ismainworker) {
    job->job_status[0]   = THREAD_JOB_RECIEVED;
    jobqueue->current_data[0] = job;
    PetscRunKernel(0,job->nargs, jobqueue->current_data[0]);
    job->job_status[0]   = THREAD_JOB_COMPLETED;
  }
  if (ptcomm->synchronizeafter) {
    ierr = (*tcomm->ops->barrier)(tcomm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

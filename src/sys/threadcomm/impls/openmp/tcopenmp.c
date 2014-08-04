#define PETSC_DESIRE_FEATURE_TEST_MACROS
#include <../src/sys/threadcomm/impls/openmp/tcopenmpimpl.h>
#include <omp.h>

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommGetRank_OpenMP"
/*
   PetscThreadCommGetRank_OpenMP - Get rank of calling thread

   Not Collective

   Output Parameters:
.  trank - Rank of calling thread

   Level: developer

*/
PetscErrorCode PetscThreadCommGetRank_OpenMP(PetscInt *trank)
{
  *trank =  omp_get_thread_num();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommSetAffinity_OpenMP"
/*
   PetscThreadCommSetAffinity_OpenMPLoop - Set thread affinity for an openmp
                                           thread for the loop thread model

   Not Collective

   Input Parameters:
+  pool   - Threadpool containing affinity settings
-  thread - Thread to set the affinity for (unused for this setaffinity routine)

   Level: developer

   Notes:
   This routine assumes that there is one threadcomm and one threadpool, and
   is therefore only used by the loop user model.

*/
PETSC_EXTERN PetscErrorCode PetscThreadCommSetAffinity_OpenMP(PetscThreadPool pool,PetscThread thread)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)

  #pragma omp parallel num_threads(pool->npoolthreads)
  {
    cpu_set_t      cpuset;
    PetscInt       trank;
    PetscBool      set;
    PetscErrorCode ierr;
    trank = omp_get_thread_num();

    ierr = PetscThreadPoolSetAffinity(pool,&cpuset,trank,&set);CHKERRCONTINUE(ierr);
    if (set) sched_setaffinity(0,sizeof(cpu_set_t),&cpuset);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadInit_OpenMP"
PETSC_EXTERN PetscErrorCode PetscThreadInit_OpenMP()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ThreadType = THREAD_TYPE_OPENMP;
  ierr = PetscThreadLockInitialize_OpenMP();CHKERRQ(ierr);
  PetscThreadLockAcquire = PetscThreadLockAcquire_OpenMP;
  PetscThreadLockRelease = PetscThreadLockRelease_OpenMP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolInit_OpenMP"
/*
   PetscThreadPoolInit_OpenMP - Initialize the threadpool to use OpenMP as
                                the threading type

   Not Collective

   Input Parameters:
.  pool - Threadpool to initialize

   Level: developer

*/
PETSC_EXTERN PetscErrorCode PetscThreadPoolInit_OpenMP(PetscThreadPool pool)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pool->model == THREAD_MODEL_AUTO) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to use auto thread model with OpenMP. Use loop or user model with OpenMP");

  ierr = PetscStrcpy(pool->type,OPENMP);CHKERRQ(ierr);
  pool->threadtype = THREAD_TYPE_OPENMP;
  pool->ops->setaffinities = PetscThreadCommSetAffinity_OpenMP;
  pool->ops->pooldestroy = PetscThreadPoolDestroy_OpenMP;
  if (pool->model == THREAD_MODEL_LOOP) {
    // Initialize each thread
    #pragma omp parallel num_threads(pool->npoolthreads)
    {
      ierr = PetscThreadInitialize();CHKERRCONTINUE(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInit_OpenMP"
/*
   PetscThreadCommInit_OpenMP - Initialize the threadcomm to use OpenMP as
                                the threading type
*/
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_OpenMP(PetscThreadComm tcomm)
{
  PetscThreadComm_OpenMP ptcomm;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&ptcomm);CHKERRQ(ierr);
  ptcomm->barrier_threads = 0;
  ptcomm->wait_inc = PETSC_TRUE;
  ptcomm->wait_dec = PETSC_TRUE;

  tcomm->data = (void*)ptcomm;
  if (tcomm->model == THREAD_MODEL_LOOP) {
    tcomm->ops->runkernel = PetscThreadCommRunKernel_OpenMPLoop;
  } else if (tcomm->model == THREAD_MODEL_USER) {
    tcomm->ops->runkernel = PetscThreadCommRunKernel_OpenMPUser;
  }
  tcomm->ops->commdestroy = PetscThreadCommDestroy_OpenMP;
  tcomm->ops->barrier     = PetscThreadCommBarrier_OpenMP;
  tcomm->ops->getrank     = PetscThreadCommGetRank_OpenMP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommDestroy_OpenMP"
/*
   PetscThreadCommDestroy_OpenMP - Destroy OpenMP threadcomm structs
*/
PETSC_EXTERN PetscErrorCode PetscThreadCommDestroy_OpenMP(PetscThreadComm tcomm)
{
  PetscThreadComm_OpenMP ptcomm = (PetscThreadComm_OpenMP)tcomm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if(!ptcomm) PetscFunctionReturn(0);
  /* Destroy openmp threadcomm data */
  ierr = PetscFree(ptcomm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolDestroy_OpenMP"
/*
   PetscThreadPoolDestroy_OpenMP - Destroy OpenMP thread structs
*/
PETSC_EXTERN PetscErrorCode PetscThreadPoolDestroy_OpenMP(PetscThreadPool pool)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Destroy openmp thread data */
  if (pool->model == THREAD_MODEL_LOOP) {
    #pragma omp parallel num_threads(pool->npoolthreads)
    {
      ierr = PetscThreadFinalize();CHKERRCONTINUE(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_OpenMPLoop"
/*
   PetscThreadCommRunKernel_OpenMPLoop - Run kernel routine for OpenMP loop thread model

   Not Collective

   Input Parameters:
+  tcomm - Threadcomm to run the kernel
-  job   - job for the kernel to run

   Level: developer

   Notes:
   Creates an OpenMP parallel region for the threadcomm and runs the jobs on the threads
   in the threadcomm. OpenMP automatically synchronizes at the end of the OpenMP region.

*/
PetscErrorCode PetscThreadCommRunKernel_OpenMPLoop(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   threadjob;
  PetscInt                trank;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
#pragma omp parallel num_threads(tcomm->ncommthreads) private(trank,jobqueue,threadjob)
  {
    trank = omp_get_thread_num();
    /* Get thread specific jobqueue and job */
    jobqueue = tcomm->commthreads[trank]->jobqueue;
    threadjob = &jobqueue->jobs[jobqueue->newest_job_index];
    /* Run kernel and update thread status */
    threadjob->job_status = THREAD_JOB_RECEIVED;
    ierr = PetscRunKernel(trank,threadjob->nargs,threadjob);CHKERRCONTINUE(ierr);
    threadjob->job_status = THREAD_JOB_COMPLETED;
    jobqueue->current_job_index = (jobqueue->current_job_index+1)%tcomm->nkernels;
    jobqueue->completed_jobs_ctr++;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_OpenMPUser"
/*
   PetscThreadCommRunKernel_OpenMPUser - Run kernel routine for OpenMP user thread model

   Not Collective

   Input Parameters:
+  tcomm - Threadcomm to run the kernel
-  job   - job for the kernel to run

   Level: developer

   Notes:
   This routine must be called from within an OpenMP parallel region. Runs the jobs on
   the calling thread. Synchronization at the end of the kernel is optional. If the user
   does not synchronize in this function, they will need to call a synchronization function
   in their code to verify that the job has finished.

*/
PetscErrorCode PetscThreadCommRunKernel_OpenMPUser(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscThreadCommJobQueue jobqueue;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  if (tcomm->ismainworker) {
    job->job_status = THREAD_JOB_RECEIVED;
    ierr = PetscRunKernel(0,job->nargs,job);CHKERRCONTINUE(ierr);
    job->job_status = THREAD_JOB_COMPLETED;
    jobqueue = tcomm->commthreads[tcomm->lleader]->jobqueue;
    jobqueue->current_job_index = (jobqueue->current_job_index+1)%tcomm->nkernels;
    jobqueue->completed_jobs_ctr++;
  }
  if (tcomm->syncafter) {
    ierr = PetscThreadCommJobBarrier(tcomm);CHKERRCONTINUE(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommBarrier_OpenMP"
/*
   PetscThreadCommBarrier_OpenMP - Barrier that ensures all threads in a threadcomm
                                   have reached the barrier

   Collective on threadcomm

   Input Parameters:
.  tcomm - Threadcomm

   Level: developer

   Notes:
   The OpenMP barrier only works with all OpenMP threads. This routine implements a
   reusable barrier that can block threads in one threadcomm while threads
   in other threadcomms continue executing. This barrier has each thread increment a
   counter and wait until all threads have incremented the counter, then has each
   thread decrement the counter and wait until all threads have decremented the counter.
   This ensures that the barrier is ready to be called again.

*/
PetscErrorCode PetscThreadCommBarrier_OpenMP(PetscThreadComm tcomm)
{
  PetscThreadComm_OpenMP ptcomm = (PetscThreadComm_OpenMP)tcomm->data;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_Barrier,0,0,0,0);CHKERRQ(ierr);
  // Increment counter one thread at a time
  #pragma omp atomic
  ptcomm->barrier_threads++;

  // Make sure all threads increment counter
  while (ptcomm->wait_inc) {
    if (PetscReadOnce(int,ptcomm->barrier_threads) == tcomm->ncommthreads) {
      #pragma omp critical
      {
        ptcomm->wait_dec = PETSC_TRUE;
        ptcomm->wait_inc = PETSC_FALSE;
      }
    }
  }

  // Decrement counter one thread at a time
  #pragma omp atomic
  ptcomm->barrier_threads--;

  // Make sure all threads decrement counter
  while (ptcomm->wait_dec) {
    if (PetscReadOnce(int,ptcomm->barrier_threads) == 0) {
      #pragma omp critical
      {
        ptcomm->wait_inc = PETSC_TRUE;
        ptcomm->wait_dec = PETSC_FALSE;
      }
    }
  }
  ierr = PetscLogEventEnd(ThreadComm_Barrier,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadLockCreate_OpenMP"
PetscErrorCode PetscThreadLockCreate_OpenMP(void **lock)
{
  PetscThreadLock_OpenMP omplock;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&omplock);CHKERRQ(ierr);
  omp_init_lock(omplock);
  *lock = (void*)omplock;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadLockInitialize_OpenMP"
PetscErrorCode PetscThreadLockInitialize_OpenMP(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscLocks) PetscFunctionReturn(0);

  ierr = PetscNew(&PetscLocks);CHKERRQ(ierr);
  ierr = PetscThreadLockCreate_OpenMP(&PetscLocks->trmalloc_lock);
  ierr = PetscThreadLockCreate_OpenMP(&PetscLocks->vec_lock);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadLockAcquire_OpenMP"
PetscErrorCode PetscThreadLockAcquire_OpenMP(void *lock)
{
  PetscThreadLock_OpenMP omplock;

  PetscFunctionBegin;
  omplock = (PetscThreadLock_OpenMP)lock;
  omp_set_lock(omplock);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadLockRelease_OpenMP"
PetscErrorCode PetscThreadLockRelease_OpenMP(void *lock)
{
  PetscThreadLock_OpenMP omplock;

  PetscFunctionBegin;
  omplock = (PetscThreadLock_OpenMP)lock;
  omp_unset_lock(omplock);
  PetscFunctionReturn(0);
}

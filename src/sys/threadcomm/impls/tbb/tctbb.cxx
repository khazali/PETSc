#include <../src/sys/threadcomm/impls/tbb/tctbbimpl.h>
#include "tbb/tbb.h"

using namespace tbb;

/* TBBRunKernel - Class to run a TBB kernel in parallel */
class TBBRunKernel {
  PetscThreadCommJobCtx job; /* Job to run on the kernel */

public:
  /* Run kernel. Use blocked_range to pass thread id to kernel. */
  void operator()(blocked_range<size_t>& r) const
  {
    PetscErrorCode ierr;
    PetscInt       trank= r.begin();

    job->job_status = THREAD_JOB_RECEIVED;
    ierr = PetscRunKernel(trank,job->nargs,job);CHKERRCONTINUE(ierr);
    job->job_status = THREAD_JOB_COMPLETED;
  }

  TBBRunKernel(PetscThreadCommJobCtx ijob) : job(ijob) {}
};

#undef __FUNCT__
#define __FUNCT__ "PetscThreadInit_TBB"
PetscErrorCode PetscThreadInit_TBB()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ThreadType = THREAD_TYPE_TBB;
  ierr = PetscThreadLockInitialize_TBB();CHKERRQ(ierr);
  PetscThreadLockAcquire = PetscThreadLockAcquire_TBB;
  PetscThreadLockRelease = PetscThreadLockRelease_TBB;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadPoolInit_TBB"
/*
   PetscThreadPoolInit_TBB - Initialize threadpool to use tbb

   Not Collective

   Input Parameters:
.  pool - Threadpool to initialize

   Level: developer

*/
PetscErrorCode PetscThreadPoolInit_TBB(PetscThreadPool pool)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pool->model == THREAD_MODEL_AUTO || pool->model == THREAD_MODEL_USER) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to use auto or user thread model with TBB. Use loop model with TBB");

  ierr = PetscStrcpy(pool->type,TBB);CHKERRQ(ierr);
  pool->threadtype = THREAD_TYPE_TBB;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommInit_TBB"
/*
   PetscThreadCommInit_TBB - Initialize threadcomm to use tbb

   Not Collective

   Input Parameters:
.  tcomm - Threadcomm to initialize

   Level: developer

*/
PetscErrorCode PetscThreadCommInit_TBB(PetscThreadComm tcomm)
{
  PetscFunctionBegin;
  tcomm->ops->runkernel = PetscThreadCommRunKernel_TBB;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_TBB"
/*
   PetscThreadCommRunKernel_TBB - Run kernel using tbb

   Not Collective

   Input Parameters:
+  tcomm - Threadcomm to run kernel on
-  job   - Job to run on threadcomm

   Level: developer

   Notes:
   Uses affinity_partitioner to automatically set the thread affinities efficiently.
*/
PetscErrorCode PetscThreadCommRunKernel_TBB(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  static affinity_partitioner affinity;

  PetscFunctionBegin;
  task_scheduler_init init(tcomm->ncommthreads);
  parallel_for(blocked_range<size_t>(0,tcomm->ncommthreads,1),TBBRunKernel(job),affinity);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadLockInitialize_TBB"
PetscErrorCode PetscThreadLockInitialize_TBB(void)
{
  PetscThreadLock_TBB ptlock;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscLocks) PetscFunctionReturn(0);

  ierr = PetscNew(&PetscLocks);CHKERRQ(ierr);
  ptlock = (PetscThreadLock_TBB)PetscLocks->trmalloc_lock;
  //omp_init_lock(&ptlock->lock);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadLockAcquire_TBB"
PetscErrorCode PetscThreadLockAcquire_TBB(void *lock)
{
  PetscThreadLock_TBB ptlock;

  PetscFunctionBegin;
  ptlock = (PetscThreadLock_TBB)lock;
  //omp_set_lock(&ptlock->lock);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadLockRelease_TBB"
PetscErrorCode PetscThreadLockRelease_TBB(void *lock)
{
  PetscThreadLock_TBB ptlock;

  PetscFunctionBegin;
  ptlock = (PetscThreadLock_TBB)lock;
  //omp_unset_lock(&ptlock->lock);
  PetscFunctionReturn(0);
}

#include <../src/sys/threadcomm/impls/tbb/tctbbimpl.h>
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"

using namespace tbb;

class TBBRunKernel {
  PetscThreadCommJobCtx job;

public:
  void operator()(blocked_range<size_t>& r) const {
    PetscInt trank= r.begin();
    job->job_status[trank] = THREAD_JOB_RECIEVED;
    PetscRunKernel(trank,job->nargs,job);
    job->job_status[trank]= THREAD_JOB_COMPLETED;
  }

  TBBRunKernel(PetscThreadCommJobCtx ijob) : job(ijob) {}
};

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_TBB"
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_TBB(PetscThreadComm tcomm)
{
  PetscThreadPool pool = PETSC_THREAD_POOL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(pool->type,TBB);CHKERRQ(ierr);
  pool->ops->runkernel = PetscThreadCommRunKernel_TBB;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_TBB"
PetscErrorCode PetscThreadCommRunKernel_TBB(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscFunctionBegin;
  task_scheduler_init init(tcomm->ncommthreads);
  parallel_for(blocked_range<size_t>(0,tcomm->ncommthreads,1),TBBRunKernel(job));
  PetscFunctionReturn(0);
}

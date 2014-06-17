#define PETSC_DESIRE_FEATURE_TEST_MACROS
#include <../src/sys/threadcomm/impls/openmp/tcopenmpimpl.h>
#include <omp.h>

PetscErrorCode PetscThreadCommGetRank_OpenMP(PetscInt *trank)
{
  *trank =  omp_get_thread_num();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommSetAffinity_OpenMP"
PETSC_EXTERN PetscErrorCode PetscThreadCommSetAffinity_OpenMP(PetscThreadPool pool)
{
  PetscErrorCode ierr;
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  cpu_set_t *cpuset;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
  ierr = PetscMalloc1(pool->npoolthreads,&cpuset);
#pragma omp parallel num_threads(pool->npoolthreads) shared(pool)
  {
    PetscInt trank;
    PetscBool set;
    trank = omp_get_thread_num();
    PetscThreadPoolSetAffinity(pool,&cpuset[trank],trank,&set);
    if(set) sched_setaffinity(0,sizeof(cpu_set_t),&cpuset[trank]);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_OpenMPLoop"
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_OpenMPLoop(PetscThreadPool pool)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr                 = PetscStrcpy(pool->type,OPENMP);CHKERRQ(ierr);
  pool->ops->runkernel = PetscThreadCommRunKernel_OpenMPLoop;
  pool->ops->getrank   = PetscThreadCommGetRank_OpenMP;
  ierr                 = PetscThreadCommSetAffinity_OpenMP(pool);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_OpenMPUser"
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_OpenMPUser(PetscThreadPool pool)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr                       = PetscStrcpy(pool->type,OPENMP);CHKERRQ(ierr);
  pool->ops->runkernel       = PetscThreadCommRunKernel_OpenMPUser;
  pool->ops->getrank         = PetscThreadCommGetRank_OpenMP;
  pool->ops->kernelbarrier   = PetscThreadPoolBarrier;
  pool->ops->globalbarrier   = PetscThreadCommBarrier_OpenMP;
  pool->ops->atomicincrement = PetscThreadCommAtomicIncrement_OpenMP;
  ierr                       = PetscThreadCommSetAffinity_OpenMP(pool);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_OpenMPLoop"
PetscErrorCode PetscThreadCommRunKernel_OpenMPLoop(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscInt        trank=0;

  PetscFunctionBegin;
#pragma omp parallel num_threads(tcomm->ncommthreads) shared(job) private(trank)
  {
    trank = omp_get_thread_num();
    PetscRunKernel(trank,job->nargs,job);
    job->job_status[trank] = THREAD_JOB_COMPLETED;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_OpenMPUser"
PetscErrorCode PetscThreadCommRunKernel_OpenMPUser(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscThreadPool pool = PETSC_THREAD_POOL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("Running OpenMP User kernel\n");
  if(pool->ismainworker) {
    job->job_status[0] = THREAD_JOB_RECIEVED;
    PetscRunKernel(0,job->nargs,job);
    job->job_status[0] = THREAD_JOB_COMPLETED;
  }
  if(pool->synchronizeafter) {
    ierr = (*pool->ops->kernelbarrier)(tcomm);CHKERRCONTINUE(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommBarrier_OpenMP"
PetscErrorCode PetscThreadCommBarrier_OpenMP(PetscThreadComm tcomm)
{
  PetscFunctionBegin;
  #pragma omp barrier
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommAtomicIncrement_OpenMP"
PetscErrorCode PetscThreadCommAtomicIncrement_OpenMP(PetscThreadComm tcomm,PetscInt *val,PetscInt inc)
{
  PetscFunctionBegin;
  #pragma omp atomic
  (*val)+=inc;
  PetscFunctionReturn(0);
}

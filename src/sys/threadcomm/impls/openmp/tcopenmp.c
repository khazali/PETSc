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
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_OpenMPLoop(PetscThreadComm tcomm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr                  = PetscStrcpy(tcomm->type,OPENMP);CHKERRQ(ierr);
  tcomm->ops->runkernel = PetscThreadCommRunKernel_OpenMPLoop;
  tcomm->ops->getrank   = PetscThreadCommGetRank_OpenMP;
  ierr                  = PetscThreadCommSetAffinity_OpenMP(tcomm->pool);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_OpenMPUser"
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_OpenMPUser(PetscThreadComm tcomm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr                        = PetscStrcpy(tcomm->type,OPENMP);CHKERRQ(ierr);
  tcomm->ops->runkernel       = PetscThreadCommRunKernel_OpenMPUser;
  tcomm->ops->getrank         = PetscThreadCommGetRank_OpenMP;
  tcomm->ops->kernelbarrier   = PetscThreadPoolBarrier;
  tcomm->ops->globalbarrier   = PetscThreadCommBarrier_OpenMP;
  tcomm->ops->atomicincrement = PetscThreadCommAtomicIncrement_OpenMP;
  ierr                        = PetscThreadCommSetAffinity_OpenMP(tcomm->pool);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_OpenMPLoop"
PetscErrorCode PetscThreadCommRunKernel_OpenMPLoop(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscInt        trank=0;

  PetscFunctionBegin;
#pragma omp parallel num_threads(tcomm->ncommthreads) private(trank)
  {
    trank = omp_get_thread_num();
    PetscRunKernel(trank,job->nargs,job);
    job->job_status = THREAD_JOB_COMPLETED;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_OpenMPUser"
PetscErrorCode PetscThreadCommRunKernel_OpenMPUser(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("Running OpenMP User kernel\n");
  if(tcomm->ismainworker) {
    job->job_status = THREAD_JOB_RECIEVED;
    PetscRunKernel(0,job->nargs,job);
    job->job_status = THREAD_JOB_COMPLETED;
  }
  if(tcomm->syncafter) {
    ierr = (*tcomm->ops->kernelbarrier)(tcomm);CHKERRCONTINUE(ierr);
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

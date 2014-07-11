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
#define __FUNCT__ "PetscThreadCommInit_OpenMP"
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_OpenMP(PetscThreadPool pool)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if(pool->model==THREAD_MODEL_AUTO) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to use auto thread model with OpenMP. Use loop or user model with OpenMP");

  ierr                     = PetscStrcpy(pool->type,OPENMP);CHKERRQ(ierr);
  pool->threadtype         = THREAD_TYPE_OPENMP;
  pool->ops->setaffinities = PetscThreadCommSetAffinity_OpenMP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommCreate_OpenMP"
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_OpenMP(PetscThreadComm tcomm)
{
  PetscThreadComm_OpenMP ptcomm;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&ptcomm);CHKERRQ(ierr);
  ptcomm->barrier_threads = 0;
  ptcomm->wait_inc = PETSC_TRUE;
  ptcomm->wait_dec = PETSC_TRUE;

  tcomm->data           = (void*)ptcomm;
  if(tcomm->model==THREAD_MODEL_LOOP) {
    tcomm->ops->runkernel = PetscThreadCommRunKernel_OpenMPLoop;
  } else if(tcomm->model==THREAD_MODEL_USER) {
    tcomm->ops->runkernel = PetscThreadCommRunKernel_OpenMPUser;
  }
  tcomm->ops->barrier   = PetscThreadCommBarrier_OpenMP;
  tcomm->ops->getrank   = PetscThreadCommGetRank_OpenMP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_OpenMPLoop"
PetscErrorCode PetscThreadCommRunKernel_OpenMPLoop(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscThreadCommJobQueue jobqueue;
  PetscThreadCommJobCtx   threadjob;
  PetscInt                trank;

  PetscFunctionBegin;
#pragma omp parallel num_threads(tcomm->ncommthreads) private(trank,jobqueue,threadjob)
  {
    trank = omp_get_thread_num();
    /* Get thread specific jobqueue and job */
    jobqueue = tcomm->commthreads[trank]->jobqueue;
    threadjob = &jobqueue->jobs[jobqueue->newest_job_index];
    /* Run kernel and update thread status */
    threadjob->job_status = THREAD_JOB_RECIEVED;
    PetscRunKernel(trank,threadjob->nargs,threadjob);
    threadjob->job_status = THREAD_JOB_COMPLETED;
    jobqueue->current_job_index = (jobqueue->current_job_index+1)%tcomm->nkernels;
    jobqueue->completed_jobs_ctr++;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRunKernel_OpenMPUser"
PetscErrorCode PetscThreadCommRunKernel_OpenMPUser(PetscThreadComm tcomm,PetscThreadCommJobCtx job)
{
  PetscThreadCommJobQueue jobqueue;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("Running OpenMP User kernel\n");
  if(tcomm->ismainworker) {
    job->job_status = THREAD_JOB_RECIEVED;
    PetscRunKernel(0,job->nargs,job);
    job->job_status = THREAD_JOB_COMPLETED;
    jobqueue = tcomm->commthreads[tcomm->leader]->jobqueue;
    jobqueue->current_job_index = (jobqueue->current_job_index+1)%tcomm->nkernels;
    jobqueue->completed_jobs_ctr++;
  }
  if(tcomm->syncafter) {
    ierr = PetscThreadCommJobBarrier(tcomm);CHKERRCONTINUE(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommBarrier_OpenMP"
/* Reusable barrier that can block threads in one threadcomm while threads
 in other threadcomms continue executing. */
PetscErrorCode PetscThreadCommBarrier_OpenMP(PetscThreadComm tcomm)
{
  PetscThreadComm_OpenMP ptcomm = (PetscThreadComm_OpenMP)tcomm->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(ThreadComm_Barrier,0,0,0,0);CHKERRQ(ierr);
  // Increment counter one thread at a time
  #pragma omp atomic
  ptcomm->barrier_threads++;

  // Make sure all threads increment counter
  while(ptcomm->wait_inc) {
    if(PetscReadOnce(int,ptcomm->barrier_threads) == tcomm->ncommthreads) {
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
  while(ptcomm->wait_dec) {
    if(PetscReadOnce(int,ptcomm->barrier_threads) == 0) {
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

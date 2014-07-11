
#ifndef __THREADCOMMIMPL_H
#define __THREADCOMMIMPL_H

#include <petscthreadcomm.h>
#include <petsc-private/petscimpl.h>

#if defined(PETSC_HAVE_SYS_SYSINFO_H)
#include <sys/sysinfo.h>
#endif
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_SYS_SYSCTL_H)
#include <sys/sysctl.h>
#endif
#if defined(PETSC_HAVE_WINDOWS_H)
#include <windows.h>
#endif

PETSC_EXTERN PetscMPIInt Petsc_ThreadComm_keyval;

/* Max. number of arguments for kernel */
#define PETSC_KERNEL_NARGS_MAX 10

/* Reduction status of threads */
#define THREADCOMM_THREAD_WAITING_FOR_NEWRED 0
#define THREADCOMM_THREAD_POSTED_LOCALRED    1
/* Status of the reduction */
#define THREADCOMM_REDUCTION_NONE           -1
#define THREADCOMM_REDUCTION_NEW             0
#define THREADCOMM_REDUCTION_COMPLETE        1

/* Job status for threads */
#define THREAD_JOB_NONE       -1
#define THREAD_JOB_POSTED      1
#define THREAD_JOB_RECIEVED    2
#define THREAD_JOB_COMPLETED   0

/* Thread status */
#define THREAD_TERMINATE      0
#define THREAD_INITIALIZED    1
#define THREAD_CREATED        0

/* Thread model */
#define THREAD_MODEL_LOOP   0
#define THREAD_MODEL_AUTO   1
#define THREAD_MODEL_USER   2

/* Thread type */
#define THREAD_TYPE_NOTHREAD 0
#define THREAD_TYPE_PTHREAD  1
#define THREAD_TYPE_OPENMP   2
#define THREAD_TYPE_TBB      3

#define PetscReadOnce(type,val) (*(volatile type *)&val)

#if defined(PETSC_MEMORY_BARRIER)
#define PetscMemoryBarrier() do {PETSC_MEMORY_BARRIER();} while(0)
#else
#define PetscMemoryBarrier()
#endif
#if defined(PETSC_READ_MEMORY_BARRIER)
#define PetscReadMemoryBarrier() do {PETSC_READ_MEMORY_BARRIER();} while(0)
#else
#define PetscReadMemoryBarrier()
#endif
#if defined(PETSC_WRITE_MEMORY_BARRIER)
#define PetscWriteMemoryBarrier() do {PETSC_WRITE_MEMORY_BARRIER();} while(0)
#else
#define PetscWriteMemoryBarrier()
#endif

#if defined(PETSC_CPU_RELAX)
#define PetscCPURelax() do {PETSC_CPU_RELAX();} while (0)
#else
#define PetscCPURelax() do { } while (0)
#endif

typedef enum {THREADAFFPOLICY_ALL,THREADAFFPOLICY_ONECORE,THREADAFFPOLICY_NONE} PetscThreadCommAffPolicyType;
extern const char *const PetscThreadCommAffPolicyTypes[];

typedef struct _p_PetscThreadCommRedCtx *PetscThreadCommRedCtx;
struct _p_PetscThreadCommRedCtx{
  PetscThreadComm               tcomm;          /* The associated threadcomm */
  PetscInt                      red_status;     /* Reduction status */
  PetscInt                      *thread_status; /* Reduction status of each thread */
  void                          *local_red;     /* Array to hold local reduction contribution from each thread */
  PetscThreadCommReductionOp    op;             /* The reduction operation */
  PetscDataType                 type;           /* The reduction data type */
};

struct _p_PetscThreadCommReduction{
  PetscInt              nreds;                              /* Number of reductions in operation */
  PetscThreadCommRedCtx redctx;                             /* Reduction objects */
  PetscInt               ctr;                               /* Global Reduction counter */
  PetscInt              *thread_ctr;                        /* Reduction counter for each thread */
};

typedef struct _p_PetscThreadCommJobCtx* PetscThreadCommJobCtx;
struct  _p_PetscThreadCommJobCtx{
  PetscThreadComm   tcomm;                         /* The thread communicator */
  PetscInt          commrank;                      /* Rank of thread in communicator */
  PetscInt          nargs;                         /* Number of arguments for the kernel */
  PetscThreadKernel pfunc;                         /* Kernel function */
  void              *args[PETSC_KERNEL_NARGS_MAX]; /* Array of void* to hold the arguments */
  PetscScalar       scalars[3];                    /* Array to hold three scalar values */
  PetscInt          ints[3];                       /* Array to hold three integer values */
  PetscInt          job_status;                   /* Thread job status */
};

/* Structure to manage job queue */
typedef struct _p_PetscThreadCommJobQueue* PetscThreadCommJobQueue;
struct _p_PetscThreadCommJobQueue{
  PetscInt current_job_index;        /* Index of current job this thread is working on */
  PetscInt newest_job_index;         /* Index of newest job added to the jobqueue */
  PetscInt next_job_index;           /* Index of next available job slot */
  PetscInt total_jobs_ctr;           /* Total number of jobs added to jobqueue */
  PetscInt completed_jobs_ctr;       /* Total number of jobs completed in this jobqueue thread */
  PetscThreadCommJobCtx jobs;        /* Queue of jobs */
};

typedef struct _PetscThreadCommOps* PetscThreadCommOps;
struct _PetscThreadCommOps {
  PetscErrorCode (*runkernel)(PetscThreadComm,PetscThreadCommJobCtx);
  PetscErrorCode (*view)(PetscThreadComm,PetscViewer);
  PetscErrorCode (*barrier)(PetscThreadComm);
  PetscErrorCode (*getcores)(PetscThreadComm,PetscInt,PetscInt*);
  PetscErrorCode (*getrank)(PetscInt*);
  PetscErrorCode (*commdestroy)(PetscThreadComm);
};

typedef struct _p_PetscThread* PetscThread;
struct _p_PetscThread{
  PetscInt                grank;    /* Thread rank in pool */
  PetscThreadPool         pool;     /* Threadpool for current thread */
  PetscInt                status;   /* Status of current job for each thread */
  PetscThreadCommJobCtx   jobdata;  /* Data for current job for each thread */
  PetscInt                affinity; /* Core affinity of each thread */
  PetscThreadCommJobQueue jobqueue; /* Job queue */
  void                    *data;    /* Implementation specific thread data */
};

typedef struct _PetscThreadPoolOps* PetscThreadPoolOps;
struct _PetscThreadPoolOps {
  PetscErrorCode (*tcomminit)(PetscThreadComm);     /* Function to initialize threadcomm */
  PetscErrorCode (*createthread)(PetscThread);      /* Function to allocate thread struct */
  PetscErrorCode (*startthreads)(PetscThreadPool);  /* Function to initialize and create threads */
  PetscErrorCode (*setaffinities)(PetscThreadPool); /* Function to set thread affinities */
  PetscErrorCode (*pooldestroy)(PetscThreadPool);   /* Function to destroy threads */
};

struct _p_PetscThreadPool{
  PetscInt                refct;           /* Number of ThreadComm references */
  PetscInt                npoolthreads;    /* Max number of threads pool can hold */
  PetscThread             *poolthreads;    /* Array of all threads */

  char                    type[256];       /* Thread model type */
  PetscInt                model;           /* Threading model used */
  PetscInt                threadtype;      /* Threading type used */
  PetscThreadCommAffPolicyType aff;        /* Affinity policy */
  PetscInt                nkernels;        /* Maximum kernels launched */
  PetscInt                thread_start;    /* Pool rank of first thread*/
  PetscInt                ismainworker;    /* Is main thread a worker thread? */
  PetscThreadPoolOps      ops;             /* Threadpool operatioins table */
};

struct _p_PetscThreadComm{
  // Threadcomm information
  PetscInt                 model;        /* Threading model used */
  PetscInt                 threadtype;   /* Thread type used */
  PetscInt                 nkernels;     /* Maximum kernels launched */
  PetscInt                 refct;        /* Number of MPI_Comm references */
  PetscInt                 leader;       /* Rank of the leader thread. This thread manages
                                            the synchronization for collective operatons like reductions. */
  PetscInt                 thread_start; /* Index for the first created thread (=1 if main thread is a worker, else 0 */
  PetscThreadCommReduction red;          /* Reduction context */
  PetscBool                active;       /* Does this threadcomm have access to the threads? */
  PetscThreadCommOps       ops;          /* Threadcomm operations table */
  void                     *data;        /* Implementation specific threadcomm data */
  PetscBool                syncafter;    /* Whether the main thread should block until all threads complete kernel */
  PetscBool                ismainworker; /* Is the main thread also a work thread? */

  // Thread information
  PetscThreadPool         pool;         /* Threadpool containing threads for this comm */
  PetscInt                ncommthreads; /* Max threads comm can use */
  PetscThread             *commthreads; /* Threads that this comm can use */
};

/* register thread communicator models */
PETSC_EXTERN PetscErrorCode PetscThreadCommModelRegister(const char[],PetscErrorCode(*)(PetscThreadComm));
PETSC_EXTERN PetscErrorCode PetscThreadCommInitTypeRegister(const char sname[],PetscErrorCode (*function)(PetscThreadPool));
PETSC_EXTERN PetscErrorCode PetscThreadCommTypeRegister(const char[],PetscErrorCode(*)(PetscThreadComm));
PETSC_EXTERN PetscErrorCode PetscThreadCommRegisterAllModels(void);
PETSC_EXTERN PetscErrorCode PetscThreadCommRegisterAllTypes(PetscThreadPool);

#undef __FUNCT__
#define __FUNCT__
PETSC_STATIC_INLINE PetscErrorCode PetscRunKernel(PetscInt trank,PetscInt nargs,PetscThreadCommJobCtx job)
{
  printf("Running kernel with trank=%d\n",trank);
  switch(nargs) {
  case 0:
    (*job->pfunc)(trank);
    break;
  case 1:
    (*job->pfunc)(trank,job->args[0]);
    break;
  case 2:
    (*job->pfunc)(trank,job->args[0],job->args[1]);
    break;
  case 3:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2]);
    break;
  case 4:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3]);
    break;
  case 5:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4]);
    break;
  case 6:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5]);
    break;
  case 7:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5],job->args[6]);
    break;
  case 8:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5],job->args[6],job->args[7]);
    break;
  case 9:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5],job->args[6],job->args[7],job->args[8]);
    break;
  case 10:
    (*job->pfunc)(trank,job->args[0],job->args[1],job->args[2],job->args[3],job->args[4],job->args[5],job->args[6],job->args[7],job->args[8],job->args[9]);
    break;
  }
  return 0;
}

PETSC_EXTERN PetscErrorCode PetscThreadCommReductionCreate(PetscThreadComm,PetscThreadCommReduction*);
PETSC_EXTERN PetscErrorCode PetscThreadCommReductionDestroy(PetscThreadCommReduction);

PETSC_EXTERN PetscLogEvent ThreadComm_RunKernel, ThreadComm_Barrier;
#endif

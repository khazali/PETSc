
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
#define THREAD_MODEL_USER   1

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

typedef enum {PTHREADAFFPOLICY_ALL,PTHREADAFFPOLICY_ONECORE,PTHREADAFFPOLICY_NONE} PetscPThreadCommAffinityPolicyType;
extern const char *const PetscPTheadCommAffinityPolicyTypes[];

typedef enum {PTHREADPOOLSPARK_SELF} PetscThreadPoolSparkType;
extern const char *const PetscThreadPoolSparkTypes[];

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
  PetscInt          nargs;                         /* Number of arguments for the kernel */
  PetscThreadKernel pfunc;                         /* Kernel function */
  void              *args[PETSC_KERNEL_NARGS_MAX]; /* Array of void* to hold the arguments */
  PetscScalar       scalars[3];                    /* Array to hold three scalar values */
  PetscInt          ints[3];                       /* Array to hold three integer values */
  PetscInt          *job_status;                   /* Thread job status */
};

typedef struct _p_PetscThreadInfo* PetscThreadInfo;
struct _p_PetscThreadInfo{
  PetscInt        rank;       /* Rank of thread */
  PetscThreadComm tcomm;      /* Thread comm for current thread */
  PetscInt        status;     /* Status of current job for each thread */
  PetscThreadCommJobCtx data; /* Data for current job for each thread */
};

/* Structure to manage job queue */
typedef struct _p_PetscThreadCommJobQueue* PetscThreadCommJobQueue;
struct _p_PetscThreadCommJobQueue{
  PetscInt ctr;                      /* Job counter */
  PetscInt kernel_ctr;               /* Kernel counter .. need this otherwise race conditions are unavoidable */
  PetscThreadCommJobCtx jobs;        /* Queue of jobs */
  PetscThreadInfo *tinfo;            /* Data to pass to pthread worker */
};

typedef struct _PetscThreadCommOps* PetscThreadCommOps;
struct _PetscThreadCommOps {
  PetscErrorCode (*destroy)(PetscThreadComm);
  PetscErrorCode (*runkernel)(PetscThreadComm,PetscThreadCommJobCtx);
  PetscErrorCode (*view)(PetscThreadComm,PetscViewer);
  PetscErrorCode (*barrier)(PetscThreadComm);
  PetscErrorCode (*getrank)(PetscInt*);
};

struct _p_PetscThreadPool{
  PetscInt                nthreads;   /* Number of threads in pool */
  PetscInt                maxthreads; /* Max number of threads pool can hold */
  PetscInt                master;     /* Track master thread */
  PetscThreadCommJobQueue jobqueue;   /* Job queue */

  PetscInt                *granks;    /* Track thread ranks in pool */
  PetscInt                thread_num_start; /* Index for the first created thread (=1 if main thread is a worker, else 0 */
  PetscThreadPoolSparkType spark;  /* Type for sparking threads */
  PetscPThreadCommAffinityPolicyType  aff;    /* affinity policy */
  PetscBool                synchronizeafter; /* Whether the main thread should block until all threads complete kernel */
};

struct _p_PetscThreadComm{
  PetscInt                refct;
  PetscInt                nworkThreads; /* Number of threads in the pool */
  PetscInt                *affinities;  /* Thread affinity */
  PetscThreadCommOps      ops;          /* Operations table */
  void                    *data;        /* implementation specific data */
  char                    type[256];    /* Thread model type */
  PetscInt                model;        /* Threading model used */
  PetscInt                leader;       /* Rank of the leader thread. This thread manages
                                           the synchronization for collective operatons like reductions.
                                        */
  PetscThreadCommReduction red;         /* Reduction context */
  PetscInt                job_ctr;      /* which job is this threadcomm running in the job queue */
  PetscBool               isnothread;   /* No threading model used */
  PetscInt                nkernels;     /* Maximum kernels launched */
  PetscBool               ismainworker; /* Is the main thread also a work thread? */
  PetscThreadPool         pool;         /* Thread pool */
};

/* Global thread communicator that manages all the threads. Other threadcomms
   use threads from PETSC_THREAD_COMM_WORLD
*/
extern PetscThreadComm PETSC_THREAD_COMM_WORLD;

/* register thread communicator models */
PETSC_EXTERN PetscErrorCode PetscThreadModelRegister(const char[],PetscErrorCode(*)(PetscThreadComm));
PETSC_EXTERN PetscErrorCode PetscThreadCommRegister(const char[],PetscErrorCode(*)(PetscThreadComm));
PETSC_EXTERN PetscErrorCode PetscThreadCommRegisterAllModels(void);
PETSC_EXTERN PetscErrorCode PetscThreadCommRegisterAllTypes(PetscThreadComm tcomm);

#undef __FUNCT__
#define __FUNCT__
PETSC_STATIC_INLINE PetscErrorCode PetscRunKernel(PetscInt trank,PetscInt nargs,PetscThreadCommJobCtx job)
{
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

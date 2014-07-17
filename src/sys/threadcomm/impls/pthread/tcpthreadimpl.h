
#if !defined(__TCPTHREADIMPLH)
#define __TCPTHREADIMPLH

#include <petsc-private/threadcommimpl.h>

#if defined(PETSC_HAVE_PTHREAD_H)
#include <pthread.h>
#elif defined(PETSC_HAVE_WINPTHREADS_H)
#include <winpthreads.h>       /* http://locklessinc.com/downloads/winpthreads.h */
#endif

/* PetscThread_PThread - Contains PThread specific data structures for thread */
struct _p_PetscThread_PThread {
  pthread_t      tid;                       /* thread ids */
  pthread_attr_t attr;                      /* thread attributes */
};
typedef struct _p_PetscThread_PThread *PetscThread_PThread;

/* PetscThreadComm_PThread - Contains PThread specific data structures for threadcomm */
struct _p_PetscThreadComm_PThread {
  pthread_barrier_t barr;                    /* pthread barrier */
  pthread_mutex_t   threadmutex;             /* mutex for nthreads variable */
};
typedef struct _p_PetscThreadComm_PThread *PetscThreadComm_PThread;

/* Rank of the calling thread - thread local variable */
#if defined(PETSC_PTHREAD_LOCAL)
PETSC_EXTERN PETSC_PTHREAD_LOCAL PetscInt PetscPThreadRank;
#else
PETSC_EXTERN pthread_key_t PetscPThreadRankkey;
#endif

PETSC_EXTERN PetscErrorCode PetscThreadPoolInit_PThread(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommInit_PThread(PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadCommInitialize_PThread(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommDestroy_PThread(PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadPoolDestroy_PThread(PetscThreadPool);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel_PThread(PetscThreadComm,PetscThreadCommJobCtx);
PETSC_EXTERN PetscErrorCode PetscThreadCommBarrier_PThread(PetscThreadComm);

#endif

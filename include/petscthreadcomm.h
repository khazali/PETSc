
#if !defined(__PETSCTHREADCOMM_H)
#define __PETSCTHREADCOMM_H
#include <petscsys.h>

/* Function pointer cast for the kernel function */
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*PetscThreadKernel)(PetscInt,...);

/*
  PetscThreadComm - Abstract object that manages all thread communication models

  Level: developer

  Concepts: threads

.seealso: PetscThreadCommCreate(), PetscThreadCommDestroy
*/
typedef struct _p_PetscThreadComm *PetscThreadComm;
typedef struct _p_PetscThreadPool *PetscThreadPool;

/*
   PetscThreadCommReduction - Context used for managing threaded reductions

   Level: developer
*/
typedef struct _p_PetscThreadCommReduction *PetscThreadCommReduction;

typedef const char* PetscThreadCommModel;
#define LOOP                "loop"
#define AUTO                "auto"
#define USER                "user"

typedef const char* PetscThreadCommType;
#define NOTHREAD            "nothread"
#define PTHREAD             "pthread"
#define OPENMP              "openmp"
#define TBB                 "tbb"

PETSC_EXTERN PetscFunctionList PetscThreadCommInitTypeList;
PETSC_EXTERN PetscFunctionList PetscThreadCommTypeList;
PETSC_EXTERN PetscFunctionList PetscThreadCommModelList;

typedef enum {THREADCOMM_SUM,THREADCOMM_PROD,THREADCOMM_MAX,THREADCOMM_MIN,THREADCOMM_MAXLOC,THREADCOMM_MINLOC} PetscThreadCommReductionOp;
PETSC_EXTERN const char* const PetscThreadCommReductionOps[];

/* Max. number of reductions */
#define PETSC_REDUCTIONS_MAX 32

/* Package routines in dlregisthreadcomm.c */
PETSC_EXTERN PetscErrorCode PetscThreadCommInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscThreadCommFinalizePackage(void);

/* Threadcomm routines in threadcomm.c */
/* Initialization/Destruction routines */
PETSC_EXTERN PetscErrorCode PetscThreadCommInitialize(PetscInt,PetscInt*,PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadCommDestroy(PetscThreadComm*);
/* General routines */
PETSC_EXTERN PetscErrorCode PetscCommGetThreadComm(MPI_Comm,PetscThreadComm*);
PETSC_EXTERN PetscErrorCode PetscCommCheckGetThreadComm(MPI_Comm,PetscThreadComm*,PetscBool*);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetNThreads(MPI_Comm,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetRank(PetscThreadComm,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetOwnershipRanges(MPI_Comm,PetscInt,PetscInt*[]);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetAffinities(MPI_Comm,PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscThreadCommView(MPI_Comm,PetscViewer);
/* Routines to create threadcomms */
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate(MPI_Comm,PetscInt,MPI_Comm*);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreateShare(MPI_Comm,PetscInt,PetscInt*,MPI_Comm*);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreateAttach(MPI_Comm,PetscInt);
PETSC_EXTERN PetscErrorCode PetscThreadCommSplitEvenly(MPI_Comm,PetscInt,MPI_Comm**);
PETSC_EXTERN PetscErrorCode PetscThreadCommSplit(MPI_Comm,PetscInt,PetscInt*,MPI_Comm**);
PETSC_EXTERN PetscErrorCode PetscThreadCommDetach(MPI_Comm);
PETSC_EXTERN PetscErrorCode PetscThreadCommAttach(MPI_Comm,PetscThreadComm);
/* Kernel routines */
PETSC_EXTERN PetscErrorCode PetscThreadCommGetScalars(MPI_Comm,PetscScalar**,PetscScalar**,PetscScalar**);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetInts(MPI_Comm,PetscInt**,PetscInt**,PetscInt**);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel(MPI_Comm,PetscErrorCode (*)(PetscInt,...),PetscInt,...);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel0(MPI_Comm,PetscErrorCode (*)(PetscInt,...));
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel1(MPI_Comm,PetscErrorCode (*)(PetscInt,...),void*);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel2(MPI_Comm,PetscErrorCode (*)(PetscInt,...),void*,void*);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel3(MPI_Comm,PetscErrorCode (*)(PetscInt,...),void*,void*,void*);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel4(MPI_Comm,PetscErrorCode (*)(PetscInt,...),void*,void*,void*,void*);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel6(MPI_Comm,PetscErrorCode (*)(PetscInt,...),void*,void*,void*,void*,void*,void*);
/* Thread barrier routines */
PETSC_EXTERN PetscErrorCode PetscThreadCommBarrier(MPI_Comm);
PETSC_EXTERN PetscErrorCode PetscThreadCommUserBarrier(MPI_Comm);
PETSC_EXTERN PetscErrorCode PetscThreadCommJobBarrier(PetscThreadComm);
/* Thread Join/Return routines */
PETSC_EXTERN PetscErrorCode PetscThreadCommJoin(MPI_Comm*,PetscInt,PetscInt, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscThreadCommReturn(MPI_Comm*, PetscInt,PetscInt,PetscInt*);
/* Debug routines */
PETSC_EXTERN PetscErrorCode PetscThreadCommStackCreate(PetscInt);
PETSC_EXTERN PetscErrorCode PetscThreadCommStackDestroy(PetscInt);

/* Reduction operations in threadcommred.c */
PETSC_EXTERN PetscErrorCode PetscThreadReductionKernelPost(PetscInt,PetscThreadCommReduction,void*);
PETSC_EXTERN PetscErrorCode PetscThreadReductionKernelEnd(PetscInt,PetscThreadCommReduction,void*);
PETSC_EXTERN PetscErrorCode PetscThreadReductionBegin(MPI_Comm,PetscThreadCommReductionOp,PetscDataType,PetscInt,PetscThreadCommReduction*);
PETSC_EXTERN PetscErrorCode PetscThreadReductionEnd(PetscThreadCommReduction,void*);

/* Threadpool routines in threadpool.c */
/* Initialization/Destruction routines */
PETSC_EXTERN PetscErrorCode PetscThreadPoolAlloc(PetscThreadPool *pool);
PETSC_EXTERN PetscErrorCode PetscThreadPoolInitialize();
PETSC_EXTERN PetscErrorCode PetscThreadPoolDestroy(PetscThreadPool pool);
PETSC_EXTERN PetscErrorCode PetscThreadPoolSetModel(PetscThreadPool pool,PetscThreadCommModel model);
PETSC_EXTERN PetscErrorCode PetscThreadPoolSetType(PetscThreadPool pool,PetscThreadCommType type);
/* General routines */
PETSC_EXTERN PetscErrorCode PetscGetNCores(PetscInt*);
PETSC_EXTERN PetscErrorCode PetscThreadPoolGetPool(MPI_Comm comm,PetscThreadPool *pool);
PETSC_EXTERN PetscErrorCode PetscThreadPoolSetNThreads(PetscThreadPool pool,PetscInt nthreads);
PETSC_EXTERN PetscErrorCode PetscThreadPoolGetNThreads(MPI_Comm comm,PetscInt *nthreads);
PETSC_EXTERN PetscErrorCode PetscThreadPoolSetAffinities(PetscThreadPool pool,const PetscInt affinities[]);
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
PETSC_EXTERN PetscErrorCode PetscThreadPoolSetAffinity(PetscThreadPool pool,cpu_set_t *cpuset,PetscInt trank,PetscBool *set);
#endif
/* Worker thread routines */
PETSC_EXTERN PetscErrorCode PetscThreadPoolCreate(PetscThreadComm tcomm,PetscInt *nthreads);
PETSC_EXTERN void* PetscThreadPoolFunc(void *arg);

#endif

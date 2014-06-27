
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

PETSC_EXTERN PetscFunctionList PetscThreadCommTypeList;
PETSC_EXTERN PetscFunctionList PetscThreadCommModelList;

typedef enum {THREADCOMM_SUM,THREADCOMM_PROD,THREADCOMM_MAX,THREADCOMM_MIN,THREADCOMM_MAXLOC,THREADCOMM_MINLOC} PetscThreadCommReductionOp;
PETSC_EXTERN const char* const PetscThreadCommReductionOps[];

/* Max. number of reductions */
#define PETSC_REDUCTIONS_MAX 32

PETSC_EXTERN PetscErrorCode PetscCommGetThreadComm(MPI_Comm,PetscThreadComm*);
PETSC_EXTERN PetscErrorCode PetscThreadCommInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscThreadCommFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate(MPI_Comm,PetscInt,PetscBool,MPI_Comm*);
PETSC_EXTERN PetscErrorCode PetscThreadCommCreateShare(MPI_Comm comm,PetscInt nthreads,PetscInt *granks,MPI_Comm *mpicomm);
PETSC_EXTERN PetscErrorCode PetscThreadCommInitialize(PetscInt,PetscInt*,PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetNThreads(MPI_Comm,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetAffinities(MPI_Comm,PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscThreadCommView(MPI_Comm,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetScalars(MPI_Comm,PetscScalar**,PetscScalar**,PetscScalar**);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetInts(MPI_Comm,PetscInt**,PetscInt**,PetscInt**);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel(MPI_Comm,PetscErrorCode (*)(PetscInt,...),PetscInt,...);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel0(MPI_Comm,PetscErrorCode (*)(PetscInt,...));
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel1(MPI_Comm,PetscErrorCode (*)(PetscInt,...),void*);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel2(MPI_Comm,PetscErrorCode (*)(PetscInt,...),void*,void*);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel3(MPI_Comm,PetscErrorCode (*)(PetscInt,...),void*,void*,void*);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel4(MPI_Comm,PetscErrorCode (*)(PetscInt,...),void*,void*,void*,void*);
PETSC_EXTERN PetscErrorCode PetscThreadCommRunKernel6(MPI_Comm,PetscErrorCode (*)(PetscInt,...),void*,void*,void*,void*,void*,void*);
PETSC_EXTERN PetscErrorCode PetscThreadCommBarrier(MPI_Comm);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetOwnershipRanges(MPI_Comm,PetscInt,PetscInt*[]);
PETSC_EXTERN PetscErrorCode PetscThreadCommGetRank(PetscThreadComm,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscThreadCommDetach(MPI_Comm);
PETSC_EXTERN PetscErrorCode PetscThreadCommAttach(MPI_Comm,PetscThreadComm);
PETSC_EXTERN PetscErrorCode PetscThreadCommDestroy(PetscThreadComm*);

PETSC_EXTERN PetscErrorCode PetscThreadCommSplit(MPI_Comm comm,PetscInt ncomms,PetscInt *commsizes,MPI_Comm **splitcomms);
PETSC_EXTERN PetscErrorCode PetscThreadCommSetModel(PetscThreadComm tcomm,PetscThreadCommModel model);
PETSC_EXTERN PetscErrorCode PetscThreadCommSetType(PetscThreadComm tcomm,PetscThreadCommType type);

/* Reduction operations */
PETSC_EXTERN PetscErrorCode PetscThreadReductionKernelPost(PetscInt,PetscThreadCommReduction,void*);
PETSC_EXTERN PetscErrorCode PetscThreadReductionKernelEnd(PetscInt,PetscThreadCommReduction,void*);
PETSC_EXTERN PetscErrorCode PetscThreadReductionBegin(MPI_Comm,PetscThreadCommReductionOp,PetscDataType,PetscInt,PetscThreadCommReduction*);
PETSC_EXTERN PetscErrorCode PetscThreadReductionEnd(PetscThreadCommReduction,void*);

// Threadpool functions
PETSC_EXTERN PetscErrorCode PetscThreadPoolInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscThreadPoolFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscGetNCores(PetscInt*);
PETSC_EXTERN PetscErrorCode PetscThreadPoolGetPool(MPI_Comm comm,PetscThreadPool *pool);
PETSC_EXTERN PetscErrorCode PetscThreadPoolInitialize();
PETSC_EXTERN PetscErrorCode PetscThreadPoolAlloc(PetscThreadPool *pool);
PETSC_EXTERN PetscErrorCode PetscThreadPoolCreate(PetscThreadComm tcomm,PetscInt nthreads, PetscBool createthreads);
PETSC_EXTERN PetscErrorCode PetscThreadPoolSetNThreads(PetscThreadPool pool,PetscInt nthreads);
PETSC_EXTERN PetscErrorCode PetscThreadPoolGetNThreads(MPI_Comm comm,PetscInt *nthreads);
PETSC_EXTERN PetscErrorCode PetscThreadPoolSetAffinities(PetscThreadPool pool,const PetscInt affinities[]);
PETSC_EXTERN PetscErrorCode PetscThreadPoolJoin(MPI_Comm comm, PetscInt trank, PetscInt *poolrank);
PETSC_EXTERN PetscErrorCode PetscThreadCommLocalBarrier(PetscThreadComm tcomm);
PETSC_EXTERN void* PetscThreadPoolFunc(void *arg);
PETSC_EXTERN PetscErrorCode PetscThreadPoolReturn(MPI_Comm comm, PetscInt *poolrank);
PETSC_EXTERN PetscErrorCode PetscThreadPoolBarrier(PetscThreadComm comm);
PETSC_EXTERN PetscErrorCode PetscThreadPoolDestroy(PetscThreadComm comm);
#if defined(PETSC_HAVE_SCHED_CPU_SET_T)
PETSC_EXTERN PetscErrorCode PetscThreadPoolSetAffinity(PetscThreadPool pool,cpu_set_t *cpuset,PetscInt trank,PetscBool *set);
#endif
#endif
